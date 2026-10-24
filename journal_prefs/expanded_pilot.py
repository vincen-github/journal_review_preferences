"""Bounded same-year expansion and frozen paper-group learning-curve cohorts."""

import hashlib
import json
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .data import fingerprint, read_json, write_json
from .peerread_pilot import NoRedirect, band, parse_review_file, parsed_text

BANDS = ("low", "middle", "high")


def balanced_counts(capacities, required, target):
    counts = {b: required.get(b, 0) for b in BANDS}
    if any(counts[b] > capacities[b] for b in BANDS) or sum(counts.values()) > target:
        raise ValueError("Mandatory cohort exceeds capacity/target")
    while sum(counts.values()) < target:
        available = [b for b in BANDS if counts[b] < capacities[b]]
        if not available:
            raise ValueError("Insufficient eligible papers for requested target")
        chosen = min(available, key=lambda b: (counts[b], BANDS.index(b)))
        counts[chosen] += 1
    return counts


def proportional_counts(n, weights, capacities, initial=None):
    counts = {b: (initial or {}).get(b, 0) for b in BANDS}
    if sum(counts.values()) > n or any(counts[b] > capacities[b] for b in BANDS):
        raise ValueError("Held-out minimum quotas exceed capacity")
    for _ in range(n - sum(counts.values())):
        available = [b for b in BANDS if counts[b] < capacities[b]]
        if not available:
            raise ValueError("Insufficient unused papers for fixed held-out group")
        b = max(available, key=lambda b: (n * weights[b] / sum(weights.values()) - counts[b], -BANDS.index(b)))
        counts[b] += 1
    return counts


def freeze_cohort(records, mandatory_ids, seed=20261023, target=100, train_n=60, validation_n=20):
    """Metadata-only seeded selection; editorial decisions never enter selection."""
    if target - train_n - validation_n <= 0 or len(mandatory_ids) > min(20, train_n):
        raise ValueError("Invalid fixed cohort sizes")
    if len(set(mandatory_ids)) != len(mandatory_ids):
        raise ValueError("Duplicate mandatory paper IDs")
    eligible, exclusions, seen_ids, seen_titles = [], [], set(), set()
    for row in sorted(records, key=lambda r: (r["paper_id"], r.get("source_review_path", ""))):
        title = re.sub(r"\W+", " ", row["title"].lower()).strip()
        reason = None
        if row["paper_id"] in seen_ids:
            reason = "duplicate_paper_id"
        elif title and title in seen_titles:
            reason = "duplicate_normalized_title"
        elif row["mean_score"] is None:
            reason = "missing_recommendations"
        elif not row.get("source_text_blob"):
            reason = "missing_indexed_paired_text"
        if reason:
            exclusions.append({"paper_id": row["paper_id"], "reason": reason})
            continue
        seen_ids.add(row["paper_id"])
        if title:
            seen_titles.add(title)
        eligible.append(row)
    by_id = {r["paper_id"]: r for r in eligible}
    if set(mandatory_ids) - set(by_id):
        raise ValueError("A previously explored mandatory paper is not eligible")
    mandatory = [by_id[i] for i in mandatory_ids]
    pools = {b: [r for r in eligible if band(r["mean_score"]) == b and r["paper_id"] not in mandatory_ids] for b in BANDS}
    required = Counter(band(r["mean_score"]) for r in mandatory)
    capacities = {b: required[b] + len(pools[b]) for b in BANDS}
    counts = balanced_counts(capacities, required, target)
    rng = random.Random(seed)
    selected_new = {}
    for b in BANDS:
        rng.shuffle(pools[b])
        selected_new[b] = pools[b][:counts[b] - required[b]]
    # Reserve scarce bands jointly: validation cannot consume the only test example.
    test_min = {b: int(bool(selected_new[b])) for b in BANDS}
    val_min = {b: int(len(selected_new[b]) >= 2) for b in BANDS}
    val_counts = proportional_counts(validation_n, counts,
                                    {b: len(selected_new[b]) - test_min[b] for b in BANDS}, val_min)
    test_counts = proportional_counts(target - train_n - validation_n, counts,
                                      {b: len(selected_new[b]) - val_counts[b] for b in BANDS}, test_min)
    groups = {"train": list(mandatory), "validation": [], "test": []}
    train_new = {}
    for b in BANDS:
        group_rows = selected_new[b]
        groups["validation"] += group_rows[:val_counts[b]]
        groups["test"] += group_rows[val_counts[b]:val_counts[b] + test_counts[b]]
        train_new[b] = group_rows[val_counts[b] + test_counts[b]:]
    prefix_counts = Counter(band(r["mean_score"]) for r in mandatory)
    while len(groups["train"]) < train_n:
        choices = [b for b in BANDS if train_new[b]]
        b = min(choices, key=lambda b: (prefix_counts[b], BANDS.index(b)))
        groups["train"].append(train_new[b].pop())
        prefix_counts[b] += 1
    rng.shuffle(groups["validation"])
    rng.shuffle(groups["test"])
    ids = {g: [r["paper_id"] for r in rows] for g, rows in groups.items()}
    flat = sum(ids.values(), [])
    assert len(flat) == len(set(flat)) == target
    assert not set(mandatory_ids) & set(ids["validation"] + ids["test"])
    return groups, {"seed": seed, "source_count": len(records), "eligible_count": len(eligible),
                    "exclusions": exclusions, "frame_band_counts": capacities, "selected_band_counts": counts,
                    "group_band_counts": {g: dict(Counter(band(r["mean_score"]) for r in rows)) for g, rows in groups.items()},
                    "mandatory_ids": mandatory_ids, "ids": ids,
                    "nested_train_ids": {str(n): ids["train"][:n] for n in (20, 40, 60)},
                    "decision_filter_applied": False, "stratified_not_population_representative": True}


class SourceReader:
    """Pinned, hash-checked public JSON; 2 workers, <=1 request start/second."""
    def __init__(self, root, old_root, revision):
        self.root, self.old_root = Path(root).resolve(), Path(old_root).resolve()
        if self.root == self.old_root or self.root in self.old_root.parents or self.old_root in self.root.parents:
            raise ValueError("New and original data directories must be separate, nonnested paths")
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("Pin a full source commit")
        self.revision = revision
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.last_request, self.log = 0, []

    def network(self, url):
        # Serialize the entire network operation and stop flag. No thread can begin
        # a request after another request has reported a denial/failure.
        with self.lock:
            if self.stop.is_set():
                raise ValueError("Source access already stopped")
            time.sleep(max(0, 1 - (time.monotonic() - self.last_request)))
            if self.stop.is_set():
                raise ValueError("Source access already stopped")
            self.last_request = time.monotonic()
            opener = build_opener(ProxyHandler({}), NoRedirect())
            try:
                with opener.open(Request(url, headers={"User-Agent": "JournalReviewPreferences-bounded-public-research"}), timeout=30) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ValueError("Source JSON exceeds 2 MB")
                return raw
            except Exception:
                self.stop.set()
                raise

    def file(self, entry):
        path, expected = entry["path"], entry["sha"]
        if not re.fullmatch(r"data/iclr_2017/(train|dev|test)/(reviews|parsed_pdfs)/\d+(\.pdf)?\.json", path):
            raise ValueError("Only indexed ICLR2017 review/parsed-text JSON is allowed")
        # Keep Windows path resolution/directory creation together: other workers
        # must not create a path while it is being resolved and checked.
        with self.lock:
            destination = (self.root / "raw" / path).resolve()
            original = (self.old_root / "raw" / path).resolve()
            if self.root not in destination.parents or self.old_root not in original.parents:
                raise ValueError("Resolved cache path escapes its data directory: " + path
                                 + "; destination=" + str(destination) + "; original=" + str(original))
            destination.parent.mkdir(parents=True, exist_ok=True)
            cached = destination if destination.exists() else original
            cache_hit = cached.exists()
            raw = cached.read_bytes() if cache_hit else None
        if raw is None:
            raw = self.network("https://raw.githubusercontent.com/allenai/PeerRead/" + self.revision + "/" + path)
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected:
            self.stop.set()
            raise ValueError("Pinned source Git blob mismatch")
        with self.lock:
            if self.root not in destination.resolve().parents:
                raise ValueError("Cache destination changed before write: " + path)
            destination.write_bytes(raw)
            self.log.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "git_blob": blob,
                             "bytes": len(raw), "cached": cache_hit, "cached_from": str(cached) if cache_hit else None})
            write_json(self.root / "download_log.json", self.log)
            if len(self.log) % 50 == 0:
                print(json.dumps({"source_files_verified": len(self.log)}), flush=True)
        return json.loads(raw)


def build_expansion(root, old_root, allow_network=False):
    if allow_network is not True:
        raise ValueError("Explicit public-network opt-in required")
    root, old_root = Path(root), Path(old_root)
    if (root / "partition.json").exists():
        raise ValueError("Expanded frozen cohort already exists")
    protocol = read_json(root / "pre_ingestion_protocol.json")
    index = read_json(old_root / "source_index.json")
    if index["revision"] != protocol["revision"]:
        raise ValueError("Source revision differs from the pre-ingestion protocol")
    if (protocol["target"], protocol["split_counts"], protocol["nested_train_counts"]) != (
            100, {"train": 60, "validation": 20, "test": 20}, [20, 40, 60]):
        raise ValueError("This bounded implementation requires the declared 100/60-20-20/20-40-60 protocol")
    reader = SourceReader(root, old_root, index["revision"])
    try:
        cached_index = read_json(root / "source_index.json") if (root / "source_index.json").exists() else None
        if cached_index and cached_index["revision"] == index["revision"]:
            for split in ("dev", "test"):
                path = "data/iclr_2017/" + split + "/parsed_pdfs"
                if path in cached_index["paths"]:
                    index["paths"][path] = cached_index["paths"][path]
        for split in ("dev", "test"):
            path = "data/iclr_2017/" + split + "/parsed_pdfs"
            if path not in index["paths"]:
                url = "https://api.github.com/repos/allenai/PeerRead/contents/" + path + "?ref=" + index["revision"]
                index["paths"][path] = json.loads(reader.network(url))
        write_json(root / "source_index.json", index)
        entries, text_entries = [], {}
        for split in ("train", "dev", "test"):
            entries += index["paths"]["data/iclr_2017/" + split + "/reviews"]
            for entry in index["paths"]["data/iclr_2017/" + split + "/parsed_pdfs"]:
                text_entries[entry["path"]] = entry
        with ThreadPoolExecutor(max_workers=2) as pool:
            raw = list(pool.map(reader.file, entries))
        records = []
        for entry, data in zip(entries, raw):
            row = parse_review_file(data)
            text_path = entry["path"].replace("/reviews/", "/parsed_pdfs/").replace(".json", ".pdf.json")
            row.update(source_review_path=entry["path"], source_text_path=text_path,
                       source_text_blob=text_entries.get(text_path, {}).get("sha"))
            records.append(row)
        write_json(root / "metadata_frame.json", records)
        old_partition = read_json(old_root / "pilot/partition.json")["ids"]
        old_ids = sum((old_partition[g] for g in ("train", "validation", "test")), [])
        groups, partition = freeze_cohort(records, old_ids, protocol["seed"])
        selected = sum(groups.values(), [])
        with ThreadPoolExecutor(max_workers=2) as pool:
            texts = list(pool.map(reader.file, [text_entries[r["source_text_path"]] for r in selected]))
        seen_text = set()
        for row, data in zip(selected, texts):
            try:
                text = parsed_text(data)
            except ValueError as exc:
                raise ValueError("Invalid paired text for paper " + row["paper_id"] + ": " + str(exc)) from None
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_text:
                raise ValueError("Duplicate full manuscript text for paper " + row["paper_id"] + "; stop before freezing or scoring")
            seen_text.add(digest)
            row.update(text=text, text_sha256=digest, text_characters=len(text))
        out = root / "cohort"
        out.mkdir(exist_ok=False)
        for group, rows in groups.items():
            inputs = [{k: r[k] for k in ("paper_id", "title", "text", "text_sha256", "manuscript_version_status")} for r in rows]
            labels = [{k: r[k] for k in ("paper_id", "accepted", "reviews", "individual_scores", "mean_score", "score_stage")} for r in rows]
            write_json(out / (group + ".inputs.json"), inputs)
            write_json(out / (group + ".labels.json"), labels)
        partition.update(protocol_sha256=fingerprint(protocol), partition_sha256=fingerprint(partition["ids"]),
                         paired_count=len(selected), unique_recommendations=sum(len(r["reviews"]) for r in selected),
                         retained_unanimous_high_rejected=sum(r["all_high_rejected"] for r in selected),
                         source_revision=index["revision"], source_manuscript_stages="unverified; unchanged from initial pilot")
        write_json(root / "partition.json", partition)
        write_json(root / "ingestion_status.json", {"status": "complete", "paired_count": len(selected)})
        return partition
    except Exception as exc:
        write_json(root / "ingestion_status.json", {"status": "stopped_no_retry", "error_type": type(exc).__name__,
                                                   "error": str(exc), "verified_files": len(reader.log)})
        raise
