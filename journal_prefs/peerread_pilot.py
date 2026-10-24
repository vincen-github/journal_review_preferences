"""Small author-released PeerRead ICLR 2017 paper-group ingestion pilot.

No OpenReview access, model calls, source scripts, training or experiments.
Source content stays in the caller's gitignored local data directory.
"""

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .data import fingerprint, read_json, write_json

REPOSITORY = "https://github.com/allenai/PeerRead"
RAW_HOST = "https://raw.githubusercontent.com/allenai/PeerRead/"
SPLITS = {"train": "train", "validation": "dev", "test": "test"}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Public dataset redirects are disabled; stopped")


def parse_review_file(data):
    if not isinstance(data, dict) or not str(data.get("id", "")).isdigit():
        raise ValueError("Expected a numeric PeerRead paper ID")
    if not isinstance(data.get("accepted"), bool):
        raise ValueError("PeerRead acceptance label must be boolean")
    seen, rated, meta, unrated = set(), [], [], 0
    duplicates = 0
    for review in data.get("reviews", []):
        if not isinstance(review, dict):
            raise ValueError("Review entry must be an object")
        # Drop unused identifiers before deduplicating, without merging differing reviews.
        projected = {k: review[k] for k in ("TITLE", "comments", "DATE", "IS_META_REVIEW", "RECOMMENDATION", "REVIEWER_CONFIDENCE") if k in review}
        key = fingerprint(projected)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if review.get("IS_META_REVIEW") is True:
            meta.append(projected)
            continue
        score = review.get("RECOMMENDATION")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 1 <= score <= 10:
            unrated += 1
            continue  # Unscored discussions are not zero-score official reviews.
        projected["score_numeric"] = float(score)
        rated.append(projected)
    scores = [r["score_numeric"] for r in rated]
    return {"paper_id": str(data["id"]), "title": data.get("title", ""), "abstract": data.get("abstract", ""),
            "source_conference": data.get("conference"), "accepted": data["accepted"],
            "reviews": rated, "meta_reviews": meta, "individual_scores": scores,
            "mean_score": sum(scores) / len(scores) if scores else None,
            "duplicate_entries_removed": duplicates, "unrated_discussion_entries": unrated,
            "all_high_rejected": bool(scores) and not data["accepted"] and all(x >= 8 for x in scores),
            "score_stage": "PeerRead extraction snapshot; initial/final stage not reconstructable from normalized fields",
            "manuscript_version_status": "unverified; source histories do not certify initial pre-review PDF",
            "histories_present": bool(data.get("histories")), "evaluation_ready": False}


def band(score):
    return "low" if score < 4 else ("middle" if score < 7 else "high")


def select_group(records, group, seed):
    rng = random.Random(seed)
    pools = {key: [] for key in ("low", "middle_boundary", "middle_other", "high")}
    excluded = []
    for record in sorted(records, key=lambda r: r["paper_id"]):
        if record["mean_score"] is None:
            excluded.append({"paper_id": record["paper_id"], "reason": "no_recommendation_bearing_review"})
            continue
        if group == "train" and record["all_high_rejected"]:
            excluded.append({"paper_id": record["paper_id"], "reason": "unanimous_high_score_rejected"})
            continue
        key = band(record["mean_score"])
        if key == "middle":
            key = "middle_boundary" if 5 <= record["mean_score"] <= 6 else "middle_other"
        pools[key].append(record)
    quotas = {"low": 2, "middle_boundary": 1, "middle_other": 1, "high": 2}
    selected, strata = [], {}
    for key, quota in quotas.items():
        pool = pools[key]
        take = min(quota, len(pool))
        chosen = rng.sample(pool, take)
        strata[key] = {"N": len(pool), "requested": quota, "n": take, "shortfall": quota - take,
                       "selection_probability_conditional_on_frame": take / len(pool) if pool else None}
        for record in chosen:
            selected.append(dict(record, group=group, stratum=key,
                                 conditional_selection_probability=take / len(pool),
                                 conditional_inverse_probability_weight=len(pool) / take))
    return selected, {"seed": seed, "strata": strata, "exclusions": excluded,
                      "exclusion_counts": dict(Counter(x["reason"] for x in excluded)),
                      "high_rejected_filter_applied": group == "train"}


def parsed_text(data):
    sections = data.get("metadata", {}).get("sections")
    if not isinstance(sections, list):
        raise ValueError("Parsed manuscript has no section list")
    output = []
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("Malformed manuscript section")
        heading, text = section.get("heading"), section.get("text")
        if isinstance(heading, str) and heading.strip():
            output.append(heading)
        if isinstance(text, str) and text.strip():
            output.append(text)
    result = "\n\n".join(output)
    if len(result) < 500:
        raise ValueError("Parsed manuscript text too short; not substituting abstract")
    return result


def fetch_file(root, path, revision, log, expected_blob=None):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Pin a full source commit")
    if not (path == "README.md" or re.fullmatch(r"data/iclr_2017/(train|dev|test)/(reviews|parsed_pdfs)/\d+(\.pdf)?\.json", path)):
        raise ValueError("Only bounded ICLR 2017 review/text files may be fetched")
    local = root / "raw" / path
    cached = local.exists()
    if cached and not expected_blob:
        raise ValueError("Cached source needs an expected Git blob; unverified cache refused")
    url = RAW_HOST + revision + "/" + path
    if cached:
        raw = local.read_bytes()
    else:
        opener = build_opener(ProxyHandler({}), NoRedirect())
        with opener.open(Request(url, headers={"User-Agent": "JournalReviewPreferences-public-dataset-research"}), timeout=30) as response:
            raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Source file exceeds 2 MB; stopped")
    if expected_blob:
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != expected_blob:
            raise ValueError("Source Git blob checksum mismatch")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(raw)
    log.append({"url": url, "local_file": str(local.relative_to(root)), "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "git_blob_verified": bool(expected_blob), "cached": cached})
    write_json(root / "pilot_download_log.json", log)
    if not cached:
        time.sleep(1.1)
    return json.loads(raw) if path.endswith(".json") else raw.decode("utf-8")


def build_pilot(root, allow_network=False, seed=20261023):
    if allow_network is not True:
        raise ValueError("Explicit --allow-public-network required; no automatic network")
    root = Path(root)
    if (root / "pilot_manifest.json").exists():
        raise ValueError("Pilot manifest exists; no overwrite or automatic resume")
    index = read_json(root / "source_index.json")
    revision = index["revision"]
    log, frame, selected, summaries = [], {}, [], {}
    # Fixed metadata caps are chosen before reading scores; never expand adaptively.
    caps = {"train": 48, "validation": 24, "test": 24}
    manifest = {"schema_version": 1, "source": REPOSITORY, "source_revision": revision,
                "split_mode": "same_year_paper_group", "chronological_deployment_simulation": False,
                "seed": seed, "frame_caps": caps, "status": "in_progress", "groups": {}, "selected": [],
                "evaluation_ready_count": 0, "model_calls": 0, "preference_induction_implemented": False}
    try:
        for group, source_split in SPLITS.items():
            listing = index["paths"]["data/iclr_2017/" + source_split + "/reviews"]
            rng = random.Random(seed + list(SPLITS).index(group))
            candidates = rng.sample(sorted(listing, key=lambda x: x["name"]), min(caps[group], len(listing)))
            records = []
            for entry in candidates:
                data = fetch_file(root, entry["path"], revision, log, entry.get("sha"))
                record = parse_review_file(data)
                record["source_review_path"] = entry["path"]
                records.append(record)
            frame[group] = records
            chosen, info = select_group(records, group, seed + 10 + list(SPLITS).index(group))
            info.update({"source_partition": source_split, "source_partition_total": len(listing),
                         "metadata_frame_count": len(records), "frame_paper_ids": [r["paper_id"] for r in records]})
            summaries[group] = info
            selected.extend(chosen)
            print(f"Parsed {group} metadata frame: {len(records)}; selected {len(chosen)}", flush=True)
        ids = [r["paper_id"] for r in selected]
        if len(ids) != len(set(ids)):
            raise ValueError("Same manuscript occurs across source partitions")
        text_hashes = set()
        for record in selected:
            source_split = SPLITS[record["group"]]
            path = f"data/iclr_2017/{source_split}/parsed_pdfs/{record['paper_id']}.pdf.json"
            text_listing = index["paths"].get(f"data/iclr_2017/{source_split}/parsed_pdfs", [])
            expected_text_blob = next((x.get("sha") for x in text_listing if x["path"] == path), None)
            raw_text = fetch_file(root, path, revision, log, expected_text_blob)
            text = parsed_text(raw_text)
            text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_sha in text_hashes:
                raise ValueError("Duplicate manuscript text across selected paper IDs")
            text_hashes.add(text_sha)
            record.update({"text": text, "source_text_path": path, "text_sha256": text_sha,
                           "text_characters": len(text), "manuscript_year": 2017,
                           "source_pdf_or_version_timestamp": None, "public_availability_time": None})
        output = root / "pilot"
        output.mkdir(exist_ok=False)
        for group in SPLITS:
            records = [r for r in selected if r["group"] == group]
            # Training records may contain labels; held-out model inputs do not.
            inputs = [{k: r[k] for k in ("paper_id", "title", "text", "text_sha256", "manuscript_version_status")} for r in records]
            write_json(output / (group + ".inputs.json"), inputs)
            labels = [{k: r[k] for k in ("paper_id", "accepted", "reviews", "meta_reviews", "individual_scores", "mean_score", "score_stage")} for r in records]
            write_json(output / (group + ".labels.json"), labels)
        write_json(output / "training_examples.json", [r for r in selected if r["group"] == "train"])
        frame_meta = {g: [{k: r[k] for k in ("paper_id", "individual_scores", "mean_score", "accepted", "all_high_rejected", "duplicate_entries_removed", "score_stage")} for r in rows] for g, rows in frame.items()}
        write_json(root / "sampling_frame.json", frame_meta)
        partition = {g: [r["paper_id"] for r in selected if r["group"] == g] for g in SPLITS}
        write_json(output / "partition.json", {"split_mode": "same_year_paper_group", "ids": partition,
                                              "note": "PeerRead original paper partitions preserved; score-stratified subsampling, no deployment-time chronology",
                                              "partition_sha256": fingerprint(partition)})
        manifest.update({"status": "paired_ingestion_complete_version_unverified", "groups": summaries,
                         "selected": [{k: r[k] for k in ("paper_id", "group", "stratum", "conditional_selection_probability", "accepted", "individual_scores", "mean_score", "source_review_path", "source_text_path", "text_sha256", "text_characters")} for r in selected],
                         "partition_sha256": fingerprint(partition), "paired_manuscript_count": len(selected),
                         "unique_scored_reviews_in_pilot": sum(len(r["reviews"]) for r in selected),
                         "accepted_in_pilot": sum(r["accepted"] for r in selected),
                         "rejected_in_pilot": sum(not r["accepted"] for r in selected),
                         "version_caveat": "No per-paper pre-review PDF/version time certified; histories empty or insufficient. Not evaluation ready.",
                         "scale": {"raw_range": [1, 10], "normalization": "none", "bands": [1, 4, 7, 10],
                                   "high_filter_numeric_threshold": 8, "high_category_semantics_verified": False,
                                   "threshold_caveat": "Conservative upper-score pilot definition, not a verified 2017 official category boundary"},
                         "license_caveat": "Official repository describes research use; no root LICENSE verified. Article/review rights are not replaced. Local inspection only, no redistribution or MIT relicensing.",
                         "pretraining_contamination": "Old public papers/reviews may already be in real LLM pretraining; held-out IDs alone do not prevent this.",
                         "evaluation_claim": "No model experiment, rule learning, acceptance probability or improvement claim. Small stratified pilot is not representative."})
    except Exception as exc:
        manifest.update({"status": "stopped_error", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        manifest["downloads"] = log
        write_json(root / "pilot_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bounded author-released PeerRead paired-data pilot; no model inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--allow-public-network", action="store_true")
    args = parser.parse_args()
    result = build_pilot(args.data_dir, args.allow_public_network)
    print(json.dumps({k: result[k] for k in ("status", "paired_manuscript_count", "unique_scored_reviews_in_pilot", "accepted_in_pilot", "rejected_in_pilot", "evaluation_ready_count")}, indent=2))
