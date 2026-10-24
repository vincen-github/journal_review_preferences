"""Public-only OpenReview API v2 import; separate from model inference.

Authored implementation, no OpenReview SDK dependency. No authentication,
profiles, identity lookup, retries, proxy use, or fallback endpoints.
"""

import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .data import fingerprint, read_json, write_json

API = "https://api2.openreview.net"
TERMS = "https://openreview.net/legal/terms"
DOCS = "https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
TIMES = ("cdate", "tcdate", "mdate", "tmdate", "pdate", "odate")
REVIEW_FIELDS = ("title", "review", "summary", "summary_of_the_paper", "strengths", "weaknesses",
                 "questions", "limitations", "soundness", "presentation", "contribution",
                 "rating", "recommendation", "confidence", "justification", "decision", "metareview", "meta_review")


def value(field):
    if isinstance(field, dict):
        # A public note may contain fields with narrower readership.
        if field.get("readers") and "everyone" not in field["readers"]:
            return None
        return field.get("value")
    return field


def content(note, key, default=None):
    result = value(note.get("content", {}).get(key))
    return default if result is None else result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def iso_ms(milliseconds):
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float)) or not math.isfinite(milliseconds):
        return None
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def creation_ms(note):
    values = [note.get(k) for k in ("cdate", "tcdate") if isinstance(note.get(k), (int, float))]
    return min(values) if values else None


def conservative_available_ms(note):
    values = [note.get(k) for k in TIMES if isinstance(note.get(k), (int, float))]
    return max(values) if values else None


def public(note):
    return isinstance(note, dict) and "everyone" in note.get("readers", []) and not note.get("ddate")


def invitations(note):
    result = note.get("invitations", [])
    if isinstance(result, str):
        result = [result]
    if isinstance(note.get("invitation"), str):
        result = list(result) + [note["invitation"]]
    return [i for i in result if isinstance(i, str)]


def has_stage(note, stage):
    return any(i.endswith("/-/" + stage) for i in invitations(note))


def numeric_rating(raw):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) else None
    if isinstance(raw, str):
        match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?::|$)", raw)
        return float(match.group(1)) if match else None
    return None


class SourceAccessDenied(RuntimeError):
    def __init__(self, url, status):
        self.url, self.status = url, status
        super().__init__(f"OpenReview access stopped: HTTP {status} at {url}; no retry or fallback")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise SourceAccessDenied(req.full_url, code)


class PublicOpenReviewClient:
    def __init__(self, allow_network=False, acknowledge_terms=False, interval=1.5, max_requests=80):
        if allow_network is not True or acknowledge_terms is not True:
            raise ValueError("Public data import requires --allow-public-network and --ack-source-terms")
        if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 1:
            raise ValueError("Request interval must be at least one second")
        if not isinstance(max_requests, int) or isinstance(max_requests, bool) or not 1 <= max_requests <= 150:
            raise ValueError("Request budget must be within 1..150")
        self.interval, self.max_requests = interval, max_requests
        self.log, self.last_request, self.stopped = [], None, False
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def get(self, path, params):
        if self.stopped:
            raise RuntimeError("This import client has stopped; no further requests are allowed")
        if path not in ("/notes", "/notes/edits", "/groups", "/invitations"):
            raise ValueError("Only public note, edit, venue-group and form GET endpoints are allowed")
        if len(self.log) >= self.max_requests:
            raise ValueError("Public import request budget exhausted")
        if self.last_request is not None:
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
        url = API + path + "?" + urlencode(params)
        entry = {"url": url, "requested_at": utc_now(), "method": "GET"}
        self.log.append(entry)
        self.last_request = time.monotonic()
        try:
            request = Request(url, headers={"User-Agent": "JournalReviewPreferences/0.1 public-local-research-pilot",
                                            "Accept": "application/json"}, method="GET")
            with self.opener.open(request, timeout=30) as response:
                data = response.read(20_000_001)
                entry["status"] = response.status
            if len(data) > 20_000_000:
                raise ValueError("Public metadata response exceeded 20 MB; stopped")
            entry["response_sha256"] = hashlib.sha256(data).hexdigest()
            entry["bytes"] = len(data)
            parsed = json.loads(data)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object from OpenReview")
            return parsed
        except HTTPError as exc:
            self.stopped = True
            entry["status"] = exc.code
            # Every HTTP error stops this run, including denial/rate limiting.
            raise SourceAccessDenied(url, exc.code) from None
        except Exception:
            self.stopped = True
            entry.setdefault("status", "transport_or_parse_error")
            raise


def parse_venue_group(group):
    """Project only configuration; never retain members, profiles or recruitment data."""
    keep = ("submission_id", "submission_name", "submission_venue_id", "withdrawn_venue_id",
            "rejected_venue_id", "desk_rejected_venue_id", "review_name", "review_rating",
            "review_confidence", "meta_review_name", "decision_name")
    return {"id": group["id"], "configuration": {k: content(group, k) for k in keep if content(group, k) is not None}}


def find_rating_enum(form, score_field):
    """Find the official rating enum; never infer a scale from observed scores."""
    found = []

    def visit(node):
        if isinstance(node, dict):
            field = node.get(score_field)
            if isinstance(field, dict):
                spec = field.get("value", field)
                if isinstance(spec, dict):
                    spec = spec.get("param", spec)
                    if isinstance(spec, dict) and isinstance(spec.get("enum"), list):
                        found.append(spec["enum"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(form)
    unique = {json.dumps(x, sort_keys=True) for x in found}
    if len(unique) != 1:
        raise ValueError("Cannot establish one unambiguous official recommendation scale")
    labels = json.loads(next(iter(unique)))
    categories = [{"value": numeric_rating(label), "label": label} for label in labels]
    if any(c["value"] is None or not isinstance(c["label"], str) for c in categories):
        raise ValueError("Official categories must include numeric values and descriptive labels")
    if len({c["value"] for c in categories}) != len(categories):
        raise ValueError("Duplicate numeric recommendation categories")
    high = [c["value"] for c in categories if re.search(r"\baccept\b", c["label"].lower())
            and not any(w in c["label"].lower() for w in ("reject", "below", "borderline", "marginal", "weak", "threshold", "not accept"))]
    boundary = [c["value"] for c in categories if any(w in c["label"].lower() for w in ("borderline", "marginal", "weak accept", "weak reject"))]
    if not high or not boundary:
        raise ValueError("Cannot map explicit above-borderline acceptance and boundary categories; manual schema review required")
    numbers = [c["value"] for c in categories]
    low, high_limit = min(numbers), max(numbers)
    if low == high_limit:
        raise ValueError("Official score scale must vary")
    return {"categories": categories, "raw_min": low, "raw_max": high_limit,
            "high_values": high, "boundary_values": boundary,
            "band_edges": [low, low + (high_limit - low) / 3, low + 2 * (high_limit - low) / 3, high_limit],
            "normalization": "none; each venue-year is sampled separately",
            "high_rule": "Official labels containing accept, excluding reject/below/borderline/marginal/weak/threshold"}


def project_note(note, fields=REVIEW_FIELDS):
    keep = {key: note.get(key) for key in ("id", "forum", "number", "license") if note.get(key) is not None}
    keep.update({k: note[k] for k in TIMES if note.get(k) is not None})
    keep["invitations"] = invitations(note)
    keep["content"] = {k: content(note, k) for k in fields if content(note, k) is not None}
    keep["url"] = "https://openreview.net/forum?id=" + str(note.get("forum", note.get("id", ""))) + "&noteId=" + str(note.get("id", ""))
    keep["available_at_conservative"] = iso_ms(conservative_available_ms(note))
    return keep


def classify_status(submission, decision_notes, venue):
    venue_id = str(content(submission, "venueid", ""))
    if "withdraw" in venue_id.lower():
        return "withdrawn"
    if "desk_reject" in venue_id.lower():
        return "desk_rejected"
    if venue_id == venue:
        return "ordinary_accepted"
    if venue_id == venue + "/Rejected_Submission":
        return "ordinary_rejected"
    labels = [str(content(n, "decision", "")).lower() for n in decision_notes]
    if any("desk reject" in label for label in labels):
        return "desk_rejected"
    if any("withdraw" in label for label in labels):
        return "withdrawn"
    accepted = any("accept" in label for label in labels)
    rejected = any("reject" in label for label in labels)
    if accepted != rejected:
        return "ordinary_accepted" if accepted else "ordinary_rejected"
    return "unknown_or_missing_decision"


def parse_forum(submission, replies, venue, scale, score_field="rating", review_stage="Official_Review"):
    if not public(submission):
        raise ValueError("Submission is not explicitly public")
    forum = submission["id"]
    if not SAFE_ID.fullmatch(forum):
        raise ValueError("Unsupported forum identifier")
    # API details may return more than the direct replies; restrict to this forum.
    notes = [n for n in replies if public(n) and n.get("forum") == forum and n.get("replyto") == forum]
    review_notes = [n for n in notes if has_stage(n, review_stage)]
    if len({n["id"] for n in review_notes}) != len(review_notes):
        raise ValueError("Duplicate official review IDs in forum")
    decisions = [n for n in notes if has_stage(n, "Decision")]
    metas = [n for n in notes if has_stage(n, "Meta_Review")]
    valid_values = {c["value"] for c in scale["categories"]}
    reviews, missing = [], 0
    for note in sorted(review_notes, key=lambda n: n["id"]):
        raw = content(note, score_field)
        score = numeric_rating(raw)
        if score not in valid_values:
            score, missing = None, missing + 1
        projected = project_note(note)
        projected.update({"score_raw": raw, "score_numeric": score,
                          "confidence_raw": content(note, "confidence"),
                          "confidence_numeric": numeric_rating(content(note, "confidence"))})
        reviews.append(projected)
    observed = [r["score_numeric"] for r in reviews if r["score_numeric"] is not None]
    mean = sum(observed) / len(observed) if observed else None
    status = classify_status(submission, decisions, venue)
    return {"forum_id": forum, "venue_id": venue, "submission": project_note(submission, ("title", "abstract", "pdf", "venueid", "venue")),
            "decision_status": status, "official_reviews": reviews,
            "meta_reviews": [project_note(n) for n in metas], "decisions": [project_note(n) for n in decisions],
            "review_stage": "latest_public_" + review_stage + "_snapshot_at_download; not initial pre-discussion ratings",
            "mean_official_score_observed": mean, "review_count": len(reviews), "missing_score_count": missing,
            "mean_definition": "Unweighted mean of numeric official overall recommendations; missing excluded from mean and flagged",
            "all_scores_present": bool(reviews) and missing == 0,
            "unanimous_high_score_rejected": status == "ordinary_rejected" and bool(reviews) and missing == 0
                and all(r["score_numeric"] in scale["high_values"] for r in reviews),
            "evaluation_ready": False, "evaluation_exclusion_reasons": ["initial_pre_review_fulltext_not_imported"]}


def score_band(mean, scale):
    low, a, b, high = scale["band_edges"]
    if not low <= mean <= high:
        raise ValueError("Mean score outside official scale")
    return "low" if mean < a else ("middle" if mean < b else "high")


def select_pilot(records, scale, seed=20261023, per_band=3):
    """Pilot-only filters. General parse_forum retains discordant/withdrawn records."""
    if per_band != 3:
        raise ValueError("This bounded pilot selects three per band (nine per venue-year)")
    rng = random.Random(seed)
    pools = {key: [] for key in ("low", "middle_boundary", "middle_other", "high")}
    exclusions = []
    for record in sorted(records, key=lambda r: r["forum_id"]):
        reason = None
        if record["decision_status"] not in ("ordinary_accepted", "ordinary_rejected"):
            reason = record["decision_status"]
        elif not record["all_scores_present"]:
            reason = "missing_official_reviews_or_scores"
        elif not record["decisions"]:
            reason = "missing_official_decision_note"
        elif record["unanimous_high_score_rejected"]:
            reason = "unanimous_high_score_rejected"
        if reason:
            exclusions.append({"forum_id": record["forum_id"], "reason": reason})
            continue
        band = score_band(record["mean_official_score_observed"], scale)
        key = band
        if band == "middle":
            boundary = scale["boundary_values"]
            is_boundary = min(boundary) <= record["mean_official_score_observed"] <= max(boundary)
            key = "middle_boundary" if is_boundary else "middle_other"
        pools[key].append(record)
    target = {"low": 3, "middle_boundary": 1, "middle_other": 2, "high": 3}
    selected, strata = [], {}
    for key, count in target.items():
        pool = pools[key]
        take = min(count, len(pool))
        chosen = rng.sample(pool, take)
        strata[key] = {"frame_eligible": len(pool), "requested": count, "selected": take,
                       "shortfall": count - take, "inclusion_probability": take / len(pool) if pool else None}
        for record in chosen:
            selected.append({"forum_id": record["forum_id"], "band": "middle" if key.startswith("middle") else key,
                             "sub_stratum": key, "mean_raw_score": record["mean_official_score_observed"],
                             "decision_status": record["decision_status"],
                             "selection_probability_conditional_on_eligible_frame": take / len(pool),
                             "inverse_probability_weight_within_frame": len(pool) / take})
    return selected, {"seed": seed, "method": "Fixed-seed uniform random selection within disjoint venue-year score strata",
                      "strata": strata, "exclusions": exclusions,
                      "exclusion_counts": dict(Counter(e["reason"] for e in exclusions)),
                      "notice": "Score-balanced and exclusion-filtered pilot; not representative prevalence, acceptance probability, or an unbiased test set",
                      "general_importer_filter": False, "pilot_exclusion_filter": "unanimous_high_score_rejected",
                      "boundary_rule": "Middle-band mean between min and max official borderline/marginal/weak-accept categories, inclusive",
                      "shortfall_policy": "Do not relax filters, move papers across strata, or replace missing strata with high-score papers"}


def initial_version(submission, edits, reviews):
    """Only immutable PDF paths in public edits can certify a pre-review version.

    No use of the mutable /pdf?id=forum link and no fallback to current text.
    Full text is intentionally not downloaded by this metadata pilot.
    """
    review_times = [creation_ms(r) for r in reviews]
    if not review_times or any(t is None for t in review_times):
        return {"status": "unavailable_or_unverified", "initial_pdf_url": None,
                "reason": "Every public official review needs a creation timestamp to verify pre-review ordering",
                "evaluation_ready": False}
    first_review = min(review_times)
    candidates = []
    for edit in edits:
        if not public(edit) or edit.get("note", {}).get("id") != submission["id"]:
            continue
        note = edit.get("note", {})
        pdf = content(note, "pdf")
        when = conservative_available_ms(edit)
        if not isinstance(pdf, str) or not re.fullmatch(r"/pdf/[a-fA-F0-9]{40,64}\.pdf", pdf):
            continue
        if when is None or first_review is None or when >= first_review:
            continue
        if any("Camera_Ready" in i or "Revision" in i for i in invitations(edit)):
            continue
        candidates.append((when, edit["id"], pdf))
    if not candidates:
        return {"status": "unavailable_or_unverified", "initial_pdf_url": None,
                "reason": "No public, immutable, non-revision PDF edit proved to precede the earliest official review",
                "evaluation_ready": False}
    when, edit_id, pdf = min(candidates)
    return {"status": "pre_review_immutable_pdf_reference_identified_not_downloaded",
            "edit_id": edit_id, "version_time_conservative": iso_ms(when),
            "first_official_review_created_at": iso_ms(first_review),
            "initial_pdf_url": API + pdf, "pdf_source_path": pdf,
            "evaluation_ready": False, "reason": "Full text not downloaded; historical public visibility and per-article rights still need verification"}


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("Public import profile schema_version must be 1")
    venues = profile.get("venues", [])
    if not 2 <= len(venues) <= 3:
        raise ValueError("Pilot profile must specify two or three completed venue-years")
    ids = []
    for item in venues:
        venue = item.get("venue_id", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/\d{4}/[A-Za-z0-9_.-]+", venue):
            raise ValueError("Unsupported venue ID")
        if item.get("completed") is not True:
            raise ValueError("Venue-year must be explicitly marked completed")
        ids.append(venue)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate venue-years")
    maximum = profile.get("max_frame_records_per_venue", 300)
    if not isinstance(maximum, int) or isinstance(maximum, bool) or not 9 <= maximum <= 1000:
        raise ValueError("Metadata sampling frame cap must be within 9..1000 per venue-year")
    if len(venues) * 9 > 20:
        raise ValueError("This score-balanced profile uses nine per year; choose two years to remain <=20")
    return profile


def fetch_venue_frame(client, item, cap):
    venue = item["venue_id"]
    groups = client.get("/groups", {"id": venue}).get("groups", [])
    if not groups:
        raise ValueError("Public venue configuration missing")
    group = parse_venue_group(groups[0])
    configuration = group["configuration"]
    stage = configuration.get("review_name", "Official_Review")
    score_field = configuration.get("review_rating", "rating")
    form_id = venue + "/-/" + stage
    forms = client.get("/invitations", {"id": form_id}).get("invitations", [])
    if not forms:
        raise ValueError("Official recommendation form unavailable; no guessed scale allowed")
    scale = find_rating_enum(forms[0], score_field)
    scale.update({"source_url": API + "/invitations?" + urlencode({"id": form_id}),
                  "official_form_sha256": fingerprint(forms[0]), "score_field": score_field,
                  "review_stage": stage, "verified_from_public_official_form": True})
    invitation = configuration.get("submission_id", venue + "/-/" + configuration.get("submission_name", "Submission"))
    records, offset, total = [], 0, None
    while offset < cap:
        limit = min(100, cap - offset)
        payload = client.get("/notes", {"invitation": invitation, "details": "directReplies",
                                        "sort": "number:asc", "limit": limit, "offset": offset})
        page = payload.get("notes", [])
        total = payload.get("count", total)
        for note in page:
            if not public(note):
                continue
            replies = note.get("details", {}).get("directReplies", [])
            records.append(parse_forum(note, replies, venue, scale, score_field, stage))
        offset += len(page)
        if len(page) < limit or (isinstance(total, int) and offset >= total):
            break
    if len({r["forum_id"] for r in records}) != len(records):
        raise ValueError("Duplicate forum across pages; sampling frame is not stable")
    return records, {"venue_id": venue, "group": group, "scale": scale,
                     "frame_order": "number:asc", "frame_cap": cap, "api_count_reported": total,
                     "notes_scanned": offset, "public_records_parsed": len(records),
                     "frame_is_complete": isinstance(total, int) and offset >= total,
                     "status_counts": dict(Counter(r["decision_status"] for r in records))}


def import_public_pilot(profile, output, allow_network=False, acknowledge_terms=False):
    profile = validate_profile(profile)
    output = Path(output)
    if output.exists():
        raise ValueError("Public import output must be a new directory")
    client = PublicOpenReviewClient(allow_network, acknowledge_terms)
    output.mkdir(parents=True)
    # This directory must never be included in a source-code publication.
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    manifest = {"schema_version": 1, "started_at": utc_now(), "status": "in_progress", "source": API,
                "terms_url": TERMS, "docs_url": DOCS, "profile": profile,
                "data_license": "Source licenses retained; no corpus relicensing or redistribution",
                "review_stage": "Latest public Official_Review note at retrieval, including public revisions; not initial ratings",
                "network_purpose": "Explicit public data import only; never called by offline model inference",
                "venues": [], "selected": [], "evaluation_ready_count": 0,
                "evaluation_notice": "No current/camera-ready full text is substituted; this metadata pilot is excluded from backtest inputs"}
    all_records = []
    try:
        for item in profile["venues"]:
            records, info = fetch_venue_frame(client, item, profile.get("max_frame_records_per_venue", 300))
            selected, sampling = select_pilot(records, info["scale"], profile.get("seed", 20261023))
            info["sampling"] = sampling
            manifest["venues"].append(info)
            by_id = {r["forum_id"]: r for r in records}
            # Save projected frame metadata, including exclusion flags; not full unselected review text.
            excluded = {e["forum_id"]: e["reason"] for e in sampling["exclusions"]}
            frame = []
            for record in records:
                row = {k: record[k] for k in ("forum_id", "venue_id", "decision_status", "mean_official_score_observed",
                                              "review_count", "missing_score_count", "all_scores_present", "unanimous_high_score_rejected")}
                row["pilot_exclusion_reason"] = excluded.get(record["forum_id"])
                row["individual_official_scores"] = [
                    {k: review[k] for k in ("id", "score_raw", "score_numeric", "confidence_raw", "available_at_conservative")}
                    for review in record["official_reviews"]]
                frame.append(row)
            slug = item["venue_id"].replace("/", "_")
            write_json(output / (slug + ".frame.json"), frame)
            for selection in selected:
                record = by_id[selection["forum_id"]]
                note = {"id": record["forum_id"]}
                edits_payload = client.get("/notes/edits", {"note.id": note["id"], "limit": 100, "sort": "tcdate:asc"})
                edits = edits_payload.get("edits", [])
                record["initial_version"] = initial_version(note, edits, record["official_reviews"])
                record["selection"] = selection
                record["content_sha256"] = fingerprint(record)
                record_path = slug + "." + note["id"] + ".json"
                write_json(output / record_path, record)
                selection.update({"venue_id": item["venue_id"], "record_file": record_path,
                                  "record_sha256": hashlib.sha256((output / record_path).read_bytes()).hexdigest(),
                                  "source_url": "https://openreview.net/forum?id=" + note["id"],
                                  "article_license": record["submission"].get("license", "unknown; do not infer"),
                                  "initial_version_status": record["initial_version"]["status"]})
                manifest["selected"].append(selection)
                all_records.append(record)
        manifest["status"] = "completed" if len(all_records) == 18 else "completed_with_stratum_shortfalls"
    except SourceAccessDenied as exc:
        manifest["status"] = "stopped_source_denied"
        manifest["blocked_action"] = {"method": "GET", "url": exc.url, "http_status": exc.status,
                                      "outcome": "denied", "retried": False, "fallback_used": False}
    except Exception as exc:
        manifest["status"] = "stopped_error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
    finally:
        manifest["finished_at"] = utc_now()
        manifest["request_log"] = client.log
        manifest["selected_count"] = len(all_records)
        manifest["downloaded_pdfs"] = 0
        manifest["actual_fulltext_records"] = 0
        manifest["accepted_count"] = sum(r["decision_status"] == "ordinary_accepted" for r in all_records)
        manifest["rejected_count"] = sum(r["decision_status"] == "ordinary_rejected" for r in all_records)
        write_json(output / "manifest.json", manifest)
    return manifest
