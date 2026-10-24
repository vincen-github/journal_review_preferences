"""Exact cache reuse, merged scoring receipts, and fixed-raw calibration curves."""

import json
import hashlib
from pathlib import Path

from .data import fingerprint, read_json, write_json
from .models import validate_response
from .prompts import load_template, normalize_response, render_instructions
from .score_calibration import fit_state, save_state
from .evidence_format import restore_source_whitespace, quarantine_unmatched_quotes


def validate_partition(partition, plan):
    ids = partition["ids"]
    flat = sum((ids[g] for g in ("train", "validation", "test")), [])
    if (fingerprint(ids) != partition["partition_sha256"] or plan["partition_sha256"] != partition["partition_sha256"]
            or len(flat) != len(set(flat)) or [len(ids[g]) for g in ("train", "validation", "test")] != [60, 20, 20]
            or any(partition["nested_train_ids"][str(n)] != ids["train"][:n] for n in (20, 40, 60))
            or not set(partition["mandatory_ids"]) <= set(ids["train"][:20])):
        raise ValueError("Changed, overlapping or non-nested fixed partition")


def lock_scoring_inputs(root, old_root, template_path):
    """Recheck original cache, then bind analysis inputs without changing the plan."""
    root, template = Path(root), load_template(template_path)
    path = root / "scoring_integrity.json"
    if path.exists():
        raise ValueError("Scoring integrity receipt already frozen")
    plan, partition = read_json(root / "scoring_plan.json"), read_json(root / "partition.json")
    validate_partition(partition, plan)
    cached = read_json(root / "cached_predictions.json")
    ledger = read_json(root / "cache_ledger.json")
    if cached:
        inputs = read_json(root / "cohort/train.inputs.json")
        checked, checked_ledger = compatible_cache(old_root, inputs, template, plan["settings"])
        if cached != list(checked.values()) or ledger != checked_ledger:
            raise ValueError("Prepared cache differs from its original validated responses")
    receipt = {"plan_sha256": fingerprint(plan), "partition_sha256": fingerprint(partition),
               "cached_predictions_sha256": fingerprint(cached), "cache_ledger_sha256": fingerprint(ledger),
               "development_sha256": {g + "." + suffix: fingerprint(read_json(root / "cohort" / (g + "." + suffix + ".json")))
                    for g in ("train", "validation") for suffix in ("inputs", "labels")}}
    write_json(path, receipt)
    return receipt


def verify_integrity(root):
    root = Path(root)
    receipt = read_json(root / "scoring_integrity.json")
    for filename, key in (("scoring_plan.json", "plan_sha256"), ("partition.json", "partition_sha256"),
                          ("cached_predictions.json", "cached_predictions_sha256"), ("cache_ledger.json", "cache_ledger_sha256")):
        if fingerprint(read_json(root / filename)) != receipt[key]:
            raise ValueError("Scoring integrity receipt mismatch: " + filename)
    for name, checksum in receipt["development_sha256"].items():
        if fingerprint(read_json(root / "cohort" / (name + ".json"))) != checksum:
            raise ValueError("Development inputs/labels changed after integrity freeze")


def checked_combined(root, group, plan, partition):
    directory = Path(root) / ("combined_" + group)
    manifest, predictions = read_json(directory / "manifest.json"), read_json(directory / "predictions.json")
    if (manifest["status"] != "completed_predictions_frozen_before_label_evaluation"
            or fingerprint(predictions) != manifest["predictions_sha256"] or manifest["settings"] != plan["settings"]
            or manifest["instructions_sha256"] != plan["instructions_sha256"] or manifest["schema_sha256"] != plan["schema_sha256"]
            or [p["paper_id"] for p in predictions] != partition["ids"][group]):
        raise ValueError("Combined development scores/configuration/IDs changed")
    return predictions


def request_fingerprint(paper, template):
    payload = {k: paper[k] for k in ("paper_id", "title", "text")}
    return fingerprint(render_instructions(template) + "\nMANUSCRIPT_DATA_JSON:\n" + json.dumps(payload, ensure_ascii=False))


def compatible_cache(old_root, inputs, template, settings):
    by_id = {p["paper_id"]: p for p in inputs}
    cached, ledger = {}, []
    for directory in ("sol_iclr2017_train_20261023", "sol_iclr2017_validation_20261023", "sol_iclr2017_test_20261023"):
        root = Path(old_root) / directory
        manifest, predictions = read_json(root / "manifest.json"), read_json(root / "predictions.json")
        if (manifest["status"] != "completed_predictions_frozen_before_label_evaluation"
                or fingerprint(predictions) != manifest["predictions_sha256"]
                or manifest["settings"] != settings
                or manifest["instructions_sha256"] != fingerprint(render_instructions(template))
                or manifest["schema_sha256"] != fingerprint(template["output_schema"])
                or manifest["prompt_template_sha256"] != fingerprint(template)):
            raise ValueError("Prior scoring cache differs in model/settings/prompt/schema or frozen predictions")
        for prediction in predictions:
            paper_id = prediction["paper_id"]
            if paper_id not in by_id:
                continue
            run = next(r for r in manifest["runs"] if r["paper_id"] == paper_id)
            paper = by_id[paper_id]
            if hashlib.sha256(paper["text"].encode("utf-8")).hexdigest() != paper["text_sha256"]:
                raise ValueError("Current manuscript text checksum mismatch")
            if not run["reported_sol_slug"] or run["tool_events"] or request_fingerprint(paper, template) != prediction["request_sha256"]:
                raise ValueError("Prior per-paper cache provenance mismatch")
            response = json.loads((root / ("response_" + paper_id + ".json.txt")).read_text(encoding="utf-8"))
            if fingerprint(response) != run["response_sha256"]:
                raise ValueError("Prior raw response changed")
            checked = validate_response(normalize_response(response, template), {"paper": paper, "rubric": {"score_range": template["score_range"]}})
            if any(checked[k] != prediction[k] for k in checked):
                raise ValueError("Prior prediction differs from validated raw response")
            if paper_id in cached:
                raise ValueError("Duplicate prior prediction")
            cached[paper_id] = dict(checked, paper_id=paper_id, request_sha256=prediction["request_sha256"], arm=template["template_id"])
            ledger.append({"paper_id": paper_id, "source_run": str(root), "source_manifest_sha256": fingerprint(manifest),
                           "request_sha256": prediction["request_sha256"], "text_sha256": paper["text_sha256"],
                           "response_sha256": run["response_sha256"], "new_model_call": False})
    return cached, ledger


def prepare_scoring(root, old_root, template_path, settings):
    root = Path(root)
    if (root / "scoring_plan.json").exists():
        raise ValueError("Scoring plan already frozen; do not overwrite")
    partition, template = read_json(root / "partition.json"), load_template(template_path)
    protocol = read_json(root / "pre_ingestion_protocol.json")
    if fingerprint(template) != protocol["template_sha256"]:
        raise ValueError("Historical prompt differs from protocol")
    groups = {g: read_json(root / "cohort" / (g + ".inputs.json")) for g in ("train", "validation", "test")}
    cached, ledger = compatible_cache(old_root, sum(groups.values(), []), template, settings)
    if set(cached) != set(partition["mandatory_ids"]) or not set(cached) <= set(partition["ids"]["train"]):
        raise ValueError("Only all 18 previously explored TRAIN papers may use the old cache")
    batches = root / "batch_inputs"
    batches.mkdir(exist_ok=False)
    train = [p for p in groups["train"] if p["paper_id"] not in cached]
    jobs = [("train_a", "train", train[:len(train)//2]), ("train_b", "train", train[len(train)//2:]),
            ("validation", "validation", groups["validation"]),
            ("test_a", "test", groups["test"][:len(groups["test"])//2]),
            ("test_b", "test", groups["test"][len(groups["test"])//2:])]
    plan_jobs = []
    for name, group, inputs in jobs:
        write_json(batches / (name + ".json"), inputs)
        plan_jobs.append({"name": name, "group": group, "count": len(inputs), "ids": [p["paper_id"] for p in inputs],
                          "input_sha256": fingerprint(inputs), "run_directory": "model_runs/" + name})
    write_json(root / "cached_predictions.json", list(cached.values()))
    write_json(root / "cache_ledger.json", ledger)
    plan = {"schema_version": 1, "partition_sha256": partition["partition_sha256"], "template_sha256": fingerprint(template),
            "instructions_sha256": fingerprint(render_instructions(template)), "schema_sha256": fingerprint(template["output_schema"]),
            "settings": settings, "jobs": plan_jobs, "cache_reused": len(cached),
            "new_calls_planned": sum(j["count"] for j in plan_jobs), "test_requires_frozen_selection": True}
    write_json(root / "scoring_plan.json", plan)
    return plan


def combine_group(root, group, template_path):
    root, template = Path(root), load_template(template_path)
    if group == "test":
        from .expanded_evaluation import verify_test_release
        verify_test_release(root)
    plan = read_json(root / "scoring_plan.json")
    partition = read_json(root / "partition.json")
    validate_partition(partition, plan)
    verify_integrity(root)
    if fingerprint(template) != plan["template_sha256"]:
        raise ValueError("Prompt changed after scoring plan freeze")
    inputs = read_json(root / "cohort" / (group + ".inputs.json"))
    by_id, results, origins = {p["paper_id"]: p for p in inputs}, {}, []
    for prediction in read_json(root / "cached_predictions.json"):
        if prediction["paper_id"] in by_id:
            results[prediction["paper_id"]] = prediction
    for job in [j for j in plan["jobs"] if j["group"] == group]:
        directory = root / job["run_directory"]
        manifest, predictions = read_json(directory / "manifest.json"), read_json(directory / "predictions.json")
        if (manifest["status"] != "completed_predictions_frozen_before_label_evaluation"
                or fingerprint(predictions) != manifest["predictions_sha256"]
                or manifest["settings"] != plan["settings"] or manifest["instructions_sha256"] != plan["instructions_sha256"]
                or manifest["schema_sha256"] != plan["schema_sha256"] or set(manifest["paper_ids"]) != set(job["ids"])):
            raise ValueError("Scoring job incomplete or configuration changed")
        runs = {r["paper_id"]: r for r in manifest["runs"]}
        if (len(runs) != len(manifest["runs"]) or set(runs) != set(job["ids"])
                or len(predictions) != len(job["ids"]) or fingerprint(read_json(root / "batch_inputs" / (job["name"] + ".json"))) != job["input_sha256"]
                or manifest.get("prompt_template_sha256") != plan["template_sha256"]):
            raise ValueError("Scoring runs do not correspond exactly to the frozen job")
        for prediction in predictions:
            paper_id = prediction["paper_id"]
            if paper_id in results or paper_id not in by_id or prediction["request_sha256"] != request_fingerprint(by_id[paper_id], template):
                raise ValueError("Duplicate, foreign or altered per-paper prediction")
            record = runs[paper_id]
            if (record["status"] != "completed" or record["exit_code"] != 0 or record["tool_events"]
                    or not record["reported_sol_slug"] or record["request_sha256"] != prediction["request_sha256"]):
                raise ValueError("Incomplete, tool-using or incompatible per-paper scoring receipt")
            response = json.loads((directory / ("response_" + paper_id + ".json.txt")).read_text(encoding="utf-8"))
            normalized, repairs = restore_source_whitespace(normalize_response(response, template), by_id[paper_id]["text"])
            if repairs != record.get("evidence_whitespace_repairs", []):
                raise ValueError("Evidence whitespace projection differs from recorded repair")
            normalized, rejected = quarantine_unmatched_quotes(normalized, by_id[paper_id]["text"])
            if rejected != record.get("rejected_evidence", []):
                raise ValueError("Quarantined evidence differs from the raw-response audit")
            checked = validate_response(normalized, {"paper": by_id[paper_id], "rubric": {"score_range": template["score_range"]}})
            if fingerprint(response) != record["response_sha256"] or any(checked[k] != prediction[k] for k in checked):
                raise ValueError("Prediction differs from the recorded raw model response")
            results[paper_id] = prediction
        origins.append({"directory": job["run_directory"], "manifest_sha256": fingerprint(manifest), "calls": len(predictions)})
    if set(results) != set(by_id):
        raise ValueError("Incomplete combined prediction group")
    predictions = [results[p["paper_id"]] for p in inputs]
    output = root / ("combined_" + group)
    output.mkdir(exist_ok=False)
    write_json(output / "predictions.json", predictions)
    manifest = {"schema_version": 1, "status": "completed_predictions_frozen_before_label_evaluation", "group": group,
                "settings": plan["settings"], "instructions_sha256": plan["instructions_sha256"], "schema_sha256": plan["schema_sha256"],
                "predictions_sha256": fingerprint(predictions), "paper_ids": [p["paper_id"] for p in inputs],
                "source_runs": origins, "cache_reused": len(predictions) - sum(o["calls"] for o in origins),
                "new_calls": sum(o["calls"] for o in origins), "receipt_kind": "aggregate; not new CLI call events"}
    write_json(output / "manifest.json", manifest)
    return manifest


def fit_learning_curve(root):
    """Read TRAIN/VALIDATION only. No test file is opened or scored here."""
    root = Path(root)
    if (root / "selection.json").exists():
        raise ValueError("Learning-curve selection already frozen")
    partition, plan = read_json(root / "partition.json"), read_json(root / "scoring_plan.json")
    if (root / "test_release.json").exists() or any((root / j["run_directory"]).exists() for j in plan.get("jobs", []) if j["group"] == "test"):
        raise ValueError("Cannot select calibration after test release/scoring")
    validate_partition(partition, plan)
    verify_integrity(root)
    predictions = {p["paper_id"]: p for p in checked_combined(root, "train", plan, partition)}
    labels = {p["paper_id"]: p for p in read_json(root / "cohort/train.labels.json")}
    val_predictions = checked_combined(root, "validation", plan, partition)
    val_labels = read_json(root / "cohort/validation.labels.json")
    if (len(labels) != len(partition["ids"]["train"]) or set(labels) != set(partition["ids"]["train"])
            or len(val_labels) != len(partition["ids"]["validation"]) or {p["paper_id"] for p in val_labels} != set(partition["ids"]["validation"])):
        raise ValueError("Development label IDs differ from frozen partition")
    signature = fingerprint({"settings": plan["settings"], "instructions": plan["instructions_sha256"], "schema": plan["schema_sha256"]})
    candidates = []
    for n in (20, 40, 60):
        ids = partition["nested_train_ids"][str(n)]
        state = fit_state([predictions[i] for i in ids], [labels[i] for i in ids], val_predictions, val_labels,
                          provenance={"scoring_signature": signature, "label_target": "ICLR2017 paper mean recommendation 1..10",
                                      "prior_analyst_label_exposure": True, "blind_claim": False, "manuscript_stage": "unverified"})
        save_state(state, root / ("calibration_n" + str(n) + ".json"))
        selected = next(c for c in state["candidate_results"] if c["calibration"] == state["calibration"])
        candidates.append({"n": n, "state_sha256": state["state_sha256"], "selected_calibration": state["calibration"],
                           "validation": selected["validation"], "candidate_results": state["candidate_results"]})
    best_mae = min(c["validation"]["mae"] for c in candidates)
    selected = min((c for c in candidates if c["validation"]["mae"] <= best_mae + 1e-12), key=lambda c: c["n"])
    result = {"schema_version": 1, "frozen": True, "selected_n": selected["n"], "state_sha256": selected["state_sha256"],
              "candidates": candidates, "rule": "Lowest validation MAE; numerical ties within 1e-12 prefer smaller N, then identity within N",
              "note": "Fixed raw scores; sensitivity of external calibration, not base-model learning. Identity is the same effective candidate at all N.",
              "new_test_scoring_started": False, "partition_sha256": partition["partition_sha256"]}
    write_json(root / "selection.json", result)
    return result
