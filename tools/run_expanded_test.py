"""Run a fixed new-test batch only under its previously released selection."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from journal_prefs.data import fingerprint, read_json
from journal_prefs.expanded_evaluation import verify_test_release
from journal_prefs.prompts import load_template
from run_codex_baseline import SETTINGS, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--batch", choices=["test_a", "test_b"], required=True)
    parser.add_argument("--codex", required=True)
    parser.add_argument("--allow-codex-quota", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    _, _, plan, _ = verify_test_release(root)
    template = ROOT / "prompts/iclr2017_official_form.v1.json"
    job = next(j for j in plan["jobs"] if j["name"] == args.batch and j["group"] == "test")
    inputs = root / "batch_inputs" / (args.batch + ".json")
    if (plan["settings"] != SETTINGS or fingerprint(load_template(template)) != plan["template_sha256"]
            or fingerprint(read_json(inputs)) != job["input_sha256"]):
        raise ValueError("Frozen test batch settings/template/input changed")
    run(inputs, root / job["run_directory"], args.codex, args.allow_codex_quota,
        "test", template, root / "scoring_abort.json", repair_evidence_whitespace=True, audit_invalid_evidence=True)
