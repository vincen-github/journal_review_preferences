"""Minimal CLI: no dependency installation or third-party executable code."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import URLError

from .data import load_history, read_json, validate_paper, validate_rubric, write_json
from .engine import review_with_state, run_backtest, save_run
from .models import make_model

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Institution-owned peer-review preferences (prototype)")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run only authored synthetic data and deterministic mock, fully offline")
    demo.add_argument("--out", default="demo_output")
    assessment = commands.add_parser("assess", help="Institution-configured structured assessment; numeric scores are optional")
    assessment.add_argument("--task", required=True, help="Versioned instructions, output schema and optional task metrics")
    assessment.add_argument("--request", required=True, help="Unlabelled target and explicitly selected TRAIN examples")
    assessment.add_argument("--config", default=str(ROOT / "configs" / "offline.json"))
    assessment.add_argument("--allow-cloud", action="store_true")
    assessment.add_argument("--out", required=True, help="NEW local output directory")
    preference_demo = commands.add_parser("preference-demo", help="Synthetic preference-interface demo; not ICAI reproduction or a real experiment")
    preference_demo.add_argument("--out", required=True)
    validate = commands.add_parser("validate", help="Validate institution-owned JSONL history; no model call")
    validate.add_argument("--data", required=True)
    validate.add_argument("--rubric", required=True)
    public_import = commands.add_parser("import-openreview", help="Explicit public API v2 metadata import, separate from model inference")
    public_import.add_argument("--profile", default=str(ROOT / "configs" / "openreview_iclr_pilot.json"))
    public_import.add_argument("--out", required=True, help="NEW local directory; downloaded records are gitignored")
    public_import.add_argument("--allow-public-network", action="store_true")
    public_import.add_argument("--ack-source-terms", action="store_true")
    for name in ("backtest", "review"):
        command = commands.add_parser(name)
        command.add_argument("--data", required=True, help="Historical JSONL")
        command.add_argument("--rubric", required=True)
        command.add_argument("--config", default=str(ROOT / "configs" / "offline.json"))
        command.add_argument("--prompt", help="Versioned JSON instructions, score anchors and output schema")
        command.add_argument("--allow-cloud", action="store_true", help="Explicitly authorize configured cloud transfer for this run")
        command.add_argument("--out", required=True, help="NEW output directory")
        if name == "backtest":
            command.add_argument("--train-before", required=True, help="Exclusive TRAIN submission and label cutoff, timezone required")
            command.add_argument("--validation-before", required=True, help="Exclusive VALIDATION cutoff; TEST starts here")
            command.add_argument("--calibration", choices=["none", "affine", "auto"], default="auto")
            command.add_argument("--top-k", type=int, default=3)
        else:
            command.add_argument("--paper", required=True, help="Unlabelled target manuscript JSON")
            command.add_argument("--state", required=True, help="Previously frozen model_state.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "assess":
            from .assessment import assess
            from .data import fingerprint
            if Path(args.out).exists():
                raise ValueError("Output directory already exists; choose a new path")
            task, request = read_json(args.task), read_json(args.request)
            result = assess(task, request, read_json(args.config), args.allow_cloud)
            Path(args.out).mkdir(parents=True, exist_ok=False)
            write_json(Path(args.out) / "assessment.json", result)
            write_json(Path(args.out) / "task_snapshot.json", {"task": task, "task_sha256": fingerprint(task)})
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "preference-demo":
            from .comparison import run_preference_demo
            rubric = validate_rubric(read_json(ROOT / "examples" / "rubric.json"))
            rows = load_history(ROOT / "examples" / "history.synthetic.jsonl", rubric)
            report = run_preference_demo(rows, rubric, args.out)
            print(report["notice"])
            print(json.dumps(report["metrics"], indent=2))
            return 0
        elif args.command == "import-openreview":
            from .openreview_import import import_public_pilot
            report = import_public_pilot(read_json(args.profile), args.out,
                                         args.allow_public_network, args.ack_source_terms)
            print(json.dumps({k: report[k] for k in ("status", "selected_count", "evaluation_ready_count", "downloaded_pdfs")}, indent=2))
            print("Saved import manifest:", Path(args.out).resolve() / "manifest.json")
            return 0 if report["status"].startswith("completed") else 2
        elif args.command == "demo":
            rubric = validate_rubric(read_json(ROOT / "examples" / "rubric.json"))
            rows = load_history(ROOT / "examples" / "history.synthetic.jsonl", rubric)
            config = {"mode": "offline", "adapter": "mock"}
            model = make_model(config)
            state, report, predictions = run_backtest(rows, rubric, model, config,
                                                      "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
            save_run(args.out, state, report, predictions)
        elif args.command == "validate":
            rubric = validate_rubric(read_json(args.rubric))
            rows = load_history(args.data, rubric)
            print(f"Valid: {len(rows)} historical versions. Backtest additionally requires one version per manuscript.")
            return 0
        else:
            if Path(args.out).exists():
                raise ValueError("Output directory already exists; choose a new path")
            rubric = validate_rubric(read_json(args.rubric))
            rows = load_history(args.data, rubric)
            config = read_json(args.config)
            if args.prompt:
                from .prompts import load_template
                config["prompt_template"] = load_template(args.prompt)
            # Permission/config gate is checked before any model request, with no fallback.
            model = make_model(config, args.allow_cloud)
            if args.command == "backtest":
                state, report, predictions = run_backtest(rows, rubric, model, config, args.train_before,
                                                          args.validation_before, args.calibration, args.top_k)
                save_run(args.out, state, report, predictions)
            else:
                paper = validate_paper(read_json(args.paper), rubric)
                # Target JSON must be unlabelled, even though projection is already whitelisted.
                if "reviews" in paper or "decision" in paper:
                    raise ValueError("Review target must omit reviews and decision")
                result = review_with_state(paper, rows, rubric, model, config, read_json(args.state))
                Path(args.out).mkdir(parents=True, exist_ok=False)
                write_json(Path(args.out) / "review.json", result)
                if "prompt_template" in config:
                    from .prompts import snapshot
                    write_json(Path(args.out) / "prompt_snapshot.json", snapshot(config["prompt_template"]))
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 0
        print(report["notice"])
        print(json.dumps(report["metrics"], indent=2))
        print("Saved local results:", Path(args.out).resolve())
        return 0
    except (ValueError, KeyError, TypeError, OSError, URLError) as exc:
        # Do not expose endpoint response bodies or credentials in error output.
        message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        print("Error:", message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
