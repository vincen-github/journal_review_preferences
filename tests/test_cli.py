import contextlib
import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_prefs.__main__ import main

ROOT = Path(__file__).resolve().parent.parent


class CLITests(unittest.TestCase):
    def test_full_demo_validation_and_new_review_with_network_forbidden(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(socket, "socket", side_effect=AssertionError("No network permitted")), \
             patch.object(socket, "getaddrinfo", side_effect=AssertionError("No DNS permitted")):
            output = Path(temp) / "demo"
            self.assertEqual(main(["demo", "--out", str(output)]), 0)
            self.assertTrue((output / "report.md").exists())
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["privacy_mode"], "offline")
            self.assertEqual(main(["validate", "--data", str(ROOT / "examples/history.synthetic.jsonl"),
                                   "--rubric", str(ROOT / "examples/rubric.json")]), 0)
            reviewed = Path(temp) / "review"
            args = ["review", "--data", str(ROOT / "examples/history.synthetic.jsonl"),
                    "--rubric", str(ROOT / "examples/rubric.json"),
                    "--paper", str(ROOT / "examples/submission.synthetic.json"),
                    "--state", str(output / "model_state.json"), "--out", str(reviewed)]
            self.assertEqual(main(args), 0)
            result = json.loads((reviewed / "review.json").read_text(encoding="utf-8"))
            self.assertTrue(result["synthetic_output"])
            self.assertTrue(result["human_review_required"])
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)  # Existing results are never overwritten.

    def test_cli_backtest_command(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["backtest", "--data", str(ROOT / "examples/history.synthetic.jsonl"),
                                   "--rubric", str(ROOT / "examples/rubric.json"),
                                   "--train-before", "2024-01-01T00:00:00Z",
                                   "--validation-before", "2025-01-01T00:00:00Z",
                                   "--calibration", "none", "--out", str(Path(temp) / "run")]), 0)


if __name__ == "__main__":
    unittest.main()
