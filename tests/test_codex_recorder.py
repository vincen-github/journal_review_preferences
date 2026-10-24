import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("codex_baseline_tool", ROOT / "tools/run_codex_baseline.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class RecorderTests(unittest.TestCase):
    def inputs(self, root, count=6):
        papers = []
        for i in range(count):
            text = "SYNTHETIC manuscript body " + str(i)
            papers.append({"paper_id": str(i), "title": "Synthetic", "text": text,
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "manuscript_version_status": "synthetic"})
        path = root / "inputs.json"
        path.write_text(json.dumps(papers), encoding="utf-8")
        return path

    def response(self, args, input, **kwargs):
        paper = json.loads(input.split("MANUSCRIPT_DATA_JSON:\n")[1])
        answer = {"score": 5, "review": "SYNTHETIC", "evidence": [{"quote": paper["text"], "comment": "SYNTHETIC"}]}
        events = [{"type": "item.completed", "item": {"type": "error", "message": "Authored startup warning"}},
                  {"type": "turn.started"},
                  {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}},
                  {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}]
        return SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(e) for e in events), stderr='route slug="gpt-6.1-sol"')

    def test_startup_notifications_are_not_tool_calls_and_all_outputs_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.inputs(root)
            with patch.object(tool.subprocess, "run", side_effect=self.response) as calls, patch("builtins.print"):
                result = tool.run(path, root / "out", "fake-no-executable", True)
            self.assertEqual(calls.call_count, 6)
            self.assertEqual(result["status"], "completed_predictions_frozen_before_label_evaluation")
            self.assertEqual(len(list((root / "out").glob("response_*.json.txt"))), 6)
            self.assertTrue(all(not r["tool_events"] for r in result["runs"]))

    def test_no_quota_optin_means_no_process_or_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(tool.subprocess, "run") as calls:
            root = Path(directory)
            with self.assertRaises(ValueError):
                tool.run(root / "unused", root / "out", "unused", False)
            calls.assert_not_called()
            self.assertFalse((root / "out").exists())

    def test_explicit_prompt_saves_actual_snapshot_and_schema_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.inputs(root)
            prompt = ROOT / "prompts/iclr2017_official_form.v1.json"
            def fake(args, input, **kwargs):
                saved = json.loads((root / "out/prompt_snapshot.json").read_text(encoding="utf-8"))
                self.assertIn("acceptance boundary", input)
                self.assertEqual(saved["template"]["template_id"], "iclr2017_official_form")
                schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text(encoding="utf-8"))
                self.assertEqual(schema["properties"]["score"]["type"], "integer")
                paper = json.loads(input.split("MANUSCRIPT_DATA_JSON:\n")[1])
                self.assertEqual(set(paper), {"paper_id", "title", "text"})
                return self.response(args, input, **kwargs)
            with patch.object(tool.subprocess, "run", side_effect=fake), patch("builtins.print"):
                result = tool.run(path, root / "out", "fake-no-executable", True, "test", prompt)
            self.assertEqual(result["arm"], "iclr2017_official_form")
            saved = json.loads((root / "out/prompt_snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(result["prompt_template_sha256"], saved["template_sha256"])

    def test_larger_fixed_batch_scores_each_paper_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.inputs(root, 21)
            with patch.object(tool.subprocess, "run", side_effect=self.response) as calls, patch("builtins.print"):
                result = tool.run(path, root / "out", "fake-no-executable", True)
            self.assertEqual(calls.call_count, 21)
            self.assertEqual(len(result["runs"]), 21)

    def test_shared_abort_prevents_calls_and_failure_creates_stop_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.inputs(root)
            abort = root / "abort.json"
            failed = SimpleNamespace(returncode=1, stdout="", stderr="authored failure, no credential")
            with patch.object(tool.subprocess, "run", return_value=failed) as calls:
                with self.assertRaises(ValueError):
                    tool.run(path, root / "first", "fake", True, abort_file=abort)
                self.assertEqual(calls.call_count, 1)
                self.assertTrue(abort.exists())
                calls.reset_mock()
                with self.assertRaises(ValueError):
                    tool.run(path, root / "second", "fake", True, abort_file=abort)
                calls.assert_not_called()
                self.assertFalse((root / "second").exists())

    def test_whitespace_recovery_never_repeats_completed_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.inputs(root)
            papers = json.loads(path.read_text(encoding="utf-8"))
            papers[0]["text"] = "SYNTHETIC manuscript\nbody 0"
            papers[0]["text_sha256"] = hashlib.sha256(papers[0]["text"].encode()).hexdigest()
            path.write_text(json.dumps(papers), encoding="utf-8")
            invoked = []
            def fake(args, input, **kwargs):
                paper = json.loads(input.split("MANUSCRIPT_DATA_JSON:\n")[1])
                invoked.append(paper["paper_id"])
                answer = {"score": 5, "review": "SYNTHETIC", "evidence": [{"quote": paper["text"].replace("\n", " "), "comment": "SYNTHETIC"}]}
                return SimpleNamespace(returncode=0, stdout=json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}}), stderr='slug="gpt-6.1-sol"')
            with patch.object(tool.subprocess, "run", side_effect=fake), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "quote is absent"):
                    tool.run(path, root / "out", "fake", True)
                raw = (root / "out/response_0.json.txt").read_bytes()
                result = tool.run(path, root / "out", "fake", True, resume=True, repair_evidence_whitespace=True)
            self.assertEqual(invoked, [str(i) for i in range(6)])
            self.assertEqual(raw, (root / "out/response_0.json.txt").read_bytes())
            self.assertEqual(result["runs"][0]["initial_status"], "failed_or_uncertain_no_retry")
            self.assertEqual(result["runs"][0]["score"], 5)
            self.assertEqual(len(result["runs"][0]["evidence_whitespace_repairs"]), 1)
            self.assertTrue((root / "out/manifest.before_resume.json").exists())

    def test_whitespace_projection_does_not_change_nonwhitespace_characters(self):
        from journal_prefs.evidence_format import restore_source_whitespace
        raw = {"score": 5, "review": "SYNTHETIC", "evidence": [{"quote": "missing-word", "comment": "SYNTHETIC"}]}
        projected, repairs = restore_source_whitespace(raw, "missing word")
        self.assertEqual(projected, raw)
        self.assertEqual(repairs, [])

    def test_invalid_fragment_is_retained_in_audit_without_changing_score(self):
        from journal_prefs.evidence_format import quarantine_unmatched_quotes
        raw = {"score": 6, "review": "SYNTHETIC", "evidence": [{"quote": "valid source", "comment": "SYNTHETIC"}, {"quote": "invented z", "comment": "SYNTHETIC"}]}
        checked, rejected = quarantine_unmatched_quotes(raw, "valid source")
        self.assertEqual(checked["score"], 6)
        self.assertEqual(checked["review"], raw["review"])
        self.assertEqual(checked["evidence"], raw["evidence"][:1])
        self.assertEqual(rejected[0]["quote"], "invented z")
        self.assertEqual(len(raw["evidence"]), 2)
