"""Synthetic subprocess receipts only. These tests never invoke a real model."""

import io
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from journal_prefs.data import fingerprint, read_json
from tools.codex_json_generation import generate


SCHEMA = {"type": "object", "properties": {"rules": {"type": "array", "items": {"type": "string"}}},
          "required": ["rules"], "additionalProperties": False}
PAYLOAD = {"training_examples": [{"paper_id": "SYNTHETIC_TRAIN", "text": "SYNTHETIC manuscript",
                                   "review": "SYNTHETIC historical review"}]}
INSTRUCTIONS = "Return synthetic rule strings using the supplied synthetic TRAIN examples only."


def events(raw='{"rules": ["SYNTHETIC rule"]}', extra=None):
    rows = [{"type": "thread.started", "thread_id": "DO_NOT_STORE_THREAD_ID"},
            {"type": "turn.started"}]
    rows.extend(extra or [])
    if raw is not None:
        rows.append({"type": "item.completed", "item": {"type": "agent_message", "text": raw}})
    rows.append({"type": "turn.completed", "usage": {"input_tokens": 120, "output_tokens": 30,
                  "cached_input_tokens": 10, "reasoning_output_tokens": 20, "cache_write_input_tokens": 5,
                  "account_id": "DO_NOT_STORE_ACCOUNT", "malicious": "DO_NOT_STORE_USAGE_TEXT"}})
    return "\n".join(json.dumps(row) for row in rows)


def result(raw='{"rules": ["SYNTHETIC rule"]}', extra=None, returncode=0, stderr='route slug="gpt-6.1-sol"'):
    return SimpleNamespace(returncode=returncode, stdout=events(raw, extra), stderr=stderr)


class RuleGenerationTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.guards = []
        for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo",
                       "urllib.request.urlopen", "http.client.HTTPConnection.connect",
                       "http.client.HTTPSConnection.connect", "subprocess.Popen"):
            self.guards.append(stack.enter_context(patch(target, side_effect=AssertionError("Real process/network forbidden"))))

    def tearDown(self):
        for guard in self.guards:
            guard.assert_not_called()

    def call(self, output, **kwargs):
        return generate(PAYLOAD, INSTRUCTIONS, SCHEMA, output, "SYNTHETIC_CODEX.exe", allow_quota=True, **kwargs)

    def test_success_preserves_started_snapshot_before_single_isolated_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation"

            def mocked(args, **kwargs):
                self.assertEqual(read_json(output / "receipt.json")["status"], "started")
                snapshot = read_json(output / "prompt_snapshot.json")
                self.assertEqual(snapshot["payload"], PAYLOAD)
                self.assertEqual(snapshot["request"], kwargs["input"])
                self.assertEqual(snapshot["request_sha256"], fingerprint(kwargs["input"]))
                self.assertEqual(read_json(output / "response.schema.json"), SCHEMA)
                self.assertEqual(read_json(output / "isolated/response.schema.json"), SCHEMA)
                self.assertEqual(list((output / "isolated").iterdir()), [output / "isolated/response.schema.json"])
                self.assertEqual(kwargs["cwd"], output / "isolated")
                self.assertEqual(kwargs["timeout"], 300)
                self.assertTrue(kwargs["capture_output"])
                self.assertEqual(args[0], "SYNTHETIC_CODEX.exe")
                for flag in ("--ephemeral", "--ignore-user-config", "--skip-git-repo-check", "--json"):
                    self.assertIn(flag, args)
                self.assertEqual(args[args.index("--model") + 1], "gpt-6.1-sol")
                for setting in ('forced_login_method="chatgpt"', 'model_reasoning_effort="low"',
                                'service_tier="default"', 'permissions.scoring.network.enabled=false',
                                'features.shell_tool=false', 'features.apps=false', 'features.plugins=false',
                                'features.multi_agent=false', 'features.memories=false', 'mcp_servers={}',
                                'project_doc_max_bytes=0',
                                'permissions.scoring.filesystem={":root"="deny",":minimal"="read",":workspace_roots"={"."="read"}}'):
                    self.assertIn(setting, args)
                self.assertEqual(args[-1], "-")
                return result()

            with patch("tools.codex_json_generation.subprocess.run", side_effect=mocked) as run:
                response, receipt = self.call(output)
            run.assert_called_once()
            self.assertEqual(response, {"rules": ["SYNTHETIC rule"]})
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["subprocess_attempts"], 1)
            self.assertFalse(receipt["automatic_retry"])
            self.assertTrue(receipt["reported_sol_slug"])
            self.assertEqual(receipt["tool_events"], [])
            self.assertEqual(receipt["response_sha256"], fingerprint(response))
            self.assertEqual(receipt, read_json(output / "receipt.json"))
            self.assertEqual(read_json(output / "response.json"), response)
            self.assertEqual(json.loads((output / "response.raw.txt").read_text(encoding="utf-8")), response)

    def test_explicit_literal_opt_in_required_before_any_output_or_process(self):
        with tempfile.TemporaryDirectory() as directory:
            for permission in (False, None, 1, "true"):
                output = Path(directory) / str(permission)
                with self.subTest(permission=permission), patch("tools.codex_json_generation.subprocess.run") as run:
                    with self.assertRaisesRegex(ValueError, "allow_quota=True"):
                        generate(PAYLOAD, INSTRUCTIONS, SCHEMA, output, "SYNTHETIC_CODEX.exe", permission)
                    run.assert_not_called()
                    self.assertFalse(output.exists())

    def test_existing_output_refuses_repeated_or_uncertain_generation(self):
        with tempfile.TemporaryDirectory() as directory, patch("tools.codex_json_generation.subprocess.run") as run:
            with self.assertRaises(FileExistsError):
                self.call(Path(directory))
            run.assert_not_called()

    def test_usage_and_notices_drop_identity_and_raw_stderr(self):
        extra = [{"type": "item.completed", "item": {"type": "error",
                  "message": "Invalid profile for DO_NOT_STORE_EMAIL@example.invalid DO_NOT_STORE_SECRET"}},
                 {"type": "unrecognized_DO_NOT_STORE_IDENTITY", "account": "DO_NOT_STORE_ACCOUNT"}]
        stderr = 'slug="gpt-6.1-sol" DO_NOT_STORE_STDERR DO_NOT_STORE_EMAIL@example.invalid'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation"
            stream = io.StringIO()
            with patch("tools.codex_json_generation.subprocess.run", return_value=result(extra=extra, stderr=stderr)), \
                    redirect_stdout(stream), redirect_stderr(stream):
                _, receipt = self.call(output)
            self.assertEqual(stream.getvalue(), "")
            self.assertEqual(receipt["usage"], {"input_tokens": 120, "output_tokens": 30, "cached_input_tokens": 10,
                                               "reasoning_output_tokens": 20, "cache_write_input_tokens": 5})
            self.assertNotIn("total_tokens", receipt["usage"])  # Preserve components; do not invent/add a total.
            self.assertEqual(receipt["cli_notices"], ["invalid_configuration"])
            self.assertEqual(receipt["unrecognized_event_count"], 1)
            saved = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*") if path.is_file())
            self.assertNotIn("DO_NOT_STORE", saved)

    def test_invalid_json_keeps_original_final_and_hash_with_safe_failure(self):
        raw = "SYNTHETIC malformed JSON final"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation"
            with patch("tools.codex_json_generation.subprocess.run", return_value=result(raw)) as run:
                with self.assertRaisesRegex(ValueError, "no automatic retry"):
                    self.call(output)
            run.assert_called_once()
            receipt = read_json(output / "receipt.json")
            self.assertEqual(receipt["error_code"], "invalid_json_object")
            self.assertEqual(receipt["status"], "failed_or_uncertain_no_retry")
            self.assertEqual((output / "response.raw.txt").read_text(encoding="utf-8"), raw)
            self.assertIn("raw_response_sha256", receipt)
            self.assertFalse((output / "response.json").exists())

    def test_nonobject_and_nonfinite_json_are_rejected_without_retry(self):
        for raw in ("[]", "null", '"string"', '{"rules": NaN}', '{"rules": Infinity}'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "generation"
                with patch("tools.codex_json_generation.subprocess.run", return_value=result(raw)) as run:
                    with self.assertRaises(ValueError):
                        self.call(output)
                run.assert_called_once()
                self.assertEqual((output / "response.raw.txt").read_text(encoding="utf-8"), raw)

    def test_nonzero_exit_retains_returned_final_but_does_not_return_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation"
            with patch("tools.codex_json_generation.subprocess.run", return_value=result(returncode=2)) as run:
                with self.assertRaises(ValueError):
                    self.call(output)
            run.assert_called_once()
            self.assertTrue((output / "response.raw.txt").exists())
            receipt = read_json(output / "receipt.json")
            self.assertEqual(receipt["exit_code"], 2)
            self.assertEqual(receipt["error_code"], "nonzero_exit")

    def test_any_tool_or_unknown_item_stops_and_preserves_final(self):
        for item_type in ("command_execution", "mcp_tool_call", "web_search", "file_read", "new_unknown_tool"):
            extra = [{"type": "item.completed", "item": {"type": item_type, "output": "DO_NOT_STORE_TOOL_OUTPUT"}}]
            with self.subTest(item_type=item_type), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "generation"
                with patch("tools.codex_json_generation.subprocess.run", return_value=result(extra=extra)) as run:
                    with self.assertRaises(ValueError):
                        self.call(output)
                run.assert_called_once()
                receipt = read_json(output / "receipt.json")
                self.assertEqual(receipt["error_code"], "tool_or_unrecognized_item")
                self.assertTrue(receipt["tool_events"])
                self.assertNotIn("DO_NOT_STORE_TOOL_OUTPUT", repr(receipt))
                self.assertTrue((output / "response.raw.txt").exists())

    def test_model_route_absent_or_lookalike_slug_is_not_accepted(self):
        for stderr in ("", 'slug="another-model"', 'slug="gpt-6.1-sol-fake"'):
            with self.subTest(stderr=stderr), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "generation"
                with patch("tools.codex_json_generation.subprocess.run", return_value=result(stderr=stderr)):
                    with self.assertRaises(ValueError):
                        self.call(output)
                self.assertEqual(read_json(output / "receipt.json")["error_code"], "model_route_not_reported")

    def test_missing_final_or_failed_turn_cannot_succeed(self):
        cases = [(result(raw=None), "missing_final_response"),
                 (result(extra=[{"type": "turn.failed", "error": {"message": "rate limit DO_NOT_STORE_SECRET"}}]), "turn_failed")]
        for reply, code in cases:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "generation"
                with patch("tools.codex_json_generation.subprocess.run", return_value=reply) as run:
                    with self.assertRaises(ValueError):
                        self.call(output)
                run.assert_called_once()
                receipt = read_json(output / "receipt.json")
                self.assertEqual(receipt["error_code"], code)
                self.assertNotIn("DO_NOT_STORE_SECRET", repr(receipt))

    def test_timeout_preserves_partial_final_without_stderr_or_exception_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation"
            timeout = subprocess.TimeoutExpired("DO_NOT_STORE_COMMAND_IDENTITY", 300,
                output=events().encode(), stderr=b'slug="gpt-6.1-sol" DO_NOT_STORE_SECRET')
            with patch("tools.codex_json_generation.subprocess.run", side_effect=timeout) as run:
                with self.assertRaises(ValueError) as raised:
                    self.call(output)
            run.assert_called_once()
            receipt = read_json(output / "receipt.json")
            self.assertEqual(receipt["error_code"], "timeout_uncertain")
            self.assertIsNone(receipt["exit_code"])
            self.assertTrue((output / "response.raw.txt").exists())
            self.assertNotIn("DO_NOT_STORE", repr(receipt) + str(raised.exception))

    def test_launch_error_has_safe_receipt_and_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation"
            with patch("tools.codex_json_generation.subprocess.run", side_effect=OSError("DO_NOT_STORE_SECRET")) as run:
                with self.assertRaises(ValueError) as raised:
                    self.call(output)
            run.assert_called_once()
            receipt = read_json(output / "receipt.json")
            self.assertEqual(receipt["error_code"], "process_launch_error")
            self.assertNotIn("DO_NOT_STORE", repr(receipt) + str(raised.exception))

    def test_invalid_inputs_fail_before_output_or_subprocess(self):
        cases = [(None, INSTRUCTIONS, SCHEMA), (PAYLOAD, " ", SCHEMA), (PAYLOAD, INSTRUCTIONS, []),
                 ({"nan": float("nan")}, INSTRUCTIONS, SCHEMA),
                 ({1: "not-string-key"}, INSTRUCTIONS, SCHEMA), ({"tuple": (1, 2)}, INSTRUCTIONS, SCHEMA)]
        with tempfile.TemporaryDirectory() as directory:
            for index, (payload, instructions, schema) in enumerate(cases):
                output = Path(directory) / str(index)
                with self.subTest(index=index), patch("tools.codex_json_generation.subprocess.run") as run:
                    with self.assertRaises(ValueError):
                        generate(payload, instructions, schema, output, "SYNTHETIC_CODEX.exe", True)
                    run.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
