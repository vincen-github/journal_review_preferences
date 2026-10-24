"""Authored synthetic privacy checks; every network entry point is blocked."""

import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler

from journal_prefs import __main__ as cli
from journal_prefs.data import paper_input
from journal_prefs.engine import predict, retrieve
from journal_prefs.models import (MockModel, OpenAICompatibleModel, make_model,
                                  validate_config, validate_response)


def rubric():
    return {
        "schema_version": 1, "institution_id": "synthetic-institution",
        "venue_id": "synthetic-venue", "name": "Synthetic rubric", "source": "authored fixture",
        "available_at": "2020-01-01T00:00:00Z", "score_range": [0, 10],
        "accept_threshold": 5, "positive_decision": "accept", "negative_decision": "reject",
        "criteria": [{"id": "method", "description": "Synthetic keyword", "weight": 1,
                      "keywords": ["method"]}],
    }


def paper(manuscript_id="target", submitted="2025-01-01T00:00:00Z"):
    text_available = (datetime.fromisoformat(submitted.replace("Z", "+00:00")) - timedelta(seconds=1)).isoformat()
    return {
        "schema_version": 1, "manuscript_id": manuscript_id, "version_id": "v1",
        "institution_id": "synthetic-institution", "venue_id": "synthetic-venue",
        "title": "Synthetic method paper", "text": "A synthetic method and result.",
        "submitted_at": submitted, "version_time": text_available, "text_available_at": text_available,
        "rights_confirmed": True, "synthetic": True,
        "reviews": [{"text": "SYNTHETIC PRIVATE HUMAN REVIEW", "score": 8,
                     "available_at": "2023-01-02T00:00:00Z"}],
        "decision": {"label": "accept", "available_at": "2023-01-03T00:00:00Z"},
        "private_contact": "synthetic-contact@example.invalid",
    }


def local_config(endpoint="http://127.0.0.1:8000/v1/chat/completions"):
    return {"mode": "local", "adapter": "openai_compatible", "endpoint": endpoint,
            "model": "authored-test-model", "timeout_seconds": 7}


def cloud_config():
    return dict(local_config("https://synthetic-provider.example.invalid/v1/chat/completions"),
                mode="cloud", api_key_env="SYNTHETIC_TEST_KEY", allow_private_data_transfer=True)


def request():
    return {"paper": paper_input(paper()), "rubric": rubric(), "historical_examples": []}


def response():
    return {"score": 5, "review": "Synthetic review.",
            "evidence": [{"quote": "synthetic method", "comment": "Exact synthetic excerpt."}]}


def transport_response(payload):
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.read.return_value = payload
    return result


class NetworkBlockedTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.network_guards = []
        for target in (
            "socket.socket", "socket.create_connection", "socket.getaddrinfo",
            "urllib.request.urlopen", "http.client.HTTPConnection.connect",
            "http.client.HTTPSConnection.connect",
        ):
            guard = self.stack.enter_context(patch(target, side_effect=AssertionError("Network forbidden in unit tests")))
            self.network_guards.append(guard)

    def tearDown(self):
        for guard in self.network_guards:
            guard.assert_not_called()


class PrivacyConfigurationTests(NetworkBlockedTestCase):
    def test_offline_mock_runs_without_network_or_environment_keys(self):
        with patch("journal_prefs.models.os.environ.get", side_effect=AssertionError("No credential access")):
            model = make_model({"mode": "offline", "adapter": "mock"})
            prediction = predict(paper(), [], rubric(), model)
        self.assertIsInstance(model, MockModel)
        self.assertTrue(prediction["synthetic_output"])
        self.assertIn("SYNTHETIC MOCK OUTPUT", prediction["review"])

    def test_offline_rejects_endpoint_or_fallback_settings(self):
        for extra in ({"endpoint": "https://example.invalid/chat/completions"},
                      {"fallback": "cloud"}, {"api_key_env": "SYNTHETIC_TEST_KEY"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                make_model(dict(mode="offline", adapter="mock", **extra))

    def test_cloud_requires_both_explicit_config_and_run_permission(self):
        with self.assertRaisesRegex(ValueError, "Cloud is disabled"):
            make_model(cloud_config())
        for consent in (None, False, "true", 1):
            config = cloud_config()
            config["allow_private_data_transfer"] = consent
            with self.subTest(consent=consent), self.assertRaises(ValueError):
                make_model(config, allow_cloud=True)
        with redirect_stderr(io.StringIO()):
            self.assertFalse(make_model(cloud_config(), allow_cloud=True).synthetic)

    def test_cloud_disclosure_contains_destination_but_no_key_or_manuscript(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"SYNTHETIC_TEST_KEY": "AUTHORED_FAKE_SECRET"}, clear=True):
            with redirect_stderr(output):
                make_model(cloud_config(), allow_cloud=True)
        self.assertIn("synthetic-provider.example.invalid", output.getvalue())
        self.assertIn("historical", output.getvalue())
        self.assertNotIn("AUTHORED_FAKE_SECRET", output.getvalue())
        self.assertNotIn(paper()["text"], output.getvalue())

    def test_local_accepts_only_http_literal_loopback(self):
        for endpoint in ("http://127.0.0.1:8000/v1/chat/completions",
                         "http://[::1]:8000/v1/chat/completions"):
            with self.subTest(endpoint=endpoint):
                validate_config(local_config(endpoint))
        for host in ("localhost", "127.0.0.2", "127.1", "2130706433", "0x7f000001",
                     "0.0.0.0", "192.168.1.1", "example.invalid", "[::ffff:127.0.0.1]"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                make_model(local_config("http://" + host + "/v1/chat/completions"))
        for endpoint in ("https://127.0.0.1/chat/completions", "file:///chat/completions"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_config(local_config(endpoint))

    def test_endpoint_credentials_queries_fragments_and_wrong_paths_are_rejected(self):
        for endpoint in ("http://user:fake-secret@127.0.0.1/chat/completions",
                         "http://127.0.0.1/chat/completions?api_key=fake-secret",
                         "http://127.0.0.1/chat/completions#fake-secret",
                         "http://127.0.0.1/v1/completions"):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_config(local_config(endpoint))
        with self.assertRaises(ValueError):
            validate_config(dict(local_config(), api_key_env="SYNTHETIC_TEST_KEY"))

    def test_cloud_rejects_plain_http_and_missing_key_name(self):
        for updates in ({"endpoint": "http://example.invalid/chat/completions"},
                        {"api_key_env": ""}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                make_model(dict(cloud_config(), **updates), allow_cloud=True)

    def test_invalid_timeout_rejected_before_transport(self):
        for timeout in (0, -1, 121, True, float("nan"), float("inf"), "30"):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                make_model(dict(local_config(), timeout_seconds=timeout))

    def test_adapter_configuration_cannot_be_mutated_into_remote_transfer(self):
        config = local_config()
        model = make_model(config)
        config["endpoint"] = "http://remote.example.invalid/v1/chat/completions"
        model.config["endpoint"] = "http://another-remote.example.invalid/v1/chat/completions"
        opener = Mock()
        opener.open.return_value = transport_response(json.dumps({"choices": [{"message": {
            "content": json.dumps(response())}}]}).encode())
        with patch("urllib.request.build_opener", return_value=opener):
            try:
                model.review(request())
            except ValueError:
                opener.open.assert_not_called()
            else:
                self.assertEqual(opener.open.call_args.args[0].full_url, local_config()["endpoint"])

    def test_direct_adapter_construction_enforces_privacy_gate(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleModel(cloud_config())
        with self.assertRaises(ValueError):
            OpenAICompatibleModel(local_config("http://remote.example.invalid/chat/completions"))


class PrivacyTransportTests(NetworkBlockedTestCase):
    def test_local_transport_disables_proxy_redirect_and_authorization(self):
        opener = Mock()
        raw = json.dumps({"choices": [{"message": {"content": json.dumps(response())}}]}).encode()
        opener.open.return_value = transport_response(raw)
        with patch("urllib.request.build_opener", return_value=opener) as build:
            with patch("journal_prefs.models.os.environ.get", side_effect=AssertionError("No key access")):
                actual = make_model(local_config()).review(request())
        self.assertEqual(actual, response())
        handlers = build.call_args.args
        proxy = next(h for h in handlers if isinstance(h, ProxyHandler))
        self.assertEqual(proxy.proxies, {})
        redirect = next(h for h in handlers if isinstance(h, HTTPRedirectHandler))
        with self.assertRaisesRegex(ValueError, "redirects are disabled"):
            redirect.redirect_request(None, None, 302, "redirect", {}, "https://example.invalid/")
        sent = opener.open.call_args.args[0]
        self.assertEqual(sent.full_url, local_config()["endpoint"])
        self.assertEqual(sent.get_method(), "POST")
        self.assertFalse(any(key.lower() == "authorization" for key in sent.headers))
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 7)
        payload = json.loads(sent.data)
        self.assertEqual(json.loads(payload["messages"][1]["content"]), request())
        self.assertIn("untrusted data", payload["messages"][0]["content"])

    def test_missing_cloud_key_fails_before_building_transport(self):
        with redirect_stderr(io.StringIO()):
            model = make_model(cloud_config(), allow_cloud=True)
        with patch.dict(os.environ, {}, clear=True), patch("urllib.request.build_opener") as build:
            with self.assertRaisesRegex(ValueError, "no request sent"):
                model.review(request())
        build.assert_not_called()

    def test_cloud_transport_uses_only_the_named_fake_key_and_logs_no_credentials(self):
        output = io.StringIO()
        opener = Mock()
        opener.open.return_value = transport_response(json.dumps({"choices": [{"message": {
            "content": json.dumps(response())}}]}).encode())
        with patch.dict(os.environ, {"SYNTHETIC_TEST_KEY": "AUTHORED_FAKE_SECRET",
                                    "UNRELATED_TEST_KEY": "UNRELATED_FAKE_SECRET"}, clear=True), \
                patch("urllib.request.build_opener", return_value=opener), \
                redirect_stderr(output), redirect_stdout(output):
            model = make_model(cloud_config(), allow_cloud=True)
            model.review(request())
        sent = opener.open.call_args.args[0]
        self.assertEqual(sent.full_url, cloud_config()["endpoint"])
        self.assertEqual(sent.get_header("Authorization"), "Bearer AUTHORED_FAKE_SECRET")
        self.assertNotIn("AUTHORED_FAKE_SECRET", output.getvalue())
        self.assertNotIn("UNRELATED_FAKE_SECRET", output.getvalue())
        self.assertNotIn("AUTHORED_FAKE_SECRET", sent.data.decode())

    def test_invalid_credential_is_rejected_without_logging_or_transport(self):
        with redirect_stderr(io.StringIO()):
            model = make_model(cloud_config(), allow_cloud=True)
        fake_secret = "AUTHORED_FAKE_SECRET\r\nInjected: synthetic"
        output = io.StringIO()
        with patch.dict(os.environ, {"SYNTHETIC_TEST_KEY": fake_secret}, clear=True), \
                patch("urllib.request.build_opener") as build, redirect_stderr(output):
            with self.assertRaises(ValueError) as raised:
                model.review(request())
        build.assert_not_called()
        self.assertNotIn("AUTHORED_FAKE_SECRET", str(raised.exception) + output.getvalue())

    def test_transport_failure_has_one_attempt_and_no_fallback(self):
        opener = Mock()
        opener.open.side_effect = URLError("AUTHORED_FAKE_SERVER_FAILURE")
        with patch("urllib.request.build_opener", return_value=opener) as build:
            with self.assertRaises(URLError):
                make_model(local_config()).review(request())
        self.assertEqual(build.call_count, 1)
        self.assertEqual(opener.open.call_count, 1)

    def test_response_read_is_bounded(self):
        opener = Mock()
        reply = transport_response(b"x" * 2_000_001)
        opener.open.return_value = reply
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "exceeds 2 MB"):
                make_model(local_config()).review(request())
        reply.read.assert_called_once_with(2_000_001)

    def test_cli_transport_failure_hides_server_body_and_credential(self):
        output = io.StringIO()
        failure = HTTPError("https://example.invalid", 401, "AUTHORED_FAKE_SECRET", {},
                            io.BytesIO(b"AUTHORED_PRIVATE_SERVER_BODY"))
        with tempfile.TemporaryDirectory() as directory:
            argv = ["backtest", "--data", "synthetic.jsonl", "--rubric", "rubric.json",
                    "--config", "config.json", "--out", str(Path(directory) / "new-output"),
                    "--train-before", "2023-01-01T00:00:00Z",
                    "--validation-before", "2024-01-01T00:00:00Z"]
            with patch.object(cli, "read_json", side_effect=[rubric(), local_config()]), \
                    patch.object(cli, "load_history", return_value=[]), \
                    patch.object(cli, "run_backtest", side_effect=failure), \
                    redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(cli.main(argv), 2)
            self.assertFalse((Path(directory) / "new-output").exists())
        self.assertIn("HTTPError", output.getvalue())
        self.assertNotIn("AUTHORED_FAKE_SECRET", output.getvalue())
        self.assertNotIn("AUTHORED_PRIVATE_SERVER_BODY", output.getvalue())


class InferenceDataPrivacyTests(NetworkBlockedTestCase):
    def test_target_projection_excludes_labels_and_arbitrary_private_fields(self):
        projected = paper_input(paper())
        self.assertEqual(set(projected), {"manuscript_id", "version_id", "title", "text",
                                          "submitted_at", "version_time"})
        model = Mock(synthetic=True)
        model.review.return_value = response()
        predict(paper(), [], rubric(), model)
        sent = model.review.call_args.args[0]
        self.assertEqual(sent["paper"], projected)
        encoded = json.dumps(sent)
        self.assertNotIn("SYNTHETIC PRIVATE HUMAN REVIEW", encoded)
        self.assertNotIn("synthetic-contact", encoded)

    def test_history_projection_excludes_same_paper_future_labels_and_other_venue(self):
        old = paper("eligible", "2023-01-01T00:00:00Z")
        same = paper("target", "2023-01-01T00:00:00Z")
        future = paper("future", "2023-01-01T00:00:00Z")
        future["decision"]["available_at"] = "2026-01-01T00:00:00Z"
        other = paper("other-venue", "2023-01-01T00:00:00Z")
        other["venue_id"] = "different-venue"
        examples = retrieve(paper(), [old, same, future, other])
        self.assertEqual([e["manuscript_id"] for e in examples], ["eligible"])
        self.assertNotIn("private_contact", json.dumps(examples))
        self.assertLessEqual(len(examples[0]["text_excerpt"]), 2000)

    def test_exact_target_evidence_required(self):
        for evidence in ([], [{"quote": "not present", "comment": "Synthetic"}],
                         [{"quote": "SYNTHETIC PRIVATE HUMAN REVIEW", "comment": "Historical only"}],
                         [{"quote": "   ", "comment": "Synthetic"}],
                         [{"quote": "synthetic method", "comment": ""}]):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate_response(dict(response(), evidence=evidence), request())

    def test_nonfinite_out_of_range_or_boolean_scores_rejected(self):
        for score in (-1, 11, True, float("nan"), float("inf"), "5"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                validate_response(dict(response(), score=score), request())

    def test_valid_response_is_projected_to_validated_fields_only(self):
        candidate = response()
        candidate["private_payload"] = "UNVALIDATED_PRIVATE_TOP_LEVEL"
        candidate["evidence"][0]["private_payload"] = "UNVALIDATED_PRIVATE_NESTED"
        validated = validate_response(candidate, request())
        self.assertEqual(set(validated), {"score", "review", "evidence"})
        self.assertEqual(set(validated["evidence"][0]), {"quote", "comment"})

    def test_malformed_evidence_shapes_fail_with_validation_error(self):
        for evidence in ([None], ["synthetic method"], [42], [{}]):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                validate_response(dict(response(), evidence=evidence), request())

    def test_malformed_top_level_response_fails_with_validation_error(self):
        for candidate in (None, [], "SYNTHETIC INVALID RESPONSE", 5):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_response(candidate, request())


if __name__ == "__main__":
    unittest.main()
