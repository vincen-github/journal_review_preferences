"""Synthetic generic contracts only; model and network entrypoints are blocked."""
import copy
import math
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from journal_prefs.assessment import assess, evaluate_metrics, validate_output, validate_request, validate_task
from journal_prefs.data import fingerprint, read_json

ROOT = Path(__file__).resolve().parent.parent


class AssessmentTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.guards = []
        for name in ("socket.socket", "socket.create_connection", "socket.getaddrinfo", "urllib.request.urlopen",
                     "http.client.HTTPConnection.connect", "http.client.HTTPSConnection.connect",
                     "journal_prefs.models.OpenAICompatibleModel.generate_json"):
            self.guards.append(stack.enter_context(patch(name, side_effect=AssertionError("No real model/network"))))
        self.task = read_json(ROOT / "examples/assessment_task.synthetic.json")
        self.request = read_json(ROOT / "examples/assessment_request.synthetic.json")

    def tearDown(self):
        for guard in self.guards:
            guard.assert_not_called()

    def test_offline_no_score_example_has_explicit_synthetic_output_and_snapshots(self):
        with patch("journal_prefs.assessment.make_model", side_effect=AssertionError("Offline model construction forbidden")) as model:
            result = assess(self.task, self.request)
        model.assert_not_called()
        self.assertTrue(result["synthetic_output"])
        self.assertTrue(result["human_review_required"])
        self.assertFalse(result["automatic_editorial_decision"])
        self.assertNotIn("score", result["output"])
        self.assertEqual(result["output"], self.task["offline_response"]["output"])
        for name, snapshot in (("task", self.task), ("request", self.request), ("output", result["output"])):
            self.assertEqual(result[name + "_sha256"], fingerprint(snapshot))
        self.assertEqual(result["task_snapshot"], self.task)
        self.assertEqual(result["request_snapshot"], self.request)
        result["task_snapshot"]["instructions"] = "changed"
        self.assertNotEqual(result["task_snapshot"], self.task)

    def test_offline_requires_schema_valid_explicit_synthetic_response(self):
        cases = [dict(self.task, offline_response={"synthetic": False, "output": {}}),
                 dict(self.task, offline_response={"synthetic": True, "output": {}})]
        missing = copy.deepcopy(self.task)
        missing.pop("offline_response")
        cases.append(missing)
        for task in cases:
            with self.subTest(task=task), self.assertRaises(ValueError):
                assess(task, self.request)

    def test_target_own_labels_and_unexpected_request_fields_are_rejected_before_model(self):
        for key in ("score", "decision", "accepted", "reviews", "output", "unrecognized"):
            request = copy.deepcopy(self.request)
            request["target"][key] = "OWN_LABEL"
            with self.subTest(key=key), self.assertRaises(ValueError):
                assess(self.task, request)
        with self.assertRaises(ValueError):
            validate_request(dict(self.request, test_labels=[]), self.task)

    def test_history_requires_explicit_train_membership_disjoint_unique_ids(self):
        for split in ("test", "TEST", "VALIDATION", None):
            request = copy.deepcopy(self.request)
            request["history"][0]["split"] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                validate_request(request, self.task)
        request = copy.deepcopy(self.request)
        request["history"][0]["manuscript"]["id"] = request["target"]["id"]
        with self.assertRaisesRegex(ValueError, "Same-paper"):
            validate_request(request, self.task)
        request = copy.deepcopy(self.request)
        request["history"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_request(request, self.task)
        self.assertEqual(validate_request(dict(self.request, history=[]), self.task)["history"], [])

    def test_no_required_score_review_or_evidence_names_for_text_only_task(self):
        task = {"schema_version": 1, "task_id": "synthetic-text", "version": "1", "instructions": "SYNTHETIC text only",
                "output_schema": {"type": "object", "properties": {"editorial_memo": {"type": "string", "maxLength": 500}},
                                  "required": ["editorial_memo"], "additionalProperties": False},
                "offline_response": {"synthetic": True, "output": {"editorial_memo": "SYNTHETIC free-text memo"}}}
        request = dict(self.request, history=[])
        result = assess(task, request)
        self.assertEqual(set(result["output"]), {"editorial_memo"})
        self.assertEqual(evaluate_metrics(task, [result["output"]], [result["output"]]), {})

    def test_nested_custom_schema_rejects_wrong_types_unknown_fields_and_bounds(self):
        schema = self.task["output_schema"]
        output = self.task["offline_response"]["output"]
        cases = [dict(output, unexpected=True), dict(output, recommendation="not-an-enum-choice"),
                 dict(output, editor_note=""), dict(output, editor_note="x" * 1001),
                 dict(output, custom={"needs_specialist": 1, "tags": []}),
                 dict(output, custom={"needs_specialist": True, "tags": ["x"] * 4}),
                 dict(output, custom={"needs_specialist": True, "tags": [42]})]
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_output(candidate, schema)

    def test_unsupported_schema_keywords_and_unbounded_fields_are_rejected(self):
        for keyword in ("$ref", "oneOf", "pattern", "format", "default"):
            task = copy.deepcopy(self.task)
            task["output_schema"]["properties"]["editor_note"][keyword] = "unsupported"
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                validate_task(task)
        for mutate in (lambda s: s.update(additionalProperties=True),
                       lambda s: s["properties"]["editor_note"].pop("maxLength"),
                       lambda s: s["properties"]["custom"]["properties"]["tags"].update(maxItems=101)):
            task = copy.deepcopy(self.task)
            mutate(task["output_schema"])
            with self.assertRaises(ValueError):
                validate_task(task)

    def test_accuracy_is_explicit_and_free_text_has_no_invented_metric(self):
        first = self.task["offline_response"]["output"]
        second = dict(first, recommendation="decline")
        self.assertEqual(evaluate_metrics(self.task, [first, second], [first, first]), {"recommendation": {"accuracy": 0.5}})
        task = dict(self.task, metrics=[])
        self.assertEqual(evaluate_metrics(task, [first], [second]), {})
        for config in ({"field": "editor_note", "metric": "accuracy"},
                       {"field": "recommendation", "metric": "MAE"},
                       {"field": "custom", "metric": "text_quality"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_task(dict(self.task, metrics=[config]))

    def test_numeric_fields_are_optional_and_named_metrics_use_declared_scale(self):
        task = {"schema_version": 1, "task_id": "synthetic-custom-number", "version": "1", "instructions": "Synthetic",
                "output_schema": {"type": "object", "properties": {
                    "priority": {"type": "number", "minimum": -5, "maximum": 5},
                    "section_count": {"type": "integer", "minimum": 0, "maximum": 30}},
                    "additionalProperties": False},
                "metrics": [{"field": "priority", "metric": "MAE"}, {"field": "priority", "metric": "RMSE"}]}
        self.assertEqual(validate_output({}, task["output_schema"]), {})
        metrics = evaluate_metrics(task, [{"priority": 0}, {"priority": 2}], [{"priority": -1}, {"priority": 0}])
        self.assertEqual(metrics["priority"]["MAE"], 1.5)
        self.assertAlmostEqual(metrics["priority"]["RMSE"], math.sqrt(2.5))
        for value in (True, float("nan"), float("inf"), 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_output({"priority": value}, task["output_schema"])
        with self.assertRaises(ValueError):
            validate_output({"section_count": 1.5}, task["output_schema"])
        with self.assertRaises(ValueError):
            evaluate_metrics(task, [{}], [{}])

    def test_local_backend_receives_only_validated_request_and_schema_not_offline_answer(self):
        config = {"mode": "local", "adapter": "openai_compatible", "endpoint": "http://127.0.0.1:1234/v1/chat/completions", "model": "institution-selected"}
        model = Mock()
        model.generate_json.return_value = self.task["offline_response"]["output"]
        with patch("journal_prefs.assessment.make_model", return_value=model) as make:
            result = assess(self.task, self.request, config)
        make.assert_called_once_with(config, allow_cloud=False)
        payload, instructions = model.generate_json.call_args.args
        self.assertEqual(payload["target"], self.request["target"])
        self.assertEqual(payload["history"], self.request["history"])
        self.assertNotIn("offline_response", payload)
        self.assertNotIn("metrics", payload)
        self.assertIn("untrusted data", instructions)
        self.assertFalse(result["synthetic_output"])

    def test_existing_local_cloud_and_offline_privacy_gates_remain_active(self):
        local = {"mode": "local", "adapter": "openai_compatible", "endpoint": "http://remote.invalid/v1/chat/completions", "model": "chosen"}
        with self.assertRaises(ValueError):
            assess(self.task, self.request, local)
        cloud = dict(local, mode="cloud", endpoint="https://provider.invalid/v1/chat/completions",
                     api_key_env="SYNTHETIC_UNUSED_KEY_NAME", allow_private_data_transfer=True)
        with self.assertRaisesRegex(ValueError, "Cloud is disabled"):
            assess(self.task, self.request, cloud)
        with self.assertRaises(ValueError):
            assess(self.task, self.request, {"mode": "offline", "adapter": "mock", "fallback": "cloud"})


if __name__ == "__main__":
    unittest.main()
