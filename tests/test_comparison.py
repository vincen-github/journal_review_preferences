import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_prefs.comparison import (SyntheticRuleGenerator, compare_inputs, evaluate_comparison,
                                      normalize_history, run_preference_demo)
from journal_prefs.data import load_history, read_json, validate_rubric
from journal_prefs.engine import split_history
from journal_prefs.models import MockModel
from journal_prefs.preferences import EvidenceRules

ROOT = Path(__file__).resolve().parent.parent


class RecordingModel(MockModel):
    def __init__(self):
        self.requests = []

    def review(self, request):
        self.requests.append(copy.deepcopy(request))
        return super().review(request)


class ComparisonTests(unittest.TestCase):
    def test_configured_custom_algorithm_can_replace_preference_scaffold(self):
        from examples.custom_preferences import ConfiguredPreferences
        plugin = ConfiguredPreferences(["SYNTHETIC configured criterion"])
        state = plugin.freeze(plugin.fit([{"paper_id": "learning"}]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            plugin.save(state, path)
            self.assertEqual(plugin.condition(plugin.load(path), {"paper_id": "new"}),
                             [{"rule_id": "1", "principle": "SYNTHETIC configured criterion"}])

    def setUp(self):
        self.rubric = validate_rubric(read_json(ROOT / "examples/rubric.json"))
        rows = load_history(ROOT / "examples/history.synthetic.jsonl", self.rubric)
        train, validation, test = split_history(rows, "2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z")
        self.rows = rows
        self.algorithm = EvidenceRules(SyntheticRuleGenerator())
        self.artifact = self.algorithm.freeze(self.algorithm.refine(self.algorithm.fit(normalize_history(train)), normalize_history(validation)))
        self.inputs = [{"paper_id": r["manuscript_id"], "title": r["title"], "text": r["text"]} for r in test]
        self.labels = [{"paper_id": r["manuscript_id"], "individual_scores": [x["score"] for x in r["reviews"]],
                        "accepted": r["decision"]["label"] == self.rubric["positive_decision"]} for r in test]
        self.guide = {"text": "SYNTHETIC GUIDELINE", "source": "authored fixture", "kind": "synthetic"}

    def test_three_arms_change_only_context_and_do_not_receive_labels(self):
        model = RecordingModel()
        with patch("socket.socket", side_effect=AssertionError("Offline")):
            result = compare_inputs(self.inputs, self.rubric, self.guide, self.algorithm, self.artifact, model, {"model": "same-fixture"}, True)
        self.assertEqual(len(model.requests), 12)
        for request in model.requests:
            self.assertEqual(set(request["paper"]), {"paper_id", "title", "text"})
            self.assertEqual(request["historical_examples"], [])
            self.assertEqual(request["rubric"], self.rubric)
            self.assertNotIn("score", request["evaluation_context"])
        self.assertFalse(result["blind_evaluation_claim"])
        self.assertTrue(result["prior_label_exposure"])
        for paper_id in result["paper_ids"]:
            arms = [r for r in model.requests if r["paper"]["paper_id"] == paper_id]
            self.assertEqual(sum(r["evaluation_context"]["guideline"] is None for r in arms), 1)
            self.assertEqual(sum(bool(r["evaluation_context"]["rules"]) for r in arms), 1)

    def test_evaluation_label_changes_do_not_change_completed_predictions(self):
        result = compare_inputs(self.inputs, self.rubric, self.guide, self.algorithm, self.artifact, MockModel(), {}, True)
        frozen = copy.deepcopy(result)
        before = evaluate_comparison(result, self.labels, 5)
        poisoned = copy.deepcopy(self.labels)
        for row in poisoned:
            row["individual_scores"] = [1]
            row["accepted"] = not row["accepted"]
        after = evaluate_comparison(result, poisoned, 5)
        self.assertEqual(result, frozen)
        self.assertNotEqual(before["metrics"], after["metrics"])
        self.assertEqual(before["learned_minus_guideline_mae"], 0)

    def test_labelled_or_duplicate_targets_are_refused_before_model_calls(self):
        for inputs in ([dict(self.inputs[0], accepted=True)], [self.inputs[0], self.inputs[0]]):
            model = RecordingModel()
            with self.assertRaises(ValueError):
                compare_inputs(inputs, self.rubric, self.guide, self.algorithm, self.artifact, model, {}, True)
            self.assertEqual(model.requests, [])

    def test_full_synthetic_interface_demo_is_offline_and_honest_about_zero_difference(self):
        with tempfile.TemporaryDirectory() as directory, patch("socket.socket", side_effect=AssertionError("Offline")):
            output = Path(directory) / "new"
            report = run_preference_demo(self.rows, self.rubric, output)
            self.assertEqual(report["learned_minus_guideline_mae"], 0)
            self.assertIn("SYNTHETIC", report["notice"])
            self.assertTrue((output / "preferences.json").exists())
            with self.assertRaises(ValueError):
                run_preference_demo(self.rows, self.rubric, output)
