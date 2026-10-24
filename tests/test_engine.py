import copy
import unittest
from pathlib import Path

from journal_prefs.data import (available_at, load_history, read_json, timestamp,
                                validate_history, validate_paper, validate_rubric)
from journal_prefs.engine import (accepted_percentile, fit_affine, retrieve,
                                  review_with_state, run_backtest, split_history)
from journal_prefs.models import MockModel

ROOT = Path(__file__).resolve().parent.parent
CONFIG = {"mode": "offline", "adapter": "mock"}
TRAIN = "2024-01-01T00:00:00Z"
VALIDATION = "2025-01-01T00:00:00Z"


class CaptureModel(MockModel):
    def __init__(self):
        self.requests = []

    def review(self, request):
        self.requests.append(copy.deepcopy(request))
        return super().review(request)


class ChronologyTests(unittest.TestCase):
    def setUp(self):
        self.rubric = validate_rubric(read_json(ROOT / "examples/rubric.json"))
        self.rows = load_history(ROOT / "examples/history.synthetic.jsonl", self.rubric)

    def run_demo(self, rows=None, **kwargs):
        return run_backtest(rows or self.rows, self.rubric, MockModel(), CONFIG, TRAIN, VALIDATION, **kwargs)

    def test_all_adapter_requests_exclude_heldout_labels(self):
        model = CaptureModel()
        state, report, predictions = run_backtest(self.rows, self.rubric, model, CONFIG, TRAIN, VALIDATION)
        self.assertEqual([len(predictions[s]) for s in ("train", "validation", "test")], [8, 4, 4])
        train = self.rows[:8]
        lookup = {r["manuscript_id"]: r for r in train}
        for request in model.requests:
            self.assertNotIn("reviews", request["paper"])
            self.assertNotIn("decision", request["paper"])
            self.assertNotIn("score", request["paper"])
            for example in request["historical_examples"]:
                self.assertIn(example["manuscript_id"], state["training_ids"])
                self.assertNotEqual(example["manuscript_id"], request["paper"]["manuscript_id"])
                self.assertLess(available_at(lookup[example["manuscript_id"]]), timestamp(request["paper"]["submitted_at"]))
        self.assertEqual(report["frozen_state_sha256_before_test"], report["frozen_state_sha256_after_test"])

    def test_test_label_mutations_cannot_change_predictions_or_state(self):
        baseline = self.run_demo()
        mutated = copy.deepcopy(self.rows)
        for row in mutated[12:]:
            row["reviews"][0]["text"] = "HELD_OUT_SECRET_REVIEW_DO_NOT_READ"
            row["reviews"][0]["score"] = 1.0
            row["decision"]["label"] = "decline"
        state, report, predictions = self.run_demo(mutated)
        self.assertEqual(state, baseline[0])
        self.assertEqual(predictions, baseline[2])
        self.assertNotEqual(report["metrics"]["untouched_test"], baseline[1]["metrics"]["untouched_test"])

    def test_future_train_labels_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["reviews"][0]["available_at"] = TRAIN
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            self.run_demo(rows)

    def test_future_validation_labels_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[8]["decision"]["available_at"] = VALIDATION
        with self.assertRaisesRegex(ValueError, "VALIDATION"):
            self.run_demo(rows)

    def test_manuscript_versions_cannot_cross_cohorts(self):
        rows = copy.deepcopy(self.rows)
        rows[12]["manuscript_id"] = rows[0]["manuscript_id"]
        rows[12]["version_id"] = "v2-after-review"
        with self.assertRaisesRegex(ValueError, "ONE"):
            self.run_demo(rows)

    def test_historical_availability_must_strictly_precede_query(self):
        query = copy.deepcopy(self.rows[1])
        prior = copy.deepcopy(self.rows[0])
        prior["decision"]["available_at"] = query["submitted_at"]
        self.assertEqual(retrieve(query, [prior]), [])

    def test_split_cutoffs_are_exclusive_and_ordered(self):
        train, validation, test = split_history(self.rows, TRAIN, VALIDATION)
        self.assertTrue(all(timestamp(r["submitted_at"]) < timestamp(TRAIN) for r in train))
        self.assertTrue(all(timestamp(TRAIN) <= timestamp(r["submitted_at"]) < timestamp(VALIDATION) for r in validation))
        self.assertTrue(all(timestamp(r["submitted_at"]) >= timestamp(VALIDATION) for r in test))
        with self.assertRaises(ValueError):
            split_history(self.rows, VALIDATION, TRAIN)

    def test_train_affine_fit_and_validation_selection_are_explicit(self):
        state, report, _ = self.run_demo()
        self.assertEqual(len(report["candidate_results"]), 2)
        self.assertIn(state["calibration"], [x["parameters"] for x in report["candidate_results"]])
        self.assertEqual(report["metrics"]["untouched_test"]["n"], 4)
        self.assertTrue(state["synthetic_model"])
        self.assertIn("SYNTHETIC", report["notice"])
        state_none, _, _ = self.run_demo(calibration="none")
        self.assertEqual(state_none["calibration"]["kind"], "none")

    def test_affine_fit_recovers_known_score_scale_and_handles_degeneracy(self):
        rows = [{"reviews": [{"score": y}]} for y in (3, 5, 7)]
        fit = fit_affine([{"raw_score": x} for x in (1, 2, 3)], rows)
        self.assertAlmostEqual(fit["slope"], 2)
        self.assertAlmostEqual(fit["intercept"], 1)
        self.assertIsNone(fit_affine([{"raw_score": 1}] * 3, rows))

    def test_accepted_quantile_uses_training_only(self):
        self.rubric["threshold_mode"] = "accepted_quantile"
        state, _, _ = self.run_demo()
        accepted = [r["reviews"][0]["score"] for r in self.rows[:8] if r["decision"]["label"] == "accept"]
        self.assertEqual(state["threshold"], accepted_percentile(accepted, 0.25))
        self.assertEqual(accepted_percentile([2, 4, 6, 8, 10], 0.25), 4)

    def test_state_is_tied_to_rubric_model_and_training_corpus(self):
        state, _, _ = self.run_demo()
        paper = read_json(ROOT / "examples/submission.synthetic.json")
        result = review_with_state(paper, self.rows, self.rubric, MockModel(), CONFIG, state)
        self.assertTrue(result["human_review_required"])
        changed = copy.deepcopy(self.rubric)
        changed["criteria"][0]["weight"] = 0.7
        with self.assertRaisesRegex(ValueError, "rubric changed"):
            review_with_state(paper, self.rows, changed, MockModel(), CONFIG, state)
        changed_history = copy.deepcopy(self.rows)
        changed_history[0]["text"] += " Changed after fitting."
        with self.assertRaisesRegex(ValueError, "training corpus changed"):
            review_with_state(paper, changed_history, self.rubric, MockModel(), CONFIG, state)

    def test_invalid_state_parameters_are_rejected(self):
        state, _, _ = self.run_demo()
        paper = read_json(ROOT / "examples/submission.synthetic.json")
        modifications = [{"threshold": 100}, {"top_k": -1}, {"top_k": True}, {"calibration": None},
                         {"calibration": {"kind": "affine", "slope": -1, "intercept": 0}},
                         {"calibration": {"kind": "affine", "slope": float("nan"), "intercept": 0}},
                         {"calibration": {"kind": "none", "slope": 1, "intercept": 2}},
                         {"training_ids": ["unexpected"]}]
        for change in modifications:
            with self.subTest(change=change), self.assertRaises(ValueError):
                review_with_state(paper, self.rows, self.rubric, MockModel(), CONFIG, dict(state, **change))

    def test_previously_evaluated_paper_cannot_be_reused_as_fresh_review(self):
        state, _, _ = self.run_demo()
        with self.assertRaisesRegex(ValueError, "already used"):
            review_with_state(self.rows[-1], self.rows, self.rubric, MockModel(), CONFIG, state)

    def test_schema_rejects_impossible_times_and_nonfinite_scores(self):
        invalid = copy.deepcopy(self.rows[0])
        invalid["reviews"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            validate_history(invalid, self.rubric)
        for field in ("text_available_at", "version_time"):
            invalid = copy.deepcopy(self.rows[0])
            invalid[field] = "2030-01-01T00:00:00Z"
            with self.assertRaises(ValueError):
                validate_paper(invalid, self.rubric)
        invalid = copy.deepcopy(self.rows[0])
        invalid["text_available_at"] = invalid["submitted_at"]
        with self.assertRaises(ValueError):
            validate_paper(invalid, self.rubric)

    def test_configured_arbitrary_venue_and_institution_rights(self):
        rubric = copy.deepcopy(self.rubric)
        rubric["venue_id"] = "a-humanities-venue"
        row = copy.deepcopy(self.rows[0])
        row["venue_id"] = rubric["venue_id"]
        self.assertEqual(validate_history(row, rubric), row)
        row["rights_confirmed"] = False
        with self.assertRaises(ValueError):
            validate_history(row, rubric)


if __name__ == "__main__":
    unittest.main()
