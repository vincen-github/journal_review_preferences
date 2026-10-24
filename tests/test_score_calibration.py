"""Synthetic-only calibration checks; no models, credentials, or networking."""

import copy
import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from journal_prefs.data import fingerprint
from journal_prefs.score_calibration import (evaluate_test, fit_state, load_state,
                                             save_state, validate_state)


def cohort(prefix, predictions=(2, 4, 6), targets=(3, 5, 7)):
    raw = [{"paper_id": prefix + str(i), "score": value} for i, value in enumerate(predictions)]
    labels = [{"paper_id": prefix + str(i), "individual_scores": [value]} for i, value in enumerate(targets)]
    return raw, labels


def fit_affine_example():
    train, training_labels = cohort("train")
    validation, validation_labels = cohort("validation", (3, 5), (4, 6))
    return fit_state(train, training_labels, validation, validation_labels,
                     provenance={"scoring_signature": "SYNTHETIC_SIGNATURE", "fixture": True})


def reseal(state):
    state["state_sha256"] = fingerprint({key: value for key, value in state.items() if key != "state_sha256"})
    return state


class ScoreCalibrationTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.guards = []
        for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo",
                       "urllib.request.urlopen", "http.client.HTTPConnection.connect",
                       "http.client.HTTPSConnection.connect"):
            self.guards.append(stack.enter_context(patch(target, side_effect=AssertionError("Network forbidden"))))

    def tearDown(self):
        for guard in self.guards:
            guard.assert_not_called()

    def test_known_positive_affine_is_recovered_from_training_and_selected_on_validation(self):
        state = fit_affine_example()
        self.assertTrue(state["frozen"])
        self.assertEqual(state["algorithm"], "absolute_score_calibration")
        self.assertEqual(state["calibration"]["kind"], "affine")
        self.assertAlmostEqual(state["calibration"]["slope"], 1)
        self.assertAlmostEqual(state["calibration"]["intercept"], 1)
        self.assertEqual(state["candidate_results"][0]["calibration"]["kind"], "identity")
        self.assertEqual(state["candidate_results"][1]["validation"]["mae"], 0)
        self.assertEqual(state, validate_state(state))
        self.assertEqual(state["provenance"]["scoring_signature"], "SYNTHETIC_SIGNATURE")

    def test_validation_selects_identity_when_train_affine_is_worse(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3, 5), (3, 5))
        state = fit_state(train, labels, validation, validation_labels)
        self.assertEqual(state["calibration"]["kind"], "identity")
        self.assertEqual(state["candidate_results"][0]["validation"]["mae"], 0)
        self.assertEqual(state["candidate_results"][1]["validation"]["mae"], 1)

    def test_validation_ties_favor_identity_including_numerical_roundoff(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3, 5), (3.5, 5.5))
        state = fit_state(train, labels, validation, validation_labels)
        self.assertEqual(state["calibration"]["kind"], "identity")
        train, labels = cohort("train", (1.1, 2.2, 3.3), (1.1, 2.2, 3.3))
        validation, validation_labels = cohort("validation", (4.4,), (4.4,))
        self.assertEqual(fit_state(train, labels, validation, validation_labels)["calibration"]["kind"], "identity")

    def test_clipping_is_used_during_selection_and_test(self):
        train, labels = cohort("train", (2, 3, 4), (4, 6, 8))
        validation, validation_labels = cohort("validation", (1, 6), (2, 10))
        state = fit_state(train, labels, validation, validation_labels)
        self.assertEqual(state["calibration"]["kind"], "affine")
        self.assertEqual(state["candidate_results"][1]["validation"]["mae"], 0)
        test, test_labels = cohort("test", (9,), (10,))
        result = evaluate_test(state, test, test_labels)
        self.assertEqual(result["per_paper"][0]["calibrated_score"], 10)
        # A positive affine with a negative intercept exercises lower clipping.
        train, labels = cohort("train", (3, 4, 5), (1, 3, 5))
        validation, validation_labels = cohort("validation", (2, 6), (1, 7))
        state = fit_state(train, labels, validation, validation_labels)
        test, test_labels = cohort("test", (1,), (1,))
        self.assertEqual(evaluate_test(state, test, test_labels)["per_paper"][0]["calibrated_score"], 1)

    def test_unavailable_affine_uses_identity_for_constant_negative_or_too_small_train(self):
        for raw, target in (((4, 4, 4), (2, 4, 8)), ((2, 4, 6), (8, 6, 4)),
                            ((2, 4, 6), (5, 5, 5)), ((2, 4), (3, 5))):
            train, labels = cohort("train", raw, target)
            validation, validation_labels = cohort("validation", (4,), (5,))
            state = fit_state(train, labels, validation, validation_labels)
            with self.subTest(raw=raw, target=target):
                self.assertEqual(state["calibration"]["kind"], "identity")
                self.assertEqual(len(state["candidate_results"]), 1)

    def test_prediction_and_label_orders_align_by_unique_ids_and_hashes_are_stable(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3, 5), (4, 6))
        first = fit_state(train, labels, validation, validation_labels)
        second = fit_state(list(reversed(train)), labels[1:] + labels[:1],
                           list(reversed(validation)), validation_labels)
        self.assertEqual(first, second)
        self.assertEqual(first["train_ids"], sorted(p["paper_id"] for p in train))
        test, test_labels = cohort("test", (2, 7), (3, 8))
        self.assertEqual(evaluate_test(first, test, test_labels),
                         evaluate_test(first, list(reversed(test)), test_labels))

    def test_targets_use_individual_mean_and_preserve_dispersion(self):
        state = fit_affine_example()
        predictions = [{"paper_id": "test0", "score": 4}]
        labels = [{"paper_id": "test0", "individual_scores": [1, 5, 9], "accepted": False}]
        result = evaluate_test(state, predictions, labels)
        row = result["per_paper"][0]
        self.assertEqual(row["human_mean"], 5)
        self.assertEqual(row["individual_scores"], [1, 5, 9])
        self.assertAlmostEqual(row["score_std_population"], math.sqrt(32 / 3))
        self.assertEqual((row["score_min"], row["score_max"]), (1, 9))
        self.assertEqual(result["raw"], {"n": 1, "mae": 1, "rmse": 1, "signed_bias": -1})
        self.assertEqual(result["calibrated"], {"n": 1, "mae": 0, "rmse": 0, "signed_bias": 0})
        self.assertEqual(result["improvement"], {"mae": 1, "rmse": 1})

    def test_zero_and_negative_improvement_are_reported_without_suppression(self):
        state = fit_affine_example()
        predictions, labels = cohort("test", (4,), (4,))
        result = evaluate_test(state, predictions, labels)
        self.assertEqual(result["improvement"], {"mae": -1, "rmse": -1})
        train, training_labels = cohort("train")
        validation, validation_labels = cohort("validation", (3,), (3,))
        identity = fit_state(train, training_labels, validation, validation_labels)
        self.assertEqual(evaluate_test(identity, predictions, labels)["improvement"], {"mae": 0, "rmse": 0})

    def test_held_out_label_poisoning_changes_metrics_only_not_state_or_predictions(self):
        state = fit_affine_example()
        snapshot = copy.deepcopy(state)
        predictions, labels = cohort("test", (2, 5), (3, 6))
        original_predictions = copy.deepcopy(predictions)
        first = evaluate_test(state, predictions, labels)
        poisoned = [{"paper_id": label["paper_id"], "individual_scores": [10], "accepted": True} for label in labels]
        second = evaluate_test(state, predictions, poisoned)
        self.assertEqual(state, snapshot)
        self.assertEqual(predictions, original_predictions)
        self.assertEqual([r["calibrated_score"] for r in first["per_paper"]],
                         [r["calibrated_score"] for r in second["per_paper"]])
        self.assertEqual(first["state_sha256"], second["state_sha256"])
        self.assertNotEqual(first["calibrated"]["mae"], second["calibrated"]["mae"])

    def test_acceptance_labels_do_not_determine_calibration_or_score_metrics(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3, 5), (4, 6))
        first = fit_state(train, [dict(row, accepted=True) for row in labels], validation, validation_labels)
        second = fit_state(train, [dict(row, accepted=False) for row in labels], validation, validation_labels)
        self.assertEqual(first["calibration"], second["calibration"])
        self.assertEqual(first["candidate_results"], second["candidate_results"])
        self.assertNotEqual(first["train_labels_sha256"], second["train_labels_sha256"])
        predictions, test_labels = cohort("test", (3,), (5,))
        self.assertEqual(evaluate_test(first, predictions, [dict(row, accepted=True) for row in test_labels]),
                         evaluate_test(first, predictions, [dict(row, accepted=False) for row in test_labels]))

    def test_train_validation_and_test_groups_are_disjoint(self):
        train, labels = cohort("train")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_state(train, labels, train, labels)
        state = fit_affine_example()
        for identifier in ("train0", "validation0"):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "disjoint"):
                evaluate_test(state, [{"paper_id": identifier, "score": 5}],
                              [{"paper_id": identifier, "individual_scores": [5]}])

    def test_duplicate_mismatched_and_empty_cohorts_are_rejected(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3,), (4,))
        invalid = [(train + train[:1], labels), (train, labels + labels[:1]),
                   (train, labels[:-1]), ([], []), (None, labels),
                   ([dict(train[0], paper_id=" ")], labels[:1]), ([None], labels[:1])]
        for raw, target in invalid:
            with self.subTest(raw=raw, target=target), self.assertRaises(ValueError):
                fit_state(raw, target, validation, validation_labels)

    def test_bad_scores_and_missing_individual_recommendations_fail(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3,), (4,))
        for value in (None, True, "5", 0, 11, float("nan"), float("inf"), 10 ** 1000):
            bad_prediction = copy.deepcopy(train)
            bad_prediction[0]["score"] = value
            bad_label = copy.deepcopy(labels)
            bad_label[0]["individual_scores"] = [value]
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                fit_state(bad_prediction, labels, validation, validation_labels)
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                fit_state(train, bad_label, validation, validation_labels)
        for scores in (None, [], "5", [True]):
            bad = copy.deepcopy(labels)
            bad[0]["individual_scores"] = scores
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                fit_state(train, bad, validation, validation_labels)
        with self.assertRaises(ValueError):
            fit_state(train, [dict(row, accepted="yes") for row in labels], validation, validation_labels)

    def test_only_schema_fields_are_retained_and_input_mutation_cannot_change_state(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3,), (4,))
        provenance = {"scoring_signature": "SYNTHETIC", "nested": {"prompt_version": 1}}
        state = fit_state([dict(row, unrelated="DROP") for row in train],
                          [dict(row, manuscript_text="DROP") for row in labels], validation, validation_labels, provenance)
        canonical = fit_state(train, labels, validation, validation_labels, provenance)
        self.assertEqual(state, canonical)
        provenance["nested"]["prompt_version"] = 99
        train[0]["score"] = 10
        labels[0]["individual_scores"].append(10)
        self.assertEqual(state, canonical)
        self.assertNotIn("DROP", repr(state))

    def test_state_save_load_and_checksum_validation(self):
        state = fit_affine_example()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            save_state(state, path)
            loaded = load_state(path)
            self.assertEqual(state, loaded)
            path.write_text(path.read_text(encoding="utf-8").replace("SYNTHETIC_SIGNATURE", "ALTERED_SIGNATURE"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_state(path)
        loaded["calibration"]["intercept"] = 9
        self.assertNotEqual(loaded, state)

    def test_state_types_and_partition_rules_are_checked_even_with_recomputed_checksum(self):
        state = fit_affine_example()
        mutations = [lambda s: s.update(schema_version=True), lambda s: s.update(frozen=False),
                     lambda s: s.update(score_range=[True, 10]), lambda s: s.update(algorithm="other"),
                     lambda s: s.update(provenance=["bad"]), lambda s: s.update(train_sha256="invalid"),
                     lambda s: s.update(train_ids=["train0", "train0"]),
                     lambda s: s.update(validation_ids=["train0"]),
                     lambda s: s["calibration"].update(slope=True),
                     lambda s: s["candidate_results"][0]["validation"].update(mae=float("inf")),
                     lambda s: s["candidate_results"][0]["validation"].update(n=True),
                     lambda s: s["candidate_results"].reverse(),
                     lambda s: s.update(calibration={"kind": "identity", "slope": 1, "intercept": 0})]
        for mutate in mutations:
            bad = copy.deepcopy(state)
            mutate(bad)
            # NaN/Infinity cannot be checksummed as valid JSON in the first place.
            try:
                reseal(bad)
            except ValueError:
                pass
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_state(bad)

    def test_invalid_provenance_is_rejected(self):
        train, labels = cohort("train")
        validation, validation_labels = cohort("validation", (3,), (4,))
        for provenance in (["bad"], {"number": float("nan")}, {1: "non-string-key"}, {"tuple": (1, 2)}):
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                fit_state(train, labels, validation, validation_labels, provenance)


if __name__ == "__main__":
    unittest.main()
