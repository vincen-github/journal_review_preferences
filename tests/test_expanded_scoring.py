import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_prefs.data import fingerprint, read_json, write_json
from journal_prefs.expanded_scoring import fit_learning_curve, lock_scoring_inputs
from journal_prefs.expanded_evaluation import release_test, verify_test_release, evaluate_expanded

PROMPT = Path(__file__).resolve().parent.parent / "prompts/iclr2017_official_form.v1.json"


class CurveTests(unittest.TestCase):
    def fixture(self, root, offset=1):
        ids = ["train" + str(i) for i in range(60)]
        predictions = [{"paper_id": i, "score": 2 + n % 5} for n, i in enumerate(ids)]
        labels = [{"paper_id": p["paper_id"], "individual_scores": [p["score"] + offset]} for p in predictions]
        validation = [{"paper_id": "validation" + str(i), "score": 2 + i % 5} for i in range(20)]
        val_labels = [{"paper_id": p["paper_id"], "individual_scores": [p["score"] + offset]} for p in validation]
        groups = {"train": ids, "validation": [p["paper_id"] for p in validation], "test": ["test" + str(i) for i in range(20)]}
        partition = {"ids": groups, "mandatory_ids": ids[:18], "nested_train_ids": {str(n): ids[:n] for n in (20, 40, 60)}, "partition_sha256": fingerprint(groups)}
        plan = {"partition_sha256": fingerprint(groups), "settings": {"model": "SYNTHETIC"}, "instructions_sha256": "SYNTHETIC", "schema_sha256": "SYNTHETIC",
                "jobs": [{"group": "test", "run_directory": "model_runs/test_a"}]}
        write_json(root / "partition.json", partition)
        write_json(root / "scoring_plan.json", plan)
        for directory in ("combined_train", "combined_validation", "cohort"):
            (root / directory).mkdir()
        for path, value in [("combined_train/predictions.json", predictions), ("combined_validation/predictions.json", validation),
                             ("cohort/train.labels.json", labels), ("cohort/validation.labels.json", val_labels)]:
            write_json(root / path, value)
        for group, values in (("train", predictions), ("validation", validation)):
            manifest = dict(plan, status="completed_predictions_frozen_before_label_evaluation", predictions_sha256=fingerprint(values))
            write_json(root / ("combined_" + group) / "manifest.json", manifest)
            write_json(root / "cohort" / (group + ".inputs.json"), [{"paper_id": p["paper_id"]} for p in values])
        write_json(root / "cached_predictions.json", [])
        write_json(root / "cache_ledger.json", [])
        lock_scoring_inputs(root, None, PROMPT)

    def test_curve_freezes_before_test_and_tie_prefers_smaller_n(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            def guarded(path):
                self.assertNotIn("test", Path(path).name)
                return read_json(path)
            with patch("journal_prefs.expanded_scoring.read_json", side_effect=guarded):
                result = fit_learning_curve(root)
            self.assertEqual(result["selected_n"], 20)
            self.assertFalse(result["new_test_scoring_started"])
            self.assertTrue(all(c["selected_calibration"]["kind"] == "affine" for c in result["candidates"]))
            self.assertEqual([len(read_json(root / ("calibration_n" + str(n) + ".json"))["train_ids"]) for n in (20, 40, 60)], [20, 40, 60])
            with self.assertRaises(ValueError):
                fit_learning_curve(root)

    def test_identity_is_retained_when_no_adjustment_is_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, offset=0)
            result = fit_learning_curve(root)
            self.assertEqual(result["selected_n"], 20)
            self.assertTrue(all(c["selected_calibration"]["kind"] == "identity" for c in result["candidates"]))

    def test_changed_partition_scores_or_cache_cannot_be_frozen(self):
        for filename in ("partition.json", "combined_train/predictions.json", "cached_predictions.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                value = read_json(root / filename)
                if filename == "partition.json":
                    value["nested_train_ids"]["20"] = value["ids"]["train"][40:60]
                elif filename.endswith("predictions.json") and isinstance(value, list) and value:
                    value[0]["score"] += 1
                else:
                    value.append({"paper_id": "changed"})
                write_json(root / filename, value)
                with self.assertRaises(ValueError):
                    fit_learning_curve(root)
                self.assertFalse((root / "selection.json").exists())

    def test_test_release_requires_selection_and_prevents_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with self.assertRaises(FileNotFoundError):
                release_test(root)
            fit_learning_curve(root)
            release_test(root)
            verify_test_release(root)
            with self.assertRaises(ValueError):
                release_test(root)
            selection = read_json(root / "selection.json")
            selection["selected_n"] = 40
            write_json(root / "selection.json", selection)
            with self.assertRaises(ValueError):
                verify_test_release(root)

    def test_test_scoring_before_fit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / "model_runs/test_a").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "after test"):
                fit_learning_curve(root)

    def test_evaluation_uses_only_selected_state_and_training_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            fit_learning_curve(root)
            release_test(root)
            predictions = [{"paper_id": "test" + str(i), "score": 3} for i in range(20)]
            labels = [{"paper_id": p["paper_id"], "individual_scores": [4, 4]} for p in predictions]
            (root / "combined_test").mkdir()
            write_json(root / "combined_test/predictions.json", predictions)
            plan = read_json(root / "scoring_plan.json")
            write_json(root / "combined_test/manifest.json", dict(plan, status="completed_predictions_frozen_before_label_evaluation", predictions_sha256=fingerprint(predictions)))
            write_json(root / "cohort/test.labels.json", labels)
            result = evaluate_expanded(root)
            self.assertEqual(result["selected_n"], 20)
            self.assertEqual(result["raw"]["mae"], 1)
            self.assertAlmostEqual(result["calibrated"]["mae"], 0)
            self.assertEqual(result["training_mean_baseline"]["n_train"], 20)
            with self.assertRaises(ValueError):
                evaluate_expanded(root)
