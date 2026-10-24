import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_prefs.data import fingerprint, read_json, write_json
from journal_prefs.nonlinear_scores import fit_score_model, predict_score_model

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("cached_comparison", ROOT / "tools/compare_cached_scores.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class NonlinearTests(unittest.TestCase):
    def test_quadratic_recovers_curvature_with_unpenalized_intercept(self):
        x = list(range(1, 11)) * 2
        y = [4 + (v - 5.5) / 4.5 + 2 * ((v - 5.5) / 4.5) ** 2 for v in x]
        model = fit_score_model("quadratic_ridge", x, y, alpha=1e-8)
        self.assertAlmostEqual(model["coefficients"][0], 1, places=6)
        self.assertAlmostEqual(model["coefficients"][1], 2, places=6)
        self.assertLess(max(abs(a - b) for a, b in zip(predict_score_model(model, x), y)), 1e-6)
        shifted = fit_score_model("quadratic_ridge", x, [v + 1 for v in y], alpha=1e-8)
        self.assertAlmostEqual(shifted["intercept"] - model["intercept"], 1)
        regularized = fit_score_model("quadratic_ridge", x, y, alpha=100)
        self.assertLess(sum(c * c for c in regularized["coefficients"]), sum(c * c for c in model["coefficients"]))

    def test_tree_leaf_means_depth_and_minimum_leaf_are_enforced(self):
        x, y = list(range(1, 11)), [2] * 5 + [8] * 5
        tree = fit_score_model("tree", x, y, max_depth=1, min_leaf=2)
        self.assertEqual(tree["tree"]["threshold"], 5.5)
        self.assertEqual(tree["actual_depth"], 1)
        self.assertEqual(predict_score_model(tree, x), y)
        shallow = fit_score_model("tree", x, y, max_depth=1, min_leaf=6)
        self.assertEqual(shallow["leaves"], 1)
        self.assertEqual(predict_score_model(shallow, x), [5] * 10)
        self.assertEqual(predict_score_model(json.loads(json.dumps(tree)), x), y)

    def test_degenerate_or_invalid_features_do_not_silently_fit(self):
        with self.assertRaises(ValueError):
            fit_score_model("affine", [5, 5, 5], [3, 5, 8])
        with self.assertRaises(ValueError):
            fit_score_model("tree", [1, 2], [1, 2], max_depth=10, min_leaf=2)
        for x in ([float("nan")], [True], [11]):
            with self.assertRaises(ValueError):
                fit_score_model("mean", x, [5])

    def fixture(self, root):
        source = root / "source"
        (source / "cohort").mkdir(parents=True)
        groups = {"train": ["train" + str(i) for i in range(60)], "validation": ["val" + str(i) for i in range(20)], "test": ["test" + str(i) for i in range(20)]}
        write_json(source / "partition.json", {"ids": groups, "partition_sha256": fingerprint(groups)})
        for group, ids in groups.items():
            directory = source / ("combined_" + group)
            directory.mkdir()
            predictions = [{"paper_id": i, "score": 2 + n % 7, "review": "SYNTHETIC unused text"} for n, i in enumerate(ids)]
            labels = [{"paper_id": p["paper_id"], "individual_scores": [p["score"] + 1]} for p in predictions]
            write_json(directory / "predictions.json", predictions)
            write_json(directory / "manifest.json", {"status": "completed_predictions_frozen_before_label_evaluation", "predictions_sha256": fingerprint(predictions)})
            write_json(source / "cohort" / (group + ".labels.json"), labels)
        write_json(source / "selection.json", {"selected_n": 20})
        write_json(source / "calibration_n20.json", {"calibration": {"kind": "affine", "slope": 1, "intercept": 1}})
        return source

    def test_selection_never_opens_test_and_one_pass_evaluation_retains_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            def guard(path):
                self.assertNotIn("combined_test", str(path))
                self.assertNotIn("test.labels", str(path))
                return read_json(path)
            with patch.object(tool, "read_json", side_effect=guard), patch("socket.socket", side_effect=AssertionError("No network")):
                selected = tool.select(source, root / "experiment")
            self.assertFalse(selected["test_read_by_select"])
            before = (root / "experiment/selection.json").read_bytes()
            result = tool.evaluate(source, root / "experiment")
            self.assertTrue(result["no_new_llm_calls"])
            self.assertEqual(before, (root / "experiment/selection.json").read_bytes())
            self.assertIn("Already viewed", result["test_status"])
            with self.assertRaises(ValueError):
                tool.evaluate(source, root / "experiment")

    def test_changed_frozen_model_is_rejected_before_test_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture(root)
            tool.select(source, root / "experiment")
            path = root / "experiment/selection.json"
            state = read_json(path)
            state["selected_family"] = "changed"
            write_json(path, state)
            with self.assertRaisesRegex(ValueError, "Selection changed"):
                tool.evaluate(source, root / "experiment")
