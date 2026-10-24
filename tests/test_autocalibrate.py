import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_prefs.autocalibrate import AutoCalibrateAlgorithm, criteria_artifact, condition, validate_artifact
from journal_prefs.data import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("autocalibrate_pilot", ROOT / "tools/autocalibrate_pilot.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CriteriaTests(unittest.TestCase):
    def test_algorithm1_filters_and_refines_using_candidate_own_train_errors(self):
        history = [{"paper_id": str(i), "title": "SYNTHETIC", "text": "SYNTHETIC body", "reviews": [], "individual_scores": [5 + i % 2]} for i in range(6)]
        targets = {r["paper_id"]: r["individual_scores"][0] for r in history}
        class Provider:
            drafts = 0
            feedback = None
            def draft(self, examples):
                self.drafts += 1
                return {"rules": [{"principle": "SYNTHETIC draft " + str(self.drafts), "source_paper_ids": [examples[0]["paper_id"]]}]}
            def revise(self, artifact, errors):
                self.feedback = errors
                return {"rules": [{"principle": "SYNTHETIC refined", "source_paper_ids": [errors[0]["paper_id"]]}]}
            def score(self, artifact, papers):
                assert all(set(p) == {"paper_id", "title", "text"} for p in papers)
                principle = artifact["rules"][0]["principle"]
                return [targets[p["paper_id"]] if "refined" in principle else (3 if principle.endswith("2") else 2) for p in papers]
        provider = Provider()
        result = AutoCalibrateAlgorithm(provider).fit(history)
        self.assertIn("refined", result["selected"]["candidate_id"])
        self.assertTrue(all(row["predicted_score"] == 3 for row in provider.feedback))
        self.assertEqual(len(result["drafts"]), 2)
        self.assertEqual(len(result["pool"]), 2)
        self.assertNotIn("individual_scores", condition(result["selected"]))

    def test_unknown_rule_source_or_changed_artifact_is_rejected(self):
        with self.assertRaises(ValueError):
            criteria_artifact({"rules": [{"principle": "bad", "source_paper_ids": ["TEST"]}]}, ["TRAIN"], "A")
        a = criteria_artifact({"rules": [{"principle": "SYNTHETIC evidence-sensitive rule", "source_paper_ids": ["TRAIN"]}]}, ["TRAIN"], "A")
        self.assertEqual(validate_artifact(a), a)
        a["rules"][0]["principle"] = "changed"
        with self.assertRaises(ValueError):
            validate_artifact(a)

    def test_prepare_samples_only_train_and_preserves_full_scoring_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source", Path(directory) / "pilot"
            (source / "cohort").mkdir(parents=True)
            groups = {g: [g + str(i) for i in range(n)] for g, n in (("train", 60), ("validation", 20), ("test", 20))}
            write_json(source / "partition.json", {"ids": groups, "partition_sha256": fingerprint(groups)})
            frame = []
            for group, ids in groups.items():
                inputs = [{"paper_id": i, "title": "SYNTHETIC", "text": "SYNTHETIC" * 3000,
                           "text_sha256": hashlib.sha256(("SYNTHETIC" * 3000).encode()).hexdigest()} for i in ids]
                write_json(source / "cohort" / (group + ".inputs.json"), inputs)
                for n, i in enumerate(ids):
                    score = [2, 5, 8][n % 3]
                    frame.append({"paper_id": i, "mean_score": score, "individual_scores": [score], "reviews": [{"comments": "SYNTHETIC review", "score_numeric": score}]})
            write_json(source / "metadata_frame.json", frame)
            write_json(source / "cohort/train.labels.json", [r for r in frame if r["paper_id"] in groups["train"]])
            def guard(path):
                self.assertNotIn("metadata_frame", str(path))
                self.assertNotIn("test.labels", str(path))
                self.assertNotIn("validation.labels", str(path))
                return read_json(path)
            with patch.object(tool, "read_json", side_effect=guard):
                protocol = tool.prepare(source, output)
            self.assertTrue(set(protocol["golden_train_ids"]) <= set(groups["train"]))
            self.assertEqual(protocol["score_calls_planned"], 78)
            full = read_json(output / "inputs/train.json")
            clipped = read_json(output / "train_history.json")
            self.assertTrue(all(len(p["text"]) == 27000 for p in full))
            self.assertTrue(all(len(p["text"]) < 17000 for p in clipped))
            with patch.object(tool, "run") as process:
                with self.assertRaises(FileNotFoundError):
                    tool.score(source, output, "A", "test", "a", "fake", True)
                process.assert_not_called()

    def test_phase_input_validation_rejects_extra_papers_and_label_fields(self):
        paper = {"paper_id": "TRAIN", "title": "SYNTHETIC", "text": "SYNTHETIC", "text_sha256": hashlib.sha256(b"SYNTHETIC").hexdigest()}
        with self.assertRaises(ValueError):
            tool.checked_inputs([paper, paper], ["TRAIN"])
        with self.assertRaises(ValueError):
            tool.checked_inputs([dict(paper, human_mean=7)], ["TRAIN"])
