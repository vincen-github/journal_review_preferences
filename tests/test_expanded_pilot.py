import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.error import HTTPError
from copy import deepcopy

from journal_prefs.expanded_pilot import SourceReader, balanced_counts, freeze_cohort, proportional_counts


def records():
    return [{"paper_id": str(i), "title": "Unique manuscript " + str(i),
             "mean_score": [3, 5.5, 8][i % 3], "accepted": i % 2 == 0,
             "source_text_blob": "fixture-blob", "source_review_path": str(i)} for i in range(180)]


class ExpansionTests(unittest.TestCase):
    def test_fixed_nested_cohorts_keep_all_explored_papers_in_training(self):
        old = [str(i) for i in range(18)]
        groups, partition = freeze_cohort(records(), old)
        self.assertEqual([len(groups[g]) for g in ("train", "validation", "test")], [60, 20, 20])
        self.assertEqual(partition["ids"]["train"][:18], old)
        nested = partition["nested_train_ids"]
        self.assertTrue(set(nested["20"]) < set(nested["40"]) < set(nested["60"]))
        self.assertFalse(set(old) & set(partition["ids"]["validation"] + partition["ids"]["test"]))

    def test_selection_is_seeded_and_decisions_cannot_remove_difficult_cases(self):
        data = records()
        _, first = freeze_cohort(data, [str(i) for i in range(18)])
        altered = deepcopy(data)
        for row in altered:
            row["accepted"] = not row["accepted"]
            row["all_high_rejected"] = True
        _, second = freeze_cohort(list(reversed(altered)), [str(i) for i in range(18)])
        self.assertEqual(first["ids"], second["ids"])
        self.assertFalse(first["decision_filter_applied"])
        _, other_seed = freeze_cohort(data, [str(i) for i in range(18)], seed=20261024)
        self.assertNotEqual(first["ids"], other_seed["ids"])

    def test_missing_scores_and_title_duplicates_are_explicit_exclusions(self):
        data = records()
        data[100]["mean_score"] = None
        data[101]["title"] = data[102]["title"]
        _, partition = freeze_cohort(data, [str(i) for i in range(18)])
        self.assertIn("missing_recommendations", [r["reason"] for r in partition["exclusions"]])
        self.assertIn("duplicate_normalized_title", [r["reason"] for r in partition["exclusions"]])

    def test_band_capacity_shortfalls_are_bounded_and_visible(self):
        self.assertEqual(balanced_counts({"low": 8, "middle": 80, "high": 80}, {"low": 6, "middle": 6, "high": 6}, 100),
                         {"low": 8, "middle": 46, "high": 46})
        allocated = proportional_counts(20, {"low": 8, "middle": 46, "high": 46}, {"low": 2, "middle": 40, "high": 40})
        self.assertEqual(sum(allocated.values()), 20)
        self.assertLessEqual(allocated["low"], 2)
        with self.assertRaises(ValueError):
            balanced_counts({"low": 1, "middle": 1, "high": 1}, {}, 100)

    def test_invalid_mandatory_and_cross_group_duplicate_ids_fail(self):
        with self.assertRaises(ValueError):
            freeze_cohort(records(), ["0", "0"])
        with self.assertRaises(ValueError):
            freeze_cohort(records(), ["missing"])
        data = records()
        data.append(deepcopy(data[50]))
        _, partition = freeze_cohort(data, [str(i) for i in range(18)])
        self.assertEqual(len(sum(partition["ids"].values(), [])), 100)
        self.assertIn("duplicate_paper_id", [r["reason"] for r in partition["exclusions"]])

    def test_rare_new_band_is_reserved_for_both_heldout_groups(self):
        data = records()
        # Eight low papers overall, six already explored: reserve one new low each.
        low = [r for r in data if r["mean_score"] == 3]
        remove = {r["paper_id"] for r in low[8:]}
        data = [r for r in data if r["paper_id"] not in remove]
        _, partition = freeze_cohort(data, [str(i) for i in range(18)])
        self.assertEqual(partition["group_band_counts"]["validation"]["low"], 1)
        self.assertEqual(partition["group_band_counts"]["test"]["low"], 1)

    def test_reader_rejects_path_aliases_and_unpinned_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for new, old, revision in [(root, root, "a" * 40), (root / "new", root, "a" * 40),
                                        (root / "new", root / "old", "main")]:
                with self.assertRaises(ValueError):
                    SourceReader(new, old, revision)

    def test_cache_is_hash_checked_without_modifying_original_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = SourceReader(root / "new", root / "old", "a" * 40)
            path = "data/iclr_2017/train/reviews/1.json"
            old = root / "old/raw" / path
            old.parent.mkdir(parents=True)
            body = b'{"fixture": true}'
            old.write_bytes(body)
            blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
            with patch.object(reader, "network") as network:
                self.assertEqual(reader.file({"path": path, "sha": blob}), {"fixture": True})
                self.assertEqual(old.read_bytes(), body)
                with self.assertRaises(ValueError):
                    reader.file({"path": path, "sha": "0" * 40})
                network.assert_not_called()

    def test_queued_network_operation_stops_after_denial_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = SourceReader(root / "new", root / "old", "a" * 40)
            class Deny:
                calls = 0
                def open(self, *args, **kwargs):
                    self.calls += 1
                    raise HTTPError("https://fixture.invalid", 403, "authored denial", {}, None)
            transport = Deny()
            with patch("journal_prefs.expanded_pilot.build_opener", return_value=transport), patch("journal_prefs.expanded_pilot.time.sleep"):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(reader.network, "https://fixture.invalid") for _ in range(2)]
                    for future in futures:
                        with self.assertRaises((HTTPError, ValueError)):
                            future.result()
            self.assertEqual(transport.calls, 1)
