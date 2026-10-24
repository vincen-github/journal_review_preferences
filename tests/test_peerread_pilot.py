import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from journal_prefs.peerread_pilot import NoRedirect, build_pilot, fetch_file, parse_review_file, parsed_text, select_group


def fixture(identifier="1", scores=(3, 4), accepted=False):
    return {"id": identifier, "title": "Synthetic manuscript", "abstract": "Synthetic abstract", "accepted": accepted,
            "conference": "ICLR 2017 conference submission", "authors": ["UNNEEDED IDENTITY"], "histories": [],
            "reviews": [{"IS_META_REVIEW": False, "RECOMMENDATION": score, "comments": "Synthetic review", "DATE": "16 Dec 2016"} for score in scores]}


class PeerReadPilotTests(unittest.TestCase):
    def test_cached_source_is_verified_before_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = "data/iclr_2017/train/reviews/1.json"
            local = root / "raw" / path
            local.parent.mkdir(parents=True)
            raw = json.dumps(fixture()).encode("utf-8")
            local.write_bytes(raw)
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            with patch("journal_prefs.peerread_pilot.build_opener", side_effect=AssertionError("No network")):
                self.assertEqual(fetch_file(root, path, "a" * 40, [], blob)["id"], "1")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetch_file(root, path, "a" * 40, [], "b" * 40)
                with self.assertRaisesRegex(ValueError, "unverified cache refused"):
                    fetch_file(root, path, "a" * 40, [])

    def test_network_is_explicit_and_redirects_stop(self):
        with self.assertRaisesRegex(ValueError, "allow-public-network"):
            build_pilot("unused")
        with self.assertRaisesRegex(ValueError, "redirects are disabled"):
            NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://unapproved.invalid/data")

    def test_exact_duplicate_source_entries_are_removed_and_discussions_not_zero(self):
        raw = fixture()
        raw["reviews"] += copy.deepcopy(raw["reviews"])
        raw["reviews"].append({"IS_META_REVIEW": False, "comments": "Unscored discussion"})
        result = parse_review_file(raw)
        self.assertEqual(result["individual_scores"], [3, 4])
        self.assertEqual(result["mean_score"], 3.5)
        self.assertEqual(result["duplicate_entries_removed"], 2)
        self.assertEqual(result["unrated_discussion_entries"], 1)
        self.assertNotIn("UNNEEDED IDENTITY", json.dumps(result))
        self.assertFalse(result["evaluation_ready"])

    def test_requested_filter_only_applies_to_training_selection(self):
        record = parse_review_file(fixture(scores=(8, 9), accepted=False))
        self.assertTrue(record["all_high_rejected"])
        training, summary = select_group([record], "train", 42)
        test, test_summary = select_group([record], "test", 42)
        self.assertEqual(training, [])
        self.assertEqual(summary["exclusion_counts"], {"unanimous_high_score_rejected": 1})
        self.assertEqual(len(test), 1)
        self.assertFalse(test_summary["high_rejected_filter_applied"])
        mixed = parse_review_file(fixture(scores=(10, 6), accepted=False))
        self.assertFalse(mixed["all_high_rejected"])

    def test_seeded_selection_is_paper_grouped_and_keeps_boundary(self):
        records = [parse_review_file(fixture(str(i), (score,), accepted=score >= 6))
                   for i, score in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 1)]
        chosen, report = select_group(records, "test", 7)
        self.assertEqual(chosen, select_group(list(reversed(records)), "test", 7)[0])
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({r["paper_id"] for r in chosen}), 6)
        self.assertEqual(sum(r["stratum"] == "middle_boundary" for r in chosen), 1)
        self.assertEqual(sum(r["stratum"] == "low" for r in chosen), 2)

    def test_full_parsed_sections_are_used_without_abstract_substitution(self):
        body = "Synthetic manuscript body. " * 30
        data = {"metadata": {"sections": [{"heading": "Method", "text": body}], "authors": ["PRIVATE"], "abstractText": "not used"}}
        self.assertIn(body, parsed_text(data))
        self.assertNotIn("PRIVATE", parsed_text(data))
        with self.assertRaises(ValueError):
            parsed_text({"metadata": {"sections": [], "abstractText": body}})
