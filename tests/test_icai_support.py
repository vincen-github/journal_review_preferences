import csv
import tempfile
import unittest
from pathlib import Path

from journal_prefs.icai_support import ICAIUpstreamAdapter, export_upstream_csv, filter_principles, validate_pairs


def pair():
    return {"comparison_id": "synthetic-pair-1", "paper_id": "synthetic-paper-1",
            "input": "SYNTHETIC shared manuscript assessment task", "text_a": "Assessment A",
            "text_b": "Assessment B", "preferred_text": "text_a", "preference_source": "authored synthetic fixture"}


class ICAISupportTests(unittest.TestCase):
    def test_real_pair_export_matches_authors_documented_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "pairs.csv"
            receipt = export_upstream_csv([pair()], file)
            with file.open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(set(rows[0]), {"input", "text_a", "text_b", "preferred_text"})
            self.assertEqual(receipt["paper_ids"], ["synthetic-paper-1"])

    def test_accept_reject_records_and_cross_manuscript_pseudo_pairs_are_not_adapted(self):
        with self.assertRaises(ValueError):
            validate_pairs([{"paper_id": "1", "accepted": True}])
        with self.assertRaises(ValueError):
            validate_pairs([dict(pair(), response_b_paper_id="other-paper")])
        with self.assertRaises(ValueError):
            validate_pairs([dict(pair(), preferred_text="tie")])

    def test_upstream_filter_strict_majorities_relevance_and_order(self):
        def counts(a, b, abstain=0, invalid=0):
            return {"for": a, "against": b, "abstain": abstain, "invalid": invalid, "both": 0, "neither": 0}
        votes = {"tie": counts(2, 2), "best": counts(4, 0), "second": counts(3, 1),
                 "invalid": counts(2, 0, invalid=3), "sparse": counts(1, 0, abstain=20)}
        self.assertEqual(filter_principles(votes), ["best", "second"])
        self.assertEqual(filter_principles(votes, max_principles=1), ["best"])
        with self.assertRaises(ValueError):
            filter_principles({"bad": counts(-1, 0)})

    def test_unconfigured_original_runner_has_no_fallback(self):
        with self.assertRaisesRegex(ValueError, "Original ICAI runner not configured"):
            ICAIUpstreamAdapter().fit([pair()])

    def test_explicit_mock_backend_roundtrip_and_conditioning_is_not_original_run(self):
        adapter = ICAIUpstreamAdapter(lambda rows: {"final_principles": ["SYNTHETIC principle"]}, "authored mock, not an original ICAI run")
        artifact = adapter.freeze(adapter.fit([pair()]))
        self.assertEqual(adapter.condition(artifact, {"paper_id": "new"})[0]["principle"], "SYNTHETIC principle")
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "artifact.json"
            adapter.save(artifact, file)
            self.assertEqual(adapter.load(file), artifact)
        with self.assertRaises(ValueError):
            adapter.condition(artifact, {"paper_id": "synthetic-paper-1"})
        with self.assertRaises(ValueError):
            adapter.refine(artifact, [pair()])
