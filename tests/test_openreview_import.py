"""Authored API-shaped fixtures only; all real networking is forbidden."""

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from journal_prefs.__main__ import main
from journal_prefs.openreview_import import (API, PublicOpenReviewClient, SourceAccessDenied,
    classify_status, find_rating_enum, import_public_pilot, initial_version, numeric_rating,
    parse_forum, parse_venue_group, score_band, select_pilot)

VENUE = "Example.test/2024/Conference"
LABELS = ["1: strong reject", "3: reject", "5: marginally below the acceptance threshold",
          "6: marginally above the acceptance threshold", "8: accept", "10: strong accept"]


def form():
    return {"id": VENUE + "/-/Official_Review", "edit": {"note": {"content": {
        "rating": {"value": {"param": {"enum": LABELS}}}}}}}


def note(identifier="paper1", score=6, status="accept"):
    submission = {"id": identifier, "forum": identifier, "number": 1, "readers": ["everyone"],
                  "cdate": 1000, "tcdate": 1000, "mdate": 9000, "license": "CC BY 4.0",
                  "content": {"title": {"value": "SYNTHETIC importer fixture"},
                              "pdf": {"value": "/pdf/" + "b" * 40 + ".pdf"},
                              "venueid": {"value": VENUE if status == "accept" else VENUE + "/Rejected_Submission"},
                              "authors": {"value": ["DO_NOT_STORE_AUTHOR"]},
                              "authorids": {"value": ["DO_NOT_STORE_PROFILE"]}}}
    review = {"id": identifier + "review", "forum": identifier, "replyto": identifier,
              "readers": ["everyone"], "cdate": 5000, "tcdate": 5000, "mdate": 6000,
              "invitations": [VENUE + "/Submission1/-/Official_Review"],
              "signatures": ["DO_NOT_STORE_REVIEWER"],
              "content": {"rating": {"value": next(label for label in LABELS if label.startswith(str(score) + ":"))},
                          "confidence": {"value": "4: confident"}, "review": {"value": "SYNTHETIC review text"}}}
    decision = {"id": identifier + "decision", "forum": identifier, "replyto": identifier,
                "readers": ["everyone"], "cdate": 10000, "mdate": 10000,
                "invitations": [VENUE + "/Submission1/-/Decision"],
                "content": {"decision": {"value": "Accept" if status == "accept" else "Reject"}}}
    return submission, [review, decision]


def record(identifier="paper1", score=6, status="accept"):
    submission, replies = note(identifier, score, status)
    return parse_forum(submission, replies, VENUE, find_rating_enum(form(), "rating"))


class PublicImportTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("socket.socket", side_effect=AssertionError("Network forbidden"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.scale = find_rating_enum(form(), "rating")

    def test_parse_rating_and_official_scale_without_normalization(self):
        self.assertEqual(numeric_rating("6: marginally above"), 6)
        for raw in (None, "unknown", True, float("nan")):
            self.assertIsNone(numeric_rating(raw))
        self.assertEqual(self.scale["high_values"], [8, 10])
        self.assertEqual(self.scale["boundary_values"], [5, 6])
        self.assertEqual(self.scale["band_edges"], [1, 4, 7, 10])
        self.assertEqual(score_band(4, self.scale), "middle")
        self.assertEqual(score_band(7, self.scale), "high")
        with self.assertRaises(ValueError):
            find_rating_enum({"content": {}}, "rating")

    def test_public_projection_and_latest_stage_preserve_scores_confidence_times(self):
        result = record()
        self.assertEqual(result["mean_official_score_observed"], 6)
        self.assertEqual(result["official_reviews"][0]["confidence_numeric"], 4)
        self.assertEqual(result["official_reviews"][0]["mdate"], 6000)
        self.assertIn("not initial", result["review_stage"])
        self.assertNotIn("DO_NOT_STORE", json.dumps(result))
        self.assertFalse(result["evaluation_ready"])

    def test_private_note_and_field_are_not_imported(self):
        submission, replies = note()
        replies[0]["content"]["rating"]["readers"] = ["private-only"]
        result = parse_forum(submission, replies, VENUE, self.scale)
        self.assertIsNone(result["mean_official_score_observed"])
        self.assertEqual(result["missing_score_count"], 1)
        replies[0]["readers"] = ["private-only"]
        self.assertEqual(parse_forum(submission, replies, VENUE, self.scale)["review_count"], 0)
        submission["readers"] = ["private-only"]
        with self.assertRaises(ValueError):
            parse_forum(submission, replies, VENUE, self.scale)

    def test_missing_scores_never_become_zero_or_unanimous(self):
        submission, replies = note(score=8, status="reject")
        extra = copy.deepcopy(replies[0])
        extra["id"] = "missingReview"
        extra["content"].pop("rating")
        result = parse_forum(submission, replies + [extra], VENUE, self.scale)
        self.assertEqual(result["mean_official_score_observed"], 8)
        self.assertFalse(result["all_scores_present"])
        self.assertFalse(result["unanimous_high_score_rejected"])

    def test_unanimous_high_filter_is_individual_and_pilot_only(self):
        high_rejected = record("unanimous", 8, "reject")
        self.assertTrue(high_rejected["unanimous_high_score_rejected"])
        selected, summary = select_pilot([high_rejected], self.scale)
        self.assertEqual(selected, [])
        self.assertEqual(summary["exclusion_counts"]["unanimous_high_score_rejected"], 1)
        self.assertEqual(high_rejected["official_reviews"][0]["score_numeric"], 8)
        submission, replies = note("mixed", 10, "reject")
        lower = copy.deepcopy(replies[0])
        lower["id"] = "lowerReview"
        lower["content"]["rating"]["value"] = "5: marginally below the acceptance threshold"
        mixed = parse_forum(submission, replies + [lower], VENUE, self.scale)
        self.assertEqual(mixed["mean_official_score_observed"], 7.5)
        self.assertFalse(mixed["unanimous_high_score_rejected"])
        self.assertEqual(len(select_pilot([mixed], self.scale)[0]), 1)

    def test_withdrawn_desk_rejected_missing_decision_are_separate(self):
        for suffix, expected in (("Withdrawn_Submission", "withdrawn"), ("Desk_Rejected_Submission", "desk_rejected")):
            submission, replies = note()
            submission["content"]["venueid"]["value"] = VENUE + "/" + suffix
            self.assertEqual(classify_status(submission, replies[1:], VENUE), expected)
        submission, replies = note()
        submission["content"]["venueid"]["value"] = VENUE + "/Submission"
        self.assertEqual(classify_status(submission, [], VENUE), "unknown_or_missing_decision")

    def test_seeded_score_strata_boundaries_and_inclusion_probabilities(self):
        records = []
        for score in (1, 3, 5, 6, 8, 10):
            for i in range(6):
                records.append(record(f"p{score}n{i}", score, "reject" if score <= 5 else "accept"))
        # Middle non-boundary means 4.0 and 6.5, built from valid individual scores.
        for i in range(6):
            submission, replies = note(f"other{i}", 3, "reject")
            extra = copy.deepcopy(replies[0])
            extra["id"] += "extra"
            extra["content"]["rating"]["value"] = "5: marginally below the acceptance threshold"
            records.append(parse_forum(submission, replies + [extra], VENUE, self.scale))
        selected, summary = select_pilot(records, self.scale, seed=42)
        self.assertEqual(selected, select_pilot(list(reversed(records)), self.scale, seed=42)[0])
        self.assertEqual([sum(r["band"] == band for r in selected) for band in ("low", "middle", "high")], [3, 3, 3])
        self.assertEqual(sum(r["sub_stratum"] == "middle_boundary" for r in selected), 1)
        self.assertEqual(summary["strata"]["low"]["inclusion_probability"], 3 / 12)
        for item in selected:
            self.assertAlmostEqual(item["selection_probability_conditional_on_eligible_frame"] * item["inverse_probability_weight_within_frame"], 1)

    def test_shortfalls_do_not_relax_filter_or_reallocate(self):
        selected, summary = select_pilot([record("high", 8)], self.scale)
        self.assertEqual(len(selected), 1)
        self.assertEqual(summary["strata"]["low"]["shortfall"], 3)
        self.assertEqual(summary["strata"]["middle_boundary"]["shortfall"], 1)

    def test_initial_pdf_requires_public_immutable_pre_review_nonrevision_edit(self):
        submission, replies = note()
        edit = {"id": "initialEdit", "readers": ["everyone"], "cdate": 2000, "tcdate": 2000,
                "invitation": VENUE + "/-/Submission", "note": {"id": submission["id"],
                "content": {"pdf": {"value": "/pdf/" + "a" * 40 + ".pdf"}}}}
        version = initial_version(submission, [edit], replies[:1])
        self.assertEqual(version["edit_id"], "initialEdit")
        self.assertIn("not_downloaded", version["status"])
        self.assertFalse(version["evaluation_ready"])
        for change in ({"tcdate": 5000}, {"readers": ["private"]}, {"invitation": VENUE + "/-/Camera_Ready_Revision"}):
            candidate = dict(edit, **change)
            self.assertIsNone(initial_version(submission, [candidate], replies[:1])["initial_pdf_url"])
        self.assertIsNone(initial_version(submission, [], replies[:1])["initial_pdf_url"])
        unknown_time = copy.deepcopy(replies[0])
        unknown_time.pop("cdate")
        unknown_time.pop("tcdate")
        self.assertIsNone(initial_version(submission, [edit], [replies[0], unknown_time])["initial_pdf_url"])

    def test_client_needs_explicit_network_and_terms_and_stops_after_403(self):
        for args in ({}, {"allow_network": True}, {"acknowledge_terms": True}):
            with self.assertRaises(ValueError):
                PublicOpenReviewClient(**args)
        client = PublicOpenReviewClient(True, True)
        client.opener = Mock()
        client.opener.open.side_effect = HTTPError(API + "/notes", 403, "Forbidden", {}, io.BytesIO(b"private error body"))
        with self.assertRaises(SourceAccessDenied) as caught:
            client.get("/notes", {"invitation": VENUE + "/-/Submission"})
        self.assertEqual(caught.exception.status, 403)
        self.assertTrue(client.stopped)
        with self.assertRaises(RuntimeError):
            client.get("/groups", {"id": VENUE})
        client.opener.open.assert_called_once()

    def test_profile_group_projection_does_not_keep_members(self):
        group = {"id": VENUE, "members": ["PRIVATE_MEMBER"], "content": {
            "review_rating": {"value": "rating"}, "submission_id": {"value": VENUE + "/-/Submission"}}}
        self.assertNotIn("PRIVATE_MEMBER", json.dumps(parse_venue_group(group)))

    def test_denial_is_saved_without_real_requests_or_paper_records(self):
        profile = {"schema_version": 1, "venues": [
            {"venue_id": "Example.test/2024/Conference", "completed": True},
            {"venue_id": "Example.test/2025/Conference", "completed": True}]}
        with tempfile.TemporaryDirectory() as temp, patch("journal_prefs.openreview_import.PublicOpenReviewClient") as client:
            client.return_value.log = []
            client.return_value.get.side_effect = SourceAccessDenied(API + "/groups", 403)
            result = import_public_pilot(profile, Path(temp) / "new", True, True)
            self.assertEqual(result["status"], "stopped_source_denied")
            self.assertEqual(result["selected_count"], 0)
            self.assertEqual(result["downloaded_pdfs"], 0)
            self.assertTrue((Path(temp) / "new/manifest.json").exists())
            self.assertEqual((Path(temp) / "new/.gitignore").read_text().strip(), "*")

    def test_offline_cli_import_has_no_implicit_network(self):
        with tempfile.TemporaryDirectory() as temp, patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["import-openreview", "--out", str(Path(temp) / "no-permission")]), 2)
            self.assertFalse((Path(temp) / "no-permission").exists())

    def test_complete_two_year_import_with_authored_transport_fixture(self):
        profile = {"schema_version": 1, "seed": 7, "max_frame_records_per_venue": 300,
                   "venues": [{"venue_id": f"Example.test/{year}/Conference", "completed": True} for year in (2024, 2025)]}

        class FixtureClient:
            def __init__(self, *args):
                self.log = []

            def get(self, path, params):
                self.log.append({"fixture_only": True, "path": path, "params": params})
                if path == "/groups":
                    return {"groups": [{"id": params["id"], "content": {}}]}
                if path == "/invitations":
                    return {"invitations": [form()]}
                if path == "/notes/edits":
                    return {"edits": []}
                venue = params["invitation"].split("/-/")[0]
                year = venue.split("/")[1]
                notes = []
                for index, score in enumerate((1, 1, 3, 5, 8, 8, 10, 3, 3)):
                    submission, replies = note("y" + year + "p" + str(index), score, "accept" if score >= 6 else "reject")
                    if index >= 7:
                        extra = copy.deepcopy(replies[0])
                        extra["id"] += "extra"
                        extra["content"]["rating"]["value"] = LABELS[2]
                        replies.append(extra)  # middle mean 4, outside boundary mean [5,6]
                    submission["details"] = {"directReplies": replies}
                    notes.append(json.loads(json.dumps(submission).replace(VENUE, venue)))
                return {"notes": notes, "count": 9}

        with tempfile.TemporaryDirectory() as temp, patch("journal_prefs.openreview_import.PublicOpenReviewClient", FixtureClient):
            output = Path(temp) / "synthetic"
            result = import_public_pilot(profile, output, True, True)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["selected_count"], 18)
            self.assertEqual(result["evaluation_ready_count"], 0)
            self.assertEqual(result["accepted_count"] + result["rejected_count"], 18)
            self.assertEqual(result["downloaded_pdfs"], 0)
            self.assertTrue(all(item["initial_version_status"] == "unavailable_or_unverified" for item in result["selected"]))
            self.assertTrue(all(item["frame_is_complete"] for item in result["venues"]))
            for selected in result["selected"]:
                self.assertTrue((output / selected["record_file"]).is_file())
                self.assertEqual(json.loads((output / selected["record_file"]).read_text())["review_count"], 2 if "p7" in selected["forum_id"] or "p8" in selected["forum_id"] else 1)


if __name__ == "__main__":
    unittest.main()
