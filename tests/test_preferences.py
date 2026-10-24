"""Authored synthetic preference fixtures. No real model or network is used."""

import copy
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from journal_prefs.data import fingerprint
from journal_prefs.preferences import (EvidenceRules, PreferenceAlgorithm, export_artifact,
                                       load_artifact, validate_artifact)


def paper(identifier="train1", score=4, decision=False):
    return {"paper_id": identifier, "title": "SYNTHETIC title " + identifier,
            "text": "SYNTHETIC manuscript " + identifier + " describes a controlled comparison.",
            "reviews": [{"text": "SYNTHETIC reviewer " + identifier + " requests a stronger comparison.",
                         "score": score, "private_reviewer_id": "DROP_REVIEWER_ID"}],
            "decision": decision, "contact": "DROP_PRIVATE_CONTACT", "extra": {"unused": True}}


def rule(source, principle="SYNTHETIC preference supported by the quoted comparison comment."):
    return {"principle": principle, "evidence": [{"paper_id": source["paper_id"],
            "review_quote": source["reviews"][0]["text"], "manuscript_quote": source["text"],
            "extra": "DROP_EVIDENCE_EXTRA"}], "extra": "DROP_RULE_EXTRA"}


class SyntheticGenerator:
    """Scripted test double; does not discover real institutional preferences."""

    def __init__(self):
        self.calls = []

    def __call__(self, purpose, payload):
        self.calls.append((purpose, copy.deepcopy(payload)))
        if purpose == "per_paper":
            source = payload["paper"]
        elif purpose == "synthesis":
            source = payload["papers"][0]
        elif purpose == "refinement":
            source = payload["validation"][0]
        else:
            raise AssertionError("Unknown generation purpose")
        return {"rules": [rule(source)], "unused_generator_field": "DROP_GENERATOR_EXTRA"}


class PreferenceTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.network = []
        for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo",
                       "urllib.request.urlopen", "http.client.HTTPConnection.connect",
                       "http.client.HTTPSConnection.connect"):
            self.network.append(stack.enter_context(patch(target, side_effect=AssertionError("Network forbidden"))))
        self.generator = SyntheticGenerator()
        self.algorithm = EvidenceRules(self.generator)

    def tearDown(self):
        for guard in self.network:
            guard.assert_not_called()

    def test_fit_generates_per_paper_then_synthesis_from_projected_training_only(self):
        history = [paper("train2", 8, "accept"), paper()]
        original = copy.deepcopy(history)
        artifact = self.algorithm.fit(history)
        self.assertEqual(history, original)
        self.assertEqual([p for p, _ in self.generator.calls], ["per_paper", "per_paper", "synthesis"])
        self.assertEqual(artifact["train_ids"], ["train1", "train2"])
        self.assertEqual(artifact["selection_ids"], [])
        self.assertFalse(artifact["frozen"])
        self.assertEqual(artifact["refinement_count"], 0)
        self.assertEqual(artifact["train_sha256"], fingerprint(artifact["source_records"]["train"]))
        self.assertEqual(artifact, validate_artifact(artifact))
        self.assertEqual(set(artifact["rules"][0]), {"rule_id", "principle", "evidence"})
        self.assertEqual(set(artifact["rules"][0]["evidence"][0]), {"paper_id", "review_quote", "manuscript_quote"})
        for _, payload in self.generator.calls:
            self.assertNotIn("DROP_", repr(payload))
        self.assertNotIn("DROP_", repr(artifact))

    def test_fit_rejects_duplicate_ids_and_invalid_history_before_generation(self):
        invalid = [[], [paper(), paper()], [dict(paper(), paper_id="")],
                   [dict(paper(), reviews=[])], [dict(paper(), decision=1)],
                   [dict(paper(), text=" ")], [dict(paper(), title=None)], [None]]
        for history in invalid:
            generator = Mock()
            with self.subTest(history=history), self.assertRaises(ValueError):
                EvidenceRules(generator).fit(history)
            generator.assert_not_called()
        for score in (True, float("nan"), float("inf"), "7"):
            candidate = paper()
            candidate["reviews"][0]["score"] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.algorithm.fit([candidate])

    def test_constructor_validates_configuration_bounds_and_generator(self):
        for kwargs in ({"max_rules": 0}, {"max_rules": True}, {"max_rules": 65},
                       {"max_refinements": 2}, {"max_refinements": True}, {"max_refinements": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvidenceRules(self.generator, **kwargs)
        with self.assertRaises(ValueError):
            EvidenceRules(None)

    def test_per_paper_generation_cannot_cite_another_training_paper(self):
        other = paper("train2")
        generator = Mock(return_value={"rules": [rule(other)]})
        with self.assertRaisesRegex(ValueError, "unknown source paper"):
            EvidenceRules(generator).fit([paper(), other])
        self.assertEqual(generator.call_count, 1)

    def test_generated_evidence_requires_exact_quotes_in_the_cited_source(self):
        for field, wrong in (("paper_id", "unseen-test"), ("review_quote", "fabricated review"),
                             ("manuscript_quote", "fabricated manuscript"), ("review_quote", " ")):
            candidate = rule(paper())
            candidate["evidence"][0][field] = wrong
            generator = Mock(return_value={"rules": [candidate]})
            with self.subTest(field=field), self.assertRaises(ValueError):
                EvidenceRules(generator).fit([paper()])
        candidate = rule(paper())
        candidate["evidence"][0]["review_quote"] = paper()["text"]
        with self.assertRaisesRegex(ValueError, "human reviews"):
            EvidenceRules(Mock(return_value={"rules": [candidate]})).fit([paper()])

    def test_synthesis_output_is_validated_again_not_just_candidate_rules(self):
        generator = SyntheticGenerator()

        def generate(purpose, payload):
            if purpose == "synthesis":
                return {"rules": [rule(paper("unseen-test"))]}
            return generator(purpose, payload)

        with self.assertRaisesRegex(ValueError, "unknown source paper"):
            EvidenceRules(generate).fit([paper()])

    def test_rule_shapes_counts_and_duplicate_principles_are_rejected(self):
        valid = rule(paper())
        duplicate_evidence = copy.deepcopy(valid)
        duplicate_evidence["evidence"] *= 2
        candidates = [None, {"rules": {}}, {"rules": [None]},
                      {"rules": [dict(valid, principle="")]}, {"rules": [dict(valid, evidence=[])]},
                      {"rules": [dict(valid, evidence=[None])]}, {"rules": [duplicate_evidence]},
                      {"rules": [valid, copy.deepcopy(valid)]}]
        for candidate in candidates:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                EvidenceRules(Mock(return_value=candidate)).fit([paper()])
        with self.assertRaisesRegex(ValueError, "max_rules"):
            EvidenceRules(Mock(return_value={"rules": [valid, rule(paper(), "Different principle")]}), max_rules=1).fit([paper()])

    def test_generator_mutation_does_not_modify_sources_or_original_history(self):
        history = [paper()]
        original = copy.deepcopy(history)

        def generate(purpose, payload):
            source = copy.deepcopy(payload["paper"] if purpose == "per_paper" else payload["papers"][0])
            if purpose == "per_paper":
                payload["paper"]["text"] = "MUTATED"
            else:
                payload["papers"][0]["reviews"][0]["text"] = "MUTATED"
            return {"rules": [rule(source)]}

        artifact = EvidenceRules(generate).fit(history)
        self.assertEqual(history, original)
        self.assertNotIn("MUTATED", repr(artifact))

    def test_refinement_uses_validation_labels_once_and_records_selection_provenance(self):
        before = self.algorithm.fit([paper()])
        snapshot = copy.deepcopy(before)
        validation = [paper("validation", 8, "accept")]
        artifact = self.algorithm.refine(before, validation,
            [{"paper_id": "validation", "score": 6, "review": "SYNTHETIC prediction", "decision": False,
              "unknown_test_context": "DROP_ASSESSMENT_EXTRA"}])
        self.assertEqual(before, snapshot)
        self.assertEqual(artifact["selection_ids"], ["validation"])
        self.assertEqual(artifact["refinement_count"], 1)
        self.assertEqual(artifact["selection_sha256"], fingerprint(artifact["source_records"]["selection"]))
        self.assertEqual(artifact["rules"][0]["evidence"][0]["paper_id"], "validation")
        purpose, payload = self.generator.calls[-1]
        self.assertEqual(purpose, "refinement")
        self.assertEqual(payload["validation"][0]["decision"], "accept")
        self.assertNotIn("DROP_", repr(payload))
        calls = len(self.generator.calls)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.algorithm.refine(artifact, [paper("second-validation")])
        self.assertEqual(len(self.generator.calls), calls)

    def test_refinement_may_retain_training_and_add_validation_evidence(self):
        artifact = self.algorithm.fit([paper()])

        def generate(purpose, payload):
            self.assertEqual(purpose, "refinement")
            result = rule(payload["training"][0])
            result["evidence"] += rule(payload["validation"][0])["evidence"]
            return {"rules": [result]}

        algorithm = EvidenceRules(generate)
        refined = algorithm.refine(artifact, [paper("validation")])
        self.assertEqual({e["paper_id"] for e in refined["rules"][0]["evidence"]}, {"train1", "validation"})

    def test_refinement_rejects_training_overlap_and_foreign_assessment_ids_before_generation(self):
        artifact = self.algorithm.fit([paper()])
        calls = len(self.generator.calls)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            self.algorithm.refine(artifact, [paper()])
        with self.assertRaisesRegex(ValueError, "validation-only"):
            self.algorithm.refine(artifact, [paper("validation")], [{"paper_id": "unseen-test", "score": 9}])
        with self.assertRaisesRegex(ValueError, "validation-only"):
            self.algorithm.refine(artifact, [paper("validation")], [{"paper_id": "validation"}] * 2)
        self.assertEqual(len(self.generator.calls), calls)

    def test_zero_refinement_budget_and_frozen_artifacts_do_not_call_generator(self):
        algorithm = EvidenceRules(self.generator, max_refinements=0)
        artifact = algorithm.fit([paper()])
        calls = len(self.generator.calls)
        with self.assertRaisesRegex(ValueError, "budget"):
            algorithm.refine(artifact, [paper("validation")])
        frozen = algorithm.freeze(artifact)
        with self.assertRaisesRegex(ValueError, "Frozen"):
            algorithm.refine(frozen, [paper("validation")])
        self.assertEqual(len(self.generator.calls), calls)

    def test_freeze_is_immutable_and_condition_contains_rules_only_for_unused_target(self):
        artifact = self.algorithm.refine(self.algorithm.fit([paper()]), [paper("validation")])
        with self.assertRaisesRegex(ValueError, "Freeze"):
            self.algorithm.condition(artifact, {"paper_id": "unseen-test"})
        frozen = self.algorithm.freeze(artifact)
        self.assertFalse(artifact["frozen"])
        self.assertTrue(frozen["frozen"])
        self.assertEqual(frozen, self.algorithm.freeze(frozen))
        for identifier in ("train1", "validation"):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "used"):
                self.algorithm.condition(frozen, {"paper_id": identifier})
        target = dict(paper("unseen-test"), test_only_secret="DROP_TARGET_LABEL")
        calls = len(self.generator.calls)
        conditioning = self.algorithm.condition(frozen, target)
        self.assertEqual(len(self.generator.calls), calls)
        self.assertEqual(conditioning, [{"rule_id": "R001", "principle": frozen["rules"][0]["principle"]}])
        self.assertEqual(set(conditioning[0]), {"rule_id", "principle"})
        conditioning[0]["principle"] = "MUTATED"
        self.assertNotEqual(conditioning, self.algorithm.condition(frozen, target))

    def test_empty_evidence_backed_rule_set_is_allowed_without_invented_preferences(self):
        algorithm = EvidenceRules(Mock(return_value={"rules": []}))
        frozen = algorithm.freeze(algorithm.fit([paper()]))
        self.assertEqual(algorithm.condition(frozen, {"paper_id": "unseen-test"}), [])

    def test_export_load_preserves_artifact_and_detects_changed_content(self):
        frozen = self.algorithm.freeze(self.algorithm.fit([paper()]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            export_artifact(frozen, path)
            loaded = load_artifact(path)
            self.assertEqual(frozen, loaded)
            path.write_text(path.read_text(encoding="utf-8").replace("SYNTHETIC preference", "ALTERED preference"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_artifact(path)
        loaded["rules"][0]["principle"] = "MUTATED"
        self.assertNotEqual(loaded, frozen)

    def test_artifact_schema_evidence_source_hash_and_rule_ids_are_validated(self):
        artifact = self.algorithm.fit([paper()])
        mutations = [lambda a: a.update(schema_version=True),
                     lambda a: a.update(frozen="true"), lambda a: a.update(algorithm="other"),
                     lambda a: a.update(unknown="discard?"), lambda a: a.update(train_sha256="0" * 64),
                     lambda a: a.update(train_ids=["wrong"]),
                     lambda a: a["rules"][0].update(rule_id="R999"),
                     lambda a: a["rules"][0]["evidence"][0].update(manuscript_quote="fabricated"),
                     lambda a: a["source_records"]["train"][0].update(unrecognized="extra")]
        for mutate in mutations:
            candidate = copy.deepcopy(artifact)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_artifact(candidate)

    def test_protocol_accepts_minimal_custom_algorithm_without_provider_or_calibration(self):
        class CustomAlgorithm:
            def fit(self, history):
                return {"custom": "SYNTHETIC", "ids": [row["paper_id"] for row in history]}

            def refine(self, artifact, validation, assessments=None):
                return dict(artifact, selected=[row["paper_id"] for row in validation])

            def freeze(self, artifact):
                return dict(artifact, frozen=True)

            def condition(self, artifact, paper):
                return [{"rule_id": "custom", "principle": "SYNTHETIC custom conditioning"}]

        custom = CustomAlgorithm()
        self.assertIsInstance(custom, PreferenceAlgorithm)
        self.assertIsInstance(self.algorithm, PreferenceAlgorithm)
        self.assertEqual(custom.condition(custom.freeze(custom.fit([paper()])), {"paper_id": "new"})[0]["rule_id"], "custom")


if __name__ == "__main__":
    unittest.main()
