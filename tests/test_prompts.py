import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from journal_prefs.data import fingerprint
from journal_prefs.models import OpenAICompatibleModel
from journal_prefs.prompts import load_template, normalize_response, render_instructions, snapshot, validate_template

ROOT = Path(__file__).resolve().parent.parent


class PromptTests(unittest.TestCase):
    def test_all_examples_load_and_historical_categories_differ_from_generic(self):
        generic = load_template(ROOT / "prompts/institution_neutral.v1.json")
        historical = load_template(ROOT / "prompts/iclr2017_official_form.v1.json")
        self.assertEqual([a["score"] for a in historical["score_anchors"]], list(range(1, 11)))
        self.assertIn("boundary", historical["score_anchors"][4]["meaning"])
        self.assertNotEqual(historical["score_anchors"], generic["score_anchors"])
        self.assertIn("2016", render_instructions(historical))
        load_template(ROOT / "examples/custom_prompt.json")

    def test_snapshot_is_independent_and_any_template_edit_changes_hash(self):
        template = load_template(ROOT / "prompts/institution_neutral.v1.json")
        before = snapshot(template)
        template["score_anchors"][0]["meaning"] = "Institution replacement"
        self.assertNotEqual(before["template_sha256"], snapshot(template)["template_sha256"])
        self.assertEqual(before["template_sha256"], fingerprint(before["template"]))
        self.assertNotIn("Institution replacement", before["rendered_instructions"])

    def test_custom_output_names_normalize_without_provider_changes(self):
        template = load_template(ROOT / "examples/custom_prompt.json")
        response = {"rating": 6, "assessment": "SYNTHETIC", "passages": [{"quote": "text", "comment": "SYNTHETIC"}]}
        self.assertEqual(normalize_response(response, template)["score"], 6)
        with self.assertRaises(ValueError):
            normalize_response({"score": 6}, template)

    def test_invalid_ranges_aliases_and_noninteger_scores_are_rejected(self):
        original = load_template(ROOT / "prompts/iclr2017_official_form.v1.json")
        for mutate in (lambda t: t.update(score_range=[10, 1]),
                       lambda t: t["score_anchors"].append(deepcopy(t["score_anchors"][0])),
                       lambda t: t["response_fields"].update(review="score"),
                       lambda t: t["output_schema"]["properties"]["score"].update(maximum=9)):
            bad = deepcopy(original)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_template(bad)
        with self.assertRaises(ValueError):
            normalize_response({"score": 5.5, "review": "SYNTHETIC", "evidence": []}, original)

    def test_local_adapter_uses_prompt_and_field_mapping_without_network(self):
        template = load_template(ROOT / "examples/custom_prompt.json")
        model = OpenAICompatibleModel({"mode": "local", "adapter": "openai_compatible", "model": "synthetic-local",
                                       "endpoint": "http://127.0.0.1:9999/v1/chat/completions", "prompt_template": template})
        response = {"rating": 6, "assessment": "SYNTHETIC", "passages": [{"quote": "text", "comment": "SYNTHETIC"}]}
        with patch.object(model, "generate_json", return_value=response) as generate:
            result = model.review({"rubric": {"score_range": [1, 10]}, "paper": {"text": "text"}})
            self.assertEqual(result["score"], 6)
            self.assertEqual(generate.call_args.args[1], render_instructions(template))
        with patch.object(model, "generate_json") as generate:
            with self.assertRaises(ValueError):
                model.review({"rubric": {"score_range": [0, 1]}})
            generate.assert_not_called()

    def test_loading_prompt_is_data_only_no_expression_evaluation(self):
        template = load_template(ROOT / "prompts/institution_neutral.v1.json")
        template["instructions"] = "Literal ${not_code} and {not_interpolation}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompt.json"
            path.write_text(json.dumps(template), encoding="utf-8")
            self.assertTrue(render_instructions(load_template(path)).startswith(template["instructions"]))
