import io
import json
import unittest
from unittest.mock import Mock, patch

from journal_prefs.models import make_model, validate_config


def config(**updates):
    return dict({"mode": "local", "adapter": "openai_compatible",
                 "endpoint": "http://127.0.0.1:1234/v1/chat/completions", "model": "institution-selected-open-model"}, **updates)


def reply(data):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(data).encode()
    return response


class LocalModelOptionsTests(unittest.TestCase):
    def test_configured_open_model_id_generation_and_json_mode_are_forwarded(self):
        opener = Mock()
        opener.open.return_value = reply({"choices": [{"message": {"content": '{"score": 5}'}}]})
        with patch("socket.socket", side_effect=AssertionError("No real network")), \
             patch("urllib.request.build_opener", return_value=opener):
            model = make_model(config(temperature=0.2, max_tokens=900, response_format={"type": "json_object"}))
            result = model.review({"paper": {"text": "synthetic fixture"}, "rubric": {}, "historical_examples": []})
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(result, {"score": 5})
        self.assertEqual(payload["model"], "institution-selected-open-model")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 900)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(opener.open.call_count, 1)

    def test_invalid_generation_options_fail_before_transport(self):
        invalid = [{"temperature": float("nan")}, {"temperature": -1}, {"temperature": 3},
                   {"max_tokens": True}, {"max_tokens": 0}, {"max_tokens": 16385},
                   {"response_format": "json_object"}, {"response_format": {"type": "json_schema"}}]
        for option in invalid:
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_config(config(**option))

    def test_malformed_compatible_server_responses_have_clear_bounded_errors(self):
        for envelope in ({}, {"choices": []}, {"choices": [None]},
                         {"choices": [{"message": {"content": None}}]},
                         {"choices": [{"message": {"content": "[]"}}]}):
            opener = Mock()
            opener.open.return_value = reply(envelope)
            with self.subTest(envelope=envelope), patch("socket.socket", side_effect=AssertionError("No network")), \
                 patch("urllib.request.build_opener", return_value=opener), self.assertRaises(ValueError):
                make_model(config()).review({"paper": {}, "rubric": {}, "historical_examples": []})
            self.assertEqual(opener.open.call_count, 1)
