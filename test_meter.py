"""Test usage accounting, concurrent task handling and CLI configuration."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import batch_call as batch


class UsageTests(unittest.TestCase):
    """Exercise request and result handling with temporary files and mock calls."""

    def setUp(self):
        """Create a temporary directory that is cleaned up after each test."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def provider(name="primary", model="model"):
        """Return a provider fixture with per-million input and output rates."""
        return {"name": name, "model": model, "base_url": "https://api.example.com/v1",
                "endpoint": "/chat/completions", "key": "sensitive-value", "extra_body": {},
                "max_tokens_field": "max_tokens", "send_temperature": True,
                "pricing": {model: {"in": 1, "out": 5}}}

    @staticmethod
    def response(text="ok", reason="stop", incoming=100, outgoing=20, model="model"):
        """Return a chat response fixture with configurable content and usage."""
        return {"model": model, "usage": {"prompt_tokens": incoming, "completion_tokens": outgoing},
                "choices": [{"message": {"content": text}, "finish_reason": reason}]}

    def run_task(self, providers=None, rescue=0, retries=0):
        """Return the result of one fixture task with the requested retry limits."""
        return batch.run_task({"custom_id": "item", "prompt": "hello"},
                              providers or [self.provider()], 32, 0, 1, retries, rescue)

    def test_nested_usage_and_metadata(self):
        """Check recursive counter sums, metadata replacement and input retention."""
        first = {"prompt_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 4}, "cached": True}
        second = {"prompt_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 6}, "cached": False}
        self.assertEqual(batch._add_usage(first, second), {
            "prompt_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 10}, "cached": False})
        self.assertEqual(first["prompt_tokens"], 10)
        self.assertEqual(batch._add_usage(first, None), first)

    def test_prices_are_configured_per_response_model(self):
        """Check cached-input pricing, multipliers and unknown response models."""
        usage = {"prompt_tokens": 100, "completion_tokens": 20,
                 "prompt_tokens_details": {"cached_tokens": 80},
                 "completion_tokens_details": {"reasoning_tokens": 15}}
        pricing = {"model": {"in": 1, "out": 5, "cached_in": 0.25}}
        self.assertEqual(batch._meter_cost(usage, pricing, "model"), 0.00014)
        self.assertEqual(batch._meter_cost(usage, pricing, "model", 0.5), 0.00007)
        self.assertIsNone(batch._meter_cost(usage, pricing, "unknown-model"))
        self.assertIsNone(batch._meter_cost(None, pricing, "model"))
        self.assertIsNone(batch._meter_cost(usage, {}, "model"))

    def test_empty_response_rescue_accounts_for_both_responses(self):
        """Check token-budget doubling and usage totals across rescued responses."""
        responses = [self.response("", "length", 10, 20), self.response("ok", "stop", 7, 2)]
        budgets = []

        def fake(provider, body, timeout):
            budgets.append(body["max_tokens"])
            return responses.pop(0)

        with patch.object(batch, "call_once", side_effect=fake) as call:
            row = self.run_task(rescue=1)
        self.assertIsNone(row["error"])
        self.assertEqual(row["usage"], {"prompt_tokens": 17, "completion_tokens": 22})
        self.assertEqual(row["usage_last"], {"prompt_tokens": 7, "completion_tokens": 2})
        self.assertEqual((row["responses"], row["attempts"]), (2, 2))
        self.assertEqual(row["usd"], 0.000127)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(budgets, [32, 64])

    def test_fallback_uses_configured_models_and_keeps_prior_usage(self):
        """Check provider fallback after invalid output and retention of its usage."""
        broken = {"model": "model", "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        fallback = self.provider("secondary", "other-model")
        with patch.object(batch, "call_once", side_effect=[broken, self.response(model="other-model")]):
            row = self.run_task([self.provider(), fallback])
        self.assertEqual((row["provider"], row["model"]), ("secondary", "other-model"))
        self.assertEqual(row["usage"]["prompt_tokens"], 105)
        self.assertEqual(row["responses"], 2)
        self.assertEqual(row["usd"], 0.00022)

    def test_unpriced_response_keeps_total_cost_unknown(self):
        """Check that a response without matching pricing produces a null cost."""
        with patch.object(batch, "call_once", return_value=self.response(model="unknown-model")):
            row = self.run_task()
        self.assertIsNone(row["usd"])
        self.assertEqual(row["model"], "unknown-model")

    def test_http_retry_and_retry_after(self):
        """Check retry counts and a numeric Retry-After delay after HTTP 429."""
        error = urllib.error.HTTPError("https://api.example.com/v1/chat/completions", 429,
                                       "rate limited", {"Retry-After": "3"}, io.BytesIO(b"busy"))
        with patch.object(batch, "call_once", side_effect=[error, self.response()]), patch.object(
                batch.time, "sleep") as sleep:
            row = self.run_task(retries=1)
        self.assertIsNone(row["error"])
        self.assertEqual((row["attempts"], row["responses"]), (2, 1))
        sleep.assert_called_once_with(3)

    def test_authentication_errors_are_not_retried_or_exposed(self):
        """Check that HTTP 401 is not retried and its API key is redacted."""
        error = urllib.error.HTTPError("https://api.example.com/v1/chat/completions", 401,
                                       "unauthorized", {}, io.BytesIO(b"sensitive-value"))
        with patch.object(batch, "call_once", side_effect=error) as call, patch.object(
                batch.time, "sleep") as sleep:
            row = self.run_task(retries=3)
        self.assertEqual(call.call_count, 1)
        sleep.assert_not_called()
        self.assertIn("HTTP 401", row["error"])
        self.assertIn("[REDACTED]", row["error"])
        self.assertNotIn("sensitive-value", row["error"])

    def test_message_and_request_configuration(self):
        """Check system and image messages, token field choice and task overrides."""
        provider = self.provider()
        provider.update(max_tokens_field="max_completion_tokens", send_temperature=False)
        task = {"system": "Be concise", "prompt": "Describe the image", "max_tokens": 64,
                "image": "https://example.com/image.png"}
        body = batch.request_body(provider, task, 32, 0)
        self.assertEqual(body["max_completion_tokens"], 64)
        self.assertNotIn("temperature", body)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["messages"][1]["content"][1]["type"], "image_url")

    def test_local_images_are_encoded(self):
        """Check local PNG embedding as a base64 data URL."""
        image = self.root / "image.png"
        image.write_bytes(b"sample")
        content = batch.image_to_content(str(image))
        self.assertEqual(content["image_url"]["url"], "data:image/png;base64,c2FtcGxl")

    def test_output_recovery_preserves_records_after_partial_tail(self):
        """Check result recovery when a partial tail precedes appended records."""
        output = self.root / "results.jsonl"
        output.write_text('{"custom_id":"done","error":null}\n{"custom_id":', encoding="utf-8")
        self.assertEqual(batch.done_ids(str(output)), {"done"})
        with batch.open_output(str(output)) as stream:
            stream.write('{"custom_id":"next","error":null}\n')
        self.assertEqual(batch.done_ids(str(output)), {"done", "next"})

    def test_input_requires_unique_ids_and_valid_jsonl(self):
        """Check rejection of duplicate IDs, missing IDs and invalid input JSON."""
        source = self.root / "tasks.jsonl"
        for content in (
            '{"custom_id":"a"}\n{"custom_id":"a"}\n',
            '{"prompt":"hello"}\n',
            '{broken}\n',
        ):
            source.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                batch.read_tasks(str(source))

    def test_optional_parsing_records_failures(self):
        """Check embedded JSON, score extraction and parse-error recording."""
        self.assertEqual(batch.extract_json('```json\n{"text":"brace }"}\n```'), {"text": "brace }"})
        text = 'Example: {"score":1}\n```json\n{"score":4}\n```'
        self.assertEqual(batch.postprocess({"text": text}, "score")["parsed"], 4)
        self.assertEqual(batch.extract_json("Result: [1, 2]"), [1, 2])
        self.assertEqual(batch.postprocess({"text": '{"score":4}'}, "score")["parsed"], 4)
        self.assertTrue(batch.postprocess({"text": "no result"}, "json")["parse_error"])

    def test_cli_configuration_and_resume_without_network(self):
        """Check CLI provider loading and skipping successful IDs on resume."""
        config = self.root / "providers.json"
        config.write_text(json.dumps({"compatible": {"base_url": "https://api.example.com/v1",
                                                   "key_env": "BATCH_TEST_KEY", "model": "model"}}), encoding="utf-8")
        source, output = self.root / "tasks.jsonl", self.root / "results.jsonl"
        source.write_text('{"custom_id":"item","prompt":"hello"}\n', encoding="utf-8")
        arguments = ["batch_call.py", "--config", str(config), "--provider", "compatible",
                     "--in", str(source), "--out", str(output)]
        with patch.dict(os.environ, {"BATCH_TEST_KEY": "sensitive-value"}), contextlib.redirect_stderr(
                io.StringIO()):
            with patch.object(sys, "argv", arguments), patch.object(batch, "call_once", return_value=self.response()):
                batch.main()
            with patch.object(sys, "argv", arguments + ["--resume"]), patch.object(batch, "call_once") as call:
                batch.main()
            call.assert_not_called()
        self.assertEqual(len(batch.load_results(str(output))), 1)

    def test_help_exits_before_reading_configuration(self):
        """Check that --help exits without opening the provider configuration."""
        with patch.object(sys, "argv", ["batch_call.py", "--help"]), contextlib.redirect_stdout(
                io.StringIO()), patch.object(batch, "load_providers") as load:
            with self.assertRaises(SystemExit) as raised:
                batch.main()
        self.assertEqual(raised.exception.code, 0)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
