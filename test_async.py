"""Test server-side batch submission and recovery with mocked HTTP requests."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import batch_async as asynchronous


@unittest.skipUnless(os.name == "posix", "Server-side batch recovery requires POSIX file locking")
class AsyncTests(unittest.TestCase):
    """Exercise batch state, result export and request limits on POSIX systems."""

    def setUp(self):
        """Create temporary task paths and an in-memory provider fixture."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = argparse.Namespace(inp=str(self.root / "tasks.jsonl"), out=str(self.root / "results.jsonl"),
                                       max_tokens=128, temperature=0, timeout=2,
                                       batch_wait=0, resume=False, parse="text")
        Path(self.args.inp).write_text("", encoding="utf-8")
        self.tasks = [{"custom_id": "first", "prompt": "hello"}, {"custom_id": "second", "prompt": "hello"}]
        self.provider = {"name": "compatible", "model": "model", "base_url": "https://api.example.com/v1",
                         "key": "sensitive-value", "key_env": "BATCH_TEST_KEY", "extra_body": {},
                         "pricing": {"model": {"in": 1, "out": 5}},
                         "max_tokens_field": "max_completion_tokens", "send_temperature": False,
                         "batch": {"endpoint": "/v1/chat/completions", "completion_window": "24h",
                                   "max_requests": 50000, "max_bytes": 200000000, "pricing_multiplier": 0.5}}

    def run_batch(self):
        """Run the fixture batch while suppressing progress messages."""
        with contextlib.redirect_stderr(io.StringIO()):
            asynchronous.run_async_batch(self.args, self.provider, self.tasks)

    def rows(self):
        """Return recovered output rows, or an empty list before output exists."""
        path = Path(self.args.out)
        return asynchronous.batch.load_results(str(path)) if path.exists() else []

    @staticmethod
    def result(identity):
        """Return a successful batch response for the given transport ID."""
        return {"custom_id": identity, "response": {"status_code": 200, "body": {
            "model": "model", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}}}

    def test_upload_and_results_return_in_input_id_order(self):
        """Check upload fields, input ordering and the configured cost multiplier."""
        calls = []

        def fake(provider, path, method="GET", payload=None, **options):
            calls.append(path)
            if path == "/files":
                self.assertIn(b'name="purpose"\r\n\r\nbatch', payload)
                self.assertIn(b'"max_completion_tokens":128', payload)
                self.assertNotIn(b'"temperature"', payload)
                return {"id": "file-input"}
            if path == "/batches":
                self.assertEqual(json.loads(payload)["endpoint"], "/v1/chat/completions")
                return {"id": "batch-complete"}
            if path.endswith("/content"):
                return b"\n".join(asynchronous.batch.encode(self.result(f"r-{index}")) for index in (1, 0))
            return {"status": "completed", "output_file_id": "file-output"}

        with patch.object(asynchronous, "http", side_effect=fake):
            self.run_batch()
        self.assertEqual([row["custom_id"] for row in self.rows()], ["first", "second"])
        self.assertTrue(all(row["usd"] == 0.0001 for row in self.rows()))
        self.assertEqual(calls[-1], "/files/file-output/content")

    def test_resume_pending_then_retry_only_failed(self):
        """Check pending-job reuse and resubmission of collected failures."""
        created, ready = [], False

        def fake(provider, path, method="GET", payload=None, **options):
            if path == "/files":
                created.append(payload)
                return {"id": "file-input"}
            if path == "/batches":
                return {"id": f"batch-{len(created)}"}
            if path.endswith("/content"):
                results = [self.result("r-0")]
                if len(created) == 1:
                    results.append({"custom_id": "r-1", "error": {"type": "expired"}})
                return b"\n".join(asynchronous.batch.encode(result) for result in results)
            return {"status": "completed" if ready else "in_progress", "output_file_id": "file-output"}

        with patch.object(asynchronous, "http", side_effect=fake):
            self.run_batch()
            self.assertEqual(len(created), 1)
            self.args.resume, ready = True, True
            with self.assertRaises(SystemExit) as raised:
                self.run_batch()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(len(created), 1)
            self.run_batch()
            self.run_batch()
        self.assertEqual(len(created), 2)
        self.assertEqual(len(self.rows()), 3)
        self.assertEqual(self.rows()[-1]["custom_id"], "second")

    def test_interruption_after_export_does_not_duplicate_output(self):
        """Check that exported rows prevent duplicate output after interruption."""
        def fake(provider, path, method="GET", payload=None, **options):
            if path == "/files":
                return {"id": "file-input"}
            if path == "/batches":
                return {"id": "batch-complete"}
            if path.endswith("/content"):
                return b"\n".join(asynchronous.batch.encode(self.result(f"r-{index}")) for index in range(2))
            return {"status": "completed", "output_file_id": "file-output"}

        original = asynchronous.save

        def interrupted_save(path, state):
            if state["batches"] and state["batches"][0].get("collected"):
                raise KeyboardInterrupt
            original(path, state)

        with patch.object(asynchronous, "http", side_effect=fake), patch.object(
                asynchronous, "save", side_effect=interrupted_save):
            with self.assertRaises(KeyboardInterrupt):
                self.run_batch()
        self.args.resume = True
        with patch.object(asynchronous, "http", side_effect=fake):
            self.run_batch()
        self.assertEqual(len(self.rows()), 2)

    def test_ambiguous_submission_is_not_repeated(self):
        """Check that a lost submission reply blocks automatic resubmission."""
        def fake(provider, path, method="GET", payload=None, **options):
            if path == "/files":
                return {"id": "file-input"}
            raise urllib.error.URLError("connection lost")

        with patch.object(asynchronous, "http", side_effect=fake), self.assertRaises(urllib.error.URLError):
            self.run_batch()
        self.args.resume = True
        with patch.object(asynchronous, "http") as request, self.assertRaises(ValueError):
            self.run_batch()
        request.assert_not_called()

    def test_split_limits_count_and_encoded_bytes(self):
        """Check request-count and JSONL-byte limits, including oversized rows."""
        requests = [
            (str(index), {"custom_id": str(index), "body": {"text": "sample"}})
            for index in range(5)
        ]
        chunks = list(asynchronous.split_requests(iter(requests), {"max_requests": 2, "max_bytes": 150}))
        self.assertEqual(sum(map(len, chunks)), 5)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 2)
            self.assertLessEqual(
                sum(len(asynchronous.batch.encode(request)) + 1 for _, request in chunk), 150)
        with self.assertRaises(ValueError):
            list(asynchronous.split_requests(iter(requests), {"max_requests": 10, "max_bytes": 1}))

    def test_batch_errors_are_redacted_and_unpriced(self):
        """Check API-key redaction and unknown cost for an error-only result."""
        entry = {"id": "batch-result", "submitted_at": 0}
        row = asynchronous.result_row({"error": {"message": "sensitive-value"}}, "first", entry, self.provider)
        self.assertIn("[REDACTED]", row["error"])
        self.assertNotIn("sensitive-value", row["error"])
        self.assertIsNone(row["usd"])

    def test_custom_messages_preserve_image_and_system_blocks(self):
        """Check that explicit messages retain system and image content."""
        task = {"custom_id": "image", "messages": [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]}]}
        identity, request = next(asynchronous.requests_for([task], self.args, self.provider))
        self.assertEqual(identity, "image")
        self.assertEqual(request["body"]["messages"], task["messages"])

    def test_resume_rejects_changed_model_before_network(self):
        """Check that a model change rejects saved state before HTTP requests."""
        def fake(provider, path, method="GET", payload=None, **options):
            if path == "/files":
                return {"id": "file-input"}
            if path == "/batches":
                return {"id": "batch-pending"}
            return {"status": "in_progress"}

        with patch.object(asynchronous, "http", side_effect=fake):
            self.run_batch()
        self.args.resume = True
        self.provider["model"] = "other-model"
        with patch.object(asynchronous, "http") as request, self.assertRaises(ValueError):
            self.run_batch()
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
