import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from langsmith_export import build_export
from scbench_codex import attach_inference_payloads, verify_inference_bindings
from scbench_results import import_run


class CodexInferenceCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "arm" / "run"
        agent = self.root / "case/checkpoint_1/agent"
        self.capture_dir = agent / "workspace/codex-api"
        self.trace = self.capture_dir / "trace-example"
        (self.trace / "payloads").mkdir(parents=True)
        (self.root / "checkpoint_results.jsonl").write_text(
            json.dumps({"problem": "case", "idx": 1, "checkpoint": "checkpoint_1"})
        )
        self.manifest = {
            "schema_version": 2,
            "root_thread_id": "thread",
            "rollout_id": "r",
        }
        (self.trace / "manifest.json").write_text(json.dumps(self.manifest))
        self.request = {"input": [{"text": "private request"}], "model": "model"}
        self.output = {
            "response_id": "resp",
            "token_usage": {"input_tokens": 11, "output_tokens": 4},
            "output_items": [
                {
                    "type": "function_call",
                    "call_id": "tool",
                    "arguments": "private args",
                }
            ],
        }
        for n, body in [(1, self.request), (2, self.output)]:
            (self.trace / f"payloads/{n}.json").write_text(json.dumps(body))
        self.start = {
            "type": "inference_started",
            "model": "model",
            "thread_id": "thread",
            "inference_call_id": "c",
            "provider_name": "openai",
            "request_payload": {
                "kind": {"type": "inference_request"},
                "path": "payloads/1.json",
            },
        }
        self.end = {
            "type": "inference_completed",
            "response_id": "resp",
            "inference_call_id": "c",
            "response_payload": {
                "kind": {"type": "inference_response"},
                "path": "payloads/2.json",
            },
        }
        self.write_trace(self.start, self.end)
        rollout = [
            {
                "type": "session_meta",
                "payload": {
                    "id": "thread",
                    "model_provider": "openai",
                    "source": "exec",
                },
            },
            {"type": "turn_context", "payload": {"model": "model"}},
            {
                "type": "token_usage_record",
                "payload": {
                    "response_id": "resp",
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "tool",
                    "name": "shell",
                    "arguments": "{}",
                },
            },
        ]
        (agent / "rollout-one.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rollout)
        )

    def write_trace(self, *events):
        log = [
            {
                "schema_version": 2,
                "rollout_id": "r",
                "thread_id": "thread",
                "wall_time_unix_ms": 1700000000000 + i,
                "payload": e,
            }
            for i, e in enumerate(events)
        ]
        (self.trace / "trace.jsonl").write_text(
            "\n".join(json.dumps(row) for row in log)
        )

    def test_bindings_preserve_ids_and_link_observed_tools(self):
        (attempt,) = import_run(self.root)["attempts"]
        (llm,) = [s for s in attempt["spans"] if s["kind"] == "llm"]
        self.assertEqual(llm["id"], "session:thread:response:resp")
        self.assertEqual(llm["status"], "completed")
        self.assertEqual(
            llm["request_body_digest"],
            hashlib.sha256(json.dumps(self.request).encode()).hexdigest(),
        )
        self.assertEqual(llm["response_body_format"], "codex_observed_output_items")
        self.assertEqual(attempt["events"][0]["parent_id"], llm["id"])
        self.assertNotIn("private", json.dumps(attempt))
        verify_inference_bindings([llm], self.capture_dir, self.root)
        forged = {
            **llm,
            "request_body_source": llm["response_body_source"],
            "request_body_digest": llm["response_body_digest"],
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_inference_bindings([forged], self.capture_dir, self.root)
        with self.assertRaisesRegex(ValueError, "aliases an SCBench input"):
            import_run(self.root, output=self.trace / "payloads/1.json")

    def test_recorder_retains_calls_alongside_unattributed_usage(self):
        path = self.root / "case/checkpoint_1/agent/rollout-one.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for usage in [
            {"type": "token_usage_record", "payload": {"usage": {"input_tokens": 11}}},
            {
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": {"input_tokens": 11}},
                },
            },
        ]:
            with self.subTest(usage=usage["type"]):
                rows[2] = usage
                path.write_text("\n".join(json.dumps(row) for row in rows))
                spans = import_run(self.root)["attempts"][0]["spans"]
                (call,) = [s for s in spans if s["kind"] == "llm"]
                self.assertTrue(call["response_body_available"])
                (metric,) = [s for s in spans if s["kind"] == "metric"]
                self.assertEqual(metric["usage"]["input_tokens"], 11)

    def test_missing_failed_and_partial_capture(self):
        self.write_trace(
            self.start,
            {
                **self.end,
                "type": "inference_failed",
                "partial_response_payload": self.end["response_payload"],
            },
            {**self.start, "inference_call_id": "retry"},
        )
        spans = []
        attach_inference_payloads(spans, self.capture_dir, self.root)
        failed, pending = [s for s in spans if s["kind"] == "llm"]
        self.assertEqual(
            (failed["status"], failed["response_body_format"]),
            ("failed", "codex_partial_output_items"),
        )
        self.assertEqual(
            (
                pending["status"],
                pending["request_body_available"],
                pending["response_body_available"],
            ),
            ("in_progress", True, False),
        )
        plan = build_export(
            import_run(self.root),
            project="85c33318-924b-47d1-b11e-ad3ea7b39c0d",
            study="interrupted",
        )
        models = [r["body"] for r in plan if r["body"].get("run_type") == "llm"]
        self.assertEqual(sum("end_time" not in model for model in models), 1)
        self.assertEqual(sum("error" in model for model in models), 1)
        (self.trace / "payloads/2.json").unlink()
        llm = next(
            s
            for s in import_run(self.root)["attempts"][0]["spans"]
            if s["id"] == failed["id"]
        )
        self.assertFalse(llm["response_body_available"])
        self.assertNotIn("response_body_source", llm)

    def test_truncated_capture_preserves_rollout_and_other_side(self):
        (self.trace / "payloads/2.json").write_text("{")
        (call,) = [
            s
            for s in import_run(self.root)["attempts"][0]["spans"]
            if s["kind"] == "llm"
        ]
        self.assertTrue(call["request_body_available"])
        self.assertFalse(call["response_body_available"])
        (self.trace / "manifest.json").write_text("{")
        (call,) = [
            s
            for s in import_run(self.root)["attempts"][0]["spans"]
            if s["kind"] == "llm"
        ]
        self.assertEqual(call["evidence"], "token_usage_record")

    def test_reject_crosslinks_and_unknown_recording(self):
        bad = copy.deepcopy(self.start)
        bad["request_payload"]["path"] = "../sensitive.json"
        self.write_trace(bad, self.end)
        with self.assertRaisesRegex(ValueError, "invalid Codex inference"):
            import_run(self.root)
        self.write_trace(self.start, self.end)
        self.output["response_id"] = "foreign"
        (self.trace / "payloads/2.json").write_text(json.dumps(self.output))
        (llm,) = [
            s
            for s in import_run(self.root)["attempts"][0]["spans"]
            if s["kind"] == "llm"
        ]
        self.assertFalse(llm["response_body_available"])
        (self.trace / "manifest.json").write_text(
            json.dumps({**self.manifest, "root_thread_id": "unrelated"})
        )
        (llm,) = [
            s
            for s in import_run(self.root)["attempts"][0]["spans"]
            if s["kind"] == "llm"
        ]
        self.assertNotIn("request_body_source", llm)
