import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from claude_inspection import _Inspector, inspect_stdout
from scbench_results import _Archive, _claude_events


class ClaudeInspectionTests(unittest.TestCase):
    def test_streamed_fragments_use_request_identity_and_correlate_skill_result(self):
        fixture = (
            Path(__file__).parent / "fixtures/claude/stream-json.plugin-skill.jsonl"
        )
        rows = [json.loads(line) for line in fixture.read_text().splitlines()]
        events, spans = inspect_stdout(enumerate(rows, 1))
        llms = [s for s in spans if s["kind"] == "llm"]
        self.assertEqual(len(llms), 2)
        first = llms[0]
        self.assertEqual(first["usage"]["input_tokens"], 2 + 9177 + 10364)
        self.assertEqual(first["usage"]["output_tokens"], 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["parent_id"], first["id"])
        self.assertEqual(events[0]["evidence"], "skill_completed")
        self.assertEqual(events[0]["skill_name"], "probe-plugin:tidy-commit")
        (prompt,) = [s for s in spans if s["kind"] == "prompt"]
        self.assertTrue(prompt["synthetic"])
        self.assertNotIn("prompt_text", prompt)

    def test_reconciles_message_only_stdout_with_request_identified_project(self):
        def assistant(**fields):
            return {
                "type": "assistant",
                "message": {"model": "claude", "id": "msg_a", "content": []},
                **fields,
            }

        for requests in (["req_a"], ["req_a", "req_b"]):
            with self.subTest(requests=requests):
                inspector = _Inspector()
                row = assistant()
                row["message"]["content"] = [
                    {"type": "tool_use", "name": "Read", "id": "read_a", "input": {}}
                ]
                inspector.read([(1, row)], "stdout")
                inspector.read(
                    enumerate((assistant(requestId=req) for req in requests), 1),
                    "project",
                    project=True,
                )
                events, spans = inspector.finish()
                model_ids = {s["id"] for s in spans if s["kind"] == "llm"}
                self.assertEqual(model_ids, {"claude-llm:" + r for r in requests})
                if len(requests) == 1:
                    self.assertIn(events[0]["parent_id"], model_ids)
                    self.assertEqual(inspector.message_ids["msg_a"], "claude-llm:req_a")
                else:
                    self.assertNotIn("msg_a", inspector.message_ids)
                    parent = next(s for s in spans if s["id"] == events[0]["parent_id"])
                    self.assertEqual(parent["evidence"], "ambiguous_request")

    def test_joins_native_requests_agents_and_opt_in_body_by_provider_identity(self):
        def write(root, name, rows):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps(row) for row in rows))
            return path

        def assistant(req, content, **fields):
            return {
                "type": "assistant",
                "message": {"model": "claude", "id": "msg-" + req, "content": content},
                **fields,
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            agent = root / "x/checkpoint_1/agent"
            tool = {
                "type": "tool_use",
                "id": "dispatch",
                "name": "Agent",
                "input": {"prompt": "private"},
            }
            write(
                agent,
                "stdout.jsonl",
                [
                    assistant("lead", [tool], session_id="s", request_id="lead"),
                    {
                        "type": "user",
                        "session_id": "s",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "dispatch",
                                    "content": "private",
                                }
                            ]
                        },
                        "tool_use_result": {"agentId": "worker"},
                    },
                ],
            )
            write(
                agent,
                "workspace/projects/-workspace/lead.jsonl",
                [
                    {
                        "type": "api-request",
                        "id": "a",
                        "sessionId": "s",
                        "timestamp": "2026-01-02T00:00:00Z",
                    },
                    assistant(
                        "lead", [tool], sessionId="s", requestId="lead", requestRef="a"
                    ),
                ],
            )
            write(
                agent,
                "workspace/projects/-workspace/subagents/worker.jsonl",
                [
                    {
                        "type": "api-request",
                        "id": "b",
                        "sessionId": "s",
                        "agentId": "worker",
                    },
                    assistant(
                        "worker",
                        [],
                        sessionId="s",
                        agentId="worker",
                        requestId="worker",
                        requestRef="b",
                    ),
                ],
            )
            open_index = write(
                agent,
                "workspace/raw-api/test/index.jsonl",
                [
                    {
                        "request_id": "lead",
                        "message_id": "msg-lead",
                        "request_file": "lead.request.json",
                        "response_file": "/container/raw-api/test/lead.response.json",
                    },
                    {
                        "request_id": "worker",
                        "request_file": "../escape.request.json",
                        "message_id": "msg-worker",
                        "response_file": "lead.response.json",
                    },
                ],
            )
            payload = b'{"messages":[]}'
            open_index.with_name("lead.request.json").write_bytes(payload)
            response = b'{"id":"msg-lead","content":[]}'
            open_index.with_name("lead.response.json").write_bytes(response)
            events, spans = _claude_events(_Archive(root, None), Path("x/checkpoint_1"))
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["status"], "completed")
            by_id = {s["id"]: s for s in spans}
            self.assertEqual(len([s for s in spans if s["kind"] == "llm"]), 2)
            child = by_id["claude-llm:worker"]
            self.assertEqual(by_id[child["parent_id"]]["parent_id"], events[0]["id"])
            self.assertFalse(child["request_body_available"])
            self.assertFalse(child["response_body_available"])
            self.assertEqual(
                by_id["claude-llm:lead"]["response_body_digest"],
                hashlib.sha256(response).hexdigest(),
            )
            self.assertEqual(
                by_id["claude-llm:lead"]["request_body_digest"],
                hashlib.sha256(payload).hexdigest(),
            )
