import json
import tempfile
import unittest
from pathlib import Path

from scbench_codex import inspect_rollouts
from scbench_results import attach_tool_content


def rows(*records):
    return list(
        enumerate(
            (
                {
                    "type": kind,
                    "timestamp": f"2026-11-01T00:00:{n:02}Z",
                    "payload": body,
                }
                for n, (kind, body) in enumerate(records)
            ),
            1,
        )
    )


class CodexRolloutTests(unittest.TestCase):
    def test_in_prose_skill_links_keep_targets_distinct_from_invocation(self):
        from scbench_codex import inspect_rollouts

        text = "Use [$one](/skills/one/SKILL.md) then [$two](/skills/two/SKILL.md) and [$skip](/other.md)"
        _, spans = inspect_rollouts(
            [
                (
                    "rollout.jsonl",
                    [
                        (
                            1,
                            {
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"text": text}],
                                },
                            },
                        )
                    ],
                )
            ]
        )
        prompt = next(span for span in spans if span["kind"] == "prompt")
        self.assertEqual(prompt["skill_names"], ["one", "two"])
        self.assertEqual(prompt["evidence"], "explicit_skill_request")

    def test_fork_history_and_retry_ids(self):
        request = {
            "thread_id": "lead",
            "turn_id": "lead-turn",
            "response_id": "resp1",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 30,
                "output_tokens": 4,
            },
        }
        call = {
            "type": "function_call",
            "call_id": "call1",
            "name": "shell",
            "arguments": "private args",
        }
        result = {
            "type": "function_call_output",
            "call_id": "call1",
            "output": "private output",
        }
        lead = rows(
            (
                "session_meta",
                {"id": "lead", "source": "exec", "model_provider": "openai"},
            ),
            ("turn_context", {"model": "lead-model"}),
            ("response_item", call),
            ("token_usage_record", request),
            ("response_item", result),
            ("token_usage_record", request),
        )
        sub = rows(
            (
                "session_meta",
                {"id": "sub", "parent_thread_id": "lead", "agent_role": "reviewer"},
            ),
            ("session_meta", {"id": "lead", "source": "exec"}),
            ("response_item", call),
            ("event_msg", {"type": "task_started", "turn_id": "review-turn"}),
            ("turn_context", {"model": "review-model"}),
            (
                "token_usage_record",
                {
                    **request,
                    "thread_id": "sub",
                    "turn_id": "review-turn",
                    "response_id": "resp2",
                },
            ),
        )
        early, _ = inspect_rollouts([("one", lead)])
        events, spans = inspect_rollouts([("two", sub), ("one", lead)])
        self.assertEqual(early, events)
        (tool,) = events
        self.assertTrue(tool["result_observed"])
        self.assertEqual(tool["result_source"], "one:5")
        self.assertNotIn("private", str(events))
        agents = {s["id"]: s for s in spans if s["kind"] == "agent"}
        models = [s for s in spans if s["kind"] == "llm"]
        self.assertEqual([s["model"] for s in models], ["review-model", "lead-model"])
        self.assertEqual(
            agents[models[0]["parent_id"]]["parent_id"], models[1]["parent_id"]
        )
        self.assertEqual(models[1]["usage"]["input_token_details"], {"cache_read": 30})

    def test_prompt_evidence_and_fallback_usage(self):
        def message(role, text):
            return (
                "response_item",
                {"type": "message", "role": role, "content": [{"text": text}]},
            )

        count = {
            "type": "token_count",
            "info": {
                "last_token_usage": {"output_tokens": 10},
                "total_token_usage": {"output_tokens": 50},
            },
        }
        parsed = [{"type": "read", "path": "/home/test/skills/verify/SKILL.md"}]

        def command(id_, code):
            return (
                "event_msg",
                {
                    "type": "item_completed",
                    "item": {
                        "type": "CommandExecution",
                        "id": id_,
                        "exit_code": code,
                        "parsed_cmd": parsed,
                    },
                },
            )

        lead = rows(
            ("session_meta", {"id": "lead"}),
            message("user", "$verify do the task"),
            message("user", "<skill>\n<name>verify</name>\n<body>private</body>"),
            message("developer", "private instructions"),
            command("a", 0),
            command("b", 1),
            ("event_msg", count),
            ("event_msg", count),
            (
                "event_msg",
                {
                    "type": "token_count",
                    "info": {"total_token_usage": {"output_tokens": 300}},
                },
            ),
        )
        _, spans = inspect_rollouts([("one", lead)])
        self.assertEqual(
            [s["prompt_text"] for s in spans if "prompt_text" in s],
            ["$verify do the task"],
        )
        self.assertNotIn("private", str(spans))
        self.assertEqual(
            [s.get("evidence") for s in spans if "skill_name" in s],
            ["explicit_skill_request", "injected_skill", "skill_file_read"],
        )
        self.assertEqual(sum(s["kind"] == "llm" for s in spans), 1)
        self.assertEqual(
            next(s for s in spans if s.get("exit_code") == 1)["status"], "failed"
        )


class SCBenchToolContentTests(unittest.TestCase):
    def test_verified_only_and_confined_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "arm" / "run"
            agent = root / "p/checkpoint_1/agent"
            agent.mkdir(parents=True)
            trace = agent / "rollout-1.jsonl"
            rel = trace.relative_to(root).as_posix()
            records = rows(
                ("session_meta", {"id": "lead"}),
                (
                    "response_item",
                    {
                        "type": "custom_tool_call",
                        "name": "check",
                        "call_id": "x",
                        "input": "private command",
                    },
                ),
                (
                    "response_item",
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "x",
                        "output": "private result",
                    },
                ),
            )
            trace.write_text(
                "\n".join(json.dumps(row) for _, row in records), encoding="utf-8"
            )
            events, spans = inspect_rollouts([(rel, records)])
            attempt = {
                "identity": {
                    "arm": "arm",
                    "run_id": "run",
                    "problem": "p",
                    "checkpoint": 1,
                },
                "events": events,
                "spans": spans,
            }
            self.assertNotIn("private", str(attempt))
            spans[0]["tool_input"] = "forged"
            events[0]["tool_output"] = "forged"
            attach_tool_content(attempt, root)
            self.assertNotIn("tool_input", spans[0])
            self.assertEqual(events[0]["tool_input"], "private command")
            self.assertEqual(events[0]["tool_output"], "private result")
            events[0]["output_digest"] = "different"
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                attach_tool_content(attempt, root)
            events[0]["source"] = "p/checkpoint_2/agent/rollout-1.jsonl:2"
            with self.assertRaises(ValueError):
                attach_tool_content(attempt, root)
