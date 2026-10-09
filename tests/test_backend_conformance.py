from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from helpers import make_eval_repo, run_cli

import skill_benchmark as sb
from agent_capabilities import BACKENDS


def stream(*records: dict) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


@dataclass(frozen=True)
class BackendProfile:
    healthy: str
    no_usage: str
    invalid: dict[str, str]
    served_model: str | None = None
    sidecar_flag: str | None = None


CLAUDE_RESULT = {"type": "result", "subtype": "success", "result": "token ok",
                 "is_error": False, "stop_reason": "end_turn"}
CLAUDE_MESSAGE = {"type": "assistant", "message": {
    "role": "assistant", "model": "fixture-model",
    "content": [{"type": "text", "text": "token ok"}]}}
CODEX_MESSAGE = {"type": "item.completed", "item": {
    "id": "answer", "type": "agent_message", "text": "trace answer"}}
GEMINI_INIT = {"type": "init", "timestamp": "t", "session_id": "s", "model": "fixture-model"}
GEMINI_MESSAGE = {"type": "message", "timestamp": "t", "role": "assistant", "content": "token ok"}
GEMINI_RESULT = {"type": "result", "timestamp": "t", "status": "success"}
GEMINI_STATS = {"total_tokens": 3, "input_tokens": 1, "output_tokens": 2,
                "cached": 0, "input": 1, "duration_ms": 25, "tool_calls": 0,
                "models": {"fixture-model": {"total_tokens": 3, "input_tokens": 1,
                                              "output_tokens": 2, "cached": 0, "input": 1}}}
VIBE_MESSAGE = {"role": "assistant", "content": "token ok"}
PROFILES = {
    "claude": BackendProfile(
        healthy=stream(CLAUDE_MESSAGE, {**CLAUDE_RESULT, "usage": {"input_tokens": 1, "output_tokens": 2}},
                       {"type": "system", "subtype": "task_summary"}),
        no_usage=stream(CLAUDE_MESSAGE, CLAUDE_RESULT,
                        {"type": "system", "subtype": "task_summary"}),
        invalid={"non-string result": stream({**CLAUDE_RESULT, "result": {"text": "token ok"}}),
                 "missing result": stream({"type": "result"}), "bad JSON": "{not json"},
        served_model="fixture-model"),
    "codex": BackendProfile(
        healthy=stream(CODEX_MESSAGE, {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}),
        no_usage=stream(CODEX_MESSAGE, {"type": "turn.completed"}),
        invalid={"missing sidecar": stream(CODEX_MESSAGE, {"type": "turn.completed"})},
        sidecar_flag="--output-last-message"),
    "gemini": BackendProfile(
        healthy=stream(GEMINI_INIT, GEMINI_MESSAGE, {**GEMINI_RESULT, "stats": GEMINI_STATS}),
        no_usage=stream(GEMINI_INIT, GEMINI_MESSAGE, GEMINI_RESULT),
        invalid={"bad JSON": "{not json", "missing terminal": stream(GEMINI_INIT, GEMINI_MESSAGE),
                 "bad token total": stream(GEMINI_INIT, GEMINI_MESSAGE,
                                           {**GEMINI_RESULT, "stats": {**GEMINI_STATS, "total_tokens": 0}})},
        served_model="fixture-model"),
    "vibe": BackendProfile(
        healthy=stream(VIBE_MESSAGE), no_usage=stream(VIBE_MESSAGE),
        invalid={"bad JSON": "{not json", "no assistant": stream({"role": "tool", "content": "token ok"}),
                 "non-string content": stream({**VIBE_MESSAGE, "content": {"text": "token ok"}})}),
}


@dataclass(frozen=True)
class RunArtifacts:
    output: str
    metadata: dict
    trace: str
    valid: bool


def run_backend(backend: str, stdout: str, *, sidecar: str = "token ok") -> RunArtifacts:
    profile = PROFILES[backend]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        manifest = make_eval_repo(root, cases=[{
            "id": "c", "split": "tune", "prompt": "do it",
            "assertions": [{"name": "answer", "type": "contains", "value": "token"}],
        }])
        tasks = root / "tasks.jsonl"
        code, _, stderr = run_cli("prepare", manifest, "--out", tasks)
        if code != 0:
            raise AssertionError(f"prepare exited {code}: {stderr}")
        row = next(json.loads(line) for line in tasks.read_text().splitlines()
                   if json.loads(line)["variant"] == "with_skill")
        tasks.write_text(json.dumps(row) + "\n", encoding="utf-8")
        cli = root / "fake-cli"
        body = [f"#!{sys.executable}", "import pathlib, sys",
                "if '--version' in sys.argv:", "    print('fixture-cli 1.0')", "    sys.exit(0)",
                "_ = sys.stdin.read()"]
        if profile.sidecar_flag:
            body.append(f"pathlib.Path(sys.argv[sys.argv.index({profile.sidecar_flag!r}) + 1]).write_text({sidecar!r})")
        body.append(f"sys.stdout.write({stdout!r})")
        cli.write_text("\n".join(body) + "\n", encoding="utf-8")
        cli.chmod(0o755)
        binding = BACKENDS[backend].answer
        option = binding.cli_options[0].flags[0]
        runs = root / "runs"
        code, _, stderr = run_cli("run-agent", "--agent", backend, "--tasks", tasks,
                                  "--runs", runs, "--model", "fixture-model", option, cli)
        if code != 0:
            raise AssertionError(f"run-agent exited {code}: {stderr}")
        base = runs / row["run_dir"]
        output = (base / "output.md").read_text(encoding="utf-8")
        metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
        metrics = sb.read_metrics_base(base)
        return RunArtifacts(output, metadata,
                            (base / "trace.jsonl").read_text(encoding="utf-8"),
                            sb.execution_valid(metrics, output))


def assert_failed_closed(case: unittest.TestCase, backend: str, run: RunArtifacts) -> None:
    case.assertEqual(run.metadata["returncode"], 0)
    case.assertIs(run.metadata["provider_response_complete"], False)
    case.assertTrue(run.output.startswith(BACKENDS[backend].failure_marker))
    case.assertFalse(run.valid)


class BackendConformanceTests(unittest.TestCase):
    def test_profiles_cover_current_native_answer_registry(self):
        self.assertEqual(set(PROFILES), {name for name, row in BACKENDS.items()
                                       if row.answer_route == "native"})

    def test_zero_exit_protocol_failures_do_not_become_answers(self):
        for backend, profile in PROFILES.items():
            for label, raw in profile.invalid.items():
                with self.subTest(backend=backend, shape=label):
                    run = run_backend(backend, raw, sidecar="")
                    assert_failed_closed(self, backend, run)
                    self.assertEqual(run.trace, raw)

    def test_healthy_answers_keep_raw_stdout_completion_and_telemetry(self):
        for backend, profile in PROFILES.items():
            with self.subTest(backend=backend):
                run = run_backend(backend, profile.healthy)
                self.assertEqual(run.output, "token ok")
                self.assertEqual(run.trace, profile.healthy)
                self.assertEqual(run.metadata["returncode"], 0)
                self.assertIs(run.metadata["provider_response_complete"], True)
                self.assertTrue(run.valid)
                self.assertEqual(run.metadata["served_model"], profile.served_model)
                self.assertEqual(run.metadata["served_model_check"],
                                 "match" if profile.served_model else "unavailable")
                self.assertEqual(run.metadata["cost_normalized"], {"source": "missing"})
                if BACKENDS[backend].capabilities.token_usage:
                    self.assertEqual(run.metadata["usage_normalized"]["total_tokens"], 3)
                    self.assertIn(run.metadata["usage_normalized"]["source"],
                                  {"provider_reported", "trace_normalized"})

    def test_missing_usage_does_not_become_a_measured_zero(self):
        for backend, profile in PROFILES.items():
            with self.subTest(backend=backend):
                run = run_backend(backend, profile.no_usage)
                self.assertEqual(run.output, "token ok")
                self.assertTrue(run.valid)
                self.assertEqual(run.metadata["usage_normalized"], {"source": "missing"})

    def test_shared_contract_rejects_a_parser_that_invents_success(self):
        raw = PROFILES["claude"].invalid["bad JSON"]
        invented = sb.RunnerOutcome(provider="claude", answer="token ok", returncode=0,
                                     trace_text=raw, model="fixture-model")
        with mock.patch.object(sb.AGENT_BACKENDS["claude"], "invoke_answer", return_value=invented):
            run = run_backend("claude", raw)
        self.assertEqual(run.output, "token ok")
        self.assertIs(run.metadata["provider_response_complete"], True)
        with self.assertRaisesRegex(AssertionError, "True is not False"):
            assert_failed_closed(self, "claude", run)
        assert_failed_closed(self, "claude", run_backend("claude", raw))
