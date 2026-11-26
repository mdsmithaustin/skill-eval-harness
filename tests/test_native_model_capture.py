import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import make_eval_repo, run_cli

from agent_clis import model_body_capture
from claude_inspection import inspect_stdout
from langsmith_export import native_inspection
from skill_benchmark import RunnerOutcome, write_runner_outcome


class NativeModelCaptureTests(unittest.TestCase):
    def test_capture_is_private_per_invocation_and_only_archives_provider_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            secret = root / "private.txt"
            secret.write_text("unrelated")
            for provider in ("codex", "claude"):
                with self.subTest(provider=provider):
                    with model_body_capture(provider, None) as disabled:
                        self.assertEqual(disabled, {})
                    destination = root / f"{provider}-api"
                    with model_body_capture(provider, destination) as environment:
                        location = Path(
                            next(iter(environment.values())).removeprefix("file:")
                        )
                        if provider == "codex":
                            parent = location / "trace-fixture" / "payloads"
                            parent.mkdir(parents=True)
                            saved, linked, hard = "1.json", "2.json", "3.json"
                        else:
                            parent = location
                            saved, linked, hard = (
                                "a.request.json",
                                "b.request.json",
                                "c.request.json",
                            )
                        (parent / saved).write_text("{}")
                        (parent / "unrelated.json").write_text("unrelated")
                        (parent / linked).symlink_to(secret)
                        os.link(secret, parent / hard)
                    self.assertFalse(location.exists())
                    files = [p for p in destination.rglob("*") if p.is_file()]
                    self.assertEqual([f.name for f in files], [saved])
                    self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)

    def test_codex_answer_capture_propagates_recorder_and_session_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            tasks, runs = root / "tasks.jsonl", root / "runs"
            self.assertEqual(
                run_cli("prepare", make_eval_repo(root), "--out", tasks)[0], 0
            )
            provider = root / "provider.py"
            provider.write_text(
                """import json, os, sys
from pathlib import Path
assert "--ephemeral" not in sys.argv
sys.stdin.read()
capture = Path(os.environ["CODEX_ROLLOUT_TRACE_ROOT"]) / "trace-sample"
capture.mkdir()
(capture / "manifest.json").write_text(json.dumps({"schema_version": 2}))
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text("alpha")
print(json.dumps({"type": "thread.started", "thread_id": "t"}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}}))
"""
            )
            with patch.dict(
                os.environ,
                {
                    "SKILL_HARNESS_MODEL_BODIES": "1",
                    "CODEX_HOME": str(root / "empty-home"),
                },
            ):
                code, _, stderr = run_cli(
                    "run-codex",
                    "--tasks",
                    tasks,
                    "--runs",
                    runs,
                    "--codex-cmd",
                    f'"{sys.executable}" "{provider}"',
                )
            self.assertEqual(code, 0, stderr)
            for arm in ("with_skill", "without_skill"):
                base = runs / "case-1" / arm
                self.assertTrue(
                    (base / "codex-api/trace-sample/manifest.json").is_file()
                )
                self.assertFalse(list(base.rglob("auth.json")))

    def test_claude_index_only_success_and_unfinished_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            capture = root / "claude-api" / "invocation"
            capture.mkdir(parents=True)
            req = {"model": "claude-test", "messages": []}
            resp = {
                "id": "msg",
                "model": "claude-test",
                "content": [],
                "usage": {"output_tokens": 4},
            }
            (capture / "a.request.json").write_text(json.dumps(req))
            (capture / "r.response.json").write_text(json.dumps(resp))
            (capture / "b.request.json").write_text(json.dumps(req))
            records = [
                {
                    "request_id": "r",
                    "message_id": "msg",
                    "timestamp": 1795735390,
                    "request_file": "a.request.json",
                    "response_file": "r.response.json",
                },
                {
                    "request_id": "x",
                    "message_id": "other",
                    "request_file": "b.request.json",
                    "response_file": "missing.response.json",
                },
            ]
            (capture / "index.jsonl").write_text(
                "\n".join(json.dumps(row) for row in records)
            )
            _, spans = inspect_stdout([], capture_root=root)
            llm, attempt = spans
            self.assertEqual(
                (llm["kind"], llm["model"], llm["usage"]["output_tokens"]),
                ("llm", "claude-test", 4),
            )
            self.assertEqual(
                (attempt["kind"], attempt["response_body_available"]),
                ("model_request_attempt", False),
            )
            for span in spans:
                for side in ("request", "response"):
                    if span[side + "_body_available"]:
                        path = root / span[side + "_body_source"]
                        self.assertEqual(
                            hashlib.sha256(path.read_bytes()).hexdigest(),
                            span[side + "_body_digest"],
                        )
            base = root / "runs/case-1/with_skill"
            sidecars = root / "sidecars"
            sidecars.mkdir()
            (root / "claude-api").rename(sidecars / "claude-api")
            write_runner_outcome(
                base,
                RunnerOutcome(
                    provider="claude",
                    returncode=0,
                    answer="alpha",
                    trace_text='{"type":"result","result":"alpha"}\n',
                ),
                sidecars=sidecars,
            )
            row = {
                "case_id": "case-1",
                "variant": "with_skill",
                "run_number": 1,
                "run_base": str(base),
            }
            self.assertEqual(
                native_inspection(root / "runs", row)[3], "provider_requests"
            )
            (base / "claude-api/invocation/a.request.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "inventory"):
                native_inspection(root / "runs", row)
