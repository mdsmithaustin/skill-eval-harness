import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import attest_answer_design, make_eval_repo, run_cli

from langsmith_export import build_export
from skill_benchmark import RunnerOutcome, write_runner_outcome
from trace_normalization import detect_trigger_records, normalize_trace_records

PROJECT = "85c33318-924b-47d1-b11e-ad3ea7b39c0d"
FIXTURES = Path(__file__).parent / "fixtures"


class NativeInspectionTests(unittest.TestCase):
    def test_successful_target_operations_have_distinct_evidence(self):
        target = Path("/tmp/example/SKILL.md")
        for name, args, error, expected in [
            ("Skill", {"skill": "example"}, False, True),
            ("Read", {"file_path": str(target)}, True, False),
            ("Bash", {"command": f"echo {target}"}, False, False),
            ("Read", {"file_path": "/tmp/different/SKILL.md"}, False, False),
        ]:
            with self.subTest(tool=name):
                records = [
                    {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "id": "x", "input": args}]}},
                    {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "x", "content": "", "is_error": error}]}},
                ]
                self.assertEqual(detect_trigger_records(records, [target], source="claude").triggered, expected)
        raw = [json.loads(line) for line in (FIXTURES / "claude/stream-json.plugin-skill.jsonl").read_text().splitlines()]
        events, _ = normalize_trace_records(raw, source="claude")
        self.assertTrue(any(e.get("synthetic") and e.get("timestamp") for e in events["events"]))

    def test_recorded_claude_requests_nested_skill_and_injection(self):
        source = FIXTURES / "claude/stream-json.plugin-skill.jsonl"
        records = [json.loads(line) for line in source.read_text().splitlines()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            manifest = make_eval_repo(root, skill_name="probe-plugin:tidy-commit",
                                      skill_paths=["skills/tidy-commit/SKILL.md"])
            for arm in ("with_skill", "without_skill"):
                raw = records
                if arm == "without_skill":
                    # Preserve the injected body and the request fragments, dropping the Skill operation.
                    raw = [r for r in records if not any(
                        b.get("type") in {"tool_use", "tool_result"}
                        for b in r.get("message", {}).get("content", []) if isinstance(b, dict))]
                write_runner_outcome(runs / "case-1" / arm / "run-1", RunnerOutcome(
                    provider="claude", returncode=0, answer="alpha",
                    metadata_extra={"target_skill_names": ["tidy-commit", "probe-plugin:tidy-commit"]},
                    trace_text="".join(json.dumps(r) + "\n" for r in raw)))
            attest_answer_design(manifest, runs)
            report = root / "benchmark.json"
            code, _, stderr = run_cli("benchmark", manifest, "--runs", runs, "--out", report)
            self.assertEqual(code, 0, stderr)
            doc = json.loads(report.read_text())
            plan = build_export(doc, project=PROJECT, study="capture", runs=runs, include_prompts=True)
            self.assertNotIn("Base directory for this skill", json.dumps(plan))
            for arm, invoked in (("with_skill", True), ("without_skill", False)):
                body = next(r["body"] for r in plan if r["body"].get("inputs", {}).get("variant") == arm)
                detail = body["extra"]["metadata"]
                self.assertEqual(detail["skill_invoked"], invoked)
                self.assertTrue(detail["skill_injected"])
                llms = [r["body"] for r in plan if r["body"].get("run_type") == "llm"
                        and r["body"]["trace_id"] == body["id"]]
                self.assertEqual(len(llms), 2)
                self.assertTrue(all(r["extra"]["metadata"]["ls_model_name"] == "claude-sonnet-5" for r in llms))

    def test_opt_in_codex_answer_captures_only_observed_thread_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = make_eval_repo(root)
            tasks, runs = root / "tasks.jsonl", root / "runs"
            self.assertEqual(run_cli("prepare", manifest, "--out", tasks)[0], 0)
            provider = root / "codex.py"
            provider.write_text("""import json, os, sys
from pathlib import Path
assert '--ephemeral' not in sys.argv
sys.stdin.read()
base=Path(os.environ['CODEX_HOME'])/'sessions'
base.mkdir()
ids=[f'12345678-1234-1234-1234-{i:012}' for i in range(3)]
for i,uid in enumerate(ids):
    rows=[{'type':'session_meta','payload':{'id':uid,'model_provider':'openai','parent_thread_id':ids[0] if i==1 else None}},
          {'type':'turn_context','payload':{'model':'test-model'}},
          {'type':'token_usage_record','payload':{'response_id':uid,'usage':{'input_tokens':3,'output_tokens':2}}}]
    (base/f'rollout-{uid}.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('alpha')
for r in [{'type':'thread.started','thread_id':ids[0]},{'type':'turn.started'},
          {'type':'item.completed','item':{'id':'i','type':'agent_message','text':'alpha'}},
          {'type':'turn.completed','usage':{'input_tokens':6,'output_tokens':4}}]:
    print(json.dumps(r))
""")
            with patch.dict(os.environ, {"SKILL_HARNESS_CODEX_ROLLOUTS": "1", "CODEX_HOME": str(root / "stub-home")}):
                code, _, stderr = run_cli("run-codex", "--tasks", tasks, "--runs", runs,
                                         "--codex-cmd", f'"{sys.executable}" "{provider}"')
            self.assertEqual(code, 0, stderr)
            self.assertEqual(len(list(runs.rglob("codex-rollouts/*.jsonl"))), 4)
            report = root / "benchmark.json"
            code, _, stderr = run_cli("benchmark", manifest, "--runs", runs, "--out", report)
            self.assertEqual(code, 0, stderr)
            plan = build_export(json.loads(report.read_text()), project=PROJECT, study="capture", runs=runs)
            runs_out = [r["body"] for r in plan if r["path"] == "runs"]
            llms = [r for r in runs_out if r["run_type"] == "llm"]
            self.assertEqual(len(llms), 4)
            self.assertEqual(len({r["parent_run_id"] for r in llms}), 4)
            self.assertEqual(runs_out[0]["extra"]["metadata"]["model_evidence"], "provider_requests")
            tasks_out = [r for r in runs_out if r["name"] == "Task prompt"]
            self.assertEqual(len(tasks_out), 2)
            self.assertTrue(all(r["inputs"] == {} for r in tasks_out))
            prompt_text = next(runs.rglob("task-prompt.txt")).read_text()
            explicit = build_export(json.loads(report.read_text()), project=PROJECT, study="capture",
                                    runs=runs, include_prompts=True)
            disclosed = [r["body"]["inputs"]["messages"][0]["content"] for r in explicit
                         if r["body"].get("name") == "Task prompt"]
            self.assertEqual(disclosed, [prompt_text, prompt_text])

            for rollout in runs.rglob("codex-rollouts/*.jsonl"):
                rollout.unlink()
            archived = build_export(json.loads(report.read_text()), project=PROJECT, study="capture", runs=runs)
            lead = archived[0]["body"]["extra"]["metadata"]
            self.assertEqual(lead["model_evidence"], "turn_aggregate_only")
            self.assertEqual(lead["observed_model_calls"], 0)
