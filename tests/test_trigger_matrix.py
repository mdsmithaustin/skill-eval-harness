"""The trigger matrix (run_trigger_matrix.py) measured offline and live.

Offline: the stub adapter runs the whole pipeline in CI with no model — the
demo skill's should-fire query triggers, the should-not-fire query doesn't,
and weakening the mounted description measurably under-triggers (the tuning
loop's core signal, reproduced deterministically). Claude-specific detection
and the observation-window rule are covered with canned event streams; Codex is
covered through its adapter contract and shared path-evidence detector.

Live (manual): RUN_AGENT_INVOKE_SMOKE=1 runs one cheap invocation for every
supported live trigger adapter/model to verify auth/network/process plumbing.
RUN_TRIGGER_SMOKE=1 runs the fuller Claude trigger matrix across haiku, sonnet,
and opus; RUN_CODEX_TRIGGER_SMOKE=1, RUN_PI_TRIGGER_SMOKE=1, and
RUN_VIBE_TRIGGER_SMOKE=1 run the same trigger path for those adapters:

    RUN_AGENT_INVOKE_SMOKE=1 python3 -m unittest tests.test_trigger_matrix.AgentInvokeSmokeTests -v
    RUN_TRIGGER_SMOKE=1 python3 -m unittest tests.test_trigger_matrix -v

Live smokes need the relevant CLI and API credentials, and spend real tokens.
The cheap agent smoke asserts invocation only; the trigger-matrix smokes assert
observed trigger-eval runs and at least one autonomous load.
"""
import contextlib
import functools
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from helpers import (
    CLAUDE_POST_RESULT_RECORDS,
    assert_dies,
    claude_streams_ending_after_result,
    run_cli,
    skill_markdown,
    stub_agent_cli,
)

import run_pi_trigger_eval as tr
import run_trigger_matrix as tm
import skill_benchmark as sb
from trigger_contracts import (
    InvocationOutcome,
    InvocationState,
    TriggerEvidenceKind,
    TriggerObservation,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO_MANIFEST = ROOT / "examples" / "demo-skill" / "evals" / "shared-benchmark.json"


def demo_trigger_rows():
    manifest = tm.load_manifest(DEMO_MANIFEST)
    return tm.cases_from_manifest(manifest, "tune")


def completed_invocation(stdout: str) -> InvocationOutcome:
    return InvocationOutcome.from_process(stdout=stdout, stderr="", returncode=0, elapsed_ms=1)


PI_STOP = {"type": "agent_end", "messages": [{"role": "assistant", "stopReason": "stop"}]}


def pi_stream(*events) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def pi_runs(fake):
    """Replace the Pi process boundary the matrix's Pi adapter calls."""
    return mock.patch.object(tm.PiAdapter, "_run_argv", staticmethod(fake))


def write_rows(root: Path, rows) -> Path:
    path = root / "rows.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def run_pi_cli(extra_argv, fake, out: Path):
    """Run `skill-pi-trigger-eval` on the demo manifest; return (exit code, report)."""
    argv = ["skill-pi-trigger-eval", str(DEMO_MANIFEST), *extra_argv, "--out", str(out)]
    with mock.patch.object(sys, "argv", argv), pi_runs(fake), mock.patch("builtins.print"):
        code = tr.main()
    return code, json.loads(out.read_text(encoding="utf-8"))


def matrix_cli(*argv: str | Path) -> tuple[int | str | None, str]:
    """Run `skill-trigger-matrix ARGV` in process through its real parser.
    Returns (exit status, stderr): main()'s return value, or the code of the
    SystemExit it raised (a refusal's message is that code)."""
    stderr = io.StringIO()
    with mock.patch.object(sys, "argv", ["skill-trigger-matrix", *map(str, argv)]), \
         contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
        try:
            code: int | str | None = tm.main()
        except SystemExit as exc:
            code = exc.code
    return code, stderr.getvalue()


def fake_trigger_claude(path: Path, probe: Path, *, invoke_project_skill: bool = False,
                        trailing_records: list[dict] | None = None) -> Path:
    """A fake `claude` for trigger runs. Its init event lists the skills Claude
    Code offers the model: one bundled skill, the project skills mounted in the
    working directory (by directory name, as Claude Code 2.1.269 lists them),
    the personal skills in its config dir, and an organisation skill when
    CLAUDE_CODE_SYNC_SKILLS is set. It records what it was offered and where it
    looked, and echoes any auth token it was given, as a leaky CLI would. With
    invoke_project_skill it calls the first project skill by that name;
    trailing_records are written after the `result` record."""
    path.write_text(f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
skills = ["update-config"]
skills += sorted(p.name for p in (Path.cwd() / ".claude" / "skills").iterdir())
if (config / "skills").is_dir():
    skills += sorted(p.name for p in (config / "skills").iterdir())
if os.environ.get("CLAUDE_CODE_SYNC_SKILLS"):
    skills.append("org-synced-skill")
project = sorted(p.name for p in (Path.cwd() / ".claude" / "skills").iterdir())
with open({str(probe)!r}, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({{"config": str(config), "cwd": os.getcwd(), "skills": skills,
                             "sync_skills": "CLAUDE_CODE_SYNC_SKILLS" in os.environ}}) + "\\n")
token = " ".join(os.environ.get(name, "") for name in
                 ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")).strip()
sys.stderr.write("auth: " + token + "\\n")
records = [{{"type": "system", "subtype": "init", "session_id": "s", "skills": skills}}]
if {invoke_project_skill!r}:
    records += [{{"type": "assistant", "message": {{"role": "assistant", "content": [
                    {{"type": "tool_use", "id": "toolu_1", "name": "Skill", "input": {{"skill": project[0]}}}}]}}}},
                {{"type": "user", "message": {{"role": "user", "content": [
                    {{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Launching skill"}}]}}}}]
records += [{{"type": "assistant", "message": {{"role": "assistant", "content": [
                {{"type": "text", "text": "Answered with " + token}}]}}}},
            {{"type": "result", "subtype": "success", "is_error": False,
              "result": "Answered.", "total_cost_usd": 0.001}}]
records += json.loads({json.dumps(trailing_records or [])!r})
for record in records:
    print(json.dumps(record))
""", encoding="utf-8")
    path.chmod(0o755)
    return path


def observe_pi(query, should_trigger, fake, *, trace_dir=None):
    """One Pi cell of the matrix on the demo skill's canonical tree."""
    manifest = tm.load_manifest(DEMO_MANIFEST)
    with tempfile.TemporaryDirectory() as td:
        tree, tree_hash, _ = tm.trigger_tree_for_manifest(
            sb.repo_root_for_manifest(DEMO_MANIFEST), manifest, Path(td), None)
        with pi_runs(fake):
            return tm.observe_cell_query(
                tm.PiAdapter(), tree, query, should_trigger, None, 12,
                trace_dir=trace_dir, metadata={"skill_tree_hash": tree_hash})


class TriggerRowBoundaryTests(unittest.TestCase):
    def test_eval_set_requires_real_boolean_should_trigger(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.json"
            path.write_text(json.dumps([{"query": "review this", "should_trigger": "false"}]), encoding="utf-8")
            code, _ = matrix_cli(DEMO_MANIFEST, "--eval-set", path, "--split", "tune", "--agent", "stub",
                                 "--out", Path(td) / "report.json")
        self.assertIn("should_trigger must be true or false", str(code))

    def test_the_protocol_rejects_nonpositive_concurrency_limits(self):
        for field, mutation in (
            ("timeout_seconds", {"timeout": 0, "runs_per_query": 1, "workers": 1}),
            ("runs_per_query", {"timeout": 1, "runs_per_query": 0, "workers": 1}),
            ("workers", {"timeout": 1, "runs_per_query": 1, "workers": 0}),
            ("workers", {"timeout": 1, "runs_per_query": 1, "workers": False}),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                tm.trigger_protocol([], None, **mutation)

    def test_matrix_rejects_zero_workers_before_constructing_an_executor(self):
        with self.assertRaisesRegex(SystemExit, "workers must be a positive integer"):
            tm.run_matrix(
                DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                models=[None], runs_per_query=1, timeout=30, workers=0,
            )

    def test_the_protocol_rejects_ambiguous_model_identities(self):
        for model in ("", "   ", False):
            with self.subTest(model=model), self.assertRaises(ValueError):
                tm.trigger_protocol(
                    [tm.AgentAdapter()], [model],
                    timeout=1, runs_per_query=1, workers=1)

    def test_eval_set_preserves_false_boolean(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.json"
            path.write_text(json.dumps({"evals": [{"query": "hello", "should_trigger": False}]}), encoding="utf-8")
            out = Path(td) / "report.json"
            code, stderr = matrix_cli(DEMO_MANIFEST, "--eval-set", path, "--split", "tune", "--agent", "stub",
                                      "--runs-per-query", "1", "--out", out)
            self.assertEqual(code, 0, stderr)
            rows = json.loads(out.read_text(encoding="utf-8"))["design"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "hello")
        self.assertIs(rows[0]["should_trigger"], False)
        self.assertRegex(rows[0]["query_id"], r"^query-[0-9a-f]{64}$")

    def test_duplicate_query_id_is_rejected_before_runs_are_scheduled(self):
        rows = [
            {"query_id": "same", "query": "one", "should_trigger": True},
            {"query_id": "same", "query": "two", "should_trigger": False},
        ]
        with self.assertRaisesRegex(SystemExit, "conflicting queries"):
            tm.validate_trigger_rows(rows, "fixture")

    def test_exact_duplicate_query_id_is_rejected_before_runs_are_scheduled(self):
        rows = [
            {"query_id": "same", "query": "one", "should_trigger": True},
            {"query_id": "same", "query": "one", "should_trigger": True},
        ]
        with self.assertRaisesRegex(SystemExit, "duplicate query_id"):
            tm.validate_trigger_rows(rows, "fixture")

    def test_distinct_ids_cannot_alias_the_same_authored_query(self):
        rows = [
            {"query_id": "first", "query": "one", "should_trigger": True},
            {"query_id": "second", "query": "one", "should_trigger": True},
        ]
        with self.assertRaisesRegex(SystemExit, "alias the same query"):
            tm.validate_trigger_rows(rows, "fixture")

    def test_cosmetic_query_variants_are_one_inference_identity(self):
        rows = [
            {"query_id": "first", "query": "Caf\u00e9   prompt", "should_trigger": True},
            {"query_id": "second", "query": "  CAFE\u0301 prompt\t", "should_trigger": True},
        ]
        with self.assertRaisesRegex(SystemExit, "alias the same query"):
            tm.validate_trigger_rows(rows, "fixture")

    def test_conflicting_id_aliases_are_rejected(self):
        rows = [{"id": "first", "query_id": "second",
                 "query": "one", "should_trigger": True}]
        with self.assertRaisesRegex(SystemExit, "conflicting query_id and id"):
            tm.validate_trigger_rows(rows, "fixture")

    def test_eval_set_rejects_evals_and_queries_aliases_together(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.json"
            row = {"query": "one", "should_trigger": True}
            path.write_text(json.dumps({"evals": [row], "queries": [row]}), encoding="utf-8")
            code, _ = matrix_cli(DEMO_MANIFEST, "--eval-set", path, "--split", "tune", "--agent", "stub",
                                 "--out", Path(td) / "report.json")
            self.assertIn("exactly one of evals or queries", str(code))

    def test_pi_cli_is_the_matrix_with_pi_home_outside_its_working_directory(self):
        seen = {}

        def fake_run(plan):
            config, cwd = Path(dict(plan.environment or {})["PI_CODING_AGENT_DIR"]), Path(plan.cwd)
            seen.update({
                "argv": list(plan.argv), "cwd": cwd, "config": config,
                "auth_copied": (config / "auth.json").is_file(),
                "home": sorted(path.name for path in config.iterdir()),
                "mounted": sorted(path.name for path in (config / "skills").iterdir()),
                # What Pi's read/grep/find/ls tools can reach from where it runs.
                "reachable": sorted(path.name for path in cwd.rglob("*")),
            })
            return completed_invocation(pi_stream(PI_STOP))

        with tempfile.TemporaryDirectory() as td:
            user_home = Path(td) / "user-pi"
            user_home.mkdir()
            (user_home / "auth.json").write_text('{"token": "user-token-123"}', encoding="utf-8")
            # Settings and a system prompt change behaviour; only auth is copied.
            (user_home / "settings.json").write_text('{"defaultThinkingLevel": "high"}', encoding="utf-8")
            (user_home / "AGENTS.md").write_text("Always load every skill.\n", encoding="utf-8")
            eval_set = write_rows(Path(td), [{"query_id": "negative", "query": "ordinary chat",
                                              "should_trigger": False}])
            with mock.patch.dict(os.environ, {"PI_CODING_AGENT_DIR": str(user_home)}):
                code, report = run_pi_cli(
                    ["--eval-set", str(eval_set), "--runs-per-query", "1", "--workers", "1"],
                    fake_run, Path(td) / "report.json")
        self.assertEqual(code, 0)
        self.assertEqual(report["protocol"]["producer"], "skill-trigger-matrix")
        self.assertEqual([adapter["agent"] for adapter in report["protocol"]["adapters"]], ["pi"])
        self.assertEqual(report["results"][0]["protocol_observation"],
                         {"config_isolated": True, "pi_home_outside_workdir": True})
        self.assertEqual(seen["argv"][0], "pi")
        self.assertNotEqual(seen["cwd"].resolve(), ROOT.resolve())
        self.assertNotEqual(seen["config"], user_home)
        self.assertFalse(seen["config"].is_relative_to(seen["cwd"]))
        self.assertTrue(seen["auth_copied"])
        self.assertEqual(seen["home"], ["auth.json", "skills"])
        self.assertTrue(seen["mounted"])
        self.assertNotIn("auth.json", seen["reachable"])
        self.assertNotIn("SKILL.md", seen["reachable"])
        self.assertFalse(seen["config"].exists(), "the Pi home and its copied auth are removed")

    def test_an_agent_home_is_removed_when_the_mounted_tree_fails_its_hash_check(self):
        for adapter in (tm.PiAdapter(), tm.CodexAdapter()):
            with self.subTest(agent=adapter.name), tempfile.TemporaryDirectory() as td:
                tree = Path(td) / "tree" / "demo"
                tree.mkdir(parents=True)
                (tree / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
                created = []
                real_mount = type(adapter).mount

                def spy(self, tree_dir, workspace, _real=real_mount, _created=created):
                    _created.append(workspace)
                    return _real(self, tree_dir, workspace)

                with mock.patch.object(type(adapter), "mount", spy), \
                     self.assertRaisesRegex(ValueError, "does not match"):
                    tm.observe_cell_query(adapter, tree.parent, "q", True, None, 5,
                                          metadata={"skill_tree_hash": "0" * 64})
                home = (adapter._pi_home if adapter.name == "pi" else adapter._codex_home)(created[0])
                self.assertFalse(home.exists())

    def test_an_agent_home_is_removed_when_the_agent_crashes(self):
        # The home outlives invoke() so its credentials can be scanned for
        # redaction; the cell still removes it when the agent process raises.
        def crash(plan):
            homes.append(Path(dict(plan.environment or {})[home_var]))
            raise RuntimeError("provider unavailable")

        for adapter_cls, home_var in ((tm.PiAdapter, "PI_CODING_AGENT_DIR"),
                                      (tm.CodexAdapter, "CODEX_HOME")):
            homes = []
            with self.subTest(agent=adapter_cls.name), tempfile.TemporaryDirectory() as td, \
                 mock.patch.object(adapter_cls, "_run_argv", staticmethod(crash)):
                tree = Path(td) / "tree"
                (tree / "demo").mkdir(parents=True)
                (tree / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                    tm.observe_cell_query(adapter_cls(), tree, "q", True, None, 5,
                                          metadata={"skill_tree_hash": sb.skill_tree_hash(tree)})
                self.assertEqual(len(homes), 1)
                self.assertFalse(homes[0].exists())

    def test_pi_ablation_report_names_the_edited_tree_on_every_repetition(self):
        with tempfile.TemporaryDirectory() as td:
            eval_set = write_rows(Path(td), [{"query_id": "pi-query", "query": "ordinary chat",
                                              "should_trigger": False}])
            code, report = run_pi_cli(
                ["--eval-set", str(eval_set), "--runs-per-query", "2", "--workers", "1",
                 "--ablation", "weaker-description"],
                lambda plan: completed_invocation(pi_stream(PI_STOP)), Path(td) / "report.json")
        self.assertEqual(code, 0)
        provenance = report["provenance"]
        self.assertEqual(report["skill_tree_hash"], provenance["skill_hash"])
        self.assertNotEqual(report["skill_tree_hash"], provenance["parent_skill_hash"])
        self.assertEqual(sorted((row["query_id"], row["run_number"]) for row in report["results"]),
                         [("pi-query", 1), ("pi-query", 2)])
        self.assertEqual({row["skill_tree_hash"] for row in report["results"]},
                         {report["skill_tree_hash"]})

    def test_pi_main_reports_are_accepted_by_trigger_comparer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eval_set = write_rows(root, [{"query_id": "negative", "query": "ordinary chat",
                                          "should_trigger": False}])
            reports = []
            for ablation in (None, "weaker-description"):
                extra = ["--eval-set", str(eval_set), "--runs-per-query", "1",
                         "--workers", "1", "--timeout", "12"]
                if ablation:
                    extra.extend(["--ablation", ablation])
                code, report = run_pi_cli(
                    extra, lambda plan: completed_invocation(pi_stream(PI_STOP)),
                    root / ("ablation.json" if ablation else "baseline.json"))
                self.assertEqual(code, 0)
                reports.append(report)
        compared = sb.build_trigger_comparison(reports[0], reports[1])
        self.assertTrue(compared["provenance"]["verified"])
        self.assertEqual(compared["paired"]["blocked"], [])

    def test_pi_main_excludes_incomplete_runs_from_pass_rate_denominator(self):
        timed_out = InvocationOutcome.from_process(
            stdout=json.dumps({"type": "agent_start"}) + "\n",
            stderr="timeout", returncode=124, elapsed_ms=1,
        )
        with tempfile.TemporaryDirectory() as td:
            eval_set = write_rows(Path(td), [{"query_id": "negative", "query": "ordinary chat",
                                              "should_trigger": False}])
            code, report = run_pi_cli(
                ["--eval-set", str(eval_set), "--runs-per-query", "1", "--workers", "1",
                 "--timeout", "1"],
                lambda plan: timed_out, Path(td) / "report.json")
        summary = report["summary"]
        self.assertEqual(code, 1)
        self.assertEqual(summary["measurement_status"], "incomplete")
        self.assertEqual(
            (summary["complete"], summary["incomplete"], summary["total"]),
            (0, 1, 1),
        )
        self.assertNotIn("pass_rate", summary)
        self.assertNotIn("observed_pass_rate", summary)

    def test_pi_json_provider_error_cannot_pass_a_negative_trigger(self):
        provider_error = json.dumps({
            "type": "agent_end", "willRetry": False,
            "messages": [{"role": "assistant", "content": [], "stopReason": "error",
                          "errorMessage": "Mistral API error (400): Invalid model",
                          "usage": {"input": 7, "output": 2, "totalTokens": 9,
                                    "cost": {"total": 0.009}}}],
        })

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.PiStream, "parse", wraps=tm.PiStream.parse) as parse_stream:
            trace_dir = Path(td) / "trace"
            result = observe_pi(
                "ordinary chat", False,
                lambda plan: completed_invocation(provider_error + "\n"),
                trace_dir=trace_dir).as_row()
            artifacts = [
                json.loads((trace_dir / name).read_text(encoding="utf-8"))
                for name in ("metrics.json", "metadata.json")
            ]
        # Detection, telemetry and the trace artifacts share one parsed stream.
        self.assertEqual(parse_stream.call_count, 1)
        self.assertFalse(result["observation_complete"])
        self.assertIsNone(result["pass"])
        self.assertIsNone(result["triggered"])
        self.assertEqual(result["usage_normalized"], {"source": "missing"})
        self.assertEqual(result["cost_normalized"], {"source": "missing"})
        self.assertIn("Invalid model", result["provider_error"])
        for artifact in artifacts:
            self.assertEqual(artifact["usage_normalized"], {"source": "missing"})
            self.assertEqual(artifact["cost_normalized"], {"source": "missing"})
            measurements = artifact["telemetry"]["measurements"]
            self.assertEqual(measurements["total_tokens"]["availability"], "unavailable")
            self.assertEqual(measurements["cost"]["availability"], "unavailable")

    def test_pi_runner_redacts_ambient_and_auth_secrets_before_writing(self):
        # One secret from the environment, one from the Pi auth the run copies;
        # the model echoes both into its stream and stderr.
        env_secret, auth_secret = "ambient-env-secret-123", "pi-auth-secret-456"

        def leaky_pi(plan):
            assistant = {"role": "assistant", "stopReason": "stop",
                         "content": [{"type": "text", "text": f"{env_secret} {auth_secret}"}]}
            return InvocationOutcome.from_process(
                stdout=json.dumps({"type": "agent_end", "messages": [assistant]}) + "\n",
                stderr=f"debug {env_secret} {auth_secret}", returncode=0, elapsed_ms=1)

        with tempfile.TemporaryDirectory() as td:
            pi_home = Path(td) / "pi-home"
            pi_home.mkdir()
            (pi_home / "auth.json").write_text(json.dumps({"token": auth_secret}), encoding="utf-8")
            trace_dir = Path(td) / "trace"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": env_secret,
                                              "PI_CODING_AGENT_DIR": str(pi_home)}):
                row = observe_pi("ordinary chat", False, leaky_pi, trace_dir=trace_dir).as_row()
            written = {path.name: path.read_text(encoding="utf-8")
                       for path in trace_dir.iterdir()}
        written["row"] = json.dumps(row)
        self.assertIn("[REDACTED]", written["trace.jsonl"])
        self.assertIn("[REDACTED]", row["stderr"])
        for name, text in written.items():
            with self.subTest(artifact=name):
                self.assertNotIn(env_secret, text)
                self.assertNotIn(auth_secret, text)

    def test_pi_adapter_propagates_json_provider_error_as_incomplete(self):
        provider_error = json.dumps({
            "type": "agent_end", "willRetry": False,
            "messages": [{"stopReason": "error", "errorMessage": "provider rejected model"}],
        })
        run = completed_invocation(provider_error)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm.PiAdapter, "_run_argv", staticmethod(lambda *args, **kwargs: run)):
            workspace = Path(td) / "workspace"
            workspace.mkdir()
            result = tm.PiAdapter().invoke("ordinary chat", None, workspace, 12)
        self.assertFalse(result.observation_complete)
        self.assertEqual(result.provider_error, "provider rejected model")

    def test_pi_matrix_detection_and_telemetry_share_one_parsed_stream(self):
        def successful_pi(plan):
            skill = Path(plan.environment["PI_CODING_AGENT_DIR"]) / "skills" / "demo" / "SKILL.md"
            assistant = {"role": "assistant", "stopReason": "stop",
                         "usage": {"input": 4, "output": 1, "totalTokens": 5}}
            stdout = "\n".join([
                json.dumps({"type": "tool_execution_start", "toolCallId": "call_1", "toolName": "read", "args": {"path": str(skill)}}),
                json.dumps({"type": "tool_execution_end", "toolCallId": "call_1", "toolName": "read", "result": "ok", "isError": False}),
                json.dumps({"type": "agent_end", "messages": [assistant]}),
            ]) + "\n"
            return InvocationOutcome.from_process(
                stdout=stdout, stderr="", returncode=0, elapsed_ms=1,
            )

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.PiAdapter, "_run_argv", staticmethod(successful_pi)), \
             mock.patch.object(tm.PiStream, "parse", wraps=tm.PiStream.parse) as parse_stream:
            tree = Path(td) / "tree"
            skill = tree / "demo"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
            row = tm.observe_cell_query(
                tm.PiAdapter(), tree, "review this", True, None, 12,
                metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
            ).as_row()
        self.assertEqual(parse_stream.call_count, 1)
        self.assertTrue(row["triggered"])
        self.assertEqual(row["usage_normalized"]["total_tokens"], 5)

    def test_pi_trace_artifacts_carry_the_detector_evidence_and_the_query(self):
        def reads_skill(plan):
            skill = Path(plan.environment["PI_CODING_AGENT_DIR"]) / "skills"
            mounted = next(skill.rglob("SKILL.md"))
            usage = {"input": 3, "output": 2, "totalTokens": 5}
            return completed_invocation(pi_stream(
                {"type": "tool_execution_start", "toolName": "read", "args": {"path": str(mounted)}},
                {"type": "tool_execution_end", "toolName": "read", "args": {"path": str(mounted)},
                 "result": "ok"},
                {"type": "agent_end", "messages": [{"role": "assistant", "stopReason": "stop",
                                                    "usage": usage}]}))

        with tempfile.TemporaryDirectory() as td:
            trace_dir = Path(td) / "trace"
            observe_pi("demo", True, reads_skill, trace_dir=trace_dir)
            metrics = json.loads((trace_dir / "metrics.json").read_text(encoding="utf-8"))
            meta = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["skill_invoked"])
        self.assertEqual(metrics["total_tokens"], 5)
        self.assertEqual((meta["query"], meta["should_trigger"], meta["pass"]), ("demo", True, True))

    def test_pi_timeout_with_parseable_partial_trace_is_not_telemetry_complete(self):
        def timed_out(plan):
            return InvocationOutcome.from_process(
                stdout=json.dumps({"type": "command", "command": "partial"}) + "\n",
                stderr="timeout", returncode=124, elapsed_ms=10,
            )

        with tempfile.TemporaryDirectory() as td:
            trace_dir = Path(td) / "trace"
            observe_pi("ordinary chat", False, timed_out, trace_dir=trace_dir)
            meta = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(meta["observation_complete"])
        self.assertEqual(meta["telemetry"]["measurements"]["commands"]["availability"], "unavailable")

    def test_a_token_the_agent_refreshes_during_the_run_is_redacted(self):
        # Pi and Codex rewrite auth.json in their home when they refresh an
        # OAuth token. The refreshed token exists only in the run's copy of the
        # home, never in the user's source auth, and the model echoes it.
        refreshed = "refreshed-token-written-during-the-run"
        home_vars = {"pi": "PI_CODING_AGENT_DIR", "codex": "CODEX_HOME"}

        def refreshing(agent):
            def run(plan):
                home = Path(dict(plan.environment or {})[home_vars[agent]])
                (home / "auth.json").write_text(json.dumps({"access_token": refreshed}),
                                                encoding="utf-8")
                if agent == "pi":
                    return completed_invocation(pi_stream({"type": "agent_end", "messages": [
                        {"role": "assistant", "stopReason": "stop",
                         "content": [{"type": "text", "text": f"token {refreshed}"}]}]}))
                return completed_invocation(pi_stream(
                    {"type": "thread.started", "thread_id": "t"}, {"type": "turn.started"},
                    {"type": "item.completed", "item": {"id": "item_0", "type": "agent_message",
                                                        "text": f"token {refreshed}"}},
                    {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}))
            return run

        rows = [{"query_id": "negative", "query": "ordinary chat", "should_trigger": False}]
        for agent, adapter_cls in (("pi", tm.PiAdapter), ("codex", tm.CodexAdapter)):
            with self.subTest(agent=agent), tempfile.TemporaryDirectory() as td:
                source = Path(td) / "user-home"
                source.mkdir()
                (source / "auth.json").write_text(json.dumps({"access_token": "token-before-refresh"}),
                                                  encoding="utf-8")
                traces = Path(td) / "traces"
                with mock.patch.dict(os.environ, {home_vars[agent]: str(source)}), \
                     mock.patch.object(adapter_cls, "_run_argv", staticmethod(refreshing(agent))):
                    report = tm.run_matrix(DEMO_MANIFEST, rows, agents=[agent], models=[None],
                                           runs_per_query=1, timeout=30, workers=1,
                                           trace_runs=traces)
                written = {str(path.relative_to(traces)): path.read_text(encoding="utf-8")
                           for path in traces.rglob("*") if path.is_file()}
                written["report"] = json.dumps(report)
                self.assertTrue(report["results"][0]["observation_complete"])
                for name, text in written.items():
                    with self.subTest(artifact=name):
                        self.assertNotIn(refreshed, text)
                self.assertTrue(any(name.endswith("trace.jsonl") and "[REDACTED]" in text
                                    for name, text in written.items()))

    def test_a_pi_cli_baseline_pairs_with_a_matrix_ablation_at_default_settings(self):
        # skill-pi-trigger-eval is the matrix with the Pi adapter; left at their
        # defaults, the two entry points must run one experimental protocol.
        def stops(plan):
            return completed_invocation(pi_stream(PI_STOP))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eval_set = write_rows(root, [{"query_id": "negative", "query": "ordinary chat",
                                          "should_trigger": False}])
            common = ["--eval-set", str(eval_set), "--runs-per-query", "1", "--workers", "1"]
            code, baseline = run_pi_cli(common, stops, root / "baseline.json")
            self.assertEqual(code, 0)
            argv = ["skill-trigger-matrix", str(DEMO_MANIFEST), "--agent", "pi", *common,
                    "--ablation", "weaker-description", "--out", str(root / "ablation.json")]
            with mock.patch.object(sys, "argv", argv), pi_runs(stops), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tm.main(), 0)
            code, stdout, stderr = run_cli("trigger-compare", "--baseline", root / "baseline.json",
                                           "--ablation", root / "ablation.json")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["provenance"]["reasons"], [])
        self.assertEqual(baseline["protocol"]["timeout_seconds"], 240)


class StubMatrixOfflineTests(unittest.TestCase):
    def test_every_matrix_adapter_has_an_explicit_trace_dialect(self):
        self.assertLessEqual(set(tm.ADAPTERS), set(sb.TRACE_DIALECTS))

    def test_stub_matrix_passes_both_polarities_per_model(self):
        report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows(), agents=["stub"],
                               models=["haiku", "sonnet", "opus"], runs_per_query=2,
                               timeout=30, workers=2)
        self.assertEqual(report["evidence_class"], "raw_autonomous_trigger_measurement")
        self.assertTrue(report["skill_tree_hash"])
        self.assertEqual(len(report["matrix"]), 3)   # one cell per model
        for cell in report["matrix"]:
            s = cell["summary"]
            self.assertEqual((s["should_trigger"]["passed"], s["should_trigger"]["total"]), (2, 2), cell["model"])
            self.assertEqual((s["should_not_trigger"]["passed"], s["should_not_trigger"]["total"]), (2, 2), cell["model"])
            self.assertEqual(s["incomplete_observations"], 0)
        self.assertEqual(report["summary"]["pass_rate"], 1.0)
        for row in report["results"]:
            self.assertEqual(row["usage_normalized"], {"source": "not_applicable"})
            self.assertEqual(row["cost_normalized"], {"source": "not_applicable"})
            self.assertIsInstance(row["elapsed_ms"], int)

    def test_trace_runs_are_written_for_matrix_agents(self):
        with tempfile.TemporaryDirectory() as td:
            trace_root = Path(td) / "traces"
            report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                   models=["offline"], runs_per_query=1,
                                   timeout=30, workers=1, trace_runs=trace_root)
            trace_dir = Path(report["results"][0]["trace_dir"])
            self.assertTrue((trace_dir / "trace.jsonl").is_file())
            self.assertTrue((trace_dir / "events.json").is_file())
            self.assertTrue((trace_dir / "metrics.json").is_file())
            meta = json.loads((trace_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["provider"], "stub")
            self.assertEqual(meta["population"], "trigger")
            self.assertEqual(meta["telemetry"]["population"], "trigger")
            self.assertEqual(meta["measurement"], "raw_measurement")
            self.assertEqual(report["results"][0]["measurement"], "raw_measurement")
            self.assertTrue(trace_dir.is_relative_to(trace_root))

    def test_trace_runs_use_unique_matrix_root_per_invocation(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm.time, "time", return_value=1234567890):
            trace_root = Path(td) / "traces"
            first = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                  models=["offline"], runs_per_query=2,
                                  timeout=30, workers=1, trace_runs=trace_root)
            second = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                   models=["offline"], runs_per_query=1,
                                   timeout=30, workers=1, trace_runs=trace_root)
        first_roots = {Path(r["trace_dir"]).relative_to(trace_root).parts[0] for r in first["results"]}
        second_roots = {Path(r["trace_dir"]).relative_to(trace_root).parts[0] for r in second["results"]}
        self.assertEqual(len(first_roots), 1)
        self.assertEqual(len(second_roots), 1)
        self.assertNotEqual(first_roots, second_roots)

    def test_trace_model_segment_is_path_safe(self):
        with tempfile.TemporaryDirectory() as td:
            trace_root = Path(td) / "traces"
            report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                   models=["../bad/model"], runs_per_query=1,
                                   timeout=30, workers=1, trace_runs=trace_root)
            trace_dir = Path(report["results"][0]["trace_dir"])
        self.assertTrue(trace_dir.is_relative_to(trace_root))
        parts = trace_dir.relative_to(trace_root).parts
        self.assertNotIn("..", parts)
        self.assertTrue(any(part.startswith("bad-model-") for part in parts), parts)

    def test_baseline_provenance_records_skill_tree_hash(self):
        report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                               models=[None], runs_per_query=1, timeout=30, workers=1)
        self.assertEqual(report["provenance"], {"mode": "baseline", "skill_tree_hash": report["skill_tree_hash"]})

    def test_demo_manifest_contains_documented_discovery_ablation(self):
        manifest = tm.load_manifest(DEMO_MANIFEST)
        repo_root = tm.repo_root_for_manifest(DEMO_MANIFEST)
        with tempfile.TemporaryDirectory() as td:
            _, tree_hash, provenance = tm.trigger_tree_for_manifest(repo_root, manifest, Path(td), "weaker-description")
        self.assertEqual(provenance["id"], "weaker-description")
        self.assertEqual(provenance["population"], "trigger")
        self.assertEqual(tree_hash, provenance["skill_hash"])
        self.assertNotIn("dir", provenance)
        self.assertNotIn("skill_files", provenance)

    def test_duplicate_agents_or_models_are_rejected(self):
        with self.assertRaises(SystemExit) as agent_ctx:
            tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub", "stub"],
                          models=[None], runs_per_query=1, timeout=30, workers=1)
        self.assertIn("duplicate --agent", str(agent_ctx.exception))
        with self.assertRaises(SystemExit) as model_ctx:
            tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                          models=["same", "same"], runs_per_query=1, timeout=30, workers=1)
        self.assertIn("duplicate --model", str(model_ctx.exception))

    def test_trace_write_failure_does_not_discard_observation(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm, "write_trace_artifacts", side_effect=OSError("ENOSPC")):
            report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                   models=[None], runs_per_query=1, timeout=30, workers=1,
                                   trace_runs=Path(td) / "traces")
        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertIn("ENOSPC", report["results"][0]["trace_error"])

    def test_worker_exception_becomes_incomplete_row(self):
        class FailingAdapter(tm.AgentAdapter):
            name = "stub"

            def mount(self, tree_dir, workspace):
                raise OSError("disk full")

            def invoke(self, query, model, workspace, timeout):
                raise AssertionError("unreachable")

        old = tm.ADAPTERS["stub"]
        try:
            tm.ADAPTERS["stub"] = FailingAdapter
            report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                   models=[None], runs_per_query=1, timeout=30, workers=1)
        finally:
            tm.ADAPTERS["stub"] = old
        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["summary"]["complete"], 0)
        self.assertEqual(report["summary"]["incomplete"], 1)
        self.assertEqual(report["summary"]["passed"], 0)
        self.assertNotIn("pass_rate", report["summary"])
        self.assertEqual(report["matrix"][0]["summary"]["incomplete_observations"], 1)
        self.assertEqual(report["matrix"][0]["queries"][0]["complete"], 0)
        self.assertNotIn("trigger_rate", report["matrix"][0]["queries"][0])
        self.assertIsNone(report["results"][0]["pass"])
        self.assertIsNone(report["results"][0]["triggered"])
        self.assertIn("disk full", report["results"][0]["error"])

        with mock.patch("sys.stdout") as stdout:
            tm.print_matrix(report["matrix"])
        rendered = " ".join(
            str(call.args[0]) for call in stdout.write.call_args_list if call.args
        )
        self.assertIn("INCOMPLETE", rendered)

    def test_trace_redacts_workspace_credentials(self):
        class SecretEchoAdapter(tm.AgentAdapter):
            name = "generic"

            def mount(self, tree_dir, workspace):
                (workspace / ".codex").mkdir(parents=True)
                (workspace / ".codex" / "auth.json").write_text('{"token":"SECRET-TOKEN-12345"}', encoding="utf-8")
                return self._mount_tree(tree_dir, workspace / "skills")

            def invoke(self, query, model, workspace, timeout):
                skill = next((workspace / "skills").glob("*/SKILL.md"))
                stdout = json.dumps({
                    "type": "command",
                    "command": ["bash", "-lc", f"cat {skill}; echo SECRET-TOKEN-12345"],
                }) + "\n"
                return {"stdout": stdout, "stderr": "", "returncode": 0, "timed_out": False,
                        "elapsed_ms": 1, "observation_complete": True}

        with tempfile.TemporaryDirectory() as td:
            tree = tm.build_canonical_skill_tree(tm.repo_root_for_manifest(DEMO_MANIFEST), tm.load_manifest(DEMO_MANIFEST), Path(td) / "tree")
            trace_dir = Path(td) / "trace"
            row = tm.observe_cell_query(
                SecretEchoAdapter(), tree, "q", True, None, 12, trace_dir,
                metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
            ).as_row()
            trace_text = (trace_dir / "trace.jsonl").read_text(encoding="utf-8")
            metadata_text = (trace_dir / "metadata.json").read_text(encoding="utf-8")
            metrics_text = (trace_dir / "metrics.json").read_text(encoding="utf-8")
        self.assertFalse(row["triggered"])
        self.assertNotIn("SECRET-TOKEN-12345", trace_text)
        self.assertNotIn("SECRET-TOKEN-12345", json.dumps(row))
        self.assertNotIn("SECRET-TOKEN-12345", metadata_text)
        self.assertNotIn("SECRET-TOKEN-12345", metrics_text)
        self.assertIn("[REDACTED]", trace_text)

    def test_weakened_description_under_triggers_offline(self):
        """The loop's core signal, deterministic: strip the description of the
        words users actually type and the (stub) agent stops loading the skill
        on the should-fire query."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            skill_dir = root / "skills" / "demo"
            (skill_dir / "references").mkdir(parents=True)
            source = (ROOT / "examples" / "demo-skill" / "skills" / "demo" / "SKILL.md").read_text(encoding="utf-8")
            weakened = source.replace(
                "description: Demo skill for the Skill Eval Harness example. Use it to review a proposed change and label the severity of each finding.",
                "description: General assistance helper.")
            (skill_dir / "SKILL.md").write_text(weakened, encoding="utf-8")
            (skill_dir / "references" / "checklist.md").write_text("checklist\n", encoding="utf-8")
            evals = root / "evals"
            evals.mkdir()
            manifest_path = evals / "shared-benchmark.json"
            manifest_path.write_text(json.dumps({
                "version": 1, "skill_name": "demo-reviewer",
                "skill_paths": ["skills/demo/SKILL.md"],
                "variants": ["with_skill", "without_skill"], "cases": [],
            }), encoding="utf-8")
            rows = [r for r in demo_trigger_rows() if r["should_trigger"]]
            report = tm.run_matrix(manifest_path, rows, agents=["stub"], models=["haiku"],
                                   runs_per_query=1, timeout=30, workers=1)
            cell = report["matrix"][0]
            self.assertEqual(cell["summary"]["should_trigger"]["passed"], 0,
                             "a description without the user's words must stop triggering the stub")

    def test_unknown_agent_names_the_extension_seam(self):
        with self.assertRaises(SystemExit) as ctx:
            tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows(), agents=["missing-agent"], models=None,
                          runs_per_query=1, timeout=30, workers=1)
        self.assertIn("AgentAdapter", str(ctx.exception))

    def test_models_that_sanitise_alike_keep_separate_trace_directories(self):
        models = ["vendor/model-a", "vendor:model-a"]
        with tempfile.TemporaryDirectory() as td:
            report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["stub"],
                                   models=models, runs_per_query=1, timeout=30, workers=1,
                                   trace_runs=Path(td) / "traces")
            recorded = {
                row["model"]: json.loads((Path(row["trace_dir"]) / "metadata.json")
                                         .read_text(encoding="utf-8"))["model"]
                for row in report["results"]}
            trace_dirs = {row["trace_dir"] for row in report["results"]}
        self.assertEqual(len(trace_dirs), 2)
        self.assertEqual(recorded, {model: model for model in models})


class TriggerCliStatusTests(unittest.TestCase):
    def test_matrix_cli_exits_nonzero_for_an_incomplete_report(self):
        # Only the Pi process boundary is replaced: the matrix itself decides
        # that crashed queries leave the measurement incomplete.
        def crash(plan):
            raise RuntimeError("provider unavailable")

        with tempfile.TemporaryDirectory() as td, pi_runs(crash), \
             contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(sys, "argv", [
                 "skill-trigger-matrix", str(DEMO_MANIFEST), "--agent", "pi",
                 "--runs-per-query", "1", "--out", str(Path(td) / "report.json"),
             ]):
            self.assertEqual(tm.main(), 1)
            report = json.loads((Path(td) / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["measurement_status"], "incomplete")

    def test_a_crashed_pi_query_is_an_incomplete_row_not_a_crashed_run(self):
        def crash(plan):
            raise RuntimeError("provider unavailable")

        with tempfile.TemporaryDirectory() as td:
            code, report = run_pi_cli(["--runs-per-query", "1"], crash, Path(td) / "report.json")
        self.assertEqual(code, 1)
        self.assertEqual(report["summary"]["measurement_status"], "incomplete")
        self.assertTrue(report["results"])
        self.assertEqual({row["error"] for row in report["results"]},
                         {"RuntimeError: provider unavailable"})

    def test_an_empty_model_is_an_argument_error_not_a_traceback(self):
        def unreachable(plan):
            raise AssertionError("no agent may run with an empty model")

        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "report.json")
            for main, argv in (
                    (tm.main, ["skill-trigger-matrix", str(DEMO_MANIFEST), "--agent", "pi",
                               "--model", "", "--out", out]),
                    (tr.main, ["skill-pi-trigger-eval", str(DEMO_MANIFEST), "--model", "",
                               "--out", out])):
                with self.subTest(argv[0]):
                    with mock.patch.object(sys, "argv", argv), pi_runs(unreachable), \
                         self.assertRaises(SystemExit) as ctx:
                        main()
                    self.assertIn("must be None or a non-empty string", str(ctx.exception.code))


class ClaudeDetectionTests(unittest.TestCase):
    """Canned claude -p stream-json fragments; no subprocess."""

    def _adapter(self):
        return tm.ClaudeAdapter()

    def test_completed_skill_tool_use_by_name_is_trigger_evidence(self):
        call = {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "load-1", "name": "Skill",
             "input": {"skill": "demo-reviewer", "args": "..."}}]}}
        completed = {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "load-1", "content": "loaded"}]}}
        stream = "\n".join(json.dumps(event) for event in [call, completed])
        detection = self._adapter().detect(completed_invocation(stream), ["demo-reviewer"], [])
        self.assertEqual(detection.legacy_evidence, ["Skill tool invoked: demo-reviewer"])
        incomplete = self._adapter().detect(completed_invocation(json.dumps(call)), ["demo-reviewer"], [])
        self.assertFalse(incomplete.triggered)
        completed["message"]["content"][0]["is_error"] = True
        failed_stream = "\n".join(json.dumps(event) for event in [call, completed])
        failed = self._adapter().detect(completed_invocation(failed_stream), ["demo-reviewer"], [])
        self.assertFalse(failed.triggered)

    def test_other_skills_and_plain_answers_are_not_evidence(self):
        stream = "\n".join([
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Skill", "input": {"skill": "code-review"}}]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "I would use the demo-reviewer skill here."}]}}),
        ])
        detection = self._adapter().detect(completed_invocation(stream), ["demo-reviewer"], [])
        self.assertFalse(detection.triggered, "a different skill firing, or the name in prose, is not load evidence")

    def test_reading_the_mounted_skill_md_is_fallback_evidence(self):
        mounted = Path("/tmp/trigger-x/.claude/skills/demo-reviewer/SKILL.md")
        stream = "\n".join([
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "read-1", "name": "Read",
                 "input": {"file_path": str(mounted)}}]}}),
            json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "read-1", "content": "ok"}]}}),
            json.dumps({"type": "result", "subtype": "success", "result": "done"}),
        ])
        detection = self._adapter().detect(completed_invocation(stream), ["demo-reviewer"], [mounted])
        self.assertTrue(detection.triggered)

    def test_max_turns_is_a_completed_observation_window(self):
        # Hitting --max-turns exits 1, but the model had its whole window to
        # load the skill, so a no-trigger here is a valid negative observation.
        stdout = "\n".join([
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Still thinking."}]}}),
            json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True}),
        ]) + "\n"

        def fake_run(*args, **kwargs):
            return InvocationOutcome.from_process(
                stdout=stdout, stderr="", returncode=1, elapsed_ms=3)

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            result = tm.ClaudeAdapter().invoke("q", "haiku", Path(td) / "run", 1)
        self.assertIs(result.state, InvocationState.COMPLETE)
        self.assertTrue(result.observation_complete)
        self.assertIsNone(result.provider_error)
        self.assertEqual(result.returncode, 1)

    def test_max_turns_subtype_does_not_reclassify_timeout_or_spawn_failure(self):
        stdout = json.dumps({"type": "result", "subtype": "error_max_turns"})
        for returncode, state in ((124, InvocationState.TIMED_OUT), (127, InvocationState.SPAWN_FAILED)):
            def fake_run(*args, _returncode=returncode, **kwargs):
                if _returncode == 124:
                    return InvocationOutcome.from_timeout(
                        stdout=stdout, stderr="failure", elapsed_ms=3)
                return InvocationOutcome.spawn_failed(
                    stdout=stdout, stderr="failure", elapsed_ms=3)

            with self.subTest(returncode=returncode), \
                 tempfile.TemporaryDirectory() as td, \
                 mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)), \
                 mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                result = tm.ClaudeAdapter().invoke("q", "haiku", Path(td) / "run", 1)
            self.assertIs(result.state, state)

    def test_mounted_skill_names_reports_folder_and_frontmatter(self):
        with tempfile.TemporaryDirectory() as td:
            skill_md = Path(td) / "some-dir" / "SKILL.md"
            skill_md.parent.mkdir()
            skill_md.write_text("---\nname: demo-reviewer\ndescription: x\n---\n", encoding="utf-8")
            bare = Path(td) / "bare"
            bare.mkdir()
            (bare / "SKILL.md").write_text("---\ndescription: x\n---\n", encoding="utf-8")
            self.assertEqual(tm.mounted_skill_names([skill_md, bare]), [
                tm.MountedSkillName(folder="some-dir", frontmatter="demo-reviewer"),
                tm.MountedSkillName(folder="bare", frontmatter="bare"),
            ])

    def test_claude_invoke_seeds_portable_auth_into_isolated_config(self):
        seen = {}

        def fake_run(plan):
            env = dict(plan.environment or {})
            config_dir = Path(env["CLAUDE_CONFIG_DIR"])
            seen["config_dir"] = config_dir
            seen["credentials"] = (config_dir / ".credentials.json").read_text(encoding="utf-8")
            return completed_invocation(json.dumps({"type": "result", "subtype": "success"}) + "\n")

        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)):
            root = Path(td)
            source = root / "user-claude"
            source.mkdir()
            (source / ".credentials.json").write_text('{"token":"t"}', encoding="utf-8")
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(source)}, clear=True):
                result = tm.ClaudeAdapter().invoke("q", "haiku", root / "run", 12)
        self.assertEqual(seen["credentials"], '{"token":"t"}')
        self.assertFalse(seen["config_dir"].is_relative_to(root / "run"))
        self.assertTrue(result.metadata["config_isolated"])
        self.assertNotIn("config_isolation_warning", result.metadata)

    def test_claude_invoke_preserves_nonportable_oauth_config(self):
        seen = {}

        def fake_run(plan):
            env = dict(plan.environment or {})
            seen["config_dir"] = env.get("CLAUDE_CONFIG_DIR")
            seen["sync_skills"] = env.get("CLAUDE_CODE_SYNC_SKILLS")
            return completed_invocation(json.dumps({"type": "result", "subtype": "success"}) + "\n")

        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)):
            root = Path(td)
            source = root / "oauth-backed-claude"
            source.mkdir()
            workspace = root / "run"
            env = {"CLAUDE_CONFIG_DIR": str(source), "CLAUDE_CODE_SYNC_SKILLS": "1"}
            with mock.patch.dict(os.environ, env, clear=True):
                result = tm.ClaudeAdapter().invoke("q", "haiku", workspace, 12)
        self.assertEqual(seen["config_dir"], str(source))
        self.assertFalse(tm.ClaudeAdapter._config_dir(workspace).exists())
        self.assertFalse(result.metadata["config_isolated"])
        self.assertIn("personal config may influence", result.metadata["config_isolation_warning"])
        # Not isolated: the run is left as the user's own CLI would see it, and
        # a stream with no init skill list is no evidence about competitors.
        self.assertEqual(seen["sync_skills"], "1")
        self.assertNotIn("competing_skills", result.metadata)

    def test_claude_malformed_stream_is_not_a_valid_negative_observation(self):
        def fake_run(*args, **kwargs):
            return InvocationOutcome.from_process(
                stdout="\n".join([
                    json.dumps({
                        "type": "assistant",
                        "message": {"content": {"type": "tool_use", "name": "Skill"}},
                    }),
                    json.dumps({"type": "result", "subtype": "success"}),
                ]) + "\n",
                stderr="", returncode=0, elapsed_ms=1,
            )

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            result = tm.ClaudeAdapter().invoke("q", "haiku", Path(td) / "run", 1)
        self.assertIs(result.state, InvocationState.PROVIDER_FAILED)
        self.assertIn("protocol error", result.provider_error or "")

    def test_a_recorded_claude_stream_is_a_complete_observation_with_skill_evidence(self):
        # Real Claude Code output (tests/fixtures/claude/README.md): system
        # events, thinking blocks and parent_tool_use_id: null that the
        # canned fragments above never carry must not fail the protocol checks.
        recorded = (Path(__file__).parent / "fixtures" / "claude"
                    / "stream-json.plugin-skill.jsonl").read_text(encoding="utf-8")

        def fake_run(*args, **kwargs):
            return InvocationOutcome.from_process(stdout=recorded, stderr="", returncode=0, elapsed_ms=1)

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            result = tm.ClaudeAdapter().invoke("q", None, Path(td) / "run", 1)
        self.assertIs(result.state, InvocationState.COMPLETE)
        self.assertIsNone(result.provider_error)
        detection = self._adapter().detect(result, ["probe-plugin:tidy-commit"], [])
        self.assertEqual(detection.legacy_evidence, ["Skill tool invoked: probe-plugin:tidy-commit"])
        # Every recording that continues after `result` is a complete
        # observation too; the skills it invoked are read off the stream itself.
        for source, text in claude_streams_ending_after_result():
            invoked = [str(block["input"]["skill"])
                       for record in map(json.loads, filter(str.strip, text.splitlines()))
                       if record.get("type") == "assistant"
                       for block in record["message"]["content"]
                       if block.get("type") == "tool_use" and block.get("name") == "Skill"]
            with self.subTest(source=source), tempfile.TemporaryDirectory() as td, \
                 mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(
                     lambda plan, text=text: completed_invocation(text))), \
                 mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
                result = tm.ClaudeAdapter().invoke("q", None, Path(td) / "run", 1)
                self.assertIs(result.state, InvocationState.COMPLETE, result.provider_error)
                self.assertIsNone(result.provider_error)
                detection = self._adapter().detect(result, invoked, [])
                self.assertEqual(detection.legacy_evidence, [f"Skill tool invoked: {name}" for name in invoked][:5])

    def test_skill_tool_called_by_mounted_directory_name_is_trigger_evidence(self):
        # Claude Code 2.1.269 invokes a project skill by the directory it is
        # mounted under, `demo` for skills/demo/SKILL.md, not by its declared
        # name `demo-reviewer` (#85 recorded 0/3 should-fire on Haiku and Sonnet
        # for a skill a traced run showed being invoked). The flattened mount
        # key of earlier versions is not a name anything is mounted under now.
        def invoking(skill):
            records = [
                {"type": "system", "subtype": "init", "session_id": "s"},
                {"type": "assistant", "message": {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Skill", "input": {"skill": skill}}]}},
                {"type": "user", "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Launching skill"}]}},
                {"type": "assistant", "message": {"role": "assistant", "content": [
                    {"type": "text", "text": "Reviewed."}]}},
                {"type": "result", "subtype": "success", "is_error": False, "result": "Reviewed."},
            ]
            stdout = "".join(json.dumps(record) + "\n" for record in records)
            return lambda plan: completed_invocation(stdout)

        should_fire = [row for row in demo_trigger_rows() if row["should_trigger"]][:1]
        for skill, triggered in (("demo", True), ("skills_demo_SKILL.md", False), ("other", False)):
            with self.subTest(skill=skill), \
                 mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(invoking(skill))), \
                 mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                report = tm.run_matrix(DEMO_MANIFEST, should_fire, agents=["claude"], models=["haiku"],
                                       runs_per_query=1, timeout=30, workers=1)
                (row,) = report["results"]
                self.assertEqual(report["summary"]["measurement_status"], "complete")
                self.assertIs(row["triggered"], triggered)
                self.assertEqual(row["evidence"], [f"Skill tool invoked: {skill}"] if triggered else [])

    def test_claude_auth_tokens_from_the_environment_are_redacted(self):
        # Claude Code authenticates from these variables too; a CLI that
        # prints one (a debug line, an auth error) must not write it out.
        tokens = {"CLAUDE_CODE_OAUTH_TOKEN": "oauth-token-from-setup-token",
                  "ANTHROPIC_AUTH_TOKEN": "bearer-token-for-a-gateway"}

        def leaky(plan):
            env = dict(plan.environment or {})
            echoed = " ".join(env[name] for name in tokens)
            stdout = json.dumps({"type": "result", "subtype": "success", "result": echoed}) + "\n"
            return InvocationOutcome.from_process(stdout=stdout, stderr=f"auth: {echoed}",
                                                  returncode=0, elapsed_ms=1)

        rows = [{"query_id": "negative", "query": "ordinary chat", "should_trigger": False}]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(leaky)), \
             mock.patch.dict(os.environ, {**tokens, "CLAUDE_CONFIG_DIR": str(Path(td) / "no-config")}):
            traces = Path(td) / "traces"
            report = tm.run_matrix(DEMO_MANIFEST, rows, agents=["claude"], models=["haiku"],
                                   runs_per_query=1, timeout=30, workers=1, trace_runs=traces)
            written = {str(path.relative_to(traces)): path.read_text(encoding="utf-8")
                       for path in traces.rglob("*") if path.is_file()}
        written["report"] = json.dumps(report)
        self.assertEqual(report["results"][0]["stderr"], "auth: [REDACTED] [REDACTED]")
        for name, text in written.items():
            for token in tokens.values():
                with self.subTest(artifact=name, token=token):
                    self.assertNotIn(token, text)

    def test_copied_claude_credentials_sit_outside_the_working_directory(self):
        # Claude runs with Read and Glob from its working directory, so the
        # config dir holding the copied OAuth credentials must not be in it.
        seen = {}

        def fake_run(plan):
            config, cwd = Path(dict(plan.environment or {})["CLAUDE_CONFIG_DIR"]), Path(plan.cwd)
            seen.update({
                "config": config, "cwd": cwd,
                "credentials": (config / ".credentials.json").read_text(encoding="utf-8"),
                "reachable": sorted(path.name for path in cwd.rglob("*")),
            })
            return completed_invocation(
                json.dumps({"type": "result", "subtype": "success", "result": "ok"}) + "\n")

        rows = [{"query_id": "negative", "query": "ordinary chat", "should_trigger": False}]
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.ClaudeAdapter, "_run_argv", staticmethod(fake_run)):
            source = Path(td) / "user-claude"
            source.mkdir()
            (source / ".credentials.json").write_text(
                '{"claudeAiOauth": {"accessToken": "user-oauth-access-token"}}', encoding="utf-8")
            paths = {}
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": str(source)}):
                for arm, ablation in (("baseline", None), ("ablation", "weaker-description")):
                    report = tm.run_matrix(DEMO_MANIFEST, rows, agents=["claude"], models=["haiku"],
                                           runs_per_query=1, timeout=30, workers=1, ablation=ablation)
                    paths[arm] = Path(td) / f"{arm}.json"
                    paths[arm].write_text(json.dumps(report), encoding="utf-8")
            code, stdout, stderr = run_cli("trigger-compare", "--baseline", paths["baseline"],
                                           "--ablation", paths["ablation"])
        self.assertIn("user-oauth-access-token", seen["credentials"])
        self.assertFalse(seen["config"].is_relative_to(seen["cwd"]))
        self.assertNotIn(".credentials.json", seen["reachable"])
        self.assertFalse(seen["config"].exists(), "the copied config is removed with the cell")
        self.assertEqual(report["results"][0]["protocol_observation"],
                         {"config_isolated": True, "claude_config_outside_workdir": True})
        # trigger-compare requires the same controls the adapter declares.
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["paired"]["blocked"], [])

    def test_env_auth_isolates_the_claude_config_so_trigger_compare_accepts_the_cells(self):
        # CI logs Claude in through an environment variable, with no credentials
        # file to copy. An empty config dir still authenticates then, so the run
        # is isolated: the user's personal skill and the organisation's synced
        # skills must not compete with the skill under test.
        for var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            with self.subTest(auth=var), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                token = f"{var.lower()}-secret-value"
                personal = root / "user-claude"
                (personal / "skills" / "personal-helper").mkdir(parents=True)
                probe = root / "probe.jsonl"
                claude = fake_trigger_claude(root / "claude", probe)
                env = {"PATH": os.environ.get("PATH", ""), "HOME": str(root), var: token,
                       "CLAUDE_CONFIG_DIR": str(personal), "CLAUDE_CODE_SYNC_SKILLS": "1"}
                reports = {}
                with mock.patch.dict(os.environ, env, clear=True), \
                     contextlib.redirect_stdout(io.StringIO()):
                    for arm, extra in (("baseline", []), ("ablation", ["--ablation", "weaker-description"])):
                        reports[arm] = root / f"{arm}.json"
                        with mock.patch.object(sys, "argv", [
                                "skill-trigger-matrix", str(DEMO_MANIFEST), "--agent", "claude",
                                "--model", "haiku", "--runs-per-query", "1", "--workers", "1",
                                "--claude-bin", str(claude), "--trace-runs", str(root / "traces" / arm),
                                "--out", str(reports[arm]), *extra]):
                            self.assertEqual(tm.main(), 0)
                    code, stdout, stderr = run_cli("trigger-compare", "--baseline", reports["baseline"],
                                                   "--ablation", reports["ablation"])
                rows = [row for path in reports.values()
                        for row in json.loads(path.read_text(encoding="utf-8"))["results"]]
                runs = [json.loads(line) for line in probe.read_text(encoding="utf-8").splitlines()]
                artifacts = {str(path): path.read_text(encoding="utf-8")
                             for path in [*reports.values(), *(root / "traces").rglob("*")] if path.is_file()}
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(row["protocol_observation"],
                                     {"config_isolated": True, "claude_config_outside_workdir": True})
                    self.assertNotIn("config_isolation_warning", row)
                    # Only the skill bundled with the CLI competed with the mounted one.
                    self.assertEqual(row.get("competing_skills"), ["update-config"])
                self.assertEqual(len(runs), len(rows))
                for run in runs:
                    config = Path(run["config"])
                    self.assertNotEqual(config, personal)
                    self.assertFalse(config.is_relative_to(Path(run["cwd"])))
                    self.assertFalse(config.exists(), "the isolated config is removed with the cell")
                    self.assertFalse(run["sync_skills"])
                self.assertEqual(code, 0, stderr)
                self.assertEqual(json.loads(stdout)["paired"]["blocked"], [])
                for name, text in artifacts.items():
                    self.assertNotIn(token, text, name)

    def test_the_model_is_offered_the_skill_under_its_own_directory_name(self):
        # A user who installs skills/demo/ sees a skill named `demo`; the trigger
        # run must offer the model that name, not the flattened manifest path.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            probe = root / "probe.jsonl"
            claude = fake_trigger_claude(root / "claude", probe, invoke_project_skill=True)
            out = root / "report.json"
            with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}), \
                 contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(sys, "argv", [
                     "skill-trigger-matrix", str(DEMO_MANIFEST), "--agent", "claude", "--model", "haiku",
                     "--runs-per-query", "1", "--workers", "1", "--claude-bin", str(claude),
                     "--out", str(out)]):
                self.assertEqual(tm.main(), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            offered = [json.loads(line)["skills"] for line in probe.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(offered)
        for skills in offered:
            self.assertEqual(skills, ["update-config", "demo"])
        for row in report["results"]:
            self.assertEqual(row["evidence"], ["Skill tool invoked: demo"])
            self.assertIs(row["trigger_evidence_observed"], True)

    def test_the_trigger_adapter_applies_the_answer_parsers_rule_after_the_result(self):
        # The matrix reads Claude's stream by the rule the answer parser and the
        # trace dialect share: metadata after the one `result` keeps the cell a
        # complete observation; a second result or a late turn leaves it
        # incomplete, so it cannot count as a trigger or a miss.
        should_fire = [row for row in demo_trigger_rows() if row["should_trigger"]][:1]
        for label, trailing, allowed in CLAUDE_POST_RESULT_RECORDS:
            with self.subTest(trailing=label), tempfile.TemporaryDirectory() as td, \
                 mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                claude = fake_trigger_claude(Path(td) / "claude", Path(td) / "probe.jsonl",
                                             invoke_project_skill=True, trailing_records=[trailing])
                report = tm.run_matrix(DEMO_MANIFEST, should_fire, agents=["claude"], models=["haiku"],
                                       runs_per_query=1, timeout=30, workers=1, claude_bin=str(claude))
                row = report["results"][0]
                self.assertIs(row["observation_complete"], allowed, row.get("provider_error"))
                if allowed:
                    self.assertEqual((row["triggered"], row["evidence"]), (True, ["Skill tool invoked: demo"]))
                    self.assertEqual(report["summary"]["measurement_status"], "complete")
                else:
                    self.assertEqual(row["provider_error"],
                                     "Claude JSON stream must contain exactly one terminal result event, "
                                     "with no session content after it")
                    self.assertEqual(report["summary"]["measurement_status"], "incomplete")


CODEX_FIXTURES = ROOT / "tests" / "fixtures" / "codex"
CODEX_THREAD_ID = "01a09c3f-0000-7000-8000-000000000001"


class CodexRolloutDetectionTests(unittest.TestCase):
    """Skill loads read from the Codex session rollout, with the stream-only
    detector as fallback.

    The fixtures under tests/fixtures/codex are a redacted copy of one real run
    of `codex exec --json --skip-git-repo-check -m gpt-5.6-sol 'Use $unslop on
    ...'` (Codex CLI 0.150.1, 2026-09-13). Its event stream carried only
    thread.started / turn.started / agent_message / turn.completed and no tool
    event, while the rollout under $CODEX_HOME/sessions showed the CLI injecting
    the skill body as a user-role response_item opening with `<skill>`. The
    thread id, paths, cwd, base instructions, encrypted reasoning, and the skill
    body are replaced or truncated; record types, roles, and field names are
    verbatim. `/CODEX_HOME` in the rollout fixture stands for the isolated home."""

    def _events(self) -> str:
        return (CODEX_FIXTURES / "exec-skill-events.jsonl").read_text(encoding="utf-8")

    def _rollout(self, home: Path) -> str:
        text = (CODEX_FIXTURES / "rollout-skill-injection.jsonl").read_text(encoding="utf-8")
        return text.replace("/CODEX_HOME", str(home))

    def _mounted(self, root: Path):
        tree = root / "tree" / "unslop"
        tree.mkdir(parents=True)
        (tree / "SKILL.md").write_text(
            "---\nname: unslop\ndescription: Cut AI tells from any writing.\n---\n\n# Unslop\n", encoding="utf-8")
        workspace = root / "workspace"
        workspace.mkdir()
        adapter = tm.CodexAdapter(codex_cmd="codex exec --json")
        return adapter, workspace, adapter.mount(root / "tree", workspace)

    def _observe(self, td: str, *, rollout: bool | str = True, stream_extra=None,
                 events: str | None = None):
        root = Path(td)
        adapter, workspace, copied = self._mounted(root)
        extra = stream_extra(copied) if callable(stream_extra) else (stream_extra or "")
        seen: dict = {}

        def fake_run(plan):
            home = Path(dict(plan.environment)["CODEX_HOME"])
            seen["home"] = home
            if rollout:
                day = home / "sessions" / "2026" / "09" / "13"
                day.mkdir(parents=True)
                text = self._rollout(home) if rollout is True else (
                    rollout(copied) if callable(rollout) else rollout)
                (day / f"rollout-2026-09-13T13-30-23-{CODEX_THREAD_ID}.jsonl").write_text(text, encoding="utf-8")
            lines = (events if events is not None else self._events()).splitlines()
            if extra:
                lines.insert(len(lines) - 1, extra)
            return InvocationOutcome.from_process(
                stdout="\n".join(lines) + "\n", stderr="", returncode=0, elapsed_ms=1)

        with mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"CODEX_HOME": str(root / "ambient-codex")}):
            invocation = adapter.invoke("Use $unslop on this sentence", None, workspace, 5)
            detection = adapter.detect(invocation, ["unslop"], copied)
            self.assertTrue(adapter._codex_home(workspace).exists())
            adapter.release(workspace)
        return invocation, detection, copied, seen

    def test_skill_injection_in_rollout_counts_as_load(self):
        with tempfile.TemporaryDirectory() as td:
            invocation, detection, copied, seen = self._observe(td, rollout=True)
            self.assertFalse(seen["home"].exists())
        self.assertTrue(detection.triggered)
        self.assertEqual({item.kind for item in detection.evidence}, {TriggerEvidenceKind.CODEX_ROLLOUT})
        self.assertIn("unslop", detection.evidence[0].text)
        self.assertEqual(invocation.metadata["codex_rollout_status"], "found")
        self.assertEqual(invocation.metadata["codex_rollout_home"], "isolated")
        self.assertTrue(invocation.metadata["codex_rollout_file"].endswith(f"-{CODEX_THREAD_ID}.jsonl"))
        # The stream alone shows no tool event: the rollout is what decided.
        stream_only = tm.CodexAdapter().detect(completed_invocation(self._events()), ["unslop"], copied)
        self.assertFalse(stream_only.triggered)

    def test_missing_rollout_falls_back_to_path_evidence(self):
        def read_of_mounted(copied):
            return json.dumps({"type": "item.completed", "item": {
                "id": "item_1", "type": "command_execution", "status": "completed",
                "exit_code": 0, "command": f"cat {copied[0]}"}})

        with tempfile.TemporaryDirectory() as td:
            invocation, detection, _, _ = self._observe(td, rollout=False, stream_extra=read_of_mounted)
        self.assertTrue(detection.triggered)
        self.assertEqual({item.kind for item in detection.evidence}, {TriggerEvidenceKind.MOUNTED_PATH})
        self.assertEqual(invocation.metadata["codex_rollout_status"], "not_found")
        self.assertNotIn("codex_rollout_home", invocation.metadata)

    def test_failed_rollout_skill_read_does_not_trigger(self):
        def failed_read(copied):
            def item(payload: dict) -> str:
                return json.dumps({"timestamp": "t", "ordinal": 1,
                                   "type": "response_item", "payload": payload})

            return "\n".join([
                item({"type": "function_call", "name": "shell", "call_id": "c1",
                      "arguments": json.dumps({"command": ["cat", str(copied[0])]})}),
                item({"type": "function_call_output", "call_id": "c1",
                      "output": "Process exited with code 1. Permission denied."}),
            ])

        with tempfile.TemporaryDirectory() as td:
            invocation, detection, _, _ = self._observe(td, rollout=failed_read)
        self.assertTrue(invocation.observation_complete, invocation.provider_error)
        self.assertEqual(invocation.metadata["codex_rollout_status"], "found")
        self.assertFalse(detection.triggered)
        self.assertEqual(detection.evidence, ())

    def test_stream_line_with_a_duplicate_id_does_not_lose_the_row(self):
        # Observed live 2026-09-13: 10 of 30 Codex trigger rows died with
        # `ValueError: duplicate object key: 'id'` because `codex exec --json`
        # repeats `id` inside some items. The stream is the CLI's, not ours:
        # the line is kept (last value wins) and the row says which lines were.
        events = (CODEX_FIXTURES / "exec-duplicate-id-events.jsonl").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate object key: 'id'"):
            list(sb.iter_json_objects(events))
        with tempfile.TemporaryDirectory() as td:
            invocation, detection, copied, _ = self._observe(td, rollout=True, events=events)
        self.assertTrue(invocation.observation_complete, invocation.provider_error)
        self.assertIsNone(invocation.provider_error)
        self.assertEqual(sb.codex_thread_id(events), CODEX_THREAD_ID)
        self.assertEqual(invocation.metadata["codex_rollout_status"], "found")
        self.assertTrue(detection.triggered)
        self.assertEqual({item.kind for item in detection.evidence}, {TriggerEvidenceKind.CODEX_ROLLOUT})
        self.assertFalse(sb.detect_trigger_detection(events, copied).triggered)

    def test_row_records_which_stream_lines_the_lenient_rule_kept(self):
        events = (CODEX_FIXTURES / "exec-duplicate-id-events.jsonl").read_text(encoding="utf-8")

        def fake_run(plan):
            return InvocationOutcome.from_process(stdout=events, stderr="", returncode=0, elapsed_ms=1)

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"CODEX_HOME": str(Path(td) / "ambient-codex")}):
            tree = Path(td) / "tree"
            skill = tree / "unslop"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: unslop\ndescription: Cut AI tells.\n---\n", encoding="utf-8")
            row = tm.observe_cell_query(
                tm.CodexAdapter(codex_cmd="codex exec --json"), tree, "Use $unslop", True, None, 12,
                metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
            ).as_row()
        self.assertTrue(row["observation_complete"])
        self.assertEqual(list(row["invocation_metadata"]["stream_duplicate_keys"]), ["line 3: id"])
        self.assertEqual(list(row["stream_duplicate_keys"]), ["line 3: id"])
        self.assertEqual(row["usage_normalized"]["input_tokens"], 22215)
        clean = (CODEX_FIXTURES / "exec-skill-events.jsonl").read_text(encoding="utf-8")
        self.assertEqual(sb.stream_duplicate_keys(clean), [])

    def test_rollout_detector_keeps_the_strict_artifact_rule(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            rollout = self._rollout(home)
            lines = rollout.splitlines()
            lines.insert(2, json.dumps({"timestamp": "t", "ordinal": 9, "type": "event_msg",
                                        "payload": {"type": "x"}})[:-1] + ',"type":"event_msg"}')
            copied = [home / "skills" / "unslop" / "SKILL.md"]
            with self.assertRaisesRegex(ValueError, "duplicate object key: 'type'"):
                sb.codex_rollout_skill_loads("\n".join(lines) + "\n", ["unslop"], copied)
            self.assertTrue(sb.codex_rollout_skill_loads(rollout, ["unslop"], copied))

    def test_rollout_without_a_mounted_skill_load_stays_negative(self):
        other = (CODEX_FIXTURES / "rollout-skill-injection.jsonl").read_text(encoding="utf-8")
        other = other.replace("<name>unslop</name>", "<name>other-skill</name>").replace(
            "/CODEX_HOME/skills/unslop/SKILL.md", "/elsewhere/skills/other-skill/SKILL.md")
        with tempfile.TemporaryDirectory() as td:
            invocation, detection, _, _ = self._observe(td, rollout=other)
        self.assertFalse(detection.triggered)
        self.assertEqual(invocation.metadata["codex_rollout_status"], "found")

    def test_injection_naming_a_mounted_skill_from_elsewhere_is_not_a_load(self):
        mounted = Path("/tmp/trigger-x-codex-home/skills/unslop/SKILL.md")
        injected = self._rollout(Path("/Users/someone/.agents"))
        self.assertEqual(sb.codex_rollout_skill_loads(injected, ["unslop"], [mounted]), [])
        self.assertEqual(sb.codex_rollout_skill_loads(self._rollout(mounted.parents[2]), ["unslop"], [mounted]),
                         [f"rollout skill injection: unslop ({mounted})"])

    def test_rollout_tool_calls_and_listings_do_not_count_as_loads(self):
        mounted = Path("/tmp/trigger-x-codex-home/skills/unslop/SKILL.md")

        def item(payload: dict) -> str:
            return json.dumps({"timestamp": "t", "ordinal": 1, "type": "response_item", "payload": payload})

        call = item({"type": "function_call", "name": "shell", "call_id": "c1",
                     "arguments": json.dumps({"command": ["cat", str(mounted)]})})
        failed_read = item({"type": "function_call_output", "call_id": "c1",
                            "output": "Permission denied."})
        listing_call = item({"type": "function_call", "name": "shell", "call_id": "c2",
                             "arguments": json.dumps({"command": ["ls", str(mounted.parent)]})})
        listing_output = item({"type": "function_call_output", "call_id": "c2",
                               "output": "SKILL.md"})
        listing = item({"type": "message", "role": "developer", "content": [
            {"type": "input_text", "text": f"- unslop: Cut AI tells. (file: {mounted})"}]})
        prose = item({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": f"I would load {mounted}"}]})
        for record in (call, failed_read, call + "\n" + failed_read, listing_call,
                       listing_output, listing_call + "\n" + listing_output, listing, prose):
            self.assertEqual(sb.codex_rollout_skill_loads(record, ["unslop"], [mounted]), [])

    def test_thread_id_and_ambient_home_lookup(self):
        self.assertEqual(sb.codex_thread_id(self._events()), CODEX_THREAD_ID)
        self.assertIsNone(sb.codex_thread_id('{"type":"turn.completed"}\n'))
        self.assertEqual(sb.locate_codex_rollout('{"type":"turn.completed"}\n', None).status, "no_thread_id")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ambient = root / "ambient"
            day = ambient / "sessions" / "2026" / "09" / "13"
            day.mkdir(parents=True)
            (day / f"rollout-2026-09-13T13-30-23-{CODEX_THREAD_ID}.jsonl").write_text(
                self._rollout(ambient), encoding="utf-8")
            (root / "isolated").mkdir()
            with mock.patch.dict(os.environ, {"CODEX_HOME": str(ambient)}):
                found = sb.locate_codex_rollout(self._events(), root / "isolated")
                self.assertEqual((found.status, found.home), ("found", "ambient"))
                self.assertIn(sb.CODEX_ROLLOUT_SKILL_TAG, found.text)
                (root / "other").mkdir()
                with mock.patch.dict(os.environ, {"CODEX_HOME": str(root / "other")}):
                    self.assertEqual(sb.locate_codex_rollout(self._events(), root / "isolated").status, "not_found")


class TriggerContextIsolationTests(unittest.TestCase):
    """A trigger run measures whether the agent loads the mounted skill, so the
    agent must see that skill and nothing from the operator's host: no host
    skills, bundled skills, host agents, instruction files, or MCP servers."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        self.tree = self.root / "tree"
        (self.tree / "demo").mkdir(parents=True)
        (self.tree / "demo" / "SKILL.md").write_text(skill_markdown(), encoding="utf-8")
        self.home = self.root / "home"
        self.host_skill = self.home / ".agents" / "skills" / "host-skill" / "SKILL.md"
        self.host_skill.parent.mkdir(parents=True)
        self.host_skill.write_text(skill_markdown("host-skill"), encoding="utf-8")
        self.probe = self.root / "argv.json"
        self.bindir = self.root / "bin"
        self.bindir.mkdir()

    def host_env(self, **extra):
        return mock.patch.dict(os.environ, {"HOME": str(self.home), **extra})

    def run_row(self, adapter):
        return tm.observe_cell_query(
            adapter, self.tree, "review this diff", True, None, 30,
            metadata={"skill_tree_hash": sb.skill_tree_hash(self.tree)}).as_row()

    def seen_argv(self):
        return json.loads(self.probe.read_text(encoding="utf-8"))

    @staticmethod
    def recorded_isolation(row):
        return json.loads(json.dumps(row)).get("context_isolation")

    def test_claude_trigger_run_keeps_project_skills_and_drops_host_context(self):
        claude = stub_agent_cli(self.bindir / "claude", probe_path=self.probe,
                                stdout_records=[{"type": "result", "subtype": "success"}])
        with self.host_env(CLAUDE_CONFIG_DIR=str(self.root / "user-claude")):
            row = self.run_row(tm.ClaudeAdapter(claude_bin=str(claude)))
        expected = ["--setting-sources", "project", "--strict-mcp-config",
                    "--settings", '{"disableBundledSkills":true,"autoMemoryEnabled":false}']
        self.assertEqual(self.recorded_isolation(row), expected)
        argv = self.seen_argv()
        start = argv.index("--setting-sources")
        self.assertEqual(argv[start:start + 5], expected)
        self.assertNotIn("--safe-mode", argv)
        self.assertNotIn("--disable-slash-commands", argv)

    def test_codex_trigger_run_disables_host_and_bundled_skills_but_not_the_mount(self):
        codex = stub_agent_cli(self.bindir / "codex", probe_path=self.probe,
                               stdout_records=[{"type": "turn.completed"}])
        with self.host_env(CODEX_HOME=str(self.root / "user-codex")):
            row = self.run_row(tm.CodexAdapter(codex_cmd=f"{codex} exec --json"))
        self.assertEqual(self.recorded_isolation(row), [
            "-c", "skills.bundled.enabled=false",
            "-c", "skills.config=<1 host skill(s) disabled>",
            "--disable", "apps"])
        argv = self.seen_argv()
        self.assertIn(f'skills.config=[{{path="{self.host_skill}",enabled=false}}]', argv)
        self.assertIn("skills.bundled.enabled=false", argv)
        self.assertNotIn("skills.include_instructions=false", argv)

    def test_pi_trigger_run_loads_only_the_mounted_skills_dir(self):
        stub_agent_cli(self.bindir / "pi", probe_path=self.probe, stdout_records=[
            {"type": "agent_end", "messages": [{"stopReason": "stop"}]}])
        path = f"{self.bindir}{os.pathsep}{os.environ['PATH']}"
        with self.host_env(PATH=path, PI_CODING_AGENT_DIR=str(self.root / "user-pi")):
            row = self.run_row(tm.PiAdapter())
        self.assertEqual(self.recorded_isolation(row), [
            "--no-context-files", "--no-prompt-templates", "--no-extensions",
            "--no-skills", "--skill", "<mounted skills dir>"])
        argv = self.seen_argv()
        self.assertIn("--no-skills", argv)
        self.assertEqual(Path(argv[argv.index("--skill") + 1]).name, "skills")
        self.assertIn("pi-home", Path(argv[argv.index("--skill") + 1]).parent.name)

    def test_pi_trigger_eval_row_records_the_same_isolation(self):
        stub_agent_cli(self.bindir / "pi", probe_path=self.probe, stdout_records=[
            {"type": "agent_end", "messages": [{"stopReason": "stop"}]}])
        path = f"{self.bindir}{os.pathsep}{os.environ['PATH']}"
        with self.host_env(PATH=path, PI_CODING_AGENT_DIR=str(self.root / "user-pi")):
            rows_file = self.root / "pi-queries.json"
            rows_file.write_text(json.dumps([{"query": "ordinary chat", "should_trigger": False}]))
            output = self.root / "pi-report.json"
            with mock.patch.object(sys, "argv", ["skill-pi-trigger-eval", str(DEMO_MANIFEST),
                    "--eval-set", str(rows_file), "--runs-per-query", "1", "--workers", "1", "--out", str(output)]), \
                    mock.patch("builtins.print"):
                self.assertEqual(tr.main(), 0)
            row = json.loads(output.read_text())["results"][0]
        self.assertEqual(self.recorded_isolation(row), [
            "--no-context-files", "--no-prompt-templates", "--no-extensions",
            "--no-skills", "--skill", "<mounted skills dir>"])
        argv = self.seen_argv()
        self.assertEqual(Path(argv[argv.index("--skill") + 1]).name, "skills")


@unittest.skipIf(sys.version_info < (3, 11), "tomllib is stdlib from Python 3.11")
class CodexSkillConfigTomlEncodingTests(unittest.TestCase):
    """The `-c skills.config=[...]` argv value must be a TOML value Codex can
    parse. `json.dumps` escapes characters above U+FFFF as UTF-16 surrogate
    pairs, which is a JSON string, not a TOML one: Codex reads the whole
    value as an opaque string and dies with `Error loading config.toml:
    invalid type: string ...`."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.home = Path(td.name)
        (self.home / ".agents").mkdir()

    def _skill(self, dirname: str, name: str) -> Path:
        d = self.home / ".agents" / "skills" / dirname
        d.mkdir(parents=True)
        skill_md = d / "SKILL.md"
        skill_md.write_text(skill_markdown(name), encoding="utf-8")
        return skill_md

    def _parsed_skills_config(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.home)}):
            args, _recorded = sb.codex_trigger_context_isolation_args()
        cfg = next(a for a in args if a.startswith("skills.config="))
        import tomllib
        return cfg, tomllib.loads(cfg + "\n")["skills"]["config"]

    def test_astral_and_bmp_unicode_directory_names_parse_as_toml(self):
        emoji = self._skill("rocket-\U0001F680", "astral-emoji")
        accent = self._skill("café", "bmp-accent")
        cfg, entries = self._parsed_skills_config()
        self.assertNotIn("\\u", cfg.lower().replace("http", ""))
        paths = {e["path"] for e in entries}
        self.assertEqual(paths, {str(emoji), str(accent)})
        self.assertTrue(all(e["enabled"] is False for e in entries))

    def test_quote_and_backslash_directory_names_parse_as_toml(self):
        quoted = self._skill('my skill "q"', "quoted-space")
        backslash = self._skill("back\\slash", "backslash-name")
        _cfg, entries = self._parsed_skills_config()
        paths = {e["path"] for e in entries}
        self.assertEqual(paths, {str(quoted), str(backslash)})

    def test_newline_in_directory_name_parses_as_toml_if_filesystem_allows(self):
        directory = self.home / ".agents" / "skills" / "line\nbreak"
        try:
            directory.mkdir(parents=True)
        except OSError:
            self.skipTest("filesystem rejects newline in a directory name")
        newline = directory / "SKILL.md"
        newline.write_text(skill_markdown("newline-name"), encoding="utf-8")
        _cfg, entries = self._parsed_skills_config()
        self.assertEqual({e["path"] for e in entries}, {str(newline)})

    def test_argv_carries_the_exact_paths_for_a_plain_and_a_control_char_free_mix(self):
        plain = self._skill("host-a", "host-a")
        cfg, entries = self._parsed_skills_config()
        self.assertIn(str(plain), cfg)
        self.assertEqual(len(entries), 1)


class CodexConfigFailureStderrRedactionTests(unittest.TestCase):
    """A Codex config-load failure must not let the operator's host skill
    paths reach a saved row through stderr recorded verbatim."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.home = Path(td.name) / "home"
        self.host_skill = self.home / ".agents" / "skills" / "very-secret-project-name" / "SKILL.md"
        self.host_skill.parent.mkdir(parents=True)
        self.host_skill.write_text(skill_markdown("very-secret-project-name"), encoding="utf-8")
        self.workspace_root = Path(td.name) / "ws"
        self.workspace_root.mkdir()

    def _invoke_with_fake_config_error(self):
        # Shaped like the real codex error: `serde`'s message quotes the
        # whole offending argv value, which itself embeds every host path.
        argv_value = f'[{{path="{self.host_skill}",enabled=false}}]'
        fake_stderr = (
            f'Error: invalid type: string "{argv_value}", expected a sequence\n'
            f'in `skills.config`\n'
        )

        def fake_run(plan):
            return {"stdout": "", "stderr": fake_stderr, "returncode": 1, "timed_out": False,
                    "elapsed_ms": 5, "observation_complete": False}

        with mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"HOME": str(self.home)}):
            workspace = self.workspace_root / "workspace"
            workspace.mkdir()
            return tm.CodexAdapter(codex_cmd="codex exec --json").invoke("q", None, workspace, 5)

    def test_config_load_failure_stderr_does_not_carry_the_host_path(self):
        result = self._invoke_with_fake_config_error()
        self.assertNotIn(str(self.host_skill), result.stderr)
        self.assertNotIn("very-secret-project-name", result.stderr)

    def test_config_load_failure_stderr_keeps_the_error_shape(self):
        result = self._invoke_with_fake_config_error()
        self.assertIn("invalid type: string", result.stderr)
        self.assertIn("skills.config", result.stderr)


class CodexRealHostPathStderrRedactionTests(unittest.TestCase):
    """Every spelling of a host skill path that real Codex puts in stderr must
    be gone from the saved row, while the mounted skill path stays.

    tests/fixtures/codex/host-path-stderr.json holds stderr from codex-cli
    0.156.1 run through CodexAdapter.invoke against a local 400 sink, captured
    before the adapter redacted anything. `@ROOT@` stands for the directory
    that held each fake HOME; every other byte is verbatim. `forcefail` was
    captured with the pre-TOML JSON encoder, so Codex echoed the whole
    skills.config value back with its own escaping on top. The cases cover
    the spellings Codex prints: the raw path, the realpath of a symlinked skill
    or root, a `file://` URL of the containing directory (percent-encoded,
    with tab and newline dropped), and the config-load echo."""

    FIXTURE = CODEX_FIXTURES / "host-path-stderr.json"

    def _host_tree(self, case: str, home: Path) -> None:
        skills = home / ".agents" / "skills"
        skills.mkdir(parents=True)

        def skill(d: Path, text: str) -> None:
            d.mkdir(parents=True, exist_ok=True)
            (d / "SKILL.md").write_text(text, encoding="utf-8")

        no_front = "no frontmatter\n"
        if case == "badfront":
            skill(skills / "bad-front", no_front)
            skill(skills / 'bad "q" front', "---\nname: [unclosed\n---\n")
        elif case == "symbad":
            skill(home / "elsewhere" / "symtarget", no_front)
            (skills / "symlinked").symlink_to(home / "elsewhere" / "symtarget")
        elif case == "symbadspecial":
            target = home / "else where" / 'sym "q" café-\U0001F680'
            skill(target, no_front)
            (skills / "symlinked").symlink_to(target)
        elif case in ("tabdir", "special-home-file-url"):
            skill(skills / "tab\there", skill_markdown("tabbed"))
        elif case == "badspecial":
            for name in ['q "quote"', "back\\slash", "ctl\x01", "rocket-\U0001F680", "café",
                         "sp ace", "pct%41", "new\nline", "tab\tx"]:
                skill(skills / name, no_front)
        elif case == "forcefail":
            for name in ["plain-ascii-secret", 'quote "secret"', "rocket-\U0001F680", "café-secret"]:
                skill(skills / name, skill_markdown("p"))
        elif case == "symroot":
            skills.rmdir()
            skill(home / "real skills" / "bad", no_front)
            skills.symlink_to(home / "real skills")
        else:
            raise AssertionError(case)

    def _row(self, case: str, fixture: dict, root: Path) -> tuple[dict, Path]:
        home = root / fixture["home"]
        self._host_tree(case, home)
        stderr = fixture["stderr"].replace("@ROOT@", str(root))
        seen: dict = {}

        def fake_run(plan):
            mounted = next((Path(dict(plan.environment)["CODEX_HOME"]) / "skills").rglob("SKILL.md"))
            seen["mounted"] = mounted
            read = json.dumps({"type": "item.completed", "item": {
                "id": "item_1", "type": "command_execution", "status": "completed",
                "exit_code": 0, "command": f"cat {mounted}"}})
            return {"stdout": read + "\n", "stderr": stderr, "returncode": 1, "timed_out": False,
                    "elapsed_ms": 5, "observation_complete": False}

        tree = root / "tree"
        if not tree.exists():
            (tree / "demo").mkdir(parents=True)
            (tree / "demo" / "SKILL.md").write_text(skill_markdown("demo"), encoding="utf-8")
        with mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"HOME": str(home), "CODEX_HOME": str(root / "ambient-codex")}):
            observation = tm.observe_cell_query(
                tm.CodexAdapter(codex_cmd="codex exec --json"), tree, "q", False, None, 5,
                trace_dir=root / "trace" / case, metadata={"skill_tree_hash": sb.skill_tree_hash(tree)})
        # The row keeps only the last 1,000 characters of stderr; check the
        # full redacted stderr too so an early line cannot hide a leak.
        return {**observation.as_row(), "full_stderr": observation.invocation.stderr}, seen["mounted"]

    def _saved_text(self, row: dict) -> str:
        texts = [json.dumps(row, ensure_ascii=False)]
        texts += [p.read_text(encoding="utf-8") for p in Path(row["trace_dir"]).rglob("*") if p.is_file()]
        return "\n".join(texts)

    def test_no_host_path_spelling_from_real_codex_stderr_reaches_the_saved_row(self):
        cases = json.loads(self.FIXTURE.read_text(encoding="utf-8"))["cases"]
        self.assertEqual(set(cases), {"badfront", "badspecial", "forcefail", "special-home-file-url",
                                      "symbad", "symbadspecial", "symroot", "tabdir"})
        for case, fixture in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                root = Path(td).resolve()
                row, mounted = self._row(case, fixture, root)
                saved = self._saved_text(row)
                raw_saved = json.dumps(row)
                for spelling in fixture["host_path_spellings"]:
                    spelling = spelling.replace("@ROOT@", str(root))
                    self.assertNotIn(spelling, row["full_stderr"])
                    self.assertNotIn(spelling, saved)
                    self.assertNotIn(json.dumps(spelling)[1:-1], raw_saved)
                self.assertNotIn(str(root / fixture["home"] / ".agents"), saved)
                self.assertIn("[REDACTED]", row["full_stderr"])
                self.assertEqual(row["evidence"], [f"cat {mounted}"])
                self.assertIn(str(mounted), (Path(row["trace_dir"]) / "trace.jsonl").read_text(encoding="utf-8"))

    def test_redaction_keeps_the_codex_error_text_around_the_path(self):
        cases = json.loads(self.FIXTURE.read_text(encoding="utf-8"))["cases"]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            row, _ = self._row("badfront", cases["badfront"], root)
            tab_row, _ = self._row("tabdir", cases["tabdir"], root)
        self.assertEqual(row["stderr"].splitlines()[1:], [
            ("2026-09-29T21:28:43.535905Z ERROR codex_core::session::session: "
             "failed to load skill [REDACTED]: missing field `description`"),
            ("2026-09-29T21:28:43.536274Z ERROR codex_core::session::session: "
             "failed to load skill [REDACTED]: missing YAML frontmatter delimited by ---"),
        ])
        self.assertEqual(tab_row["stderr"].splitlines()[1],
                         "2026-09-29T21:28:46.401772Z ERROR codex_skills_extension::loader::host: "
                         "failed to scan skill path [REDACTED]: No such file or directory (os error 2)")


class CodexStderrCapCannotSplitAHostPathTests(unittest.TestCase):
    """`invoke_argv_with_timeout` caps captured stderr before `CodexAdapter`
    ever sees it, so a stderr longer than the cap used to be cut mid-path: the
    redactor only recognizes a host path whole, so the surviving fragment (the
    operator's home directory name, part of a project id) reached the saved
    row unredacted. This reproduces that shape directly against a real
    subprocess and the real cap, not a mocked stderr already under it."""

    CAP = 4000

    def _fake_codex_script(self, root: Path) -> Path:
        script = root / "fake_codex.py"
        script.write_text(
            "import os, pathlib, sys\n"
            "sys.stdin.read()\n"
            "sys.stderr.write(pathlib.Path(os.environ['FAKE_CODEX_STDERR_FILE']).read_text(encoding='utf-8'))\n"
            "sys.exit(1)\n",
            encoding="utf-8",
        )
        return script

    def _stderr_straddling_the_cap(self, skill_md: Path, home: Path) -> tuple[str, int, int]:
        """A repeated 'failed to load skill' line, padded so one occurrence's
        path crosses index `CAP` while the cut still lands inside the `home`
        segment, before `.agents/skills` even starts. That is what the real
        Codex repro looked like (the saved row's stderr ended mid-way through
        the session directory name, short of `.agents/skills`): the
        redactor's own root fallback needs `home/.agents/skills` present
        *whole* to catch a span with no trailing delimiter, so a cut that
        never reaches the root literal defeats it too."""
        prefix = ("2026-09-29T21:40:49.748679Z ERROR codex_core::session::session: "
                  "failed to load skill ")
        suffix = ": missing YAML frontmatter delimited by ---\n"
        path_text = str(skill_md)
        home_text = str(home)
        line = f"{prefix}{path_text}{suffix}"
        for pad_len in range(len(line)):
            for k in range(40):
                path_start = pad_len + len(prefix) + k * len(line)
                offset = self.CAP - path_start
                if 20 <= offset < len(home_text):
                    body = (" " * pad_len) + line * (k + 2)
                    return body, path_start, offset
        raise AssertionError("could not align a mid-home cut for this fixture")

    def test_no_host_path_fragment_survives_a_stderr_cap_that_lands_mid_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            home = root / "home"
            skill_dir = home / ".agents" / "skills" / "very-secret-codename-zulu"
            skill_dir.mkdir(parents=True)
            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("no frontmatter\n", encoding="utf-8")

            stderr_text, path_start, offset = self._stderr_straddling_the_cap(skill_md, home)
            self.assertGreater(len(stderr_text), self.CAP)
            self.assertLess(path_start, self.CAP)
            # The cut falls inside `home`, short of `.agents/skills`: this is
            # the exact prefix a hard character cut used to leave behind.
            leaked_prefix = str(skill_md)[:offset]
            self.assertNotIn(".agents", leaked_prefix)

            stderr_file = root / "fake-stderr.txt"
            stderr_file.write_text(stderr_text, encoding="utf-8")
            fake_codex = self._fake_codex_script(root)
            workspace = root / "workspace"
            workspace.mkdir()

            with mock.patch.dict(os.environ, {"HOME": str(home), "FAKE_CODEX_STDERR_FILE": str(stderr_file)}):
                os.environ.pop("CODEX_HOME", None)
                result = tm.CodexAdapter(codex_cmd=f"{sys.executable} {fake_codex}").invoke(
                    "q", None, workspace, 10)

        self.assertNotIn(leaked_prefix, result.stderr)
        self.assertNotIn(str(home), result.stderr)
        self.assertNotIn("very-secret-codename-zulu", result.stderr)
        self.assertNotIn(str(skill_dir), result.stderr)


class CodexConfigEchoLongerThanTheCapTests(unittest.TestCase):
    """A rejected skills.config echoes every host path on one stderr line, so
    with enough host skills that single line runs past the stderr cap. No cut
    of that line may leave a host path fragment in the saved row.

    tests/fixtures/codex/forcefail-bulk-stderr.json is real codex-cli 0.156.1
    stderr for 60 host skills plus one astral-named one under the pre-TOML
    encoder. The fake HOME's name is padded so the cap lands inside it."""

    FIXTURE = CODEX_FIXTURES / "forcefail-bulk-stderr.json"
    CAP = 4000

    def _home_cut_by_the_cap(self, root: Path, template: str) -> tuple[Path, str]:
        for pad in range(400):
            home = root / ("operator-" + "x" * pad)
            text = template.replace("@HOME@", str(home))
            for start in (i for i in range(len(text)) if text.startswith(str(home), i)):
                marker_end = start + len(str(root)) + len("/operator")
                if marker_end < self.CAP - 20 and self.CAP < start + len(str(home)):
                    return home, text
        raise AssertionError("no padding puts the cap inside the home path")

    def test_no_host_path_fragment_survives_a_single_line_config_echo_past_the_cap(self):
        fixture = json.loads(self.FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            home, stderr_text = self._home_cut_by_the_cap(root, fixture["stderr"])
            self.assertEqual(stderr_text.count("\n"), 3)
            self.assertGreater(stderr_text.index("\n"), self.CAP)
            for name in fixture["host_skill_dirs"]:
                (home / ".agents" / "skills" / name).mkdir(parents=True)
                (home / ".agents" / "skills" / name / "SKILL.md").write_text(skill_markdown("h"), encoding="utf-8")
            stderr_file = root / "fake-stderr.txt"
            stderr_file.write_text(stderr_text, encoding="utf-8")
            fake_codex = root / "fake_codex.py"
            fake_codex.write_text(
                "import os, pathlib, sys\n"
                "sys.stdin.read()\n"
                "sys.stderr.write(pathlib.Path(os.environ['FAKE_CODEX_STDERR_FILE']).read_text(encoding='utf-8'))\n"
                "sys.exit(1)\n",
                encoding="utf-8")
            tree = root / "tree"
            (tree / "demo").mkdir(parents=True)
            (tree / "demo" / "SKILL.md").write_text(skill_markdown("demo"), encoding="utf-8")
            with mock.patch.dict(os.environ, {"HOME": str(home), "FAKE_CODEX_STDERR_FILE": str(stderr_file)}):
                os.environ.pop("CODEX_HOME", None)
                observation = tm.observe_cell_query(
                    tm.CodexAdapter(codex_cmd=f"{sys.executable} {fake_codex}"), tree, "q", False, None, 10,
                    trace_dir=root / "trace", metadata={"skill_tree_hash": sb.skill_tree_hash(tree)})
            row = observation.as_row()
            invocation = observation.invocation
            saved = "\n".join([json.dumps(row, ensure_ascii=False), json.dumps(row)]
                              + [p.read_text(encoding="utf-8") for p in (root / "trace").rglob("*") if p.is_file()])

        self.assertEqual(invocation.state, InvocationState.PROCESS_FAILED)
        for label, text in {"stderr": invocation.stderr, "stdout": invocation.stdout,
                            "provider_error": invocation.provider_error or "", "saved row": saved}.items():
            with self.subTest(label):
                self.assertEqual([m for m in ("operator-", "hmk", str(home)) if m in text], [])
        self.assertTrue(invocation.stderr.startswith('Error loading config.toml: invalid type: string "[{path=\\"[REDACTED]\\",enabled=false},'))
        self.assertLessEqual(len(invocation.stderr), self.CAP)


class CodexAdapterTests(unittest.TestCase):
    """Codex trigger support without a live codex binary."""

    def test_codex_completed_read_of_the_mounted_skill_is_a_trigger(self):
        # `codex exec --json` reports a finished shell command as an
        # item.completed command_execution; reading the SKILL.md mounted under
        # the isolated $CODEX_HOME/skills and then ending the turn is load evidence.
        def fake_run(plan):
            argv = list(plan.argv)
            skills_dir = Path(argv[argv.index("--add-dir") + 1])
            skill_md = next(skills_dir.glob("*/SKILL.md"))
            command = f"bash -lc 'cat {skill_md}'"
            stream = [
                {"type": "thread.started", "thread_id": "t"},
                {"type": "turn.started"},
                {"type": "item.completed", "item": {
                    "id": "item_0", "type": "command_execution", "command": command,
                    "aggregated_output": skill_md.read_text(encoding="utf-8"),
                    "exit_code": 0, "status": "completed"}},
                {"type": "item.completed", "item": {
                    "id": "item_1", "type": "agent_message", "text": "Reviewed."}},
                {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
            ]
            return InvocationOutcome.from_process(
                stdout="".join(json.dumps(record) + "\n" for record in stream),
                stderr="", returncode=0, elapsed_ms=1)

        should_fire = [row for row in demo_trigger_rows() if row["should_trigger"]]
        with mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)):
            report = tm.run_matrix(DEMO_MANIFEST, should_fire, agents=["codex"], models=[None],
                                   runs_per_query=1, timeout=30, workers=1)
        (row,) = report["results"]
        self.assertTrue(row["observation_complete"])
        self.assertTrue(row["triggered"])
        self.assertTrue(row["pass"])
        self.assertEqual(len(row["evidence"]), 1)
        self.assertRegex(row["evidence"][0], r"^bash -lc 'cat .*-codex-home/skills/.*/SKILL\.md'$")

    def test_codex_statusless_command_is_not_load_evidence(self):
        # Without a completion status the command may never have run.
        mounted = Path("/tmp/trigger-x/.codex/skills/demo-reviewer/SKILL.md")
        stream = json.dumps({"type": "command", "command": ["bash", "-lc", f"cat {mounted}"]})
        detection = tm.CodexAdapter().detect(completed_invocation(stream), ["demo-reviewer"], [mounted])
        self.assertFalse(detection.triggered)
        self.assertFalse(detection.evidence)

    def test_codex_skill_name_in_prose_is_not_load_evidence(self):
        mounted = Path("/tmp/trigger-x/.codex/skills/demo-reviewer/SKILL.md")
        prose = json.dumps({"type": "message", "content": "I would use demo-reviewer."})
        self.assertFalse(tm.CodexAdapter().detect(completed_invocation(prose), ["demo-reviewer"], [mounted]).triggered)

    def test_malformed_or_unterminated_streams_are_not_valid_negative_observations(self):
        # Parseable JSON is not enough: Codex must end its turn and Vibe must
        # end with an assistant answer before absence of evidence counts.
        for adapter_cls in (tm.CodexAdapter, tm.VibeAdapter):
            for stdout, reason in (("not-json\n", "is malformed"), ("{}\n", "JSON stream must")):
                def fake_run(*args, _stdout=stdout, **kwargs):
                    return InvocationOutcome.from_process(
                        stdout=_stdout, stderr="", returncode=0, elapsed_ms=1)

                with self.subTest(adapter=adapter_cls.name, stdout=stdout), \
                     tempfile.TemporaryDirectory() as td, \
                     mock.patch.object(adapter_cls, "_run_argv", staticmethod(fake_run)):
                    workspace = Path(td) / "workspace"
                    workspace.mkdir()
                    result = adapter_cls().invoke("q", None, workspace, 1)
                    self.assertIs(result.state, InvocationState.PROVIDER_FAILED)
                    self.assertIn(reason, result.provider_error or "")

    def test_codex_invoke_appends_raw_query_model_and_external_skill_dir(self):
        seen = {}

        def fake_run(plan):
            argv, cwd = list(plan.argv), plan.cwd
            env, timeout = dict(plan.environment or {}), int(plan.timeout_s)
            seen.update({"argv": argv, "cwd": cwd, "env": env, "timeout": timeout})
            return completed_invocation('{"type":"turn.completed"}\n')

        with mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)):
            with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"CODEX_HOME": str(Path(td) / "source-codex")}):
                workspace = Path(td) / "workspace"
                workspace.mkdir()
                result = tm.CodexAdapter(codex_cmd="codex exec --json").invoke("raw trigger query", "o4-mini", workspace, 12)
        self.assertEqual(seen["argv"][:3], ["codex", "exec", "--json"])
        self.assertIn("--add-dir", seen["argv"])
        skill_dir = Path(seen["argv"][seen["argv"].index("--add-dir") + 1])
        self.assertEqual(skill_dir, Path(seen["env"]["CODEX_HOME"]) / "skills")
        self.assertFalse(Path(seen["env"]["CODEX_HOME"]).is_relative_to(seen["cwd"]))
        self.assertEqual(seen["argv"][-3:], ["--model", "o4-mini", "raw trigger query"])
        self.assertEqual(seen["timeout"], 12)
        self.assertTrue(result.metadata["codex_home_outside_workdir"])

    def test_codex_invoke_seeds_auth_without_copying_user_skills(self):
        seen = {}

        def fake_run(plan):
            cwd = plan.cwd
            env = dict(plan.environment or {})
            codex_home = Path(env["CODEX_HOME"])
            seen["auth"] = (codex_home / "auth.json").read_text(encoding="utf-8")
            seen["config"] = (codex_home / "config.toml").read_text(encoding="utf-8")
            seen["mounted_skills_survive"] = (codex_home / "skills" / "demo").is_dir()
            seen["user_skills_not_copied"] = not (codex_home / "skills" / "personal").exists()
            seen["workspace_auth_present"] = (Path(cwd) / ".codex" / "auth.json").exists()
            seen["workspace_config_present"] = (Path(cwd) / ".codex" / "config.toml").exists()
            return completed_invocation('{"type":"turn.completed"}\n')

        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm.CodexAdapter, "_run_argv", staticmethod(fake_run)):
            root = Path(td)
            source = root / "user-codex"
            (source / "skills" / "personal").mkdir(parents=True)
            (source / "auth.json").write_text('{"token":"t"}', encoding="utf-8")
            (source / "config.toml").write_text("model = 'm'\n", encoding="utf-8")
            workspace = root / "run"
            workspace.mkdir()
            tree = root / "tree"
            (tree / "demo").mkdir(parents=True)
            (tree / "demo" / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
            adapter = tm.CodexAdapter(codex_cmd="codex exec --json")
            adapter.mount(tree, workspace)
            with mock.patch.dict(os.environ, {"CODEX_HOME": str(source)}):
                result = adapter.invoke("q", None, workspace, 12)
        self.assertEqual(seen["auth"], '{"token":"t"}')
        self.assertEqual(seen["config"], "model = 'm'\n")
        self.assertTrue(seen["mounted_skills_survive"])
        self.assertTrue(seen["user_skills_not_copied"])
        self.assertTrue(result.metadata["codex_home_outside_workdir"])
        self.assertFalse(seen["workspace_auth_present"])
        self.assertFalse(seen["workspace_config_present"])

    def test_cell_observation_redacts_ambient_env_secrets(self):
        class LeakyAdapter(tm.AgentAdapter):
            name = "stub"

            def mount(self, tree_dir, workspace):
                return self._mount_tree(tree_dir, workspace / "skills")

            def invoke(self, query, model, workspace, timeout):
                secret = os.environ["MISTRAL_API_KEY"]
                return {"stdout": f"leaked {secret}\n", "stderr": f"err {secret}", "returncode": 0,
                        "timed_out": False, "elapsed_ms": 1, "observation_complete": True,
                        "debug": {"auth": secret}}

        with tempfile.TemporaryDirectory() as td, mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "ambient-secret-token"}):
            tree = Path(td) / "tree"
            skill = tree / "demo"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
            row = tm.observe_cell_query(
                LeakyAdapter(), tree, "q", False, None, 12,
                trace_dir=Path(td) / "trace",
                metadata={"skill_tree_hash": sb.skill_tree_hash(tree),
                          "external": {"token": "ambient-secret-token"}},
            ).as_row()
            trace_text = (Path(row["trace_dir"]) / "trace.jsonl").read_text(encoding="utf-8")
            trace_metadata = json.loads((Path(row["trace_dir"]) / "metadata.json").read_text(encoding="utf-8"))
        self.assertNotIn("ambient-secret-token", row["stderr"])
        self.assertEqual(row["stderr"], "err [REDACTED]")
        self.assertEqual(row["debug"], {"auth": "[REDACTED]"})
        self.assertEqual(row["external"], {"token": "[REDACTED]"})
        self.assertEqual(trace_metadata["debug"], {"auth": "[REDACTED]"})
        self.assertEqual(trace_metadata["external"], {"token": "[REDACTED]"})
        self.assertNotIn("ambient-secret-token", trace_text)

    def test_cell_observation_requires_invoke_contract(self):
        class BrokenAdapter(tm.AgentAdapter):
            name = "broken"

            def mount(self, tree_dir, workspace):
                return self._mount_tree(tree_dir, workspace / "skills")

            def invoke(self, query, model, workspace, timeout):
                return {"stdout": "{}\n", "stderr": "", "returncode": 0, "timed_out": False, "elapsed_ms": 1}

        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "tree"
            skill = tree / "demo"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
            with self.assertRaises(KeyError) as ctx:
                tm.observe_cell_query(
                    BrokenAdapter(), tree, "q", True, None, 12,
                    metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
                )
        self.assertIn("observation_complete", str(ctx.exception))

    def test_cell_observation_rejects_mount_bytes_that_differ_from_scheduled_tree(self):
        class MutatingAdapter(tm.AgentAdapter):
            name = "stub"

            def mount(self, tree_dir, workspace):
                copied = self._mount_tree(tree_dir, workspace / "skills")
                copied[0].write_text("mutated after scheduling\n", encoding="utf-8")
                return copied

            def invoke(self, query, model, workspace, timeout):
                raise AssertionError("mismatched mounted bytes must fail before invocation")

        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "tree"
            skill = tree / "demo"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: demo\n---\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mounted skill tree hash"):
                tm.observe_cell_query(
                    MutatingAdapter(), tree, "q", True, None, 12,
                    metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
                )

    def test_missing_capability_row_fails_before_any_runs(self):
        class UnregisteredAdapter(tm.AgentAdapter):
            name = "my-agent"

            def mount(self, tree_dir, workspace):
                raise AssertionError("mount should not run before capability validation")

            def invoke(self, query, model, workspace, timeout):
                raise AssertionError("invoke should not run before capability validation")

        old = dict(tm.ADAPTERS)
        try:
            tm.ADAPTERS["my-agent"] = UnregisteredAdapter
            with self.assertRaises(SystemExit) as ctx:
                tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows()[:1], agents=["my-agent"],
                              models=[None], runs_per_query=1, timeout=30, workers=1)
        finally:
            tm.ADAPTERS.clear()
            tm.ADAPTERS.update(old)
        self.assertIn("AGENT_CAPABILITIES", str(ctx.exception))

    def test_interpreter_wrapper_identity_binds_script_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "wrapper.py"
            script.write_text("print('first')\n", encoding="utf-8")
            first = tm.executable_identity(f"{sys.executable} {script}")
            script.write_text("print('second')\n", encoding="utf-8")
            second = tm.executable_identity(f"{sys.executable} {script}")
        self.assertNotEqual(first, second)
        self.assertIn(str(script.resolve()), first["argument_files"])

    def test_codex_baseline_and_ablation_reports_pair_in_trigger_compare(self):
        # A fake `codex exec --json` that ends its turn without loading the skill.
        fake_codex = (
            "import json\n"
            "for record in ({'type': 'thread.started', 'thread_id': 't'}, {'type': 'turn.started'},\n"
            "               {'type': 'item.completed', 'item': {'id': 'i', 'type': 'agent_message', 'text': 'ok'}},\n"
            "               {'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}}):\n"
            "    print(json.dumps(record))\n")
        rows = [{"query_id": "negative", "query": "ordinary chat", "should_trigger": False}]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "fake_codex.py").write_text(fake_codex, encoding="utf-8")
            user_codex = root / "user-codex"
            user_codex.mkdir()
            (user_codex / "auth.json").write_text('{"token": "codex-user-token"}', encoding="utf-8")
            paths = {}
            with mock.patch.dict(os.environ, {"CODEX_HOME": str(user_codex)}):
                for arm, ablation in (("baseline", None), ("ablation", "weaker-description")):
                    report = tm.run_matrix(
                        DEMO_MANIFEST, rows, agents=["codex"], models=[None], runs_per_query=1,
                        timeout=30, workers=1, codex_cmd=f"{sys.executable} {root / 'fake_codex.py'}",
                        ablation=ablation)
                    self.assertEqual(report["summary"]["measurement_status"], "complete")
                    paths[arm] = root / f"{arm}.json"
                    paths[arm].write_text(json.dumps(report), encoding="utf-8")
            code, stdout, stderr = run_cli("trigger-compare", "--baseline", paths["baseline"],
                                           "--ablation", paths["ablation"])
        self.assertEqual(code, 0, stderr)
        compared = json.loads(stdout)
        self.assertEqual(compared["paired"]["blocked"], [])
        self.assertTrue(compared["provenance"]["verified"])


class VibeAdapterTests(unittest.TestCase):
    """Mistral Vibe trigger support without a live API key."""

    def test_vibe_cmd_flag_defaults_to_the_shared_vibe_command(self):
        parser = tm.build_arg_parser()
        vibe_action = next(a for a in parser._actions if "--vibe-cmd" in getattr(a, "option_strings", ()))
        self.assertEqual(vibe_action.default, tm.VIBE_DEFAULT_CMD)

    def test_vibe_mounts_project_agent_skills(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tree = root / "tree"
            skill = tree / "demo-root"
            skill.mkdir(parents=True)
            (skill / "SKILL.md").write_text("---\nname: demo-reviewer\n---\n", encoding="utf-8")
            copied = tm.VibeAdapter().mount(tree, root / "workspace")
        self.assertEqual(copied[0].parts[-4:], ("workspace", ".agents", "skills", "demo-root", "SKILL.md")[-4:])
        self.assertIn(".agents", str(copied[0]))

    def test_vibe_detects_native_skill_tool_call(self):
        stream = "\n".join([
            json.dumps({"role": "assistant", "tool_calls": [{
                "id": "call-1", "function": {"name": "skill",
                "arguments": json.dumps({"name": "demo-reviewer"})}}]}),
            json.dumps({"role": "tool", "tool_call_id": "call-1", "content": "loaded"}),
            json.dumps({"role": "assistant", "content": "done"}),
        ])
        detection = tm.VibeAdapter().detect(completed_invocation(stream), ["demo-reviewer"], [])
        self.assertTrue(detection.triggered)
        self.assertIn("Vibe skill tool invoked: demo-reviewer", detection.legacy_evidence)
        other = json.dumps({"role": "assistant", "tool_calls": [{"function": {"name": "skill", "arguments": json.dumps({"name": "other"})}}]})
        self.assertFalse(tm.VibeAdapter().detect(completed_invocation(other), ["demo-reviewer"], []).triggered)

    def test_vibe_invoke_uses_isolated_home_model_env_and_prompt_arg(self):
        seen = {}

        def fake_run(plan):
            argv, cwd = list(plan.argv), plan.cwd
            env, timeout = dict(plan.environment or {}), int(plan.timeout_s)
            input_text = plan.input_text
            seen.update({"argv": argv, "cwd": cwd, "env": env, "timeout": timeout, "input_text": input_text})
            seen["vibe_home_inside_workdir"] = Path(env["VIBE_HOME"]).is_relative_to(Path(cwd))
            seen["workspace_vibe_env_present"] = (Path(cwd) / ".vibe-home" / ".env").exists()
            return completed_invocation(json.dumps({"role": "assistant", "content": "ok"}) + "\n")

        with tempfile.TemporaryDirectory() as td, mock.patch.object(tm.VibeAdapter, "_run_argv", staticmethod(fake_run)):
            workspace = Path(td) / "workspace"
            workspace.mkdir()
            result = tm.VibeAdapter(vibe_cmd=f"{sys.executable} fake_vibe.py", max_turns=4).invoke("raw trigger query", "mistral-small", workspace, 12)
        self.assertIn("--prompt", seen["argv"])
        self.assertEqual(seen["argv"][seen["argv"].index("--prompt") + 1], "raw trigger query")
        self.assertIn("--output", seen["argv"])
        self.assertIn("--workdir", seen["argv"])
        self.assertIn("--enabled-tools", seen["argv"])
        self.assertIn("skill", seen["argv"])
        self.assertEqual(seen["input_text"], "")
        self.assertEqual(seen["env"]["VIBE_ACTIVE_MODEL"], "mistral-small")
        self.assertFalse(seen["vibe_home_inside_workdir"])
        self.assertFalse(seen["workspace_vibe_env_present"])
        self.assertTrue(result.metadata["config_isolated"])
        self.assertTrue(result.metadata["vibe_home_outside_workdir"])

    def test_a_vibe_2_23_history_entry_stream_is_a_complete_cell_with_skill_tool_evidence(self):
        # Vibe 2.23 and later write public history entries (built from Vibe
        # 2.25.8's own code, tests/fixtures/vibe/README.md). A stream that
        # loads the mounted `demo` skill triggers; one that answers without a
        # tool is a complete observation that did not trigger.
        fixtures = ROOT / "tests" / "fixtures" / "vibe"
        rows = [{"query_id": "q", "query": "Review this pull request description.", "should_trigger": True}]
        for name, triggered, evidence in (
                ("streaming.2.25.8.skill-load.jsonl", True, ["Vibe skill tool invoked: demo"]),
                ("streaming.2.25.8.no-tools.jsonl", False, [])):
            with self.subTest(fixture=name), tempfile.TemporaryDirectory() as td:
                fake_vibe = Path(td) / "fake_vibe.py"
                fake_vibe.write_text(
                    f"import sys\nsys.stdout.write(open({str(fixtures / name)!r}, encoding='utf-8').read())\n",
                    encoding="utf-8")
                report = tm.run_matrix(DEMO_MANIFEST, rows, agents=["vibe"], models=[None],
                                       runs_per_query=1, timeout=30, workers=1,
                                       vibe_cmd=f"{sys.executable} {fake_vibe}")
                row = report["results"][0]
                self.assertIs(row["observation_complete"], True, row.get("provider_error"))
                self.assertEqual((row["triggered"], row["evidence"]), (triggered, evidence))


DEMO_SKILLS = ROOT / "examples" / "demo-skill" / "skills"


class MountedSkillNameTests(unittest.TestCase):
    """Each CLI names a mounted skill its own way, and detection must accept
    exactly that name. The demo tree mounts under folder `demo` with frontmatter
    `name: demo-reviewer`, so the two names diverge.

    Recorded evidence per CLI:
    - Claude Code 2.1.284 (live haiku run): the init event listed
      `"skills": ["demo", ...]` and the model called `Skill {"skill": "demo"}`.
    - Codex 0.156.1 (requests captured by a local sink, no model call): the
      skills listing read `demo-reviewer: ... (file: .../skills/demo/SKILL.md)`,
      `$demo-reviewer` produced the user-role injection
      `<skill>\\n<name>demo-reviewer</name>\\n<path>.../skills/demo/SKILL.md</path>`,
      and `$demo` produced no injection at all.
    - Mistral Vibe 2.25.8 source: `SkillManager` keys skills by frontmatter
      `name` (a folder mismatch only logs a warning), and the `skill` tool
      looks its `name` argument up in that map."""

    def _row(self, adapter, fake_run, tree_dir=DEMO_SKILLS):
        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(type(adapter), "_run_argv", staticmethod(fake_run)), \
             mock.patch.dict(os.environ, {"CODEX_HOME": str(Path(td) / "ambient-codex")}):
            return tm.observe_cell_query(
                adapter, tree_dir, "Review this proposed change", True, None, 12,
                metadata={"skill_tree_hash": sb.skill_tree_hash(tree_dir)},
            ).as_row()

    def _claude_row(self, invoked: str):
        stream = "\n".join(json.dumps(record) for record in [
            {"type": "system", "subtype": "init", "session_id": "s", "skills": ["demo", "design", "doctor"]},
            {"type": "assistant", "message": {"role": "assistant", "content": [{
                "type": "tool_use", "id": "toolu_01Ttn2zkj2An4sgwBUhBdCZ7", "name": "Skill",
                "input": {"skill": invoked, "args": "Review this proposed change"}}]}},
            {"type": "user", "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "toolu_01Ttn2zkj2An4sgwBUhBdCZ7",
                "content": f"Launching skill: {invoked}"}]}},
            {"type": "result", "subtype": "success", "is_error": False, "result": "done"},
        ]) + "\n"
        return self._row(tm.ClaudeAdapter(), lambda plan: completed_invocation(stream))

    def _codex_row(self, injected: str, tree_dir=DEMO_SKILLS):
        events = (CODEX_FIXTURES / "exec-skill-events.jsonl").read_text(encoding="utf-8")

        def fake_run(plan):
            home = Path(dict(plan.environment)["CODEX_HOME"])
            rollout = (CODEX_FIXTURES / "rollout-skill-injection.jsonl").read_text(encoding="utf-8")
            rollout = rollout.replace("<name>unslop</name>", f"<name>{injected}</name>").replace(
                "/CODEX_HOME/skills/unslop/SKILL.md", f"{home}/skills/demo/SKILL.md")
            day = home / "sessions" / "2026" / "09" / "13"
            day.mkdir(parents=True)
            (day / f"rollout-2026-09-13T13-30-23-{CODEX_THREAD_ID}.jsonl").write_text(rollout, encoding="utf-8")
            return completed_invocation(events)

        return self._row(tm.CodexAdapter(codex_cmd="codex exec --json"), fake_run, tree_dir)

    def _vibe_row(self, invoked: str):
        stream = "\n".join(json.dumps(record) for record in [
            {"role": "assistant", "tool_calls": [{
                "id": "call-1", "function": {"name": "skill", "arguments": json.dumps({"name": invoked})}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "loaded"},
            {"role": "assistant", "content": "done"},
        ]) + "\n"
        return self._row(tm.VibeAdapter(), lambda plan: completed_invocation(stream))

    def test_claude_skill_call_by_mount_folder_triggers(self):
        row = self._claude_row("demo")
        self.assertTrue(row["observation_complete"], row.get("provider_error"))
        self.assertTrue(row["triggered"])
        self.assertEqual(row["evidence"], ["Skill tool invoked: demo"])

    def test_claude_skill_call_by_host_name_does_not_trigger(self):
        for invoked in ("design",):
            with self.subTest(invoked=invoked):
                row = self._claude_row(invoked)
                self.assertTrue(row["observation_complete"], row.get("provider_error"))
                self.assertFalse(row["triggered"])

    def test_claude_skill_call_by_frontmatter_alias_triggers(self):
        row = self._claude_row("demo-reviewer")
        self.assertTrue(row["triggered"])
        self.assertEqual([item["kind"] for item in row["evidence_typed"]], ["skill_tool"])

    def test_vibe_skill_call_by_folder_alias_triggers(self):
        row = self._vibe_row("demo")
        self.assertTrue(row["triggered"])
        self.assertEqual([item["kind"] for item in row["evidence_typed"]], ["vibe_skill_tool"])

    def test_codex_rollout_injection_by_frontmatter_name_triggers(self):
        row = self._codex_row("demo-reviewer")
        self.assertTrue(row["observation_complete"], row.get("provider_error"))
        self.assertTrue(row["triggered"])
        self.assertEqual(len(row["evidence"]), 1)
        self.assertRegex(row["evidence"][0], r"^rollout skill injection: demo-reviewer \(.*/skills/demo/SKILL\.md\)$")

    def test_codex_rollout_injection_by_folder_or_other_name_does_not_trigger(self):
        for injected in ("demo", "unslop"):
            with self.subTest(injected=injected):
                row = self._codex_row(injected)
                self.assertTrue(row["observation_complete"], row.get("provider_error"))
                self.assertFalse(row["triggered"])

    def test_vibe_skill_call_by_frontmatter_name_triggers(self):
        row = self._vibe_row("demo-reviewer")
        self.assertTrue(row["observation_complete"], row.get("provider_error"))
        self.assertTrue(row["triggered"])
        self.assertEqual(row["evidence"], ["Vibe skill tool invoked: demo-reviewer"])

    def test_vibe_skill_call_by_other_name_does_not_trigger(self):
        for invoked in ("other",):
            with self.subTest(invoked=invoked):
                row = self._vibe_row(invoked)
                self.assertTrue(row["observation_complete"], row.get("provider_error"))
                self.assertFalse(row["triggered"])

    def test_codex_rollout_injection_matches_padded_frontmatter_name(self):
        # A SKILL.md frontmatter `name:` value can carry stray whitespace
        # (`" demo-reviewer "`). Codex strips it before listing/injecting the
        # skill, so detection must strip it too, or a real injection of the
        # trimmed name never matches the padded needle `mounted_skill_names`
        # read straight from frontmatter.
        with tempfile.TemporaryDirectory() as copy_root:
            tree_dir = Path(copy_root) / "skills"
            shutil.copytree(DEMO_SKILLS, tree_dir)
            skill_md = tree_dir / "demo" / "SKILL.md"
            skill_md.write_text(
                skill_md.read_text(encoding="utf-8").replace(
                    "name: demo-reviewer", 'name: " demo-reviewer "', 1),
                encoding="utf-8",
            )
            row = self._codex_row("demo-reviewer", tree_dir=tree_dir)
        self.assertTrue(row["observation_complete"], row.get("provider_error"))
        self.assertTrue(row["triggered"])
        self.assertEqual(len(row["evidence"]), 1)
        self.assertRegex(row["evidence"][0], r"^rollout skill injection: demo-reviewer \(.*/skills/demo/SKILL\.md\)$")


def _csv_env(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [part.strip() or None for part in raw.split(",")]


def _live_invoke_smoke_agents():
    configured = [name for name in _csv_env("AGENT_INVOKE_SMOKE_AGENTS", []) if name]
    if configured:
        return [str(name) for name in configured if name]
    return [
        name for name in tm.ADAPTERS
        if name != "stub" and tm.require_agent_capabilities(name).autonomous_trigger
    ]


def _live_invoke_smoke_models(agent_name, adapter):
    upper = agent_name.upper().replace("-", "_")
    models = _csv_env(f"{upper}_INVOKE_SMOKE_MODELS", adapter.default_models)
    if os.environ.get(f"{upper}_INVOKE_SMOKE_MODEL") is not None:
        models = [os.environ.get(f"{upper}_INVOKE_SMOKE_MODEL") or None]
    return models


@unittest.skipUnless(os.environ.get("RUN_AGENT_INVOKE_SMOKE") == "1",
                     "cheap manual smoke: set RUN_AGENT_INVOKE_SMOKE=1 (needs live agent CLIs + credentials, spends tiny requests)")
class AgentInvokeSmokeTests(unittest.TestCase):
    def test_live_agents_complete_trivial_prompt_for_each_model(self):
        query = os.environ.get("AGENT_INVOKE_SMOKE_QUERY", "Reply with exactly: OK")
        timeout = int(os.environ.get("AGENT_INVOKE_SMOKE_TIMEOUT", "90"))
        manifest = tm.load_manifest(DEMO_MANIFEST)
        repo_root = tm.repo_root_for_manifest(DEMO_MANIFEST)
        results = []
        failures = []

        for agent_name in _live_invoke_smoke_agents():
            try:
                adapter = tm.adapter_instance(
                    agent_name,
                    claude_bin=os.environ.get("CLAUDE_INVOKE_SMOKE_BIN", "claude"),
                    codex_cmd=os.environ.get("CODEX_INVOKE_SMOKE_CMD", tm.DEFAULT_CODEX_CMD),
                    vibe_cmd=os.environ.get("VIBE_INVOKE_SMOKE_CMD", tm.VIBE_DEFAULT_CMD),
                    max_turns=int(os.environ.get("CLAUDE_INVOKE_SMOKE_MAX_TURNS", "1")),
                )
                models = _live_invoke_smoke_models(agent_name, adapter)
            except Exception as exc:
                failures.append(f"{agent_name}/(setup): {exc!r}")
                results.append({"agent": agent_name, "model": "(setup)", "ok": False, "error": repr(exc)})
                continue

            for model in models:
                model_label = model or "(default)"
                row = {"agent": agent_name, "model": model_label}
                try:
                    with tempfile.TemporaryDirectory(prefix=f"{agent_name}-invoke-smoke-") as td:
                        workspace = Path(td)
                        tree = tm.build_canonical_skill_tree(repo_root, manifest, workspace / "tree")
                        try:
                            copied = adapter.mount(tree, workspace)
                            result = tm.validate_invoke_result(
                                agent_name,
                                adapter.invoke(query, model, workspace, timeout),
                            )
                        finally:
                            adapter.release(workspace)
                    row.update({
                        "returncode": result.returncode,
                        "timed_out": result.timed_out,
                        "observation_complete": result.observation_complete,
                        "elapsed_ms": result.elapsed_ms,
                        "stdout_bytes": len(result.stdout),
                        "stdout_tail": result.stdout[-300:],
                        "stderr_tail": result.stderr[-300:],
                        "mounted_paths": len(copied),
                    })
                    ok = (
                        not result.timed_out and
                        result.returncode == 0 and
                        result.observation_complete and
                        bool(result.stdout.strip())
                    )
                    row["ok"] = ok
                    if not ok:
                        failures.append(
                            f"{agent_name}/{model_label}: returncode={result.returncode} "
                            f"timed_out={result.timed_out} observation_complete={result.observation_complete} "
                            f"stdout_bytes={len(result.stdout)} "
                            f"stdout_tail={result.stdout[-300:]!r} "
                            f"stderr_tail={result.stderr[-300:]!r}"
                        )
                except Exception as exc:
                    row.update({"ok": False, "error": repr(exc)})
                    failures.append(f"{agent_name}/{model_label}: {exc!r}")
                results.append(row)

        if not results:
            failures.append("no live agent/model smoke targets were selected")
        print(json.dumps({"cheap_invoke_smoke": results}, indent=2, sort_keys=True))
        self.assertFalse(failures, "\n".join(failures))


class AgentInvokeSmokeConfigTests(unittest.TestCase):
    def test_default_cheap_smoke_targets_every_live_supported_agent_and_model(self):
        clean_env = {
            "AGENT_INVOKE_SMOKE_AGENTS": "",
            "CLAUDE_INVOKE_SMOKE_MODELS": "",
            "CODEX_INVOKE_SMOKE_MODEL": "",
            "PI_INVOKE_SMOKE_MODEL": "",
            "VIBE_INVOKE_SMOKE_MODEL": "",
        }
        with mock.patch.dict(os.environ, clean_env, clear=False):
            for key in clean_env:
                os.environ.pop(key, None)
            agents = _live_invoke_smoke_agents()
            models = {
                name: _live_invoke_smoke_models(name, tm.adapter_instance(name))
                for name in agents
            }
        self.assertEqual(agents, ["claude", "codex", "pi", "vibe"])
        self.assertEqual(models["claude"], ["haiku", "sonnet", "opus"])
        self.assertEqual(models["codex"], [None])
        self.assertEqual(models["pi"], [None])
        self.assertEqual(models["vibe"], [None])


@unittest.skipUnless(os.environ.get("RUN_TRIGGER_SMOKE") == "1",
                     "manual smoke: set RUN_TRIGGER_SMOKE=1 (needs claude CLI + credentials, spends tokens)")
class ClaudeMatrixSmokeTests(unittest.TestCase):
    def test_haiku_sonnet_opus_matrix_end_to_end(self):
        runs = int(os.environ.get("TRIGGER_SMOKE_RUNS", "1"))
        report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows(), agents=["claude"],
                               models=["haiku", "sonnet", "opus"], runs_per_query=runs,
                               timeout=300, workers=3)
        tm.print_matrix(report["matrix"])
        self.assertEqual(len(report["matrix"]), 3)
        incomplete = [r for r in report["results"] if not r["observation_complete"]]
        self.assertFalse(incomplete, f"broken runs (crash/timeout), not trigger signal: {incomplete}")
        self.assertTrue(any(r["triggered"] for r in report["results"]),
                        "no model loaded the skill on any run — detection or mounting is broken")


@unittest.skipUnless(os.environ.get("RUN_CODEX_TRIGGER_SMOKE") == "1",
                     "manual smoke: set RUN_CODEX_TRIGGER_SMOKE=1 (needs codex CLI + credentials, spends tokens)")
class CodexMatrixSmokeTests(unittest.TestCase):
    def test_codex_matrix_end_to_end(self):
        runs = int(os.environ.get("CODEX_TRIGGER_SMOKE_RUNS", "1"))
        model = os.environ.get("CODEX_TRIGGER_SMOKE_MODEL")
        report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows(), agents=["codex"],
                               models=[model] if model else [None], runs_per_query=runs,
                               timeout=300, workers=1)
        tm.print_matrix(report["matrix"])
        incomplete = [r for r in report["results"] if not r["observation_complete"]]
        self.assertFalse(incomplete, f"broken runs (crash/timeout), not trigger signal: {incomplete}")
        self.assertTrue(any(r["triggered"] for r in report["results"]),
                        "no Codex run loaded the skill — detection, auth, or mounting is broken")


@unittest.skipUnless(os.environ.get("RUN_PI_TRIGGER_SMOKE") == "1",
                     "manual smoke: set RUN_PI_TRIGGER_SMOKE=1 (needs Pi CLI + credentials, spends tokens)")
class PiMatrixSmokeTests(unittest.TestCase):
    def test_pi_matrix_end_to_end(self):
        runs = int(os.environ.get("PI_TRIGGER_SMOKE_RUNS", "1"))
        model = os.environ.get("PI_TRIGGER_SMOKE_MODEL")
        report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows(), agents=["pi"],
                               models=[model] if model else [None], runs_per_query=runs,
                               timeout=300, workers=1)
        tm.print_matrix(report["matrix"])
        incomplete = [r for r in report["results"] if not r["observation_complete"]]
        self.assertFalse(incomplete, f"broken runs (crash/timeout), not trigger signal: {incomplete}")
        self.assertTrue(any(r["triggered"] for r in report["results"]),
                        "no Pi run loaded the skill — detection, auth, or mounting is broken")


@unittest.skipUnless(os.environ.get("RUN_VIBE_TRIGGER_SMOKE") == "1",
                     "manual smoke: set RUN_VIBE_TRIGGER_SMOKE=1 (needs vibe CLI + MISTRAL_API_KEY, spends tokens)")
class VibeMatrixSmokeTests(unittest.TestCase):
    def test_vibe_matrix_end_to_end(self):
        runs = int(os.environ.get("VIBE_TRIGGER_SMOKE_RUNS", "1"))
        model = os.environ.get("VIBE_TRIGGER_SMOKE_MODEL")
        report = tm.run_matrix(DEMO_MANIFEST, demo_trigger_rows(), agents=["vibe"],
                               models=[model] if model else [None], runs_per_query=runs,
                               timeout=300, workers=1,
                               vibe_cmd=os.environ.get("VIBE_TRIGGER_SMOKE_CMD", tm.VIBE_DEFAULT_CMD))
        tm.print_matrix(report["matrix"])
        incomplete = [r for r in report["results"] if not r["observation_complete"]]
        self.assertFalse(incomplete, f"broken runs (crash/timeout), not trigger signal: {incomplete}")
        self.assertTrue(any(r["triggered"] for r in report["results"]),
                        "no Vibe run loaded the skill — detection, auth, or mounting is broken")


def trigger_row(query, should, *, triggered, complete=True, agent="stub", model=None,
                query_id=None, run_number=1):
    """One persisted trigger-matrix result row, valid under
    TriggerObservation.from_row's contract."""
    evidence = ["skills/demo/SKILL.md"] if (complete and triggered) else []
    return {
        "population": "trigger", "agent": agent, "model": model, "query": query,
        "query_id": query_id or query, "run_number": run_number,
        "should_trigger": should, "triggered": complete and triggered,
        "pass": complete and (triggered == should),
        "observation_complete": complete,
        "returncode": 0 if complete else 124, "timed_out": not complete,
        "elapsed_ms": 5, "completion_evidence": "normal_exit" if complete else None,
        "evidence": evidence,
        "evidence_typed": [{"kind": "mounted_path", "text": t} for t in evidence],
        "protocol_observation": {},
        "usage_normalized": {"source": "missing"}, "cost_normalized": {"source": "missing"},
        "stderr": "",
    }


BASE_HASH = "sha256:base-revision"
EDIT_HASH = "sha256:edited-revision"
ABLATION_PROVENANCE = {
    "id": "drop-description", "mode": "materialized", "population": "trigger",
    "skill_hash": EDIT_HASH, "parent_skill_hash": BASE_HASH,
    "components": [{"class": "discovery", "mechanism": "frontmatter_field",
                    "skill_root": "skills/demo", "target": {"field": "description"}}],
}

TRIGGER_MANIFEST = {
    "skill_name": "demo",
    "skill_paths": ["skills/demo"],
    "ablations": [{
        "id": "drop-description", "population": "trigger",
        "components": [{"class": "discovery", "mechanism": "frontmatter_field",
                        "skill_root": "skills/demo", "target": {"field": "description"}}],
    }],
}


def trigger_report(rows, *, ablation=None, provenance=None, tree_hash=BASE_HASH,
                   runs_per_query=2):
    design = []
    seen = set()
    for row in rows:
        key = (row["agent"], row["model"], row["query_id"])
        if key not in seen:
            design.append({k: row[k] for k in (
                "agent", "model", "query_id", "query", "should_trigger")})
            seen.add(key)
    adapter_models = {}
    for row in rows:
        adapter_models.setdefault(row["agent"], [])
        if row["model"] not in adapter_models[row["agent"]]:
            adapter_models[row["agent"]].append(row["model"])
    protocol = {
        "schema_version": 1, "producer": "skill-trigger-matrix",
        "harness_identity": sb.trigger_harness_identity(),
        "timeout_seconds": 30, "runs_per_query": runs_per_query, "workers": 1,
        "adapters": [
            {"adapter": f"run_trigger_matrix.{agent.title()}Adapter", "agent": agent,
             "trace_dialect": agent,
             "implementation_sha256": "sha256:" + ("0" * 64),
             "producer_sha256": "sha256:" + ("1" * 64),
             "required_observations": {}, "models": models}
            for agent, models in sorted(adapter_models.items())
        ],
    }
    protocol_sha256 = sb.canonical_json_sha256(protocol)
    rows = [{**row, "skill_tree_hash": tree_hash,
             "protocol_sha256": protocol_sha256,
             "protocol_observation": row.get("protocol_observation", {})}
            for row in rows]
    return {"skill_name": "demo", "generated_at": 1,
            "evidence_class": tm.TRIGGER_MEASUREMENT_EVIDENCE_CLASS,
            "skill_tree_hash": tree_hash, "ablation": ablation,
            "provenance": provenance if provenance is not None else {"mode": "baseline", "skill_tree_hash": tree_hash},
            "manifest_identity": sb.trigger_manifest_identity(TRIGGER_MANIFEST),
            "protocol": protocol, "protocol_sha256": protocol_sha256,
            "runs_per_query": runs_per_query, "design": design, "results": rows}


class PiProtocolRequirementTests(unittest.TestCase):
    """A Pi report declares the isolation controls it ran under, and only the
    controls the adapter requires today are accepted. Reports from before Pi's
    home moved out of its working directory carry an older harness identity
    and are refused before this check."""

    def protocol(self, required):
        adapter = tm.PiAdapter().protocol_parameters()
        return {"schema_version": 1, "producer": "skill-trigger-matrix",
                "harness_identity": sb.trigger_harness_identity(),
                "timeout_seconds": 30, "runs_per_query": 1, "workers": 1,
                "adapters": [{**adapter, "required_observations": required, "models": [None]}]}

    def validate(self, required):
        return sb._validated_trigger_protocol(
            self.protocol(required), label="report", runs_per_query=1,
            design_pairs={("pi", None)})

    def test_the_adapter_declares_its_home_outside_the_working_directory(self):
        declared = tm.PiAdapter().protocol_parameters()["required_observations"]
        self.assertEqual(self.validate(declared), {"pi": declared})

    def test_any_other_control_set_is_refused(self):
        for required in ({"config_isolated": True}, {"pi_home_outside_workdir": True}):
            stderr = io.StringIO()
            with self.subTest(required=required), contextlib.redirect_stderr(stderr), \
                    self.assertRaises(SystemExit):
                self.validate(required)
            self.assertIn("must require", stderr.getvalue())

    def test_a_report_from_an_older_harness_identity_is_refused_by_name(self):
        identity = sb.trigger_harness_identity()
        payload = {key: value for key, value in identity.items() if key != "identity_sha256"}
        payload["schema_version"] = sb.TRIGGER_HARNESS_IDENTITY_VERSION - 1
        older = {**payload, "identity_sha256": sb.canonical_json_sha256(payload)}
        with self.assertRaisesRegex(ValueError, "identity v2; this harness reads v3. Regenerate"):
            sb.validate_trigger_harness_identity(older, "baseline")


class TriggerComparisonTests(unittest.TestCase):
    """build_trigger_comparison pairs a baseline matrix run with an --ablation
    run of the SAME canonical revision, mirroring the answer path's
    causal-confirmation gate: provenance verified + coverage + an observed,
    sign-flip-significant pass-rate drop across queries."""

    QUERIES = [f"query {n}" for n in range(1, 7)]   # 6 paired deltas: exact p ~= 0.031

    def _baseline_rows(self):
        return [trigger_row(q, True, triggered=True, run_number=run_number)
                for q in self.QUERIES for run_number in range(1, 3)]

    def _ablation_rows(self, *, triggered=False):
        return [trigger_row(q, True, triggered=triggered, run_number=run_number)
                for q in self.QUERIES for run_number in range(1, 3)]

    def _compare(self, base_rows=None, abl_rows=None, *, provenance=None,
                 abl_hash=EDIT_HASH, base_runs_per_query=2,
                 abl_runs_per_query=None):
        baseline = trigger_report(
            base_rows if base_rows is not None else self._baseline_rows(),
            runs_per_query=base_runs_per_query)
        ablation = trigger_report(abl_rows if abl_rows is not None else self._ablation_rows(),
                                  ablation="drop-description",
                                  provenance=provenance if provenance is not None else ABLATION_PROVENANCE,
                                  tree_hash=abl_hash,
                                  runs_per_query=(abl_runs_per_query
                                                  if abl_runs_per_query is not None
                                                  else base_runs_per_query))
        return sb.build_trigger_comparison(baseline, ablation)

    def test_verified_significant_drop_confirms_causal(self):
        out = self._compare()
        self.assertEqual(out["population"], "trigger")
        self.assertEqual(out["evidence_class"], "confirmed_causal")
        self.assertTrue(out["provenance"]["verified"])
        self.assertEqual(out["summary"]["comparable"], 6)
        self.assertEqual(len(out["regressed_queries"]), 6)
        self.assertTrue(out["paired"]["significance"]["significant_at_0_05"])
        self.assertEqual(out["paired"]["comparable_queries"][0]["pass_delta"], -1.0)

    def test_comparer_keeps_sub_millionth_rate_deltas(self):
        epsilon = 1 / 3_000_000
        pass_rates = iter(
            value for _ in self.QUERIES for value in (1.0, 1.0 - epsilon)
        )
        trigger_rates = iter(
            value for _ in self.QUERIES for value in (1.0, 1.0 - epsilon)
        )
        with mock.patch.object(
            sb.CompleteTriggerCohort, "pass_rate",
            new_callable=mock.PropertyMock, side_effect=pass_rates,
        ) as pass_rate, mock.patch.object(
            sb.CompleteTriggerCohort, "trigger_rate",
            new_callable=mock.PropertyMock, side_effect=trigger_rates,
        ) as trigger_rate:
            out = self._compare()
        self.assertEqual(pass_rate.call_count, 2 * len(self.QUERIES))
        self.assertEqual(trigger_rate.call_count, 2 * len(self.QUERIES))
        self.assertLess(out["paired"]["comparable_queries"][0]["pass_delta"], 0)
        self.assertAlmostEqual(
            out["paired"]["comparable_queries"][0]["pass_delta"], -epsilon)
        self.assertLess(out["summary"]["mean_pass_delta"], 0)

    def test_no_drop_is_refuted(self):
        out = self._compare(abl_rows=self._ablation_rows(triggered=True))
        self.assertEqual(out["evidence_class"], "refuted")
        self.assertEqual(out["regressed_queries"], [])

    def test_observed_but_insignificant_drop_is_indeterminate(self):
        # one regressed query out of six cannot clear the sign-flip bar
        abl = [trigger_row(q, True, triggered=(q != "query 1"), run_number=run_number)
               for q in self.QUERIES for run_number in range(1, 3)]
        out = self._compare(abl_rows=abl)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertEqual(len(out["regressed_queries"]), 1)
        self.assertIn("not significant", out["note"])

    def test_significant_change_in_wrong_direction_is_refuted(self):
        queries = [f"direction {n}" for n in range(10)]
        baseline = []
        ablation = []
        for i, query in enumerate(queries):
            # One regression, nine improvements: the old two-sided gate called
            # this confirmed merely because at least one cell was negative.
            baseline.append(trigger_row(query, True, triggered=(i == 0)))
            ablation.append(trigger_row(query, True, triggered=(i != 0)))
        out = self._compare(base_rows=baseline, abl_rows=ablation,
                            base_runs_per_query=1)
        self.assertTrue(out["paired"]["significance"]["significant_at_0_05"])
        self.assertGreater(out["summary"]["mean_pass_delta"], 0)
        self.assertEqual(out["evidence_class"], "refuted")
        self.assertIn("aggregate mean pass delta is non-negative", out["note"])
        self.assertNotIn("not significant", out["note"])

    def test_models_do_not_multiply_one_query_into_six_units(self):
        baseline = [trigger_row("one query", True, triggered=True, model=f"m{n}")
                    for n in range(6)]
        ablation = [trigger_row("one query", True, triggered=False, model=f"m{n}")
                    for n in range(6)]
        out = self._compare(base_rows=baseline, abl_rows=ablation,
                            base_runs_per_query=1)
        self.assertEqual(out["summary"]["comparable_cells"], 6)
        self.assertEqual(out["paired"]["significance"]["n"], 1)
        self.assertEqual(out["evidence_class"], "indeterminate")

    def test_revision_mismatch_is_indeterminate_with_reason(self):
        provenance = {**ABLATION_PROVENANCE, "parent_skill_hash": "sha256:other-revision"}
        out = self._compare(provenance=provenance)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertFalse(out["provenance"]["verified"])
        self.assertTrue(any("different skill revision" in r for r in out["provenance"]["reasons"]))

    def test_baseline_provenance_must_attest_its_top_level_hash(self):
        baseline = trigger_report(self._baseline_rows())
        baseline["provenance"]["skill_tree_hash"] = "sha256:other"
        ablation = trigger_report(self._ablation_rows(), ablation="drop-description",
                                  provenance=ABLATION_PROVENANCE, tree_hash=EDIT_HASH)
        out = sb.build_trigger_comparison(baseline, ablation)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertTrue(any("baseline provenance" in reason
                            for reason in out["provenance"]["reasons"]))

    def test_skill_name_mismatch_is_indeterminate(self):
        baseline = trigger_report(self._baseline_rows())
        ablation = trigger_report(self._ablation_rows(), ablation="drop-description",
                                  provenance=ABLATION_PROVENANCE, tree_hash=EDIT_HASH)
        ablation["skill_name"] = "other"
        ablation["manifest_identity"] = sb.trigger_manifest_identity(
            {**TRIGGER_MANIFEST, "skill_name": "other"})
        out = sb.build_trigger_comparison(baseline, ablation)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertTrue(any("different skills" in reason
                            for reason in out["provenance"]["reasons"]))

    def test_top_level_ablation_id_must_match_provenance(self):
        provenance = {**ABLATION_PROVENANCE, "id": "some-other-ablation"}
        out = self._compare(provenance=provenance)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertFalse(out["provenance"]["verified"])
        self.assertTrue(any("does not match provenance id" in reason
                            for reason in out["provenance"]["reasons"]))

    def test_missing_and_incomplete_arms_are_blocked_pairs(self):
        base = self._baseline_rows() + [
            trigger_row("only baseline", True, triggered=True, run_number=n)
            for n in range(1, 3)
        ]
        abl = self._ablation_rows() + [
            trigger_row("timed out", True, triggered=False,
                        complete=(n == 1), run_number=n)
            for n in range(1, 3)
        ]
        base += [trigger_row("timed out", True, triggered=True, run_number=n)
                 for n in range(1, 3)]
        out = self._compare(base_rows=base, abl_rows=abl)
        reasons = {b["query"]: b["reason"] for b in out["paired"]["blocked"]}
        self.assertEqual(reasons["only baseline"], "missing_ablation_arm")
        self.assertEqual(reasons["timed out"], "ablation_observations_incomplete")
        self.assertEqual(out["summary"]["comparable"], 6)
        self.assertFalse(out["paired"]["significance"]["significant_at_0_05"])
        self.assertTrue(out["paired"]["observed_significance"]["significant_at_0_05"])
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertIn("coverage incomplete", out["note"])

    def test_incomplete_repetition_is_blocked(self):
        base = [trigger_row("partial", True, triggered=True, run_number=n)
                for n in range(1, 3)]
        abl = [trigger_row("partial", True, triggered=False,
                           complete=(n == 1), run_number=n)
               for n in range(1, 3)]
        out = self._compare(base_rows=base, abl_rows=abl)
        self.assertEqual(out["paired"]["blocked"][0]["reason"],
                         "ablation_observations_incomplete")
        self.assertEqual(out["summary"]["comparable"], 0)
        self.assertEqual(out["evidence_class"], "indeterminate")

    def test_mismatched_declared_repetition_sets_are_blocked(self):
        base = [trigger_row("mismatched", True, triggered=True, run_number=n)
                for n in range(1, 3)]
        abl = [trigger_row("mismatched", True, triggered=False)]
        out = self._compare(base_rows=base, abl_rows=abl,
                            abl_runs_per_query=1)
        self.assertEqual(out["paired"]["blocked"][0]["reason"],
                         "repetition_count_mismatch")
        self.assertEqual(out["evidence_class"], "indeterminate")

    def test_manifest_declared_component_target_is_authoritative(self):
        wrong = {
            **ABLATION_PROVENANCE,
            "components": [{**ABLATION_PROVENANCE["components"][0],
                            "target": {"field": "name"}}],
        }
        out = self._compare(provenance=wrong)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertTrue(any("manifest-declared treatment" in reason
                            for reason in out["provenance"]["reasons"]))

    def test_invalid_skill_ablation_can_never_confirm_behavioral_causality(self):
        invalid_manifest = {
            **TRIGGER_MANIFEST,
            "ablations": [{**TRIGGER_MANIFEST["ablations"][0], "invalid_skill": True}],
        }
        invalid_provenance = {**ABLATION_PROVENANCE, "mode": "invalid_skill"}
        baseline = trigger_report(self._baseline_rows())
        ablation = trigger_report(self._ablation_rows(), ablation="drop-description",
                                  provenance=invalid_provenance, tree_hash=EDIT_HASH)
        identity = sb.trigger_manifest_identity(invalid_manifest)
        baseline["manifest_identity"] = identity
        ablation["manifest_identity"] = identity
        out = sb.build_trigger_comparison(baseline, ablation)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertIn("invalid-skill experiment", out["note"])

    def test_protocol_drift_is_indeterminate_even_when_rows_regress(self):
        baseline = trigger_report(self._baseline_rows())
        ablation = trigger_report(self._ablation_rows(), ablation="drop-description",
                                  provenance=ABLATION_PROVENANCE, tree_hash=EDIT_HASH)
        ablation["protocol"] = {**ablation["protocol"], "timeout_seconds": 31}
        ablation["protocol_sha256"] = sb.canonical_json_sha256(ablation["protocol"])
        for row in ablation["results"]:
            row["protocol_sha256"] = ablation["protocol_sha256"]
        out = sb.build_trigger_comparison(baseline, ablation)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertTrue(any("experimental protocols" in reason
                            for reason in out["provenance"]["reasons"]))

    def test_observed_isolation_drift_blocks_the_cell(self):
        base = [trigger_row("isolation", True, triggered=True, run_number=n)
                for n in range(1, 3)]
        abl = [trigger_row("isolation", True, triggered=False, run_number=n)
               for n in range(1, 3)]
        for row in base:
            row["protocol_observation"] = {"config_isolated": True}
        for row in abl:
            row["protocol_observation"] = {"config_isolated": False}
        out = self._compare(base_rows=base, abl_rows=abl)
        self.assertEqual(out["paired"]["blocked"][0]["reason"],
                         "protocol_observation_unsafe")
        self.assertEqual(out["evidence_class"], "indeterminate")

    def test_matching_unsafe_isolation_cannot_confirm(self):
        base = self._baseline_rows()
        abl = self._ablation_rows()
        unsafe = {"config_isolated": False,
                  "config_isolation_warning": "personal config may influence this measurement"}
        for row in base + abl:
            row["protocol_observation"] = unsafe
        out = self._compare(base_rows=base, abl_rows=abl)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertTrue(out["paired"]["blocked"])
        self.assertTrue(all(item["reason"] == "protocol_observation_unsafe"
                            for item in out["paired"]["blocked"]))

    def test_negative_polarity_overtriggering_is_a_causal_regression(self):
        queries = [f"negative {n}" for n in range(1, 7)]
        baseline = [trigger_row(q, False, triggered=False, run_number=n)
                    for q in queries for n in range(1, 3)]
        ablation = [trigger_row(q, False, triggered=True, run_number=n)
                    for q in queries for n in range(1, 3)]
        out = self._compare(base_rows=baseline, abl_rows=ablation)
        self.assertEqual(out["evidence_class"], "confirmed_causal")
        self.assertTrue(all(row["should_trigger"] is False
                            for row in out["regressed_queries"]))

    def test_query_id_definition_mismatch_is_blocked(self):
        base = [trigger_row("baseline text", True, triggered=True,
                            query_id="shared", run_number=n) for n in range(1, 3)]
        abl = [trigger_row("different text", True, triggered=False,
                           query_id="shared", run_number=n) for n in range(1, 3)]
        out = self._compare(base_rows=base, abl_rows=abl)
        self.assertEqual(out["paired"]["blocked"][0]["reason"],
                         "query_definition_mismatch")
        self.assertEqual(out["evidence_class"], "indeterminate")

    def test_real_matrix_reports_feed_the_comparer_without_provenance_drift(self):
        baseline = tm.run_matrix(
            DEMO_MANIFEST, demo_trigger_rows(), agents=["stub"], models=["offline"],
            runs_per_query=1, timeout=30, workers=1)
        ablation = tm.run_matrix(
            DEMO_MANIFEST, demo_trigger_rows(), agents=["stub"], models=["offline"],
            runs_per_query=1, timeout=30, workers=1, ablation="weaker-description")
        out = sb.build_trigger_comparison(baseline, ablation)
        self.assertTrue(out["provenance"]["verified"])
        self.assertEqual(out["paired"]["blocked"], [])
        self.assertEqual(ablation["skill_tree_hash"],
                         ablation["provenance"]["skill_hash"])

    def test_a_baseline_that_contradicts_itself_is_rejected_by_its_guard(self):
        def baseline(rows=None, **report):
            return trigger_report(
                self._baseline_rows() if rows is None else rows, **report)

        def changed(mutate, *, rehash_protocol=False):
            def make():
                report = baseline()
                mutate(report)
                if rehash_protocol:
                    report["protocol_sha256"] = sb.canonical_json_sha256(report["protocol"])
                    for row in report["results"]:
                        row["protocol_sha256"] = report["protocol_sha256"]
                return report
            return make

        def omit_identity_module(report):
            identity = report["protocol"]["harness_identity"]
            self.assertIn("trigger_reporting.py", identity["modules"])
            identity["modules"].pop("trigger_reporting.py")
            identity["identity_sha256"] = sb.canonical_json_sha256(
                {key: value for key, value in identity.items() if key != "identity_sha256"})

        rejected = {
            # label: (the baseline report, the guard's message)
            "a declared repetition is missing": (
                lambda: baseline([trigger_row("short", True, triggered=True, run_number=1)]),
                "--baseline has incomplete repetition identities for (stub, None, short): expected [1, 2], got [1]"),
            "a declared cell has no results": (
                changed(lambda r: r.update(results=[row for row in r["results"]
                                                    if row["query_id"] != self.QUERIES[0]])),
                "--baseline has incomplete repetition identities for (stub, None, query 1): expected [1, 2], got []"),
            "a repetition is recorded twice": (
                lambda: baseline([trigger_row("duplicate", True, triggered=True, run_number=n)
                                  for n in (1, 1, 2)]),
                "--baseline duplicates repetition 1 for (stub, None, duplicate)"),
            "the design omits a result cell": (
                changed(lambda r: r.update(design=r["design"][1:])),
                "--baseline results row 1 is not present in the declared design"),
            "the protocol declares other repetitions": (
                changed(lambda r: r["protocol"].update(runs_per_query=999), rehash_protocol=True),
                "--baseline protocol runs_per_query disagrees with its report"),
            "the protocol ran an agent the design never declared": (
                changed(lambda r: r["protocol"]["adapters"][0].update(
                    agent="other", trace_dialect="other"), rehash_protocol=True),
                "--baseline protocol agent/model design disagrees with its report"),
            "the harness identity omits a module": (
                changed(omit_identity_module, rehash_protocol=True),
                "harness_identity must identify exactly"),
            "cosmetic aliases of one query": (
                lambda: baseline([trigger_row("identical prompt" + " " * n, True, triggered=True,
                                              query_id=f"q{n}") for n in range(6)],
                                 runs_per_query=1),
                "--baseline design canonical query aliases must share one query ID, polarity, and scope"),
            "a row records another tree": (
                changed(lambda r: r["results"][0].update(skill_tree_hash="sha256:other")),
                "--baseline results row 1: skill_tree_hash disagrees with its report"),
            "a row's pass contradicts its observation": (
                changed(lambda r: r["results"][0].update({"pass": True, "triggered": False})),
                "--baseline results row 1: persisted triggered flag disagrees with the typed observation"),
            "the baseline declares an ablation": (
                lambda: baseline(ablation="drop-description", provenance=ABLATION_PROVENANCE),
                "--baseline must be an unablated trigger run (it declares an ablation)"),
            "an answer report": (
                changed(lambda r: r.update(evidence_class="answer")),
                "--baseline is not a skill-trigger-matrix report"),
        }
        for label, (make_baseline, message) in rejected.items():
            with self.subTest(label):
                ablation = trigger_report(
                    self._ablation_rows(), ablation="drop-description",
                    provenance=ABLATION_PROVENANCE, tree_hash=EDIT_HASH)
                compare = functools.partial(sb.build_trigger_comparison, make_baseline(), ablation)
                assert_dies(self, compare, message)

    def test_reports_from_the_direct_script_entry_point_pair(self):
        # examples/demo-skill/README.md runs `python3 ../../run_trigger_matrix.py`,
        # where the adapters are defined in `__main__`.
        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for arm, extra in (("baseline", []), ("ablation", ["--ablation", "weaker-description"])):
                paths[arm] = Path(td) / f"{arm}.json"
                subprocess.run(
                    [sys.executable, str(ROOT / "run_trigger_matrix.py"),
                     "evals/shared-benchmark.json", "--agent", "stub", "--runs-per-query", "1",
                     *extra, "--out", str(paths[arm])],
                    cwd=DEMO_MANIFEST.parents[1], check=True, capture_output=True)
            adapter = json.loads(paths["baseline"].read_text(encoding="utf-8"))["protocol"]["adapters"][0]
            code, stdout, stderr = run_cli("trigger-compare", "--baseline", paths["baseline"],
                                           "--ablation", paths["ablation"])
        self.assertEqual(code, 0, stderr)
        self.assertTrue(json.loads(stdout)["provenance"]["verified"])
        self.assertEqual(adapter["adapter"], "run_trigger_matrix.StubAdapter")




class CatalogAttributionTests(unittest.TestCase):
    REVIEW = "skills/z-reviewer/SKILL.md"
    RELEASE = "skills/a-release/SKILL.md"
    OTHER = "skills/m-extra/SKILL.md"

    def catalog(self, directory):
        from helpers import make_eval_repo
        path = make_eval_repo(Path(directory), skill_name="catalog-report-label",
                             skill_paths=[self.REVIEW, self.OTHER, self.RELEASE], cases=[],
                             ablations=[{"id": "drop-description", "class": "discovery",
                                         "mechanism": "frontmatter_field",
                                         "target": {"field": "when_to_use", "skill_root": self.REVIEW}}])
        for identity, name, description in (
            (self.REVIEW, "review-name", "Review change"),
            (self.RELEASE, "release-name", "Release deploy"),
            (self.OTHER, "extra-name", "Travel planning"),
        ):
            (path.parent.parent / identity).write_text(skill_markdown(name, description).replace("description:", "when_to_use: Extra discovery hint\ndescription:"), encoding="utf-8")
        return path

    def run_rows(self, manifest, rows, **kwargs):
        return tm.run_matrix(manifest, rows, agents=["stub"], models=[None],
                             runs_per_query=2, timeout=2, workers=1, **kwargs)

    def test_catalog_verdicts_keep_full_mount_and_hash_with_reversed_declarations(self):
        specs = [
            ("review change", [self.REVIEW], [self.RELEASE], True, True),
            ("release deploy", [self.REVIEW], [self.RELEASE], False, False),
            ("review change release deploy", [self.REVIEW], [self.RELEASE], True, False),
            ("weather forecast", [self.REVIEW], [self.RELEASE], False, False),
            ("review change travel planning", [self.REVIEW], [self.RELEASE], True, True),
            ("review change missing extra", [self.REVIEW, self.OTHER], [self.RELEASE], True, False),
            ("travel planning negative", [], [self.RELEASE], False, True),
            ("release deploy negative", [], [self.RELEASE], True, False),
        ]
        rows = [{"query_id": f"case-{i}", "query": query, "should_trigger": bool(expected),
                 "expected_skills": expected, "forbidden_skills": forbidden}
                for i, (query, expected, forbidden, _, _) in enumerate(specs)]
        with tempfile.TemporaryDirectory() as td:
            manifest = self.catalog(td)
            report = self.run_rows(manifest, rows)
            legacy = self.run_rows(manifest, [{"query": "travel planning", "should_trigger": True}])
        self.assertEqual(report["skill_tree_hash"], legacy["skill_tree_hash"])
        self.assertEqual(report["summary"]["measurement_status"], "complete")
        self.assertEqual(report["summary"]["passed"], 6)
        self.assertTrue(legacy["results"][0]["pass"])
        self.assertNotIn("expected_skills", legacy["results"][0])
        for row in report["results"]:
            _, expected, forbidden, triggered, passed = specs[int(row["query_id"].split("-")[1])]
            self.assertEqual((row["triggered"], row["pass"]), (triggered, passed))
            self.assertEqual(row["expected_skills"], sorted(expected))
            self.assertEqual(set(row["skill_detections"]), set(expected + forbidden))
            self.assertEqual(row["skill_tree_hash"], report["skill_tree_hash"])
        parsed = sb._trigger_report_rows(report, "catalog")
        self.assertEqual(len(parsed.observations), 16)

    def test_harness_failure_retains_scope_and_null_verdicts(self):
        class Broken(tm.StubAdapter):
            def invoke(self, *args):
                raise RuntimeError("fixture invoke failed")
        with tempfile.TemporaryDirectory() as td, mock.patch.dict(tm.ADAPTERS, {"stub": Broken}):
            report = self.run_rows(self.catalog(td), [{"query": "review change", "should_trigger": True,
                                                      "expected_skills": [self.REVIEW]}])
        row = report["results"][0]
        self.assertEqual(row["expected_skills"], [self.REVIEW])
        self.assertEqual(row["skill_detections"], {self.REVIEW: []})
        self.assertIsNone(row["pass"])
        self.assertIsNone(row["triggered"])
        self.assertEqual(report["summary"]["measurement_status"], "incomplete")
        self.assertEqual(TriggerObservation.from_row(row).constraints.expected, frozenset({self.REVIEW}))

    def test_selected_duplicate_exposed_name_rejected_before_any_invocation(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = self.catalog(td)
            (manifest.parent.parent / self.OTHER).write_text(skill_markdown('" review-name "', "Travel planning"))
            for agent in ("codex", "vibe"):
                with self.subTest(agent=agent), \
                     mock.patch.object(tm.ADAPTERS[agent], "invoke", side_effect=AssertionError("must not invoke")), \
                     self.assertRaisesRegex(ValueError, "duplicate exposed skill name"):
                    tm.run_matrix(manifest, [{"query": "review change", "should_trigger": True,
                                              "expected_skills": [self.REVIEW]}], agents=[agent], models=[None],
                                  runs_per_query=1, timeout=1, workers=1)
            report = self.run_rows(manifest, [{"query": "review change", "should_trigger": True,
                                              "expected_skills": [self.REVIEW]}])
        self.assertTrue(report["results"][0]["pass"])

    def test_provider_specific_names_are_attributed_to_selected_root(self):
        from trigger_contracts import parse_skill_constraints
        with tempfile.TemporaryDirectory() as td:
            manifest = self.catalog(td)
            tree = Path(sb.build_canonical_skill_tree(manifest.parent.parent, tm.load_manifest(manifest), Path(td) / "tree"))
            root_keys = {self.REVIEW: "z-reviewer", self.RELEASE: "a-release", self.OTHER: "m-extra"}
            constraints = parse_skill_constraints({"should_trigger": True, "expected_skills": [self.REVIEW],
                                                   "forbidden_skills": [self.RELEASE]})
            for adapter, invoked, should_pass in (
                (tm.ClaudeAdapter(), "z-reviewer", True),
                (tm.ClaudeAdapter(), "review-name", True),
                (tm.CodexAdapter(), "review-name", True),
                (tm.CodexAdapter(), "z-reviewer", False),
                (tm.VibeAdapter(), "review-name", True),
                (tm.VibeAdapter(), "z-reviewer", True),
            ):
                if adapter.name == "claude":
                    events = [
                        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "s1", "name": "Skill", "input": {"skill": invoked}}]}},
                        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "s1", "content": "loaded"}]}},
                    ]
                    invocation = completed_invocation("\n".join(map(json.dumps, events)))
                elif adapter.name == "vibe":
                    events = [
                        {"role": "assistant", "tool_calls": [{"id": "s1", "function": {"name": "skill", "arguments": json.dumps({"name": invoked})}}]},
                        {"role": "tool", "tool_call_id": "s1", "content": "loaded"},
                        {"role": "assistant", "content": "done"},
                    ]
                    invocation = completed_invocation("\n".join(map(json.dumps, events)))
                else:
                    text = json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
                        {"text": f"<skill>\n<name>{invoked}</name>\n</skill>"}]}})
                    invocation = completed_invocation("").with_provider_payload(sb.CodexRollout("found", text=text))
                def invoke(query, model, workspace, timeout, adapter=adapter, invocation=invocation):
                    self.assertEqual(query, "review change")
                    if isinstance(adapter, tm.CodexAdapter):
                        shutil.rmtree(adapter._codex_home(workspace))
                    return invocation

                with self.subTest(agent=adapter.name, invoked=invoked), \
                     mock.patch.object(adapter, "invoke", side_effect=invoke):
                    row = tm.observe_cell_query(adapter, tree, "review change", True, None, 1,
                                            metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
                                            constraints=constraints, root_keys=root_keys).as_row()
                self.assertEqual(row["pass"], should_pass)
                self.assertEqual(row["activated_skills"], [self.REVIEW] if should_pass else [])

    def test_directory_and_sibling_prefix_paths_require_the_selected_root(self):
        selected = Path("/tmp/catalog/review")
        for path, expected in (("/tmp/catalog/review/SKILL.md", True),
                               ("/tmp/catalog/review/references/check.md", True),
                               ("/tmp/catalog/review-extra/SKILL.md", False),
                               ("/tmp/catalog/other/SKILL.md", False)):
            with self.subTest(path=path):
                records = [{"type": "file_read", "path": path, "status": "completed"}]
                self.assertEqual(sb.detect_trigger_records(records, [selected]).triggered, expected)
                rollout = json.dumps({"type": "response_item", "payload": {"type": "message", "role": "user", "content": [
                    {"text": f"<skill>\n<name>review</name>\n<path>{path}</path>\n</skill>"}]}})
                self.assertEqual(bool(sb.codex_rollout_skill_loads(rollout, ["review"], [selected])), expected)
        for status in ("in_progress", "failed"):
            self.assertFalse(sb.detect_trigger_records([
                {"type": "file_read", "path": "/tmp/catalog/review/SKILL.md", "status": status}], [selected]).triggered)

    def test_codex_failed_commands_do_not_credit_mounted_path_reads(self):
        selected = Path("/tmp/catalog/review/SKILL.md")
        for exit_code, expected in ((0, True), (1, False), (-1, False), (True, False), (False, False)):
            with self.subTest(exit_code=exit_code):
                records = [{"type": "item.completed", "item": {
                    "type": "command_execution", "status": "completed",
                    "exit_code": exit_code, "command": f"cat {selected}",
                    "aggregated_output": "loaded" if expected else "Permission denied",
                }}]
                detection = sb.detect_trigger_records(records, [selected], source="codex")
                self.assertEqual(detection.triggered, expected)
                self.assertEqual(detection.legacy_evidence, [f"cat {selected}"] if expected else [])

    def test_completed_file_reads_without_exit_code_remain_eligible(self):
        selected = Path("/tmp/catalog/review/SKILL.md")
        for exit_evidence in ({}, {"exit_code": None}):
            for status, expected in (("completed", True), ("failed", False), (False, False)):
                with self.subTest(status=status, exit_evidence=exit_evidence):
                    records = [{"type": "file_read", "path": str(selected), "status": status,
                                **exit_evidence}]
                    detection = sb.detect_trigger_records(records, [selected])
                    self.assertEqual(detection.triggered, expected)
                    self.assertEqual(detection.legacy_evidence, [str(selected)] if expected else [])

    def test_scope_changes_block_comparison_and_missing_repetitions_are_rejected(self):
        rows = [{"query_id": "review", "query": "review change", "should_trigger": True,
                 "expected_skills": [self.REVIEW]}]
        with tempfile.TemporaryDirectory() as td:
            manifest = self.catalog(td)
            base = self.run_rows(manifest, rows)
            changed = self.run_rows(manifest, [{**rows[0], "forbidden_skills": [self.RELEASE]}], ablation="drop-description")
            matching = self.run_rows(manifest, rows, ablation="drop-description")
        comparison = sb.build_trigger_comparison(base, matching)
        self.assertEqual(comparison["summary"]["comparable"], 1)
        self.assertEqual(comparison["paired"]["query_units"][0]["expected_skills"], [self.REVIEW])
        self.assertEqual(comparison["paired"]["query_units"][0]["pass_delta"], 0.0)
        blocked = sb.build_trigger_comparison(base, changed)
        self.assertEqual(blocked["paired"]["blocked"][0]["reason"], "query_definition_mismatch")
        missing = json.loads(json.dumps(base))
        missing["results"].pop()
        with self.assertRaises(SystemExit):
            sb.build_trigger_comparison(missing, matching)
        forged = json.loads(json.dumps(base))
        forged["design"][0]["expected_skills"] = [self.RELEASE]
        with self.assertRaises(SystemExit):
            sb.build_trigger_comparison(forged, matching)

    def test_loader_parity_dataset_scope_and_prompt_independence(self):
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as td:
            path = self.catalog(td)
            manifest = tm.load_manifest(path)
            manifest["datasets"] = {"targets": [{"id": "one", "target": self.REVIEW}]}
            manifest["cases"] = [{"id": "scoped", "kind": "trigger", "split": "tune", "template": "targets",
                                  "prompt": "review change", "should_trigger": True,
                                  "expected_skills": ["{target}"]}]
            path.write_text(json.dumps(manifest))
            rows = tr.eval_rows_from_args(SimpleNamespace(eval_set=None, split="tune"), path)
            self.assertEqual(rows[0]["expected_skills"], [self.REVIEW])
            self.assertEqual(rows[0]["forbidden_skills"], [])
            eval_set = Path(td) / "eval.json"
            eval_set.write_text(json.dumps(rows))
            explicit = tr.eval_rows_from_args(SimpleNamespace(eval_set=str(eval_set), split="tune"), path)
            self.assertEqual(explicit, rows)
            with self.assertRaisesRegex(SystemExit, "canonical query aliases"):
                tm.validate_trigger_rows([rows[0], {**rows[0], "query_id": "different", "expected_skills": [self.RELEASE]}],
                                         "fixture", frozenset([self.REVIEW, self.RELEASE]))
            malformed = deepcopy(manifest)
            malformed["cases"][0]["expected_skills"] = ["missing"]
            path.write_text(json.dumps(malformed))
            for loader in (sb.load_manifest_source, sb.validate_manifest, tm.load_manifest):
                with self.subTest(loader=loader.__name__), self.assertRaises(SystemExit):
                    loader(path)

    def test_legacy_generated_ids_are_unchanged_and_scope_changes_new_ids(self):
        query = "review change"
        old_id = "query-653729a748d6a08f2bdf340ab1df29140ef5869777772561e7466345da82f6fc"
        legacy = tm.validate_trigger_rows([{"query": query, "should_trigger": True}], "fixture")[0]
        self.assertEqual(legacy["query_id"], old_id)
        declared = frozenset([self.REVIEW, self.RELEASE])
        scoped = tm.validate_trigger_rows([{**legacy, "query_id": "", "expected_skills": [self.REVIEW]}], "fixture", declared)[0]
        forbidden = tm.validate_trigger_rows([{**legacy, "query_id": "", "expected_skills": [self.REVIEW], "forbidden_skills": [self.RELEASE]}], "fixture", declared)[0]
        self.assertNotEqual(scoped["query_id"], old_id)
        self.assertNotEqual(scoped["query_id"], forbidden["query_id"])

    def test_pi_entrypoint_persists_the_same_scope_and_per_skill_evidence(self):
        from trigger_contracts import parse_skill_constraints
        with tempfile.TemporaryDirectory() as td:
            manifest = self.catalog(td)
            rows = [{"query_id": "pi-scoped", "query": "review change", "should_trigger": True,
                     "expected_skills": [self.REVIEW], "forbidden_skills": [self.RELEASE]}]
            eval_set = Path(td) / "rows.json"
            eval_set.write_text(json.dumps(rows))
            output = Path(td) / "report.json"

            def invoke(plan):
                self.assertEqual(plan.argv[-1], "review change")
                skills = Path(plan.argv[plan.argv.index("--skill") + 1])
                self.assertFalse(skills.resolve().is_relative_to(plan.cwd.resolve()))
                self.assertEqual(sorted(path.name for path in skills.iterdir()),
                                 ["a-release", "m-extra", "z-reviewer"])
                fixture = (ROOT / "tests/fixtures/pi/lifecycle-success.jsonl").read_text()
                return completed_invocation(fixture.replace("/tmp/pi-config/skills/demo/SKILL.md",
                                                             str(skills / "z-reviewer/SKILL.md")))

            argv = ["skill-pi-trigger-eval", str(manifest), "--eval-set", str(eval_set),
                    "--runs-per-query", "1", "--workers", "1", "--out", str(output)]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(tm.PiAdapter, "_run_argv", staticmethod(invoke)), \
                 mock.patch("builtins.print"):
                self.assertEqual(tr.main(), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["design"][0]["expected_skills"], [self.REVIEW])
            self.assertEqual(report["results"][0]["activated_skills"], [self.REVIEW])
            self.assertTrue(report["results"][0]["pass"])
            self.assertEqual(sb._trigger_report_rows(report, "pi-scoped").queries["pi-scoped"][2],
                             parse_skill_constraints(rows[0]))

    def test_directory_without_skill_md_cannot_credit_another_root(self):
        from trigger_contracts import parse_skill_constraints
        with tempfile.TemporaryDirectory() as td:
            tree = Path(td) / "tree"
            empty = tree / "review"
            empty.mkdir(parents=True)
            (empty / "instructions.txt").write_text("Review instructions")
            sibling = tree / "review-extra"
            sibling.mkdir()
            (sibling / "SKILL.md").write_text(skill_markdown("review-extra", "Release deploy"))
            row = tm.observe_cell_query(
                tm.StubAdapter(), tree, "release deploy", True, None, 1,
                metadata={"skill_tree_hash": sb.skill_tree_hash(tree)},
                constraints=parse_skill_constraints({"should_trigger": True, "expected_skills": ["skills/review"]}),
                root_keys={"skills/review": "review", "skills/review-extra/SKILL.md": "review-extra"},
            ).as_row()
        self.assertEqual(row["missing_expected_skills"], ["skills/review"])
        self.assertFalse(row["pass"])
        self.assertFalse(row["triggered"])


if __name__ == "__main__":
    unittest.main()
