import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import make_eval_repo, run_cli, write_run

import skill_benchmark as sb

ROOT = Path(__file__).resolve().parents[1]
PLANTED = "planted-private-value-b8bbf"
EXTRA = "explicit-private-value-4b394"
EXPECTED = "total: 13"
WRONG = "total: 12"


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rollout(records):
    return "".join(json.dumps(record) + "\n" for record in records)


def record(record_type, **payload):
    return {"type": record_type, "payload": payload}


def message(role, text, **fields):
    item_type = "input_text" if role == "user" else "output_text"
    return record("response_item", type="message", role=role,
                  content=[{"type": item_type, "text": text}], **fields)


class TraceCaseTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.fixtures = self.root / "replay"
        self.drafts = self.root / "drafts"
        self.trace = self.root / "rollout.jsonl"
        self.secrets = self.root / "redact.txt"
        self.review = self.root / "review.json"
        self.datasets = {"existing": "dataset.jsonl"}
        self.manifest = make_eval_repo(
            self.root, skill_name="arithmetic", version=2, cases=[],
            extra={"dataset_files": self.datasets},
        )
        self.skill = self.manifest.parent.parent / "skills" / "arithmetic" / "SKILL.md"
        (self.manifest.parent / "dataset.jsonl").write_text('{"id":"existing"}\n', encoding="utf-8")
        json_file(self.fixtures / "sample.json", {"api_key": PLANTED, "value": 4})
        (self.fixtures / "unlisted.txt").write_text("not a dependency", encoding="utf-8")
        self.secrets.write_text(f"{EXTRA}\n", encoding="utf-8")
        self.trace.write_text(rollout([
            record("session_meta", id="example-thread", cwd=f"/private/{EXTRA}"),
            record("event_msg", type="task_started"),
            message("user", "Prior turn"),
            record("response_item", type="function_call", name="read_file",
                   arguments=json.dumps({"path": "prior-turn.txt"})),
            message("assistant", "Prior output", phase="final_answer"),
            record("event_msg", type="task_started"),
            message("user", f"Add the entries in numbers.txt. API key: {PLANTED}. Ref {EXTRA}"),
            message("user", f"<skill>\nInjected private instructions {PLANTED}\n</skill>"),
            record("response_item", type="function_call", name="exec_command",
                   arguments=json.dumps({"cmd": "cat numbers.txt"})),
            record("response_item", type="function_call", name="exec_command",
                   arguments=json.dumps({"cmd": "cat ../outside.txt; echo nope"})),
            record("response_item", type="function_call_output", output=f"private raw tool log {PLANTED}"),
            record("event_msg", type="task_complete", last_agent_message=f"{WRONG}\nRef {EXTRA}"),
        ]), encoding="utf-8")

    def cli(self, *args, failure=None):
        code, stdout, stderr = run_cli(*args)
        self.assertEqual(code, 1 if failure else 0, stdout + stderr)
        if failure:
            self.assertIn(failure, stderr)
        return stdout

    def import_args(self):
        return ("import-case", "--trace", self.trace, "--out-dir", self.drafts,
                "--fixture-root", self.fixtures, "--require-file", "sample.json",
                "--redact-file", self.secrets)

    def draft(self):
        return next(self.drafts.glob("session-*"))

    def approve(self, expected=EXPECTED, approved=True, unresolved_reviewed=True):
        json_file(self.review, {
            "approved": approved,
            "unresolved_inputs_reviewed": unresolved_reviewed,
            "expected_behavior": [f"Return {expected}."],
            "assertions": [{"type": "contains", "value": expected}],
            "fixtures": {p.relative_to(self.draft() / "fixtures").as_posix():
                         hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (self.draft() / "fixtures").rglob("*") if p.is_file()},
        })

    def promote(self, failure=None):
        return self.cli("promote-case", "--manifest", self.manifest, "--draft", self.draft(),
                        "--review", self.review, failure=failure)

    def add_input(self):
        (self.draft() / "fixtures" / "numbers.txt").write_text("4\n9\n", encoding="utf-8")

    def test_import_review_promote_prepare_and_grade(self):
        original = self.manifest.read_bytes()
        imported = self.cli(*self.import_args())
        self.assertIn("Missing required inputs: numbers.txt", imported)
        draft = self.draft()
        data = json.loads((draft / "draft.json").read_bytes())
        self.assertEqual(data["available_inputs"], ["sample.json"])
        self.assertEqual(data["missing_inputs"], {"numbers.txt": "file unavailable"})
        self.assertEqual(data["provenance"]["trace_sha256"], hashlib.sha256(self.trace.read_bytes()).hexdigest())
        self.assertIn("Add the entries in numbers.txt", data["prompt"])
        self.assertNotIn("Prior turn", data["prompt"])
        self.assertEqual(json.loads((draft / "fixtures" / "sample.json").read_bytes())["api_key"], "[REDACTED]")
        for path in draft.rglob("*"):
            if path.is_file():
                content = path.read_bytes()
                for secret in (PLANTED, EXTRA, "Injected private instructions", "private raw tool log", "example-thread"):
                    self.assertNotIn(secret.encode(), content)
        self.assertEqual(self.manifest.read_bytes(), original)

        self.promote(failure="missing required inputs: numbers.txt")
        self.assertEqual(self.manifest.read_bytes(), original)
        self.add_input()
        self.assertIn("Reused draft", self.cli(*self.import_args()))
        self.assertEqual(list(self.drafts.glob("session-*")), [draft])
        self.assertEqual((draft / "fixtures" / "numbers.txt").read_bytes(), b"4\n9\n")
        self.approve()
        self.promote()
        saved = self.manifest.read_bytes()
        self.assertIn("Reused case", self.promote())
        self.assertEqual(self.manifest.read_bytes(), saved)
        promoted = json.loads(saved)
        self.assertEqual(promoted["dataset_files"], self.datasets)
        case, = promoted["cases"]
        self.assertEqual(case["id"], draft.name)
        self.assertEqual(case["eval_intent"], "regression")
        self.assertEqual(case["source"], "production")
        self.cli("validate", self.manifest, "--strict-leakage", "--leakage-min-chars", 1)
        tasks = self.cli("prepare", self.manifest)
        for local_only in (EXPECTED, WRONG, PLANTED, EXTRA, draft.as_posix(), '"answer_key"'):
            self.assertNotIn(local_only, tasks)
        prepared = [json.loads(line) for line in tasks.splitlines()]
        self.assertEqual(len(prepared), 2)
        for task in prepared:
            self.assertEqual({Path(path).name for path in task["input_files"]}, {"numbers.txt", "sample.json"})
        runs = self.root / "runs"
        recorded = (draft / "recorded-output.txt").read_text(encoding="utf-8")
        output = write_run(runs / case["id"] / "with_skill", recorded) / "output.md"
        self.assertIn(WRONG, output.read_text(encoding="utf-8"))
        bad, = json.loads(self.cli("grade", self.manifest, "--runs", runs, "--variant", "with_skill"))["results"]
        self.assertEqual(bad["outcome_passed"], 0)
        self.assertIs(bad["assertions"][0]["passed"], False)
        output.write_text(EXPECTED, encoding="utf-8")
        fixed, = json.loads(self.cli("grade", self.manifest, "--runs", runs, "--variant", "with_skill"))["results"]
        self.assertEqual(fixed["outcome_passed"], 1)
        self.assertIs(fixed["assertions"][0]["passed"], True)

    def test_review_must_approve_exact_fixtures_and_pass_leakage_and_validation(self):
        original = self.manifest.read_bytes()
        self.cli(*self.import_args())
        self.add_input()
        self.approve(approved=False)
        self.promote(failure="review must set approved to true")
        self.approve()
        (self.draft() / "fixtures" / "numbers.txt").write_text("changed", encoding="utf-8")
        self.promote(failure="review.fixtures must match")
        self.approve(expected="numbers.txt")
        self.promote(failure="appears in the prompt (leakage)")
        self.approve()
        malformed = json.loads(self.review.read_bytes())
        malformed["assertions"] = [{"type": "regex", "value": "["}]
        json_file(self.review, malformed)
        self.promote(failure="regex")
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(list(self.manifest.parent.rglob("numbers.txt")), [])

    def test_repeat_promotion_checks_the_installed_fixture_bytes(self):
        self.cli(*self.import_args())
        self.add_input()
        self.approve()
        self.promote()
        original = self.manifest.read_bytes()
        case, = json.loads(original)["cases"]
        installed = next(self.manifest.parent / path for path in case["files"]
                         if Path(path).name == "numbers.txt")
        installed.write_text("altered after approval", encoding="utf-8")
        self.promote(failure="existing regression fixture snapshot differs")
        self.assertEqual(self.manifest.read_bytes(), original)

    def test_fixture_paths_cannot_escape_the_selected_root(self):
        self.cli(*self.import_args(), "--require-file", "../outside.txt",
                 failure="fixture paths must be portable relative paths")
        (self.fixtures / "external.txt").symlink_to(self.trace)
        self.cli(*self.import_args(), "--require-file", "external.txt",
                 failure="fixture paths must not contain symlinks")
        self.assertFalse(self.drafts.exists())

    def test_review_inside_mounted_skill_blocks_promotion(self):
        self.cli(*self.import_args())
        self.add_input()
        self.review = self.skill.parent / "review.json"
        self.approve()
        self.promote(failure="the review would be copied into the runner's skill directory")
        self.assertEqual(json.loads(self.manifest.read_bytes())["cases"], [])

    def write_rollout(self, request, *events):
        self.trace.write_text(rollout([
            record("session_meta", id="one-thread", cwd="/session/project"),
            record("event_msg", type="task_started"),
            message("user", request),
            *events,
        ]), encoding="utf-8")

    def test_codex_exec_wrapper_uses_recorded_workdir(self):
        (self.fixtures / "totals.txt").write_text("other file", encoding="utf-8")
        args = json.dumps({"cmd": "cat totals.txt", "workdir": "/session/project/team"})
        self.write_rollout(
            "Read totals.txt.",
            record("response_item", type="custom_tool_call", name="exec",
                   input=f"const r = await tools.exec_command({args}); text(r.output);"),
            message("assistant", WRONG, phase="final_answer"),
        )
        self.assertIn("Missing required inputs: team/totals.txt", self.cli(*self.import_args()))
        self.assertFalse((self.draft() / "fixtures" / "totals.txt").exists())
        self.promote(failure="missing required inputs: team/totals.txt")
        team = self.draft() / "fixtures" / "team"
        team.mkdir()
        (team / "totals.txt").write_text("4\n9\n", encoding="utf-8")
        self.approve()
        self.promote()
        row, = json.loads(self.manifest.read_bytes())["cases"]
        installed = next(self.manifest.parent / file for file in row["files"] if file.endswith("totals.txt"))
        self.assertEqual(installed.read_text(encoding="utf-8"), "4\n9\n")

    def test_prompt_file_is_required_even_without_a_tool_call(self):
        self.write_rollout("Count warnings in missing.csv. API key: planted-sensitive.csv.",
                           message("assistant", WRONG))
        self.assertIn("Missing required inputs: missing.csv", self.cli(*self.import_args()))
        draft = (self.draft() / "draft.json").read_bytes()
        self.assertNotIn(b"planted-sensitive.csv", draft)
        self.assertEqual(json.loads(draft)["missing_inputs"], {"missing.csv": "file unavailable"})
        self.promote(failure="missing required inputs: missing.csv")

    def test_opaque_input_requires_explicit_review(self):
        self.write_rollout(
            "Count the warnings.",
            record("response_item", type="custom_tool_call", name="exec",
                   input="await tools.exec_command(makeArgs());"),
            message("assistant", WRONG),
        )
        self.assertIn("Review unmapped inputs", self.cli(*self.import_args()))
        self.approve(unresolved_reviewed=False)
        self.promote(failure="resolve unmapped inputs")
        self.approve()
        self.promote()

    def test_other_shell_calls_require_explicit_review_in_both_encodings(self):
        commands = ("head -n 1 totals.txt", "ls >/dev/null && tail -n 1 totals.txt", "")
        for index, command in enumerate(commands):
            for wrapped in (False, True):
                with self.subTest(command=command, wrapped=wrapped):
                    self.drafts = self.root / f"opaque-{index}-{wrapped}"
                    args = json.dumps({"cmd": command, "workdir": "/session/project"})
                    event = (
                        record("response_item", type="custom_tool_call", name="exec",
                               input=f"const result = await tools.exec_command({args}); text(result.output);")
                        if wrapped else
                        record("response_item", type="function_call", name="exec_command", arguments=args)
                    )
                    self.write_rollout("Read the first ledger row.", event, message("assistant", WRONG))
                    imported = self.cli("import-case", "--trace", self.trace, "--out-dir", self.drafts)
                    data = json.loads((self.draft() / "draft.json").read_bytes())
                    self.assertEqual(data["required_inputs"], [])
                    self.assertEqual(data["missing_inputs"], {})
                    self.assertTrue(data["unresolved_inputs"])
                    self.assertIn("Review unmapped inputs", imported)
                    self.approve(unresolved_reviewed=False)
                    self.promote(failure="resolve unmapped inputs")

    def test_drafts_from_the_previous_importer_must_be_reimported(self):
        self.cli(*self.import_args())
        self.add_input()
        self.approve()
        path = self.draft() / "draft.json"
        data = json.loads(path.read_bytes())
        data["schema_version"] = 2
        json_file(path, data)
        failure = "import this session into a new private draft directory"
        self.cli(*self.import_args(), failure=failure)
        self.promote(failure=failure)

    def test_later_image_turn_never_reuses_previous_text_or_answer(self):
        image = record("response_item", type="message", role="user",
                       content=[{"type": "input_image", "image_url": "data:image/png;base64,example"}])
        self.write_rollout("Prior prompt.", message("assistant", WRONG),
                           record("event_msg", type="task_started"), image)
        self.cli(*self.import_args(), failure="latest Codex turn has no supported text request")
        self.assertFalse(self.drafts.exists())

    def test_interrupted_turn_needs_a_replay_of_its_own(self):
        self.write_rollout("Prior prompt.", message("assistant", "prior answer"),
                           record("event_msg", type="task_started"),
                           message("user", "Latest request."),
                           record("event_msg", type="turn_aborted"),
                           record("event_msg", type="task_complete", last_agent_message="older final"))
        self.assertIn("No recorded final answer", self.cli(*self.import_args()))
        draft = self.draft()
        self.assertFalse((draft / "recorded-output.txt").exists())
        data = json.loads((draft / "draft.json").read_bytes())
        self.assertEqual(data["prompt"], "Latest request.")
        self.approve()
        self.promote(failure="supply a redacted replay as recorded-output.txt")
        (draft / "recorded-output.txt").write_text("Recovered failed answer.", encoding="utf-8")
        self.promote()

    def test_new_trace_options_or_redaction_require_a_fresh_draft(self):
        self.cli(*self.import_args())
        draft = self.draft() / "draft.json"
        original = draft.read_bytes()
        self.assertIn("Reused draft", self.cli(*self.import_args()))
        mismatch = "trace or import options changed"
        self.cli(*self.import_args(), "--require-file", "other.txt", failure=mismatch)
        secrets = self.secrets.read_bytes()
        self.secrets.write_text("another sensitive value\n", encoding="utf-8")
        self.cli(*self.import_args(), failure=mismatch)
        self.secrets.write_bytes(secrets)
        self.trace.write_text(self.trace.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.cli(*self.import_args(), failure=mismatch)
        self.assertEqual(draft.read_bytes(), original)
        fresh = self.root / "fresh-drafts"
        self.cli(*self.import_args(), "--out-dir", fresh)
        self.assertTrue((fresh / self.draft().name / "draft.json").is_file())

    def test_duplicate_basenames_have_a_review_recovery(self):
        self.write_rollout("Compare east/data.csv and west/data.csv.", message("assistant", WRONG))
        self.cli(*self.import_args())
        self.approve()
        self.promote(failure="rename the replay copies and update required_inputs and the prompt")
        draft = self.draft() / "draft.json"
        data = json.loads(draft.read_bytes())
        data["required_inputs"] = ["east-data.csv", "west-data.csv", "sample.json"]
        data["prompt"] = "Compare east-data.csv and west-data.csv."
        json_file(draft, data)
        for name in ("east-data.csv", "west-data.csv"):
            (self.draft() / "fixtures" / name).write_text("reviewed bytes", encoding="utf-8")
        self.approve()
        self.promote()

    def test_runner_preparation_rejects_an_alias_into_excluded_files(self):
        self.cli(*self.import_args())
        self.add_input()
        self.approve()
        self.promote()
        excluded = self.skill.parent / "evals"
        excluded.mkdir()
        (excluded / "local.txt").write_text("grading data", encoding="utf-8")
        (self.skill.parent / "reference.txt").symlink_to(excluded / "local.txt")
        with self.assertRaisesRegex(sb.AblationError, "symlink into an excluded path"):
            run_cli("prepare", self.manifest)

    def test_import_existing_rollout_ignores_skill_injection(self):
        trace = ROOT / "tests" / "fixtures" / "codex" / "rollout-skill-injection.jsonl"
        self.cli("import-case", "--trace", trace, "--out-dir", self.drafts)
        data = json.loads((self.draft() / "draft.json").read_bytes())
        self.assertTrue(data["prompt"].startswith("Use $unslop on this sentence"))
        self.assertNotIn("skills_instructions", data["prompt"])
        self.assertEqual((self.draft() / "recorded-output.txt").read_text(encoding="utf-8"),
                         "Combining complementary strengths can make a major difference.")


if __name__ == "__main__":
    unittest.main()
