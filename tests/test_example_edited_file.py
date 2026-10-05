from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import load_example_module

import skill_benchmark as sb

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/edited-file-demo"
ORACLE = load_example_module("edited_file_oracle", "examples/edited-file-demo/evals/oracles/check_edit.py")


class EditedFileExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="edited-file-example-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.directory = Path(cls.tmp.name)
        cls.runs = cls.directory / "runs"
        tasks = cls.directory / "tasks.jsonl"
        benchmark = cls.directory / "benchmark.json"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
               "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
        commands = [
            ["prepare", str(DEMO / "evals/shared-benchmark.json"), "--out", str(tasks)],
            ["run-codex", "--tasks", str(tasks), "--runs", str(cls.runs), "--codex-cmd",
             shlex.join([sys.executable, "-B", str(DEMO / "stub_runner.py")])],
            ["benchmark", str(DEMO / "evals/shared-benchmark.json"), "--runs", str(cls.runs),
             "--allow-scripts", "--out", str(benchmark)],
            ["report", "--benchmark", str(benchmark), "--format", "github",
             "--out", str(cls.directory / "summary.md"), "--fail-on-failures"],
        ]
        for args in commands:
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "skill_benchmark.py"), *args],
                cwd=ROOT, env=env, text=True, capture_output=True, check=False, timeout=60,
            )
            if completed.returncode != 0:
                raise AssertionError(f"{args[0]} exited {completed.returncode}\n{completed.stdout}{completed.stderr}")
        cls.benchmark = json.loads(benchmark.read_text(encoding="utf-8"))
        cls.treatment = cls.runs / "normalize-name/with_skill"

    def copied_run(self):
        tmp = tempfile.TemporaryDirectory(prefix="edited-file-mutation-")
        self.addCleanup(tmp.cleanup)
        run = Path(tmp.name) / "run"
        shutil.copytree(self.treatment, run)
        return run

    def change_capture(self, mutate):
        run = self.copied_run()
        path = run / "workspace-changes.json"
        capture = json.loads(path.read_text(encoding="utf-8"))
        mutate(capture)
        path.write_text(json.dumps(capture), encoding="utf-8")
        sb.write_artifact_commit(run)
        return run

    def test_cli_sequence_has_opposite_product_verdicts_and_complete_evidence(self):
        self.assertEqual(self.benchmark["availability"], "complete")
        self.assertEqual(
            {row["variant"]: row["objective_pass_rate"] for row in self.benchmark["results"]},
            {"with_skill": 1.0, "without_skill": 0.0},
        )
        self.assertTrue(all(row["execution_valid"] for row in self.benchmark["results"]))
        self.assertEqual(ORACLE.verify_edit(self.treatment), "Verified committed edit passes trusted product tests.")
        capture = json.loads((self.treatment / "workspace-changes.json").read_text())
        self.assertEqual([change["path"] for change in capture["changes"]], ["inputs/name_tools.py"])
        events = json.loads((self.treatment / "events.json").read_text())["events"]
        tests = [event for event in events if event["type"] == "command" and event["status"] == "completed"]
        self.assertEqual([(event["input_summary"], event["exit_code"]) for event in tests],
                         [("python3 -B inputs/test_name_tools.py", 0)])

    def test_empty_baseline_and_claimed_success_do_not_prove_an_edit(self):
        baseline = self.runs / "normalize-name/without_skill"
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ORACLE.verify_edit(baseline)
        run = self.copied_run()
        (run / "output.md").write_text("I edited the file and all tests passed.")
        (run / "workspace-changes.json").write_bytes((baseline / "workspace-changes.json").read_bytes())
        (run / "candidate.patch").unlink()
        sb.write_artifact_commit(run)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            ORACLE.verify_edit(run)

    def test_tampered_committed_patch_is_rejected(self):
        run = self.copied_run()
        with (run / "candidate.patch").open("a") as handle:
            handle.write("tampered\n")
        with self.assertRaisesRegex(TypeError, "inventory"):
            ORACLE.verify_edit(run)

    def test_missing_capture_is_rejected_even_with_a_valid_inventory(self):
        run = self.copied_run()
        (run / "workspace-changes.json").unlink()
        sb.write_artifact_commit(run)
        with self.assertRaisesRegex(ValueError, "capture"):
            ORACLE.verify_edit(run)

    def test_wrong_before_and_after_digests_are_rejected(self):
        for side in ("before", "after"):
            with self.subTest(side=side):
                run = self.change_capture(lambda capture, side=side: capture["changes"][0][side].update(sha256="0" * 64))
                with self.assertRaisesRegex(ValueError, f"{side} digest"):
                    ORACLE.verify_edit(run)

    def test_omitted_unsafe_symlink_binary_and_extra_edits_are_rejected(self):
        def extra(capture):
            change = json.loads(json.dumps(capture["changes"][0]))
            change["path"] = "inputs/test_name_tools.py"
            capture["changes"].append(change)

        mutations = {
            "omitted": lambda capture: capture["changes"][0].update(evidence={"kind": "omitted", "reason": "total_cap"}),
            "unsafe": lambda capture: capture["changes"][0].update(path="../name_tools.py"),
            "wrong-file": lambda capture: capture["changes"][0].update(path="inputs/test_name_tools.py"),
            "symlink": lambda capture: capture["changes"][0].update(after={"kind": "symlink", "target": "elsewhere"}),
            "binary": lambda capture: capture["changes"][0]["after"].update(text=False),
            "executable": lambda capture: capture["changes"][0]["after"].update(executable=True),
            "extra": extra,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                ORACLE.verify_edit(self.change_capture(mutate))

    def test_unsafe_patch_paths_are_rejected_before_candidate_execution(self):
        run = self.copied_run()
        patch = run / "candidate.patch"
        patch.write_text(patch.read_text().replace("inputs/name_tools.py", "../escape.py"))
        capture = json.loads((run / "workspace-changes.json").read_text())
        capture["patch"]["sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
        (run / "workspace-changes.json").write_text(json.dumps(capture))
        sb.write_artifact_commit(run)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            ORACLE.verify_edit(run)

    def test_regular_edit_with_wrong_product_and_claimed_success_is_rejected(self):
        import runner_contracts as rc
        import workspace_contracts as wc

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def build(workspace):
                (workspace / "inputs").mkdir()
                shutil.copyfile(DEMO / "evals/fixtures/name_tools.py", workspace / "inputs/name_tools.py")

            changes = root / "changes"
            changes.mkdir()
            with wc.captured_workspace(prefix="wrong-product-", changes_dir=changes, build=build) as (workspace, _):
                (workspace / "inputs/name_tools.py").write_text(
                    'def normalize_name(value: str) -> str:\n    return value.strip()\n')
            run = root / "run"
            sb.write_runner_outcome(run, rc.Completed(rc.OutcomeContext(provider="codex"),
                                                    answer="Edited the file. All tests passed."), sidecars=changes)
            with self.assertRaisesRegex(ValueError, "trusted product tests failed"):
                ORACLE.verify_edit(run)

    def test_artifact_symlink_is_rejected(self):
        run = self.copied_run()
        patch = run / "candidate.patch"
        data = patch.read_bytes()
        patch.unlink()
        elsewhere = run.parent / "patch"
        elsewhere.write_bytes(data)
        patch.symlink_to(elsewhere)
        with self.assertRaisesRegex(TypeError, "inventory"):
            ORACLE.verify_edit(run)


if __name__ == "__main__":
    unittest.main()
