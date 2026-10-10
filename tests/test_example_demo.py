"""The bundled offline example is executable documentation: prepare -> run (with the
deterministic stub 'model') -> report, and the two materialized ablations each
confirm a regression on a distinct assertion. Runs in CI with no model/API."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import run_cli

import run_trigger_matrix as tm
import skill_benchmark as sb

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo-skill"

DEMO_MANIFEST = DEMO / "evals" / "shared-benchmark.json"
TRAJECTORY_MANIFEST = DEMO / "trajectory-benchmark.json"
TRIGGER_EVAL_SET = DEMO / "evals" / "trigger-eval-set.json"


def run_demo_suite(test: unittest.TestCase, manifest_path: Path, *,
                   runner_flags: str = "", judge_flags: str = "") -> tuple[dict, list[dict]]:
    tmp = tempfile.TemporaryDirectory(prefix="demo-journey-")
    test.addCleanup(tmp.cleanup)
    td = Path(tmp.name)
    commands = [
        ("prepare", manifest_path, "--split", "tune", "--out", td / "tasks.jsonl"),
        ("run-codex", "--tasks", td / "tasks.jsonl", "--runs", td / "runs",
         "--codex-cmd", f"{sys.executable} {DEMO / 'stub_runner.py'}{runner_flags}", "--timeout", "120"),
        ("judge", manifest_path, "--runs", td / "runs", "--out", td / "judge.jsonl",
         "--judge-cmd", f"{sys.executable} {DEMO / 'stub_judge.py'}{judge_flags}"),
        ("benchmark", manifest_path, "--runs", td / "runs", "--out", td / "bench.json",
         "--judge-results", td / "judge.jsonl"),
    ]
    for command in commands:
        code, _, stderr = run_cli(*command)
        test.assertEqual(code, 0, stderr)
    report = json.loads((td / "bench.json").read_text(encoding="utf-8"))
    verdicts = [json.loads(line) for line in (td / "judge.jsonl").read_text(encoding="utf-8").splitlines()]
    return report, verdicts


def assertion_row(report: dict, case_id: str, variant: str, name: str) -> dict:
    result = next(r for r in report["results"]
                  if r["case_id"] == case_id and r["variant"] == variant)
    return next(a for a in result["assertions"] + result["qualitative_assertions"]
                if a["name"] == name)


def _min_runs_for_significance() -> int:
    """The smallest per-case matched-pair count at which a unanimous regression
    (every paired delta pointing the same way) clears the two-sided sign-flip
    gate — computed against the real `sign_flip_significance`, not hardcoded,
    so this tracks the gate if its threshold ever moves."""
    for n in range(1, 15):
        if sb.sign_flip_significance([-1.0] * n)["significant_at_0_05"]:
            return n
    raise AssertionError("sign_flip_significance never reached significance up to n=14")


class DemoExampleTests(unittest.TestCase):
    """One offline pipeline run shared by every assertion about its report."""

    MANIFEST = DEMO / "evals" / "shared-benchmark.json"

    @classmethod
    def setUpClass(cls):
        mp = cls.MANIFEST
        manifest = sb.validate_manifest(mp)
        tmp = tempfile.TemporaryDirectory(prefix="demo-eval-")
        cls.addClassCleanup(tmp.cleanup)
        td = Path(tmp.name)
        # 6 matched runs per arm clear the two-sided paired sign-flip floor
        # (2/2^6 = 0.03125); fewer unanimous pairs stay INDETERMINATE.
        rows = sb.prepared_task_rows(mp, manifest, include_ablations=True, ablation_dir=str(td / "abl"), runs_per_variant=6)
        (td / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        stub = f"{sys.executable} {DEMO / 'stub_runner.py'}"
        code, _, stderr = run_cli("run-codex", "--tasks", td / "tasks.jsonl", "--runs", td / "runs",
                                  "--codex-cmd", stub, "--timeout", "120")
        if code != 0:
            raise AssertionError(f"run-codex exited {code}: {stderr}")
        variants = sorted({r["variant"] for r in rows})   # include the ablation arms, not just the manifest variants
        # The example declares one judge assertion, so an executable end-to-end
        # report must also materialize its verdicts. Leaving them deferred would
        # correctly make the report partial and would make objective ablation
        # evidence look complete only by projecting away a declared grader.
        judge_tasks = sb.collect_judge_tasks(mp, td / "runs", variants=variants)
        judge_cmd = f"{sys.executable} {DEMO / 'stub_judge.py'}"
        verdicts = [sb.run_one_judge_task(task, judge_cmd, None, 1)
                    for task in judge_tasks]
        judge_results = td / "judge-results.jsonl"
        judge_results.write_text(
            "\n".join(json.dumps(verdict) for verdict in verdicts) + "\n",
            encoding="utf-8",
        )
        cls.runs_dir, cls.judge_results = td / "runs", judge_results
        cls.report = sb.build_benchmark_report(
            mp, td / "runs", variants_arg=variants,
            judge_results_path=str(judge_results),
        )

    def test_materialized_ablations_confirm_offline(self):
        regs = {e["id"]: e for e in self.report["ablation_regressions"]}
        for aid, assertion in (("no-severity", "severity-label"), ("no-checklist", "cite-checklist")):
            entry = regs[aid]
            self.assertEqual(entry["status"], "measured", f"{aid} should be measured")
            self.assertTrue(entry["provenance_verified"], f"{aid} provenance must verify (materialized, same revision)")
            confirmed = [r for r in entry["regressions"] if r.get("expected_regression_confirmed")]
            self.assertTrue(confirmed, f"{aid} should confirm a regression")

    def test_token_overhead_joins_lift_once_judge_verdicts_are_supplied(self):
        # docs/is-my-skill-worth-its-tokens.md: before --judge-results existed the
        # judged demo could only report partial coverage, and a RunNumber pair
        # key crashed judge_task_id.
        partial = sb.paired_token_overhead_report(self.MANIFEST, runs=self.runs_dir)
        complete = sb.paired_token_overhead_report(
            self.MANIFEST, runs=self.runs_dir, judge_results_path=str(self.judge_results))
        self.assertEqual(partial["summary"]["availability"], "partial")
        self.assertEqual(len(complete["pairs"]), 12)
        self.assertEqual(complete["summary"]["objective_delta"]["mean"], 1.0)

    def test_with_skill_beats_without_on_the_demo(self):
        s = self.report["summary"]
        self.assertEqual(s["with_skill"]["objective_pass_rate"]["mean"], 1.0)      # skill present -> both assertions pass
        self.assertEqual(s["without_skill"]["objective_pass_rate"]["mean"], 0.0)   # no skill -> both fail

    def test_audit_rates_the_marks_on_runs_only_with_the_judge_verdicts(self):
        # The demo declares a judge assertion. Without its verdicts the audit's
        # benchmark is partial: readiness names that as a blocker and the marks
        # measured on runs read unavailable, instead of passing on empty lists.
        partial = sb.audit_manifest_report(self.MANIFEST, runs=str(self.runs_dir))
        complete = sb.audit_manifest_report(
            self.MANIFEST, runs=str(self.runs_dir), judge_results_path=str(self.judge_results))
        self.assertEqual([b["kind"] for b in partial["readiness"]["blocker_findings"]],
                         ["benchmark-incomplete"])
        self.assertEqual(complete["readiness"]["blocker_findings"], [])
        status = {mark["id"]: mark["status"] for mark in complete["eval_health"]["marks"]}
        partial_status = {mark["id"]: mark["status"] for mark in partial["eval_health"]["marks"]}
        for mark in ("baseline-headroom", "noise-below-min-lift", "arms-differ-only-in-skill"):
            with self.subTest(mark=mark):
                self.assertEqual(partial_status[mark], "unavailable")
                self.assertNotEqual(status[mark], "unavailable")
        # Two cases cannot move six the same way: the noise mark says so.
        self.assertEqual(status["noise-below-min-lift"], "concern")
        # The partial audit says what is missing: the judge verdicts, not --runs.
        noise_notes = next(mark["notes"] for mark in partial["eval_health"]["marks"]
                           if mark["id"] == "noise-below-min-lift")
        self.assertEqual(noise_notes, [
            "the benchmark is incomplete: judge assertions have no verdicts; pass --judge-results"])


class DemoReadmeTests(unittest.TestCase):
    """Any doc whose demo walkthrough claims an ablation confirms a regression at a
    specific `prepare --runs-per-variant` count is only honest if that count actually
    clears the per-case significance gate. Each doc here makes exactly that claim for
    its first --runs-per-variant occurrence, so a drift between the two can't ship
    silently."""

    def _assert_documented_runs_per_variant_clears_significance_gate(self, path: Path, label: str) -> None:
        text = path.read_text(encoding="utf-8")
        match = re.search(r"--runs-per-variant (\d+)", text)
        self.assertIsNotNone(match, f"{label} should document a --runs-per-variant value")
        documented = int(match.group(1))
        minimum = _min_runs_for_significance()
        self.assertGreaterEqual(
            documented, minimum,
            f"{label} documents --runs-per-variant {documented}, but a case needs "
            f">= {minimum} matched pairs to clear the significance gate and report "
            "expected_regression_confirmed",
        )

    def test_documented_runs_per_variant_clears_significance_gate(self):
        self._assert_documented_runs_per_variant_clears_significance_gate(
            DEMO / "README.md", "examples/demo-skill/README.md")

    def test_did_my_skill_edit_regress_doc_runs_per_variant_clears_significance_gate(self):
        self._assert_documented_runs_per_variant_clears_significance_gate(
            ROOT / "docs" / "did-my-skill-edit-regress.md", "docs/did-my-skill-edit-regress.md")


class WalkthroughJudgeStepTests(unittest.TestCase):
    """Any doc that claims an ablation confirms a regression
    (`expected_regression_confirmed`, `CONFIRMED_CAUSAL`, "confirms/confirm a
    regression") is only honest if the ablation `benchmark` call it pastes actually
    graded the judge assertion behind that confirmation. Otherwise the claimed
    evidence is a "partial" report artifact, not a real verdict. This is a structural
    check over fenced code blocks (which commands run in which order), not a
    full-text snapshot, so prose can be reworded freely without breaking it."""

    CONFIRM_MARKERS = (
        "expected_regression_confirmed", "CONFIRMED_CAUSAL",
        "confirms a regression", "confirm a regression",
    )
    _CODE_BLOCK = re.compile(r"```(?:bash|sh|yaml)\n(.*?)\n```", re.DOTALL)
    _JUDGE_CALL = re.compile(r"(?:skill[-_]benchmark(?:\.py)?|\$H(?:ARNESS)?)\s+judge\b(?!-)")
    _BENCHMARK_CALL = re.compile(r"(?:skill[-_]benchmark(?:\.py)?|\$H(?:ARNESS)?)\s+benchmark\b")

    @classmethod
    def _flatten(cls, block: str) -> list[str]:
        joined = re.sub(r"\\\n\s*", " ", block)
        return [line.strip() for line in joined.split("\n")
                if line.strip() and not line.strip().startswith("#")]

    @staticmethod
    def _flag_value(line: str, flag: str) -> str | None:
        m = re.search(re.escape(flag) + r"[= ]+(\S+)", line)
        return m.group(1) if m else None

    def _assert_confirming_ablation_benchmarks_are_judged(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        if not any(marker in text for marker in self.CONFIRM_MARKERS):
            return
        judge_outs: set[str] = set()
        ablation_benchmark_calls = 0
        for block in self._CODE_BLOCK.findall(text):
            for line in self._flatten(block):
                if self._JUDGE_CALL.search(line):
                    out = self._flag_value(line, "--out")
                    if out:
                        judge_outs.add(out)
                if self._BENCHMARK_CALL.search(line) and "ablation" in line:
                    ablation_benchmark_calls += 1
                    judge_results = self._flag_value(line, "--judge-results")
                    self.assertIsNotNone(
                        judge_results,
                        f"{path}: an ablation benchmark call claims a confirmed "
                        f"regression but has no --judge-results: {line!r}",
                    )
                    self.assertIn(
                        judge_results, judge_outs,
                        f"{path}: --judge-results {judge_results!r} is not the --out "
                        f"of a preceding `judge` step in the same doc: {line!r}",
                    )
        if ablation_benchmark_calls == 0:
            self.fail(
                f"{path} claims a confirmed regression but pastes no ablation "
                "benchmark command to check"
            )

    def _assert_some_benchmark_is_judged(self, path: Path) -> None:
        judge_outs: set[str] = set()
        judged = []
        for block in self._CODE_BLOCK.findall(path.read_text(encoding="utf-8")):
            for line in self._flatten(block):
                if self._JUDGE_CALL.search(line):
                    out = self._flag_value(line, "--out")
                    if out:
                        judge_outs.add(out)
                if self._BENCHMARK_CALL.search(line):
                    judge_results = self._flag_value(line, "--judge-results")
                    if judge_results and judge_results in judge_outs:
                        judged.append(line)
        self.assertTrue(
            judged, f"{path}: no benchmark call is fed by a preceding judge step's --out")

    def test_why_did_this_run_fail_benchmarks_judged_runs(self):
        self._assert_some_benchmark_is_judged(ROOT / "docs" / "why-did-this-run-fail.md")

    def test_gating_ci_on_evals_benchmarks_judged_runs(self):
        self._assert_some_benchmark_is_judged(ROOT / "docs" / "gating-ci-on-evals.md")

    def test_demo_readme_judges_before_confirming(self):
        self._assert_confirming_ablation_benchmarks_are_judged(DEMO / "README.md")

    def test_did_my_skill_edit_regress_judges_before_confirming(self):
        self._assert_confirming_ablation_benchmarks_are_judged(
            ROOT / "docs" / "did-my-skill-edit-regress.md")

    def test_ablation_study_walkthrough_judges_before_confirming(self):
        self._assert_confirming_ablation_benchmarks_are_judged(
            ROOT / "docs" / "ablation-study-walkthrough.md")


class DemoJudgeTests(unittest.TestCase):
    """Pins the stub-judge pair's calibration signature that
    docs/can-i-trust-my-judge.md pastes: the careful judge aligns with the human
    labels and rejects the negative controls; the --lenient rubber-stamp leaks
    every control and scores kappa 0.0 despite 0.5 raw agreement."""

    VARIANTS = ["with_skill", "without_skill", "ablation:no-severity", "ablation:no-checklist"]
    # The human gold labels the journey doc records for the four c-review arms.
    HUMAN = {
        "with_skill": True,
        "without_skill": False,
        "ablation:no-severity": False,
        "ablation:no-checklist": True,
    }

    def _judge_rows(self, lenient: bool):
        mp = DEMO / "evals" / "shared-benchmark.json"
        manifest = sb.validate_manifest(mp)
        tmp = tempfile.TemporaryDirectory(prefix="demo-judge-")
        self.addCleanup(tmp.cleanup)
        td = Path(tmp.name)
        rows = sb.prepared_task_rows(mp, manifest, include_ablations=True, ablation_dir=str(td / "abl"))
        rows = [r for r in rows if r["variant"] in self.VARIANTS]
        (td / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        code, _, stderr = run_cli("run-codex", "--tasks", td / "tasks.jsonl", "--runs", td / "runs",
                                  "--codex-cmd", f"{sys.executable} {DEMO / 'stub_runner.py'}",
                                  "--timeout", "120")
        self.assertEqual(code, 0, stderr)
        tasks = sb.collect_judge_tasks(mp, td / "runs", variants=self.VARIANTS)
        self.assertEqual(len(tasks), 4)   # one actionable-review task per c-review arm
        cmd = f"{sys.executable} {DEMO / 'stub_judge.py'}" + (" --lenient" if lenient else "")
        verdicts = [sb.run_one_judge_task(t, cmd, None, 1) for t in tasks]
        return tasks, verdicts, td

    @staticmethod
    def _keyed(rows):
        return {r["judge_task_id"]: r for r in rows}

    def test_careful_judge_aligns_and_rejects_controls(self):
        tasks, verdicts, td = self._judge_rows(lenient=False)
        human = {t["judge_task_id"]: {"passed": self.HUMAN[t["variant"]]} for t in tasks}
        align = sb.judge_alignment_report(human, self._keyed(verdicts), min_labels=4)
        self.assertEqual(align["cohen_kappa"], 1.0)
        self.assertEqual(align["confusion"], {"tp": 2, "fp": 0, "fn": 0, "tn": 2})
        robust = sb.judge_robustness_report(tasks, tmp_dir=td,
                                            judge_cmd=f"{sys.executable} {DEMO / 'stub_judge.py'}")
        self.assertEqual(robust["summary"]["order_flip_consistency"], 1.0)
        self.assertEqual(robust["summary"]["control_leak_rate"], 0.0)
        self.assertEqual(robust["findings"], [])

    def test_lenient_judge_is_caught_by_both_probes(self):
        tasks, verdicts, td = self._judge_rows(lenient=True)
        human = {t["judge_task_id"]: {"passed": self.HUMAN[t["variant"]]} for t in tasks}
        align = sb.judge_alignment_report(human, self._keyed(verdicts), min_labels=4)
        self.assertEqual(align["agreement"], 0.5)      # right whenever the answer deserves to pass...
        self.assertEqual(align["cohen_kappa"], 0.0)    # ...but no better than chance once corrected
        self.assertEqual(align["recall"], 1.0)
        self.assertEqual(align["precision"], 0.5)
        robust = sb.judge_robustness_report(tasks, tmp_dir=td,
                                            judge_cmd=f"{sys.executable} {DEMO / 'stub_judge.py'} --lenient")
        self.assertEqual(robust["summary"]["control_leak_rate"], 1.0)
        kinds = {f["kind"] for f in robust["findings"]}
        self.assertEqual(kinds, {"passes-empty-control", "passes-master-key-control"})


class DemoTrajectoryJourneyTests(unittest.TestCase):
    def test_no_lift_case_still_shows_the_skill_changing_the_path(self):
        report, _ = run_demo_suite(self, TRAJECTORY_MANIFEST)
        for variant in ("with_skill", "without_skill"):
            self.assertTrue(assertion_row(report, "c-weak-outcome", variant, "gives-a-review")["passed"])
        flags = next(f["flags"] for f in report["case_flags"] if f["case_id"] == "c-weak-outcome")
        self.assertIn("no objective lift", flags)
        case = next(c for c in report["trajectory_diff"]["cases"] if c["case_id"] == "c-weak-outcome")
        self.assertEqual(case["commands_only_with_skill"],
                         ["cat skills/demo/SKILL.md",
                          "cat skills/demo/references/checklist.md"])
        self.assertEqual(case["commands_only_without_skill"], [])
        self.assertEqual(case["skill_invoked"], {"with_skill": 1.0, "without_skill": 0.0})
        self.assertEqual(case["mean_deltas"]["commands"], 2.0)

    def test_careful_path_passes_the_process_and_per_step_checks(self):
        report, verdicts = run_demo_suite(self, TRAJECTORY_MANIFEST)
        for name in ("severity-label", "skill-read", "no-reread-loop", "sound-steps"):
            with self.subTest(name=name):
                self.assertTrue(assertion_row(report, "c-review-path", "with_skill", name)["passed"])
        per_step = next(v for v in verdicts if v["judge_task_id"].endswith("::sound-steps"))
        self.assertEqual(per_step["criteria"], [{"name": "step-1", "met": True},
                                                {"name": "step-2", "met": True}])

    def test_looping_path_keeps_the_answer_and_fails_only_the_path_checks(self):
        report, verdicts = run_demo_suite(self, TRAJECTORY_MANIFEST, runner_flags=" --loop")
        self.assertTrue(assertion_row(report, "c-review-path", "with_skill", "severity-label")["passed"])
        self.assertTrue(assertion_row(report, "c-review-path", "with_skill", "skill-read")["passed"])
        self.assertFalse(assertion_row(report, "c-review-path", "with_skill", "no-reread-loop")["passed"])
        self.assertFalse(assertion_row(report, "c-review-path", "with_skill", "sound-steps")["passed"])
        per_step = next(v for v in verdicts if v["judge_task_id"].endswith("::sound-steps"))
        self.assertEqual([c["met"] for c in per_step["criteria"]], [True, True, False, False])
        case = next(c for c in report["trajectory_diff"]["cases"] if c["case_id"] == "c-review-path")
        self.assertEqual(case["mean_deltas"]["commands"], 4.0)

    def test_lenient_per_step_judge_misses_the_loop(self):
        report, _ = run_demo_suite(self, TRAJECTORY_MANIFEST, runner_flags=" --loop",
                                   judge_flags=" --lenient")
        self.assertTrue(assertion_row(report, "c-review-path", "with_skill", "sound-steps")["passed"])
        self.assertFalse(assertion_row(report, "c-review-path", "with_skill", "no-reread-loop")["passed"])


class DemoDiscoveryJourneyTests(unittest.TestCase):
    @staticmethod
    def _compare(rows):
        kwargs = {"agents": ["stub"], "models": None, "runs_per_query": 3, "timeout": 30, "workers": 1}
        baseline = tm.run_matrix(DEMO_MANIFEST, rows, **kwargs)
        ablated = tm.run_matrix(DEMO_MANIFEST, rows, ablation="weaker-description", **kwargs)
        return sb.build_trigger_comparison(baseline, ablated)

    @staticmethod
    def _eval_set_rows():
        return tm.validate_trigger_rows(
            json.loads(TRIGGER_EVAL_SET.read_text(encoding="utf-8"))["queries"],
            str(TRIGGER_EVAL_SET))

    def test_manifest_queries_refute_the_ablation(self):
        rows = tm.cases_from_manifest(tm.load_manifest(DEMO_MANIFEST), None)
        out = self._compare(rows)
        self.assertTrue(out["provenance"]["verified"])
        self.assertEqual(out["evidence_class"], "refuted")
        self.assertEqual(out["regressed_queries"], [])

    def test_three_queries_in_the_users_words_are_indeterminate(self):
        rows = [r for r in self._eval_set_rows() if r["query_id"].startswith("wtu-")][:3]
        out = self._compare(rows)
        self.assertEqual(out["evidence_class"], "indeterminate")
        self.assertEqual(len(out["regressed_queries"]), 3)
        self.assertIn("not significant", out["note"])

    def test_full_eval_set_confirms_the_ablation(self):
        rows = self._eval_set_rows()
        out = self._compare(rows)
        self.assertEqual(out["evidence_class"], "confirmed_causal")
        regressed = {r["query_id"] for r in out["regressed_queries"]}
        self.assertEqual(regressed, {r["query_id"] for r in rows if r["query_id"].startswith("wtu-")})
        self.assertEqual(len(regressed), 6)


if __name__ == "__main__":
    unittest.main()
