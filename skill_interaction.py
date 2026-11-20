"""Four-configuration experiments on task outcomes."""
from __future__ import annotations

import inspect
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import grading_contracts
import skill_benchmark as bench
import text_contracts
from ablation_model import PreparedTask, execution_valid
from content_digests import (
    canonical_json_sha256,
    directory_tree_sha256,
    file_sha256,
    tree_sha256,
)
from effect_estimates import sign_flip_interval
from json_contracts import strict_json_loads
from manifest_contracts import RunNumber
from skill_inputs import (
    AblationError,
    _copy_skill_root,
    frontmatter_value,
    skill_root_key,
)

ARMS = {"neither": (), "a": ("a",), "b": ("b",), "both": ("a", "b")}
PLAN_NAME = "interaction-plan.json"
DIGEST_SCHEME = "skill-tree-sha256-v1"


def _dataset(path: Path, split: str | None) -> tuple[list[dict[str, Any]], str]:
    manifest = bench.validate_manifest(path, allow_missing_holdback=split != "holdback")
    cases = [case for case in bench.iter_cases(manifest, split) if not bench.is_trigger_case(case)]
    identity = bench.eval_contract_sha256(
        {"version": manifest["version"], "cases": cases}, path, cases=cases)
    return cases, identity


def _grader() -> dict[str, str]:
    return {
        "id": "builtin-outcome-v1",
        "sha256": canonical_json_sha256({
            "grading": inspect.getsource(bench.grade_case_variant),
            "assertions": inspect.getsource(bench.assertion_result),
            "contracts": {module.__name__: file_sha256(Path(module.__file__))
                          for module in (grading_contracts, text_contracts)},
        }),
    }


def _check_cases(cases: list[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("interaction requires at least one answer case")
    for case in cases:
        cid = case["id"]
        if case.get("turns") or case.get("tool_faults"):
            raise ValueError(f"{cid}: interaction requires single-turn cases without tool faults")
        assertions = case.get("assertions", [])
        for assertion in assertions:
            atype = assertion["type"]
            if (atype in bench.QUALITATIVE_ASSERTIONS or atype == "script"
                    or (atype == "similarity" and assertion.get("mode") == "embedding")):
                raise ValueError(f"{cid}: interaction requires built-in deterministic assertions")
            if any(field in assertion for field in ("variants", "only_variants", "except_variants")):
                raise ValueError(f"{cid}: interaction uses the same assertions in all four arms")
        if not any(assertion["type"] in bench.TEXT_ASSERTIONS
                   and bench.assertion_severity(assertion) in {"gate", "critical"}
                   for assertion in assertions):
            raise ValueError(f"{cid}: interaction requires a gate or critical outcome assertion")


def _read_skill(root: Path) -> tuple[Path, str, str]:
    source = root.expanduser().resolve()
    if source.is_dir():
        source /= "SKILL.md"
    if source.name != "SKILL.md" or not source.is_file():
        raise ValueError(f"expected a directory containing SKILL.md or a SKILL.md file: {root}")
    name = frontmatter_value(source.read_text(encoding="utf-8"), "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"SKILL.md must declare a skill name: {root}")
    return source.parent, name, skill_root_key(source.parent.parent, source.parent.name)


def _prepare(
    *,
    manifest: Path,
    cases: list[dict[str, Any]],
    dataset: str,
    runs: Path,
    sources: dict[str, tuple[Path, str, str]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    inputs = runs / "_inputs"
    skills_root = inputs / "skills"
    skills: dict[str, dict[str, str]] = {}
    for label, (source, name, key) in sources.items():
        target = skills_root / key
        _copy_skill_root(source, target)
        skills[label] = {
            "name": name, "mount_name": key, "digest_scheme": DIGEST_SCHEME,
            "digest": directory_tree_sha256(target),
        }

    oracles = inputs / "oracles"
    oracles.mkdir(parents=True)
    golden_paths = {
        assertion["reference"] for case in cases for assertion in case.get("assertions", [])
        if assertion["type"] == "golden_output"
    }
    for relative in sorted(golden_paths):
        src = bench.resolved_assertion_path(manifest.parent, relative)
        target = oracles / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)

    prompts = {}
    files: dict[str, tuple[str, ...]] = {}
    for index, case in enumerate(cases):
        cid = case["id"]
        prompts[cid] = bench.case_prompt(case, manifest)
        copied = []
        for relative in case.get("files", []):
            src = (manifest.parent / relative).resolve()
            target = inputs / "fixtures" / str(index) / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"{cid}: multiple fixtures mount as {src.name}")
            shutil.copyfile(src, target)
            copied.append(str(target))
        files[cid] = tuple(copied)

    tasks = {}
    designs = {}
    for arm, labels in ARMS.items():
        keys = tuple(skills[label]["mount_name"] for label in labels)
        paths = tuple(str(skills_root / key / "SKILL.md") for key in keys)
        mounted_hash = tree_sha256(
            (path.relative_to(skills_root).as_posix(), path)
            for key in keys for path in (skills_root / key).rglob("*") if path.is_file()
        ) if keys else None
        variant = bench.WITH_SKILL if keys else bench.WITHOUT_SKILL
        rows = []
        for case in cases:
            cid = case["id"]
            for n in range(1, config["repetitions"] + 1):
                task = PreparedTask(
                    case_id=cid,
                    split=case["split"],
                    kind=case.get("kind", "behavior"),
                    variant_truth=variant,
                    run_number=RunNumber(n),
                    skill_name="interaction",
                    repo_root=str(manifest.parent),
                    skill_paths=paths,
                    skill_root_keys=keys,
                    input_files=files[cid],
                    run_dir=f"{cid}/{variant}/run-{n}",
                    instruction=bench.variant_instruction(variant, {"skill_name": "provided"}),
                    prompt=prompts[cid],
                    tags=tuple(case.get("tags", [])),
                    skill_tree_hash=mounted_hash,
                )
                rows.append({**task.harness_record(), "model": config["model"],
                             "eval_contract_sha256": dataset})
        tasks[arm] = rows
        designs[arm] = bench.answer_design_from_tasks(rows)["design_sha256"]
    if _dataset(manifest, config["split"])[1] != dataset:
        raise ValueError("dataset changed while preparing interaction inputs")

    plan = {
        "schema_version": 1, **config, "skills": skills, "dataset_sha256": dataset,
        "case_ids": [case["id"] for case in cases], "grader": _grader(),
        "oracle_tree_sha256": directory_tree_sha256(oracles),
        "arm_designs": designs,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    bench.write_json(runs / PLAN_NAME, plan)
    return plan, tasks


def _load_plan(runs: Path) -> dict[str, Any]:
    plan = strict_json_loads((runs / PLAN_NAME).read_bytes())
    if (not isinstance(plan, dict) or plan.get("schema_version") != 1
            or plan.get("plan_sha256") != canonical_json_sha256(
                {key: value for key, value in plan.items() if key != "plan_sha256"})):
        raise ValueError("interaction plan has an unsupported version or mismatched digest")
    return plan


def _run_outcome(
    base: Path, plan: dict[str, Any], design: dict[str, Any], row: dict[str, Any],
    case: dict[str, Any], oracles: Path,
) -> tuple[float | None, str | None, str | None, str | None]:
    if not base.is_dir():
        return None, "missing_run", None, None
    metadata = bench.read_metrics_base(base)
    if metadata.get("artifact_set_complete") is not True or metadata.get("metadata_error"):
        return None, "invalid_run_artifacts", None, None
    served = metadata.get("served_model")
    check = metadata.get("served_model_check")
    if (metadata.get("provider") != plan["agent"]
            or metadata.get("model") != plan["model"]
            or metadata.get("requested_model") != plan["model"]
            or check in {"mismatch", "mixed"}):
        return None, "model_or_agent_mismatch", served, check
    expected = {
        "case_id": row["case_id"], "run_number": row["run_number"], "variant": row["variant"],
        "answer_design_sha256": design["design_sha256"], "answer_task_sha256": row["task_sha256"],
        "answer_instruction_sha256": row["instruction_sha256"],
        "fixture_tree_hash": row["fixture_tree_hash"], "skill_tree_hash": row["planned_skill_tree_hash"],
    }
    if any(metadata.get(field) != value for field, value in expected.items()):
        return None, "run_identity_mismatch", served, check
    text, output = bench.read_output_base(base)
    if not text or not text.strip() or not execution_valid(metadata, text):
        return None, "incomplete_execution", served, check
    result, _ = bench.grade_case_variant(
        case, row["variant"], text, output, metadata, run_number=row["run_number"],
        run_base=base, manifest_dir=oracles, model=plan["model"])
    outcome = result["outcome_pass_rate"]
    return outcome, "incomplete_grading" if outcome is None else None, served, check


def _estimate(deltas: list[float]) -> dict[str, Any]:
    interval = sign_flip_interval(deltas)
    return {"estimate": round(statistics.mean(deltas), 12), "interval": interval,
            "conclusion": _conclusion(interval)}


def _conclusion(interval: dict[str, Any]) -> str:
    if not interval["bounded"]:
        return "inconclusive"
    if interval["lower"] > 0:
        return "positive"
    if interval["upper"] < 0:
        return "negative"
    return "inconclusive"


def _complete_report(plan: dict[str, Any], runs: list[dict[str, Any]],
                     issues: list[dict[str, Any]]) -> dict[str, Any]:
    cases, k = plan["case_ids"], plan["repetitions"]
    buckets: dict[tuple[str, str], list[float]] = {
        (arm, cid): [] for arm in ARMS for cid in cases}
    for row in runs:
        if row["outcome_rate"] is not None:
            buckets[row["arm"], row["case_id"]].append(row["outcome_rate"])
    rates = {arm: {cid: statistics.mean(buckets[arm, cid])
                   for cid in cases if len(buckets[arm, cid]) == k} for arm in ARMS}
    arm_report = {
        arm: {
            "rate": round(statistics.mean(rates[arm].values()), 12)
            if len(rates[arm]) == len(cases) else None,
            "complete_cases": len(rates[arm]),
            "scored_runs": sum(len(buckets[arm, cid]) for cid in cases),
        }
        for arm in ARMS
    }
    comparisons = {}
    required = {
        "both_vs_neither": ("both", "neither"),
        "both_vs_better_individual": ("both", "a", "b"),
        "interaction": ("both", "a", "b", "neither"),
    }
    for comparison, arms in required.items():
        present = [cid for cid in cases if all(cid in rates[arm] for arm in arms)]
        entry: dict[str, Any] = {
            "complete_cases": len(present), "required_arms": list(arms),
        }
        if len(present) != len(cases):
            entry.update(status="withheld", estimate=None, interval=None, conclusion="incomplete_evidence",
                         incomplete_cases=[cid for cid in cases if cid not in present])
        elif comparison == "both_vs_neither":
            entry.update(status="estimated", **_estimate([
                rates["both"][cid] - rates["neither"][cid] for cid in present]))
        elif comparison == "interaction":
            entry.update(status="estimated", **_estimate([
                (rates["both"][cid] - rates["a"][cid])
                - (rates["b"][cid] - rates["neither"][cid]) for cid in present]))
        else:
            a = [rates["both"][cid] - rates["a"][cid] for cid in present]
            b = [rates["both"][cid] - rates["b"][cid] for cid in present]
            intervals = [sign_flip_interval(deltas, confidence=0.975) for deltas in (a, b)]
            bounded = all(interval["bounded"] for interval in intervals)
            interval = {
                "confidence": 0.95, "method": "bonferroni-sign-flip-minimum", "bounded": bounded,
                "n": len(present),
                "lower": min(i["lower"] for i in intervals) if bounded else None,
                "upper": min(i["upper"] for i in intervals) if bounded else None,
                **({"reason": "an individual comparison has no bounded interval"} if not bounded else {}),
            }
            entry.update(status="estimated", estimate=round(min(statistics.mean(a), statistics.mean(b)), 12),
                         interval=interval, conclusion=_conclusion(interval))
        comparisons[comparison] = entry
    a_rate, b_rate = arm_report["a"]["rate"], arm_report["b"]["rate"]
    better = None
    if a_rate is not None and b_rate is not None:
        better = "tie" if a_rate == b_rate else ("a" if a_rate > b_rate else "b")
    return {
        "schema_version": 1, "plan_sha256": plan["plan_sha256"], "skills": plan["skills"],
        "dataset_sha256": plan["dataset_sha256"], "grader": plan["grader"],
        "agent": plan["agent"], "model": plan["model"], "split": plan["split"],
        "repetitions": k, "planned_cases": len(cases),
        "complete_cases": comparisons["interaction"]["complete_cases"],
        "better_individual": better, "arms": arm_report, "comparisons": comparisons,
        "issues": issues, "runs": runs,
        "served_model_checks": dict(Counter(row["served_model_check"] or "unrecorded" for row in runs)),
    }


def _oracle_digest(path: Path) -> str | None:
    try:
        return directory_tree_sha256(path, reject_symlinks=True) if path.is_dir() else None
    except (OSError, ValueError):
        return None


def _report(manifest: Path, root: Path, plan: dict[str, Any],
            expected: dict[str, Any]) -> dict[str, Any]:
    issues = []
    for field, value in expected.items():
        if value is not None and plan[field] != value:
            issues.append({"code": f"{field}_mismatch", "recorded": plan[field], "expected": value})
    if plan["grader"] != _grader():
        issues.append({"code": "grader_mismatch"})
    try:
        cases, dataset = _dataset(manifest, plan["split"])
        if dataset != plan["dataset_sha256"]:
            issues.append({"code": "dataset_mismatch", "current_sha256": dataset})
    except (OSError, ValueError, SystemExit) as exc:
        issues.append({"code": "dataset_unavailable", "detail": str(exc)})
        cases = []
    oracles = root / "_inputs" / "oracles"
    if _oracle_digest(oracles) != plan["oracle_tree_sha256"]:
        issues.append({"code": "grader_inputs_mismatch"})
    if issues:
        return _complete_report(plan, [], issues)

    by_id = {case["id"]: case for case in cases}
    reported_runs = []
    for arm, digest in plan["arm_designs"].items():
        arm_root = root / arm
        try:
            design = bench.validate_answer_design(
                strict_json_loads((arm_root / bench.ANSWER_DESIGN_NAME).read_bytes()))
            if design["design_sha256"] != digest:
                raise ValueError("answer design digest disagrees with the interaction plan")
        except (OSError, ValueError) as exc:
            issues.append({"code": "missing_or_incompatible_arm_design", "arm": arm, "detail": str(exc)})
            continue
        for row in design["identities"]:
            cid, n = row["case_id"], row["run_number"]
            base = bench.safe_child_path(arm_root, row["run_dir"])
            try:
                rate, problem, served, check = _run_outcome(base, plan, design, row, by_id[cid], oracles)
            except (OSError, ValueError) as exc:
                rate, problem, served, check = None, "invalid_run_artifacts", None, None
                issues.append({"code": problem, "arm": arm, "case_id": cid, "run_number": n,
                               "detail": str(exc)})
            else:
                if problem:
                    issues.append({"code": problem, "arm": arm, "case_id": cid, "run_number": n})
            reported_runs.append({
                "arm": arm, "case_id": cid, "run_number": n,
                "outcome_rate": rate, "served_model": served, "served_model_check": check,
            })

    served_models = {row["served_model"] for row in reported_runs
                     if row["outcome_rate"] is not None and row["served_model"] is not None}
    if len(served_models) > 1:
        issues.append({"code": "served_model_drift", "served_models": sorted(served_models)})
        return _complete_report(plan, [], issues)
    if _oracle_digest(oracles) != plan["oracle_tree_sha256"]:
        issues.append({"code": "grader_inputs_mismatch"})
        return _complete_report(plan, [], issues)
    return _complete_report(plan, reported_runs, issues)


def run_interaction(
    *, manifest: Path, runs: Path, sources: tuple[str | None, str | None],
    model: str | None, agent: str | None, split: str | None, repetitions: int | None,
    timeout: int, out: str | None, report_only: bool, options: dict[str, Any],
) -> int:
    expected = {"model": model, "agent": agent, "split": split, "repetitions": repetitions}
    try:
        if out:
            output = Path(out).resolve()
            if runs in output.parents:
                top = output.relative_to(runs).parts[0]
                if top in {*ARMS, "_inputs", PLAN_NAME}:
                    raise ValueError("--out would overwrite retained interaction evidence")
        if report_only:
            plan = _load_plan(runs)
        else:
            skill_a, skill_b = sources
            if not (agent and model and skill_a and skill_b):
                raise ValueError("running an interaction requires --agent, --model, --skill-a and --skill-b")
            roots = {"a": _read_skill(Path(skill_a)), "b": _read_skill(Path(skill_b))}
            a, b = roots.values()
            if a[0] == b[0] or a[0] in b[0].parents or b[0] in a[0].parents or a[1] == b[1] or a[2] == b[2]:
                raise ValueError("interaction requires distinct skill roots, names and mount directories")
            if any(source == runs or source in runs.parents or runs in source.parents
                   for source, _, _ in roots.values()):
                raise ValueError("--runs must be outside both skill roots")
            cases, dataset = _dataset(manifest, split)
            _check_cases(cases)
            backend = bench.registered_agent_backend(agent)
            runs.mkdir(parents=True, exist_ok=False)
            plan, tasks = _prepare(
                manifest=manifest, cases=cases, dataset=dataset, runs=runs, sources=roots,
                config={**expected, "repetitions": repetitions or 1, "timeout": timeout})
            for arm, rows in tasks.items():
                bench.run_agent_tasks(rows, runs / arm, backend, timeout=timeout, **options)
        report = _report(manifest, runs, plan, expected)
        bench.emit_report(report, out)
        return 2 if report["complete_cases"] != report["planned_cases"] else 0
    except (AblationError, OSError, ValueError) as exc:
        bench.die(str(exc))
