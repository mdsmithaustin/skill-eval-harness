"""Fail unless the built wheel works on its own, outside the checkout.

CI's test jobs run from the source tree, where every top-level module is
importable whether or not the wheel ships it. A module missing from
``[tool.setuptools] py-modules`` passes every other step and breaks only for
users of the published package. This script:

1. builds a wheel from a copy of the checkout (or takes ``--wheel``);
2. installs it into a fresh virtual environment;
3. from a working directory outside the checkout, imports every top-level
   ``*.py`` module of the source tree and requires each to load from the
   environment's site-packages, then runs every ``[project.scripts]`` console
   script with ``--help``;
4. deletes one installed module and requires step 3's import check to fail,
   so the check cannot pass vacuously.

It also runs two offline Stub comparison pairs, recorded provider fixtures,
and trigger source ownership, relocation, required edits and missing bytes.

Installing the wheel resolves its dependencies through pip, as CI's install
step already does. Exit status: 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

_COPY_IGNORE = shutil.ignore_patterns(
    ".git", "build", "dist", "*.egg-info", "__pycache__", ".hypothesis",
    ".pytest_cache", ".ruff_cache", ".venv", "venv")

# Runs inside the installed environment: import each module named on the
# command line and report where it loaded from, or why it failed.
_IMPORT_PROBE = """
import importlib, json, sys
report = {}
for name in sys.argv[1:]:
    try:
        report[name] = {"file": importlib.import_module(name).__file__}
    except Exception as exc:
        report[name] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(report))
"""

_TRIGGER_PROBE = """
import json, sys
from pathlib import Path
import run_trigger_matrix, run_pi_trigger_eval, trigger_identity
root = Path(trigger_identity.__file__).resolve().parent
identity = trigger_identity.trigger_harness_identity()
loaded = sorted({Path(module.__file__).resolve().relative_to(root).as_posix()
                 for module in sys.modules.values()
                 if getattr(module, '__file__', None) and Path(module.__file__).resolve().parent == root})
assert set(loaded) <= set(identity['modules']), loaded
assert 'skill_benchmark' not in sys.modules
assert 'skill_benchmark.py' not in identity['modules']
assert identity['schema_version'] == 4
print(json.dumps({'root': str(root), 'loaded': loaded, 'identity': identity}))
"""

_DIRECT_MATRIX_PROBE = """
import runpy, sys
from pathlib import Path
import trigger_identity
sys.argv = ['run_trigger_matrix.py', '--help']
try:
    runpy.run_path(str(Path(trigger_identity.__file__).with_name('run_trigger_matrix.py')), run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
module = sys.modules['run_trigger_matrix']
assert module.ADAPTERS['stub'] is module.StubAdapter
assert module.StubAdapter.__module__ == '__main__'
assert 'skill_benchmark' not in sys.modules
print('OK: direct Matrix has one defining module')
"""

_EXECUTION_PROBE = """
import runpy, sys
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0, exc.code
"""

_OFFLINE_PROVIDER = """
import json, os, sys
from pathlib import Path
agent = Path(sys.argv[0]).name
fixtures = Path(__file__).with_name('fixtures')
if agent == 'pi':
    home = Path(os.environ['PI_CODING_AGENT_DIR'])
    text = (fixtures / 'pi/lifecycle-success.jsonl').read_text().replace('/tmp/pi-config', str(home))
elif agent == 'claude':
    skill = next((Path.cwd() / '.claude/skills').iterdir())
    text = (fixtures / 'claude/stream-json.plugin-skill.jsonl').read_text()
    text = text.replace('probe-plugin:tidy-commit', skill.name).replace('/work/probe-plugin/skills/tidy-commit', str(skill))
elif agent == 'codex':
    home = Path(os.environ['CODEX_HOME'])
    skill = next((home / 'skills').iterdir())
    text = (fixtures / 'codex/exec-skill-events.jsonl').read_text()
    thread = json.loads(text.splitlines()[0])['thread_id']
    rollout = (fixtures / 'codex/rollout-skill-injection.jsonl').read_text()
    rollout = rollout.replace('/CODEX_HOME/skills/unslop/SKILL.md', str(skill / 'SKILL.md'))
    rollout = rollout.replace('/CODEX_HOME', str(home)).replace('unslop', 'demo-reviewer')
    target = home / 'sessions/2026/09/13' / ('rollout-fixture-' + thread + '.jsonl')
    target.parent.mkdir(parents=True)
    target.write_text(rollout)
elif agent == 'vibe':
    text = (fixtures / 'vibe/streaming.2.25.8.skill-load.jsonl').read_text().replace('/work', str(Path.cwd()))
else:
    raise SystemExit('unknown offline provider fixture')
sys.stdout.write(text)
"""


class CheckFailed(Exception):
    pass


def source_modules() -> list[str]:
    modules = sorted(path.stem for path in ROOT.glob("*.py"))
    if not modules:
        raise CheckFailed(f"no top-level modules found under {ROOT}")
    return modules


def console_scripts() -> list[str]:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    table = re.search(r"(?ms)^\[project\.scripts\]\s*$\n(?P<body>.*?)(?=^\[|\Z)", text)
    names = re.findall(r"(?m)^([A-Za-z0-9_.-]+)\s*=", table.group("body")) if table else []
    if not names:
        raise CheckFailed("pyproject.toml declares no [project.scripts]")
    return names


def run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None, receipt: Path | None = None) -> str:
    if receipt is not None and receipt.exists():
        raise CheckFailed(f"command receipt already exists: {receipt}")
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=180)
    if receipt is not None:
        receipt.write_text(json.dumps({"command": argv, "cwd": str(cwd), "exit_code": proc.returncode,
                                       "deadline_seconds": 180, "stdout": proc.stdout, "stderr": proc.stderr,
                                       "failure_signal": "nonzero exit or the enclosing artifact assertion"}, indent=2) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise CheckFailed(
            f"{' '.join(argv)} exited {proc.returncode}\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    return proc.stdout


def build_wheel(work: Path) -> Path:
    source = work / "source"
    shutil.copytree(ROOT, source, ignore=_COPY_IGNORE)
    dist = work / "dist"
    run([sys.executable, "-m", "pip", "wheel", "--quiet", "--no-deps",
         "--wheel-dir", str(dist), str(source)], cwd=work)
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise CheckFailed(f"expected one wheel in {dist}, found {[w.name for w in wheels]}")
    return wheels[0]


def import_failures(python: Path, modules: list[str], purelib: Path, *,
                    cwd: Path, env: dict[str, str]) -> list[str]:
    """Modules that fail to import, or import from outside ``purelib``."""
    report = json.loads(run([str(python), "-I", "-c", _IMPORT_PROBE, *modules],
                            cwd=cwd, env=env))
    failures = []
    for name in modules:
        entry = report[name]
        if "error" in entry:
            failures.append(f"{name}: {entry['error']}")
        elif Path(entry["file"]).resolve().parent != purelib:
            failures.append(f"{name}: loaded from {entry['file']}, not the installed wheel")
    return failures


def check_trigger_execution(python: Path, purelib: Path, bindir: Path, outside: Path,
                            env: dict[str, str], artifacts: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    probe = json.loads(run([str(python), "-I", "-c", _TRIGGER_PROBE], cwd=outside, env=env,
                           receipt=artifacts / "installed-identity.command.json"))
    if Path(probe["root"]) != purelib or not probe["loaded"]:
        raise CheckFailed("trigger probe did not exercise the owning installation")
    (artifacts / "installed-identity.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
    run([str(python), "-I", "-c", _DIRECT_MATRIX_PROBE], cwd=outside, env=env,
        receipt=artifacts / "direct-matrix.command.json")
    run([str(python), "-I", str(purelib / "run_pi_trigger_eval.py"), "--help"], cwd=outside, env=env,
        receipt=artifacts / "direct-pi.command.json")
    demo = outside / "demo"
    shutil.copytree(ROOT / "examples/demo-skill", demo)
    manifest = demo / "evals/shared-benchmark.json"
    for cohort, query_count, extra, expected_class, regressions in (
            ("demo", 2, [], "refuted", 0),
            ("guide", 11, ["--eval-set", str(demo / "evals/trigger-eval-set.json")], "confirmed_causal", 6)):
        reports = []
        for arm, ablation in (("baseline", []), ("ablation", ["--ablation", "weaker-description"])):
            target = artifacts / f"{cohort}-{arm}.json"
            run([str(bindir / "skill-trigger-matrix"), str(manifest), "--agent", "stub",
                 "--runs-per-query", "3", "--workers", "1", *extra, *ablation, "--out", str(target)], cwd=outside, env=env,
                receipt=artifacts / f"{cohort}-{arm}.command.json")
            report = json.loads(target.read_text(encoding="utf-8"))
            if (len(report["results"]) != query_count * 3 or len(report["design"]) != query_count
                    or report["runs_per_query"] != 3 or report["protocol"]["harness_identity"] != probe["identity"]
                    or report["summary"]["measurement_status"] != "complete"):
                raise CheckFailed(f"{cohort}-{arm} has incomplete population or source identity")
            reports.append(target)
        comparison = artifacts / f"{cohort}-comparison.json"
        run([str(bindir / "skill-benchmark"), "trigger-compare", "--baseline", str(reports[0]),
             "--ablation", str(reports[1]), "--out", str(comparison)], cwd=outside, env=env,
            receipt=artifacts / f"{cohort}-comparison.command.json")
        value = json.loads(comparison.read_text(encoding="utf-8"))
        if (value["evidence_class"] != expected_class or value["summary"]["comparable"] != query_count
                or len(value["regressed_queries"]) != regressions or value["summary"]["blocked"] != 0):
            raise CheckFailed(f"{cohort} comparison does not match its offline treatment")
    check_provider_fixtures(python, purelib, outside, manifest, env, artifacts, probe["identity"])
    identity_command = [str(python), "-I", "-c", _TRIGGER_PROBE]
    for name, must_change in (("skill_benchmark.py", False), ("process_runtime.py", True)):
        path = purelib / name
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n# Installed identity control.\n")
            changed = json.loads(run(identity_command, cwd=outside, env=env))["identity"]
            if (changed != probe["identity"]) != must_change:
                raise CheckFailed(f"source identity responds incorrectly to {name}")
        finally:
            path.write_bytes(original)
    relocated = outside / "relocated"
    relocated.mkdir()
    for name in source_modules():
        shutil.copy2(purelib / f"{name}.py", relocated)
    metadata = list(purelib.glob("skill_eval_harness_ext-*.dist-info"))
    if len(metadata) != 1:
        raise CheckFailed("cannot locate exactly one owning distribution metadata directory")
    shutil.copytree(metadata[0], relocated / metadata[0].name)
    relocated_probe = json.loads(run([str(python), "-I", "-c",
                                     "import sys; sys.path.insert(0, sys.argv[1]);\n" + _TRIGGER_PROBE,
                                     str(relocated)], cwd=outside, env=env))
    (artifacts / "relocated-identity.json").write_text(json.dumps(relocated_probe, indent=2) + "\n", encoding="utf-8")
    if relocated_probe["identity"] != probe["identity"] or Path(relocated_probe["root"]) != relocated.resolve():
        raise CheckFailed("relocation changed source identity or used the wrong distribution")
    missing = purelib / "process_runtime.py"
    original = missing.read_bytes()
    try:
        missing.unlink()
        result = subprocess.run([str(python), "-I", "-c",
                                 "import trigger_identity; trigger_identity.trigger_harness_identity()"],
                                cwd=outside, env=env, capture_output=True, text=True, timeout=30, check=False)
        (artifacts / "missing-source.stdout").write_text(result.stdout, encoding="utf-8")
        (artifacts / "missing-source.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode == 0 or "required trigger owner process_runtime" not in result.stderr:
            raise CheckFailed(f"missing packaged bytes did not identify their required owner:\n{result.stderr[-2000:]}")
    finally:
        missing.write_bytes(original)
    print("OK: four installed Stub reports, both comparisons, source closure, relocation, and missing-byte controls")


def check_provider_fixtures(python: Path, purelib: Path, outside: Path, manifest: Path,
                            env: dict[str, str], artifacts: Path, identity: dict[str, Any]) -> None:
    fixture_bin = outside / "fixture-bin"
    fixture_bin.mkdir()
    fixtures = fixture_bin / "fixtures"
    for name in ("pi", "claude", "codex", "vibe"):
        shutil.copytree(ROOT / "tests/fixtures" / name, fixtures / name)
        executable = fixture_bin / name
        executable.write_text(f"#!{python}\n" + _OFFLINE_PROVIDER, encoding="utf-8")
        executable.chmod(0o755)
    shutil.copytree(fixture_bin, artifacts / "offline-provider-inputs")
    query = outside / "provider-query.json"
    query.write_text(json.dumps([{"query_id": "installed-fixture", "query": "Review this proposed change.",
                                  "should_trigger": True}]), encoding="utf-8")
    home = outside / "fixture-home"
    home.mkdir()
    fixture_env = {**env, "HOME": str(home), "PATH": str(fixture_bin) + os.pathsep + env.get("PATH", ""),
                   "ANTHROPIC_API_KEY": "offline-fixture", "CLAUDE_CONFIG_DIR": str(home / "claude"),
                   "CODEX_HOME": str(home / "codex"), "PI_CODING_AGENT_DIR": str(home / "pi"),
                   "VIBE_HOME": str(home / "vibe")}
    if Path(shutil.which("pi", path=fixture_env["PATH"]) or "") != fixture_bin / "pi":
        raise CheckFailed("Pi fixture command is not selected")
    for agent, extra in (
            ("pi", []),
            ("claude", ["--agent", "claude", "--model", "claude-sonnet-5", "--claude-bin", str(fixture_bin / "claude")]),
            ("codex", ["--agent", "codex", "--codex-cmd", shlex.quote(str(fixture_bin / "codex")) + " exec --json"]),
            ("vibe", ["--agent", "vibe", "--vibe-cmd", shlex.quote(str(fixture_bin / "vibe"))])):
        entrypoint = "run_pi_trigger_eval.py" if agent == "pi" else "run_trigger_matrix.py"
        target, traces = artifacts / f"{agent}-fixture.json", artifacts / f"{agent}-traces"
        output = run([str(python), "-I", "-c", _EXECUTION_PROBE + _TRIGGER_PROBE,
                      str(purelib / entrypoint), str(manifest), "--eval-set", str(query),
                      "--runs-per-query", "1", "--workers", "1", "--timeout", "30",
                      "--trace-runs", str(traces), "--out", str(target), *extra], cwd=outside, env=fixture_env,
                     receipt=artifacts / f"{agent}-fixture.command.json")
        loaded = json.loads(output.splitlines()[-1])
        report = json.loads(target.read_text(encoding="utf-8"))
        if (loaded["identity"] != identity or len(report["results"]) != 1
                or report["protocol"]["harness_identity"] != identity
                or report["summary"]["measurement_status"] != "complete"
                or report["results"][0]["triggered"] is not True):
            raise CheckFailed(f"installed {agent} fixture did not complete its trigger observation")
        events = list(traces.rglob("events.json"))
        if len(events) != 1 or not json.loads(events[0].read_text(encoding="utf-8"))["events"]:
            raise CheckFailed(f"installed {agent} fixture did not exercise trace callbacks")
        (artifacts / f"{agent}-loaded.json").write_text(json.dumps(loaded, indent=2) + "\n", encoding="utf-8")
    print("OK: installed Pi, Claude, Codex and Vibe offline children and trace callbacks")


def check(wheel: Path | None, artifacts_dir: Path | None = None) -> None:
    modules = source_modules()
    scripts = console_scripts()
    with tempfile.TemporaryDirectory(prefix="installed-wheel-check-") as td:
        work = Path(td)
        wheel = wheel or build_wheel(work)
        venv = work / "venv"
        run([sys.executable, "-m", "venv", "--without-pip", str(venv)], cwd=work)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([sys.executable, "-m", "pip", "--python", str(python), "install", "--quiet",
             "--disable-pip-version-check", str(wheel)], cwd=work)
        paths = json.loads(run([str(python), "-c",
                                "import json, sysconfig; print(json.dumps(sysconfig.get_paths()))"],
                               cwd=work))
        purelib, bindir = Path(paths["purelib"]).resolve(), Path(paths["scripts"])
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        outside = work / "elsewhere"
        outside.mkdir()

        failures = import_failures(python, modules, purelib, cwd=outside, env=env)
        for name in scripts:
            try:
                usage = run([str(bindir / name), "--help"], cwd=outside, env=env)
            except (CheckFailed, OSError) as exc:
                failures.append(f"console script {name}: {exc}")
                continue
            if not usage.startswith("usage:"):
                failures.append(f"console script {name}: --help printed no usage line")
        if failures:
            raise CheckFailed("installed wheel is broken:\n  " + "\n  ".join(failures))
        print(f"OK: {wheel.name} imports all {len(modules)} modules from site-packages "
              f"and runs {len(scripts)} console scripts")
        check_trigger_execution(python, purelib, bindir, outside, env, artifacts_dir or work / "artifacts")

        # Teeth: the same probe must notice a module the wheel lacks.
        missing = modules[0]
        (purelib / f"{missing}.py").unlink()
        if not import_failures(python, [missing], purelib, cwd=outside, env=env):
            raise CheckFailed(f"import check still passed after deleting {missing}.py")
        print(f"OK: the import check fails once {missing}.py is removed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--wheel", type=Path,
                        help="check this wheel instead of building one from the checkout")
    parser.add_argument("--artifacts-dir", type=Path, help="retain offline trigger reports and source proof here")
    args = parser.parse_args(argv)
    try:
        check(args.wheel.resolve() if args.wheel else None, args.artifacts_dir.resolve() if args.artifacts_dir else None)
    except CheckFailed as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
