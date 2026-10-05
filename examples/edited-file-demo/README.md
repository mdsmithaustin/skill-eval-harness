# Verify a captured file edit

This offline example fixes a Python name normalizer and grades the captured edit.
The treatment stub edits the copied module and runs its tests. The baseline stub
runs the same tests without editing the module. Neither arm calls a model.
This is a one-case plumbing example. `audit-manifest --fail-on-blockers` reports
missing adversarial coverage.

From the repository root, use an active Python 3.10 or newer environment with the
repository dependencies installed. `python3` must resolve to that environment.
Git must be on `PATH` for patch replay.

```sh
python3 skill_benchmark.py prepare examples/edited-file-demo/evals/shared-benchmark.json \
	--out /tmp/edit-tasks.jsonl
python3 skill_benchmark.py run-codex --tasks /tmp/edit-tasks.jsonl --runs /tmp/edit-runs \
	--codex-cmd "python3 -B '$PWD/examples/edited-file-demo/stub_runner.py'"
python3 skill_benchmark.py benchmark examples/edited-file-demo/evals/shared-benchmark.json \
	--runs /tmp/edit-runs --allow-scripts --out /tmp/edit-benchmark.json
python3 skill_benchmark.py report --benchmark /tmp/edit-benchmark.json --format github \
	--fail-on-failures --gate-variant with_skill
```

Each command must exit zero. The benchmark records a passing `verified-edit`
assertion for `with_skill` and a failing assertion for `without_skill`. The report
gate selects `with_skill`, which is also its default. The baseline failure stays
visible in the report.

The oracle first verifies the committed artifact inventory and complete capture.
It permits exactly one modification to the regular text file `inputs/name_tools.py`.
It checks the original digest, applies the committed patch to the immutable
fixture in a fresh directory, checks the resulting digest, and runs trusted tests.
The tests include whitespace collapse and Unicode case folding. Empty, missing,
tampered, omitted, unsafe, extra, and symlink edits fail before product execution.
Both the stub and oracle prevent Python bytecode writes.

The offline stub proves capture and grading. To check live Codex write permission,
run this separate opt-in command from the repository root. It can spend money.

```sh
python3 scripts/smoke_supported_clis.py --live --permission-edit --agents codex \
	--out-dir /tmp/edit-permission-smoke
```

A nonzero exit or `status: failed` in `smoke.json` means the check failed. This mode
uses the resolved Codex executable with `exec --sandbox workspace-write`. The
ordinary smoke command keeps its existing provider defaults. The permission check
requires the same run's verified edit and a completed native command event for
exactly `python3 -B inputs/test_name_tools.py`, with integer exit code zero.
An assistant claim, an unrelated command, or oracle success alone cannot pass.
The output directory retains each attempt's tasks, trace, command prefix,
benchmark, and run artifacts. `smoke.json` describes the latest attempt.
