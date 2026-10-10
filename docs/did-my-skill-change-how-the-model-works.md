# Did my skill change how the model works?

A benchmark can report no lift while the skill changes the path to the answer.
`trajectory_diff` compares completed commands and tool calls between paired arms.
Process assertions grade patterns in that trace. A `per_step` judge grades each
completed action. This tutorial produces all three with the offline demo.

## Produce traced runs

Run these commands from the repository root after the [test setup](../CONTRIBUTING.md#local-setup).
The examples use `.venv/bin/python3.12.14`. Use the Python interpreter in your
virtual environment if its name differs. Every CLI command must exit zero.
A non-zero exit means the workflow failed, even if an earlier output file exists.

```bash
S=$(mktemp -d /tmp/skill-trajectory.XXXXXX)
PY="$(pwd)/.venv/bin/python3.12.14"
M=examples/demo-skill/trajectory-benchmark.json
"$PY" skill_benchmark.py prepare "$M" --split tune --out "$S/tasks.jsonl"
"$PY" skill_benchmark.py run-codex --tasks "$S/tasks.jsonl" --runs "$S/runs" \
	--codex-cmd "$PY $(pwd)/examples/demo-skill/stub_runner.py"
"$PY" skill_benchmark.py judge "$M" --runs "$S/runs" \
	--judge-cmd "$PY $(pwd)/examples/demo-skill/stub_judge.py" --out "$S/judge.jsonl"
"$PY" skill_benchmark.py benchmark "$M" --runs "$S/runs" \
	--judge-results "$S/judge.jsonl" --out "$S/bench.json"
```

The manifest has two cases with the same prompt. `c-weak-outcome` checks only
that the answer starts with a review. Both arms pass. In `bench.json`, its flags
include `saturated/non-discriminating` and `no objective lift`.
The same case's `trajectory_diff` entry shows this subset of fields, reproduced
on 2026-10-09 with the deterministic stub:

```json
{
  "case_id": "c-weak-outcome",
  "pairs": 1,
  "mean_deltas": {"steps": 2, "commands": 2, "tool_calls": 2, "file_reads": 0, "file_writes": 0},
  "skill_invoked": {"with_skill": 1.0, "without_skill": 0.0},
  "commands_only_with_skill": [
    "cat skills/demo/SKILL.md",
    "cat skills/demo/references/checklist.md"
  ],
  "commands_only_without_skill": []
}
```

The stub records each file it reads as a Codex `command_execution` event.
The skill arm reads the skill and checklist. The baseline reads neither.
The fork mounts the source directory `demo` under `skills/demo/`.
These command events explain why the `file_reads` delta is zero.
The skill changed the path, but the outcome assertion cannot distinguish the arms.
Add a discriminating outcome assertion or grade the path directly.

## Grade a redundant path

`c-review-path` checks the severity label, the skill read, repeated commands,
and a `per_step` judge scoped to `with_skill`. All four checks pass on the normal
skill-arm run. Now run a stub that reads `SKILL.md` twice more before answering.
Its answer remains byte-identical.

```bash
"$PY" skill_benchmark.py run-codex --tasks "$S/tasks.jsonl" --runs "$S/loop-runs" \
	--codex-cmd "$PY $(pwd)/examples/demo-skill/stub_runner.py --loop"
"$PY" skill_benchmark.py judge "$M" --runs "$S/loop-runs" \
	--judge-cmd "$PY $(pwd)/examples/demo-skill/stub_judge.py" --out "$S/loop-judge.jsonl"
"$PY" skill_benchmark.py benchmark "$M" --runs "$S/loop-runs" \
	--judge-results "$S/loop-judge.jsonl" --out "$S/loop-bench.json"
"$PY" skill_benchmark.py judge "$M" --runs "$S/loop-runs" \
	--judge-cmd "$PY $(pwd)/examples/demo-skill/stub_judge.py --lenient" --out "$S/lenient-judge.jsonl"
"$PY" skill_benchmark.py benchmark "$M" --runs "$S/loop-runs" \
	--judge-results "$S/lenient-judge.jsonl" --out "$S/lenient-bench.json"
```

Each command must exit zero. Read the assertions in each report to detect an
unsound path. Successful report generation does not mean the assertions passed.
The skill-arm results for `c-review-path` are:

| Assertion | Normal path | Loop, careful judge | Loop, lenient judge |
|---|---|---|---|
| `severity-label` | pass | pass | pass |
| `skill-read` | pass | pass | pass |
| `no-reread-loop` | pass | fail | fail |
| `sound-steps` | pass | fail | pass |

The loop trace reads `SKILL.md`, `checklist.md`, `SKILL.md`, and `SKILL.md`.
`no_repeated_command_loop` counts adjacent repeats. Its evidence changes from
`repeated_command_max=1; max=1` to `repeated_command_max=2; max=1`.
The careful judge also catches the non-adjacent repeat at step 3. Its criteria
are `true, true, false, false`, with evidence `2/4 trajectory steps sound`.
The lenient judge passes all four steps. The command delta grows from 2 to 4.

## Decide what to change

- If the report shows no lift but different commands, improve the assertions
  before concluding that the skill has no effect.
- If the answer passes but a process assertion fails, inspect the path.
- If a deterministic check and judge disagree, read each check's definition.
  [Calibrate the judge](can-i-trust-my-judge.md) before using its verdict.
- If a pair is blocked with `missing_trace_evidence`, repair the trace capture.
  Missing evidence is not an empty path.

## What keeps the measurement honest

Only completed events enter counts and judge steps. A `per_step` assertion with
no completed steps fails closed and emits no judge task. The demo scopes that
assertion to `with_skill` because its baseline has no actions.
`trajectory_diff` describes paired means and has no significance test.

## Where this stops

The stubs prove the offline workflow. They do not establish live model behavior,
billing, or containment. To test which component caused a change, use an
[ablation study](ablation-study-walkthrough.md). To test whether an agent discovers
the skill, use the [discovery journey](did-removing-this-break-discovery.md).
