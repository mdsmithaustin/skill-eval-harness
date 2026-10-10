# How do I stop an answer, subagent, or judge batch at a dollar ceiling?

Use `--max-cost-usd` on `run-agent`, `run-claude`, or `run-codex` to limit admission for one command invocation. A call can exceed the ceiling. The command finishes that call and refuses later calls.

This offline example uses the demo runner. From the repository root, prepare its ordinary answer tasks:

```sh
skill-benchmark prepare examples/demo-skill/evals/shared-benchmark.json \
  --split tune --out /tmp/spend-demo-tasks.jsonl
```

Assign each unpriced answer call an explicit assumed charge:

```sh
skill-benchmark run-codex --tasks /tmp/spend-demo-tasks.jsonl \
  --runs /tmp/spend-demo-runs \
  --codex-cmd "python3 $(pwd)/examples/demo-skill/stub_runner.py" \
  --max-cost-usd 0.02 --assumed-cost-per-run-usd 0.01
```

The command prints a ledger path under `/tmp/spend-demo-runs/spend/<invocation-id>/spend-ceiling.json` and exits 2 after two calls. Later calls have `not_started` records. The runner writes no answer or process return code for those refusals.

Inspect the retained experiment:

```sh
skill-benchmark benchmark examples/demo-skill/evals/shared-benchmark.json \
  --runs /tmp/spend-demo-runs --out /tmp/spend-demo-report.json
```

The report's `spend_invocations` field includes the ledger and its path. Missing answers remain incomplete under the existing artifact and grade rules. `cost-summary` exposes the same invocation records separately from observed model cost totals.

For a provider that reports complete dollars, such as Claude, omit the assumption to use observed cost. Codex, Gemini, and Vibe require an assumption before a paid start because their registered native backends do not promise dollar telemetry. A zero ceiling starts no calls and needs no assumption.

If a call unexpectedly returns no price, admission closes and the ledger reports a partial known subtotal. An assumption permits later calls until the assumed total reaches the ceiling. Assumptions always have the `assumed` charge basis and cannot reduce any observed subtotal. They never become observed telemetry.

Each invocation gets its own exclusive directory and a fresh ceiling. Repeating a command against the same runs root preserves both ledgers. It does not resume or replay an old spend ledger. The immutable `answer-design.json` continues to describe the experiment, independent of spend policy.

The harness publishes `in_flight` before dispatch and publishes the charge before workspace capture and cleanup. An observed provider charge survives a later capture, cleanup, or output-write error. The error still stops the command and prevents later calls. Refused calls build no model workspace or baseline.

Each publication flushes and syncs the file before atomic replacement, then syncs the directory where the platform supports it. File publication failures stop dispatch. An abrupt process kill or failed settlement publication can leave unresolved `in_flight` evidence. The ledger then reports partial spend. A raised callback exception retains an attempted, unpriced receipt, or a labeled assumption if one was supplied. Raised setup, provider, or pricing exceptions skip workspace capture. A settlement publication error also skips capture. Returned failure and timeout outcomes still capture workspace evidence before deletion.

For capped batches, every selected task must be an ordinary native answer task. A recovery row anywhere in the batch rejects the command before provider calls or writes to the runs root. Uncapped recovery keeps its existing phase behavior. The native rules above apply to the serial native answer commands. `run-subagent` has the external-turn rules below. Judge calls have the rules below. Jetty submissions, trigger matrices, and Pi trigger evaluations keep their existing execution policies.

## Subagent external turns

`run-subagent` accepts the same ceiling and assumption flags. It admits each external backend callback, including each scripted turn. A provider's internal maximum turn setting stays within one callback.

Use the prepared demo tasks above and create an offline JSON backend:

```sh
cat > /tmp/subagent-spend-demo.py <<'PYTHON'
import json
import sys

json.load(sys.stdin)
print(json.dumps({"answer": "demo answer", "usage": {"cost_usd": 0.03},
                  "telemetry_scope": "turn_delta"}))
PYTHON
skill-benchmark run-subagent --tasks /tmp/spend-demo-tasks.jsonl \
  --runs /tmp/subagent-spend-demo-runs \
  --agent-cmd "python3 /tmp/subagent-spend-demo.py" --max-cost-usd 0.02
```

The first callback starts and records an observed $0.03 charge. Later required calls are refused, and the command exits 2. Each newly refused root has a failure body and false provider completeness. Refused calls have no process return code or provider turn directory. Inspect `spend_invocations` with `benchmark` or `cost-summary` as above, using `/tmp/subagent-spend-demo-runs`.

For a scripted multi-turn task, the runner retains started turn artifacts, safe partial delta totals, replay data when recorded, and workspace edits. It publishes an incomplete conversation root when later required turns are refused. Multi-turn pricing requires explicit `turn_delta` scope. Cumulative or unspecified dollars remain diagnostic evidence and close later admission unless you supply an assumption. Assumptions do not become provider telemetry. Safe dollars survive failed process exits and rejected response schemas, and rejection diagnostics retain the raw envelope.

Native Claude and built-in subagent backends retain trustworthy dollars captured before a process timeout as an observed subtotal. The whole-call price remains unavailable because the process may incur later cost. Without an assumption, that partial price closes later admission. An assumption charges at least the subtotal, so $0.01 cannot reduce a known $0.06 floor. Timeout artifacts retain return code 124, false provider completeness, and unavailable whole-call cost. Shell capture requires valid original UTF-8 and one strict complete JSON document. Claude capture requires one unambiguous terminal result with no later session content. Bad sibling token fields do not erase independent dollars. Malformed token records remain raw diagnostics.

Claude judges use the same shared price distinction in their admission ledger and verdict rows.
Their timeout member rows retain `observed_subtotal_usd` and unavailable full price. Repeat,
panel, benchmark, and saved cost reports preserve partial known dollars. Two $0.06 judge
floors report a partial $0.12 subtotal and zero whole-price observations. See the
[judge command reference](commands.md#judge-backends).

A zero ceiling starts no callback. With no prior destination content, the runner publishes an incomplete terminal root. With any prior destination content, it preserves every existing file and records this invocation's refusals only in its new ledger. A new invocation does not claim the preserved old output as its own result. Capped recovery anywhere in the selected population rejects the whole batch before runs-root writes.


## Judge calls

`judge` accepts the same ceiling and assumption flags for native backends and `--judge-cmd`.
Each ready task, requested model, and repeat is one paid call. The command checks all evidence
guards before creating its ledger. Guarded results invoke no model and have no paid ledger entry.
An all-guarded batch needs no cost assumption. A zero ceiling retains those results and refuses
all ready calls.

Using the demo answer runs above, limit the offline shell judge to one assumed charge:

```sh
skill-benchmark judge examples/demo-skill/evals/shared-benchmark.json \
  --runs /tmp/spend-demo-runs \
  --judge-cmd "python3 $(pwd)/examples/demo-skill/stub_judge.py" \
  --judge-runs 2 --max-cost-usd 0.02 --assumed-cost-per-run-usd 0.03 \
  --out /tmp/spend-demo-judge.jsonl
```

The command exits 2 after its first ready call. The shell judge supplies no provider dollars,
so its $0.03 charge has the ledger's `assumed` basis. Later ready rows carry
`spend_refusal_reason`, `invocation_state: not_started`, and their effective input hashes.
They have no process return code or served-model evidence. Every requested repeat and panel
member remains in the results. Consensus remains incomplete when any member is refused.

A positive ceiling on ready shell, Codex, Gemini, or Vibe work requires an assumption before
any launch or output-file write. Claude uses reported dollars when available. Missing whole-call
dollars close later admission without an assumption, including a timeout with a safe subtotal.
An assumption cannot lower that subtotal. Assumptions never fill verdict price telemetry.
The command exits 2 for refusals or partial unpriced accounting.

The judge retains the effective prompt and consumed evidence before its first call. Declared
text-only input checks remain separate from trajectory and exploration bindings. Exploration
hashes retain the source view, which excludes oracle names and symlinks. The provider's copy
also excludes agent context files. Each admitted invocation receives a disposable copy of a
private retained template. Provider edits cannot change a later repeat's evidence. Ordinary
text-only judging copies no unrelated files. The template lasts through the batch. All scratch
evidence is removed after success, refusal, or failure.

Settlement precedes verdict parsing and transcript publication. A paid malformed verdict or
failed transcript write retains its settled price. The output error still stops execution.
`benchmark` and `cost-summary` include the invocation ledger and retain the existing recursive
judge price accounting. A refused slot cannot establish a billed whole-price observation.
