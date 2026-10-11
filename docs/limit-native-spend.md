# How do I limit paid calls at a dollar ceiling?

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

For capped batches, every selected task must be an ordinary native answer task. A recovery row anywhere in the batch rejects the command before provider calls or writes to the runs root. Uncapped recovery keeps its existing phase behavior. The native rules above apply to the serial native answer commands. `run-subagent` has the external-turn rules below. Judge calls have the rules below. Jetty submissions have the rules below. Trigger matrices and Pi trigger evaluations use the bounded concurrent rules below.

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
The command exits 2 for an incomplete observation or partial unpriced accounting.

Missing results use `judge_observation_kind: missing`, incomplete availability, and a compatibility
`consensus` false shell without a score. A paid scored result keeps its score even beside a refusal.
An incomplete group carries all members and no aggregate score or agreement. Saved per-step
consensus validates every leaf's exact steps, minimum, and trajectory binding before using the
parent's decision and median.

`cost-summary.judge.requested_calls` partitions into `billed_calls`, `not_started_calls`,
`nonbillable_calls`, and `unverified_calls`. Billed counts affirmative launches, including unknown
and zero dollars. Guards and spawn failures are nonbillable. Historical rows without launch facts
are unverified. `requested_calls_basis` is `requested_slots` for fresh calls and `retained_leaves`
when the historical requested population is unknown; that uncertainty makes `counts_availability`
partial. The demo has four requested slots, one launch, and three refusals. Its assumed charge
stays in the ledger, while verdict price telemetry remains unavailable.

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

## Jetty submissions

`run-jetty` accepts the ceiling and assumption flags. It validates every selected payload,
attested upload, route, saved result, and output slot before creating a ledger or replacing
output. Already uploaded tasks use their stored contract and receipt. Their original local
files need not remain available.

Apply a ceiling to exported executable payloads:

```sh
skill-benchmark run-jetty --payloads jetty-payloads.jsonl --out jetty-runs.jsonl \
  --max-cost-usd 1 --assumed-cost-per-run-usd 0.10
```

This command requires `JETTY_API_TOKEN` and can incur provider spend. The assumption is a
ledger policy amount. If a response establishes a $0.60 cumulative floor, the ledger charges
at least $0.60. The assumption never becomes provider telemetry.

A fresh zero ceiling makes no HTTP request, uploads no bundle, and appends no submission
attempt. Refused rows have `invocation_state: not_started`, requested routing, and the
prospective paid call identity. They have no provider status, return code, served model, or
trajectory acknowledgement. An explicit unknown replacement refused at zero preserves the
old journal bytes and reports its prior receipt separately. `import-jetty-results` accepts
refusal rows as incomplete metadata without creating an answer or provider execution facts.

Polling, artifact retrieval, and durable replay reconcile earlier attempts without new
admission. A zero ceiling permits those operations. Replay makes no HTTP request and emits
the original saved record without changing its hash. A reconciliation-only invocation has
an empty paid plan. Historical incomplete accounting does not make its current ledger partial.

The ledger first records `in_flight`. Its callback uploads when needed, persists a new UUID
at the journal's `submitting` transition, and submits. Uncapped submissions also receive a
UUID and create no spend directory. Unknown submissions never retry automatically.
`--resubmit-unknown` requests a distinct admitted attempt and retains the predecessor's
identity and receipts. Internal HTTP 429 retries keep one logical attempt. Final 429 and
possibly accepted 5xx responses do not prove zero billing.

Supported `cost` and `cost_usd` fields, including nested `usage` aliases, are cumulative
provider-reported observations. Equal aliases count once. Conflicting or malformed dollar
aliases contribute no invented amount. Valid earlier floors survive later conflicts, missing
fields, malformed sibling tokens, and local artifact errors. Floors use the maximum safe
observation, never the sum of snapshots. Terminal execution status does not establish complete
billing. Jetty's current provider contract supplies no billing finality marker, so current
records and imported telemetry leave the complete price unavailable. `benchmark` and
`cost-summary` exclude these floors from complete observed dollar totals. The floor remains
in `jetty_price_evidence.observed_subtotal_usd`, imported `observed_subtotal_usd`, and the
current invocation ledger. Historical replay retains its original cost claims and the
importer labels their billing finality `historical_unverified`.

Every capped invocation writes `<out>.spend/<invocation-id>/spend-ceiling.json`. Earlier
ledgers remain immutable when later reconciliation adds journal evidence. Inspect those
admission snapshots separately from provider receipts. They describe the same attempt and
must not be added as separate charges. Missing complete price closes later fresh admission
unless an assumption applies. Earlier reconciliation and replay continue. Current refusal
or partial accounting exits 2. Other lifecycle exits remain unchanged.

Journal schema 2 stores digest-keyed tasks with shared upload facts and ordered submission
attempts. Schema 1 loads without a write. A real transition persists migration while retaining
surviving receipts, exact record hashes, opaque abandonment counts, and unclassified metadata.
Its scoped `legacy_current` reference identifies the surviving local receipt. It cannot recover
an original submission UUID or vanished prior attempts and costs. Load, replay, and refusal
alone do not rewrite that journal.

Persistence failures and interruptions stop execution. When settlement publication succeeds,
known floors settle before a deferred error propagates. A failed submitting checkpoint sends
no submit request. A hard process kill or settlement failure can leave the ledger `in_flight`.
No local error proves free model execution. Output publication still precedes journal commit.


## Trigger matrices and Pi evaluations

Both `skill-trigger-matrix` and `skill-pi-trigger-eval` accept `--max-cost-usd` and
`--assumed-cost-per-run-usd`. Each agent, requested model, query, and repetition is one
call. The complete population remains in the report. The ledger contains only potentially
paid calls and declares `scope: paid_matrix_calls`.

Run the deterministic offline matrix at a zero ceiling:

```sh
skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json \
  --agent stub --model default --runs-per-query 2 --workers 2 \
  --max-cost-usd 0 --out /tmp/trigger-spend-demo.json
```

The concrete built-in Stub runs all four repetitions and exits 0. It invokes no model.
A replacement registered under the same name receives no offline authority. A zero ceiling
refuses every potentially paid repetition without mounting its workspace or invoking its
provider. Refused rows have `execution_state: not_started`, null trigger and pass verdicts,
and no process return code, timing, or invocation fields.

The command validates every selected route, model, query scope, treatment, protocol, and cost
capability before ledger or trace writes and cell execution. Preparing the shared treatment
requires one temporary tree. Under a positive ceiling, Codex and Vibe need an explicit
assumption. Claude and Pi use their supported dollar envelopes when those envelopes establish
complete billing. Missing prices close later paid admission unless an assumption applies.
Assumptions charge at least a retained safe subtotal and never become provider telemetry.

One controller keeps at most `--workers` cells active across paid and offline work. It settles
all ready cells in population order before refilling a slot and checks readiness again before
each admission. Completions after that last empty scan belong to the next turn. Active calls
can overshoot the ceiling. Reached ceilings and unknown prices stop later paid calls while
trusted offline cells continue. Reaching the ceiling on the final cell alone is not refusal.

The supervisor retains private original process facts before redaction, parsing, detection,
release, and cleanup. Independent Claude and Pi dollar readers do not normalize sibling token
fields. Ambiguous, duplicate, malformed, or invalid UTF-8 output cannot establish complete
billing. Timeout and unfinished billing evidence can retain a safe floor. Codex and Vibe
cost-shaped fields do not establish a supported dollar contract.

Settlement precedes trace publication. A failed discovery stage retains its real process facts
in `original_invocation` with missing discovery telemetry and null verdicts. Ordinary trace
write errors retain the existing `trace_error` diagnostic. Required ledger publication failures
stop all replenishment and drain active work. Matrix directory sync treats supported POSIX
failures as fatal, while recognized unsupported-directory operations retain the platform limit.
Serial answer, subagent, judge, and Jetty publication behavior stays unchanged.

The report's `spend` block names its ledger under `<out>.spend/<invocation-id>/spend-ceiling.json`.
An interrupted or fatally stopped execution attempts to write a full-population report before
propagating the original stop. Its spend snapshot is labeled `in_memory_after_abort` and does
not prove that a settlement reached disk. The last durable ledger may remain `in_flight`.
Signals keep their existing exits. Budget refusal exits 2, other incomplete measurement exits
1, and complete measurement exits 0. Incomplete cohort summaries omit every rate field.
Saved trigger comparison retains labeled observed diagnostics, blocks complete causal
inference for incomplete arms, and keeps its existing exit 0.

The ceiling and assumption do not enter the prompt, treatment, or protocol identity. Pi uses
the same matrix operation and retains its existing model, repetition, and timeout defaults.
