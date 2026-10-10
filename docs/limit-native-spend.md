# How do I stop a native answer batch at a dollar ceiling?

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

The harness publishes `in_flight` before dispatch and publishes the charge after the callback. An abrupt process kill can leave unresolved `in_flight` evidence. The ledger then reports partial spend. A raised callback exception also retains an attempted, unpriced receipt, or a labeled assumption if one was supplied.

For capped batches, every selected task must be an ordinary native answer task. A recovery row anywhere in the batch rejects the command before provider calls or writes to the runs root. Uncapped recovery keeps its existing phase behavior. These flags apply only to the native serial answer commands. Subagent turns, judges, Jetty submissions, trigger matrices, and Pi trigger evaluations have their existing execution policies.
