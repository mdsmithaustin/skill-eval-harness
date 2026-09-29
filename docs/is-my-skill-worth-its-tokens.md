# Is my skill worth its tokens?

Every skill you ship rides in the model's context on every request that loads it —
the `SKILL.md`, its frontmatter, and whichever `references/` it pulls in. That is a
standing cost paid on every run, forever. The naive version of the question wants one
number ("my skill adds 9 KB, is that OK?"), but 9 KB is only the *bill*; whether it is
*worth* it is the bill weighed against the **lift** those tokens buy — the with-skill
minus without-skill pass-rate delta from the same paired cases the benchmark already
runs. A skill that adds 4 KB and lifts nothing is worse than one that adds 12 KB and
turns a 0.2 pass rate into 0.9. So the question is not "how big is it" but "what is the
lift per token, and is any of that footprint buying nothing?"

That splits into two measurements, and they need different evidence:

- **Static footprint** is deterministic and free — no model, no run. `profile-skill`
  counts it. This is the numerator's denominator: the tokens you pay unconditionally.
- **Runtime lift and dollar cost** need real runs with telemetry. `token-overhead`
  joins the static footprint to the measured objective lift and (when the runner
  recorded it) the dollar delta; `cost-summary` rolls up spend across a whole suite.
  These are only as real as the token/cost numbers your runner actually wrote.

## Run the static half offline

The static footprint costs nothing to measure and needs no runner. On the bundled
demo ([`examples/demo-skill/`](../examples/demo-skill/)):

```bash
cd examples/demo-skill
python3 ../../skill_benchmark.py profile-skill evals/shared-benchmark.json --format markdown
```

Real output (2026-07-05, Python 3.11):

```text
# Skill profile — demo-reviewer

## Summary

| Metric | Value |
|---|---:|
| skill_files | 1 |
| skill_tokens | 144 |
| reference_files | 1 |
| reference_tokens | 51 |
| modules | 2 |

## Findings

- No profile findings.
```

That is the whole standing cost of the demo skill: 144 tokens of `SKILL.md` plus 51
tokens of one reference, two loadable modules. `profile-skill` is where a "my skill is
getting big" worry starts — set `--max-skill-tokens` / `--max-reference-tokens` /
`--max-references` and it emits a finding when a component crosses your budget, so a
reference that has quietly grown past its keep shows up here before you pay for a run.

## Run the runtime half — and see why the demo can't fake it

Now join footprint to lift. `token-overhead` reads the same paired runs the benchmark
graded and reports lift-per-token and lift-per-dollar per skill:

```bash
cd examples/demo-skill
python3 ../../skill_benchmark.py prepare evals/shared-benchmark.json --split tune \
  --runs-per-variant 2 --out /tmp/demo-tasks.jsonl
python3 ../../skill_benchmark.py run-codex --tasks /tmp/demo-tasks.jsonl \
  --runs /tmp/demo-runs --codex-cmd "python3 stub_runner.py"
python3 ../../skill_benchmark.py token-overhead evals/shared-benchmark.json \
  --runs /tmp/demo-runs --format markdown
```

Real output against the offline stub runs (2026-09-29):

```text
# Token overhead report

| Skill | Static SKILL tokens | Reference tokens | Runtime pairs | Mean total delta | ... | Mean cost delta USD | Lift per $ | Saturated/no-lift cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| demo-reviewer | None | None | None | None | ... | None | None | None |
```

**Every column is `None` — but not because the runs are empty.** `c-review` carries one
`type: "judge"` assertion (`actionable-review`), and `token-overhead` checks design
completeness through the same gate `benchmark` uses. That gate reports
`deferred_judge_verdicts` for any manifest with an ungraded qualitative assertion, and
`token-overhead` has no `--judge-results` flag to clear it — so this top-level row stays
`None` for a manifest with a judge assertion whether or not you ran `judge` separately.
The real numbers are one level down, in `reports[0].summary.observed`:

```json
"paired_runtime_rows": 4,
"total_token_delta": {"mean": null, "n": 0},
"objective_delta": {"mean": 1.0, "n": 4}
```

`objective_delta` is real: the skill wins every paired run (`with_skill` passes,
`without_skill` fails). `total_token_delta` stays `null` for a different reason than
missing telemetry — the stub *does* report tokens (0, honestly), but a token or cost
delta also requires both arms to agree on a `model` identity, and the stub never sets
one. Missing model identity blocks the comparison (`reason: "basis_missing"`) the same
way missing usage would.

`cost-summary` shows the usage half is actually present:

```bash
python3 ../../skill_benchmark.py cost-summary \
  --manifest evals/shared-benchmark.json --runs /tmp/demo-runs
```

```json
"coverage": {
  "runs_seen": 8,
  "runs_with_token_usage": 8,
  "runs_with_dollar_cost": 0,
  "runs_with_non_usd_cost": 0,
  "runs_missing_usage": 0,
  "runs_missing_cost": 8
}
```

Eight runs on disk, all eight carrying token usage — the stub reports an honest zero
(`source: "trace_normalized"` in each run's `metrics.json`), which the ledger counts as
*available*, not missing. A missing number and a zero number are different claims, and
`runs_with_token_usage` only counts the latter as present. Only *cost* is missing here
(`runs_missing_cost: 8`): the stub never emits a `cost_normalized` block at all, so that
half is correctly `source: "missing"`.

To get real runtime numbers — including a bound token/cost delta — run the same cases
through a runner that records a model identity alongside usage. `run-claude` parses the
`claude -p` JSON envelope and records `usage_normalized` / `cost_normalized` with a
`model` field; the Pi smoke runner does the same. Re-run `token-overhead` / `cost-summary`
against *those* runs and the blocked deltas above become real numbers.

## Reading the numbers, symptom by symptom

Once the runtime pairs are real, read the row for the keep/trim/cut decision:

- **High `Lift per 1k total tokens` / `Lift per $`** → the footprint is earning its
  keep. Leave it. This is the case the skill exists for.
- **Positive footprint, `Mean objective lift` ≈ 0** → you are paying tokens for nothing
  measurable. Either the cases are **saturated** (the base model already passes them —
  the benchmark's `saturated`/`no-lift` case flags catch this) so the eval can't *see*
  the lift, or the skill genuinely isn't helping. Check the flags before you cut:
  saturation is an eval problem, no-lift is a skill problem. `Saturated/no-lift cost
  USD` totals exactly the spend on cases that bought no lift — that column is the
  trim list.
- **Large `Reference tokens`, small lift** → suspect a reference. `profile-skill`
  tells you which module carries the bytes; drop it from the skill, re-run, and if the
  lift holds, the reference was dead weight. (This is a footprint ablation you can do
  by hand; the [ablation study](ablation-study-walkthrough.md) does the causal version.)
- **`audit-manifest --runs <dir>`** folds the same signal into review findings —
  expensive-but-saturated cases, high-cost judge-only cases with no deterministic
  oracle — so the cost view shows up next to the manifest hygiene view.

A worked scale for what "expensive" looks like on real models: a 2026 ten-skill suite
run recorded in [issue #21] billed **$175.21** across 2,568 generation calls, and a
*single* skill — `swiss-poster-skill` at 1,272 runs and ~44.8M tokens — was **~$124**
of it. The ablation arms alone were $157.91 against $17.30 for the plain with/without
baseline. That is the shape of the decision: most of the money is in a few skills and
in the ablation matrix, so "is my skill worth its tokens" is usually really "is *this*
skill, on *these* cases, at *this* repeat count, worth its share of the suite."

## What keeps the measurement honest

- **Footprint is deterministic; lift is not.** `profile-skill` is exact and repeatable.
  A single-shot lift number is a coin flip — the same repetition discipline the rest of
  the harness insists on applies here. Read a one-run `token-overhead` delta as a hint,
  not a verdict.
- **Missing telemetry is not zero cost.** The ledger's `source: "missing"` /
  `not_applicable` markers exist because a runner that forgot to record usage would
  otherwise report a skill as free. If `runs_with_dollar_cost` is below `runs_seen`,
  your dollar totals are *underreported*, not low — fix the runner before you trust the
  bill.
- **Provider-reported beats estimated.** When a runner records the provider's own cost,
  the ledger stamps `source: "provider_reported"`; a price-table estimate is stamped
  `price_table_estimated` with its table version. Don't compare a provider-reported
  total to an estimated one and call it a regression.
- **Exclude execution errors from the lift, keep them in the bill.** A timed-out run
  cost money but proves nothing about quality; the reports count it in operational cost
  and drop it from the pass-rate denominator.

## Where this stops

This journey decides whether a skill's footprint is *worth it on the cases you have*.
It does not tell you which single component is load-bearing — that is a causal claim,
and cutting a reference and eyeballing the lift is not the same as a provenance-gated,
significance-tested removal. When `token-overhead` says a reference looks like dead
weight, confirm it with a materialized ablation in the
[ablation study walkthrough](ablation-study-walkthrough.md) before you delete it.

[issue #21]: https://github.com/adewale/skill-eval-harness/issues/21
