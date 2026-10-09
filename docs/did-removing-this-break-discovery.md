# Did removing this description actually break discovery?

`trigger-compare` pairs a baseline matrix with a discovery ablation of the same
skill revision. It reports `refuted`, `indeterminate`, or `confirmed_causal`.
This tutorial produces each class with the offline demo.

## Compare the manifest queries

Run these commands from the repository root after the [test setup](../CONTRIBUTING.md#local-setup).
The examples use `.venv/bin/python3.12.14`. Use your virtual environment's Python
interpreter if its name differs. Each CLI command must exit zero.
A non-zero exit means the workflow failed.

```bash
S=$(mktemp -d /tmp/skill-discovery.XXXXXX)
PY="$(pwd)/.venv/bin/python3.12.14"
M=examples/demo-skill/evals/shared-benchmark.json
"$PY" run_trigger_matrix.py "$M" --agent stub --runs-per-query 3 --out "$S/base.json"
"$PY" run_trigger_matrix.py "$M" --agent stub --runs-per-query 3 \
	--ablation weaker-description --out "$S/abl.json"
"$PY" skill_benchmark.py trigger-compare --baseline "$S/base.json" \
	--ablation "$S/abl.json" --out "$S/compare.json"
```

`weaker-description` removes `when_to_use` from the demo skill's frontmatter.
The stub reads the mounted `description` and `when_to_use`. It fires when the
query shares at least two content words with that text.
The comparison contains these fields, reproduced on 2026-10-09 offline:

```json
{
  "evidence_class": "refuted",
  "summary": {"comparable": 2, "comparable_cells": 2, "blocked": 0, "regressed": 0,
              "availability": "complete", "mean_pass_delta": 0.0}
}
```

The should-fire query matches words in `description`. Removing `when_to_use`
does not change its result. `refuted` means no regression on these queries.
It does not establish that the removed text has no value.

## Ask in the removed field's words

Keep the same query set in both arms. These three queries use words from
`when_to_use` that the description lacks.

```bash
cat > "$S/three.json" <<'QUERIES'
{"queries": [
  {"id": "wtu-inspect-diff", "query": "Can you inspect this diff?", "should_trigger": true},
  {"id": "wtu-serious-code", "query": "How serious is this bug in my code?", "should_trigger": true},
  {"id": "wtu-inspect-code", "query": "Please inspect the code in this diff.", "should_trigger": true}
]}
QUERIES
"$PY" run_trigger_matrix.py "$M" --agent stub --runs-per-query 3 --eval-set "$S/three.json" \
	--out "$S/base-three.json"
"$PY" run_trigger_matrix.py "$M" --agent stub --runs-per-query 3 --eval-set "$S/three.json" \
	--ablation weaker-description --out "$S/abl-three.json"
"$PY" skill_benchmark.py trigger-compare --baseline "$S/base-three.json" \
	--ablation "$S/abl-three.json" --out "$S/compare-three.json"
```

Each command must exit zero. The report has `evidence_class: "indeterminate"`,
three regressed queries, and `mean_pass_delta: -1.0`. Its note explains that
p = 0.25 and at least six consistently regressed queries are needed to confirm.
The inference unit is the authored query. More repetitions sharpen each query's
rate, but do not add queries. Three unanimous drops give p = 2/2³ = 0.25.

Now use the bundled set with two positive queries in the description's words,
six in `when_to_use`'s words, and three negative queries.

```bash
E=examples/demo-skill/evals/trigger-eval-set.json
"$PY" run_trigger_matrix.py "$M" --agent stub --runs-per-query 3 --eval-set "$E" \
	--out "$S/base-full.json"
"$PY" run_trigger_matrix.py "$M" --agent stub --runs-per-query 3 --eval-set "$E" \
	--ablation weaker-description --out "$S/abl-full.json"
"$PY" skill_benchmark.py trigger-compare --baseline "$S/base-full.json" \
	--ablation "$S/abl-full.json" --out "$S/compare-full.json"
```

Each command must exit zero. The baseline passes 24/24 should-fire observations
and 9/9 should-not-fire observations. The ablated arm passes 6/24 and 9/9.
The comparison contains:

```json
{
  "evidence_class": "confirmed_causal",
  "summary": {"comparable": 11, "comparable_cells": 11, "blocked": 0, "regressed": 6,
              "availability": "complete", "mean_pass_delta": -0.5454545454545454}
}
```

`regressed_queries` contains all six `wtu-` queries. The other five have zero
delta. Six unanimous drops give p = 2/2⁶ = 0.03125, below 0.05.
Removing the hints caused this deterministic stub to miss those requests.

## Decide what to change

| Evidence | Action |
|---|---|
| `confirmed_causal` | Restore the removed field or put the required phrases in the description. Run both arms again. |
| `indeterminate`, not significant | Add distinct queries in the phrasing that regressed. |
| `indeterminate`, provenance unverified | Use one checkout, revision, query set, and protocol for both arms. |
| `refuted` | Check whether the queries exercised the removed text. |
| Blocked pairs | Read the reason in `paired.blocked`. Repair incomplete observations or the missing arm. |

## What keeps the measurement honest

The comparison checks the ablation's parent skill hash against the baseline and
requires a matching protocol. Repetitions and model cells become one delta per
query before significance testing. Incomplete observations block pairs.
Only a significant regression can confirm causality. Negative queries regress
when they start firing, because the comparison uses pass rates, not trigger rates.

## Where this stops

The stub proves the offline workflow and its own routing rule. It does not
establish live provider behavior, billing, or containment.
[Activation tuning](tuning-skill-activation.md) covers real agent measurements.
[The trajectory journey](did-my-skill-change-how-the-model-works.md) covers answer
paths after a skill loads. A discovery ablation removes text. Comparing a rewritten
description with its original is a separate, unsupported swap workflow.
