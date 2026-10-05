# How do I gate my skill repo's CI on this?

You have a skill repo and a benchmark that passes today. The question is how to make CI
*stay* honest — fail a PR that regresses the skill, or that ships a manifest too weak to
catch a regression at all. The instinct is to treat it like a unit test: one green
check, merge on green. But an eval is not a test (see
[evals-are-not-tests.md](evals-are-not-tests.md)) — a single run is a sample, and a
manifest can be green because it is *good* or because it is *too weak to fail*. So a
useful gate has two independent jobs, and they key on different things:

1. **Do the selected checks pass?** — turn `benchmark.json` into a CI-native
   pass/fail with `report --fail-on-failures --format junit|github`.
2. **Is the manifest itself strong enough to trust the green?** — `audit-manifest
   --fail-on-blockers` fails when the suite has structural blockers (no adversarial
   coverage, a leak-saturated case, an instruction-simulated ablation masquerading
   as evidence).

The saved report gate and manifest audit do not call a model. Objective grading
also runs without a model. A live `judge` step calls its configured model and needs
that provider's credentials. The bundled demo uses a local stub for its judge.

## Run both gates offline on the demo

Grade the bundled demo the normal way, then serialize the result for CI. This is the
whole loop, runnable with no key:

```bash
cd examples/demo-skill
HARNESS=../../skill_benchmark.py

# (assumes /tmp/demo-runs exists from the demo README's prepare + run-codex + judge
#  steps — c-review carries a judge assertion, so skipping judge leaves grading
#  "partial" and `report` below prints "incomplete experiment evidence" instead)
python3 $HARNESS benchmark evals/shared-benchmark.json --runs /tmp/demo-runs \
  --variant with_skill --variant without_skill \
  --judge-results /tmp/demo-judge-results.jsonl --out /tmp/demo-benchmark.json

python3 $HARNESS report --benchmark /tmp/demo-benchmark.json --format github \
  --fail-on-failures
```

Real output (2026-09-29, Python 3.12, the demo README's 6-run walkthrough):

```text
# Skill eval — demo-reviewer

**Lift (with − without, objective):** 1.00 − 0.00 = **1.00**

| variant | cases | runs | mean objective | mean combined | missing | exec errors |
|---|---|---|---|---|---|---|
| with_skill | 2 | 12 | 1.00 | 1.00 | 0 | 0 |
| without_skill | 2 | 12 | 0.00 | 0.00 | 0 | 0 |
```

`--format github` writes a job-summary table (and annotations) straight into a GitHub
Actions run. `--format junit` writes the same result as JUnit XML, one `<testcase>` per
case/variant/run, which a CI JUnit viewer can render (trimmed to the
first two runs of each case/variant; the demo's 6 runs each produce `run-1` .. `run-6`):

```text
<testsuite name="skill-eval:demo-reviewer" tests="24" failures="12" errors="0" ...>
  <testcase classname="demo-reviewer.c-review.default-model" name="default-model/with_skill/run-1" />
  <testcase classname="demo-reviewer.c-review.default-model" name="default-model/with_skill/run-2" />
  <testcase classname="demo-reviewer.c-review.default-model" name="default-model/without_skill/run-1">
    <failure message="3 failing check(s)">severity-label: none matched: ['Blocking', 'Minor', 'Clean']
cite-checklist: none matched: ['file and line']
actionable-review: no justification for the finding, or the concrete gap (the missing test) is never named</failure>
  </testcase>
  <testcase classname="demo-reviewer.c-review.default-model" name="default-model/without_skill/run-2">
    <failure message="3 failing check(s)">severity-label: none matched: ['Blocking', 'Minor', 'Clean']
cite-checklist: none matched: ['file and line']
actionable-review: no justification for the finding, or the concrete gap (the missing test) is never named</failure>
  </testcase>
  <testcase classname="demo-reviewer.c-adversarial.default-model" name="default-model/with_skill/run-1" />
  <testcase classname="demo-reviewer.c-adversarial.default-model" name="default-model/without_skill/run-1">
    <failure message="1 failing check(s)">severity-label: none matched: ['Blocking', 'Minor', 'Clean']</failure>
  </testcase>
</testsuite>
```

`c-review`'s failures now list all three assertions (the two objective checks plus the
`actionable-review` judge verdict) because the benchmark above passed `--judge-results`.
Drop that flag and `--format junit` still emits the objective testcases (minus the judge
assertion) plus one extra `<testcase classname="demo-reviewer.experiment"
name="answer-design-coverage">` `<error>`; `--format github` is stricter and blanks the
whole table instead — `**Experiment status:** incomplete (deferred judge verdicts, blocked
grading evidence)`, lift `— − — = —`, and an `::error title=skill-eval demo-reviewer::`
annotation.

The `without_skill` failures are *expected* here — that arm exists to prove the skill is
what passes the cases. Which is the first subtlety of gating an eval: you do not gate on
"all testcases green." `--fail-on-failures` checks `with_skill` by default. It
requires complete evidence across the experiment and passing applicable non-soft
checks in the selected arm.
It preserves critical vetoes and reference-floor failures. Baseline assertion
failures remain visible without rejecting that default gate.

Use repeatable `--gate-variant` options to select other arms. For example, add
`--gate-variant with_skill --gate-variant ablation:preserve-behavior` to require both.
This replaces the default selection. A destructive ablation belongs in the gate
only when you intend its assertions to pass.

The command renders identical bytes with or without the gate. Exit 0 means the
selected checks pass. Exit 1 means incomplete evidence or a selected failure.
Exit 2 means invalid arguments, unreadable or malformed JSON, or an unrenderable
report shape. Diagnostics go to stderr, including when `--out` writes a report.
A missing baseline, a deferred judge, a duplicate identity, or blocked grading in
any arm rejects the gate. An absent selected variant also rejects it. Pairing uses
the report's existing contrast rules. A report with a missing control arm stays
partial. A report with no applicable presence contrast can remain complete.

The gate reads saved evidence only. It does not reopen artifacts or verify their
current contents. It applies no lift, rate, or statistical significance threshold.
Without `--fail-on-failures`, `report` retains its render-only exit behavior.

## The second gate: is the manifest strong enough?

A benchmark can be green because the manifest can't fail. `audit-manifest` scores that,
and `--fail-on-blockers` turns it into an exit code:

```bash
python3 $HARNESS audit-manifest evals/shared-benchmark.json --fail-on-blockers
echo "exit=$?"
```

Real output (2026-07-05) — the demo is a *ready* manifest, so it passes:

```text
exit=0
```

with a readiness block reporting:

```json
"readiness": {
  "ablations": { "total": 3, "materialized": 3, "instruction_simulated": 0 },
  "leak_saturated_cases": [],
  "blockers": []
}
```

`--fail-on-blockers` keys on `readiness.blockers` — the structural problems that make a
green meaningless: no adversarial cases (nothing tests whether the skill holds under
pressure), a leak-saturated case (an assertion the base model passes from the prompt
alone), an ablation that is only *instruction-simulated* and so can never confirm a
causal regression, or — once run data is supplied — a base-saturated case. The demo has
none, so it gates clean.

Note the distinction the exit code draws: `audit-manifest` *also* emitted eight
`findings` at `recommended`/`required` severity on this same run (missing domain tags,
missing difficulty tags, …). Those are advice, not blockers — `--fail-on-blockers`
deliberately does **not** fail on them, so your CI fails on "this suite can't be trusted"
without nagging on "this suite could be richer." Add `--strict-judge` to also fail when
the declared judge model is the model under test.

## A workflow that ties it together

The recipe for a skill repo's `.github/workflows/`:

```yaml
- name: Grade skill eval
  run: |
    skill-benchmark judge evals/shared-benchmark.json \
      --runs eval-runs/latest --variant with_skill --variant without_skill \
      --judge-cmd "$JUDGE_CMD" --out judge-results.jsonl
    skill-benchmark benchmark evals/shared-benchmark.json \
      --runs eval-runs/latest --variant with_skill --variant without_skill \
      --judge-results judge-results.jsonl --out benchmark.json
    skill-benchmark report --benchmark benchmark.json --format github \
      --fail-on-failures >> "$GITHUB_STEP_SUMMARY"

- name: Fail if the manifest is too weak to trust
  run: skill-benchmark audit-manifest evals/shared-benchmark.json --fail-on-blockers
```

Skip the `judge` step only when no case in the manifest carries a judge assertion; a
`benchmark` run without `--judge-results` against a manifest that does leaves grading
`"partial"` for every judged case, exactly like the demo walkthrough above.

For a full-suite gate across many skills, `suite-run` adds a preflight with cost
ceilings (`--max-estimated-cost-usd`) so a PR job can refuse to start a run that would
blow its budget — the operational half of the same gate.

## Reading a failing gate, symptom by symptom

- **`report` shows lift dropped vs. the last run** → a real regression, or a flaky
  sample. The `benchmark.json` `significance` block (a sign-flip permutation test on the
  paired scores) tells you which. Gate on *confirmed* regressions, not on a one-run dip;
  an ablation cohort needs ≥6 exact repetition pairs to clear its two-sided sign-flip gate.
- **`audit-manifest --fail-on-blockers` exits non-zero** → read the `blockers` list. A
  `leak-saturated` blocker means an assertion passes from the prompt alone; a
  no-adversarial blocker means nothing tests the skill under pressure. Fix the
  manifest, not the threshold. (Missing hidden splits surface as a `required`
  *finding*, not a blocker — advice the exit code deliberately does not fail on.)
- **JUnit shows `errors` > 0** → experiment evidence is incomplete. Inspect answer
  design, execution, grading, and pairing evidence. A crash also appears as a run
  failure. Repair missing evidence before interpreting quality results.
- **Everything green but lift ≈ 0** → the gate is passing on a saturated suite. The
  benchmark's `saturated`/`no-lift` case flags are the tell; a suite that can't fail
  isn't guarding anything.

## What keeps the gate honest

- **Saved evidence checks do not call a model.** `report` and `audit-manifest`
  inspect existing evidence. `benchmark` merges saved judge verdicts. The explicit
  runner and live `judge` steps call models. Opt-in script assertions execute their
  configured commands.
- **Select the variants whose checks must pass.** The default gate checks
  `with_skill`. Expected `without_skill` and destructive ablation failures remain
  visible. Lift and significance require a separate policy decision.
- **A green benchmark is not a green skill-loads.** The answer runners force-load the
  skill; passing them says nothing about autonomous activation. If activation matters for
  your gate, add a `skill-trigger-matrix` check — see
  [tuning-skill-activation.md](tuning-skill-activation.md).
- **`--fail-on-blockers` gates trust, not taste.** It fails on structural blockers that
  void the measurement, and stays quiet on `recommended` findings, so the gate means
  "this result is trustworthy," not "this suite is perfect."

## Where this stops

This journey gets a PR to fail on a regression or an untrustworthy manifest. It does not
decide *whether the regression is worth blocking on* — a confirmed drop on a
regression-guard case is a hard stop, but a soft-severity dip may be acceptable. That
judgment lives in the severity tiers you set on each assertion
([authoring-evals.md](authoring-evals.md)); this gate only enforces the tiers you
already chose.
