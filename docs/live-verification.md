# Live verification

CI never runs a real agent. These checks need a machine with credentialed CLIs, so an agent or a
person runs them by hand after any change to a runner, an adapter or the stop, served-model or
effort checks, and after a CLI upgrade. Each check gives the command, what output proves the code
right, and what to send back when it does not.

## Before you start

Fixed prepared recovery rows have a separate [capability and evidence contract](recovery.md).
Their model-free process tests do not establish live provider eligibility.
Before any consumer certification, inspect actual checkpoint write and active-turn traces, wrapper/runtime identity, evidence protection, and enforcing refusal.
Existing read-only backend restrictions remain unchanged and may block the checkpoint write.
Do not relax permissions or infer denial from an absent file to make a recovery case pass.
Live recovery eligibility is deferred to consumer integration and is not a release claim of this runner feature.

- Check out the branch under test, run `python3 -m pip uninstall -y skill-eval-harness`
  (it prints a warning and exits 0 when the old distribution is not installed), then
  `python3 -m pip install -e ".[test]"` and `python3 -m unittest discover tests` (it must
  pass before any live run).
- Record the versions you run against: `claude --version`, `codex --version`, `pi --version`,
  `vibe --version`, `gemini --version`, and the OS. Put them in your report.
- Use the cheapest model each CLI offers unless a check says otherwise. Every check below is a
  few runs; the whole list costs a few dollars at most.
- Work under a scratch directory such as `/tmp/live`. Never commit a credential; before you
  commit any recorded file, read it for tokens, emails, home paths and session ids.
- Prepare the demo tasks once:

  ```
  skill-benchmark prepare examples/demo-skill/evals/shared-benchmark.json --out /tmp/live/tasks.jsonl
  ```

## Reporting back

For each check, report: the command, the CLI version, pass or fail, and for a failure the
relevant lines of `metadata.json`, the report JSON or stderr. Commit only recorded fixtures (with
their provenance file) and code fixes; put the rest in a PR comment.

## 1. The live smokes

Each smoke runs only when its variable is set, and fails (never skips) when the CLI or its
credentials are missing. Run them from the repository root:

```
RUN_TRIGGER_SMOKE=1 python3 -m unittest discover -s tests -p 'test_trigger_matrix.py' -k ClaudeMatrixSmokeTests -v
RUN_CODEX_TRIGGER_SMOKE=1 python3 -m unittest discover -s tests -p 'test_trigger_matrix.py' -k CodexMatrixSmokeTests -v
RUN_PI_TRIGGER_SMOKE=1 python3 -m unittest discover -s tests -p 'test_trigger_matrix.py' -k PiMatrixSmokeTests -v
RUN_VIBE_TRIGGER_SMOKE=1 python3 -m unittest discover -s tests -p 'test_trigger_matrix.py' -k VibeMatrixSmokeTests -v
RUN_AGENT_INVOKE_SMOKE=1 python3 -m unittest discover -s tests -p 'test_trigger_matrix.py' -k AgentInvokeSmokeTests -v
RUN_GEMINI_SMOKE=1 python3 -m unittest discover -s tests -p 'test_gemini_backend.py' -k GeminiLiveSmokeTests -v
RUN_JETTY_SMOKE=1 JETTY_API_TOKEN=… python3 -m unittest discover -s tests -p 'test_smoke_jetty.py' -k JettyLiveSmokeTests -v
```

Right: each enabled smoke passes. A failure is a finding; report the CLI version and the failure.

## 2. Claude answer runs: stop reason, served model, effort

1. `skill-benchmark run-claude --tasks /tmp/live/tasks.jsonl --runs /tmp/live/rc --model claude-haiku-4-5 --effort high`
   - Right: every `metadata.json` shows `stop_class: completed`, `stop_reason: end_turn`, a dated
     `served_models` entry, `served_model_check: match`, and `effort: {requested: high,
     applied_by: "claude --effort"}`; `environment.json` shows `--effort high`, and `stderr` has no
     `Unknown --effort value` line.
2. Truncation: repeat one long-answer case with `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64` in the
   environment.
   - Right: `stop_class: truncated`, `stop_reason: max_tokens`, and `skill-benchmark benchmark`
     marks the run `unscorable_reason: stopped:truncated`. If Claude Code ignores the variable,
     report the result event's `stop_reason` instead.
3. Aliases: run one task with `--model 'sonnet[1m]'`, and with a `-latest` alias if the account
   serves one.
   - Right: `sonnet[1m]` reads `served_model_check: match`; a `-latest` request reads
     `unverifiable` and the run is scored. A `mismatch` on either is a bug in
     `completion_contracts.ServedModel`; send the `served_models` value.
4. Effort levels: `claude --help | grep -A1 -- --effort` must list exactly `low, medium, high,
   xhigh, max` (otherwise update `ClaudeBackend.effort_levels`), and
   `skill-benchmark run-claude … --effort minimal` must exit 1 with `claude --effort accepts low,
   medium, high, xhigh, max; got minimal` and write no runs directory.

## 3. run-subagent

1. `skill-benchmark run-subagent --tasks /tmp/live/tasks.jsonl --runs /tmp/live/rs --model claude-haiku-4-5`
   - Right: the same `stop_class`, `stop_reason`, `served_models` and `served_model_check` values
     as the `run-claude` runs in check 2.1.
   - Each with_skill `metrics.json` reads `source: "claude"` and `trace_observation_complete:
     true`, with the same `tool_calls` and `skill_invoked` as the `run-claude` run, and
     `environment.json` shows the run's workspace as the working directory, not a temp dir.
2. Repeat check 2.2 under `run-subagent`: right if the run is `truncated` and unscorable.
3. A case with `turns`: each `turn-N/metadata.json` carries its own stop and served model, the
   run's `served_models` lists every turn's, and `multi_turn_telemetry` reads `complete` for
   usage and cost, with run totals equal to the sum of the turns'.

## 4. Codex

1. Record `codex --version`. Run `skill-benchmark run-codex --tasks /tmp/live/tasks.jsonl --runs
   /tmp/live/cx --effort max` and again with `--effort minimal` on the default model.
   - Right: both complete. If Codex warns about, ignores or clamps a level, send the stderr: the
     harness cannot see that, and `CodexBackend.effort_levels` should then drop the level.
2. Trigger cells: `skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json --agent
   codex --runs-per-query 3 --out /tmp/live/cx-base.json`, the same with `--ablation
   weaker-description --out /tmp/live/cx-abl.json`, then `skill-benchmark trigger-compare
   --baseline /tmp/live/cx-base.json --ablation /tmp/live/cx-abl.json`.
   - Right: no cell blocked as `protocol_observation_unsafe`; rows carry `codex_home_files`.

## 5. Claude trigger runs: login from the environment, mount name

1. With `CLAUDE_CODE_OAUTH_TOKEN` set (and again with `ANTHROPIC_AUTH_TOKEN`), and with
   `CLAUDE_CODE_SYNC_SKILLS=1` also set:

   ```
   skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json --agent claude --model haiku \
     --runs-per-query 3 --trace-runs /tmp/live/t-base --out /tmp/live/base.json
   skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json --agent claude --model haiku \
     --runs-per-query 3 --ablation weaker-description --out /tmp/live/abl.json
   skill-benchmark trigger-compare --baseline /tmp/live/base.json --ablation /tmp/live/abl.json
   ```

   - Right: every row's `protocol_observation` is `{"config_isolated": true,
     "claude_config_outside_workdir": true}`, `trigger-compare` blocks no cell, rows are
     `observation_complete`, and `competing_skills` lists only Claude Code's bundled skills.
     `grep -r "$CLAUDE_CODE_OAUTH_TOKEN" /tmp/live/base.json /tmp/live/t-base` finds nothing.
   - Wrong: rows incomplete with "not logged in" means environment auth does not carry into an
     empty config.
2. Mount name: in `/tmp/live/t-base/**/trace.jsonl`, the init event's `skills` list contains
   `demo` and not `skills_demo_SKILL.md`, any `Skill` call names `demo` or `demo-reviewer`, and
   should-fire rows show `Skill tool invoked: demo`.

## 6. Record Claude's trailing record

`python3 scripts/record_claude_stream.py --model haiku` (Claude Code 2.1.269 or later).

- Exit 0 prints the records that follow `result` (expected `system`/`task_summary`) and writes
  `tests/fixtures/claude/<name>.jsonl` plus `<name>.provenance.json`. Exit 1 means no record
  followed `result`: retry, or pass a `--prompt` that starts a background task.
- Check the rule for what may follow `result` against it:
  `python3 -c "import json, skill_benchmark as sb; r = [json.loads(l) for l in open('tests/fixtures/claude/<name>.jsonl')]; print(sb.claude_terminal_result_index(r))"`
  prints an index, not `None`. Note which record types follow `result`.
- Read the `.jsonl` for ids, paths, emails or tokens before committing it, then run
  `python3 -m unittest discover tests`: the trailing-record tests pick the recording up and show
  subTests labelled `recorded <name>.jsonl`. A failure there is real evidence against the
  trailing-record rule; send it with the recording.

## 7. Vibe

1. `skill-benchmark run-agent --agent vibe --tasks /tmp/live/tasks.jsonl --runs /tmp/live/rv --model devstral-small-latest`
   - Right: `stop_source` reads `vibe output carries no stop reason`, and
     `grep -iE 'stop|finish' /tmp/live/rv/**/trace.jsonl` finds no stop field. If one exists, send
     the record so it can be mapped. Record `vibe --version` and the shape of one record.
2. `run-agent --agent vibe … --effort high` must be refused before any run.
3. Vibe 2.23 and later write public history entries, which the dialect reads from fixtures built
   from Vibe's source, not recorded. Record a real stream with the steps in
   `tests/fixtures/vibe/README.md` ("Replacing them with a recording") and run its check script.
   Right: no parse or protocol errors, the skill evidence names the skill, and the answer is the
   final message. Then `RUN_VIBE_TRIGGER_SMOKE=1` (check 1) passes, a `run-agent --agent vibe` run
   reads `trace_observation_complete: true`, and a `judge --judge-backend vibe` verdict parses.

## 8. Judges

1. On a soft judge with `score_scale: [1, 5], threshold: 4`, run `skill-benchmark judge … --judge-backend claude`
   and again with `--judge-backend codex`.
   - Right: verdict rows have `score` in [1, 5], `availability: complete`, and
     `passed == (score >= 4)`; `benchmark --judge-results` reads `paired_summary.graded.availability:
     complete`, and the qualitative rows carry `raw_score` with a normalised `score`.
   - On Codex, a verdict with `"score": null` must read `complete`, not a schema error.
2. A panel: `judge … --judge-panel <model-a> --judge-panel <model-b> --judge-panel <model-c> --quorum 3`
   on an `atLeast` judge. Right if a task where two of three members pass reads `passed: false`
   after `grade`. `judge … --judge-cmd … --quorum 2` must exit 1.
3. On a real suite with a judge-only case, `case_flags` entries read `signal: combined`, and
   `readiness.floor_cases` is a subset of the cases flagged `floor`.

## 9. Stop a run

1. Start `skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json --agent claude --agent codex --runs-per-query 1 --out /tmp/live/stop.json`.
   Press Ctrl-C after both agents start. Repeat with `kill -TERM <pid>` from another terminal.
   - Right: the command prints only `stopped by SIGINT` (or `SIGTERM`), exits 130 (or 143) within
     about 3 seconds, and `pgrep -fl 'claude|codex'` lists none of the run's agents.
2. Do the same with `skill-benchmark run-agent --agent claude --tasks /tmp/live/tasks.jsonl --runs /tmp/live/rstop`
   and with `--agent codex`. Right: the same line, exit code, and no agent left running.
