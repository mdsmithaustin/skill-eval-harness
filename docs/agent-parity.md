# Agent parity matrix

The harness supports several agent surfaces, but not every agent supports every surface. This table is the reader-facing copy of `agent_capabilities.BACKENDS`; `AGENT_CAPABILITIES`, native answer/judge dispatch, trigger adapters, workspace builders, smoke targets, failure markers, and provider-specific CLI options are compatibility projections of those rows. When an agent gains a surface, update its one registry row, this table, and its conformance fixtures.

Run `skill-benchmark agent-capabilities` for the machine-readable registry view.

| Agent | Answer runs | Answer route | Autonomous trigger | Trigger ablation | Trace artifacts | Token usage | Dollar cost | Judge backend | Tool replay | Live smoke |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---|
| `claude` | yes (`run-claude`, `run-agent --agent claude`) | `native` | yes (`skill-trigger-matrix --agent claude`) | yes | yes | yes | `provider_reported` | yes (`judge --judge-backend claude` / `--judge-model`) | yes through `run-subagent` | `RUN_TRIGGER_SMOKE` |
| `codex` | yes (`run-codex`, `run-agent --agent codex`) | `native` | yes (`skill-trigger-matrix --agent codex`) | yes | yes | yes, when stream reports it | `missing` unless a wrapper emits/estimates cost | yes (`judge --judge-backend codex`) | no native replay | `RUN_CODEX_TRIGGER_SMOKE` |
| `gemini` | yes (`run-agent --agent gemini`) | `native` | no (headless `activate_skill` consent gate not live-proven) | no | yes | yes, when JSON stats report it | `missing` (CLI has no cost field) | yes (`judge --judge-backend gemini`) | no native replay | `RUN_GEMINI_SMOKE` |
| `pi` | no core answer runner | `none` | yes (`skill-pi-trigger-eval`, `skill-trigger-matrix --agent pi`) | yes | yes | yes, when stream reports it | `trace_normalized` from the stream when available | no | no | `RUN_PI_TRIGGER_SMOKE` |
| `jetty` | yes (`export-jetty` / `run-jetty` / `import-jetty-results`; answer-path ablations only) | `export_import` | no | no | yes, imported | yes, imported | `provider_reported`, imported | no (planned in Jetty TODO) | no | `RUN_JETTY_SMOKE` |
| `vibe` | yes (`run-agent --agent vibe`) | `native` | yes (`skill-trigger-matrix --agent vibe`) | yes | yes | no in current CLI output | `missing` in current CLI output | yes (`judge --judge-backend vibe`) | no native replay | `RUN_VIBE_TRIGGER_SMOKE` |
| `subagent` | yes (`run-subagent`) | `subagent` | no | no | yes | yes, when backend returns it | `missing` unless backend emits cost | no | yes | n/a |
| `stub` | no native answer runner; demo stub uses `run-codex --codex-cmd` | `none` | yes | yes | yes | no (not applicable) | `not_applicable` | no | no | n/a |

## What changed for Gemini CLI

Gemini is a first-class native answer and judge backend, with its unproven surface kept out of the registry:

- `skill-benchmark run-agent --agent gemini` runs `gemini --prompt "$PROMPT" --output-format stream-json`, takes final answer text only from a schema-valid terminal stream, and retains the raw stream for trace normalization.
- `skill-benchmark judge --judge-backend gemini` uses `--output-format stream-json`, parses only the final validated assistant segment as verdict JSON, and preserves the raw lifecycle stream plus provider metadata in judge transcripts.
- Every invocation uses a fresh `GEMINI_CLI_HOME` outside the model workdir. A valid configured `security.auth.selectedType` wins before environment selection; the harness copies only credential material and supporting environment required by that one planned auth mode, forces portable file storage when needed, suppresses interactive browser auth, and fails closed on invalid settings or nonportable credentials. User skills, extensions, MCP configuration, hooks, context, policies, history, and sessions are not copied.
- A user-tier TOML policy denies every tool, then answer runs allow only `glob`, `grep_search`, `list_directory`, `read_file`, and `read_many_files`; judges keep the deny-all rule and reject any observed tool lifecycle or nonzero aggregate tool counter. The harness requests sandboxing only when a supported engine exists and the chosen credentials have a proven transport, and records disabled reasons plus the administrator settings/policy override risk.
- Workspace `.gemini`, `.agents`, `.geminiignore`, and `GEMINI.md` controls are rejected case-insensitively before invocation so an eval fixture cannot silently replace the harness control plane. Usage is normalized from `stats` when present; absent usage and unsupported dollar cost remain explicit `missing` rather than zero.
- `--gemini-cmd` is one caller-trusted executable token. Every run probes and records `gemini --version`; the live smoke requires that evidence, while offline fixtures name their exact upstream commit/package snapshot.
- Autonomous trigger remains `false`. Current Gemini skills activate through `activate_skill`, whose headless consent behavior has not been proven safe and noninteractive in a token-backed run. No adapter or trigger claim is published until that gate passes.

Offline tests mirror the official Gemini CLI stream/JSON conformance shapes. The opt-in token-backed answer smoke is `RUN_GEMINI_SMOKE=1`; it is not part of default CI. Gemini judge explore remains rejected until a separately proven read-only implementation exists.

## What changed for Vibe

Mistral Vibe is now a first-class native backend alongside Claude and Codex for the surfaces Vibe exposes safely in programmatic mode:

- `skill-benchmark run-agent --agent vibe` runs prepared answer rows through `vibe --prompt "$PROMPT" --output streaming` from an isolated workspace.
- `skill-benchmark judge --judge-backend vibe` runs native Vibe judges with tools disabled (`--enabled-tools re:^$`) and validates the final assistant message against the harness verdict schema.
- `skill-trigger-matrix --agent vibe` mounts skills under project `.agents/skills`, runs raw trigger queries, detects native `skill` tool calls by skill name, and falls back to mounted-path evidence.
- Every invocation sets a fresh `VIBE_HOME` outside the model workdir; `MISTRAL_API_KEY` is read from the environment, and if absent the harness copies only `.env` from the current `VIBE_HOME` (falling back to `~/.vibe/.env`) into the isolated home. User skills/config are never copied.
- Model selection uses `VIBE_ACTIVE_MODEL` when `--model` / `--judge-model` is supplied. Current Vibe `json`/`streaming` messages do not include usage/cost fields, so telemetry is marked explicit `missing` until the CLI exports it or the harness adds an estimator.

Live smoke is gated by `RUN_VIBE_TRIGGER_SMOKE=1` plus `MISTRAL_API_KEY`; token-backed smokes passed for Vibe 2.19.1 on 2026-07-09.

## What changed for Codex

Codex is no longer only an answer runner. The trigger matrix now accepts `--agent codex`, mounts the same canonical or materialized skill tree used by Claude/Pi/stub under isolated `$CODEX_HOME/skills`, exposes that skills directory as a skills-only extra read root, and runs the raw trigger query through `codex exec --json` with `--ignore-user-config --ignore-rules` by default. It detects activation from the session rollout when the CLI writes a user-role `<skill>` injection for a mounted skill. Tool calls, outputs, and skill listings in the rollout do not count as loads. Without an injection, the shared completed path-evidence detector decides the result. Each row records `codex_rollout_status` (`found`, `not_found`, `no_thread_id`) and the evidence kind (`codex_rollout` or `mounted_path`) so a reader can tell which detector decided. Native answer and judge runs also use an isolated `CODEX_HOME`; answer/judge final text comes from `--output-last-message` while JSONL remains the trace/usage stream. Credential-bearing Codex homes are outside the model workdir. The raw query is appended to the command prefix:

```bash
skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json \
  --agent codex \
  --runs-per-query 3 \
  --out /tmp/trigger-codex.json
```

The same command can write per-run traces and run a materialized discovery/trigger-population ablation for any matrix agent:

```bash
skill-trigger-matrix examples/demo-skill/evals/shared-benchmark.json \
  --agent codex \
  --trace-runs /tmp/trigger-traces \
  --ablation weaker-description \
  --out /tmp/trigger-codex-ablation.json
```

The report-level evidence class is `raw_autonomous_trigger_measurement`; individual result rows use the shared `raw_measurement` enum value. These are rates for tuning descriptions, not provenance-confirmed causal lift claims.

## Trigger context isolation

A trigger run measures whether the agent loads the mounted skill on its own. Anything else the agent is shown can win that routing decision, so each trigger adapter hides the operator's host context and keeps only the mounted skill. Claude and Codex answer and judge runs use the same flags, so skills and agents an answer run mounts in its workspace (`.claude/skills`, `.claude/agents`, `.agents/skills`) stay visible there too. `--safe-mode --disable-slash-commands` and `-c skills.include_instructions=false` are not used anywhere, because they also hide mounted skills.

- `claude` adds `--setting-sources project --strict-mcp-config --settings '{"disableBundledSkills":true}'`. That removes `~/.claude` skills, user agents, `~/.claude/CLAUDE.md`, user settings (hooks, env, permissions), every MCP server including claude.ai connectors, and the skills Claude Code bundles. The workspace `.claude/skills` still loads.
- `codex` adds `-c skills.bundled.enabled=false`, a `-c skills.config=[{path=...,enabled=false}, ...]` entry naming each `SKILL.md` under `~/.agents/skills`, and `--disable apps`. That removes host skills under `~/.agents/skills`, Codex's bundled system skills, the apps connector, and plugin recommendations. The isolated `$CODEX_HOME/skills` still loads.
- `pi` adds `--no-skills --skill <mounted skills dir>` next to the existing `--no-context-files --no-prompt-templates --no-extensions`. That stops Pi's own discovery of `~/.agents/skills` and project `.agents/skills`. The mounted skills directory still loads, listed like a discovered skill and not force-loaded.

Each trigger row records the flags it ran with as `context_isolation`. A Codex row records `skills.config=<N host skill(s) disabled>` in place of the operator's skill paths.

This was checked on Claude Code CLI 2.1.284, codex-cli 0.156.1, and Pi 0.73.1. A census of each adapter's real invocation path found the following after the change.

- Claude lists the mounted skill plus Claude Code's own `design` and `doctor` entries, only built-in agents, and no MCP servers. A live run with these flags does not see `~/.claude/CLAUDE.md` and still signs in through OAuth.
- Codex lists only the mounted skill.
- Pi lists only the mounted skill. Pi was checked through its own resource loader, not a live run.

Limits. When Claude auth is not file-portable, the run still uses the operator's `CLAUDE_CONFIG_DIR`, so the row keeps its `config_isolation_warning`. Codex accepts an unknown `-c` key silently, so a Codex build that renames `skills.config` or `skills.bundled` would show host skills again while the row still records `context_isolation`; re-run the census after a Codex upgrade. Codex's admin skill root (`/etc/codex/skills`) is not disabled. Vibe was not measured.

Host paths in Codex output. Codex names host skills in its own stderr: a skill that fails to load (by its path, or by the symlink target's real path), a directory it cannot scan (as a percent-encoded `file://` URL with tab and newline dropped), and a rejected `skills.config` value echoed back with Rust's escaping on top. Before the row is written, `CodexAdapter.invoke` redacts from stdout, stderr, and `provider_error` every spelling of each `SKILL.md` and directory the host-skill walk visits and of their real paths (raw, TOML-escaped, JSON-escaped, any of those through Rust's `{:?}`, and `file://`), plus any span that starts with `~/.agents/skills` or its real path in one of those spellings, in each case up to the next quote, whitespace, `]`, or `,`. The mounted skill lives in the isolated `$CODEX_HOME`, so its path and the detection evidence are kept. Two cases can still leave part of a host path in a row. A spelling not listed here of a path outside `~/.agents/skills` (a symlinked skill's target) is not caught at all, because nothing anchors it. A spelling not listed here of a path under `~/.agents/skills` loses only the part before the first quote, whitespace, `]`, or `,` after the root; the rest stays. Rust's `{:?}` is approximated with Python's `str.isprintable`, so a non-printable character the two classify differently produces such an unlisted spelling.

A third case is closed, not residual: `invoke_argv_with_timeout` caps captured stderr at 4000 characters, and a rejected `skills.config` echoes every host path on one line, so with enough host skills the cap used to cut a path in half before the redactor ran. The fragment it left (as little as the start of the operator's home directory) matched no listed spelling and no root-anchored span, so it reached the row verbatim. The same held for a timed-out run with empty stderr, which records the argv, `skills.config` included. `CodexAdapter.invoke` now passes the redactor to the subprocess owner as `ProcessInvocationPlan.redact_output`, which applies it to the whole captured stdout and stderr before any cap. The cap only ever cuts text that has already been redacted. Other callers pass no redactor, and their capture is unchanged. One gap remains in the timed-out, empty-stderr case: Python's timeout message prints the argv with `repr` escaping, which is not a listed spelling, so a host skill directory whose name contains whitespace or a quote can still leave part of its name in the row. Real Codex writes to stderr at startup, so this fallback text is rarely recorded.

Scale limit. Each disabled host skill adds one `{path="...",enabled=false}` entry to the single `skills.config=[...]` argv element, roughly 81 bytes per entry for a typical `~/.agents/skills/<name>/SKILL.md` path. Linux limits a single argument to 128 KiB, which caps a Codex trigger run near ~1,600 host skills. macOS has no per-argument limit; its cap is the total size of argv plus the environment (`ARG_MAX`, 1,048,576 bytes), near ~12,900 host skills at ~81 bytes each, less whatever the rest of argv and the environment take. Past either limit the run does not degrade quietly: spawning Codex fails before it starts, and the row records `invocation_state` `spawn_failed` with `OSError: [Errno 7] Argument list too long` in its stderr.
