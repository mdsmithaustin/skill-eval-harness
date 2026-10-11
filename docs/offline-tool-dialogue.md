# Replay tool responses through a local child

Run a deterministic child against prepared tool challenges. This walkthrough uses no model or live tool provider.
The bridge requires POSIX `waitid` with `WNOWAIT`, nonblocking pipes, process-group controls, and Linux procfs.
The fixtures pass on Linux ARM64 with Python 3.12.15. macOS Python 3.12.14 lacks `os.waitid` and refuses before spawn.
Windows and systems without Linux procfs also refuse. Other platforms need their own lifecycle verification.

From the checkout root, prepare the example and run its child with absolute paths:

```sh
skill-benchmark prepare examples/demo-skill/tool-challenge.json --out /tmp/tool-challenge-tasks.jsonl
skill-benchmark run-subagent --tasks /tmp/tool-challenge-tasks.jsonl --runs /tmp/tool-challenge-runs \
  --tool-bridge --tool-replay strict \
  --agent-cmd "python $(pwd)/examples/demo-skill/offline_tool_bridge.py" \
  --max-cost-usd 1 --assumed-cost-per-run-usd 0.01
skill-benchmark grade examples/demo-skill/tool-challenge.json --runs /tmp/tool-challenge-runs
```

Each answer is `alpha error=known:True`. Both the answer and delivered-error assertions pass.
The explicit monetary assumption labels each external turn. Offline tool resolution does not establish a free child invocation.
Use a fresh runs directory when changing between the ordinary and bridge routes.
An existing replay-v1 `tool-replay.json` under each prepared run directory also works.
Legacy outputs cross the bridge with unavailable error data.

## Protocol reference

The child reads and writes strict UTF-8 JSONL version 1. Object keys cannot repeat.
The parent sends an opening with exactly `type`, `protocol_version`, `prompt`, `model`, `workspace`, and `history`.
`type` is `opening`. The command uses argv parsed from `--agent-cmd` and runs in the isolated workspace.
Shell operators are unsupported on the bridge route.

The child sends one request:

```json
{"type":"call","protocol_version":1,"id":1,"tool":"search","input":{"q":"alpha"}}
```

IDs are positive integers at most `2**63-1`, unique within the turn.
The parent resolves declarations before replay FIFO. Strict misses fail without a live fallback.
One outstanding exchange is allowed. The parent persists its offer before writing the response.

The `result` response has exactly `type`, `protocol_version`, `id`, `task_sha256`, `run`,
`turn_number`, `exchange_number`, `tool`, `input_sha256`, `source`, `fault_id`, `output`, `error`, and `payload_sha256`.
The run is the prepared answer coordinate. Error is either `{"availability":"known","value":true}`
with an exact boolean, or `{"availability":"unavailable","reason":"legacy_error_unavailable"}`.
Output is an ordinary JSON value. A dictionary's own `is_error` key has no special meaning.

`payload_sha256` hashes `tool-bridge-response-v1`, a newline, then the canonical response body excluding that hash.
Canonical JSON sorts keys, uses compact separators and UTF-8, and retains non-ASCII text.
The resolution hash is separate. The child copies the response's coordinates and hash into its ACK:

```json
{"type":"ack","protocol_version":1,"id":1,"turn_number":1,"exchange_number":1,"payload_sha256":"sha256:<response hash>"}
```

The parent accepts the exact ACK only after the entire response write completes.
It persists the acknowledgment before accepting another request or final response.
ACK proves protocol acceptance by the child. It does not prove model consumption.

The child finishes with `type`, `protocol_version`, and a nonblank `answer`.
Only `trace`, `usage`, and `telemetry_scope` are optional.
Those fields use the existing subagent response contract. Provider trace stays diagnostic.

```json
{"type":"final","protocol_version":1,"answer":"alpha","usage":{"cost_usd":0.02},"telemetry_scope":"turn_delta"}
```

Completion also requires natural zero exit, no pending response, full required pipe drain, leader reap,
and observed original-group shutdown. A final followed by a hang, nonzero exit, or held pipes remains incomplete.
Previously ACKed outputs can still establish positive prefix claims. Exact counts and absence require closure.
Closed legacy output counts remain available. Claims that depend on unknown delivered errors remain unavailable.
Generic tool responses cannot establish file-read, command-exit, or skill-open success.

## Bounds and retained evidence

Defaults are 65,536 bytes per frame including its newline, 8,388,608 bytes of aggregate parent writes,
and 8,388,608 bytes of combined stdout and stderr. The `--tool-bridge-max-frame-bytes`,
`--tool-bridge-max-outbound-bytes`, and `--tool-bridge-max-stream-bytes` flags can lower those limits.
All pipe activity shares `--timeout`. Partial writes and unterminated input consume the same deadline.
Outgoing values pass a finite minimum-size walk before encoding; oversized scalars or containers
are rejected before string escaping. The encoded UTF-8 bytes must also fit the frame bound.
Reads and writes use at most 16 KiB chunks. Cleanup has a separate 64 KiB drain bound and a bounded grace period.
Crossing bytes remain recorded as failed transport. They cannot become a valid truncated frame.

The `mediated_tools.journal_path` in `events.json` points into `runs/.tool-observations/`.
Its sibling `turn-N/` directories retain exclusive `stdout.bin`, `stderr.bin`, and `process.json` files.
Process facts record actual OS return code, natural exit, forced termination, group observation, drain, reap,
and retained byte hashes. Cleanup finishes before workspace capture or deletion.
The existing journal is the only transition writer, and the existing supervisor is the only process owner.

Successful scoped file and directory fsync calls establish the retained prefix on the tested filesystem.
Failed writes withhold the affected evidence. SIGKILL can lose unobserved bytes.
Filesystem stalls have no hard I/O deadline. Detached sessions escape original-group control.
Linux procfs ignores zombie and dead group members. Other POSIX group-zero probes have different limits.
Same-user edits are outside the durability claim.

The price owner isolates finite nonnegative nonboolean `usage.cost_usd` from a complete strict final object.
Later invalid siblings or journal closure failure do not erase naturally completed whole dollars.
Actual forced termination or unresolved process continuation retains only an eligible reported floor.
Multi-turn dollars require `telemetry_scope: turn_delta`. Tool exchanges add no paid invocation.
SIGINT and SIGTERM retain cleanup and paid facts, settle the admitted turn, then re-raise the original interruption.
Ledger persistence failure remains unresolved.
