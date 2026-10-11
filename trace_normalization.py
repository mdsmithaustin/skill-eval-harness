from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Any, Protocol

import telemetry as telemetry_domain
import tool_fault_contracts as tool_fault_domain
from agent_capabilities import trace_dialect_implementations
from gemini_contracts import GeminiStream
from json_contracts import (
    StrictJSONViolation,
    iter_json_objects,
    parse_stream_json,
    stream_json_loads,
    strict_json_loads,
    write_json,
)
from observation_contracts import COST_SOURCES, USAGE_SOURCES
from trace_contracts import EventState, event_is_completed, parse_event_state
from trigger_contracts import TraceEventKind, TriggerDetection, TriggerEvidenceKind


def string_keyed_dict(value: Any, label: str) -> dict[str, Any]:
    """Reify an untrusted JSON/YAML object without coercing or losing keys."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} object keys must be strings")
    return {
        key: item for key, item in value.items() if isinstance(key, str)
    }


GEMINI_READ_ONLY_TOOLS = (
    "glob", "grep_search", "list_directory", "read_file", "read_many_files",
)


def _jetty_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    aggregate = next((record.get("total_tool_calls") for record in records
                      if record.get("type") == "usage"), None)
    if aggregate is None:
        if not any(str(record.get("type") or "").casefold() in {
                "command", "tool_call", "tool_use", "file_read", "file_write",
                "skill_load"} for record in records):
            return "Jetty trajectory has no event stream or explicit zero aggregate"
        return None
    detailed = sum(1 for record in records if str(record.get("type") or "").casefold()
                   in {"command", "tool_call", "tool_use", "file_read", "file_write", "skill_load"})
    if detailed != aggregate:
        return ("Jetty aggregate total_tool_calls has no complete matching "
                "event-level trajectory")
    return None


def _tool_measurements(observation: tool_fault_domain.ToolObservation,
                       basis: Mapping[str, Any] | None = None) -> dict[str, telemetry_domain.Measurement[Any]]:
    result: dict[str, telemetry_domain.Measurement[Any]] = {}
    for key in tool_fault_domain.TOOL_METRIC_KEYS:
        assessed = observation.assess(tool_fault_domain.Count(key))
        result[key] = (telemetry_domain.Measurement.available(assessed.value, provenance="trace_normalized", basis=basis)
                       if isinstance(assessed, tool_fault_domain.Observed) else
                       telemetry_domain.Measurement.unavailable(assessed.reason, basis=basis))
    return result


def command_text(event: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ["input_summary", "command", "cmd", "name"]:
        value = event.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.append(" ".join(str(v) for v in value))
    details = event.get("details")
    if isinstance(details, dict):
        for key in ["command", "cmd", "input", "args"]:
            value = details.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.append(" ".join(str(v) for v in value))
    return " ".join(p for p in parts if p).strip()


def command_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Commands proven completed; failed/unknown/start events never satisfy execution."""
    return [e for e in events if e.get("type") == "command" and event_is_completed(e)]


def event_mentions_skill_file(event: dict[str, Any]) -> bool:
    if event.get("type") not in {"skill_load", "file_read", "tool_call", "command"}:
        return False
    if str(event.get("name") or "").casefold() in {
            "glob", "grep_search", "list_directory", "read_many_files"}:
        return False
    hay = " ".join(str(event.get(key, ""))
                   for key in ["input_summary", "name"])
    return "SKILL.md" in hay or "/skills/" in hay or "\\skills\\" in hay


TRAJECTORY_STEP_TYPES = {"command", "tool_call", "file_read", "file_write", "skill_load"}


def trace_event_counts(events: list[dict[str, Any]], *, mediated_tools: tool_fault_domain.ToolObservation | None = None) -> dict[str, Any]:
    """Single owner of the completed-events-only counting rules. metrics.json
    (normalize_trace_records) and the report's trajectory diff both derive
    their counters here, so the two surfaces cannot drift: a diff delta is a
    delta of exactly the numbers metrics.json reports. Errors count over ALL
    events — a failed operation is an error precisely because it never
    completed."""
    if mediated_tools is not None:
        measured = _tool_measurements(mediated_tools)
        return {key: value.value for key, value in measured.items()}

    completed = [e for e in events if event_is_completed(e)]
    commands = command_events(events)
    return {
        "steps": sum(1 for e in completed if e.get("type") in TRAJECTORY_STEP_TYPES),
        "tool_calls": sum(1 for e in completed
                          if e.get("type") in TRAJECTORY_STEP_TYPES),
        "commands": len(commands),
        "file_reads": sum(1 for e in completed if e.get("type") in {"file_read", "skill_load"}),
        "file_writes": sum(1 for e in completed if e.get("type") == "file_write"),
        "errors": sum(1 for e in events if e.get("type") == "error" or e.get("is_error") is True),
        "skill_events": [e for e in completed if e.get("type") == "skill_load" or event_mentions_skill_file(e)],
    }


EVENT_TEXT_KEYS = {"file_path", "path", "skill", "input", "input_summary", "partial_json", "command", "cmd", "args", "argv"}


def _flatten_event_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_flatten_event_text(item) for item in value).strip()
    return ""


def event_texts_for_tool_input(obj: Any) -> list[str]:
    """Recursively collect file-path-ish strings from a runner event (tool inputs),
    used to detect whether the model actually opened a mounted skill file."""
    out: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in EVENT_TEXT_KEYS:
                text = _flatten_event_text(value)
                if text:
                    out.append(text)
            out.extend(event_texts_for_tool_input(value))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(event_texts_for_tool_input(item))
    return out


def text_mentions_mounted_skill(text: str, copied_paths: Sequence[Path]) -> bool:
    for path in copied_paths:
        root = path.parent if path.name == "SKILL.md" else path
        pattern = r"(?<![\w./-])" + re.escape(str(root)) + r"(?=$|[/\s\"'<>),;])"
        if re.search(pattern, text):
            return True
    return False


def detect_trigger_records(records: Iterable[dict[str, Any]], copied_paths: list[Path],
                           *, source: str = "generic",
                           pi_stream: PiStream | None = None, mediated_tools: tool_fault_domain.ToolObservation | None = None) -> TriggerDetection:
    """Derive mounted-path evidence from provider-aware completed operations."""
    if mediated_tools is not None:
        assessed = mediated_tools.assess(tool_fault_domain.SuccessfulOperations())
        if isinstance(assessed, tool_fault_domain.Unavailable):
            return TriggerDetection.absent()
        return TriggerDetection.from_texts(TriggerEvidenceKind.MOUNTED_PATH, assessed.value)

    evidence: list[str] = []
    materialized = list(records)
    event_doc, _ = normalize_trace_records(
        materialized, source=source, pi_stream=pi_stream)
    for event in event_doc["events"]:
        if not event_is_completed(event) or event.get("is_error") is True:
            continue
        exit_code = event.get("exit_code")
        if exit_code is not None and (type(exit_code) is not int or exit_code != 0):
            continue
        if event.get("type") not in {"skill_load", "file_read", "command"}:
            continue
        for text in event_texts_for_tool_input(event):
            if text_mentions_mounted_skill(text, copied_paths):
                evidence.append(text[:500])
    return TriggerDetection.from_texts(TriggerEvidenceKind.MOUNTED_PATH, evidence[:5])


def detect_trigger_detection(stdout: str, copied_paths: list[Path],
                             *, source: str = "generic") -> TriggerDetection:
    """Typed skill-invocation detector for a raw JSON event stream."""
    records = [event for event in iter_json_objects(stdout, strict=False) if isinstance(event, dict)]
    return detect_trigger_records(records, copied_paths, source=source)


def detect_trigger(stdout: str, copied_paths: list[Path]) -> tuple[bool, list[str]]:
    """Compatibility wire helper; internal trigger runners use TriggerDetection."""
    detection = detect_trigger_detection(stdout, copied_paths)
    return detection.triggered, detection.legacy_evidence


def safe_trace_label(text: str, fallback: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)[:80].strip("-")
    return label or fallback


def repeated_command_max(commands: list[str]) -> int:
    last = None
    current = 0
    best = 0
    for command in commands:
        normed = re.sub(r"\s+", " ", command.strip().casefold())
        if normed and normed == last:
            current += 1
        else:
            last = normed
            current = 1 if normed else 0
        best = max(best, current)
    return best


USAGE_ALIASES = telemetry_domain.USAGE_ALIASES


COST_PART_ALIASES: dict[str, tuple[str, ...]] = {
    "input_cost": ("input_cost", "prompt_cost"),
    "output_cost": ("output_cost", "completion_cost"),
    "cache_read_cost": ("cache_read_cost",),
    "cache_write_cost": ("cache_write_cost", "cache_creation_cost"),
    "reasoning_cost": ("reasoning_cost",),
}


def _num(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_usage(raw: Any, *, source: str = "provider_reported") -> dict[str, Any]:
    """The per-run usage_normalized block (issue #21): alias-normalized token
    counts with explicit provenance. No usable numbers means {"source":
    "missing"} — missing telemetry is never silently zero."""
    if source not in USAGE_SOURCES:
        raise ValueError(f"unknown usage source {source!r}; expected one of {sorted(USAGE_SOURCES)}")
    out: dict[str, Any] = telemetry_domain.canonical_usage_counts(raw)
    if source == "not_applicable":
        return {"source": "not_applicable"}
    if ("total_tokens" not in out and "input_tokens" in out
            and "output_tokens" in out):
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    if not out:
        return {"source": "missing"}
    out["source"] = source
    return out


def normalize_cost(raw: Any, *, source: str = "provider_reported", currency: str = "USD",
                   pricing_model: str | None = None, pricing_table_version: str | None = None,
                   pricing_notes: list[str] | None = None) -> dict[str, Any]:
    """The per-run cost_normalized block (issue #21): a provider-reported number
    or cost object, or a price-table estimate, with currency and provenance.
    Missing cost is marked missing, never zero."""
    if source not in COST_SOURCES:
        raise ValueError(f"unknown cost source {source!r}; expected one of {sorted(COST_SOURCES)}")
    if source == "not_applicable":
        return {"source": "not_applicable"}
    if isinstance(raw, (int, float)) and (isinstance(raw, bool) or _num(raw) is None):
        raise ValueError("cost measurement must be a finite number")
    total = _num(raw)
    parts: dict[str, float] = {}
    resolved_currency = currency
    if isinstance(raw, dict):
        if raw.get("currency") is not None:
            raw_currency = raw.get("currency")
            if not isinstance(raw_currency, str) or not re.fullmatch(r"[A-Z]{3}", raw_currency):
                raise ValueError("cost currency must be a three-letter uppercase code")
            resolved_currency = raw_currency
        total_observations: list[tuple[str, float]] = []
        for key in ("total_cost", "total_cost_usd", "cost_usd", "total", "cost", "amount"):
            if key not in raw or raw[key] is None:
                continue
            value = _num(raw[key])
            if value is None:
                raise ValueError(f"cost.{key} must be a finite number")
            if key.endswith("_usd") and resolved_currency != "USD":
                raise ValueError(f"cost.{key} cannot be labelled {resolved_currency}")
            total_observations.append((key, value))
        if total_observations:
            values = {value for _, value in total_observations}
            if len(values) != 1:
                raise ValueError(
                    f"conflicting total cost aliases: {dict(total_observations)}")
            total = total_observations[0][1]
        for norm_key, aliases in COST_PART_ALIASES.items():
            observations: list[tuple[str, float]] = []
            for alias in aliases:
                if alias not in raw or raw[alias] is None:
                    continue
                value = _num(raw[alias])
                if value is None:
                    raise ValueError(f"cost.{alias} must be a finite number")
                observations.append((alias, value))
            if observations:
                values = {value for _, value in observations}
                if len(values) != 1:
                    raise ValueError(
                        f"conflicting aliases for {norm_key}: {dict(observations)}")
                parts[norm_key] = observations[0][1]
        if "components_complete" in raw and not isinstance(raw["components_complete"], bool):
            raise ValueError("cost.components_complete must be boolean")
        if total is None and parts and raw.get("components_complete") is True:
            total = sum(parts.values())
        elif total is not None and parts and raw.get("components_complete") is True:
            component_total = sum(parts.values())
            if not math.isclose(total, component_total, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    "complete cost components do not sum to the reported total")
    if total is None:
        return ({"source": "missing", "currency": resolved_currency,
                 "observed_parts": parts, "reason": "partial_cost_components"}
                if parts else {"source": "missing"})
    if total < 0 or any(value < 0 for value in parts.values()):
        raise ValueError("cost measurements must be nonnegative")
    out: dict[str, Any] = {"currency": resolved_currency, **parts,
                           "total_cost": total, "source": source}
    if pricing_model:
        out["pricing_model"] = pricing_model
    if pricing_table_version:
        out["pricing_table_version"] = pricing_table_version
    if pricing_notes:
        out["pricing_notes"] = list(pricing_notes)
    return out


def raw_trace_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    for container_key in ["message", "delta", "data", "item"]:
        nested = record.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def raw_trace_input_value(record: dict[str, Any], *keys: str) -> Any:
    """Invocation arguments may supply paths/commands, never lifecycle facts."""
    value = raw_trace_value(record, *keys)
    if value is not None:
        return value
    for container_key in ("tool_input", "input", "details", "args"):
        nested = record.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    return nested[key]
    return None


def raw_trace_has_key(record: dict[str, Any], *keys: str) -> bool:
    if any(key in record for key in keys):
        return True
    for container_key in ["message", "delta", "data", "item"]:
        nested = record.get(container_key)
        if isinstance(nested, dict) and any(key in nested for key in keys):
            return True
    return False


def stringify_trace_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(stringify_trace_value(v) for v in value)
    if isinstance(value, dict):
        for key in ["command", "cmd", "content", "text", "path"]:
            if key in value:
                return stringify_trace_value(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def parse_trace_jsonl_text_with_lines(
    text: str, *, strict_json_errors: bool = True, strict: bool = True,
) -> tuple[list[dict[str, Any]], list[str], list[int], list[str]]:
    """Parse JSONL while retaining each object's physical source line.

    Blank, malformed, and non-object lines do not become records, but they do
    occupy source lines. Keeping that mapping makes every emitted raw_ref
    resolvable against the original trace.jsonl rather than a filtered ordinal.

    `strict=True` is the artifact rule (`strict_json_loads`: a repeated object
    key is a defect). `strict=False` is the external-CLI stream rule
    (`parse_stream_json`: a repeated key resolves last-value-wins and is
    reported in the fourth result as `line N: key`), so a valid line an agent
    CLI emitted is never dropped. Every other failure is unchanged: a line that
    is not JSON is skipped and recorded, and a non-finite constant raises (or
    is recorded when `strict_json_errors` is off).
    """
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    record_lines: list[int] = []
    duplicate_keys: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            if strict:
                obj = strict_json_loads(line)
            else:
                parsed = parse_stream_json(line)
                obj = parsed.value
                duplicate_keys.extend(f"line {line_number}: {key}" for key in parsed.duplicate_keys)
        except json.JSONDecodeError as exc:
            if isinstance(exc, StrictJSONViolation) and strict_json_errors:
                raise ValueError(exc.msg) from exc
            errors.append(f"line {line_number}: {exc}")
            continue
        except (TypeError, ValueError, RecursionError) as exc:
            if strict_json_errors:
                raise ValueError(f"line {line_number}: {exc}") from exc
            errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(obj, dict):
            records.append(obj)
            record_lines.append(line_number)
        else:
            errors.append(f"line {line_number}: JSON value is not an object")
    return records, errors, record_lines, duplicate_keys


def parse_trace_jsonl_text(
    text: str, *, strict: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    records, errors, _, _ = parse_trace_jsonl_text_with_lines(text, strict=strict)
    return records, errors


def stream_duplicate_keys(text: str) -> list[str]:
    """`line N: key` for every duplicate object key the stream rule resolved in
    an external CLI stream; empty when the strict rule would have accepted
    every line. Never raises: it only annotates a row."""
    _, _, _, duplicates = parse_trace_jsonl_text_with_lines(
        text, strict_json_errors=False, strict=False)
    return duplicates


def load_trace_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    return parse_trace_jsonl_text(path.read_text(encoding="utf-8", errors="replace"), strict=False)


def nested_item_type(record: dict[str, Any]) -> str:
    item = record.get("item")
    if isinstance(item, dict):
        value = item.get("type") or item.get("kind") or item.get("name")
        return str(value or "")
    return ""


_RAW_TYPE_WORD_BOUNDARY = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def raw_type_tokens(raw_type: str) -> frozenset[str]:
    """Whole-word tokens of a provider type name, split on non-alphanumerics
    and camelCase boundaries, each casefolded. Pass the original-case string;
    casefolding first collapses the camelCase boundary this exists to find."""
    return frozenset(part.casefold() for part in _RAW_TYPE_WORD_BOUNDARY.split(raw_type) if part)


def usage_number(usage: dict[str, Any], *keys: str) -> int | None:
    normalized = normalize_usage(usage, source="trace_normalized")
    for key in keys:
        value = normalized.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def normalize_trace_record(record: dict[str, Any], *, source: str, index: int, line: int) -> dict[str, Any]:
    top_type = str(raw_trace_value(record, "type", "event", "name", "kind") or "")
    item_type = nested_item_type(record)
    raw_type = f"{top_type} {item_type}".casefold()
    raw_type_tok = raw_type_tokens(f"{top_type} {item_type}")

    def has_keyword(keyword: str) -> bool:
        return any(token.startswith(keyword) for token in raw_type_tok)

    path = stringify_trace_value(raw_trace_input_value(record, "path", "file", "file_path"))
    command = stringify_trace_value(raw_trace_input_value(record, "command", "cmd", "args"))
    content = stringify_trace_value(raw_trace_value(record, "content", "text", "message"))
    raw_status = raw_trace_value(record, "status", "state")
    parsed_state = parse_event_state(
        raw_status, raw_type=top_type or item_type,
        status_present=raw_trace_has_key(record, "status", "state"))
    status = parsed_state.state.value
    event_type = TraceEventKind.EVENT
    name = stringify_trace_value(raw_trace_value(
        record, "tool", "tool_name", "toolName", "name"))
    tool_name = name.casefold()
    is_write = (has_keyword("write") or has_keyword("edit")
                or tool_name in {"write", "edit", "multiedit", "notebookedit", "write_file"})
    is_read = (has_keyword("read") or tool_name in {"read", "read_file"})
    skill_path = path.endswith("SKILL.md") or "/SKILL.md" in path or "\\SKILL.md" in path
    explicit_skill_load = has_keyword("skill") and (has_keyword("load") or has_keyword("read"))
    if is_write:
        event_type = TraceEventKind.FILE_WRITE
    elif explicit_skill_load or (is_read and skill_path):
        event_type = TraceEventKind.SKILL_LOAD
    elif has_keyword("command") or has_keyword("exec") or command:
        event_type = TraceEventKind.COMMAND
        name = name or "bash"
    elif is_read:
        event_type = TraceEventKind.FILE_READ
    elif has_keyword("tool") or raw_trace_value(
            record, "tool", "tool_name", "toolName", "tool_call_id") is not None:
        event_type = TraceEventKind.TOOL_CALL
    elif has_keyword("error") or str(status).casefold() in {"failed", "error", "errored"}:
        event_type = TraceEventKind.ERROR
    elif raw_trace_value(record, "role") or content or "agent_message" in raw_type:
        event_type = TraceEventKind.MESSAGE
    elif has_keyword("usage") or has_keyword("metric") or raw_trace_value(record, "usage", "tokens"):
        event_type = TraceEventKind.METRIC
    input_summary = command or path or content[:500]
    output_summary = stringify_trace_value(raw_trace_value(record, "output", "stdout", "stderr", "result"))[:1000]
    raw_call_line = record.get("_raw_call_line")
    raw_line = raw_call_line if isinstance(raw_call_line, int) and not isinstance(raw_call_line, bool) and raw_call_line > 0 else line
    event = {
        "index": index,
        "type": event_type.value,
        "status": status,
        "state_source": parsed_state.source.value,
        "raw_ref": {"file": "trace.jsonl", "line": raw_line},
    }
    raw_result_line = record.get("_raw_result_line")
    if isinstance(raw_result_line, int) and not isinstance(raw_result_line, bool) and raw_result_line > 0:
        event["raw_result_ref"] = {"file": "trace.jsonl", "line": raw_result_line}
    if record.get("is_error") is True:
        event["is_error"] = True
    if isinstance(raw_status, str) and raw_status.casefold() != status:
        event["raw_status"] = raw_status
    role = raw_trace_value(record, "role")
    if not role and "agent_message" in raw_type:
        role = "assistant"
    if role:
        event["role"] = str(role)
    if name:
        event["name"] = name
    change_kind = record.get("change_kind")
    if isinstance(change_kind, str) and change_kind:
        event["change_kind"] = change_kind
    if input_summary:
        event["input_summary"] = input_summary[:1000]
    if output_summary:
        event["output_summary"] = output_summary
    timestamp = raw_trace_value(record, "timestamp", "time", "created_at")
    if timestamp:
        event["timestamp"] = str(timestamp)
    exit_code = raw_trace_value(record, "exit_code", "returncode")
    if isinstance(exit_code, int):
        event["exit_code"] = exit_code
    duration = _num(raw_trace_value(record, "duration_ms", "elapsed_ms"))
    if duration is not None:
        event["duration_ms"] = duration
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else record.get("tokens")
    if isinstance(usage, dict):
        token_doc = {k: v for k, raw in usage.items() if (v := _num(raw)) is not None}
        normalized_total = usage_number(usage, "total_tokens")
        normalized_input = usage_number(usage, "input_tokens")
        normalized_output = usage_number(usage, "output_tokens")
        if normalized_total is None and normalized_input is not None and normalized_output is not None:
            normalized_total = normalized_input + normalized_output
        if normalized_input is not None:
            token_doc["input_tokens"] = int(normalized_input)
        if normalized_output is not None:
            token_doc["output_tokens"] = int(normalized_output)
        if normalized_total is not None:
            token_doc["total_tokens"] = int(normalized_total)
        event["tokens"] = token_doc
    event["source"] = source
    event["otel"] = otel_attributes_for_event(event)
    return event


def otel_attributes_for_event(event: dict[str, Any]) -> dict[str, Any]:
    """OTel GenAI semantic-convention attributes for one normalized event
    (roadmap 2.4). Additive: the harness's own keys stay authoritative for
    grading; these make the trace boundary a standard target instead of a
    bespoke schema. Events schema_version 2 carries them; version-1 files
    still grade unchanged."""
    attrs: dict[str, Any] = {}
    event_type = event.get("type")
    if event_type in {"command", "tool_call"}:
        attrs["gen_ai.operation.name"] = "execute_tool"
        attrs["gen_ai.tool.name"] = str(event.get("name") or "bash")
        if event.get("input_summary"):
            attrs["gen_ai.tool.call.arguments"] = event["input_summary"]
        if event.get("output_summary"):
            attrs["gen_ai.tool.call.result"] = event["output_summary"]
    elif event_type == "message":
        attrs["gen_ai.operation.name"] = "chat"
        if event.get("role"):
            attrs["gen_ai.message.role"] = event["role"]
    elif event_type == "error":
        attrs["error.type"] = str(event.get("name") or event.get("input_summary") or "error")[:120]
    elif event_type in {"file_read", "file_write", "skill_load"}:
        if event.get("input_summary"):
            attrs["file.path"] = event["input_summary"][:500]
    tokens = event.get("tokens")
    if isinstance(tokens, dict):
        if (input_tokens := _num(tokens.get("input_tokens"))) is not None:
            attrs["gen_ai.usage.input_tokens"] = int(input_tokens)
        if (output_tokens := _num(tokens.get("output_tokens"))) is not None:
            attrs["gen_ai.usage.output_tokens"] = int(output_tokens)
    if isinstance(event.get("exit_code"), int):
        attrs["process.exit_code"] = event["exit_code"]
    if event.get("is_error") is True:
        attrs["error.type"] = str(event.get("name") or "tool_result_error")[:120]
    return attrs


def _claude_tool_flat_record(name: str, tool_input: Any) -> dict[str, Any]:
    """One Claude tool_use block as a normalizer-native record. Bash is a shell
    command; Read/Write/Edit are file operations (a SKILL.md path classifies as
    skill_load in normalize_trace_record); the Skill tool is a skill load; any
    other tool stays a generic tool call with its input preserved."""
    inp = tool_input if isinstance(tool_input, dict) else {}
    if name == "Bash":
        return {"type": "command", "tool": name, "command": str(inp.get("command") or "")}
    if name == "Read":
        return {"type": "file_read", "name": name, "path": str(inp.get("file_path") or "")}
    if name in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
        return {"type": "file_write", "name": name, "path": str(inp.get("file_path") or inp.get("notebook_path") or "")}
    if name == "Skill":
        return {"type": "skill_load", "name": name, "path": str(inp.get("skill") or "")}
    return {"type": "tool_use", "tool": name, "input": inp}


def _claude_protocol_error(message: str) -> dict[str, Any]:
    return {
        "type": "error", "status": "failed", "message": message,
        "_trace_protocol_invalid": True,
    }


def claude_stream_flat_records(records: list[dict[str, Any]], *,
                               record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Flatten `claude -p --output-format stream-json` events into (line, record)
    pairs the generic normalizer understands. A message wraps several content
    blocks, so one raw line can yield several records — each keeps the RAW line
    for its raw_ref. Lifecycle is paired: a tool_use OPENS a call (in progress)
    and the matching tool_result COMPLETES it, so an orphaned call (crash
    mid-tool) counts zero under the completed-events-only metrics contract.
    Usage rides ONLY the terminal result event — per-assistant-message usage is
    API-request-level and would double count the cumulative total."""
    flat: list[tuple[int, dict[str, Any]]] = []
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    open_calls: dict[str, tuple[int, dict[str, Any]]] = {}
    seen_call_ids: set[str] = set()
    for ordinal, record in enumerate(records, 1):
        line = record_lines[ordinal - 1] if record_lines is not None else ordinal
        rtype = str(record.get("type") or "")
        if rtype not in {"assistant", "user"}:
            flat.append((line, record))
            continue
        if not isinstance(record.get("message"), dict):
            flat.append((line, _claude_protocol_error(
                f"Claude {rtype} record has no message object")))
            continue
        message = record["message"]
        role = str(message.get("role") or rtype)
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                flat.append((line, {"type": "message", "role": role, "text": content}))
            continue
        if not isinstance(content, list):
            flat.append((line, _claude_protocol_error(
                "Claude message content must be a string or list")))
            continue
        for block in content:
            if not isinstance(block, dict):
                flat.append((line, _claude_protocol_error(
                    "Claude message content blocks must be objects")))
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    flat.append((line, _claude_protocol_error(
                        "Claude text block text must be a string")))
                elif text.strip():
                    flat.append((line, {"type": "message", "role": role, "text": text}))
            elif btype == "thinking":
                if not isinstance(block.get("thinking"), str):
                    flat.append((line, _claude_protocol_error(
                        "Claude thinking block must contain thinking text")))
            elif btype == "redacted_thinking":
                if not isinstance(block.get("data"), str):
                    flat.append((line, _claude_protocol_error(
                        "Claude redacted_thinking block must contain data")))
            elif btype == "tool_use":
                name, raw_input, call_id = block.get("name"), block.get("input"), block.get("id")
                if not isinstance(name, str) or not name.strip():
                    flat.append((line, _claude_protocol_error(
                        "Claude tool_use name must be a non-empty string")))
                    continue
                if not isinstance(raw_input, dict):
                    flat.append((line, _claude_protocol_error(
                        "Claude tool_use input must be an object")))
                    continue
                if not isinstance(call_id, str) or not call_id.strip():
                    flat.append((line, _claude_protocol_error(
                        "Claude tool_use id must be a non-empty string")))
                    continue
                spec = _claude_tool_flat_record(name, raw_input)
                if call_id in seen_call_ids:
                    lifecycle = "open " if call_id in open_calls else "reused "
                    flat.append((line, _claude_protocol_error(
                        f"duplicate {lifecycle}Claude tool_use call id {call_id!r}")))
                    continue
                flat.append((line, {**spec, "status": "in_progress"}))
                seen_call_ids.add(call_id)
                open_calls[call_id] = (line, spec)
            elif btype == "tool_result":
                call_id = block.get("tool_use_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    flat.append((line, _claude_protocol_error(
                        "Claude tool_result tool_use_id must be a non-empty string")))
                    continue
                if "is_error" in block and not isinstance(block["is_error"], bool):
                    flat.append((line, _claude_protocol_error(
                        "Claude tool_result is_error must be boolean")))
                    continue
                if not isinstance(block.get("content"), (str, list)):
                    flat.append((line, _claude_protocol_error(
                        "Claude tool_result content must be a string or list")))
                    continue
                matched_call = open_calls.pop(call_id, None)
                if matched_call is None:
                    flat.append((line, {
                        **_claude_protocol_error(
                            f"unmatched Claude tool_result for call id {call_id!r}"),
                        "result": stringify_trace_value(block.get("content"))[:1000],
                        "_raw_result_line": line,
                    }))
                    continue
                call_line, spec = matched_call
                completion = {**spec, "status": "completed",
                              "output": stringify_trace_value(block.get("content"))[:1000],
                              "_raw_call_line": call_line, "_raw_result_line": line}
                if block.get("is_error"):
                    completion["is_error"] = True
                flat.append((line, completion))
            else:
                flat.append((line, _claude_protocol_error(
                    f"unsupported Claude content block type {btype!r}")))
    for call_id, (call_line, _) in sorted(open_calls.items()):
        flat.append((call_line, _claude_protocol_error(
            f"Claude tool_use call id {call_id!r} has no matching tool_result")))
    return flat


def identity_flat_records(records: list[dict[str, Any]], *,
                          record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """One raw record per physical line, unchanged — the default flatten."""
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    return [((record_lines[i - 1] if record_lines is not None else i), record)
            for i, record in enumerate(records, 1)]


def pi_stream_flat_records(records: list[dict[str, Any]], *,
                           record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Pi's terminal `tool_execution_end`/`tool_execution_update` carries only
    toolCallId, toolName, result, and isError — the invocation's path or
    command lives on the matching `tool_execution_start`. Without correlating
    the two by toolCallId, a completed read or edit has no path to classify
    against and can only ever read as a bare command."""
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    pending_args: dict[str, Any] = {}
    flat: list[tuple[int, dict[str, Any]]] = []
    for i, record in enumerate(records, 1):
        line = record_lines[i - 1] if record_lines is not None else i
        top_type = str(record.get("type") or "")
        call_id = record.get("toolCallId")
        args = record.get("args")
        if top_type == "tool_execution_start" and isinstance(call_id, str) and isinstance(args, dict):
            pending_args[call_id] = args
        elif (top_type in {"tool_execution_end", "tool_execution_update"}
              and isinstance(call_id, str) and "args" not in record
              and call_id in pending_args):
            record = {**record, "args": pending_args[call_id]}
        flat.append((line, record))
    return flat


def codex_stream_flat_records(records: list[dict[str, Any]], *,
                              record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]:
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    flat: list[tuple[int, dict[str, Any]]] = []
    for i, record in enumerate(records, 1):
        line = record_lines[i - 1] if record_lines is not None else i
        item = record.get("item")
        changes = item.get("changes") if isinstance(item, dict) else None
        if (record.get("type") == "item.completed" and isinstance(item, dict)
                and item.get("type") == "file_change" and isinstance(changes, list) and changes
                and all(isinstance(change, dict) for change in changes)
                and parse_event_state(item.get("status"), raw_type="item.completed",
                                      status_present="status" in item).state is EventState.COMPLETED):
            flat.extend((line, {
                "type": "file_write",
                "name": "file_change",
                "path": stringify_trace_value(change.get("path")),
                "change_kind": stringify_trace_value(change.get("kind")),
                "status": "completed",
            }) for change in changes)
        else:
            flat.append((line, record))
    return flat


VIBE_HISTORY_CONTENT_ENTRY_TYPES = frozenset({"message", "reasoning", "effect", "callback"})


VIBE_HISTORY_METADATA_ENTRY_TYPES = frozenset({"checkpoint", "notice"})


def vibe_records_are_history_entries(records: Sequence[Mapping[str, Any]]) -> bool:
    """True for a Vibe 2.23+ stream of public history entries, False for the
    LLMMessage dumps Vibe 2.22 and earlier wrote (they carry no `type`)."""
    return bool(records) and "type" in records[0]


def vibe_entry_text(entry: Mapping[str, Any]) -> str:
    """The text of a Vibe 2.23+ `message` entry, joined as Vibe's own
    PublicMessageEntry.text joins it."""
    content = entry.get("content")
    if not isinstance(content, list):
        return ""
    return "\n\n".join(block["text"] for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                        and isinstance(block.get("text"), str))


def _vibe_tool_flat_record(name: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """One Vibe tool call in the harness trace vocabulary, for the tools the
    harness enables (VIBE_READ_ONLY_TOOLS); None for any other tool. Reads the
    LLMMessage argument spellings and the 2.23+ camelCase effect input."""
    if name == "skill":
        return {"type": "skill_load", "name": name,
                "path": str(arguments.get("name") or arguments.get("skill") or "")}
    if name == "read_file":
        return {"type": "file_read", "name": name,
                "path": str(arguments.get("path") or arguments.get("file_path")
                            or arguments.get("filePath") or "")}
    if name == "grep":
        return {"type": "tool_use", "tool": name, "input": dict(arguments)}
    return None


def vibe_stream_flat_records(records: list[dict[str, Any]], *,
                             record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Flatten a Vibe stream in whichever record shape it is in."""
    if vibe_records_are_history_entries(records):
        return vibe_history_entry_flat_records(records, record_lines=record_lines)
    return vibe_llm_message_flat_records(records, record_lines=record_lines)


def vibe_history_entry_flat_records(
    records: list[dict[str, Any]], *, record_lines: list[int] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Flatten Vibe 2.23+ public history entries. Vibe writes an entry once it
    is completed, so an `effect` entry is a whole tool call: its input under
    `detail`, its outcome under `state`."""
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    flat: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()

    def invalid(line: int, message: str) -> None:
        flat.append((line, _claude_protocol_error(f"Vibe {message}")))

    for ordinal, record in enumerate(records, 1):
        line = record_lines[ordinal - 1] if record_lines is not None else ordinal
        kind, entry_id = record.get("type"), record.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            invalid(line, "history entry id must be a non-empty string")
            continue
        if entry_id in seen:
            invalid(line, f"history entry id {entry_id!r} is duplicated")
            continue
        seen.add(entry_id)
        if record.get("generationStatus") != "completed":
            invalid(line, f"history entry {entry_id!r} is not completed")
            continue
        if kind == "message":
            role, content = record.get("role"), record.get("content")
            if role not in {"user", "assistant", "system"}:
                invalid(line, f"message role {role!r} is unsupported")
            elif not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
                invalid(line, "message content must be a list of content blocks")
            elif vibe_entry_text(record).strip():
                flat.append((line, {"type": "message", "role": role, "text": vibe_entry_text(record)}))
        elif kind == "effect":
            detail, state = record.get("detail"), record.get("state")
            name = detail.get("toolName") if isinstance(detail, dict) else None
            arguments = detail.get("input") if isinstance(detail, dict) else None
            if not isinstance(name, str) or not name.strip() or not isinstance(state, dict):
                invalid(line, "effect must carry a detail.toolName and a state")
                continue
            spec = _vibe_tool_flat_record(name, arguments if isinstance(arguments, dict) else {})
            if spec is None:
                invalid(line, f"tool call function {name!r} is unsupported")
                continue
            status = state.get("status")
            if status not in {"completed", "failed", "cancelled", "skipped"}:
                invalid(line, f"effect {entry_id!r} state {status!r} is not a finished tool call")
                continue
            flat.append((line, {**spec, "status": "completed" if status == "completed" else "failed",
                                **({} if status == "completed" else {"is_error": True}),
                                "output": stringify_trace_value(
                                    state.get("output") if status == "completed"
                                    else state.get("error") or state.get("reason"))[:1000],
                                "_raw_call_line": line, "_raw_result_line": line}))
        elif kind == "reasoning":
            if not isinstance(record.get("text"), str):
                invalid(line, "reasoning entry text must be a string")
        elif kind == "callback" or kind in VIBE_HISTORY_METADATA_ENTRY_TYPES:
            flat.append((line, record))
        else:
            invalid(line, f"history entry type {kind!r} is unsupported")
    return flat


def vibe_llm_message_flat_records(records: list[dict[str, Any]], *,
                                  record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]:
    """Flatten Vibe 2.22's OpenAI-style message/tool lifecycle without dropping it."""
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    flat: list[tuple[int, dict[str, Any]]] = []
    open_calls: dict[str, tuple[int, dict[str, Any]]] = {}
    seen: set[str] = set()

    def invalid(line: int, message: str) -> None:
        flat.append((line, _claude_protocol_error(f"Vibe {message}")))

    for ordinal, record in enumerate(records, 1):
        line = record_lines[ordinal - 1] if record_lines is not None else ordinal
        role = record.get("role")
        tool_calls = record.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant" or not isinstance(tool_calls, list):
                invalid(line, "tool_calls must be a list on an assistant message")
                continue
            for call in tool_calls:
                if not isinstance(call, dict):
                    invalid(line, "tool_calls entries must be objects")
                    continue
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not call_id.strip():
                    invalid(line, "tool call id must be a non-empty string")
                    continue
                if call_id in seen:
                    invalid(line, f"tool call id {call_id!r} is duplicated or reused")
                    continue
                if not isinstance(function, dict):
                    invalid(line, "tool call function must be an object")
                    continue
                name, arguments = function.get("name"), function.get("arguments")
                if not isinstance(name, str) or not name.strip():
                    invalid(line, "tool call function name must be a non-empty string")
                    continue
                if isinstance(arguments, str):
                    try:
                        arguments = stream_json_loads(arguments)
                    except json.JSONDecodeError:
                        invalid(line, "tool call arguments must be a JSON object")
                        continue
                if not isinstance(arguments, dict):
                    invalid(line, "tool call arguments must be an object")
                    continue
                spec = _vibe_tool_flat_record(name, arguments)
                if spec is None:
                    invalid(line, f"tool call function {name!r} is unsupported")
                    continue
                flat.append((line, {**spec, "status": "in_progress"}))
                open_calls[call_id] = (line, spec)
                seen.add(call_id)
        content = record.get("content")
        if role == "tool":
            call_id = record.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                invalid(line, "tool result tool_call_id must be a non-empty string")
                continue
            matched = open_calls.pop(call_id, None)
            if matched is None:
                invalid(line, f"tool result for unknown call id {call_id!r}")
                continue
            call_line, spec = matched
            flat.append((line, {**spec, "status": "completed",
                                "output": stringify_trace_value(content)[:1000],
                                "_raw_call_line": call_line,
                                "_raw_result_line": line}))
        elif role in {"assistant", "user", "system"}:
            if content is not None and not isinstance(content, str):
                invalid(line, "message content must be a string")
            elif isinstance(content, str) and content.strip():
                flat.append((line, {"type": "message", "role": role,
                                    "text": content,
                                    **({"usage": record["usage"]}
                                       if isinstance(record.get("usage"), dict) else {})}))
        elif role is not None:
            invalid(line, f"message role {role!r} is unsupported")
        elif tool_calls is None:
            invalid(line, "record must have a supported role")
    for call_id, (line, _) in sorted(open_calls.items()):
        invalid(line, f"tool call id {call_id!r} has no matching tool result")
    return flat


def _gemini_tool_flat_record(name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Map one official Gemini tool name into the harness trace vocabulary."""
    values = dict(parameters)
    common = {"parameters": values}
    if name == "run_shell_command":
        return {"type": "command", "tool": name,
                "command": str(values.get("command") or ""), **common}
    if name == "activate_skill":
        return {"type": "skill_load", "name": name,
                "path": str(values.get("name") or ""), **common}
    if name in {"write_file", "replace"}:
        return {"type": "file_write", "name": name,
                "path": str(values.get("file_path") or ""), **common}
    if name == "read_file":
        path = str(values.get("file_path") or "")
    elif name in {"list_directory", "grep_search", "glob"}:
        path = str(values.get("dir_path") or "")
    elif name == "read_many_files":
        path = ""
    else:
        path = ""
    if name in GEMINI_READ_ONLY_TOOLS:
        return {"type": "file_read", "name": name, "path": path, **common}
    return {"type": "tool_use", "tool": name, "input": values}


def gemini_stream_flat_records(
    records: list[dict[str, Any]], *, record_lines: list[int] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Flatten Gemini's paired tool events without discarding raw line refs."""
    if record_lines is not None and len(record_lines) != len(records):
        raise ValueError("record_lines must have one physical line per trace record")
    flat: list[tuple[int, dict[str, Any]]] = []
    open_calls: dict[str, tuple[int, dict[str, Any]]] = {}
    for ordinal, record in enumerate(records, 1):
        line = record_lines[ordinal - 1] if record_lines is not None else ordinal
        kind = record.get("type")
        if kind == "tool_use":
            call_id = record.get("tool_id")
            name = record.get("tool_name")
            parameters = record.get("parameters")
            if (not isinstance(call_id, str) or not call_id.strip()
                    or not isinstance(name, str) or not name.strip()
                    or not isinstance(parameters, Mapping)):
                flat.append((line, _claude_protocol_error(
                    "Gemini tool_use has an invalid id, name, or parameters")))
                continue
            spec = _gemini_tool_flat_record(name, parameters)
            flat.append((line, {**spec, "status": "in_progress"}))
            open_calls[call_id] = (line, spec)
            continue
        if kind == "tool_result":
            call_id = record.get("tool_id")
            matched = open_calls.pop(call_id, None) if isinstance(call_id, str) else None
            if matched is None:
                flat.append((line, _claude_protocol_error(
                    "Gemini tool_result has no matching tool_use")))
                continue
            call_line, spec = matched
            flat.append((line, {
                **spec,
                "status": (
                    "failed" if record.get("status") == "error"
                    else "completed"
                ),
                "output": stringify_trace_value(
                    record.get("output") or record.get("error"))[:1000],
                "is_error": record.get("status") == "error",
                "_raw_call_line": call_line,
                "_raw_result_line": line,
            }))
            continue
        if kind == "error" and record.get("severity") == "warning":
            flat.append((line, {
                "type": "warning",
                "severity": "warning",
                "message": record.get("message"),
                "timestamp": record.get("timestamp"),
            }))
            continue
        if kind == "message":
            flat.append((line, {
                "type": "message",
                "role": record.get("role"),
                "text": record.get("content"),
                "timestamp": record.get("timestamp"),
            }))
            continue
        flat.append((line, record))
    for call_id, (line, _) in sorted(open_calls.items()):
        flat.append((line, _claude_protocol_error(
            f"Gemini tool_use call id {call_id!r} has no matching tool_result")))
    return flat


def _gemini_records_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "\n".join(json.dumps(dict(record), ensure_ascii=False)
                     for record in records) + ("\n" if records else "")


def _gemini_stream_semantics(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> tuple[dict[str, Any] | None, str | None]:
    _ = pi_stream
    parsed = GeminiStream.parse(_gemini_records_text(records))
    failure = parsed.protocol_error or parsed.provider_error
    return (dict(parsed.usage) if parsed.usage is not None and failure is None
            else None), failure


def _gemini_usage_and_cost_blocks(
    raw_text: str, pi_stream: PiStream | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _ = pi_stream
    parsed = GeminiStream.parse(raw_text)
    if not parsed.complete or parsed.usage is None:
        return {"source": "missing"}, {"source": "missing"}
    return (
        normalize_usage(dict(parsed.usage), source="trace_normalized"),
        {"source": "missing"},
    )


def _no_stream_semantics(records: list[dict[str, Any]], pi_stream: PiStream | None) -> tuple[dict[str, Any] | None, str | None]:
    """No terminal cumulative usage and no stream-level failure: per-record
    token accumulation applies."""
    return None, None


def _no_retry_observation(records: list[dict[str, Any]], pi_stream: PiStream | None) -> int | None:
    return None


def _pi_retries(records: list[dict[str, Any]], pi_stream: PiStream | None) -> int | None:
    return (pi_stream or PiStream.from_records(records)).retries


def _pi_stream_semantics(records: list[dict[str, Any]], pi_stream: PiStream | None) -> tuple[dict[str, Any] | None, str | None]:
    """Pi repeats final cumulative usage on message_end, turn_end, and
    agent_end — one response, not several token deltas — and a terminal
    failure invalidates the stream's usage entirely. An already-parsed
    PiStream may be passed to avoid re-parsing."""
    parsed = pi_stream or PiStream.from_records(records)
    error = parsed.failure_error
    return (parsed.terminal_usage if not error else None), error


def _generic_usage_and_cost_blocks(raw_text: str, pi_stream: PiStream | None) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [obj for obj in iter_json_objects(raw_text, strict=False) if isinstance(obj, dict)]
    return _generic_stream_usage_and_cost(records)


def _pi_usage_and_cost_blocks(raw_text: str, pi_stream: PiStream | None) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = pi_stream or PiStream.parse(raw_text)
    return dict(parsed.usage_normalized), dict(parsed.cost_normalized)


def _generic_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    return None


def _codex_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    terminals = [i for i, record in enumerate(records)
                 if str(record.get("type") or "").casefold() == "turn.completed"]
    if terminals != [len(records) - 1]:
        return "Codex trace must contain exactly one final turn.completed event"
    return None


def _claude_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    if claude_terminal_result_index(records) is None:
        return "Claude trace must contain exactly one final result event"
    return None


def _vibe_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    if vibe_records_are_history_entries(records):
        content = [record for record in records
                   if record.get("type") in VIBE_HISTORY_CONTENT_ENTRY_TYPES]
        terminal_answer = (bool(content) and content[-1].get("type") == "message"
                           and content[-1].get("role") == "assistant"
                           and bool(vibe_entry_text(content[-1]).strip()))
    else:
        terminal_answer = (bool(records) and records[-1].get("role") == "assistant"
                           and isinstance(records[-1].get("content"), str)
                           and bool(records[-1]["content"].strip()))
    if not terminal_answer:
        return "Vibe trace must end with one non-empty assistant response"
    return None


def _gemini_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    _ = pi_stream
    return GeminiStream.parse(_gemini_records_text(records)).protocol_error


def _pi_trace_protocol_error(
    records: list[dict[str, Any]], pi_stream: PiStream | None,
) -> str | None:
    parsed = pi_stream or PiStream.from_records(records)
    return parsed.protocol_error


class TraceFlattener(Protocol):
    def __call__(self, records: list[dict[str, Any]], *,
                 record_lines: list[int] | None = None) -> list[tuple[int, dict[str, Any]]]: ...


@_dataclass(frozen=True)
class TraceDialect:
    """Per-provider trace semantics: how raw records flatten into
    normalizer-native (line, record) pairs — claude's block-structured stream
    expands to several records per raw line — how a stream's terminal
    usage/failure resolve — Pi's cumulative repeats must not be summed — and
    how a raw stream normalizes into usage/cost blocks. ONE registry instead
    of per-source branches scattered through the normalizer and
    stream_usage_and_cost. Generic-semantics providers are registered
    explicitly; misspelled or unsupported sources are rejected before a
    provider-specific stream can be silently normalized with the wrong rules."""

    flatten: TraceFlattener = identity_flat_records
    stream_semantics: Callable[[list[dict[str, Any]], PiStream | None], tuple[dict[str, Any] | None, str | None]] = _no_stream_semantics
    usage_and_cost: Callable[[str, PiStream | None], tuple[dict[str, Any], dict[str, Any]]] = _generic_usage_and_cost_blocks
    protocol_error: Callable[[list[dict[str, Any]], PiStream | None], str | None] = _generic_trace_protocol_error
    retries: Callable[[list[dict[str, Any]], PiStream | None], int | None] = _no_retry_observation


GENERIC_TRACE_DIALECT = TraceDialect()


CODEX_TRACE_DIALECT = TraceDialect(flatten=codex_stream_flat_records, protocol_error=_codex_trace_protocol_error)


JETTY_TRACE_DIALECT = TraceDialect(protocol_error=_jetty_trace_protocol_error)


VIBE_TRACE_DIALECT = TraceDialect(
    flatten=vibe_stream_flat_records,
    protocol_error=_vibe_trace_protocol_error,
)


CLAUDE_TRACE_DIALECT = TraceDialect(
    flatten=claude_stream_flat_records,
    protocol_error=_claude_trace_protocol_error,
)


PI_TRACE_DIALECT = TraceDialect(
    flatten=pi_stream_flat_records,
    stream_semantics=_pi_stream_semantics,
    usage_and_cost=_pi_usage_and_cost_blocks,
    protocol_error=_pi_trace_protocol_error,
    retries=_pi_retries,
)


GEMINI_TRACE_DIALECT = TraceDialect(
    flatten=gemini_stream_flat_records,
    stream_semantics=_gemini_stream_semantics,
    usage_and_cost=_gemini_usage_and_cost_blocks,
    protocol_error=_gemini_trace_protocol_error,
)


TRACE_DIALECTS: dict[str, TraceDialect] = {
    "generic": GENERIC_TRACE_DIALECT,
    **trace_dialect_implementations(),
}


def trace_dialect_for(source: str) -> TraceDialect:
    if not isinstance(source, str) or not source:
        raise ValueError("trace source must be a non-empty string")
    key = source.casefold()
    try:
        return TRACE_DIALECTS[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported trace source {source!r}; known: {sorted(TRACE_DIALECTS)}") from exc


def normalize_trace_records(records: list[dict[str, Any]], *, source: str = "generic",
                            pi_stream: PiStream | None = None,
                            record_lines: list[int] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    dialect = trace_dialect_for(source)
    flat = dialect.flatten(records, record_lines=record_lines)
    events = [normalize_trace_record(record, source=source, index=i, line=line)
              for i, (line, record) in enumerate(flat, 1)]
    commands = [command_text(e) for e in command_events(events)]
    token_totals: dict[str, int | None] = {
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
    }
    elapsed_ms = 0.0
    terminal_usage, stream_error = dialect.stream_semantics(records, pi_stream)
    if terminal_usage is not None:
        for key in token_totals:
            value = usage_number(terminal_usage, key)
            if value is not None:
                token_totals[key] = value
    elif not stream_error:
        for (_, record), event in zip(flat, events):
            usage = record.get("usage") if isinstance(record.get("usage"), dict) else record.get("tokens")
            if isinstance(usage, dict):
                input_tokens = usage_number(usage, "input_tokens")
                output_tokens = usage_number(usage, "output_tokens")
                total_tokens = usage_number(usage, "total_tokens")
                if total_tokens is None and input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens
                for key, value in [("input_tokens", input_tokens), ("output_tokens", output_tokens), ("total_tokens", total_tokens)]:
                    if value is not None:
                        token_totals[key] = (token_totals[key] or 0) + value
            duration = _num(raw_trace_value(record, "duration_ms", "elapsed_ms"))
            if duration is not None:
                elapsed_ms += duration
            tokens = event.get("tokens")
            if isinstance(tokens, dict) and not isinstance(usage, dict):
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    value = usage_number(tokens, key)
                    if value is not None:
                        token_totals[key] = (token_totals[key] or 0) + value
    if (token_totals["total_tokens"] is None
            and token_totals["input_tokens"] is not None
            and token_totals["output_tokens"] is not None):
        token_totals["total_tokens"] = (
            token_totals["input_tokens"] + token_totals["output_tokens"])
    counts = trace_event_counts(events)
    skill_events = counts["skill_events"]
    metrics: dict[str, Any] = {
        "schema_version": 2,
        "source": source,
        "tool_calls": counts["tool_calls"],
        "commands": counts["commands"],
        "file_reads": counts["file_reads"],
        "file_writes": counts["file_writes"],
        "errors": counts["errors"],
        "repeated_command_max": repeated_command_max(commands),
        "skill_invoked": bool(skill_events),
        "skill_invocation_evidence": [command_text(e) or e.get("input_summary", "") for e in skill_events[:10]],
    }
    retries = dialect.retries(records, pi_stream)
    if retries is not None:
        metrics["retries"] = retries
    protocol_errors = [
        str(record.get("message") or "trace protocol error")
        for _, record in flat if record.get("_trace_protocol_invalid") is True
    ]
    if protocol_errors:
        metrics["trace_protocol_errors"] = protocol_errors[:20]
    if elapsed_ms:
        metrics["elapsed_ms"] = int(elapsed_ms)
    for key, value in token_totals.items():
        if value is not None:
            metrics[key] = value
    observed_tokens = {key: value for key, value in token_totals.items()
                       if value is not None}
    if observed_tokens:
        metrics["usage_normalized"] = normalize_usage(
            observed_tokens, source="trace_normalized")
    otel_usage = {}
    if token_totals["input_tokens"] is not None:
        otel_usage["gen_ai.usage.input_tokens"] = token_totals["input_tokens"]
    if token_totals["output_tokens"] is not None:
        otel_usage["gen_ai.usage.output_tokens"] = token_totals["output_tokens"]
    if otel_usage:
        metrics["otel"] = otel_usage
    event_doc = {"schema_version": 2, "source": source, "events": events}
    return event_doc, metrics


def _pi_final_message(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    message = record.get("message")
    if isinstance(message, dict):
        return string_keyed_dict(message, "Pi final message")
    messages = record.get("messages")
    if isinstance(messages, list):
        for candidate in reversed(messages):
            if isinstance(candidate, dict) and candidate.get("role") == "assistant":
                return string_keyed_dict(candidate, "Pi assistant message")
        for candidate in reversed(messages):
            if isinstance(candidate, dict):
                return string_keyed_dict(candidate, "Pi fallback message")
    return None


def _pi_final_agent_end(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((record for record in reversed(records) if str(record.get("type") or "") == "agent_end"), None)


def _pi_terminal_error(records: list[dict[str, Any]]) -> str | None:
    """Read only Pi's final retry attempt, never a historical failed message."""
    last_agent_end = _pi_final_agent_end(records)
    fallback = next((record for record in reversed(records) if str(record.get("type") or "") == "turn_end"), None)
    candidate = _pi_final_message(last_agent_end if last_agent_end is not None else fallback)
    if candidate is None:
        return None
    error = candidate.get("errorMessage") or candidate.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()
    if str(candidate.get("stopReason") or "").casefold() == "error":
        return "Pi provider ended the turn with stopReason:error"
    return None


def pi_stream_terminal_error(raw_text: str) -> str | None:
    """Compatibility boundary backed by the typed, single-pass Pi parser."""
    return PiStream.parse(raw_text).failure_error


def _pi_terminal_usage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return cumulative usage from Pi's final retry attempt only."""
    last_agent_end = _pi_final_agent_end(records)
    fallback = next((record for record in reversed(records)
                     if str(record.get("type") or "") in {"turn_end", "message_end"}), None)
    message = _pi_final_message(last_agent_end if last_agent_end is not None else fallback)
    return message.get("usage") if isinstance(message, dict) and isinstance(message.get("usage"), dict) else None


def _stream_usage_doc(record: dict[str, Any]) -> dict[str, Any] | None:
    usage = raw_trace_value(record, "usage", "tokens")
    return usage if isinstance(usage, dict) else None


def _is_cumulative_usage_record(record: dict[str, Any]) -> bool:
    top_type = str(raw_trace_value(record, "type", "event", "name", "kind") or "").casefold()
    item_type = nested_item_type(record).casefold()
    return top_type in {"result", "response.completed"} or item_type in {"result", "response.completed"}


def _sum_stream_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for record in records:
        usage = _stream_usage_doc(record)
        if not usage:
            continue
        normalized = normalize_usage(usage, source="trace_normalized")
        for key in USAGE_ALIASES:
            value = normalized.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + int(value)
    return normalize_usage(totals if totals else None, source="trace_normalized")


def _cost_value_from_record(record: dict[str, Any]) -> Any:
    """Return a validated raw cost value without discarding object currency."""
    raw = raw_trace_value(record, "cost", "cost_usd", "total_cost_usd")
    if raw is None:
        usage_obj = _stream_usage_doc(record)
        if isinstance(usage_obj, dict):
            raw = usage_obj.get("cost", usage_obj.get("cost_usd"))
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return raw if _num(raw) is not None else None
    if isinstance(raw, dict):
        return raw if normalize_cost(raw).get("source") != "missing" else None
    return None


def _generic_stream_usage_and_cost(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    cumulative_usage: list[dict[str, Any]] = []
    for record in records:
        if not _is_cumulative_usage_record(record):
            continue
        usage_doc = _stream_usage_doc(record)
        if not usage_doc:
            continue
        normalized = normalize_usage(usage_doc, source="trace_normalized")
        if normalized.get("source") != "missing":
            cumulative_usage.append(normalized)
    usage = cumulative_usage[-1] if cumulative_usage else _sum_stream_usage(records)

    cumulative_cost: list[dict[str, Any]] = []
    for record in records:
        if not _is_cumulative_usage_record(record):
            continue
        value = _cost_value_from_record(record)
        if value is None:
            continue
        normalized = normalize_cost(value, source="trace_normalized")
        if normalized.get("source") != "missing":
            cumulative_cost.append(normalized)
    if cumulative_cost:
        return usage, cumulative_cost[-1]
    cost_values = [_cost_value_from_record(record) for record in records]
    cost_blocks = [normalize_cost(value, source="trace_normalized") for value in cost_values if value is not None]
    cost_blocks = [block for block in cost_blocks if block.get("source") != "missing"]
    currencies = {str(block.get("currency")) for block in cost_blocks}
    if len(currencies) == 1:
        currency = next(iter(currencies))
        cost_total = sum(float(block["total_cost"]) for block in cost_blocks)
        cost = normalize_cost({"amount": round(cost_total, 6), "currency": currency}, source="trace_normalized")
    else:
        cost = {"source": "missing"}
    return usage, cost


@_dataclass(frozen=True)
class PiStream:
    """One parsed Pi JSON stream with intrinsic terminal semantics.

    Provider status and cumulative telemetry are derived together once. Callers
    pass this object to detection, telemetry, and trace normalization rather than
    reinterpreting the same wire text with independent policies.
    """

    records: tuple[dict[str, Any], ...]
    parse_errors: tuple[str, ...]
    terminal_error: str | None
    protocol_error: str | None
    terminal_usage: dict[str, Any] | None
    usage_normalized: dict[str, Any]
    cost_normalized: dict[str, Any]
    retries: int | None = None

    def __post_init__(self) -> None:
        if self.failure_error and (
            self.usage_normalized != {"source": "missing"}
            or self.cost_normalized != {"source": "missing"}
        ):
            raise ValueError("failed Pi streams cannot carry measured usage or cost")
        if self.retries is not None and (
            isinstance(self.retries, bool) or not isinstance(self.retries, int) or self.retries < 0
        ):
            raise ValueError("Pi retry count must be a non-negative integer or None")
        if self.protocol_error and self.retries is not None:
            raise ValueError("protocol-invalid Pi streams cannot carry a retry count")

    @property
    def failure_error(self) -> str | None:
        return self.terminal_error or self.protocol_error

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]],
                     parse_errors: Iterable[str] = ()) -> PiStream:
        materialized = [dict(record) for record in records]
        terminal_error = _pi_terminal_error(materialized)
        final_agent_end = _pi_final_agent_end(materialized)
        terminal_seen = final_agent_end is not None and final_agent_end.get("willRetry") is not True
        errors = tuple(parse_errors)
        protocol_error = (
            f"Pi JSON stream parse error: {errors[0]}" if errors
            else None if terminal_seen
            else "Pi JSON stream ended without a final agent_end event"
        )
        terminal_usage = _pi_terminal_usage(materialized)
        if terminal_error or protocol_error:
            usage, cost = {"source": "missing"}, {"source": "missing"}
        elif terminal_usage is not None:
            usage = normalize_usage(terminal_usage, source="trace_normalized")
            cost = normalize_cost(terminal_usage.get("cost"), source="trace_normalized")
        else:
            usage, cost = _generic_stream_usage_and_cost(materialized)
        attempt_ends = [record for record in materialized if record.get("type") == "agent_end"]
        last_attempt_event = next((record for record in reversed(materialized)
                                   if record.get("type") in {"agent_start", "agent_end"}), None)
        retries = None
        if (not protocol_error and last_attempt_event is final_agent_end
                and all("willRetry" not in record or isinstance(record["willRetry"], bool)
                        for record in attempt_ends)):
            retries = sum(record.get("willRetry") is True for record in attempt_ends)
        return cls(tuple(materialized), errors, terminal_error, protocol_error,
                   dict(terminal_usage) if terminal_usage is not None else None,
                   usage, cost, retries)

    @classmethod
    def parse(cls, raw_text: str) -> PiStream:
        if not isinstance(raw_text, str):
            raise TypeError("Pi stream must be text")
        records, errors = parse_trace_jsonl_text(raw_text, strict=False)
        return cls.from_records(records, errors)


def stream_usage_and_cost(raw_text: str, *, source: str | None = None,
                          pi_stream: PiStream | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize one runner stream without confusing Pi cumulative events for
    deltas — resolved by the source's registered trace dialect."""
    resolved_source = "generic" if source is None else source
    return trace_dialect_for(resolved_source).usage_and_cost(raw_text, pi_stream)


def write_trace_artifacts(
    run_dir: Path,
    trace_text: str,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    write_metadata: bool = True,
    out_events: Path | None = None,
    out_metrics: Path | None = None,
    write_raw_trace: bool = True,
    pi_stream: PiStream | None = None,
    process_observation_complete: bool | None = None,
    provider_response_complete: bool | None = None,
    artifact_set_complete: bool | None = None,
    retain_invalid_provider_trace: bool = False,
    trace_utf8_valid: bool = True,
    mediated_tools: tool_fault_domain.ToolObservation | None = None,
    tool_journal_path: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if tool_fault_domain.MEDIATED_KEYS & set(metadata or {}):
        raise ValueError("metadata cannot author mediated tool evidence")
    if not isinstance(retain_invalid_provider_trace, bool):
        raise TypeError("retain_invalid_provider_trace must be boolean")
    if not isinstance(trace_utf8_valid, bool):
        raise TypeError("trace_utf8_valid must be boolean")
    for label, value in (("process_observation_complete", process_observation_complete),
                         ("provider_response_complete", provider_response_complete),
                         ("artifact_set_complete", artifact_set_complete)):
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"{label} must be boolean or None")
    run_dir.mkdir(parents=True, exist_ok=True)
    if write_raw_trace:
        (run_dir / "trace.jsonl").write_text(trace_text, encoding="utf-8")
    parsed_records, parsed_errors, record_lines, duplicate_keys = parse_trace_jsonl_text_with_lines(
        trace_text, strict_json_errors=not retain_invalid_provider_trace, strict=False)
    if source.casefold() == "pi" and pi_stream is not None:
        records, parse_errors = list(pi_stream.records), list(pi_stream.parse_errors)
        if len(record_lines) != len(records):
            record_lines = list(range(1, len(records) + 1))
    else:
        records, parse_errors = parsed_records, parsed_errors
    if source.casefold() == "claude" and retain_invalid_provider_trace:
        retained_records = []
        for line, record in zip(record_lines, records):
            try:
                for holder in (record, record.get("message")):
                    if isinstance(holder, dict) and isinstance(holder.get("usage"), dict):
                        normalize_usage(holder["usage"], source="trace_normalized")
            except ValueError as exc:
                parse_errors.append(f"line {line}: invalid Claude usage: {exc}")
                retained_records.append(_claude_protocol_error(f"invalid Claude usage: {exc}"))
            else:
                retained_records.append(record)
        records = retained_records
    events, metrics = normalize_trace_records(
        records, source=source, pi_stream=pi_stream, record_lines=record_lines)
    semantic_protocol_error = trace_dialect_for(source).protocol_error(records, pi_stream)
    if semantic_protocol_error:
        existing_protocol_errors = metrics.get("trace_protocol_errors")
        protocol_errors = (list(existing_protocol_errors)
                           if isinstance(existing_protocol_errors, list) else [])
        protocol_errors.append(semantic_protocol_error)
        metrics["trace_protocol_errors"] = protocol_errors[:20]
    if not trace_utf8_valid:
        protocol_errors = list(metrics.get("trace_protocol_errors") or [])
        protocol_errors.append("trace transport is not valid UTF-8")
        metrics["trace_protocol_errors"] = protocol_errors[:20]
    if parse_errors:
        metrics["parse_errors"] = parse_errors[:20]
        metrics["errors"] = int(metrics.get("errors", 0) or 0) + len(parse_errors)
    if duplicate_keys:
        metrics["stream_duplicate_keys"] = duplicate_keys[:20]
    trace_observation_complete = (
        bool(records) and not parse_errors and not metrics.get("trace_protocol_errors"))
    reserved = set(metrics) | {
        "observation_complete", "trace_observation_complete", "process_observation_complete",
        "provider_response_complete", "operation_observation_complete",
        "artifact_set_complete", "observation_evidence", "telemetry",
        "telemetry_schema_version", "usage_normalized", "cost_normalized",
        "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens",
        "cache_write_tokens", "cache_creation_tokens", "cost_usd", "otel",
        "parse_errors", "trace_protocol_errors", "stream_duplicate_keys",
        "skill_invoked", "skill_invocation_evidence", "retries",
        "repeated_command_max", "commands", "tool_calls", "file_reads",
        "file_writes", "errors", "schema_version", "source",
    }
    process_metric_keys = {"returncode", "timed_out", "elapsed_ms"}
    reserved.update(tool_fault_domain.MEDIATED_KEYS)
    collisions = (reserved - process_metric_keys) & set(extra_metrics or {})
    if collisions:
        raise ValueError(f"extra_metrics cannot override derived evidence: {', '.join(sorted(collisions))}")
    if extra_metrics:
        metrics.update(extra_metrics)
    generic_complete = metrics.get("observation_complete")
    if not isinstance(generic_complete, bool):
        generic_complete = (metadata or {}).get("observation_complete")
    if process_observation_complete is None:
        explicit = (metadata or {}).get("process_observation_complete")
        process_observation_complete = explicit if isinstance(explicit, bool) else generic_complete if isinstance(generic_complete, bool) else None
    if provider_response_complete is None:
        explicit = (metadata or {}).get("provider_response_complete")
        provider_response_complete = explicit if isinstance(explicit, bool) else generic_complete if isinstance(generic_complete, bool) else None
    evidence = telemetry_domain.ObservationEvidence(
        telemetry_domain.ObservationEvidence.state(process_observation_complete),
        telemetry_domain.ObservationEvidence.state(provider_response_complete),
        telemetry_domain.ObservationEvidence.state(trace_observation_complete),
        telemetry_domain.ObservationEvidence.state(artifact_set_complete),
    )
    legacy_observation_complete = (
        provider_response_complete if isinstance(provider_response_complete, bool)
        else generic_complete if isinstance(generic_complete, bool)
        else False
    )
    metrics.update({
        "observation_complete": legacy_observation_complete,
        "trace_observation_complete": trace_observation_complete,
        "process_observation_complete": process_observation_complete,
        "provider_response_complete": provider_response_complete,
        "operation_observation_complete": evidence.operation_complete,
        "artifact_set_complete": artifact_set_complete,
        "observation_evidence": evidence.to_dict(),
    })
    for key in ("usage_normalized", "cost_normalized"):
        block = (metadata or {}).get(key)
        if isinstance(block, dict) and block.get("source") in {"provider_reported", "trace_normalized", "price_table_estimated", "estimated", "not_applicable"}:
            metrics[key] = block
        metrics.setdefault(key, {"source": "missing"})
    usage_block = metrics["usage_normalized"]
    if isinstance(usage_block, dict):
        for key in ("input_tokens", "output_tokens", "total_tokens",
                    "cache_read_tokens", "cache_write_tokens"):
            value = usage_block.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                metrics[key] = value
        if isinstance(usage_block.get("cache_write_tokens"), int):
            metrics["cache_creation_tokens"] = usage_block["cache_write_tokens"]
    cost_block = metrics["cost_normalized"]
    if (isinstance(cost_block, dict) and cost_block.get("currency") == "USD"
            and isinstance(cost_block.get("total_cost"), (int, float))
            and not isinstance(cost_block.get("total_cost"), bool)):
        metrics["cost_usd"] = float(cost_block["total_cost"])
    envelope_input = {**(metadata or {}), **metrics}
    if not evidence.operation_complete:
        for key in ("usage_normalized", "cost_normalized"):
            block = envelope_input.get(key)
            if isinstance(block, dict) and block.get("source") == "trace_normalized":
                envelope_input[key] = {"source": "missing"}
    tool_measurements = _tool_measurements(mediated_tools, telemetry_domain.basis_from_run(envelope_input)) if mediated_tools is not None else None
    if mediated_tools is not None:
        if not tool_journal_path:
            raise ValueError("mediated tools require a stable journal path")
        events["mediated_tools"] = mediated_tools.as_member(tool_journal_path)
        if parse_errors or metrics.get("errors"):
            metrics["provider_trace_errors"] = metrics.get("errors")
        for key, value in (tool_measurements or {}).items():
            metrics.pop(key, None)
            if value.availability == telemetry_domain.AVAILABLE:
                metrics[key] = value.value
    envelope = telemetry_domain.telemetry_envelope(
        envelope_input,
        tool_measurements=tool_measurements,
        source=source,
        population=str(envelope_input.get("population") or "answer"),
    )
    metrics["telemetry_schema_version"] = 3
    metrics["telemetry"] = envelope
    write_json(out_events or run_dir / "events.json", events)
    write_json(out_metrics or run_dir / "metrics.json", metrics)
    if environment:
        write_json(run_dir / "environment.json", environment)
    if write_metadata:
        existing: dict[str, Any] = {}
        if metadata:
            existing.update(metadata)
        existing.update({k: v for k, v in metrics.items() if k not in {"schema_version", "source"}})
        existing.setdefault("usage_normalized", {"source": "missing"})
        existing.setdefault("cost_normalized", {"source": "missing"})
        existing["telemetry_schema_version"] = 3
        existing["telemetry"] = envelope
        existing["trace_source"] = source
        write_json(run_dir / "metadata.json", existing)
    return events, metrics


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_vibe_messages_with_errors(
    stdout: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse Vibe --output json (one list) or --output streaming (JSONL).

    The Vibe CLI emits LLMMessage dictionaries (through 2.22) or public
    history entries (2.23 and later), not a provider-enforced answer schema. The harness therefore treats the final assistant message content as
    the answer/verdict, while preserving all parsed messages as trace JSONL."""
    text = coerce_text(stdout).strip()
    if not text:
        return [], ["Vibe stream is empty"]
    try:
        parsed = stream_json_loads(text)
    except json.JSONDecodeError:
        return parse_trace_jsonl_text(text, strict=False)
    if isinstance(parsed, list):
        errors = [f"Vibe message {index} is not an object"
                  for index, item in enumerate(parsed, 1)
                  if not isinstance(item, dict)]
        return [item for item in parsed if isinstance(item, dict)], errors
    if isinstance(parsed, dict):
        if isinstance(parsed.get("messages"), list):
            values = parsed["messages"]
            errors = [f"Vibe message {index} is not an object"
                      for index, item in enumerate(values, 1)
                      if not isinstance(item, dict)]
            return [item for item in values if isinstance(item, dict)], errors
        return [parsed], []
    return [], ["Vibe output must be a message object, message list, or JSONL"]


def parse_vibe_messages(stdout: str) -> list[dict[str, Any]]:
    """Compatibility projection; execution boundaries consume parser errors too."""
    messages, _ = parse_vibe_messages_with_errors(stdout)
    return messages


def _vibe_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def vibe_final_answer(messages: list[dict[str, Any]]) -> str:
    """Return only a validated assistant message; trace bytes are not answers."""
    for msg in reversed(messages):
        if str(msg.get("role", "")).casefold() != "assistant":
            continue
        text = _vibe_content_text(msg.get("content")).strip()
        if text:
            return text
    return ""


def vibe_usage_and_cost(messages: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None]:
    terminal = next((message for message in reversed(messages)
                     if message.get("role") == "assistant"
                     and isinstance(message.get("content"), str)
                     and message["content"].strip()), None)
    if terminal is None:
        return None, None
    candidate = terminal.get("usage") or terminal.get("tokens")
    usage = candidate if (isinstance(candidate, dict)
                          and normalize_usage(candidate).get("source") != "missing") else None
    cost: float | None = None
    for key in ("cost_usd", "total_cost_usd", "total_cost", "cost"):
        value = terminal.get(key)
        normalized = normalize_cost(value)
        if normalized.get("source") != "missing":
            cost = float(normalized["total_cost"])
            break
    return usage, cost


def vibe_trace_text(messages: list[dict[str, Any]], stdout: str) -> str:
    return coerce_text(stdout)


def vibe_skill_tool_evidence(stdout: str, skill_names: list[str]) -> list[str]:
    """Detect only completed, schema-valid Vibe `skill` tool lifecycles."""
    records, errors = parse_trace_jsonl_text(stdout, strict=False)
    if errors:
        return []
    events, metrics = normalize_trace_records(records, source="vibe")
    if metrics.get("trace_protocol_errors"):
        return []
    names = set(skill_names)
    evidence = []
    for event in events["events"]:
        invoked = str(event.get("input_summary") or "")
        if (event.get("type") == "skill_load" and event_is_completed(event)
                and event.get("is_error") is not True and invoked in names):
            evidence.append(f"Vibe skill tool invoked: {invoked}")
    return evidence[:5]


CLAUDE_SESSION_CONTENT_RECORD_TYPES = frozenset({"assistant", "user", "result", "stream_event"})


def claude_terminal_result_index(records: Sequence[Mapping[str, Any]]) -> int | None:
    """The one owner of Claude's terminal-event rule, shared by the answer
    parser, the trace dialect and the trigger adapter: exactly one `result`
    record, and no session content after it. Anything else (no result, two
    results, a turn after the result) is None: the stream has no final word."""
    results = [i for i, record in enumerate(records) if record.get("type") == "result"]
    if len(results) != 1:
        return None
    for record in records[results[0] + 1:]:
        if (record.get("type") in CLAUDE_SESSION_CONTENT_RECORD_TYPES
                or isinstance(record.get("message"), Mapping)):
            return None
    return results[0]
