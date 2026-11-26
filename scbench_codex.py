"""Read observed requests and operations from Codex rollout JSONL."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from json_contracts import stream_json_loads

_VERDICT = re.compile(
    r"(?im)^(?:- )?(?:\*\*)?(?:spec verdict: |specification )(pass|fail)"
    r"(?:ed)?(?:[.!](?:\*\*)?(?=\s|$)|\*\*(?=\s|$)|$)"
)


def _digest(value: Any) -> str:
    data = (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _usage(raw: dict) -> dict:
    result = {
        k: raw[k]
        for k in ("input_tokens", "output_tokens", "total_tokens")
        if type(raw.get(k)) is int
    }
    for group, fields in (
        (
            "input_token_details",
            {
                "cached_input_tokens": "cache_read",
                "cache_write_input_tokens": "cache_creation",
            },
        ),
        ("output_token_details", {"reasoning_output_tokens": "reasoning"}),
    ):
        if details := {
            name: raw[k] for k, name in fields.items() if type(raw.get(k)) is int
        }:
            result[group] = details
    return result


def explicit_skill_names(text: str) -> list[str]:
    names = []
    if invoked := re.match(r"^\s*\$([\w-]+)\b", text):
        names.append(invoked[1])
    for link in re.finditer(r"\[\$([\w.-]+)\]\(([^)\n]+)\)", text):
        path = link[2].strip("<>").replace("\\", "/")
        if path.endswith(f"/{link[1]}/SKILL.md") and link[1] not in names:
            names.append(link[1])
    return names


def _message(common: dict, payload: dict) -> dict:
    content = payload.get("content", [])
    blocks = [b for b in content if isinstance(b, dict)]
    text = "\n".join(b["text"] for b in blocks if isinstance(b.get("text"), str))
    complete = len(blocks) == len(content) and all(
        isinstance(b.get("text"), str) for b in blocks
    )
    node = {**common, "digest": _digest(content), "content_complete": complete}
    role = payload.get("role")
    if payload["type"] == "agent_message":
        to_lead = payload.get("recipient") == "/root"
        node.update(
            kind="agent_message",
            name="message_to_lead" if to_lead else "agent_message",
            author=payload.get("author"),
            role="agent",
        )
        if to_lead:
            node["text"] = text
            verdicts = {v.lower() for v in _VERDICT.findall(text)}
            if len(verdicts) == 1:
                node["spec_verdict"] = verdicts.pop()
        elif text.startswith("Message Type: NEW_TASK\n"):
            node.update(
                kind="prompt",
                role="agent_brief",
                name="delegation",
                evidence="agent_message",
            )
            if complete:
                node["prompt_text"] = text
        return node
    if role == "assistant":
        return {
            **node,
            "kind": "message",
            "name": "assistant_message",
            "role": role,
            "response_texts": [
                b["text"]
                for b in blocks
                if b.get("type") in {"output_text", "text"}
                and isinstance(b.get("text"), str)
            ],
        }
    node.update(kind="prompt", role=role, name="prompt", synthetic=role != "user")
    if role != "user":
        return node
    if matched := re.match(r"^<skill>\s*<name>([\w.:-]+)</name>", text):
        node.update(
            name="skill_injection",
            skill_name=matched[1],
            evidence="injected_skill",
            synthetic=True,
        )
    elif text.startswith(
        (
            "<environment_context>",
            "# AGENTS.md instructions",
            "<permissions instructions>",
        )
    ):
        node.update(name="environment", synthetic=True)
    else:
        if complete:
            node["prompt_text"] = text
        names = explicit_skill_names(text)
        if names:
            node.update(
                skill_name=names[0],
                skill_names=names,
                evidence="explicit_skill_request",
            )
    return node


def _read_session(
    path: str, rows: list, events: list, spans: list
) -> tuple[dict, str | None]:
    first, meta_row = next(
        ((n, row) for n, row in rows if row.get("type") == "session_meta"), (1, {})
    )
    meta = meta_row.get("payload", {})
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    spawn = spawn if isinstance(spawn, dict) else {}
    parent = meta.get("parent_thread_id") or spawn.get("parent_thread_id")
    thread = meta.get("id") or meta.get("session_id") or path
    session_id = f"session:{thread}"
    role = (
        meta.get("agent_role")
        or spawn.get("agent_role")
        or ("subagent" if parent else "lead" if meta else "unknown")
    )
    session = {
        "id": session_id,
        "kind": "agent",
        "name": role,
        "actor": role,
        "role": role,
        "source": f"{path}:{first}",
        "timestamp": meta_row.get("timestamp"),
    }
    spans.append(session)
    turns = {
        p.get("turn_id")
        for _, r in rows
        if isinstance(p := r.get("payload"), dict)
        and p.get("thread_id") == thread
        and p.get("turn_id")
    }
    foreign = False
    model = effort = None
    calls: dict[str, dict] = {}
    recorded: dict[str, dict] = {}
    fallback = []
    previous_count = None
    for number, row in rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        group, kind = row.get("type"), payload.get("type")
        if group == "session_meta":
            foreign = bool(meta.get("id") and payload.get("id") != thread)
            continue
        if (
            foreign
            and group == "event_msg"
            and kind == "task_started"
            and payload.get("turn_id") in turns
        ):
            foreign = False
        if foreign or (payload.get("thread_id") and payload["thread_id"] != thread):
            continue
        timestamp = row.get("timestamp")
        if timestamp:
            session["end_time"] = timestamp
            if not session["timestamp"]:
                session["timestamp"] = timestamp
        location = f"{path}:{number}"
        key = payload.get("id") or location
        common = {
            "id": f"{session_id}:{group}:{key}",
            "parent_id": session_id,
            "actor": role,
            "timestamp": timestamp,
            "source": location,
        }
        if group == "turn_context":
            model, effort = payload.get("model"), payload.get("effort")
        elif group == "token_usage_record" and isinstance(
            raw := payload.get("usage"), dict
        ):
            response = payload.get("response_id") or location
            recorded.setdefault(
                response,
                {
                    **common,
                    "id": f"{session_id}:response:{response}",
                    "kind": "llm",
                    "name": model or "model_request",
                    "model": model,
                    "effort": effort,
                    "provider": meta.get("model_provider"),
                    "usage": _usage(raw),
                    "evidence": "token_usage_record"
                    if payload.get("response_id")
                    else "unkeyed_token_usage_record",
                },
            )
        elif group == "event_msg" and kind == "token_count":
            info = payload.get("info")
            if isinstance(info, dict):
                marker = (info.get("total_token_usage"), info.get("last_token_usage"))
                if (
                    marker != previous_count
                    and isinstance(marker[1], dict)
                    and any(type(v) is int and v > 0 for v in marker[1].values())
                ):
                    fallback.append(
                        {
                            **common,
                            "kind": "llm",
                            "name": model or "model_request",
                            "model": model,
                            "effort": effort,
                            "provider": meta.get("model_provider"),
                            "usage": _usage(marker[1]),
                            "evidence": "last_token_usage",
                        }
                    )
                previous_count = marker
        elif group == "event_msg" and kind == "item_completed":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "CommandExecution":
                command = {
                    **common,
                    "id": f"{session_id}:command:{item.get('id') or location}",
                    "kind": "command",
                    "name": "command",
                    "timestamp": payload.get("started_at_ms") or timestamp,
                    "end_time": payload.get("completed_at_ms") or timestamp,
                }
                if type(exit_code := item.get("exit_code")) is int:
                    command.update(
                        exit_code=exit_code,
                        is_error=exit_code != 0,
                        status="failed" if exit_code else "completed",
                    )
                if "aggregated_output" in item:
                    command["output_digest"] = _digest(item["aggregated_output"])
                if "command" in item:
                    command["input_digest"] = _digest(item["command"])
                parsed = item.get("parsed_cmd")
                if (
                    command.get("exit_code") == 0
                    and isinstance(parsed, list)
                    and len(parsed) == 1
                ):
                    read = parsed[0]
                    if isinstance(read, dict) and read.get("type") == "read":
                        skill = re.search(
                            r"(?:^|/)skills/([\w.-]+)/SKILL\.md$",
                            str(read.get("path", "")),
                        )
                        if skill:
                            command.update(
                                skill_name=skill[1], evidence="skill_file_read"
                            )
                spans.append(command)
        elif group == "response_item":
            call_id = payload.get("call_id")
            if kind in ("function_call", "custom_tool_call"):
                name = payload.get("name") or "tool_call"
                arg = (
                    payload.get("input")
                    if kind == "custom_tool_call"
                    else payload.get("arguments")
                )
                if name == "exec" and "tools.apply_patch(" in str(arg):
                    name = "exec_with_apply_patch"
                tool = {
                    **common,
                    "id": f"{session_id}:tool:{call_id or key}",
                    "kind": "tool_call",
                    "name": name,
                    "result_observed": False,
                }
                if arg is not None:
                    tool["input_digest"] = _digest(arg)
                if call_id:
                    if call_id in calls:
                        continue
                    calls[call_id] = tool
                events.append(tool)
                if name == "spawn_agent" and isinstance(arg, str):
                    try:
                        request = json.loads(arg)
                    except ValueError:
                        request = None
                    if isinstance(request, dict) and isinstance(
                        brief := request.get("message"), str
                    ):
                        spans.append(
                            {
                                **common,
                                "id": f"{tool['id']}:brief",
                                "parent_id": tool["id"],
                                "kind": "prompt",
                                "name": "delegation",
                                "role": "agent_brief",
                                "evidence": "spawn_argument",
                                "digest": _digest(brief),
                                "prompt_text": brief,
                            }
                        )
            elif (
                kind in ("function_call_output", "custom_tool_call_output")
                and call_id in calls
            ):
                tool = calls[call_id]
                tool.update(
                    result_observed=True, end_time=timestamp, result_source=location
                )
                if "output" in payload:
                    tool["output_digest"] = _digest(payload["output"])
            elif kind in ("message", "agent_message"):
                node = _message(common, payload)
                if node["name"] == "message_to_lead":
                    events.append(node)
                else:
                    spans.append(node)
    spans.extend(recorded.values() if recorded else fallback)
    return session, parent


def inspect_rollouts(
    files: Iterable[tuple[str, Iterable[tuple[int, dict]]]],
) -> tuple[list[dict], list[dict]]:
    events: list[dict] = []
    spans: list[dict] = []
    sessions = []
    for path, rows in files:
        records = list(rows)
        if records:
            sessions.append(_read_session(path, records, events, spans))
    ids = {session["id"] for session, _ in sessions}
    for session, parent in sessions:
        if parent and f"session:{parent}" in ids:
            session["parent_id"] = f"session:{parent}"
        elif parent:
            session["evidence"] = "parent_rollout_missing"
    events.sort(key=lambda e: e["timestamp"] or "")
    return events, spans


def inspect_exec(
    rows: list[tuple[int, dict]],
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Pair Codex exec item lifecycles while retaining other normalized observations."""
    retained = {}
    replaced = set()
    agents = {}
    briefs = {}
    by_ref = {}
    thread = None
    turn = 0
    for line, row in rows:
        ref = f"trace.jsonl:{line}"
        typ = row.get("type")
        stamp = row.get("timestamp")
        if typ == "thread.started" and isinstance(uid := row.get("thread_id"), str):
            thread = _digest(uid)
            agents[thread] = {
                "id": f"exec-thread:{thread}",
                "kind": "agent",
                "name": "Codex thread",
                "actor": "lead",
                "timestamp": stamp,
                "source": ref,
                "sender_thread": thread,
            }
            replaced.add(ref)
            turn = 0
            continue
        if thread is not None:
            by_ref[ref] = agents[thread]["id"]
            if stamp:
                agents[thread]["end_time"] = stamp
        if typ == "turn.started":
            turn += 1
        item = row.get("item")
        if typ not in {
            "item.started",
            "item.updated",
            "item.completed",
        } or not isinstance(item, dict):
            continue
        kind = item.get("type")
        uid = item.get("id")
        if (
            kind not in {"command_execution", "collab_tool_call"}
            or not isinstance(uid, str)
            or not uid
        ):
            continue
        sender = (
            _digest(item["sender_thread_id"])
            if isinstance(item.get("sender_thread_id"), str)
            else thread
        )
        key = f"exec:{sender}:{turn}:{kind}:{uid}"
        collab = kind == "collab_tool_call"
        node = retained.setdefault(
            key,
            {
                "id": key,
                "kind": "tool_call" if collab else "command",
                "name": (item.get("tool") or "collaboration") if collab else "command",
                "source": ref,
                "timestamp": stamp,
                "result_observed": False,
            },
        )
        replaced.add(ref)
        if sender:
            node["sender_thread"] = sender
            if sender in agents:
                node["parent_id"] = agents[sender]["id"]
        if collab:
            if isinstance(targets := item.get("receiver_thread_ids"), list):
                node["receiver_threads"] = sorted(
                    set(node.get("receiver_threads", []))
                    | {
                        _digest(value)
                        for value in targets
                        if isinstance(value, str) and value
                    }
                )
            if isinstance(prompt := item.get("prompt"), str) and prompt:
                digest = _digest(prompt)
                brief_id = f"{key}:prompt:{digest}"
                briefs.setdefault(
                    brief_id,
                    {
                        "id": brief_id,
                        "parent_id": key,
                        "kind": "prompt",
                        "name": "Delegation brief",
                        "evidence": "collab_tool_call",
                        "role": "agent_brief",
                        "prompt_text": prompt,
                        "digest": digest,
                        "source": ref,
                        "timestamp": stamp,
                    },
                )
                node.update(input_digest=digest, input_source=ref)
        elif isinstance(command := item.get("command"), str):
            node.update(input_digest=_digest(command), input_source=ref)
        node["status"] = item.get("status") or (
            "completed" if typ == "item.completed" else "in_progress"
        )
        if typ == "item.completed":
            node.update(result_observed=True, result_source=ref, end_time=stamp)
            if isinstance(output := item.get("aggregated_output"), str):
                node["output_digest"] = _digest(output)
            elif collab and isinstance(states := item.get("agents_states"), dict):
                node["output_digest"] = _digest(states)
            if type(code := item.get("exit_code")) is int:
                node.update(exit_code=code, is_error=code != 0)
                if code != 0:
                    node["status"] = "failed"
    result = list(retained.values())
    for event in events:
        ref = event.get("source")
        if ref not in replaced:
            if ref in by_ref:
                event = {**event, "parent_id": by_ref[ref]}
            result.append(event)
    return result, [*agents.values(), *briefs.values()]


def _inference_pairs(trace: Path, rollout_id: str) -> dict:
    calls: dict = {}
    with trace.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = stream_json_loads(line)
            except ValueError:
                if not line.endswith("\n"):
                    break
                raise
            if not isinstance(row, dict) or row.get("rollout_id") != rollout_id:
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict) or payload.get("type") not in {
                "inference_started",
                "inference_completed",
                "inference_failed",
                "inference_cancelled",
                "inference_canceled",
            }:
                continue
            thread, uid = row.get("thread_id"), payload.get("inference_call_id")
            if (
                not isinstance(thread, str)
                or not thread
                or not isinstance(uid, str)
                or not uid
            ):
                continue
            if payload.get("thread_id", thread) != thread:
                continue
            pair = calls.setdefault((thread, uid), {})
            side = "start" if payload["type"] == "inference_started" else "end"
            if side in pair:
                raise ValueError("duplicate Codex inference endpoint")
            pair[side] = (payload, row.get("wall_time_unix_ms"), number)
    return calls


def _inference_body(
    span: dict,
    side: str,
    record: dict,
    directory: Path,
    root: Path,
    checked: Callable[[Path], Path],
    tool_producers: dict[str, set[str]],
    thread: str,
) -> None:
    partial = side == "response" and record.get("type") in (
        "inference_failed",
        "inference_cancelled",
        "inference_canceled",
    )
    ref = record.get("partial_response_payload" if partial else f"{side}_payload")
    if not isinstance(ref, dict) or ref.get("kind") != {"type": f"inference_{side}"}:
        return
    relative = ref.get("path")
    if not isinstance(relative, str) or not re.fullmatch(
        r"payloads/[0-9]+\.json", relative
    ):
        raise ValueError("invalid Codex inference payload path")
    path = checked(directory / relative)
    if not path.is_file():
        return
    data = path.read_bytes()
    try:
        body = stream_json_loads(data)
    except ValueError:
        return
    if not isinstance(body, dict):
        return
    if side == "request":
        if not isinstance(body.get("input"), list) or body.get("model") != record.get(
            "model"
        ):
            return
        fmt = "codex_inference_request"
    else:
        if (
            not isinstance(body.get("output_items"), list)
            or (not partial and not record.get("response_id"))
            or body.get("response_id") != record.get("response_id")
        ):
            return
        fmt = "codex_partial_output_items" if partial else "codex_observed_output_items"
        for item in body["output_items"]:
            if (
                isinstance(item, dict)
                and item.get("type") in {"function_call", "custom_tool_call"}
                and isinstance(call_id := item.get("call_id"), str)
            ):
                tool_producers.setdefault(
                    f"session:{thread}:tool:{call_id}", set()
                ).add(span["id"])
        if isinstance(body.get("token_usage"), dict) and "usage" not in span:
            span["usage"] = _usage(body["token_usage"])
    span.update(
        {
            f"{side}_body_available": True,
            f"{side}_body_source": path.relative_to(root).as_posix(),
            f"{side}_body_digest": hashlib.sha256(data).hexdigest(),
            f"{side}_body_format": fmt,
        }
    )


def attach_inference_payloads(
    spans: list[dict],
    capture_dir: Path,
    root: Path,
    *,
    input_path: Callable[[Path], Path] | None = None,
    events: list[dict] | None = None,
) -> None:
    """Bind recorder payloads by thread, inference call and provider response ID."""
    root = root.resolve()

    def checked(path: Path) -> Path:
        if not path.is_relative_to(root) or path.resolve() != path:
            raise ValueError(
                "Codex inference capture path escapes its archive or traverses a symlink"
            )
        return input_path(path) if input_path else path

    captures = sorted(checked(capture_dir).glob("trace-*/manifest.json"))
    if not captures:
        return
    nodes = {s["id"]: s for s in spans if "id" in s}
    recorder_sessions = set()
    known_sessions = {s["id"] for s in spans if s["kind"] == "agent"}
    owners = {}
    tool_producers: dict[str, set[str]] = {}
    for manifest in captures:
        raw_manifest = checked(manifest).read_bytes()
        try:
            meta = stream_json_loads(raw_manifest)
        except ValueError:
            continue
        if not isinstance(meta, dict) or meta.get("schema_version") != 2:
            continue
        root_thread, rollout_id = meta.get("root_thread_id"), meta.get("rollout_id")
        if not isinstance(root_thread, str) or not isinstance(rollout_id, str):
            continue
        if known_sessions and not (
            {f"session:{root_thread}", f"exec-thread:{_digest(root_thread)}"}
            & known_sessions
        ):
            continue
        trace = checked(manifest.parent / "trace.jsonl")
        if not trace.is_file():
            continue
        for (thread, call), pair in _inference_pairs(trace, rollout_id).items():
            start, start_ms, line = pair.get("start", ({}, None, None))
            end, end_ms, end_line = pair.get("end", ({}, None, None))
            session = f"session:{thread}"
            if (stdout_session := f"exec-thread:{_digest(thread)}") in nodes:
                session = stdout_session
            response = end.get("response_id")
            uid = (
                f"session:{thread}:response:{response}"
                if response
                else f"session:{thread}:inference:{call}"
            )
            recorder_sessions.add(session)
            if uid in owners and owners[uid] != (rollout_id, call):
                raise ValueError("ambiguous Codex inference response binding")
            owners[uid] = (rollout_id, call)
            ref = f"{trace.relative_to(root).as_posix()}:{line or end_line}"
            if session not in nodes:
                role = "lead" if thread == root_thread else "subagent"
                agent = {
                    "id": session,
                    "kind": "agent",
                    "name": role,
                    "role": role,
                    "source": ref,
                    "evidence": "inference_trace",
                    "timestamp": start_ms or end_ms,
                }
                nodes[session] = agent
                spans.append(agent)
            span = nodes.get(uid)
            if span is None:
                span = {
                    "id": uid,
                    "kind": "llm",
                    "name": start.get("model") or "model_request",
                    "model": start.get("model"),
                    "provider": start.get("provider_name"),
                    "parent_id": session,
                    "evidence": "inference_trace",
                    "source": ref,
                }
                nodes[uid] = span
                spans.append(span)
            span.update(request_body_available=False, response_body_available=False)
            if start_ms is not None:
                span["timestamp"] = start_ms
            if end_ms is not None:
                span["end_time"] = end_ms
            span["status"] = {
                "inference_completed": "completed",
                "inference_failed": "failed",
                "inference_cancelled": "cancelled",
                "inference_canceled": "cancelled",
            }.get(end.get("type"), "in_progress")
            for side, record in (("request", start), ("response", end)):
                _inference_body(
                    span,
                    side,
                    record,
                    manifest.parent,
                    root,
                    checked,
                    tool_producers,
                    thread,
                )
    for span in spans:
        if span.get("parent_id") in recorder_sessions and span.get("evidence") in (
            "last_token_usage",
            "unkeyed_token_usage_record",
        ):
            span.update(kind="metric", name="unattributed_token_usage")
    for event in events or ():
        producers = tool_producers.get(event.get("id", ""), set())
        if len(producers) == 1 and event.get("kind") == "tool_call":
            event["parent_id"] = next(iter(producers))


def verify_inference_bindings(spans: list[dict], capture_dir: Path, root: Path) -> None:
    """Re-read the recorder's inference records before exporting indexed file references."""
    captured: list[dict] = []
    attach_inference_payloads(captured, capture_dir, root)
    by_id = {item["id"]: item for item in captured if item["kind"] == "llm"}
    for span in spans:
        for side in ("request", "response"):
            key = f"{side}_body_source"
            if key not in span:
                continue
            bound = by_id.get(span.get("id"))
            if not bound or any(
                bound.get(f"{side}_body_{field}") != span.get(f"{side}_body_{field}")
                for field in ("source", "digest", "format")
            ):
                raise ValueError(
                    "Codex model body reference does not match the recorded inference"
                )
