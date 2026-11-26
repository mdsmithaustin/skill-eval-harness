"""Project and stream-json observations from Claude Code, correlated by provider IDs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

Rows = Iterable[tuple[int, dict]]


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _usage(raw: dict) -> dict:
    counts = {k: v for k, v in raw.items() if type(v) is int and v >= 0}
    result: dict = {}
    details = {}
    for field, key in (
        ("cache_read_input_tokens", "cache_read"),
        ("cache_creation_input_tokens", "cache_creation"),
    ):
        if field in counts:
            details[key] = counts[field]
    if details:
        result["input_token_details"] = details
    if "input_tokens" in counts:
        result["input_tokens"] = counts["input_tokens"] + sum(details.values())
    if "output_tokens" in counts:
        result["output_tokens"] = counts["output_tokens"]
        if "input_tokens" in result:
            result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    thinking = (raw.get("output_tokens_details") or {}).get("thinking_tokens")
    if type(thinking) is int and thinking >= 0:
        result["output_token_details"] = {"reasoning": thinking}
    return result


class _Inspector:
    def __init__(self):
        self.events: dict[str, dict] = {}
        self.spans: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.child_agents: dict[str, str] = {}
        self.request_ids: dict[str, str] = {}
        self.message_ids: dict[str, str] = {}
        self.message_requests: dict[str, set[str]] = {}
        self.request_refs: dict[str, str] = {}
        self.lead_sessions: set[str] = set()

    def read(self, rows: Rows, source: str, *, project: bool = False) -> None:
        requests = {}
        for number, row in rows:
            typ = row.get("type")
            if typ not in ("system", "user", "assistant", "api-request"):
                continue
            session = row.get("sessionId") or row.get("session_id")
            if typ == "system" and not session:
                continue
            agent = row.get("agentId")
            if not project and session:
                self.lead_sessions.add(session)
            actor = "subagent" if agent or row.get("parent_tool_use_id") else "lead"
            if project and not agent and session not in self.lead_sessions:
                actor = "unknown"
            group = "claude-agent:" + str(agent or session or source)
            parent = group
            if row.get("parent_tool_use_id"):
                parent = "claude-tool:" + row["parent_tool_use_id"]
            stamp = row.get("timestamp")
            ref = f"{source}:{number}"
            if group not in self.spans:
                self.spans[group] = {
                    "kind": "agent",
                    "name": "Claude session",
                    "id": group,
                    "actor": actor,
                    "ancestry": "recorded" if actor == "lead" else "unknown",
                    "source": ref,
                }
            group_span = self.spans[group]
            if stamp:
                group_span["timestamp"] = min(group_span.get("timestamp", stamp), stamp)
                group_span["end_time"] = max(group_span.get("end_time", stamp), stamp)
            if typ == "api-request" and isinstance(row.get("id"), str):
                requests[row["id"]] = (row, ref, group, actor)
            if typ not in ("assistant", "user") or not isinstance(
                msg := row.get("message"), dict
            ):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                continue
            req = row.get("requestId") or row.get("request_id")
            msg_id = msg.get("id")
            request_ref = row.get("requestRef")
            if (
                typ == "assistant"
                and (req or request_ref or msg_id)
                and msg.get("model") != "<synthetic>"
            ):
                mid = "claude-llm:" + str(req or request_ref or msg_id)
                if req:
                    self.request_ids[req] = mid
                if msg_id:
                    candidates = self.message_requests.setdefault(msg_id, set())
                    if req or request_ref:
                        candidates.add(mid)
                if request_ref:
                    self.request_refs[request_ref] = mid
                model = msg.get("model")
                span = self.spans.setdefault(
                    mid,
                    {
                        "id": mid,
                        "kind": "llm",
                        "name": model or "Claude request",
                        "timestamp": stamp,
                        "source": ref,
                        "parent_id": parent,
                        "provider": "anthropic",
                        "evidence": "assistant_response",
                    },
                )
                span["response_source"] = ref
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ):
                        texts = span.setdefault("response_texts", [])
                        if block["text"] not in texts:
                            texts.append(block["text"])
                skill = row.get("attributionSkill")
                if isinstance(skill, str) and re.fullmatch(r"[\w.:-]{1,160}", skill):
                    span["skill_name"] = skill
                    span["skill_evidence"] = "provider_attribution"
                if project:
                    span.update(parent_id=parent, actor=actor)
                if model:
                    span["model"] = model
                if stamp:
                    span["end_time"] = max(span.get("end_time", stamp), stamp)
                usage = _usage(msg.get("usage") or {})
                if usage and usage.get("output_tokens", 0) >= span.get("usage", {}).get(
                    "output_tokens", 0
                ):
                    span["usage"] = usage
                if isinstance(row.get("effort"), str):
                    span["effort"] = row["effort"]
                parent = mid
            for index, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if typ == "assistant" and block_type == "tool_use":
                    key = "claude-tool:" + str(block.get("id") or f"{ref}:{index}")
                    tool = self.events.setdefault(
                        key,
                        {
                            "id": key,
                            "kind": "tool_call",
                            "name": block.get("name", "tool"),
                            "actor": actor,
                            "source": ref,
                            "input_digest": _digest(block.get("input")),
                            "timestamp": stamp,
                            "status": "in_progress",
                            "parent_id": parent,
                        },
                    )
                    if project:
                        tool.update(parent_id=parent, actor=actor)
                    args = block.get("input")
                    if tool["name"] == "Skill" and isinstance(args, dict):
                        skill = args.get("skill")
                        if isinstance(skill, str) and re.fullmatch(
                            r"[\w.:-]{1,160}", skill
                        ):
                            tool.update(skill_name=skill, evidence="skill_requested")
                elif (
                    typ == "user"
                    and block_type == "tool_result"
                    and block.get("tool_use_id")
                ):
                    key = "claude-tool:" + block["tool_use_id"]
                    detail = (
                        row.get("tool_use_result") or row.get("toolUseResult") or {}
                    )
                    failed = (
                        block.get("is_error") is True
                        or isinstance(detail, dict)
                        and detail.get("success") is False
                    )
                    self.results[key] = {
                        "end_time": stamp,
                        "is_error": failed,
                        "status": "failed" if failed else "completed",
                        "output_digest": _digest(block.get("content")),
                        "result_source": ref,
                    }
                    if isinstance(detail, dict) and isinstance(
                        child := detail.get("agentId"), str
                    ):
                        self.child_agents["claude-agent:" + child] = key
                elif (
                    typ == "user"
                    and block_type == "text"
                    and isinstance(block.get("text"), str)
                ):
                    text = block["text"]
                    synthetic = (
                        row.get("isSynthetic") is True
                        or row.get("isMeta") is True
                        or text.lstrip().startswith(
                            (
                                "<system-reminder>",
                                "<task-notification>",
                                "<local-command-stdout>",
                                "<local-command-caveat>",
                            )
                        )
                    )
                    # Project transcripts also contain CLI-generated notices without a user origin.
                    observed_user = (
                        not project
                        or row.get("turnOrigin") in ("sdk", "human", "user")
                        or bool(agent)
                    )
                    role = (
                        "injected"
                        if synthetic
                        else "agent_brief"
                        if agent
                        else "user"
                        if observed_user
                        else "unknown"
                    )
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    key = f"claude-prompt:{row.get('uuid') or ref}:{index}:{digest}"
                    prompt = self.spans.setdefault(
                        key,
                        {
                            "id": key,
                            "kind": "prompt",
                            "name": "Claude prompt",
                            "role": role,
                            "synthetic": synthetic,
                            "digest": digest,
                            "timestamp": stamp,
                            "source": ref,
                            "parent_id": parent,
                        },
                    )
                    if synthetic and text.startswith("Base directory for this skill: "):
                        directory = (
                            text.splitlines()[0]
                            .removeprefix("Base directory for this skill: ")
                            .strip()
                        )
                        skill = re.split(r"[/\\]", directory.rstrip("/\\"))[-1]
                        if re.fullmatch(r"[\w.:-]{1,160}", skill):
                            prompt.update(
                                evidence="skill_directory_injection", skill_name=skill
                            )
                    elif role == "user" and (
                        command := re.match(
                            r"^<command-message>([\w.:-]{1,160})</command-message>\s*<command-name>/\1</command-name>",
                            text,
                        )
                    ):
                        prompt.update(
                            evidence="explicit_command", skill_name=command.group(1)
                        )
                    if not synthetic and role in ("agent_brief", "user"):
                        prompt["prompt_text"] = text
        for key, (row, ref, group, actor) in requests.items():
            mid = self.request_refs.get(key, "claude-llm:" + key)
            span = self.spans.setdefault(
                mid,
                {
                    "id": mid,
                    "kind": "llm",
                    "name": "Claude request",
                    "provider": "anthropic",
                },
            )
            span.update(
                timestamp=row.get("timestamp"),
                source=ref,
                parent_id=group,
                actor=actor,
                evidence="api_request",
            )
            if isinstance(row.get("params"), dict) and isinstance(
                model := row["params"].get("model"), str
            ):
                span.setdefault("model", model)
            if isinstance(row.get("requestDigest"), str):
                span["digest"] = row["requestDigest"]

    def finish(self) -> tuple[list[dict], list[dict]]:
        aliases = {}
        for message, candidates in self.message_requests.items():
            fallback = "claude-llm:" + message
            if not candidates:
                self.message_ids[message] = fallback
            elif len(candidates) == 1:
                chosen = next(iter(candidates))
                self.message_ids[message] = chosen
                if fallback != chosen and fallback in self.spans:
                    original = self.spans.pop(fallback)
                    richer = self.spans[chosen]
                    self.spans[chosen] = {**original, **richer}
                    if "response_texts" in original:
                        self.spans[chosen]["response_texts"] = list(
                            dict.fromkeys(
                                [
                                    *original["response_texts"],
                                    *richer.get("response_texts", []),
                                ]
                            )
                        )
                    aliases[fallback] = chosen
            elif fallback in self.spans and fallback not in candidates:
                self.spans[fallback].update(
                    kind="assistant_message",
                    name="Claude response",
                    evidence="ambiguous_request",
                )
        for key, result in self.results.items():
            if tool := self.events.get(key):
                tool.update(result)
                if tool.get("skill_name"):
                    tool["evidence"] = (
                        "skill_failed" if tool["is_error"] else "skill_completed"
                    )
        for group, parent in self.child_agents.items():
            if group in self.spans and parent in self.events:
                self.spans[group].update(parent_id=parent, ancestry="recorded")
        known = self.spans.keys() | self.events.keys()
        for item in [*self.spans.values(), *self.events.values()]:
            if item.get("parent_id") in aliases:
                item["parent_id"] = aliases[item["parent_id"]]
            if item.get("parent_id") is not None and item["parent_id"] not in known:
                item.pop("parent_id")
                item["ancestry"] = "unknown"
        return list(self.events.values()), list(self.spans.values())


def inspect_stdout(
    rows: Rows, *, source: str = "trace.jsonl", capture_root: Path | None = None
) -> tuple[list[dict], list[dict]]:
    inspector = _Inspector()
    inspector.read(rows, source)
    events, spans = inspector.finish()
    if capture_root is not None:

        def checked(path: Path) -> Path:
            if path.resolve() != path or not path.is_relative_to(capture_root):
                raise ValueError("Claude capture outside native run")
            return path

        _bind_capture_files(
            inspector, spans, capture_root, capture_root / "claude-api", checked
        )
    return events, spans


def inspect_archive(
    archive, checkpoint: Path, read_rows
) -> tuple[list[dict], list[dict]]:
    inspector = _Inspector()
    stdout = archive.input(checkpoint / "agent/stdout.jsonl")
    if stdout.is_file():
        inspector.read(
            read_rows(stdout, agent_stream=True), str(stdout.relative_to(archive.root))
        )
    workspace = checkpoint / "agent/workspace"
    for path in sorted(archive.input(workspace / "projects").rglob("*.jsonl")):
        path = archive.input(path)
        inspector.read(
            read_rows(path, agent_stream=True),
            str(path.relative_to(archive.root)),
            project=True,
        )
    prompt = archive.input(checkpoint / "prompt.txt")
    if prompt.is_file():
        payload = prompt.read_bytes()
        inspector.spans["checkpoint-prompt"] = {
            "kind": "prompt",
            "id": "checkpoint-prompt",
            "name": "Checkpoint task",
            "role": "user",
            "source": str(prompt.relative_to(archive.root)),
            "digest": hashlib.sha256(payload).hexdigest(),
            "evidence": "rendered_checkpoint_prompt",
            "prompt_text": payload.decode("utf-8"),
        }
    events, spans = inspector.finish()
    _bind_capture_files(
        inspector,
        spans,
        archive.root,
        archive.input(workspace / "raw-api"),
        archive.input,
    )
    return events, spans


def _bind_capture_files(
    inspector: _Inspector,
    spans: list[dict],
    root: Path,
    capture: Path,
    checked: Callable[[Path], Path],
) -> None:
    if not checked(capture).is_dir():
        return
    for span in spans:
        if span["kind"] == "llm":
            span.update(request_body_available=False, response_body_available=False)
    linked = set()
    for index in sorted(capture.glob("*/index.jsonl")):
        with checked(index).open("rb") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                req, msg = row.get("request_id"), row.get("message_id")
                if not isinstance(req, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", req):
                    continue
                if not isinstance(msg, str):
                    continue
                request = inspector.request_ids.get(req)
                message = inspector.message_ids.get(msg)
                if message and (
                    (request and request != message)
                    or (not request and message in inspector.request_ids.values())
                ):
                    continue
                mid = request or message or f"claude-llm:{req}"
                existing = inspector.spans.get(mid)
                span = (
                    existing
                    if existing is not None
                    else {
                        "id": mid,
                        "kind": "llm",
                        "name": "Claude request",
                        "provider": "anthropic",
                        "evidence": "captured_response",
                        "ancestry": "unknown",
                        "request_body_available": False,
                        "response_body_available": False,
                        "source": index.relative_to(root).as_posix(),
                    }
                )
                bodies = {}
                for kind in ("request", "response"):
                    recorded = row.get(f"{kind}_file")
                    if not isinstance(recorded, str):
                        continue
                    filename = Path(recorded)
                    if filename.name != recorded and not (
                        filename.is_absolute()
                        and ".." not in filename.parts
                        and filename.parts[-3:-1] == (capture.name, index.parent.name)
                    ):
                        continue
                    if not re.fullmatch(
                        rf"[a-zA-Z0-9_-]+\.{kind}\.json", filename.name
                    ):
                        continue
                    if kind == "response" and filename.name != f"{req}.response.json":
                        continue
                    path = checked(index.parent / filename.name)
                    payload = _captured_body(path, kind)
                    if payload is None:
                        continue
                    body, digest = payload
                    if kind == "response" and body.get("id") != msg:
                        continue
                    bodies[kind] = body
                    span.update(
                        {
                            f"{kind}_body_available": True,
                            f"{kind}_body_source": path.relative_to(root).as_posix(),
                            f"{kind}_body_digest": digest,
                        }
                    )
                if "request" in bodies and (
                    existing is not None or "response" in bodies
                ):
                    linked.add(root / span["request_body_source"])
                if existing is None and "response" in bodies:
                    response = bodies["response"]
                    span["model"] = response.get("model") or row.get("model")
                    stamp = str(row.get("timestamp", ""))
                    if re.fullmatch(r"\d{10}(?:\.\d+)?|\d{13}", stamp):
                        span["timestamp"] = stamp
                    else:
                        try:
                            datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                        except ValueError:
                            pass
                        else:
                            span["timestamp"] = stamp
                    if isinstance(usage := response.get("usage"), dict):
                        span["usage"] = _usage(usage)
                    group = "claude-agent:" + str(row.get("session_id"))
                    if group in inspector.spans:
                        span["parent_id"] = group
                        span["ancestry"] = "recorded_session"
                    inspector.spans[mid] = span
                    spans.append(span)
    for path in sorted(capture.glob("*/*.request.json")):
        if path in linked or not re.fullmatch(
            r"[A-Za-z0-9_-]+\.request\.json", path.name
        ):
            continue
        data = _captured_body(checked(path), "request")
        if data is None:
            continue
        body, digest = data
        source = path.relative_to(root).as_posix()
        spans.append(
            {
                "id": f"claude-attempt:{source}",
                "kind": "model_request_attempt",
                "name": "Claude request attempt",
                "provider": "anthropic",
                "model": body.get("model"),
                "source": source,
                "evidence": "unmatched_request_file",
                "ancestry": "unknown",
                "request_body_available": True,
                "response_body_available": False,
                "request_body_source": source,
                "request_body_digest": digest,
            }
        )


def _captured_body(path: Path, kind: str) -> tuple[dict, str] | None:
    if not path.is_file() or path.resolve() != path:
        return None
    payload = path.read_bytes()
    try:
        body = json.loads(payload)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(body, dict) or not isinstance(
        body.get("messages" if kind == "request" else "content"), list
    ):
        return None
    return body, hashlib.sha256(payload).hexdigest()
