from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypedDict, TypeVar, overload

import regex

from json_contracts import (
    freeze_json_mapping,
    freeze_json_value,
    strict_json_equal,
    thaw_json_value,
    validate_json_text,
)
from manifest_contracts import RunCoordinate

FAULT_MATCH_TIMEOUT_SECONDS = 0.25
MEDIATED_KEYS = frozenset({"mediated_tools", "mediated_tool_measurements", "tool_observation_expected"})
TOOL_METRIC_KEYS = ("steps", "tool_calls", "commands", "file_reads", "file_writes", "errors",
                   "retries", "repeated_command_max", "skill_invoked")
T = TypeVar("T")


def _digest(domain: str, value: Any) -> str:
    raw = json.dumps(freeze_json_value(value, domain), sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256((domain + "\n" + raw).encode("utf-8")).hexdigest()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("tool observation requires a SHA-256 identity")
    return value


def _positive(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("tool observation coordinates must be positive integers")
    return value


def canonical_input(value: Any) -> str:
    return json.dumps(freeze_json_value(value, "tool input"), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class FaultDefinition:
    tool: str
    output: Any
    declared_error: bool = False
    match: str | None = None
    times: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool, str) or not self.tool.strip():
            raise ValueError("tool_faults tool must be non-empty text")
        validate_json_text(self.tool, "tool_faults tool")
        if type(self.declared_error) is not bool:
            raise ValueError("tool_faults is_error must be boolean")
        if self.times is not None and (type(self.times) is not int or self.times < 1):
            raise ValueError("tool_faults times must be a positive nonboolean integer")
        if self.match is not None:
            if not isinstance(self.match, str) or not self.match:
                raise ValueError("tool_faults match must be non-empty text")
            validate_json_text(self.match, "tool_faults match")
            try:
                regex.compile(self.match, regex.VERSION0)
            except regex.error as exc:
                raise ValueError(f"tool_faults invalid match regex: {exc}") from exc
        object.__setattr__(self, "output", freeze_json_value(self.output, "tool_faults output"))

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "output": thaw_json_value(self.output),
                "is_error": self.declared_error,
                **({"match": self.match} if self.match is not None else {}),
                **({"times": self.times} if self.times is not None else {})}

    def matches(self, tool: str, payload: Any) -> bool:
        if tool != self.tool:
            return False
        text = canonical_input(payload)
        return self.match is None or regex.search(
            self.match, text, regex.VERSION0, timeout=FAULT_MATCH_TIMEOUT_SECONDS) is not None


def parse_faults(raw: Any) -> tuple[FaultDefinition, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("tool_faults must be a non-empty ordered list")
    result = []
    for item in raw:
        if not isinstance(item, Mapping) or not {"tool", "output"} <= set(item) or set(item) - {
                "tool", "output", "is_error", "match", "times"}:
            raise ValueError("tool_faults declaration requires tool/output and only match/is_error/times")
        if "times" in item and item["times"] is None or "match" in item and item["match"] is None:
            raise ValueError("tool_faults optional fields must be omitted rather than null")
        result.append(FaultDefinition(item["tool"], item["output"], item.get("is_error", False),
                                      item.get("match"), item.get("times")))
    return tuple(result)


def fault_rows(faults: tuple[FaultDefinition, ...]) -> list[dict[str, Any]]:
    return [fault.as_dict() for fault in faults]


def fault_identity(case_input_sha256: str, ordinal: int) -> str:
    return _digest("tool-fault-declaration-v1", {"case_input_sha256": _sha(case_input_sha256),
                                               "ordinal": _positive(ordinal)})


@dataclass(frozen=True)
class Observed(Generic[T]):
    value: T


@dataclass(frozen=True)
class Unavailable:
    reason: str


@dataclass(frozen=True)
class Count:
    metric: str
    tool: str | None = None

    def __post_init__(self) -> None:
        if self.metric not in TOOL_METRIC_KEYS:
            raise ValueError("unsupported mediated tool count")
        if self.tool is not None and (not isinstance(self.tool, str) or not self.tool.strip()):
            raise ValueError("tool count selector must be non-empty text")


@dataclass(frozen=True)
class ToolClaim:
    assertion: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assertion", freeze_json_mapping(self.assertion, "tool claim"))


@dataclass(frozen=True)
class SuccessfulOperations:
    pass


@dataclass(frozen=True)
class ExactOutput:
    exchange_number: int


@dataclass(frozen=True)
class _Resolution:
    source: Literal["fault", "recorded", "live"]
    fault_id: str | None
    output: Any
    resolution_error: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.source not in {"fault", "recorded", "live"}:
            raise ValueError("invalid tool resolution source")
        if self.source == "fault":
            _sha(self.fault_id)
            if set(self.resolution_error) != {"availability", "value"} or self.resolution_error.get("availability") != "known" or type(self.resolution_error.get("value")) is not bool:
                raise ValueError("fault resolution requires a declared error")
        elif self.fault_id is not None or dict(self.resolution_error) != {
                "availability": "unavailable", "reason": "legacy_error_unavailable"}:
            raise ValueError("ordinary resolution has no declared error")
        object.__setattr__(self, "output", freeze_json_value(self.output, "resolution output"))
        object.__setattr__(self, "resolution_error", freeze_json_mapping(self.resolution_error, "resolution error"))

    def as_dict(self) -> dict[str, Any]:
        body = {"source": self.source, "fault_id": self.fault_id,
                "output": thaw_json_value(self.output), "resolution_error": dict(self.resolution_error)}
        return {**body, "resolution_sha256": _digest("tool-resolution-v1", body)}


def callable_preparation(identity: Mapping[str, Any], turn: int, exchange: int,
                         tool: str, payload: Any, resolution: _Resolution) -> dict[str, Any]:
    if not isinstance(tool, str) or not tool.strip():
        raise ValueError("tool must be non-empty text")
    return {"type": "prepared", "identity": dict(identity), "turn": _positive(turn),
            "exchange": _positive(exchange), "tool": tool,
            "input": thaw_json_value(freeze_json_value(payload, "tool input")),
            "input_sha256": _digest("tool-input-v1", payload),
            "resolution": resolution.as_dict(),
            "payload": {"kind": "callable_output", "output": thaw_json_value(resolution.output),
                        "delivered_error": {"availability": "unavailable", "reason": "output_only_api"},
                        "payload_sha256": _digest("callable-output-v1", resolution.output)}}


class BridgeResponseBody(TypedDict):
    type: Literal["result"]
    protocol_version: Literal[1]
    id: int
    task_sha256: str
    run: Any
    turn_number: int
    exchange_number: int
    tool: str
    input_sha256: str
    source: Literal["fault", "recorded", "live"]
    fault_id: str | None
    output: Any
    error: dict[str, Any]


def bridge_response(identity: Mapping[str, Any], turn: int, exchange: int,
                    tool: str, payload: Any, resolution: _Resolution, request_id: int) -> BridgeResponseBody:
    return {"type": "result", "protocol_version": 1, "id": _positive(request_id),
            "task_sha256": identity["task_sha256"], "run": thaw_json_value(identity["run"]),
            "turn_number": turn, "exchange_number": exchange, "tool": tool,
            "input_sha256": _digest("tool-input-v1", payload), "source": resolution.source,
            "fault_id": resolution.fault_id, "output": thaw_json_value(resolution.output),
            "error": dict(resolution.resolution_error)}


def bridge_offer(identity: Mapping[str, Any], turn: int, exchange: int,
                 tool: str, payload: Any, resolution: _Resolution, request_id: int) -> dict[str, Any]:
    response = bridge_response(identity, turn, exchange, tool, payload, resolution, request_id)
    prepared = callable_preparation(identity, turn, exchange, tool, payload, resolution)
    digest = _digest("tool-bridge-response-v1", response)
    return {**prepared, "type": "offered", "request_id": request_id,
            "payload": {"kind": "bridge_response", "output": response["output"],
                        "delivered_error": response["error"], "payload_sha256": digest,
                        "response": {**response, "payload_sha256": digest}}}


def journal_record(rows: Sequence[Mapping[str, Any]], transition: Mapping[str, Any]) -> dict[str, Any]:
    body = {"version": 1, "sequence": len(rows) + 1,
            "previous_sha256": rows[-1]["record_sha256"] if rows else None,
            **dict(transition)}
    return {**body, "record_sha256": _digest("tool-journal-v1", body)}


@dataclass(frozen=True)
class _ExchangeState:
    prepared: Mapping[str, Any]
    state: Literal["prepared", "confirmed", "aborted", "offered", "acknowledged"]


@dataclass(frozen=True)
class ToolObservation:
    _identity: Mapping[str, Any]
    _rows: tuple[Mapping[str, Any], ...]
    turn: int | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        identity = freeze_json_mapping(self._identity, "tool run identity")
        bridge = identity.get("route") == "offline_bridge"
        if set(identity) != {"task_sha256", "case_input_sha256", "run", "faults", "expected_turns"} | ({"route"} if bridge else set()):
            raise ValueError("invalid tool run identity")
        _sha(identity["task_sha256"])
        _sha(identity["case_input_sha256"])
        run = identity["run"]
        if not isinstance(run, Mapping) or set(run) != {"case_id", "model", "variant", "run_number", "population"}:
            raise ValueError("invalid tool run coordinate")
        if run["population"] != "answer" or not all(isinstance(run[k], str) and run[k] for k in ("case_id", "variant")) or run["model"] is not None and not isinstance(run["model"], str):
            raise ValueError("invalid tool run coordinate values")
        RunCoordinate.of(run["case_id"], run["variant"], run["run_number"], run["model"])
        _positive(identity["expected_turns"])
        if identity["faults"] or not bridge:
            parse_faults(identity["faults"])
        elif not isinstance(identity["faults"], (list, tuple)):
            raise ValueError("bridge faults must be an ordered list")
        if self.turn is not None:
            _positive(self.turn)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_rows", tuple(freeze_json_mapping(row, "tool journal row") for row in self._rows))
        self._fold()

    def _fold(self) -> tuple[dict[int, _ExchangeState], set[int], bool, str | None]:
        exchanges: dict[int, _ExchangeState] = {}
        completed: set[int] = set()
        active: int | None = None
        closed = False
        gap: str | None = None
        for index, row in enumerate(self._rows):
            body = {key: value for key, value in row.items() if key != "record_sha256"}
            if type(row.get("version")) is not int or row["version"] != 1 or type(row.get("sequence")) is not int or row["sequence"] != index + 1 or row.get("previous_sha256") != (self._rows[index - 1]["record_sha256"] if index else None) or row.get("record_sha256") != _digest("tool-journal-v1", body):
                raise ValueError("invalid tool journal chain")
            if closed:
                raise ValueError("tool journal transition after close")
            common = {"version", "sequence", "previous_sha256", "record_sha256", "type"}
            kind = row.get("type")
            if index == 0:
                if kind != "opened" or set(row) != common | {"identity"} or not strict_json_equal(row["identity"], self._identity):
                    raise ValueError("tool journal opening identity mismatch")
                continue
            if kind == "turn_opened":
                if set(row) != common | {"turn"}:
                    raise ValueError("invalid tool turn opening")
                turn = _positive(row["turn"])
                if active is not None or turn in completed or turn != len(completed) + 1 or turn > self._identity["expected_turns"]:
                    raise ValueError("invalid tool turn order")
                active = turn
            elif kind in {"prepared", "offered"}:
                if (kind == "offered") != (self._identity.get("route") == "offline_bridge"):
                    raise ValueError("tool payload route mismatch")
                if active != row.get("turn") or type(row.get("turn")) is not int or set(row) != common | {"identity", "turn", "exchange", "tool", "input", "input_sha256", "resolution", "payload"} | ({"request_id"} if kind == "offered" else set()) or not strict_json_equal(row["identity"], self._identity):
                    raise ValueError("tool preparation identity mismatch")
                if any(exchange.state in {"prepared", "offered"} for exchange in exchanges.values()):
                    raise ValueError("callable preparation before prior output confirmation")
                number = _positive(row["exchange"])
                if number != len(exchanges) + 1:
                    raise ValueError("tool exchange order mismatch")
                raw = row["resolution"]
                if not isinstance(raw, Mapping) or set(raw) != {"source", "fault_id", "output", "resolution_error", "resolution_sha256"}:
                    raise ValueError("invalid tool resolution")
                resolution = _Resolution(raw["source"], raw["fault_id"], raw["output"], raw["resolution_error"])
                definitions = parse_faults(self._identity["faults"]) if self._identity["faults"] else ()
                used = Counter(e.prepared["resolution"]["fault_id"] for e in exchanges.values())
                eligible = [(fault_identity(self._identity["case_input_sha256"], n), f) for n, f in enumerate(definitions, 1)
                            if (f.times is None or used[fault_identity(self._identity["case_input_sha256"], n)] < f.times)
                            and f.matches(row["tool"], row["input"])]
                if bool(eligible) != (resolution.source == "fault") or eligible and eligible[0][0] != resolution.fault_id:
                    raise ValueError("fault resolution violates ordered budget")
                if resolution.source == "fault":
                    matched = [f for n, f in enumerate(definitions, 1) if fault_identity(self._identity["case_input_sha256"], n) == resolution.fault_id]
                    if not matched or matched[0].tool != row["tool"] or not strict_json_equal(matched[0].output, resolution.output) or matched[0].declared_error != resolution.resolution_error["value"]:
                        raise ValueError("fault resolution disagrees with declaration")
                expected = callable_preparation(self._identity, _positive(row["turn"]), number, row["tool"], row["input"], resolution)
                if kind == "offered":
                    expected = bridge_offer(self._identity, row["turn"], number, row["tool"], row["input"], resolution, row["request_id"])
                    if resolution.source == "live" or any(e.prepared.get("request_id") == row["request_id"] and e.prepared["turn"] == active for e in exchanges.values()):
                        raise ValueError("invalid offline bridge resolution or repeated request")
                if not strict_json_equal({k: row[k] for k in expected}, expected):
                    raise ValueError("tool resolution or delivered payload mismatch")
                exchanges[number] = _ExchangeState(row, "offered" if kind == "offered" else "prepared")
            elif kind == "acknowledged":
                if set(row) != common | {"turn", "exchange", "offered_sha256", "ack"}:
                    raise ValueError("invalid bridge acknowledgment")
                number = _positive(row["exchange"])
                previous = exchanges.get(number)
                if previous is None or previous.state != "offered" or type(row["turn"]) is not int or row["turn"] != active or previous.prepared["turn"] != active or row["offered_sha256"] != previous.prepared["record_sha256"]:
                    raise ValueError("invalid bridge receipt conservation")
                response = previous.prepared["payload"]["response"]
                expected_ack = {"type": "ack", "protocol_version": 1, "id": response["id"],
                                "turn_number": active, "exchange_number": number,
                                "payload_sha256": response["payload_sha256"]}
                if not strict_json_equal(row["ack"], expected_ack):
                    raise ValueError("bridge ACK identity mismatch")
                exchanges[number] = _ExchangeState(previous.prepared, "acknowledged")
            elif kind in {"confirmed", "aborted"}:
                if set(row) != common | {"turn", "exchange", "prepared_sha256"}:
                    raise ValueError("invalid callable receipt transition")
                number = _positive(row["exchange"])
                previous = exchanges.get(number)
                if previous is None or previous.state != "prepared" or row["turn"] != active or type(row["turn"]) is not int or previous.prepared["turn"] != active or row["prepared_sha256"] != previous.prepared["record_sha256"]:
                    raise ValueError("invalid callable receipt conservation")
                exchanges[number] = _ExchangeState(previous.prepared, kind)
                if kind == "aborted":
                    gap = "aborted_callable_output"
            elif kind == "gap":
                if set(row) != common | {"turn", "reason", "tool", "input_sha256"} or row["turn"] != active or type(row["turn"]) is not int or not isinstance(row["reason"], str) or not row["reason"]:
                    raise ValueError("invalid tool evidence gap")
                _sha(row["input_sha256"])
                gap = row["reason"]
            elif kind == "turn_closed":
                if set(row) != common | {"turn"} or row["turn"] != active or type(row["turn"]) is not int or any(e.state in {"prepared", "offered"} for e in exchanges.values() if e.prepared["turn"] == active):
                    raise ValueError("invalid tool turn closure")
                completed.add(_positive(row["turn"]))
                active = None
            elif kind == "closed":
                if set(row) != common or active is not None or len(completed) != self._identity["expected_turns"] or gap is not None:
                    raise ValueError("invalid tool run closure")
                closed = True
            else:
                raise ValueError("unknown tool journal transition")
        return exchanges, completed, closed, gap or self.reason

    @property
    def coverage(self) -> Literal["missing", "invalid", "open", "closed"]:
        if not self._rows:
            return "invalid" if self.reason and self.reason.startswith("invalid_tool_observation") else "missing"
        _, completed, closed, gap = self._fold()
        return "closed" if gap is None and (closed if self.turn is None else self.turn in completed) else "open"

    def for_turn(self, turn: int) -> ToolObservation:
        return ToolObservation(self._identity, self._rows, turn, self.reason)

    def canonical_rows(self) -> list[dict[str, Any]]:
        exchanges, _, _, _ = self._fold()
        return [{"identity": thaw_json_value(self._identity), "turn": e.prepared["turn"],
                 "exchange": number, "tool": e.prepared["tool"],
                 "input": thaw_json_value(e.prepared["input"]),
                 "input_sha256": e.prepared["input_sha256"],
                 "source": e.prepared["resolution"]["source"],
                 "fault_id": e.prepared["resolution"]["fault_id"],
                 "payload": thaw_json_value(e.prepared["payload"])}
                for number, e in exchanges.items() if e.state in {"confirmed", "acknowledged"}
                and (self.turn is None or e.prepared["turn"] == self.turn)]

    def as_member(self, journal_path: str) -> dict[str, Any]:
        return {"version": 1, "identity": thaw_json_value(self._identity), "turn": self.turn,
                "journal_path": journal_path, "journal_rows": len(self._rows),
                "journal_sha256": self._rows[-1]["record_sha256"] if self._rows else None,
                "canonical_rows": self.canonical_rows()}

    @overload
    def assess(self, claim: Count) -> Observed[int | bool] | Unavailable: ...

    @overload
    def assess(self, claim: ToolClaim) -> Observed[tuple[bool, str, float | None]] | Unavailable: ...

    @overload
    def assess(self, claim: SuccessfulOperations) -> Observed[tuple[str, ...]] | Unavailable: ...

    @overload
    def assess(self, claim: ExactOutput) -> Observed[Any] | Unavailable: ...

    def assess(self, claim: Count | ToolClaim | SuccessfulOperations | ExactOutput) -> Observed[Any] | Unavailable:
        rows = self.canonical_rows()
        closed = self.coverage == "closed"
        reason = self.reason or "tool_population_not_closed"
        if isinstance(claim, ExactOutput):
            hit = next((row for row in rows if row["exchange"] == claim.exchange_number), None)
            return Observed(freeze_json_value(hit["payload"]["output"], "delivered output")) if hit else Unavailable(reason)
        if isinstance(claim, SuccessfulOperations):
            return Observed(()) if closed and not rows else Unavailable(
                "bridge_has_no_operation_semantics" if self._identity.get("route") == "offline_bridge"
                else "output_only_api_has_no_operation_success")
        if isinstance(claim, Count):
            selected = [row for row in rows if claim.tool is None or row["tool"].casefold() == claim.tool.casefold()]
            if not closed:
                return Unavailable(reason)
            if claim.metric in {"steps", "tool_calls"}:
                return Observed(len(selected))
            if claim.metric == "errors" and selected:
                error = next((row["payload"]["delivered_error"] for row in selected
                              if row["payload"]["delivered_error"]["availability"] != "known"), None)
                return Unavailable(error["reason"]) if error else Observed(sum(
                    row["payload"]["delivered_error"]["value"] for row in selected))
            if not selected:
                return Observed(False if claim.metric == "skill_invoked" else 0)
            return Unavailable("output_only_api" if claim.metric == "errors" else "output_only_api_has_no_operation_semantics")
        assertion = claim.assertion
        atype = assertion.get("type")
        if atype not in {"tool_call", "tool_sequence", "tool_count_le"}:
            return Unavailable("output_only_api_has_no_operation_semantics")
        tool = assertion.get("tool")
        selected = [row for row in rows if not tool or row["tool"].casefold() == str(tool).casefold()]
        case_insensitive = bool(assertion.get("ci", True))
        def hit(pattern: str, text: str) -> bool:
            return regex.search(pattern, text, regex.VERSION0 | (regex.IGNORECASE if case_insensitive else 0),
                                timeout=FAULT_MATCH_TIMEOUT_SECONDS) is not None
        if "is_error" in assertion:
            if assertion.get("expected_no_call"):
                raise ValueError("expected_no_call cannot use is_error")
            if atype == "tool_call" and assertion.get("pattern") and not any(
                    assertion.get(key) for key in ("required_calls", "call_set", "order")):
                selected = [row for row in selected if hit(str(assertion["pattern"]),
                    row["input"] if isinstance(row["input"], str) else canonical_input(row["input"]))]
            error = next((row["payload"]["delivered_error"] for row in selected
                          if row["payload"]["delivered_error"]["availability"] != "known"), None)
            if error:
                return Unavailable(error["reason"])
            selected = [row for row in selected if row["payload"]["delivered_error"]["value"] == assertion["is_error"]]
        names = [row["tool"].casefold() for row in (selected if "is_error" in assertion else rows)]
        text = [row["input"] if isinstance(row["input"], str) else canonical_input(row["input"]) for row in selected]
        score: float | None = None
        sequence_detail: str | None = None
        positive_prefix = False
        if atype == "tool_count_le":
            passed = len(selected) <= int(assertion.get("max", assertion.get("value", 0)))
        elif atype == "tool_sequence":
            expected = [str(value).casefold() for value in assertion.get("expected", ())]
            actual_counts, expected_counts = Counter(names), Counter(expected)
            matches = sum((actual_counts & expected_counts).values())
            precision = matches / len(names) if names else 0.0
            recall = matches / len(expected) if expected else 0.0
            score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if closed:
                sequence_detail = f"; precision={precision:.4f}; recall={recall:.4f}; f1={score:.4f}"
            mode = assertion.get("mode", "strict")
            passed = (names == expected if mode == "strict" else actual_counts == expected_counts if mode == "unordered"
                      else not actual_counts - expected_counts if mode == "subset" else not expected_counts - actual_counts)
            passed = bool(passed) and (assertion.get("min_f1") is None or score >= float(assertion["min_f1"]))
            positive_prefix = passed and mode == "superset" and assertion.get("min_f1") is None
            if positive_prefix and not closed:
                score = None
        elif assertion.get("expected_no_call"):
            pattern = assertion.get("pattern")
            passed = not any(hit(str(pattern), row["tool"].casefold()) for row in rows) if pattern else not selected
        elif assertion.get("required_calls"):
            passed = all(str(value).casefold() in names for value in assertion["required_calls"])
            positive_prefix = passed
        elif assertion.get("call_set"):
            passed = Counter(names) == Counter(str(value).casefold() for value in assertion["call_set"])
        elif assertion.get("order"):
            cursor = 0
            passed = True
            for pattern in assertion["order"]:
                found = next((n for n in range(cursor, len(text)) if hit(str(pattern), text[n])), None)
                if found is None:
                    passed = False
                    break
                cursor = found + 1
            positive_prefix = passed
        else:
            pattern = assertion.get("pattern")
            count = sum(hit(str(pattern), value) for value in text) if pattern else len(selected)
            passed = count >= int(assertion.get("min_count", 1)) and (assertion.get("max_count") is None or count <= int(assertion["max_count"]))
            positive_prefix = passed and assertion.get("max_count") is None
        if not closed and not positive_prefix:
            return Unavailable(reason)
        detail = f"mediated {atype}; outputs={names}; coverage={self.coverage}"
        if sequence_detail is not None:
            detail += sequence_detail
        return Observed((passed, detail, score))


def reconstruct_tool_observation(rows: Sequence[Mapping[str, Any]], member: Any,
                                 expected_identity: Mapping[str, Any], *, reason: str | None = None) -> ToolObservation:
    if not isinstance(member, Mapping) or set(member) != {"version", "identity", "turn", "journal_path", "journal_rows", "journal_sha256", "canonical_rows"} or type(member["version"]) is not int or member["version"] != 1 or not strict_json_equal(member["identity"], expected_identity):
        raise ValueError("invalid mediated tool observation member")
    if type(member["journal_rows"]) is not int or member["journal_rows"] < 1 or member["journal_rows"] > len(rows):
        raise ValueError("missing mediated tool journal prefix")
    observation = ToolObservation(expected_identity, tuple(rows[:member["journal_rows"]]), member["turn"], reason)
    if member["journal_sha256"] != observation._rows[-1]["record_sha256"] or not strict_json_equal(member["canonical_rows"], observation.canonical_rows()):
        raise ValueError("mediated tool observation conservation mismatch")
    return observation
