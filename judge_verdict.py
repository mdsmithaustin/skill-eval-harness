"""Strict discriminated judge verdicts and stored-result boundary parsing."""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, TypeAlias

from invocation_contracts import InvocationState, validate_invocation_lifecycle
from json_contracts import (
    freeze_json_mapping,
    freeze_json_value,
    strict_json_equal,
    thaw_json_value,
    validate_json_text,
)
from spend_contracts import JudgeCall


class VerdictKind(str, Enum):
    BOOLEAN = "boolean"
    SCORED = "scored"
    DIMENSIONS = "dimensions"
    DYNAMIC = "dynamic"
    CONSENSUS = "consensus"


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _passed(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("passed must be a boolean")
    return value


@dataclass(frozen=True)
class BooleanVerdict:
    passed: bool
    kind: VerdictKind = field(default=VerdictKind.BOOLEAN, init=False)

    def __post_init__(self) -> None:
        _passed(self.passed)


@dataclass(frozen=True)
class ScoredVerdict:
    score: float
    threshold: float
    passed: bool
    kind: VerdictKind = field(default=VerdictKind.SCORED, init=False)

    def __post_init__(self) -> None:
        score, threshold = _number(self.score, "score"), _number(self.threshold, "threshold")
        if _passed(self.passed) != (score >= threshold):
            raise ValueError("passed contradicts score and threshold")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class DimensionVerdict:
    dimension_scores: tuple[tuple[str, float], ...]
    score: float
    threshold: float
    passed: bool
    kind: VerdictKind = field(default=VerdictKind.DIMENSIONS, init=False)

    def __post_init__(self) -> None:
        if not self.dimension_scores or len({name for name, _ in self.dimension_scores}) != len(self.dimension_scores):
            raise ValueError("dimension names must be non-empty and unique")
        values = []
        for name, value in self.dimension_scores:
            if not isinstance(name, str) or not name:
                raise ValueError("dimension name must be non-empty")
            number = _number(value, f"dimension {name}")
            if not 1 <= number <= 5:
                raise ValueError("dimension scores must be in [1,5]")
            values.append(number)
        derived = round(statistics.mean((value - 1) / 4 for value in values), 4)
        threshold = _number(self.threshold, "threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("dimension threshold must be normalized to [0,1]")
        if abs(_number(self.score, "score") - derived) > 1e-9:
            raise ValueError("dimension aggregate score contradicts dimension_scores")
        if _passed(self.passed) != (derived >= threshold):
            raise ValueError("dimension passed contradicts score and threshold")
        object.__setattr__(self, "dimension_scores", tuple((name, float(value)) for name, value in self.dimension_scores))
        object.__setattr__(self, "score", derived)
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class DynamicVerdict:
    criteria: tuple[tuple[str, bool], ...]
    minimum_criteria: int
    score: float
    passed: bool
    kind: VerdictKind = field(default=VerdictKind.DYNAMIC, init=False)

    def __post_init__(self) -> None:
        if (isinstance(self.minimum_criteria, bool) or not isinstance(self.minimum_criteria, int)
                or self.minimum_criteria < 1):
            raise ValueError("minimum_criteria must be a positive integer")
        if not self.criteria or len({name for name, _ in self.criteria}) != len(self.criteria):
            raise ValueError("dynamic criteria names must be non-empty and unique")
        if self.minimum_criteria > len(self.criteria):
            raise ValueError("minimum_criteria cannot exceed the number of criteria")
        for name, met in self.criteria:
            if not isinstance(name, str) or not name or not isinstance(met, bool):
                raise ValueError("dynamic criteria require non-empty names and boolean met")
        met_count = sum(1 for _, met in self.criteria if met)
        derived = round(met_count / len(self.criteria), 4)
        expected = len(self.criteria) >= self.minimum_criteria and met_count >= self.minimum_criteria
        if abs(_number(self.score, "score") - derived) > 1e-9 or _passed(self.passed) != expected:
            raise ValueError("dynamic score/passed contradict criteria")
        object.__setattr__(self, "score", derived)


@dataclass(frozen=True)
class ConsensusVerdict:
    passed: bool
    score: float | None = None
    kind: VerdictKind = field(default=VerdictKind.CONSENSUS, init=False)

    def __post_init__(self) -> None:
        _passed(self.passed)
        if self.score is not None:
            object.__setattr__(self, "score", _number(self.score, "score"))


@dataclass(frozen=True)
class Consensus:
    """Several verdicts on one judge task, folded by one rule.

    Repeated runs of one judge and a panel of judge models used to fold ties
    differently: repeats turned an even split into a silent fail, while the
    panel reported it as unresolved. Both now use ``resolve_consensus``, and
    both report ``agreement``, so a judge that disagrees with itself is
    visible instead of averaged away.
    """

    passed: bool
    unresolved: bool
    concur: int
    n: int
    median_score: float | None = None
    quorum: int | None = None

    def __post_init__(self) -> None:
        _passed(self.passed)
        if not isinstance(self.unresolved, bool):
            raise ValueError("unresolved must be a boolean")
        for label, value in (("concur", self.concur), ("n", self.n)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.n < 1 or self.concur > self.n:
            raise ValueError("consensus needs 1 <= n and concur <= n")
        if self.unresolved and self.passed:
            raise ValueError("an unresolved consensus cannot pass")
        if self.median_score is not None:
            object.__setattr__(self, "median_score", _number(self.median_score, "median_score"))

    def agreement(self) -> dict[str, Any]:
        return {"concur": self.concur, "n": self.n,
                "concur_fraction": round(self.concur / self.n, 4),
                "unanimous": self.concur in (0, self.n), "unresolved": self.unresolved,
                "quorum": self.quorum}

    def verdict(self) -> ConsensusVerdict:
        return ConsensusVerdict(self.passed, self.median_score)


def resolve_consensus(passed: list[bool], scores: list[float], *,
                      threshold: float | None = None,
                      quorum: int | None = None) -> Consensus:
    """Strict majority passes; an explicit quorum overrides the majority.

    An exact tie with no quorum is decided by the median score only against
    an explicit threshold; otherwise it is ``unresolved`` and does not pass.
    It is never a silent coin flip.
    """
    if not passed or not all(isinstance(item, bool) for item in passed):
        raise ValueError("consensus needs at least one boolean verdict")
    n, concur = len(passed), sum(passed)
    median = statistics.median(scores) if scores else None
    unresolved = False
    if isinstance(quorum, int) and not isinstance(quorum, bool) and quorum > 0:
        outcome = concur >= quorum
    elif concur * 2 > n:
        outcome = True
    elif concur * 2 < n:
        outcome = False
    elif (median is not None and isinstance(threshold, (int, float))
          and not isinstance(threshold, bool)):
        outcome = median >= threshold
    else:
        outcome, unresolved = False, True
    return Consensus(outcome, unresolved, concur, n, median,
                     quorum if isinstance(quorum, int) and not isinstance(quorum, bool) and quorum > 0 else None)


JudgeVerdict: TypeAlias = BooleanVerdict | ScoredVerdict | DimensionVerdict | DynamicVerdict | ConsensusVerdict


_SEMANTIC_FIELDS = frozenset({
    "passed", "score", "threshold", "dimension_scores", "criteria", "minimum_criteria",
})
_ALLOWED_FIELDS = {
    VerdictKind.BOOLEAN: frozenset({"passed"}),
    VerdictKind.SCORED: frozenset({"passed", "score", "threshold"}),
    VerdictKind.DIMENSIONS: frozenset({"passed", "score", "threshold", "dimension_scores"}),
    VerdictKind.DYNAMIC: frozenset({"passed", "score", "criteria", "minimum_criteria"}),
    VerdictKind.CONSENSUS: frozenset({"passed", "score"}),
}


def verdict_from_dict(raw: Mapping[str, Any], *, strict_stored: bool = True,
                      expected_dimensions: tuple[str, ...] | None = None) -> JudgeVerdict:
    if not isinstance(raw, Mapping):
        raise TypeError("judge verdict must be an object")
    explicit = raw.get("verdict_kind")
    try:
        kind = VerdictKind(explicit) if explicit is not None else None
    except ValueError as exc:
        raise ValueError(f"unknown verdict_kind {explicit!r}") from exc
    has_dims, has_dynamic = "dimension_scores" in raw, "criteria" in raw
    if has_dims and has_dynamic:
        raise ValueError("judge verdict cannot mix dimensions and dynamic criteria")
    if kind is None:
        if raw.get("judge_model") == "consensus" or "judge_panel" in raw:
            kind = VerdictKind.CONSENSUS
        elif has_dims:
            kind = VerdictKind.DIMENSIONS
        elif has_dynamic:
            kind = VerdictKind.DYNAMIC
        elif raw.get("score") is not None:
            kind = VerdictKind.SCORED
        elif "passed" in raw:
            kind = VerdictKind.BOOLEAN
        else:
            raise ValueError("judge verdict has no recognized payload")
    present_semantic = {
        key for key in (set(raw) & _SEMANTIC_FIELDS)
        if not (key == "score" and raw.get(key) is None)
        and not (key == "threshold" and kind is VerdictKind.BOOLEAN and raw.get("score") is None)
    }
    foreign_fields = sorted(present_semantic - _ALLOWED_FIELDS[kind])
    if foreign_fields:
        raise ValueError(f"{kind.value} verdict cannot carry fields: {', '.join(foreign_fields)}")
    if kind is VerdictKind.BOOLEAN:
        return BooleanVerdict(_passed(raw.get("passed")))
    if kind is VerdictKind.SCORED:
        if raw.get("threshold") is None:
            if strict_stored:
                raise ValueError("stored scored verdict requires threshold")
            threshold = 1.0
        else:
            threshold = _number(raw.get("threshold"), "threshold")
        score = _number(raw.get("score"), "score")
        passed = raw.get("passed", score >= threshold)
        return ScoredVerdict(score, threshold, _passed(passed))
    if kind is VerdictKind.DIMENSIONS:
        dims = raw.get("dimension_scores")
        if not isinstance(dims, Mapping):
            raise ValueError("dimension_scores must be an object")
        pairs = tuple((name, _number(value, f"dimension {name}")) for name, value in dims.items())
        if not all(isinstance(name, str) and name for name, _ in pairs):
            raise ValueError("dimension names must be non-empty strings")
        if expected_dimensions is not None and {name for name, _ in pairs} != set(expected_dimensions):
            raise ValueError("dimension_scores must exactly match the declared dimensions")
        normalized = round(statistics.mean((value - 1) / 4 for _, value in pairs), 4) if pairs else float("nan")
        threshold = raw.get("threshold")
        if threshold is None:
            raise ValueError("stored dimension verdict requires normalized threshold")
        passed = raw.get("passed", normalized >= _number(threshold, "threshold"))
        return DimensionVerdict(pairs, raw.get("score", normalized), threshold, _passed(passed))
    if kind is VerdictKind.DYNAMIC:
        criteria = raw.get("criteria")
        if not isinstance(criteria, list):
            raise ValueError("criteria must be a list")
        pairs = []
        for item in criteria:
            if not isinstance(item, Mapping):
                raise ValueError("criterion must be an object")
            pairs.append((item.get("name"), item.get("met")))
        minimum = raw.get("minimum_criteria")
        if minimum is None:
            raise ValueError("stored dynamic verdict requires minimum_criteria")
        met = sum(1 for _, value in pairs if value is True)
        score = round(met / len(pairs), 4) if pairs else float("nan")
        passed = raw.get("passed", len(pairs) >= minimum and met >= minimum)
        return DynamicVerdict(tuple(pairs), minimum, raw.get("score", score), _passed(passed))
    return ConsensusVerdict(_passed(raw.get("passed")), raw.get("score"))


def verdict_fields(verdict: JudgeVerdict) -> dict[str, Any]:
    out: dict[str, Any] = {"verdict_kind": verdict.kind.value, "passed": verdict.passed}
    if isinstance(verdict, ScoredVerdict):
        out.update(score=verdict.score, threshold=verdict.threshold)
    elif isinstance(verdict, DimensionVerdict):
        out.update(score=verdict.score, threshold=verdict.threshold,
                   dimension_scores={name: value for name, value in verdict.dimension_scores})
    elif isinstance(verdict, DynamicVerdict):
        out.update(score=verdict.score, minimum_criteria=verdict.minimum_criteria,
                   criteria=[{"name": name, "met": met} for name, met in verdict.criteria])
    elif isinstance(verdict, ConsensusVerdict) and verdict.score is not None:
        out["score"] = verdict.score
    return out


def validated_result_row(raw: Mapping[str, Any], *,
                         expected_dimensions: tuple[str, ...] | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("judge result row must be an object")
    verdict = verdict_from_dict(
        raw, strict_stored=True, expected_dimensions=expected_dimensions)
    nonsemantic = {key: value for key, value in raw.items()
                   if key not in _SEMANTIC_FIELDS and key != "verdict_kind"}
    return {**nonsemantic, **verdict_fields(verdict)}


@dataclass(frozen=True)
class _JudgeAttempt:
    kind: Literal["process", "not_started", "guard", "historical"]
    facts: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", freeze_json_mapping(self.facts, "judge execution"))
        code = self.facts.get("returncode")
        state = self.facts.get("invocation_state")
        if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
            raise ValueError("judge returncode must be an integer")
        if self.kind == "not_started":
            if (state != "not_started" or code is not None or self.facts.get("judge_served_model") is not None
                    or self.facts.get("timed_out") is not None or self.facts.get("provider_error") is not None):
                raise ValueError("not_started judge cannot carry process code or served model")
        elif self.kind == "guard":
            if (state is not None or self.facts.get("judge_served_model") is not None or code not in (None, 0)
                    or self.facts.get("timed_out") is not None or self.facts.get("provider_error") is not None):
                raise ValueError("guarded judge cannot carry process facts")
        elif self.kind == "process":
            validate_invocation_lifecycle(InvocationState(state), code, self.facts.get("provider_error"))
            if self.facts.get("timed_out", state == "timed_out") != (state == "timed_out"):
                raise ValueError("judge timed_out contradicts invocation state")


@dataclass(frozen=True)
class _JudgeLeaf:
    attempt: _JudgeAttempt
    call: JudgeCall | None = None


@dataclass(frozen=True)
class _AbsentSummary:
    pass


@dataclass(frozen=True)
class _NullSummary:
    pass


@dataclass(frozen=True)
class _IntegerSummary:
    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise ValueError("saved judge aggregate integer status must be an integer")


@dataclass(frozen=True)
class _MalformedScalarSummary:
    value: bool | float | str

    def __post_init__(self) -> None:
        if type(self.value) not in (bool, float, str) or (
                isinstance(self.value, float) and not math.isfinite(self.value)):
            raise ValueError("saved judge aggregate malformed scalar must be boolean, finite float or string")


@dataclass(frozen=True)
class _MalformedContainerSummary:
    json_type: Literal["array", "object"]

    def __post_init__(self) -> None:
        if self.json_type not in ("array", "object"):
            raise ValueError("saved judge aggregate malformed container must be array or object")


_SavedAggregateSummary: TypeAlias = (
    _AbsentSummary | _NullSummary | _IntegerSummary | _MalformedScalarSummary | _MalformedContainerSummary
)


@dataclass(frozen=True)
class _JudgeRepeats:
    members: tuple[_JudgeObservation, ...]
    expected_count: int | None = None
    policy: _JudgeConsensusPolicy | None = None
    saved_summary: _SavedAggregateSummary | None = None


@dataclass(frozen=True)
class _JudgePanel:
    members: tuple[_JudgeObservation, ...]
    requested_models: tuple[str, ...]
    policy: _JudgeConsensusPolicy | None = None
    models_recorded: bool = True
    saved_summary: _SavedAggregateSummary | None = None


@dataclass(frozen=True)
class _JudgeConsensusPolicy:
    threshold: float | None = None
    quorum: int | None = None

    def __post_init__(self) -> None:
        if self.threshold is not None:
            _number(self.threshold, "consensus threshold")
        if self.quorum is not None and (isinstance(self.quorum, bool)
                or not isinstance(self.quorum, int) or self.quorum < 1):
            raise ValueError("consensus quorum must be a positive integer")


_JudgePopulation: TypeAlias = _JudgeLeaf | _JudgeRepeats | _JudgePanel


@dataclass(frozen=True)
class _AbsentSuppliedDiagnostics:
    pass


@dataclass(frozen=True)
class _SuppliedDiagnostics:
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_json_value(self.value, "supplied judge diagnostics"))


_SuppliedDiagnosticEvidence: TypeAlias = _AbsentSuppliedDiagnostics | _SuppliedDiagnostics


@dataclass(frozen=True)
class _JudgeCause:
    path: tuple[int, ...]
    scope: Literal["observation", "aggregate"]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, tuple) or any(type(index) is not int or index < 1 for index in self.path):
            raise ValueError("judge diagnostic path requires positive integer ordinals")
        if self.scope not in ("observation", "aggregate"):
            raise ValueError("judge diagnostic scope must be observation or aggregate")
        validate_json_text(self.reason, "judge diagnostic reason")
        if not self.reason:
            raise ValueError("judge diagnostic reason must be nonempty")


def _check_local_reasons(reasons: tuple[str, ...]) -> None:
    if not isinstance(reasons, tuple) or not reasons:
        raise ValueError("unavailable judge leaf requires nonempty local reasons")
    for reason in reasons:
        validate_json_text(reason, "judge local reason")
        if not reason:
            raise ValueError("judge local reason must be nonempty")


@dataclass(frozen=True)
class _CompleteJudgeObservation:
    fields: Mapping[str, Any]
    population: _JudgePopulation
    verdict: JudgeVerdict
    explicit_kind: VerdictKind | None
    fresh: bool = True
    agreement: Consensus | Mapping[str, Any] | None = None
    supplied_diagnostics: _SuppliedDiagnosticEvidence = _AbsentSuppliedDiagnostics()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", freeze_json_mapping(self.fields, "judge metadata"))
        _check_observation_fields(self.fields)
        if self.agreement is not None and not isinstance(self.agreement, Consensus):
            object.__setattr__(self, "agreement", freeze_json_mapping(self.agreement, "judge agreement"))
        if isinstance(self.population, _JudgeLeaf):
            if self.fresh and self.population.call is None:
                raise ValueError("fresh judge leaf requires a requested call identity")
            reason = _judge_leaf_incomplete_reason(self.fields, self.population.attempt)
            if reason is not None:
                raise ValueError(reason)
            if self.population.attempt.kind == "guard" and (
                    self.fields.get("judge_guard_reason") != "empty_steps"
                    or not isinstance(self.verdict, BooleanVerdict) or self.verdict.passed):
                raise ValueError("only an empty-step local false verdict can be complete without a call")
        else:
            if self.fresh and self.population.saved_summary is not None:
                raise ValueError("fresh judge group cannot carry saved historical summary")
            reason = _aggregate_summary_reason(self.population.saved_summary)
            if reason is not None:
                raise ValueError(reason)
            if not all(isinstance(member, _CompleteJudgeObservation) for member in self.population.members):
                raise ValueError("complete judge group requires complete children")


@dataclass(frozen=True)
class _PartialJudgeObservation:
    fields: Mapping[str, Any]
    population: _JudgeLeaf
    verdict: JudgeVerdict
    explicit_kind: VerdictKind | None
    reasons: tuple[str, ...]
    fresh: bool = True
    supplied_diagnostics: _SuppliedDiagnosticEvidence = _AbsentSuppliedDiagnostics()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", freeze_json_mapping(self.fields, "judge metadata"))
        _check_observation_fields(self.fields)
        _check_local_reasons(self.reasons)


@dataclass(frozen=True)
class _MissingJudgeObservation:
    fields: Mapping[str, Any]
    population: _JudgePopulation
    local_reasons: tuple[str, ...]
    fresh: bool = True
    supplied_diagnostics: _SuppliedDiagnosticEvidence = _AbsentSuppliedDiagnostics()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", freeze_json_mapping(self.fields, "judge metadata"))
        _check_observation_fields(self.fields)
        if (self.fresh and isinstance(self.population, (_JudgeRepeats, _JudgePanel))
                and self.population.saved_summary is not None):
            raise ValueError("fresh judge group cannot carry saved historical summary")
        if isinstance(self.population, _JudgeLeaf):
            _check_local_reasons(self.local_reasons)
        elif not isinstance(self.local_reasons, tuple) or self.local_reasons:
            raise ValueError("missing judge group cannot carry leaf local reasons")

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(_render_judge_reason(cause.path, cause.reason) for cause in _judge_diagnostics(self))


_JudgeObservation: TypeAlias = _CompleteJudgeObservation | _PartialJudgeObservation | _MissingJudgeObservation
_ATTEMPT_FIELDS = frozenset({"returncode", "invocation_state", "timed_out", "provider_error", "judge_served_model"})
_RESULT_FIELDS = _SEMANTIC_FIELDS | frozenset({
    "verdict_kind", "judge_observation_kind", "judge_observation_complete", "availability",
    "judge_runs", "judge_panel", "agreement", "judge_expected_repeats", "judge_models",
    "judge_consensus_policy", "judge_execution_kind", "judge_aggregate_summary",
    "incomplete_judge_members", "judge_diagnostic_evidence",
}) | _ATTEMPT_FIELDS
_BINDING_FIELDS = (
    "judge_task_id", "id", "case_id", "variant", "run_number", "judge_backend",
    "judge_input_sha256", "judge_prompt_sha256", "judge_evidence_mode",
    "judge_context_sha256", "trajectory_steps_sha256",
)


def _check_observation_fields(fields: Mapping[str, Any]) -> None:
    reserved = set(fields) & _RESULT_FIELDS
    if reserved:
        raise ValueError(f"judge metadata cannot override reserved fields: {sorted(reserved)}")


def _aggregate_summary_reason(summary: _SavedAggregateSummary | None) -> str | None:
    prefix = "judge aggregate summary returncode is "
    if summary is None:
        return None
    if isinstance(summary, _IntegerSummary):
        return f"{prefix}nonzero ({summary.value})" if summary.value else None
    if isinstance(summary, _AbsentSummary):
        return prefix + "absent"
    if isinstance(summary, _NullSummary):
        return prefix + "null"
    if isinstance(summary, _MalformedScalarSummary):
        category = "boolean" if isinstance(summary.value, bool) else "number" if isinstance(summary.value, float) else "string"
    else:
        category = summary.json_type
    return f"{prefix}noninteger ({category})"


def _aggregate_summary_fields(summary: _SavedAggregateSummary | None) -> dict[str, Any]:
    if summary is None:
        return {"version": 1, "kind": "derived"}
    if isinstance(summary, _AbsentSummary):
        status = {"kind": "absent"}
    elif isinstance(summary, _NullSummary):
        status = {"kind": "null"}
    elif isinstance(summary, _IntegerSummary):
        status = {"kind": "integer", "value": summary.value}
    elif isinstance(summary, _MalformedScalarSummary):
        status = {"kind": "malformed-scalar", "value": summary.value}
    else:
        status = {"kind": "malformed-container", "json_type": summary.json_type}
    return {"version": 1, "kind": "saved", "status": status}


def _saved_aggregate_summary(raw: Mapping[str, Any], *, fresh: bool) -> _SavedAggregateSummary | None:
    canonical = "judge_aggregate_summary" in raw
    if fresh or canonical:
        code = raw.get("returncode")
        if type(code) is not int:
            raise ValueError("judge aggregate returncode must be an integer")
        if code != (0 if raw.get("judge_observation_complete") is True else 1):
            raise ValueError("judge aggregate returncode contradicts completeness")
    if not canonical:
        if fresh:
            return None
        if "returncode" not in raw:
            return _AbsentSummary()
        value = raw["returncode"]
        if value is None:
            return _NullSummary()
        if type(value) is int:
            return _IntegerSummary(value)
        if type(value) in (bool, float, str):
            return _MalformedScalarSummary(value)
        if isinstance(value, (list, Mapping)):
            return _MalformedContainerSummary("array" if isinstance(value, list) else "object")
        raise ValueError("judge aggregate returncode must be a JSON value")
    envelope = raw["judge_aggregate_summary"]
    if (not isinstance(envelope, Mapping) or type(envelope.get("version")) is not int
            or envelope["version"] != 1):
        raise ValueError("judge aggregate summary requires integer version 1")
    kind = envelope.get("kind")
    if kind not in ("derived", "saved") or set(envelope) != (
            {"version", "kind"} if kind == "derived" else {"version", "kind", "status"}):
        raise ValueError("judge aggregate summary has invalid kind or fields")
    complete = raw.get("judge_observation_complete") is True
    if (raw.get("judge_observation_complete") is not complete
            or raw.get("availability") != ("complete" if complete else "partial")):
        raise ValueError("canonical judge aggregate summary contradicts completeness flags")
    if not complete and ({key: raw[key] for key in _SEMANTIC_FIELDS | {"verdict_kind"} if key in raw}
                         != verdict_fields(ConsensusVerdict(False)) or "agreement" in raw):
        raise ValueError("canonical missing judge aggregate requires exact consensus false shell without agreement")
    if kind == "derived":
        return None
    if fresh:
        raise ValueError("fresh judge group cannot carry saved historical summary")
    status = envelope["status"]
    if not isinstance(status, Mapping):
        raise ValueError("judge aggregate saved summary status must be an object")
    status_kind = status.get("kind")
    summary: _SavedAggregateSummary
    if status_kind in ("absent", "null") and set(status) == {"kind"}:
        summary = _AbsentSummary() if status_kind == "absent" else _NullSummary()
    elif status_kind == "integer" and set(status) == {"kind", "value"}:
        summary = _IntegerSummary(status["value"])
    elif status_kind == "malformed-scalar" and set(status) == {"kind", "value"}:
        summary = _MalformedScalarSummary(status["value"])
    elif (status_kind == "malformed-container" and set(status) == {"kind", "json_type"}
          and status["json_type"] in ("array", "object")):
        summary = _MalformedContainerSummary(status["json_type"])
    else:
        raise ValueError("judge aggregate saved summary status has invalid kind or fields")
    if complete and _aggregate_summary_reason(summary) is not None:
        raise ValueError("complete judge aggregate cannot carry unavailable saved summary")
    return summary


def _judge_leaf_incomplete_reason(fields: Mapping[str, Any], attempt: _JudgeAttempt) -> str | None:
    if fields.get("judge_evidence_mode") not in {"text-only", "trajectory", "explore", "trajectory+explore"}:
        return "judge evidence mode is missing or invalid"
    if not isinstance(fields.get("judge_input_sha256"), str) or re.fullmatch(
            r"sha256:[0-9a-f]{64}", fields["judge_input_sha256"]) is None:
        return "judge input fingerprint is missing or invalid"
    if not isinstance(fields.get("judge_prompt_sha256"), str) or re.fullmatch(
            r"[0-9a-f]{64}", fields["judge_prompt_sha256"]) is None:
        return "judge prompt fingerprint is missing or invalid"
    if fields.get("judge_evidence_mode") in {"explore", "trajectory+explore"} and (
            not isinstance(fields.get("judge_context_sha256"), str) or re.fullmatch(
                r"sha256:[0-9a-f]{64}", fields["judge_context_sha256"]) is None):
        return "judge explore context fingerprint is missing"
    if attempt.facts.get("returncode") != 0 or attempt.kind == "not_started":
        return "judge call did not exit successfully"
    if attempt.kind == "process" and attempt.facts.get("invocation_state") != "complete":
        return "judge process is not complete"
    if fields.get("schema_errors") or fields.get("verdict_validation_error"):
        return "judge verdict failed validation"
    return None


def _judge_leaf_observation(verdict: JudgeVerdict | None, raw: Mapping[str, Any], *,
                            attempt_kind: Literal["process", "not_started", "guard", "historical"],
                            complete: bool, reasons: tuple[str, ...] = (),
                            fresh: bool = True, explicit_kind: VerdictKind | None = None) -> _JudgeObservation:
    fields = freeze_json_mapping({key: value for key, value in raw.items()
                                  if key not in _RESULT_FIELDS}, "judge metadata")
    attempt = _JudgeAttempt(attempt_kind, freeze_json_mapping(
        {key: value for key, value in raw.items() if key in _ATTEMPT_FIELDS}, "judge execution"))
    call = None
    if fresh:
        if not {"judge_task_id", "judge_input_sha256", "judge_backend", "judge_requested_model", "judge_repeat"} <= raw.keys():
            raise ValueError("fresh judge requires complete requested slot identity")
        call = JudgeCall(raw["judge_task_id"], raw["judge_input_sha256"],
                         raw["judge_backend"], raw["judge_requested_model"], raw["judge_repeat"])
        if raw.get("spend_call_id") != call.call_id:
            raise ValueError("judge spend call identity contradicts requested slot")
    population = _JudgeLeaf(attempt, call)
    if verdict is None:
        return _MissingJudgeObservation(fields, population, reasons or ("judge observation is missing",), fresh)
    if complete:
        return _CompleteJudgeObservation(fields, population, verdict,
                                         verdict.kind if fresh else explicit_kind, fresh)
    return _PartialJudgeObservation(fields, population, verdict,
                                    verdict.kind if fresh else explicit_kind,
                                    reasons or ("judge observation is not explicitly complete",), fresh)


def _judge_observation_reason(observation: _JudgeObservation) -> str | None:
    causes = _judge_diagnostics(observation)
    if not causes:
        return None
    reason = _render_judge_reason(causes[0].path, causes[0].reason)
    return reason if len(reason) <= 512 else reason[:509] + "..."


def _render_judge_reason(path: tuple[int, ...], reason: str) -> str:
    return ", ".join(f"member {index}" for index in path) + ": " + reason if path else reason


def _judge_diagnostic_records(causes: tuple[_JudgeCause, ...]) -> list[dict[str, Any]]:
    return [{"member": cause.path[0] if cause.path else "aggregate",
             "reason": _render_judge_reason(cause.path[1:], cause.reason)} for cause in causes]


def _judge_diagnostic_tree(observation: _JudgeObservation) -> dict[int, tuple[_JudgeCause, ...]]:
    subtrees: dict[int, tuple[_JudgeCause, ...]] = {}

    def visit(node: _JudgeObservation) -> tuple[_JudgeCause, ...]:
        population = node.population
        children = tuple(visit(member) for member in population.members) if not isinstance(population, _JudgeLeaf) else ()
        if isinstance(node, _CompleteJudgeObservation):
            causes = ()
        elif isinstance(population, _JudgeLeaf):
            local = node.local_reasons if isinstance(node, _MissingJudgeObservation) else node.reasons
            causes = tuple(_JudgeCause((), "observation", reason) for reason in local)
        else:
            own = []
            if node.fields.get("schema_errors") or node.fields.get("verdict_validation_error"):
                own.append(_JudgeCause((), "aggregate", "judge verdict failed validation"))
            reason = _aggregate_summary_reason(population.saved_summary)
            if reason is not None:
                own.append(_JudgeCause((), "aggregate", reason))
            causes = tuple(own) + _judge_population_causes(population, node.fields, children)
            if not causes:
                causes = (_JudgeCause((), "aggregate", "judge observation is not explicitly complete"),)
        subtrees[id(node)] = causes
        return causes

    visit(observation)
    return subtrees


def _judge_diagnostics(observation: _JudgeObservation) -> tuple[_JudgeCause, ...]:
    return _judge_diagnostic_tree(observation)[id(observation)]


def _judge_diagnostic_evidence_fields(observation: _JudgeObservation) -> dict[str, Any]:
    supplied = observation.supplied_diagnostics
    evidence = {"version": 1, "supplied": {"kind": "present", "value": thaw_json_value(supplied.value)}
                if isinstance(supplied, _SuppliedDiagnostics) else {"kind": "absent"}}
    if isinstance(observation.population, _JudgeLeaf) and not isinstance(observation, _CompleteJudgeObservation):
        evidence["leaf_reasons"] = list(observation.local_reasons if isinstance(observation, _MissingJudgeObservation)
                                        else observation.reasons)
    return evidence


def _admit_judge_diagnostic_evidence(observation: _JudgeObservation, raw: Mapping[str, Any]) -> _JudgeObservation:
    if "judge_diagnostic_evidence" not in raw:
        return replace(observation, supplied_diagnostics=_SuppliedDiagnostics(raw["incomplete_judge_members"])
                       if "incomplete_judge_members" in raw else _AbsentSuppliedDiagnostics())
    envelope = raw["judge_diagnostic_evidence"]
    if (not isinstance(envelope, Mapping) or type(envelope.get("version")) is not int
            or envelope["version"] != 1):
        raise ValueError("judge diagnostic evidence requires integer version 1")
    unavailable_leaf = isinstance(observation.population, _JudgeLeaf) and not isinstance(observation, _CompleteJudgeObservation)
    if "leaf_reasons" in envelope and not unavailable_leaf:
        raise ValueError("judge leaf reasons must belong to an unavailable leaf")
    if set(envelope) != ({"version", "supplied", "leaf_reasons"} if unavailable_leaf else {"version", "supplied"}):
        raise ValueError("judge diagnostic evidence has invalid fields")
    supplied = envelope["supplied"]
    if not isinstance(supplied, Mapping):
        raise ValueError("judge supplied diagnostics must be an object")
    kind = supplied.get("kind")
    if kind not in ("absent", "present") or set(supplied) != (
            {"kind"} if kind == "absent" else {"kind", "value"}):
        raise ValueError("judge supplied diagnostics has invalid kind or fields")
    observation = replace(observation, supplied_diagnostics=_SuppliedDiagnostics(supplied["value"])
                          if kind == "present" else _AbsentSuppliedDiagnostics())
    if unavailable_leaf:
        reasons = envelope["leaf_reasons"]
        if not isinstance(reasons, list):
            raise ValueError("judge leaf reasons must be a nonempty array")
        local = tuple(reasons)
        _check_local_reasons(local)
        if isinstance(observation, _MissingJudgeObservation):
            observation = replace(observation, local_reasons=local)
        elif isinstance(observation, _PartialJudgeObservation):
            observation = replace(observation, reasons=local)
    if isinstance(observation.population, _JudgeLeaf):
        if "incomplete_judge_members" in raw:
            raise ValueError("canonical judge member diagnostics must belong to a group")
    elif "incomplete_judge_members" not in raw or not strict_json_equal(
            raw["incomplete_judge_members"], _judge_diagnostic_records(_judge_diagnostics(observation))):
        raise ValueError("judge calculated diagnostics contradict retained population")
    return observation


def _judge_model(observation: _JudgeObservation) -> Any:
    return observation.fields.get("judge_requested_model", observation.fields.get("judge_model"))


def _validate_judge_group_identity(members: tuple[_JudgeObservation, ...], *, panel: bool) -> None:
    if not members:
        raise ValueError("judge group requires at least one member")
    ids = {member.fields.get("judge_task_id", member.fields.get("id")) for member in members}
    if len(ids) != 1 or None in ids:
        raise ValueError("judge panel rows must share one task id" if panel else
                         "judge repeats must share one task id and verdict kind")
    if panel:
        models = [_judge_model(member) for member in members]
        if any(not isinstance(model, str) or not model for model in models) or len(set(models)) != len(models):
            raise ValueError("judge panel models must be non-empty and unique")
    else:
        models = {member.fields.get("judge_requested_model") for member in members
                  if "judge_requested_model" in member.fields}
        if len(models) > 1:
            raise ValueError("judge repeats must share one requested model")
        repeats = [member.fields.get("judge_repeat") for member in members]
        if any(value is not None for value in repeats) and (
                any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in repeats)
                or len(set(repeats)) != len(repeats)):
            raise ValueError("judge repeat identities must be positive and unique")
    for key in _BINDING_FIELDS:
        if key in {"judge_task_id", "id", "judge_input_sha256"}:
            continue
        values = [member.fields.get(key) for member in members]
        if any(member.fresh for member in members) and any(value != values[0] for value in values):
            raise ValueError(f"judge members must share one {key}")


def _judge_population_causes(population: _JudgeRepeats | _JudgePanel, fields: Mapping[str, Any],
                              children: tuple[tuple[_JudgeCause, ...], ...]) -> tuple[_JudgeCause, ...]:
    members = population.members
    causes = [_JudgeCause((index,) + cause.path, cause.scope, cause.reason)
              for index, child in enumerate(children, 1) for cause in child]
    fingerprints = [member.fields.get("judge_input_sha256") for member in members]
    if any(not isinstance(value, str) for value in fingerprints) or len(set(fingerprints)) != 1:
        causes.append(_JudgeCause((), "aggregate", "judge members must share one explicit judge_input_sha256"))
    kinds = {member.explicit_kind for member in members
             if isinstance(member, (_CompleteJudgeObservation, _PartialJudgeObservation))
             and member.explicit_kind is not None}
    if len(kinds) > 1 and not causes:
        raise ValueError("judge panel rows must share one verdict kind" if isinstance(population, _JudgePanel) else
                         "judge repeats must share one task id and verdict kind")
    if "judge_input_sha256" in fields and any(value != fields["judge_input_sha256"] for value in fingerprints):
        causes.append(_JudgeCause((), "aggregate", "judge parent input fingerprint contradicts members"))
    if len(kinds) > 1:
        causes.extend(_JudgeCause((index,), "observation", "judge member has incompatible explicit verdict kind")
                       for index, member in enumerate(members, 1)
                       if isinstance(member, (_CompleteJudgeObservation, _PartialJudgeObservation))
                       and member.explicit_kind is not None)
    return tuple(causes)


def _judge_group_errors(members: tuple[_JudgeObservation, ...], *, panel: bool) -> list[dict[str, Any]]:
    _validate_judge_group_identity(members, panel=panel)
    population = _JudgePanel(members, tuple(_judge_model(member) for member in members)) if panel else _JudgeRepeats(members)
    return _judge_diagnostic_records(_judge_population_causes(population, {}, tuple(_judge_diagnostics(member) for member in members)))


def _fold_judge_observations(members: tuple[_JudgeObservation, ...], *, panel: bool = False,
                             quorum: int | None = None) -> _JudgeObservation:
    errors = _judge_group_errors(members, panel=panel)
    if len(members) == 1:
        return members[0]
    fresh = all(member.fresh for member in members)
    fields: dict[str, Any] = {key: members[0].fields[key] for key in _BINDING_FIELDS
                              if key in members[0].fields
                              and all(member.fields.get(key) == members[0].fields[key] for member in members)}
    if panel:
        fields["judge_model"] = "consensus"
    elif fresh:
        fields["judge_model"] = _judge_model(members[0]) or fields.get("judge_backend")
    elif all(member.fields.get("judge_model") == members[0].fields.get("judge_model") for member in members):
        fields["judge_model"] = members[0].fields.get("judge_model")
    if not panel and fresh:
        fields["judge_requested_model"] = _judge_model(members[0])
    threshold = None
    first = members[0]
    if isinstance(first, (_CompleteJudgeObservation, _PartialJudgeObservation)) and isinstance(
            first.verdict, (ScoredVerdict, DimensionVerdict)):
        threshold = first.verdict.threshold
    policy = _JudgeConsensusPolicy(threshold, quorum)
    population = (_JudgePanel(members, tuple(_judge_model(member) for member in members), policy if fresh else None)
                  if panel else _JudgeRepeats(members, len(members) if fresh else None, policy if fresh else None))
    if errors:
        fields["evidence"] = "judge aggregate incomplete: " + "; ".join(str(error["reason"]) for error in errors[:5])
        return _MissingJudgeObservation(freeze_json_mapping(fields, "judge group"), population,
                                        (), fresh)
    complete_members = tuple(member for member in members if isinstance(member, _CompleteJudgeObservation))
    consensus = _resolve_judge_observations(complete_members, policy)
    fields["evidence"] = " | ".join(str(member.fields.get("evidence", "")) for member in members
                                    if member.fields.get("evidence"))[:4000]
    return _CompleteJudgeObservation(freeze_json_mapping(fields, "judge group"), population,
                                     consensus.verdict(), VerdictKind.CONSENSUS, fresh, consensus)


def _resolve_judge_observations(members: tuple[_CompleteJudgeObservation, ...],
                                policy: _JudgeConsensusPolicy) -> Consensus:
    return resolve_consensus([member.verdict.passed for member in members],
                             [member.verdict.score for member in members
                              if not isinstance(member.verdict, BooleanVerdict) and member.verdict.score is not None],
                             threshold=policy.threshold, quorum=policy.quorum)


def _judge_observation_fields(observation: _JudgeObservation) -> dict[str, Any]:
    subtrees = _judge_diagnostic_tree(observation)

    def project(observation: _JudgeObservation) -> dict[str, Any]:
        out = thaw_json_value(observation.fields, "judge metadata")
        if isinstance(observation, _MissingJudgeObservation):
            out.update(verdict_fields(ConsensusVerdict(False)))
            kind, complete = "missing", False
        else:
            out.update(verdict_fields(observation.verdict))
            if not observation.fresh and observation.explicit_kind is None:
                out.pop("verdict_kind", None)
            kind, complete = ("complete", True) if isinstance(observation, _CompleteJudgeObservation) else ("partial", False)
        out.update(judge_observation_complete=complete, availability="complete" if complete else "partial")
        if observation.fresh:
            out["judge_observation_kind"] = kind
        population = observation.population
        if isinstance(population, _JudgeLeaf):
            out.update(thaw_json_value(population.attempt.facts, "judge execution"))
            if observation.fresh:
                out["judge_execution_kind"] = population.attempt.kind
        else:
            key = "judge_panel" if isinstance(population, _JudgePanel) else "judge_runs"
            out[key] = [project(member) for member in population.members]
            out["incomplete_judge_members"] = _judge_diagnostic_records(subtrees[id(observation)])
            out["returncode"] = 0 if complete else 1
            out["judge_aggregate_summary"] = _aggregate_summary_fields(population.saved_summary)
            if isinstance(population, _JudgePanel):
                if population.models_recorded:
                    out["judge_models"] = list(population.requested_models)
            elif population.expected_count is not None:
                out["judge_expected_repeats"] = population.expected_count
            if population.policy is not None:
                out["judge_consensus_policy"] = {"threshold": population.policy.threshold, "quorum": population.policy.quorum}
            if isinstance(observation, _CompleteJudgeObservation) and observation.agreement is not None:
                out["agreement"] = (observation.agreement.agreement() if isinstance(observation.agreement, Consensus)
                                    else thaw_json_value(observation.agreement, "judge agreement"))
        out["judge_diagnostic_evidence"] = _judge_diagnostic_evidence_fields(observation)
        return out

    return project(observation)


def _judge_observation_from_row(raw: Mapping[str, Any]) -> _JudgeObservation:
    if not isinstance(raw, Mapping):
        raise TypeError("judge result row must be an object")
    marker = raw.get("judge_observation_kind")
    if marker is not None and marker not in {"complete", "partial", "missing"}:
        raise ValueError("unknown judge_observation_kind")
    fresh = marker is not None
    explicit = VerdictKind(raw["verdict_kind"]) if raw.get("verdict_kind") is not None else None
    verdict = verdict_from_dict(raw)
    if marker == "missing" and ({key: raw[key] for key in _SEMANTIC_FIELDS | {"verdict_kind"} if key in raw}
                                != verdict_fields(ConsensusVerdict(False))):
        raise ValueError("missing judge observation requires exact consensus false shell")
    if fresh and (raw.get("judge_observation_complete") is not (marker == "complete")
                  or raw.get("availability") != ("complete" if marker == "complete" else "partial")):
        raise ValueError("judge observation marker contradicts completeness")
    memberships = [key for key in ("judge_runs", "judge_panel") if key in raw]
    if len(memberships) > 1:
        raise ValueError("judge row cannot contain both membership paths")
    if not memberships:
        if "judge_aggregate_summary" in raw:
            raise ValueError("judge aggregate summary must belong to a group")
        kind = raw.get("judge_execution_kind")
        if kind is None:
            kind = "not_started" if raw.get("invocation_state") == "not_started" else "historical"
        if kind not in {"process", "not_started", "guard", "historical"}:
            raise ValueError("unknown judge execution kind")
        if fresh:
            repeat = raw.get("judge_repeat")
            if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
                raise ValueError("judge repeat identity must be positive")
            if "judge_requested_model" not in raw:
                raise ValueError("fresh judge requires requested model identity")
            requested = raw["judge_requested_model"]
            if requested is not None and (not isinstance(requested, str) or not requested):
                raise ValueError("judge requested model must be non-empty or null")
            if kind == "historical":
                raise ValueError("fresh judge cannot have historical execution")
        attempt = _JudgeAttempt(kind, freeze_json_mapping(
            {key: value for key, value in raw.items() if key in _ATTEMPT_FIELDS}, "judge execution"))
        fields = {key: value for key, value in raw.items() if key not in _RESULT_FIELDS}
        reason = None
        if raw.get("judge_observation_complete") is not True:
            reason = "judge observation is not explicitly complete"
        elif raw.get("availability") != "complete":
            reason = "judge result availability is not explicitly complete"
        else:
            reason = _judge_leaf_incomplete_reason(fields, attempt)
        if marker == "complete" and reason is not None:
            raise ValueError(reason)
        observation = _judge_leaf_observation(None if marker == "missing" else verdict, raw,
                                              attempt_kind=kind, complete=reason is None,
                                              reasons=(reason,) if reason else (), fresh=fresh, explicit_kind=explicit)
        return _admit_judge_diagnostic_evidence(observation, raw)
    if marker == "partial":
        raise ValueError("partial judge observation must be a leaf")
    key = memberships[0]
    children = raw[key]
    if not isinstance(children, list) or not children:
        raise ValueError("judge members must be a non-empty list")
    members = tuple(_judge_observation_from_row(child) for child in children)
    _validate_judge_group_identity(members, panel=key == "judge_panel")
    provisional = _JudgePanel(members, tuple(_judge_model(member) for member in members)) if key == "judge_panel" else _JudgeRepeats(members)
    errors = _judge_population_causes(provisional, raw, tuple(_judge_diagnostics(member) for member in members))
    expected = raw.get("judge_expected_repeats")
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)
                                 or expected != len(members) or expected < 1):
        raise ValueError("judge repeat population contradicts expected count")
    if fresh and key == "judge_runs" and {member.fields.get("judge_repeat") for member in members} != set(range(1, len(members) + 1)):
        raise ValueError("judge repeat identities do not match requested population")
    if fresh and (not all(member.fresh for member in members)
                  or (key == "judge_runs" and expected is None)
                  or (key == "judge_panel" and "judge_models" not in raw)):
        raise ValueError("fresh judge group requires its full declared population")
    models = tuple(_judge_model(member) for member in members)
    if key == "judge_panel" and "judge_models" in raw and raw["judge_models"] != list(models):
        raise ValueError("judge panel models contradict requested population")
    if key == "judge_runs" and "judge_requested_model" in raw and any(
            "judge_requested_model" in member.fields
            and member.fields["judge_requested_model"] != raw["judge_requested_model"] for member in members):
        raise ValueError("judge parent requested model contradicts members")
    for name in _BINDING_FIELDS:
        if name != "judge_input_sha256" and name in raw and any(member.fields.get(name) != raw[name] for member in members):
            raise ValueError(f"judge parent {name} contradicts members")
    if fresh and any(name in raw for name in ("spend_call_id", "judge_repeat", "judge_served_model", "invocation_state", "timed_out", "provider_error", "judge_execution_kind")):
        raise ValueError("judge aggregate cannot carry leaf execution facts")
    policy_raw = raw.get("judge_consensus_policy")
    policy = None
    if policy_raw is not None:
        if not isinstance(policy_raw, Mapping) or set(policy_raw) != {"threshold", "quorum"}:
            raise ValueError("judge consensus policy must record threshold and quorum")
        policy = _JudgeConsensusPolicy(policy_raw["threshold"], policy_raw["quorum"])
    if fresh and policy is None:
        raise ValueError("fresh judge group requires recorded consensus policy")
    summary = _saved_aggregate_summary(raw, fresh=fresh)
    population = (_JudgePanel(members, models, policy, "judge_models" in raw, summary) if key == "judge_panel" else
                  _JudgeRepeats(members, expected, policy, summary))
    fields = freeze_json_mapping({name: value for name, value in raw.items() if name not in _RESULT_FIELDS}, "judge group")
    validation_failed = bool(raw.get("schema_errors") or raw.get("verdict_validation_error"))
    if marker == "complete" and validation_failed:
        raise ValueError("judge verdict failed validation")
    complete = (raw.get("judge_observation_complete") is True and raw.get("availability") == "complete"
                and not errors and not validation_failed)
    if raw.get("judge_observation_complete") is True and errors:
        raise ValueError("complete judge group requires complete compatible children")
    agreement: Consensus | Mapping[str, Any] | None = None
    if complete:
        if not isinstance(verdict, ConsensusVerdict):
            raise ValueError("judge group requires consensus verdict")
        if policy is not None:
            agreement = _resolve_judge_observations(tuple(member for member in members
                                                        if isinstance(member, _CompleteJudgeObservation)), policy)
            if verdict != agreement.verdict() or raw.get("agreement") != agreement.agreement():
                raise ValueError("recorded judge consensus contradicts members and policy")
        elif "agreement" in raw:
            if not isinstance(raw["agreement"], Mapping):
                raise ValueError("judge agreement must be an object")
            agreement = raw["agreement"]
    summary_reason = _aggregate_summary_reason(summary)
    if not complete or summary_reason is not None:
        observation = _MissingJudgeObservation(fields, population, (), fresh)
    else:
        observation = _CompleteJudgeObservation(fields, population, verdict, explicit, fresh, agreement)
    return _admit_judge_diagnostic_evidence(observation, raw)


def _judge_observation_matches_steps(observation: _JudgeObservation, *, fingerprint: str | None,
                                     names: tuple[str, ...], minimum: int) -> bool:
    if not isinstance(observation, _CompleteJudgeObservation):
        return False
    if isinstance(observation.population, _JudgeLeaf):
        verdict = observation.verdict
        return (isinstance(verdict, DynamicVerdict)
                and tuple(name for name, _ in verdict.criteria) == names
                and verdict.minimum_criteria == minimum
                and observation.fields.get("trajectory_steps_sha256") == fingerprint)
    return all(_judge_observation_matches_steps(member, fingerprint=fingerprint,
                                               names=names, minimum=minimum)
               for member in observation.population.members)
