"""Strict discriminated judge verdicts and stored-result boundary parsing."""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias

from invocation_contracts import InvocationState, validate_invocation_lifecycle
from json_contracts import freeze_json_mapping, thaw_json_value
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
class _JudgeRepeats:
    members: tuple[_JudgeObservation, ...]
    expected_count: int | None = None
    policy: _JudgeConsensusPolicy | None = None


@dataclass(frozen=True)
class _JudgePanel:
    members: tuple[_JudgeObservation, ...]
    requested_models: tuple[str, ...]
    policy: _JudgeConsensusPolicy | None = None
    models_recorded: bool = True


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
class _CompleteJudgeObservation:
    fields: Mapping[str, Any]
    population: _JudgePopulation
    verdict: JudgeVerdict
    explicit_kind: VerdictKind | None
    fresh: bool = True
    agreement: Consensus | Mapping[str, Any] | None = None

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
        elif not all(isinstance(member, _CompleteJudgeObservation) for member in self.population.members):
            raise ValueError("complete judge group requires complete children")


@dataclass(frozen=True)
class _PartialJudgeObservation:
    fields: Mapping[str, Any]
    population: _JudgeLeaf
    verdict: JudgeVerdict
    explicit_kind: VerdictKind | None
    reasons: tuple[str, ...]
    fresh: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", freeze_json_mapping(self.fields, "judge metadata"))
        _check_observation_fields(self.fields)
        if not self.reasons:
            raise ValueError("partial judge observation requires a reason")


@dataclass(frozen=True)
class _MissingJudgeObservation:
    fields: Mapping[str, Any]
    population: _JudgePopulation
    reasons: tuple[str, ...]
    fresh: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", freeze_json_mapping(self.fields, "judge metadata"))
        _check_observation_fields(self.fields)
        if not self.reasons:
            raise ValueError("missing judge observation requires a reason")


_JudgeObservation: TypeAlias = _CompleteJudgeObservation | _PartialJudgeObservation | _MissingJudgeObservation
_ATTEMPT_FIELDS = frozenset({"returncode", "invocation_state", "timed_out", "provider_error", "judge_served_model"})
_RESULT_FIELDS = _SEMANTIC_FIELDS | frozenset({
    "verdict_kind", "judge_observation_kind", "judge_observation_complete", "availability",
    "judge_runs", "judge_panel", "agreement", "judge_expected_repeats", "judge_models",
    "judge_consensus_policy", "judge_execution_kind",
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
    if isinstance(observation, _CompleteJudgeObservation):
        return None
    return observation.reasons[0]


def _judge_model(observation: _JudgeObservation) -> Any:
    return observation.fields.get("judge_requested_model", observation.fields.get("judge_model"))


def _judge_group_errors(members: tuple[_JudgeObservation, ...], *, panel: bool) -> list[dict[str, Any]]:
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
    errors = [{"member": index, "reason": reason} for index, member in enumerate(members, 1)
              if (reason := _judge_observation_reason(member)) is not None]
    fingerprints = [member.fields.get("judge_input_sha256") for member in members]
    if any(not isinstance(value, str) for value in fingerprints) or len(set(fingerprints)) != 1:
        errors.append({"member": "aggregate", "reason": "judge members must share one explicit judge_input_sha256"})
    kinds = {member.explicit_kind for member in members
             if isinstance(member, (_CompleteJudgeObservation, _PartialJudgeObservation))
             and member.explicit_kind is not None}
    if len(kinds) > 1:
        if not errors:
            raise ValueError("judge panel rows must share one verdict kind" if panel else
                             "judge repeats must share one task id and verdict kind")
        errors.extend({"member": index, "reason": "judge member has incompatible explicit verdict kind"}
                      for index, member in enumerate(members, 1)
                      if isinstance(member, (_CompleteJudgeObservation, _PartialJudgeObservation))
                      and member.explicit_kind is not None)
    return errors


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
        fields["incomplete_judge_members"] = errors
        fields["evidence"] = "judge aggregate incomplete: " + "; ".join(str(error["reason"]) for error in errors[:5])
        return _MissingJudgeObservation(freeze_json_mapping(fields, "judge group"), population,
                                        tuple(str(error["reason"]) for error in errors), fresh)
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
        out[key] = [_judge_observation_fields(member) for member in population.members]
        out["returncode"] = 0 if complete else 1
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
    return out


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
        return _judge_leaf_observation(None if marker == "missing" else verdict, raw,
                                       attempt_kind=kind, complete=reason is None,
                                       reasons=(reason,) if reason else (), fresh=fresh, explicit_kind=explicit)
    if marker == "partial":
        raise ValueError("partial judge observation must be a leaf")
    key = memberships[0]
    children = raw[key]
    if not isinstance(children, list) or not children:
        raise ValueError("judge members must be a non-empty list")
    members = tuple(_judge_observation_from_row(child) for child in children)
    errors = _judge_group_errors(members, panel=key == "judge_panel")
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
    for name in _BINDING_FIELDS:
        if name in raw and any(member.fields.get(name) != raw[name] for member in members):
            if name != "judge_input_sha256":
                raise ValueError(f"judge parent {name} contradicts members")
            errors.append({"member": "aggregate", "reason": "judge parent input fingerprint contradicts members"})
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
    if fresh and raw.get("returncode") != (0 if marker == "complete" else 1):
        raise ValueError("judge aggregate returncode contradicts completeness")
    population = (_JudgePanel(members, models, policy, "judge_models" in raw) if key == "judge_panel" else
                  _JudgeRepeats(members, expected, policy))
    fields = freeze_json_mapping({name: value for name, value in raw.items() if name not in _RESULT_FIELDS}, "judge group")
    complete = raw.get("judge_observation_complete") is True and raw.get("availability") == "complete" and not errors
    if raw.get("judge_observation_complete") is True and errors:
        raise ValueError("complete judge group requires complete compatible children")
    if not complete:
        return _MissingJudgeObservation(fields, population,
                                        tuple(str(error["reason"]) for error in errors) or ("judge observation is not explicitly complete",), fresh)
    if not isinstance(verdict, ConsensusVerdict):
        raise ValueError("judge group requires consensus verdict")
    agreement: Consensus | Mapping[str, Any] | None = None
    if policy is not None:
        agreement = _resolve_judge_observations(tuple(member for member in members
                                                    if isinstance(member, _CompleteJudgeObservation)), policy)
        if verdict != agreement.verdict() or raw.get("agreement") != agreement.agreement():
            raise ValueError("recorded judge consensus contradicts members and policy")
    elif "agreement" in raw:
        if not isinstance(raw["agreement"], Mapping):
            raise ValueError("judge agreement must be an object")
        agreement = raw["agreement"]
    return _CompleteJudgeObservation(fields, population, verdict, explicit, fresh, agreement)


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
