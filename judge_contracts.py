"""Immutable invocation results for qualitative judge backends.

Provider adapters and the shell-command escape hatch construct this type once at
the process boundary.  Verdict parsing may therefore consume one closed shape
instead of repairing independently assembled dictionaries.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import telemetry
from invocation_contracts import InvocationState, validate_invocation_lifecycle
from json_contracts import freeze_json_mapping, validate_json_text

JudgeUsageSource = Literal["provider_reported", "trace_normalized"]
JUDGE_USAGE_SOURCES = frozenset({"provider_reported", "trace_normalized"})
JUDGE_METADATA_RESERVED_FIELDS = frozenset({
    "cost_usd", "cost_normalized", "cost_availability", "observed_subtotal_usd",
    "cost_reason", "cost_aggregate", "telemetry", "returncode", "timed_out",
    "invocation_state", "provider_error",
})


@dataclass(frozen=True)
class JudgeInvocation:
    """One completed attempt to obtain a verdict from a judge backend.

    A nonzero ``returncode`` is a valid invocation record, but never a successful
    observation.  Provider output may be empty on failure; identity and telemetry
    remain independently validated and immutable for diagnostics and accounting.
    """

    stdout: str
    stderr: str
    returncode: int
    invocation_state: InvocationState
    provider_error: str | None = None
    usage: Mapping[str, Any] | None = None
    cost_usd: float | None = None
    usage_source: JudgeUsageSource = "provider_reported"
    model_label: str | None = None
    raw_response: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    observed_subtotal_usd: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("judge stdout and stderr must be strings")
        validate_json_text(self.stdout, "judge stdout")
        validate_json_text(self.stderr, "judge stderr")
        if type(self.returncode) is not int:
            raise TypeError("judge returncode must be an integer")
        validate_invocation_lifecycle(
            self.invocation_state, self.returncode, self.provider_error)
        if self.provider_error is not None:
            validate_json_text(self.provider_error, "judge provider_error")
        if self.usage_source not in JUDGE_USAGE_SOURCES:
            raise ValueError(
                f"unknown judge usage source {self.usage_source!r}")
        if self.model_label is not None and (
                not isinstance(self.model_label, str) or not self.model_label.strip()):
            raise ValueError("judge model label must be None or a non-empty string")
        if self.model_label is not None:
            validate_json_text(self.model_label, "judge model label")
        if self.raw_response is not None and not isinstance(self.raw_response, str):
            raise TypeError("judge raw_response must be text or None")
        if self.raw_response is not None:
            validate_json_text(self.raw_response, "judge raw_response")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("judge metadata must be a mapping")
        reserved = JUDGE_METADATA_RESERVED_FIELDS.intersection(self.metadata)
        if reserved:
            raise ValueError(f"judge metadata contains reserved fields: {sorted(reserved)}")
        object.__setattr__(self, "metadata", freeze_json_mapping(
            self.metadata, "judge metadata"))
        for key in ("cost_usd", "observed_subtotal_usd"):
            value = getattr(self, key)
            if value is not None:
                telemetry.finite_nonnegative(value, f"judge {key}")
                object.__setattr__(self, key, float(value))
        if self.cost_usd is not None and self.observed_subtotal_usd is not None:
            raise ValueError("judge full cost and observed subtotal cannot coexist")
        if self.cost_usd is not None and self.invocation_state in {
                InvocationState.TIMED_OUT, InvocationState.SPAWN_FAILED}:
            raise ValueError("judge full cost requires a naturally exited process")
        if self.observed_subtotal_usd is not None and (
                self.invocation_state is not InvocationState.TIMED_OUT or self.returncode != 124):
            raise ValueError("judge observed subtotal requires an actual timeout with code 124")
        if self.usage is not None:
            if not isinstance(self.usage, Mapping):
                raise TypeError("judge usage must be a mapping or None")
            telemetry.validate_raw_usage(self.usage, "judge usage")
            frozen_usage = freeze_json_mapping(self.usage, "judge usage")
            telemetry.canonical_usage_counts(frozen_usage)
            object.__setattr__(self, "usage", frozen_usage)

    @property
    def succeeded(self) -> bool:
        return self.invocation_state is InvocationState.COMPLETE
