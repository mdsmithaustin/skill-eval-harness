"""One invocation's immutable spend plan, receipts, and admission state."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from manifest_contracts import RunCoordinate
from telemetry import AVAILABLE, PROVENANCE, Measurement, Money

SPEND_LEDGER_NAME = "spend-ceiling.json"


def _usd(value: Money) -> None:
    if not isinstance(value, Money):
        raise TypeError("spend amounts must be Money")
    if value.currency != "USD":
        raise ValueError("spend amounts must use USD")


def _reason(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("spend reason must be nonempty")


class SpendStopReason(str, Enum):
    COST_CEILING = "cost_ceiling"
    COST_UNOBSERVABLE = "cost_unobservable"


@dataclass(frozen=True)
class SpendPolicy:
    ceiling: Money
    assumed_cost_per_run: Money | None = None

    def __post_init__(self) -> None:
        _usd(self.ceiling)
        if self.assumed_cost_per_run is not None:
            _usd(self.assumed_cost_per_run)

    @classmethod
    def from_raw(cls, ceiling: Any, assumed_cost_per_run: Any = None) -> SpendPolicy:
        return cls(Money.from_raw(ceiling), None if assumed_cost_per_run is None
                   else Money.from_raw(assumed_cost_per_run))

    def as_dict(self) -> dict[str, Any]:
        return {"ceiling_usd": format(self.ceiling.amount, "f"),
                "assumed_cost_per_run_usd": None if self.assumed_cost_per_run is None
                else format(self.assumed_cost_per_run.amount, "f")}


@dataclass(frozen=True)
class AnswerCall:
    task_sha256: str
    coordinate: RunCoordinate

    def __post_init__(self) -> None:
        if not isinstance(self.task_sha256, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", self.task_sha256) is None:
            raise ValueError("answer call requires a task SHA-256")
        if not isinstance(self.coordinate, RunCoordinate):
            raise TypeError("answer call requires a RunCoordinate")

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "answer", "task_sha256": self.task_sha256,
                "coordinate": self.coordinate.as_dict()}

    @property
    def call_id(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SpendPlan:
    calls: tuple[AnswerCall, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.calls, tuple) or not all(isinstance(call, AnswerCall) for call in self.calls):
            raise TypeError("spend plan requires a tuple of answer calls")
        if len({call.call_id for call in self.calls}) != len(self.calls):
            raise ValueError("spend plan contains duplicate call identities")


@dataclass(frozen=True)
class ObservedCharge:
    amount: Money
    provenance: str

    def __post_init__(self) -> None:
        _usd(self.amount)
        if self.provenance not in PROVENANCE:
            raise ValueError("observed charge requires known provenance")


@dataclass(frozen=True)
class AssumedCharge:
    amount: Money
    reason: str
    observed_subtotal: Money | None = None

    def __post_init__(self) -> None:
        _usd(self.amount)
        _reason(self.reason)
        if self.observed_subtotal is not None:
            _usd(self.observed_subtotal)
            if self.amount.amount < self.observed_subtotal.amount:
                raise ValueError("assumed charge cannot be below the observed subtotal")


@dataclass(frozen=True)
class UnpricedCall:
    reason: str
    observed_subtotal: Money | None = None

    def __post_init__(self) -> None:
        _reason(self.reason)
        if self.observed_subtotal is not None:
            _usd(self.observed_subtotal)


@dataclass(frozen=True)
class NoModelSpend:
    reason: Literal["offline_adapter", "spawn_failed_before_process"]

    def __post_init__(self) -> None:
        if self.reason not in {"offline_adapter", "spawn_failed_before_process"}:
            raise ValueError("nonbillable charge requires explicit no-model evidence")


Charge = ObservedCharge | AssumedCharge | UnpricedCall | NoModelSpend


def price_measurement(policy: SpendPolicy, cost: Measurement[Money], *,
                      observed_subtotal: Money | None = None,
                      no_model_spend: NoModelSpend | None = None) -> Charge:
    if not isinstance(cost, Measurement):
        raise TypeError("spend cost requires a Measurement")
    if observed_subtotal is not None:
        _usd(observed_subtotal)
    if no_model_spend is not None:
        if cost.availability == AVAILABLE or observed_subtotal is not None:
            raise ValueError("nonbillable call cannot carry observed dollars")
        return no_model_spend
    if cost.availability == AVAILABLE:
        if not isinstance(cost.value, Money):
            raise TypeError("spend cost requires Money")
        if cost.value.currency == "USD" and observed_subtotal is None:
            return ObservedCharge(cost.value, str(cost.provenance))
    reason = cost.reason or "incomplete_or_non_usd_cost"
    if policy.assumed_cost_per_run is None:
        return UnpricedCall(reason, observed_subtotal)
    assumed = policy.assumed_cost_per_run.amount
    amount = max(assumed, observed_subtotal.amount) if observed_subtotal is not None else assumed
    return AssumedCharge(Money(amount, "USD"), reason, observed_subtotal)


@dataclass(frozen=True)
class Planned:
    call: AnswerCall


@dataclass(frozen=True)
class InFlight:
    call: AnswerCall


@dataclass(frozen=True)
class Settled:
    call: AnswerCall
    charge: Charge


@dataclass(frozen=True)
class Refused:
    call: AnswerCall
    reason: SpendStopReason


CallState = Planned | InFlight | Settled | Refused


def _charge_dict(charge: Charge) -> dict[str, Any]:
    if isinstance(charge, ObservedCharge):
        return {"basis": "observed", "amount_usd": format(charge.amount.amount, "f"),
                "provenance": charge.provenance}
    if isinstance(charge, NoModelSpend):
        return {"basis": "no_model_spend", "reason": charge.reason}
    subtotal = None if charge.observed_subtotal is None else format(charge.observed_subtotal.amount, "f")
    if isinstance(charge, AssumedCharge):
        return {"basis": "assumed", "amount_usd": format(charge.amount.amount, "f"),
                "reason": charge.reason, "observed_subtotal_usd": subtotal}
    return {"basis": "unpriced", "reason": charge.reason, "observed_subtotal_usd": subtotal}


@dataclass(frozen=True)
class SpendLedger:
    invocation_id: str
    policy: SpendPolicy
    states: Mapping[str, CallState]

    def __post_init__(self) -> None:
        if not isinstance(self.invocation_id, str) or re.fullmatch(r"[0-9a-f]{32}", self.invocation_id) is None:
            raise ValueError("spend invocation id must be a UUID hex string")
        if not isinstance(self.policy, SpendPolicy):
            raise TypeError("spend ledger requires a SpendPolicy")
        for key, state in self.states.items():
            if not isinstance(state, (Planned, InFlight, Settled, Refused)) or key != state.call.call_id:
                raise ValueError("spend states must match their canonical call identities")
            if isinstance(state, Settled) and not isinstance(
                    state.charge, (ObservedCharge, AssumedCharge, UnpricedCall, NoModelSpend)):
                raise TypeError("settled call requires a closed charge state")
            if isinstance(state, Refused) and not isinstance(state.reason, SpendStopReason):
                raise TypeError("refused call requires a SpendStopReason")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))

    @classmethod
    def planned(cls, invocation_id: str, policy: SpendPolicy, plan: SpendPlan) -> SpendLedger:
        return cls(invocation_id, policy, {call.call_id: Planned(call) for call in plan.calls})

    @property
    def spent(self) -> Money:
        amounts = []
        for state in self.states.values():
            if isinstance(state, Settled):
                if isinstance(state.charge, (ObservedCharge, AssumedCharge)):
                    amounts.append(state.charge.amount.amount)
                elif isinstance(state.charge, UnpricedCall) and state.charge.observed_subtotal is not None:
                    amounts.append(state.charge.observed_subtotal.amount)
        return Money(sum(amounts, Decimal(0)), "USD")

    @property
    def partial(self) -> bool:
        return any(isinstance(state, InFlight) or
                   isinstance(state, Settled) and isinstance(state.charge, UnpricedCall)
                   for state in self.states.values())

    @property
    def refusal(self) -> SpendStopReason | None:
        if self.partial:
            return SpendStopReason.COST_UNOBSERVABLE
        if self.spent.amount >= self.policy.ceiling.amount:
            return SpendStopReason.COST_CEILING
        return None

    def transition(self, state: CallState) -> SpendLedger:
        previous = self.states.get(state.call.call_id)
        if previous == state:
            return self
        if not (isinstance(previous, Planned) and isinstance(state, (InFlight, Refused))
                or isinstance(previous, InFlight) and isinstance(state, Settled)):
            raise TypeError("invalid spend call transition")
        return replace(self, states={**self.states, state.call.call_id: state})

    def as_dict(self) -> dict[str, Any]:
        calls = []
        for state in self.states.values():
            row = {"call_id": state.call.call_id, "call": state.call.as_dict()}
            if isinstance(state, Planned):
                row["state"] = "planned"
            elif isinstance(state, InFlight):
                row["state"] = "in_flight"
            elif isinstance(state, Refused):
                row.update(state="not_started", reason=state.reason.value,
                           invocation_state="not_started")
            else:
                row.update(state="settled", charge=_charge_dict(state.charge))
            calls.append(row)
        return {"schema_version": 1, "invocation_id": self.invocation_id,
                **self.policy.as_dict(), "spent_usd": format(self.spent.amount, "f"),
                "spent_availability": "partial" if self.partial else "complete",
                "calls": calls}

    @classmethod
    def from_dict(cls, raw: Any) -> SpendLedger:
        if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
            raise ValueError("unsupported spend ledger schema")
        policy = SpendPolicy.from_raw(raw.get("ceiling_usd"), raw.get("assumed_cost_per_run_usd"))
        states: dict[str, CallState] = {}
        if not isinstance(raw.get("calls"), list):
            raise TypeError("spend ledger calls must be a list")
        for row in raw["calls"]:
            if not isinstance(row, dict) or not isinstance(row.get("call"), dict):
                raise TypeError("spend call must be an object")
            call_raw = row["call"]
            if call_raw.get("kind") != "answer":
                raise ValueError("unsupported spend call kind")
            call = AnswerCall(call_raw.get("task_sha256"), RunCoordinate.from_row(call_raw.get("coordinate")))
            if call.call_id in states or row.get("call_id") != call.call_id:
                raise ValueError("duplicate or mismatched spend call identity")
            state: CallState
            if row.get("state") == "planned":
                state = Planned(call)
            elif row.get("state") == "in_flight":
                state = InFlight(call)
            elif row.get("state") == "not_started":
                state = Refused(call, SpendStopReason(row.get("reason")))
            elif row.get("state") == "settled":
                charge_raw = row.get("charge")
                if not isinstance(charge_raw, dict):
                    raise ValueError("settled call requires a charge")
                subtotal_raw = charge_raw.get("observed_subtotal_usd")
                subtotal = None if subtotal_raw is None else Money.from_raw(subtotal_raw)
                charge: Charge
                basis = charge_raw.get("basis")
                if basis == "observed":
                    charge = ObservedCharge(Money.from_raw(charge_raw.get("amount_usd")), charge_raw.get("provenance"))
                elif basis == "assumed":
                    charge = AssumedCharge(Money.from_raw(charge_raw.get("amount_usd")), charge_raw.get("reason"), subtotal)
                elif basis == "unpriced":
                    charge = UnpricedCall(charge_raw.get("reason"), subtotal)
                elif basis == "no_model_spend":
                    charge = NoModelSpend(charge_raw.get("reason"))
                else:
                    raise ValueError("unknown spend charge basis")
                state = Settled(call, charge)
            else:
                raise ValueError("unknown spend call state")
            states[call.call_id] = state
        ledger = cls(raw.get("invocation_id"), policy, states)
        if ledger.as_dict() != raw:
            raise ValueError("spend ledger contradicts its canonical records")
        return ledger
