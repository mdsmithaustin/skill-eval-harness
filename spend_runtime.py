from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from spend_contracts import (
    SPEND_LEDGER_NAME,
    AnswerCall,
    InFlight,
    NoModelSpend,
    Refused,
    Settled,
    SpendLedger,
    SpendPlan,
    SpendPolicy,
    SpendStopReason,
    price_measurement,
)
from telemetry import AVAILABLE, Measurement, Money

T = TypeVar("T")


@dataclass(frozen=True)
class Priced(Generic[T]):
    value: T
    cost: Measurement[Money]
    observed_subtotal: Money | None = None
    no_model_spend: NoModelSpend | None = None

    def __post_init__(self) -> None:
        if self.no_model_spend is not None and (
                self.cost.availability == AVAILABLE or self.observed_subtotal is not None):
            raise ValueError("nonbillable call cannot carry observed dollars")


@dataclass(frozen=True)
class Started(Generic[T]):
    call: AnswerCall
    value: T


@dataclass(frozen=True)
class NotStarted:
    call: AnswerCall
    reason: SpendStopReason


class SpendAdmission:
    def __init__(self, ledger: SpendLedger | None, path: Path | None) -> None:
        self.ledger = ledger
        self.path = path

    @classmethod
    @contextmanager
    def open(cls, policy: SpendPolicy | None, plan: SpendPlan, *, root: Path) -> Iterator[SpendAdmission]:
        if policy is None:
            yield cls(None, None)
            return
        invocation_id = uuid.uuid4().hex
        directory = root / invocation_id
        directory.mkdir(parents=True, exist_ok=False)
        admission = cls(SpendLedger.planned(invocation_id, policy, plan), directory / SPEND_LEDGER_NAME)
        admission._persist()
        yield admission

    def _persist(self) -> None:
        if self.ledger is None or self.path is None:
            return
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.ledger.as_dict(), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def run(self, call: AnswerCall, invoke: Callable[[], Priced[T]]) -> Started[T] | NotStarted:
        if self.ledger is None:
            return Started(call, invoke().value)
        refusal = self.ledger.refusal
        if refusal is not None:
            self.ledger = self.ledger.transition(Refused(call, refusal))
            self._persist()
            return NotStarted(call, refusal)
        self.ledger = self.ledger.transition(InFlight(call))
        self._persist()
        try:
            priced = invoke()
            charge = price_measurement(self.ledger.policy, priced.cost,
                                       observed_subtotal=priced.observed_subtotal,
                                       no_model_spend=priced.no_model_spend)
        except BaseException:
            self.ledger = self.ledger.transition(Settled(call, price_measurement(
                self.ledger.policy, Measurement.unavailable("invocation_raised"))))
            self._persist()
            raise
        self.ledger = self.ledger.transition(Settled(call, charge))
        self._persist()
        return Started(call, priced.value)


def spend_invocations(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob(f"*/{SPEND_LEDGER_NAME}")):
        ledger = SpendLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))
        records.append({"ledger_path": str(path), **ledger.as_dict()})
    return records
