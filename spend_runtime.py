from __future__ import annotations

import errno
import json
import os
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from spend_contracts import (
    SPEND_LEDGER_NAME,
    InFlight,
    MatrixCall,
    MatrixPopulation,
    NoModelSpend,
    Refused,
    Settled,
    SpendCall,
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
    interruption: BaseException | None = None

    def __post_init__(self) -> None:
        if self.no_model_spend is not None and (
                self.cost.availability == AVAILABLE or self.observed_subtotal is not None):
            raise ValueError("nonbillable call cannot carry observed dollars")


@dataclass(frozen=True)
class Started(Generic[T]):
    call: SpendCall
    value: T


@dataclass(frozen=True)
class NotStarted:
    call: SpendCall
    reason: SpendStopReason


@dataclass(frozen=True)
class WorkerRaised:
    call: MatrixCall
    error: BaseException


class MatrixStopReason(str, Enum):
    INTERRUPTED = "interrupted"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True)
class MatrixNotStarted:
    call: MatrixCall
    reason: MatrixStopReason


class BoundedRunError(BaseException):
    def __init__(self, cause: BaseException,
                 executions: tuple[Started[Any] | NotStarted | MatrixNotStarted | WorkerRaised, ...]) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.executions = executions


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
        except OSError as exc:
            if (self.ledger.scope == "paid_matrix_calls" and os.name != "nt"
                    and exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}):
                raise
            return
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if (self.ledger.scope == "paid_matrix_calls" and os.name != "nt"
                    and exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS}):
                raise
        finally:
            os.close(descriptor)

    def run(self, call: SpendCall, invoke: Callable[[], Priced[T]]) -> Started[T] | NotStarted:
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


    def bounded_map(self, population: MatrixPopulation,
                    invoke: Callable[[MatrixCall], Priced[T]], *, workers: int,
                    publish: Callable[[T], T] | None = None,
                    stopped: Callable[[], bool] = lambda: False,
                    ) -> tuple[Started[T] | NotStarted | MatrixNotStarted | WorkerRaised, ...]:
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be positive")
        if self.ledger is not None and tuple(self.ledger.states) != tuple(
                call.call_id for call in population.paid_plan.calls):
            raise ValueError("matrix ledger must match the derived paid plan")
        results: dict[int, Started[T] | NotStarted | MatrixNotStarted | WorkerRaised] = {}
        active: dict[int, Future[Priced[T]]] = {}
        cursor = 0
        fatal: BaseException | None = None
        executor = ThreadPoolExecutor(max_workers=workers)

        def settle_ready(*, drain: bool = False) -> None:
            nonlocal fatal
            while True:
                ready = [index for index, future in active.items() if drain or future.done()]
                if not ready:
                    return
                retained: list[tuple[int, T]] = []
                for index in sorted(ready):
                    cell = population.cells[index]
                    future = active.pop(index)
                    try:
                        priced = future.result()
                    except Exception as exc:  # noqa: BLE001
                        results[index] = WorkerRaised(cell.call, exc)
                        priced = Priced(None, Measurement.unavailable("invocation_raised"))
                    except BaseException as exc:  # noqa: BLE001
                        fatal = fatal or exc
                        results[index] = WorkerRaised(cell.call, exc)
                        continue
                    else:
                        fatal = fatal or priced.interruption
                        results[index] = Started(cell.call, priced.value)
                        retained.append((index, priced.value))
                    if self.ledger is not None and not cell.offline:
                        charge = price_measurement(self.ledger.policy, priced.cost,
                                                   observed_subtotal=priced.observed_subtotal,
                                                   no_model_spend=priced.no_model_spend)
                        self.ledger = self.ledger.transition(Settled(cell.call, charge))
                if fatal is None:
                    self._persist()
                    if publish is not None:
                        for index, value in retained:
                            results[index] = Started(population.cells[index].call, publish(value))
                if drain:
                    return

        try:
            while cursor < len(population.cells) or active:
                if stopped():
                    raise KeyboardInterrupt
                settle_ready()
                if fatal is not None:
                    raise fatal
                while cursor < len(population.cells) and len(active) < workers:
                    settle_ready()
                    if stopped():
                        raise KeyboardInterrupt
                    if fatal is not None:
                        raise fatal
                    cell = population.cells[cursor]
                    refusal = (self.ledger.settled_refusal
                               if self.ledger is not None and not cell.offline else None)
                    if refusal is not None and self.ledger is not None:
                        self.ledger = self.ledger.transition(Refused(cell.call, refusal))
                        self._persist()
                        results[cursor] = NotStarted(cell.call, refusal)
                        cursor += 1
                        continue
                    if self.ledger is not None and not cell.offline:
                        if any(isinstance(state, InFlight) and state.call.call_id not in {
                                population.cells[index].call.call_id for index in active}
                               for state in self.ledger.states.values()):
                            raise RuntimeError("admitted matrix call has no active handle")
                        self.ledger = self.ledger.transition(InFlight(cell.call))
                        self._persist()
                    try:
                        active[cursor] = executor.submit(invoke, cell.call)
                    except BaseException as exc:
                        results[cursor] = WorkerRaised(cell.call, exc)
                        raise
                    cursor += 1
                if active:
                    wait(tuple(active.values()), timeout=0.05, return_when=FIRST_COMPLETED)
        except BaseException as exc:  # noqa: BLE001
            fatal = fatal or exc
            try:
                settle_ready(drain=True)
                if isinstance(fatal, KeyboardInterrupt):
                    self._persist()
            except BaseException as drain_error:  # noqa: BLE001
                exc.__context__ = drain_error
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if fatal is not None:
            reason = (MatrixStopReason.INTERRUPTED if isinstance(fatal, KeyboardInterrupt)
                      else MatrixStopReason.INFRASTRUCTURE_FAILED)
            for index, cell in enumerate(population.cells):
                results.setdefault(index, MatrixNotStarted(cell.call, reason))
            raise BoundedRunError(fatal, tuple(results[index] for index in range(len(population.cells))))
        return tuple(results[index] for index in range(len(population.cells)))


def spend_invocations(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob(f"*/{SPEND_LEDGER_NAME}")):
        ledger = SpendLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))
        records.append({"ledger_path": str(path), **ledger.as_dict()})
    return records
