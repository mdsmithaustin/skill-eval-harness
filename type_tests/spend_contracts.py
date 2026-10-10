from collections.abc import Callable
from typing import Literal, NoReturn

from typing_extensions import assert_type

from spend_contracts import (
    AnswerCall,
    AssumedCharge,
    CallState,
    Charge,
    InFlight,
    NoModelSpend,
    ObservedCharge,
    Planned,
    Refused,
    Settled,
    UnpricedCall,
)
from spend_runtime import NotStarted, Priced, SpendAdmission, Started
from telemetry import Measurement, Money


def unreachable(value: NoReturn) -> NoReturn:
    raise AssertionError(value)


def charge_precision(charge: Charge) -> None:
    if isinstance(charge, (ObservedCharge, AssumedCharge)):
        assert_type(charge.amount, Money)
    elif isinstance(charge, UnpricedCall):
        assert_type(charge.observed_subtotal, Money | None)
    elif isinstance(charge, NoModelSpend):
        assert_type(charge.reason, Literal["offline_adapter", "spawn_failed_before_process"])
    else:
        unreachable(charge)


def state_precision(state: CallState) -> None:
    assert_type(state.call, AnswerCall)
    if isinstance(state, (Planned, InFlight, Refused)):
        return
    if isinstance(state, Settled):
        assert_type(state.charge, Charge)
        return
    unreachable(state)


def admission_precision(admission: SpendAdmission, call: AnswerCall,
                        invoke: Callable[[], Priced[str]]) -> None:
    result = admission.run(call, invoke)
    assert_type(result, Started[str] | NotStarted)
    if isinstance(result, NotStarted):
        assert_type(result.call, AnswerCall)
    else:
        assert_type(result.value, str)
    assert_type(invoke().cost, Measurement[Money])
