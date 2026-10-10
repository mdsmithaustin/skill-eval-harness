import json
import tempfile
import unittest
from pathlib import Path

from manifest_contracts import RunCoordinate
from spend_contracts import (
    AnswerCall,
    AssumedCharge,
    InFlight,
    NoModelSpend,
    ObservedCharge,
    Planned,
    Refused,
    Settled,
    SpendLedger,
    SpendPlan,
    SpendPolicy,
    SpendStopReason,
    SubagentTurnCall,
    UnpricedCall,
    price_measurement,
)
from spend_runtime import NotStarted, Priced, SpendAdmission
from telemetry import Measurement, Money


class SpendContractTests(unittest.TestCase):
    def call(self, run=1):
        return AnswerCall("sha256:" + "a" * 64, RunCoordinate.of("case-1", "with_skill", run))

    def test_money_rejects_invalid_operator_values(self):
        for value in (True, False, -1, "NaN", "Infinity", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpendPolicy.from_raw(value)
            with self.subTest(assumption=value), self.assertRaises(ValueError):
                SpendPolicy.from_raw("2", value)
        with self.assertRaisesRegex(ValueError, "USD"):
            SpendPolicy(Money.from_raw("2", "EUR"))

    def test_unknown_subtotal_is_retained_and_assumption_cannot_reduce_it(self):
        cost = Measurement.unavailable("partial_provider_cost")
        subtotal = Money.from_raw("0.75")
        unknown = price_measurement(SpendPolicy.from_raw("10"), cost, observed_subtotal=subtotal)
        self.assertEqual(unknown, UnpricedCall("partial_provider_cost", subtotal))
        assumed = price_measurement(SpendPolicy.from_raw("10", "0.10"), cost,
                                    observed_subtotal=subtotal)
        self.assertEqual(assumed, AssumedCharge(subtotal, "partial_provider_cost", subtotal))
        with self.assertRaisesRegex(ValueError, "below"):
            AssumedCharge(Money.from_raw("0.1"), "missing", subtotal)
        call = self.call()
        ledger = SpendLedger.planned("a" * 32, SpendPolicy.from_raw("10"), SpendPlan((call,)))
        ledger = ledger.transition(InFlight(call)).transition(Settled(call, unknown))
        self.assertEqual(ledger.as_dict()["spent_usd"], "0.75")
        self.assertEqual(ledger.refusal, SpendStopReason.COST_UNOBSERVABLE)

    def test_zero_observed_and_proven_nonbillable_are_distinct_from_unknown(self):
        policy = SpendPolicy.from_raw("1")
        zero = price_measurement(policy, Measurement.available(Money.from_raw("0"), provenance="provider_reported"))
        self.assertEqual(zero, ObservedCharge(Money.from_raw("0"), "provider_reported"))
        no_spend = NoModelSpend("offline_adapter")
        self.assertEqual(price_measurement(policy, Measurement.not_applicable("offline"),
                                           no_model_spend=no_spend), no_spend)
        self.assertEqual(price_measurement(policy, Measurement.not_applicable("unproven")),
                         UnpricedCall("unproven"))
        with self.assertRaisesRegex(ValueError, "observed dollars"):
            price_measurement(policy, Measurement.available(Money.from_raw("1"), provenance="provider_reported"),
                              no_model_spend=no_spend)
        with self.assertRaisesRegex(ValueError, "observed dollars"):
            Priced("invalid", Measurement.available(Money.from_raw("1"), provenance="provider_reported"),
                   no_model_spend=no_spend)

    def test_ledger_transitions_freeze_inputs_and_reject_double_charges(self):
        call = self.call()
        states = {call.call_id: Planned(call)}
        ledger = SpendLedger("b" * 32, SpendPolicy.from_raw("2"), states)
        states.clear()
        self.assertEqual(tuple(ledger.states), (call.call_id,))
        with self.assertRaises(TypeError):
            ledger.states[call.call_id] = Planned(call)
        settled = Settled(call, ObservedCharge(Money.from_raw("1"), "provider_reported"))
        with self.assertRaisesRegex(TypeError, "transition"):
            ledger.transition(settled)
        admitted = ledger.transition(InFlight(call))
        result = admitted.transition(settled)
        self.assertEqual(result.as_dict()["spent_usd"], "1")
        self.assertIs(result.transition(settled), result)
        with self.assertRaisesRegex(TypeError, "transition"):
            result.transition(Settled(call, ObservedCharge(Money.from_raw("2"), "provider_reported")))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SpendPlan((call, call))

    def test_snapshot_round_trip_retains_all_states_and_checks_derived_fields(self):
        calls = tuple(self.call(run) for run in range(1, 5))
        ledger = SpendLedger.planned("c" * 32, SpendPolicy.from_raw("2"), SpendPlan(calls))
        ledger = ledger.transition(InFlight(calls[0]))
        ledger = ledger.transition(Settled(calls[0], UnpricedCall("missing")))
        ledger = ledger.transition(Refused(calls[1], SpendStopReason.COST_UNOBSERVABLE))
        ledger = ledger.transition(InFlight(calls[2]))
        raw = ledger.as_dict()
        self.assertEqual(SpendLedger.from_dict(json.loads(json.dumps(raw))), ledger)
        for key, value in (("spent_usd", "5"), ("spent_availability", "complete"),
                           ("schema_version", True), ("extra", "value")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                SpendLedger.from_dict({**raw, key: value})

    def test_runtime_settles_raised_baseexception_before_propagating(self):
        call = self.call()
        with tempfile.TemporaryDirectory() as td:
            with SpendAdmission.open(SpendPolicy.from_raw("1"), SpendPlan((call, self.call(2))), root=Path(td)) as admission:
                def interrupt():
                    raw = json.loads(admission.path.read_text())
                    self.assertEqual(raw["calls"][0]["state"], "in_flight")
                    raise KeyboardInterrupt("original interruption")
                with self.assertRaisesRegex(KeyboardInterrupt, "original interruption"):
                    admission.run(call, interrupt)
                result = admission.run(self.call(2), lambda: self.fail("refused callback ran"))
                self.assertEqual(result, NotStarted(self.call(2), SpendStopReason.COST_UNOBSERVABLE))
                raw = json.loads(admission.path.read_text())
                self.assertEqual(raw["calls"][0]["charge"]["basis"], "unpriced")
                self.assertEqual(raw["calls"][1]["state"], "not_started")

    def test_uncapped_executes_without_creating_storage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "spend"
            with SpendAdmission.open(None, SpendPlan((self.call(),)), root=root) as admission:
                result = admission.run(self.call(), lambda: Priced("answer", Measurement.unavailable("missing")))
                self.assertEqual(result.value, "answer")
            self.assertFalse(root.exists())

    def test_subagent_turn_identity_round_trips_without_changing_answer_bytes(self):
        answer = self.call()
        self.assertEqual(json.dumps(answer.as_dict(), sort_keys=True, separators=(",", ":")),
                         '{"coordinate":{"case_id":"case-1","run_number":1,"variant":"with_skill"},'
                         '"kind":"answer","task_sha256":"sha256:' + "a" * 64 + '"}')
        self.assertEqual(answer.call_id, "0400b704dc143723721d3e73986247cb739228e2ca0e7edb54ec684aed1cb39e")
        call = SubagentTurnCall(answer.task_sha256, answer.coordinate, 2)
        self.assertEqual(call.as_dict(), {"kind": "subagent_turn", "task_sha256": "sha256:" + "a" * 64,
                                        "coordinate": {"case_id": "case-1", "variant": "with_skill", "run_number": 1},
                                        "external_turn": 2})
        ledger = SpendLedger.planned("a" * 32, SpendPolicy.from_raw("1"), SpendPlan((answer, call)))
        ledger = ledger.transition(InFlight(call)).transition(Settled(call, ObservedCharge(Money.from_raw("0.2"), "provider_reported")))
        self.assertEqual(SpendLedger.from_dict(ledger.as_dict()).as_dict(), ledger.as_dict())
        for value in (0, -1, True, False, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                SubagentTurnCall(answer.task_sha256, answer.coordinate, value)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SpendPlan((call, call))

    def test_subagent_ledger_rejects_unknown_kind_invalid_turn_and_contradictory_id(self):
        answer = self.call()
        call = SubagentTurnCall(answer.task_sha256, answer.coordinate, 1)
        ledger = SpendLedger.planned("a" * 32, SpendPolicy.from_raw("1"), SpendPlan((call,))).as_dict()
        for field, value, message in (("kind", "subagent", "unsupported spend call kind"),
                                      ("external_turn", True, "positive integer"),
                                      ("external_turn", 0, "positive integer")):
            raw = json.loads(json.dumps(ledger))
            raw["calls"][0]["call"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                SpendLedger.from_dict(raw)
        raw = json.loads(json.dumps(ledger))
        raw["calls"][0]["call_id"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "mismatched spend call identity"):
            SpendLedger.from_dict(raw)
