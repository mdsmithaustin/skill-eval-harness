"""Lift intervals, the noise check, and the floor/ceiling split.

The interval is the sign-flip test inverted, so it must exclude zero exactly
when the exact test rejects "no lift". The noise check must say when an eval
could not have shown a lift at all. A case both arms fail must never be sent
to suggest-cases for hardening."""
import bisect
import collections
import itertools
import json
import math
import random
import statistics
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from helpers import (
    attest_answer_design,
    demo_manifest,
    judge_with_stub,
    make_eval_repo,
    result_row,
    run_cli,
    write_demo_manifest,
    write_run,
)

import effect_estimates as ee
import skill_benchmark as sb
from findings import CaseFlag


def thirds(values: list[int]) -> list[float]:
    """Per-case deltas at three repeats per arm, the blog's customer-support shape."""
    return [value / 3 for value in values]


def grouped_p(deltas: list[float], shift: float) -> float:
    """The two-sided sign-flip p-value of ``d - shift``, counted over how many
    of each distinct delta are flipped (zeros included), each count weighted
    by its binomial coefficient: no pattern sums are grouped or sorted."""
    counts = sorted(collections.Counter(deltas).items())
    n = len(deltas)
    observed = abs(math.fsum(deltas) - n * shift) - 1e-12
    hits = 0
    for flips in itertools.product(*(range(count + 1) for _, count in counts)):
        statistic = math.fsum((value - shift) * (count - 2 * j)
                              for (value, count), j in zip(counts, flips))
        if abs(statistic) >= observed:
            hits += math.prod(math.comb(count, j) for (_, count), j in zip(counts, flips))
    return hits / 2 ** n


def brute_force_p(deltas: list[float]):
    """The two-sided sign-flip p-value of ``d - shift`` by visiting all 2**n
    sign patterns one at a time: the definition, with no grouping."""
    n = len(deltas)
    sums = [(math.fsum(s * d for s, d in zip(signs, deltas)), sum(signs))
            for signs in itertools.product((1, -1), repeat=n)]

    def p(shift: float) -> float:
        observed = abs(math.fsum(deltas) - n * shift)
        return sum(1 for a, b in sums if abs(a - shift * b) >= observed - 1e-12) / 2 ** n
    return p


class IntervalAgreesWithTheTestTests(unittest.TestCase):
    def test_interval_excludes_zero_exactly_when_the_exact_test_rejects(self):
        rng = random.Random(7)
        choices = [-3, -2, -1, 0, 0, 1, 2, 3]
        checked = 0
        for n in range(6, 13):
            for _ in range(25):
                deltas = thirds([rng.choice(choices) for _ in range(n)])
                if all(abs(value) < 1e-12 for value in deltas):
                    continue
                interval = ee.sign_flip_interval(deltas)
                significance = sb.sign_flip_significance(deltas)
                with self.subTest(deltas=deltas):
                    if not interval["bounded"]:
                        self.assertFalse(significance["significant_at_0_05"])
                        continue
                    zero_inside = interval["lower"] <= 0 <= interval["upper"]
                    self.assertEqual(zero_inside, not significance["significant_at_0_05"])
                    checked += 1
        self.assertGreater(checked, 100)

    def test_sampled_interval_agrees_with_the_sampled_test(self):
        # Past the exact budget both sample sign patterns. The test gates on a
        # conservative upper bound, so an interval that gated on the point
        # estimate excluded zero where the test reported no significance.
        # These deltas repeat, which keeps them exact by default, so the
        # budget is set to 2**0 to put every one on the sampled path.
        rng = random.Random(11)
        choices = [-3, -2, -1, 0, 0, 1, 2, 3]
        checked = 0
        for n in (15, 18, 22, 30):
            for _ in range(40):
                deltas = thirds([rng.choice(choices) for _ in range(n)])
                if all(abs(value) < 1e-12 for value in deltas):
                    continue
                interval = ee.sign_flip_interval(deltas, max_exact_n=0)
                significance = sb.sign_flip_significance(deltas, max_exact_n=0)
                self.assertEqual(interval["method"], "sign-flip-inversion-sampled")
                if not interval["bounded"]:
                    continue
                with self.subTest(deltas=deltas):
                    zero_inside = interval["lower"] <= 0 <= interval["upper"]
                    self.assertEqual(zero_inside, not significance["significant_at_0_05"])
                    checked += 1
        self.assertGreater(checked, 100)

    def test_grouped_enumeration_matches_every_sign_pattern(self):
        # Equal deltas are counted as one binomial group and zeros as a spread
        # over the pattern weight, so evals with repeats or unchanged cases
        # stay exact past 14 units. The p-value at the shifts the interval
        # inverts, and the interval itself, must match visiting all 2**n
        # patterns; the accepted shifts must form one interval that excludes
        # zero exactly when the test rejects.
        inputs = [
            [1.0] * 6 + [0.0] * 8,
            [1.0] * 6 + [0.0] * 9,
            thirds([1, 1, 2, 0, 0, -1, 1, 2, 0, 1, 1, -2, 0, 1]),
            [0.5, 0.5, -0.5, 1.0, 1.0, 1.0, 0.0, 0.25, 0.25, -1.0, 0.5, 0.0],
            thirds([3, 2, 1, -1, -2, 0, 1, 2, 3, 3]),
        ]
        for deltas in inputs:
            p = brute_force_p(deltas)
            interval = ee.sign_flip_interval(deltas)
            significance = ee.sign_flip_test(deltas)
            with self.subTest(deltas=deltas):
                self.assertTrue(interval["method"].endswith("exact"))
                self.assertTrue(interval["bounded"])
                lower, upper = interval["lower"], interval["upper"]
                shifts = sorted({0.0, statistics.fmean(deltas), *deltas,
                                 lower - 1e-5, lower + 1e-5, upper - 1e-5, upper + 1e-5})
                for shift in shifts:
                    shifted = [d - shift for d in deltas]
                    self.assertAlmostEqual(ee.sign_flip_test(shifted)["p_value"], p(shift),
                                           places=12, msg=f"shift={shift}")
                self.assertGreater(p(lower + 1e-5), 0.05)
                self.assertGreater(p(upper - 1e-5), 0.05)
                self.assertLessEqual(p(lower - 1e-5), 0.05)
                self.assertLessEqual(p(upper + 1e-5), 0.05)
                grid = [min(deltas) - 0.5 + step * 0.05
                        for step in range(int((max(deltas) - min(deltas) + 1) / 0.05) + 1)]
                accepted = [shift for shift in grid if p(shift) > 0.05]
                self.assertEqual(accepted, [s for s in grid if accepted[0] <= s <= accepted[-1]])
                self.assertTrue(lower - 1e-5 <= accepted[0] and accepted[-1] <= upper + 1e-5)
                self.assertEqual(not lower <= 0 <= upper, significance["significant_at_0_05"])
                self.assertEqual(significance["p_value"], p(0.0))

    def test_interval_contains_the_observed_mean(self):
        deltas = thirds([1, 2, 0, 3, 1, -1, 2, 1])
        interval = ee.sign_flip_interval(deltas)
        mean = sum(deltas) / len(deltas)
        self.assertTrue(interval["bounded"])
        self.assertLess(interval["lower"], mean)
        self.assertGreater(interval["upper"], mean)

    def test_too_few_cases_cannot_bound_anything(self):
        # 2 / 2**5 = 0.0625 > 0.05: five cases can never exclude any shift.
        interval = ee.sign_flip_interval([0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertFalse(interval["bounded"])
        self.assertIsNone(interval["lower"])
        self.assertIn("at least 6", interval["reason"])
        # 2 / 2**6 = 0.03125 <= 0.05: the sixth case makes a bound possible
        # (one that differs, since a constant sample cannot be bounded).
        self.assertTrue(ee.sign_flip_interval([0.5] * 5 + [0.25])["bounded"])

    def test_sampled_path_is_deterministic_and_order_invariant(self):
        rng = random.Random(3)
        deltas = [rng.choice([-1.0, 0.0, 0.5, 1.0]) for _ in range(30)]
        first = ee.sign_flip_interval(deltas, max_exact_n=0)
        ee._patterns_of.cache_clear()   # as in a fresh re-grade
        second = ee.sign_flip_interval(list(reversed(deltas)), max_exact_n=0)
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "sign-flip-inversion-sampled")

    def test_empty_and_non_finite_input(self):
        self.assertFalse(ee.sign_flip_interval([])["bounded"])
        with self.assertRaises(ValueError):
            ee.sign_flip_interval([float("nan"), 1.0])

    def test_a_300_case_exact_interval_takes_few_evaluations(self):
        # 300 cases: 120 gained a run, 60 lost one, 120 did not move. Equal
        # deltas group and unchanged cases only spread the pattern weight, so
        # the path is exact, but each p-value evaluation visited every (pattern
        # weight, unchanged-case count) pair, and the search bisected 60 times
        # a side: 122 evaluations and about 1.5 s for one interval.
        deltas = [1.0] * 120 + [-1.0] * 60 + [0.0] * 120
        # The deterministic guards: how many evaluations, and how many binary
        # searches each takes (one per unchanged-case count, not one per
        # pattern weight as well).
        with mock.patch.object(ee, "_tail", wraps=ee._tail) as tail, \
                mock.patch.object(bisect, "bisect_left", wraps=bisect.bisect_left) as search:
            ee.sign_flip_interval(deltas)
        self.assertLessEqual(tail.call_count, 60)
        self.assertLessEqual(search.call_count, tail.call_count * (120 + 1))
        started = time.perf_counter()
        interval = ee.sign_flip_interval(deltas)
        elapsed = time.perf_counter() - started
        # The endpoints the 60-step search recorded before the change.
        self.assertEqual((interval["method"], interval["lower"], interval["upper"]),
                         ("sign-flip-inversion-exact", 0.114865, 0.284848))
        self.assertLess(elapsed, 0.5)   # about 0.1 s here; generous, not a tight timer

    def test_unchanged_cases_keep_the_interval_exact(self):
        # An independent count over how many +1, -1 and 0 deltas flip agrees
        # with the interval on both sides of each endpoint.
        deltas = [1.0] * 14 + [-1.0] * 6 + [0.0] * 10
        interval = ee.sign_flip_interval(deltas)
        self.assertEqual(interval["method"], "sign-flip-inversion-exact")
        for end, inside in ((interval["lower"], 1e-5), (interval["upper"], -1e-5)):
            with self.subTest(end=end):
                self.assertGreater(grouped_p(deltas, end + inside), 0.05)
                self.assertLessEqual(grouped_p(deltas, end - inside), 0.05)

    def test_pass_rate_deltas_past_the_outcome_budget_stay_exact(self):
        # 29 cases at three repeats: six distinct non-zero deltas, 5 or 4 of
        # each, take 6*5*6*5*6*5 = 27000 > 2**14 flip-count outcomes, so the
        # test and the interval sampled. As thirds they are whole numbers of
        # runs, and their pattern sums take far fewer values, so both are
        # exact: the p-value and the interval's endpoints match a count over
        # every combination of flips.
        deltas = thirds([1] * 5 + [-1] * 4 + [2] * 5 + [-2] * 4 + [3] * 5 + [-3] * 4 + [0] * 2)
        significance = ee.sign_flip_test(deltas)
        interval = ee.sign_flip_interval(deltas)
        self.assertEqual((significance["method"], interval["method"]),
                         ("sign-flip-exact", "sign-flip-inversion-exact"))
        self.assertAlmostEqual(significance["p_value"], grouped_p(deltas, 0.0), places=12)
        self.assertTrue(interval["bounded"])
        for end, inside in ((interval["lower"], 1e-5), (interval["upper"], -1e-5)):
            with self.subTest(end=end):
                self.assertGreater(grouped_p(deltas, end + inside), 0.05)
                self.assertLessEqual(grouped_p(deltas, end - inside), 0.05)

    # How far an exact p-value may sit from alpha before the sampled decision
    # must match it: past 2**18 patterns the bound is about 0.002 above p.
    AMBIGUOUS = 0.004

    def test_the_sampled_test_decides_like_brute_force_below_the_old_floor(self):
        # Graded-score deltas take arbitrary values, so they sample. The
        # sampled test decided on a Hoeffding bound that never fell below
        # about 0.03, so it could not reject at alpha 0.01 even when the
        # exact p was 0.0001. On 84 evals of 8-14 cases, forced onto the
        # sampled path, it must decide as visiting every pattern does at
        # 0.05, 0.01 and 0.005, wherever the exact p is clear of alpha.
        rng = random.Random(23)
        checked = 0
        for n in range(8, 15):
            for _ in range(12):
                deltas = [round(rng.uniform(-0.4, 0.9), 4) for _ in range(n)]
                exact = brute_force_p(deltas)(0.0)
                for alpha in (0.05, 0.01, 0.005):
                    if abs(exact - alpha) < self.AMBIGUOUS:
                        continue
                    sampled = ee.sign_flip_test(deltas, max_exact_n=0, alpha=alpha)
                    with self.subTest(deltas=deltas, alpha=alpha, exact=exact):
                        self.assertEqual(sampled["method"], "sign-flip-sampled")
                        self.assertEqual(sampled["significant_at_0_05"], exact <= alpha)
                        self.assertGreaterEqual(sampled["p_value_upper_bound"], exact)
                    checked += 1
        self.assertGreater(checked, 200)
        # Twenty equal moves reach exact p = 2 / 2**20; the sampled bound
        # used to stop at 0.029.
        self.assertLess(ee.sign_flip_test([0.5] * 20, max_exact_n=0)["p_value_upper_bound"],
                        0.003)

    def test_the_sampled_path_agrees_with_the_exact_path_where_both_apply(self):
        # 15-30 cases at three repeats are exact; sampled instead, their
        # decision at 0.05 and 0.01 must match the exact one wherever the
        # exact p is clear of alpha, and the sampled interval, which rejects
        # a shift only when its upper bound clears alpha, must contain the
        # exact one.
        rng = random.Random(29)
        decided = contained = 0
        for _ in range(40):
            deltas = thirds([rng.choice([-3, -2, -1, 0, 1, 1, 2, 3]) for _ in range(rng.randint(15, 30))])
            for alpha in (0.05, 0.01):
                exact = ee.sign_flip_test(deltas, alpha=alpha)
                sampled = ee.sign_flip_test(deltas, max_exact_n=0, alpha=alpha)
                self.assertEqual((exact["method"], sampled["method"]),
                                 ("sign-flip-exact", "sign-flip-sampled"))
                if abs(exact["p_value"] - alpha) >= self.AMBIGUOUS:
                    with self.subTest(deltas=deltas, alpha=alpha):
                        self.assertEqual(sampled["significant_at_0_05"],
                                         exact["significant_at_0_05"])
                    decided += 1
            exact_interval = ee.sign_flip_interval(deltas)
            sampled_interval = ee.sign_flip_interval(deltas, max_exact_n=0)
            if exact_interval["bounded"] and sampled_interval["bounded"]:
                with self.subTest(deltas=deltas):
                    self.assertLessEqual(sampled_interval["lower"], exact_interval["lower"])
                    self.assertGreaterEqual(sampled_interval["upper"], exact_interval["upper"])
                contained += 1
        self.assertGreater(decided, 60)
        self.assertGreater(contained, 25)


class NoiseCheckTests(unittest.TestCase):
    def check(self, deltas, without, **options):
        return ee.noise_check(deltas, without, interval=ee.sign_flip_interval(deltas), **options)

    def test_blog_held_out_split_cannot_reach_significance(self):
        # 14 held-out tickets, 3 repeats each, net +5 runs: at best five
        # tickets moved one run each, so p can never go below 2/2**5.
        deltas = thirds([1, 1, 1, 1, 1] + [0] * 9)
        result = self.check(deltas, [0.79] * 14)
        self.assertEqual(result["verdict"], "too-few-cases-moved")
        self.assertEqual(result["cases_moved"], 5)
        self.assertEqual(result["smallest_achievable_p"], 0.0625)
        self.assertEqual(result["cases_needed_for_alpha"], 6)

    def test_noise_larger_than_headroom(self):
        deltas = thirds([3, -3, 3, -3, 3, -3, 2, -2])
        result = self.check(deltas, [0.95] * len(deltas))
        self.assertEqual(result["verdict"], "noise-exceeds-headroom")
        self.assertIn("projected_cases", result)

    def test_min_lift_the_author_would_act_on(self):
        deltas = thirds([1, 2, 1, 0, 1, 2, 1, 1, 0, 1])
        result = self.check(deltas, [0.3] * len(deltas), min_lift=0.05)
        self.assertEqual(result["verdict"], "noise-exceeds-min-lift")
        self.assertEqual(result["min_lift"], 0.05)
        self.assertGreater(result["projected_cases"], len(deltas))

    def test_resolvable_when_the_floor_is_small(self):
        deltas = [0.5] * 11 + [0.25]
        result = self.check(deltas, [0.2] * 12, min_lift=0.3)
        self.assertEqual(result["verdict"], "resolvable")

    def test_smallest_achievable_p_and_cases_needed(self):
        self.assertEqual(ee.smallest_achievable_p(0), 1.0)
        self.assertEqual(ee.smallest_achievable_p(6), 0.03125)
        self.assertEqual(ee.cases_needed_for_alpha(0.05), 6)
        self.assertEqual(ee.cases_needed_for_alpha(0.01), 8)

    def test_six_cases_moving_together_are_enough(self):
        # The boundary next to the blog split: six cases that all improve reach
        # p = 2/2**6 = 0.03125 <= 0.05. Five gain all three runs and one gains
        # two: every shift outside [2/3, 1] leaves six same-sign deltas the
        # test rejects, so the noise floor is (1 - 2/3) / 2.
        result = self.check([1.0] * 5 + [2 / 3], [0.0] * 6)
        self.assertEqual(result["cases_moved"], 6)
        self.assertEqual(result["smallest_achievable_p"], 0.03125)
        self.assertAlmostEqual(result["noise_floor"], 1 / 6, places=5)
        self.assertEqual(result["verdict"], "resolvable")


    def test_a_constant_sample_cannot_be_bounded(self):
        # The sign-flip test reads only signs, so it rejects every shift but
        # the common value of a constant sample; the point [v, v] that
        # inverting it gives is no measure of how large the lift is. The
        # interval says so, and the noise check cannot call any --min-lift
        # resolvable, though the lift itself is significant.
        for deltas in ([1.0] * 6, [0.5] * 12, [0.0] * 7):
            with self.subTest(deltas=deltas):
                interval = ee.sign_flip_interval(deltas)
                self.assertFalse(interval["bounded"])
                self.assertEqual((interval["lower"], interval["upper"]), (None, None))
                self.assertIn("every paired delta is the same", interval["reason"])
        result = self.check([1.0] * 6, [0.0] * 6, min_lift=0.5)
        self.assertEqual(result["verdict"], "unbounded")
        self.assertIsNone(result["noise_floor"])
        self.assertIn("every paired delta is the same", result["reason"])
        self.assertTrue(ee.sign_flip_test([1.0] * 6)["significant_at_0_05"])

class HolmTests(unittest.TestCase):
    def test_textbook_example(self):
        # Sorted p 0.01, 0.02, 0.03, 0.04 with m = 4 step down to 0.04, 0.06, 0.06, 0.04:
        # the last is raised to 0.06 so the adjusted values never decrease.
        adjusted = ee.holm_adjust([0.04, 0.01, 0.03, 0.02])
        for got, want in zip(adjusted, [0.06, 0.04, 0.06, 0.06]):
            self.assertAlmostEqual(got, want)

    def test_adjusted_values_cap_at_one_and_keep_input_order(self):
        self.assertEqual(ee.holm_adjust([0.5, 0.9, 0.01]), [1.0, 1.0, 0.03])
        self.assertEqual(ee.holm_adjust([]), [])
        self.assertEqual(ee.holm_adjust([0.03]), [0.03])

    def test_p_values_outside_the_unit_interval_are_rejected(self):
        for values in ([1.2], [-0.1, 0.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ee.holm_adjust(values)

    def test_one_lucky_model_among_four_does_not_survive(self):
        # p = 0.03125 is six cases all improving. Alone it is significant;
        # as the best of four models it adjusts to 0.125.
        blocks = {"a": {"p_value": 0.03125, "p_value_upper_bound": 0.03125},
                  "b": {"p_value": 0.5, "p_value_upper_bound": 0.5},
                  "c": {"p_value": 1.0, "p_value_upper_bound": 1.0},
                  "d": {"p_value": 0.25, "p_value_upper_bound": 0.25}}
        result = ee.multiplicity(blocks, family="by_model")
        self.assertEqual(result["tests"], 4)
        self.assertEqual(result["by_test"]["a"]["adjusted_p_value"], 0.125)
        self.assertEqual(result["significant_after_adjustment"], [])

    def test_the_sampled_bound_decides_and_untested_blocks_do_not_count(self):
        blocks = {"sampled": {"p_value": 0.001, "p_value_upper_bound": 0.02},
                  "withheld": {"method": "unavailable", "p_value": None},
                  "exact": {"p_value": 0.0002}}
        result = ee.multiplicity(blocks, family="by_model")
        self.assertEqual((result["tests"], result["untested"]), (2, ["withheld"]))
        self.assertEqual(result["by_test"]["sampled"]["p_value"], 0.02)
        self.assertEqual(result["significant_after_adjustment"], ["exact", "sampled"])


PILOT = [((3, 3), (3, 1)), ((3, 2), (3, 2)), ((3, 3), (3, 0)), ((3, 1), (3, 1)),
         ((3, 2), (3, 0)), ((3, 3), (3, 3)), ((3, 0), (3, 1)), ((3, 2), (3, 1))]


class PlanningTests(unittest.TestCase):
    def test_lattice_p_value_is_the_exact_test_p_value(self):
        rng = random.Random(7)
        for _ in range(200):
            repetitions = rng.choice([1, 2, 3, 5])
            steps = [rng.randint(-repetitions, repetitions) for _ in range(rng.randint(1, 14))]
            exact = ee.sign_flip_test([step / repetitions for step in steps])
            with self.subTest(steps=steps):
                self.assertAlmostEqual(ee.lattice_sign_flip_p(steps), exact["p_value"], places=12)
        self.assertEqual(ee.lattice_sign_flip_p([]), 1.0)
        self.assertEqual(ee.lattice_sign_flip_p([0, 0]), 1.0)
        self.assertEqual(ee.lattice_sign_flip_p([1] * 6), 0.03125)

    def test_variance_split_separates_repetition_noise_from_case_spread(self):
        # Every case is a coin flip in both arms, so all the variance of the
        # deltas is repetition noise: 2 * (0.5 * 0.5) / (2 - 1) per case.
        noisy = ee.variance_split([((2, 1), (2, 1))] * 6)
        self.assertEqual(noisy["within_case_variance"], 0.5)
        self.assertEqual((noisy["between_case_variance"], noisy["within_case_share"]), (0.0, 1.0))
        # Deterministic arms that disagree by case: no repetition noise at all.
        spread = ee.variance_split([((3, 3), (3, 0)), ((3, 0), (3, 3))] * 3)
        self.assertEqual((spread["within_case_variance"], spread["within_case_share"]), (0.0, 0.0))
        self.assertGreater(spread["between_case_variance"], 1)

    def test_one_run_per_arm_cannot_split_the_variance(self):
        split = ee.variance_split([((1, 1), (1, 0)), ((1, 0), (1, 0)), ((1, 1), (1, 1))])
        self.assertIsNotNone(split["observed_delta_variance"])
        self.assertIsNone(split["within_case_variance"])
        self.assertIsNone(split["within_case_share"])

    def test_power_grows_with_cases_and_the_cheapest_sufficient_cell_is_recommended(self):
        plan = ee.plan_power(PILOT, min_lift=0.15, cases_grid=[8, 32, 128],
                             repetitions_grid=[3], trials=200)
        powers = [cell["power"] for cell in plan["grid"]]
        self.assertEqual(powers, sorted(powers))
        self.assertLess(powers[0], 0.2)
        self.assertGreater(powers[-1], 0.9)
        self.assertEqual(plan["verdict"], "plan-found")
        self.assertEqual(plan["recommendation"]["cases"], 128)
        self.assertEqual(plan["recommendation"]["runs_per_arm"], 384)

    def test_a_larger_target_lift_is_easier_to_show_and_a_loss_never_counts(self):
        def power(min_lift):
            return ee.plan_power(PILOT, min_lift=min_lift, cases_grid=[40], repetitions_grid=[3],
                                 trials=2000)["grid"][0]["power"]
        self.assertLess(power(0.1), 0.5)
        self.assertGreater(power(0.4), 0.99)
        # With almost no lift, half the significant evals point the wrong way.
        # Counting them would report about twice this share.
        self.assertLess(power(0.001), 0.028)

    def test_shrunk_rates_shed_repetition_noise_when_it_can_be_measured(self):
        # Three observed rates whose spread is no wider than four coin flips
        # would give: nothing separates the cases, so each gets the arm mean.
        self.assertEqual(set(ee._shrunk_rates([(4, 2)] * 3 + [(4, 3)] * 3 + [(4, 1)] * 2)), {0.53125})
        # Deterministic cases keep their rates: there is no noise to remove.
        self.assertEqual(ee._shrunk_rates([(3, 3), (3, 0), (3, 3)]), [1.0, 0.0, 1.0])
        # One run per case cannot show noise.
        self.assertIsNone(ee._shrunk_rates([(1, 1), (1, 0)]))
        self.assertEqual(ee._smoothed_rates([(1, 1), (1, 0)]), [0.75, 0.25])

    def test_each_cell_carries_two_readings_and_the_observed_one_decides(self):
        # Every case is the same coin in each arm (0.7 against 0.4), so the
        # pilot's spread is all repetition noise. Observed rates treat that
        # noise as case spread and read lower than the noise-free reading.
        rng = random.Random(11)
        pilot = [((3, sum(rng.random() < 0.7 for _ in range(3))),
                  (3, sum(rng.random() < 0.4 for _ in range(3)))) for _ in range(30)]
        plan = ee.plan_power(pilot, min_lift=0.3, cases_grid=[20, 30], repetitions_grid=[3],
                             power=0.8, trials=600)
        small, large = plan["grid"]
        self.assertLess(small["power"], small["power_shrunk"])
        self.assertLess(small["power"], 0.8)
        self.assertGreaterEqual(small["power_shrunk"], 0.8)
        # The recommendation waits for the conservative reading to clear the target.
        self.assertEqual(plan["recommendation"]["cases"], 30)
        self.assertGreaterEqual(large["power"], 0.8)

    def test_a_one_repetition_pilot_has_no_noise_free_reading(self):
        single = [((1, 1), (1, 0)), ((1, 0), (1, 0)), ((1, 1), (1, 1))] * 4
        mixed = [((1, 1), (3, 2)), ((1, 0), (3, 1)), ((1, 1), (3, 0))] * 3
        for pilot in (single, mixed):
            with self.subTest(pilot=pilot[0]):
                cell = ee.plan_power(pilot, min_lift=0.2, cases_grid=[8], repetitions_grid=[1],
                                     trials=5)["grid"][0]
                self.assertIsNone(cell["power_shrunk"])
                self.assertIsNotNone(cell["power"])

    def test_five_cases_can_never_reach_alpha_whatever_the_lift(self):
        # 2 / 2**5 = 0.0625: no simulated eval of five cases is significant.
        plan = ee.plan_power(PILOT, min_lift=0.5, cases_grid=[5], repetitions_grid=[5], trials=100)
        self.assertEqual(plan["grid"][0]["power"], 0.0)
        self.assertEqual((plan["verdict"], plan["recommendation"]), ("grid-too-small", None))

    def test_a_lift_the_baseline_leaves_no_room_for_is_named(self):
        ceiling = [((3, 3), (3, 3))] * 8
        plan = ee.plan_power(ceiling, min_lift=0.5, cases_grid=[8], repetitions_grid=[3])
        self.assertEqual(plan["verdict"], "min-lift-exceeds-headroom")
        self.assertLess(plan["headroom"], 0.5)

    def test_headroom_reads_the_raw_baseline_and_a_cell_ignores_its_neighbours(self):
        # 22 of 24 baseline runs pass: one twelfth of headroom, whatever
        # smoothing would add.
        ceiling = [((3, 3), (3, 3))] * 6 + [((3, 3), (3, 2))] * 2
        plan = ee.plan_power(ceiling, min_lift=0.1, cases_grid=[8], repetitions_grid=[3], trials=5)
        self.assertEqual((plan["verdict"], plan["headroom"]), ("min-lift-exceeds-headroom", 0.083333))
        alone = ee.plan_power(PILOT, min_lift=0.15, cases_grid=[32], repetitions_grid=[3], trials=100)
        beside = ee.plan_power(PILOT, min_lift=0.15, cases_grid=[8, 32], repetitions_grid=[1, 3],
                               trials=100)
        self.assertEqual(alone["grid"][0], next(cell for cell in beside["grid"]
                                                if (cell["cases"], cell["repetitions"]) == (32, 3)))

    def test_the_plan_is_reproducible_and_rejects_bad_targets(self):
        options = {"min_lift": 0.15, "cases_grid": [16], "repetitions_grid": [3], "trials": 50}
        self.assertEqual(ee.plan_power(PILOT, **options), ee.plan_power(PILOT, **options))
        self.assertEqual(ee.plan_power([], **options)["verdict"], "no-pilot-pairs")
        for bad in ({"min_lift": 0}, {"power": 1.0}, {"trials": 0}, {"cases_grid": [0]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ee.plan_power(PILOT, **{**options, **bad})


class PlanEvalCommandTests(unittest.TestCase):
    def pilot(self, root):
        cases = [{"id": f"c{index}", "split": "tune", "prompt": "Do it.",
                  "assertions": [{"name": "has-alpha", "type": "contains", "value": "alpha"}]}
                 for index in range(6)]
        manifest = make_eval_repo(root, cases=cases)
        runs = root / "runs"
        for index in range(6):
            for run_number in (1, 2):
                write_run(runs / f"c{index}" / "with_skill" / f"run-{run_number}",
                          "alpha" if index < 4 or run_number == 1 else "beta")
                write_run(runs / f"c{index}" / "without_skill" / f"run-{run_number}",
                          "alpha" if index == 0 else "beta")
        attest_answer_design(manifest, runs)
        benchmark = root / "benchmark.json"
        self.assertEqual(run_cli("benchmark", manifest, "--runs", runs, "--out", benchmark)[0], 0)
        return benchmark

    def test_plan_reads_a_pilot_benchmark_through_the_cli(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "plan.json"
            code, _, stderr = run_cli("plan-eval", "--benchmark", self.pilot(root), "--min-lift", "0.3",
                                      "--cases", "6,24", "--repetitions", "2", "--trials", "100",
                                      "--out", out)
            self.assertEqual((code, stderr), (0, ""))
            plan = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((plan["pilot"]["cases"], plan["pilot"]["repetitions"], plan["unit"]), (6, 2, "case"))
        self.assertEqual(plan["pilot"]["limiting_variance"], "within-case")
        self.assertEqual([(cell["cases"], cell["repetitions"]) for cell in plan["grid"]], [(6, 2), (24, 2)])
        self.assertLess(plan["grid"][0]["power"], plan["grid"][1]["power"])

    def test_runs_that_pass_part_of_their_checks_are_counted_and_explained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = json.loads(self.pilot(root).read_text(encoding="utf-8"))
            whole = root / "whole.json"
            self.assertEqual(run_cli("plan-eval", "--benchmark", root / "benchmark.json", "--min-lift", "0.3",
                                     "--cases", "6", "--repetitions", "2", "--trials", "5",
                                     "--out", whole)[0], 0)
            report["results"][0]["objective_pass_rate"] = 0.5
            partial = root / "partial.json"
            partial.write_text(json.dumps(report), encoding="utf-8")
            out = root / "plan.json"
            self.assertEqual(run_cli("plan-eval", "--benchmark", partial, "--min-lift", "0.3",
                                     "--cases", "6", "--repetitions", "2", "--trials", "5",
                                     "--out", out)[0], 0)
            clean = json.loads(whole.read_text(encoding="utf-8"))
            plan = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((clean["pilot"]["partially_passing_runs"], clean["notes"]), (0, []))
        self.assertEqual(plan["pilot"]["partially_passing_runs"], 1)
        self.assertIn("a lift in fully passing runs", plan["notes"][0])

    def test_bad_grids_and_reports_are_usage_errors(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            benchmark = self.pilot(root)
            empty = root / "empty.json"
            empty.write_text("{}", encoding="utf-8")
            for argv, message in (
                    (("--benchmark", benchmark, "--min-lift", "0.3", "--cases", "6,x"),
                     "--cases must be a comma-separated list of whole numbers"),
                    (("--benchmark", benchmark, "--min-lift", "0.3", "--repetitions", "0"),
                     "--repetitions must be distinct whole numbers of 1 or more"),
                    (("--benchmark", benchmark, "--min-lift", "0.3", "--power", "1"),
                     "--power must lie strictly between 0 and 1"),
                    (("--benchmark", benchmark, "--min-lift", "2"),
                     "min-lift must be a pass-rate difference in (0, 1]"),
                    (("--benchmark", empty, "--min-lift", "0.3"), "has no results rows")):
                with self.subTest(message=message):
                    code, _, stderr = run_cli("plan-eval", *argv)
                    self.assertNotEqual(code, 0)
                    self.assertIn(message, stderr)


class HarmCheckTests(unittest.TestCase):
    def check(self, deltas, max_harm):
        return ee.harm_check(ee.sign_flip_interval(deltas), max_harm=max_harm)

    def test_no_lift_is_not_no_harm(self):
        # Eight cases that swing both ways: the lift is not significant, and
        # the interval still reaches a third of a run below zero, so a 0.05
        # margin is not cleared.
        deltas = thirds([3, -3, 3, -3, 3, -3, 2, -2])
        self.assertFalse(ee.sign_flip_test(deltas)["significant_at_0_05"])
        result = self.check(deltas, 0.05)
        self.assertEqual(result["verdict"], "harm-not-excluded")
        self.assertLess(result["lower"], -0.05)

    def test_lower_bound_at_or_above_the_margin_clears_it(self):
        deltas = [0.5] * 11 + [0.25]
        result = self.check(deltas, 0.05)
        self.assertEqual(result["verdict"], "within-margin")
        interval = ee.sign_flip_interval(deltas)
        self.assertEqual((result["lower"], result["upper"]), (interval["lower"], interval["upper"]))

    def test_the_margin_is_inclusive_at_the_lower_bound(self):
        deltas = thirds([1, -1, 1, 0, -1, 1, 0, 1, -1, 0])
        lower = ee.sign_flip_interval(deltas)["lower"]
        self.assertLess(lower, 0)
        self.assertEqual(self.check(deltas, -lower)["verdict"], "within-margin")
        self.assertEqual(self.check(deltas, -lower - 1e-6)["verdict"], "harm-not-excluded")

    def test_a_loss_inside_the_margin_at_its_upper_end_is_not_yet_harm(self):
        # The interval is [-1/3, -1/9]: a certain loss, but the margin 0.2
        # lies inside it, so a loss larger than the margin is neither shown
        # nor ruled out.
        deltas = thirds([-1, -1, -1, 0, -1, 0, -1, -1, 0, -1])
        result = self.check(deltas, 0.2)
        self.assertEqual((result["lower"], result["upper"]), (-0.333333, -0.111111))
        self.assertEqual(result["verdict"], "harm-not-excluded")
        self.assertEqual(self.check(deltas, 0.1)["verdict"], "harm-exceeds-margin")

    def test_upper_bound_below_the_margin_is_harm(self):
        deltas = [-0.5] * 11 + [-0.25]
        self.assertEqual(self.check(deltas, 0.05)["verdict"], "harm-exceeds-margin")

    def test_an_unbounded_interval_never_clears_a_margin(self):
        result = self.check(thirds([1, 1, 1, 1, 1]), 1.0)
        self.assertEqual(result["verdict"], "unbounded")
        self.assertIsNone(result["lower"])
        self.assertIn("cannot exclude any lift", result["reason"])

    def test_margin_outside_the_pass_rate_range_is_rejected(self):
        interval = ee.sign_flip_interval([0.5] * 11 + [0.25])
        for value in (0, -0.1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ee.harm_check(interval, max_harm=value)


class PairedSummaryTests(unittest.TestCase):
    def rows(self, pairs):
        out = []
        for index, (with_rate, without_rate) in enumerate(pairs):
            case = f"c{index}"
            out.append(result_row(case, "with_skill", rate=with_rate, run_number=1))
            out.append(result_row(case, "without_skill", rate=without_rate, run_number=1))
        return out

    def test_summary_carries_interval_and_noise_check(self):
        summary = sb.build_paired_summary(self.rows([(1.0, 0.0)] * 7 + [(1.0, 0.5)]), min_lift=0.2)
        self.assertTrue(summary["interval"]["bounded"])
        self.assertGreater(summary["interval"]["lower"], 0)
        self.assertEqual(summary["noise_check"]["cases_moved"], 8)
        self.assertEqual(summary["noise_check"]["min_lift"], 0.2)

    def test_summary_carries_harm_check_only_when_a_margin_is_given(self):
        rows = self.rows([(1.0, 0.0)] * 7 + [(1.0, 0.5)])
        self.assertNotIn("harm_check", sb.build_paired_summary(rows))
        harm = sb.build_paired_summary(rows, max_harm=0.05)["harm_check"]
        self.assertEqual((harm["verdict"], harm["max_harm"], harm["unit"]),
                         ("within-margin", 0.05, "case"))

    def test_blocked_pairing_moves_harm_check_to_observed(self):
        rows = self.rows([(1.0, 0.0)] * 7 + [(1.0, 0.5)])
        rows.append(result_row("orphan", "with_skill", rate=1.0, run_number=1))
        summary = sb.build_paired_summary(rows, max_harm=0.05)
        self.assertEqual(summary["harm_check"]["availability"], "unavailable")
        self.assertEqual(summary["observed_harm_check"]["verdict"], "within-margin")

    def model_rows(self, pairs_by_model):
        out = []
        for model, pairs in pairs_by_model.items():
            for row in self.rows(pairs):
                out.append({**row, "model": model})
        return out

    def test_several_models_get_a_holm_family_and_one_model_does_not(self):
        lucky = [(1.0, 0.0)] * 6
        flat = [(1.0, 0.0), (0.0, 1.0)] * 3
        one = sb.build_paired_summary(self.model_rows({"m1": lucky}))
        self.assertNotIn("by_model_multiplicity", one)
        summary = sb.build_paired_summary(
            self.model_rows({"m1": lucky, "m2": flat, "m3": flat, "m4": flat}))
        self.assertTrue(summary["by_model"]["m1"]["significance"]["significant_at_0_05"])
        family = summary["by_model_multiplicity"]
        self.assertEqual((family["method"], family["family"], family["tests"]),
                         ("holm", "by_model", 4))
        self.assertEqual(family["by_test"]["m1"]["adjusted_p_value"], 0.125)
        self.assertEqual(family["significant_after_adjustment"], [])

    def test_a_withheld_report_withdraws_the_family_with_its_tests(self):
        lucky = [(1.0, 0.0)] * 7
        summary = sb.build_paired_summary(self.model_rows({"m1": lucky, "m2": lucky}))
        self.assertEqual(summary["by_model_multiplicity"]["significant_after_adjustment"], ["m1", "m2"])
        withheld = sb.invalidate_report_pairing(summary, "grading_evidence_incomplete")
        family = withheld["by_model_multiplicity"]
        self.assertEqual((family["tests"], family["by_test"], family["untested"]), (0, {}, ["m1", "m2"]))
        self.assertEqual(family["significant_after_adjustment"], [])

    def test_blocked_pairing_moves_both_to_observed(self):
        rows = self.rows([(1.0, 0.0)] * 7 + [(1.0, 0.5)])
        rows.append(result_row("orphan", "with_skill", rate=1.0, run_number=1))
        summary = sb.build_paired_summary(rows)
        self.assertEqual(summary["availability"], "partial")
        self.assertEqual(summary["interval"]["availability"], "unavailable")
        self.assertTrue(summary["observed_interval"]["bounded"])
        self.assertIn("verdict", summary["observed_noise_check"])

    def test_incomplete_grading_moves_every_estimate_to_observed(self):
        # Every pair forms, but a script oracle that did not run (no
        # --allow-scripts) leaves each row's grading incomplete. The report
        # withholds the lift, and with it the interval, the noise check and
        # the graded channel, on the pooled block and on each model's.
        cases = [{"id": f"c{i}", "split": "tune", "kind": "behavior", "prompt": "Do it.",
                  "assertions": [
                      {"name": "has-alpha", "type": "contains", "value": "alpha"},
                      {"name": "oracle", "type": "script", "command": ["true"]},
                      {"name": "quality", "type": "judge", "rubric": ["Names the first Greek letter"]}]}
                 for i in range(6)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = write_demo_manifest(root, demo_manifest(cases=cases))
            runs = root / "runs"
            for i in range(6):
                for model in ("m1", "m2"):
                    write_run(runs / f"c{i}" / model / "with_skill", "alpha")
                    # c0 passes in both arms, so one delta per model is 0.
                    write_run(runs / f"c{i}" / model / "without_skill", "none" if i else "alpha")
            attest_answer_design(path, runs)
            verdicts = judge_with_stub(path, runs, root / "verdicts.jsonl",
                                       passes_on="alpha", scored=True)
            out = root / "benchmark.json"
            code, _, stderr = run_cli("benchmark", path, "--runs", runs,
                                      "--judge-results", verdicts, "--out", out)
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report["incomplete_reasons"], ["grading_evidence_incomplete"])
        withheld = {"availability": "unavailable", "reason": "grading_evidence_incomplete"}
        lift = report["paired_summary"]
        for label, block in (("pooled", lift), *lift["by_model"].items()):
            with self.subTest(block=label):
                self.assertEqual(block["interval"], withheld)
                self.assertEqual(block["noise_check"], withheld)
                # Each model pairs 6 cases, 5 of which moved; pooled, 12 and 10.
                self.assertEqual(block["observed_interval"]["n"], 12 if label == "pooled" else 6)
                self.assertEqual(block["observed_noise_check"]["cases_moved"],
                                 10 if label == "pooled" else 5)
        self.assertEqual(lift["graded"]["availability"], "partial")
        self.assertIsNone(lift["graded"]["delta"])
        self.assertEqual(lift["graded"]["reason"], "grading_evidence_incomplete")
        self.assertEqual(lift["observed_graded"]["delta"], round(10 / 12, 4))


class FloorCeilingTests(unittest.TestCase):
    def test_classification(self):
        self.assertIs(ee.ceiling_or_floor(1.0, 1.0), ee.DiscriminationFailure.CEILING)
        self.assertIs(ee.ceiling_or_floor(0.0, 0.0), ee.DiscriminationFailure.FLOOR)
        self.assertIsNone(ee.ceiling_or_floor(0.5, 0.5))
        self.assertIsNone(ee.ceiling_or_floor(None, 0.0))

    def test_suggest_cases_never_hardens_a_floor_case(self):
        report = {"case_flags": [
            {"case_id": "floor", "flags": [CaseFlag.FLOOR.value, "no objective lift", "with-skill failure"]},
            {"case_id": "ceiling", "flags": ["saturated/non-discriminating", "no objective lift"]},
        ]}
        manifest = {"cases": [{"id": "floor", "prompt": "p", "assertions": []},
                              {"id": "ceiling", "prompt": "q", "assertions": []}]}
        seeds = sb.suggest_case_candidates(report, manifest)
        self.assertEqual([seed["case_id"] for seed in seeds], ["ceiling"])
        self.assertIn("why the case is hard", seeds[0]["instruction"])

    def res(self, case, variant, rate, intent="capability"):
        return {"case_id": case, "variant": variant, "run_number": 1,
                "objective_pass_rate": rate, "combined_pass_rate": rate,
                "missing_output": False, "execution_valid": True, "eval_intent": intent}

    def test_readiness_separates_floor_from_base_saturation(self):
        report = {"availability": "complete", "results": [
            self.res("floor", "with_skill", 0.0), self.res("floor", "without_skill", 0.0),
            self.res("guard", "with_skill", 0.0, "regression"),
            self.res("guard", "without_skill", 0.0, "regression"),
            self.res("ceiling", "with_skill", 1.0), self.res("ceiling", "without_skill", 1.0),
        ]}
        signals = sb.readiness_run_signals(report)
        self.assertEqual(signals["floor_cases"], ["floor", "guard"])
        self.assertEqual(signals["base_saturated_cases"], ["ceiling"])
        # A regression guard nothing passes is broken, not holding.
        self.assertEqual(signals["base_saturated_expected_cases"], [])


class FloorEndToEndTests(unittest.TestCase):
    def test_graded_floor_case_is_flagged_and_audited(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = write_demo_manifest(root, demo_manifest())
            runs = root / "runs"
            for variant in ("with_skill", "without_skill"):
                base = runs / "case-1" / variant
                base.mkdir(parents=True)
                (base / "output.md").write_text("no match here", encoding="utf-8")
            attest_answer_design(path, runs)
            report = sb.build_benchmark_report(path, runs)
            audit = sb.audit_manifest_report(path, runs=str(runs))
        flags = report["case_flags"][0]["flags"]
        self.assertIn(CaseFlag.FLOOR.value, flags)
        self.assertNotIn("saturated/non-discriminating", flags)
        kinds = {finding["kind"] for finding in audit["findings"]}
        self.assertIn("floor-eval", kinds)
        self.assertNotIn("no-lift-eval", kinds)
        self.assertIn("noise_check", report["paired_summary"])

    def test_a_gate_judge_that_passes_in_one_arm_keeps_the_case_off_the_floor(self):
        # The contains check fails in both arms, but the gate judge passes the
        # with-skill answer: combined 0.5 against 0. The flag, the audit and
        # readiness decide the floor on one rule, so none of them calls this
        # case a floor, and readiness keeps it as qualitative-only.
        manifest = demo_manifest()
        manifest["cases"][0]["assertions"].append(
            {"name": "quality", "type": "judge", "severity": "gate",
             "rubric": ["Names the third Greek letter"]})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = write_demo_manifest(root, manifest)
            runs = root / "runs"
            write_run(runs / "case-1" / "with_skill", "gamma")
            write_run(runs / "case-1" / "without_skill", "none")
            attest_answer_design(path, runs)
            verdicts = judge_with_stub(path, runs, root / "verdicts.jsonl", passes_on="gamma")
            report = sb.build_benchmark_report(path, runs, judge_results_path=str(verdicts))
            audit = sb.audit_manifest_report(path, runs=str(runs),
                                             judge_results_path=str(verdicts))
        self.assertEqual([(row["variant"], row["objective_pass_rate"], row["combined_pass_rate"])
                          for row in report["results"]],
                         [("with_skill", 0.0, 0.5), ("without_skill", 0.0, 0.0)])
        flags = report["case_flags"][0]["flags"]
        self.assertNotIn(CaseFlag.FLOOR.value, flags)
        self.assertNotIn("floor-eval", {finding["kind"] for finding in audit["findings"]})
        self.assertEqual(audit["readiness"]["floor_cases"], [])
        self.assertEqual(audit["readiness"]["qualitative_only_cases"], ["case-1"])

    def test_a_case_gated_only_by_judges_is_flagged_on_the_score_readiness_reads(self):
        # A judge-only case has no objective rate, so it is outside the
        # objective pairing. Its flags must come from the combined score that
        # readiness reads, or readiness lists a floor case no flag or finding
        # names. Each case below exercises one flag; the stub judge passes an
        # answer that says "gamma" (score 1.0) and fails the rest (score 0.0).
        def judge(name, severity):
            return {"name": name, "type": "judge", "severity": severity,
                    "rubric": ["Names the third Greek letter"]}
        outputs = {  # case -> (with_skill runs, without_skill runs)
            "floor": (["none", "none"], ["none", "none"]),
            "ceiling": (["gamma", "gamma"], ["gamma", "gamma"]),
            "flaky": (["gamma", "none"], ["none", "none"]),
            "critical": (["gamma", "gamma"], ["none", "none"]),
            "graded": (["none", "none"], ["gamma", "gamma"]),
        }
        assertions = {"floor": [judge("quality", "gate")], "ceiling": [judge("quality", "gate")],
                      "flaky": [judge("quality", "gate")],
                      "critical": [judge("quality", "critical")],
                      "graded": [judge("quality", "gate"), judge("polish", "soft")]}
        cases = [{"id": cid, "split": "tune", "kind": "behavior", "prompt": "Do the task.",
                  "assertions": assertions[cid],
                  **({"reference_score": 0.5} if cid == "graded" else {})} for cid in outputs]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = write_demo_manifest(root, demo_manifest(cases=cases))
            runs = root / "runs"
            for cid, arms in outputs.items():
                for variant, texts in zip(("with_skill", "without_skill"), arms):
                    for number, text in enumerate(texts, 1):
                        write_run(runs / cid / variant / f"run-{number}", text)
            attest_answer_design(path, runs)
            verdicts = judge_with_stub(path, runs, root / "verdicts.jsonl",
                                       passes_on="gamma", scored=True)
            code, _, stderr = run_cli("benchmark", path, "--runs", runs, "--judge-results",
                                      verdicts, "--out", root / "benchmark.json")
            self.assertEqual(code, 0, stderr)
            report = json.loads((root / "benchmark.json").read_text(encoding="utf-8"))
            code, _, stderr = run_cli("audit-manifest", path, "--runs", runs, "--judge-results",
                                      verdicts, "--out", root / "audit.json")
            self.assertEqual(code, 0, stderr)
            audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["availability"], "complete")
        flags = {entry["case_id"]: entry for entry in report["case_flags"]}
        self.assertEqual({cid: flags[cid]["flags"] for cid in flags}, {
            "floor": [CaseFlag.FLOOR.value, "no objective lift", "with-skill failure"],
            "ceiling": ["saturated/non-discriminating", "no objective lift"],
            "flaky": ["with-skill failure", "flaky repeated pass rates: with_skill"],
            "critical": ["critical-failure: without_skill (quality)"],
            "graded": ["no objective lift", "with-skill failure",
                       "below-reference-floor: polish"],
        })
        self.assertEqual({cid: (flags[cid]["with_skill"], flags[cid]["without_skill"])
                          for cid in flags},
                         {"floor": (0.0, 0.0), "ceiling": (1.0, 1.0), "flaky": (0.5, 0.0),
                          "critical": (1.0, 0.0), "graded": (0.0, 1.0)})
        self.assertEqual({flags[cid]["signal"] for cid in flags}, {"combined"})
        # Readiness and the flags name the same floor and ceiling cases.
        self.assertEqual(audit["readiness"]["floor_cases"], ["floor"])
        self.assertEqual(audit["readiness"]["base_saturated_cases"], ["ceiling"])
        found = {(finding["kind"], finding["evidence"].get("case_id"))
                 for finding in audit["findings"] if isinstance(finding.get("evidence"), dict)}
        self.assertLessEqual({("floor-eval", "floor"), ("saturated-eval", "ceiling"),
                              ("no-lift-eval", "ceiling"), ("no-lift-eval", "graded"),
                              ("flaky-eval", "flaky")}, found)
        self.assertNotIn(("no-lift-eval", "floor"), found)


class EstimateTests(unittest.TestCase):
    def test_test_interval_and_noise_come_from_one_set_of_deltas(self):
        deltas = thirds([1, 2, 1, 0, 1, 2, 1, 1])
        estimate = ee.Estimate.from_deltas(deltas, unit=ee.InferenceUnit.CASE,
                                           without_rates=[0.3] * 8, min_lift=0.2)
        blocks = estimate.blocks()
        self.assertEqual({k: v for k, v in blocks["significance"].items() if k != "unit"},
                         sb.sign_flip_significance(deltas))
        self.assertEqual({k: v for k, v in blocks["interval"].items() if k != "unit"},
                         ee.sign_flip_interval(deltas))
        self.assertEqual({blocks[name]["unit"] for name in blocks}, {"case"})
        self.assertEqual(blocks["noise_check"]["min_lift"], 0.2)

    def test_unchanged_cases_do_not_move_a_lift_onto_the_sampled_path(self):
        # Flipping a zero delta changes no sum, so six cases that all moved +1
        # reach the exact p of 2 / 2**6 whatever number of cases did not move,
        # and the noise check's floor is the p the test reports.
        for zeros in (8, 9, 40):
            with self.subTest(zeros=zeros):
                deltas = [1.0] * 6 + [0.0] * zeros
                estimate = ee.Estimate.from_deltas(deltas, unit=ee.InferenceUnit.CASE,
                                                   without_rates=[0.0] * len(deltas))
                self.assertEqual(estimate.significance["method"], "sign-flip-exact")
                self.assertEqual(estimate.significance["p_value"], 0.03125)
                self.assertTrue(estimate.significant)
                self.assertEqual(estimate.interval["method"], "sign-flip-inversion-exact")
                self.assertGreater(estimate.interval["lower"], 0)
                self.assertEqual(estimate.noise["smallest_achievable_p"], 0.03125)
                self.assertEqual(estimate.noise["verdict"], "resolvable")

    def test_without_baseline_rates_there_is_no_noise_check(self):
        estimate = ee.Estimate.from_deltas([1.0] * 6, unit=ee.InferenceUnit.QUERY)
        self.assertNotIn("noise_check", estimate.blocks())
        self.assertTrue(estimate.significant)

    def test_the_sample_size_note_names_the_unit_and_what_repeats_cannot_do(self):
        case_note = ee.minimum_units_note(ee.InferenceUnit.CASE)
        self.assertIn("at least 6 cases", case_note)
        self.assertIn("do not add cases", case_note)
        replicate_note = ee.minimum_units_note(ee.InferenceUnit.REPLICATE_PAIR)
        self.assertIn("at least 6 matched replicate pairs", replicate_note)
        self.assertNotIn("do not add cases", replicate_note)
        self.assertIn("at least 8 authored queries",
                      ee.minimum_units_note(ee.InferenceUnit.QUERY, alpha=0.01))


if __name__ == "__main__":
    unittest.main()
