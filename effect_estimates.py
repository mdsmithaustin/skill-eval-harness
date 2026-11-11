"""Effect estimates for paired lift: an interval and a noise check.

The benchmark already tests lift with a sign-flip permutation test over
per-case deltas. This module adds the two numbers that test cannot give:

* ``sign_flip_interval`` inverts that same test. The interval is every shift
  ``delta`` the test would not reject at the chosen confidence, so the
  interval excludes zero exactly when the test rejects "no lift". The test
  (``sign_flip_test``) and the interval read one set of sign patterns and one
  decision rule, including the conservative Monte Carlo bound once there are
  too many sign outcomes to enumerate, so they cannot disagree.
* ``noise_check`` says whether the eval could have seen a lift at all. It
  reports the smallest p-value the observed data could ever produce, the
  interval half-width (the noise floor), and the headroom left in the
  ``without_skill`` arm, then names which of them limits the eval.

``ceiling_or_floor`` separates the two ways a case stops discriminating: both
arms always pass (ceiling, the case is too easy) and both arms always fail
(floor, which is more often a broken case or assertion than a hard task).

Everything here is deterministic. Sampling is seeded so a re-grade stays
byte-identical (CF.3).
"""
from __future__ import annotations

import bisect
import functools
import itertools
import math
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any

DEFAULT_ALPHA = 0.05
DEFAULT_CONFIDENCE = 0.95
# Exact enumeration while the non-zero deltas' sign patterns take at most
# 2**14 distinct (sum, weight) outcomes: 14 distinct non-zero deltas, or many
# more when deltas repeat or are whole numbers of runs. Zero deltas cost a
# factor of (zeros + 1) in the interval search rather than 2**zeros, bounded
# at 2**16 against the moved units. The pattern sums are grouped by pattern
# weight and sorted once, so each p-value evaluation during the interval
# search is a handful of binary searches.
MAX_EXACT_CASES = 14
# Past the exact budget the test samples sign patterns: 4,096 at first, four
# times as many while the decision at alpha is still open, up to 2**18. The
# decision uses an upper confidence bound on p that fails with probability at
# most FAILURE_PROBABILITY over all those looks together.
SAMPLED_PATTERNS = 4096
MAX_SAMPLED_PATTERNS = 1 << 18
FAILURE_PROBABILITY = 0.001
_ESCALATION = 4
# Merging outcomes stops (and the test samples) past this many dict updates.
_EXACT_WORK = 1 << 18
# Deltas within this relative distance of a fraction with a denominator up to
# _MAX_DENOMINATOR are counted on that fraction's lattice.
_MAX_DENOMINATOR = 10**6
_LATTICE_SLACK = 1e-12
_TOLERANCE = 1e-12
_SEARCH_STEPS = 60
_DECIMALS = 6


class DiscriminationFailure(str, Enum):
    """Why a case measured no lift when both arms scored the same extreme."""

    CEILING = "ceiling"
    FLOOR = "floor"


class NoiseVerdict(str, Enum):
    """What limits the eval's ability to show a lift, in the order checked."""

    NO_DATA = "no-data"
    TOO_FEW_CASES_MOVED = "too-few-cases-moved"
    UNBOUNDED = "unbounded"
    NOISE_EXCEEDS_HEADROOM = "noise-exceeds-headroom"
    NOISE_EXCEEDS_MIN_LIFT = "noise-exceeds-min-lift"
    RESOLVABLE = "resolvable"


class HarmVerdict(str, Enum):
    """Whether the interval rules out a loss larger than the margin."""

    UNBOUNDED = "unbounded"
    WITHIN_MARGIN = "within-margin"
    HARM_NOT_EXCLUDED = "harm-not-excluded"
    HARM_EXCEEDS_MARGIN = "harm-exceeds-margin"


class InferenceUnit(str, Enum):
    """What one delta in a paired test is a delta of.

    The sign-flip test counts units, not runs. Which unit it counts depends on
    the report: benchmark lift pairs per (case, model), ablation confirmation
    pairs repetitions within one case, and trigger comparison pairs authored
    queries. Extra repetitions sharpen a case's rate but never add a case, so
    the advice for an underpowered eval depends on the unit.
    """

    CASE = "case"
    REPLICATE_PAIR = "replicate_pair"
    QUERY = "query"

    @property
    def plural(self) -> str:
        return {InferenceUnit.CASE: "cases",
                InferenceUnit.REPLICATE_PAIR: "matched replicate pairs",
                InferenceUnit.QUERY: "authored queries"}[self]


def minimum_units_note(unit: InferenceUnit, alpha: float = DEFAULT_ALPHA) -> str:
    """How many units must move before the paired test can reach ``alpha``."""
    needed = cases_needed_for_alpha(alpha)
    note = (f"p <= {alpha:g} needs at least {needed} {unit.plural} that move the same way "
            f"(the smallest reachable p with {needed} is {smallest_achievable_p(needed):g})")
    if unit is InferenceUnit.CASE:
        note += "; more repetitions sharpen each case's rate but do not add cases"
    return note


def ceiling_or_floor(with_rate: float | None, without_rate: float | None,
                     *, eps: float = 1e-9) -> DiscriminationFailure | None:
    """Classify a case whose two arms sit at the same extreme.

    Both arms at 1.0 is a ceiling: the base model already does the task.
    Both arms at 0.0 is a floor: nothing passes, so suspect the case or its
    assertions before making it harder. Anything else is neither.
    """
    if with_rate is None or without_rate is None:
        return None
    if with_rate >= 1 - eps and without_rate >= 1 - eps:
        return DiscriminationFailure.CEILING
    if with_rate <= eps and without_rate <= eps:
        return DiscriminationFailure.FLOOR
    return None


def smallest_achievable_p(nonzero_cases: int) -> float:
    """The smallest two-sided exact sign-flip p-value k moved cases can reach.

    Cases with a zero delta add nothing to the test: flipping the sign of zero
    changes no sum. With ``k`` non-zero deltas all pointing the same way, only
    the observed pattern and its mirror image are as extreme, so the p-value
    can never go below ``2 / 2**k``.
    """
    if isinstance(nonzero_cases, bool) or not isinstance(nonzero_cases, int) or nonzero_cases < 0:
        raise ValueError("nonzero_cases must be a non-negative integer")
    if nonzero_cases == 0:
        return 1.0
    return min(1.0, 2.0 / (1 << nonzero_cases))


def cases_needed_for_alpha(alpha: float = DEFAULT_ALPHA) -> int:
    """How many cases must move (all the same way) before p <= alpha is possible."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return max(1, math.ceil(math.log2(2.0 / alpha)))


@dataclass(frozen=True)
class _SignPatterns:
    """Sign-pattern sums grouped by pattern weight.

    A sign pattern ``s`` flips some cases. For a shift ``delta`` its test
    statistic is ``|A_s - delta * B_s|`` where ``A_s = sum(s_i * d_i)`` and
    ``B_s = sum(s_i)``. ``B_s`` takes few values, so grouping the ``A_s`` by
    ``B_s`` and sorting each group turns a tail count into binary searches.
    Each group carries the running count of patterns behind its sorted sums,
    and ``zero_offsets`` the ``(B, count)`` spread of the zero deltas, which
    move ``B`` but never ``A``.

    A tail count visits every (group, zero offset) pair. With many groups and
    many zeros that is slow, so ``cells`` then keeps the ``(A, B, count)``
    triples flat: an evaluation sorts ``A - delta * B`` once and visits each
    zero offset with two binary searches instead.
    """

    groups: tuple[tuple[int, tuple[float, ...], tuple[int, ...]], ...]
    zero_offsets: tuple[tuple[int, int], ...]
    total: int
    exact: bool
    cells: tuple[tuple[float, ...], tuple[int, ...], tuple[int, ...]] | None = None
    # Sampled patterns only: the failure probability their upper bound carries.
    failure_probability: float = 0.0
    # The last evaluation's sort order: the search's shifts converge, so the
    # next sort starts nearly sorted.
    last_order: list[int] = field(default_factory=list, compare=False, repr=False)


def _lattice_scale(values: Sequence[float]) -> int | None:
    """The common denominator of the deltas when every one is a fraction with
    a small denominator, as pass-rate deltas are (whole runs and assertions
    over the repeats), so pattern sums can be counted as whole numbers."""
    scale = 1
    for value in values:
        fraction = Fraction(value).limit_denominator(_MAX_DENOMINATOR)
        if fraction == 0 or abs(float(fraction) - value) > _LATTICE_SLACK * max(1.0, abs(value)):
            return None
        scale = math.lcm(scale, fraction.denominator)
        if scale > _MAX_DENOMINATOR:
            return None
    return scale


def _exact_patterns(deltas: Sequence[float], max_exact_n: int) -> _SignPatterns | None:
    """Every sign pattern's (A, B), counted, or None past the exact budget.

    Flipping j of c equal deltas v gives the same (A, B) in comb(c, j) ways:
    A gains v * (c - 2j) and B gains c - 2j. Zeros only move B, so they stay
    one binomial spread instead of multiplying the outcomes. Outcomes with
    the same (A, B) merge, so the budget is on distinct outcomes: on a
    fraction lattice (pass-rate deltas) A is a whole number of steps and
    many more evals fit than the product of the counts suggests.
    """
    counts = Counter(v for v in deltas if v != 0)
    moved = sum(counts.values())
    zeros = len(deltas) - moved
    if (moved + 1) * (zeros + 1) > 1 << (max_exact_n + 2):
        return None
    # Bound the distinct outcomes before counting them. Without a lattice
    # they are the product of the counts; on one, A also lies among the
    # whole numbers of steps up to sum(c * |v|), and merging leaves a few
    # times fewer than that bound, so it may be tried with some slack.
    outcomes_bound = math.prod(count + 1 for count in counts.values())
    scale = _lattice_scale(list(counts))
    if scale:
        steps = sum(count * abs(round(value * scale)) for value, count in counts.items())
        outcomes_bound = min(outcomes_bound, (2 * steps + 1) * (moved + 1))
    if outcomes_bound > 1 << (max_exact_n + (2 if scale else 0)):
        return None
    cells: dict[tuple[float, int], int] = {(0, 0): 1}
    work = 0
    for value, count in sorted(counts.items()):
        step = round(value * scale) if scale else value
        work += len(cells) * (count + 1)
        if work > _EXACT_WORK:
            return None
        merged: dict[tuple[float, int], int] = {}
        for (a, b), w in cells.items():
            for j in range(count + 1):
                key = (a + step * (count - 2 * j), b + count - 2 * j)
                merged[key] = merged.get(key, 0) + w * math.comb(count, j)
        if len(merged) > 1 << max_exact_n:
            return None
        cells = merged
    outcomes = [(a / scale if scale else float(a), b, w) for (a, b), w in cells.items()]
    by_b: dict[int, list[tuple[float, int]]] = {}
    for a, b, w in outcomes:
        by_b.setdefault(b, []).append((a, w))
    groups = []
    for b, items in sorted(by_b.items()):
        items.sort()
        groups.append((b, tuple(a for a, _ in items),
                       tuple(itertools.accumulate((w for _, w in items), initial=0))))
    # A shifted tail count costs one pair of binary searches per (group, zero
    # offset), or one sort of the outcomes plus a pair per zero offset.
    flat = len(groups) * (zeros + 1) > len(outcomes) + zeros + 1
    return _SignPatterns(groups=tuple(groups),
                         zero_offsets=tuple((zeros - 2 * j, math.comb(zeros, j))
                                            for j in range(zeros + 1)),
                         total=1 << len(deltas), exact=True,
                         cells=(tuple(a for a, _, _ in outcomes), tuple(b for _, b, _ in outcomes),
                                tuple(w for _, _, w in outcomes)) if flat else None)


class _SignDraws:
    """One seeded stream of sign patterns, extended a block at a time, so the
    patterns behind each look are a prefix of the next look's.

    Signs are drawn against the deltas in a canonical order (by magnitude,
    then value), so the patterns depend on the multiset of deltas, not their
    order. The deltas are cut into bytes of eight; for each byte a block of
    random bytes is drawn at once, and two 256-entry tables turn each random
    byte into the sum of the deltas it keeps positive and how many it keeps,
    so a pattern costs a table lookup per eight deltas. With s_i = +1 for a
    set bit: A = 2 * sum(kept) - W and B = 2 * kept - n.
    """

    def __init__(self, deltas: Sequence[float], seed: int) -> None:
        ordered = sorted(deltas, key=lambda value: (abs(value), value))
        self.n = len(ordered)
        self.whole = math.fsum(ordered)
        self.chunks = []
        for start in range(0, self.n, 8):
            chunk = ordered[start:start + 8]
            self.chunks.append((
                [math.fsum(v for bit, v in enumerate(chunk) if byte >> bit & 1) for byte in range(256)],
                [(byte & ((1 << len(chunk)) - 1)).bit_count() for byte in range(256)]))
        self.rng = random.Random(seed)
        self.sums: list[float] = []
        self.weights: list[int] = []

    def extend(self, size: int) -> None:
        block = size - len(self.sums)
        if block <= 0:
            return
        raws = [self.rng.getrandbits(8 * block).to_bytes(block, "little") for _ in self.chunks]
        kept_sums = map(math.fsum, zip(*(map(sums.__getitem__, raw)
                                         for (sums, _), raw in zip(self.chunks, raws))))
        kept = map(sum, zip(*(map(counts.__getitem__, raw)
                              for (_, counts), raw in zip(self.chunks, raws))))
        self.sums.extend(2 * value - self.whole for value in kept_sums)
        self.weights.extend(2 * count - self.n for count in kept)

    def patterns(self, size: int, failure_probability: float) -> _SignPatterns:
        grouped: dict[int, list[float]] = {}
        for a, b in zip(self.sums[:size], self.weights[:size]):
            grouped.setdefault(b, []).append(a)
        groups = tuple((b, tuple(sorted(values)), tuple(range(len(values) + 1)))
                       for b, values in sorted(grouped.items()))
        return _SignPatterns(groups=groups, zero_offsets=((0, 1),), total=size, exact=False,
                             failure_probability=failure_probability)


def _sampled_patterns(deltas: Sequence[float], samples: int, seed: int,
                      alpha: float) -> _SignPatterns:
    """Seeded sign patterns, as many as the decision at ``alpha`` needs.

    The first look draws ``samples`` patterns. While the bounds on p leave
    "no lift" undecided at alpha (the lower one at or below it, the upper one
    above), the next look draws four times as many, up to
    ``MAX_SAMPLED_PATTERNS``. Every look shares one failure probability
    split evenly, so the decision is as conservative as one look. The test
    and the interval both read the patterns of the look that decided, so
    their one decision rule is the test's at that look.
    """
    sizes = [samples]
    while sizes[-1] * _ESCALATION <= MAX_SAMPLED_PATTERNS:
        sizes.append(sizes[-1] * _ESCALATION)
    failure = FAILURE_PROBABILITY / len(sizes)
    draws = _SignDraws(deltas, seed)
    threshold = abs(draws.whole) - _TOLERANCE
    for size in sizes:
        draws.extend(size)
        hits = sum(1 for a in draws.sums[:size] if abs(a) >= threshold)
        if (size == sizes[-1]
                or monte_carlo_upper_bound(hits, size, failure_probability=failure) <= alpha
                or monte_carlo_lower_bound(hits, size, failure_probability=failure) > alpha):
            return draws.patterns(size, failure)
    raise AssertionError("unreachable: the last look always decides")


def _relative_entropy(p: float, q: float) -> float:
    """KL(p || q) between Bernoulli distributions, for q strictly inside (0, 1)."""
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def _chernoff_bound(hits: int, samples: int, failure_probability: float, *, upper: bool) -> float:
    """The farthest tail probability q the sample cannot rule out: by
    Chernoff, hits this far from q * samples happen with probability at most
    exp(-samples * KL(hits / samples || q)), and q stays while that exceeds
    the failure probability. Bisection, rounded outward."""
    if samples < 1:
        raise ValueError("Monte Carlo samples must be positive")
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must lie strictly between 0 and 1")
    observed = hits / samples
    budget = math.log(1.0 / failure_probability) / samples
    if upper:
        if observed >= 1:
            return 1.0
        near, far = observed, 1.0
    else:
        if observed <= 0:
            return 0.0
        near, far = observed, 0.0
    for _ in range(_SEARCH_STEPS):
        middle = (near + far) / 2
        if _relative_entropy(observed, middle) <= budget:
            near = middle
        else:
            far = middle
    return far


def monte_carlo_upper_bound(hits: int, samples: int, *,
                            failure_probability: float = FAILURE_PROBABILITY) -> float:
    """Distribution-free upper confidence bound for a sampled tail probability.

    The relative-entropy (Chernoff) bound is about ``p + sqrt(2 p ln(1/f) / m)``
    for small p, where the additive (Hoeffding) bound adds
    ``sqrt(ln(1/f) / 2m)`` whatever p is: about 0.03 at 4,096 patterns,
    which no stricter alpha could clear.
    """
    return _chernoff_bound(hits, samples, failure_probability, upper=True)


def monte_carlo_lower_bound(hits: int, samples: int, *,
                            failure_probability: float = FAILURE_PROBABILITY) -> float:
    """Distribution-free lower confidence bound for a sampled tail probability."""
    return _chernoff_bound(hits, samples, failure_probability, upper=False)


def _tail(patterns: _SignPatterns, whole: float, n: int, delta: float) -> tuple[float, float]:
    """Two-sided p-value of the sign-flip test on ``d_i - delta``, and its upper bound.

    Exact enumeration returns the p-value twice. A sampled p-value uses the
    (b + 1) / (m + 1) estimator, because the observed pattern is always a valid
    permutation under the null, and a distribution-free upper bound that the
    decision uses: a point estimate just under alpha is not evidence.
    """
    threshold = abs(whole - n * delta) - _TOLERANCE
    # Unshifted, the zeros' B never matters, so their spread folds into one.
    offsets = (patterns.zero_offsets if delta != 0
               else ((0, sum(count for _, count in patterns.zero_offsets)),))
    hits = 0
    if patterns.cells is not None and delta != 0 and threshold > 0:
        # |A - delta * (B + z)| >= t  <=>  u >= z * delta + t  or  u <= z * delta - t,
        # with u = A - delta * B sorted once for this delta.
        sums, weights, multiplicity = patterns.cells
        shifted = [a - delta * b for a, b in zip(sums, weights)]
        order = sorted(patterns.last_order or range(len(shifted)), key=shifted.__getitem__)
        patterns.last_order[:] = order
        ordered = list(map(shifted.__getitem__, order))
        running = list(itertools.accumulate(map(multiplicity.__getitem__, order), initial=0))
        for offset, weight in offsets:
            centre = offset * delta
            hits += weight * (running[-1] - running[bisect.bisect_left(ordered, centre + threshold)]
                              + running[bisect.bisect_right(ordered, centre - threshold)])
        return hits / patterns.total, hits / patterns.total
    for b, values, counts in patterns.groups:
        for offset, weight in offsets:
            centre = (b + offset) * delta
            # |A - centre| >= threshold  <=>  A >= centre + t  or  A <= centre - t
            if threshold <= 0:
                hits += weight * counts[-1]
                continue
            hits += weight * (counts[-1] - counts[bisect.bisect_left(values, centre + threshold)]
                              + counts[bisect.bisect_right(values, centre - threshold)])
    if patterns.exact:
        p = hits / patterns.total
        return p, p
    return ((hits + 1) / (patterns.total + 1),
            monte_carlo_upper_bound(hits, patterns.total,
                                    failure_probability=patterns.failure_probability))


def _rejects(patterns: _SignPatterns, whole: float, n: int, delta: float, alpha: float) -> bool:
    return _tail(patterns, whole, n, delta)[1] <= alpha


def _patterns(values: Sequence[float], max_exact_n: int, samples: int,
              seed: int, alpha: float) -> _SignPatterns:
    return _patterns_of(tuple(sorted(values)), max_exact_n, samples, seed, alpha)


@functools.lru_cache(maxsize=2)
def _patterns_of(values: tuple[float, ...], max_exact_n: int, samples: int,
                 seed: int, alpha: float) -> _SignPatterns:
    # Both depend on the multiset of deltas only, and Estimate asks for the
    # test and then the interval of one set, so the second reuses the first.
    exact = _exact_patterns(values, max_exact_n)
    if exact is not None:
        return exact
    return _sampled_patterns(values, samples, seed, alpha)


def sign_flip_test(deltas: Sequence[float], *, max_exact_n: int = MAX_EXACT_CASES,
                   samples: int = SAMPLED_PATTERNS, seed: int = 0,
                   alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Two-sided sign-flip permutation test over per-case paired deltas.

    Under the null (the skill does nothing) each case's delta is equally
    likely to have either sign, so p is the share of sign patterns whose
    |mean| reaches the observed |mean|. Exact enumeration while the non-zero
    deltas' patterns reach at most ``2**max_exact_n`` distinct (sum, weight)
    outcomes (zeros never count, since flipping one changes no sum), then a
    seeded sample, so a re-grade stays byte-identical (CF.3). The sampled
    decision uses the upper bound, and ``sampled_patterns`` says how many
    patterns it took to reach it (more while the decision at alpha is open).
    """
    n = len(deltas)
    if n == 0:
        return {"method": "sign-flip", "n": 0, "observed_mean_delta": None,
                "p_value": None, "p_value_upper_bound": None,
                "significant_at_0_05": False}
    observed = statistics.mean(deltas)
    if all(abs(d) < _TOLERANCE for d in deltas):
        return {"method": "sign-flip", "n": n, "observed_mean_delta": 0.0,
                "p_value": 1.0, "p_value_upper_bound": 1.0,
                "significant_at_0_05": False}
    values = [float(value) for value in deltas]
    patterns = _patterns(values, max_exact_n, samples, seed, alpha)
    p, p_upper = _tail(patterns, math.fsum(values), n, 0.0)
    return {"method": "sign-flip-exact" if patterns.exact else "sign-flip-sampled",
            "n": n, "observed_mean_delta": observed,
            "p_value": p, "p_value_upper_bound": p_upper,
            **({} if patterns.exact else {"sampled_patterns": patterns.total}),
            "significant_at_0_05": p_upper <= alpha}


def _bound(patterns: _SignPatterns, whole: float, n: int, centre: float,
           far: float, alpha: float) -> float | None:
    """Search from the accepted centre toward ``far`` for the last accepted shift.

    Moving away from the mean, the observed statistic grows with slope ``n``
    while every pattern's statistic grows with slope ``|B| <= n``, so the
    p-value never rises again once it falls. That makes the accepted region
    an interval and bisection exact. The endpoint is reported to six
    decimals, so the search stops once both ends of the bracket round alike:
    the boundary lies between them and rounds the same way.
    """
    if not _rejects(patterns, whole, n, far, alpha):
        return None
    accepted, rejected = centre, far
    for _ in range(_SEARCH_STEPS):
        if round(accepted, _DECIMALS) == round(rejected, _DECIMALS):
            break
        middle = (accepted + rejected) / 2
        if not _rejects(patterns, whole, n, middle, alpha):
            accepted = middle
        else:
            rejected = middle
    return accepted


def sign_flip_interval(deltas: Sequence[float], *, confidence: float = DEFAULT_CONFIDENCE,
                       max_exact_n: int = MAX_EXACT_CASES, samples: int = SAMPLED_PATTERNS,
                       seed: int = 0) -> dict[str, Any]:
    """Confidence interval for the mean per-case delta, by inverting the sign-flip test.

    Returns ``bounded: False`` with null endpoints when the test cannot reject
    any shift at all, which happens when there are too few cases (five or fewer
    at 95%, since 2 / 2**5 > 0.05): the data cannot rule anything out, and
    printing an interval would claim a precision the eval does not have. It
    does the same when every delta is equal: the test reads only signs, so it
    rejects every shift but that value, and the point it leaves is no bound.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    # Sorted so the result depends on the multiset of deltas, not their order;
    # the seeded sampled path would otherwise vary with case ordering.
    values = sorted(float(value) for value in deltas)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("deltas must be finite")
    n = len(values)
    # Rounded so that confidence 1 - alpha gives back the test's own alpha:
    # the sampled path chooses its look by the decision at alpha.
    alpha = round(1 - confidence, 12)
    base: dict[str, Any] = {"confidence": confidence, "n": n}
    if n == 0:
        return {**base, "method": "unavailable", "lower": None, "upper": None,
                "bounded": False, "reason": "no paired cases"}
    centre = math.fsum(values) / n
    patterns = _patterns(values, max_exact_n, samples, seed, alpha)
    method = ("sign-flip-inversion-exact" if patterns.exact
              else "sign-flip-inversion-sampled")
    whole = math.fsum(values)
    spread = max(values) - min(values)
    # Past every observed delta the shifted deltas all share one sign, which is
    # the most extreme pattern the test can see; one spread further is safely
    # beyond any boundary.
    margin = max(spread, 1.0)
    lower = _bound(patterns, whole, n, centre, min(values) - margin, alpha)
    upper = _bound(patterns, whole, n, centre, max(values) + margin, alpha)
    if lower is None or upper is None:
        return {**base, "method": method, "lower": None, "upper": None, "bounded": False,
                "reason": (f"{n} paired case(s) cannot exclude any lift at "
                           f"{confidence:.0%}; the test needs at least "
                           f"{cases_needed_for_alpha(alpha)} cases that differ between arms")}
    if values[0] == values[-1]:
        return {**base, "method": method, "lower": None, "upper": None, "bounded": False,
                "reason": (f"every paired delta is the same ({values[0]:g}); a sign-flip "
                           "test reads only signs, so it cannot bound a constant sample")}
    return {**base, "method": method, "lower": round(lower, _DECIMALS),
            "upper": round(upper, _DECIMALS), "bounded": True}


def noise_check(deltas: Sequence[float], without_rates: Sequence[float], *,
                interval: dict[str, Any], alpha: float = DEFAULT_ALPHA,
                min_lift: float | None = None) -> dict[str, Any]:
    """Could this eval have shown a lift? Names the first limit that says no.

    * ``smallest_achievable_p``: with ``k`` cases whose delta is non-zero, the
      exact test can never report less than ``2 / 2**k``. Below
      ``cases_needed_for_alpha`` moved cases, significance is impossible. The
      test decides on the same non-zero deltas, and up to ``MAX_EXACT_CASES``
      of them it is always exact, so this floor is the decision's floor.
    * ``noise_floor``: the interval half-width. A lift smaller than this is
      indistinguishable from run-to-run noise at the current case count.
    * ``headroom``: ``1 - without_skill`` rate, the largest lift the eval can
      show. When the noise floor exceeds it, no skill could clear the noise.
    * ``min_lift`` (optional): the smallest lift the author would act on.

    ``projected_cases`` scales the case count by ``(noise_floor / target)**2``,
    the usual square-root law, and is an estimate, not a guarantee.
    """
    values = [float(value) for value in deltas]
    n = len(values)
    moved = sum(1 for value in values if abs(value) > _TOLERANCE)
    needed = cases_needed_for_alpha(alpha)
    floor = None
    if interval.get("bounded"):
        floor = round((float(interval["upper"]) - float(interval["lower"])) / 2, 6)
    headroom = None
    if without_rates:
        headroom = round(1 - statistics.fmean(float(rate) for rate in without_rates), 6)
    if n == 0:
        verdict = NoiseVerdict.NO_DATA
    elif moved < needed:
        verdict = NoiseVerdict.TOO_FEW_CASES_MOVED
    elif floor is None:
        verdict = NoiseVerdict.UNBOUNDED
    elif headroom is not None and floor >= headroom:
        verdict = NoiseVerdict.NOISE_EXCEEDS_HEADROOM
    elif min_lift is not None and floor >= min_lift:
        verdict = NoiseVerdict.NOISE_EXCEEDS_MIN_LIFT
    else:
        verdict = NoiseVerdict.RESOLVABLE
    out: dict[str, Any] = {
        "verdict": verdict.value,
        "cases": n,
        "cases_moved": moved,
        "cases_needed_for_alpha": needed,
        "alpha": alpha,
        "smallest_achievable_p": smallest_achievable_p(moved),
        "noise_floor": floor,
        "headroom": headroom,
    }
    if verdict is NoiseVerdict.UNBOUNDED and interval.get("reason"):
        out["reason"] = interval["reason"]
    if min_lift is not None:
        out["min_lift"] = min_lift
    target = min_lift if min_lift is not None else headroom
    if floor is not None and target is not None and target > 0 and floor >= target:
        out["projected_cases"] = math.ceil(n * (floor / target) ** 2)
    return out


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the order given.

    The i-th smallest of m p-values is multiplied by ``m - i`` (counting from
    zero), then made monotone and capped at one. Rejecting where the adjusted
    value is at or below alpha keeps the chance of any false rejection in the
    family at or below alpha, with no assumption about how the tests depend
    on each other.
    """
    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must lie in [0, 1]")
    m = len(p_values)
    order = sorted(range(m), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def multiplicity(significance_by_name: Mapping[str, Mapping[str, Any]], *, family: str,
                 alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Holm-adjust a family of significance blocks that one report prints together.

    Each block's deciding p-value is its ``p_value_upper_bound`` when it has
    one (the sampled path decides on the bound), else its ``p_value``. A block
    with no p-value ran no test, so it is listed under ``untested`` and does
    not count toward the family size.
    """
    tested: dict[str, float] = {}
    untested: list[str] = []
    for name, block in significance_by_name.items():
        p = block.get("p_value_upper_bound")
        if p is None:
            p = block.get("p_value")
        if p is None:
            untested.append(name)
        else:
            tested[name] = float(p)
    adjusted = dict(zip(tested, holm_adjust(list(tested.values()))))
    return {
        "method": "holm",
        "family": family,
        "tests": len(tested),
        "alpha": alpha,
        "by_test": {name: {"p_value": tested[name], "adjusted_p_value": adjusted[name],
                           "significant": adjusted[name] <= alpha}
                    for name in sorted(tested)},
        "significant_after_adjustment": sorted(name for name, value in adjusted.items()
                                               if value <= alpha),
        "untested": sorted(untested),
    }


def lattice_sign_flip_p(steps: Sequence[int]) -> float:
    """Exact two-sided sign-flip p-value for whole-number deltas.

    The planner simulates thousands of evals whose per-case deltas are whole
    runs out of the repetitions, so the sum of signed deltas lives on the
    integers and one convolution counts every sign pattern. It is the same
    p-value ``sign_flip_test`` reports on its exact path.
    """
    moved = [abs(step) for step in steps if step]
    if not moved:
        return 1.0
    reach = sum(moved)
    counts = [0] * (2 * reach + 1)
    counts[reach] = 1
    for size in moved:
        shifted = [0] * (2 * reach + 1)
        for index, count in enumerate(counts):
            if count:
                shifted[index - size] += count
                shifted[index + size] += count
        counts = shifted
    observed = abs(sum(steps))
    extreme = sum(count for index, count in enumerate(counts) if abs(index - reach) >= observed)
    return extreme / (1 << len(moved))


def variance_split(case_counts: Sequence[tuple[tuple[int, int], tuple[int, int]]]) -> dict[str, Any]:
    """Split the variance of per-case deltas into repetition noise and case spread.

    Each entry is ``((runs, passes), (runs, passes))`` for the skill arm and
    the baseline arm of one case. ``within_case_variance`` is the mean
    binomial variance of a case's delta, which more repetitions shrink.
    ``between_case_variance`` is what is left of the observed variance, which
    only more cases average away. With one run per arm the within-case share
    cannot be estimated and both parts are null.
    """
    deltas = [with_passes / with_runs - without_passes / without_runs
              for (with_runs, with_passes), (without_runs, without_passes) in case_counts]
    out: dict[str, Any] = {
        "cases": len(deltas),
        "observed_delta_variance": round(statistics.variance(deltas), _DECIMALS) if len(deltas) > 1 else None,
        "within_case_variance": None,
        "between_case_variance": None,
        "within_case_share": None,
    }
    if len(deltas) < 2 or any(runs < 2 for arms in case_counts for runs, _ in arms):
        return out
    within = statistics.fmean(
        sum((passes / runs) * (1 - passes / runs) / (runs - 1) for runs, passes in arms)
        for arms in case_counts)
    total = statistics.variance(deltas)
    between = max(0.0, total - within)
    out["within_case_variance"] = round(within, _DECIMALS)
    out["between_case_variance"] = round(between, _DECIMALS)
    if within + between > 0:
        out["within_case_share"] = round(within / (within + between), _DECIMALS)
    return out


def _shifted_rates(with_rates: Sequence[float], without_rates: Sequence[float],
                   min_lift: float) -> list[float] | None:
    """The pilot's skill-arm rates moved by one shift so the mean lift is ``min_lift``.

    Clipping at 0 and 1 keeps each case's rate a probability, and bisection on
    the shift recovers the target mean. None when even a skill arm that always
    passes cannot reach it: the baseline leaves too little headroom.
    """
    def lift(shift: float) -> float:
        return statistics.fmean(min(1.0, max(0.0, rate + shift)) - base
                                for rate, base in zip(with_rates, without_rates))
    if lift(1.0) < min_lift - _TOLERANCE:
        return None
    low, high = -1.0, 1.0
    for _ in range(_SEARCH_STEPS):
        middle = (low + high) / 2
        if lift(middle) < min_lift:
            low = middle
        else:
            high = middle
    return [min(1.0, max(0.0, rate + high)) for rate in with_rates]


def _smoothed_rates(arm: Sequence[tuple[int, int]]) -> list[float]:
    """Observed per-case pass rates, moved half a run toward one half so a
    case seen once is not treated as certain."""
    return [(passes + 0.5) / (runs + 1) for runs, passes in arm]


def _shrunk_rates(arm: Sequence[tuple[int, int]]) -> list[float] | None:
    """Per-case pass rates with repetition noise removed, or None when one run
    per case leaves that noise unmeasurable.

    Each rate is pulled toward the arm mean by the share of the observed
    variance that repetition noise explains. That treats every case as having
    the same true lift apart from noise, so it erases a lift that differs by
    case.
    """
    rates = [passes / runs for runs, passes in arm]
    if len(rates) < 2 or any(runs < 2 for runs, _ in arm):
        return None
    mean = statistics.fmean(rates)
    observed = statistics.variance(rates)
    noise = statistics.fmean(rate * (1 - rate) / (runs - 1) for rate, (runs, _) in zip(rates, arm))
    kept = max(0.0, observed - noise) / observed if observed > 0 else 0.0
    return [mean + kept * (rate - mean) for rate in rates]


def _simulated_power(with_rates: Sequence[float], without_rates: Sequence[float], *,
                     cases: int, repetitions: int, trials: int, rng: random.Random,
                     alpha: float) -> float:
    wins = 0
    for _ in range(trials):
        steps = []
        for index in rng.choices(range(len(without_rates)), k=cases):
            gained = sum(rng.random() < with_rates[index] for _ in range(repetitions))
            kept = sum(rng.random() < without_rates[index] for _ in range(repetitions))
            steps.append(gained - kept)
        if sum(steps) > 0 and lattice_sign_flip_p(steps) <= alpha:
            wins += 1
    return wins / trials


def plan_power(case_counts: Sequence[tuple[tuple[int, int], tuple[int, int]]], *,
               min_lift: float, cases_grid: Sequence[int], repetitions_grid: Sequence[int],
               power: float = 0.8, trials: int = 400, seed: int = 0,
               alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """How many cases and repetitions would let the paired test show ``min_lift``?

    Each grid cell simulates ``trials`` evals: draw cases from the pilot with
    replacement, draw each arm's passes, and run the exact sign-flip test.
    The skill-arm rates are shifted until the mean lift equals ``min_lift``.
    Power is the share of trials that reach ``alpha`` with a positive lift.

    A small pilot cannot pin down each case's true rates, so each cell carries
    two readings and neither is a bound. ``power`` simulates from the observed
    rates. Those carry the pilot's own repetition noise as if it were case
    spread, so on average ``power`` does not overstate, and it decides the
    recommendation. ``power_shrunk`` removes that noise. It is closer when the
    cases are alike in baseline and in gain, and it can read high when the two
    arms vary independently across cases. Both read low when baselines differ
    widely. ``power_shrunk`` is null when the pilot cannot measure repetition
    noise or the shrunk rates leave no room for ``min_lift``. The draws are
    seeded per cell, so the plan is reproducible.
    """
    if not 0 < min_lift <= 1:
        raise ValueError("min_lift must be a pass-rate difference in (0, 1]")
    if not 0 < power < 1:
        raise ValueError("power must lie strictly between 0 and 1")
    if trials < 1 or any(value < 1 for value in (*cases_grid, *repetitions_grid)):
        raise ValueError("trials, cases and repetitions must be positive")
    target: dict[str, Any] = {"min_lift": min_lift, "alpha": alpha, "power": power}
    base: dict[str, Any] = {"target": target, "pilot": variance_split(case_counts),
                            "trials": trials, "seed": seed}
    if not case_counts:
        return {**base, "verdict": "no-pilot-pairs", "grid": [], "recommendation": None}
    treatment = [arm for arm, _ in case_counts]
    control = [arm for _, arm in case_counts]
    without = _smoothed_rates(control)
    # Headroom is a fact about the pilot's baseline, so it reads the raw rates:
    # smoothing would invent room above a baseline that already passes.
    # Compared as reported, so a target copied from the reported headroom fits.
    headroom = round(1 - statistics.fmean(passes / runs for runs, passes in control), _DECIMALS)
    if min_lift > headroom + _TOLERANCE:
        return {**base, "verdict": "min-lift-exceeds-headroom",
                "headroom": headroom,
                "grid": [], "recommendation": None}
    # Smoothing lifts a low baseline, which can leave the smoothed rates less
    # room than the pilot showed. The skill arm then always passes: the
    # largest lift the smoothed rates allow, a little under the target.
    shifted = (_shifted_rates(_smoothed_rates(treatment), without, min_lift)
               or [1.0] * len(without))
    shrunk_with, shrunk_without = _shrunk_rates(treatment), _shrunk_rates(control)
    shrunk_shifted = (None if shrunk_with is None or shrunk_without is None
                      else _shifted_rates(shrunk_with, shrunk_without, min_lift))
    grid: list[dict[str, Any]] = []
    for cases in sorted(set(cases_grid)):
        for repetitions in sorted(set(repetitions_grid)):
            # One stream per reading per cell, so a cell's power does not
            # depend on which other cells the grid holds.
            share = _simulated_power(
                shifted, without, cases=cases, repetitions=repetitions, trials=trials,
                rng=random.Random(f"{seed}:{cases}:{repetitions}:observed"), alpha=alpha)
            shrunk_share = None
            if shrunk_shifted is not None and shrunk_without is not None:
                shrunk_share = round(_simulated_power(
                    shrunk_shifted, shrunk_without, cases=cases, repetitions=repetitions,
                    trials=trials, rng=random.Random(f"{seed}:{cases}:{repetitions}:shrunk"),
                    alpha=alpha), 4)
            grid.append({"cases": cases, "repetitions": repetitions,
                         "runs_per_arm": cases * repetitions, "power": round(share, 4),
                         "power_standard_error": round(math.sqrt(share * (1 - share) / trials), 4),
                         "power_shrunk": shrunk_share})
    enough = [cell for cell in grid if cell["power"] >= power]
    recommendation = min(enough, key=lambda cell: (cell["runs_per_arm"], cell["cases"]), default=None)
    return {**base, "verdict": "plan-found" if recommendation else "grid-too-small",
            "grid": grid, "recommendation": recommendation}


# Fewer labels than this in either human class leave the judge's error rate
# too rough to divide by.
MIN_CLASS_LABELS = 10


def _judge_error_rates(confusion: Mapping[str, int]) -> tuple[float, float, int, int] | str:
    """Sensitivity, specificity and the label count behind each, or why not."""
    human_pass = confusion["tp"] + confusion["fn"]
    human_fail = confusion["tn"] + confusion["fp"]
    if min(human_pass, human_fail) < MIN_CLASS_LABELS:
        return (f"needs at least {MIN_CLASS_LABELS} human pass and {MIN_CLASS_LABELS} human fail "
                f"labels; found {human_pass} pass and {human_fail} fail")
    sensitivity = confusion["tp"] / human_pass
    specificity = confusion["tn"] / human_fail
    if sensitivity + specificity <= 1:
        return ("the judge agrees with the human labels no better than chance "
                f"(sensitivity {sensitivity:.3f} + specificity {specificity:.3f} <= 1)")
    return sensitivity, specificity, human_pass, human_fail


def _normal_interval(estimate: float, variance: float, confidence: float,
                     low: float, high: float) -> dict[str, Any]:
    reach = statistics.NormalDist().inv_cdf((1 + confidence) / 2) * math.sqrt(variance)
    return {"confidence": confidence, "method": "normal-approximation",
            "lower": round(max(low, estimate - reach), 4),
            "upper": round(min(high, estimate + reach), 4)}


def judge_corrected_rate(passes: int, total: int, *, confusion: Mapping[str, int],
                         confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
    """A judge's pass rate corrected for the judge's own error rates.

    A judge that passes some answers a human fails, and fails some a human
    passes, reports ``p = theta * sens + (1 - theta) * (1 - spec)`` for a true
    rate ``theta``. Solving for theta is the Rogan-Gladen estimator. Its
    interval is a delta-method normal approximation that carries three
    sources of noise: the judged sample and the two labelled samples behind
    sensitivity and specificity.
    """
    if total < 1 or not 0 <= passes <= total:
        raise ValueError("passes must lie between 0 and a positive total")
    observed = passes / total
    out: dict[str, Any] = {"n": total, "observed_pass_rate": round(observed, 4)}
    rates = _judge_error_rates(confusion)
    if isinstance(rates, str):
        return {**out, "corrected_pass_rate": None, "interval": None, "reason": rates}
    sensitivity, specificity, human_pass, human_fail = rates
    youden = sensitivity + specificity - 1
    theta = min(1.0, max(0.0, (observed + specificity - 1) / youden))
    variance = (observed * (1 - observed) / total
                + theta ** 2 * sensitivity * (1 - sensitivity) / human_pass
                + (1 - theta) ** 2 * specificity * (1 - specificity) / human_fail) / youden ** 2
    return {**out, "corrected_pass_rate": round(theta, 4),
            "interval": _normal_interval(theta, variance, confidence, 0.0, 1.0)}


def judge_corrected_lift(treatment: tuple[int, int], control: tuple[int, int], *,
                         confusion: Mapping[str, int],
                         confidence: float = DEFAULT_CONFIDENCE) -> dict[str, Any]:
    """The difference of two judged pass rates, corrected for judge error.

    With one judge for both arms the false-positive rate cancels and the
    observed difference is the true one scaled by ``sens + spec - 1``, so an
    imperfect judge shrinks a real lift toward zero. Each of ``treatment`` and
    ``control`` is ``(passes, total)``. The interval treats the arms as
    independent samples, which is wider than a paired interval when the same
    cases feed both arms.
    """
    for passes, total in (treatment, control):
        if total < 1 or not 0 <= passes <= total:
            raise ValueError("passes must lie between 0 and a positive total")
    with_rate = treatment[0] / treatment[1]
    without_rate = control[0] / control[1]
    out: dict[str, Any] = {"observed_delta": round(with_rate - without_rate, 4)}
    rates = _judge_error_rates(confusion)
    if isinstance(rates, str):
        return {**out, "corrected_delta": None, "interval": None, "reason": rates}
    sensitivity, specificity, human_pass, human_fail = rates
    youden = sensitivity + specificity - 1
    lift = max(-1.0, min(1.0, (with_rate - without_rate) / youden))
    variance = ((with_rate * (1 - with_rate) / treatment[1]
                 + without_rate * (1 - without_rate) / control[1])
                + lift ** 2 * (sensitivity * (1 - sensitivity) / human_pass
                               + specificity * (1 - specificity) / human_fail)) / youden ** 2
    return {**out, "corrected_delta": round(lift, 4),
            "interval": _normal_interval(lift, variance, confidence, -1.0, 1.0)}


def harm_check(interval: dict[str, Any], *, max_harm: float) -> dict[str, Any]:
    """Does the interval rule out a loss larger than ``max_harm``?

    A lift that is not significant is not evidence of no harm: the interval
    may still reach far below zero. This is the non-inferiority reading of the
    same interval. ``within-margin`` needs the lower bound at or above
    ``-max_harm``; ``harm-exceeds-margin`` needs the upper bound below it; an
    interval that straddles the margin leaves harm ``harm-not-excluded``. An
    unbounded interval excludes nothing, so it can never clear the margin.
    """
    if not 0 < max_harm <= 1:
        raise ValueError("max_harm must be a pass-rate difference in (0, 1]")
    out: dict[str, Any] = {"max_harm": max_harm, "lower": None, "upper": None}
    if not interval.get("bounded"):
        return {**out, "verdict": HarmVerdict.UNBOUNDED.value,
                **({"reason": interval["reason"]} if interval.get("reason") else {})}
    lower, upper = float(interval["lower"]), float(interval["upper"])
    if lower >= -max_harm:
        verdict = HarmVerdict.WITHIN_MARGIN
    elif upper < -max_harm:
        verdict = HarmVerdict.HARM_EXCEEDS_MARGIN
    else:
        verdict = HarmVerdict.HARM_NOT_EXCLUDED
    return {**out, "verdict": verdict.value, "lower": lower, "upper": upper}


@dataclass(frozen=True)
class Estimate:
    """One paired effect: the test, its interval and the noise check, from one set of deltas.

    Every lift the harness reports is built here, so the p-value, the interval
    and the noise check can never be computed from different deltas or
    described in different units.
    """

    unit: InferenceUnit
    deltas: tuple[float, ...]
    significance: dict[str, Any]
    interval: dict[str, Any]
    noise: dict[str, Any] | None
    harm: dict[str, Any] | None = None

    @classmethod
    def from_deltas(cls, deltas: Sequence[float], *, unit: InferenceUnit,
                    without_rates: Sequence[float] | None = None,
                    min_lift: float | None = None,
                    max_harm: float | None = None,
                    alpha: float = DEFAULT_ALPHA) -> Estimate:
        values = tuple(float(value) for value in deltas)
        interval = sign_flip_interval(values, confidence=1 - alpha)
        noise = None
        if without_rates is not None:
            noise = noise_check(values, list(without_rates), interval=interval,
                                alpha=alpha, min_lift=min_lift)
        harm = None if max_harm is None else harm_check(interval, max_harm=max_harm)
        return cls(InferenceUnit(unit), values, sign_flip_test(values, alpha=alpha),
                   interval, noise, harm)

    @property
    def significant(self) -> bool:
        return bool(self.significance.get("significant_at_0_05"))

    def blocks(self) -> dict[str, Any]:
        """The report fields: ``significance``, ``interval``, ``noise_check``
        when the baseline rates were given and ``harm_check`` when a margin
        was, each naming its unit."""
        out: dict[str, Any] = {
            "significance": {**self.significance, "unit": self.unit.value},
            "interval": {**self.interval, "unit": self.unit.value},
        }
        if self.noise is not None:
            out["noise_check"] = {**self.noise, "unit": self.unit.value}
        if self.harm is not None:
            out["harm_check"] = {**self.harm, "unit": self.unit.value}
        return out
