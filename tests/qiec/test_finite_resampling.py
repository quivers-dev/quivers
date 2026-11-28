"""Verify the ordered ancestry law against independent finite calculations."""

from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Bind,
    ExactJvp,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    Perform,
    ResamplingParticle,
    SiteEstimator,
    plan_bernoulli_importance,
    plan_multinomial_resampling,
)

_SOURCE = """
instance random : Random

define likelihood(p : Real) : Real !{random} =
    let selected <- perform random.sample[Bool](site("weight"), Bernoulli(p))
    if selected then
        return 3.0
    else
        return 1.0

define objective(a : Real, b : Real, theta : Real) : Real !{} =
    return a * b + 2.0 * a - b * b + theta * a - 4.0

define raw(value : Real) : Real !{} =
    return value

define logged(value : Real) : Real !{} =
    return log(value)

define multi_step(a : Real, b : Real) : Real !{} =
    let x <- raw(a)
    return x - b

define branch(a : Real, b : Real) : Real !{} =
    if a > b then
        return a
    else
        return b
"""


def _jvp(value, tangent):
    return ExactJvp(Fraction(value), Fraction(tangent))


_PARTICLES = (
    ResamplingParticle(_jvp(2, 1), _jvp(Fraction(1, 2), 2)),
    ResamplingParticle(_jvp(2, -1), _jvp(Fraction(3, 2), -1)),
    ResamplingParticle(_jvp(-1, 2), _jvp(1, Fraction(1, 4))),
)
_POINT = Fraction(1, 3)
_DIRECTION = Fraction(-2, 3)


@pytest.fixture(scope="module")
def module():
    lowered = Compiler(parse(_SOURCE), module_name="finite-resampling").qiec_module
    assert lowered is not None
    return lowered


def _objective(a, b, theta):
    return a * b + 2 * a - b * b + theta * a - 4


def _stencil(function):
    return (-function(2) + 8 * function(1) - 8 * function(-1) + function(-2)) / 12


def _plan(module):
    return plan_multinomial_resampling(
        module,
        "objective",
        _PARTICLES,
        arguments=(_POINT,),
        direction=(_DIRECTION,),
    )


def test_ordered_multinomial_objective_matches_an_independent_rational_oracle(module):
    plan = _plan(module)
    assert plan.draws == 2 and plan.source_steps > 0
    assert plan.total_weight == _jvp(3, Fraction(5, 4))
    assert len(plan.ancestors) == len(_PARTICLES)
    assert len(plan.outcomes) == 9
    assert [item.position for item in plan.ancestors] == [0, 1, 2]
    assert plan.ancestors[0].value.value == plan.ancestors[1].value.value
    assert plan.ancestors[0].score != plan.ancestors[1].score

    def weight(index, t):
        particle = _PARTICLES[index]
        return particle.weight.value + t * particle.weight.tangent

    def value(index, t):
        particle = _PARTICLES[index]
        return particle.value.value + t * particle.value.tangent

    def denominator(t):
        return sum(weight(index, t) for index in range(3)) ** 2

    def cost(i, j, t):
        return _objective(value(i, t), value(j, t), _POINT + t * _DIRECTION)

    def numerator(t):
        return sum(
            weight(i, t) * weight(j, t) * cost(i, j, t)
            for i, j in product(range(3), repeat=2)
        )

    num, den = numerator(0), denominator(0)
    dnum, dden = _stencil(numerator), _stencil(denominator)
    assert plan.exact == _jvp(num / den, (dnum * den - num * dden) / den**2)
    assert plan.exact.tangent != sum(
        occurrence.probability.value * occurrence.objective.tangent
        for occurrence in plan.outcomes
    )
    assert plan.exact.tangent == sum(
        occurrence.probability.value * occurrence.estimate.tangent
        for occurrence in plan.outcomes
    )
    for (i, j), outcome in zip(product(range(3), repeat=2), plan.outcomes, strict=True):
        assert outcome.positions == (i, j)

        def raw(t):
            return weight(i, t) * weight(j, t)

        primal, tangent = raw(0), _stencil(raw)
        expected_score = tangent / primal - dden / den
        assert outcome.ancestry_score == expected_score
        assert outcome.probability == _jvp(
            primal / den, (tangent * den - primal * dden) / den**2
        )
        pointwise = cost(i, j, 0)
        direct = _stencil(lambda t: cost(i, j, t))
        assert outcome.objective == _jvp(pointwise, direct)
        assert outcome.estimate == _jvp(pointwise, direct + pointwise * expected_score)
    outcomes = {item.positions: item for item in plan.outcomes}
    pairs = outcomes[(0, 0)], outcomes[(0, 1)]
    assert pairs[0].objective.value == pairs[1].objective.value
    assert pairs[0].probability.value != pairs[1].probability.value
    assert pairs[0].ancestry_score != pairs[1].ancestry_score
    assert outcomes[(0, 2)].objective.value != outcomes[(2, 0)].objective.value
    assert pairs[0].ancestry_score == 2 * plan.ancestors[0].score
    assert pairs[0].estimate.tangent != (
        pairs[0].objective.tangent + pairs[0].objective.value * plan.ancestors[0].score
    )


def test_integer_sampler_has_exact_independent_slot_mass(module, monkeypatch):
    plan = _plan(module)
    seen = {item.positions: 0 for item in plan.outcomes}
    p, r = Fraction(1, 6), Fraction(3, 5)
    events = tuple(product(range(p.denominator), range(r.denominator)))

    for choices in product(events, repeat=2):
        expected, script = [], []
        for first, second in choices:
            script.append((p.denominator, first))
            if first < p.numerator:
                expected.append(0)
            else:
                script.append((r.denominator, second))
                expected.append(1 if second < r.numerator else 2)
        pending = iter(script)

        class Script:
            def randrange(self, stop):
                bound, draw = next(pending)
                assert bound == stop
                return draw

        monkeypatch.setattr(
            "quivers.qiec.finite_resampling.Random", lambda seed: Script()
        )
        outcome = plan.sample(seed=2)
        assert next(pending, None) is None
        assert outcome.positions == tuple(expected)
        seen[outcome.positions] += 1
    for outcome in plan.outcomes:
        assert (
            Fraction(seen[outcome.positions], len(events) ** 2)
            == outcome.probability.value
        )


def test_unbiased_raw_normalizer_does_not_make_its_log_unbiased(module):
    owner = next(item for item in module.computations if item.name == "likelihood")
    assert isinstance(owner.body, Bind) and isinstance(owner.body.first, Perform)
    site = owner.body.first.request.origin.static_site
    policy = SiteEstimator(site=site, estimator="enumerate")
    importance = plan_bernoulli_importance(
        module,
        owner.id,
        (Fraction(1, 2),),
        (0,),
        (policy,),
        proposal_arguments=(Fraction(1, 2),),
        proposal_policies=(policy,),
    )
    particles = tuple(
        ResamplingParticle(atom.estimate, _jvp(atom.proposal, 0))
        for atom in importance.outcomes
    )
    plan = plan_multinomial_resampling(module, "raw", particles)
    left, right = plan.outcomes
    assert plan.exact.value == importance.exact.value == 2
    assert left.probability.value == right.probability.value == Fraction(1, 2)
    assert (left.estimate.value, right.estimate.value) == (1, 3)
    # Strict concavity: log(sqrt(1*3)) < log(2), all exact before applying log.
    assert left.estimate.value * right.estimate.value < plan.exact.value**2
    with pytest.raises(JvpUnsupported, match="primitive"):
        plan_multinomial_resampling(module, "logged", particles)


def test_support_objective_boundary_and_actual_limits_are_checked(module):
    assert (
        len(
            plan_multinomial_resampling(
                module,
                "objective",
                _PARTICLES,
                arguments=(_POINT,),
                direction=(_DIRECTION,),
                max_outcomes=9,
            ).outcomes
        )
        == 9
    )
    with pytest.raises(JvpLimit, match="exceeds 8 ordered assignments"):
        plan_multinomial_resampling(
            module,
            "objective",
            _PARTICLES,
            arguments=(_POINT,),
            direction=(_DIRECTION,),
            max_outcomes=8,
        )
    for bad in (-1, 0):
        with pytest.raises(JvpUnsupported, match="positive particle weights"):
            plan_multinomial_resampling(
                module, "raw", (ResamplingParticle(_jvp(1, 0), _jvp(bad, 1)),)
            )
    with pytest.raises(JvpBoundaryError, match="equality boundary"):
        plan_multinomial_resampling(module, "branch", _PARTICLES)
    with pytest.raises(JvpLimit, match="budget of 1 steps"):
        plan_multinomial_resampling(module, "multi_step", _PARTICLES, fuel=1)
    with pytest.raises(ValueError, match="equal arity"):
        plan_multinomial_resampling(module, "objective", _PARTICLES, arguments=(1,))
    with pytest.raises(ValueError, match="one offspring slot"):
        plan_multinomial_resampling(
            module, "raw", _PARTICLES, arguments=(1,), direction=(0,)
        )
    with pytest.raises(ValueError, match="nonempty"):
        plan_multinomial_resampling(module, "raw", ())


def test_one_position_and_a_rare_position_keep_exact_scores(module, monkeypatch):
    only = plan_multinomial_resampling(
        module,
        "objective",
        (_PARTICLES[0],),
        arguments=(_POINT,),
        direction=(_DIRECTION,),
    )
    assert (only.draws, only.outcomes[0].positions) == (2, (0, 0))
    assert only.ancestors[0].score == 0
    assert only.sample(seed=17).estimate == only.exact

    tiny = Fraction(1, 2**220)
    rare = ResamplingParticle(_jvp(5, 0), _jvp(tiny, 1))
    usual = ResamplingParticle(_jvp(1, 0), _jvp(1 - tiny, -1))
    plan = plan_multinomial_resampling(module, "raw", (rare, usual))
    assert plan.ancestors[0].score == 2**220
    assert plan.outcomes[0].estimate.tangent == 5 * 2**220
    assert plan.exact.tangent == 4

    class DrawRare:
        def randrange(self, stop):
            assert stop == 2**220
            return 0

    monkeypatch.setattr(
        "quivers.qiec.finite_resampling.Random", lambda seed: DrawRare()
    )
    assert plan.sample(seed=17) == plan.outcomes[0]
