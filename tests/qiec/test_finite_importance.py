"""Exercise held-fixed proposal actions, exact coverage and sampling."""

from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Bind,
    If,
    ImportanceCoverageError,
    JvpLimit,
    JvpUnsupported,
    Perform,
    SiteEstimator,
    SiteValue,
    plan_bernoulli_importance,
)

_SOURCE = """
instance random : Random

define coin(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("coin"), Bernoulli(p))

define observed(p : Real, x : Real, tilt : Real) : Real !{random} =
    let a <- coin(p)
    if a then
        let b <- coin(p * x)
        if b then
            return tilt + x * x + p
        else
            return tilt + 1.0 + p * x
    else
        return tilt + 2.0 + x

define alternative(p : Real, side : Real) : Real !{random} =
    if side > 0.0 then
        let bit <- perform random.sample[Bool](site("left"), Bernoulli(p))
        if bit then
            return 3.0
        else
            return 2.0
    else
        let bit <- perform random.sample[Bool](site("right"), Bernoulli(p))
        if bit then
            return 3.0
        else
            return 2.0

define baseline() : Real !{} =
    return 2.0

define shifted(p : Real, side : Real) : Real !{random} =
    if side > 0.0 then
        let y <- baseline()
        let a <- coin(p)
        if a then
            return y + 1.0
        else
            return y
    else
        let a <- coin(p)
        if a then
            return 3.0
        else
            return 2.0

define rare(p : Real) : Real !{random} =
    let a <- coin(p)
    if a then
        return 3.0
    else
        return 1.0
"""


@pytest.fixture(scope="module")
def module():
    result = Compiler(parse(_SOURCE), module_name="finite-importance").qiec_module
    assert result is not None
    return result


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _policies(term):
    if isinstance(term, Bind):
        return _policies(term.first) | _policies(term.then)
    if isinstance(term, If):
        return _policies(term.then) | _policies(term.otherwise)
    if isinstance(term, Perform):
        label = term.request.arguments[0]
        assert isinstance(label, SiteValue)
        return {
            label.label: SiteEstimator(
                site=term.request.origin.static_site, estimator="enumerate"
            )
        }
    return {}


def _proposal(p, x):
    r = p * x
    return 1 - p, p * (1 - r), p * r


def _joint(point):
    p, x, tilt = point
    factors = tilt + 2 + x, tilt + 1 + p * x, tilt + x * x + p
    return tuple(
        q * likelihood for q, likelihood in zip(_proposal(p, x), factors, strict=True)
    )


def _directional(function, point, direction):
    def at(step):
        return function(
            tuple(value + step * d for value, d in zip(point, direction, strict=True))
        )

    return Fraction(45 * (at(1) - at(-1)) - 9 * (at(2) - at(-2)) + at(3) - at(-3), 60)


_POINT = (Fraction(1, 3), Fraction(3, 2), 1)
_DIRECTION = (Fraction(-2, 3), 1, 2)
_PROPOSAL = (Fraction(2, 7), 3, -10)


def _plan(module, proposal):
    owner = _entry(module, "observed")
    policy = _policies(_entry(module, "coin").body)["coin"]
    return plan_bernoulli_importance(
        module,
        owner.id,
        _POINT,
        _DIRECTION,
        (policy,),
        proposal_arguments=proposal,
        proposal_policies=(policy,),
    )


@pytest.mark.parametrize("proposal", [_PROPOSAL, _POINT, (Fraction(3, 5), 1, 5)])
def test_original_address_coverage_and_held_fixed_weight_oracle(module, proposal):
    plan = _plan(module, proposal)
    q = _proposal(*proposal[:2])
    weights = _joint(_POINT)
    slopes = tuple(
        _directional(lambda point: _joint(point)[index], _POINT, _DIRECTION)
        for index in range(3)
    )
    expected_guards = ((False,), (True, False), (True, True))
    assert plan.coverage.computation == _entry(module, "observed").id
    assert plan.coverage.target_point == _POINT
    assert plan.coverage.proposal_point == proposal
    assert plan.coverage.histories == tuple(item.history for item in plan.outcomes)
    assert sum(q) == 1
    assert plan.target_steps > 0 and plan.proposal_steps > 0
    assert (plan.exact.value, plan.exact.tangent) == (sum(weights), sum(slopes))
    policy = _policies(_entry(module, "coin").body)["coin"]
    for index, atom in enumerate(plan.outcomes):
        assert tuple(choice for _, choice in atom.history) == expected_guards[index]
        assert tuple(address for address, _ in atom.history) == tuple(
            (str(policy.site), (("call", f"coin#{number}"),), ())
            for number in range(1, len(atom.history) + 1)
        )
        assert atom.proposal == q[index]
        assert (atom.target.value, atom.target.tangent) == (
            weights[index],
            slopes[index],
        )
        assert (atom.estimate.value, atom.estimate.tangent) == (
            weights[index] / q[index],
            slopes[index] / q[index],
        )
    assert sum(item.proposal * item.estimate.value for item in plan.outcomes) == sum(
        weights
    )
    assert sum(item.proposal * item.estimate.tangent for item in plan.outcomes) == sum(
        slopes
    )
    wrong = sum(
        atom.proposal
        * (
            atom.estimate.tangent
            - atom.estimate.value
            * _directional(lambda point: _proposal(*point[:2])[i], proposal, _DIRECTION)
            / atom.proposal
        )
        for i, atom in enumerate(plan.outcomes)
    )
    assert wrong != plan.exact.tangent


def test_proposal_ignores_its_observation_factor_and_has_exact_draws(
    module, monkeypatch
):
    plan = _plan(module, _PROPOSAL)
    same_prior = _plan(module, (*_PROPOSAL[:2], 42))
    assert plan.outcomes == same_prior.outcomes
    p, x = _PROPOSAL[:2]
    r = p * x
    sizes = p.denominator, r.denominator
    seen = {outcome.history: 0 for outcome in plan.outcomes}
    for first, second in product(*(range(size) for size in sizes)):
        positions = iter((first, second))

        class Controlled:
            def randrange(self, stop):
                index = next(positions)
                assert 0 <= index < stop
                return index

        monkeypatch.setattr(
            "quivers.qiec.finite_importance.Random", lambda seed: Controlled()
        )
        result = plan.sample(seed=19)
        prefix = tuple(bit for _, bit in result.history)
        expected = (True, second < r.numerator) if first < p.numerator else (False,)
        assert prefix == expected
        seen[result.history] += 1
    for result in plan.outcomes:
        assert (
            Fraction(seen[result.history], p.denominator * r.denominator)
            == result.proposal
        )


@pytest.mark.parametrize("name", ["alternative", "shifted"])
def test_positive_target_atom_has_a_canonical_uncovered_witness(module, name):
    owner = _entry(module, name)
    sites = _policies(owner.body)
    if name == "alternative":
        target, proposal = sites["left"], sites["right"]
    else:
        target = proposal = _policies(_entry(module, "coin").body)["coin"]
    point = (Fraction(2, 5), 1)
    with pytest.raises(
        ImportanceCoverageError, match="does not cover positive target history"
    ) as caught:
        plan_bernoulli_importance(
            module,
            owner.id,
            point,
            (1, 0),
            (target,),
            proposal_arguments=(point[0], -1),
            proposal_policies=(proposal,),
        )
    assert caught.value.target_weight == Fraction(6, 5)
    ((address, choice),) = caught.value.history
    assert choice is False and address[0] == str(target.site)
    if name == "shifted":
        assert address[1] == (("call", "coin#2"),)


def test_invalid_proposal_support_and_budget_refuse(module):
    owner = _entry(module, "rare")
    policy = _policies(_entry(module, "coin").body)["coin"]
    call = (module, owner.id, (Fraction(1, 4),), (1,), (policy,))
    with pytest.raises(JvpUnsupported, match="strictly between zero and one"):
        plan_bernoulli_importance(
            *call,
            proposal_arguments=(0,),
            proposal_policies=(policy,),
        )
    with pytest.raises(JvpUnsupported, match="every site"):
        plan_bernoulli_importance(
            *call,
            proposal_arguments=(Fraction(1, 4),),
            proposal_policies=(policy.with_(estimator="measure"),),
        )
    with pytest.raises(JvpLimit, match="complete paths"):
        plan_bernoulli_importance(
            *call,
            proposal_arguments=(Fraction(1, 4),),
            proposal_policies=(policy,),
            max_paths=1,
        )
    observed = _entry(module, "observed")
    with pytest.raises(JvpUnsupported, match="positive observation factors"):
        plan_bernoulli_importance(
            module,
            observed.id,
            (*_POINT[:2], -10),
            _DIRECTION,
            (policy,),
            proposal_arguments=_PROPOSAL,
            proposal_policies=(policy,),
        )


def test_rare_proposal_draw_preserves_the_raw_weight_and_direction(module):
    owner = _entry(module, "rare")
    policy = _policies(_entry(module, "coin").body)["coin"]
    denominator = 2**190
    plan = plan_bernoulli_importance(
        module,
        owner.id,
        (Fraction(1, 4),),
        (1,),
        (policy,),
        proposal_arguments=(Fraction(1, denominator),),
        proposal_policies=(policy,),
    )
    assert (plan.exact.value, plan.exact.tangent) == (Fraction(3, 2), 2)
    rare = plan.outcomes[1]
    assert rare.estimate.value == Fraction(3 * denominator, 4)
    assert rare.estimate.tangent == 3 * denominator
