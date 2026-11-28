"""Exact signed clipping bias under actual score and weak selector randomness."""

from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Bind,
    ExactJvp,
    If,
    JvpLimit,
    Perform,
    SiteEstimator,
    SiteValue,
    audit_clipped_bernoulli_gradient,
)

_SOURCE = """
instance random : Random

define rare(p : Real, reward : Real) : Real !{random} =
    let bit <- perform random.sample[Bool](site("rare"), Bernoulli(p))
    if bit then
        return reward
    else
        return 0.0

define adaptive(p : Real, q : Real, s : Real) : Real !{random} =
    let b <- perform random.sample[Bool](site("outer"), Bernoulli(p))
    if b then
        let c <- perform random.sample[Bool](site("inner"), Bernoulli(q))
        if c then
            return 2.0 * s
        else
            return -s
    else
        return 3.0 * s
"""
_POINT = Fraction(2, 5), Fraction(1, 3), 2
_DIRECTION = (1, -2, 3)
_POLICIES = ("enumerate", "score", "measure")
_BOOL = (False, True)


@pytest.fixture(scope="module")
def module():
    result = Compiler(parse(_SOURCE), module_name="finite-clip").qiec_module
    assert result is not None
    return result


def _entry(module, name):
    return next(entry for entry in module.computations if entry.name == name)


def _sites(term):
    if isinstance(term, Bind):
        return _sites(term.first) | _sites(term.then)
    if isinstance(term, If):
        return _sites(term.then) | _sites(term.otherwise)
    if isinstance(term, Perform):
        label = term.request.arguments[0]
        assert isinstance(label, SiteValue)
        return {label.label: term.request.origin.static_site}
    return {}


def _policy(owner, **choices):
    return tuple(
        SiteEstimator(site=site, estimator=choices[name])
        for name, site in _sites(owner.body).items()
    )


@pytest.mark.parametrize("sign", (-1, 1))
def test_rare_clipping_preserves_signed_bias_and_the_checked_scope(module, sign):
    owner = _entry(module, "rare")
    p = Fraction(1, 2**200)
    values, direction, cap = (p, sign), (1, 0), 7
    policy = _policy(owner, rare="score")
    audit = audit_clipped_bernoulli_gradient(
        module, owner.id, values, direction, policy, clip_limit=cap
    )
    assert audit.module is module and audit.source.computation == owner.id
    assert (
        audit.point == values
        and audit.direction == direction
        and audit.clip_limit == cap
    )
    assert audit.source.exact == ExactJvp(p * sign, Fraction(sign))
    assert audit.mean == ExactJvp(p * sign, p * cap * sign)
    assert audit.signed_bias == sign * (p * cap - 1)
    common, rare = audit.outcomes
    assert common.probability == 1 - p and rare.probability == p
    assert common.raw.tangent == common.clipped.tangent == 0
    assert rare.raw.tangent == sign / p and rare.clipped.tangent == cap * sign
    assert rare.decisions[0].visit.address == audit.source.visits[0].address
    assert rare.decisions[0].visit.estimator == "score" and rare.decisions[0].choice


@pytest.mark.parametrize("reward", (-2, 2))
def test_exact_cap_boundary_and_zero_cap(module, reward):
    owner = _entry(module, "rare")
    policy = _policy(owner, rare="score")
    call = (module, owner.id, (Fraction(1, 4), reward), (1, 0), policy)
    tied = audit_clipped_bernoulli_gradient(*call, clip_limit=8)
    assert tied.signed_bias == 0
    assert tied.outcomes[1].raw == tied.outcomes[1].clipped
    smaller = audit_clipped_bernoulli_gradient(*call, clip_limit=Fraction(15, 2))
    assert smaller.signed_bias == -Fraction(reward, 16)
    zero = audit_clipped_bernoulli_gradient(*call, clip_limit=0)
    assert zero.signed_bias == -reward and zero.mean.tangent == 0
    assert zero.mean.value == tied.mean.value


def _finite_oracle(outer, inner):
    p, q, s = _POINT
    dp, dq, ds = _DIRECTION
    left, dl = 3 * s, 3 * ds
    right0, dr0 = -s, -ds
    right1, dr1 = 2 * s, 2 * ds

    def inside(choice):
        if choice is None:
            return (
                (1 - q) * right0 + q * right1,
                (1 - q) * dr0 + q * dr1 + dq * (right1 - right0),
            )
        value, tangent = (right1, dr1) if choice else (right0, dr0)
        if inner == "measure":
            return value, tangent + dq * (right1 - right0)
        return value, tangent + value * dq / (q if choice else q - 1)

    oracle = {}
    for first in (None,) if outer == "enumerate" else _BOOL:
        skip = outer == "score" and first is False
        seconds = (None,) if skip or inner == "enumerate" else _BOOL
        p_first = 1 if first is None else p if first else 1 - p
        for second in seconds:
            p_second = 1 if second is None else q if second else 1 - q
            if skip:
                value, tangent = left, dl - left * dp / (1 - p)
            else:
                right, dr = inside(second)
                if outer == "enumerate":
                    value = (1 - p) * left + p * right
                    tangent = (1 - p) * dl + p * dr + dp * (right - left)
                elif outer == "score":
                    value, tangent = right, dr + right * dp / p
                else:
                    value, tangent = (right, dr) if first else (left, dl)
                    tangent += dp * (right - left)
            script = tuple(bit for bit in (first, second) if bit is not None)
            locations = tuple(
                prefix
                for bit, prefix in ((first, ()), (second, (True,)))
                if bit is not None
            )
            oracle[script] = (
                ExactJvp(Fraction(value), Fraction(tangent)),
                p_first * p_second,
                locations,
            )
    return oracle


@pytest.mark.parametrize(("outer", "inner"), product(_POLICIES, repeat=2))
def test_audit_enumerates_the_estimator_randomness_instead_of_source_paths(
    module,
    outer,
    inner,
):
    owner = _entry(module, "adaptive")
    policy = _policy(owner, outer=outer, inner=inner)
    audit = audit_clipped_bernoulli_gradient(
        module,
        owner.id,
        _POINT,
        _DIRECTION,
        policy,
        clip_limit=4,
        max_paths=3,
    )
    expected = _finite_oracle(outer, inner)
    actual = {
        tuple(draw.choice for draw in item.decisions): item for item in audit.outcomes
    }
    assert actual.keys() == expected.keys()
    assert len(audit.outcomes) == len(expected)
    assert audit.source.paths == 3
    expected_mean = Fraction(0)
    for key, (raw, probability, positions) in expected.items():
        output = actual[key]
        want = 4 if raw.tangent > 4 else -4 if raw.tangent < -4 else raw.tangent
        assert output.raw == raw
        assert output.clipped == ExactJvp(raw.value, Fraction(want))
        assert output.probability == probability
        assert tuple(item.visit.prefix for item in output.decisions) == positions
        expected_mean += probability * want
    assert audit.mean.tangent == expected_mean
    assert audit.signed_bias == expected_mean - audit.source.exact.tangent
    for seed in (1, 2, 7):
        sampled = audit.sample(seed=seed)
        uncapped = audit.source.sample(seed=seed)
        assert sampled.value == uncapped.value
        assert sampled.tangent == max(-4, min(4, uncapped.tangent))
        assert (sampled.draws, sampled.selectors, sampled.virtual_branches) == (
            uncapped.draws,
            uncapped.selectors,
            uncapped.virtual_branches,
        )


def test_source_rebinding_caps_and_work_limits_are_checked(module):
    owner = _entry(module, "adaptive")
    policies = _policy(owner, outer="measure", inner="score")
    arguments = (module, owner.id, _POINT, _DIRECTION, policies)
    result = audit_clipped_bernoulli_gradient(*arguments, clip_limit=4, max_outcomes=4)
    assert len(result.outcomes) == 4
    with pytest.raises(JvpLimit, match="exceeds 3 complete estimator scripts"):
        audit_clipped_bernoulli_gradient(*arguments, clip_limit=4, max_outcomes=3)
    with pytest.raises(JvpLimit, match="exceeds 2 complete paths"):
        audit_clipped_bernoulli_gradient(*arguments, clip_limit=4, max_paths=2)
    with pytest.raises(ValueError, match="nonnegative"):
        audit_clipped_bernoulli_gradient(*arguments, clip_limit=-1)
    alternate = Compiler(
        parse(_SOURCE.replace("return 3.0 * s", "return 4.0 * s")),
        module_name="finite-clip",
    ).qiec_module
    assert alternate is not None
    revised = _entry(alternate, "adaptive")
    assert revised.id == owner.id
    fresh = audit_clipped_bernoulli_gradient(
        alternate,
        revised.id,
        _POINT,
        _DIRECTION,
        _policy(revised, outer="measure", inner="score"),
        clip_limit=4,
    )
    assert fresh.module is alternate
    assert fresh.source.exact != result.source.exact
    assert fresh.signed_bias != result.signed_bias
