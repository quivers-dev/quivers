"""Check interventions and observation in an exact confounded source model."""

from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    AddressedBernoulliEvent,
    Bind,
    If,
    JvpLimit,
    JvpUnsupported,
    Perform,
    QiecModule,
    SiteEstimator,
    SiteValue,
    compare_bernoulli_do_and_condition,
)

_SOURCE = """
instance random : Random

define propensity(c : Bool, p : Real, q : Real) : Real !{} =
    if c then
        return q
    else
        return p

define response(c : Bool, treated : Bool, p : Real, b : Real) : Real !{} =
    if treated then
        if c then
            return (1.0 + b + p) / 3.0
        else
            return (b + p) / 4.0
    else
        if c then
            return (1.0 + b) / 3.0
        else
            return b / 4.0

define model(alpha : Real, p : Real, q : Real, b : Real, e : Real) : Real !{random} =
    let c <- perform random.sample[Bool](site("confounder"), Bernoulli(alpha))
    let propensity_value <- propensity(c, p, q)
    let t <- perform random.sample[Bool](site("treatment"), Bernoulli(propensity_value))
    let response_value <- response(c, t, p, b)
    let y <- perform random.sample[Bool](site("outcome"), Bernoulli(response_value))
    if y then
        if c then
            return 2.0 + e * p
        else
            return 1.0 + e
    else
        if c then
            return 3.0 + p
        else
            return e * p + 1.0

define draw(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("repeat"), Bernoulli(p))

define repeated(p : Real, q : Real) : Real !{random} =
    let first <- draw(p)
    let second <- draw(q)
    return 1.0

define absent(p : Real) : Real !{random} =
    let earlier <- perform random.sample[Bool](site("earlier"), Bernoulli(p))
    if earlier then
        let later <- perform random.sample[Bool](site("later"), Bernoulli(p))
        return 1.0
    else
        return 1.0
"""
_BOOL = (False, True)
_POINT = (
    Fraction(1, 3),
    Fraction(1, 4),
    Fraction(2, 3),
    Fraction(1, 2),
    Fraction(3, 2),
)
_DIRECTION = (2, -1, 3, Fraction(2, 3), -2)


@pytest.fixture(scope="module")
def module() -> QiecModule:
    module = Compiler(parse(_SOURCE), module_name="finite-causal").qiec_module
    assert module is not None
    return module


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _policies(node):
    if isinstance(node, Bind):
        return _policies(node.first) | _policies(node.then)
    if isinstance(node, If):
        return _policies(node.then) | _policies(node.otherwise)
    if isinstance(node, Perform):
        site = node.request.arguments[0]
        assert isinstance(site, SiteValue)
        return {
            site.label: SiteEstimator(
                site=node.request.origin.static_site, estimator="enumerate"
            )
        }
    return {}


def _event(policy, value, scopes=()):
    return AddressedBernoulliEvent((str(policy.site), scopes, ()), value)


def _product(point, c, t, y, *, intervention, keep_evidence=True):
    alpha, p, q, b, e = point
    a = q if c else p
    r = ((1 + b + p * t) / 3) if c else ((b + p * t) / 4)
    f = (2 + e * p if c else 1 + e) if y else (3 + p if c else e * p + 1)
    return (
        (alpha if c else 1 - alpha)
        * (1 if intervention else a if t else 1 - a)
        * (r if y else 1 - r)
        * (f if keep_evidence else 1)
    )


def _derivative(function, point, direction):
    # The polynomial in the perturbation has degree at most five.
    def at(t):
        return function(
            tuple(x + t * dx for x, dx in zip(point, direction, strict=True))
        )

    return Fraction(45 * (at(1) - at(-1)) - 9 * (at(2) - at(-2)) + at(3) - at(-3), 60)


def _model(module, t, y):
    owner = _entry(module, "model")
    policies = _policies(owner.body)
    return (
        owner.id,
        tuple(policies.values()),
        _event(policies["treatment"], t),
        _event(policies["outcome"], y),
    )


@pytest.mark.parametrize(("t", "y"), product(_BOOL, repeat=2))
def test_confounded_source_matches_two_distinct_exact_posterior_queries(module, t, y):
    owner, policies, treatment, outcome = _model(module, t, y)
    result = compare_bernoulli_do_and_condition(
        module,
        owner,
        _POINT,
        _DIRECTION,
        policies,
        treatment=treatment,
        outcome=outcome,
    )
    assert result.source_steps > 0
    assert len(result.histories) == 4
    assert result.conditioned != result.intervened
    assert result.conditioned.value != result.intervened.value
    assert result.conditioned.tangent != result.intervened.tangent
    for mode, fit, reported_evidence in (
        (False, result.conditioned, result.observational_evidence),
        (True, result.intervened, result.interventional_evidence),
    ):

        def evidence(point):
            return sum(
                _product(point, c, t, bit, intervention=mode)
                for c, bit in product(_BOOL, repeat=2)
            )

        def match(point):
            return sum(_product(point, c, t, y, intervention=mode) for c in _BOOL)

        mass, selected = evidence(_POINT), match(_POINT)
        dm = _derivative(evidence, _POINT, _DIRECTION)
        ds = _derivative(match, _POINT, _DIRECTION)
        score = (ds * mass - dm * selected) / mass**2
        assert (reported_evidence.value, reported_evidence.tangent) == (mass, dm)
        assert (fit.value, fit.tangent) == (selected / mass, score)
        for atom, (c, bit) in zip(
            result.histories, product(_BOOL, repeat=2), strict=True
        ):
            assert tuple(choice for _, choice in atom.history) == (c, t, bit)
            assert atom.matches_outcome == (bit == y)
            weight = atom.interventional_weight if mode else atom.observational_weight

            def branch(point):
                return _product(point, c, t, bit, intervention=mode)

            assert (weight.value, weight.tangent) == (
                branch(_POINT),
                _derivative(branch, _POINT, _DIRECTION),
            )
        lost_evidence = sum(
            _product(_POINT, c, t, y, intervention=mode, keep_evidence=False)
            for c in _BOOL
        ) / sum(
            _product(_POINT, c, t, bit, intervention=mode, keep_evidence=False)
            for c, bit in product(_BOOL, repeat=2)
        )
        assert fit.value != lost_evidence


def test_only_the_intervened_probability_and_its_score_are_removed(module):
    owner, policies, treatment, outcome = _model(module, True, True)
    direction = (0, 0, 1, 0, 0)  # q affects the treated kernel on one history only.
    answer = compare_bernoulli_do_and_condition(
        module, owner, _POINT, direction, policies, treatment=treatment, outcome=outcome
    )
    assert answer.conditioned.tangent != 0
    assert answer.intervened.tangent == 0
    assert answer.interventional_evidence.tangent == 0
    for mode, fit in ((False, answer.conditioned), (True, answer.intervened)):

        def derivative_with_extra_treatment_score(multiplicity):
            rates = [
                (
                    atom.interventional_weight if mode else atom.observational_weight,
                    atom.matches_outcome,
                    atom.history[0][1],
                )
                for atom in answer.histories
            ]
            total = sum(weight.value for weight, _, _ in rates)
            posterior = sum(weight.value for weight, match, _ in rates if match) / total
            return (
                sum(
                    (int(match) - posterior)
                    * (
                        weight.tangent
                        + multiplicity * weight.value * int(c) / _POINT[2]
                    )
                    for weight, match, c in rates
                )
                / total
            )

        assert derivative_with_extra_treatment_score(0) == fit.tangent
        assert all(
            derivative_with_extra_treatment_score(count) != fit.tangent
            for count in (-1, 1)
        )

    shared = compare_bernoulli_do_and_condition(
        module,
        owner,
        _POINT,
        (0, 1, 0, 0, 0),
        policies,
        treatment=treatment,
        outcome=outcome,
    )
    assert shared.intervened.tangent != 0
    assert shared.interventional_evidence.tangent != 0


def test_addressing_one_call_preserves_the_other_call_to_the_same_site(module):
    owner = _entry(module, "repeated")
    policy = _policies(_entry(module, "draw").body)["repeat"]
    result = compare_bernoulli_do_and_condition(
        module,
        owner.id,
        (Fraction(1, 4), Fraction(2, 3)),
        (2, -1),
        (policy,),
        treatment=_event(policy, True, (("call", "draw#1"),)),
        outcome=_event(policy, True, (("call", "draw#2"),)),
    )
    assert result.conditioned == result.intervened
    assert (result.intervened.value, result.intervened.tangent) == (Fraction(2, 3), -1)
    assert (
        result.observational_evidence.value,
        result.observational_evidence.tangent,
    ) == (Fraction(1, 4), 2)
    assert (
        result.interventional_evidence.value,
        result.interventional_evidence.tangent,
    ) == (1, 0)


def test_refuses_missing_or_non_boolean_addresses_and_budget(module):
    owner = _entry(module, "absent")
    policies = _policies(owner.body)
    call = (module, owner.id, (Fraction(1, 2),), (1,), tuple(policies.values()))
    earlier = _event(policies["earlier"], True)
    later = _event(policies["later"], True)
    with pytest.raises(JvpUnsupported, match="treatment address must occur on every"):
        compare_bernoulli_do_and_condition(*call, treatment=later, outcome=earlier)
    with pytest.raises(JvpUnsupported, match="outcome address must occur on every"):
        compare_bernoulli_do_and_condition(*call, treatment=earlier, outcome=later)
    with pytest.raises(ValueError, match="must be Boolean"):
        AddressedBernoulliEvent(earlier.address, 1)  # type: ignore[arg-type]
    with pytest.raises(JvpLimit, match="complete paths"):
        compare_bernoulli_do_and_condition(
            *call, treatment=earlier, outcome=earlier, max_paths=1
        )
