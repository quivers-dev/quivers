"""Exact source marginal scores, posterior directions, and support failures."""

from fractions import Fraction

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    Perform,
    SiteEstimator,
    SiteId,
    marginalize_bernoulli_unit,
)

_SOURCE = """
instance random : Random

define coin(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("coin"), Bernoulli(p))

define adaptive(p : Real, x : Real) : Real !{random} =
    let selected <- coin(p)
    if selected then
        let continuation <- coin(p * x)
        if continuation then
            return x * x + p
        else
            return 1.0 + x * p
    else
        return 2.0 + x

define nonpositive(p : Real, x : Real) : Real !{random} =
    let selected <- coin(p)
    if selected then
        return x
    else
        return 3.0

define zero_stationary(p : Real, x : Real) : Real !{random} =
    let selected <- coin(p)
    if selected then
        return x * x
    else
        return 3.0

define switching(p : Real, x : Real) : Real !{random} =
    let selected <- coin(p)
    if selected then
        if x > 0.0 then
            return 1.0
        else
            return 2.0
    else
        return 3.0

define neutral_response(p : Real) : Real !{random} =
    let selected <- coin(p)
    if selected then
        return 2.0
    else
        return 2.0
"""
_PATHS = ((False,), (True, False), (True, True))


@pytest.fixture(scope="module")
def module():
    lowered = Compiler(parse(_SOURCE), module_name="finite-fisher").qiec_module
    assert lowered is not None
    return lowered


def _entry(module, name):
    return next(comp for comp in module.computations if comp.name == name)


def _policy(module):
    term = _entry(module, "coin").body
    assert isinstance(term, Perform)
    return SiteEstimator(site=term.request.origin.static_site, estimator="enumerate")


def _prior(parameters):
    p, x = parameters
    return 1 - p, p * (1 - p * x), p * p * x


def _observations(parameters):
    p, x = parameters
    return 2 + x, 1 + p * x, x * x + p


def _directional(function, point, tangent):
    # Exact centered differentiation on this degree-at-most-five line.
    def at(t):
        return function(tuple(p + t * v for p, v in zip(point, tangent, strict=True)))

    return Fraction(45 * (at(1) - at(-1)) - 9 * (at(2) - at(-2)) + at(3) - at(-3), 60)


@pytest.mark.parametrize(
    ("point", "direction"),
    [
        ((Fraction(2, 5), Fraction(3, 2)), (Fraction(-3, 5), 2)),
        ((Fraction(1, 8), 3), (3, -2)),
        ((Fraction(3, 4), Fraction(1, 4)), (-3, Fraction(4, 3))),
    ],
)
def test_fisher_scores_agree_with_exact_evidence_and_normalized_path_derivatives(
    module,
    point,
    direction,
):
    owner = _entry(module, "adaptive")
    result = marginalize_bernoulli_unit(
        module, owner.id, point, direction, (_policy(module),), max_paths=3
    )

    def weights(parameters):
        return tuple(
            prior * likelihood
            for prior, likelihood in zip(
                _prior(parameters), _observations(parameters), strict=True
            )
        )

    weights_at_point = weights(point)
    expected_value = sum(weights_at_point)
    expected_dz = _directional(lambda at: sum(weights(at)), point, direction)
    assert (result.normalizer.value, result.normalizer.tangent) == (
        expected_value,
        expected_dz,
    )
    assert result.marginal_score == expected_dz / expected_value
    assert [item.prefix for item in result.outcomes] == list(_PATHS)
    assert result.source_steps > 0 and result.replays == 5
    for index, outcome in enumerate(result.outcomes):
        prior, likelihood = _prior(point)[index], _observations(point)[index]
        dp = _directional(lambda at: _prior(at)[index], point, direction)
        dl = _directional(lambda at: _observations(at)[index], point, direction)
        dw = _directional(lambda at: weights(at)[index], point, direction)
        w = weights_at_point[index]
        mass = w / expected_value
        assert (outcome.prior.value, outcome.prior.tangent) == (prior, dp)
        assert (outcome.observation.value, outcome.observation.tangent) == (
            likelihood,
            dl,
        )
        assert (outcome.joint.value, outcome.joint.tangent) == (w, dw)
        assert outcome.complete_score == dp / prior + dl / likelihood
        assert outcome.complete_score == dw / w
        assert (outcome.posterior.value, outcome.posterior.tangent) == (
            mass,
            (dw * expected_value - w * expected_dz) / expected_value**2,
        )
    assert sum(item.prior.value for item in result.outcomes) == 1
    assert sum(item.posterior.value for item in result.outcomes) == 1
    assert sum(item.posterior.tangent for item in result.outcomes) == 0
    assert (
        sum(item.posterior.value * item.complete_score for item in result.outcomes)
        == result.marginal_score
    )
    prior_log_term = sum(
        item.posterior.value * item.prior.tangent / item.prior.value
        for item in result.outcomes
    )
    observation_log_term = sum(
        item.posterior.value * item.observation.tangent / item.observation.value
        for item in result.outcomes
    )
    assert prior_log_term != 0 and observation_log_term != 0
    assert result.marginal_score not in (
        prior_log_term,
        observation_log_term,
        2 * prior_log_term + observation_log_term,
        prior_log_term + 2 * observation_log_term,
    )


@pytest.mark.parametrize(
    ("name", "weight"),
    [("nonpositive", 0), ("nonpositive", -1), ("zero_stationary", 0)],
)
def test_positive_marginal_cannot_license_a_nonpositive_complete_history(
    module, name, weight
):
    owner = _entry(module, name)
    with pytest.raises(
        JvpUnsupported, match=r"positive observation factors.*\(True,\)"
    ):
        marginalize_bernoulli_unit(
            module, owner.id, (Fraction(1, 4), weight), (1, 0), (_policy(module),)
        )


def test_refuses_changed_support_unstable_control_and_undeclared_policies(module):
    owner = _entry(module, "switching")
    policy = _policy(module)
    with pytest.raises(JvpUnsupported, match="strictly between zero and one"):
        marginalize_bernoulli_unit(module, owner.id, (0, 1), (0, 1), (policy,))
    with pytest.raises(JvpBoundaryError, match="equality boundary"):
        marginalize_bernoulli_unit(
            module, owner.id, (Fraction(1, 2), 0), (0, 1), (policy,)
        )
    active = (module, owner.id, (Fraction(1, 2), 1), (0, 1))
    with pytest.raises(JvpUnsupported, match="every site"):
        marginalize_bernoulli_unit(*active, (policy.with_(estimator="score"),))
    with pytest.raises(JvpUnsupported, match="missing estimator"):
        marginalize_bernoulli_unit(*active, ())
    with pytest.raises(JvpUnsupported, match="unreached sites"):
        marginalize_bernoulli_unit(
            *active,
            (policy, policy.with_(site=SiteId.derive("unused"))),
        )
    with pytest.raises(JvpLimit, match="exceeds 1 complete paths"):
        marginalize_bernoulli_unit(*active, (policy,), max_paths=1)
    with pytest.raises(JvpLimit, match="step budget of 1"):
        marginalize_bernoulli_unit(*active, (policy,), fuel=1)


def test_tiny_positive_prior_keeps_large_signed_complete_score(module):
    owner = _entry(module, "neutral_response")
    point = Fraction(1, 2**200)
    result = marginalize_bernoulli_unit(
        module, owner.id, (point,), (1,), (_policy(module),)
    )
    no, yes = result.outcomes
    assert (
        result.normalizer.value,
        result.normalizer.tangent,
        result.marginal_score,
    ) == (2, 0, 0)
    assert (yes.complete_score, yes.posterior.value, yes.posterior.tangent) == (
        2**200,
        point,
        1,
    )
    assert (no.complete_score, no.posterior.value, no.posterior.tangent) == (
        -1 / (1 - point),
        1 - point,
        -1,
    )
