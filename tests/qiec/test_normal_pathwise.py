"""Exact Normal pathwise results against ordinary Gaussian moment identities."""

from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EMPTY_ROW,
    RANDOM_SAMPLE,
    Bind,
    ComputationId,
    Handle,
    If,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    Perform,
    QiecModule,
    RuntimeConfiguration,
    RuntimeSelection,
    SiteEstimator,
    SiteId,
    SiteValue,
    host_value,
    plan_normal_pathwise,
    run_named,
)

_SOURCE = """
instance random : Random

handler replayed for Random : Real -> Real [coverage=total, implementation=foreign]
    sample resumes 1

define affine_pair(m : Real, s : Real, t : Real, k : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(m, s))
    let y <- perform random.sample[Real](site("y"), Normal(k * x, t))
    return x * y + y * y

define regular_branch(theta : Real, s : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(0.0, s))
    if theta > 0.0 then
        return (x * x) / (theta + 1.0)
    else
        return x / 2.0

define indicator(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(theta, 1.0))
    if x > 0.0 then
        return 1.0
    else
        return 0.0

define locally_cancelled_guard(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(0.0, 1.0))
    if theta * theta * x > 0.0 then
        return 1.0
    else
        return 0.0

define random_scale(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(theta, 1.0))
    perform random.sample[Real](site("y"), Normal(0.0, x * x + 1.0))

define random_divisor(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(theta, 1.0))
    return 1.0 / (1.0 + x * x)

define zero_divisor(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(theta, 1.0))
    return x / theta

define moments(m : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(m, 1.0))
    return pow(x, 4.0) + pow(x, 3.0)

define reciprocal(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(0.0, 1.0))
    return pow(theta, -2.0) * x * x

define negative_noise_power(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(theta, 1.0))
    return pow(1.0 + x * x, -1.0)

define dynamic_power(theta : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(0.0, 1.0))
    return pow(x, theta)

define draw(m : Real, s : Real) : Real !{random} =
    perform random.sample[Real](site("both"), Normal(m, s))

define difference(m : Real, s : Real) : Real !{random} =
    let x <- draw(m, s)
    let y <- draw(m, s)
    return (x - y) * (x - y)

define wrong_family(theta : Real) : Real !{random} =
    perform random.sample[Real](site("uniform"), Uniform(0.0, theta))

define non_scalar_request(p : Real) : Real !{random} =
    let chosen <- perform random.sample[Bool](site("bool"), Bernoulli(p))
    if chosen then
        return 1.0
    else
        return 0.0
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    lowered = Compiler(parse(_SOURCE), module_name="normal-pathwise-test").qiec_module
    assert lowered is not None
    return lowered


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _sites(term):
    if isinstance(term, Bind):
        yield from _sites(term.first)
        yield from _sites(term.then)
    if isinstance(term, If):
        yield from _sites(term.then)
        yield from _sites(term.otherwise)
    if isinstance(term, Perform):
        label = term.request.arguments[0]
        assert isinstance(label, SiteValue)
        yield label.label, term.request.origin.static_site


def _policies(module, name):
    return tuple(
        SiteEstimator(site=site, estimator="reparam")
        for _, site in _sites(_entry(module, name).body)
    )


@pytest.mark.parametrize(
    ("point", "direction"),
    [
        ((Fraction(2, 3), 2, Fraction(1, 2), -2), (1, -2, 3, 4)),
        ((-2, Fraction(3, 2), 2, Fraction(1, 3)), (2, 3, -1, 2)),
        ((0, 1, 1, 1), (0, 1, 1, -1)),
    ],
)
def test_adaptive_location_matches_independent_exact_moments_and_pointwise_action(
    module,
    point,
    direction,
):
    owner = _entry(module, "affine_pair")
    plan = plan_normal_pathwise(
        module, owner.id, point, direction, _policies(module, "affine_pair")
    )
    m, s, t, k = point
    dm, ds, dt, dk = direction
    second_moment = m * m + s * s
    expected_value = (k + k * k) * second_moment + t * t
    expected_tangent = (
        (1 + 2 * k) * dk * second_moment
        + (k + k * k) * (2 * m * dm + 2 * s * ds)
        + 2 * t * dt
    )
    assert (plan.exact.value, plan.exact.tangent) == (expected_value, expected_tangent)
    assert [visit.scale for visit in plan.visits] == [s, t]
    assert [visit.scale_tangent for visit in plan.visits] == [ds, dt]
    assert len({visit.address for visit in plan.visits}) == 2
    for noise in product((Fraction(-3, 2), 0, 2), repeat=2):
        zx, zy = noise
        x, dx = m + s * zx, dm + ds * zx
        y, dy = k * x + t * zy, dk * x + k * dx + dt * zy
        action = plan.at_noise(noise)
        assert (action.value, action.tangent) == (
            x * y + y * y,
            dx * y + (x + 2 * y) * dy,
        )
    assert plan.source_steps > 0 and plan.polynomial_work > 0


def test_gaussian_moments_integrate_odd_terms_and_differentiate_the_mean(module):
    owner = _entry(module, "moments")
    policies = _policies(module, "moments")
    for m in (-2, Fraction(1, 3), 0):
        plan = plan_normal_pathwise(module, owner.id, (m,), (2,), policies)
        assert plan.exact.value == m**4 + 6 * m**2 + 3 + m**3 + 3 * m
        assert plan.exact.tangent == 2 * (4 * m**3 + 12 * m + 3 * m**2 + 3)
        at_zero = plan.at_noise((0,))
        assert (at_zero.value, at_zero.tangent) == (
            m**4 + m**3,
            2 * (4 * m**3 + 3 * m**2),
        )


def test_shared_static_sampling_site_still_allocates_independent_coordinates(module):
    owner = _entry(module, "difference")
    policy = _policies(module, "draw")
    plan = plan_normal_pathwise(module, owner.id, (2, 3), (1, 2), policy)
    site = policy[0].site
    assert [visit.site for visit in plan.visits] == [site, site]
    assert [visit.address for visit in plan.visits] == [
        (str(site), (("call", "draw#1"),), ()),
        (str(site), (("call", "draw#2"),), ()),
    ]
    assert (plan.exact.value, plan.exact.tangent) == (18, 24)
    pointwise = plan.at_noise((1, -1))
    assert (pointwise.value, pointwise.tangent) == (36, 48)


def test_canonical_sites_and_conditional_values_agree_with_checked_runtime_replay(
    module,
):
    owner = _entry(module, "difference")
    handler = next(item for item in module.handlers if item.name == "replayed")
    reference = module.with_(
        computations=tuple(
            item.with_(
                type=item.type.with_(effects=EMPTY_ROW),
                body=Handle(
                    instance=owner.type.effects.entries[0].instance,
                    handler=handler.id,
                    computation=owner.body,
                ),
            )
            if item.id == owner.id
            else item
            for item in module.computations
        )
    )
    plan = plan_normal_pathwise(
        module, owner.id, (2, 3), (1, 2), _policies(module, "draw")
    )
    for noise in ((1, -1), (0, 0), (Fraction(1, 2), -1)):
        configuration = RuntimeConfiguration(
            selections=(
                RuntimeSelection(
                    provider="core",
                    options={
                        "handlers": {
                            "replayed": {
                                "kind": "replay",
                                "policy": "clamp-no-score",
                                "values": {
                                    f"both@{index}": float(2 + 3 * coordinate)
                                    for index, coordinate in enumerate(noise)
                                },
                            }
                        }
                    },
                ),
            )
        )
        result = run_named(reference, "difference", (2, 3), runtime=configuration)
        actual = tuple(
            event.detail["address"]
            for event in result.trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(RANDOM_SAMPLE)
        )
        assert actual == tuple(visit.address for visit in plan.visits)
        assert host_value(result.value) == plan.at_noise(noise).value


def test_negative_power_of_nonzero_deterministic_coefficient(module):
    owner = _entry(module, "reciprocal")
    policy = _policies(module, "reciprocal")
    plan = plan_normal_pathwise(module, owner.id, (2,), (3,), policy)
    assert (plan.exact.value, plan.exact.tangent) == (Fraction(1, 4), Fraction(-3, 4))
    action = plan.at_noise((2,))
    assert (action.value, action.tangent) == (1, -3)
    with pytest.raises(JvpUnsupported, match="nonzero base"):
        plan_normal_pathwise(module, owner.id, (0,), (0,), policy)


def test_stable_parameter_control_and_noise_independent_rational_coefficients(module):
    owner = _entry(module, "regular_branch")
    policy = _policies(module, "regular_branch")
    positive = plan_normal_pathwise(module, owner.id, (2, 3), (1, 2), policy)
    assert (positive.exact.value, positive.exact.tangent) == (3, 3)
    negative = plan_normal_pathwise(module, owner.id, (-2, 3), (1, 2), policy)
    assert (negative.exact.value, negative.exact.tangent) == (0, 0)
    assert negative.at_noise((2,)).value == 3
    assert negative.at_noise((2,)).tangent == 2
    with pytest.raises(JvpBoundaryError, match="equality boundary"):
        plan_normal_pathwise(module, owner.id, (0, 3), (0, 1), policy)


@pytest.mark.parametrize(
    ("name", "point", "match"),
    [
        ("indicator", (0,), "noise-dependent comparison"),
        ("locally_cancelled_guard", (0,), "noise-dependent comparison"),
        ("random_scale", (0,), "positive and noise-independent"),
        ("random_divisor", (0,), "noise-independent divisor"),
        ("zero_divisor", (0,), "division by zero"),
        ("negative_noise_power", (0,), "noise-independent base"),
        ("dynamic_power", (2,), "fixed integral exponent"),
        ("wrong_family", (1,), "unplated Normal"),
        ("non_scalar_request", (Fraction(1, 2),), r"Random.sample\[Real\]"),
    ],
)
def test_unlicensed_pathwise_operations_refuse_in_actual_checked_source(
    module, name, point, match
):
    owner = _entry(module, name)
    with pytest.raises(JvpUnsupported, match=match):
        plan_normal_pathwise(module, owner.id, point, (1,), _policies(module, name))


def test_scale_and_the_policy_are_required_at_every_reached_occurrence(module):
    owner = _entry(module, "affine_pair")
    policies = _policies(module, "affine_pair")
    direction = (1, 1, 1, 1)
    for point in ((0, 0, 1, 0), (0, 1, -1, 0)):
        with pytest.raises(JvpUnsupported, match="positive and noise-independent"):
            plan_normal_pathwise(module, owner.id, point, direction, policies)
    call = (module, owner.id, (0, 1, 1, 0), direction)
    with pytest.raises(JvpUnsupported, match="missing pathwise estimator"):
        plan_normal_pathwise(*call, policies[:-1])
    with pytest.raises(ValueError, match="duplicate pathwise estimator"):
        plan_normal_pathwise(*call, (*policies, policies[0]))
    with pytest.raises(JvpUnsupported, match="explicit reparam"):
        plan_normal_pathwise(*call, (policies[0].with_(estimator="score"),))
    extra = SiteEstimator(site=SiteId.derive("not-a-source-site"), estimator="reparam")
    with pytest.raises(JvpUnsupported, match="unreached sites"):
        plan_normal_pathwise(*call, (*policies, extra))


def test_noise_coordinate_work_and_fuel_bounds_are_checked(module):
    owner = _entry(module, "affine_pair")
    policies = _policies(module, "affine_pair")
    call = (module, owner.id, (0, 1, 1, 1), (0, 1, 1, 1), policies)
    with pytest.raises(JvpLimit, match="exceeds 1 noise coordinates"):
        plan_normal_pathwise(*call, max_sites=1)
    with pytest.raises(JvpLimit, match="work limit"):
        plan_normal_pathwise(*call, max_work=1)
    with pytest.raises(JvpLimit, match="step budget of 1"):
        plan_normal_pathwise(*call, fuel=1)
    plan = plan_normal_pathwise(*call, max_sites=2)
    assert len(plan.visits) == 2
    assert (
        plan_normal_pathwise(*call, max_sites=2, max_work=plan.polynomial_work) == plan
    )
    with pytest.raises(JvpLimit, match="work limit"):
        plan_normal_pathwise(*call, max_sites=2, max_work=plan.polynomial_work - 1)
    with pytest.raises(ValueError, match="noise coordinates must match"):
        plan.at_noise((0,))
    with pytest.raises(ValueError, match="must be integers or Fractions"):
        plan.at_noise((True, 1))
    with pytest.raises(ValueError, match="unknown computation"):
        plan_normal_pathwise(module, ComputationId.derive("absent"), (), (), ())
