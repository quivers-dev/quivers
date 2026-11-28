"""Check signed mixed-estimator source behavior with exact independent oracles."""

from fractions import Fraction
from itertools import product
from random import Random

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EMPTY_ROW,
    RANDOM_SAMPLE,
    Bind,
    Handle,
    JvpLimit,
    JvpUnsupported,
    Perform,
    RuntimeConfiguration,
    RuntimeSelection,
    SiteEstimator,
    SiteValue,
    host_value,
    plan_mixed_site_gradient,
    run_named,
)
from quivers.qiec.bernoulli_estimators import _draw

_SOURCE = """
instance random : Random

handler replayed for Random : Real -> Real [coverage=total, implementation=foreign]
    sample resumes 1

define next_probability(keep : Bool, p : Real, q : Real) : Real !{} =
    if keep then
        return p * q
    else
        return 1.0 - q

define second_location(flip : Bool, x : Real, m : Real, k : Real) : Real !{} =
    if flip then
        return k * x
    else
        return m - x

define mixed(p : Real, q : Real, m : Real, s : Real, c : Real, k : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("first_normal"), Normal(m, s))
    let keep <- perform random.sample[Bool](site("outer"), Bernoulli(p))
    let r <- next_probability(keep, p, q)
    let flip <- perform random.sample[Bool](site("inner"), Bernoulli(r))
    let loc <- second_location(flip, x, m, k)
    let y <- perform random.sample[Real](site("second_normal"), Normal(loc, c))
    if keep then
        return x * y - y * y
    else
        return (y - m) * (y - m) - x

define cancelled_probability(m : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("first_normal"), Normal(0.0, 1.0))
    let keep <- perform random.sample[Bool](site("outer"), Bernoulli(0.5 + m * m * x))
    if keep then
        return x
    else
        return -x
"""
_BOOLEAN = (False, True)
_QUADRATURE = (
    (-2, Fraction(1, 12)),
    (-1, Fraction(1, 6)),
    (0, Fraction(1, 2)),
    (1, Fraction(1, 6)),
    (2, Fraction(1, 12)),
)
_POINT = (Fraction(2, 5), Fraction(3, 7), 3, 2, 1, Fraction(1, 2))
_DIRECTION = (1, -2, -1, 3, -2, 2)


@pytest.fixture(scope="module")
def module():
    lowered = Compiler(parse(_SOURCE), module_name="mixed-site-gradient").qiec_module
    assert lowered is not None
    return lowered


def _policies(module, name, outer="score", inner="measure"):
    owner = next(item for item in module.computations if item.name == name)
    choices = {
        "first_normal": "reparam",
        "second_normal": "reparam",
        "outer": outer,
        "inner": inner,
    }
    result = []
    term = owner.body
    while isinstance(term, Bind):
        if isinstance(term.first, Perform):
            request = term.first.request
            label = request.arguments[0]
            assert isinstance(label, SiteValue)
            result.append(
                SiteEstimator(
                    site=request.origin.static_site,
                    estimator=choices[label.label],
                )
            )
        term = term.then
    return owner, tuple(result)


def _reward(point, keep, flip, z, w):
    _, _, m, s, c, k = point
    x = m + s * z
    y = (k * x if flip else m - x) + c * w
    return x * y - y * y if keep else (y - m) ** 2 - x


def _at(step):
    return tuple(x + step * dx for x, dx in zip(_POINT, _DIRECTION, strict=True))


def _derivative(f):
    # The joint branch-weighted source is degree at most seven on this line.
    return (
        672 * (f(1) - f(-1))
        - 168 * (f(2) - f(-2))
        + 32 * (f(3) - f(-3))
        - 3 * (f(4) - f(-4))
    ) / 840


def _weights(point, keep, flip):
    p, q = point[:2]
    r = p * q if keep else 1 - q
    return (p if keep else 1 - p) * (r if flip else 1 - r)


def _mean(point, keep, flip):
    return sum(
        u * v * _reward(point, keep, flip, z, w)
        for (z, u), (w, v) in product(_QUADRATURE, repeat=2)
    )


def _joint(step, conditional_noise=None):
    point = _at(step)
    return sum(
        _weights(point, *path)
        * (
            _mean(point, *path)
            if conditional_noise is None
            else _reward(point, *path, *conditional_noise[path])
        )
        for path in product(_BOOLEAN, repeat=2)
    )


class _Script(Random):
    def __init__(self, expected):
        super().__init__(0)
        self.expected = iter(expected)

    def randrange(self, start, stop=None, step=1):
        p, value = next(self.expected)
        assert stop is None and step == 1 and p.denominator == start
        return 0 if value else start - 1


def _estimator_oracle(outer, inner, outcomes, conditional_noise):
    p, q = _POINT[:2]
    ps = {(): p, (False,): 1 - q, (True,): p * q}
    dp, dq = _DIRECTION[:2]
    slopes = {(): dp, (False,): -dq, (True,): dp * q + p * dq}
    draws, selectors, virtual, calls = [], [], [], []

    def visit(path):
        if len(path) == 2:

            def pay(step):
                point = _at(step)
                if conditional_noise is None:
                    return _mean(point, *path)
                return _reward(point, *path, *conditional_noise[path])

            return pay(0), _derivative(pay)
        method = inner if path else outer
        probability, slope = ps[path], slopes[path]
        selector = outcomes[path]
        if method == "score":
            draws.append((path, selector))
            calls.append((probability, selector))
            value, tangent = visit((*path, selector))
            denominator = probability if selector else probability - 1
            return value, tangent + (slope / denominator) * value
        if method == "measure":
            selectors.append((path, selector))
            calls.append((probability, selector))
            virtual.append((*path, False))
        left, tangent_left = visit((*path, False))
        if method == "measure":
            virtual.append((*path, True))
        right, tangent_right = visit((*path, True))
        shift = slope * (right - left)
        if method == "measure":
            return (
                (right, tangent_right + shift)
                if selector
                else (left, tangent_left + shift)
            )
        return (
            left + probability * (right - left),
            tangent_left + probability * (tangent_right - tangent_left) + shift,
        )

    result = visit(())
    mass = Fraction(1)
    for path, answer in outcomes.items():
        mass *= ps[path] if answer else 1 - ps[path]
    return result, mass, draws, selectors, virtual, calls


@pytest.mark.parametrize(
    ("outer", "inner"), tuple(product(("enumerate", "score", "measure"), repeat=2))
)
def test_every_finite_policy_combines_with_conditional_normal_pathwise(
    module, outer, inner
):
    owner, policies = _policies(module, "mixed", outer, inner)
    plan = plan_mixed_site_gradient(module, owner.id, _POINT, _DIRECTION, policies)
    assert (plan.exact.value, plan.exact.tangent) == (_joint(0), _derivative(_joint))
    assert (plan.finite.paths, plan.finite.replays) == (4, 7)
    assert [visit.prefix for visit in plan.finite.visits] == [(), (False,), (True,)]
    paths = tuple(product(_BOOLEAN, repeat=2))
    assert [item.prefix for item in plan.normal_visits] == [(), *paths]
    assert [leaf.coordinates for leaf in plan.normal_paths] == [
        (0, i) for i in range(1, 5)
    ]
    assert len({item.visit.address for item in plan.normal_visits}) == 2
    assert len(plan.sample(seed=42).visits) >= 2

    noise = (Fraction(3, 2), -3, 2, -1, 4)
    conditional_noise = {
        path: (noise[0], noise[index]) for index, path in enumerate(paths, 1)
    }
    explicit = plan.at_noise(noise)
    for tree, sources in ((plan.finite, None), (explicit, conditional_noise)):
        mass_sum, value_sum, gradient_sum = Fraction(0), Fraction(0), Fraction(0)
        for answers in product(_BOOLEAN, repeat=3):
            assignment = dict(zip(((), (False,), (True,)), answers, strict=True))
            expected, mass, draws, selectors, virtual, calls = _estimator_oracle(
                outer,
                inner,
                assignment,
                sources,
            )
            script = _Script(calls)
            result = _draw(tree.tree, script)
            assert next(script.expected, None) is None
            assert (result.value, result.tangent) == expected
            assert [(x.visit.prefix, x.choice) for x in result.draws] == draws
            assert [(x.visit.prefix, x.choice) for x in result.selectors] == selectors
            assert result.virtual_branches == tuple(virtual)
            mass_sum += mass
            value_sum += mass * result.value
            gradient_sum += mass * result.tangent
        assert mass_sum == 1
        assert (tree.exact.value, tree.exact.tangent) == (value_sum, gradient_sum)
        assert value_sum == _joint(0, sources)
        assert gradient_sum == _derivative(lambda step: _joint(step, sources))


def test_signed_cost_distinguishes_missing_and_doubled_probability_ownership(module):
    owner, policies = _policies(module, "mixed")
    plan = plan_mixed_site_gradient(module, owner.id, _POINT, _DIRECTION, policies)
    paths = tuple(product(_BOOLEAN, repeat=2))
    contributions = [
        _derivative(lambda step: _weights(_at(step), *path)) * _mean(_POINT, *path)
        for path in paths
    ]
    conditional = sum(
        _weights(_POINT, *path) * _derivative(lambda step: _mean(_at(step), *path))
        for path in paths
    )
    assert min(contributions) < 0 < max(contributions)
    correct = conditional + sum(contributions)
    assert plan.exact.tangent == correct
    assert all(correct != wrong for wrong in (conditional, 2 * correct - conditional))
    assert any(leaf.normal.exact.value < 0 for leaf in plan.normal_paths)


def test_mixed_source_replay_keeps_canonical_finite_and_normal_addresses(module):
    owner, policies = _policies(module, "mixed")
    plan = plan_mixed_site_gradient(module, owner.id, _POINT, _DIRECTION, policies)
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
    _, _, m, s, c, k = _POINT
    for path in plan.normal_paths:
        keep, flip = path.prefix
        z, w = Fraction(3, 2), -2
        x = m + s * z
        y = (k * x if flip else m - x) + c * w
        runtime = RuntimeConfiguration(
            selections=(
                RuntimeSelection(
                    provider="core",
                    options={
                        "handlers": {
                            "replayed": {
                                "kind": "replay",
                                "policy": "clamp-no-score",
                                "values": {
                                    "first_normal@0": float(x),
                                    "outer@0": keep,
                                    "inner@0": flip,
                                    "second_normal@0": float(y),
                                },
                            }
                        }
                    },
                ),
            )
        )
        result = run_named(
            reference, "mixed", tuple(map(float, _POINT)), runtime=runtime
        )
        actual = tuple(
            event.detail["address"]
            for event in result.trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(RANDOM_SAMPLE)
        )
        outer, inner = (
            visit for visit in plan.finite.visits if visit.prefix in ((), (keep,))
        )
        first, second = path.normal.visits
        assert actual == (first.address, outer.address, inner.address, second.address)
        assert host_value(result.value) == path.normal.at_noise((z, w)).value


def test_joint_refusals_do_not_erase_noise_or_missing_policies(module):
    wrong, declared = _policies(module, "cancelled_probability")
    with pytest.raises(JvpUnsupported, match="probabilities must be noise-independent"):
        plan_mixed_site_gradient(module, wrong.id, (0,), (0,), declared)
    owner, policies = _policies(module, "mixed")
    call = (module, owner.id, _POINT, _DIRECTION)
    for index, wrong in ((0, "score"), (1, "reparam"), (2, "reparam"), (3, "measure")):
        bad = list(policies)
        bad[index] = bad[index].with_(estimator=wrong)
        with pytest.raises(JvpUnsupported, match="missing .*estimator"):
            plan_mixed_site_gradient(*call, tuple(bad))
    with pytest.raises(JvpUnsupported, match="strictly between"):
        plan_mixed_site_gradient(
            module, owner.id, (0, *_POINT[1:]), _DIRECTION, policies
        )
    with pytest.raises(JvpUnsupported, match="strictly positive"):
        plan_mixed_site_gradient(
            module, owner.id, (*_POINT[:4], 0, _POINT[5]), _DIRECTION, policies
        )


def test_joint_work_limits_and_exact_noise_arity(module):
    owner, policies = _policies(module, "mixed")
    call = (module, owner.id, _POINT, _DIRECTION, policies)
    plan = plan_mixed_site_gradient(*call, max_sites=2)
    assert plan.polynomial_work > sum(
        leaf.normal.polynomial_work for leaf in plan.normal_paths
    )
    assert (
        plan_mixed_site_gradient(*call, max_sites=2, max_work=plan.polynomial_work)
        == plan
    )
    with pytest.raises(JvpLimit, match="exceeds 3 complete paths"):
        plan_mixed_site_gradient(*call, max_paths=3)
    with pytest.raises(JvpLimit, match="exceeds 1 noise coordinates"):
        plan_mixed_site_gradient(*call, max_sites=1)
    with pytest.raises(JvpLimit, match="work limit"):
        plan_mixed_site_gradient(*call, max_sites=2, max_work=plan.polynomial_work - 1)
    with pytest.raises(JvpLimit, match="step budget of 1"):
        plan_mixed_site_gradient(*call, fuel=1)
    with pytest.raises(ValueError, match="must match"):
        plan.at_noise((0,))
    with pytest.raises(ValueError, match="integers or Fractions"):
        plan.at_noise((True, 0, 0, 0, 0))
