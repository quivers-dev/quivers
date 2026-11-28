"""Checked Bernoulli site estimates against an independent analytic objective."""

from fractions import Fraction
from itertools import product
from random import Random

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EMPTY_ROW,
    RANDOM_SAMPLE,
    Bind,
    ComputationId,
    DistributionValue,
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
    run_named,
)
from quivers.qiec.bernoulli_estimators import _draw, plan_bernoulli_gradient

_SOURCE = """
instance random : Random

handler replayed for Random : Real -> Real [coverage=total, implementation=foreign]
    sample resumes 1

define adaptive(p : Real, q : Real) : Real !{random} =
    let outer <- perform random.sample[Bool](site("outer"), Bernoulli(p))
    if outer then
        let right <- perform random.sample[Bool](site("right"), Bernoulli(p * q))
        if right then
            return p + 3.0 * q
        else
            return 2.0 * p
    else
        let left <- perform random.sample[Bool](site("left"), Bernoulli((1.0 - p) * q))
        if left then
            return q - p
        else
            return p * q

define flip(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("shared"), Bernoulli(p))

define single(p : Real) : Real !{random} =
    let selected <- perform random.sample[Bool](site("single"), Bernoulli(p))
    if selected then
        return 1.0
    else
        return 0.0

define shared(p : Real) : Real !{random} =
    let first <- flip(p)
    let second <- flip(p * p)
    if first then
        if second then
            return 1.0
        else
            return 0.0
    else
        return 0.0

define early_exit(remaining : Real, p : Real) : Real !{random} =
    if remaining > 0.0 then
        let stop <- flip(p)
        if stop then
            return remaining
        else
            early_exit(remaining - 1.0, p)
    else
        return 0.0

define uncertain_sign(p : Real) : Real !{random} =
    let value <- perform random.sample[Bool](site("value"), Bernoulli(p))
    if p > 0.5 then
        return p
    else
        return -p

define non_boolean(p : Real) : Real !{random} =
    perform random.sample[Real](site("continuous"), Normal(p, 1.0))

define duplicate(p : Real) : Real !{random} =
    let first <- perform random.sample[Bool](site("first"), Bernoulli(p))
    let second <- perform random.sample[Bool](site("second"), Bernoulli(p))
    if first && second then
        return 1.0
    else
        return 0.0
"""

_BOOLEAN = (False, True)


@pytest.fixture(scope="module")
def module() -> QiecModule:
    result = Compiler(
        parse(_SOURCE), module_name="bernoulli-estimators-test"
    ).qiec_module
    assert result is not None
    return result


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _body_sites(term):
    if isinstance(term, Bind):
        yield from _body_sites(term.first)
        yield from _body_sites(term.then)
    if isinstance(term, If):
        yield from _body_sites(term.then)
        yield from _body_sites(term.otherwise)
    if isinstance(term, Perform):
        argument = term.request.arguments[0]
        assert isinstance(argument, SiteValue)
        yield argument.label, term.request.origin.static_site


def _policies(module, name, **declared):
    sites = dict(_body_sites(_entry(module, name).body))
    return tuple(
        SiteEstimator(site=sites[label], estimator=kind)
        for label, kind in declared.items()
    )


def _closed_form(p, q, dp, dq):
    # Expand all four path masses and rewards first, then differentiate.
    quadratic = 3 * p**2 + (1 - p) ** 3
    cross = p**2 - 2 * p**3
    expected = 2 * p**2 + cross * q + quadratic * q**2
    dp_expected = 4 * p + (2 * p - 6 * p**2) * q + (6 * p - 3 * (1 - p) ** 2) * q**2
    dq_expected = cross + 2 * quadratic * q
    return expected, dp * dp_expected + dq * dq_expected


class _ScriptedRandom(Random):
    def __init__(self, answers):
        super().__init__(0)
        self.answers = iter(answers)

    def randrange(self, start, stop=None, step=1):
        assert stop is None and step == 1
        return 0 if next(self.answers) else start - 1


def _probability(p_true, value):
    return p_true if value else 1 - p_true


def _outcomes(p, left_p, right_p, outer, inner):
    if outer == "enumerate" and inner == "enumerate":
        return (((), Fraction(1)),)
    if outer == "score" and inner == "enumerate":
        return tuple(((choice,), _probability(p, choice)) for choice in _BOOLEAN)
    if outer == "enumerate":
        return tuple(
            ((left, right), _probability(left_p, left) * _probability(right_p, right))
            for left, right in product(_BOOLEAN, repeat=2)
        )
    return tuple(
        ((a, b), _probability(p, a) * _probability(right_p if a else left_p, b))
        for a, b in product(_BOOLEAN, repeat=2)
    )


@pytest.mark.parametrize(("outer", "inner"), product(("enumerate", "score"), repeat=2))
@pytest.mark.parametrize(
    ("p", "q", "dp", "dq"),
    [
        (Fraction(1, 4), Fraction(2, 3), 1, -2),
        (Fraction(3, 5), Fraction(1, 7), -3, 2),
    ],
)
def test_adaptive_mixed_estimators_match_independent_expectation(
    module, outer, inner, p, q, dp, dq
):
    owner = _entry(module, "adaptive")
    policies = _policies(module, "adaptive", outer=outer, left=inner, right=inner)
    plan = plan_bernoulli_gradient(module, owner.id, (p, q), (dp, dq), policies)
    expected = _closed_form(p, q, dp, dq)
    assert (plan.exact.value, plan.exact.tangent) == expected
    assert plan.paths == 4
    assert plan.replays == 7 and 0 < plan.source_steps <= plan.replays * 65_536
    assert [visit.prefix for visit in plan.visits] == [(), (False,), (True,)]
    assert [visit.probability for visit in plan.visits] == [p, (1 - p) * q, p * q]
    assert [visit.tangent for visit in plan.visits] == [
        dp,
        (1 - p) * dq - dp * q,
        p * dq + dp * q,
    ]
    assert len({visit.site for visit in plan.visits}) == 3
    assert all(visit.address == (str(visit.site), (), ()) for visit in plan.visits)

    value = tangent = Fraction()
    for script, mass in _outcomes(p, (1 - p) * q, p * q, outer, inner):
        chosen = _draw(plan.tree, _ScriptedRandom(script))
        assert tuple(draw.choice for draw in chosen.draws) == script
        assert len(chosen.visits) == (3 if outer == "enumerate" else 2)
        if outer == "enumerate" and inner == "score":
            assert tuple(draw.visit.prefix for draw in chosen.draws) == (
                (False,),
                (True,),
            )
        elif outer == "score" and inner == "score":
            assert len(script) == 2
            assert tuple(draw.visit.prefix for draw in chosen.draws) == (
                (),
                (script[0],),
            )
        value += mass * chosen.value
        tangent += mass * chosen.tangent
    assert (value, tangent) == expected
    assert plan.sample(seed=390) == plan.sample(seed=390)
    if outer == inner == "enumerate":
        exact_draw = plan.sample(seed=390)
        assert (exact_draw.value, exact_draw.tangent, exact_draw.draws) == (
            *expected,
            (),
        )


def test_shared_parameters_keep_distinct_dynamic_addresses(module):
    owner = _entry(module, "shared")
    policies = _policies(module, "flip", shared="score")
    plan = plan_bernoulli_gradient(module, owner.id, (Fraction(2, 3),), (2,), policies)
    site = policies[0].site
    assert [visit.site for visit in plan.visits] == [site, site, site]
    assert [visit.address for visit in plan.visits] == [
        (str(site), (("call", "flip#1"),), ()),
        (str(site), (("call", "flip#2"),), ()),
        (str(site), (("call", "flip#2"),), ()),
    ]
    assert (plan.exact.value, plan.exact.tangent) == (Fraction(8, 27), Fraction(8, 3))
    always_true = _draw(plan.tree, _ScriptedRandom((True, True)))
    assert len(always_true.draws) == 2
    assert always_true.tangent == 9


def test_counted_addresses_and_values_match_ordinary_reference_replay(module):
    owner = _entry(module, "shared")
    policies = _policies(module, "flip", shared="score")
    point = Fraction(1, 2)
    plan = plan_bernoulli_gradient(module, owner.id, (point,), (1,), policies)
    handler = next(item for item in module.handlers if item.name == "replayed")
    body = Handle(
        instance=owner.type.effects.entries[0].instance,
        handler=handler.id,
        computation=owner.body,
    )
    reference = module.with_(
        computations=tuple(
            item.with_(type=item.type.with_(effects=EMPTY_ROW), body=body)
            if item.id == owner.id
            else item
            for item in module.computations
        )
    )
    for choices in product(_BOOLEAN, repeat=2):
        answer = _draw(plan.tree, _ScriptedRandom(choices))
        addresses = tuple(draw.visit.address for draw in answer.draws)
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
                                    f"shared@{index}": choice
                                    for index, choice in enumerate(choices)
                                },
                            }
                        }
                    },
                ),
            )
        )
        original = run_named(reference, "shared", (float(point),), runtime=runtime)
        actual = tuple(
            event.detail["address"]
            for event in original.trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(RANDOM_SAMPLE)
        )
        assert actual == addresses
        assert host_value(original.value) == float(answer.value) == float(all(choices))


def test_asymmetric_recursive_trace_tree_has_the_measured_path_count(module):
    owner = _entry(module, "early_exit")
    remaining, p = Fraction(41, 2), Fraction(1, 2)
    paths = 22
    plan = plan_bernoulli_gradient(
        module,
        owner.id,
        (remaining, p),
        (0, 1),
        _policies(module, "flip", shared="score"),
        max_paths=paths,
    )
    r = 1 - p
    expected = sum((remaining - k) * p * r**k for k in range(paths - 1))
    tangent = sum(
        (remaining - k) * (r**k - p * k * r ** (k - 1)) for k in range(paths - 1)
    )
    assert (plan.exact.value, plan.exact.tangent) == (expected, tangent)
    assert plan.paths == paths and plan.replays == 2 * paths - 1
    assert [visit.prefix for visit in plan.visits] == [
        (False,) * count for count in range(paths - 1)
    ]
    assert len({visit.address for visit in plan.visits}) == paths - 1
    last = _draw(plan.tree, _ScriptedRandom((False,) * (paths - 1)))
    assert last.value == last.tangent == 0
    assert len(last.draws) == paths - 1


def test_probabilities_and_comparisons_must_be_locally_regular(module):
    owner = _entry(module, "adaptive")
    policies = _policies(module, "adaptive", outer="score", right="score", left="score")
    for p in (0, 1, -1, 2):
        with pytest.raises(JvpUnsupported, match="strictly between"):
            plan_bernoulli_gradient(
                module, owner.id, (p, Fraction(1, 3)), (1, 0), policies
            )
    with pytest.raises(JvpUnsupported, match="strictly between"):
        plan_bernoulli_gradient(
            module,
            owner.id,
            (Fraction(1, 2), 2),
            (1, 0),
            policies,
        )

    boundary = _entry(module, "uncertain_sign")
    condition = _policies(module, "uncertain_sign", value="enumerate")
    with pytest.raises(JvpBoundaryError, match="equality boundary"):
        plan_bernoulli_gradient(module, boundary.id, (Fraction(1, 2),), (1,), condition)
    regular = plan_bernoulli_gradient(
        module, boundary.id, (Fraction(1, 3),), (1,), condition
    )
    assert regular.exact.value == Fraction(-1, 3) and regular.exact.tangent == -1


def test_a_rare_large_score_outcome_is_not_censored_after_planning(module):
    rate = 2**10_000
    owner = _entry(module, "single")
    plan = plan_bernoulli_gradient(
        module,
        owner.id,
        (Fraction(1, rate),),
        (rate,),
        _policies(module, "single", single="score"),
    )
    rare = _draw(plan.tree, _ScriptedRandom((True,)))
    usual = _draw(plan.tree, _ScriptedRandom((False,)))
    assert rare.tangent == rate**2 and usual.tangent == 0
    assert Fraction(rare.tangent, rate) == plan.exact.tangent == rate


def test_unsupported_sample_variants_fail_before_planning(module):
    owner = _entry(module, "non_boolean")
    policies = _policies(module, "non_boolean", continuous="score")
    with pytest.raises(JvpUnsupported, match=r"Random.sample\[Bool\]"):
        plan_bernoulli_gradient(module, owner.id, (Fraction(1, 2),), (1,), policies)

    owner = _entry(module, "single")
    assert isinstance(owner.body, Bind) and isinstance(owner.body.first, Perform)
    request = owner.body.first.request
    site, distribution = request.arguments
    assert isinstance(distribution, DistributionValue)
    logits = distribution.with_(arguments=(("logits", distribution.arguments[0][1]),))
    body = owner.body.with_(
        first=owner.body.first.with_(request=request.with_(arguments=(site, logits)))
    )
    changed = _replace_body(module, owner, body)
    policies = _policies(module, "single", single="score")
    with pytest.raises(JvpUnsupported, match=r"Bernoulli\(probs="):
        plan_bernoulli_gradient(changed, owner.id, (Fraction(1, 2),), (1,), policies)


def test_policy_complete_and_unique_and_checked_before_estimation(module):
    owner = _entry(module, "adaptive")
    call = (module, owner.id, (Fraction(1, 3), Fraction(1, 2)), (1, 0))
    all_sites = _policies(
        module, "adaptive", outer="enumerate", left="score", right="score"
    )
    with pytest.raises(JvpUnsupported, match="missing estimator"):
        plan_bernoulli_gradient(*call, all_sites[:-1])
    with pytest.raises(ValueError, match="duplicate estimator"):
        plan_bernoulli_gradient(*call, (*all_sites, all_sites[0]))
    with pytest.raises(JvpUnsupported, match="unsupported Bernoulli estimator"):
        plan_bernoulli_gradient(*call, (all_sites[0].with_(estimator="reparam"),))
    extra = SiteEstimator(site=SiteId.derive("not-a-source-site"), estimator="score")
    with pytest.raises(JvpUnsupported, match="unreached sites"):
        plan_bernoulli_gradient(*call, (*all_sites, extra))


def test_bounded_expansion_measures_complete_paths_and_replay_fuel(module):
    owner = _entry(module, "adaptive")
    policies = _policies(module, "adaptive", outer="score", left="score", right="score")
    call = (module, owner.id, (Fraction(1, 3), Fraction(1, 2)), (1, 0), policies)
    with pytest.raises(JvpLimit, match="exceeds 3 complete paths"):
        plan_bernoulli_gradient(*call, max_paths=3)
    assert plan_bernoulli_gradient(*call, max_paths=4).paths == 4
    with pytest.raises(JvpLimit, match="step budget of 1"):
        plan_bernoulli_gradient(*call, fuel=1)
    with pytest.raises(ValueError, match="positive integer"):
        plan_bernoulli_gradient(*call, max_paths=True)
    with pytest.raises(ValueError, match="unknown computation"):
        plan_bernoulli_gradient(module, ComputationId.derive("absent"), (), (), ())


def test_duplicate_full_address_in_one_source_path_is_refused(module):
    owner = _entry(module, "duplicate")
    assert isinstance(owner.body, Bind) and isinstance(owner.body.first, Perform)
    other = owner.body.then
    assert isinstance(other, Bind) and isinstance(other.first, Perform)
    other_request = other.first.request.with_(origin=owner.body.first.request.origin)
    body = owner.body.with_(
        then=other.with_(first=other.first.with_(request=other_request))
    )
    candidate = _replace_body(module, owner, body)
    policy = _policies(module, "duplicate", first="score")
    with pytest.raises(JvpUnsupported, match="repeated Bernoulli address"):
        plan_bernoulli_gradient(candidate, owner.id, (Fraction(1, 2),), (1,), policy)


def _replace_body(module, owner, body):
    return module.with_(
        computations=tuple(
            item.with_(body=body) if item.id == owner.id else item
            for item in module.computations
        )
    )
