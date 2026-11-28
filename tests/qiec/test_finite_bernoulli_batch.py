"""Compare independent row axes with scalar replay, gradients and joint laws."""

from dataclasses import replace
from fractions import Fraction
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    EMPTY_ROW,
    RANDOM_SAMPLE,
    Handle,
    JvpLimit,
    JvpUnsupported,
    Perform,
    RuntimeConfiguration,
    RuntimeSelection,
    SiteEstimator,
    exact_gradient,
    host_value,
    plan_bernoulli_batch,
    run_named,
)
from quivers.qiec.bernoulli_estimators import plan_bernoulli_gradient

_SOURCE = """
instance random : Random

handler replayed for Random : Real -> Real [coverage=total, implementation=foreign]
    sample resumes 1

define toss(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("toss"), Bernoulli(p))

define row(s : Real, p : Real, x : Real) : Real !{random} =
    let a <- toss(p)
    let b <- toss(p * p)
    if a then
        if b then
            return s * x + s * s
        else
            return -x + s
    else
        if b then
            return x * x + s
        else
            return s * x - 2.0

define ff(s : Real, p : Real, x : Real) : Real !{} =
    return s * x - 2.0

define ft(s : Real, p : Real, x : Real) : Real !{} =
    return x * x + s

define tf(s : Real, p : Real, x : Real) : Real !{} =
    return -x + s

define tt(s : Real, p : Real, x : Real) : Real !{} =
    return s * x + s * s

define expected(s : Real, p : Real, x : Real) : Real !{} =
    return (1.0 - p) * (1.0 - p * p) * (s * x - 2.0) + (1.0 - p) * p * p * (x * x + s) + p * (1.0 - p * p) * (-x + s) + p * p * p * (s * x + s * s)

define single(p : Real) : Real !{random} =
    let bit <- toss(p)
    if bit then
        return 1.0
    else
        return 0.0

define constant() : Real !{random} =
    single(0.5)

define probability(previous : Bool, p : Real) : Real !{} =
    if previous then
        return p
    else
        return 1.0 - p

define dependent(p : Real) : Real !{random} =
    let earlier <- toss(p)
    let q <- probability(earlier, p)
    let later <- toss(q)
    if later then
        return 1.0
    else
        return 0.0

define optional(p : Real) : Real !{random} =
    let bit <- toss(p)
    if bit then
        let extra <- toss(p)
        return 1.0
    else
        return 2.0

define neutral(x : Real) : Real !{} =
    return x

define shifted(p : Real, x : Real) : Real !{random} =
    if x > 0.0 then
        let y <- neutral(x)
        let bit <- toss(p)
        return y
    else
        let bit <- toss(p)
        return x

define recursive(p : Real, n : Real) : Real !{random} =
    if n > 0.0 then
        recursive(p, n - 1.0)
    else
        single(p)
"""
_SHARED = (Fraction(2, 3),)
_ROWS = ((Fraction(1, 3), 2), (Fraction(3, 5), -1))
_D_SHARED = (3,)
_D_ROWS = ((2, -1), (-3, 1))
_COTANGENTS = (-2, Fraction(5, 3))
_BOOL = (False, True)


@pytest.fixture(scope="module")
def module():
    module = Compiler(parse(_SOURCE), module_name="finite-bernoulli-batch").qiec_module
    assert module is not None
    return module


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _policy(module):
    request = _entry(module, "toss").body
    assert isinstance(request, Perform)
    return SiteEstimator(site=request.request.origin.static_site, estimator="enumerate")


def _plan(module):
    return plan_bernoulli_batch(
        module, _entry(module, "row").id, "Items", _SHARED, _ROWS, (_policy(module),)
    )


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def _compare_linearization(linearization, exact_rows):
    assert linearization.values == tuple(row.value for row in exact_rows)
    assert linearization.jvp(_D_SHARED, _D_ROWS) == tuple(
        _dot(row.gradient, (*_D_SHARED, *seed))
        for row, seed in zip(exact_rows, _D_ROWS, strict=True)
    )
    adjoint = linearization.vjp(_COTANGENTS)
    assert adjoint.shared == (
        sum(
            row.gradient[0] * w for row, w in zip(exact_rows, _COTANGENTS, strict=True)
        ),
    )
    assert adjoint.rows == tuple(
        tuple(w * slope for slope in row.gradient[1:])
        for row, w in zip(exact_rows, _COTANGENTS, strict=True)
    )
    assert _dot(linearization.jvp(_D_SHARED, _D_ROWS), _COTANGENTS) == _dot(
        adjoint.shared, _D_SHARED
    ) + sum(_dot(a, b) for a, b in zip(adjoint.rows, _D_ROWS, strict=True))


def test_ordered_values_and_jvp_vjp_agree_with_exact_scalar_expectation(module):
    plan = _plan(module)
    policy = _policy(module)
    assert plan.shape == (
        (str(policy.site), (("call", "toss#1"),), ()),
        (str(policy.site), (("call", "toss#2"),), ()),
    )
    rows = tuple(exact_gradient(module, "expected", (*_SHARED, *row)) for row in _ROWS)
    _compare_linearization(plan.expected, rows)
    scalar = tuple(
        plan_bernoulli_gradient(
            module, plan.computation, (*_SHARED, *row), (*_D_SHARED, *seed), (policy,)
        ).exact
        for row, seed in zip(_ROWS, _D_ROWS, strict=True)
    )
    assert plan.expected.values == tuple(item.value for item in scalar)
    assert plan.expected.jvp(_D_SHARED, _D_ROWS) == tuple(
        item.tangent for item in scalar
    )
    assert all(row.gradient[1] != 0 for row in plan.expected.rows)


def test_every_joint_trace_keeps_distinct_row_addresses_and_exact_replay_ad(module):
    plan = _plan(module)
    names = dict(zip(product(_BOOL, repeat=2), ("ff", "ft", "tf", "tt"), strict=True))
    probability = Fraction(0)
    expected = [Fraction(0), Fraction(0)]
    for selected in product(*plan.paths):
        tapes = tuple(path.tape for path in selected)
        replay = plan.replay(tapes)
        exact_rows = []
        expected_joint = Fraction(1)
        for index, tape in enumerate(tapes):
            assert tape.addresses == tuple(
                ("Items", index, local) for local in plan.shape
            )
            bits = tuple(bit for _, bit in tape.choices)
            p, x = _ROWS[index]
            expected_joint *= (p if bits[0] else 1 - p) * (
                p * p if bits[1] else 1 - p * p
            )
            exact_rows.append(exact_gradient(module, names[bits], (*_SHARED, p, x)))
        assert replay.probability == expected_joint
        assert len(set(tapes[0].addresses + tapes[1].addresses)) == 4
        _compare_linearization(replay.linearization, exact_rows)
        assert all(row.gradient[1] == 0 for row in replay.linearization.rows)
        probability += replay.probability
        for index, value in enumerate(replay.linearization.values):
            expected[index] += replay.probability * value
    assert probability == 1
    assert tuple(expected) == plan.expected.values


def test_joint_rng_independence_separates_shared_draw_with_the_same_marginals(
    module, monkeypatch
):
    owner = _entry(module, "single")
    p = Fraction(1, 2)
    plan = plan_bernoulli_batch(
        module, owner.id, "Items", (p,), ((), ()), (_policy(module),)
    )
    assert plan.expected.values == (p, p)
    counts = dict.fromkeys(product(_BOOL, repeat=2), 0)
    for scripted in product(range(2), repeat=2):
        draws = iter(scripted)

        class Script:
            def randrange(self, stop):
                assert stop == 2
                return next(draws)

        monkeypatch.setattr(
            "quivers.qiec.finite_bernoulli_batch.Random", lambda seed: Script()
        )
        trace = plan.sample(seed=42)
        bits = tuple(tape.choices[0][1] for tape in trace.tapes)
        assert bits == tuple(bool(bit) for bit in scripted)
        assert trace.linearization.values == tuple(Fraction(bit) for bit in bits)
        assert next(draws, None) is None
        counts[bits] += 1
    assert set(counts.values()) == {1}
    independent_agreement = Fraction(counts[(False, False)] + counts[(True, True)], 4)
    assert independent_agreement == Fraction(1, 2)
    joint = {bits: Fraction(count, 4) for bits, count in counts.items()}
    shared_draw = {(False, False): p, (True, True): p}
    for index in range(2):
        assert sum(mass for bits, mass in joint.items() if bits[index]) == p
        assert sum(mass for bits, mass in shared_draw.items() if bits[index]) == p
    assert sum(mass for bits, mass in shared_draw.items() if bits[0] == bits[1]) == 1
    assert joint != shared_draw
    assert plan.expected.jvp((1,), ((), ())) == (1, 1)
    assert plan.expected.vjp((2, -3)).shared == (-1,)


def test_fixed_replay_agrees_with_ordinary_scalar_machine_request_addresses(module):
    plan = _plan(module)
    owner = _entry(module, "row")
    replayed = next(item for item in module.handlers if item.name == "replayed")
    candidate = module.with_(
        computations=tuple(
            item.with_(
                type=item.type.with_(effects=EMPTY_ROW),
                body=Handle(
                    instance=item.type.effects.entries[0].instance,
                    handler=replayed.id,
                    computation=item.body,
                ),
            )
            if item.id == owner.id
            else item
            for item in module.computations
        )
    )
    replay = plan.replay((plan.paths[0][1].tape, plan.paths[1][2].tape))
    for tape, x, value in zip(
        replay.tapes, _ROWS, replay.linearization.values, strict=True
    ):
        options = RuntimeConfiguration(
            selections=(
                RuntimeSelection(
                    provider="core",
                    options={
                        "handlers": {
                            "replayed": {
                                "kind": "replay",
                                "policy": "clamp-no-score",
                                "values": {
                                    f"toss@{i}": bit
                                    for i, (_, bit) in enumerate(tape.choices)
                                },
                            }
                        }
                    },
                ),
            )
        )
        source = run_named(
            candidate, "row", tuple(map(float, (*_SHARED, *x))), runtime=options
        )
        observed = tuple(
            event.detail["address"]
            for event in source.trace
            if (
                event.event == "operation.requested"
                and event.detail["operation"] == str(RANDOM_SAMPLE)
            )
        )
        assert observed == tuple(address for address, _ in tape.choices)
        assert host_value(source.value) == pytest.approx(float(value))


def test_rejects_wrong_tape_axes_shapes_and_replay_cotangents(module):
    batch = _plan(module)
    a, b = batch.paths[0][0].tape, batch.paths[1][0].tape
    with pytest.raises(ValueError, match="one addressed tape"):
        batch.replay((a,))
    for wrong in (replace(a, axis="Other"), replace(a, ordinal=1)):
        with pytest.raises(ValueError, match="axis and row order"):
            batch.replay((wrong, b))
    for choices in (a.choices[::-1], a.choices[:-1]):
        with pytest.raises(ValueError, match="original addresses"):
            batch.replay((replace(a, choices=choices), b))
    with pytest.raises(ValueError, match="shared prefix"):
        batch.expected.jvp((), _D_ROWS)
    with pytest.raises(ValueError, match="local tangent"):
        batch.expected.jvp(_D_SHARED, ((1,), (2, 3)))
    with pytest.raises(ValueError, match="row count"):
        batch.expected.vjp((1,))


def test_data_dependent_draw_shape_cycles_and_work_refuse(module):
    policy = _policy(module)
    for name, rows in (
        ("optional", ((Fraction(1, 2),),)),
        ("shifted", ((Fraction(1, 2), 1), (Fraction(1, 2), -1))),
    ):
        with pytest.raises(JvpUnsupported, match="same ordered draw addresses"):
            plan_bernoulli_batch(
                module, _entry(module, name).id, "Items", (), rows, (policy,)
            )
    with pytest.raises(JvpUnsupported, match="acyclic call graph"):
        plan_bernoulli_batch(
            module,
            _entry(module, "recursive").id,
            "Items",
            (),
            ((Fraction(1, 2), -1),),
            (policy,),
        )
    batch = _plan(module)
    call = (module, batch.computation, "Items", _SHARED, _ROWS, (policy,))
    with pytest.raises(JvpLimit, match="exceeds 1 rows"):
        plan_bernoulli_batch(*call, max_rows=1)
    with pytest.raises(JvpLimit, match="exceeds 3 complete paths"):
        plan_bernoulli_batch(*call, max_paths=3)
    with pytest.raises(JvpLimit, match="reference steps"):
        plan_bernoulli_batch(*call, max_work=batch.source_steps - 1)
    assert plan_bernoulli_batch(*call, max_work=batch.source_steps) == batch


def test_zero_arity_random_rows_keep_independent_axis_positions(module):
    batch = plan_bernoulli_batch(
        module, _entry(module, "constant").id, "Items", (), ((), ()), (_policy(module),)
    )
    assert batch.expected.values == (Fraction(1, 2), Fraction(1, 2))
    assert batch.expected.jvp((), ((), ())) == (0, 0)
    assert batch.expected.vjp((1, 1)).rows == ((), ())
    assert batch.paths[0][0].tape.addresses != batch.paths[1][0].tape.addresses


def test_fixed_addresses_admit_an_adaptive_conditional_probability(module):
    points = (Fraction(1, 3), Fraction(3, 4))
    plan = plan_bernoulli_batch(
        module,
        _entry(module, "dependent").id,
        "Items",
        (),
        tuple((p,) for p in points),
        (_policy(module),),
    )
    assert plan.expected.values == tuple(p * p + (1 - p) * (1 - p) for p in points)
    assert plan.expected.jvp((), ((1,), (1,))) == tuple(4 * p - 2 for p in points)
    assert plan.shape[1][1] == (("call", "toss#3"),)
    for p, support in zip(points, plan.paths, strict=True):
        for case in support:
            first, second = tuple(bit for _, bit in case.tape.choices)
            q = p if first else 1 - p
            assert case.probability == (p if first else 1 - p) * (
                q if second else 1 - q
            )
            assert case.action.gradient == (0,)


def test_joint_replay_keeps_a_product_beyond_each_scalar_coefficient_limit(module):
    p = Fraction(1, 2**10000)
    plan = plan_bernoulli_batch(
        module,
        _entry(module, "single").id,
        "Items",
        (p,),
        ((), ()),
        (_policy(module),),
    )
    tapes = tuple(support[-1].tape for support in plan.paths)
    replay = plan.replay(tapes)
    assert replay.linearization.values == (1, 1)
    assert replay.probability == Fraction(1, 2**20000)
