"""Scalar, tensor and independently calculated Bernoulli axis observations."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from itertools import product

import pytest
import torch

from quivers._values import HeldValue
from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    BernoulliTensorAxis,
    Evaluator,
    JvpBoundaryError,
    JvpLimit,
    JvpUnsupported,
    Perform,
    RuntimeAttachments,
    RuntimeRequest,
    SiteEstimator,
    exact_gradient,
    exact_jvp,
    plan_bernoulli_batch,
    validate_module,
)
from quivers.qiec import evaluator as evaluator_module
from quivers.qiec.distributions import RuntimeDistribution
from quivers.qiec.evaluator import Scope, _Continue, _Frame, _Returned
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.terms import Local

_SOURCE = """
instance random : Random

define toss(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("toss"), Bernoulli(p))

define unit(s : Real, p : Real, x : Real) : Real !{random} =
    let earlier <- toss(p)
    let q = p * (s + 0.25 * real(earlier))
    let got <- toss(q)
    if true then
        return (x + real(earlier)) * (s + real(got)) + x * x
    else
        return x / 0.0

define same_p(p : Real) : Real !{random} =
    let bit <- toss(p)
    return real(bit)

define constant(p : Real) : Real !{random} =
    let bit <- toss(p)
    return 1.0

define no_draw() : Real !{random} =
    if true then
        return 2.0
    else
        let bit <- toss(0.5)
        return real(bit)

define row_branch(p : Real) : Real !{random} =
    let bit <- toss(p)
    if p > 0.5 then
        return real(bit)
    else
        return p / 0.0

define random_branch(p : Real) : Real !{random} =
    let bit <- toss(p)
    if bit then
        return 1.0
    else
        return 0.0

define active_partial(p : Real) : Real !{random} =
    let bit <- toss(p)
    if false then
        return real(bit)
    else
        return p / 0.0

define stable_bool(p : Real) : Real !{} =
    return p * real(p > 0.5)
"""
_SHARED = (Fraction(1, 2),)
_ROWS = ((Fraction(1, 4), 0), (Fraction(3, 4), -2))
_D_SHARED = (Fraction(2, 3),)
_D_ROWS = ((Fraction(3, 5), 1), (2, Fraction(-2, 3)))
_VALUES_SEED = (-2, 3)
_SCORES_SEED = ((3, -1), (-2, 4))
_BOOL = (False, True)


@pytest.fixture(scope="module")
def module():
    module = Compiler(parse(_SOURCE), module_name="bernoulli-tensor-axis").qiec_module
    assert module is not None
    return module


def _entry(module, name):
    return next(item for item in module.computations if item.name == name)


def _tensor(values):
    def convert(item):
        if isinstance(item, (list, tuple)):
            return tuple(map(convert, item))
        return float(item)

    return torch.tensor(convert(values), dtype=torch.float64)


def _exact_batch(module, name="unit", arity=1, shared=_SHARED, rows=_ROWS):
    assert len(shared) == arity
    operation = _entry(module, "toss").body
    assert isinstance(operation, Perform)
    return plan_bernoulli_batch(
        module,
        _entry(module, name).id,
        "Items",
        shared,
        rows,
        (
            SiteEstimator(
                site=operation.request.origin.static_site, estimator="enumerate"
            ),
        ),
    )


class _ScalarReplay(Evaluator):
    """Clamp one ordinary scalar source and record the score at each request."""

    def __init__(self, module, bits):
        super().__init__(
            RuntimeAttachments(numbers=NumberDomain(), distributions=TensorBackend()),
            module=module,
        )
        self.bits = tuple(bits)
        self.trace = []

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: Scope[Local, HeldValue],
        stack: list[_Frame],
        search_index: int,
    ) -> _Continue:
        law = request.arguments[1]
        assert isinstance(law, RuntimeDistribution)
        bit = self.bits[len(self.trace)]
        value = self.numbers.number("truth", bit)
        self.trace.append((request.address, law.log_prob(value)))
        return _Continue(
            current=_Returned(value=value),
            environment=environment,
            resumption_path=request.resumption_path,
        )


def _scalar_replay(module, shared, row, bits):
    owner = _entry(module, "unit")
    machine = _ScalarReplay(module, bits)
    environment = dict(zip(owner.parameters, (*shared, *row), strict=True))
    value = machine.evaluate_validated(owner.body, validate_module(module), environment)
    assert isinstance(value, torch.Tensor)
    return (
        value,
        tuple(address for address, _ in machine.trace),
        torch.stack([score for _, score in machine.trace]),
    )


def _scores(point, bits):
    s, p, _ = point
    a, b = bits
    conditional = s + Fraction(a, 4)
    q = p * conditional
    first = (1 / p) if a else (-1 / (1 - p))
    second = (1 / q) if b else (-1 / (1 - q))
    return (p if a else 1 - p, q if b else 1 - q), (
        (0, first, 0),
        (second * p, second * conditional, 0),
    )


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_every_tensor_trace_matches_scalar_values_scores_jvp_and_vjp(module):
    axis = BernoulliTensorAxis(module, _entry(module, "unit").id, "Items", 1)
    exact = _exact_batch(module)
    shared, rows = _tensor(_SHARED), _tensor(_ROWS)
    d_shared, d_rows = _tensor(_D_SHARED), _tensor(_D_ROWS)
    v_values, v_scores = _tensor(_VALUES_SEED), _tensor(_SCORES_SEED)
    joint_mass = 0.0
    averaged = torch.zeros_like(rows[:, 0])
    for selected in product(*exact.paths):
        tapes = tuple(item.tape for item in selected)
        scalar = exact.replay(tapes)
        actual = axis.replay(shared, rows, tapes)
        assert actual.source_passes == 1
        assert actual.log_scores.shape == (2, 2)
        assert actual.tapes == tapes
        torch.testing.assert_close(actual.values, _tensor(scalar.linearization.values))
        scalar_score_gradients = []
        for index, tape in enumerate(tapes):
            bits = tuple(bit for _, bit in tape.choices)
            ordinary, addresses, scores = _scalar_replay(
                module, shared, rows[index], bits
            )
            torch.testing.assert_close(actual.values[index], ordinary)
            torch.testing.assert_close(actual.log_scores[index], scores)
            assert tape.addresses == tuple(("Items", index, item) for item in addresses)
            assert tuple(address[1] for address in addresses) == (
                (("call", "toss#1"),),
                (("call", "toss#2"),),
            )
            probabilities, gradients = _scores((*_SHARED, *_ROWS[index]), bits)
            torch.testing.assert_close(scores.exp(), _tensor(probabilities))
            scalar_score_gradients.append(gradients)
        weight = actual.log_scores.sum().exp()
        torch.testing.assert_close(weight, _tensor(scalar.probability))
        joint_mass += float(weight)
        averaged += weight * actual.values

        def observed(s, x):
            measured = axis.replay(s, x, tapes)
            return measured.values, measured.log_scores

        jvp_results = torch.func.jvp(observed, (shared, rows), (d_shared, d_rows))
        jvp_values, jvp_scores = jvp_results[1]
        torch.testing.assert_close(
            jvp_values, _tensor(scalar.linearization.jvp(_D_SHARED, _D_ROWS))
        )
        torch.testing.assert_close(
            jvp_scores,
            _tensor(
                [
                    [
                        _dot(gradient, (*_D_SHARED, *_D_ROWS[index]))
                        for gradient in matrix
                    ]
                    for index, matrix in enumerate(scalar_score_gradients)
                ]
            ),
        )
        apply_transpose = torch.func.vjp(observed, shared, rows)[1]
        vjp_shared, vjp_rows = apply_transpose((v_values, v_scores))
        exact_vjp = scalar.linearization.vjp(_VALUES_SEED)
        score_vjp = [
            tuple(
                _dot(column, _SCORES_SEED[index])
                for column in zip(*matrix, strict=True)
            )
            for index, matrix in enumerate(scalar_score_gradients)
        ]
        torch.testing.assert_close(
            vjp_shared,
            _tensor((exact_vjp.shared[0] + sum(row[0] for row in score_vjp),)),
        )
        torch.testing.assert_close(
            vjp_rows,
            _tensor(
                [
                    tuple(
                        a + b
                        for a, b in zip(exact_vjp.rows[index], row[1:], strict=True)
                    )
                    for index, row in enumerate(score_vjp)
                ]
            ),
        )
        torch.testing.assert_close(
            (jvp_values * v_values).sum() + (jvp_scores * v_scores).sum(),
            (vjp_shared * d_shared).sum() + (vjp_rows * d_rows).sum(),
        )
    assert joint_mass == pytest.approx(1.0)
    torch.testing.assert_close(averaged, _tensor(exact.expected.values))

    def finite_mixture(s, x):
        value = torch.zeros_like(x[:, 0])
        for selected in product(*exact.paths):
            tapes = tuple(item.tape for item in selected)
            replay = axis.replay(s, x, tapes)
            value += replay.log_scores.sum().exp() * replay.values
        return value

    mixture = torch.func.jvp(finite_mixture, (shared, rows), (d_shared, d_rows))
    torch.testing.assert_close(mixture[0], _tensor(exact.expected.values))
    torch.testing.assert_close(
        mixture[1], _tensor(exact.expected.jvp(_D_SHARED, _D_ROWS))
    )
    mix_back = torch.func.vjp(finite_mixture, shared, rows)[1]
    mix_shared, mix_rows = mix_back(v_values)
    mix_exact = exact.expected.vjp(_VALUES_SEED)
    torch.testing.assert_close(mix_shared, _tensor(mix_exact.shared))
    torch.testing.assert_close(mix_rows, _tensor(mix_exact.rows))


def test_tensor_machine_runs_once_and_does_not_evaluate_a_dead_partial_branch(
    module, monkeypatch
):
    active = []
    reports = []
    ordinary = Evaluator.evaluate_validated

    def invocation(self, *args, **kwargs):
        reports.append(self)
        return ordinary(self, *args, **kwargs)

    monkeypatch.setattr(Evaluator, "evaluate_validated", invocation)
    original = dict(evaluator_module.IMPLEMENTATIONS)

    def active_multiplication(domain, left, right):
        active.append((left.shape, right.shape))
        return original["mul_real"](domain, left, right)

    def partial(*args):
        pytest.fail("a lazy branch must never evaluate its division by zero")

    monkeypatch.setattr(
        evaluator_module,
        "IMPLEMENTATIONS",
        {**original, "mul_real": active_multiplication, "div_real": partial},
    )
    axis = BernoulliTensorAxis(module, _entry(module, "unit").id, "Items", 1)
    shared, rows = _tensor(_SHARED), _tensor(((0.25, 0), (0.5, 2), (0.75, -2)))
    result = axis.sample(shared, rows, generator=torch.Generator().manual_seed(5))
    assert len(reports) == result.source_passes == 1
    assert result.source_steps == reports[0]._steps > 0
    assert (
        active
        == [(torch.Size([]), torch.Size([3]))]
        + [(torch.Size([3]), torch.Size([3]))] * 3
    )
    assert torch.isfinite(result.values).all()
    with pytest.raises(ZeroDivisionError, match="divides by zero"):
        original["div_real"](
            NumberDomain(), torch.zeros(3, dtype=torch.float64), _tensor(0)
        )
    singleton = axis.sample(
        shared, rows[:1], generator=torch.Generator().manual_seed(5)
    )
    assert singleton.source_steps == result.source_steps


def test_shared_probability_still_samples_each_row_and_observes_joint_law(
    module, monkeypatch
):
    axis = BernoulliTensorAxis(module, _entry(module, "same_p").id, "Items", 1)
    shared, rows = _tensor(_SHARED), _tensor(((), ()))
    actual_support = {}
    call_shapes = []

    for prescribed in product(_BOOL, repeat=2):

        def scripted(probs, *, generator):
            call_shapes.append(probs.shape)
            torch.testing.assert_close(probs, _tensor((Fraction(1, 2),) * 2))
            return _tensor(prescribed)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "bernoulli", scripted)
            got = axis.sample(shared, rows)
        observed = tuple(row.choices[0][1] for row in got.tapes)
        assert observed == prescribed
        assert tuple(tape.addresses[0][:2] for tape in got.tapes) == (
            ("Items", 0),
            ("Items", 1),
        )
        assert got.tapes[0].choices[0][0] == got.tapes[1].choices[0][0]
        assert actual_support.setdefault(
            observed, float(got.log_scores.sum().exp())
        ) == (pytest.approx(0.25))
        replay = axis.replay(shared, rows, got.tapes)
        torch.testing.assert_close(replay.values, got.values)
        torch.testing.assert_close(replay.log_scores, got.log_scores)
    assert call_shapes == [torch.Size([2])] * 4
    merged_draw = {(False, False): 0.5, (True, True): 0.5}
    for lane in range(2):
        assert sum(mass for bits, mass in actual_support.items() if bits[lane]) == (
            sum(mass for bits, mass in merged_draw.items() if bits[lane])
        )
    assert sum(mass for bits, mass in actual_support.items() if bits[0] != bits[1]) == (
        0.5
    )


def test_dynamic_branching_active_partial_primitives_and_bounds_refuse(module):
    for name in ("row_branch", "random_branch"):
        with pytest.raises(JvpUnsupported, match="literal Boolean branch"):
            BernoulliTensorAxis(module, _entry(module, name).id, "Items", 1)
    with pytest.raises(JvpUnsupported, match="div_real"):
        BernoulliTensorAxis(module, _entry(module, "active_partial").id, "Items", 1)

    owner = _entry(module, "unit")
    shared, rows = _tensor(_SHARED), _tensor(_ROWS)
    with pytest.raises(JvpLimit, match="exceeds 1 rows"):
        BernoulliTensorAxis(module, owner.id, "Items", 1, max_rows=1).sample(
            shared, rows
        )
    with pytest.raises(JvpLimit, match="budget of 2 steps"):
        BernoulliTensorAxis(module, owner.id, "Items", 1, fuel=2).sample(shared, rows)

    axis = BernoulliTensorAxis(module, owner.id, "Items", 1)
    for bad_shared, bad_rows in (
        (shared, rows[0]),
        (shared, rows[:, :1]),
        (shared, rows[:0]),
        (shared, rows.to(dtype=torch.int64)),
        (shared, torch.full_like(rows, torch.nan)),
        (shared.float(), rows),
    ):
        with pytest.raises(ValueError):
            axis.sample(bad_shared, bad_rows)
    for p in (0.0, 1.0, 1.25):
        with pytest.raises(JvpUnsupported, match="strictly between"):
            axis.sample(shared, _tensor(((p, 0), (0.5, 1))))


def test_replay_matches_exact_lane_shape_and_original_address_order(module):
    owner = _entry(module, "unit")
    axis = BernoulliTensorAxis(module, owner.id, "Items", 1)
    plan = _exact_batch(module)
    shared, rows = _tensor(_SHARED), _tensor(_ROWS)
    a, b = tuple(support[0].tape for support in plan.paths)
    for bad in (
        (a,),
        (b, a),
        (replace(a, axis="Other"), b),
        (replace(a, ordinal=False), b),
        (replace(a, choices=((a.choices[0][0], 1), a.choices[1])), b),
        (replace(a, choices=a.choices[::-1]), b),
        (replace(a, choices=a.choices[::-1]), replace(b, choices=b.choices[::-1])),
        (replace(a, choices=a.choices[:1]), replace(b, choices=b.choices[:1])),
        (
            replace(a, choices=a.choices + a.choices[:1]),
            replace(b, choices=b.choices + b.choices[:1]),
        ),
    ):
        with pytest.raises(ValueError, match="replay"):
            axis.replay(shared, rows, bad)


def test_zero_local_arity_shared_primal_expands_and_bool_conversion_is_exact(module):
    owner = _entry(module, "constant")
    axis = BernoulliTensorAxis(module, owner.id, "Items", 1)
    shared, rows = _tensor(_SHARED), _tensor(((), (), ()))
    found = axis.sample(shared, rows, generator=torch.Generator().manual_seed(27))
    torch.testing.assert_close(found.values, _tensor((1, 1, 1)))
    exact = _exact_batch(module, name="constant", rows=((), (), ()))
    torch.testing.assert_close(found.values, _tensor(exact.expected.values))

    for p, value, slope in (
        (Fraction(3, 4), Fraction(3, 4), 1),
        (Fraction(1, 4), 0, 0),
    ):
        forward = exact_jvp(module, "stable_bool", (p,), (3,))
        reverse = exact_gradient(module, "stable_bool", (p,))
        assert (forward.value, forward.tangent) == (value, 3 * slope)
        assert (reverse.value, reverse.gradient) == (value, (slope,))
    with pytest.raises(JvpBoundaryError, match="equality boundary"):
        exact_jvp(module, "stable_bool", (Fraction(1, 2),), (0,))


def test_empty_request_axis_preserves_row_extent_without_a_draw(module, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("the dead request must never draw")

    monkeypatch.setattr(torch, "bernoulli", unexpected)
    axis = BernoulliTensorAxis(module, _entry(module, "no_draw").id, "Items", 0)
    shared = torch.empty(0, dtype=torch.float32)
    rows = torch.empty((3, 0), dtype=torch.float32)
    observed = axis.sample(shared, rows)
    assert observed.values.tolist() == [2.0] * 3
    assert observed.values.dtype == observed.log_scores.dtype == torch.float32
    assert observed.log_scores.shape == (3, 0)
    assert tuple(tape.choices for tape in observed.tapes) == ((),) * 3
    assert tuple(tape.ordinal for tape in observed.tapes) == (0, 1, 2)
    copied = axis.replay(shared, rows, observed.tapes)
    torch.testing.assert_close(copied.values, observed.values)
    torch.testing.assert_close(copied.log_scores, observed.log_scores)
