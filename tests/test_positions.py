"""Computations written for one position, evaluated at many.

`quivers._positions` runs a computation once over many positions on
values whose position axis is no part of their shape. The tests state
each of its operations against one call per position: checks that refuse
a position or raise, reductions that steer an iteration, routines over
arrays, conditionals, and functions of an explicit leading axis.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Iterator

import pytest
import torch
from torch.distributions import (
    Distribution,
    Geometric,
    MultivariateNormal,
    Normal,
    Wishart,
)

from quivers import _positions
from quivers._positions import (
    PositionLayoutError,
    UnreadRefusalError,
    anywhere,
    branch,
    demand,
    depends_on_position,
    elementwise,
    entry_count,
    everywhere,
    largest,
    lift,
    map_positions,
    reject_unless,
    rejects_position,
)


@pytest.fixture(autouse=True)
def _double() -> Iterator[None]:
    """Compare in double precision."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _rows(*values: float) -> torch.Tensor:
    """One coordinate per position."""
    return torch.tensor(values).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def test_a_rejected_position_is_marked_and_the_others_run() -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z > 0, lambda: ValueError("not positive"))
        return z.log().sum()

    found = map_positions(density, _rows(1.0, -1.0, 2.0))
    assert found.accepted.tolist() == [True, False, True]
    assert found.values[2] == torch.tensor(2.0).log()
    with pytest.raises(ValueError, match="not positive"):
        density(torch.tensor([-1.0]))


def test_a_check_of_a_shared_value_raises_for_the_call() -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        reject_unless(torch.tensor(False), lambda: ValueError("shared"))
        return z.sum()

    with pytest.raises(ValueError, match="shared"):
        map_positions(density, _rows(1.0, 2.0))


def test_a_demand_raises_unless_its_positions_were_rejected_before() -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z < 10, lambda: ValueError("large"))
        demand(z < 5, lambda: ArithmeticError("defect"))
        return z.sum()

    found = map_positions(density, _rows(1.0, 20.0))
    assert found.accepted.tolist() == [True, False]
    with pytest.raises(ArithmeticError, match="defect"):
        map_positions(density, _rows(1.0, 7.0))


@pytest.mark.parametrize(
    "build",
    [
        lambda z: Normal(0.0, z[0]).log_prob(torch.tensor(0.3)),
        lambda z: MultivariateNormal(
            torch.zeros(2), covariance_matrix=z[0] * torch.eye(2)
        ).log_prob(torch.ones(2)),
        lambda z: torch.linalg.cholesky(z[0] * torch.eye(2)).sum(),
        lambda z: torch.linalg.inv((z[0] + 1.0) * torch.eye(2)).sum(),
        lambda z: torch.linalg.solve((z[0] + 1.0) * torch.eye(2), torch.ones(2)).sum(),
    ],
)
def test_validators_and_factorizations_refuse_their_positions(
    build: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    rows = _rows(1.0, -1.0, 2.0)
    found = map_positions(build, rows)
    assert found.accepted.tolist() == [True, False, True]
    for index in (0, 2):
        torch.testing.assert_close(
            found.values[index], build(rows[index]), rtol=1e-13, atol=0.0
        )
    with pytest.raises((ValueError, torch.linalg.LinAlgError)):
        build(rows[1])


@pytest.mark.parametrize(
    "build",
    [
        lambda z: Geometric(z[0]).mean,
        lambda z: Wishart(z[0], scale_tril=torch.eye(2)).log_prob(torch.eye(2)),
    ],
)
def test_a_validator_that_marks_a_defect_raises(
    build: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Their constructors raise from their own frame, which no sampler reads
    as a position."""
    with pytest.raises(ValueError):
        build(torch.tensor([0.0]))
    with pytest.raises(ValueError):
        map_positions(build, _rows(0.0, 0.0))


# ---------------------------------------------------------------------------
# Reductions over the positions
# ---------------------------------------------------------------------------


def test_reductions_read_every_position() -> None:
    read: list[tuple[bool, bool, float, int]] = []

    def density(z: torch.Tensor) -> torch.Tensor:
        read.append(
            (everywhere(z > 0), anywhere(z > 2), largest(z, 0.0), entry_count(z))
        )
        return z.sum()

    map_positions(density, torch.tensor([[1.0, 2.0], [0.5, 3.0]]))
    map_positions(density, torch.tensor([[1.0, 2.0], [-0.5, 1.0]]))
    assert read == [(True, True, 3.0, 4), (False, False, 2.0, 4)]
    assert everywhere(torch.tensor([True, True]))
    assert entry_count(torch.zeros(2, 3)) == 6
    assert largest(torch.zeros(0), 1.5) == 1.5


def test_reductions_leave_out_rejected_positions() -> None:
    """An iteration stops when the positions still accepted have converged."""
    read: list[tuple[bool, float]] = []

    def density(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z > 0, lambda: ValueError("not positive"))
        read.append((everywhere(z > 0), largest(z, -1.0)))
        return z.sum()

    map_positions(density, _rows(1.0, -7.0))
    map_positions(density, _rows(-1.0, -7.0))
    assert read == [(True, 1.0), (True, -1.0)]


def test_dependence_on_the_position() -> None:
    read: list[tuple[bool, bool]] = []

    def density(z: torch.Tensor) -> torch.Tensor:
        shared = torch.arange(3.0)
        read.append((depends_on_position(z + shared[0]), depends_on_position(shared)))
        return z.sum()

    map_positions(density, _rows(1.0, 2.0))
    assert read == [(True, False)]
    assert not depends_on_position(torch.zeros(2))


# ---------------------------------------------------------------------------
# Routines over arrays, explicit axes, and conditionals
# ---------------------------------------------------------------------------


@elementwise
def _selected(a: torch.Tensor, b: torch.Tensor, *, shift: float) -> torch.Tensor:
    """A routine that selects its entries by value."""
    a, b = torch.broadcast_tensors(a, b)
    out = a * b + shift
    chosen = a > 0
    if bool(chosen.any()):
        out = out.index_put((chosen,), a[chosen].log() + b[chosen])
    return out


@pytest.mark.parametrize("rows", [1, 3, 4])
def test_an_elementwise_routine_aligns_the_positions_ahead_of_its_axes(
    rows: int,
) -> None:
    """A scalar of the position against a vector of the same length as the
    batch is broadcast along the vector, not along the positions."""
    table = torch.arange(3.0)

    def density(z: torch.Tensor) -> torch.Tensor:
        return (
            _selected(z[0], table, shift=0.5).sum()
            + _selected(table, z, shift=0.0).square().sum()
        )

    positions = torch.randn(rows, 3, generator=torch.Generator().manual_seed(0))
    found = map_positions(density, positions)
    single = torch.stack([density(row) for row in positions])
    torch.testing.assert_close(found.values, single, rtol=1e-13, atol=0.0)


def test_an_elementwise_routine_is_differentiated() -> None:
    positions = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))

    def density(z: torch.Tensor) -> torch.Tensor:
        return _selected(z, z.flip(0), shift=0.0).sum()

    leaf = positions.clone().requires_grad_()
    (together,) = torch.autograd.grad(map_positions(density, leaf).values.sum(), leaf)
    for index, row in enumerate(positions):
        own = row.clone().requires_grad_()
        (alone,) = torch.autograd.grad(density(own), own)
        torch.testing.assert_close(together[index], alone, rtol=1e-13, atol=0.0)


def test_a_lifted_function_gets_the_axis_and_must_return_it() -> None:
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def total(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seen.append((tuple(values["z"].shape), tuple(values["shared"].shape)))
        return {"sum": (values["z"] * values["shared"]).sum(-1)}

    def density(z: torch.Tensor) -> torch.Tensor:
        return lift(total, "total", {"z": z, "shared": torch.ones(2)})["sum"]

    positions = torch.randn(5, 2)
    found = map_positions(density, positions)
    assert seen == [((5, 2), (2,))]
    assert torch.equal(found.values, positions.sum(-1))
    assert lift(total, "total", {"z": torch.ones(2), "shared": torch.ones(2)})[
        "sum"
    ] == torch.tensor(2.0)
    with pytest.raises(PositionLayoutError, match="reduction"):
        map_positions(lambda z: lift(torch.sum, "reduction", z), positions)


def test_a_check_inside_a_lifted_function_is_read_per_position() -> None:
    def inner(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z > 0, lambda: ValueError("inner"))
        return Normal(0.0, z + 1.0).log_prob(torch.tensor(0.2)).sum(-1)

    found = map_positions(lambda z: lift(inner, "inner", z), _rows(1.0, -0.5, 2.0))
    assert found.accepted.tolist() == [True, False, True]


def test_a_branch_is_computed_only_where_it_is_taken() -> None:
    """The square root is differentiated at no position below zero."""

    def above(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z < 10, lambda: ValueError("large"))
        return z.sqrt().sum()

    def below(z: torch.Tensor) -> torch.Tensor:
        return z.square().sum() * torch.tensor(3.0)

    def density(z: torch.Tensor) -> torch.Tensor:
        return branch((z > 0).all(), above, below, z)

    rows = _rows(4.0, -1.0, 0.0, 9.0, -50.0, 20.0)
    leaf = rows.clone().requires_grad_()
    found = map_positions(density, leaf)
    assert found.accepted.tolist() == [True, True, True, True, True, False]
    assert found.values[:5].tolist() == [2.0, 3.0, 0.0, 3.0, 7500.0]
    (grad,) = torch.autograd.grad(found.values[:5].sum(), leaf)
    assert grad[:5].squeeze(-1).tolist() == [0.25, -6.0, 0.0, 1 / 6, -300.0]
    assert map_positions(density, _rows(1.0, 4.0)).values.tolist() == [1.0, 2.0]


def test_branches_of_different_shapes_have_no_common_array() -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        return branch((z > 0).all(), lambda v: v, lambda v: torch.cat([v, v]), z).sum()

    assert density(torch.tensor([1.0])) == 1.0
    with pytest.raises(PositionLayoutError, match="different shapes"):
        map_positions(density, _rows(1.0, -1.0))


def test_a_function_runs_once_whatever_the_number_of_positions() -> None:
    calls: list[int] = []

    def density(z: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return z.square().sum()

    map_positions(density, torch.zeros(1000, 2))
    assert calls == [1]


def test_a_shared_operand_returned_as_it_was_is_not_taken_for_a_row() -> None:
    """Converting a datum to a row's type returns the datum itself, whose
    check is then the datum's and not the rows'."""
    datum = torch.tensor([1.0, 2.0])

    def inner(z: torch.Tensor) -> torch.Tensor:
        converted = datum.type_as(z)
        assert converted is datum
        return Normal(converted, 1.0).log_prob(z.unsqueeze(-1)).sum(-1)

    rows = torch.tensor([0.5, -0.5])
    found = map_positions(lambda z: lift(inner, "inner", z), rows)
    assert found.accepted.all()
    assert not vars(datum)
    in_place = torch.zeros(2)

    def changed(z: torch.Tensor) -> torch.Tensor:
        own = in_place.clone()
        own.add_(z)
        reject_unless(own > 0, lambda: ValueError("own"))
        return own

    found = map_positions(lambda z: lift(changed, "changed", z), rows)
    assert found.accepted.tolist() == [True, False]


# ---------------------------------------------------------------------------
# One refused position never affects another
# ---------------------------------------------------------------------------


@pytest.fixture
def _unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluate without the contract check, to reach the guard behind it."""
    monkeypatch.setattr(_positions, "_check_contract", lambda: None)


def _scale(z: torch.Tensor) -> torch.Tensor:
    """A density whose scale validator refuses a nonpositive position."""
    return Normal(0.0, z, validate_args=True).log_prob(torch.tensor(0.3)).sum()


def test_a_validator_asking_from_a_helper_is_read_per_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch may move a validator's question into a function it calls."""
    namespace: dict[str, object] = {
        "__name__": "torch.distributions.distribution",
        "torch": torch,
    }
    exec(  # noqa: S102 - the helper has to live in torch's namespace
        """
def all_true(valid, message):
    return bool(torch._is_all_true(valid))


def validate(self, value):
    valid = self.support.check(value)
    if not all_true(valid, "outside"):
        raise ValueError("outside the support")
""",
        namespace,
    )
    validate = namespace["validate"]
    assert isinstance(validate, types.FunctionType)
    verdicts = {
        **_positions._VALIDATORS,
        validate.__code__: _positions._Verdict(accepts=True, refuses=True),
    }
    monkeypatch.setattr(_positions, "_VALIDATORS", verdicts)
    monkeypatch.setattr(Distribution, "_validate_sample", validate)
    found = map_positions(
        lambda z: (
            torch.distributions.Exponential(1.0, validate_args=True).log_prob(z).sum()
        ),
        _rows(1.0, -1.0, 2.0),
    )
    assert found.accepted.tolist() == [True, False, True]


def test_an_unknown_validator_stops_the_evaluation(
    monkeypatch: pytest.MonkeyPatch, _unchecked: None
) -> None:
    """A question no verdict covers is not answered once for every position."""
    verdicts = dict(_positions._VALIDATORS)
    del verdicts[Distribution.__init__.__code__]
    monkeypatch.setattr(_positions, "_VALIDATORS", verdicts)
    with pytest.raises(UnreadRefusalError, match=torch.__version__.split("+")[0]):
        map_positions(_scale, _rows(1.0, -1.0))


@pytest.mark.parametrize(
    "verdict",
    [
        pytest.param(
            _positions._Verdict(accepts=False, refuses=True),
            id="refuses-every-position",
        ),
        pytest.param(
            _positions._Verdict(
                accepts=True,
                valid=lambda frame: torch.ones_like(frame.f_locals["valid"]),
                refuses=True,
            ),
            id="refuses-none",
        ),
        pytest.param(
            _positions._Verdict(
                accepts=True, valid=lambda frame: ~frame.f_locals["valid"], refuses=True
            ),
            id="refuses-the-others",
        ),
    ],
)
def test_a_wrong_verdict_fails_the_contract(
    monkeypatch: pytest.MonkeyPatch, verdict: _positions._Verdict
) -> None:
    """Refusing the wrong positions names the torch version."""
    verdicts = {**_positions._VALIDATORS, Distribution.__init__.__code__: verdict}
    monkeypatch.setattr(_positions, "_VALIDATORS", verdicts)
    _positions._check_contract.cache_clear()
    try:
        with pytest.raises(UnreadRefusalError, match=torch.__version__.split("+")[0]):
            map_positions(_scale, _rows(1.0, 2.0))
    finally:
        _positions._check_contract.cache_clear()


def test_the_contract_is_checked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def counted(z: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return _scale(z)

    monkeypatch.setattr(_positions, "_CONTRACT", (("counted", counted),))
    _positions._check_contract.cache_clear()
    try:
        map_positions(_scale, _rows(1.0, 2.0))
        checked = len(calls)
        map_positions(_scale, _rows(1.0, 2.0))
        assert checked and len(calls) == checked
    finally:
        _positions._check_contract.cache_clear()


@pytest.mark.parametrize("merge", [torch._is_all_true, torch._is_any_true])
def test_a_reduction_over_the_positions_is_not_read_as_one_answer(
    merge: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    with pytest.raises(UnreadRefusalError, match="every position"):
        map_positions(lambda z: z.sum() * merge(z > 0), _rows(1.0, -1.0))


def test_an_unread_factorization_stops_the_evaluation(
    monkeypatch: pytest.MonkeyPatch, _unchecked: None
) -> None:
    """A factorization that raises for the batch is not every row's refusal."""
    monkeypatch.setattr(_positions, "_FACTORIZATIONS", {})
    with pytest.raises((UnreadRefusalError, RuntimeError)) as stopped:
        map_positions(
            lambda z: torch.linalg.cholesky(z * torch.eye(2)).sum(), _rows(1.0, -1.0)
        )
    assert not rejects_position(stopped.value)


def test_a_refusal_raised_for_the_call_is_not_every_positions() -> None:
    """An error that marks a position, raised for all of them by a check no
    position accounts for."""

    def density(z: torch.Tensor) -> torch.Tensor:
        raise torch.linalg.LinAlgError("one of the matrices")

    with pytest.raises(UnreadRefusalError, match="one of the matrices"):
        map_positions(density, _rows(1.0, 2.0))


def test_an_unread_refusal_is_never_a_position_outside_the_support() -> None:
    assert not rejects_position(UnreadRefusalError("a check"))

    def inner(z: torch.Tensor) -> torch.Tensor:
        values = torch.tensor(z.detach().tolist())
        reject_unless(values > 0, lambda: ValueError("inner"))
        return z

    with pytest.raises(PositionLayoutError) as stopped:
        map_positions(lambda z: lift(inner, "inner", z).sum(), _rows(1.0, -1.0))
    assert not rejects_position(stopped.value)


def test_a_value_every_position_shares_refuses_them_all() -> None:
    with pytest.raises(ValueError) as shared:
        map_positions(lambda z: z.sum() + _scale(torch.tensor(-1.0)), _rows(1.0, 2.0))
    assert rejects_position(shared.value)


def test_a_shared_refusal_in_a_branch_refuses_the_positions_that_take_it() -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        return branch(
            (z > 0).all(),
            lambda v: v.sum(),
            lambda v: v.sum() + _scale(torch.tensor(-1.0)),
            z,
        )

    found = map_positions(density, _rows(1.0, -1.0, 3.0))
    assert found.accepted.tolist() == [True, False, True]
    assert found.values[found.accepted].tolist() == [1.0, 3.0]
    with pytest.raises(ValueError) as everyone:
        map_positions(density, _rows(-1.0, -2.0))
    assert rejects_position(everyone.value)


def test_a_refused_position_does_not_reach_a_routine_over_the_array() -> None:
    """The routine raises, and iterates, on the values it is given."""

    @elementwise
    def routine(value: torch.Tensor) -> torch.Tensor:
        if bool((value < 0).any()) or bool(value.isnan().any()):
            raise ArithmeticError("outside the routine's domain")
        return value.sqrt()

    def density(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z >= 0, lambda: ValueError("negative"))
        return routine(z).sum()

    rows = _rows(4.0, -1.0, float("nan"), 9.0).requires_grad_()
    found = map_positions(density, rows)
    assert found.accepted.tolist() == [True, False, False, True]
    assert found.values[found.accepted].tolist() == [2.0, 3.0]
    (grad,) = torch.autograd.grad(found.values[found.accepted].sum(), rows)
    assert grad.flatten().tolist()[:1] == [0.25] and grad[3].item() == 1 / 6
    none = map_positions(lambda z: lift(torch.neg, "neg", z).sum(), _rows(1.0, -1.0))
    assert none.values.tolist() == [-1.0, 1.0]


def test_a_defect_of_a_refused_position_does_not_raise() -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        reject_unless(z > 0, lambda: ValueError("negative"))
        demand(z > 0, lambda: ArithmeticError("defect"))
        return z.sum()

    assert map_positions(density, _rows(1.0, -1.0)).accepted.tolist() == [True, False]


@pytest.mark.parametrize(
    "decompose",
    [
        lambda m: torch.linalg.eigvalsh(m),
        lambda m: torch.linalg.eigh(m).eigenvalues,
        lambda m: torch.linalg.svdvals(m),
        lambda m: torch.linalg.svd(m).S,
    ],
)
def test_a_matrix_that_is_not_finite_has_no_decomposition_of_another(
    decompose: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    def density(z: torch.Tensor) -> torch.Tensor:
        return decompose(z[0] * torch.diag(torch.tensor([1.0, 2.0]))).sum()

    found = map_positions(density, _rows(1.0, float("nan"), 3.0))
    assert found.values.isnan().tolist() == [False, True, False]
    torch.testing.assert_close(found.values[[0, 2]], torch.tensor([3.0, 9.0]))


def test_the_gradient_of_one_position_holds_nothing_of_another() -> None:
    """A row whose derivative is not finite leaves the others' derivatives."""

    def density(z: torch.Tensor) -> torch.Tensor:
        return (_scale(z[:1].abs()) + z[1:].log().sum()) + torch.linalg.cholesky(
            z[:1] * torch.eye(2)
        ).diagonal().log().sum()

    rows = torch.tensor([[1.0, 2.0], [-1.0, -2.0], [2.0, 0.0], [3.0, 1.0]])
    leaf = rows.clone().requires_grad_()
    found = map_positions(density, leaf)
    kept = found.accepted & torch.isfinite(found.values)
    assert kept.tolist() == [True, False, False, True]
    (grad,) = torch.autograd.grad(found.values[kept].sum(), leaf)
    for index in (0, 3):
        own = rows[index].clone().requires_grad_()
        (alone,) = torch.autograd.grad(density(own), own)
        torch.testing.assert_close(grad[index], alone, rtol=1e-13, atol=0.0)
