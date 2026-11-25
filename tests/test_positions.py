"""Computations written for one position, evaluated at many.

`quivers._positions` runs a computation once over many positions on
values whose position axis is no part of their shape. The tests state
each of its operations against one call per position: checks that refuse
a position or raise, reductions that steer an iteration, routines over
arrays, conditionals, and functions of an explicit leading axis.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch
from torch.distributions import (
    Geometric,
    MultivariateNormal,
    Normal,
    Wishart,
)

from quivers._positions import (
    PositionLayoutError,
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
