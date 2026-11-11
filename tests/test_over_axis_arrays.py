"""Sites drawn over more ``over`` axes than their family's event rank.

A family of event rank ``r`` drawn ``[over=[X1, ..., Xn]]`` with
``n > r`` is an independent array over the leading ``n - r`` axes of
one variate on the trailing ``r``, after the plate of an index
annotation. The checked type, the compiled runtime's draw, and the
scored density all read the declaration that way, so a body may index
the site by every listed axis.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.dsl import loads
from quivers.dsl.compiler._prelude import CompileError
from quivers.inference import trace
from quivers.inference.registry import LatentRegistry

HEAD = """
object A : FinSet 2
object B : FinSet 3
object C : FinSet 4
object K : FinSet 5
object Val : Real 1
"""

#: (step, expected value shape, independent log density of a value)
CASES: dict[str, tuple[str, tuple[int, ...], str]] = {
    "scalar-rank-1": ("sample w : B <- Normal(loc=0.0, scale=1.5)", (3,), "normal"),
    "scalar-rank-2-plated": (
        "sample w : A <- Normal(loc=0.0, scale=1.5) [over=B]",
        (2, 3),
        "normal",
    ),
    "scalar-rank-2": (
        "sample w <- Normal(loc=0.0, scale=1.5) [over=[B, C]]",
        (3, 4),
        "normal",
    ),
    "scalar-rank-3-plated": (
        "sample w : A <- Normal(loc=0.0, scale=1.5) [over=[B, C]]",
        (2, 3, 4),
        "normal",
    ),
    "scalar-rank-3": (
        "sample w <- Normal(loc=0.0, scale=1.5) [over=[A, B, C]]",
        (2, 3, 4),
        "normal",
    ),
    "vector-rank-1": (
        "sample w <- Dirichlet(concentration=2.0) [over=K]",
        (5,),
        "dirichlet",
    ),
    "vector-rank-2-plated": (
        "sample w : A <- Dirichlet(concentration=2.0) [over=K]",
        (2, 5),
        "dirichlet",
    ),
    "vector-rank-2": (
        "sample w <- Dirichlet(concentration=2.0) [over=[B, K]]",
        (3, 5),
        "dirichlet",
    ),
    "vector-rank-3-plated": (
        "sample w : A <- Dirichlet(concentration=2.0) [over=[B, K]]",
        (2, 3, 5),
        "dirichlet",
    ),
}


def _program(step: str, extra: str = "") -> str:
    return HEAD + f"program p : A -> Val\n    {step}\n{extra}    return w\nexport p\n"


def _density(kind: str, value: torch.Tensor) -> torch.Tensor:
    if kind == "normal":
        return D.Normal(0.0, 1.5).log_prob(value).sum()
    return D.Dirichlet(torch.full((5,), 2.0)).log_prob(value).sum()


def _checked_shape(source: str) -> tuple[int, ...]:
    program = loads(source)
    entry = next(e for e in program.qiec.entries if e.name == "p")
    site = next(s for s in entry.sites if s.name == "w")
    return tuple(int(axis.size.value) for axis in (*site.batch, *site.event))


@pytest.mark.parametrize("case", sorted(CASES))
def test_checked_type_runtime_draw_and_density_agree(case: str) -> None:
    step, shape, kind = CASES[case]
    source = _program(step)
    assert _checked_shape(source) == shape
    model = loads(source).morphism
    x = torch.zeros(1, 1)
    value = trace(model, x).sites["w"].value
    assert tuple(value.shape[-len(shape) :]) == shape
    assert value.numel() == torch.Size(shape).numel()
    assert tuple(model.rsample(x).shape[-len(shape) :]) == shape
    drawn = value.reshape(shape)
    torch.testing.assert_close(
        model.log_joint(x, {"w": drawn}).sum(), _density(kind, drawn)
    )


def test_an_independent_array_is_indexed_by_every_listed_axis() -> None:
    source = (
        HEAD
        + """
program cube : A -> Val
    sample w : A <- Normal(loc=0.0, scale=1.0) [over=[B, C]]
    let t = factor a : A, b : B, c : C in w[a, b, c] * 2.0
    return t

export cube
"""
    )
    model = loads(source).morphism
    run = trace(model, torch.zeros(1, 1))
    w = run.sites["w"].value
    assert tuple(w.shape) == (2, 3, 4)
    torch.testing.assert_close(model.rsample(torch.zeros(1, 1)).shape[-3:], w.shape)
    source_value = loads(source).morphism.log_joint(torch.zeros(1, 1), {"w": w})
    assert torch.isfinite(source_value).all()


def test_an_independent_simplex_array_is_indexed_entrywise() -> None:
    source = _program(
        "sample w : A <- Dirichlet(concentration=2.0) [over=[B, K]]",
        "    let t = factor a : A, b : B in w[a, b, 4] - w[a, b, 0]\n",
    )
    model = loads(source).morphism
    w = trace(model, torch.zeros(1, 1)).sites["w"].value
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 3))


def test_the_latent_registry_reports_the_array_shape() -> None:
    model = loads(
        _program("sample w : A <- Normal(loc=0.0, scale=1.0) [over=[B, C]]")
    ).morphism
    site = LatentRegistry.from_model(model, frozenset()).sites["w"]
    assert site.constrained_shape == (2, 3, 4)
    assert site.plate_index_size == 24


def test_a_matrix_family_takes_no_independent_axes() -> None:
    source = (
        "object B : FinSet 2\nobject K : FinSet 3\nobject Val : Real 1\n"
        "program p : B -> Val\n"
        "    sample L <- LKJCholesky(concentration=2.0) [over=[B, K, K]]\n"
        "    return L\nexport p\n"
    )
    with pytest.raises(CompileError, match="takes no further independent axes"):
        loads(source)
