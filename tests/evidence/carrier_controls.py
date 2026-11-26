"""Checks and deliberately wrong versions of the statistics carriers.

The checks compare the carriers on every path of a layered graph with
enumeration of the paths in exact rational arithmetic (O1, O5), with the
gradient autograd gives through the plain sum (O7), and with closed
forms of geometric series for the stars; one more reduces over an empty
axis.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from fractions import Fraction

import torch

from quivers.core.algebras import LOG_PROB, REAL, FreeSemimodule
from quivers.core.carriers import (
    ExpectationSemiring,
    ProductSemiring,
    SecondOrderExpectationSemiring,
)
from quivers.core.algebras import MAX_PLUS
from tests.evidence.checks._registry import Check
from tests.evidence.compare import Accuracy, Mismatch, compare_number
from tests.evidence.mutations import Mutation, Rewrite

LAYERS = 3
"""Edges of a path."""

WIDTH = 3
"""Nodes per layer."""

WEIGHT = [
    [
        [Fraction(1 + (3 * a + 5 * b + 7 * c) % 5, 4) for c in range(WIDTH)]
        for b in range(WIDTH)
    ]
    for a in range(LAYERS)
]
"""``WEIGHT[layer][source][target]``: positive rational edge weights."""

FEATURE = [
    [
        [
            (Fraction((a + 2 * b + c) % 3 - 1), Fraction((2 * a + b + 3 * c) % 4, 2))
            for c in range(WIDTH)
        ]
        for b in range(WIDTH)
    ]
    for a in range(LAYERS)
]
"""Two features of each edge, signed."""

type Path = tuple[int, ...]
"""The nodes of a path, one per layer boundary."""


def paths() -> list[Path]:
    """Every path through the layers.

    Returns
    -------
    list of Path
        ``WIDTH ** (LAYERS + 1)`` paths.
    """
    return list(itertools.product(range(WIDTH), repeat=LAYERS + 1))


def path_weight(path: Path) -> Fraction:
    """The product of a path's edge weights.

    Parameters
    ----------
    path : Path
        The path.

    Returns
    -------
    Fraction
        Its weight.
    """
    weight = Fraction(1)
    for layer in range(LAYERS):
        weight *= WEIGHT[layer][path[layer]][path[layer + 1]]
    return weight


def path_feature(path: Path, index: int) -> Fraction:
    """The sum of one feature over a path's edges.

    Parameters
    ----------
    path : Path
        The path.
    index : int
        The feature.

    Returns
    -------
    Fraction
        Its total.
    """
    return sum(
        (
            FEATURE[layer][path[layer]][path[layer + 1]][index]
            for layer in range(LAYERS)
        ),
        Fraction(0),
    )


def _tensor(values: list[list[list[Fraction]]]) -> torch.Tensor:
    """A float64 tensor of nested rationals.

    Parameters
    ----------
    values : list
        The rationals.

    Returns
    -------
    torch.Tensor
        The tensor.
    """
    return torch.tensor(
        [[[float(v) for v in row] for row in layer] for layer in values],
        dtype=torch.float64,
    )


def _features() -> torch.Tensor:
    """The features as a tensor of shape ``(LAYERS, WIDTH, WIDTH, 2)``.

    Returns
    -------
    torch.Tensor
        The features.
    """
    return torch.tensor(
        [
            [[[float(f) for f in cell] for cell in row] for row in layer]
            for layer in FEATURE
        ],
        dtype=torch.float64,
    )


def _chain(
    carrier: ExpectationSemiring | SecondOrderExpectationSemiring,
    layers: torch.Tensor,
) -> torch.Tensor:
    """The sum over every path of the product of its elements.

    Parameters
    ----------
    carrier : ExpectationSemiring or SecondOrderExpectationSemiring
        The carrier.
    layers : torch.Tensor
        Packed elements of shape ``(LAYERS, WIDTH, WIDTH, size)``.

    Returns
    -------
    torch.Tensor
        One packed element.
    """
    total = layers[0]
    for layer in range(1, LAYERS):
        total = carrier.compose(total, layers[layer], 1)
    return carrier.join(total, (0, 1))


def _require(condition: bool, what: str) -> None:
    """Fail a check with an infinite deviation.

    Parameters
    ----------
    condition : bool
        What has to hold.
    what : str
        The failure.

    Raises
    ------
    Mismatch
        If the condition does not hold.
    """
    if not condition:
        raise Mismatch("value", math.inf, what)


def check_expectation() -> None:
    """Require the weight and the expected features of every path.

    Raises
    ------
    Mismatch
        If either differs from enumeration in rationals.
    """
    carrier = ExpectationSemiring(FreeSemimodule(REAL, 2))
    weights = _tensor(WEIGHT)
    features = _features()
    total = _chain(carrier, carrier.pack(weights, weights.unsqueeze(-1) * features))
    every = paths()
    partition = sum((path_weight(p) for p in every), Fraction(0))
    accuracy = Accuracy(domain="linear", operations=64)
    compare_number(float(total[0]), partition, accuracy, "weight")
    for index in range(2):
        expected = sum(
            (path_weight(p) * path_feature(p, index) for p in every), Fraction(0)
        )
        compare_number(float(total[1 + index]), expected, accuracy, f"feature {index}")


def check_gradient() -> None:
    """Require the expected features to equal the gradient through the sum.

    With edge weights ``exp(θ · f)`` the partition's gradient in ``θ`` is
    the sum over paths of the path weight times its features, which the
    carrier computes forward.

    Raises
    ------
    Mismatch
        If the carrier's statistic differs from autograd's gradient.
    """
    theta = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    features = _features()
    weights = torch.exp(features @ theta)
    product = weights[0]
    for layer in range(1, LAYERS):
        product = product @ weights[layer]
    (gradient,) = torch.autograd.grad(product.sum(), theta)
    carrier = ExpectationSemiring(FreeSemimodule(REAL, 2))
    plain = weights.detach()
    total = _chain(carrier, carrier.pack(plain, plain.unsqueeze(-1) * features))
    for index in range(2):
        compare_number(
            float(total[1 + index]),
            float(gradient[index]),
            Accuracy(domain="linear", operations=128),
            f"gradient {index}",
        )


def check_second_order() -> None:
    """Require the expected product of two path features.

    Each edge carries ``<p, p f0, p f1, p f0 f1>`` (Li and Eisner 2009,
    section 4), so the last part sums ``p(d) f0(d) f1(d)`` over paths.

    Raises
    ------
    Mismatch
        If the last part differs from enumeration in rationals.
    """
    carrier = SecondOrderExpectationSemiring(REAL, 1, 1)
    weights = _tensor(WEIGHT)
    features = _features()
    r = weights.unsqueeze(-1) * features[..., :1]
    s = weights.unsqueeze(-1) * features[..., 1:]
    total = _chain(
        carrier,
        carrier.pack(
            weights,
            r,
            s,
            (weights * features[..., 0] * features[..., 1]).reshape(
                *weights.shape, 1, 1
            ),
        ),
    )
    every = paths()
    expected = sum(
        (path_weight(p) * path_feature(p, 0) * path_feature(p, 1) for p in every),
        Fraction(0),
    )
    compare_number(
        float(total[-1]), expected, Accuracy(domain="linear", operations=128), "rs"
    )


def check_stars() -> None:
    """Require the stars to be the sums of the geometric series.

    For ``x = <p, r>`` the series has weight ``1 / (1 - p)`` and statistic
    ``sum_k k p^(k-1) r = r / (1 - p)^2``; for the second order the last
    part is ``t / (1 - p)^2 + 2 r s / (1 - p)^3``. For a product, each
    factor's star.

    Raises
    ------
    Mismatch
        If a star differs from its series.
    """
    p, r, s, t = Fraction(1, 3), Fraction(2), Fraction(-1, 2), Fraction(5)
    star = 1 / (1 - p)
    first = ExpectationSemiring(FreeSemimodule(REAL, 1))
    element = first.pack(
        torch.tensor(float(p), dtype=torch.float64),
        torch.tensor([float(r)], dtype=torch.float64),
    )
    found = first.star(element, lambda w: 1 / (1 - w))
    accuracy = Accuracy(domain="linear", operations=16)
    compare_number(float(found[0]), star, accuracy, "expectation weight")
    compare_number(float(found[1]), r * star**2, accuracy, "expectation statistic")
    second = SecondOrderExpectationSemiring(REAL, 1, 1)
    packed = second.pack(
        torch.tensor(float(p), dtype=torch.float64),
        torch.tensor([float(r)], dtype=torch.float64),
        torch.tensor([float(s)], dtype=torch.float64),
        torch.tensor([[float(t)]], dtype=torch.float64),
    )
    starred = second.star(packed, lambda w: 1 / (1 - w))
    compare_number(
        float(starred[-1]), t * star**2 + 2 * r * s * star**3, accuracy, "second order"
    )
    product = ProductSemiring([LOG_PROB, MAX_PLUS])
    pair = torch.tensor([-1.0, -0.5], dtype=torch.float64)
    stars: list[Callable[[torch.Tensor], torch.Tensor]] = [
        lambda w: -torch.log1p(-torch.exp(w)),
        lambda w: torch.zeros_like(w),
    ]
    both = product.star(pair, stars)
    compare_number(float(both[0]), -math.log1p(-math.exp(-1.0)), accuracy, "log star")
    _require(float(both[1]) == 0.0, "the max-plus star of a negative score is 0")


def check_empty() -> None:
    """Require a sum over an empty axis to be the zero, with zero derivative.

    Raises
    ------
    Mismatch
        If the shape, dtype, value or derivative differs.
    """
    for carrier in (
        ExpectationSemiring(FreeSemimodule(LOG_PROB, 2)),
        SecondOrderExpectationSemiring(REAL, 1, 2),
        ProductSemiring([LOG_PROB, MAX_PLUS]),
    ):
        size = carrier.element_shape[0]
        empty = torch.zeros(3, 0, size, dtype=torch.float32, requires_grad=True)
        total = carrier.join(empty, 1)
        zero = carrier.zero_like((3,), empty)
        _require(
            tuple(total.shape) == (3, size) and total.dtype == torch.float32,
            f"{carrier.name}: an empty sum has shape {tuple(total.shape)} {total.dtype}",
        )
        _require(
            bool((total == zero).all()), f"{carrier.name}: an empty sum is {total}"
        )
        (gradient,) = torch.autograd.grad(
            torch.where(torch.isfinite(total), total, 0.0).sum(),
            empty,
            allow_unused=True,
        )
        _require(
            gradient is None or bool(torch.isfinite(gradient).all()),
            f"{carrier.name}: an empty sum has a derivative that is not finite",
        )


CHECKS: dict[str, Check] = {
    name: Check(name=name, executor="carriers", question="Q11", run=run)
    for name, run in (
        ("carriers/expectation", check_expectation),
        ("carriers/gradient", check_gradient),
        ("carriers/second-order", check_second_order),
        ("carriers/stars", check_stars),
        ("carriers/empty", check_empty),
    )
}
"""Every check of the carriers by name."""

MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        name="empty-carrier-sum-is-the-unit",
        defect_class="empty reduction",
        provenance="a sum over no terms returns the unit, as a product over no terms does",
        target="quivers.core.carriers:_Packed.join",
        rewrites=(
            Rewrite(
                old="return self.empty_join(t, axes)",
                new=(
                    "return self.unit_like(tuple(n for i, n in enumerate(t.shape[:-1]) "
                    "if i not in axes), t)"
                ),
            ),
        ),
        rejected_by=("carriers/empty",),
        min_effect=1.0,
    ),
    Mutation(
        name="expectation-statistic-drops-a-term",
        defect_class="gradient path",
        provenance="the product keeps only the left factor's statistic",
        target="quivers.core.carriers:ExpectationSemiring.tensor_op",
        rewrites=(
            Rewrite(
                old="self._module.act(p1, r2), self._module.act(p2, r1)",
                new="self._module.act(p2, r1), self._module.act(p2, r1)",
            ),
        ),
        rejected_by=("carriers/expectation", "carriers/gradient"),
        min_effect=0.01,
    ),
    Mutation(
        name="second-order-cross-term-dropped",
        defect_class="aggregation",
        provenance="the product of two statistics forgets one cross term",
        target="quivers.core.carriers:SecondOrderExpectationSemiring.tensor_op",
        rewrites=(
            Rewrite(
                old="outer(r2.unsqueeze(-1), s1.unsqueeze(-2)),",
                new="outer(r2.unsqueeze(-1), torch.zeros_like(s1).unsqueeze(-2)),",
            ),
        ),
        rejected_by=("carriers/second-order",),
        min_effect=0.01,
    ),
)
"""Every wrong version of the carriers and the checks that reject it."""
