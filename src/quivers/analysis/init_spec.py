"""Algebra-specific initialization recipes.

The `Algebra`'s tensor / join operations determine where in
its value space a ``k``-step composition lands when entries are
i.i.d. and roughly mean-zero. For each built-in algebra we record
the parameters of a one-dimensional initial distribution that
places the ``k``-step output in a reference region where the closed-form
approximation predicts larger gradients. The shipped recipes are:

* `ProductFuzzyAlgebra` — noisy-OR over ``k`` cells lands at
  ½ when each cell is initialized at ``p ≈ ln 2 / k``.
* `LogProbAlgebra` — log-domain; centered at ``-ln k`` so the
  exponentiated value is ``1 / k``.
* `MarkovAlgebra` — row logits centered at 0 (uniform
  prior over the row simplex).
* `MaxPlusAlgebra` / `TropicalAlgebra` — additive
  semiring on the (extended) reals; centered at 0 with spread
  ``1 / sqrt(k)``.
* `StandardMVAlgebra` — bounded-sum t-conorm hits ½ at
  roughly ``k`` terms of ``p ≈ 1 / k``.
* `BooleanAlgebra`, `GodelAlgebra` — idempotent;
  init at the algebra's mid-point ``0.5``.
* `RealAlgebra`, `BoundedSumProductAlgebra`,
  `CountingAlgebra` — sum-product semirings; we recommend
  unit-variance Normal init centered at 0 (real) or an algebra-specific
  ``1 / k`` (probability / counting). These pieces are conservative
  defaults pending a fuller writeup of the semiring case.

The `InitSpec` produced by `Algebra.init_spec` is
distribution-shaped (mean, std, lower, upper, distribution name).
Sampling itself is delegated to ``torch.nn.init`` via
`apply_init_spec`.
"""

from __future__ import annotations

import math
from typing import Literal

import didactic.api as dx
import torch
import torch.nn as nn

from quivers.analysis.chain_shape import ChainShape
from quivers.core.algebras import (
    Algebra,
    BooleanAlgebra,
    CountingAlgebra,
    GodelAlgebra,
    LogProbAlgebra,
    StandardMVAlgebra,
    MarkovAlgebra,
    MaxPlusAlgebra,
    BoundedSumProductAlgebra,
    ProductFuzzyAlgebra,
    RealAlgebra,
    TropicalAlgebra,
)
from quivers.dsl.ast_nodes import Module


class InitSpec(dx.Model):
    """Parameters of a one-dimensional initial distribution.

    Attributes
    ----------
    distribution : str
        One of ``"normal"`` / ``"uniform"`` / ``"constant"``.
    mean : float
        Location parameter for ``"normal"`` / ``"constant"``; the
        midpoint ``(lower + upper) / 2`` for ``"uniform"``.
    std : float
        Scale parameter for ``"normal"``; the half-width
        ``(upper - lower) / 2`` for ``"uniform"``; ``0.0`` for
        ``"constant"``.
    lower, upper : float
        Bounds for ``"uniform"``; for ``"normal"`` / ``"constant"``
        they record the algebra's value-space bounds if any
        (``-inf`` / ``inf`` otherwise).
    rationale : str
        One-line explanation of the recipe, suitable for surfacing
        in a warning or in ``qvr check`` output.
    """

    distribution: Literal["normal", "uniform", "constant"] = "normal"
    mean: float = 0.0
    std: float = 1.0
    lower: float = float("-inf")
    upper: float = float("inf")
    rationale: str = ""


def _algebra_init_spec(
    algebra: Algebra, depth: int, intermediate_size: int
) -> InitSpec:
    """Algebra-specific `InitSpec` for the given algebra
    at the given chain depth and per-step intermediate size.

    ``depth`` is the number of stochastic composition steps the
    initialized tensor will flow through before the observation
    step. ``intermediate_size`` is the cardinality of the shared
    axis being contracted at each step (e.g. ``|B|`` in
    ``f : A -> B`` followed by ``g : B -> C``); it controls how
    much each composition concentrates / disperses mass.
    """
    k = max(int(depth), 1)
    size = max(int(intermediate_size), 1)
    effective = k * size
    if isinstance(algebra, ProductFuzzyAlgebra):
        p = math.log(2.0) / effective
        return InitSpec(
            distribution="uniform",
            mean=p,
            std=p / 2.0,
            lower=max(0.0, p / 2.0),
            upper=min(1.0, 3.0 * p / 2.0),
            rationale=(
                f"product-fuzzy noisy-OR over k={effective} cells lands at 1/2 "
                f"when each cell is p ≈ ln(2)/k = {p:.4g}"
            ),
        )
    if isinstance(algebra, StandardMVAlgebra):
        p = 1.0 / effective
        return InitSpec(
            distribution="uniform",
            mean=p,
            std=p / 2.0,
            lower=max(0.0, p / 2.0),
            upper=min(1.0, 3.0 * p / 2.0),
            rationale=(
                f"Łukasiewicz bounded-sum hits 1/2 at p ≈ 1/k = {p:.4g} over "
                f"k={effective} terms"
            ),
        )
    if isinstance(algebra, (BooleanAlgebra, GodelAlgebra)):
        return InitSpec(
            distribution="constant",
            mean=0.5,
            std=0.0,
            lower=0.0,
            upper=1.0,
            rationale=(
                f"idempotent algebra {type(algebra).__name__}: "
                "init at the mid-point 1/2"
            ),
        )
    if isinstance(algebra, LogProbAlgebra):
        loc = -math.log(float(effective))
        return InitSpec(
            distribution="normal",
            mean=loc,
            std=0.5,
            lower=float("-inf"),
            upper=0.0,
            rationale=(
                f"log-prob centered at -ln(k) = {loc:.4g} for k={effective} steps"
            ),
        )
    if isinstance(algebra, MarkovAlgebra):
        return InitSpec(
            distribution="normal",
            mean=0.0,
            std=1.0,
            rationale=(
                "Markov row logits at zero (uniform row simplex) before "
                "softmax normalization"
            ),
        )
    if isinstance(algebra, (MaxPlusAlgebra, TropicalAlgebra)):
        spread = 1.0 / math.sqrt(float(effective))
        return InitSpec(
            distribution="normal",
            mean=0.0,
            std=spread,
            rationale=(
                f"{type(algebra).__name__} additive semiring: centered at 0, "
                f"spread 1/sqrt(k) = {spread:.4g}"
            ),
        )
    if isinstance(algebra, RealAlgebra):
        spread = 1.0 / math.sqrt(float(effective))
        return InitSpec(
            distribution="normal",
            mean=0.0,
            std=spread,
            rationale=(
                f"real sum-product semiring: centered at 0, spread "
                f"1/sqrt(k) = {spread:.4g}"
            ),
        )
    if isinstance(algebra, (BoundedSumProductAlgebra, CountingAlgebra)):
        p = 1.0 / float(effective)
        return InitSpec(
            distribution="uniform",
            mean=p,
            std=p / 2.0,
            lower=max(0.0, p / 2.0),
            upper=min(1.0, 3.0 * p / 2.0)
            if isinstance(algebra, BoundedSumProductAlgebra)
            else 3.0 * p / 2.0,
            rationale=(
                f"{type(algebra).__name__} sum-product semiring: "
                f"p ≈ 1/k = {p:.4g} keeps the k-step product at order 1"
            ),
        )
    # Unknown algebra (e.g. a user-defined CustomAlgebra). Fall back
    # to a moderate Normal that won't blow up across small chains.
    return InitSpec(
        distribution="normal",
        mean=0.0,
        std=1.0 / math.sqrt(float(effective)),
        rationale=(
            f"fallback: unknown algebra {type(algebra).__name__}, using "
            f"Normal(0, 1/sqrt(k)) with k={effective}"
        ),
    )


def recommend_init(module: Module) -> dict[str, InitSpec]:
    """Per-latent algebra-specific initialization recipe.

    Walks the program's `ChainShape`, looks up the governing
    algebra, and returns a mapping from latent variable name to the
    `InitSpec` that places the chain at the algebra's
    mid-saturation point at depth-and-size.

    Parameters
    ----------
    module : Module
        The parsed QVR module whose program is analysed.

    Returns
    -------
    dict[str, InitSpec]
        The recipe for each latent, keyed by variable name; empty when
        the module declares no governing algebra.
    """
    shape = ChainShape.from_module(module)
    algebra = shape.algebra
    if algebra is None:
        return {}
    out: dict[str, InitSpec] = {}
    for step in shape.steps:
        if step.kind != "latent":
            continue
        size = step.intermediate_size or 1
        out[step.name] = _algebra_init_spec(algebra, step.depth, size)
    return out


def apply_init_spec(parameter: nn.Parameter | torch.Tensor, spec: InitSpec) -> None:
    """Materialize an `InitSpec` onto a learnable tensor.

    Delegates to ``torch.nn.init`` for ``normal`` / ``uniform`` and
    to a plain ``fill_`` for ``constant``. The tensor's shape is
    preserved; only its entries are overwritten in-place.

    Parameters
    ----------
    parameter : nn.Parameter or torch.Tensor
        The tensor to overwrite in place.
    spec : InitSpec
        The recipe to draw the new entries from.
    """
    with torch.no_grad():
        if spec.distribution == "normal":
            nn.init.normal_(parameter, mean=spec.mean, std=spec.std)
        elif spec.distribution == "uniform":
            nn.init.uniform_(parameter, a=spec.lower, b=spec.upper)
        else:
            parameter.fill_(spec.mean)


# Patch `Algebra` with an `init_spec` method that
# returns the recipe for that algebra. Keeping the dispatch in this
# module (rather than on each Algebra subclass) keeps the algebra
# core free of analysis-layer dependencies; calling
# ``algebra.init_spec(...)`` is supported transparently.
def _init_spec_method(
    self: Algebra, depth: int, intermediate_size: int = 1
) -> InitSpec:
    """Algebra-specific `InitSpec` for ``self`` at the given
    chain depth and intermediate axis size.

    See [`quivers.analysis.init_spec.recommend_init`][quivers.analysis.init_spec.recommend_init] for the
    Module-level entry point that walks a program and returns one
    spec per latent.
    """
    return _algebra_init_spec(self, depth, intermediate_size)


Algebra.init_spec = _init_spec_method  # type: ignore[attr-defined]


__all__ = ["InitSpec", "recommend_init", "apply_init_spec"]
