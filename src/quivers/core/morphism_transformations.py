"""Change of base by shape-aware morphism transformations.

A `quivers.core.algebra_morphisms.CarrierMap`
acts pointwise on tensor entries (``log``, ``threshold``, ``max-plus
lift``). That handles a useful but narrow class of change-of-base
moves. A transformation whose action on a morphism's tensor is
**shape-aware** rather than strictly pointwise covers softmax-style normalizations,
Bayes-inversion under a prior, L1 / L2 / spectral row
normalizations, top-k truncation, and Sinkhorn balancing.

This module ships the abstract base
`MorphismTransformation` and four concrete subclasses
(Softmax, L1Normalize, L2Normalize, BayesInvert) that cover the
common practical needs. ``Morphism.change_base`` accepts either a
``CarrierMap`` (pointwise) or a ``MorphismTransformation``
(shape-aware) and dispatches accordingly.

A ``MorphismTransformation`` is a map from morphisms over ``V`` to
morphisms over ``W`` that acts on whole rows, columns or matrices and
may carry parameter data (``BayesInvert`` carries a prior). It promises
no functoriality: normalizing a composite is in general not the
composite of the normalized morphisms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from quivers.core._util import EPS
from quivers.core.objects import ProductSet, SetObject
from quivers.core.algebras import (
    MARKOV,
    PRODUCT_FUZZY,
    REAL,
    Algebra,
)


class MorphismTransformation(ABC):
    """A shape-aware change-of-base transformation.

    Concrete subclasses act on a morphism's whole tensor (not
    entry-by-entry), optionally consuming an explicit axis
    specification or external parameters (e.g. a prior). The
    ``source`` algebra is the domain of the action; the
    ``target`` algebra is the resulting morphism's algebra.

    The base contract is:

        morphism :: V-Cat(A, B)  -->  V-Cat(A', B')

    where ``(A', B')`` is determined by the transformation
    (often the same as ``(A, B)`` for normalizations; swapped
    for Bayes inversion).
    """

    @property
    @abstractmethod
    def source(self) -> Algebra:
        """The source algebra ``V``."""

    @property
    @abstractmethod
    def target(self) -> Algebra:
        """The target algebra ``W``."""

    @abstractmethod
    def apply(self, tensor: torch.Tensor, morphism) -> torch.Tensor:
        """Transform ``morphism.tensor`` into a tensor over the target
        algebra.

        Parameters
        ----------
        tensor : torch.Tensor
            The morphism's tensor at the current algebra.
        morphism : Morphism
            The full morphism, exposed so the transformation can
            consult domain / codomain shape and pick the right axis.
        """

    def new_domain(self, morphism) -> SetObject:
        """The transformed morphism's domain. Default: unchanged."""
        return morphism.domain

    def new_codomain(self, morphism) -> SetObject:
        """The transformed morphism's codomain. Default: unchanged."""
        return morphism.codomain

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.source.name} -> {self.target.name})"


def _axis_index(morphism, axis_object: SetObject) -> int:
    """Resolve a domain / codomain object to its axis index in the
    morphism's tensor. Domain axes come first; codomain axes follow.

    For a morphism ``f : A -> B`` with tensor shape
    ``(|A|, |B|)``, ``_axis_index(f, A)`` returns 0 and
    ``_axis_index(f, B)`` returns 1.
    """
    domain_objects = _decompose_object(morphism.domain)
    codomain_objects = _decompose_object(morphism.codomain)
    all_objects = domain_objects + codomain_objects
    for i, obj in enumerate(all_objects):
        if obj is axis_object or obj == axis_object:
            return i
    raise ValueError(
        f"axis object {axis_object.name!r} not found in "
        f"morphism shape {[o.name for o in all_objects]}"
    )


def _decompose_object(obj) -> list:
    """Flatten a ProductSet into its constituent atomic objects."""
    if isinstance(obj, ProductSet):
        out = []
        for c in obj.components:
            out.extend(_decompose_object(c))
        return out
    return [obj]


class Softmax(MorphismTransformation):
    """Softmax-normalize along an axis, producing a Markov-style kernel.

    Maps a ``ProductFuzzyAlgebra`` (or ``Real``) tensor to a row-stochastic
    Markov kernel by applying ``softmax`` along the axis indexed by
    ``axis_object``. The output's entries lie in ``(0, 1)`` and
    sum to 1 along that axis.

    Parameters
    ----------
    axis_object : SetObject
        The object whose axis the softmax is taken over.
    source : Algebra, optional
        The morphism's algebra before softmax. Default
        ``PRODUCT_FUZZY``.
    """

    def __init__(
        self,
        axis_object: SetObject,
        source: Algebra = PRODUCT_FUZZY,
    ) -> None:

        self._axis_object = axis_object
        self._source = source
        self._target = MARKOV

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    @property
    def axis_object(self) -> SetObject:
        return self._axis_object

    def apply(self, tensor: torch.Tensor, morphism) -> torch.Tensor:
        axis = _axis_index(morphism, self._axis_object)
        return F.softmax(tensor, dim=axis)


class L1Normalize(MorphismTransformation):
    """L1-normalize along an axis: divide each entry by the row sum.

    Maps a non-negative real tensor to one whose entries sum to
    1 along the given axis. The output is row-stochastic on that
    axis; semantically identical to a Markov kernel.

    Parameters
    ----------
    axis_object : SetObject
        The object whose axis is normalized.
    source : Algebra, optional
        Default `REAL`.
    """

    def __init__(
        self,
        axis_object: SetObject,
        source: Algebra = REAL,
    ) -> None:

        self._axis_object = axis_object
        self._source = source
        self._target = MARKOV

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, tensor: torch.Tensor, morphism) -> torch.Tensor:
        axis = _axis_index(morphism, self._axis_object)
        nonneg = tensor.clamp(min=0.0)
        total = nonneg.sum(dim=axis, keepdim=True).clamp(min=EPS)
        return nonneg / total


class L2Normalize(MorphismTransformation):
    """L2-normalize along an axis: divide each entry by the row norm.

    Maps a real-valued tensor to one whose entries lie on the unit
    L2 sphere along the given axis. The result is still in the
    source algebra (Real), with rows rescaled to unit length.

    Parameters
    ----------
    axis_object : SetObject
        The object whose axis is normalized.
    source : Algebra, optional
        Default `REAL`. The target equals the source.
    """

    def __init__(
        self,
        axis_object: SetObject,
        source: Algebra = REAL,
    ) -> None:
        self._axis_object = axis_object
        self._source = source
        self._target = source

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    def apply(self, tensor: torch.Tensor, morphism) -> torch.Tensor:
        axis = _axis_index(morphism, self._axis_object)
        norm = tensor.pow(2).sum(dim=axis, keepdim=True).sqrt().clamp(min=EPS)
        return tensor / norm


class BayesInvert(MorphismTransformation):
    """Bayes-invert a Markov kernel under a prior.

    Given a Markov kernel ``f : A -> B`` and a prior
    ``pi : 1 -> A``, the Bayes-inverse ``f^{-1}_pi : B -> A`` is
    the kernel satisfying

        f^{-1}_pi(a | b) = f(b | a) * pi(a) / sum_a' f(b | a') * pi(a')

    Mathematically this is the disintegration of the joint
    distribution along the second margin. The domain and codomain
    swap roles in the result.

    Parameters
    ----------
    prior : torch.Tensor
        Prior probabilities over the source's first object.
        Shape ``(|A|,)``; must sum to 1.
    """

    def __init__(self, prior: torch.Tensor) -> None:

        if prior.dim() != 1:
            raise ValueError(f"BayesInvert: prior must be 1-D; got {prior.dim()}-D")
        if not torch.isclose(prior.sum(), torch.ones(())):
            raise ValueError(
                f"BayesInvert: prior must sum to 1; got sum={float(prior.sum()):.4f}"
            )
        if (prior < 0).any():
            raise ValueError("BayesInvert: prior entries must be non-negative")
        self._prior = prior.clone()
        self._source = MARKOV
        self._target = MARKOV

    @property
    def source(self) -> Algebra:
        return self._source

    @property
    def target(self) -> Algebra:
        return self._target

    @property
    def prior(self) -> torch.Tensor:
        return self._prior.clone()

    def apply(self, tensor: torch.Tensor, morphism) -> torch.Tensor:
        if tensor.dim() != 2:
            raise ValueError(
                f"BayesInvert: expected 2-D Markov kernel; got {tensor.dim()}-D"
            )
        if tensor.shape[0] != self._prior.shape[0]:
            raise ValueError(
                f"BayesInvert: prior length {self._prior.shape[0]} "
                f"!= kernel rows {tensor.shape[0]}"
            )
        # Joint over (a, b) is prior[a] * tensor[a, b].
        joint = self._prior.unsqueeze(-1) * tensor
        # Marginal over b is sum_a joint[a, b].
        marginal_b = joint.sum(dim=0, keepdim=True).clamp(min=EPS)
        # Posterior is joint / marginal_b, transposed so the
        # result is (b, a) — the inverse kernel.
        posterior = (joint / marginal_b).t().contiguous()
        return posterior

    def new_domain(self, morphism) -> SetObject:
        # Bayes inversion swaps domain and codomain.
        return morphism.codomain

    def new_codomain(self, morphism) -> SetObject:
        return morphism.domain


# ---------------------------------------------------------------------------
# Factory functions exposed in the DSL transformation catalog.
#
# Each factory accepts compile-time values resolved from the DSL's
# surrounding scope (objects, morphisms) and returns a fully-
# constructed `MorphismTransformation`. The compiler's
# transformation catalog binds them so the user can write
# ``f.change_base(softmax(B))`` / ``f.change_base(bayes_invert(prior))``
# in pure QVR.
# ---------------------------------------------------------------------------


def softmax(axis_object: SetObject) -> Softmax:
    """Build a `Softmax` transformation along ``axis_object``.

    Parameters
    ----------
    axis_object : SetObject
        The object whose axis the softmax is taken over.

    Returns
    -------
    Softmax
        The ProductFuzzy-to-Markov transformation.
    """
    return Softmax(axis_object)


def l1_normalize(axis_object: SetObject) -> L1Normalize:
    """Build an `L1Normalize` transformation along ``axis_object``.

    Parameters
    ----------
    axis_object : SetObject
        The object whose axis is normalized.

    Returns
    -------
    L1Normalize
        The Real-to-Markov transformation.
    """
    return L1Normalize(axis_object)


def l2_normalize(axis_object: SetObject) -> L2Normalize:
    """Build an `L2Normalize` transformation along ``axis_object``.

    Parameters
    ----------
    axis_object : SetObject
        The object whose axis is normalized.

    Returns
    -------
    L2Normalize
        The Real-to-Real transformation.
    """
    return L2Normalize(axis_object)


def bayes_invert(prior) -> BayesInvert:
    """Build a `BayesInvert` transformation from a prior.

    ``prior`` may be a 1-D `torch.Tensor` (used directly)
    or any object that exposes a ``.tensor`` attribute (the
    morphism convention used throughout the V-Cat layer). Duck
    typing on ``.tensor`` avoids a circular import between
    [`quivers.core.morphisms`][quivers.core.morphisms] and this module. The morphism
    form is what the DSL feeds in when the user writes
    ``change_base(bayes_invert(prior_morph))``.

    Parameters
    ----------
    prior : torch.Tensor or Morphism
        The prior over the kernel's domain, as a tensor or as a
        morphism whose ``tensor`` holds it. A prior that does not sum
        to 1 is clamped to be non-negative and renormalized.

    Returns
    -------
    BayesInvert
        The Bayes-inversion transformation under ``prior``.

    Raises
    ------
    TypeError
        If ``prior`` is neither a tensor nor carries a ``tensor``.
    ValueError
        If ``prior`` sums to zero.
    """
    if isinstance(prior, torch.Tensor):
        tensor = prior
    elif hasattr(prior, "tensor"):
        tensor = prior.tensor.detach().clone()
    else:
        raise TypeError(
            f"bayes_invert: expected Tensor or .tensor-bearing "
            f"morphism prior; got {type(prior).__name__}"
        )
    flat = tensor.flatten()
    total = flat.sum()
    if total.abs() < 1e-12:
        raise ValueError("bayes_invert: prior sums to zero; cannot normalize")
    if not torch.isclose(total, torch.tensor(1.0), atol=1e-5):
        flat = flat.clamp(min=0.0)
        flat = flat / flat.sum().clamp(min=1e-12)
    return BayesInvert(flat)


__all__ = [
    "BayesInvert",
    "L1Normalize",
    "L2Normalize",
    "MorphismTransformation",
    "Softmax",
    "bayes_invert",
    "l1_normalize",
    "l2_normalize",
    "softmax",
]
