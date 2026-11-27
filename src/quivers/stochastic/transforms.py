"""Transformations on stochastic morphisms.

Provides operations like conditioning, mixing, factoring, and
normalization that wrap existing morphisms into new ones.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from quivers.core.morphisms import FiniteMorphism, _require_finite
from quivers.core._util import EPS


class _ConditionedModule(nn.Module):
    """Module wrapper for a conditioned morphism."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class ConditionedMorphism(FiniteMorphism):
    """A morphism conditioned on evidence via Bayesian update.

    Given a morphism f: A → B and evidence tensor e over B,
    produces f|e: A → B where:

        f|e(a, b) = f(a, b) · e(b) / Σ_{b'} f(a, b') · e(b')

    This is pointwise multiplication followed by row-renormalization.

    Parameters
    ----------
    inner : FiniteMorphism
        The unconditioned morphism.
    evidence : torch.Tensor
        Evidence tensor broadcastable to the codomain shape.
        Values should be non-negative (likelihoods).
    """

    def __init__(self, inner: FiniteMorphism, evidence: torch.Tensor) -> None:
        inner = _require_finite(inner)
        super().__init__(inner.domain, inner.codomain, algebra=inner._algebra)
        self._inner = inner
        self._evidence = evidence

    @property
    def tensor(self) -> torch.Tensor:
        """Conditioned (posterior) tensor.

        Returns
        -------
        torch.Tensor
            Row-normalized tensor after pointwise evidence multiplication.
        """
        t = self._inner.tensor
        n_dom = self._inner.domain.ndim

        # broadcast evidence over domain dimensions
        ev = self._evidence

        for _ in range(n_dom):
            ev = ev.unsqueeze(0)

        weighted = t * ev

        # renormalize over codomain dimensions
        cod_dims = tuple(range(n_dom, weighted.ndim))
        z = weighted.sum(dim=cod_dims, keepdim=True).clamp(min=EPS)
        return weighted / z

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return _ConditionedModule(self._inner.module())


def condition(f: FiniteMorphism, evidence: torch.Tensor) -> ConditionedMorphism:
    """Condition a morphism on evidence (Bayesian update).

    Computes the posterior: f|e(a, b) ∝ f(a, b) · e(b),
    renormalized so rows sum to 1.

    Parameters
    ----------
    f : FiniteMorphism
        The prior morphism A → B.
    evidence : torch.Tensor
        Non-negative evidence tensor over B.

    Returns
    -------
    ConditionedMorphism
        The conditioned morphism.

    Examples
    --------
    >>> A = FinSet(name="A", cardinality=2)
    >>> B = FinSet(name="B", cardinality=3)
    >>> f = StochasticMorphism(A, B)
    >>> e = torch.tensor([1.0, 0.0, 1.0])  # observe B != 1
    >>> g = condition(f, e)
    >>> g.tensor.sum(dim=-1)  # still row-stochastic
    """
    return ConditionedMorphism(f, evidence)


class _MixedModule(nn.Module):
    """Module wrapper for a mixture of morphisms."""

    def __init__(self, left_mod: nn.Module, right_mod: nn.Module) -> None:
        super().__init__()
        self.left = left_mod
        self.right = right_mod


class MixtureMorphism(FiniteMorphism):
    """Convex combination of two morphisms with a learnable weight.

    Given morphisms f, g: A → B and a mixing weight p ∈ [0, 1]:

        mix(p, f, g)(a, b) = p · f(a, b) + (1 - p) · g(a, b)

    The weight p is parameterized via a sigmoid-transformed scalar.

    Parameters
    ----------
    left : FiniteMorphism
        First component morphism.
    right : FiniteMorphism
        Second component morphism (same domain and codomain).
    learnable : bool
        If True (default), the mixing weight is a learnable parameter.
        If False, it is fixed at the initial value.
    init_logit : float
        Initial value for the unconstrained logit of p.
        Default 0.0 (p = 0.5).

    Raises
    ------
    TypeError
        If ``left`` and ``right`` differ in domain or codomain.
    """

    def __init__(
        self,
        left: FiniteMorphism,
        right: FiniteMorphism,
        learnable: bool = True,
        init_logit: float = 0.0,
    ) -> None:
        left = _require_finite(left)
        right = _require_finite(right)
        if left.domain != right.domain or left.codomain != right.codomain:
            raise TypeError(
                f"cannot mix morphisms with different types: "
                f"{left.domain!r} -> {left.codomain!r} vs "
                f"{right.domain!r} -> {right.codomain!r}"
            )

        super().__init__(left.domain, left.codomain, algebra=left._algebra)
        self._left = left
        self._right = right

        self._mix_module = _MixedModule(left.module(), right.module())

        if learnable:
            self._mix_module.register_parameter(
                "logit_p", nn.Parameter(torch.tensor(init_logit))
            )

        else:
            self._mix_module.register_buffer("logit_p", torch.tensor(init_logit))

    @property
    def weight(self) -> torch.Tensor:
        """The mixing weight p ∈ (0, 1)."""
        return torch.sigmoid(cast(torch.Tensor, self._mix_module.logit_p))

    @property
    def tensor(self) -> torch.Tensor:
        """Convex combination: p · left + (1 - p) · right.

        Returns
        -------
        torch.Tensor
            Mixed tensor, row-stochastic if inputs are.
        """
        p = self.weight
        return p * self._left.tensor + (1.0 - p) * self._right.tensor

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return self._mix_module


def mix(
    left: FiniteMorphism,
    right: FiniteMorphism,
    learnable: bool = True,
    init_logit: float = 0.0,
) -> MixtureMorphism:
    """Create a convex combination (mixture) of two morphisms.

    Produces h: A → B where h(a,b) = p·f(a,b) + (1-p)·g(a,b).
    The mixing weight p is learnable by default.

    Parameters
    ----------
    left : FiniteMorphism
        First component.
    right : FiniteMorphism
        Second component (same type as left).
    learnable : bool
        Whether the mixing weight is trainable.
    init_logit : float
        Initial logit value for the mixing weight.

    Returns
    -------
    MixtureMorphism
        The mixture morphism.
    """
    return MixtureMorphism(left, right, learnable=learnable, init_logit=init_logit)


class _FactoredModule(nn.Module):
    """Module wrapper for a factored morphism."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class FactoredMorphism(FiniteMorphism):
    """A morphism with pointwise likelihood weighting.

    Given a morphism f: A → B and a weight tensor w over B:

        factor(f, w)(a, b) = f(a, b) · w(b)

    This is unnormalized: the result is not necessarily row-stochastic.
    Use ``normalize`` afterward if normalization is needed.

    Parameters
    ----------
    inner : FiniteMorphism
        The base morphism.
    weights : torch.Tensor
        Non-negative weight tensor broadcastable to the codomain shape.
    """

    def __init__(self, inner: FiniteMorphism, weights: torch.Tensor) -> None:
        inner = _require_finite(inner)
        super().__init__(inner.domain, inner.codomain, algebra=inner._algebra)
        self._inner = inner
        self._weights = weights

    @property
    def tensor(self) -> torch.Tensor:
        """Pointwise-weighted tensor (unnormalized).

        Returns
        -------
        torch.Tensor
            f(a, b) · w(b) for each (a, b).
        """
        t = self._inner.tensor
        n_dom = self._inner.domain.ndim

        w = self._weights

        for _ in range(n_dom):
            w = w.unsqueeze(0)

        return t * w

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return _FactoredModule(self._inner.module())


def factor(f: FiniteMorphism, weights: torch.Tensor) -> FactoredMorphism:
    """Apply pointwise likelihood weighting to a morphism.

    Computes: factor(f, w)(a, b) = f(a, b) · w(b).
    The result is unnormalized.

    Parameters
    ----------
    f : FiniteMorphism
        The base morphism.
    weights : torch.Tensor
        Non-negative weight tensor over the codomain.

    Returns
    -------
    FactoredMorphism
        The weighted morphism.
    """
    return FactoredMorphism(f, weights)


class _NormalizedModule(nn.Module):
    """Module wrapper for a normalized morphism."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class NormalizedMorphism(FiniteMorphism):
    """A morphism with row-renormalized tensor.

    Divides each codomain fiber by its sum so that rows sum to 1.

    Parameters
    ----------
    inner : FiniteMorphism
        The unnormalized morphism.
    """

    def __init__(self, inner: FiniteMorphism) -> None:
        inner = _require_finite(inner)
        super().__init__(inner.domain, inner.codomain, algebra=inner._algebra)
        self._inner = inner

    @property
    def tensor(self) -> torch.Tensor:
        """Row-normalized tensor.

        Returns
        -------
        torch.Tensor
            Tensor where codomain fibers sum to 1.
        """
        t = self._inner.tensor
        n_dom = self._inner.domain.ndim
        cod_dims = tuple(range(n_dom, t.ndim))
        z = t.sum(dim=cod_dims, keepdim=True).clamp(min=EPS)
        return t / z

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return _NormalizedModule(self._inner.module())


def normalize(f: FiniteMorphism) -> NormalizedMorphism:
    """Renormalize a morphism so codomain fibers sum to 1.

    Parameters
    ----------
    f : FiniteMorphism
        The unnormalized morphism.

    Returns
    -------
    NormalizedMorphism
        The row-normalized morphism.
    """
    return NormalizedMorphism(f)


__all__ = [
    "ConditionedMorphism",
    "condition",
    "MixtureMorphism",
    "mix",
    "FactoredMorphism",
    "factor",
    "NormalizedMorphism",
    "normalize",
]
