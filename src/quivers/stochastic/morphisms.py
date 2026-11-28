"""Stochastic morphisms: learnable Markov kernels via softmax.

Provides StochasticMorphism for row-stochastic linear maps and the
stochastic() factory function.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.core.objects import SetObject
from quivers.core.morphisms import FiniteMorphism, _MorphismModule
from quivers.core.algebras import MARKOV


class StochasticMorphism(FiniteMorphism):
    """A learnable Markov kernel backed by softmax-normalized parameters.

    Stores unconstrained real-valued logits and applies softmax along
    the codomain dimensions to produce a row-stochastic tensor (each
    fiber over a domain element sums to 1).

    This is a morphism in FinStoch: a stochastic map A → B.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    temperature : float
        Softmax temperature. Lower values → sharper distributions.
        Default 1.0.

    Examples
    --------
    >>> A = FinSet(name="A", cardinality=3)
    >>> B = FinSet(name="B", cardinality=4)
    >>> f = StochasticMorphism(A, B)
    >>> t = f.tensor  # shape (3, 4), rows sum to 1
    >>> t.sum(dim=-1)  # tensor([1., 1., 1.])
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: SetObject,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(domain, codomain, algebra=MARKOV)
        self._temperature = temperature

        shape = self.tensor_shape
        self._module = _MorphismModule()
        logits = nn.Parameter(torch.randn(shape) * 0.1)
        self._module.register_parameter("logits", logits)

    @property
    def logits(self) -> nn.Parameter:
        """Unconstrained logit parameters."""
        return self._module.logits  # type: ignore[return-value]

    @property
    def tensor(self) -> torch.Tensor:
        """Row-stochastic tensor via softmax over codomain dimensions.

        Returns
        -------
        torch.Tensor
            Tensor of shape (*domain.shape, *codomain.shape) where
            each codomain fiber sums to 1.
        """
        logits = cast(torch.Tensor, self._module.logits) / self._temperature

        # reshape to (prod(domain_shape), prod(codomain_shape))
        # apply softmax along last dim, then reshape back
        dom_size = self.domain.size
        cod_size = self.codomain.size
        flat = logits.reshape(dom_size, cod_size)
        probs = F.softmax(flat, dim=-1)
        return probs.reshape(self.tensor_shape)

    def module(self) -> nn.Module:
        """Return the ``nn.Module`` holding this morphism's parameters.

        Returns
        -------
        nn.Module
            The parameter container.
        """
        return self._module


CategoricalMorphism = StochasticMorphism
"""An alias of `StochasticMorphism` named for its categorical rows."""


def stochastic(
    domain: SetObject,
    codomain: SetObject,
    temperature: float = 1.0,
) -> StochasticMorphism:
    """Create a learnable stochastic morphism (Markov kernel).

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    temperature : float
        Softmax temperature.

    Returns
    -------
    StochasticMorphism
        A learnable row-stochastic morphism.
    """
    return StochasticMorphism(domain, codomain, temperature=temperature)


__all__ = [
    "StochasticMorphism",
    "CategoricalMorphism",
    "stochastic",
]
