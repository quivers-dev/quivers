"""Traced monoidal categories and the canonical trace of V-relations.

A traced monoidal category is a symmetric monoidal category (C, ⊗, I)
equipped with a trace operator

    Tr^U_{A,B}: C(A ⊗ U, B ⊗ U) → C(A, B)

satisfying naturality, dinaturality, vanishing, superposing, and
yanking [Joyal, Street and Verity (1996)].

V-valued matrices with the monoidal structure given by the cartesian
product of sets form a compact closed category when V is a commutative
semiring, and every compact closed category has a canonical trace,

    Tr^U(f) = (id_B ⊗ ε_U) ∘ (f ⊗ id_U) ∘ (id_A ⊗ η_U),

which for V-matrices is the sum over the feedback variable:

    Tr^U(f)(a, b) = ⋁_u f((a, u), (b, u)).

For Boolean relations this is the existential trace of Rel, and for
real matrices the partial trace. The trace axioms need V to be a
`Semiring`, so the classes and functions here require one.

This module provides:

    TracedMonoidal (abstract)
    └── CompactClosedTrace — the canonical trace of V-matrices

    trace()              — apply the trace operator
    partial_trace()      — trace over a subset of feedback wires

References
----------
Joyal, A., Street, R., and Verity, D. (1996). Traced monoidal
categories. *Mathematical Proceedings of the Cambridge Philosophical
Society*, 119(3), 447-468.
[doi:10.1017/S0305004100074338](https://doi.org/10.1017/S0305004100074338)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import itertools

import torch

from quivers.core.objects import SetObject, ProductSet
from quivers.core.morphisms import Morphism, observed, identity
from quivers.core.algebras import Algebra, CompositionRule, Semiring
from quivers.categorical.monoidal import MonoidalStructure, CartesianProductMonoidal


def _require_semiring(algebra: CompositionRule) -> Semiring:
    """Return ``algebra`` as a `Semiring`, or raise.

    Parameters
    ----------
    algebra : CompositionRule
        The algebra a trace was asked to use.

    Returns
    -------
    Semiring
        ``algebra``.

    Raises
    ------
    TypeError
        If ``algebra`` is not a `Semiring`.
    """
    if not isinstance(algebra, Semiring):
        raise TypeError(
            "the trace axioms need a Semiring algebra, under which V-matrices "
            f"form a compact closed category; got {algebra!r}"
        )
    return algebra


class TracedMonoidal(ABC):
    """Abstract traced monoidal category.

    Provides the trace operator Tr^U_{A,B} on a monoidal structure.

    Parameters
    ----------
    monoidal : MonoidalStructure
        The underlying monoidal structure.
    algebra : Semiring
        The enrichment algebra.

    Raises
    ------
    TypeError
        If ``algebra`` is not a `Semiring`.
    """

    def __init__(
        self,
        monoidal: MonoidalStructure,
        algebra: Semiring,
    ) -> None:
        self._monoidal = monoidal
        self._algebra = _require_semiring(algebra)

    @property
    def monoidal(self) -> MonoidalStructure:
        """The underlying monoidal structure."""
        return self._monoidal

    @property
    def algebra(self) -> Semiring:
        """The enrichment algebra."""
        return self._algebra

    @abstractmethod
    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> Morphism:
        """Apply the trace operator.

        Given f: A ⊗ U → B ⊗ U, compute Tr^U(f): A → B.

        Parameters
        ----------
        morph : Morphism
            The morphism f: A ⊗ U → B ⊗ U.
        feedback : SetObject
            The feedback object U.
        domain : SetObject
            The external domain A.
        codomain : SetObject
            The external codomain B.

        Returns
        -------
        Morphism
            The traced morphism Tr^U(f): A → B.
        """
        ...

    def verify_yanking(
        self,
        feedback: SetObject,
        atol: float = 1e-5,
    ) -> bool:
        """Verify the yanking axiom: Tr^U(σ_{U,U}) = id_U.

        The trace of the braiding/swap should be the identity.

        Parameters
        ----------
        feedback : SetObject
            The object U to test.
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if yanking holds within tolerance.
        """
        # σ_{U,U}: U ⊗ U → U ⊗ U (swap)
        if isinstance(self._monoidal, CartesianProductMonoidal):
            swap = self._monoidal.braiding(feedback, feedback)
            traced = self.trace(swap, feedback, feedback, feedback)
            expected = identity(feedback, algebra=self._algebra).tensor

            return torch.allclose(traced.tensor, expected, atol=atol)

        return True  # skip for non-cartesian

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self._monoidal!r})"


class CompactClosedTrace(TracedMonoidal):
    """The canonical trace of V-matrices under the cartesian product of sets.

    For a morphism f: A × U → B × U,

        Tr^U(f)(a, b) = ⋁_u f((a, u), (b, u)),

    the join over the feedback dimensions of the tensor's diagonal
    (where the U components of domain and codomain agree). It is the
    trace that compact closure induces, and it satisfies the trace
    axioms when the algebra is a `Semiring`.

    Parameters
    ----------
    algebra : Semiring
        The enrichment algebra.

    Raises
    ------
    TypeError
        If ``algebra`` is not a `Semiring`.
    """

    def __init__(self, algebra: Semiring) -> None:
        q = _require_semiring(algebra)
        super().__init__(CartesianProductMonoidal(q), q)

    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> Morphism:
        """Compute the cartesian trace via diagonal extraction and join.

        Parameters
        ----------
        morph : Morphism
            f: A × U → B × U.
        feedback : SetObject
            The feedback object U.
        domain : SetObject
            The external domain A.
        codomain : SetObject
            The external codomain B.

        Returns
        -------
        ObservedMorphism
            Tr^U(f): A → B.
        """
        q = self._algebra
        t = morph.tensor

        # tensor shape: (*a.shape, *u.shape, *b.shape, *u.shape)
        # we need to extract the diagonal over the two U copies
        # and join over U

        # the U dimensions in domain are at positions [n_a, n_a + n_u)
        # the U dimensions in codomain are at positions [n_a + n_u + n_b, n_a + n_u + n_b + n_u)
        result_shape = (*domain.shape, *codomain.shape)
        result = torch.full(result_shape, q.zero)

        # extract diagonal and join over U
        for a_idx in itertools.product(*(range(s) for s in domain.shape)):
            for b_idx in itertools.product(*(range(s) for s in codomain.shape)):
                # collect f(a, u, b, u) for all u
                vals: list[torch.Tensor] = []

                for u_idx in itertools.product(*(range(s) for s in feedback.shape)):
                    src_idx = a_idx + u_idx + b_idx + u_idx
                    vals.append(t[src_idx].unsqueeze(0))

                if vals:
                    stacked = torch.cat(vals)
                    result[a_idx + b_idx] = q.join(stacked, dim=0)

        return observed(domain, codomain, result, algebra=q)


def trace(
    morph: Morphism,
    feedback: SetObject,
    domain: SetObject,
    codomain: SetObject,
    algebra: Algebra | None = None,
) -> Morphism:
    """Apply the canonical trace `CompactClosedTrace`.

    Given f: A × U → B × U, compute Tr^U(f): A → B
    using the CompactClosedTrace.

    Parameters
    ----------
    morph : Morphism
        The morphism f: A × U → B × U.
    feedback : SetObject
        The feedback object U.
    domain : SetObject
        The external domain A.
    codomain : SetObject
        The external codomain B.
    algebra : Algebra or None
        The enrichment algebra, a `Semiring`; the morphism's own when
        omitted.

    Returns
    -------
    Morphism
        The traced morphism Tr^U(f): A → B.

    Raises
    ------
    TypeError
        If the algebra is not a `Semiring`.
    """
    tracer = CompactClosedTrace(
        _require_semiring(algebra if algebra is not None else morph.algebra)
    )
    return tracer.trace(morph, feedback, domain, codomain)


def partial_trace(
    morph: Morphism,
    feedback_indices: tuple[int, ...],
    algebra: Algebra | None = None,
) -> Morphism:
    """Trace over a subset of feedback wires in a product.

    Given f: (A₁ × ... × Aₙ) → (B₁ × ... × Bₙ) where some
    pairs (Aᵢ, Bᵢ) are feedback wires (Aᵢ = Bᵢ), compute the
    partial trace over the specified pairs.

    Parameters
    ----------
    morph : Morphism
        The morphism.
    feedback_indices : tuple[int, ...]
        Indices of the product components to trace over.
        These components must have matching shapes in domain
        and codomain.
    algebra : Algebra or None
        The enrichment algebra, a `Semiring`; the morphism's own when
        omitted.

    Returns
    -------
    ObservedMorphism
        The partially traced morphism.

    Raises
    ------
    TypeError
        If the algebra is not a `Semiring`, or the domain or codomain
        is not a product.
    ValueError
        If the domain and codomain differ in their number of components
        or in a traced component.
    """
    q = _require_semiring(algebra if algebra is not None else morph.algebra)
    t = morph.tensor
    dom = morph.domain
    cod = morph.codomain

    if not isinstance(dom, ProductSet) or not isinstance(cod, ProductSet):
        raise TypeError("partial_trace requires ProductSet domain and codomain")

    if len(dom.components) != len(cod.components):
        raise ValueError("domain and codomain must have the same number of components")

    # validate feedback components match
    for idx in feedback_indices:
        if dom.components[idx] != cod.components[idx]:
            raise ValueError(
                f"component {idx}: domain component {dom.components[idx]!r} "
                f"!= codomain component {cod.components[idx]!r}"
            )

    # build feedback and external objects
    external_dom_comps = [
        c for i, c in enumerate(dom.components) if i not in feedback_indices
    ]
    external_cod_comps = [
        c for i, c in enumerate(cod.components) if i not in feedback_indices
    ]
    [dom.components[i] for i in feedback_indices]

    # compute dimension indices
    n_dom = dom.ndim
    dom_offsets: list[int] = []
    offset = 0

    for c in dom.components:
        dom_offsets.append(offset)
        offset += c.ndim

    cod_offsets: list[int] = []
    offset = n_dom

    for c in cod.components:
        cod_offsets.append(offset)
        offset += c.ndim

    # identify feedback dimension pairs
    contra_dims: list[int] = []
    co_dims: list[int] = []

    for idx in feedback_indices:
        comp = dom.components[idx]

        for d in range(comp.ndim):
            contra_dims.append(dom_offsets[idx] + d)
            co_dims.append(cod_offsets[idx] + d)

    # use diagonal extraction and join
    from quivers.enriched.ends_coends import coend

    result = coend(
        t,
        contra_dims=tuple(contra_dims),
        co_dims=tuple(co_dims),
        algebra=q,
    )

    # build result objects
    if len(external_dom_comps) == 0:
        raise ValueError("cannot trace over all components")

    elif len(external_dom_comps) == 1:
        ext_dom = external_dom_comps[0]

    else:
        ext_dom = ProductSet(components=tuple(external_dom_comps))

    if len(external_cod_comps) == 1:
        ext_cod = external_cod_comps[0]

    else:
        ext_cod = ProductSet(components=tuple(external_cod_comps))

    return observed(ext_dom, ext_cod, result, algebra=q)


__all__ = [
    "TracedMonoidal",
    "CompactClosedTrace",
    "trace",
    "partial_trace",
]
