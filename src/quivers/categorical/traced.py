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

    KleeneStarTrace       — partial coproduct trace from finite path closure

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

from quivers.core.objects import SetObject, ProductSet, CoproductSet
from quivers.core.morphisms import Morphism, ObservedMorphism, observed, identity
from quivers.core.algebras import (
    Algebra,
    CompositionRule,
    Semiring,
    BooleanAlgebra,
    CountingAlgebra,
    MarkovAlgebra,
    LogProbAlgebra,
    MaxPlusAlgebra,
)
from quivers.core._carriers import validate_carrier
from quivers.core.tensor_ops import diagonal_reduce
from quivers.categorical.monoidal import (
    MonoidalStructure,
    CartesianProductMonoidal,
    CoproductMonoidal,
)
from quivers.stochastic._kleene import DivergentClosureError, kleene_star
from quivers.stochastic.semiring import (
    ChartSemiring,
    BooleanSemiring,
    CountingSemiring,
    LogProbSemiring,
    ViterbiSemiring,
)


_STAR_SEMIRINGS: dict[type[Semiring], ChartSemiring] = {
    BooleanAlgebra: BooleanSemiring(),
    CountingAlgebra: CountingSemiring(),
    MarkovAlgebra: CountingSemiring(),
    LogProbAlgebra: LogProbSemiring(),
    MaxPlusAlgebra: ViterbiSemiring(),
}


class KleeneStarTrace:
    """Partial trace of matrices under coproduct, defined by path closure.

    For a matrix ``f: A + U -> B + U`` in row-to-column convention,
    the result is ``f_AB + f_AU (f_UU)* f_UB``, with addition and
    multiplication taken in the declared semiring. The star includes
    paths of every finite length, including the empty path.

    Boolean closure is always finite. Counting closure is finite on
    acyclic feedback graphs. Markov and log-probability closure require
    spectral radius below one in the nonnegative representation.
    Max-plus closure excludes positive cycles. Divergent feedback is
    undefined even when no external path reaches that component.

    Nonnegative path sums can be regrouped without cancellation. This
    gives the partial trace axioms, including the definedness condition
    for nested feedback. Signed real matrices are excluded: convergence
    of an individual feedback block does not establish those axioms on
    the signed matrix category. This partial operator is independent of
    the total cartesian `TracedMonoidal` interface.

    Parameters
    ----------
    algebra : Semiring
        Exactly a built-in Boolean, Counting, Markov, LogProb, or MaxPlus
        algebra. Custom subclasses are not accepted.

    Raises
    ------
    TypeError
        If the source algebra is not one of the declared semirings.

    References
    ----------
    Malherbe, O., Scott, P. J., and Selinger, P. (2012). Partially
    traced categories. Journal of Pure and Applied Algebra, 216(12),
    2563-2585. DOI: 10.1016/j.jpaa.2012.03.026. Definition 3.2 gives
    the partial trace axioms; Proposition 3.10 gives the obstruction
    for a convergence-based star on signed real matrices.
    """

    def __init__(self, algebra: Semiring) -> None:
        if type(algebra) not in _STAR_SEMIRINGS:
            raise TypeError(
                f"no coproduct Kleene-star trace is declared for {algebra!r}"
            )
        self._algebra = algebra
        self._semiring = _STAR_SEMIRINGS[type(algebra)]
        self._monoidal = CoproductMonoidal(algebra)

    @property
    def algebra(self) -> Semiring:
        """The semiring whose finite path weights are combined."""
        return self._algebra

    @property
    def monoidal(self) -> CoproductMonoidal:
        """The coproduct monoidal structure used by this partial trace."""
        return self._monoidal

    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> ObservedMorphism:
        """Sum all paths through the feedback block when its closure is finite.

        Parameters
        ----------
        morph : Morphism
            Matrix ``A + U -> B + U`` over this trace's algebra, with a
            float32 or float64 tensor. Values must lie in its carrier.
        feedback : SetObject
            Feedback object ``U``, including the empty object.
        domain : SetObject
            External domain ``A``; it may have several tensor axes.
        codomain : SetObject
            External codomain ``B``; it may have several tensor axes.

        Returns
        -------
        ObservedMorphism
            Matrix ``A -> B`` with the external tensor shapes restored.
            Input precision is retained. Markov and log-probability
            evaluations retain input gradients.

        Raises
        ------
        DivergentClosureError
            If the feedback closure has no finite value.
        TypeError
            If the input tensor is not float32 or float64.
        ValueError
            If objects, algebra, tensor shape, or carrier values disagree,
            or a finite mathematical result exceeds the numeric carrier.
        """
        q = self._algebra
        expected_domain = CoproductSet(components=(domain, feedback))
        expected_codomain = CoproductSet(components=(codomain, feedback))
        if morph.domain != expected_domain or morph.codomain != expected_codomain:
            raise ValueError(
                "Kleene-star trace requires the declared A + U -> B + U objects"
            )
        if type(morph.algebra) is not type(q):
            raise ValueError(
                "the morphism and Kleene-star trace must use the same algebra"
            )
        matrix = morph.tensor
        expected_shape = (domain.size + feedback.size, codomain.size + feedback.size)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"expected coproduct matrix shape {expected_shape}, got {tuple(matrix.shape)}"
            )
        if matrix.dtype not in (torch.float32, torch.float64):
            raise TypeError("Kleene-star trace requires a float32 or float64 tensor")
        validate_carrier(matrix, q)
        direct = matrix[: domain.size, : codomain.size]
        if feedback.size == 0:
            result = direct
        else:
            entrance = matrix[: domain.size, codomain.size :]
            exit_weights = matrix[domain.size :, : codomain.size]
            loop = matrix[domain.size :, codomain.size :]
            closure = kleene_star(loop, self._semiring)
            through = q.compose(q.compose(entrance, closure, 1), exit_weights, 1)
            result = q.join(torch.stack((direct, through)), dim=0)
        validate_carrier(result, q)
        return observed(
            domain, codomain, result.reshape(*domain.shape, *codomain.shape), algebra=q
        )

    def verify_yanking(self, feedback: SetObject, atol: float = 1e-6) -> bool:
        """Check that tracing the coproduct swap returns the identity.

        Parameters
        ----------
        feedback : SetObject
            Object whose swap is traced.
        atol : float
            Absolute comparison tolerance, with zero relative tolerance.

        Returns
        -------
        bool
            Whether the yanking equation holds.
        """
        swap = self._monoidal.braiding(feedback, feedback)
        actual = self.trace(swap, feedback, feedback, feedback).tensor
        expected = identity(feedback, algebra=self._algebra).tensor
        return torch.allclose(actual, expected, atol=atol, rtol=0)

    def __repr__(self) -> str:
        """Identify the semiring used for coproduct feedback.

        Returns
        -------
        str
            Constructor name and declared algebra.
        """
        return f"KleeneStarTrace({self._algebra!r})"


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
        swap = self._monoidal.braiding(feedback, feedback)
        traced = self.trace(swap, feedback, feedback, feedback)
        expected = identity(feedback, algebra=self._algebra).tensor
        return torch.allclose(traced.tensor, expected, atol=atol)

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

    result = diagonal_reduce(t, tuple(contra_dims), tuple(co_dims), q.join)

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
    "KleeneStarTrace",
    "DivergentClosureError",
    "TracedMonoidal",
    "CompactClosedTrace",
    "trace",
    "partial_trace",
]
