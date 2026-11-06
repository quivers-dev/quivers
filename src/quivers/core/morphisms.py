"""Morphism hierarchy: V-enriched relations between finite sets.

A morphism from domain D to codomain C is represented as a tensor of
shape (*D.shape, *C.shape) with entries in the carrier of an algebra.
Composition uses the algebra's operations (join over tensor product).

The hierarchy:

    Morphism (abstract)
    ├── ObservedMorphism    — fixed tensor (not learned)
    ├── LatentMorphism      — nn.Parameter with sigmoid constraint
    ├── ComposedMorphism    — f >> g (V-enriched composition)
    ├── ProductMorphism     — f @ g (tensor / parallel product)
    ├── MarginalizedMorphism — contract codomain dims via join
    ├── FunctorMorphism     — lazy image of a morphism under a functor
    └── RepeatMorphism      — runtime-variable iterated composition (T^n)

PyTorch boundary
================

The categorical hierarchy is intentionally *not* a subclass of
`torch.nn.Module`: a Morphism is a categorical object and a
Module is a PyTorch parameter container. When a Morphism needs to
be bound into a parameter-tracking context (a `MonadicProgram`,
a `FanMorphism`, a parametric-program parameter slot), the
adapter `as_torch_module` produces a backend-agnostic
`nn.Module` wrapping the morphism's parameters. Every
binding site funnels through this adapter so the categorical /
PyTorch boundary stays explicit and a JAX or numpy backend can
replace ``as_torch_module`` without touching the morphism
hierarchy itself.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import cast
import torch
import torch.nn as nn
from quivers.core.morphism_transformations import MorphismTransformation
from quivers.core.objects import FinSet, FreeResiduated, SetObject, ProductSet
from quivers.core.algebra_morphisms import AlgebraHomomorphism
from quivers.core.algebras import PRODUCT_FUZZY, Algebra
from quivers.core.trans import TransSeq


def _build_algebra_homomorphism_transform(phi):
    """Closure factory for the pointwise change-of-base transform.

    Splitting the closure construction into a free function avoids
    ``reportRedeclaration``
    warnings from the type checker when both arms of the ``isinstance``
    in ``change_base`` define a local with the same name.
    """

    def _transform(t: torch.Tensor) -> torch.Tensor:
        return phi.apply(t)

    return _transform


def _build_morphism_transformation_transform(phi, source):
    """Closure factory for the shape-aware change-of-base transform."""

    def _transform(t: torch.Tensor) -> torch.Tensor:
        return phi.apply(t, source)

    return _transform


class Morphism(ABC):
    """Abstract base for morphisms between finite sets.

    Subclasses must implement ``tensor`` (returns the materialized
    tensor with values in the algebra's lattice) and ``module``
    (returns the nn.Module tree for parameter collection).

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self, domain: SetObject, codomain: SetObject, algebra: Algebra | None = None
    ) -> None:
        self._domain = domain
        self._codomain = codomain
        self._algebra = algebra if algebra is not None else PRODUCT_FUZZY

    @property
    def domain(self) -> SetObject:
        """Source object."""
        return self._domain

    @property
    def codomain(self) -> SetObject:
        """Target object."""
        return self._codomain

    @property
    def algebra(self) -> Algebra:
        """The enrichment algebra for this morphism."""
        return self._algebra

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Expected shape of the materialized tensor."""
        return (*self._domain.shape, *self._codomain.shape)

    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Materialize the morphism as a tensor in the algebra's carrier."""
        ...

    @abstractmethod
    def module(self) -> nn.Module:
        """Return an nn.Module wrapping all learnable parameters."""
        ...

    def __rshift__(self, other: Morphism) -> ComposedMorphism:
        """V-enriched composition: self >> other.

        Composes self: A -> B with other: B -> C to yield
        a morphism A -> C, contracting over B using the algebra.

        Parameters
        ----------
        other : Morphism
            Right morphism whose domain must match self's codomain.

        Returns
        -------
        ComposedMorphism
            The composed morphism.
        """
        if not isinstance(other, Morphism):
            return NotImplemented
        if self.codomain != other.domain:
            raise TypeError(
                f"cannot compose: codomain {self.codomain!r} != domain {other.domain!r}"
            )
        if not self._algebra.is_compatible(other._algebra):
            raise TypeError(
                f"incompatible algebras: {self._algebra!r} and {other._algebra!r}"
            )
        return ComposedMorphism(self, other)

    def __matmul__(self, other: Morphism) -> ProductMorphism:
        """Tensor (parallel) product: self @ other.

        Given self: A -> B and other: C -> D, produces
        a morphism A×C -> B×D whose tensor is the outer product
        via the algebra's tensor_op.

        Parameters
        ----------
        other : Morphism
            Right morphism.

        Returns
        -------
        ProductMorphism
            The product morphism.
        """
        if not isinstance(other, Morphism):
            return NotImplemented
        return ProductMorphism(self, other)

    def change_base(self, phi) -> "TransformedMorphism":
        """Transport this morphism along a change-of-base functor.

        ``phi`` may be either a
        `quivers.core.algebra_morphisms.AlgebraHomomorphism`
        (pointwise: the action factors entry-by-entry through
        ``phi.apply``) or a
        `quivers.core.morphism_transformations.MorphismTransformation`
        (shape-aware: the action consumes the whole tensor plus
        the morphism for axis resolution).

        The result is a `TransformedMorphism` whose
        `tensor` is recomputed on each access by applying
        ``phi`` to the source's current tensor. The source's
        learnable parameters are exposed through ``.module()`` so
        `torch.nn.Module.parameters` walks reach them, and
        each backward through a fresh ``.tensor`` access rebuilds
        a graph rooted at those parameters; multi-step optimisation
        through change-of-base is supported.

        Parameters
        ----------
        phi : AlgebraHomomorphism or MorphismTransformation
            The change-of-base functor. Its ``source`` must match
            ``self.algebra``.

        Returns
        -------
        TransformedMorphism
            A morphism over ``phi.target`` whose tensor is the
            transported tensor of ``self``. For pointwise
            transformations the domain and codomain are preserved;
            for shape-aware ones (e.g. ``BayesInvert``) the
            transformation may swap them.
        """
        if isinstance(phi, TransSeq):
            # Apply the steps in order; each step's change_base
            # type-checks its own source against the current
            # algebra, so a malformed seam surfaces with the
            # same error a hand-written sequence would produce.
            current: Morphism = self
            for step in phi.steps:
                current = current.change_base(step)
            return cast(TransformedMorphism, current)
        if not isinstance(phi, (AlgebraHomomorphism, MorphismTransformation)):
            raise TypeError(
                f"change_base: expected AlgebraHomomorphism, "
                f"MorphismTransformation, or TransSeq; got "
                f"{type(phi).__name__}"
            )
        if type(phi.source) is not type(self._algebra):
            raise TypeError(
                f"change_base: source algebra "
                f"{phi.source.name!r} does not match this morphism's "
                f"algebra {self._algebra.name!r}"
            )
        if isinstance(phi, AlgebraHomomorphism):
            transform = _build_algebra_homomorphism_transform(phi)
            new_domain = self._domain
            new_codomain = self._codomain
        else:
            transform = _build_morphism_transformation_transform(phi, self)
            new_domain = phi.new_domain(self)
            new_codomain = phi.new_codomain(self)
        return TransformedMorphism(
            self,
            transform,
            new_domain,
            new_codomain,
            algebra=phi.target,
        )

    def refactor(self, *, domain=None, codomain=None) -> "TransformedMorphism":
        """Switch between flat and product views of this morphism.

        Given a morphism ``f : A -> B`` whose tensor storage has
        total numel matching the requested ``domain`` / ``codomain``
        objects, return an equivalent morphism ``f' : A' -> B'``
        whose tensor is the same data reshaped to the new factored
        layout. ``A'`` and ``B'`` must be isomorphic to ``A`` and
        ``B`` as objects — same cardinality, possibly different
        product structure.

        Categorically this is the action of an object iso
        ``B ≅ B'`` on the morphism's tensor. Semantically it is
        a no-op; presentation only.

        Parameters
        ----------
        domain : SetObject, optional
            New domain. Must have ``prod(shape) == prod(self.domain.shape)``.
        codomain : SetObject, optional
            New codomain. Same numel constraint.

        Returns
        -------
        ObservedMorphism
            The reshape of ``self`` into the requested type.
        """
        new_domain = domain if domain is not None else self._domain
        new_codomain = codomain if codomain is not None else self._codomain

        def _numel(shape) -> int:
            n = 1
            for s in shape:
                n *= int(s)
            return n

        if _numel(new_domain.shape) != _numel(self._domain.shape):
            raise ValueError(
                f"refactor: domain numel {_numel(new_domain.shape)} "
                f"does not match {_numel(self._domain.shape)} for "
                f"{self._domain!r} -> {new_domain!r}"
            )
        if _numel(new_codomain.shape) != _numel(self._codomain.shape):
            raise ValueError(
                f"refactor: codomain numel "
                f"{_numel(new_codomain.shape)} does not match "
                f"{_numel(self._codomain.shape)} for "
                f"{self._codomain!r} -> {new_codomain!r}"
            )
        target_shape = tuple(new_domain.shape) + tuple(new_codomain.shape)

        def _transform(t: torch.Tensor, _shape=target_shape) -> torch.Tensor:
            return t.reshape(_shape)

        return TransformedMorphism(
            self,
            _transform,
            new_domain,
            new_codomain,
            algebra=self._algebra,
        )

    def marginalize(self, *sets: SetObject) -> MarginalizedMorphism:
        """Marginalize (join-reduce) over codomain components.

        The codomain must be a ProductSet containing the sets to
        marginalize over. The result has a codomain with those
        components removed. Uses the algebra's join operation.

        Parameters
        ----------
        *sets : SetObject
            Codomain components to marginalize over.

        Returns
        -------
        MarginalizedMorphism
            The marginalized morphism.
        """
        return MarginalizedMorphism(self, sets)

    @property
    def dagger(self) -> "TransformedMorphism":
        """Transpose / dagger of ``self : A → B``, producing a
        morphism ``B → A`` whose tensor has the domain and codomain
        axes swapped.

        The compact-closed structure of V-Cat means every object is
        self-dual (``A^* = A``), so the dagger is well-defined for
        every algebra. The semantic interpretation depends on the
        algebra: ProductFuzzyAlgebra gives the tensor transpose, Markov
        the Bayes-uniform-prior inversion, Viterbi the max-plus
        reversal, Boolean the relational converse.

        Categorically:
        ``f^† = (ε_B ⊗ id_A) ∘ (id_B ⊗ f^* ⊗ id_A) ∘ (id_B ⊗ η_A)``
        — but for finite-set objects the unit / counit collapse to
        the diagonal / co-diagonal, and the dagger reduces to a
        tensor transpose along the domain/codomain axes.

        Returns
        -------
        ObservedMorphism
            Morphism ``B → A`` whose tensor is the axis-swapped
            tensor of ``self``. Gradients propagate to the original
            morphism's parameters through the transpose.
        """
        d_ndim = len(self._domain.shape)
        c_ndim = len(self._codomain.shape)
        # Build the permutation that brings the codomain axes to
        # the front and the domain axes to the back.
        perm = tuple(range(d_ndim, d_ndim + c_ndim)) + tuple(range(d_ndim))

        def _transform(t: torch.Tensor, _perm=perm) -> torch.Tensor:
            return t.permute(_perm)

        return TransformedMorphism(
            self,
            _transform,
            self._codomain,
            self._domain,
            algebra=self._algebra,
        )

    def trace(self, obj: SetObject) -> "TransformedMorphism":
        """Trace of ``self : X ⊗ A → A ⊗ Y`` along ``obj = A``,
        producing a morphism ``X → Y``.

        Concretely the trace contracts the A axis on the domain
        side with the A axis on the codomain side via the algebra's
        ``tensor_op`` and then joins (``self._algebra.join``) over
        the contracted axis. This is the categorical trace
        ``tr_A(f) : X → Y = (ε_A ⊗ id_Y) ∘ (id_A ⊗ f) ∘ (η_A ⊗ id_X)``.

        The morphism's domain must be a product set whose first
        component is ``obj``; the codomain must be a product set
        whose first component is ``obj``. If this is not the case
        a TypeError is raised — the user should call
        `ProductSet.swap` style helpers (not yet exposed) to
        reorder axes before tracing.

        Parameters
        ----------
        obj : SetObject
            The object to contract over. Must appear at the start
            of both the domain and codomain product sets.

        Returns
        -------
        ObservedMorphism
            Morphism ``X → Y`` (the trace).
        """
        if not isinstance(self._domain, ProductSet) or not isinstance(
            self._codomain, ProductSet
        ):
            raise TypeError(
                "trace: requires the morphism's domain and codomain "
                "to both be ProductSets with the contracted object "
                f"at the front; got domain={self._domain!r}, "
                f"codomain={self._codomain!r}"
            )
        if self._domain.components[0] != obj:
            raise TypeError(
                f"trace: domain's first component {self._domain.components[0]!r} "
                f"!= contraction object {obj!r}"
            )
        if self._codomain.components[0] != obj:
            raise TypeError(
                f"trace: codomain's first component {self._codomain.components[0]!r} "
                f"!= contraction object {obj!r}"
            )
        d_ndim = len(self._domain.shape)
        a_ndim = len(obj.shape)
        algebra = self._algebra

        def _transform(
            t: torch.Tensor, _d_ndim=d_ndim, _a_ndim=a_ndim, _q=algebra
        ) -> torch.Tensor:
            # Domain axes: 0..d_ndim-1 with A at 0..a_ndim-1.
            # Codomain axes: d_ndim..d_ndim+c_ndim-1 with A at
            # d_ndim..d_ndim+a_ndim-1.  Take diagonals pairing each
            # A axis on the domain side with the corresponding one
            # on the codomain side, then join along the surviving
            # diagonal axes.
            out = t
            for k in range(_a_ndim):
                out = torch.diagonal(out, dim1=0, dim2=_d_ndim - k)
            trailing = tuple(range(out.dim() - _a_ndim, out.dim()))
            return _q.join(out, dim=trailing)

        # Recover the X and Y product sets.
        x_components = tuple(self._domain.components[1:])
        y_components = tuple(self._codomain.components[1:])
        if len(x_components) == 1:
            new_domain = x_components[0]
        else:
            new_domain = ProductSet(components=x_components)
        if len(y_components) == 1:
            new_codomain = y_components[0]
        else:
            new_codomain = ProductSet(components=y_components)
        return TransformedMorphism(
            self, _transform, new_domain, new_codomain, algebra=self._algebra
        )

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.domain!r} -> {self.codomain!r})"


class _MorphismModule(nn.Module):
    """Internal nn.Module wrapper for a single morphism's parameters."""

    pass


class _SourcedModule(nn.Module):
    """Module wrapper holding a reference to a source morphism's
    nn.Module so that `torch.nn.Module.parameters` walks reach
    the upstream learnable parameters.
    """

    def __init__(self, source_module: nn.Module) -> None:
        super().__init__()
        self.source = source_module


class TransformedMorphism(Morphism):
    """A morphism whose tensor is a transformation of a source
    morphism's tensor, recomputed on each `tensor` access.

    The source morphism is registered as a submodule so
    ``self.module().parameters()`` includes the source's learnable
    parameters; the transform is applied lazily, so each backward
    through a fresh ``.tensor`` access gets its own autograd graph
    and multi-step optimisation propagates gradients correctly.

    Parameters
    ----------
    source : Morphism
        Underlying morphism whose tensor feeds the transform.
    transform : Callable[[torch.Tensor], torch.Tensor]
        Pure function applied to ``source.tensor`` to produce the
        new tensor. Must depend on ``source`` only through the
        tensor; any additional context (e.g. the source morphism
        for shape resolution) must be captured by closure.
    domain : SetObject
        Target domain (may differ from ``source.domain`` for
        shape-aware transformations like the dagger or BayesInvert).
    codomain : SetObject
        Target codomain.
    algebra : Algebra or None
        Target algebra. Defaults to the source's algebra.
    """

    def __init__(
        self,
        source: "Morphism",
        transform,  # Callable[[torch.Tensor], torch.Tensor]
        domain: SetObject,
        codomain: SetObject,
        algebra: Algebra | None = None,
    ) -> None:
        super().__init__(
            domain,
            codomain,
            algebra=algebra if algebra is not None else source.algebra,
        )
        self._source = source
        self._transform = transform
        self._module = _SourcedModule(source.module())

    @property
    def source(self) -> "Morphism":
        """The underlying morphism feeding this transform."""
        return self._source

    @property
    def tensor(self) -> torch.Tensor:
        return self._transform(self._source.tensor)

    def module(self) -> nn.Module:
        return self._module


class ObservedMorphism(Morphism):
    """A morphism with a fixed (non-learnable) tensor.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    data : torch.Tensor
        Fixed tensor of shape (*domain.shape, *codomain.shape).
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: SetObject,
        data: torch.Tensor,
        algebra: Algebra | None = None,
    ) -> None:
        super().__init__(domain, codomain, algebra=algebra)
        expected = self.tensor_shape
        if data.shape != expected:
            raise ValueError(
                f"data shape {data.shape} does not match expected {expected}"
            )
        self._module = _MorphismModule()
        self._module.register_buffer("data", data)

    @property
    def tensor(self) -> torch.Tensor:
        return cast(torch.Tensor, self._module.data)

    def module(self) -> nn.Module:
        return self._module


class LatentMorphism(Morphism):
    """A learnable morphism backed by an nn.Parameter.

    Stores unconstrained real-valued parameters and applies sigmoid
    to produce [0,1]-valued fuzzy relation entries.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    init_scale : float
        Standard deviation of the normal initialization for the
        unconstrained parameters. Default 0.5 (sigmoid maps this
        to roughly uniform over [0.3, 0.7]).
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        domain: SetObject,
        codomain: SetObject,
        init_scale: float = 0.5,
        algebra: Algebra | None = None,
    ) -> None:
        super().__init__(domain, codomain, algebra=algebra)
        shape = self.tensor_shape
        self._module = _MorphismModule()
        raw = nn.Parameter(torch.randn(shape) * init_scale)
        self._module.register_parameter("raw", raw)

    @property
    def raw(self) -> nn.Parameter:
        """Unconstrained parameter tensor."""
        return self._module.raw

    @property
    def tensor(self) -> torch.Tensor:
        """Sigmoid-constrained tensor with values in (0, 1)."""
        return torch.sigmoid(cast(torch.Tensor, self._module.raw))

    def module(self) -> nn.Module:
        return self._module


class _ComposedModule(nn.Module):
    """Module for composed morphisms that owns sub-modules."""

    def __init__(self, left_mod: nn.Module, right_mod: nn.Module) -> None:
        super().__init__()
        self.left = left_mod
        self.right = right_mod


class ComposedMorphism(Morphism):
    """V-enriched composition of two morphisms.

    Given left: A -> B and right: B -> C, produces A -> C
    by contracting over B using the algebra's compose method.

    Parameters
    ----------
    left : Morphism
        Left morphism (applied first).
    right : Morphism
        Right morphism (applied second).
    """

    def __init__(self, left: Morphism, right: Morphism) -> None:
        n_contract = left.codomain.ndim
        super().__init__(left.domain, right.codomain, algebra=left._algebra)
        self._left = left
        self._right = right
        self._n_contract = n_contract

    @property
    def left(self) -> Morphism:
        """Left (first) morphism."""
        return self._left

    @property
    def right(self) -> Morphism:
        """Right (second) morphism."""
        return self._right

    @property
    def tensor(self) -> torch.Tensor:
        return self._algebra.compose(
            self._left.tensor, self._right.tensor, self._n_contract
        )

    def module(self) -> nn.Module:
        return _ComposedModule(self._left.module(), self._right.module())


class _ProductModule(nn.Module):
    """Module for product morphisms."""

    def __init__(self, left_mod: nn.Module, right_mod: nn.Module) -> None:
        super().__init__()
        self.left = left_mod
        self.right = right_mod


class ProductMorphism(Morphism):
    """Tensor (parallel) product of two morphisms.

    Given left: A -> B and right: C -> D, produces
    A×C -> B×D. The tensor is the outer product of the
    two component tensors via the algebra's tensor_op.

    Parameters
    ----------
    left : Morphism
        Left morphism.
    right : Morphism
        Right morphism.
    """

    def __init__(self, left: Morphism, right: Morphism) -> None:
        domain = ProductSet(components=(left.domain, right.domain))
        codomain = ProductSet(components=(left.codomain, right.codomain))
        super().__init__(domain, codomain, algebra=left._algebra)
        self._left = left
        self._right = right

    @property
    def tensor(self) -> torch.Tensor:
        lt = self._left.tensor
        rt = self._right.tensor
        n_l = lt.ndim
        n_r = rt.ndim
        lt_expanded = lt.reshape(*lt.shape, *[1] * n_r)
        rt_expanded = rt.reshape(*[1] * n_l, *rt.shape)
        outer = self._algebra.tensor_op(lt_expanded, rt_expanded)
        n_dom_l = self._left.domain.ndim
        n_cod_l = self._left.codomain.ndim
        n_dom_r = self._right.domain.ndim
        dom_l_dims = list(range(n_dom_l))
        cod_l_dims = list(range(n_dom_l, n_dom_l + n_cod_l))
        dom_r_dims = list(range(n_dom_l + n_cod_l, n_dom_l + n_cod_l + n_dom_r))
        cod_r_dims = list(range(n_dom_l + n_cod_l + n_dom_r, n_l + n_r))
        perm = dom_l_dims + dom_r_dims + cod_l_dims + cod_r_dims
        return outer.permute(*perm)

    def module(self) -> nn.Module:
        return _ProductModule(self._left.module(), self._right.module())


class _MarginalizedModule(nn.Module):
    """Module for marginalized morphisms."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class MarginalizedMorphism(Morphism):
    """Morphism with codomain dimensions marginalized via the algebra's join.

    Given an inner morphism A -> B × C and a set B to marginalize,
    produces A -> C by join-reduction over B's dimensions.

    Parameters
    ----------
    inner : Morphism
        The morphism to marginalize.
    sets_to_marginalize : tuple of SetObject
        Codomain components to marginalize over.
    """

    def __init__(
        self,
        inner: Morphism,
        sets_to_marginalize: tuple[SetObject, ...] | list[SetObject],
    ) -> None:
        codomain = inner.codomain
        sets_to_marginalize = tuple(sets_to_marginalize)
        if not isinstance(codomain, ProductSet):
            raise TypeError(
                f"can only marginalize over ProductSet codomain, got {type(codomain).__name__}"
            )
        n_domain = inner.domain.ndim
        remaining_components: list[SetObject] = []
        dims_to_reduce: list[int] = []
        offset = n_domain
        for component in codomain.components:
            if component in sets_to_marginalize:
                for d in range(component.ndim):
                    dims_to_reduce.append(offset + d)
            else:
                remaining_components.append(component)
            offset += component.ndim
        if not dims_to_reduce:
            raise ValueError("none of the specified sets found in codomain components")
        if len(remaining_components) == 0:
            raise ValueError("cannot marginalize all codomain components")
        elif len(remaining_components) == 1:
            new_codomain = remaining_components[0]
        else:
            new_codomain = ProductSet(components=tuple(remaining_components))
        super().__init__(inner.domain, new_codomain, algebra=inner._algebra)
        self._inner = inner
        self._dims_to_reduce = tuple(dims_to_reduce)

    @property
    def tensor(self) -> torch.Tensor:
        return self._algebra.join(self._inner.tensor, dim=self._dims_to_reduce)

    def module(self) -> nn.Module:
        return _MarginalizedModule(self._inner.module())


class FunctorMorphism(Morphism):
    """Lazy image of a morphism under a functor.

    Recomputes the tensor on each access from the inner morphism,
    preserving gradient flow through the functor's map_tensor method.
    No additional parameters beyond those of the inner morphism.

    Parameters
    ----------
    functor : Functor
        The functor that produced this morphism.
    inner : Morphism
        The original morphism being mapped.
    domain : SetObject
        The image of the inner morphism's domain under the functor.
    codomain : SetObject
        The image of the inner morphism's codomain under the functor.
    """

    def __init__(
        self, functor: Functor, inner: Morphism, domain: SetObject, codomain: SetObject
    ) -> None:
        super().__init__(domain, codomain, algebra=inner._algebra)
        self._functor = functor
        self._inner = inner

    @property
    def inner(self) -> Morphism:
        """The original morphism being mapped."""
        return self._inner

    @property
    def tensor(self) -> torch.Tensor:
        return self._functor.map_tensor(self._inner.tensor, self._algebra)

    def module(self) -> nn.Module:
        return self._inner.module()


class _RepeatModule(nn.Module):
    """Module wrapper for RepeatMorphism parameters."""

    def __init__(self, inner_mod: nn.Module) -> None:
        super().__init__()
        self.inner = inner_mod


class RepeatMorphism(Morphism):
    """Runtime-variable iterated composition (matrix power).

    Wraps an endomorphism f : X -> X and computes f^n at runtime,
    where n can be changed between calls. Uses repeated squaring
    for O(log n) algebra compositions.

    For an endomorphism T : S -> S under an algebra, ``T^n`` is the
    n-fold Kleisli composition. Under the product_fuzzy algebra
    with stochastic matrices, this is standard matrix power.

    Parameters
    ----------
    inner : Morphism
        An endomorphism (domain must equal codomain).
    n : int
        Initial number of repetitions (default 1). Can be changed
        later via the ``n_steps`` property.

    Raises
    ------
    TypeError
        If the inner morphism is not an endomorphism.
    ValueError
        If n < 1.

    Examples
    --------
    >>> T = morphism(S, S)
    >>> rep = RepeatMorphism(T, n=5)
    >>> rep.tensor  # computes T^5
    >>> rep.n_steps = 10
    >>> rep.tensor  # now computes T^10
    """

    def __init__(self, inner: Morphism, n: int = 1) -> None:
        if inner.domain != inner.codomain:
            raise TypeError(
                f"repeat requires an endomorphism, got {inner.domain!r} -> {inner.codomain!r}"
            )
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        super().__init__(inner.domain, inner.codomain, algebra=inner._algebra)
        self._inner = inner
        self._n = n
        self._n_contract = inner.codomain.ndim

    @property
    def inner(self) -> Morphism:
        """The base endomorphism."""
        return self._inner

    @property
    def n_steps(self) -> int:
        """Number of iterated compositions."""
        return self._n

    @n_steps.setter
    def n_steps(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"n_steps must be >= 1, got {value}")
        self._n = value

    @property
    def tensor(self) -> torch.Tensor:
        """Compute the n-fold composition via repeated squaring.

        Returns
        -------
        torch.Tensor
            The tensor for f^n, same shape as the inner morphism.
        """
        t = self._inner.tensor
        if self._n == 1:
            return t
        result = None
        base = t
        n = self._n
        while n > 0:
            if n % 2 == 1:
                if result is None:
                    result = base
                else:
                    result = self._algebra.compose(result, base, self._n_contract)
            base = self._algebra.compose(base, base, self._n_contract)
            n //= 2
        assert result is not None
        return result

    def module(self) -> nn.Module:
        return _RepeatModule(self._inner.module())

    def __repr__(self) -> str:
        return f"RepeatMorphism({self._inner!r}, n={self._n})"


class CurriedMorphism(Morphism):
    """Residuation-witness curried morphism.

    For an inner morphism ``f : X * Y -> Z`` whose codomain ``Z`` lives
    in a residuated universe (a `FreeResiduated`), produces the
    morphism corresponding to the relevant residuation isomorphism:

    - ``direction='right'`` realises the right-residuation
      ``X * Y -> Z  ≅  X -> Z/Y`` (counit of the right-residual
      adjunction),
    - ``direction='left'`` realises the left-residuation
      ``X * Y -> Z  ≅  Y -> X\\Z``.

    The underlying tensor data is reinterpreted, not recomputed: the
    same V-relation is presented under a different domain/codomain
    factoring in the residuated universe.

    Parameters
    ----------
    inner : Morphism
        The base morphism. Must have a domain that factors as a
        non-trivial product (``ProductSet`` with at least two
        components) and a codomain that inhabits a residuated universe.
    direction : Literal['right', 'left']
        Which residuation to apply.

    Raises
    ------
    TypeError
        If ``inner.domain`` does not factor as a product.
    """

    def __init__(self, inner: Morphism, direction: str = "right") -> None:
        from quivers.stochastic.categories import (
            AtomicCategory,
            SlashCategory,
        )

        if direction not in ("right", "left"):
            raise ValueError(f"direction must be 'right' or 'left', got {direction!r}")
        if not isinstance(inner.domain, ProductSet) or len(inner.domain.components) < 2:
            raise TypeError(
                f"curry requires inner morphism with product domain, "
                f"got {type(inner.domain).__name__}"
            )

        # Split off the first or last factor of the domain product
        # depending on direction; the residuation moves it into the
        # codomain via the slash constructor.
        components = inner.domain.components
        if direction == "right":
            new_domain_components = components[:-1]
            absorbed = components[-1]
        else:
            new_domain_components = components[1:]
            absorbed = components[0]

        if len(new_domain_components) == 1:
            new_domain = new_domain_components[0]
        else:
            new_domain = ProductSet(components=new_domain_components)

        # Codomain is the residuation of inner.codomain by `absorbed`.
        # When inner.codomain is a FreeResiduated universe, the new
        # codomain is the same universe (closed under residuation).
        # Otherwise, the construction is interpreted in the implicit
        # residuated structure on the existing codomain.
        if isinstance(inner.codomain, FreeResiduated):
            new_codomain: SetObject = inner.codomain
        else:
            slash = "/" if direction == "right" else "\\"
            cat = SlashCategory(
                result=AtomicCategory(name=str(inner.codomain)),
                argument=AtomicCategory(name=str(absorbed)),
                direction=slash,  # type: ignore[arg-type]
            )
            new_codomain = inner.codomain  # underlying type unchanged
            self._slash_category = cat

        super().__init__(new_domain, new_codomain, algebra=inner._algebra)
        self._inner = inner
        self._direction = direction

    @property
    def inner(self) -> Morphism:
        return self._inner

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def tensor(self) -> torch.Tensor:
        # The residuation isomorphism is the identity on tensor data:
        # the same V-relation, viewed under a different factoring.
        return self._inner.tensor

    def module(self) -> nn.Module:
        return self._inner.module()

    def __repr__(self) -> str:
        return f"CurriedMorphism({self._inner!r}, direction={self._direction!r})"


def morphism(
    domain: SetObject,
    codomain: SetObject,
    init_scale: float = 0.5,
    algebra: Algebra | None = None,
) -> LatentMorphism:
    """Create a latent (learnable) morphism.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    init_scale : float
        Initialization scale for unconstrained parameters.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    LatentMorphism
        A learnable morphism.
    """
    return LatentMorphism(domain, codomain, init_scale=init_scale, algebra=algebra)


def observed(
    domain: SetObject,
    codomain: SetObject,
    data: torch.Tensor,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Create an observed (fixed) morphism.

    Parameters
    ----------
    domain : SetObject
        Source object.
    codomain : SetObject
        Target object.
    data : torch.Tensor
        Fixed tensor.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    ObservedMorphism
        A fixed morphism.
    """
    return ObservedMorphism(domain, codomain, data, algebra=algebra)


def identity(obj: SetObject, algebra: Algebra | None = None) -> ObservedMorphism:
    """Create the identity morphism on an object.

    Returns an observed morphism obj -> obj whose tensor is the
    identity: unit on the diagonal, zero elsewhere.

    Parameters
    ----------
    obj : SetObject
        The object to create an identity for.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    ObservedMorphism
        The identity morphism.
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    data = q.identity_tensor(obj.shape)
    return ObservedMorphism(obj, obj, data, algebra=q)


def as_torch_module(m: object) -> nn.Module:
    """Coerce a Morphism into an `nn.Module` for parameter
    tracking at a binding site.

    The adapter draws the line between the categorical morphism
    hierarchy (backend-agnostic; lives in `quivers.core`) and
    PyTorch's parameter-container infrastructure. Every site that
    needs to register a morphism's parameters with a parent
    `nn.Module` (the `MonadicProgram` step list, the
    submodule list of a `FanMorphism` / `StackMorphism`
    composite, a parametric-program parameter slot bound to a
    morphism) calls this function once and stores the result.

    Parameters
    ----------
    m : object
        The morphism (or other object) to wrap. Accepted forms:

        * Already an `nn.Module` — returned unchanged so a
          continuous `MonadicProgram` step can pass its
          `ContinuousMorphism` straight through.
        * A `Morphism` with a ``.module()`` method (every
          subclass of `Morphism` defined in this file
          implements it) — the result of ``m.module()`` is
          returned. The morphism object itself is attached to the
          wrapper under the synthetic attribute ``_morphism`` so
          downstream code that needs the categorical object (e.g.
          to compute ``tensor`` or apply a ``Functor``) can recover
          it without rebuilding.

    Returns
    -------
    nn.Module
        A module whose parameters / buffers are exactly the
        morphism's, suitable for ``add_module`` on a parent.

    Raises
    ------
    TypeError
        ``m`` is neither an `nn.Module` nor a
        `Morphism`-shaped object with a ``.module()`` method.
    """
    if isinstance(m, nn.Module):
        return m
    if isinstance(m, Morphism):
        wrapper = m.module()
        if not isinstance(wrapper, nn.Module):
            raise TypeError(
                f"{type(m).__name__}.module() returned "
                f"{type(wrapper).__name__}; expected nn.Module"
            )
        # Attach the original morphism on the wrapper so downstream
        # code that needs the categorical object can recover it
        # without rebuilding from the wrapped parameters.
        wrapper._morphism = m  # type: ignore[attr-defined]
        return wrapper
    raise TypeError(
        f"as_torch_module: cannot adapt {type(m).__name__} to "
        f"nn.Module; expected an nn.Module or a Morphism with a "
        f".module() method"
    )


def extract_morphism(module: nn.Module) -> Morphism | None:
    """Recover the `Morphism` previously bound through
    `as_torch_module`.

    Returns the categorical morphism stored on the wrapper, or
    ``None`` if the module was registered directly (i.e. was
    already an `nn.Module` subclass) and thus has no
    separate categorical object attached.

    Parameters
    ----------
    module : nn.Module
        A module bound into a parameter-tracking context.

    Returns
    -------
    Morphism or None
        The wrapped morphism, or ``None`` when ``module`` wraps none.
    """
    return getattr(module, "_morphism", None)


def cup(obj: SetObject, algebra: Algebra | None = None) -> ObservedMorphism:
    """The compact-closed unit ``η_A : I → A ⊗ A``.

    For finite-set objects with their natural product, ``η_A`` is
    the *diagonal*: every entry ``(a, a)`` carries the algebra's
    monoidal unit and the off-diagonal entries carry the join unit
    (``zero``). The Kronecker-like tensor produced is the identity
    morphism's tensor reshaped from ``(*A.shape, *A.shape)`` to
    ``(1, *A.shape, *A.shape)`` so that the leading axis is the
    singleton input ``I``.

    Categorically the cup and the trivial-domain identity satisfy
    ``ε ∘ (id ⊗ η) = id``; the snake equation that makes V-Cat
    compact-closed.

    Parameters
    ----------
    obj : SetObject
        The object whose dual is being introduced. Every algebra
        ships an identity tensor; this morphism reshapes it as a
        Kleisli arrow from the singleton domain.
    algebra : Algebra, optional
        Override the default (ProductFuzzyAlgebra) algebra.

    Returns
    -------
    ObservedMorphism
        Morphism ``I → A ⊗ A`` whose tensor is the diagonal of the
        target.
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    diag = q.identity_tensor(obj.shape)
    # Wrap in a leading singleton axis so the morphism's domain is
    # the unit object I (the singleton finite set).
    I = FinSet(name="1", cardinality=1)
    cod = ProductSet(components=(obj, obj))
    return ObservedMorphism(I, cod, diag.unsqueeze(0), algebra=q)


def cap(obj: SetObject, algebra: Algebra | None = None) -> ObservedMorphism:
    """The compact-closed counit ``ε_A : A ⊗ A → I``.

    The dual of `cup`. The tensor is the diagonal flattened
    into ``(*A.shape, *A.shape, 1)`` so the trailing axis is the
    unit codomain ``I``.

    Parameters
    ----------
    obj : SetObject
        The object ``A``.
    algebra : Algebra, optional
        Override the default (ProductFuzzyAlgebra) algebra.

    Returns
    -------
    ObservedMorphism
        Morphism ``A ⊗ A → I`` whose tensor is the diagonal of the
        source.
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    diag = q.identity_tensor(obj.shape)
    I = FinSet(name="1", cardinality=1)
    dom = ProductSet(components=(obj, obj))
    return ObservedMorphism(dom, I, diag.unsqueeze(-1), algebra=q)


# ``quivers.categorical.functors`` imports this module, so the import
# resolving the `FunctorMorphism` annotation follows every definition.
from quivers.categorical.functors import Functor  # noqa: E402

__all__ = [
    "Morphism",
    "TransformedMorphism",
    "ObservedMorphism",
    "LatentMorphism",
    "ComposedMorphism",
    "ProductMorphism",
    "MarginalizedMorphism",
    "FunctorMorphism",
    "RepeatMorphism",
    "CurriedMorphism",
    "morphism",
    "observed",
    "identity",
    "as_torch_module",
    "extract_morphism",
    "cup",
    "cap",
]
