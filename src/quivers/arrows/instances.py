"""Standard arrow instances for the typeclass tower.

Implements the arrow interfaces of [`quivers.arrows.typeclasses`][quivers.arrows.typeclasses]
against the V-enriched category of finite sets. Each instance carries
no per-instance state — every operation is a stateless construction
on the underlying `Morphism` algebra. The instances register
themselves against the relevant ABCs (``Arrow``, ``ArrowChoice``,
``ArrowApply``, ``ArrowLoop``) so `isinstance` works
transparently for the chart-parser's arrow-side dispatch.

Realisations:

- `VRel`: the canonical V-enriched-relation arrow. ``compose``
  is the V-Rel ``>>`` operator; ``arr`` is the identity embedding;
  ``first`` is the parallel-product factory; ``left_arr`` is the
  coproduct functorial action; ``loop_arr`` is finite diagonal contraction over the feedback component.
- `Function`: restricts to deterministic V-relations (functions
  with point-mass tensors). Pure-functional operations.
- `Stochastic`: stochastic-matrix arrows. Neither total finite functions
  nor normalized stochastic kernels have general diagonal-contraction
  feedback, so `Function` and `Stochastic` do not implement `ArrowLoop`.
"""

from __future__ import annotations

import itertools

import didactic.api as dx
import torch

from quivers.arrows.typeclasses import (
    Arrow,
    ArrowApply,
    ArrowChoice,
    ArrowLoop,
)
from quivers.core._factories import (
    coproduct_map,
    parallel,
)
from quivers.core.morphisms import Morphism, observed
from quivers.core.morphisms import identity as id_morph
from quivers.core.objects import ProductSet, SetObject
from quivers.monadic.instances import _evaluation_morphism


def _iter_indices(shape: tuple[int, ...]):
    return itertools.product(*(range(s) for s in shape))


def _trace_v_rel(f: Morphism, C: SetObject) -> Morphism:
    """Finite diagonal contraction ``Tr^C : Hom(A⊗C, B⊗C) → Hom(A, B)``.

    Computes ``trace(f)(a, b) = ⨁_c f((a, c), (b, c))``. With a
    commutative semiring or suitable quantale, this is the compact-closed
    trace on weighted relations; the general trace axioms are described in
    [Joyal et al. (1996)](<https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/traced-monoidal-categories/2BE85628D269D9FABAB41B6364E117C8>). The implementation performs the contraction for the
    selected algebra, whose class membership alone does not prove the axioms.
    """
    if not isinstance(f.domain, ProductSet) or len(f.domain.components) != 2:
        raise TypeError(f"trace requires f : A⊗C → B⊗C; got domain {f.domain!r}")
    if not isinstance(f.codomain, ProductSet) or len(f.codomain.components) != 2:
        raise TypeError(f"trace requires f : A⊗C → B⊗C; got codomain {f.codomain!r}")
    A, C_dom = f.domain.components
    B, C_cod = f.codomain.components
    if C_dom != C or C_cod != C:
        raise TypeError(
            f"trace's recurrent components must both equal {C!r}; "
            f"got domain-side {C_dom!r}, codomain-side {C_cod!r}"
        )
    q = f.algebra
    # Reshape f.tensor to (*A.shape, *C.shape, *B.shape, *C.shape).
    # Then for each (a_idx, b_idx), join over the diagonal of the two
    # C axes (where the two C-coordinates are equal).
    A_ndim = A.ndim
    B_ndim = B.ndim
    data = f.tensor  # shape: (*A, *C, *B, *C)
    # Build the trace tensor by enumerating the diagonal of C.
    result = torch.full((*A.shape, *B.shape), q.zero)
    for c_idx in _iter_indices(C.shape):
        # Index into f.tensor with c_idx in both the domain-C slot
        # and the codomain-C slot.
        # Slicing form: f[..., c_idx, ..., c_idx] but we have to
        # name the axes correctly.
        # Build the slicing tuple: pick c_idx for the domain-C axes,
        # full slice for the rest.
        a_slice = (slice(None),) * A_ndim
        c_slice_dom = c_idx
        b_slice = (slice(None),) * B_ndim
        c_slice_cod = c_idx
        index = a_slice + c_slice_dom + b_slice + c_slice_cod
        slab = data[index]  # shape: (*A, *B)
        result = q.join(torch.stack([result, slab], dim=0), dim=0)
    return observed(A, B, result, algebra=q)


class VRel(dx.Model):
    """The V-enriched-relation arrow on finite sets.

    Hom-sets are V-Rel morphisms; composition is the standard
    ``>>``; ``arr`` is the identity embedding (every V-Rel morphism
    is an arrow); ``first`` is the parallel pair; ``left_arr`` is
    the coproduct functorial action; ``loop_arr`` is finite diagonal contraction.
    """

    name: str = "VRel"

    def id_arr(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f >> g

    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # Every V-Rel morphism is an arrow; arr is the identity.
        return f

    def first(self, f: Morphism, C: SetObject) -> Morphism:
        return parallel(f, id_morph(C))

    def left_arr(self, f: Morphism, C: SetObject) -> Morphism:
        return coproduct_map((f, id_morph(C)))

    def loop_arr(self, f: Morphism, C: SetObject) -> Morphism:
        return _trace_v_rel(f, C)


Arrow.register(VRel)
ArrowChoice.register(VRel)
ArrowLoop.register(VRel)


class Function(dx.Model):
    """Deterministic V-relation arrow (point-mass tensors only).

    Restricts `VRel` to morphisms whose tensors are 0/1-valued
    (function graphs). Composition, ``first``, and ``left_arr`` preserve
    determinism on this carrier. Arbitrary feedback can have several
    fixed points or none, so this instance does not implement `ArrowLoop`.
    """

    name: str = "Function"

    def id_arr(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f >> g

    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def first(self, f: Morphism, C: SetObject) -> Morphism:
        return parallel(f, id_morph(C))

    def left_arr(self, f: Morphism, C: SetObject) -> Morphism:
        return coproduct_map((f, id_morph(C)))

    def app(self, A: SetObject, B: SetObject) -> Morphism:
        """ArrowApply.app for Function — the standard evaluation."""
        return _evaluation_morphism(A, B)


Arrow.register(Function)
ArrowChoice.register(Function)
ArrowApply.register(Function)


class Stochastic(dx.Model):
    """Stochastic-matrix arrows.

    Each arrow is a `Morphism` whose tensor rows are
    probability distributions. Operations preserve row-stochasticity:
    composition is matrix multiplication; ``first`` and ``left_arr``
    are the tensor-product and coproduct lifts. Diagonal contraction
    need not preserve row sums, so this instance does not implement
    `ArrowLoop`.
    """

    name: str = "Stochastic"

    def id_arr(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f >> g

    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def first(self, f: Morphism, C: SetObject) -> Morphism:
        return parallel(f, id_morph(C))

    def left_arr(self, f: Morphism, C: SetObject) -> Morphism:
        return coproduct_map((f, id_morph(C)))


Arrow.register(Stochastic)
ArrowChoice.register(Stochastic)


# LinearMap deferred: requires a separate Morphism stratum (linear
# maps over ℝ, not V-relations). The framework's morphism hierarchy
# is currently V-enriched; a linear-map arrow would sit on a parallel
# stratum that the V-Rel constructions don't directly serve.


__all__ = [
    "VRel",
    "Function",
    "Stochastic",
]
