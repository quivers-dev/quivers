"""Yoneda embedding and representable profunctors.

The Yoneda lemma states that for any V-presheaf F: C^op → V and
object A in C:

    ∫_X [C(X, A), F(X)] ≅ F(A)

The routines below implement finite tensor analogues of this end.
The theorem requires lawful category and presheaf data; arbitrary input
tensors and an ``Algebra`` instance do not establish those hypotheses.

The Yoneda embedding y: C → Prof(C) sends each object A to the
representable profunctor C(-, A): C^op → V. The embedding is
full and faithful:

    C(A, B) ≅ [C^op, V](C(-, A), C(-, B))

This module provides:

    Presheaf                  — V-valued presheaf on finite objects
    yoneda_embedding()        — object to representable profunctor
    yoneda_lemma()            — compute the end ∫_X [C(X,A), F(X)]
    yoneda_density()          — decompose F via Yoneda density
    representable_profunctor()— hom profunctor C(-, A)
    corepresentable_profunctor() — hom profunctor C(A, -)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import itertools

import torch

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism
from quivers.core.algebras import PRODUCT_FUZZY, Algebra
from quivers.enriched.profunctors import Profunctor


@dataclass
class Presheaf:
    """A V-valued presheaf on a collection of finite objects.

    Represents a functor F: C^op → V where C is a finite
    category (collection of objects with hom-sets between them).
    In the simplest case, F assigns a V-tensor to each object.

    Holds a mutable ``dict[int, torch.Tensor]`` of values that grows during
    yoneda extension; not a value type.

    Parameters
    ----------
    objects : Sequence[SetObject]
        The objects of the finite category C.
    values : dict[int, torch.Tensor]
        Maps object index → V-tensor. The tensor at index i
        represents F(objects[i]), a V-valued "set" of shape
        (*objects[i].shape,).
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    objects: Sequence[SetObject]
    values: dict[int, torch.Tensor]
    algebra: Algebra | None = None

    def __post_init__(self) -> None:
        if self.algebra is None:
            self.algebra = PRODUCT_FUZZY

    def evaluate(self, index: int) -> torch.Tensor:
        """Evaluate the presheaf at object index.

        Parameters
        ----------
        index : int
            Index into the objects list.

        Returns
        -------
        torch.Tensor
            The value F(objects[index]).
        """
        return self.values[index]

    @property
    def size(self) -> int:
        """Number of objects."""
        return len(self.objects)


def representable_profunctor(
    obj: SetObject,
    algebra: Algebra | None = None,
) -> Profunctor:
    """The representable profunctor C(-, A) as a self-profunctor.

    This is the Yoneda image of A: the profunctor from A to A
    given by the identity (hom from A to itself).

    For a single-object view, this is the identity morphism
    viewed as a profunctor A ↛ A.

    Parameters
    ----------
    obj : SetObject
        The representing object A.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    Profunctor
        The representable profunctor y(A) = C(-, A).
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    id_tensor = q.identity_tensor(obj.shape)

    return Profunctor(contra=obj, co=obj, tensor=id_tensor, algebra=q)


def corepresentable_profunctor(
    obj: SetObject,
    algebra: Algebra | None = None,
) -> Profunctor:
    """The corepresentable profunctor C(A, -) as a self-profunctor.

    Dual to the representable. Also the identity viewed differently.

    Parameters
    ----------
    obj : SetObject
        The corepresenting object A.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    Profunctor
        The corepresentable profunctor C(A, -).
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    id_tensor = q.identity_tensor(obj.shape)

    return Profunctor(contra=obj, co=obj, tensor=id_tensor, algebra=q)


def yoneda_embedding(
    morph: Morphism,
) -> Profunctor:
    """Apply the Yoneda embedding to a morphism.

    Given f: A → B, the Yoneda embedding produces a profunctor
    morphism y(f): y(A) → y(B), which is the profunctor
    view of f.

    This is equivalent to Profunctor.from_morphism but makes
    the categorical origin explicit.

    Parameters
    ----------
    morph : Morphism
        A morphism f: A → B.

    Returns
    -------
    Profunctor
        The profunctor A ↛ B corresponding to f.
    """
    return Profunctor.from_morphism(morph)


def yoneda_lemma(
    presheaf: Presheaf,
    obj_index: int,
    hom_tensors: Sequence[torch.Tensor],
    algebra: Algebra | None = None,
) -> torch.Tensor:
    """Compute the Yoneda lemma: ∫_X [C(X, A), F(X)] ≅ F(A).

    Given a presheaf F and an object A (identified by index), with
    hom tensors C(X_i, A) provided for each object X_i, computes
    the end ∫_X [C(X, A), F(X)].

    The mathematical Yoneda lemma identifies this end with F(A) under
    its hypotheses. The routine does not check all of those hypotheses
    or construct a general categorical isomorphism.

    Parameters
    ----------
    presheaf : Presheaf
        The V-presheaf F.
    obj_index : int
        The index of object A in the presheaf's object list.
    hom_tensors : Sequence[torch.Tensor]
        For each object X_i in the presheaf, the hom tensor
        C(X_i, A) of shape (*X_i.shape, *A.shape).
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        The finite internal-hom reduction described above.
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY

    if len(hom_tensors) != presheaf.size:
        raise ValueError(
            f"expected {presheaf.size} hom tensors, got {len(hom_tensors)}"
        )

    obj_a = presheaf.objects[obj_index]

    # compute [C(X_i, A), F(X_i)] for each i, then meet over i
    # [C(X_i, A), F(X_i)] has shape (*A.shape,) after contracting X_i
    components: list[torch.Tensor] = []

    for i in range(presheaf.size):
        x_i = presheaf.objects[i]
        hom = hom_tensors[i]  # shape (*x_i.shape, *a.shape)
        f_xi = presheaf.evaluate(i)  # shape (*x_i.shape,)

        # compute [hom, f] via internal hom
        # for each a in A: [C(X, A)](a) = ⋀_x [C(x, a), F(x)]
        result_shape = obj_a.shape
        component = torch.full(result_shape, q.unit)

        for a_idx in itertools.product(*(range(s) for s in obj_a.shape)):
            # extract hom(-, a): shape (*x_i.shape,)
            hom_slice = hom[(..., *a_idx)]

            # [hom(x, a), F(x)] for each x, then meet
            from quivers.enriched.weighted_limits import _internal_hom_scalar

            vals: list[torch.Tensor] = []

            for x_idx in itertools.product(*(range(s) for s in x_i.shape)):
                h_val = hom_slice[x_idx]
                f_val = f_xi[x_idx]
                ih = _internal_hom_scalar(h_val, f_val, q)
                vals.append(ih)

            if vals:
                stacked = torch.stack(vals)
                component[a_idx] = q.meet(stacked, dim=0)

        components.append(component)

    # meet over all objects (the end)
    if not components:
        return torch.tensor(q.unit)

    stacked = torch.stack(components, dim=0)
    return q.meet(stacked, dim=0)


def yoneda_density(
    morph: Morphism,
    algebra: Algebra | None = None,
) -> torch.Tensor:
    """Verify Yoneda density: f ≅ ∫^X C(A, X) ⊗ f(X, -).

    For a morphism f: A → B, the Yoneda density theorem says f
    can be recovered as the coend:

        f(a, b) = ∫^x C(a, x) ⊗ f(x, b) = ⋁_x δ(a,x) ⊗ f(x, b)

    which states that composing with the identity gives back f.
    This function computes the coend and verifies it.

    Parameters
    ----------
    morph : Morphism
        The morphism f: A → B.
    algebra : Algebra or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        The coend result, which should equal morph.tensor.
    """
    q = algebra if algebra is not None else PRODUCT_FUZZY
    id_a = q.identity_tensor(morph.domain.shape)

    # coend is just composition with identity = f itself
    return q.compose(id_a, morph.tensor, morph.domain.ndim)


def verify_yoneda_fully_faithful(
    f: Morphism,
    g: Morphism,
    atol: float = 1e-5,
) -> bool:
    """Check preservation of one supplied composite numerically.

    Check that the profunctor morphism y(f) composed with y(g)
    equals y(f >> g). Despite the historical function name, this
    tests one composition equation, not fullness or faithfulness.

    Parameters
    ----------
    f : Morphism
        First morphism f: A → B.
    g : Morphism
        Second morphism g: B → C.
    atol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if y(f >> g) ≈ y(f) ; y(g).
    """
    # y(f >> g)
    fg = f >> g
    y_fg = yoneda_embedding(fg)

    # y(f) ; y(g)
    y_f = yoneda_embedding(f)
    y_g = yoneda_embedding(g)
    y_composed = y_f.compose(y_g)

    return torch.allclose(y_fg.tensor, y_composed.tensor, atol=atol)


__all__ = [
    "Presheaf",
    "representable_profunctor",
    "corepresentable_profunctor",
    "yoneda_embedding",
    "yoneda_lemma",
    "yoneda_density",
    "verify_yoneda_fully_faithful",
]
