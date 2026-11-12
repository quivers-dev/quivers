"""Noisy-OR tensor contraction and marginalization.

The noisy-OR of probabilities p_1, ..., p_n is defined as:

    noisy_or(p_1, ..., p_n) = 1 - prod_i (1 - p_i)

This module implements tensor contraction (analogous to matrix multiplication)
where the summation is replaced by noisy-OR and the product is standard
multiplication.  Given tensors M of shape (*D, *S) and N of shape (*S, *C),
the contraction over the shared dimensions S produces a tensor of shape
(*D, *C) with entries:

    result[d..., c...] = 1 - prod_{s...} (1 - M[d..., s...] * N[s..., c...])

All operations use log-space for numerical stability.
"""

from __future__ import annotations

import torch

from quivers.core._util import EPS
from quivers.core.algebras import PRODUCT_FUZZY, Algebra


def noisy_or_contract(
    m: torch.Tensor,
    n: torch.Tensor,
    n_contract: int,
) -> torch.Tensor:
    """Contract two tensors via noisy-OR over shared dimensions.

    Parameters
    ----------
    m : torch.Tensor
        Left tensor of shape (*domain, *shared).
    n : torch.Tensor
        Right tensor of shape (*shared, *codomain).
    n_contract : int
        Number of trailing dimensions of m (= leading dimensions of n)
        to contract over.

    Returns
    -------
    torch.Tensor
        Result of shape (*domain, *codomain).

    Examples
    --------
    >>> M = torch.tensor([[0.5, 0.3], [0.8, 0.1]])
    >>> N = torch.tensor([[0.4, 0.6], [0.7, 0.2]])
    >>> result = noisy_or_contract(M, N, n_contract=1)
    >>> result.shape
    torch.Size([2, 2])
    """
    if n_contract < 1:
        raise ValueError(f"n_contract must be >= 1, got {n_contract}")

    # validate shared dimensions match
    shared_m = m.shape[-n_contract:]
    shared_n = n.shape[:n_contract]

    if shared_m != shared_n:
        raise ValueError(
            f"shared dimensions do not match: "
            f"m trailing {shared_m} != n leading {shared_n}"
        )

    n_domain = m.ndim - n_contract
    n_codomain = n.ndim - n_contract
    n_shared = n_contract

    # reshape for broadcasting:
    # m: (*domain, *shared, *[1]*n_codomain)
    # n: (*[1]*n_domain, *shared, *codomain)
    m_expanded = m.reshape(*m.shape, *([1] * n_codomain))

    n_expanded = n.reshape(*([1] * n_domain), *n.shape)

    # element-wise product
    product = m_expanded * n_expanded  # (*domain, *shared, *codomain)

    # noisy-OR over shared dimensions in log-space:
    # log(1 - product), sum over shared dims, then 1 - exp(sum)
    product_clamped = product.clamp(min=0.0, max=1.0 - EPS)
    log_complement = torch.log1p(-product_clamped)

    # shared dims are at positions [n_domain, n_domain + n_shared)
    contract_dims = tuple(range(n_domain, n_domain + n_shared))
    sum_log = log_complement.sum(dim=contract_dims)

    # result = 1 - exp(sum_log) = -(exp(sum_log) - 1) = -expm1(sum_log)
    result = -torch.expm1(sum_log)

    return result


def noisy_or_reduce(
    t: torch.Tensor,
    dim: int | tuple[int, ...],
) -> torch.Tensor:
    """Marginalize (reduce) a tensor over dimensions using noisy-OR.

    Computes 1 - prod_i (1 - t_i) along the specified dimension(s).
    This is the fuzzy analogue of existential quantification (∃).

    Parameters
    ----------
    t : torch.Tensor
        Input tensor with values in [0, 1].
    dim : int or tuple[int, ...]
        Dimension(s) to reduce over.

    Returns
    -------
    torch.Tensor
        Reduced tensor with the specified dimensions removed.
    """
    if isinstance(dim, int):
        dim = (dim,)

    t_clamped = t.clamp(min=0.0, max=1.0 - EPS)
    log_complement = torch.log1p(-t_clamped)
    sum_log = log_complement.sum(dim=dim)

    return -torch.expm1(sum_log)


def noisy_and_reduce(
    t: torch.Tensor,
    dim: int | tuple[int, ...],
) -> torch.Tensor:
    """Reduce a tensor over dimensions using product (fuzzy AND).

    Computes prod_i t_i along the specified dimension(s).
    This is the fuzzy analogue of universal quantification (∀).

    Parameters
    ----------
    t : torch.Tensor
        Input tensor with values in [0, 1].
    dim : int or tuple[int, ...]
        Dimension(s) to reduce over.

    Returns
    -------
    torch.Tensor
        Reduced tensor with the specified dimensions removed.
    """
    if isinstance(dim, int):
        dim = (dim,)

    # prod doesn't support tuple of dims natively; reduce iteratively
    # (sort descending so removing dims doesn't shift indices)
    result = t

    for d in sorted(dim, reverse=True):
        result = result.prod(dim=d)

    return result


def componentwise_lift(
    f: torch.Tensor,
    k: int,
    algebra: Algebra | None = None,
) -> torch.Tensor:
    """Lift a morphism tensor to the k-fold componentwise product.

    This is the functorial action of the free monoid functor on morphisms.
    Given f of shape (|A|, |B|), produces f^k of shape
    (|A|,)*k + (|B|,)*k where:

        f^k[a1, ..., ak, b1, ..., bk] = ⊗_i f[ai, bi]

    The tensor product ⊗ is determined by the algebra (defaults to
    ProductFuzzyAlgebra, where ⊗ = ordinary multiplication).

    Categorically, this is the monoidal functor action: the free monoid
    on objects sends A to A* = 1 + A + A×A + ..., and on morphisms sends
    f: A → B to f*: A* → B* acting componentwise on each length stratum.

    Parameters
    ----------
    f : torch.Tensor
        Morphism tensor of shape (|A|, |B|) — a 2D fuzzy relation.
    k : int
        String length (number of components). Must be >= 0.
    algebra : Algebra or None
        The algebra whose tensor_op is used for the componentwise
        product. If None, defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        Lifted tensor of shape (|A|,)*k + (|B|,)*k.
        For k=0, returns a tensor of shape (1, 1) filled with the
        algebra's unit value.
        For k=1, returns f unchanged.
    """
    if algebra is None:
        algebra = PRODUCT_FUZZY

    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    if k == 0:
        return torch.full((1, 1), algebra.unit, device=f.device, dtype=f.dtype)

    if k == 1:
        return f

    # build iteratively: at step i, result has shape
    # (|A|,)*i + (|B|,)*i and we extend to (|A|,)*(i+1) + (|B|,)*(i+1)
    result = f

    for _ in range(k - 1):
        n_a = result.ndim // 2

        # result: (*a_dims, *b_dims), f: (a, b)
        # target: (*a_dims, a, *b_dims, b)
        # step 1: outer product via algebra tensor_op
        shape_r = list(result.shape) + [1, 1]
        shape_f = [1] * (2 * n_a) + list(f.shape)
        outer = algebra.tensor_op(result.reshape(shape_r), f.reshape(shape_f))

        # step 2: permute [a1..an, b1..bn, a_{n+1}, b_{n+1}]
        #       to       [a1..an, a_{n+1}, b1..bn, b_{n+1}]
        perm = list(range(n_a)) + [2 * n_a] + list(range(n_a, 2 * n_a)) + [2 * n_a + 1]
        result = outer.permute(*perm)

    return result


__all__ = [
    "noisy_or_contract",
    "noisy_or_reduce",
    "noisy_and_reduce",
    "componentwise_lift",
]
