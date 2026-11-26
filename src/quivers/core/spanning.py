"""Directed spanning trees: partition functions and edge marginals.

A directed spanning tree (an arborescence) of nodes ``0, ..., n-1``
rooted at ``r`` gives every node except ``r`` exactly one parent and has
no cycle. ``weights[..., x, y]`` is the weight of the edge from parent
``x`` to child ``y``. The partition function sums, over every such tree,
the product of its edge weights; self edges, edges into the root, and
edges a mask removes are absent.

The partition is computed by eliminating the nodes other than the root
one at a time, with sums, products and quotients of the algebra and no
subtraction: for a node ``k`` with total incoming weight
``in(k) = ⊕_x W[x, k]``,

    Z(W) = in(k) ⊗ Z(W'),   W'[x, y] = W[x, y] ⊕ W[x, k] ⊗ W[k, y] ⊘ in(k),

on the nodes other than ``k``. The identity holds in every commutative
semifield; the proof is in the API page of this module. Each step costs
``O(n^2)`` operations, so the partition costs ``O(n^3)``. Determinants
need subtraction, and in floating point the determinant of a badly
scaled Laplacian can cancel to zero where the partition is not: for
edge log weights ``-60`` and ``-61`` from the root and a unit cycle
between the other two nodes the partition is ``-59.6867`` in log space,
which elimination computes exactly, while a log determinant returns
``-inf``.

Edge marginals are the derivative of the partition with respect to the
weights: in the log semifield the probability of each edge under the
normalized tree weights, in the max-plus semifield the indicator of the
best tree by the algebra's tie rule.

References
----------
Tutte, W. T. (1984). *Graph Theory*. Addison-Wesley.

Koo, T., Globerson, A., Carreras, X., and Collins, M. (2007).
Structured prediction models via the matrix-tree theorem. *EMNLP-CoNLL*,
141-150. https://aclanthology.org/D07-1015/

Andrews, N., Eisner, J., and Dredze, M. (2012). Name phylogeny: a
generative model of string variation. *EMNLP-CoNLL*, 344-355.
https://aclanthology.org/D12-1032/
"""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.core.algebras import LOG_PROB, Semifield, ZeroSumFree

__all__ = [
    "DirectedTrees",
    "directed_tree",
    "tree_marginals",
    "tree_partition",
]


def _require(algebra: Semifield, weights: torch.Tensor, root: int) -> int:
    """Check the algebra, the shape of the weights, and the root.

    Parameters
    ----------
    algebra : Semifield
        The weights' algebra.
    weights : torch.Tensor
        Weights of shape ``(*batch, n, n)``.
    root : int
        The root.

    Returns
    -------
    int
        The number of nodes.

    Raises
    ------
    TypeError
        If the algebra is not a zero-sum-free semifield with scalar
        elements.
    ValueError
        If the weights are not square in their last two axes, or the root
        is not a node.
    """
    if not isinstance(algebra, Semifield) or not isinstance(algebra, ZeroSumFree):
        raise TypeError(
            f"{algebra.name} is not a zero-sum-free semifield; elimination divides "
            "by sums of incoming weights, which must vanish only when every term does"
        )
    if algebra.element_shape != ():
        raise TypeError(f"the elements of {algebra.name} are not scalars")
    if weights.ndim < 2 or weights.shape[-1] != weights.shape[-2]:
        raise ValueError(
            f"weights have shape {tuple(weights.shape)}; the last two axes are "
            "parents and children of one set of nodes"
        )
    nodes = weights.shape[-1]
    if not 0 <= root < nodes:
        raise ValueError(f"root {root} is not one of {nodes} nodes")
    return nodes


def _allowed(
    nodes: int, root: int, mask: torch.Tensor | None, like: torch.Tensor
) -> torch.Tensor:
    """The edges that may appear in a tree.

    Parameters
    ----------
    nodes : int
        The number of nodes.
    root : int
        The root.
    mask : torch.Tensor or None
        A boolean mask broadcastable to the weights, or ``None``.
    like : torch.Tensor
        The weights, for the device.

    Returns
    -------
    torch.Tensor
        True where an edge may appear: not a self edge, not into the root,
        and allowed by the mask.
    """
    allowed = ~torch.eye(nodes, dtype=torch.bool, device=like.device)
    allowed[:, root] = False
    if mask is not None:
        allowed = allowed & mask.to(device=like.device, dtype=torch.bool)
    return allowed


def tree_partition(
    weights: torch.Tensor,
    root: int,
    *,
    algebra: Semifield = LOG_PROB,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The partition function over directed spanning trees with a given root.

    Nodes are eliminated in an order chosen per graph: at each step, a
    remaining node whose incoming weight is not zero. When every remaining
    node has zero incoming weight and two or more remain, there is no tree,
    and every first derivative of the partition vanishes, so the result is
    the zero with derivative zero. When one remains, the partition is the
    product of the pivots, the last of them zero; where the algebra's zero
    is a finite number its derivative is the derivative of that product,
    and where the zero lies at infinity, as ``-inf`` in log space, the
    partition is not differentiable there and its derivative is zero. A
    zero is decided by the algebra, so a weight that is zero but
    differentiable keeps its derivative.

    Parameters
    ----------
    weights : torch.Tensor
        Shape ``(*batch, n, n)``; ``weights[..., x, y]`` weights the edge
        from parent ``x`` to child ``y``.
    root : int
        The root, which has no parent.
    algebra : Semifield
        A zero-sum-free semifield with scalar elements; the log semifield
        by default.
    mask : torch.Tensor or None
        A boolean mask broadcastable to the weights; False removes an edge.

    Returns
    -------
    torch.Tensor
        Shape ``batch``: the partition, the algebra's unit for one node,
        and its zero when there is no tree.

    Raises
    ------
    TypeError
        If the algebra is not a zero-sum-free semifield with scalar
        elements.
    ValueError
        If the weights are not square or the root is not a node.
    """
    nodes = _require(algebra, weights, root)
    batch = tuple(weights.shape[:-2])
    allowed = _allowed(nodes, root, mask, weights)
    zero = algebra.zero_like(tuple(weights.shape), weights)
    current = torch.where(allowed, weights, zero)
    total = algebra.unit_like(batch, weights)
    remaining = torch.ones((*batch, nodes), dtype=torch.bool, device=weights.device)
    remaining[..., root] = False
    dead = torch.zeros(batch, dtype=torch.bool, device=weights.device)
    off_diagonal = ~torch.eye(nodes, dtype=torch.bool, device=weights.device)
    for _ in range(nodes - 1):
        pivots = algebra.join(current, -2)
        live = remaining & ~algebra.is_zero(pivots)
        has_live = live.any(dim=-1)
        choice = torch.where(
            has_live,
            live.to(torch.int8).argmax(dim=-1),
            remaining.to(torch.int8).argmax(dim=-1),
        )
        last = remaining.sum(dim=-1) == 1
        dead = dead | (~has_live & ~last)
        index = choice.unsqueeze(-1)
        pivot = torch.take_along_dim(pivots, index, dim=-1).squeeze(-1)
        total = algebra.tensor_op(total, pivot)
        safe = torch.where(
            algebra.is_zero(pivot), algebra.unit_like(batch, weights), pivot
        )
        column = torch.take_along_dim(current, index.unsqueeze(-2), dim=-1).squeeze(-1)
        row = torch.take_along_dim(current, index.unsqueeze(-1), dim=-2).squeeze(-2)
        through = algebra.divide(
            algebra.tensor_op(column.unsqueeze(-1), row.unsqueeze(-2)),
            safe.unsqueeze(-1).unsqueeze(-1),
        )
        removed = torch.nn.functional.one_hot(choice, nodes).to(torch.bool)
        remaining = remaining & ~removed
        # A contracted path through the eliminated node is an edge of the
        # smaller graph even where the mask removed the direct edge; only
        # self edges and the eliminated node's own edges are dropped.
        keep = off_diagonal & ~removed.unsqueeze(-1) & ~removed.unsqueeze(-2)
        current = torch.where(
            keep, algebra.join(torch.stack((current, through)), 0), zero
        )
    # Where the algebra's zero is a finite number (the linear semifields),
    # a partition of zero with a live weight keeps the derivative of the
    # product of the pivots. Where the zero lies at infinity (log space),
    # the representation is not differentiable there, and its derivative
    # is zero.
    zero_result = algebra.zero_like(batch, weights)
    at_infinity = algebra.is_zero(total) & ~torch.isfinite(zero_result)
    return torch.where(dead | at_infinity, zero_result, total)


def tree_marginals(
    weights: torch.Tensor,
    root: int,
    *,
    algebra: Semifield = LOG_PROB,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The derivative of the partition with respect to each edge weight.

    This is the derivative of the partition itself, so it is right
    wherever the partition is: in the log semifield it is the marginal
    probability of each edge; in the max-plus semifield, the indicator of
    the best tree. The derivative is part of the autograd graph, so it can
    be differentiated again.

    Parameters
    ----------
    weights : torch.Tensor
        Shape ``(*batch, n, n)``.
    root : int
        The root.
    algebra : Semifield
        A zero-sum-free semifield with scalar elements.
    mask : torch.Tensor or None
        A boolean mask broadcastable to the weights.

    Returns
    -------
    torch.Tensor
        Shape ``(*batch, n, n)``; zero for absent edges and for graphs
        with no tree.

    Raises
    ------
    TypeError
        If the algebra is not a zero-sum-free semifield with scalar
        elements.
    ValueError
        If the weights are not square or the root is not a node.
    """
    source = weights if weights.requires_grad else weights.detach().requires_grad_()
    partition = tree_partition(source, root, algebra=algebra, mask=mask)
    (gradient,) = torch.autograd.grad(
        partition.sum(), source, create_graph=weights.requires_grad
    )
    return gradient


class DirectedTrees(ValueModel):
    """The directed spanning trees of a weighted graph with a given root.

    Parameters
    ----------
    weights : torch.Tensor
        Shape ``(*batch, n, n)``; ``weights[..., x, y]`` weights the edge
        from parent ``x`` to child ``y``.
    root : int
        The root.
    algebra : Semifield
        A zero-sum-free semifield with scalar elements.
    mask : torch.Tensor or None
        A boolean mask broadcastable to the weights.
    """

    weights: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    root: int
    algebra: Semifield = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    mask: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Check the algebra, the shape, and the root.

        Raises
        ------
        TypeError
            If the algebra is not a zero-sum-free semifield with scalar
            elements.
        ValueError
            If the weights are not square or the root is not a node.
        """
        _require(self.algebra, self.weights, self.root)

    def partition(self) -> torch.Tensor:
        """The partition function; in the log semifield, the log partition.

        Returns
        -------
        torch.Tensor
            Shape ``batch``.
        """
        return tree_partition(
            self.weights, self.root, algebra=self.algebra, mask=self.mask
        )

    def edge_marginals(self) -> torch.Tensor:
        """The derivative of the partition with respect to each edge weight.

        Returns
        -------
        torch.Tensor
            Shape ``(*batch, n, n)``.
        """
        return tree_marginals(
            self.weights, self.root, algebra=self.algebra, mask=self.mask
        )


def directed_tree(
    weights: torch.Tensor,
    *,
    root: int,
    algebra: Semifield = LOG_PROB,
    mask: torch.Tensor | None = None,
) -> DirectedTrees:
    """The directed spanning trees of a graph with a given root.

    Parameters
    ----------
    weights : torch.Tensor
        Shape ``(*batch, n, n)``; in the default log semifield, edge log
        weights.
    root : int
        The root.
    algebra : Semifield
        A zero-sum-free semifield with scalar elements.
    mask : torch.Tensor or None
        A boolean mask broadcastable to the weights.

    Returns
    -------
    DirectedTrees
        The distribution, with `DirectedTrees.partition` and
        `DirectedTrees.edge_marginals`.
    """
    return DirectedTrees(weights=weights, root=root, algebra=algebra, mask=mask)
