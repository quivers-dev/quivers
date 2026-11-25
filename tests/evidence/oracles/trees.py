"""Enumeration of the directed spanning trees of a complete graph.

``weights[(x, y)]`` is the weight of the edge from parent ``x`` to child
``y``. A spanning tree gives every node except the root exactly one
parent and has no cycle.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from fractions import Fraction

import torch

type Edge = tuple[int, int]
"""An edge from a parent to a child."""

type Parents = tuple[Edge, ...]
"""The edges of one spanning tree, by child."""


def spanning_trees(nodes: int, root: int) -> list[Parents]:
    """Every directed spanning tree with a given root.

    Parameters
    ----------
    nodes : int
        The number of nodes, at most seven.
    root : int
        The root.

    Returns
    -------
    list of Parents
        The trees, each once.


    Raises
    ------
    ValueError
        If there are more than seven nodes or the root is not a node.
    """
    if not 0 <= root < nodes <= 7:
        raise ValueError("enumeration is for one to seven nodes and a root among them")
    children = [node for node in range(nodes) if node != root]
    found: list[Parents] = []
    for parents in itertools.product(range(nodes), repeat=len(children)):
        parent_of = dict(zip(children, parents, strict=True))
        if all(_reaches(child, parent_of, root) for child in children):
            found.append(tuple((parent_of[child], child) for child in children))
    return found


def _reaches(node: int, parent_of: Mapping[int, int], root: int) -> bool:
    """Whether following parents from a node ends at the root.

    Parameters
    ----------
    node : int
        The start.
    parent_of : Mapping[int, int]
        The parent of every node except the root.
    root : int
        The root.

    Returns
    -------
    bool
        Whether the walk reaches the root without repeating a node.
    """
    seen: set[int] = set()
    while node != root:
        if node in seen:
            return False
        seen.add(node)
        node = parent_of[node]
    return True


def partition[N: (Fraction, float, torch.Tensor)](
    trees: Sequence[Parents], weights: Mapping[Edge, N], zero: N, one: N
) -> N:
    """The sum over trees of the product of their edge weights.

    Parameters
    ----------
    trees : Sequence of Parents
        The trees.
    weights : Mapping[Edge, N]
        The weight of each edge.
    zero : N
        The additive unit.
    one : N
        The multiplicative unit.

    Returns
    -------
    N
        The partition value.
    """
    answer = zero
    for tree in trees:
        product = one
        for edge in tree:
            product = product * weights[edge]
        answer = answer + product
    return answer


def edge_marginals(
    trees: Sequence[Parents], weights: Mapping[Edge, Fraction]
) -> dict[Edge, Fraction]:
    """The probability of each edge under the normalized tree weights.

    Parameters
    ----------
    trees : Sequence of Parents
        The trees.
    weights : Mapping[Edge, Fraction]
        The weight of each edge.

    Returns
    -------
    dict[Edge, Fraction]
        The marginals.
    """
    mass = partition(trees, weights, Fraction(0), Fraction(1))
    marginals = dict.fromkeys(weights, Fraction(0))
    for tree in trees:
        weight = partition([tree], weights, Fraction(0), Fraction(1))
        for edge in tree:
            marginals[edge] += weight / mass
    return marginals
