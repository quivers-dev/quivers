"""Tries of relations and the intersection step of Generic Join.

A relation over variables ``v_1, ..., v_k`` is kept as a trie: a nested
dictionary from the value of ``v_1`` to a trie over the rest, whose
leaves are what each tuple stands for (an item, or a weight). Generic
Join binds the variables of a query one at a time; the values of a
variable are the intersection, over the tries that contain it, of the
values they allow below the variables already bound, found by
iterating the smallest and probing the others (Ngo, Re and Rudra 2014).
Rule-body joins, the indexed chart and the agenda share this one step.

References
----------
Ngo, H. Q., Re, C., and Rudra, A. (2014). Skew strikes back: new
developments in the theory of join algorithms. *SIGMOD Record* 42(4),
5-16. [doi:10.1145/2590989.2590991](https://doi.org/10.1145/2590989.2590991)
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Sequence

__all__ = ["Trie", "TrieCounter", "descend", "insert", "intersect"]

type Trie[V: Hashable, L] = dict[V, Trie[V, L] | L]
"""A level of a trie over values ``V`` with leaves ``L``."""


class TrieCounter:
    """Counts of the intersection step."""

    def __init__(self) -> None:
        self.iterated = 0
        self.probes = 0
        self.tuples = 0


def insert[V: Hashable, L](root: Trie[V, L], values: Sequence[V], leaf: L) -> None:
    """Add a tuple to a trie.

    Parameters
    ----------
    root : Trie
        The trie.
    values : Sequence
        The tuple's values in the trie's variable order, at least one; a
        relation of no variables is kept under one key standing for the
        empty tuple.
    leaf : L
        What the tuple stands for.

    Raises
    ------
    ValueError
        If there are no values.
    """
    path: list[V] = list(values)
    if not path:
        raise ValueError("a trie keys at least one value")
    node = root
    for value in path[:-1]:
        child = node.setdefault(value, {})
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = leaf


def descend[V: Hashable, L](
    root: Trie[V, L], values: Sequence[V]
) -> Trie[V, L] | L | None:
    """The node of a trie below a prefix of values.

    Parameters
    ----------
    root : Trie
        The trie.
    values : Sequence
        The prefix; empty for the root.

    Returns
    -------
    Trie or L or None
        The node, or ``None`` when no tuple has the prefix.
    """
    node: Trie[V, L] | L = root
    for value in values:
        if not isinstance(node, dict):
            return None
        child = node.get(value)
        if child is None:
            return None
        node = child
    return node


def intersect[V: Hashable, L](
    order: Sequence[str],
    containing: Sequence[Sequence[int]],
    nodes: list[Trie[V, L] | L],
    binding: dict[str, V],
    counter: TrieCounter,
    depth: int = 0,
) -> Iterator[tuple[dict[str, V], list[Trie[V, L] | L]]]:
    """Bind the variables from ``depth`` on, one at a time, and yield each complete binding.

    Parameters
    ----------
    order : Sequence of str
        The free variables, in order.
    containing : Sequence of Sequence of int
        For each free variable, the tries that contain it.
    nodes : list
        Each trie's node below the variables bound so far; a trie whose
        variables are all bound is at its leaf.
    binding : dict
        The variables bound so far; extended in place and restored.
    counter : TrieCounter
        The counts.
    depth : int
        The next free variable.

    Yields
    ------
    tuple of dict and list
        A copy of each complete binding and every trie's leaf under it.
    """
    counter.tuples += 1
    if depth == len(order):
        yield dict(binding), list(nodes)
        return
    name = order[depth]
    tries = containing[depth]
    sets: list[Trie[V, L]] = []
    for index in tries:
        node = nodes[index]
        assert isinstance(node, dict)
        sets.append(node)
    smallest = min(range(len(sets)), key=lambda index: len(sets[index]))
    for value in list(sets[smallest]):
        counter.iterated += 1
        children: list[Trie[V, L] | L] = []
        for index, table in enumerate(sets):
            if index != smallest:
                counter.probes += 1
            child = table.get(value)
            if child is None:
                break
            children.append(child)
        else:
            advanced = list(nodes)
            for index, child in zip(tries, children, strict=True):
                advanced[index] = child
            binding[name] = value
            yield from intersect(
                order, containing, advanced, binding, counter, depth + 1
            )
            del binding[name]
