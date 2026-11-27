"""Persistent environments of the reference machine.

A run of the machine binds a local at every ``Bind``, every call and every
case branch, and keeps the environment of each pending frame so the frame
can continue under it. A copied dictionary makes every bind cost the size
of the scope, so a program of ``n`` binds runs in time quadratic in ``n``.
`Scope` is a persistent map instead: binding returns a new scope that
shares all but one path of the old one, so a frame keeps its environment
without a copy.

The map is a hash array mapped trie of branching 32. Binding and lookup
walk one path of the trie, whose depth is at most ``ceil(64 / 5) = 13``
and, for a scope of ``n`` locals with well-spread hashes, about
``log_32 n``; each step makes at most one node of at most 32 entries. Two
locals whose 64-bit hashes are equal share a bucket, scanned in full.
"""

from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import cast

_BITS = 5
"""Hash bits consumed per level of the trie."""

_MASK = (1 << _BITS) - 1
"""The mask selecting one level's bits."""

_HASH_BITS = 64
"""The hash width, so a path never runs deeper than ``ceil(64 / 5)``."""

_WIDTH = (1 << _HASH_BITS) - 1
"""The mask taking a Python hash to an unsigned 64-bit one."""


class _Leaf:
    """One binding at the end of a path.

    Parameters
    ----------
    code : int
        The key's unsigned hash.
    key : Hashable
        The key.
    value : object
        Its value.
    """

    __slots__ = ("code", "key", "value")

    def __init__(self, code: int, key: Hashable, value: object) -> None:
        self.code = code
        self.key = key
        self.value = value


class _Bucket:
    """Bindings whose keys share one full hash.

    Parameters
    ----------
    code : int
        The shared unsigned hash.
    leaves : tuple[_Leaf, ...]
        The bindings, keys distinct.
    """

    __slots__ = ("code", "leaves")

    def __init__(self, code: int, leaves: tuple[_Leaf, ...]) -> None:
        self.code = code
        self.leaves = leaves


class _Branch:
    """An inner node: a bitmap of occupied slots and their children.

    Parameters
    ----------
    bitmap : int
        Bit ``i`` is set when slot ``i`` has a child.
    children : tuple[_Leaf | _Bucket | _Branch, ...]
        The children of the occupied slots, in slot order.
    """

    __slots__ = ("bitmap", "children")

    def __init__(
        self, bitmap: int, children: tuple[_Leaf | _Bucket | _Branch, ...]
    ) -> None:
        self.bitmap = bitmap
        self.children = children


type _Node = _Leaf | _Bucket | _Branch

_EMPTY = _Branch(0, ())
"""The root of the empty scope."""


def _code(key: Hashable) -> int:
    """A key's hash as an unsigned 64-bit number.

    Parameters
    ----------
    key : Hashable
        The key.

    Returns
    -------
    int
        Its hash.
    """
    return hash(key) & _WIDTH


def _pair(first: _Leaf | _Bucket, second: _Leaf, shift: int) -> _Node:
    """A node holding two entries whose hashes differ.

    Parameters
    ----------
    first : _Leaf or _Bucket
        An entry already in the trie.
    second : _Leaf
        The entry added.
    shift : int
        The bit position the node's slots are read at.

    Returns
    -------
    _Node
        A branch, nested until the hashes part.
    """
    left = (first.code >> shift) & _MASK
    right = (second.code >> shift) & _MASK
    if left == right:
        return _Branch(1 << left, (_pair(first, second, shift + _BITS),))
    if left < right:
        return _Branch((1 << left) | (1 << right), (first, second))
    return _Branch((1 << left) | (1 << right), (second, first))


def _bound(node: _Node, shift: int, leaf: _Leaf) -> tuple[_Node, bool]:
    """A node with one more or one replaced binding.

    Parameters
    ----------
    node : _Node
        The node.
    shift : int
        The bit position its slots are read at.
    leaf : _Leaf
        The binding.

    Returns
    -------
    tuple[_Node, bool]
        The new node, sharing every untouched child, and whether the key
        is new.
    """
    if isinstance(node, _Branch):
        bit = 1 << ((leaf.code >> shift) & _MASK)
        index = (node.bitmap & (bit - 1)).bit_count()
        children = node.children
        if not node.bitmap & bit:
            return _Branch(
                node.bitmap | bit, (*children[:index], leaf, *children[index:])
            ), True
        child, added = _bound(children[index], shift + _BITS, leaf)
        return _Branch(
            node.bitmap, (*children[:index], child, *children[index + 1 :])
        ), added
    if isinstance(node, _Leaf):
        if node.code == leaf.code:
            if node.key == leaf.key:
                return leaf, False
            return _Bucket(leaf.code, (node, leaf)), True
        return _pair(node, leaf, shift), True
    if node.code != leaf.code:
        return _pair(node, leaf, shift), True
    for position, kept in enumerate(node.leaves):
        if kept.key == leaf.key:
            leaves = (*node.leaves[:position], leaf, *node.leaves[position + 1 :])
            return _Bucket(node.code, leaves), False
    return _Bucket(node.code, (*node.leaves, leaf)), True


def _leaves(node: _Node) -> Iterator[_Leaf]:
    """Every binding under a node.

    Parameters
    ----------
    node : _Node
        The node.

    Yields
    ------
    _Leaf
        Each binding.
    """
    pending: list[_Node] = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, _Leaf):
            yield current
        elif isinstance(current, _Bucket):
            yield from current.leaves
        else:
            pending.extend(reversed(current.children))


class Scope[K: Hashable, V](Mapping[K, V]):
    """A persistent map from locals to the values they are bound to.

    A scope never changes: `bind` and `extend` return a new scope sharing
    the old one's nodes, so a frame that keeps a scope keeps it at no
    cost. Each operation walks one path and inspects the entries sharing
    the key's full hash. With well-distributed hashes, binding and lookup
    take time logarithmic in the scope's size.

    Parameters
    ----------
    entries : Mapping or iterable of pairs
        The bindings to start from; none for the empty scope.
    """

    __slots__ = ("_root", "_size")

    _root: _Node
    _size: int

    def __init__(self, entries: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        self._root = _EMPTY
        self._size = 0
        pairs = entries.items() if isinstance(entries, Mapping) else entries
        for key, value in pairs:
            self._root, added = _bound(self._root, 0, _Leaf(_code(key), key, value))
            self._size += added

    @classmethod
    def of(cls, entries: Mapping[K, V] | Iterable[tuple[K, V]]) -> Scope[K, V]:
        """A scope with given bindings, reusing one that is already a scope.

        Parameters
        ----------
        entries : Mapping or iterable of pairs
            The bindings.

        Returns
        -------
        Scope
            ``entries`` itself when it is a scope, else a new one.
        """
        if isinstance(entries, Scope):
            return cast("Scope[K, V]", entries)
        return cls(entries)

    def _with(self, root: _Node, size: int) -> Scope[K, V]:
        """A scope over another root.

        Parameters
        ----------
        root : _Node
            The root.
        size : int
            Its number of bindings.

        Returns
        -------
        Scope
            The scope.
        """
        made = object.__new__(type(self))
        made._root = root
        made._size = size
        return made

    def bind(self, key: K, value: V) -> Scope[K, V]:
        """This scope with one local bound, or rebound.

        Parameters
        ----------
        key : K
            The local.
        value : V
            Its value.

        Returns
        -------
        Scope
            The new scope; this one is unchanged.
        """
        root, added = _bound(self._root, 0, _Leaf(_code(key), key, value))
        return self._with(root, self._size + added)

    def extend(self, pairs: Iterable[tuple[K, V]]) -> Scope[K, V]:
        """This scope with several locals bound in order.

        Parameters
        ----------
        pairs : iterable of (K, V)
            The bindings; a later one of a key replaces an earlier.

        Returns
        -------
        Scope
            The new scope; this one is unchanged.
        """
        root = self._root
        size = self._size
        for key, value in pairs:
            root, added = _bound(root, 0, _Leaf(_code(key), key, value))
            size += added
        return self._with(root, size)

    def __getitem__(self, key: K) -> V:
        """The value a local is bound to.

        Parameters
        ----------
        key : K
            The local.

        Returns
        -------
        V
            Its value.

        Raises
        ------
        KeyError
            If the local is not bound.
        """
        code = _code(key)
        node = self._root
        shift = 0
        while True:
            if isinstance(node, _Branch):
                bit = 1 << ((code >> shift) & _MASK)
                if not node.bitmap & bit:
                    raise KeyError(key)
                node = node.children[(node.bitmap & (bit - 1)).bit_count()]
                shift += _BITS
                continue
            if isinstance(node, _Leaf):
                if node.code == code and node.key == key:
                    return node.value  # type: ignore[return-value]
                raise KeyError(key)
            if node.code == code:
                for leaf in node.leaves:
                    if leaf.key == key:
                        return leaf.value  # type: ignore[return-value]
            raise KeyError(key)

    def __iter__(self) -> Iterator[K]:
        """Every bound local.

        Returns
        -------
        Iterator[K]
            The locals, in an order fixed by their hashes.
        """
        return (leaf.key for leaf in _leaves(self._root))  # type: ignore[misc]

    def __len__(self) -> int:
        """The number of bound locals.

        Returns
        -------
        int
            The number.
        """
        return self._size

    def __repr__(self) -> str:
        """The bindings, as a dictionary shows them.

        Returns
        -------
        str
            The representation.
        """
        return f"Scope({dict(self.items())!r})"


__all__ = ["Scope"]
