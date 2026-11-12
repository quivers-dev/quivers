"""Objects of the category of finite sets: finite sets, products, coproducts, and more.

The set-object family is a sum type with six variants, discriminated by
``kind``: `FinSet`, `ProductSet`, `CoproductSet`, `FreeMonoid`,
`EnumSet` and `FreeResiduated`. ``SetObject`` is the `dx.TaggedUnion`
root that gathers them; ``X * Y`` and ``X + Y`` work on any pair of
variants.
"""

from math import prod
from typing import Literal
import didactic.api as dx


class SetObject(dx.TaggedUnion, discriminator="kind"):
    """A finite set, the type of objects of the category of finite sets.

    A ``SetObject`` is one object, not the category: one of the six
    variants `FinSet`, `ProductSet`, `CoproductSet`, `FreeMonoid`,
    `EnumSet` and `FreeResiduated`. Variants share two derived
    properties (`size`, `shape`) and the ``*`` / ``+`` operators for
    product and coproduct construction.
    """

    @property
    def size(self) -> int:
        """Total cardinality of the set."""
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor dimension shape."""
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return len(self.shape)

    def __mul__(self, other: "SetObject") -> "ProductSet":
        if not isinstance(other, SetObject):
            return NotImplemented
        return ProductSet(components=(self, other))

    def __add__(self, other: "SetObject") -> "CoproductSet":
        if not isinstance(other, SetObject):
            return NotImplemented
        return CoproductSet(components=(self, other))


def _check_cardinality(value: int) -> int:
    if value < 1:
        raise ValueError(f"cardinality must be >= 1, got {value}")
    return value


class FinSet(SetObject):
    """A named finite set with a fixed cardinality.

    Attributes
    ----------
    name : str
        Human-readable name for the set.
    cardinality : int
        Number of elements (must be ``>= 1``).
    """

    name: str
    cardinality: int = dx.field(converter=_check_cardinality)
    kind: Literal["finset"] = "finset"

    @property
    def size(self) -> int:
        return self.cardinality

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.cardinality,)

    def __str__(self) -> str:
        return f"FinSet(name={self.name!r}, cardinality={self.cardinality})"


Unit: FinSet = FinSet(name="1", cardinality=1)
"""The terminal object, the one-element finite set."""


def _flatten_products(items: tuple["SetObject", ...]) -> tuple["SetObject", ...]:
    """Flatten nested ProductSet so that P(A, P(B, C)) collapses to P(A, B, C)."""
    out: list[SetObject] = []
    for c in items:
        if isinstance(c, ProductSet):
            out.extend(c.components)
        else:
            out.append(c)
    return tuple(out)


class ProductSet(SetObject):
    """Cartesian product of finite sets.

    Nested products are flattened: ``ProductSet(components=(A, ProductSet(components=(B, C))))``
    constructs to ``ProductSet`` with ``components == (A, B, C)``.
    """

    components: tuple[SetObject, ...] = dx.field(
        default=(), converter=_flatten_products
    )
    kind: Literal["product_set"] = "product_set"

    @property
    def size(self) -> int:
        return prod((c.size for c in self.components))

    @property
    def shape(self) -> tuple[int, ...]:
        result: list[int] = []
        for c in self.components:
            result.extend(c.shape)
        return tuple(result)

    def __str__(self) -> str:
        inner = " × ".join((str(c) for c in self.components))
        return f"({inner})"


def _flatten_coproducts(items: tuple["SetObject", ...]) -> tuple["SetObject", ...]:
    """Flatten nested CoproductSet so that C(A, C(B, C)) collapses to C(A, B, C)."""
    out: list[SetObject] = []
    for c in items:
        if isinstance(c, CoproductSet):
            out.extend(c.components)
        else:
            out.append(c)
    return tuple(out)


class CoproductSet(SetObject):
    """Tagged union (coproduct) of finite sets.

    The flat cardinality is the sum of component cardinalities; the shape is
    a single dimension of that total size with offsets recoverable from
    `offsets`.
    """

    components: tuple[SetObject, ...] = dx.field(
        default=(), converter=_flatten_coproducts
    )
    kind: Literal["coproduct_set"] = "coproduct_set"

    @dx.derived
    def offsets(self) -> tuple[int, ...]:
        """Per-component flat offsets, computed from cumulative sizes."""
        out: list[int] = []
        running = 0
        for c in self.components:
            out.append(running)
            running += c.size
        return tuple(out)

    @property
    def size(self) -> int:
        return sum((c.size for c in self.components))

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size,)

    def offset(self, index: int) -> int:
        """Return the flat offset for the i-th component."""
        return self.offsets[index]

    def component_range(self, index: int) -> tuple[int, int]:
        """Return ``(start, end)`` flat indices for the i-th component."""
        start = self.offsets[index]
        end = start + self.components[index].size
        return (start, end)

    def __str__(self) -> str:
        inner = " + ".join((str(c) for c in self.components))
        return f"({inner})"


def _build_free_monoid_components(
    generators: FinSet, max_length: int
) -> tuple[SetObject, ...]:
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0, got {max_length}")
    components: list[SetObject] = [Unit]
    for k in range(1, max_length + 1):
        if k == 1:
            components.append(generators)
        else:
            components.append(ProductSet(components=tuple([generators] * k)))
    return tuple(components)


class FreeMonoid(SetObject):
    """Free monoid on a generator set, truncated to ``max_length``.

    Represents all tuples (strings) from elements of ``generators`` with
    length 0 through ``max_length``. Internally a coproduct
    ``Unit + G + G×G + ... + G^max_length``; obtain it via
    `as_coproduct`.

    Attributes
    ----------
    generators : FinSet
        The generator set.
    max_length : int
        Maximum string length (inclusive).
    """

    generators: FinSet
    max_length: int
    kind: Literal["free_monoid"] = "free_monoid"

    @dx.derived
    def _coproduct(self) -> CoproductSet:
        return CoproductSet(
            components=_build_free_monoid_components(self.generators, self.max_length)
        )

    def as_coproduct(self) -> CoproductSet:
        """Return the underlying coproduct view."""
        return self._coproduct

    @property
    def components(self) -> tuple[SetObject, ...]:
        return self._coproduct.components

    @property
    def size(self) -> int:
        return self._coproduct.size

    @property
    def shape(self) -> tuple[int, ...]:
        return self._coproduct.shape

    def offset(self, index: int) -> int:
        return self._coproduct.offset(index)

    def component_range(self, index: int) -> tuple[int, int]:
        return self._coproduct.component_range(index)

    def encode(self, word: tuple[int, ...]) -> int:
        """Encode a word (tuple of generator indices) to a flat index."""
        k = len(word)
        if k > self.max_length:
            raise ValueError(f"word length {k} exceeds max_length {self.max_length}")
        g = self.generators.cardinality
        base = self.offset(k)
        if k == 0:
            return base
        idx = 0
        for w in word:
            if not 0 <= w < g:
                raise ValueError(f"generator index {w} out of range [0, {g})")
            idx = idx * g + w
        return base + idx

    def decode(self, flat_index: int) -> tuple[int, ...]:
        """Decode a flat index back to a word."""
        if not 0 <= flat_index < self.size:
            raise ValueError(f"flat_index {flat_index} out of range [0, {self.size})")
        g = self.generators.cardinality
        cop = self._coproduct
        for k in range(len(cop.components)):
            start, end = cop.component_range(k)
            if start <= flat_index < end:
                local = flat_index - start
                if k == 0:
                    return ()
                digits: list[int] = []
                for _ in range(k):
                    digits.append(local % g)
                    local //= g
                digits.reverse()
                return tuple(digits)
        raise RuntimeError("unreachable")

    def __str__(self) -> str:
        return (
            f"FreeMonoid(generators={self.generators!s}, max_length={self.max_length})"
        )


class EnumSet(SetObject):
    """A finite set whose elements have explicit names.

    Unlike `FinSet`, whose elements are anonymous integers
    ``0..cardinality-1``, an `EnumSet` carries a tuple of element
    names; the cardinality is ``len(elements)``. Names must be unique.

    Used for declaring atom collections (e.g. categorial-grammar atoms
    like ``{NP, S, VP}``) that other constructions reference by name.

    Attributes
    ----------
    name : str
        Human-readable name for the set.
    elements : tuple of str
        The element names. Must be unique and non-empty.
    """

    name: str
    elements: tuple[str, ...]
    kind: Literal["enum_set"] = "enum_set"

    __axioms__ = (
        dx.axiom(
            "length elements >= 1",
            message="EnumSet must have at least one element",
        ),
    )

    @property
    def cardinality(self) -> int:
        return len(self.elements)

    @property
    def size(self) -> int:
        return len(self.elements)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.elements),)

    def index(self, name: str) -> int:
        """Return the integer index of the element ``name``."""
        try:
            return self.elements.index(name)
        except ValueError as e:
            raise KeyError(
                f"{name!r} is not an element of EnumSet {self.name!r}"
            ) from e

    def __str__(self) -> str:
        inner = ", ".join(self.elements)
        return f"{self.name}{{{inner}}}"


class FreeResiduated(SetObject):
    """The finite set of Lambek-calculus types over a generator set.

    The elements are the formulas built from the generator atoms with
    the residuated connectives (``/``, ``\\``, ``*``, optional
    ``◇`` / ``□``) up to a bounded nesting depth, the types of the
    Lambek calculus [Lambek (1958)]. The set is a finite set of
    formulas, not a category, and it is not closed under the
    connectives. The runtime carrier is a finite enumeration of all
    admissible category strings produced by
    [`quivers.stochastic.categories.CategorySystem.from_generators`][quivers.stochastic.categories.CategorySystem.from_generators].

    Attributes
    ----------
    generators : EnumSet
        The atom generators (e.g. ``EnumSet(name='Atoms', elements=('NP', 'S', 'VP'))``).
    depth : int
        Maximum nesting depth of category constructors. ``depth=0``
        gives only atoms; each increment adds the formulas whose
        outermost connective joins two formulas of smaller depth.
    ops : tuple of str
        The connectives to close under. Allowed names:
        ``"slash"`` (both ``/`` and ``\\``), ``"product"`` (``*``),
        ``"unit"`` (the monoidal unit), ``"diamond"`` (``◇``),
        ``"box"`` (``□``).

    References
    ----------
    Lambek, J. (1958). The mathematics of sentence structure. *The
    American Mathematical Monthly*, 65(3), 154-170.
    [doi:10.1080/00029890.1958.11989160](https://doi.org/10.1080/00029890.1958.11989160)
    """

    generators: EnumSet
    depth: int = 1
    ops: tuple[str, ...] = ("slash",)
    kind: Literal["free_residuated"] = "free_residuated"

    __axioms__ = (
        dx.axiom("depth >= 0", message="FreeResiduated depth must be non-negative"),
    )

    def system(self) -> object:
        """Return the underlying `CategorySystem`.

        The system is constructed on each call (cheap for small depths;
        callers expecting to use it multiple times should hold a
        reference). Storing the result inside the model would require
        a non-frozen field, which is incompatible with the dx.Model
        contract.
        """
        from quivers.stochastic.categories import CategorySystem

        return CategorySystem.from_generators(
            list(self.generators.elements),
            constructors=list(self.ops),
            max_depth=self.depth,
        )

    @property
    def size(self) -> int:
        sys = self.system()
        # CategorySystem implements __len__ as the cardinality of its
        # finite category enumeration; fall through any AttributeError
        # in the property body would mask as "missing field" via the
        # dx.Model __getattr__ fallback, so coerce to a definite int.
        return int(len(sys))  # type: ignore[arg-type]

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size,)

    def __str__(self) -> str:
        ops = ", ".join(self.ops)
        return (
            f"FreeResiduated(generators={self.generators.name}, "
            f"depth={self.depth}, ops=[{ops}])"
        )


__all__ = [
    "SetObject",
    "FinSet",
    "Unit",
    "ProductSet",
    "CoproductSet",
    "FreeMonoid",
    "EnumSet",
    "FreeResiduated",
]
