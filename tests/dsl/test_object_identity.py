"""Declared finite sets are distinct objects, as in the Python API.

``object B : FinSet 3`` and ``object C : FinSet 3`` declare two objects
of the same size. A morphism into ``B`` does not compose with one out of
``C``, exactly as `FinSet(name="B", cardinality=3)` and
`FinSet(name="C", cardinality=3)` refuse to compose in Python. An alias
of an object (``object B2 : B``) is that object, and the same object
composes with itself.
"""

from __future__ import annotations

import pytest

from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet
from quivers.dsl import CompileError, loads

_DECLS = """
object A : FinSet 2
object B : FinSet 3
object C : FinSet 3
object D : FinSet 2
object B2 : B
morphism f : A -> B
morphism g : C -> D
morphism k : B -> D
morphism m : B2 -> D
"""


def test_distinct_objects_of_one_size_do_not_compose() -> None:
    """``f : A -> B`` and ``g : C -> D`` with ``|B| = |C|`` are refused."""
    with pytest.raises(CompileError, match="cannot compose"):
        loads(_DECLS + "define h = f >> g\nexport h\n")


def test_python_api_refuses_the_same_composite() -> None:
    """The Python API refuses it with the same reason, which the DSL now matches."""
    f = morphism(FinSet(name="A", cardinality=2), FinSet(name="B", cardinality=3))
    g = morphism(FinSet(name="C", cardinality=3), FinSet(name="D", cardinality=2))
    with pytest.raises(TypeError, match="cannot compose"):
        f >> g


def test_declared_object_keeps_its_name() -> None:
    """A finite set declared by its size is named by its declaration."""
    program = loads(_DECLS + "define h = f >> k\nexport h\n")
    assert program.morphism.domain == FinSet(name="A", cardinality=2)
    assert program.morphism.codomain == FinSet(name="D", cardinality=2)


def test_an_alias_is_the_same_object() -> None:
    """``object B2 : B`` names ``B`` itself, so ``f >> m`` composes."""
    loads(_DECLS + "define h = f >> m\nexport h\n")
