"""Backend-agnostic morphism → ``nn.Module`` adapter
(:func:`quivers.core.morphisms.as_torch_module`).

A :class:`Morphism` from :mod:`quivers.core.morphisms` is a
categorical object; an :class:`nn.Module` is a PyTorch parameter
container. Every binding site that needs to register a morphism's
parameters with a parent module (:class:`MonadicProgram` step
list, fan / stack composites, parametric-program parameter slots)
funnels through the adapter, which:

* Passes through a value that is already an :class:`nn.Module`.
* Wraps a V-Cat :class:`Morphism` by calling its ``.module()``
  method and attaches the original morphism on the wrapper under
  the synthetic attribute ``_morphism`` for downstream recovery.
* Rejects anything else with a typed error.

This module verifies the adapter's contract plus the end-to-end
behavior of binding sites it unblocks:

* A ``program`` block referencing a let-composed chain (``out <- h``
  where ``h = f >> g``) compiles and runs through ``rsample`` to
  produce a finite output tensor.
* A parametric program with a morphism-typed parameter
  (``program p (k : Mor[A, B])``) compiles and instantiates.
* ``fan(...)`` over let-composed morphisms accepts non-Module
  morphisms at the binding site.
"""

from __future__ import annotations
import textwrap

import pytest
import torch

from quivers.core.morphisms import (
    as_torch_module,
    extract_morphism,
)


# ---------------------------------------------------------------------------
# Adapter unit tests
# ---------------------------------------------------------------------------


def test_as_torch_module_passes_through_nn_module() -> None:
    """An :class:`nn.Module` is returned unchanged so existing
    ContinuousMorphism bindings don't pay an extra wrapping cost."""
    import torch.nn as nn

    m = nn.Linear(3, 3)
    wrapped = as_torch_module(m)
    assert wrapped is m


def test_as_torch_module_wraps_latent_morphism() -> None:
    """A core V-Cat :class:`LatentMorphism` is wrapped in an
    :class:`nn.Module` whose ``parameters()`` exposes the
    morphism's learnable tensor."""
    from quivers.core.morphisms import LatentMorphism
    from quivers.core.objects import FinSet

    A = FinSet(name="A", cardinality=4)
    B = FinSet(name="B", cardinality=4)
    m = LatentMorphism(A, B)
    wrapped = as_torch_module(m)
    params = list(wrapped.parameters())
    assert len(params) == 1
    assert params[0].shape == (4, 4)


def test_as_torch_module_attaches_morphism_for_recovery() -> None:
    """The wrapper exposes the original morphism via
    :func:`extract_morphism` so the runtime can apply the
    categorical operation on the V-Cat side without rebuilding
    from the wrapper's parameters."""
    from quivers.core.morphisms import LatentMorphism
    from quivers.core.objects import FinSet

    A = FinSet(name="A", cardinality=4)
    B = FinSet(name="B", cardinality=4)
    m = LatentMorphism(A, B)
    wrapped = as_torch_module(m)
    recovered = extract_morphism(wrapped)
    assert recovered is m


def test_as_torch_module_rejects_non_morphism_non_module() -> None:
    """The adapter raises a type-specific error for an input that is
    neither an :class:`nn.Module` nor a :class:`Morphism`."""
    with pytest.raises(TypeError, match="cannot adapt"):
        as_torch_module("not a morphism")


def test_extract_morphism_returns_none_for_plain_module() -> None:
    """Modules that were never wrapped through ``as_torch_module``
    return ``None`` from ``extract_morphism``."""
    import torch.nn as nn

    m = nn.Linear(2, 2)
    assert extract_morphism(m) is None


def test_as_torch_module_wraps_composed_morphism() -> None:
    """A V-Cat :class:`ComposedMorphism` (``f >> g``) gets wrapped
    in a Module whose parameter set is the union of f's and g's
    parameters."""
    from quivers.core.morphisms import LatentMorphism
    from quivers.core.objects import FinSet

    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B)
    g = LatentMorphism(B, C)
    composed = f >> g
    wrapped = as_torch_module(composed)
    # ComposedMorphism's module exposes both f.module and g.module
    # as submodules; together they expose two learnable parameters.
    n_params = sum(1 for _ in wrapped.parameters())
    assert n_params == 2


# ---------------------------------------------------------------------------
# Repro 1: program binding a let-composed chain
# ---------------------------------------------------------------------------


def test_program_binding_let_composed_chain_compiles_and_runs() -> None:
    """The issue's first repro: a ``program`` block whose body
    references a let-composed chain ``h = f >> g``."""
    from quivers.dsl import loads

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 4
    object B : FinSet 4
    object C : FinSet 4

    morphism f : A -> B [role=parameter]
    morphism g : B -> C [role=parameter]

    define h = f >> g

    program p : A -> C
        sample out <- h
        return out

    export p
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None
    # End-to-end run: input is a long tensor of A-indices; the
    # program's V-Cat step applies h.tensor[idx] to produce a
    # batch-shape × C-shape output.
    x = torch.tensor([0, 1, 2, 3])
    out = m.morphism.rsample(x)
    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()


def test_program_binding_let_composed_chain_has_learnable_params() -> None:
    """The wrapped composed morphism still exposes its learnable
    parameters to the outer program's optimizer."""
    from quivers.dsl import loads

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 4
    object B : FinSet 4
    object C : FinSet 4

    morphism f : A -> B [role=parameter]
    morphism g : B -> C [role=parameter]

    define h = f >> g

    program p : A -> C
        sample out <- h
        return out

    export p
    """
    m = loads(textwrap.dedent(src))
    n_params = sum(1 for _ in m.morphism.parameters())
    # f and g each contribute one parameter tensor.
    assert n_params == 2


# ---------------------------------------------------------------------------
# Repro 2: parametric program with morphism-typed parameter
# ---------------------------------------------------------------------------


def test_parametric_program_with_morphism_typed_parameter_compiles() -> None:
    """A parametric program with ``k : Mor[A, B]`` parameter
    accepts a morphism at instantiation; the template body's
    ``<- k`` binds through the as_torch_module adapter without
    crashing."""
    from quivers.dsl import loads

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 4
    object B : FinSet 4

    morphism f : A -> B [role=parameter]

    program p(k : Mor[A, B]) : A -> B
        sample out <- k
        return out

    define applied = p(f)
    export applied
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None


# ---------------------------------------------------------------------------
# Repro 3: fan over let-composed morphisms (regression coverage)
# ---------------------------------------------------------------------------


def test_fan_over_composed_morphism_compiles() -> None:
    """``fan(f >> g, ...)`` must accept composed-let morphisms
    without crashing at the binding site."""
    from quivers.dsl import loads

    src = """
    composition product_fuzzy [level=algebra]
    object A : FinSet 4
    object B : FinSet 4
    object C : FinSet 4

    morphism f : A -> B [role=parameter]
    morphism g : B -> C [role=parameter]

    define chain = f >> g
    define split = fan(chain, chain)
    export split
    """
    m = loads(textwrap.dedent(src))
    assert m.morphism is not None
