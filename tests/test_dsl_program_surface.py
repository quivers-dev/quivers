"""End-to-end DSL coverage for the program-surface constructs:
typed program parameters (parametric templates), labeled return
tuples, score steps, and export declarations.

The canonical module under test is the gallery example
``docs/examples/source/parametric_pooling.qvr``; targeted inline
variants isolate the runtime contribution of each construct
(the bound scalar parameter, the score factor) by compiling two
modules that differ in exactly one token and comparing their
traces at identical intermediates.
"""

from __future__ import annotations


import textwrap
from pathlib import Path

import pytest
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import ProductSpace
from quivers.core.objects import FinSet
from quivers.dsl import Compiler, load, loads
from quivers.dsl.parser import parse_file
from quivers.program import Program


_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "examples"
    / "source"
    / "parametric_pooling.qvr"
)


_POOLING_SRC = """
object School : FinSet 8

program school_effects(spread : Real, K : FinSet) : K -> K
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect

program pooled : School -> School
    sample theta <- school_effects({spread}, School)
    sample sigma <- LogNormal(loc=0.0, scale=0.5)
    let total_effect = sum(theta)
    {score_line}
    observe y : School <- Normal(loc=theta, scale=sigma)
    return (effects: theta, scale: sigma)

export pooled
"""

_TEMPLATE_ONLY_SRC = """
object School : FinSet 8

program school_effects(spread : Real, K : FinSet) : K -> K
    sample z : K <- Normal(loc=0.0, scale=1.0)
    let effect = spread * z
    return effect

export school_effects
"""

_SCORE_LINE = "score centering = -50.0 * total_effect * total_effect"


def _build(spread: float, with_score: bool = True) -> MonadicProgram:
    """Compile the inline pooling module at the given spread."""
    score_line = _SCORE_LINE if with_score else "let centering_slot = 0.0"
    src = textwrap.dedent(_POOLING_SRC).format(
        spread=spread,
        score_line=score_line,
    )
    morph = loads(src).morphism
    assert isinstance(morph, MonadicProgram)
    return morph


def _intermediates(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """A trace for the pooling module's draw sites.

    ``theta`` is deliberately absent: ``log_joint`` recomputes it
    from ``z`` through the template's ``let`` arithmetic, so any
    density difference between two compiled variants at this trace
    is attributable to the bound ``spread`` value alone.
    """
    return {
        "theta$z": z,
        "sigma": torch.full((8, 1), 0.8),
        "y": torch.linspace(-1.0, 1.0, 8),
    }


def test_example_compiles_end_to_end() -> None:
    """The gallery example loads into a Program wrapping the exported
    MonadicProgram, typed at the arrow the source declares.

    `pooled : School -> Effect * Mean` reads its domain from the
    40-school plate and returns the pair of real values its labeled
    tuple names, so the codomain is that product rather than the
    domain repeated.
    """
    prog = load(str(_EXAMPLE))
    assert isinstance(prog, Program)
    morph = prog.morphism
    assert isinstance(morph, MonadicProgram)
    assert isinstance(morph.domain, FinSet)
    assert morph.domain.cardinality == 40
    assert isinstance(morph.codomain, ProductSpace)
    assert [c.dim for c in morph.codomain.components] == [1, 1]


def test_template_scalar_parameter_changes_forward_pass() -> None:
    """Instantiating the template at two different scalar arguments
    scales the forward-sampled effects by exactly the ratio of the
    bound values (the non-centered ``let`` multiplies the same
    standardized draw)."""
    tight = _build(0.6)
    loose = _build(2.5)
    # The inlined template body binds its local under the call-site
    # name: the standardized draw surfaces as ``theta$z``.
    assert "theta$z" in repr(tight)
    x = torch.zeros(8, 1)
    torch.manual_seed(7)
    out_tight = tight.rsample(x)
    torch.manual_seed(7)
    out_loose = loose.rsample(x)
    assert isinstance(out_tight, dict)
    assert isinstance(out_loose, dict)
    effects_tight = out_tight["effects"]
    effects_loose = out_loose["effects"]
    assert not torch.allclose(effects_tight, effects_loose)
    assert torch.allclose(
        effects_loose,
        effects_tight * (2.5 / 0.6),
        atol=1e-5,
    )


def test_template_scalar_parameter_changes_log_joint() -> None:
    """At identical intermediates, the two spreads yield different
    joint densities: the bound scalar moves both the observe
    likelihood (through ``theta = spread * z``) and the score
    factor."""
    torch.manual_seed(0)
    z = torch.randn(8)
    x = torch.zeros(8, 1)
    lj_tight = _build(0.6).log_joint(x, _intermediates(z))
    lj_loose = _build(2.5).log_joint(x, _intermediates(z))
    assert lj_tight.shape == (8,)
    assert torch.isfinite(lj_tight).all()
    assert torch.isfinite(lj_loose).all()
    assert not torch.allclose(lj_tight, lj_loose)


def test_labeled_return_tuple_keys_program_output() -> None:
    """``return (effects: theta, mean: mu)`` keys the compiled
    program's output dict by the labels, and ``log_joint`` accepts
    label-keyed intermediates interchangeably with variable-keyed
    ones."""
    prog = load(str(_EXAMPLE))
    model = prog.morphism
    assert isinstance(model, MonadicProgram)
    x = torch.zeros(1, 1)
    se = torch.full((40,), 10.0)
    torch.manual_seed(0)
    z = torch.randn(40)
    tau = torch.tensor(3.0)
    theta = tau * z
    mu = torch.tensor(1.5)
    out = model.rsample(x, observations={"se": se, "mu": mu, "tau": tau, "theta$z": z})
    assert isinstance(out, dict)
    assert set(out.keys()) == {"effects", "mean"}
    assert torch.allclose(out["effects"], theta)
    assert torch.allclose(out["mean"], mu)

    y = torch.linspace(-1.0, 1.0, 40)
    shared = {"tau": tau, "theta$z": z, "y": y, "se": se}
    by_label = {"effects": theta, "mean": mu, **shared}
    by_var = {"theta": theta, "mu": mu, **shared}
    assert torch.allclose(model.log_joint(x, by_label), model.log_joint(x, by_var))


def test_score_step_moves_log_joint() -> None:
    """The score step contributes exactly its expression's value to
    the joint density: removing the line shifts ``log_joint`` by
    the soft sum-to-zero penalty and nothing else."""
    torch.manual_seed(0)
    z = torch.randn(8)
    x = torch.zeros(8, 1)
    spread = 0.6
    scored = _build(spread, with_score=True)
    unscored = _build(spread, with_score=False)
    lj_scored = scored.log_joint(x, _intermediates(z))
    lj_unscored = unscored.log_joint(x, _intermediates(z))
    total_effect = (spread * z).sum()
    penalty = -50.0 * total_effect * total_effect
    assert not torch.allclose(lj_scored, lj_unscored)
    assert torch.allclose(lj_scored - lj_unscored, penalty.expand(8), atol=1e-4)


def test_export_selects_declared_morphism() -> None:
    """``export pooled`` picks that program among the module's
    candidates: the compiled Program wraps it identically (object
    identity), not a sibling host program or the template."""
    ast = parse_file(str(_EXAMPLE))
    compiler = Compiler(ast)
    prog = compiler.compile()
    morphisms = compiler.morphisms
    assert prog.morphism is morphisms["pooled"]
    for sibling in ("pooled_tight", "pooled_loose"):
        assert sibling in morphisms
        assert prog.morphism is not morphisms[sibling]
    # The parametric template is registered as a template, never as
    # a concrete morphism.
    assert "school_effects" in compiler.programs
    assert "school_effects" not in morphisms


def test_uninstantiated_template_domain_raises_typed_error() -> None:
    """A module whose export names a parametric template compiles to
    a morphism-less Program; ``.domain`` / ``.codomain`` raise a
    TypeError naming the template and how to instantiate it."""
    prog = loads(textwrap.dedent(_TEMPLATE_ONLY_SRC))
    assert prog.morphism is None
    assert set(prog.templates) == {"school_effects"}
    with pytest.raises(TypeError, match="school_effects"):
        _ = prog.domain
    with pytest.raises(TypeError, match="Instantiate the template"):
        _ = prog.domain
    with pytest.raises(TypeError, match="codomain"):
        _ = prog.codomain
    # Instantiating the template at concrete arguments resolves it.
    inst = prog.templates["school_effects"](0.6, "School")
    assert isinstance(inst, Program)
    assert isinstance(inst.domain, FinSet)
    assert inst.domain.cardinality == 8
    assert isinstance(inst.morphism, MonadicProgram)


def test_structural_container_domain_raises_typed_error() -> None:
    """A morphism-less Program with no templates (a structural
    artifact container) raises the plain typed error."""
    container = Program()
    with pytest.raises(TypeError, match="no exported morphism"):
        _ = container.domain
    with pytest.raises(TypeError, match="no exported morphism"):
        _ = container.codomain
