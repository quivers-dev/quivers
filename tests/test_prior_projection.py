"""Prior projections preserve lexical dependencies without executing likelihoods."""

import pytest
import torch

from quivers.continuous.program_steps import Call, Draw, Let, Scan, Score, reading
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean, ProductSpace
from quivers.dsl import loads
from quivers.inference._prior_projection import project_prior_sites
from quivers.inference.trace import trace


SOURCE = """object Out:Real 1
program model(h:Real):Out->Out
    sample scale <- LogNormal(loc=0.0,scale=0.3)
    let loc=scale+1.0
    sample keep <- Normal(loc=loc,scale=scale)
    let invalid=log_probability(h)
    observe y <- Normal(loc=invalid,scale=1.0)
    return y
export model
"""


def test_projection_uses_original_conditional_law_and_drops_downstream_partial_calls():
    program = loads(SOURCE).morphism
    projected = project_prior_sites(program, {"keep"})
    assert [
        step.names[0] if isinstance(step, Draw) else step.name
        for step in projected.steps[:-1]
    ] == ["scale", "loc", "keep"]
    assert projected.input_contract is program.input_contract
    assert "h" not in projected.host_data_types
    assert projected.step_module(projected.steps[0]) is program.step_module(
        program.steps[0]
    )
    recorded = trace(
        projected,
        torch.zeros(1, 1),
        {
            "scale": torch.tensor(0.7),
            "keep": torch.tensor(1.3),
            "h": torch.tensor(-1.0),
        },
    )
    expected = torch.distributions.LogNormal(0.0, 0.3).log_prob(torch.tensor(0.7))
    expected += torch.distributions.Normal(1.7, 0.7).log_prob(torch.tensor(1.3))
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    with pytest.raises(ValueError, match="unknown prior sites"):
        project_prior_sites(program, {"missing"})


def test_projection_keeps_explicit_potentials_and_their_dependencies():
    program = loads(SOURCE).morphism
    potential = Score(
        name="potential", score=reading(lambda env: -env["keep"].square(), {"keep"})
    )
    model = MonadicProgram(
        program.domain,
        program.codomain,
        (*program.steps[:3], potential, *program.steps[3:]),
        ("y",),
        params=program._params,
    )
    projected = project_prior_sites(model, {"scale"})
    assert any(
        isinstance(step, Score) and step.name == "potential" for step in projected.steps
    )
    recorded = trace(
        projected,
        torch.zeros(1, 1),
        {"scale": torch.tensor(0.7), "keep": torch.tensor(1.3)},
    )
    assert recorded.sites["potential"].value.item() == pytest.approx(-(1.3**2))


def test_projection_keeps_opaque_callable_dependencies_conservatively():
    program = loads(SOURCE).morphism
    opaque = Let(name="opaque", value=lambda env: env["loc"])
    keep = program.steps[2].with_(args=("opaque", "scale"))
    model = MonadicProgram(
        program.domain,
        program.codomain,
        (*program.steps[:2], opaque, keep),
        ("keep",),
        params=program._params,
    )
    projected = project_prior_sites(model, {"keep"})
    assert [step.name for step in projected.steps if isinstance(step, Let)][:2] == [
        "loc",
        "opaque",
    ]
    assert torch.isfinite(trace(projected, torch.zeros(1, 1)).log_joint).all()


def test_projection_descends_calls_without_executing_unused_callee_outputs():
    program = loads(SOURCE).morphism
    caller = MonadicProgram(
        program.domain,
        program.codomain,
        (Call(name="answer", program=program, args=("provided",)),),
        ("answer",),
        params=None,
    )
    projected = project_prior_sites(caller, {"keep"})
    recorded = trace(
        projected,
        torch.zeros(1, 1),
        {"h": torch.tensor(-1.0), "provided": torch.zeros(1, 1)},
    )
    assert "keep" in recorded.sites and "y" not in recorded.sites


def test_projection_preserves_scan_feedback_and_original_occurrence_addresses():
    real = Euclidean(name="R", dim=1)
    normal = (
        loads(
            "object R:Real 1\nprogram p:R->R\n    sample e <- Normal(loc=0.0,scale=1.0)\n    return e\nexport p\n"
        )
        .morphism.steps[0]
        .morphism
    )
    cell = MonadicProgram(
        ProductSpace(components=(real, real)),
        real,
        (
            Draw(names=("noise",), morphism=normal, args=("entry",)),
            Let(
                name="next",
                value=reading(
                    lambda env: env["entry"] + env["state"] + env["noise"],
                    {"entry", "state", "noise"},
                ),
            ),
            Let(
                name="unreachable",
                value=reading(
                    lambda env: (_ for _ in ()).throw(AssertionError("downstream")), ()
                ),
            ),
        ),
        ("next",),
        params=("entry", "state"),
    )
    model = MonadicProgram(
        real,
        real,
        (Scan(name="trajectory", step=cell, source=None, initial=torch.zeros(1)),),
        ("trajectory",),
        params=None,
    )
    projected = project_prior_sites(model, {"noise@1"})
    observed = {f"noise@{i}": torch.tensor(0.25) for i in range(3)}
    recorded = trace(projected, torch.tensor([[[1.0], [2.0], [3.0]]]), observed)
    assert {f"noise@{i}" for i in range(3)} <= set(recorded.sites)
    assert "unreachable@0" not in recorded.sites
    assert recorded.sites["next@2"].value.item() == pytest.approx(6.75)
