"""Typed structural tables drive the same scalar hazard in both runtimes."""

import pytest
import torch

from quivers import FinSet
from quivers.continuous import BernsteinHazard, ConditionalBernsteinHazard, Euclidean
from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.modeling import Model
from quivers.qiec.program_runtime import run_program


SOURCE = """object Out:Real 1
program survival[n:Nat,s:Nat,k:Nat](
    breaks:Tensor[Real]([nat_add(s,1)]),
    controls:Tensor[Real]([s,k]),
    eta:Tensor[Real]([n]),
    t:Tensor[Real]([n]),
):n->Out
    observe t:n <- BernsteinHazard(breaks=breaks,controls=controls,log_rate=eta)
    return t
export survival
"""


def _data():
    return {
        "breaks": torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64),
        "controls": torch.tensor(
            [[0.4, 0.6, 0.8], [0.8, 0.2, 0.4]], dtype=torch.float64
        )
        / (0.5 * 0.6 + 1.5 * (1.4 / 3)),
        "eta": torch.tensor([-0.4, 0.2, 1.0, 0.7, -0.2], dtype=torch.float64),
        "t": torch.tensor([0.2, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64),
    }


def test_source_binding_infers_structural_extents_and_matches_reference_and_oracle():
    model = Model(SOURCE)
    data = _data()
    bound = model.bind(data)
    actual = bound.trace({}).log_joint.sum()
    native = BernsteinHazard(data["breaks"], data["controls"], log_rate=data["eta"])
    torch.testing.assert_close(actual, native.log_prob(data["t"]).sum())
    host = {
        name: tuple(
            tuple(row) if isinstance(row, list) else row for row in value.tolist()
        )
        for name, value in data.items()
    }
    expected = run_program(
        model.checked_module, "survival", data=host, sites={}
    ).log_joint
    assert actual.item() == pytest.approx(expected, abs=2e-7)
    emitted = module_to_source(parse(SOURCE))
    Compiler(parse(emitted)).qiec_module
    torch.testing.assert_close(
        Model(emitted).bind(data).trace({}).log_joint, bound.trace({}).log_joint
    )


def test_all_literal_structural_tables_and_rate_schema_are_supported():
    source = """object Row:FinSet 3
object Out:Real 1
program survival(t:Tensor[Real]([Row])):Row->Out
    observe t:Row <- BernsteinHazard(breaks=[0.0,1.0],controls=[[0.5,1.5]],rate=1.2)
    return t
export survival
"""
    times = torch.tensor([0.2, 0.7, 2.0], dtype=torch.float64)
    expected = BernsteinHazard([0.0, 1.0], [[0.5, 1.5]], rate=1.2).log_prob(times).sum()
    actual = Model(source).bind({"t": times}).trace({}).log_joint.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("shape,valid", [("[2,4]", True), ("[3,3]", False)])
def test_control_width_is_free_but_segment_count_must_match_boundaries(shape, valid):
    source = (
        SOURCE.replace("[n:Nat,s:Nat,k:Nat]", "[n:Nat]")
        .replace("nat_add(s,1)", "3")
        .replace("[s,k]", shape)
    )
    if valid:
        Compiler(parse(source)).qiec_module
    else:
        with pytest.raises(QiecDiagnosticError, match="extent"):
            Compiler(parse(source)).qiec_module


def test_structural_scalar_broadcast_is_refused():
    with pytest.raises(QiecDiagnosticError, match="rank|tensor|Tensor|shape|parameter"):
        Compiler(
            parse(SOURCE.replace("controls:Tensor[Real]([s,k])", "controls:Real"))
        ).qiec_module


def test_rate_only_conditional_adapter_copies_fixed_baseline_and_keeps_gradients():
    breaks = torch.tensor([0.0, 1.0], dtype=torch.float64)
    controls = torch.tensor([[0.5, 1.5]], dtype=torch.float64)
    adapter = ConditionalBernsteinHazard(
        FinSet(name="Context", cardinality=3),
        Euclidean(name="Time", dim=1),
        breaks=breaks,
        controls=controls,
    )
    controls.fill_(100.0)
    x = torch.tensor([0, 1, 2])
    value = torch.tensor([[0.2], [0.7], [2.0]], dtype=torch.float64)
    score = adapter.log_prob(x, value).sum()
    gradients = torch.autograd.grad(score, tuple(adapter.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert adapter.rsample(x, (4,)).shape == (4, 3, 1)
    assert adapter.support.check(value).all()
    with pytest.raises(ValueError, match="explicit fixed"):
        ConditionalBernsteinHazard(
            FinSet(name="Context", cardinality=3), Euclidean(name="Time", dim=1)
        )
