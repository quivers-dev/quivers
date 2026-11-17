"""Authored tensor axes survive closed-cardinality lowering and serialization."""

import torch
import pytest

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.serialization import dumps, loads
from quivers.modeling import Model, ModelFit
from quivers.inference import PriorDraws
from quivers.diagnostics import prediction_grid

SOURCE = """object Row:FinSet 3
object Other:FinSet 3
object Out:Real 1
program p(x:Tensor[Real]([Row]),w:Tensor[Real]([Other]),y:Tensor[Real]([Row])):Row->Out
    sample b<-Normal(loc=0.0,scale=1.0)
    let shift=sum(w)
    let mu=b*x+shift
    observe y:Row<-Normal(loc=mu,scale=1.0)
    return b
export p
"""


def test_closed_named_header_axes_preserve_nominal_identity_and_wire_roundtrip():
    module = lower_qvr_to_qiec(parse(SOURCE))
    entry = module.entries[0]
    assert {p.name: p.axes for p in entry.parameters} == {
        "x": ("Row",),
        "w": ("Other",),
        "y": ("Row",),
    }
    assert loads(dumps(entry)) == entry
    assert {p.name: p.axes for p in loads(dumps(entry)).parameters} == {
        "x": ("Row",),
        "w": ("Other",),
        "y": ("Row",),
    }


def test_equal_closed_extents_do_not_align_grid_columns_or_replace_globals():
    bound = Model(SOURCE).bind(
        {"x": torch.ones(3), "y": torch.zeros(3), "w": torch.tensor([10.0, 20.0, 30.0])}
    )
    samples = {"b": torch.tensor([[0.0, 1.0, 2.0]])}
    fitted = ModelFit(model=bound, posterior=PriorDraws(samples=samples, num_samples=3))
    result = fitted.conditional_effects(
        prediction_grid({"x": [0.0, 1.0, 2.0]}, conditions={})
    )
    expected = (
        60
        + samples["b"][:, :, None, None]
        * torch.tensor([0.0, 1.0, 2.0])[None, None, :, None]
    )
    torch.testing.assert_close(result.draws, expected)
    with pytest.raises(ValueError, match="does not use.*observation-row axis"):
        fitted.conditional_effects(
            prediction_grid({"w": [0.0, 1.0, 2.0]}, conditions={"x": [1.0]})
        )
    with pytest.raises((ValueError, TypeError), match="shape|extent|size|dimension"):
        fitted.conditional_effects(prediction_grid({"x": [0.0, 1.0]}, conditions={}))


def test_multi_axis_headers_keep_each_named_closed_axis_in_authored_order():
    source = SOURCE.replace(
        "x:Tensor[Real]([Row])", "x:Tensor[Real]([Other,Row])"
    ).replace("let mu=b*x+shift", "let mu=shift")
    entry = lower_qvr_to_qiec(parse(source)).entries[0]
    assert next(p for p in entry.parameters if p.name == "x").axes == ("Other", "Row")


def test_unnamed_closed_axes_require_unique_existing_extent_proof():
    ambiguous = SOURCE.replace("x:Tensor[Real]([Row])", "x:Tensor[Real]([3])")
    entry = lower_qvr_to_qiec(parse(ambiguous)).entries[0]
    assert next(p for p in entry.parameters if p.name == "x").axes == ("axis0",)
    unique = ambiguous.replace("object Other:FinSet 3", "object Other:FinSet 4")
    entry = lower_qvr_to_qiec(parse(unique)).entries[0]
    assert next(p for p in entry.parameters if p.name == "x").axes == ("Row",)


def test_lsp_checked_contract_preserves_closed_nominal_axes_and_authored_hover():
    from quivers.lsp.document import DocumentState
    from quivers.lsp.server import _render_hover

    document = DocumentState(uri="file:///tmp/nominal-axes.qvr")
    document.update(source=SOURCE, version=1)
    assert not document.diagnostics
    assert document.qiec_module is not None
    assert {p.name: p.axes for p in document.qiec_module.entries[0].parameters} == {
        "x": ("Row",),
        "w": ("Other",),
        "y": ("Row",),
    }
    hover = _render_hover(document, "w", line=5, col=20)
    assert hover is not None and "Tensor[Real]([Other])" in hover
