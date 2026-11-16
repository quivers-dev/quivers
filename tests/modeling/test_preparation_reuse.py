"""Source preparation reuses immutable evidence without crossing rewrites."""

from unittest.mock import patch

import torch

from quivers.dsl import Compiler, module_to_source, parse
from quivers.dsl._data_inference import infer_data_objects
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.modeling import Model


SOURCE = """object Row:FinSet _
object Out:Real 1
program p[n:Nat](x:Tensor[Real]([Row]),z:Tensor[Real]([n]),y:Tensor[Real]([Row])):Row->Out
    sample b <- Normal(loc=0.0,scale=1.0)
    let mu = b*x
    observe y:Row <- Normal(loc=mu,scale=1.0)
    return b
export p
"""


def test_unchanged_source_keeps_its_checked_contract_across_bindings():
    source = SOURCE.replace("FinSet _", "FinSet 3")
    # A fully closed interface needs no source specialization.
    source = source.replace("program p[n:Nat]", "program p").replace("([n])", "([2])")
    with (
        patch(
            "quivers.modeling.model.lower_qvr_to_qiec", wraps=lower_qvr_to_qiec
        ) as initial,
        patch(
            "quivers.dsl.compiler.core.lower_qvr_to_qiec", wraps=lower_qvr_to_qiec
        ) as compiled,
    ):
        specification = Model(source)
        for _ in range(2):
            bound = specification.bind(
                {"x": torch.ones(3), "z": torch.ones(2), "y": torch.zeros(3)}
            )
            assert bound.schema.entry == specification.schema.entry
        assert initial.call_count == 1
        assert compiled.call_count == 0


def test_combined_extents_recheck_rewrites_and_keep_density_and_gradient():
    specification = Model(SOURCE)
    for rows, width in [(3, 2), (5, 4)]:
        data = {
            "x": torch.arange(1, rows + 1, dtype=torch.float64),
            "z": torch.ones(width),
            "y": torch.zeros(rows, dtype=torch.float64),
        }
        with (
            patch(
                "quivers.dsl.qiec_lowering.lower_qvr_to_qiec", wraps=lower_qvr_to_qiec
            ) as inferred,
            patch(
                "quivers.dsl.compiler.core.lower_qvr_to_qiec", wraps=lower_qvr_to_qiec
            ) as compiled,
        ):
            bound = specification.bind(data)
        assert inferred.call_count == 1  # scoped binders after global objects changed
        assert compiled.call_count == 1  # final specialized contract
        assert inferred.call_args.args[0] is not specification.syntax
        assert compiled.call_args.args[0] is not inferred.call_args.args[0]
        assert bound.schema.describe()["z"]["type"] == f"Tensor[Real]([{width}])"
        assert bound.schema.describe()["x"]["type"] == f"Tensor[Real]([{rows}])"
        b = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
        actual = bound.trace({"b": b}).log_joint.sum()
        expected = (
            torch.distributions.Normal(0.0, 1.0).log_prob(b).sum()
            + torch.distributions.Normal(b * data["x"], 1.0).log_prob(data["y"]).sum()
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, b)[0], torch.autograd.grad(expected, b)[0]
        )


def test_equal_but_distinct_source_cannot_reuse_checked_evidence():
    original = parse(SOURCE)
    checked = lower_qvr_to_qiec(original)
    replacement = parse(module_to_source(original))
    compiler = Compiler(replacement)
    assert not compiler._reuse_checked_source(original, checked)
    with patch(
        "quivers.dsl.qiec_lowering.lower_qvr_to_qiec", wraps=lower_qvr_to_qiec
    ) as lowered:
        infer_data_objects(
            replacement,
            {"x": torch.ones(3), "z": torch.ones(2), "y": torch.zeros(3)},
            checked_source=(original, checked),
        )
    assert lowered.call_count == 2


def test_replacing_model_syntax_invalidates_original_evidence():
    specification = Model(SOURCE.replace("FinSet _", "FinSet 3"))
    specification.syntax = parse(SOURCE.replace("FinSet _", "FinSet 4"))
    bound = specification.bind(
        {"x": torch.ones(4), "z": torch.ones(2), "y": torch.zeros(4)}
    )
    assert bound.schema.describe()["x"]["type"] == "Tensor[Real]([4])"
