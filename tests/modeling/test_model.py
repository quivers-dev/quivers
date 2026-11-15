"""The model interface derives a single checked contract from QVR."""

import copy

import pytest
import torch

from quivers.dsl import CompileError, loads, module_to_source, parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.inference import MCMC, NUTSKernel
from quivers.modeling import MCMCMethod, Model, VariationalMethod

SOURCE = """object Row : FinSet _
object Out : Real 1
program regression : Row -> Out
    data x : Tensor[Real]([Row])
    data y : Tensor[Real]([Row])
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""


def test_schema_is_available_before_binding_and_preserves_types():
    model = Model(SOURCE)
    assert model.schema.describe() == {
        "x": {"role": "data", "type": "Tensor[Real]([Row])"},
        "y": {"role": "observation", "type": "Tensor[Real]([Row])"},
    }
    assert copy.copy(model.schema) == model.schema
    assert model.schema.entry.parameters[0].type == model.schema.types["x"]


def test_declarations_roundtrip():
    emitted = module_to_source(parse(SOURCE))
    assert module_to_source(parse(emitted)) == emitted
    assert "data x : Tensor[Real]([Row])" in emitted


def test_inferred_size_matches_explicit_program_and_analytic_joint():
    data = {"x": torch.tensor([1.0, 2.0, 3.0]), "y": torch.tensor([0.5, -0.5, 1.0])}
    bound = Model(SOURCE).bind(data)
    exact = Model(SOURCE.replace("FinSet _", "FinSet 3")).bind(data)
    b = torch.tensor([0.25], requires_grad=True)
    actual = bound.trace({"b": b}).log_joint
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(b)
        + torch.distributions.Normal(b * data["x"], 1.0).log_prob(data["y"]).sum()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, exact.trace({"b": b}).log_joint)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), b)[0],
        torch.autograd.grad(expected.sum(), b)[0],
    )


@pytest.mark.parametrize("count", [1, 4, 9])
def test_model_rebinds_independently(count):
    model = Model(SOURCE)
    bound = model.bind({"x": torch.ones(count), "y": torch.zeros(count)})
    assert bound.schema.describe()["x"]["type"] == f"Tensor[Real]([{count}])"
    assert model.schema.describe()["x"]["type"] == "Tensor[Real]([Row])"


def test_conflicting_size_evidence_fails_before_fit():
    with pytest.raises(CompileError, match="cannot infer.*Row"):
        Model(SOURCE).bind({"x": torch.ones(3), "y": torch.ones(4)})


def test_missing_size_evidence_requests_explicit_size():
    with pytest.raises(CompileError, match="explicit size"):
        loads(SOURCE)


@pytest.mark.parametrize(
    "bad", [torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.bool)]
)
def test_dtype_mismatch_is_not_coerced(bad):
    with pytest.raises(TypeError, match="dtype"):
        Model(SOURCE).bind({"x": bad, "y": torch.zeros(3)})


def test_explicit_size_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        Model(SOURCE.replace("FinSet _", "FinSet 3")).bind(
            {"x": torch.ones(4), "y": torch.zeros(3)}
        )


def test_missing_and_extra_inputs_fail():
    model = Model(SOURCE.replace("FinSet _", "FinSet 3"))
    with pytest.raises(ValueError, match="missing external inputs.*x"):
        model.bind({"y": torch.zeros(3)})
    with pytest.raises(ValueError, match="unknown external inputs.*typo"):
        model.bind({"x": torch.ones(3), "y": torch.zeros(3), "typo": 1})


def test_implicit_data_uses_same_schema_without_separate_name_list():
    model = Model(
        SOURCE.replace("    data x : Tensor[Real]([Row])\n", "").replace(
            "    data y : Tensor[Real]([Row])\n", ""
        )
    )
    bound = model.bind({"x": torch.ones(3), "y": torch.zeros(3)})
    assert set(bound.schema.types) == {"x", "y"}
    assert bound.schema.observations == ("y",)


def test_ambiguous_reduction_requires_declaration():
    source = """object One : FinSet 1
object Out : Real 1
program m : One -> Out
    let total = sum(external)
    return total
export m
"""
    with pytest.raises(QiecDiagnosticError, match="nothing fixes its shape"):
        Model(source)
    typed = source.replace(
        "    let total", "    data external : Tensor[Real]([3])\n    let total"
    )
    bound = Model(typed).bind({"external": torch.ones(3)})
    torch.testing.assert_close(bound.trace().output, torch.tensor(3.0))


def test_declaration_conflicting_with_use_fails_at_compile_time():
    with pytest.raises(QiecDiagnosticError):
        Model(SOURCE.replace("data y : Tensor[Real]", "data y : Tensor[Int]"))


def test_duplicate_data_declaration_fails_at_compile_time():
    source = SOURCE.replace(
        "    data x : Tensor[Real]([Row])",
        "    data x : Tensor[Real]([Row])\n    data x : Tensor[Real]([Row])",
    )
    with pytest.raises(QiecDiagnosticError, match="duplicate data"):
        Model(source)


def test_inference_method_can_return_structured_certificate():
    class Solver:
        def run(self, model):
            return {"entry": model.schema.entry.name, "certified": False, "bound": 1.25}

    bound = Model(SOURCE).bind({"x": torch.ones(3), "y": torch.zeros(3)})
    assert bound.infer(Solver()) == {
        "entry": "regression",
        "certified": False,
        "bound": 1.25,
    }


def test_mcmc_and_prediction_keep_covariates_release_observations():
    bound = Model(SOURCE).bind({"x": torch.ones(3), "y": torch.full((3,), 123.0)})
    method = MCMCMethod(
        sampler=MCMC(NUTSKernel(), num_warmup=2, num_samples=3, num_chains=1, seed=10)
    )
    fit = bound.infer(method)
    assert fit.samples["b"].shape == (1, 3)
    predictive = fit.predict(num_samples=2)
    assert predictive["y"].shape[0] == 2
    assert not torch.all(predictive["y"] == 123.0)
    assert set(bound.data) == {"x", "y"}


def test_variational_method_preserves_rng_and_records_losses():
    bound = Model(SOURCE).bind({"x": torch.ones(3), "y": torch.zeros(3)})
    before = torch.random.get_rng_state().clone()
    seen = []
    fit = bound.infer(
        VariationalMethod(
            num_steps=3, callback=lambda index, loss: seen.append((index, loss))
        )
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    assert fit.losses == tuple(loss for _, loss in seen)
    assert fit.draws(num_samples=4)["b"].shape[:2] == (1, 4)
    assert fit.predict(num_samples=2)["y"].shape[0] == 2


def test_binding_preserves_data_gradients_and_cpu_rng():
    x = torch.ones(3, requires_grad=True)
    before = torch.random.get_rng_state().clone()
    bound = Model(SOURCE).bind({"x": x, "y": torch.zeros(3)})
    assert torch.equal(torch.random.get_rng_state(), before)
    bound.trace({"b": torch.ones(1)}).log_joint.sum().backward()
    assert x.grad is not None


def test_observation_alone_infers_rows():
    source = (
        SOURCE.replace("    data x : Tensor[Real]([Row])\n", "")
        .replace("    data y : Tensor[Real]([Row])\n", "")
        .replace("    let mu = b * x\n", "")
        .replace("loc=mu", "loc=b")
    )
    bound = Model(source).bind({"y": torch.zeros(5)})
    assert bound.schema.describe()["y"]["type"] == "Tensor[Real]([5])"


def test_ragged_shape_has_named_error():
    with pytest.raises(ValueError, match="x.*ragged"):
        Model(SOURCE).bind({"x": [[1.0], [2.0, 3.0]], "y": torch.zeros(2)})


GROUPED = """object Row : FinSet _
object Group : FinSet _
object Out : Real 1
program grouped : Row -> Out
    sample a : Group <- Normal(loc=0.0, scale=1.0)
    let mu = a[group_idx]
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return a
export grouped
"""


def test_group_index_range_infers_smallest_zero_based_space():
    bound = Model(GROUPED).bind(
        {"group_idx": torch.tensor([0, 0, 2]), "y": torch.zeros(3)}
    )
    assert bound.trace().sites["a"].value.numel() == 3
    assert bound.schema.describe()["group_idx"]["type"] == "Tensor[Int]([3])"


def test_matrix_shape_infers_two_finite_extents():
    source = (
        SOURCE.replace("object Out", "object Feature : FinSet _\nobject Out")
        .replace(
            "data x : Tensor[Real]([Row])", "data x : Tensor[Real]([Row, Feature])"
        )
        .replace("b * x", "b * sum(x)")
    )
    bound = Model(source).bind({"x": torch.ones(3, 2), "y": torch.zeros(3)})
    assert bound.schema.describe()["x"]["type"] == "Tensor[Real]([3, 2])"
    point = torch.tensor([0.25])
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(point) + 3 * (
        torch.distributions.Normal(2 * point, 1.0).log_prob(torch.tensor(0.0))
    )
    torch.testing.assert_close(bound.trace({"b": point}).log_joint, expected)


def test_exact_group_extent_must_contain_all_external_indices():
    source = GROUPED.replace(
        "    sample a", "    data table : Tensor[Real]([Group])\n    sample a"
    )
    with pytest.raises(CompileError, match="Group.*indices.*group_idx"):
        Model(source).bind(
            {
                "table": torch.zeros(2),
                "group_idx": torch.tensor([0, 1, 2]),
                "y": torch.zeros(3),
            }
        )


@pytest.mark.parametrize("indices", [torch.tensor([-1, 0, 1]), torch.tensor([0, 1, 3])])
def test_fixed_group_bounds_reject_invalid_external_indices(indices):
    source = GROUPED.replace("object Group : FinSet _", "object Group : FinSet 3")
    with pytest.raises(ValueError, match="index datum.*group_idx"):
        Model(source).bind({"group_idx": indices, "y": torch.zeros(3)})


def test_lda_infers_documents_and_tokens_without_python_role_lists():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2] / "docs/examples/source/lda.qvr"
    ).read_text()
    source = source.replace("FinSet 20", "FinSet _").replace("FinSet 1000", "FinSet _")
    model = Model(source)
    bound = model.bind(
        {
            "doc_idx": torch.tensor([0, 0, 1, 1]),
            "token_idx": torch.arange(4),
            "w": torch.tensor([0, 1, 2, 3]),
        },
        parameters={"alpha": 0.5, "beta": 0.5},
    )
    recorded = bound.trace()
    assert recorded.sites["theta"].value.shape == (2, 3)
    assert recorded.sites["phi"].value.shape == (3, 50)
    assert bound.schema.observations == ("w",)


def test_check_and_editor_analysis_support_symbolic_extents(tmp_path):
    from quivers.cli.check import check_file
    from quivers.dsl.qiec_tooling import analyze_module

    path = tmp_path / "regression.qvr"
    path.write_text(SOURCE)
    assert not check_file(path)
    analysis = analyze_module(parse(SOURCE), module_name="regression")
    assert analysis.compile_error is None
    assert not analysis.qiec_diagnostics


def test_explicit_tensor_cannot_use_implicit_shared_scalar_layout():
    source = SOURCE.replace("    let mu = b * x\n", "").replace("loc=mu", "loc=x")
    with pytest.raises(ValueError, match="shape"):
        Model(source).bind({"x": torch.tensor(0.0), "y": torch.zeros(3)})


def test_template_object_substitution_preserves_declared_data_types():
    source = """object Row : FinSet 3
object Group : FinSet 2
object Out : Real 1
program grouped(G : FinSet) : Row -> Out
    data group_idx : Tensor[Int]([Row])
    data y : Tensor[Real]([Row])
    sample a : G <- Normal(loc=0.0, scale=1.0)
    let mu = a[group_idx]
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return a
export grouped
"""
    data = {"group_idx": torch.tensor([0, 1, 0]), "y": torch.zeros(3)}
    model = Model(source)
    bound = model.bind(data, parameters={"G": "Group"})
    assert bound.schema.describe()["group_idx"]["type"] == "Tensor[Int]([3])"
    assert bound.trace().sites["a"].value.numel() == 2
    invalid = {**data, "group_idx": torch.tensor([0, 2, 0])}
    with pytest.raises(ValueError, match="index datum.*group_idx"):
        model.bind(invalid, parameters={"G": "Group"})
    native = loads(source).grouped("Group").morphism
    from quivers.inference import trace

    with pytest.raises(ValueError, match="index datum.*group_idx"):
        trace(native, torch.zeros(1, 1), invalid)


def test_reference_only_entry_validates_binding_before_custom_inference():
    source = """object One : FinSet 1
object Out : Real 1
define shift(x : Real) : Real !{} =
    if x <= 0.0 then
        return 0.0
    else
        let rest <- shift(x - 1.0)
        return x + rest
program m : One -> Out
    data x : Real
    let z <- shift(x)
    return z
export m
"""
    model = Model(source)
    with pytest.raises(TypeError, match="dtype"):
        model.bind({"x": torch.tensor(1, dtype=torch.long)})
    bound = model.bind({"x": torch.tensor(1.0)})
    with pytest.raises(TypeError, match="enhancement #232"):
        bound.require_tensor_program()


@pytest.mark.parametrize("invalid", ["typo", True, torch.tensor(1.0)])
def test_python_template_boundary_rejects_untyped_scalar_arguments(invalid):
    source = SOURCE.replace("FinSet _", "FinSet 3").replace(
        "program regression :", "program regression(scale : Real) :"
    )
    data = {"x": torch.ones(3), "y": torch.zeros(3)}
    with pytest.raises(TypeError, match="parameter 'scale'.*Real"):
        Model(source).bind(data, parameters={"scale": invalid})
    with pytest.raises(TypeError, match="parameter 'scale'.*Real"):
        loads(source).regression(scale=invalid)


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_python_template_boundary_checks_natural_arguments(invalid):
    source = SOURCE.replace("FinSet _", "FinSet 3").replace(
        "program regression :", "program regression(count : Nat) :"
    )
    with pytest.raises(TypeError, match="parameter 'count'"):
        loads(source).regression(count=invalid)


def test_python_template_boundary_accepts_numpy_numbers():
    import numpy as np

    source = SOURCE.replace("FinSet _", "FinSet 3")
    real = source.replace("program regression :", "program regression(scale : Real) :")
    natural = source.replace(
        "program regression :", "program regression(count : Nat) :"
    )
    assert loads(real).regression(scale=np.float32(0.5)).morphism is not None
    assert loads(natural).regression(count=np.int64(2)).morphism is not None


def test_scanned_observations_bind_at_checked_shape_and_fit_all_positions():
    from tests.dsl.test_scan_program_cells import STATE_SPACE
    from quivers.inference import trace

    drivers = torch.zeros(1, 3, 2)
    observations = torch.zeros(3, 2)
    model = Model(STATE_SPACE)
    assert model.entry == "ssm"
    bound = model.bind({"o": observations}, input=drivers)
    latents = {f"s@{t}": torch.full((1, 3), t * 0.1) for t in range(3)}
    actual = bound.trace(latents)
    expected = trace(
        bound.require_tensor_program(),
        drivers,
        {"o": observations.unsqueeze(0), **latents},
    )
    torch.testing.assert_close(actual.log_joint, expected.log_joint)
    assert bound.data["o"] is observations
    with pytest.raises(ValueError, match="never reached.*s@99"):
        bound.trace({"s@99": torch.zeros(1, 3)})
    fit = bound.infer(VariationalMethod(num_steps=2))
    assert set(fit.draws(num_samples=2)) == {"s@0", "s@1", "s@2"}
    assert len(fit.losses) == 2
