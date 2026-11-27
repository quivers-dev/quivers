"""Observation weights scale likelihood contributions before aggregation."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import parse
from quivers.dsl.ast_nodes import ObserveStep, OptionNumber, ProgramDecl
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec import QiecModule, dumps, loads
from quivers.qiec.canonical import tensor_shape
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.primitives import IMPLEMENTATIONS
from quivers.qiec.program_runtime import program_entry, run_program

HEAD = "object Row : FinSet 3\nobject Out : Real 1\n"
BODY = """    sample a <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- Normal(loc=a, scale=1.0) [weights=w]
    return a
export main
"""


def _machine(name: str, *arguments: float | bool) -> float:
    """Apply a primitive of the reference machine to host numbers.

    Parameters
    ----------
    name : str
        The primitive.
    *arguments : float | bool
        Host numbers, made ``float64`` tensors (a truth value a ``bool``
        tensor) as the machine holds them.

    Returns
    -------
    float
        The result.
    """
    tensors = [
        torch.tensor(argument)
        if isinstance(argument, bool)
        else torch.tensor(argument, dtype=torch.float64)
        for argument in arguments
    ]
    return float(IMPLEMENTATIONS[name](NumberDomain(), *tensors))


def _module(source: str) -> QiecModule:
    return lower_qvr_to_qiec(parse(source), file_path="weights.qvr")


def _normal(value: float, mean: float = 0.0) -> float:
    return -0.5 * math.log(2.0 * math.pi) - 0.5 * (value - mean) ** 2


@pytest.mark.parametrize("weights", [(1.0, 0.0, 2.0), (0.0, 0.0, 0.0)])
def test_inferred_weights_scale_rows_and_preserve_prior(weights) -> None:
    module = _module(HEAD + "program main : Row -> Out\n" + BODY)
    entry = program_entry(module, "main")
    weight = next(parameter for parameter in entry.parameters if parameter.name == "w")
    assert weight.role == "data"
    assert weight.constraint is None
    assert weight.axes == ("Row",)
    shape = tensor_shape(weight.type)
    assert shape is not None and shape[1][0].value == 3
    data = {"y": (1.0, 2.0, 3.0), "w": weights}
    actual = run_program(module, "main", data=data, sites={"a": 0.5})
    expected = _normal(0.5) + sum(
        w * _normal(y, 0.5) for y, w in zip(data["y"], weights, strict=True)
    )
    assert actual.log_joint == pytest.approx(expected)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert run_program(
        restored, "main", data=data, sites={"a": 0.5}
    ).log_joint == pytest.approx(expected)


@pytest.mark.parametrize("weight", [0.0, 0.5, 2.0])
def test_scalar_weight_broadcasts_over_batch(weight: float) -> None:
    module = _module(HEAD + "program main(w : Real) : Row -> Out\n" + BODY)
    actual = run_program(
        module, "main", data={"y": (1.0, 2.0, 3.0), "w": weight}, sites={"a": 0.5}
    )
    assert actual.log_joint == pytest.approx(
        _normal(0.5) + weight * sum(_normal(y, 0.5) for y in (1.0, 2.0, 3.0))
    )


@pytest.mark.parametrize("invalid", [-1.0, math.inf, -math.inf, math.nan, True])
def test_observation_weight_primitive_rejects_invalid_values(invalid) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _machine("observation_weight", invalid)


@pytest.mark.parametrize("invalid", [-1.0, math.inf, math.nan])
def test_named_weight_values_are_checked_at_reference_execution(invalid: float) -> None:
    module = _module(HEAD + "program main : Row -> Out\n" + BODY)
    with pytest.raises(ExecutionFailure, match="finite and nonnegative"):
        run_program(
            module,
            "main",
            data={"y": (1.0, 2.0, 3.0), "w": (1.0, invalid, 0.0)},
            sites={"a": 0.0},
        )


@pytest.mark.parametrize("invalid", [-1.0, math.inf, math.nan])
def test_literal_weights_are_rejected_before_execution(invalid: float) -> None:
    module = parse(HEAD + "program main : Row -> Out\n" + BODY)
    program = next(
        statement
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
    )
    observe = program.draws[1]
    assert isinstance(observe, ObserveStep)
    observe = observe.with_(
        options=(observe.options[0].with_(value=OptionNumber(value=invalid)),)
    )
    program = program.with_(draws=(program.draws[0], observe))
    module = module.with_(
        statements=tuple(
            program if isinstance(statement, ProgramDecl) else statement
            for statement in module.statements
        )
    )
    with pytest.raises(QiecDiagnosticError, match="finite and nonnegative"):
        lower_qvr_to_qiec(module)


@pytest.mark.parametrize(
    "type_",
    [
        "Bool",
        "Int",
        "Tensor[Int]([Row])",
        "Tensor[Real]([2])",
        "Tensor[Real]([Row, 1])",
    ],
)
def test_declared_weight_types_must_match_observation_batch(type_: str) -> None:
    with pytest.raises(QiecDiagnosticError, match="observation weights.*must be Real"):
        _module(HEAD + f"program main(w : {type_}) : Row -> Out\n" + BODY)


@pytest.mark.parametrize(
    "option", ["true", "false", "[1.0, 2.0, 3.0]", '"w"', "product(w)"]
)
def test_weights_option_accepts_only_numeric_literal_or_name(option: str) -> None:
    with pytest.raises(QiecDiagnosticError, match="numeric literal or a value name"):
        _module(
            (HEAD + "program main : Row -> Out\n" + BODY).replace(
                "weights=w", f"weights={option}"
            )
        )


def test_computed_weight_is_checked_even_when_it_is_not_an_external_input() -> None:
    source = (
        HEAD
        + """program main : Row -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
    let w = a - 1.0
    observe y : Row <- Normal(loc=0.0, scale=1.0) [weights=w]
    return a
export main
"""
    )
    module = _module(source)
    assert all(
        parameter.name != "w" for parameter in program_entry(module, "main").parameters
    )
    with pytest.raises(ExecutionFailure, match="finite and nonnegative"):
        run_program(module, "main", data={"y": (0.0, 0.0, 0.0)}, sites={"a": 0.0})
    assert run_program(
        module, "main", data={"y": (0.0, 0.0, 0.0)}, sites={"a": 2.0}
    ).log_joint == pytest.approx(_normal(2.0) + 3 * _normal(0.0))


def test_zero_weight_outside_support_is_absorbing() -> None:
    module = _module(
        HEAD
        + """program main : Row -> Out
    observe y : Row <- Exponential(rate=1.0) [weights=w]
    return y
export main
"""
    )
    actual = run_program(
        module, "main", data={"y": (-1.0, 2.0, 3.0), "w": (0.0, 1.0, 2.0)}, sites={}
    )
    assert actual.log_joint == pytest.approx(-8.0)


def test_template_weight_formal_and_local_are_capture_avoiding() -> None:
    source = (
        HEAD
        + """program sub[A : FinSet](y : Tensor[Real]([A]), weight : Real) : Row -> Out
    let w = weight + 1.0
    observe y : A <- Normal(loc=0.0, scale=1.0) [weights=w]
    return y
program main : Row -> Out
    let w = 19.0
    sample z <- sub[Row](response, 1.0)
    return z
export main
"""
    )
    module = _module(source)
    assert {
        parameter.name for parameter in program_entry(module, "main").parameters
    } == {"response"}
    actual = run_program(module, "main", data={"response": (1.0, 2.0, 3.0)}, sites={})
    assert actual.log_joint == pytest.approx(
        2.0 * sum(_normal(y) for y in (1.0, 2.0, 3.0))
    )


def test_template_weight_actual_substitutes_literal_and_external_name() -> None:
    for actual, data, weight in [
        ("2.0", {}, 2.0),
        ("importance", {"importance": 0.5}, 0.5),
    ]:
        source = (
            HEAD
            + f"""program sub[A : FinSet](y : Tensor[Real]([A]), weight : Real) : Row -> Out
    observe y : A <- Normal(loc=0.0, scale=1.0) [weights=weight]
    return y
program main : Row -> Out
    sample z <- sub[Row](response, {actual})
    return z
export main
"""
        )
        module = _module(source)
        result = run_program(
            module, "main", data={"response": (1.0, 2.0, 3.0), **data}, sites={}
        )
        assert result.log_joint == pytest.approx(
            weight * sum(_normal(y) for y in (1.0, 2.0, 3.0))
        )


def test_grouped_weights_apply_before_row_segmentation_and_class_sum() -> None:
    source = """object Class : FinSet 2
object Group : FinSet 2
object Row : FinSet 3
object Out : Real 1
program main(mu : Tensor[Real]([Class])) : Row -> Out
    marginalize cls : Class <- Categorical(probs=[0.25, 0.75]) [over=Group]
        observe y : Row <- Normal(loc=mu[cls], scale=1.0) [via=group_idx, weights=w]
    return mu
export main
"""
    module = _module(source)
    result = run_program(
        module,
        "main",
        data={
            "mu": (0.0, 3.0),
            "y": (1.0, 2.0, 4.0),
            "group_idx": (0, 0, 1),
            "w": (0.0, 2.0, 0.5),
        },
        sites={},
    )
    expected = 0.0
    for rows in [[(1.0, 0.0), (2.0, 2.0)], [(4.0, 0.5)]]:
        terms = [
            math.log(prior) + sum(w * _normal(y, mean) for y, w in rows)
            for mean, prior in [(0.0, 0.25), (3.0, 0.75)]
        ]
        peak = max(terms)
        expected += peak + math.log(sum(math.exp(term - peak) for term in terms))
    assert result.log_joint == pytest.approx(expected)


def test_weight_one_has_exactly_the_unweighted_density() -> None:
    source = HEAD + "program main : Row -> Out\n" + BODY
    weighted = _module(source.replace("weights=w", "weights=1.0"))
    unweighted = _module(source.replace(" [weights=w]", ""))
    arguments = {"data": {"y": (1.0, 2.0, 3.0)}, "sites": {"a": 0.5}}
    assert (
        run_program(weighted, "main", **arguments).log_joint
        == run_program(unweighted, "main", **arguments).log_joint
    )


def test_computed_vector_weight_infers_its_free_data_axis() -> None:
    source = (
        HEAD
        + """program main : Row -> Out
    let w = base * 2.0
    observe y : Row <- Normal(loc=0.0, scale=1.0) [weights=w]
    return y
export main
"""
    )
    module = _module(source)
    parameter = next(
        parameter
        for parameter in program_entry(module, "main").parameters
        if parameter.name == "base"
    )
    assert parameter.axes == ("Row",)
    result = run_program(
        module, "main", data={"base": (0.5, 0.0, 1.0), "y": (1.0, 2.0, 3.0)}, sites={}
    )
    assert result.log_joint == pytest.approx(_normal(1.0) + 2.0 * _normal(3.0))


def test_scalar_observation_infers_a_scalar_weight() -> None:
    source = (
        HEAD
        + """program main : Row -> Out
    observe y <- Normal(loc=0.0, scale=1.0) [weights=w]
    return y
export main
"""
    )
    module = _module(source)
    parameter = next(
        parameter
        for parameter in program_entry(module, "main").parameters
        if parameter.name == "w"
    )
    assert tensor_shape(parameter.type) is None
    assert run_program(
        module, "main", data={"w": 0.5, "y": 2.0}, sites={}
    ).log_joint == pytest.approx(0.5 * _normal(2.0))


def test_duplicate_weight_options_are_rejected() -> None:
    with pytest.raises(QiecDiagnosticError, match="one weights option"):
        _module(
            (HEAD + "program main : Row -> Out\n" + BODY).replace(
                "weights=w", "weights=w, weights=1.0"
            )
        )


def test_weights_do_not_invent_data_bindings_for_static_objects() -> None:
    with pytest.raises(QiecDiagnosticError, match="runtime value.*static"):
        _module(
            (HEAD + "program main : Row -> Out\n" + BODY).replace(
                "weights=w", "weights=Row"
            )
        )


def test_event_axes_cannot_receive_independent_observation_weights() -> None:
    source = (
        HEAD
        + """program main(w : Tensor[Real]([Row])) : Row -> Out
    observe y <- Normal(loc=0.0, scale=1.0) [over=Row, weights=w]
    return y
export main
"""
    )
    with pytest.raises(QiecDiagnosticError, match="observation weights.*must be Real"):
        _module(source)
