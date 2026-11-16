"""Distribution constructors share complete schemas across source and kernel."""

import math

import pytest
import torch

from quivers._family_schemas import _semantic_parameterizations
from quivers.dsl import Compiler, loads as compile_source, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.qiec import (
    FAMILIES,
    REAL,
    NAT,
    IndexLiteral,
    TensorValue,
    tensor_type,
    Evaluator,
    Return,
    DistributionValue,
    KernelError,
    KernelRegistry,
    LiteralValue,
    dumps,
    family,
    infer_value,
    loads,
    run_named,
    sampleable_type,
)
from quivers.inference import trace
from quivers.qiec.identifiers import SourceOrigin


ORIGIN = SourceOrigin(
    module="constructors", structural_path=(), role="test", source_protocol="test"
)


def _distribution(name, names):
    return DistributionValue(
        family=family(name).id,
        name=name,
        arguments=tuple((key, LiteralValue(value=0.5, type=REAL)) for key in names),
        result_type=sampleable_type(REAL),
        origin=ORIGIN,
    )


def test_every_semantic_parameter_belongs_to_a_complete_schema():
    # This independent registry invariant catches both undeclared keywords and
    # forgotten alternative parameterizations, including structural GP sugar.
    for name, record in FAMILIES.items():
        schemas = _semantic_parameterizations(name, record.parameter_names)
        assert schemas, name
        assert {key for schema in schemas for key in schema} == set(
            record.parameter_names
        ), name
        assert all(len(schema) == len(set(schema)) for schema in schemas), name


@pytest.mark.parametrize(
    "name,names",
    [
        ("Normal", ("loc",)),
        ("Normal", ("scale",)),
        ("Bernoulli", ("probs", "logits")),
        ("COMPoisson", ("rate", "log_rate", "dispersion")),
    ],
)
def test_kernel_rejects_missing_and_exclusive_parameters(name, names):
    with pytest.raises(KernelError, match="exactly one complete parameter schema"):
        infer_value(_distribution(name, names), KernelRegistry())


@pytest.mark.parametrize("names", [("loc", "scale"), ("scale", "loc")])
def test_kernel_schema_selection_is_order_independent_and_serializable(names):
    value = _distribution("Normal", names)
    assert infer_value(value, KernelRegistry()) == sampleable_type(REAL)
    restored = loads(dumps(value))
    assert restored == value
    assert infer_value(restored, KernelRegistry()) == sampleable_type(REAL)


def _vector(*entries):
    return TensorValue(
        items=tuple(LiteralValue(value=value, type=REAL) for value in entries),
        result_type=tensor_type(REAL, (IndexLiteral(value=len(entries), sort=NAT),)),
    )


@pytest.mark.parametrize("parameter,number", [("rate", 2.0), ("log_rate", math.log(2))])
def test_structural_constructor_accepts_exactly_one_rate_parameter(parameter, number):
    controls = TensorValue(
        items=(_vector(1.0, 1.0),),
        result_type=tensor_type(
            REAL, (IndexLiteral(value=1, sort=NAT), IndexLiteral(value=2, sort=NAT))
        ),
    )
    arguments = (
        ("breaks", _vector(0.0, 1.0)),
        ("controls", controls),
        (parameter, LiteralValue(value=number, type=REAL)),
    )
    term = DistributionValue(
        family=family("BernsteinHazard").id,
        name="BernsteinHazard",
        arguments=arguments,
        result_type=sampleable_type(REAL),
        origin=ORIGIN,
    )
    assert infer_value(term, KernelRegistry()) == sampleable_type(REAL)
    runtime = Evaluator().evaluate(Return(value=term))
    assert runtime.log_prob(0.0) == pytest.approx(math.log(2))
    for supplied in (
        arguments[:2],
        (
            *arguments,
            (
                "rate" if parameter == "log_rate" else "log_rate",
                LiteralValue(value=1.0, type=REAL),
            ),
        ),
    ):
        with pytest.raises(KernelError, match="exactly one complete parameter schema"):
            infer_value(term.with_(arguments=supplied), KernelRegistry())


@pytest.mark.parametrize("route", ["checked", "native"])
@pytest.mark.parametrize(
    "expression",
    [
        "Normal()",
        "Normal(0.)",
        "Normal(0.,1.,2.)",
        "Bernoulli(0.5,0.)",
        "COMPoisson(1.,0.,1.)",
        "BernsteinHazard([0.,1.],[[1.,1.]])",
        "BernsteinHazard([0.,1.],[[1.,1.]],1.,0.)",
        "MultivariateNormal([0.],[[1.]],[[1.]])",
    ],
)
def test_source_constructor_arity_is_checked_before_execution(route, expression):
    source = f"define law() : Sampleable[Real] !{{}} =\n    return {expression}\n"
    with pytest.raises(QiecDiagnosticError, match="complete parameter schema|at most"):
        if route == "checked":
            Compiler(parse(source)).qiec_module
        else:
            compile_source(source)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("Normal(0.,1.)", -0.5 * math.log(2 * math.pi)),
        ("BernsteinHazard([0.,1.],[[1.,1.]],2.)", math.log(2)),
    ],
)
def test_complete_pure_constructors_execute_after_source_roundtrip(
    expression, expected
):
    source = (
        f"define density() : LogWeight !{{}} =\n    return log_prob({expression},0.)\n"
    )
    module = parse(source)
    checked = Compiler(module).qiec_module
    assert run_named(checked, "density").value == pytest.approx(expected)
    emitted = module_to_source(module)
    assert module_to_source(parse(emitted)) == emitted
    restored = Compiler(parse(emitted)).qiec_module
    assert run_named(restored, "density").value == pytest.approx(expected)
    runtime_source = (
        source.replace("density()", "density(x:Real)").replace(
            f"log_prob({expression},0.)", f"log_prob({expression},x)"
        )
        + """object Out:Real 1
program model:Out->Out
    let ell <- density(0.)
    score evidence = weight_value(ell)
    return ell
export model
"""
    )
    native = compile_source(runtime_source).morphism
    actual = trace(native, torch.zeros(1)).log_joint
    assert float(actual.sum()) == pytest.approx(expected, abs=1e-6)
