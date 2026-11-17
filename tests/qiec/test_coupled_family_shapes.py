"""Source family inference uses one joint exact Nat equation system."""

import pytest

import quivers.dsl.program_elaboration as programs
import quivers.dsl.qiec_lowering as lowering
import quivers.qiec.checking as checking
from quivers.dsl import Compiler, parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.qiec import (
    NAT,
    REAL,
    FAMILIES,
    DistributionFamily,
    FamilyParameter,
    FamilyShapeSignature,
    IndexBinder,
    IndexVariable,
    add_indices,
)


@pytest.fixture(params=[False, True], ids=["forward-parameters", "reverse-parameters"])
def coupled_family(monkeypatch, request):
    n, m = (IndexVariable(name=name, sort=NAT) for name in ("n", "m"))
    record = DistributionFamily(
        name="CoupledShapeTest",
        parameters=(
            FamilyParameter(
                name="a", rank=1, constraint="real", scalar_broadcast=False
            ),
            FamilyParameter(
                name="b", rank=1, constraint="real", scalar_broadcast=False
            ),
        ),
        shape_signature=FamilyShapeSignature(
            telescope=tuple(IndexBinder(name=name, sort=NAT) for name in ("n", "m")),
            parameter_shapes=(
                ("a", (add_indices(n, m),)),
                ("b", (add_indices(n, m, m),)),
            ),
            event_shape=(),
        ),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
    )
    if request.param:
        record = record.with_(parameters=tuple(reversed(record.parameters)))
    # This is a semantic shape fixture, not a new executable distribution.
    # Patch the same immutable registry view in both source and kernel routes.
    registry = dict(FAMILIES) | {record.name: record}
    for module in (programs, lowering, checking):
        monkeypatch.setattr(module, "FAMILIES", registry)
    return record


@pytest.mark.parametrize("arguments", ["a=a,b=b", "b=b,a=a"])
@pytest.mark.parametrize(
    "headers",
    [
        "a:Tensor[Real]([5]),b:Tensor[Real]([8])",
        "b:Tensor[Real]([8]),a:Tensor[Real]([5])",
    ],
)
def test_typed_coupled_arguments_check_in_every_order_and_source_roundtrip(
    coupled_family, arguments, headers
):
    source = f"""object Out:Real 1
program model({headers}):Out->Out
    sample y <- CoupledShapeTest({arguments})
    return y
export model
"""
    module = parse(source)
    Compiler(module).qiec_module
    Compiler(parse(module_to_source(module))).qiec_module


@pytest.mark.parametrize(
    "arguments",
    [
        "a=[0.,0.,0.,0.,0.],b=[0.,0.,0.,0.,0.,0.,0.,0.]",
        "b=[0.,0.,0.,0.,0.,0.,0.,0.],a=[0.,0.,0.,0.,0.]",
    ],
)
def test_literal_coupled_arguments_are_collected_before_inference(
    coupled_family, arguments
):
    Compiler(
        parse(f"""object Out:Real 1
program model:Out->Out
    sample y <- CoupledShapeTest({arguments})
    return y
export model
""")
    ).qiec_module


@pytest.mark.parametrize("first,second", [(5, 12), (5, 4)])
def test_negative_natural_solutions_fail_at_the_source_application(
    coupled_family, first, second
):
    source = f"""object Out:Real 1
program model(a:Tensor[Real]([{first}]),b:Tensor[Real]([{second}])):Out->Out
    sample y <- CoupledShapeTest(a=a,b=b)
    return y
"""
    with pytest.raises(QiecDiagnosticError, match="extent"):
        Compiler(parse(source)).qiec_module


def test_underdetermined_family_evidence_is_not_guessed(coupled_family):
    source = """object Out:Real 1
program model(a:Tensor[Real]([5])):Out->Out
    sample y <- CoupledShapeTest(a=a,b=b)
    return y
"""
    with pytest.raises(QiecDiagnosticError, match="underdetermined|extent"):
        Compiler(parse(source)).qiec_module


def test_declared_caller_indices_are_not_solved_from_coupled_family_constraints(
    coupled_family,
):
    source = """object Out:Real 1
program model[k:Nat](a:Tensor[Real]([k]),b:Tensor[Real]([8])):Out->Out
    sample y <- CoupledShapeTest(a=a,b=b)
    return y
"""
    with pytest.raises(QiecDiagnosticError, match="extent"):
        Compiler(parse(source)).qiec_module


def test_ordinary_computation_uses_the_same_affine_sample_signature():
    source = """define law[d:Nat](loc:Tensor[Real]([d]),chol:Tensor[Real]([d,d])) : Sampleable[Tensor[Real]([nat_add(d,1)])] !{} =
    return AdditiveLogRatioNormal(loc, chol)
"""
    Compiler(parse(source)).qiec_module
    with pytest.raises(QiecDiagnosticError, match="returns.*nat_add.*expected"):
        Compiler(parse(source.replace("[nat_add(d,1)]", "[d]"))).qiec_module


def test_fractional_joint_solutions_are_refused(coupled_family, monkeypatch):
    n, m = (IndexVariable(name=name, sort=NAT) for name in ("n", "m"))
    record = coupled_family.with_(
        shape_signature=coupled_family.shape_signature.with_(
            parameter_shapes=(
                ("a", (add_indices(n, n, m),)),
                ("b", (add_indices(n, m, m),)),
            )
        )
    )
    registry = dict(FAMILIES) | {record.name: record}
    for module in (programs, lowering, checking):
        monkeypatch.setattr(module, "FAMILIES", registry)
    source = """object Out:Real 1
program model(a:Tensor[Real]([5]),b:Tensor[Real]([8])):Out->Out
    sample y <- CoupledShapeTest(a=a,b=b)
    return y
"""
    with pytest.raises(QiecDiagnosticError, match="extent"):
        Compiler(parse(source)).qiec_module


@pytest.mark.parametrize(
    "expression",
    [
        "Restrict(Poisson(2.3),1.,3.)",
        "Normalize(Restrict(Poisson(2.3),1.,3.))",
        "Truncated(Poisson(2.3),1.,3.)",
        "PointMass(2)",
    ],
)
def test_pure_compositions_preserve_integer_sample_carriers(expression):
    source = f"""define law() : Sampleable[Int] !{{}} =
    return {expression}
"""
    Compiler(parse(source)).qiec_module
    with pytest.raises(QiecDiagnosticError, match="returns.*Int.*expected.*Real"):
        Compiler(
            parse(source.replace("Sampleable[Int]", "Sampleable[Real]"))
        ).qiec_module


def test_observation_axes_use_the_same_semantic_registry_as_the_source_checker(
    coupled_family,
):
    source = """object Out:Real 1
program model(a:Tensor[Real]([5]),b:Tensor[Real]([8])):Out->Out
    observe y <- CoupledShapeTest(a=a,b=b)
    return y
export model
"""
    checked = Compiler(parse(source)).qiec_module
    assert checked is not None
    assert checked.entries[0].sites[0].family == coupled_family.name
