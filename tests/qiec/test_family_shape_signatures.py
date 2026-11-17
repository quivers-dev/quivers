"""Distribution geometry has one dependent signature across checking frontends."""

import pytest
from didactic.fields import ValidationError

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError
from quivers.qiec import (
    NAT,
    REAL,
    FAMILIES,
    DistributionFamily,
    FamilyParameter,
    FamilyShapeSignature,
    IndexBinder,
    IndexLiteral,
    IndexVariable,
    KernelError,
    add_indices,
    shift_index,
    match_family_shapes,
    instantiate_family_shapes,
    validate_family_shape_dimensions,
    sampleable_type,
    tensor_type,
)
from quivers.qiec.checking import _parameter_expectation, _plated_sample_type
from quivers.qiec.identifiers import SourceOrigin, StaticVariableId
from quivers.qiec.kinds import SHAPE, EffectBinder
from quivers.qiec.terms import DistributionValue, PlateAxis, PlateShape


def nat(n):
    return IndexLiteral(value=n, sort=NAT)


def variable(name):
    return IndexVariable(name=name, sort=NAT)


def signature():
    s, k = variable("segments"), variable("controls")
    return FamilyShapeSignature(
        telescope=(
            IndexBinder(name="segments", sort=NAT),
            IndexBinder(name="controls", sort=NAT),
        ),
        parameter_shapes=(
            ("rate", ()),
            ("breaks", (shift_index(s, 1),)),
            ("controls", (s, k)),
        ),
        event_shape=(),
    )


def record():
    return DistributionFamily(
        name="TestHazard",
        parameters=(
            FamilyParameter(name="rate", constraint="positive", rank=0),
            FamilyParameter(
                name="breaks", constraint="real", rank=1, scalar_broadcast=False
            ),
            FamilyParameter(
                name="controls", constraint="positive", rank=2, scalar_broadcast=False
            ),
        ),
        shape_signature=signature(),
        support="nonnegative",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )


def test_parameter_only_dimensions_share_the_existing_affine_solver():
    sig = signature()
    assert (
        match_family_shapes(
            sig, {"rate": (), "breaks": (nat(3),), "controls": (nat(2), nat(4))}
        )
        == ()
    )
    assert validate_family_shape_dimensions(
        sig, {"rate": (), "breaks": (3,), "controls": (2, 4)}
    ) == {"segments": 2, "controls": 4}
    with pytest.raises(ValueError, match="incompatible|inconsistent"):
        match_family_shapes(sig, {"breaks": (nat(4),), "controls": (nat(2), nat(4))})
    with pytest.raises(ValueError, match="inconsistent"):
        validate_family_shape_dimensions(sig, {"breaks": (4,), "controls": (2, 4)})


def test_symbolic_affine_shapes_preserve_nominal_caller_and_rigid_identities():
    sig = signature()
    n, k = variable("__qvr_family_shape_0"), variable("controls")
    assert (
        match_family_shapes(sig, {"breaks": (shift_index(n, 1),), "controls": (n, k)})
        == ()
    )
    rigid = IndexVariable(
        name="segments", sort=NAT, identity=StaticVariableId.derive("shape", "branch")
    )
    assert (
        match_family_shapes(
            sig, {"breaks": (shift_index(rigid, 1),), "controls": (rigid, k)}
        )
        == ()
    )
    with pytest.raises(ValueError, match="incompatible"):
        match_family_shapes(
            sig,
            {"breaks": (shift_index(rigid, 1),), "controls": (variable("segments"), k)},
        )
    patterns, _ = instantiate_family_shapes(sig, {"segments": rigid, "controls": k})
    assert patterns["controls"][0].identity == rigid.identity


def test_coupled_ground_shapes_infer_unique_integral_event_without_single_unknown_seed():
    n, m = variable("n"), variable("m")
    sig = FamilyShapeSignature(
        telescope=(IndexBinder(name="n", sort=NAT), IndexBinder(name="m", sort=NAT)),
        parameter_shapes=(("a", (add_indices(n, m),)), ("b", (add_indices(n, m, m),))),
        event_shape=(n, m),
    )
    assert match_family_shapes(sig, {"a": (nat(5),), "b": (nat(8),)}) == (
        nat(2),
        nat(3),
    )
    assert validate_family_shape_dimensions(sig, {"a": (5,), "b": (8,)}) == {
        "n": 2,
        "m": 3,
    }
    with pytest.raises(ValueError, match="evidence|incompatible"):
        match_family_shapes(sig, {"a": (nat(5),)})
    with pytest.raises(ValueError, match="nonnegative|incompatible"):
        match_family_shapes(sig, {"a": (nat(5),), "b": (nat(12),)})


@pytest.mark.parametrize("name", tuple(FAMILIES))
def test_every_registry_family_has_one_rank_consistent_effective_contract(name):
    family = FAMILIES[name]
    sig = family.effective_shape_signature
    substitutions = {binder.name: nat(3) for binder in sig.telescope}
    parameters, event = instantiate_family_shapes(sig, substitutions)
    assert len(event) == family.event_rank
    assert all(len(parameters[p.name]) == p.rank for p in family.parameters)
    assert match_family_shapes(sig, parameters, event) == event
    assert family.effective_shape_signature is sig


def test_scalar_events_still_check_the_structural_parameter_relationship():
    family = record()
    origin = SourceOrigin(
        module="shape",
        structural_path=("test",),
        role="distribution",
        source_protocol="test",
    )
    value = DistributionValue(
        family=family.id,
        name=family.name,
        arguments=(),
        result_type=sampleable_type(REAL),
        origin=origin,
    )
    actual = {
        "rate": REAL,
        "breaks": tensor_type(REAL, (nat(3),)),
        "controls": tensor_type(REAL, (nat(2), nat(4))),
    }
    assert _plated_sample_type(value, family, actual) == sampleable_type(REAL)
    actual["controls"] = tensor_type(REAL, (nat(3), nat(4)))
    with pytest.raises(KernelError, match="incompatible.*extents"):
        _plated_sample_type(value, family, actual)


def test_structural_tables_cannot_use_legacy_scalar_plate_broadcast():
    plate = PlateShape(batch=(PlateAxis(name="row", size=nat(7)),))
    structural = record().parameter("controls")
    assert _parameter_expectation(structural, REAL, plate) is not None
    assert (
        _parameter_expectation(
            FamilyParameter(name="scale_tril", rank=2, constraint="real"), REAL, plate
        )
        is None
    )
    assert (
        _parameter_expectation(
            structural, tensor_type(REAL, (nat(7), nat(2), nat(4))), plate
        )
        is None
    )


@pytest.mark.parametrize(
    "mutations",
    [
        {"event_rank": 1},
        {"event_sources": ("missing",)},
        {"parameters": (FamilyParameter(name="different", rank=0, constraint="real"),)},
        {
            "parameters": (
                FamilyParameter(name="rate", rank=0, constraint="real"),
                FamilyParameter(name="breaks", rank=2, constraint="real"),
                FamilyParameter(name="controls", rank=2, constraint="real"),
            )
        },
    ],
)
def test_explicit_signature_rank_and_names_are_validated(mutations):
    original = record()
    arguments = {
        name: getattr(original, name) for name in original.__class__.__annotations__
    }
    with pytest.raises(
        (ValueError, TypeError, ValidationError), match="rank|parameter"
    ):
        DistributionFamily(**(arguments | mutations))


@pytest.mark.parametrize(
    "telescope,shape",
    [
        ((IndexBinder(name="x", sort=SHAPE),), (variable("x"),)),
        (
            (IndexBinder(name="x", sort=NAT), IndexBinder(name="x", sort=NAT)),
            (variable("x"),),
        ),
        ((IndexBinder(name="x", sort=NAT),), (variable("other"),)),
        ((EffectBinder(name="x"),), (variable("x"),)),
    ],
)
def test_malformed_scope_and_sorts_are_refused(telescope, shape):
    with pytest.raises((ValueError, TypeError, ValidationError)):
        FamilyShapeSignature(
            telescope=telescope, parameter_shapes=(("a", shape),), event_shape=()
        )


def test_source_affine_signature_checks_both_matrix_axes_without_specialization():
    source = """object Out : Real 1
program model[d:Nat](loc:Tensor[Real]([d]), factor:Tensor[Real]([d,d])):Out->Out
    sample y <- AdditiveLogRatioNormal(loc=loc,scale_tril=factor)
    return y
export model
"""
    Compiler(parse(source)).qiec_module
    with pytest.raises(QiecDiagnosticError, match="extent"):
        Compiler(parse(source.replace("[d,d]", "[d,nat_add(d,1)]"))).qiec_module


@pytest.mark.parametrize("extent", [True, 2.5, -1])
def test_concrete_shape_validation_preserves_natural_extent_domain(extent):
    with pytest.raises((TypeError, ValueError, ValidationError)):
        validate_family_shape_dimensions(signature(), {"breaks": (extent,)})


def test_concrete_and_symbolic_underdetermination_have_the_same_boundary():
    n, m = variable("n"), variable("m")
    sig = FamilyShapeSignature(
        telescope=(IndexBinder(name="n", sort=NAT), IndexBinder(name="m", sort=NAT)),
        parameter_shapes=(("a", (add_indices(n, m),)),),
        event_shape=(),
    )
    with pytest.raises(ValueError, match="evidence|incompatible"):
        match_family_shapes(sig, {"a": (nat(5),)})
    with pytest.raises(ValueError, match="evidence"):
        validate_family_shape_dimensions(sig, {"a": (5,)})


def test_instantiation_rejects_a_non_natural_replacement():
    with pytest.raises((ValueError, TypeError, ValidationError)):
        instantiate_family_shapes(
            signature(),
            {"segments": IndexVariable(name="shape", sort=SHAPE), "controls": nat(3)},
        )
