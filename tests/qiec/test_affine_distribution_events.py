"""Distribution parameter coordinates can differ by a proved Nat offset."""

import pytest
from didactic.fields import ValidationError
import torch

from quivers.dsl import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.modeling import Model
from quivers.qiec import (
    REAL,
    DistributionValue,
    KernelError,
    family,
    sampleable_type,
    tensor_type,
)
from quivers.qiec.checking import _plated_sample_type
from quivers.qiec.families import DistributionFamily
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.kinds import NAT
from quivers.qiec.terms import PlateAxis, PlateShape
from quivers.qiec.types import IndexLiteral, IndexVariable, shift_index

ORIGIN = SourceOrigin(
    module="affine",
    structural_path=("distribution",),
    role="distribution",
    source_protocol="test",
)


def test_symbolic_affine_event_is_canonical_and_checks_both_matrix_dimensions():
    record = family("AdditiveLogRatioNormal")
    d = IndexVariable(name="d", sort=NAT)
    rows = IndexLiteral(value=5, sort=NAT)
    k = shift_index(d, 1)
    plate = PlateShape(
        batch=(PlateAxis(name="Row", size=rows),),
        event=(PlateAxis(name="Component", size=k),),
    )
    result = sampleable_type(tensor_type(REAL, (rows, k)))
    value = DistributionValue(
        family=record.id,
        name=record.name,
        arguments=(),
        result_type=result,
        origin=ORIGIN,
        plate=plate,
    )
    arguments = {
        "loc": tensor_type(REAL, (rows, d)),
        "scale_tril": tensor_type(REAL, (rows, d, d)),
    }
    assert _plated_sample_type(value, record, arguments) == result
    arguments["scale_tril"] = tensor_type(REAL, (rows, d, k))
    with pytest.raises(KernelError, match="scale_tril.*event extent"):
        _plated_sample_type(value, record, arguments)


@pytest.mark.parametrize("offsets", [(-1,), (1, 2), (True,)])
def test_invalid_affine_event_contract_is_rejected(offsets):
    record = family("AdditiveLogRatioNormal")
    with pytest.raises((ValueError, TypeError, ValidationError), match="event"):
        DistributionFamily(
            **{
                name: getattr(record, name)
                for name in record.__class__.__annotations__
                if name != "event_offsets"
            },
            event_offsets=offsets,
        )


def test_telescope_affine_response_dimension_native_bind_and_checked_source():
    source = """object Out : Real 1
program model[n:Nat,d:Nat](loc:Tensor[Real]([n,d]), factor:Tensor[Real]([n,d,d]), y:Tensor[Real]([n,nat_add(d,1)])):n->Out
    observe y : n <- AdditiveLogRatioNormal(loc=loc,scale_tril=factor) [iid_over=n]
    return y
export model
"""
    # Explicit iid axes leave the native affine event shape to the family.
    lower_qvr_to_qiec(parse(source))
    loc = torch.zeros(2, 3, dtype=torch.double)
    factor = torch.eye(3, dtype=torch.double).expand(2, 3, 3)
    y = torch.full((2, 4), 0.25, dtype=torch.double)
    fitted = Model(source).bind({"loc": loc, "factor": factor, "y": y})
    expected = torch.distributions.MultivariateNormal(loc, scale_tril=factor).log_prob(
        torch.zeros_like(loc)
    ) - y.log().sum(-1)
    torch.testing.assert_close(fitted.trace({}).log_joint.sum(), expected.sum())


def test_wrong_equal_cardinality_axis_is_not_an_affine_coordinate():
    source = """object Row : FinSet 3
object Component : FinSet 3
program model(loc:Tensor[Real]([Row,Component]), factor:Tensor[Real]([Row,Component,Component]), y:Tensor[Real]([Row,Component])):Row->Row
    observe y : Row <- AdditiveLogRatioNormal(loc=loc,scale_tril=factor) [over=Component]
    return y
export model
"""
    with pytest.raises(ValueError, match="extent|event|sample"):
        lower_qvr_to_qiec(parse(source))


def test_shared_extent_solver_handles_joint_ground_evidence_and_symbolic_cancellation():
    from quivers.dsl.program_elaboration import _bind_extents
    from quivers.qiec.types import add_indices

    n = IndexVariable(name="n", sort=NAT)
    m = IndexVariable(name="m", sort=NAT)

    def literal(value):
        return IndexLiteral(value=value, sort=NAT)

    bindings = {}
    _bind_extents(
        tensor_type(REAL, (add_indices(n, m), add_indices(n, m, m))),
        tensor_type(REAL, (literal(5), literal(8))),
        bindings,
    )
    assert bindings == {"n": literal(2), "m": literal(3)}
    d = IndexVariable(name="d", sort=NAT)
    bindings = {}
    _bind_extents(
        tensor_type(REAL, (shift_index(n, 1),)),
        tensor_type(REAL, (shift_index(d, 1),)),
        bindings,
    )
    assert bindings == {"n": d}


def test_template_static_axis_options_preserve_stage_and_list_order():
    from quivers.dsl.ast_nodes import OptionEntry, OptionList, OptionName
    from quivers.dsl.program_templates import _rename_step_options

    options = (
        OptionEntry(
            key="iid_over",
            value=OptionList(items=(OptionName(value="n"), OptionName(value="m"))),
        ),
        OptionEntry(key="weights", value=OptionName(value="n")),
    )
    actual = _rename_step_options(
        options, {"n": "Row", "m": "Column"}, {}, {"n": "runtime_n"}
    )
    assert tuple(item.value for item in actual[0].value.items) == ("Row", "Column")
    assert actual[1].value.value == "runtime_n"


def test_callee_formal_and_caller_extent_may_share_display_names():
    from quivers.dsl.program_elaboration import _bind_extents

    n = IndexVariable(name="n", sort=NAT)
    for offset in (0, 1):
        bindings = {}
        _bind_extents(
            tensor_type(REAL, (shift_index(n, offset),)),
            tensor_type(REAL, (shift_index(n, offset),)),
            bindings,
        )
        assert bindings == {"n": n}
