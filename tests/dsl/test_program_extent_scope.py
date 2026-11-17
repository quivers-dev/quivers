"""Each program quantifies its own extents independently of other entries."""

from __future__ import annotations

import math

import pytest

from quivers.dsl import parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec import dumps, loads
from quivers.qiec.kinds import NAT
from quivers.qiec.program_runtime import program_entry, run_program
from quivers.qiec.types import IndexLiteral


OPEN_REGRESSION = """program regression(x : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
"""
CONCRETE_COUNTS = """program count_model : CountRows -> CountRows
    observe count_y : CountRows <- COMPoisson(rate=2.0, dispersion=1.0)
    return count_y
"""
OBJECTS = """object Row : FinSet _
object CountRows : FinSet 2
object Out : Real 1
"""


def _module(source):
    return lower_qvr_to_qiec(parse(source), file_path="extent-scope.qvr")


@pytest.mark.parametrize("reverse", [False, True])
def test_unrelated_open_extent_does_not_change_concrete_entry_or_execution(reverse):
    declarations = [OPEN_REGRESSION, CONCRETE_COUNTS]
    if reverse:
        declarations.reverse()
    module = _module(OBJECTS + "\n".join(declarations))
    assert [
        binder.name for binder in program_entry(module, "regression").telescope
    ] == ["Row"]
    assert program_entry(module, "count_model").telescope == ()
    result = run_program(module, "count_model", data={"count_y": (1, 2)}, sites={})
    # Independently normalized Poisson masses at one and two, rate two.
    assert result.log_joint == pytest.approx(-4 + 3 * math.log(2) - math.lgamma(3))


def test_extent_used_only_by_body_plate_is_retained_and_requires_specialization():
    module = _module(
        """object Batch : FinSet _
object Unused : FinSet _
object Out : Real 1
program prior : Out -> Out
    sample latent : Batch <- Normal(loc=0.0, scale=1.0)
    let zero = 0.0
    return zero
"""
    )
    assert [binder.name for binder in program_entry(module, "prior").telescope] == [
        "Batch"
    ]
    with pytest.raises(KeyError, match="Batch"):
        run_program(
            module, "prior", data={"out": (0.0,)}, sites={"latent": (0.0, 0.0, 0.0)}
        )
    result = run_program(
        module,
        "prior",
        data={"out": (0.0,)},
        sites={"latent": (0.0, 0.0, 0.0)},
        static_arguments=(IndexLiteral(value=3, sort=NAT),),
    )
    assert result.log_joint == pytest.approx(-1.5 * math.log(2 * math.pi))


@pytest.mark.parametrize(
    "header,names",
    [
        ("[n : Nat, A : Type]", ["n", "A"]),
        ("[G : FinSet]", ["G"]),
        ("[A : Type, n : Nat, E : Effect]", ["A", "n", "E"]),
    ],
)
def test_explicit_unused_static_binder_remains_part_of_public_interface(header, names):
    module = _module(
        f"""object Unused : FinSet _
object Out : Real 1
program constant{header}() : Out -> Out
    let one = 1.0
    return one
"""
    )
    assert [
        binder.name for binder in program_entry(module, "constant").telescope
    ] == names


@pytest.mark.parametrize("position", ["domain", "codomain"])
def test_extent_used_only_by_declared_boundary_is_retained(position):
    domain, codomain = ("Row", "Out") if position == "domain" else ("Out", "Row")
    module = _module(
        OBJECTS
        + f"""program constant : {domain} -> {codomain}
    let one = 1.0
    return one
"""
    )
    assert [binder.name for binder in program_entry(module, "constant").telescope] == [
        "Row"
    ]


def test_extent_scoping_survives_source_and_serialized_module_roundtrips():
    syntax = parse(OBJECTS + OPEN_REGRESSION + CONCRETE_COUNTS)
    source = module_to_source(syntax)
    module = _module(source)
    restored = loads(dumps(module))
    assert restored == module
    assert program_entry(restored, "count_model").telescope == ()
    assert [
        binder.name for binder in program_entry(restored, "regression").telescope
    ] == ["Row"]
    assert module_to_source(parse(source)) == source


def test_scope_scan_distinguishes_free_motive_and_rigid_branch_extents():
    from quivers.dsl.ast_nodes import TypeName
    from quivers.dsl.program_elaboration import ObjectInfo, _program_extent_mentions
    from quivers.qiec import REAL, tensor_type
    from quivers.qiec.identifiers import StaticVariableId
    from quivers.qiec.kinds import IndexBinder
    from quivers.qiec.terms import CaseMotive
    from quivers.qiec.types import IndexVariable

    bound = IndexVariable(name="local", sort=NAT)
    free = IndexVariable(name="rows", sort=NAT)
    rigid = IndexVariable(
        name="unrelated",
        sort=NAT,
        identity=StaticVariableId.derive("extent-scope", "branch"),
    )
    shared = tensor_type(REAL, (bound, free, rigid))
    motive = CaseMotive(
        indices=(IndexBinder(name="local", sort=NAT),), result_type=shared
    )
    assert _program_extent_mentions((motive,), {}) == {"rows"}
    # The same shared type outside the motive sees 'local' as free.
    assert _program_extent_mentions((motive, shared), {}) == {"local", "rows"}
    aliases = {"Rows": ObjectInfo(extent=free), "Branch": ObjectInfo(extent=rigid)}
    assert _program_extent_mentions(
        (TypeName(name="Rows"), TypeName(name="Branch")), aliases
    ) == {"rows"}


def test_scope_scan_reads_literal_type_without_accessing_data_buffer(monkeypatch):
    from quivers.dsl.program_elaboration import _program_extent_mentions
    from quivers.qiec import REAL, tensor_type
    from quivers.qiec.terms import LiteralValue
    from quivers.qiec.types import IndexVariable

    literal = LiteralValue(
        value=(0.0,) * 1024,
        type=tensor_type(REAL, (IndexVariable(name="columns", sort=NAT),)),
    )
    original = LiteralValue.__getattribute__

    def guard_buffer(self, name):
        if self is literal and name == "value":
            raise AssertionError("extent traversal accessed an opaque literal buffer")
        return original(self, name)

    monkeypatch.setattr(LiteralValue, "__getattribute__", guard_buffer)
    assert _program_extent_mentions((literal,), {}) == {"columns"}
