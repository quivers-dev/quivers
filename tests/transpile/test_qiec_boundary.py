"""Conformance tests for the shared QIEC-to-transpiler boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.plan import Lower
from quivers.transpile.qiec_ir import (
    IRQiecModule,
    IRQiecReturn,
    analyze_qiec_capabilities,
)


_OLD_PROGRAM = """\
object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Beta(concentration1=2.0, concentration0=2.0)
    return x
export prog
"""

_OLD_PROGRAM_WITH_QIEC_METADATA = "index Nat = Z | S(Nat)\n\n" + _OLD_PROGRAM

_QIEC_COMPUTATION = """\
define answer() : Int !{} =
    return 42

object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Bernoulli(probs=0.5)
    return x
export prog
"""

_ILL_TYPED_QIEC_METADATA = """\
effect Broken
    ask : Unit -> Missing

object Resp : FinSet 4
program prog : Resp -> Resp
    sample x <- Bernoulli(probs=0.5)
    return x
export prog
"""

_EFFECTFUL_QIEC = """\
effect State[S : Type]
    get : Unit -> S

instance cell : State[Int]

define read() : Int !{cell} =
    perform cell.get()
"""

_PARAMETERIZED_QIEC = """\
define identity(value : Int) : Int !{} =
    return value
"""


@pytest.mark.parametrize("target", available_targets())
def test_checked_qiec_metadata_can_accompany_old_program(target: str) -> None:
    """Static QIEC declarations preserve the accompanying program."""
    output = transpile(parse(_OLD_PROGRAM_WITH_QIEC_METADATA), target=target)

    assert output
    assert b"x" in output
    if target in {"bugs", "jags"}:
        # These graph languages reject an empty declaration-only model. The
        # neutral sentinel also keeps a mixed model syntactically uniform.
        assert b"qiec_declarations" in output


def test_qiec_computation_enters_typed_ir_without_blanket_refusal() -> None:
    ir = Lower().forward(parse(_QIEC_COMPUTATION))

    assert [item.name for item in ir.module.computations] == ["prog", "answer"]
    computation = ir.module.computations[1]
    assert computation.name == "answer"
    assert isinstance(computation.body, IRQiecReturn)
    assert analyze_qiec_capabilities(ir.module, "pyro") == ()
    assert analyze_qiec_capabilities(ir.module, "stan") == ()


@pytest.mark.parametrize("target", available_targets())
def test_invalid_qiec_metadata_fails_exact_check_before_render(target: str) -> None:
    """A mixed module cannot use metadata treatment to bypass QIEC checking."""
    with pytest.raises(QiecDiagnosticError) as exc_info:
        transpile(parse(_ILL_TYPED_QIEC_METADATA), target=target)

    assert exc_info.value.code == "qiec-kind"
    assert "unknown or unsaturated type 'Missing'" in exc_info.value.message


def test_qiec_declarations_lower_without_a_probabilistic_program() -> None:
    ir = Lower().forward(parse("index Nat = Z | S(Nat)\n"))

    assert ir.body == ()
    assert ir.inputs == ()
    assert [sort.name for sort in ir.module.index_sorts] == ["Nat"]


def test_checked_qiec_metadata_is_preserved_in_structural_ir() -> None:
    ir = Lower().forward(parse(_OLD_PROGRAM_WITH_QIEC_METADATA))

    assert isinstance(ir.module, IRQiecModule)
    assert [sort.name for sort in ir.module.index_sorts] == ["Nat"]


def test_typed_qiec_ir_round_trips_through_didactic_json() -> None:
    ir = Lower().forward(parse(_EFFECTFUL_QIEC, "state.qvr"))

    assert type(ir).model_validate_json(ir.model_dump_json()) == ir


def test_capability_analysis_is_target_specific_and_source_located() -> None:
    qiec = Lower().forward(parse(_EFFECTFUL_QIEC, "state.qvr")).module

    assert analyze_qiec_capabilities(qiec, "pyro") == ()
    diagnostics = analyze_qiec_capabilities(qiec, "stan")
    assert {diagnostic.feature for diagnostic in diagnostics} == {
        "effectful-row",
        "perform",
    }
    assert all(diagnostic.code == "qiec-capability" for diagnostic in diagnostics)
    assert all(diagnostic.computation == "read" for diagnostic in diagnostics)
    assert all(diagnostic.origin is not None for diagnostic in diagnostics)
    assert all(
        diagnostic.origin.line == 6 for diagnostic in diagnostics if diagnostic.origin
    )
    message = str(
        UnsupportedConstruct(
            "qvr-stan", [diagnostic.kind for diagnostic in diagnostics]
        )
    )
    assert "QIEC computation `read`" in message
    assert "`perform` QIEC capability" in message
    assert "silently erasing it would change the program" in message


def test_graphical_targets_refuse_named_qiec_parameters() -> None:
    qiec = Lower().forward(parse(_PARAMETERIZED_QIEC)).module

    assert analyze_qiec_capabilities(qiec, "stan") == ()
    for target in ("bugs", "jags"):
        diagnostics = analyze_qiec_capabilities(qiec, target)
        assert [diagnostic.kind for diagnostic in diagnostics] == [
            "qiec:capability:named-parameter:identity"
        ]


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        ("schema_chart_parser", "qiec:capability:search:lp_parser__run"),
        ("term_autoencoder", "qiec:capability:neural-attachment:reconstruct"),
    ],
)
def test_host_runtime_capabilities_match_the_transpile_boundary(
    example: str,
    expected: str,
) -> None:
    """Every user-facing diagnostic surface selects the same entry root."""

    path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "examples"
        / "source"
        / f"{example}.qvr"
    )
    parsed = parse(path.read_text(), str(path))
    module = lower_qvr_to_qiec(parsed, module_name=example, file_path=str(path))
    assert [
        diagnostic.kind for diagnostic in analyze_qiec_capabilities(module, "pyro")
    ] == [expected]
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parsed, target="pyro")
    assert caught.value.kinds == [expected]


_PROGRAM_CALLING_PROGRAM = """\
object X : FinSet 2
object R : FinSet 2
program sub : X -> R * R
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- Normal(loc=a, scale=1.0)
    return (a, b)
program main : X -> R
    sample (u, v) <- sub
    let s = u + v
    sample w <- Normal(loc=s, scale=1.0)
    return w
export main
"""


@pytest.mark.parametrize("target", sorted(available_targets()))
def test_a_program_drawn_from_is_planned_in_place(target: str) -> None:
    """A program another program draws from runs in place on every
    target: the names it returns take the pattern's, so its draws are
    the sites ``u`` and ``v`` of the root, and no host computation is
    defined for it."""
    emitted = transpile(parse(_PROGRAM_CALLING_PROGRAM), target=target).decode("utf-8")
    assert "qiec_sub" not in emitted
    assert "_draw_u_v" not in emitted
    for site in ("u", "v", "w"):
        assert site in emitted


_PREFIXED = _PROGRAM_CALLING_PROGRAM.replace(
    "    sample (u, v) <- sub\n    let s = u + v\n",
    "    sample pair <- sub\n    let s = pair[0] + pair[1]\n",
)


@pytest.mark.parametrize("target", ("pyro", "stan", "turing", "webppl", "church"))
def test_a_prefixed_site_is_spelled_with_the_target_separator(target: str) -> None:
    """A local of a program drawn under one name is the site ``pair$a``
    on the reference machine and ``pair__a`` in every emission, since
    no target admits ``$`` in an identifier."""
    emitted = transpile(parse(_PREFIXED), target=target).decode("utf-8")
    assert "$" not in emitted
    assert "pair__a" in emitted
    assert "pair__b" in emitted
