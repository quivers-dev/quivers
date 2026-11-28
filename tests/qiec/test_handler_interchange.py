"""Exhaust finite value observations of opposite authored handler stacks."""

from itertools import product
from pathlib import Path

import pytest

from quivers.dsl import Compiler, parse
from quivers.dsl import loads as load_source
from quivers.qiec import FiniteHandlerInterchange, check_finite_handler_interchange
from quivers.qiec.execution import run_named
from quivers.qiec.laws import FiniteLawLimit, FiniteLawUnsupported
from quivers.qiec.module import QiecModule
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.numbers import host_value


_DOC = Path(__file__).resolve().parents[2] / "docs/api/qiec/effects.md"


def _source() -> str:
    section = _DOC.read_text().split("## Finite handler interchange", 1)[1]
    return section.split("\n```qvr\n", 1)[1].split("\n```", 1)[0] + "\n"


def _module(source: str) -> QiecModule:
    module = load_source(source).qiec
    assert isinstance(module, QiecModule)
    return module


def _statement(module: QiecModule, left: str, right: str) -> FiniteHandlerInterchange:
    names = {entry.name: entry.id for entry in module.computations}
    return FiniteHandlerInterchange(left=names[left], right=names[right])


def test_independent_cells_agree_on_all_inputs_and_round_trip():
    module = _module(_source())
    law = _statement(module, "cells_lr", "cells_rl")
    decoded = loads(dumps(law))
    assert isinstance(decoded, FiniteHandlerInterchange) and decoded == law
    decoded_module = loads(dumps(module))
    assert isinstance(decoded_module, QiecModule)
    decision = check_finite_handler_interchange(decoded_module, decoded)
    assert decision.counterexample is None and decision.checked == 8
    for arguments in product((False, True), repeat=3):
        _, initial_right, written = arguments
        for name in ("cells_lr", "cells_rl"):
            value = host_value(run_named(decoded_module, name, arguments).value)
            assert value is (initial_right == written)
    with pytest.raises(FiniteLawLimit, match="8 assignments"):
        check_finite_handler_interchange(decoded_module, decoded, max_assignments=7)


def test_choice_and_state_have_a_separating_assignment():
    module = _module(_source())
    law = _statement(module, "state_first", "choice_first")
    decision = check_finite_handler_interchange(module, law)
    assert decision.counterexample is not None
    witness = decision.counterexample
    assert (witness.arguments, witness.left, witness.right) == ((False,), True, False)


@pytest.mark.parametrize(
    ("changed", "needle", "replacement", "diagnostic"),
    (
        (
            "cells_rl",
            "handle left_cell with state(a)",
            "handle left_cell with state(b)",
            "change a handler installation",
        ),
        ("cells_rl", "two_cells(new)", "two_cells(a)", "same named call"),
    ),
)
def test_equal_return_values_do_not_validate_a_different_statement(
    changed, needle, replacement, diagnostic
):
    prefix, suffix = _source().split(f"define {changed}", 1)
    mutant = _module(
        prefix + f"define {changed}" + suffix.replace(needle, replacement, 1)
    )
    with pytest.raises(FiniteLawUnsupported, match=diagnostic):
        check_finite_handler_interchange(
            mutant, _statement(mutant, "cells_lr", changed)
        )


def test_compiler_plan_keeps_each_interchange_status_and_refusal():
    compiler = Compiler(parse(_source()), module_name="finite-interchange")
    module = compiler.qiec_module
    assert isinstance(module, QiecModule)
    statements = (
        _statement(module, "cells_lr", "cells_rl"),
        _statement(module, "state_first", "choice_first"),
        _statement(module, "cells_lr", "state_first"),
    )
    working = compiler.effect_plan("cells_lr", interchanges=statements)
    assert tuple(report.status for report in working.interchanges) == (
        "checked",
        "refuted",
        "unresolved",
    )
    assert (
        working.interchanges[2].reason
        == "interchange sides need identical named inputs"
    )
    bounded = compiler.effect_plan(
        "cells_lr", interchanges=statements, max_assignments=7
    )
    assert tuple(report.status for report in bounded.interchanges) == (
        "unresolved",
        "refuted",
        "unresolved",
    )
