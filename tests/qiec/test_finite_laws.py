"""The finite checker against exact enumeration of authored state handlers."""

from __future__ import annotations

from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec.discharge import discharge_module
from quivers.qiec.laws import (
    FiniteLawCounterexample,
    FiniteLawLimit,
    FiniteLawUnsupported,
    check_finite_law,
    finite_continuation_type,
)
from quivers.qiec.module import QiecModule
from quivers.qiec.types import BOOL, INT, UNIT, product_type

STATE = """
effect BoolCell
    read : Unit -> Bool
    write : Bool -> Unit

instance cell : BoolCell

handler state for BoolCell : Bool -> (Bool * Bool) with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, current)
    read resumes 1 =>
        resume(current)
    write(v : Bool) resumes 1 =>
        resume() with (v)

handler constant for BoolCell : Bool -> (Bool * Bool) with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, current)
    read resumes 1 =>
        resume(current)
    write(v : Bool) resumes 1 =>
        resume()

handler late for BoolCell : Bool -> (Bool * Bool) with (previous : Bool, current : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return (x, current)
    read resumes 1 =>
        resume(previous)
    write(v : Bool) resumes 1 =>
        resume() with (current, v)

define lookup(pair : (Bool * Bool), answers : (Bool * Bool * Bool * Bool)) : Bool !{} =
    if pair[0] then
        if pair[1] then
            return answers[3]
        else
            return answers[2]
    else
        if pair[1] then
            return answers[1]
        else
            return answers[0]
"""

EXTRA = """
handler external for BoolCell : Bool -> Bool [coverage=total, implementation=foreign]
    read resumes 1
    write resumes 1

define foreign_read() : Bool !{} =
    handle cell with external in
        let value <- perform cell.read()
        return value

define delegated_foreign() : Bool !{} =
    foreign_read()

define residual_read() : Bool !{cell} =
    let value <- perform cell.read()
    return value

define traced_read(initial : Bool) : (Bool * Bool) !{} =
    with instance local : BoolCell in
        handle cell with state(initial) in
            residual_read()

define explicit_read(initial : Bool) : (Bool * Bool) !{} =
    return (initial, initial)

define computed_real() : Bool !{} =
    if 0.1 + 0.2 < 0.3 then
        return true
    else
        return false

define forever(value : Bool) : Bool !{} =
    forever(value)

define same_cell(answers : (Bool * Bool)) : Bool !{} =
    if answers[0] then
        return answers[0]
    else
        return false

define one_cell(answers : (Bool * Bool)) : Bool !{} =
    return answers[0]
"""


def test_discharged_trace_addresses_do_not_change_a_finite_law(
    module: QiecModule,
) -> None:
    rewritten = discharge_module(module).module
    for owner in (module, rewritten):
        decision = check_finite_law(owner, "traced_read", "explicit_read")
        assert decision.counterexample is None


def _equations(model: str, initial: str) -> str:
    return f"""
define {model}_observed(initial : Bool, value : Bool) : (Bool * Bool) !{{}} =
    handle cell with {model}({initial}) in
        perform cell.write(value)
        let observed <- perform cell.read()
        return observed

define {model}_expected(initial : Bool, value : Bool) : (Bool * Bool) !{{}} =
    handle cell with {model}({initial}) in
        perform cell.write(value)
        return value

define {model}_left(initial : Bool, value : Bool, answers : (Bool * Bool * Bool * Bool)) : Bool !{{}} =
    let result <- {model}_observed(initial, value)
    lookup(result, answers)

define {model}_right(initial : Bool, value : Bool, answers : (Bool * Bool * Bool * Bool)) : Bool !{{}} =
    let result <- {model}_expected(initial, value)
    lookup(result, answers)
"""


@pytest.fixture(scope="module")
def module() -> QiecModule:
    source = STATE + "".join(
        _equations(model, initial)
        for model, initial in (
            ("state", "initial"),
            ("constant", "initial"),
            ("late", "initial, initial"),
        )
    )
    compiler = Compiler(
        parse(source + EXTRA), module_name="finite-law", file_path="law.qvr"
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def _state_oracle(model: str) -> tuple[int, FiniteLawCounterexample | None]:
    """Enumerate total maps independently over state and a four-point domain."""
    pairs = tuple(product((False, True), repeat=2))
    cases = product(pairs, product((False, True), repeat=4))
    for index, ((initial, requested), values) in enumerate(cases, 1):
        continuation = dict(zip(pairs, values, strict=True))
        current = initial if model == "constant" else requested
        observed = requested if model == "state" else initial
        left, right = continuation[observed, current], continuation[requested, current]
        if left != right:
            return index, FiniteLawCounterexample(
                (initial, requested, values), left, right
            )
    return index, None


@pytest.mark.parametrize("model", ("state", "constant", "late"))
def test_finite_put_get_quantifies_over_total_continuations(
    module: QiecModule, model: str
) -> None:
    decision = check_finite_law(module, model + "_left", model + "_right")
    assert (decision.checked, decision.counterexample) == _state_oracle(model)
    if model == "state":
        assert decision.checked == 64 and decision.counterexample is None
    else:
        assert decision.counterexample is not None


def test_a_total_table_keeps_repeated_lookups_consistent(module: QiecModule) -> None:
    decision = check_finite_law(module, "same_cell", "one_cell")
    assert decision.checked == 4 and decision.counterexample is None


def test_table_type_agrees_with_the_checked_arity(module: QiecModule) -> None:
    left = next(item for item in module.computations if item.name == "state_left")
    assert finite_continuation_type((BOOL, BOOL), BOOL) == left.parameters[-1].type
    assert finite_continuation_type((), BOOL) == product_type(BOOL)
    assert finite_continuation_type(
        (UNIT, product_type(BOOL, UNIT)), UNIT
    ) == product_type(UNIT, UNIT)
    with pytest.raises(FiniteLawUnsupported, match="Int"):
        finite_continuation_type((INT,), BOOL)
    with pytest.raises(FiniteLawLimit, match="4 cells"):
        finite_continuation_type((BOOL, BOOL), BOOL, max_cells=3)


@pytest.mark.parametrize(
    ("name", "reason"),
    (
        ("foreign_read", "foreign handler"),
        ("delegated_foreign", "foreign handler"),
        ("residual_read", "empty residual effect row"),
        ("computed_real", "finite law fragment"),
    ),
)
def test_unverified_code_cannot_yield_a_positive_law_decision(
    module: QiecModule, name: str, reason: str
) -> None:
    with pytest.raises(FiniteLawUnsupported, match=reason):
        check_finite_law(module, name, name)


def test_resource_limits_are_undecided(module: QiecModule) -> None:
    with pytest.raises(FiniteLawLimit, match="64 assignments"):
        check_finite_law(module, "state_left", "state_right", max_assignments=63)
    with pytest.raises(FiniteLawLimit, match="fuel exhausted"):
        check_finite_law(module, "forever", "forever", fuel=16)
