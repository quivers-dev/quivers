"""Check nominal effect equations on actual handlers and against spoofed receipts."""

from dataclasses import replace
from itertools import product

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    ComputationId,
    EffectEquation,
    EffectId,
    FiniteLawDecision,
    FiniteLawLimit,
    FiniteLawUnsupported,
    QiecModule,
    check_finite_effect_law,
    check_finite_law,
    dumps,
    loads,
)

_SOURCE = """
effect Cell
    read_cell : Unit -> Bool
    write_cell : Bool -> Unit

instance cell : Cell
instance other_cell : Cell

handler state for Cell : Bool -> (Bool * Bool) with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return answer =>
        return (answer, current)
    read_cell resumes 1 =>
        resume(current)
    write_cell(v : Bool) resumes 1 =>
        resume() with (v)

handler constant for Cell : Bool -> (Bool * Bool) with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return answer =>
        return (answer, current)
    read_cell resumes 1 =>
        resume(current)
    write_cell(v : Bool) resumes 1 =>
        resume()

define write_read(value : Bool, answers : (Bool * Bool)) : Bool !{cell} =
    perform cell.write_cell(value)
    let observed <- perform cell.read_cell()
    if observed then
        return answers[1]
    else
        return answers[0]

define write_immediate(value : Bool, answers : (Bool * Bool)) : Bool !{cell} =
    perform cell.write_cell(value)
    if value then
        return answers[1]
    else
        return answers[0]

define ignore_choice(value : Bool, answers : (Bool * Bool)) : Bool !{cell} =
    perform cell.write_cell(value)
    return answers[0]

define delegated(value : Bool, answers : (Bool * Bool)) : Bool !{cell} =
    write_immediate(value, answers)

define conditional_read(value : Bool, answers : (Bool * Bool)) : Bool !{cell} =
    if value then
        return answers[1]
    else
        let observed <- perform cell.read_cell()
        if observed then
            return answers[1]
        else
            return answers[0]

define read_and_true() : Bool !{cell} =
    let ignored <- perform cell.read_cell()
    return true

define pure_true() : Bool !{} =
    return true

define write_and_true() : Bool !{cell} =
    perform cell.write_cell(true)
    return true
"""


def _compile(source: str = _SOURCE, name: str = "cell-effect-laws") -> QiecModule:
    compiler = Compiler(parse(source), module_name=name, file_path="equations.qvr")
    assert compiler.qiec_module is not None
    return compiler.qiec_module


@pytest.fixture(scope="module")
def module() -> QiecModule:
    return _compile()


def _equation(module: QiecModule) -> EffectEquation:
    instance = next(item.entry for item in module.instances if item.name == "cell")
    effect = next(item for item in module.effects if item.ref.id == instance.effect.id)
    roots = {item.name: item.id for item in module.computations}
    return EffectEquation(
        effect.ref.id,
        "write-read",
        instance.instance,
        tuple(item.id for item in effect.operations),
        roots["write_read"],
        roots["write_immediate"],
    )


def test_checks_all_values_of_the_actual_handler_and_continuation(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    state, constant = module.handlers
    lawful = check_finite_effect_law(module, equation, state.id)
    assert lawful.certified
    assert lawful.result.checked == 16
    assert lawful.handler == state.id
    assert lawful.answer_type == state.output_type
    assert len(lawful.parameter_types) == 3
    assert lawful.parameter_types[-1] == state.parameters[0].type

    broken = check_finite_effect_law(module, equation, constant.id)
    assert not broken.certified
    witness = broken.result.counterexample
    assert witness is not None
    value, answer_cells, initial = witness.arguments
    assert type(value) is bool
    assert isinstance(answer_cells, tuple)
    assert type(initial) is bool
    assert witness.left == (answer_cells[initial], initial)
    assert witness.right == (answer_cells[value], initial)
    independent = [
        (write, cells, state)
        for write, cells, state in product(
            (False, True), product((False, True), repeat=2), (False, True)
        )
        if cells[state] != cells[write]
    ]
    assert witness.arguments == independent[0]


def test_same_handler_names_and_ids_do_not_convey_changed_bodies(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    previous = check_finite_effect_law(module, equation, module.handlers[0].id)
    altered = _compile(_SOURCE.replace("resume() with (v)", "resume()"))
    assert altered.handlers[0].id == previous.handler
    assert _equation(altered) == equation
    with pytest.raises(FiniteLawUnsupported, match="evidence does not match"):
        check_finite_effect_law(altered, equation, previous.handler, prior=previous)
    broken = check_finite_effect_law(altered, equation, previous.handler)
    assert not broken.certified
    assert broken.fingerprint != previous.fingerprint


def test_same_computation_names_and_ids_do_not_convey_changed_bodies(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    handler = module.handlers[0].id
    prior = check_finite_effect_law(module, equation, handler)
    altered = _compile(
        _SOURCE.replace(
            "if value then\n        return answers[1]",
            "if value then\n        return answers[0]",
        )
    )
    assert _equation(altered) == equation
    assert altered.handlers == module.handlers
    with pytest.raises(FiniteLawUnsupported, match="evidence does not match"):
        check_finite_effect_law(altered, equation, handler, prior=prior)
    assert not check_finite_effect_law(altered, equation, handler).certified


def test_prior_evidence_checks_the_equation_handler_and_finite_domain(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    state, constant = module.handlers
    good = check_finite_effect_law(module, equation, state.id)
    unrelated = next(
        item.id for item in module.computations if item.name == "ignore_choice"
    )
    revised = replace(equation, right=unrelated)
    for wanted, handler, receipt in (
        (revised, state.id, good),
        (equation, constant.id, good),
        (equation, state.id, replace(good, parameter_types=())),
    ):
        with pytest.raises(FiniteLawUnsupported, match="evidence does not match"):
            check_finite_effect_law(module, wanted, handler, prior=receipt)
    assert not check_finite_effect_law(module, revised, state.id).certified


def test_forged_success_is_recomputed_from_actual_nominal_subject(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    constant = module.handlers[1]
    refuted = check_finite_effect_law(module, equation, constant.id)
    forged = replace(refuted, result=FiniteLawDecision(checked=16, counterexample=None))
    assert (
        check_finite_effect_law(module, equation, constant.id, prior=forged) == refuted
    )


def test_nominal_effect_instance_operations_and_computations_are_checked(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    state = module.handlers[0]
    foreign_effect = EffectId.derive("an-effect-with-the-same-display-name", "Cell")
    other_instance = next(
        item.entry.instance for item in module.instances if item.name == "other_cell"
    )
    unrelated = ComputationId.derive("unrelated", "write_read")
    for invalid, reason in (
        (replace(equation, effect=foreign_effect), "nominal instance"),
        (replace(equation, instance=other_instance), "incompatible row"),
        (replace(equation, operations=equation.operations[:1]), "performed operations"),
        (replace(equation, left=unrelated), "unknown computation"),
    ):
        with pytest.raises(FiniteLawUnsupported, match=reason):
            check_finite_effect_law(module, invalid, state.id)
    delegated = next(
        item.id for item in module.computations if item.name == "delegated"
    )
    with pytest.raises(FiniteLawUnsupported, match="unsupported Call"):
        check_finite_effect_law(module, replace(equation, right=delegated), state.id)


def test_same_subject_binds_across_canonical_module_roundtrip(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    state = module.handlers[0]
    proven = check_finite_effect_law(module, equation, state.id)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert check_finite_effect_law(restored, equation, state.id, prior=proven) == proven
    with pytest.raises(FiniteLawLimit, match="16 assignments"):
        check_finite_effect_law(
            restored, equation, state.id, prior=proven, max_assignments=15
        )


def test_operation_set_includes_requests_from_both_branches(module: QiecModule) -> None:
    equation = _equation(module)
    branch = next(
        item.id for item in module.computations if item.name == "conditional_read"
    )
    effect = next(item for item in module.effects if item.ref.id == equation.effect)
    write = next(item.id for item in effect.operations if item.name == "write_cell")
    incomplete = replace(equation, left=branch, operations=(write,))
    with pytest.raises(FiniteLawUnsupported, match="performed operations"):
        check_finite_effect_law(module, incomplete, module.handlers[0].id)


def test_pure_side_still_runs_the_actual_handlers_return_clause(
    module: QiecModule,
) -> None:
    equation = _equation(module)
    effect = next(item for item in module.effects if item.ref.id == equation.effect)
    read = next(item.id for item in effect.operations if item.name == "read_cell")
    ids = {item.name: item.id for item in module.computations}
    pure = replace(
        equation,
        name="ignore-read",
        operations=(read,),
        left=ids["read_and_true"],
        right=ids["pure_true"],
    )
    actual = check_finite_effect_law(module, pure, module.handlers[0].id)
    assert actual.certified
    assert actual.answer_type == module.handlers[0].output_type
    assert actual.result.checked == 2


def test_pure_side_does_not_execute_the_row_witness(module: QiecModule) -> None:
    equation = _equation(module)
    effect = next(item for item in module.effects if item.ref.id == equation.effect)
    write = next(item.id for item in effect.operations if item.name == "write_cell")
    ids = {item.name: item.id for item in module.computations}
    ignored = replace(
        equation,
        name="ignore-write",
        operations=(write,),
        left=ids["write_and_true"],
        right=ids["pure_true"],
    )
    checked = check_finite_effect_law(module, ignored, module.handlers[0].id)
    witness = checked.result.counterexample
    assert witness is not None
    assert (witness.arguments, witness.left, witness.right) == (
        (False,),
        (True, True),
        (True, False),
    )


def test_state_and_choice_exchange_changes_the_reference_result() -> None:
    shared = """
effect Store
    read_cell : Unit -> Bool
    write_cell : Bool -> Unit

effect Choice
    choose : Unit -> Bool

instance store : Store
instance choice : Choice

handler cell for Store : Bool -> Bool with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    read_cell resumes 1 =>
        resume(current)
    write_cell(v : Bool) resumes 1 =>
        resume() with (v)

handler alternatives for Choice : Bool -> Bool [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    choose resumes omega =>
        let first <- resume(false)
        let second <- resume(true)
        if first then
            return true
        else
            return second
"""
    branch = """
            let ignored <- perform choice.choose()
            let old <- perform store.read_cell()
            perform store.write_cell(true)
            return old
"""
    first = """
define state_inside() : Bool !{} =
    handle choice with alternatives in
        handle store with cell(false) in"""
    second = """
define state_outside() : Bool !{} =
    handle store with cell(false) in
        handle choice with alternatives in"""
    module = _compile(shared + first + branch + second + branch, "commutation-witness")
    comparison = check_finite_law(module, "state_inside", "state_outside")
    assert comparison.counterexample is not None
    witness = comparison.counterexample
    assert witness.arguments == ()
    assert (witness.left, witness.right) == (False, True)
