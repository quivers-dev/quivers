"""Integer lexemes retain exact values through each program boundary."""

import pytest
import torch

from quivers.dsl import loads, lower_qvr_to_qiec, module_to_source, parse
from quivers.dsl.ast_nodes import DrawArgScalar, LetExprLiteral, ProgramDecl
from quivers.dsl.ast_nodes.program_steps import atom_to_draw_arg
from quivers.dsl.program_templates import draw_argument_expression
from quivers.dsl.step_resolution import ResolvedDist, _atom_to_text
from quivers.qiec import run_named
from quivers.transpile.ir import IRArgNumber


@pytest.mark.parametrize("number", [2**24 + 1, 2**53, 2**53 + 1, 2**63 - 1, -(2**63)])
def test_integer_expression_parse_emit_and_reference_are_exact(number):
    source = f"define exact() : Int !{{}} =\n    return {number}\n"
    rendered = module_to_source(parse(source))
    assert str(number) in rendered
    assert module_to_source(parse(rendered)) == rendered
    assert run_named(lower_qvr_to_qiec(parse(rendered)), "exact").value == number


@pytest.mark.parametrize("number", [2**53 + 1, 2**63 - 1])
def test_draw_wire_and_template_actuals_preserve_large_integer(number):
    argument = atom_to_draw_arg(number)
    assert isinstance(argument, DrawArgScalar)
    assert isinstance(argument.value, int) and argument.value == number
    expression = draw_argument_expression(argument)
    assert isinstance(expression, LetExprLiteral)
    assert expression.integral and expression.value == number
    resolved = ResolvedDist(
        family="Binomial", args=(number, 0.5), original_morphism_name="Binomial"
    )
    assert resolved.args[0] == number and isinstance(resolved.args[0], int)
    assert _atom_to_text(number) == str(number)
    lowered = IRArgNumber(value=number)
    assert isinstance(lowered.value, int) and lowered.value == number


@pytest.mark.parametrize("number", [2**53 + 1, 2**63 - 1])
@pytest.mark.parametrize(
    "surface", ["inline", "let", "program", "python", "computation"]
)
def test_trial_bound_rejects_exact_authored_integer_before_rounding(number, surface):
    if surface == "inline":
        body = f"    observe y : Row <- Binomial(total_count={number}, probs=0.5)\n    return y\n"
        declarations = ""
    elif surface == "let":
        body = f"    let n = {number}\n    observe y : Row <- Binomial(total_count=n, probs=0.5)\n    return y\n"
        declarations = ""
    elif surface == "computation":
        declarations = f"define count() : Int !{{}} =\n    return {number}\n"
        body = "    let n <- count()\n    observe y : Row <- Binomial(total_count=n, probs=0.5)\n    return y\n"
    else:
        declarations = "program part(n : Nat) : Row -> Row\n    observe y : Row <- Binomial(total_count=n, probs=0.5)\n    return y\n"
        if surface == "program":
            body = f"    sample z <- part({number})\n    return z\n"
        else:
            with pytest.raises(ValueError, match="2\\*\\*53"):
                compiled = loads(
                    "object Row : FinSet 1\n" + declarations + "export part\n"
                )
                compiled.part(number)
            return
    source = (
        "object Row : FinSet 1\n"
        + declarations
        + "program main : Row -> Row\n"
        + body
        + "export main\n"
    )
    with pytest.raises(ValueError, match="2\\*\\*53"):
        compiled = loads(source)
        # Computed values are checked by the native family at execution.
        compiled.morphism.log_joint(torch.zeros(1, 1), {"y": torch.zeros(1)})


def test_integer_draw_literal_roundtrip_keeps_its_ast_numeric_kind():
    number = 2**53 + 1
    source = f"object Row : FinSet 1\nprogram main : Row -> Row\n    observe y : Row <- Binomial(total_count={number}, probs=0.5)\n    return y\nexport main\n"
    module = parse(module_to_source(parse(source)))
    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    assert str(number) in module_to_source(module)
    assert program.draws


@pytest.mark.parametrize("number", [4, 2**53 + 1])
def test_static_extent_in_distribution_value_position_has_a_located_type_error(number):
    source = f"""object Row : FinSet 1
program part[n : Nat] : Row -> Row
    observe y : Row <- Binomial(total_count=n, probs=0.5)
    return y
define applied = part[{number}]()
export applied
"""
    with pytest.raises(ValueError, match="static index 'n'.*runtime.*\\(n : Nat\\)"):
        loads(source)
