"""One checked constructor vocabulary for source rule and lexical forms."""

from math import log
from pathlib import Path

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec import DeductionOutcomes, QiecModule, dumps, loads, run_program

GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Forms : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, Box, On, Off, plain
    rule first : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain)
    rule second : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain)
"""
INLINE = """\
    lexicon
        "x" : A = Box(On) #[learnable]
        "x" : A = Box(Off) #[learnable]
        "x" : A = On #[learnable]
        "y" : B = plain #[learnable]
"""
PROGRAMS = """
define read_mass(rows : Forms__Outcomes) : Real !{} =
    case rows motive => Real
        Forms__outcome_end =>
            return log(0.0)
        Forms__outcome_entry(key, weight, tail) =>
            return weight

program scalar : Token -> Result
    let chart = parse(Forms, sentence)
    score answer = weight_value(chart)
    return answer

program packed : Token -> Result
    let rows = parse_outcomes(Forms, sentence, "item")
    let weight <- read_mass(rows)
    score answer = weight
    return answer

export scalar
export packed
"""


@pytest.mark.parametrize("file_lexicon", (False, True))
def test_closed_lexical_application_has_the_same_rank_and_mass_in_both_routes(
    tmp_path: Path, file_lexicon: bool
) -> None:
    lexicon = INLINE
    if file_lexicon:
        path = tmp_path / "forms.tsv"
        path.write_text("x\tA\tBox(On)\nx\tA\tBox(Off)\nx\tA\tOn\ny\tB\tplain\n")
        lexicon = f'    lexicon from "{path}" #[learnable]\n'
    compiled = load_source(GRAMMAR + lexicon + PROGRAMS)
    wire = dumps(compiled.qiec)
    module = loads(wire)
    assert isinstance(module, QiecModule) and dumps(module) == wire
    (chart,) = [
        item.foreign
        for item in module.handlers
        if isinstance(item.foreign, DeductionOutcomes)
    ]
    (term_sort,) = chart.presentation.index_sorts
    box = next(item for item in module.constructors if item.name == "Forms__Box")
    assert box.id == dict(chart.item_constructors)["Box"]
    assert term_sort.constructor_arity("Box") == len(box.fields) == 1
    assert box.fields[0].type == chart.item_type
    assert term_sort.constructor_arity("On") == 0

    # A single split gives three left occurrences and one right occurrence.
    # Two rule occurrences prove the same root for each left occurrence.
    logs = torch.tensor(
        (-0.4, -0.9, -0.2, -0.125), dtype=torch.float64, requires_grad=True
    )
    for _ in range(2):
        parameters = {
            f"Forms.lex.{index}": value for index, value in enumerate(logs.unbind())
        }
        expected = log(2) + logs[:3].logsumexp(0) + logs[3]
        gradient = torch.cat((logs[:3].softmax(0), logs.new_ones(1)))
        for name in ("scalar", "packed"):
            result = run_program(
                module,
                name,
                data={"sentence": ("x", "y")},
                sites={},
                parameters=parameters,
            )
            torch.testing.assert_close(
                result.log_joint, expected, rtol=1e-12, atol=1e-12
            )
            torch.testing.assert_close(
                torch.autograd.grad(result.log_joint, logs)[0],
                gradient,
                rtol=1e-12,
                atol=1e-12,
            )
        with torch.no_grad():
            logs.add_(0.2 * gradient)


def test_lexical_form_rank_conflicts_with_an_authored_rule() -> None:
    grammar = GRAMMAR.replace("span(I, K, A, F)", "span(I, K, A, Box(On, Off))", 1)
    with pytest.raises(
        QiecDiagnosticError, match="constructor 'Box' has conflicting arities 2 and 1"
    ):
        lower_qvr_to_qiec(parse(grammar + INLINE + PROGRAMS))
