"""Packed proof meanings reuse the ordinary program production table."""

from math import log

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.modeling import Model
from quivers.qiec import (
    REAL,
    QiecModule,
    RuleError,
    dumps,
    loads,
    run_program,
    validate_module,
)
from quivers.qiec.terms import Call, LiteralValue, TensorValue, Var
from tests.dsl.test_packed_interpretation_schema import _replace_body


GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Vote : Token -> Token [semiring=LogProb, start=S, normalize=lhs]
    atoms span, A, B, S, word
    rule no : span(I, J, A) |- span(I, J, S)
    rule yes : span(I, J, B) |- span(I, J, S)
    lexicon
        "x" : A = word
        "y" : A = word
        "x" : B = word
        "y" : B = word

interpret decision : Vote -> Bool
    axiom span = lexical
    rule no = reject
    rule yes = accept

define lexical(word : String) : Bool !{} =
    return false

define reject(value : Bool) : Bool !{} =
    return false

define accept(value : Bool) : Bool !{} =
    return true
"""

MASS = """\
define true_mass(rows : Vote__BooleanOutcomes) : Real !{} =
    case rows motive => Real
        Vote__boolean_outcome_end =>
            return 0.0
        Vote__boolean_outcome_entry(key, weight, tail) =>
            if key then
                return exp(weight)
            else
                true_mass(tail)
"""

PRIORS = """\
    sample p_s <- Dirichlet(concentration=[1.0, 1.0])
    sample p_a <- Dirichlet(concentration=[1.0, 1.0])
    sample p_b <- Dirichlet(concentration=[1.0, 1.0])
"""

BINDINGS = '("B", p_b), ("S", p_s), ("A", p_a)'

PACKED = f"""\
program fit : Token -> Result
{PRIORS}    let rows = parse_normalized_packed_boolean_outcomes(Vote, sentence, decision, {BINDINGS})
    let mass <- true_mass(rows)
    score observed = log(mass)
    return observed

export fit
"""

RAW = (
    PACKED.replace("program fit", "program raw")
    .replace("export fit", "export raw")
    .replace(
        "parse_normalized_packed_boolean_outcomes", "parse_packed_boolean_outcomes"
    )
)

EXPLICIT = (
    MASS.replace("Vote__BooleanOutcomes", "Vote__decision__Outcomes")
    .replace("Vote__boolean_outcome_", "Vote__decision__outcome_")
    .replace("true_mass", "explicit_mass")
    + "\n"
    + PACKED.replace("program fit", "program explicit")
    .replace("export fit", "export explicit")
    .replace("true_mass", "explicit_mass")
    .replace(
        "parse_normalized_packed_boolean_outcomes(Vote, sentence, decision",
        "parse_normalized_proof_outcomes(Vote, sentence, 4, decision",
    )
)

STORED = """\
program default_weights : Token -> Result
    let rows = parse_normalized_packed_boolean_outcomes(Vote, sentence, decision)
    let mass <- true_mass(rows)
    score observed = log(mass)
    return observed

export default_weights
"""

SOURCE = GRAMMAR + "\n" + MASS + "\n" + PACKED + RAW + EXPLICIT + STORED


def test_program_supplied_rule_and_lexical_probabilities_remain_live():
    compiled = load_source(SOURCE)
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule) and module == compiled.qiec
    data = {"sentence": ("x",)}
    assert compiled.run("default_weights", data=data).log_joint == pytest.approx(
        log(0.5)
    )

    named = ("p_s", "p_a", "p_b")
    logits = torch.tensor(
        [[-0.23, 0.41], [0.62, -0.37], [-0.31, 0.77]],
        dtype=torch.float64,
        requires_grad=True,
    )
    backend = TensorBackend()
    for phase in range(2):
        sites = dict(zip(named, logits.softmax(-1).unbind(), strict=True))
        lp_s, lp_a, lp_b = logits.log_softmax(-1).unbind()
        # The only proofs of x use A or B. The local choice S -> B has
        # probability p_s[1] and B's production of x has probability p_b[0].
        # The other proof uses p_s[0] * p_a[0]. Each uniform Dirichlet on
        # the two-simplex has density one, so its log density is zero.
        rejected, accepted = lp_s[0] + lp_a[0], lp_s[1] + lp_b[0]
        normalized = accepted - torch.logaddexp(rejected, accepted)
        assert abs((normalized - accepted).item()) > 0.1
        plain = compiled.run(
            "fit",
            data=data,
            sites={key: value.detach() for key, value in sites.items()},
        )
        torch.testing.assert_close(plain.log_joint, normalized)
        directions = {}
        for name, oracle in (
            ("fit", normalized),
            ("raw", accepted),
            ("explicit", normalized),
        ):
            result = run_program(
                module, name, data=data, sites=sites, distributions=backend
            )
            torch.testing.assert_close(result.log_joint, oracle, atol=1e-12, rtol=1e-12)
            gradient = torch.autograd.grad(result.log_joint, logits, retain_graph=True)[
                0
            ]
            expected = torch.autograd.grad(oracle, logits, retain_graph=True)[0]
            torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
            directions[name] = gradient
            if name != "explicit":
                calls = [
                    event.detail["computation"]
                    for event in result.result.trace
                    if event.event == "call.entered"
                ]
                assert calls.count("Vote__weighted_boolean_outcomes") == 1
                assert "Vote__boolean_outcomes" not in calls
        if phase == 0:
            with torch.no_grad():
                logits.add_(0.5 * directions["fit"])


def test_program_weights_keep_the_same_nominal_interpretation():
    weighted = Model(SOURCE, entry="fit")
    stored = Model(SOURCE, entry="default_weights")
    assert (
        weighted.schema.closed_interpretations == stored.schema.closed_interpretations
    )
    (ordinary,) = weighted.schema.closed_interpretations
    (dynamic,) = weighted.schema.packed_interpretations
    (fixed,) = stored.schema.packed_interpretations
    assert dynamic.interpretation == fixed.interpretation == ordinary.interpretation.id
    assert dynamic.computation != fixed.computation
    assert dynamic.query != fixed.query
    restored = loads(dumps(dynamic))
    restored.validate(weighted.checked_module, ordinary)
    ordinary.validate(weighted.checked_module)


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        (SOURCE.replace(", normalize=lhs", ""), "declare normalize=lhs"),
        (SOURCE.replace('("B", p_b), ', ""), "binds no probabilities for B"),
        (SOURCE.replace('("B", p_b), ', '("S", p_b), '), "binds 'S' twice"),
        (SOURCE.replace('("B", p_b), ', '("unknown", p_b), '), "unknown"),
        (
            SOURCE.replace(
                "sample p_s <- Dirichlet(concentration=[1.0, 1.0])",
                "sample p_s <- Dirichlet(concentration=[1.0, 1.0, 1.0])",
            ),
            "takes a probability vector",
        ),
        (
            SOURCE.replace(
                "sample p_s <- Dirichlet(concentration=[1.0, 1.0])",
                "sample p_s <- Normal(loc=[0.25, 0.75], scale=[1.0, 1.0])",
            ),
            "production probabilities lie on the simplex",
        ),
    ],
    ids=(
        "free-weights",
        "missing-group",
        "repeated-group",
        "wrong-label",
        "wrong-size",
        "wrong-support",
    ),
)
def test_packed_weights_share_normalization_refusals(source, diagnostic):
    with pytest.raises(QiecDiagnosticError, match=diagnostic):
        load_source(source)


def test_weighted_companion_refuses_substituting_the_program_production_table():
    model = Model(SOURCE, entry="fit")
    (ordinary,) = model.schema.closed_interpretations
    (packed,) = model.schema.packed_interpretations

    def substitute(terminal):
        assert isinstance(terminal, Call)
        *arguments, production_table = terminal.arguments
        assert isinstance(production_table, Var)
        replacement = TensorValue(
            items=tuple(LiteralValue(value=0.5, type=REAL) for _ in range(6)),
            result_type=production_table.local.type,
        )
        return terminal.with_(arguments=(*arguments, replacement))

    forged = _replace_body(model.checked_module, packed.computation, substitute)
    validate_module(forged)
    with pytest.raises(RuleError, match="or forward the production table"):
        packed.validate(forged, ordinary)
