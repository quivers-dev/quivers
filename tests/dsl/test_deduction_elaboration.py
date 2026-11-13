"""Deductions elaborate to computations that agree with the agenda engine.

A ``deduction`` becomes a closed item family, a recursive derivation
computation searching through the module's ``Search`` effect, and an
entry that handles the search and the ``Weight`` instance to answer the
inside weight of the goal. Run on the reference machine, the entry gives
the number the agenda-driven chart gives, in each semiring, with and
without learned weights, and a program that calls the deduction scores
that number.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import cast

import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.qiec import QiecModule, dumps, loads
from quivers.qiec.execution import run_named
from quivers.qiec.module import validate_module
from quivers.qiec.program_runtime import (
    deduction_item,
    program_entry,
    run_deduction,
    run_program,
)
from quivers.qiec.types import render_static

_EXAMPLES = Path(__file__).resolve().parents[2] / "docs" / "examples" / "source"

AB = """\
object Term : FinSet 16

deduction AB : Term -> Term [semiring=SEMIRING, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the  #[learnable]
        "dog"  : N          = dog  #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]
"""

AMBIGUOUS = """\
object Term : FinSet 8

deduction Attach : Term -> Term [semiring=SEMIRING, start=S, depth=6]
    atoms S, NP, VP, PP, V, P, N, span, saw, man, telescope, with, a
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule vp_v_np : span(I, K, V), span(K, J, NP) |- span(I, J, VP) #[learnable]
    rule vp_vp_pp : span(I, K, VP), span(K, J, PP) |- span(I, J, VP) #[learnable]
    rule np_np_pp : span(I, K, NP), span(K, J, PP) |- span(I, J, NP) #[learnable]
    rule pp_p_np : span(I, K, P), span(K, J, NP) |- span(I, J, PP) #[learnable]
    lexicon
        "I"         : NP = a
        "saw"       : V  = saw
        "man"       : NP = man
        "telescope" : NP = telescope
        "with"      : P  = with
"""

PROVER = """\
object Term : FinSet 8

deduction Prover : Term -> Term [semiring=SEMIRING, start=Claim, depth=3]
    atoms Claim, Every, Some, Nonempty, dog, animal, mammal
    rule barbara : Claim(Every(P, Q)), Claim(Every(Q, R)) |- Claim(Every(P, R)) #[learnable]
    rule darii : Claim(Some(P, Q)), Claim(Every(Q, R)) |- Claim(Some(P, R)) #[learnable]
    rule ex_import : Claim(Every(P, Q)), Claim(Nonempty(P)) |- Claim(Some(P, Q)) #[learnable]
"""

FIT = """\
object Term : FinSet 16
object LogWeight : Real 1

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the  #[learnable]
        "dog"  : N          = dog  #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]

program fit : Term -> LogWeight
    let chart = parse(AB, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit
"""


def _module(source: str, semiring: str = "LogProb") -> QiecModule:
    """Lower a source with the deduction's semiring filled in.

    Parameters
    ----------
    source : str
        QVR text with ``SEMIRING`` where the semiring goes.
    semiring : str
        The semiring.

    Returns
    -------
    QiecModule
        The checked module.
    """
    return lower_qvr_to_qiec(
        parse(source.replace("SEMIRING", semiring)), file_path="deduction.qvr"
    )


def _classic(source: str, name: str, semiring: str = "LogProb"):
    """Compile a source with the classic compiler and return one deduction.

    Parameters
    ----------
    source : str
        QVR text with ``SEMIRING`` where the semiring goes.
    name : str
        The deduction's name.
    semiring : str
        The semiring.

    Returns
    -------
    DeductionSystem
        The agenda-driven system.
    """
    program = Compiler(parse(source.replace("SEMIRING", semiring))).compile()
    return program.deductions[name]


def _render_binding(value: object) -> str:
    """Render a classic binding value the way the elaboration keys weights.

    Parameters
    ----------
    value : object
        An atom pair, a constructor tuple, or an integer.

    Returns
    -------
    str
        The rendering.
    """
    if isinstance(value, tuple):
        if value[0] == "atom":
            return cast(str, value[1])
        head = cast(str, value[0])
        return f"{head}({','.join(_render_binding(item) for item in value[1:])})"
    return str(value)


def test_a_deduction_becomes_a_checked_family_and_computations() -> None:
    module = _module(AB)
    validate_module(module)
    assert loads(dumps(module)) == module
    assert [family.name for family in module.families] == ["AB__Item"]
    constructors = {
        constructor.name: [render_static(field.type) for field in constructor.fields]
        for constructor in module.constructors
    }
    assert constructors["AB__span"] == ["Int", "Int", "AB__Item"]
    assert constructors["AB__Fwd"] == ["AB__Item", "AB__Item"]
    assert constructors["AB__S"] == []
    names = [computation.name for computation in module.computations]
    assert names == [
        "AB__eq",
        "AB__show",
        "AB__goal",
        "AB__fail",
        "AB__depth",
        "AB__axiom",
        "AB__derive",
        "AB__run",
    ]
    entry = next(item for item in module.computations if item.name == "AB__run")
    assert render_static(entry.type.result) == "LogWeight"
    assert [instance.name for instance in module.instances] == [
        "params",
        "AB__choice",
        "AB__weight",
    ]
    assert [handler.name for handler in module.handlers] == [
        "AB__search_logprob",
        "AB__collect_logprob",
    ]


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [("the dog runs", 0.0), ("the dog", -math.inf), ("dog runs the", -math.inf)],
)
def test_the_reference_machine_scores_the_goal_like_the_agenda(
    sentence: str, expected: float
) -> None:
    module = _module(AB)
    tokens = sentence.split()
    run = run_deduction(module, "AB", tokens=tokens)
    assert run.weight == expected
    classic = float(_classic(AB, "AB")(tokens).goal_weight())
    assert run.weight == classic


def test_an_ambiguous_sentence_sums_over_its_derivations() -> None:
    module = _module(AMBIGUOUS)
    tokens = ["I", "saw", "man", "with", "telescope"]
    run = run_deduction(module, "Attach", tokens=tokens)
    assert run.weight == pytest.approx(math.log(2.0))
    # The chart also derives the prefix "I saw man" as an S, but the
    # deduction's answer is the whole sentence's: the agenda's goal
    # weight is the chart's weight at that one item.
    view = _classic(AMBIGUOUS, "Attach")(tokens)
    whole = view.weight(("span", 0, 5, ("atom", "S")))
    assert run.weight == pytest.approx(float(whole))
    assert float(view.goal_weight()) == pytest.approx(math.log(2.0))
    assert [item for item, _ in view.goal_items] == [("span", 0, 5, ("atom", "S"))]


@pytest.mark.parametrize(
    ("semiring", "expected"),
    [("Viterbi", 0.0), ("Boolean", True), ("Counting", 2)],
)
def test_every_semiring_aggregates_the_derivations_its_own_way(
    semiring: str, expected: object
) -> None:
    module = _module(AMBIGUOUS, semiring)
    tokens = ["I", "saw", "man", "with", "telescope"]
    run = run_deduction(module, "Attach", tokens=tokens)
    assert run.weight == expected
    view = _classic(AMBIGUOUS, "Attach", semiring)(tokens)
    whole = view.weight(("span", 0, 5, ("atom", "S")))
    if semiring == "Boolean":
        assert bool(whole) is True
    else:
        assert float(whole) == pytest.approx(float(expected))  # type: ignore[arg-type]


def test_lexical_weights_are_read_by_the_classic_parameter_names() -> None:
    module = _module(AB)
    classic = _classic(AB, "AB")
    tokens = ["the", "dog", "runs"]
    values = {"lex_weight_0": -0.3, "lex_weight_1": 0.7, "lex_weight_2": -1.1}
    with torch.no_grad():
        for name, parameter in classic.named_parameters():
            stem = name.rsplit(".", 1)[-1]
            if stem in values:
                parameter.fill_(values[stem])
    run = run_deduction(
        module,
        "AB",
        tokens=tokens,
        parameters={
            f"AB.lex.{index}": value for index, value in enumerate(values.values())
        },
    )
    assert run.weight == pytest.approx(sum(values.values()))
    assert run.weight == pytest.approx(float(classic(tokens).goal_weight()))


def test_rule_weights_are_keyed_by_their_bindings() -> None:
    module = _module(AB)
    classic = _classic(AB, "AB")
    tokens = ["the", "dog", "runs"]
    classic(tokens)
    store: dict[str, float] = {}
    with torch.no_grad():
        for rule_name, parameters in (
            (name, module_) for name, module_ in classic._rule_module.named_children()
        ):
            for key, parameter in parameters.items():
                parameter.fill_(0.25 if rule_name == "rule_fwd_app" else -0.5)
    # The key covers the category variables only; the span positions
    # I, J, K never enter it, on either evaluator.
    bindings = {
        "fwd_app": {"X": ("atom", "NP"), "Y": ("atom", "N")},
        "bwd_app": {"X": ("atom", "S"), "Y": ("atom", "NP")},
    }
    classic_keys = {
        f"{rule_name}.{key}"
        for rule_name, parameters in classic._rule_module.named_children()
        for key in parameters.keys()
    }
    assert classic_keys == {
        "rule_fwd_app.X_TSatom_SNPE_Y_TSatom_SNE",
        "rule_bwd_app.X_TSatom_SSE_Y_TSatom_SNPE",
    }
    for rule_name, binding in bindings.items():
        suffix = "|".join(_render_binding(binding[name]) for name in sorted(binding))
        store[f"AB.rule.{rule_name}:{suffix}"] = (
            0.25 if rule_name == "fwd_app" else -0.5
        )
    assert set(store) == {"AB.rule.fwd_app:NP|N", "AB.rule.bwd_app:S|NP"}
    run = run_deduction(module, "AB", tokens=tokens, parameters=store)
    assert run.weight == pytest.approx(-0.25)
    assert run.weight == pytest.approx(float(classic(tokens).goal_weight()))


def test_a_deduction_without_a_lexicon_takes_its_axioms() -> None:
    module = _module(PROVER)

    def claim(quantifier: str, left: str, right: str):
        """Build a claim item.

        Parameters
        ----------
        quantifier : str
            ``Every`` or ``Some``.
        left : str
            The restrictor atom.
        right : str
            The scope atom.

        Returns
        -------
        RuntimeConstructor
            The item.
        """
        return deduction_item(
            module,
            "Prover",
            "Claim",
            deduction_item(
                module,
                "Prover",
                quantifier,
                deduction_item(module, "Prover", left),
                deduction_item(module, "Prover", right),
            ),
        )

    axioms = [
        claim("Every", "dog", "mammal"),
        claim("Every", "mammal", "animal"),
        deduction_item(
            module,
            "Prover",
            "Claim",
            deduction_item(
                module, "Prover", "Nonempty", deduction_item(module, "Prover", "dog")
            ),
        ),
    ]
    run = run_deduction(module, "Prover", axioms=axioms, axiom_weights=[0.0, 0.0, 0.0])
    # Derivations of a claim within three steps: each axiom alone, the
    # two syllogisms, and the subalternation from each of the three
    # universals.
    classic = _classic(PROVER, "Prover")
    classic_axioms = [
        (("Claim", ("Every", ("atom", "dog"), ("atom", "mammal"))), torch.tensor(0.0)),
        (
            ("Claim", ("Every", ("atom", "mammal"), ("atom", "animal"))),
            torch.tensor(0.0),
        ),
        (("Claim", ("Nonempty", ("atom", "dog"))), torch.tensor(0.0)),
    ]
    view = classic(classic_axioms)
    counted = _module(PROVER, "Counting")
    count = run_deduction(
        counted,
        "Prover",
        axioms=[
            deduction_item(
                counted,
                "Prover",
                "Claim",
                deduction_item(
                    counted,
                    "Prover",
                    "Every",
                    deduction_item(counted, "Prover", "dog"),
                    deduction_item(counted, "Prover", "mammal"),
                ),
            ),
            deduction_item(
                counted,
                "Prover",
                "Claim",
                deduction_item(
                    counted,
                    "Prover",
                    "Every",
                    deduction_item(counted, "Prover", "mammal"),
                    deduction_item(counted, "Prover", "animal"),
                ),
            ),
            deduction_item(
                counted,
                "Prover",
                "Claim",
                deduction_item(
                    counted,
                    "Prover",
                    "Nonempty",
                    deduction_item(counted, "Prover", "dog"),
                ),
            ),
        ],
        axiom_weights=[1, 1, 1],
    )
    assert count.weight == 7
    assert run.weight == pytest.approx(math.log(7.0))
    assert float(view.goal_weight()) == pytest.approx(math.log(7.0))


def test_a_program_scores_the_weight_a_deduction_answers() -> None:
    module = _module(FIT)
    entry = program_entry(module, "fit")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("sentence", "data")
    ]
    assert [binder.name for binder in entry.telescope] == ["sentence_extent"]
    computation = next(item for item in module.computations if item.name == "fit")
    assert {entry.effect.name for entry in computation.type.effects.entries} == {
        "Score",
        "Param",
    }
    store = {"AB.lex.0": 0.5, "AB.lex.1": 0.25, "AB.lex.2": -1.0}
    run = run_program(
        module,
        "fit",
        data={"sentence": ("the", "dog", "runs")},
        sites={},
        parameters=store,
    )
    assert run.log_joint == pytest.approx(-0.25)
    assert run.value == pytest.approx(-0.25)


@pytest.mark.parametrize(
    "example",
    [
        "ccg",
        "custom_rules",
        "type_logical",
        "quantifier_scope",
        "multimodal_tlg",
        "wmcfg",
        "pcfg",
        "montague_nli",
    ],
)
def test_every_gallery_deduction_elaborates(example: str) -> None:
    path = _EXAMPLES / f"{example}.qvr"
    module = lower_qvr_to_qiec(parse(path.read_text()), file_path=str(path))
    validate_module(module)
    assert loads(dumps(module)) == module
    assert module.gap == ""


def test_schema_chart_parser_elaborates_as_a_checked_deduction() -> None:
    path = _EXAMPLES / "schema_chart_parser.qvr"
    module = lower_qvr_to_qiec(
        parse(path.read_text()),
        module_name="schema_chart_parser",
        file_path=str(path),
    )
    validate_module(module)
    assert loads(dumps(module)) == module
    assert [family.name for family in module.families] == ["lp_parser__Item"]
    assert [computation.name for computation in module.computations] == [
        "lp_parser__eq",
        "lp_parser__show",
        "lp_parser__goal",
        "lp_parser__fail",
        "lp_parser__axiom",
        "lp_parser__derive",
        "lp_parser__run",
    ]
    assert [handler.name for handler in module.handlers] == [
        "lp_parser__search_logprob",
        "lp_parser__collect_logprob",
    ]


def test_schema_chart_parser_agrees_with_the_classic_chart() -> None:
    path = _EXAMPLES / "schema_chart_parser.qvr"
    parsed = parse(path.read_text())
    compiler = Compiler(
        parsed,
        module_name="schema_chart_parser",
        file_path=str(path),
    )
    classic_parser = compiler.compile().morphism
    with torch.no_grad():
        classic_parser.axiom.lexicon_logits.zero_()
    classic = float(classic_parser(torch.tensor([1, 3])).detach())

    module = compiler.qiec_module
    assert module is not None
    n_categories = classic_parser.rule_system.n_categories
    n_terminals = classic_parser.axiom.lexicon_logits.shape[0]
    uniform = -math.log(n_categories)
    parameters = {
        f"lp_parser.lex.{index}": uniform for index in range(n_categories * n_terminals)
    }
    run = run_deduction(
        module,
        "lp_parser",
        tokens=("dog", "sleeps"),
        parameters=parameters,
        fuel=1_000_000,
    )
    assert run.weight == pytest.approx(classic)


def test_builtin_schema_parser_elaborates_through_the_same_core() -> None:
    source = """\
object Atoms : {N, S}
object Cat : FreeResiduated(Atoms)
object Token : {word}

define builtin_parser = parser(rules=[evaluation], terminal=Token, start=S, depth=1)
export builtin_parser
"""
    module = lower_qvr_to_qiec(
        parse(source),
        module_name="builtin_schema_parser",
        file_path="builtin_schema_parser.qvr",
    )
    validate_module(module)
    assert loads(dumps(module)) == module
    assert any(
        computation.name == "builtin_parser__run" for computation in module.computations
    )


@pytest.mark.parametrize(
    ("example", "name", "sentence"),
    [
        ("pcfg", "PCFG", "the dog runs"),
        ("ccg", "CCG", "the cat sleeps"),
        ("type_logical", "Lambek", "Kim admires Lee"),
        ("montague_nli", "Montague", "every dog is an animal"),
    ],
)
def test_gallery_grammars_agree_with_the_agenda(
    example: str, name: str, sentence: str
) -> None:
    path = _EXAMPLES / f"{example}.qvr"
    compiler = Compiler(
        parse(path.read_text()), module_name=example, file_path=str(path)
    )
    program = compiler.compile()
    module = compiler.qiec_module
    assert module is not None
    tokens = sentence.split()
    classic = float(program.deductions[name](tokens).goal_weight())
    run = run_deduction(module, name, tokens=tokens)
    assert run.weight == pytest.approx(classic)


@pytest.mark.parametrize(
    ("edit", "fragment"),
    [
        (("semiring=SEMIRING", "semiring=Tropical"), "the semirings are"),
        (
            (
                "rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y))",
                "rule bwd_app : span(I, J, Y), span(K, J, Bwd(X, Y))",
            ),
            "over its conclusion's whole span beside other premises",
        ),
        (
            (
                "|- span(I, J, X) #[learnable]\n    rule bwd_app",
                "|- span(I, J, Z) #[learnable]\n    rule bwd_app",
            ),
            "which no premise binds",
        ),
    ],
)
def test_forms_outside_the_elaboration_are_reported(
    edit: tuple[str, str], fragment: str
) -> None:
    source = AB.replace(*edit)
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert fragment in captured.value.message


def test_an_axiom_source_morphism_leaves_the_axioms_to_the_caller() -> None:
    source = PROVER.replace("depth=3]", "depth=3, axioms=lexer]").replace(
        "\ndeduction", "\nmorphism lexer : Term -> Term [role=latent]\ndeduction"
    )
    module = _module(source)
    entry = next(item for item in module.computations if item.name == "Prover__run")
    assert [render_static(parameter.type) for parameter in entry.parameters] == [
        "Tensor[Prover__Item]([n])",
        "Tensor[LogWeight]([n])",
    ]


def test_an_authored_multi_shot_handler_resumes_at_distinct_addresses() -> None:
    module = lower_qvr_to_qiec(
        parse(
            textwrap.dedent(
                """\
                effect Pick
                    pick : Int -> Int

                instance chooser : Pick

                handler both for Pick : Int -> Int [coverage=total, forwards=none, implementation=authored]
                    return x =>
                        return x
                    pick(value : Int) resumes omega =>
                        let low <- resume(value)
                        let high <- resume(value + 1)
                        return low + high

                define searched() : Int !{} =
                    handle chooser with both in
                        let first <- perform chooser.pick(1)
                        let second <- perform chooser.pick(10)
                        return first * second
                """
            )
        ),
        file_path="search.qvr",
    )
    validate_module(module)
    result = run_named(module, "searched")
    # (1 * 10 + 1 * 11) + (2 * 10 + 2 * 11) = 63.
    assert result.value == 63
    addresses = [
        event.detail["address"]
        for event in result.trace
        if event.event == "operation.requested"
    ]
    assert len(addresses) == 3
    assert len(set(addresses)) == 3
    shots = [
        event.detail["shot"]
        for event in result.trace
        if event.event == "resumption.invoked"
    ]
    assert shots == [0, 0, 1, 1, 0, 1]
