"""Typed outcome keys preserve mass without revisiting chart proofs."""

from __future__ import annotations

import math

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.dsl.qiec_tooling import target_diagnostics
from quivers.qiec.checking import KernelError
from quivers.qiec.execution import ExecutionFailure
from quivers.qiec.foreign import DeductionOutcomeGrouping
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.types import REAL
from quivers.transpile import UnsupportedConstruct, transpile


GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb]
    atoms span, A, B, C, D, E, F, S, T, U, same, other, lex
    rule first : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, S, same)
    rule second : span(I, K, B, lex), span(K, J, D, lex) |- span(I, J, S, same)
    rule third : span(I, K, C, lex), span(K, J, D, lex) |- span(I, J, T, same)
    rule fourth : span(I, K, E, lex), span(K, J, D, lex) |- span(I, J, S, other)
    rule ghost : span(I, K, F, lex), span(K, J, D, lex) |- span(I, J, U, other)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
        "x" : C = lex #[learnable]
        "x" : E = lex #[learnable]
        "x" : F = lex #[learnable]
        "y" : D = lex

family Key : Type
    constructor Truth : Bool -> Key
    constructor Number : Int -> Key
    constructor Envelope : Key -> Key

define classify(item : Mix__Item) : Key !{} =
    let truth = construct Truth(true) as Key
    let agreement = construct Envelope(truth) as Key
    let text <- Mix__show(item)
    if text == "span(0,2,S,same)" then
        return agreement
    else
        if text == "span(0,2,T,same)" then
            return agreement
        else
            if text == "span(0,2,S,other)" then
                let number = construct Number(1) as Key
                return construct Envelope(number) as Key
            else
                let number = construct Number(2) as Key
                return construct Envelope(number) as Key

define response(key : Key, yes : Real, no : Real) : Real !{} =
    case key motive => Real
        Truth(value) =>
            return yes
        Number(value) =>
            if value == 1 then
                return no
            else
                return 0.0
        Envelope(value) =>
            let rate <- response(value, yes, no)
            return rate

define mass(rows : Mix__classify__Outcomes, yes : Real, no : Real) : Real !{} =
    case rows motive => Real
        Mix__classify__outcome_end =>
            return 0.0
        Mix__classify__outcome_entry(key, weight, tail) =>
            let rate <- response(key, yes, no)
            let rest <- mass(tail, yes, no)
            return exp(weight) * rate + rest

program normalized : Token -> Result
    let readings = parse_normalized_outcomes(Mix, sentence, classify)
    let likelihood <- mass(readings, yes, no)
    score response = log(likelihood)
    return response

program raw : Token -> Result
    let readings = parse_outcomes(Mix, sentence, classify)
    let answer <- mass(readings, yes, no)
    return answer
"""


def test_packed_goals_map_once_then_group_and_normalize_exact_typed_keys() -> None:
    compiled = load_source(GRAMMAR + "\nexport normalized\nexport raw\n")
    data = {"sentence": ("x", "y"), "yes": 0.8, "no": 0.3}
    public = compiled.run("normalized", data=data)
    assert public.log_joint == pytest.approx(math.log((3 * 0.8 + 0.3) / 5))
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert restored == module

    weights = torch.tensor(
        [-0.2, -1.1, -0.6, -0.1, -math.inf], dtype=torch.float64, requires_grad=True
    )
    rates = torch.tensor([0.8, 0.3], dtype=weights.dtype, requires_grad=True)
    parameters = {f"Mix.lex.{i}": weight for i, weight in enumerate(weights.unbind())}
    result = run_program(
        restored,
        "normalized",
        data={**data, "yes": rates[0], "no": rates[1]},
        sites={},
        parameters=parameters,
    )
    proof_rates = rates[[0, 0, 0, 1]].log()
    expected = torch.logsumexp(weights[:4] + proof_rates, 0) - torch.logsumexp(
        weights[:4], 0
    )
    torch.testing.assert_close(result.log_joint, expected, atol=1e-12, rtol=1e-12)
    for actual, reference in zip(
        torch.autograd.grad(result.log_joint, (weights, rates)),
        torch.autograd.grad(expected, (weights, rates)),
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)

    # Two proofs of the same goal share its key computation. Distinct S
    # and T goals still require calls, and the zero ghost goal needs none.
    calls = [
        event.detail["computation"]
        for event in result.result.trace
        if event.event == "call.entered"
        and event.detail["computation"] in ("Mix__outcomes", "classify")
    ]
    assert calls == ["Mix__outcomes", "classify", "classify", "classify"]
    _assert_grouping_trace(result, normalized=True)
    for yes, no, oracle in (
        (1.0, 0.0, weights[:3].exp().sum()),
        (0.0, 1.0, weights[3].exp()),
        (1.0, 1.0, weights[:4].exp().sum()),
    ):
        raw = run_program(
            restored,
            "raw",
            data={**data, "yes": yes, "no": no},
            sites={},
            parameters=parameters,
        )
        assert isinstance(raw.value, torch.Tensor)
        torch.testing.assert_close(raw.value, oracle, atol=1e-12, rtol=1e-12)
        _assert_grouping_trace(raw, normalized=False)


def _assert_grouping_trace(run, *, normalized: bool) -> None:
    (packed, grouped) = [
        event
        for event in run.result.trace
        if event.event in {"deduction.outcomes", "deduction.grouped_outcomes"}
    ]
    assert packed.event == "deduction.outcomes"
    assert (packed.detail["key"], packed.detail["normalized"]) == ("item", False)
    assert (packed.detail["goal_items"], packed.detail["positive_rows"]) == (4, 3)
    assert grouped.event == "deduction.grouped_outcomes"
    assert grouped.detail["observer"] == "weighted-typed-key-rows"
    assert grouped.detail["algorithm"] == "hash-grouped-structural-keys"
    assert grouped.detail["key_type"] == "Key"
    assert grouped.detail["normalized"] is normalized
    assert grouped.detail["input_rows"] == packed.detail["positive_rows"]
    assert grouped.detail["distinct_keys"] == grouped.detail["positive_rows"] == 2
    assert "key_hash_and_equality" in grouped.detail["unmeasured"]


def test_keyed_query_checks_empty_support_after_mapping() -> None:
    compiled = load_source(GRAMMAR + "\nexport normalized\nexport raw\n")
    data = {"sentence": ("missing",), "yes": 0.8, "no": 0.3}
    raw = compiled.run("raw", data=data)
    assert raw.value == 0.0
    (grouped,) = [
        event.detail
        for event in raw.result.trace
        if event.event == "deduction.grouped_outcomes"
    ]
    assert (
        grouped["input_rows"]
        == grouped["distinct_keys"]
        == grouped["positive_rows"]
        == 0
    )
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        compiled.run("normalized", data=data)


@pytest.mark.parametrize(
    ("key_type", "key"), (("Bool", "true"), ("Int", "1"), ("String", '"one"'))
)
def test_keys_can_be_discrete_scalars(key_type: str, key: str) -> None:
    source = (
        GRAMMAR[: GRAMMAR.index("family Key")]
        + f"""\
define classify(item : Mix__Item) : {key_type} !{{}} =
    return {key}

define count(rows : Mix__classify__Outcomes) : Real !{{}} =
    case rows motive => Real
        Mix__classify__outcome_end =>
            return 0.0
        Mix__classify__outcome_entry(value, weight, tail) =>
            let rest <- count(tail)
            return 1.0 + rest

program groups : Token -> Result
    let readings = parse_outcomes(Mix, sentence, classify)
    let answer <- count(readings)
    return answer

export groups
"""
    )
    result = load_source(source).run("groups", data={"sentence": ("x", "y")})
    assert result.value == 1.0


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        (GRAMMAR.replace("sentence, classify", "sentence, missing"), "monomorphic"),
        (
            GRAMMAR.replace("classify(item : Mix__Item)", "classify(item : Int)"),
            "taking one Mix__Item argument",
        ),
        (
            GRAMMAR.replace(
                "classify(item : Mix__Item) : Key", "classify(item : Mix__Item) : Real"
            ),
            "exact structural equality",
        ),
        (
            GRAMMAR.replace("constructor Number : Int", "constructor Number : Real"),
            "exact structural equality",
        ),
    ],
)
def test_typed_keys_refuse_unknown_incompatible_and_inexact_projectors(
    source: str, diagnostic: str
) -> None:
    with pytest.raises(QiecDiagnosticError, match=diagnostic) as error:
        load_source(source)
    assert error.value.code == "qiec-deduction-outcomes"


def test_a_closed_selector_cannot_hide_a_locally_handled_effect() -> None:
    declaration = """\
effect Probe
    inspect : Key -> Key

instance probe : Probe

handler unchanged for Probe : Key -> Key [coverage=total, forwards=none, implementation=authored]
    return key =>
        return key
    inspect(key : Key) resumes 1 =>
        resume(key)

define intercepted(key : Key) : Key !{} =
    handle probe with unchanged in
        let answer <- perform probe.inspect(key)
        return answer

"""
    source = GRAMMAR.replace(
        "define classify", declaration + "define classify"
    ).replace(
        "        return agreement\n",
        "        let result <- intercepted(agreement)\n        return result\n",
        1,
    )
    with pytest.raises(
        QiecDiagnosticError, match="'intercepted' installs an effect handler"
    ) as error:
        load_source(source)
    assert error.value.code == "qiec-deduction-outcomes"


def test_deserialized_reducer_checks_its_key_type() -> None:
    module = load_source(GRAMMAR + "\nexport raw\n").qiec
    assert isinstance(module, QiecModule)
    handlers = tuple(
        handler.with_(foreign=handler.foreign.with_(key_type=REAL))
        if isinstance(handler.foreign, DeductionOutcomeGrouping)
        else handler
        for handler in module.handlers
    )
    with pytest.raises(KernelError, match="exact structural equality"):
        validate_module(module.with_(handlers=handlers))


def test_typed_key_target_refusal_leaves_an_independent_export_available() -> None:
    source = (
        GRAMMAR
        + """\

program independent : Token -> Result
    sample z <- Normal(loc=0.0, scale=1.0)
    return z
"""
    )
    active_source = source + "\nexport normalized\n"
    active = parse(active_source, file_path="typed_keys.qvr")
    checked = lower_qvr_to_qiec(active, file_path="typed_keys.qvr")
    diagnostic = next(
        item
        for item in target_diagnostics(checked, active, "stan")
        if item.code == "qiec:capability:deduction-outcomes:normalized"
    )
    assert (
        "parse_normalized_outcomes(" in active_source.splitlines()[diagnostic.line - 1]
    )
    with pytest.raises(UnsupportedConstruct, match="deduction-outcomes"):
        transpile(active, target="stan")
    independent = parse(source + "\nexport independent\n", file_path="typed_keys.qvr")
    checked = lower_qvr_to_qiec(independent, file_path="typed_keys.qvr")
    assert not any(
        ":deduction-outcomes:" in item.code
        for item in target_diagnostics(checked, independent, "stan")
    )
    assert b"normal" in transpile(independent, target="stan")
