"""Checked weighted reading queries and their response likelihoods."""

from __future__ import annotations

import json
import math
import sys

import didactic.api as dx
import pytest
import torch

from quivers.cli import main
from quivers.core.algebras import MAX_PLUS
from quivers.dsl import loads as load_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.dsl.qiec_tooling import target_diagnostics
from quivers.inference import MCMC, NUTSKernel
from quivers.qiec import deduction_outcomes
from quivers.qiec.checking import KernelError
from quivers.qiec.execution import (
    ExecutionFailure,
    RuntimeConfiguration,
    RuntimeSelection,
    run_named,
)
from quivers.qiec.foreign import DeductionOutcomes
from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
from quivers.qiec.module import QiecModule, validate_module
from quivers.qiec.numbers import NumberDomain
from quivers.qiec.program_runtime import (
    PARAMS_HANDLER,
    foreign_handler_configuration,
    joint_module,
    program_entry,
    run_program,
    static_arguments_for,
)
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.terms import Bind, Call, Var
from quivers.qiec.types import IndexLiteral, IndexVariable
from quivers.transpile import UnsupportedConstruct, transpile


GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb]
    atoms span, A, B, C, D, S, T, same, inverse, lex
    rule first : span(I, K, A, lex), span(K, J, D, lex) |- span(I, J, S, same)
    rule second : span(I, K, B, lex), span(K, J, D, lex) |- span(I, J, T, same)
    rule third : span(I, K, C, lex), span(K, J, D, lex) |- span(I, J, S, inverse)
    lexicon
        "x" : A = lex #[learnable]
        "x" : B = lex #[learnable]
        "x" : C = lex #[learnable]
        "y" : D = lex
"""

QUERY = """\
define mixture(rows : Mix__Outcomes, a : Real, b : Real) : Real !{} =
    case rows motive => Real
        Mix__outcome_end =>
            return 0.0
        Mix__outcome_entry(key, weight, tail) =>
            let label : String <- Mix__show(key)
            let rest : Real <- mixture(tail, a, b)
            if label == "same" then
                return rest + exp(weight) * a
            else
                return rest + exp(weight) * b

program example : Token -> Result
    let outcomes = parse_normalized_outcomes(Mix, sentence, "form")
    let answer <- mixture(outcomes, a, b)
    score log_score = log(answer)
    return log_score

program raw : Token -> Result
    let outcomes = parse_outcomes(Mix, sentence, "form")
    let answer <- mixture(outcomes, a, b)
    return answer

export example
export raw
"""

CARDINALITY = """\
define cardinality(rows : Mix__Outcomes) : Real !{} =
    case rows motive => Real
        Mix__outcome_end =>
            return 0.0
        Mix__outcome_entry(key, weight, tail) =>
            let rest : Real <- cardinality(tail)
            return rest + 1.0

program keys : Token -> Result
    let outcomes = parse_outcomes(Mix, sentence, "form")
    let size <- cardinality(outcomes)
    return size
"""

PRODUCTION_GRAMMAR = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb, normalize=lhs]
    atoms span, A, B, C, D, S, T, lex
    rule first : span(I, K, A), span(K, J, D) |- span(I, J, S)
    rule second : span(I, K, B), span(K, J, D) |- span(I, J, T)
    rule third : span(I, K, C), span(K, J, D) |- span(I, J, S)
    lexicon
        "x" : A = lex
        "x" : B = lex
        "z" : C = lex
        "y" : D = lex
"""

PRODUCTION_QUERY = (
    QUERY.replace('"same"', '"S"')
    .replace('"form"', '"category"')
    .replace(
        '    let outcomes = parse_normalized_outcomes(Mix, sentence, "category")',
        "    sample p <- Dirichlet(concentration=[2.0, 3.0])\n"
        '    let outcomes = parse_normalized_outcomes(Mix, sentence, "category", ("S", p))',
    )
)


def test_computed_tokens_feed_a_weighted_response_and_its_gradients() -> None:
    """A checked tokenizer selects the sentence before exactly one outcome query."""
    producer = """\
define choose_tokens(use_x : Bool) : Tensor[String]([2]) !{} =
    if use_x then
        return ["x", "y"]
    else
        return ["z", "y"]
"""
    query = PRODUCTION_QUERY.replace(
        "program example : Token -> Result\n"
        "    sample p <- Dirichlet(concentration=[2.0, 3.0])",
        "program example(p : Tensor[Real]([2]), use_x : Bool) : Token -> Result\n"
        "    let sentence <- choose_tokens(use_x)",
    )
    source = PRODUCTION_GRAMMAR + "\n" + producer + "\n" + query
    compiled = load_source(source)
    entry = next(item for item in compiled.entry_points() if item.name == "example")
    assert {parameter.name for parameter in entry.parameters} == {
        "p",
        "use_x",
        "a",
        "b",
    }
    assert entry.statics == ()
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert restored == module

    p = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    responses = torch.tensor([0.8, 0.3], dtype=p.dtype, requires_grad=True)
    for round_tripped in (False, True):
        for use_x in (True, False):
            data = {"p": p, "use_x": use_x, "a": responses[0], "b": responses[1]}
            if round_tripped:
                result = run_program(restored, "example", data=data, sites={})
            else:
                result = compiled.run("example", data=data)
                calls = [
                    event.detail["computation"]
                    for event in result.result.trace
                    if event.event == "call.entered"
                    and event.detail["computation"]
                    in ("choose_tokens", "Mix__weighted_outcomes")
                ]
                assert calls == ["choose_tokens", "Mix__weighted_outcomes"]

            # x y has two proofs, with relative weights p[0]/sum(p) and 1.
            # z y has one proof with weight p[1]/sum(p), which cancels.
            proof = p[0 if use_x else 1] / p.sum()
            if use_x:
                mass = proof * responses[0] + responses[1]
                total = proof + 1.0
            else:
                mass = proof * responses[0]
                total = proof
            expected = mass.log() - total.log()
            torch.testing.assert_close(
                result.log_joint, expected, atol=1e-12, rtol=1e-12
            )
            actual = torch.autograd.grad(result.log_joint, (p, responses))
            wanted = torch.autograd.grad(expected, (p, responses))
            for found, reference in zip(actual, wanted, strict=True):
                torch.testing.assert_close(found, reference, atol=1e-12, rtol=1e-12)


def test_computed_tokens_reuse_a_symbolic_sentence_extent() -> None:
    source = (
        GRAMMAR
        + "\n"
        + QUERY[: QUERY.index("program example")]
        + """
program generic[n : Nat](given : Tensor[String]([n])) : Token -> Result
    let sentence = given
    let outcomes = parse_outcomes(Mix, sentence, "form")
    let answer <- mixture(outcomes, a, b)
    return answer

export generic
"""
    )
    compiled = load_source(source)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    body = next(item.body for item in module.computations if item.name == "generic")
    assert isinstance(body, Bind)
    assert body.binder.name == "sentence"
    assert isinstance(body.then, Bind)
    call = body.then.first
    assert isinstance(call, Call)
    assert call.name == "Mix__outcomes"
    assert call.static_arguments == (IndexVariable(name="n", sort=NAT),)
    assert call.arguments[0] == Var(local=body.binder)

    responses = {"a": 0.8, "b": 0.3}
    short = {"given": ("x",), **responses}
    long = {"given": ("x", "y"), **responses}
    assert compiled.run("generic", data=short).value == pytest.approx(3 * 0.3)
    assert compiled.run("generic", data=long).value == pytest.approx(2 * 0.8 + 0.3)
    with pytest.raises(ExecutionFailure, match="invalid shape evidence") as error:
        compiled.run("generic", data=short, static_arguments=("n=2",))
    assert error.value.diagnostic.code == "qiec-run-config"


@pytest.mark.parametrize(
    ("source_steps", "code", "message"),
    (
        (
            '    let sentence = [1, 2]\n    let result = parse_outcomes(Mix, sentence, "form")\n',
            "qiec-deduction-outcomes",
            "rank-one Tensor",
        ),
        (
            '    let sentence = [["x", "y"]]\n    let result = parse_outcomes(Mix, sentence, "form")\n',
            "qiec-deduction-outcomes",
            "rank-one Tensor",
        ),
        (
            '    let result = parse_outcomes(Mix, ["x", "y"], "form")\n',
            "qiec-deduction-outcomes",
            "named by a program binding",
        ),
        (
            '    let sentence = filter(["x", "y"], token -> token == "x")\n'
            '    let result = parse_outcomes(Mix, sentence, "form")\n',
            "qiec-collection:filter:dynamic-shape",
            "data-dependent output length",
        ),
        (
            '    let result = parse_outcomes(Mix, sentence, "form")\n    let sentence = ["x", "y"]\n',
            "qiec-program",
            "already bound in program",
        ),
    ),
)
def test_computed_deduction_input_requires_a_supported_binding_and_shape(
    source_steps: str, code: str, message: str
) -> None:
    source = (
        GRAMMAR
        + "\nprogram invalid : Token -> Result\n"
        + source_steps
        + ("    let answer = 1.0\n    return answer\n")
    )
    with pytest.raises(QiecDiagnosticError, match=message) as error:
        lower_qvr_to_qiec(parse(source))
    assert error.value.code == code


def test_checked_response_score_and_gradient_equal_enumeration(monkeypatch) -> None:
    """Two different goal items contribute to one reading of the sentence."""
    module = lower_qvr_to_qiec(parse(GRAMMAR + "\n" + QUERY))
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    assert restored == module
    weights = torch.tensor([-0.1, -1.2, -0.7], dtype=torch.float64, requires_grad=True)
    responses = torch.tensor([0.8, 0.3], dtype=weights.dtype, requires_grad=True)
    parameters = {
        f"Mix.lex.{index}": value for index, value in enumerate(weights.unbind())
    }
    datum = {"sentence": ("x", "y"), "a": responses[0], "b": responses[1]}
    run = run_program(restored, "example", data=datum, sites={}, parameters=parameters)
    log_response = responses[[0, 0, 1]].log()
    expected = torch.logsumexp(weights + log_response, 0) - torch.logsumexp(weights, 0)
    assert torch.allclose(run.log_joint, expected, atol=1e-12, rtol=1e-12)
    actual_gradient = torch.autograd.grad(run.log_joint, (weights, responses))
    expected_gradient = torch.autograd.grad(expected, (weights, responses))
    for actual, wanted in zip(actual_gradient, expected_gradient, strict=True):
        assert torch.allclose(actual, wanted, atol=1e-12, rtol=1e-12)

    build = deduction_outcomes.build_forest
    chart = deduction_outcomes._chart_outcomes

    def drop_second_witness(presentation, axioms):
        kept = [axiom for axiom in axioms if dict(axiom.binding)["x2"] != "B"]
        return build(presentation, kept)

    def raw_instead_of_normalized(
        semantics, tokens, terms, weights, key, normalize, *, context
    ):
        return chart(semantics, tokens, terms, weights, key, False, context=context)

    mutants = (
        ("LOG_PROB", MAX_PLUS),
        ("build_forest", drop_second_witness),
        ("_chart_outcomes", raw_instead_of_normalized),
    )
    for name, operation in mutants:
        mutant_weights = weights.detach().requires_grad_()
        with monkeypatch.context() as patch:
            patch.setattr(deduction_outcomes, name, operation)
            mutant = run_program(
                restored,
                "example",
                data={"sentence": ("x", "y"), "a": 0.8, "b": 0.3},
                sites={},
                parameters={
                    f"Mix.lex.{index}": value
                    for index, value in enumerate(mutant_weights.unbind())
                },
            ).log_joint
        assert not torch.isclose(mutant, expected), name
        assert not torch.allclose(
            torch.autograd.grad(mutant, mutant_weights)[0], actual_gradient[0]
        ), name

    for a, b, mass in (
        (1.0, 0.0, weights[:2].exp().sum()),
        (0.0, 1.0, weights[2].exp()),
        (1.0, 1.0, weights.exp().sum()),
    ):
        found = run_program(
            restored,
            "raw",
            data={"sentence": datum["sentence"], "a": a, "b": b},
            sites={},
            parameters=parameters,
        )
        assert isinstance(found.value, torch.Tensor)
        assert torch.allclose(found.value, mass, atol=1e-12, rtol=1e-12)


def test_checked_outcome_trace_reports_the_executed_selector_and_mass(
    monkeypatch,
) -> None:
    items = CARDINALITY.replace('"form"', '"item"')
    module = loads(dumps(lower_qvr_to_qiec(parse(GRAMMAR + QUERY + items))))
    assert isinstance(module, QiecModule)
    (descriptor,) = [
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionOutcomes)
    ]
    weights = torch.tensor([-0.1, -1.2, -0.7], dtype=torch.float64, requires_grad=True)
    parameters = {f"Mix.lex.{i}": value for i, value in enumerate(weights.unbind())}
    data = {"sentence": ("x", "y"), "a": 0.8, "b": 0.3}
    observed = []
    response = weights.new_tensor([0.8, 0.8, 0.3])
    raw_total = (weights.exp() * response).sum()
    normalized_total = weights.softmax(0) @ response
    for name, selector, normalized, rows, expected in (
        ("raw", "form", False, 2, raw_total),
        ("example", "form", True, 2, normalized_total.log()),
        ("keys", "item", False, 3, weights.new_tensor(3.0)),
    ):
        observed.clear()
        datum = {"sentence": data["sentence"]} if name == "keys" else data
        result = run_program(
            module,
            name,
            data=datum,
            sites={},
            parameters=parameters,
            observer=observed.append,
        )
        torch.testing.assert_close(result.value, expected)
        if name != "keys":
            torch.testing.assert_close(
                torch.autograd.grad(result.value, weights)[0],
                torch.autograd.grad(expected, weights)[0],
            )
        (event,) = [event for event in observed if event.event == "deduction.outcomes"]
        assert event in result.result.trace
        detail = event.detail
        assert detail["presentation"] == str(descriptor.presentation.id)
        assert detail["observer"] == "weighted-outcome-rows"
        assert (detail["key"], detail["normalized"]) == (selector, normalized)
        assert detail["algorithm"] == "indexed-packed-forest-topological-fold"
        assert detail["recursion"] == "finite-acyclic"
        assert detail["algebra"] == "LogProb"
        assert detail["derivative"] == "autograd-on-finite-log-weights-at-fixed-support"
        assert detail["retention"] == "packed-forest-during-query"
        assert detail["goal_items"] == 3 and detail["positive_rows"] == rows
        work = dict(detail["construction_work"])
        # Four lexical occurrences and three binary rule instances.
        assert work["provenance_records"] == 4 + 3 * (1 + 2)
        assert work["hash_operations"] > 0 and work["key_comparisons"] > 0
        assert set(work) == {"provenance_records", "hash_operations", "key_comparisons"}
        assert "peak_bytes" in detail["unmeasured"]
        assert "backward_arithmetic" in detail["unmeasured"]

    def unmeasured(*args, **kwargs):
        pytest.fail("an untraced query must not collect its report")

    monkeypatch.setattr(deduction_outcomes, "_outcome_trace", unmeasured)
    extended, entry_name = joint_module(module, "raw")
    entry = program_entry(module, "raw")
    handlers = foreign_handler_configuration(module)
    handlers[PARAMS_HANDLER] = {"kind": "param", "values": parameters}
    runtime = RuntimeConfiguration(
        selections=(RuntimeSelection(provider="core", options={"handlers": handlers}),)
    )
    plain = run_named(
        extended,
        entry_name,
        tuple(data[parameter.name] for parameter in entry.parameters),
        static_arguments=static_arguments_for(entry, data, ()),
        runtime=runtime,
        record_trace=False,
    )
    assert plain.trace == ()
    torch.testing.assert_close(plain.value[0], raw_total)


def test_zero_mass_does_not_contaminate_a_finite_reading() -> None:
    """An absent reading has no weight, even with two structural witnesses."""
    module = lower_qvr_to_qiec(parse(GRAMMAR + "\n" + QUERY))
    weights = torch.tensor(
        [-torch.inf, -torch.inf, -2.0], dtype=torch.float32, requires_grad=True
    )
    parameters = {
        f"Mix.lex.{index}": value for index, value in enumerate(weights.unbind())
    }
    observed = {"sentence": ("x", "y"), "a": 0.8, "b": 0.3}
    run = run_program(
        module,
        "example",
        data=observed,
        sites={},
        parameters=parameters,
        numbers=NumberDomain.of(weights),
    )
    assert run.log_joint.dtype == weights.dtype
    assert run.log_joint.device == weights.device
    assert torch.allclose(run.log_joint, weights.new_tensor(0.3).log())
    (gradient,) = torch.autograd.grad(run.log_joint, weights)
    assert torch.equal(gradient, weights.new_zeros(3))


def test_normalization_requires_a_positive_finite_goal_mass() -> None:
    """An unrecognized sentence has empty raw outcomes and no conditional law."""
    module = lower_qvr_to_qiec(parse(GRAMMAR + "\n" + QUERY))
    data = {"sentence": ("z", "y"), "a": 0.8, "b": 0.3}
    raw = run_program(module, "raw", data=data, sites={})
    assert raw.value == 0.0
    with pytest.raises(ExecutionFailure, match="without positive goal mass"):
        run_program(module, "example", data=data, sites={})


def test_finite_log_products_do_not_silently_become_structural_zeros() -> None:
    module = lower_qvr_to_qiec(parse(GRAMMAR + "\n" + QUERY))
    with pytest.raises(ExecutionFailure, match="finite chart range"):
        run_program(
            module,
            "raw",
            data={"sentence": ("x", "y"), "a": 0.8, "b": 0.3},
            sites={},
            parameters={"Mix.lex.0": -1e308},
        )


def test_checked_model_joint_reaches_explicit_production_probabilities() -> None:
    """A Dirichlet site controls one rule that can derive the sentence."""
    source = PRODUCTION_GRAMMAR + "\n" + PRODUCTION_QUERY
    compiled = load_source(source)
    point = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    data = {"sentence": ("x", "y"), "a": 0.8, "b": 0.3}
    prior = torch.distributions.Dirichlet(point.new_tensor([2.0, 3.0]))
    probability = point[0] / point.sum()
    likelihood = (probability * data["a"] + data["b"]) / (1.0 + probability)
    expected = prior.log_prob(point) + likelihood.log()
    actual = compiled.morphism.log_joint(point.new_zeros(1), {**data, "p": point}).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, point)[0],
        torch.autograd.grad(expected, point)[0],
    )
    sampler = MCMC(
        NUTSKernel(max_tree_depth=2), num_warmup=2, num_samples=3, num_chains=1, seed=10
    )
    samples = sampler.run(compiled.morphism, point.new_zeros(1), data).samples["p"]
    assert samples.shape == (1, 3, 2)
    assert (samples > 0).all()
    torch.testing.assert_close(samples.sum(-1), torch.ones_like(samples[..., 0]))


def test_explicit_production_zero_is_a_valid_structural_endpoint() -> None:
    """The zeroed rule contributes no key, and the finite rule still scores."""
    query = PRODUCTION_QUERY.replace(
        "    sample p <- Dirichlet(concentration=[2.0, 3.0])",
        "    let p = [0.0, 1.0]",
    )
    module = lower_qvr_to_qiec(parse(PRODUCTION_GRAMMAR + "\n" + query))
    run = run_program(
        module,
        "example",
        data={"sentence": ("x", "y"), "a": 0.8, "b": 0.3},
        sites={},
    )
    assert run.log_joint == pytest.approx(torch.tensor(0.3).log().item())


@pytest.mark.parametrize(
    ("selector", "start", "expected"),
    [
        ("item", "", 3.0),
        ("category", "", 2.0),
        ("form", "", 2.0),
        ("item", ", start=S", 2.0),
    ],
)
def test_each_key_projects_and_groups_equal_structural_values(
    selector: str, start: str, expected: float
) -> None:
    """The same span can have two categories, two forms, and three items."""
    module = lower_qvr_to_qiec(
        parse(
            GRAMMAR.replace("semiring=LogProb]", f"semiring=LogProb{start}]")
            + "\n"
            + CARDINALITY.replace('"form"', f'"{selector}"')
        )
    )
    result = run_program(module, "keys", data={"sentence": ("x", "y")}, sites={})
    assert result.value == expected


def test_chart_fold_work_tracks_edges_instead_of_proofs(monkeypatch) -> None:
    """Two binary rules produce two choices at every node of every tree."""
    grammar = """\
object Token : FinSet 2
object Result : Real 1

deduction Mix : Token -> Token [semiring=LogProb]
    atoms span, S, same
    rule first : span(I, K, S, same), span(K, J, S, same) |- span(I, J, S, same)
    rule second : span(I, K, S, same), span(K, J, S, same) |- span(I, J, S, same)
    lexicon
        "x" : S = same
"""
    module = lower_qvr_to_qiec(parse(grammar + "\n" + QUERY))
    weight = deduction_outcomes.edge_weight
    visited = 0

    def count_edge(*args, **kwargs):
        nonlocal visited
        visited += 1
        return weight(*args, **kwargs)

    monkeypatch.setattr(deduction_outcomes, "edge_weight", count_edge)
    tokens = 13
    result = run_program(
        module, "raw", data={"sentence": ("x",) * tokens, "a": 1.0, "b": 0.0}, sites={}
    )
    splits = math.comb(tokens + 1, 3)
    assert visited == tokens + 2 * splits
    internal = tokens - 1
    trees = math.comb(2 * internal, internal) // (internal + 1)
    proofs = (2**internal) * trees
    assert proofs > 100_000 * visited
    assert result.value == pytest.approx(proofs, rel=1e-12)


def test_acyclic_unary_diamond_preserves_weights_and_gradients() -> None:
    """The two unary paths build fresh forms and coalesce at one goal item."""
    grammar = (
        GRAMMAR.replace("semiring=LogProb]", "semiring=LogProb, start=Goal]")
        .replace("same, inverse, lex", "same, inverse, lex, Left, Right, Goal, wrap")
        .replace(
            "    lexicon\n",
            "    rule via_left : span(I, J, S, same) |- span(I, J, Left(same), wrap(same))\n"
            "    rule via_right : span(I, J, S, same) |- span(I, J, Right, wrap(same))\n"
            "    rule left_goal : span(I, J, Left(L), wrap(L)) |- span(I, J, Goal, L)\n"
            "    rule right_goal : span(I, J, Right, wrap(L)) |- span(I, J, Goal, L)\n"
            "    rule inverse_goal : span(I, J, S, inverse) |- span(I, J, Goal, inverse)\n"
            "    lexicon\n",
        )
    )
    module = loads(dumps(lower_qvr_to_qiec(parse(grammar + "\n" + QUERY))))
    assert isinstance(module, QiecModule)
    declaration = next(
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionOutcomes)
    )
    assert declaration.unary_height == 2
    weights = torch.tensor([-0.4, -0.8, -1.2], dtype=torch.float64, requires_grad=True)
    parameters = {
        f"Mix.lex.{index}": weight for index, weight in enumerate(weights.unbind())
    }
    data = {"sentence": ("x", "y"), "a": 0.8, "b": 0.3}
    proof_weights = weights[[0, 0, 2]]
    likelihood = torch.tensor([0.8, 0.8, 0.3], dtype=weights.dtype).log()
    expected = torch.logsumexp(proof_weights + likelihood, 0) - torch.logsumexp(
        proof_weights, 0
    )
    run = run_program(module, "example", data=data, sites={}, parameters=parameters)
    torch.testing.assert_close(run.log_joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(run.log_joint, weights)[0],
        torch.autograd.grad(expected, weights)[0],
    )
    raw = run_program(
        module,
        "raw",
        data={**data, "a": 1.0, "b": 0.0},
        sites={},
        parameters=parameters,
    )
    assert isinstance(raw.value, torch.Tensor)
    torch.testing.assert_close(raw.value, 2 * weights[0].exp())


def test_entire_rule_telescope_is_grounded_by_premises() -> None:
    """An unused natural index would leave infinitely many groundings."""
    module = lower_qvr_to_qiec(parse(GRAMMAR + "\n" + QUERY))
    declaration = next(
        handler.foreign
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionOutcomes)
    )
    first, *rest = declaration.presentation.rules
    unused = IndexBinder(name="Unused", sort=NAT)
    rule = first.with_(telescope=(*first.telescope, unused))
    unchecked = declaration.presentation.with_(rules=(rule, *rest))
    with pytest.raises(
        dx.ValidationError, match="bind its entire telescope through its premises"
    ):
        declaration.with_(presentation=unchecked)


def test_private_deduction_sorts_are_scoped_and_outer_adapter_is_checked() -> None:
    """Two different private Term sorts coexist with a third public Term."""
    second = (
        GRAMMAR[GRAMMAR.index("deduction") :]
        .replace("Mix", "Other")
        .replace("inverse", "different")
    )
    other_keys = (
        CARDINALITY.replace("Mix", "Other")
        .replace("cardinality", "other_size")
        .replace("program keys", "program other_keys")
    )
    source = (
        "index Term = Z | Succ(Term)\n\n"
        + GRAMMAR
        + "\n"
        + second
        + "\n"
        + CARDINALITY
        + "\n"
        + other_keys
    )
    module = loads(dumps(lower_qvr_to_qiec(parse(source))))
    assert isinstance(module, QiecModule)
    data = {"sentence": ("x", "y")}
    for name in ("keys", "other_keys"):
        assert run_program(module, name, data=data, sites={}).value == 2.0
    assert tuple(sort.name for sort in module.index_sorts) == ("Term",)
    handler = next(
        handler
        for handler in module.handlers
        if isinstance(handler.foreign, DeductionOutcomes)
    )
    semantics = handler.foreign
    assert isinstance(semantics, DeductionOutcomes)
    private = semantics.presentation.index_sorts[0]
    assert isinstance(private, UserIndexSort)
    wrong_argument = IndexLiteral(value="same", sort=private)
    invalid = handler.with_(
        foreign=semantics.with_(
            item_type=semantics.item_type.with_(arguments=(wrong_argument,)),
        ),
    )
    with pytest.raises(KernelError, match="embedded index sort 'Term' disagrees"):
        validate_module(
            module.with_(
                handlers=tuple(
                    invalid if item == handler else item for item in module.handlers
                )
            )
        )


def test_cli_and_published_python_entry_run_the_checked_query(
    tmp_path, capsys, monkeypatch
) -> None:
    """Both published front ends reach the scored QVR program."""
    source = GRAMMAR + "\n" + QUERY
    path = tmp_path / "readings.qvr"
    path.write_text(source)
    arguments = [
        "qvr",
        "run",
        str(path),
        "example",
        "--json",
        "--data",
        'sentence=["x","y"]',
        "--data",
        "a=0.8",
        "--data",
        "b=0.3",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    assert main() == 0
    data = {"sentence": ("x", "y"), "a": 0.8, "b": 0.3}
    compiled = load_source(source)
    expected = (2 * 0.8 + 0.3) / 3
    assert json.loads(capsys.readouterr().out)["log_joint"] == pytest.approx(
        torch.tensor(expected).log().item()
    )
    assert compiled.run("example", data=data).log_joint == pytest.approx(
        torch.tensor(expected).log().item()
    )


def test_target_refusal_is_located_and_leaves_unrelated_exports_available() -> None:
    source = (
        GRAMMAR
        + """\

program takes_readings : Token -> Result
    let rows = parse_outcomes(Mix, sentence, "form")
    let answer = 1.0
    return answer

program unrelated : Token -> Result
    sample z <- Normal(loc=0.0, scale=1.0)
    return z
"""
    )
    active_source = source + "\nexport takes_readings\n"
    active = parse(active_source, file_path="queries.qvr")
    checked = lower_qvr_to_qiec(active, file_path="queries.qvr")
    diagnostics = target_diagnostics(checked, active, "stan")
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "qiec:capability:deduction-outcomes:takes_readings"
    assert diagnostics[0].line > 0
    assert "parse_outcomes(" in active_source.splitlines()[diagnostics[0].line - 1]
    with pytest.raises(UnsupportedConstruct) as refusal:
        transpile(active, target="stan")
    assert diagnostics[0].code in refusal.value.kinds

    independent = parse(source + "\nexport unrelated\n", file_path="queries.qvr")
    checked = lower_qvr_to_qiec(independent, file_path="queries.qvr")
    assert not target_diagnostics(checked, independent, "stan")
    result = transpile(independent, target="stan")
    assert b"normal" in result


@pytest.mark.parametrize(
    ("grammar", "message"),
    [
        (
            GRAMMAR.replace(
                "span(I, J, S, same)\n", "span(I, J, S, same) #[learnable]\n"
            ),
            "Free weights",
        ),
        (
            GRAMMAR.replace("span(K, J, D, lex)", "span(M, J, D, lex)", 1),
            "combine contiguous spans",
        ),
        (
            GRAMMAR.replace(
                "    lexicon\n",
                "    rule grows : span(I, J, S, L) |- span(I, J, S, wrap(L))\n"
                "    lexicon\n",
            ),
            "unary category heads must be acyclic",
        ),
        (
            GRAMMAR.replace(
                "    lexicon\n",
                "    rule unknown : span(I, J, S, L) |- span(I, J, L, lex)\n"
                "    lexicon\n",
            ),
            "needs fixed category heads",
        ),
        (
            GRAMMAR.replace("semiring=LogProb]", "semiring=LogProb, depth=0]"),
            "category depth bound",
        ),
    ],
)
def test_an_unsupported_rule_policy_is_refused_statically(
    grammar: str, message: str
) -> None:
    with pytest.raises(QiecDiagnosticError, match=message) as error:
        lower_qvr_to_qiec(parse(grammar + "\n" + QUERY))
    assert error.value.code == "qiec-deduction-outcomes"
