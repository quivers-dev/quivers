"""Applied static proof indices retain each child when exposed in QVR."""

from collections import Counter

import didactic.api as dx
import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.qiec import QiecModule, dumps, loads, run_program
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.foreign import DeductionProofs
from quivers.qiec.kinds import NAT, UserIndexSort
from quivers.qiec.types import IndexConstructor, IndexLiteral, TypeApplication

SOURCE = """\
object Token : FinSet 2
object Result : Real 1

deduction Nested : Token -> Token [semiring=LogProb, start=S]
    atoms span, A, B, S, Pair, Box, On, Off, plain
    rule exact : span(I, K, A, Pair(Box(On), Box(On), Box(Off))), span(K, J, B, plain) |- span(I, J, S, plain)
    rule general : span(I, K, A, F), span(K, J, B, plain) |- span(I, J, S, plain)
    lexicon
        "x" : A = Pair(Box(On), Box(On), Box(Off)) #[learnable]
        "x" : A = Pair(Box(Off), Box(On), Box(On)) #[learnable]
        "x" : A = Pair(Box(On), Box(Off), Box(On)) #[learnable]
        "y" : B = plain

program listed : Token -> Result
    let proofs = parse_proofs(Nested, sentence, 4)
    return proofs

export listed
"""


def _form(sort: UserIndexSort, names: tuple[str, ...]) -> IndexConstructor:
    return IndexConstructor(
        name="Pair",
        sort=sort,
        arguments=tuple(
            IndexConstructor(
                name="Box", sort=sort, arguments=(IndexLiteral(value=name, sort=sort),)
            )
            for name in names
        ),
    )


def test_nested_proof_indices_preserve_order_and_live_weight() -> None:
    source = load_source(SOURCE).qiec
    module = loads(dumps(source))
    assert isinstance(module, QiecModule) and module == source
    (descriptor,) = [
        item.foreign
        for item in module.handlers
        if isinstance(item.foreign, DeductionProofs)
    ]
    presentation = descriptor.proof_presentation
    (sort,) = presentation.index_sorts
    assert sort.name == "Nested__Term"
    shapes = tuple(
        _form(sort, values)
        for values in (("On", "On", "Off"), ("Off", "On", "On"), ("On", "Off", "On"))
    )
    exact, general = presentation.rules
    assert exact.premises[0].judgment.arguments[3] == shapes[0]
    assert shapes[0].arguments[0] == shapes[0].arguments[1]
    assert shapes[0].arguments[0] != shapes[0].arguments[2]
    axiom = presentation.axioms[0].id

    log_weights = torch.tensor(
        [-0.2, -0.8, -1.1], dtype=torch.float64, requires_grad=True
    )
    for _ in range(2):
        result = run_program(
            module,
            "listed",
            data={"sentence": ("x", "y")},
            sites={},
            parameters={
                f"Nested.lex.{i}": value for i, value in enumerate(log_weights.unbind())
            },
        )
        row = result.value
        observed = Counter()
        found = []
        for _ in range(4):
            assert isinstance(row, RuntimeConstructor)
            assert row.constructor == descriptor.chart.entry
            proof, mass, row = row.fields
            assert isinstance(proof, RuntimeConstructor)
            assert isinstance(mass, torch.Tensor)
            left, right = proof.fields
            assert isinstance(left, RuntimeConstructor) and isinstance(
                right, RuntimeConstructor
            )
            assert left.constructor == right.constructor == axiom
            assert isinstance(left.result_type, TypeApplication)
            assert left.result_type.constructor.name == "Nested__SpanProof"
            assert left.result_type.arguments == left.static_arguments
            assert left.fields == ("x",) and right.fields == ("y",)
            left_form = left.result_type.arguments[3]
            position = shapes.index(left_form)
            torch.testing.assert_close(mass, log_weights[position], rtol=0, atol=0)
            observed[proof.constructor, position] += 1
            found.append(mass)
        assert (
            isinstance(row, RuntimeConstructor)
            and row.constructor == descriptor.chart.empty
        )
        assert observed == {
            (exact.id, 0): 1,
            (general.id, 0): 1,
            (general.id, 1): 1,
            (general.id, 2): 1,
        }
        expected_weights = log_weights + log_weights.new_tensor([2.0, 1.0, 1.0]).log()
        actual = torch.stack(found).logsumexp(0)
        torch.testing.assert_close(
            actual, expected_weights.logsumexp(0), rtol=1e-12, atol=1e-12
        )
        (gradient,) = torch.autograd.grad(actual, log_weights)
        torch.testing.assert_close(gradient, expected_weights.softmax(0))
        with torch.no_grad():
            log_weights.add_(0.15 * gradient)


def test_applied_index_still_rejects_foreign_sorts_and_wrong_rank() -> None:
    module = load_source(SOURCE).qiec
    (descriptor,) = [
        item.foreign
        for item in module.handlers
        if isinstance(item.foreign, DeductionProofs)
    ]
    (sort,) = descriptor.proof_presentation.index_sorts
    pair = _form(sort, ("On", "Off", "On"))
    other = sort.with_(name="Other__Term")
    wrong = _form(other, ("On", "Off", "On")).arguments[1]
    for child in (wrong, IndexLiteral(value=1, sort=NAT)):
        with pytest.raises(dx.ValidationError, match="wrong sort"):
            pair.with_(arguments=(pair.arguments[0], child, pair.arguments[2]))
    with pytest.raises(dx.ValidationError, match="expects 3 arguments"):
        pair.with_(arguments=pair.arguments[:-1])

    private = descriptor.chart.presentation
    with pytest.raises(dx.ValidationError, match="undeclared user index sort"):
        private.with_(index_sorts=())
    (declared,) = private.index_sorts
    changed = declared.with_(
        constructors=(*declared.constructors, "other"),
        arities=(*declared.arities, 0),
    )
    with pytest.raises(dx.ValidationError, match="disagrees with its declaration"):
        private.with_(index_sorts=(changed,))
