"""Ordinary scalar deduction consumes typed program-local token tensors."""

from math import log

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.effects.checked_program import CheckedProgram
from quivers.modeling import Model
from quivers.qiec import QiecModule, dumps, loads, run_program


GRAMMAR = """\
object Term : FinSet 2
object Result : Real 1
object Rows : FinSet 2

deduction Choice : Term -> Term [semiring=LogProb, start=S, normalize=lhs]
    atoms span, A, B, S, word
    rule first : span(I, J, A) |- span(I, J, S)
    rule second : span(I, J, B) |- span(I, J, S)
    lexicon
        "x" : A = word
        "y" : B = word

"""

LOCAL = (
    GRAMMAR
    + """\
program fit : Term -> Result
    sample p_s <- Dirichlet(concentration=[1.0, 1.0])
    let words = ["x"]
    let inside = parse(Choice, words, ("S", p_s))
    score observed = inside.goal_weight()
    return observed

export fit
"""
)

CALL = (
    LOCAL.replace('let words = ["x"]', "let words <- words_for_fit()")
    + """
define words_for_fit() : Tensor[String]([1]) !{} =
    return ["x"]
"""
)

CORPUS = LOCAL.replace('    let words = ["x"]\n', "").replace(
    'Choice, words, ("S", p_s)', 'Choice, sentences, Rows, ("S", p_s)'
)


@pytest.mark.parametrize("source", (LOCAL, CALL), ids=("literal", "named-code"))
def test_scalar_parse_preserves_checked_inference_and_lexical_gradients(source):
    compiled = load_source(source)
    assert isinstance(compiled.morphism, CheckedProgram)
    bound = Model(source).bind({}, input=torch.zeros((1, 1), dtype=torch.float64))
    assert isinstance(bound.program, CheckedProgram)
    module = loads(dumps(compiled.qiec))
    assert isinstance(module, QiecModule) and module == compiled.qiec

    logits = torch.tensor([-0.21, 0.42], dtype=torch.float64, requires_grad=True)
    p_s = logits.softmax(0)
    assert compiled.run("fit", sites={"p_s": p_s.detach()}).log_joint == pytest.approx(
        log(p_s[0].item())
    )
    result = run_program(
        module, "fit", data={}, sites={"p_s": p_s}, distributions=TensorBackend()
    )
    inferred = bound.program.log_joint(bound.input, {"p_s": p_s}).squeeze(0)
    expected = logits.log_softmax(0)[0]
    gradient = torch.autograd.grad(expected, logits)[0]
    for actual in (result.log_joint, inferred):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits, retain_graph=True)[0], gradient
        )


@pytest.mark.parametrize(
    "value", ('"x"', '[["x"]]', "[1, 1]"), ids=("scalar", "rank-two", "integer")
)
def test_scalar_parse_refuses_locals_outside_the_checked_token_type(value):
    with pytest.raises(
        QiecDiagnosticError, match=r"must be a rank-one Tensor\[String\]"
    ) as error:
        load_source(LOCAL.replace('["x"]', value))
    assert error.value.code == "qiec-program"


def test_local_sentence_cannot_enter_the_external_corpus_path():
    source = LOCAL.replace(
        'Choice, words, ("S", p_s)', 'Choice, words, Rows, ("S", p_s)'
    )
    with pytest.raises(QiecDiagnosticError, match="needs an external corpus"):
        load_source(source)


def test_external_corpus_keeps_its_routing_and_product():
    compiled = load_source(CORPUS)
    assert not isinstance(compiled.morphism, CheckedProgram)
    data = {"sentences": (("x",), ("y",))}
    logits = torch.tensor([-0.21, 0.42], dtype=torch.float64, requires_grad=True)
    p_s = logits.softmax(0)
    result = run_program(
        compiled.qiec,
        "fit",
        data=data,
        sites={"p_s": p_s},
        distributions=TensorBackend(),
    )
    expected = logits.log_softmax(0).sum()
    torch.testing.assert_close(result.log_joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.log_joint, logits)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_scalar_parse_reads_an_outer_token_binding_inside_a_marginal():
    source = LOCAL.replace(
        """
    let inside = parse(Choice, words, ("S", p_s))
    score observed = inside.goal_weight()
    return observed
""",
        """
    marginalize latent : Term <- Categorical(probs=[0.5, 0.5])
        let inside = parse(Choice, words, ("S", p_s))
        score observed = inside.goal_weight()
    let answer = 0.0
    return answer
""",
    )
    compiled = load_source(source)
    assert isinstance(compiled.morphism, CheckedProgram)
    logits = torch.tensor([-0.21, 0.42], dtype=torch.float64, requires_grad=True)
    result = run_program(
        compiled.qiec,
        "fit",
        data={},
        sites={"p_s": logits.softmax(0)},
        distributions=TensorBackend(),
    )
    # Both latent outcomes use the same sentence mass. The marginal
    # weights sum to one, leaving its log likelihood and derivative intact.
    expected = logits.log_softmax(0)[0]
    torch.testing.assert_close(result.log_joint, expected)
    torch.testing.assert_close(
        torch.autograd.grad(result.log_joint, logits)[0],
        torch.autograd.grad(expected, logits)[0],
    )
