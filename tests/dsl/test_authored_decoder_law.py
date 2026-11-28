"""Authored bounded decoders construct native checked structured laws."""

from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import loads as load_source
from quivers.dsl.compiler._prelude import CompileError
from quivers.dsl.parser import parse
from quivers.qiec.evaluator import RuntimeConstructor
from quivers.qiec.execution import ExecutionFailure, run_named
from quivers.qiec.identifiers import ConstructorId
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_program, sample_program
from quivers.qiec.serialization import dumps, loads
from quivers.qiec.structural_runtime import structural_runtime_configuration
from quivers.qiec.terms import Handle, LiteralValue, Perform, TupleValue
from quivers.structural import make_term
from quivers.transpile import UnsupportedConstruct, transpile


SOURCE = """\
signature Tiny
    sorts
        Term : object [dim=2]
    constructors
        Red : -> Term
        Blue : -> Term

decoder D : Tiny [depth=1, max_candidates=2]
    body |-> recursive

instance random : Random

define draw(code : Tensor[Real]([2])) : Tiny__TermValue !{random} =
    let law <- D__law(code)
    perform random.sample[Tiny__TermValue](site("decoded"), law)

define encode(point : Tiny__TermValue) : Real !{} =
    case point motive => Real
        Tiny__Red =>
            return 2.0
        Tiny__Blue =>
            return 4.0

define density(code : Tensor[Real]([2]), point : Tiny__TermValue) : LogWeight !{} =
    let law <- D__law(code)
    return log_prob(law, point)

object Input : FinSet 1
object Out : Real 1

program measure(code : Tensor[Real]([2])) : Input -> Out
    let term <- draw(code)
    let response <- encode(term)
    observe y <- Normal(loc=response, scale=1.0)
    return response

export measure
"""


def test_authored_law_draws_replays_and_scores_through_checked_term() -> None:
    compiled = load_source(SOURCE)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module

    decoder = compiled.decoders["D"].double()
    # Set the two relevant rows to known independent parameters.
    # The default head has an extra row for scoped variables.
    head = decoder._dec_0
    assert isinstance(head, torch.nn.Linear)
    with torch.no_grad():
        head.weight.copy_(
            torch.tensor(
                [[1.0, -0.5], [-0.25, 0.75], [20.0, 20.0]], dtype=torch.float64
            )
        )
        head.bias.zero_()
    code = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
    runtime = structural_runtime_configuration(restored, compiled)
    law = run_named(restored, "D__law", (code,), runtime=runtime).value
    (red, _), (blue, _) = law.weighted_support()
    assert isinstance(red, RuntimeConstructor) and isinstance(blue, RuntimeConstructor)
    assert red.result_type == blue.result_type == law.element
    assert law.element.constructor.name == "Tiny__TermValue"
    observed = 1.75
    forward = compiled.run(
        "measure",
        data={"code": code.detach(), "y": observed},
        seed=0,
    )
    assert forward.value in (2.0, 4.0)
    replay = run_program(
        restored,
        "measure",
        data={"code": code, "y": observed},
        sites={"decoded": red},
        providers=runtime.providers,
        distributions=TensorBackend(),
    )
    assert replay.value.item() == 2.0
    expected = (head.weight[:2] @ code + head.bias[:2]).log_softmax(dim=0)[0]
    raw = run_named(restored, "density", (code, red), runtime=runtime).value
    torch.testing.assert_close(raw, expected)
    response = torch.distributions.Normal(code.new_tensor(2.0), code.new_ones(()))
    score = expected + response.log_prob(code.new_tensor(observed))
    torch.testing.assert_close(replay.log_joint, score)
    for found, want in zip(
        torch.autograd.grad(replay.log_joint, (code, head.weight)),
        torch.autograd.grad(score, (code, head.weight)),
        strict=True,
    ):
        torch.testing.assert_close(found, want)


def test_authored_decoder_support_refuses_a_cap_below_its_exact_size() -> None:
    with pytest.raises(
        CompileError, match="complete decoder support exceeds the 1 candidate limit"
    ):
        load_source(SOURCE.replace("max_candidates=2", "max_candidates=1"))


def test_prepared_authored_decoder_support_can_be_copied() -> None:
    decoder = load_source(SOURCE).decoders["D"]
    clone = deepcopy(decoder)
    support = clone._qiec_finite_support
    law = support.distribution(clone, torch.zeros(2))
    assert len(law.weighted_support()) == 2


def test_authored_decoder_laws_share_the_signature_carrier() -> None:
    source = SOURCE.replace(
        "instance random : Random",
        "decoder Other : Tiny [depth=1, max_candidates=2]\n"
        "    body |-> recursive\n\n"
        "instance random : Random",
    )
    module = load_source(source).qiec
    assert isinstance(module, QiecModule)
    assert sum(family.name == "Tiny__TermValue" for family in module.families) == 1
    assert sum(family.name == "Tiny__Term" for family in module.families) == 1
    result = {item.name: item.type.result for item in module.computations}
    assert result["D__law"] == result["Other__law"]


def test_authored_decoder_law_names_its_target_capability() -> None:
    with pytest.raises(
        UnsupportedConstruct, match=r"`D__law` requires the `neural-attachment`"
    ):
        transpile(parse(SOURCE), target="stan")


def test_serialized_decoder_law_checks_the_original_limit() -> None:
    compiled = load_source(SOURCE)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    law = next(item for item in module.computations if item.name == "D__law")
    assert isinstance(law.body, Handle) and isinstance(law.body.computation, Perform)
    request = law.body.computation.request
    (argument,) = request.arguments
    assert isinstance(argument, TupleValue)
    code, limit = argument.items
    assert isinstance(limit, LiteralValue) and limit.value == 2
    wrong = argument.with_(items=(code, limit.with_(value=3)))
    replaced = law.with_(
        body=law.body.with_(
            computation=Perform(request=request.with_(arguments=(wrong,)))
        )
    )
    changed = module.with_(
        computations=tuple(
            replaced if item.id == law.id else item for item in module.computations
        )
    )
    restored = loads(dumps(changed))
    assert isinstance(restored, QiecModule)
    with pytest.raises(
        ExecutionFailure, match="does not match the checked support limit"
    ):
        run_named(
            restored,
            "D__law",
            (torch.zeros(2),),
            runtime=structural_runtime_configuration(restored, compiled),
        )


_ENCODE_SOURCE = SOURCE.replace(
    "decoder D : Tiny",
    "encoder E : Tiny\n\ndecoder D : Tiny",
).replace(
    "    let response <- encode(term)",
    "    let encoded <- E__encode_value(term)\n    let response = encoded[0]",
)


def test_sampled_structural_terms_feed_the_encoder_and_observation() -> None:
    compiled = load_source(_ENCODE_SOURCE)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule)
    runtime = structural_runtime_configuration(restored, compiled)
    decoder = compiled.decoders["D"].double()
    encoder = compiled.encoders["E"].double()
    head = decoder._dec_0
    assert isinstance(head, torch.nn.Linear)
    constants = dict(encoder.named_parameters())
    red, blue = constants["_op_0.const_Red"], constants["_op_1.const_Blue"]
    with torch.no_grad():
        head.weight.copy_(
            head.weight.new_tensor([[1.0, -0.5], [-0.25, 0.75], [20.0, 20.0]])
        )
        head.bias.copy_(head.bias.new_tensor([0.3, -0.1, 19.0]))
        red.copy_(red.new_tensor([-0.4, 9.0]))
        blue.copy_(blue.new_tensor([3.0, -8.0]))
    code = head.weight.new_tensor([0.7, -0.2], requires_grad=True)
    observed = 1.75
    law = run_named(restored, "D__law", (code,), runtime=runtime).value
    values = {point.constructor: point for point, _ in law.weighted_support()}
    known = {
        ConstructorId.derive("signature", "Tiny", name) for name in ("Red", "Blue")
    }
    assert values.keys() == known

    with pytest.raises(ExecutionFailure, match="differentiable backend"):
        compiled.run("measure", data={"code": code.detach(), "y": observed}, seed=0)
    # The host-reference Normal accepts frozen numeric arguments. A view of
    # a trainable encoding can retain requires_grad even under no_grad().
    red.requires_grad_(False)
    blue.requires_grad_(False)
    try:
        with torch.no_grad():
            draw = compiled.run(
                "measure", data={"code": code.detach(), "y": observed}, seed=0
            )
    finally:
        red.requires_grad_(True)
        blue.requires_grad_(True)
    assert float(draw.value.detach()) in (-0.4, 3.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        forward = sample_program(
            restored,
            "measure",
            data={"code": code, "y": observed},
            providers=runtime.providers,
            distributions=TensorBackend(),
        )
    assert float(forward.value.detach()) in (-0.4, 3.0)
    assert forward.log_joint.requires_grad
    # Enumerate the two nominal terms independently and use the declared
    # two decoder logits and the actual encoder parameters, before executing
    # either runtime conversion or the observation shortcut.
    logits = (head.weight[:2] @ code + head.bias[:2]).log_softmax(dim=0)
    means = torch.stack((red[0], blue[0]))
    expected = logits - 0.5 * ((means - observed).square() + math.log(2.0 * math.pi))
    replayed = []
    for position, name in enumerate(("Red", "Blue")):
        point = values[ConstructorId.derive("signature", "Tiny", name)]
        encoded = run_named(
            restored, "E__encode_value", (point,), runtime=runtime
        ).value
        torch.testing.assert_close(encoded, encoder(make_term(name)))
        replay = run_program(
            restored,
            "measure",
            data={"code": code, "y": observed},
            sites={"decoded": point},
            providers=runtime.providers,
            distributions=TensorBackend(),
        )
        torch.testing.assert_close(replay.value, means[position])
        torch.testing.assert_close(replay.log_joint, expected[position])
        replayed.append(replay.log_joint)
    with pytest.raises(
        UnsupportedConstruct,
        match=r"`E__encode_value` requires the `neural-attachment`",
    ):
        transpile(parse(_ENCODE_SOURCE), target="stan")
    evidence = torch.stack(replayed).logsumexp(dim=0)
    oracle = expected.logsumexp(dim=0)
    parameters = (code, head.weight, head.bias, red, blue)
    torch.testing.assert_close(evidence, oracle)
    wanted = torch.autograd.grad(oracle, parameters)
    found = torch.autograd.grad(evidence, parameters)
    for gradient, target in zip(found, wanted, strict=True):
        torch.testing.assert_close(gradient, target)
        assert torch.count_nonzero(gradient)


def test_nominal_encoder_inverse_restores_interleaved_fields_and_refuses_forgery() -> (
    None
):
    source = """\
signature Mixed
    sorts
        Term : object [dim=2]
        Atom : object [dim=2]
        Count : data [dim=2, vocab=[-1, 7]]
        Word : data [dim=2, vocab=["tag"]]
    constructors
        Leaf : -> Atom
        Tagged : Count, Atom, Word -> Term
encoder E : Mixed
decoder D : Mixed [depth=2, max_candidates=2]
    body |-> recursive
"""
    compiled = load_source(source)
    module = compiled.qiec
    assert isinstance(module, QiecModule)
    runtime = structural_runtime_configuration(module, compiled)
    compiled.decoders["D"].double()
    law = run_named(
        module, "D__law", (torch.zeros(2, dtype=torch.float64),), runtime=runtime
    ).value
    (first, _), (second, _) = law.weighted_support()
    assert isinstance(first, RuntimeConstructor)
    assert isinstance(second, RuntimeConstructor)
    assert (
        first.constructor
        == second.constructor
        == ConstructorId.derive("signature", "Mixed", "Tagged")
    )
    child, count, word = first.fields
    assert isinstance(child, RuntimeConstructor)
    assert child.constructor == ConstructorId.derive("signature", "Mixed", "Leaf")
    assert int(count) == -1 and word == "tag"
    encoder = compiled.encoders["E"]
    for point, scalar in ((first, -1), (second, 7)):
        actual = run_named(module, "E__encode_value", (point,), runtime=runtime).value
        expected = encoder(make_term("Tagged", scalar, make_term("Leaf"), "tag"))
        torch.testing.assert_close(actual, expected)
    wrong = first.with_(fields=(first, count, word))
    with pytest.raises(
        ExecutionFailure, match="invalid finite sampleable constructor point"
    ):
        run_named(module, "E__encode_value", (wrong,), runtime=runtime)
