"""Named operand calls preserve order, effects and explicit sharing."""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads as load_source
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_diagnostics import QiecDiagnosticError
from quivers.dsl.qiec_tooling import (
    call_site_at,
    call_sites,
    instantiated_call_signature,
)
from quivers.qiec.builtins import RANDOM_SAMPLE
from quivers.qiec.execution import run_named
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import sample_program
from quivers.qiec.serialization import dumps, loads


SOURCE = """\
object UnitInput : FinSet 1
object Output : Real 1

instance random : Random

effect Coin
    toss : Real -> Bool

instance weighted_coin : Coin

define coin(probability : Real) : Bool !{weighted_coin} =
    perform weighted_coin.toss(probability)

define draw(probability : Real) : Bool !{random} =
    perform random.sample[Bool](site("coin"), Bernoulli(probability))

define mark(bit : Bool) : Bool !{} =
    return bit

define score_pair(left : Bool, cue : Real, right : Bool) : Real !{} =
    if left == right then
        return cue
    else
        return 1.0 - cue

define independent(probability : Real) : Real !{weighted_coin} =
    score_pair(mark(coin(probability)), abs(-0.75), coin(probability))

define shared(probability : Real) : Real !{weighted_coin} =
    let bit <- coin(probability)
    score_pair(bit, abs(-0.75), bit)

handler exact for Coin : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return response =>
        return response
    toss(probability : Real) resumes omega =>
        let heads <- resume(true)
        let tails <- resume(false)
        return probability * heads + (1.0 - probability) * tails

define integrate_independent(probability : Real) : Real !{} =
    handle weighted_coin with exact in
        independent(probability)

define integrate_shared(probability : Real) : Real !{} =
    handle weighted_coin with exact in
        shared(probability)

define sampled(probability : Real) : Real !{random} =
    score_pair(mark(draw(probability)), abs(-0.75), draw(probability))

program inspect(probability : Real) : UnitInput -> Output
    let response <- sampled(probability)
    return response

export inspect
"""


def test_lifted_calls_enumerate_independent_and_shared_effects() -> None:
    """A ternary operation sequences two new draws and an existing pure value."""
    syntax = parse(SOURCE)
    printed = module_to_source(syntax)
    assert module_to_source(parse(printed)) == printed
    module = load_source(printed).qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module

    p = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    found = run_named(restored, "integrate_independent", (p,)).value
    shared = run_named(restored, "integrate_shared", (p,)).value
    # Sum the four ordered outcomes, each with its own pair of Bernoulli weights.
    expected = 0.75 * (p * p + (1 - p) * (1 - p)) + 0.25 * 2 * p * (1 - p)
    torch.testing.assert_close(found, expected)
    torch.testing.assert_close(shared, p.new_tensor(0.75))
    (actual_gradient,) = torch.autograd.grad(found, p)
    (expected_gradient,) = torch.autograd.grad(expected, p)
    torch.testing.assert_close(actual_gradient, expected_gradient)

    def addresses(run) -> list[str]:
        return [
            event.detail["address"]
            for event in run.result.trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(RANDOM_SAMPLE)
        ]

    draw = sample_program(restored, "inspect", data={"probability": 0.25})
    assert addresses(draw) == addresses(
        sample_program(restored, "inspect", data={"probability": 0.75})
    )
    assert len(set(addresses(draw))) == 2
    body_calls = [
        event.detail["computation"]
        for event in draw.result.trace
        if event.event == "call.entered"
        and event.detail["computation"] in {"draw", "mark", "score_pair"}
    ]
    assert body_calls == ["draw", "mark", "draw", "score_pair"]


def test_lifted_calls_have_editor_signatures() -> None:
    syntax = parse(SOURCE)
    calls = [site for site in call_sites(syntax) if site.owner == "independent"]
    assert [site.callee for site in calls] == ["score_pair", "mark", "coin", "coin"]
    assert [site.argument_count for site in calls] == [3, 1, 1, 1]
    for site in calls[2:]:
        assert call_site_at(syntax, site.line - 1, site.col) == site
        assert instantiated_call_signature(syntax, site) == (
            "coin(probability : Real) : Bool !{weighted_coin}"
        )


def test_lifted_calls_require_the_operand_effects_and_types() -> None:
    with pytest.raises(QiecDiagnosticError, match="expected Bool, got Real") as caught:
        load_source(
            SOURCE.replace(
                "define mark(bit : Bool) : Bool !{} =\n    return bit",
                "define mark(bit : Bool) : Real !{} =\n    return 1.0",
            )
        )
    assert caught.value.code == "qiec-call"
    with pytest.raises(QiecDiagnosticError, match="weighted_coin"):
        load_source(
            SOURCE.replace(
                "independent(probability : Real) : Real !{weighted_coin}",
                "independent(probability : Real) : Real !{}",
            )
        )
