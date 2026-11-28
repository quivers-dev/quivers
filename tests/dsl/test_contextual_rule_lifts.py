"""Contextual rule lifts retain one already evaluated frame."""

from __future__ import annotations

from itertools import product

import pytest
import torch

from quivers.continuous.tensor_backend import TensorBackend
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
from quivers.qiec.module import QiecModule
from quivers.qiec.program_runtime import run_program
from quivers.qiec.serialization import dumps, loads


SOURCE = """\
object Input : FinSet 1
object Output : Real 1

instance random : Random

effect Journal
    prepare : Unit -> Real
    choose : Unit -> Real
    before : Unit -> Real
    after : Unit -> Real

instance journal : Journal

handler audit for Journal : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    prepare resumes 1 =>
        resume(0.0)
    choose resumes 1 =>
        resume(0.0)
    before resumes 1 =>
        resume(0.0)
    after resumes 1 =>
        resume(0.0)

define picked(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("frame"), Bernoulli(p))

define picked_again(p : Real) : Bool !{random} =
    perform random.sample[Bool](site("fresh"), Bernoulli(p))

define prepared(xs : Tensor[Real]([2])) : Tensor[Real]([2]) !{journal} =
    let marker <- perform journal.prepare()
    return xs

define chosen(flag : Bool) : Bool !{journal} =
    let marker <- perform journal.choose()
    return flag

define measure[n : Nat](xs : Tensor[Real]([n]), keep : Bool) : Real !{journal} =
    let pre <- perform journal.before()
    let post <- perform journal.after()
    if keep then
        return xs[0] + pre + post
    else
        return xs[1] + pre + post

define evaluate(p : Real, theta : Real) : Real !{random, journal} =
    let frame <- picked(p)
    let xs = [theta, 1.0 - theta]
    let start <- lift_first[2](measure, prepared(xs), chosen(true), frame)
    let finish <- lift_second[2](measure, frame, prepared(xs), chosen(false))
    if start[1] == finish[0] then
        return start[0]
    else
        return finish[1]

define response(p : Real, theta : Real) : Real !{random} =
    handle journal with audit in
        evaluate(p, theta)

program inspect(p : Real, theta : Real) : Input -> Output
    let probability <- response(p, theta)
    observe y <- Bernoulli(probs=probability)
    return probability
"""

FIRST = "lift_first[2](measure, prepared(xs), chosen(true), frame)"
SECOND = "lift_second[2](measure, frame, prepared(xs), chosen(false))"


def _module(source: str) -> QiecModule:
    printed = module_to_source(parse(source))
    assert module_to_source(parse(printed)) == printed
    module = load_source(printed).qiec
    assert isinstance(module, QiecModule)
    restored = loads(dumps(module))
    assert isinstance(restored, QiecModule) and restored == module
    return restored


def _joint(
    module: QiecModule,
    p: torch.Tensor,
    theta: torch.Tensor,
    frame: bool,
    fresh: bool | None = None,
) -> torch.Tensor:
    sites = {"frame": frame}
    if fresh is not None:
        sites["fresh"] = fresh
    run = run_program(
        module,
        "inspect",
        data={"p": p, "theta": theta, "y": True},
        sites=sites,
        distributions=TensorBackend(),
    )
    response = theta if fresh is None or fresh == frame else 1 - theta
    torch.testing.assert_close(run.value, response)

    requested = [
        event.detail
        for event in run.result.trace
        if event.event == "operation.requested"
    ]
    samples = [
        event["address"]
        for event in requested
        if event["operation"] == str(RANDOM_SAMPLE)
    ]
    assert len(samples) == len(set(samples)) == len(sites)
    journal = next(effect for effect in module.effects if effect.ref.name == "Journal")
    names = {str(operation.id): operation.name for operation in journal.operations}
    audited = [event for event in requested if event["operation"] in names]
    assert [names[event["operation"]] for event in audited] == [
        "prepare",
        "choose",
        "before",
        "after",
    ] * 2
    assert len({event["instance"] for event in audited}) == 1
    assert len({event["address"] for event in audited}) == 8
    return run.log_joint


def test_contextual_lifts_preserve_one_draw_and_its_response_likelihood() -> None:
    shared = _module(SOURCE)
    independent = _module(
        SOURCE.replace(
            f"    let finish <- {SECOND}",
            "    let independent <- picked_again(p)\n"
            f"    let finish <- {SECOND.replace('frame', 'independent')}",
        )
    )
    expanded = _module(
        SOURCE.replace(
            f"    let start <- {FIRST}",
            "    let selected_left <- measure[2](prepared(xs), chosen(true))\n"
            "    let start = (selected_left, frame)",
        ).replace(
            f"    let finish <- {SECOND}",
            "    let selected_right <- measure[2](prepared(xs), chosen(false))\n"
            "    let finish = (frame, selected_right)",
        )
    )
    p = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    found = torch.stack([_joint(shared, p, theta, b) for b in (False, True)]).logsumexp(
        0
    )
    manual = torch.stack(
        [_joint(expanded, p, theta, b) for b in (False, True)]
    ).logsumexp(0)
    mutation = torch.stack(
        [
            _joint(independent, p, theta, a, b)
            for a, b in product((False, True), repeat=2)
        ]
    ).logsumexp(0)

    # The response must use the same saved draw twice. The mutant draws
    # independent frames and scales the opposite outcome on disagreement.
    expected = theta.log()
    disagree = 2 * p * (1 - p)
    changed = ((1 - disagree) * theta + disagree * (1 - theta)).log()
    torch.testing.assert_close(found, expected)
    torch.testing.assert_close(found, manual)
    torch.testing.assert_close(mutation, changed)
    assert not torch.isclose(found, mutation)
    torch.testing.assert_close(
        torch.autograd.grad(found, (p, theta)),
        (p.new_zeros(()), 1 / theta),
    )
    torch.testing.assert_close(
        torch.autograd.grad(mutation, (p, theta)),
        torch.autograd.grad(changed, (p, theta)),
    )


def test_contextual_lifts_keep_the_original_static_call_site() -> None:
    syntax = parse(SOURCE)
    sites = tuple(
        site
        for site in call_sites(syntax)
        if site.owner == "evaluate" and site.callee == "measure"
    )
    assert len(sites) == 2
    for site in sites:
        assert site.static_arguments == ("2",)
        assert site.argument_count == 2
        assert call_site_at(syntax, site.line - 1, site.col) == site
        assert instantiated_call_signature(syntax, site) == (
            "measure[2](xs : Tensor[Real]([2]), keep : Bool) : Real !{journal}"
        )


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            FIRST,
            FIRST.replace("frame", "picked(p)"),
            "already evaluated frame",
        ),
        (
            FIRST,
            FIRST.replace("[2]", ""),
            "supplies 0 static argument",
        ),
        (
            FIRST,
            FIRST.replace("measure", "missing"),
            "unknown computation 'missing'",
        ),
        (
            FIRST,
            FIRST.replace("chosen(true)", "1.0"),
            "expected Bool",
        ),
        (
            "define evaluate(p : Real, theta : Real) : Real !{random, journal}",
            "define evaluate(p : Real, theta : Real) : Real !{random}",
            "journal",
        ),
    ],
)
def test_contextual_lifts_recheck_types_and_scope(before, after, message) -> None:
    with pytest.raises(QiecDiagnosticError, match=message):
        load_source(SOURCE.replace(before, after, 1))
