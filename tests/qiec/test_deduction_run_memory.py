"""A deduction run holds its live state, not the history of its steps.

The reference machine enumerates a deduction's derivations, so its step
count grows with the number of derivations while the state it needs at any
moment grows with the sentence. These tests measure what a run retains per
step, and check that a run which keeps no history computes the weights and
gradients of one that does, and of the agenda chart.
"""

from __future__ import annotations

import gc
import sys

import pytest
import torch

from quivers.dsl import loads
from quivers.program import Program
from quivers.qiec.evaluator import Evaluator
from quivers.qiec.execution import ExecutionTraceEvent
from quivers.qiec.program_runtime import DeductionRun, run_deduction

COPY = """
object Term : FinSet 8
deduction Copy : Term -> Term [semiring=SEMIRING, start=S]
    atoms S, A, B, p, span, a, b
    rule pair_a : span(I1, J1, A), span(I2, J2, A) |- p(I1, J1, I2, J2) #[learnable]
    rule pair_b : span(I2, J2, B), span(I1, J1, B) |- p(I1, J1, I2, J2) #[learnable]
    rule grow_a : span(J1, K1, A), p(I1, J1, I2, J2), span(J2, K2, A) |- p(I1, K1, I2, K2) #[learnable]
    rule grow_b : p(I1, J1, I2, J2), span(J2, K2, B), span(J1, K1, B) |- p(I1, K1, I2, K2) #[learnable]
    rule copy : p(I, J, J, K) |- span(I, K, S) #[learnable]
    lexicon
        "a" : A = a
        "b" : B = b
"""

WEIGHTS = {
    "pair_a": -0.4,
    "pair_b": -1.1,
    "grow_a": 0.3,
    "grow_b": -0.6,
    "copy": -0.2,
}


def _run(
    program: Program,
    sentence: str,
    weights: dict[str, float] = WEIGHTS,
    *,
    record_trace: bool = False,
) -> DeductionRun:
    """Run the copy grammar on one sentence of one-letter words.

    Parameters
    ----------
    program : Program
        The compiled grammar.
    sentence : str
        The sentence.
    weights : dict[str, float]
        The log weight of each rule.
    record_trace : bool
        Whether the run keeps its events.

    Returns
    -------
    DeductionRun
        The run.
    """
    return run_deduction(
        program.qiec,
        "Copy",
        tokens=list(sentence),
        parameters={f"Copy.rule.{rule}:": value for rule, value in weights.items()},
        record_trace=record_trace,
    )


def _retained_per_step(
    monkeypatch: pytest.MonkeyPatch, program: Program, *, record_trace: bool
) -> float:
    """The memory blocks a long search keeps, per machine step it takes.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The patcher.
    program : Program
        The compiled grammar.
    record_trace : bool
        Whether the run keeps its events.

    Returns
    -------
    float
        The growth of the number of allocated blocks between an early and a
        late step of one run, over the number of steps between them.
    """
    early, late = 20_000, 70_000
    blocks: dict[int, int] = {}
    steps = 0
    tick = Evaluator._tick

    def counting(self: Evaluator) -> None:
        """Count a step, and the allocated blocks at two of them.

        Parameters
        ----------
        self : Evaluator
            The machine.
        """
        nonlocal steps
        steps += 1
        if steps in (early, late):
            gc.collect()
            blocks[steps] = sys.getallocatedblocks()
        tick(self)

    with monkeypatch.context() as patch:
        patch.setattr(Evaluator, "_tick", counting)
        _run(program, "abab", record_trace=record_trace)
    return (blocks[late] - blocks[early]) / (late - early)


def test_a_run_that_records_no_trace_retains_nothing_per_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = loads(COPY.replace("SEMIRING", "LogProb"))
    # A warm run first, so that the caches a process fills once are full.
    _run(program, "aba")
    bounded = _retained_per_step(monkeypatch, program, record_trace=False)
    recorded = _retained_per_step(monkeypatch, program, record_trace=True)
    # A recorded event takes several blocks; a search holds only its live
    # state, whose size does not depend on how many steps came before.
    assert recorded > 1.0
    assert abs(bounded) < 0.2


@pytest.mark.parametrize("semiring", ["LogProb", "Viterbi"])
@pytest.mark.parametrize("sentence", ["aa", "ab", "abab", "aab"])
def test_a_run_weighs_a_sentence_the_same_whether_or_not_it_records(
    semiring: str, sentence: str
) -> None:
    program = loads(COPY.replace("SEMIRING", semiring))
    system = program.deductions["Copy"]
    with torch.no_grad():
        for key, parameter in system.named_parameters():
            parameter.fill_(WEIGHTS[key.split(".")[1].removeprefix("rule_")])
    silent = _run(program, sentence)
    recorded = _run(program, sentence, record_trace=True)
    assert silent.weight == recorded.weight
    assert silent.result.trace == ()
    assert recorded.result.trace
    agenda = float(system(list(sentence)).goal_weight().detach())
    assert silent.weight == pytest.approx(agenda)


def test_an_observer_receives_every_event_of_a_run_that_records_none() -> None:
    program = loads(COPY.replace("SEMIRING", "LogProb"))
    seen: list[ExecutionTraceEvent] = []
    run = run_deduction(program.qiec, "Copy", tokens=list("aa"), observer=seen.append)
    recorded = run_deduction(program.qiec, "Copy", tokens=list("aa"), record_trace=True)
    assert run.result.trace == ()
    assert [(item.sequence, item.event) for item in seen] == [
        (item.sequence, item.event) for item in recorded.result.trace
    ]


def test_the_search_has_the_gradient_of_the_agenda_chart() -> None:
    program = loads(COPY.replace("SEMIRING", "LogProb"))
    system = program.deductions["Copy"]
    parameters = {
        key.split(".")[1].removeprefix("rule_"): parameter
        for key, parameter in system.named_parameters()
    }
    with torch.no_grad():
        for rule, parameter in parameters.items():
            parameter.fill_(WEIGHTS[rule])
    weight = system(list("abab")).goal_weight()
    gradients = torch.autograd.grad(
        weight, list(parameters.values()), allow_unused=True
    )
    step = 1e-3
    for rule, gradient in zip(parameters, gradients, strict=True):
        raised = _run(program, "abab", {**WEIGHTS, rule: WEIGHTS[rule] + step}).weight
        lowered = _run(program, "abab", {**WEIGHTS, rule: WEIGHTS[rule] - step}).weight
        expected = 0.0 if gradient is None else float(gradient)
        assert (raised - lowered) / (2 * step) == pytest.approx(expected, abs=1e-6)
