"""Learnable deduction weights are keyed by category bindings, never positions.

A ``#[learnable]`` rule has one weight per distinct binding of its
non-position variables. Chart positions (the slots the compiler types
``Int``) never enter the key, so a rule's weight is shared across every
position and sentence length, the parameter set is fixed once the
category bindings are seen, and the agenda chart and the checked QIEC
search read the same weight for the same firing.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous import Draw
from quivers.dsl import loads
from quivers.qiec.program_runtime import run_deduction
from quivers.stochastic.deduction import (
    adam_fit_deduction,
    nuts_program_from_deduction,
    sample_corpus,
)

POSITIONAL = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=NP, depth=4]
    atoms NP, A, span, a
    rule np_a : span(I, K, A), span(K, J, A) |- span(I, J, NP) #[learnable]
    rule np_r : span(I, K, NP), span(K, J, A) |- span(I, J, NP) #[learnable]
    lexicon
        "a" : A = a
"""

CATEGORIAL = """
object Term : FinSet 16
deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, big, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the
        "big"  : Fwd(N, N)  = big
        "dog"  : N          = dog
        "runs" : Bwd(S, NP) = runs
"""

CHAINED = """
object Term : FinSet 4
deduction G : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, A, span, a
    rule np_a : span(I, K, A), span(K, J, A) |- span(I, J, NP) #[learnable, bounded]
    rule relay : span(I, K, NP), span(K, J, A) |- span(I, J, S) #[parent=np_a]
    rule top : span(I, K, A), span(K, J, NP) |- span(I, J, S) #[learnable, parent=relay]
    lexicon
        "a" : A = a
"""


def _rule_parameter_names(source: str, name: str, lengths: range) -> list[list[str]]:
    """The rule-parameter names after parsing ``a`` strings of each length.

    Parameters
    ----------
    source : str
        The QVR source.
    name : str
        The deduction.
    lengths : range
        The sentence lengths to parse, in order.

    Returns
    -------
    list[list[str]]
        The names before any parse, then after each length.
    """
    system = loads(source).deductions[name]
    snapshots = [[key for key, _ in system.named_parameters()]]
    for length in lengths:
        system(["a"] * length).goal_weight()
        snapshots.append([key for key, _ in system.named_parameters()])
    return snapshots


def test_a_rule_without_category_variables_has_one_weight_at_every_position() -> None:
    snapshots = _rule_parameter_names(POSITIONAL, "G", range(1, 7))
    expected = ["_rule_module.rule_np_a.BASE", "_rule_module.rule_np_r.BASE"]
    # The weights exist before the first parse and never multiply.
    assert all(snapshot == expected for snapshot in snapshots)


def test_a_gradient_step_moves_the_same_construction_at_another_position() -> None:
    system = loads(POSITIONAL).deductions["G"]
    (weight,) = (
        parameter
        for key, parameter in system.named_parameters()
        if key == "_rule_module.rule_np_a.BASE"
    )
    item = ("span", 1, 3, ("atom", "NP"))
    before = float(system(["a"] * 4).weight(item).detach())
    optimizer = torch.optim.SGD(system.parameters(), lr=0.5)
    optimizer.zero_grad()
    (-system(["a", "a"]).goal_weight()).backward()
    optimizer.step()
    after = float(system(["a"] * 4).weight(item).detach())
    # The step was taken on the firing over (0, 2); the firing over
    # (1, 3) of a longer sentence reads the same weight.
    assert before == 0.0
    assert after == pytest.approx(0.5)
    assert after == pytest.approx(float(weight.detach()))


def test_category_bindings_still_distinguish_weights() -> None:
    system = loads(CATEGORIAL).deductions["AB"]
    for sentence in ("the dog runs", "the big dog runs", "the big big dog runs"):
        assert torch.isfinite(system(sentence.split()).goal_weight())
    names = sorted(key for key, _ in system.named_parameters())
    assert names == [
        "_rule_module.rule_bwd_app.X_TSatom_SSE_Y_TSatom_SNPE",
        "_rule_module.rule_fwd_app.X_TSatom_SNE_Y_TSatom_SNE",
        "_rule_module.rule_fwd_app.X_TSatom_SNPE_Y_TSatom_SNE",
    ]


def test_the_checked_search_reads_the_same_weights_by_category_binding() -> None:
    program = loads(CATEGORIAL)
    system = program.deductions["AB"]
    tokens = "the big big dog runs".split()
    system(tokens)
    values = {
        "_rule_module.rule_fwd_app.X_TSatom_SNE_Y_TSatom_SNE": -0.3,
        "_rule_module.rule_fwd_app.X_TSatom_SNPE_Y_TSatom_SNE": 0.2,
        "_rule_module.rule_bwd_app.X_TSatom_SSE_Y_TSatom_SNPE": -0.7,
    }
    with torch.no_grad():
        for key, parameter in system.named_parameters():
            parameter.fill_(values[key])
    agenda = float(system(tokens).goal_weight().detach())
    # Two applications of the (N, N) firing, at different positions,
    # read one weight.
    assert agenda == pytest.approx(2 * -0.3 + 0.2 - 0.7)
    run = run_deduction(
        program.qiec,
        "AB",
        tokens=tokens,
        parameters={
            "AB.rule.fwd_app:N|N": -0.3,
            "AB.rule.fwd_app:NP|N": 0.2,
            "AB.rule.bwd_app:S|NP": -0.7,
        },
    )
    assert run.weight == pytest.approx(agenda)


def test_parent_and_bounded_weights_follow_each_rules_own_flags() -> None:
    program = loads(CHAINED)
    system = program.deductions["G"]
    names = sorted(key for key, _ in system.named_parameters())
    # ``relay`` is not learnable, so only ``np_a`` and ``top`` own a weight.
    assert names == ["_rule_module.rule_np_a.BASE", "_rule_module.rule_top.BASE"]
    with torch.no_grad():
        for key, parameter in system.named_parameters():
            parameter.fill_(0.4 if key.endswith("np_a.BASE") else -0.2)
    bounded = -math.log1p(math.exp(0.4))
    # "a a a" has two derivations of S. Through ``relay`` over np(0, 2)
    # and a(2, 3): np_a's own firing, then relay's parent np_a, both
    # bounded by np_a's own flag. Through ``top`` over a(0, 1) and
    # np(1, 3): np_a's firing, top's own unbounded weight, relay (no
    # weight of its own), and relay's parent np_a.
    relay = bounded + bounded
    top = bounded + (-0.2) + bounded
    expected = math.log(math.exp(relay) + math.exp(top))
    agenda = float(system(["a", "a", "a"]).goal_weight().detach())
    assert agenda == pytest.approx(expected)
    run = run_deduction(
        program.qiec,
        "G",
        tokens=["a", "a", "a"],
        parameters={"G.rule.np_a:": 0.4, "G.rule.top:": -0.2},
    )
    assert run.weight == pytest.approx(agenda)


def test_fitting_sampling_and_posterior_sites_see_a_fixed_parameter_set() -> None:
    system = loads(POSITIONAL).deductions["G"]
    short = [["a", "a"]]
    long = [["a"] * length for length in range(2, 7)]
    adam_fit_deduction(system, short, steps=2)
    after_short = [key for key, _ in system.named_parameters()]
    adam_fit_deduction(system, long, steps=2)
    assert [key for key, _ in system.named_parameters()] == after_short
    assert len(sample_corpus(system, length=5, n_samples=3, seed=0)) == 3
    assert [key for key, _ in system.named_parameters()] == after_short
    model, _, _ = nuts_program_from_deduction(system, long)
    sites = sorted(
        name for step in model.steps if isinstance(step, Draw) for name in step.names
    )
    assert sites == [
        "log_w___rule_module__rule_np_a_BASE",
        "log_w___rule_module__rule_np_r_BASE",
    ]
