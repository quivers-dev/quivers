"""Checks of a checked program's law, sites and joint against enumeration.

A program that runs on the reference machine may reach a different set of
sites on different runs: a recursion whose depth a coin decides reaches
one coin or two. The finite-effects oracle (O9) runs the same program,
written over plain fractions, once for every sequence of values its draws
can take, and names every draw. The machine's trace of a run given those
values must reach exactly those sites, under the names a trace gives
occurrences (a label reached once by itself, a label reached more than
once as ``"<label>@<n>"``), and its joint must be the oracle's weight.
Forward runs must draw each occurrence independently, which the frequency
of each pattern of draws shows within a bound that fails with probability
below one in a thousand million.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction

import torch

from quivers.effects.checked_program import CheckedProgram
from quivers.inference import trace
from tests.evidence.checks._registry import add
from tests.evidence.checks.span import double_precision
from tests.evidence.compare import ELEMENTARY, Accuracy, Mismatch, compare_number
from tests.evidence.oracles.finite_effects import Draws, Outcome, runs
from tests.evidence.sources import program

HEADS = Fraction(3, 10)
"""The probability of a coin of the recursion."""

FIRST = Fraction(3, 5)
"""The probability of the program's first draw."""

DEPTH = 2
"""The largest number of coins a run flips."""

FORWARD_RUNS = 600
"""The forward runs of the frequency check."""


def flips(draws: Draws) -> int:
    """The coin program, over fractions.

    Parameters
    ----------
    draws : Draws
        The run's draws.

    Returns
    -------
    int
        The number of heads before the first tail, at most `DEPTH`.
    """
    draws.choose("first", (1 - FIRST, FIRST))
    heads = 0
    for occurrence in range(DEPTH):
        if draws.choose(f"coin#{occurrence}", (1 - HEADS, HEADS)) == 0:
            break
        heads += 1
    return heads


def trace_names(addresses: tuple[str, ...]) -> tuple[str, ...]:
    """The names a trace gives the oracle's draws.

    Parameters
    ----------
    addresses : tuple of str
        The oracle's names, ``"<label>#<n>"`` for the ``n``-th occurrence.

    Returns
    -------
    tuple of str
        The label alone where the run reaches it once, else
        ``"<label>@<n>"``.
    """
    labels = [address.partition("#")[0] for address in addresses]
    counts = Counter(labels)
    names = []
    for address, label in zip(addresses, labels, strict=True):
        occurrence = address.partition("#")[2]
        names.append(label if counts[label] == 1 else f"{label}@{occurrence or 0}")
    return tuple(names)


def _model() -> CheckedProgram:
    """The checked coin program.

    Returns
    -------
    CheckedProgram
        The program.
    """
    model = program("flips").morphism
    assert isinstance(model, CheckedProgram)
    return model


def _given(outcome: Outcome) -> dict[str, torch.Tensor]:
    """A run's values, as a trace names them, one run along the run axis.

    Parameters
    ----------
    outcome : Outcome
        The oracle's run.

    Returns
    -------
    dict of str to torch.Tensor
        A truth value per site.
    """
    return {
        name: torch.tensor([bool(value)])
        for name, value in zip(
            trace_names(outcome.addresses), outcome.values, strict=True
        )
    }


def _sites(recorded_sites: Mapping[str, object]) -> set[str]:
    """The draws of a trace, without the scores of the program.

    Parameters
    ----------
    recorded_sites : Mapping of str to object
        The trace's sites.

    Returns
    -------
    set of str
        The names of its draws.
    """
    return {name for name in recorded_sites if not name.startswith("score:")}


def check_sites_and_joints() -> None:
    """Every run's sites and joint against the oracle's.

    Raises
    ------
    Mismatch
        If a run reaches other sites than the oracle's, or its joint
        differs from the oracle's weight.
    """
    model = _model()
    with double_precision():
        x = torch.zeros(1, 1)
        for outcome in runs(flips):
            given = _given(outcome)
            recorded = trace(model, x, observations=given)
            found = _sites(recorded.sites)
            if found != set(given):
                raise Mismatch(
                    "support",
                    math.inf,
                    f"run {outcome.values}: sites {sorted(found)} for {sorted(given)}",
                )
            joint = model.log_joint(x, given)
            compare_number(
                float(joint.sum()),
                math.log(outcome.weight),
                Accuracy(domain="log", operations=8 * ELEMENTARY),
                f"joint of run {outcome.values}",
            )


def check_forward_runs() -> None:
    """Forward runs draw every occurrence independently, at its law.

    Each forward run's joint must be the oracle's weight of the values it
    drew, and the frequency of each pattern of draws must be the oracle's
    probability within the bound of Hoeffding's inequality, taken jointly
    over the patterns.

    Raises
    ------
    Mismatch
        If a run's joint differs from the oracle's, a run reaches sites no
        oracle run does, or a frequency is out of its bound.
    """
    model = _model()
    outcomes = {trace_names(o.addresses) + tuple(o.values): o for o in runs(flips)}
    # The frequencies are compared for the coins alone, whose patterns the
    # first draw does not change, so the bound is over fewer patterns.
    coins: dict[tuple[str | int, ...], Fraction] = {}
    for key, outcome in outcomes.items():
        pattern = _coins(key)
        coins[pattern] = coins.get(pattern, Fraction(0)) + outcome.weight
    seen: Counter[tuple[str | int, ...]] = Counter()
    with double_precision():
        torch.manual_seed(0)
        x = torch.zeros(1, 1)
        for _ in range(FORWARD_RUNS):
            recorded = trace(model, x)
            names = tuple(
                name for name in recorded.sites if name in _sites(recorded.sites)
            )
            values = tuple(
                int(bool(recorded.sites[name].value.reshape(-1)[0])) for name in names
            )
            key = names + values
            outcome = outcomes.get(key)
            if outcome is None:
                raise Mismatch("support", math.inf, f"a forward run reached {key}")
            assert recorded.log_joint is not None
            compare_number(
                float(recorded.log_joint.sum()),
                math.log(outcome.weight),
                Accuracy(domain="log", operations=8 * ELEMENTARY),
                f"joint of forward run {key}",
            )
            seen[_coins(key)] += 1
    bound = math.sqrt(math.log(2 * len(coins) * 1e9) / (2 * FORWARD_RUNS))
    worst = max(
        abs(seen[pattern] / FORWARD_RUNS - float(weight))
        for pattern, weight in coins.items()
    )
    if worst > bound:
        raise Mismatch(
            "value", worst - bound, f"a pattern's frequency is {worst} from its law"
        )


def _coins(key: tuple[str | int, ...]) -> tuple[str | int, ...]:
    """The coins of a run's names and values.

    Parameters
    ----------
    key : tuple of str and int
        The run's site names, then their values.

    Returns
    -------
    tuple of str and int
        The names and values of its coins.
    """
    half = len(key) // 2
    kept = [i for i in range(half) if not str(key[i]).startswith("first")]
    return tuple(key[i] for i in kept) + tuple(key[half + i] for i in kept)


def _register() -> None:
    """Register the checks."""
    add("checked/sites/flips", "trace", "R6", check_sites_and_joints)
    add("checked/forward/flips", "trace", "R6", check_forward_runs)


_register()

__all__ = ["DEPTH", "FIRST", "HEADS", "flips", "trace_names"]
