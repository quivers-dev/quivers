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
from functools import partial

import didactic.api as dx
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.effects.checked_program import CheckedProgram
from quivers.effects.observation_scoring import observation_scoring
from quivers.qiec import scope
from quivers.qiec.evaluator import Evaluator, InvalidHandlerError
from quivers.qiec.program_runtime import run_deduction
from quivers.inference import trace
from tests.evidence.checks._registry import add
from tests.evidence.checks.consumers import PROBLEMS, compare_distribution
from tests.evidence.checks.span import double_precision
from tests.evidence.compare import ELEMENTARY, Accuracy, Mismatch, compare_number
from tests.evidence.counters import CallCounter
from tests.evidence.oracles.finite_effects import Draws, Outcome, runs
from tests.evidence.scaling import Family, check_family
from tests.evidence.sources import LEXICAL, program

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


def check_one_pass() -> None:
    """The joints of a call's runs in one machine pass, equal to each run's.

    The recursive program's control does not read its draw, so the runs of
    one call are one pass; the coin program's recursion reads its coins,
    so runs given different coins take different branches and are made one
    by one, with the reason reported. Either way each run's joint is the
    joint of that run alone.

    Raises
    ------
    Mismatch
        If a call makes another number of passes, reports no reason for
        passes made one by one, or a run's joint differs from its own.
    """
    with double_precision():
        recursive = program("recursive").morphism
        assert isinstance(recursive, CheckedProgram)
        draws = torch.tensor([0.5, 0.1, -0.3])
        data = {"y": torch.tensor([0.5])}
        together = recursive.log_joint(torch.zeros(3, 1), {"t": draws, **data})
        if recursive.work.iterations != 1 or recursive.vectorization_obstacle:
            raise Mismatch(
                "work",
                float(recursive.work.iterations),
                f"three runs of a control independent of its draws made "
                f"{recursive.work.iterations} passes",
            )
        alone = [
            recursive.log_joint(torch.zeros(1, 1), {"t": draws[i : i + 1], **data})
            for i in range(3)
        ]
        for i, joint in enumerate(alone):
            compare_number(
                float(together[i]),
                float(joint.sum()),
                Accuracy(domain="log", operations=8 * ELEMENTARY),
                f"run {i} of a pass",
            )
        model = _model()
        given = {
            "first": torch.tensor([True, False]),
            "coin@0": torch.tensor([True, True]),
            "coin@1": torch.tensor([False, True]),
        }
        model.log_joint(torch.zeros(2, 1), given)
        if model.work.iterations != 2 or not model.vectorization_obstacle:
            raise Mismatch(
                "work",
                float(model.work.iterations),
                "runs whose control reads different draws were not made one by one "
                "with a reason",
            )


def check_observation_roles() -> None:
    """An observation is a likelihood term, drawn from its law when not given.

    On the recursive problem, whose closed forms are exact: the joint is
    prior plus likelihood; without the likelihood (the program's prior, or
    a run with observation scoring off) it is the prior alone; and a run
    not given the observation draws it, scoring the draw under its law.

    Raises
    ------
    Mismatch
        If a joint differs from its closed form, or a drawn observation is
        recorded as observed or scored under another law.
    """
    problem = PROBLEMS["recursive"]
    accuracy = Accuracy(domain="log", operations=64 * ELEMENTARY, magnitude=16.0)
    with double_precision():
        model = problem.model()
        assert isinstance(model, CheckedProgram)
        prior = model.prior_program()
        x = torch.zeros(1, 1)
        for value in (-1.0, 0.25, 2.0):
            t = torch.tensor([value])
            values = {"t": t}
            log_prior = float(problem.log_prior(values))
            log_likelihood = float(problem.log_likelihood(values))
            joint = model.log_joint(x, {"t": t, "y": torch.tensor([0.5])})
            compare_number(
                float(joint.sum()), log_prior + log_likelihood, accuracy, "joint"
            )
            try:
                prior_joint = prior.log_joint(x, values)
            except KeyError as refusal:
                raise Mismatch(
                    "support", math.inf, f"the prior reaches another site: {refusal}"
                ) from refusal
            compare_number(float(prior_joint.sum()), log_prior, accuracy, "prior")
            with observation_scoring(False):
                unscored = model.log_joint(x, {"t": t, "y": torch.tensor([0.5])})
            compare_number(
                float(unscored.sum()), log_prior, accuracy, "joint without likelihood"
            )
        torch.manual_seed(0)
        recorded = trace(model, x)
    site = recorded.sites["y"]
    if site.is_observed:
        raise Mismatch("role", math.inf, "a drawn observation recorded as observed")
    t = float(recorded.sites["t"].value.reshape(()))
    y = float(site.value.reshape(()))
    wanted = -0.5 * (y - t - 3.0) ** 2 - 0.5 * math.log(2 * math.pi)
    compare_number(float(site.log_prob.sum()), wanted, accuracy, "drawn observation")


VECTORIZED_RUNS = 2000
"""The runs of the call the vectorized forward check makes."""


def _forward(model: CheckedProgram, runs: int, *, apart: bool) -> torch.Tensor:
    """The draws of one forward call of the recursive problem.

    Parameters
    ----------
    model : CheckedProgram
        The program.
    runs : int
        The runs of the call.
    apart : bool
        Whether to make the runs one by one: a handler on the stack that
        installs nothing keeps a call from being one pass.

    Returns
    -------
    torch.Tensor
        The returned draw of each run.

    Raises
    ------
    Mismatch
        If the call did not make the passes asked for.
    """
    x = torch.zeros(runs, 1)
    if apart:
        with observation_scoring(True):
            values = model.rsample(x)
    else:
        values = model.rsample(x)
    passes = runs if apart else 1
    if model.work.iterations != passes:
        raise Mismatch(
            "work", math.inf, f"{model.work.iterations} passes, not {passes}"
        )
    return values.reshape(-1)


def check_vectorized_forward() -> None:
    """A forward call drawn in one pass draws every run on its own.

    The recursive problem's control does not depend on the run, so a
    forward call of many runs is one pass of the machine. Its draws must
    follow the law of the drawn site, N(0, 1), within the bound of
    Dvoretzky, Kiefer and Wolfowitz, as the draws of the same call made run
    by run do; and two runs of the pass must be independent: the share of
    pairs of neighbouring runs that both fall below the median is 1/4, not
    the 1/2 of runs that share one draw, within Hoeffding's bound.

    Raises
    ------
    Mismatch
        If a call makes other passes, its draws do not follow the law, or
        neighbouring runs are not independent.
    """
    model = PROBLEMS["recursive"].model()
    assert isinstance(model, CheckedProgram)
    with double_precision():
        torch.manual_seed(3)
        together = _forward(model, VECTORIZED_RUNS, apart=False)
        apart = _forward(model, VECTORIZED_RUNS // 4, apart=True)
    law = _standard_normal
    compare_distribution(together, law, "draws of one pass")
    compare_distribution(apart, law, "draws run by run")
    below = (together < 0).double()
    pairs = below[0::2] * below[1::2]
    share = float(pairs.mean())
    bound = math.sqrt(math.log(2e9) / (2 * pairs.numel()))
    if abs(share - 0.25) > bound:
        raise Mismatch(
            "value", abs(share - 0.25), f"neighbouring runs share {share} of lows"
        )


def _standard_normal(value: float) -> float:
    """The distribution function of a standard normal.

    Parameters
    ----------
    value : float
        The point.

    Returns
    -------
    float
        The probability below it.
    """
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def check_changed_sites() -> None:
    """A run that reaches other sites than its values name is refused.

    The registry of a traced run fixes its sites; a run given those values
    whose recursion goes deeper reaches a site it is not given, and one
    that stops earlier leaves a given site unreached. Both are refused by
    name, so a sampler never scores a different set of sites silently.

    Raises
    ------
    Mismatch
        If either change is scored rather than refused.
    """
    model = _model()
    x = torch.zeros(1, 1)
    deeper = {"first": torch.tensor([True]), "coin": torch.tensor([True])}
    shallower = {
        "first": torch.tensor([True]),
        "coin@0": torch.tensor([False]),
        "coin@1": torch.tensor([True]),
    }
    for name, given, wanted in (
        ("deeper", deeper, "coin"),
        ("shallower", shallower, "coin@1"),
    ):
        try:
            joint = model.log_joint(x, given)
        except (KeyError, InvalidHandlerError) as refusal:
            if wanted not in str(refusal):
                raise Mismatch("refusal", math.inf, f"{name}: {refusal}") from refusal
            continue
        raise Mismatch("refusal", math.inf, f"a {name} run scored {joint}")


def calls_source(unrelated: int, branch: int, calls: int) -> str:
    """A program whose recursion calls a computation, in a larger module.

    Parameters
    ----------
    unrelated : int
        Computations the program never calls.
    branch : int
        Lets in a branch of the called computation that the runs never
        take.
    calls : int
        The calls the recursion makes.

    Returns
    -------
    str
        The source.
    """
    extra = "".join(
        f"define unrelated_{k}(v : Real) : Real !{{}} =\n    return v + {k}.0\n"
        for k in range(unrelated)
    )
    lets = "".join(f"        let a{i} = a{i - 1} + 1.0\n" for i in range(1, branch + 1))
    return f"""
object Unit : FinSet 1
object V : Real 1
{extra}
define step(v : Real) : Real !{{}} =
    if v > 1000.0 then
        let a0 = v
{lets}        return a{branch}
    else
        return v + 1.0
define repeat(n : Int, v : Real) : Real !{{}} =
    if n <= 0 then
        return v
    else
        let w <- step(v)
        let rest <- repeat(n - 1, w)
        return rest
program calls : Unit -> V
    sample t <- Normal(loc=0.0, scale=1.0)
    let k <- repeat({calls}, t)
    observe y <- Normal(loc=k, scale=1.0)
    return t
export calls
"""


def run_work(unrelated: int, branch: int, calls: int) -> dict[str, int]:
    """The records and machine steps of the second of two joints.

    Parameters
    ----------
    unrelated : int
        Computations the program never calls.
    branch : int
        Lets of the branch the runs never take.
    calls : int
        The calls of the recursion.

    Returns
    -------
    dict of str to int
        ``"made"``, the records validated, and ``"steps"``, the machine
        steps.
    """
    model = loads(calls_source(unrelated, branch, calls)).morphism
    assert isinstance(model, CheckedProgram)
    given = {"t": torch.tensor([0.3]), "y": torch.tensor([0.5])}
    with double_precision():
        model.log_joint(torch.zeros(1, 1), given)
        with CallCounter(
            {"made": dx.Model._validate_instance, "steps": Evaluator._tick}
        ) as counter:
            model.log_joint(torch.zeros(1, 1), given)
    return dict(counter.counts)


def sequence_source(binds: int) -> str:
    """A program of a long sequence of binds, every one kept in scope.

    Parameters
    ----------
    binds : int
        The binds after the draw.

    Returns
    -------
    str
        The source.
    """
    lets = "".join(
        f"    let a{i} <- step({'t' if i == 0 else f'a{i - 1}'})\n"
        for i in range(binds)
    )
    return f"""
object Unit : FinSet 1
object V : Real 1
define step(v : Real) : Real !{{}} =
    return v + 1.0
program sequence : Unit -> V
    sample t <- Normal(loc=0.0, scale=1.0)
{lets}    observe y <- Normal(loc=a{binds - 1}, scale=1.0)
    return t
export sequence
"""


def sequence_work(binds: int) -> dict[str, int]:
    """The scope work and machine steps of the second of two joints.

    Parameters
    ----------
    binds : int
        The binds of the sequence.

    Returns
    -------
    dict of str to int
        ``"leaves"``, the new bindings, ``"nodes"``, the trie nodes visited
        while binding, and ``"steps"``, the machine steps. The initial draw
        and the enclosing call bind twice together. Each subsequent call
        binds its argument and the local for its return value. Each new
        binding visits at most ``ceil(64 / 5) + 1 = 14`` trie nodes.

    Raises
    ------
    Mismatch
        If a binding visits more nodes than the trie can have on a path.
    """
    compiled = loads(sequence_source(binds))
    program = compiled.morphism
    module = compiled.qiec
    assert isinstance(program, MonadicProgram) and module is not None
    model = CheckedProgram(program.domain, program.codomain, module, "sequence")
    given = {"t": torch.tensor([0.3]), "y": torch.tensor([0.5])}
    with double_precision():
        model.log_joint(torch.zeros(1, 1), given)
        with CallCounter(
            {
                "leaves": scope._Leaf.__init__,
                "nodes": scope._bound,
                "steps": Evaluator._tick,
            }
        ) as counter:
            model.log_joint(torch.zeros(1, 1), given)
    counts = dict(counter.counts)
    limit = 14 * counts["leaves"]
    if counts["nodes"] > limit:
        raise Mismatch("work", counts["nodes"] / limit, "scope trie path length")
    return counts


def deduction_records(unrelated: int) -> int:
    """The records one run of a deduction makes beside unrelated ones.

    Parameters
    ----------
    unrelated : int
        Deductions of the module the run never uses.

    Returns
    -------
    int
        Records validated by the second of two runs.
    """
    extra = "".join(
        f"""
deduction X{k} : Term -> Term [semiring=Counting, start=S, depth=1]
    atoms S, span, a
    rule join : span(I, K, S), span(K, J, S) |- span(I, J, S)
    lexicon
        "a" : S = a
"""
        for k in range(unrelated)
    )
    module = loads(LEXICAL.replace("SEMIRING", "Counting") + extra).qiec
    assert module is not None
    run_deduction(module, "G", tokens=["b"])
    with CallCounter({"made": dx.Model._validate_instance}) as calls:
        run_deduction(module, "G", tokens=["b"])
    return calls.counts["made"]


FAMILIES: tuple[Family, ...] = (
    Family(
        name="machine runs of a deduction beside unrelated declarations",
        executor="machine-lexical",
        question="Q24",
        varies="declarations",
        part="validation",
        sizes=(1, 2, 4, 8, 16),
        work=deduction_records,
        claim=lambda size: 1.0,
        constant=256.0,
    ),
    Family(
        name="checked runs of a module with unrelated declarations",
        executor="machine-program",
        question="Q24",
        varies="declarations",
        part="validation",
        sizes=(1, 2, 4, 8, 16),
        work=lambda size: run_work(size, 1, 8)["made"],
        claim=lambda size: 1.0,
        constant=64.0,
    ),
    Family(
        name="checked calls of a computation with a growing untaken branch",
        executor="machine-program",
        question="Q24",
        varies="branch",
        part="validation",
        sizes=(4, 8, 16, 32, 64),
        work=lambda size: run_work(1, size, 8)["made"],
        claim=lambda size: 1.0,
        constant=64.0,
    ),
    Family(
        name="checked runs of a long sequence of binds",
        executor="machine-program",
        question="Q24",
        varies="binds",
        part="forward",
        sizes=(16, 32, 64, 128, 256),
        work=lambda size: sequence_work(size)["leaves"],
        claim=lambda size: 2.0 * (size + 1),
        constant=1.0,
    ),
    Family(
        name="checked repeated calls of one computation",
        executor="machine-program",
        question="Q24",
        varies="calls",
        part="forward",
        sizes=(4, 8, 16, 32, 64),
        work=lambda size: run_work(1, 1, size)["steps"],
        claim=lambda size: float(size),
        constant=24.0,
    ),
)
"""The work of checked runs: no record of the module or of a callee's body
is made again per run or per call, a run's machine steps grow with the
calls it makes, and a bind adds a bounded number of nodes to the run's
environment however many locals are in scope."""


def _register() -> None:
    """Register the checks."""
    add("checked/sites/flips", "trace", "R6", check_sites_and_joints)
    add("checked/forward/flips", "trace", "R6", check_forward_runs)
    add("checked/runs/one-pass", "machine-program", "Q24", check_one_pass)
    add("checked/sites/changed", "trace", "R6", check_changed_sites)
    add("checked/observations/roles", "predictive", "R6", check_observation_roles)
    add("checked/forward/one-pass", "machine-program", "Q24", check_vectorized_forward)
    for family in FAMILIES:
        add(
            f"work/{family.name.replace(' ', '-')}",
            family.executor,
            family.question,
            partial(check_family, family),
        )


_register()

__all__ = ["DEPTH", "FIRST", "HEADS", "flips", "trace_names"]
