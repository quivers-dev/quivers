"""Checks of summed-out latents against enumerated draws."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from functools import partial

import torch

from quivers.continuous.families import ConditionalNormal
from quivers.continuous.plate import marginalize_categorical, marginalize_grouped
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import Euclidean, ProductSpace
from quivers.effects import collapse
from quivers.inference.trace import trace
from quivers.qiec.program_runtime import run_program
from tests.evidence.checks._registry import add
from tests.evidence.checks.span import double_precision
from tests.evidence.compare import (
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_gradients,
    compare_number,
    compare_tensor,
)
from tests.evidence.oracles import exact, finite_effects, gradients
from tests.evidence.oracles.finite_effects import Draws
from tests.evidence.problems import logarithm
from tests.evidence.sources import program

PRIOR = (Fraction(1, 4), Fraction(3, 4))
"""The distribution of a draw."""

ROWS = (1, 0, 1, 1)
"""The observed rows."""

GROUPS = (0, 1, 1, 0)
"""The group of each row."""

INITIAL = (Fraction(1, 4), Fraction(3, 4))
"""The distribution of the state before the first symbol."""

TRANSITION = ((Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 8), Fraction(7, 8)))
"""The transition rows."""

EMISSION = (
    (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)),
    (Fraction(1, 8), Fraction(3, 8), Fraction(1, 2)),
)
"""The emission rows."""

SYMBOLS = (0, 2, 1, 1)
"""The observed symbols."""

#: The log density of one uniform distribution over three categories; over
#: two categories the density is one.
UNIFORM_THREE = math.log(2)


def rate(value: int) -> Fraction:
    """The probability of a one under a draw.

    Parameters
    ----------
    value : int
        The draw.

    Returns
    -------
    Fraction
        The probability.
    """
    return Fraction(1, 8) + Fraction(value, 2)


def mixture(draws: Draws, groups: Sequence[int]) -> int:
    """The program of a mixture with one draw per group.

    Parameters
    ----------
    draws : Draws
        The run.
    groups : Sequence of int
        The group of each row.

    Returns
    -------
    int
        Always zero; the weight is what is read.
    """
    values = {
        group: draws.choose(f"a{group}", PRIOR) for group in sorted(set(groups))
    }
    for row, group in zip(ROWS, groups, strict=True):
        draws.score(rate(values[group]) if row else 1 - rate(values[group]))
    return 0


def chain(draws: Draws) -> int:
    """The program of a chain of states that emits symbols.

    Parameters
    ----------
    draws : Draws
        The run.

    Returns
    -------
    int
        Always zero; the weight is what is read.
    """
    state = draws.choose("z", INITIAL)
    for position, symbol in enumerate(SYMBOLS):
        state = draws.choose(f"z{position}", TRANSITION[state])
        draws.score(EMISSION[state][symbol])
    return 0


def mixture_accuracy(backward: bool = False) -> Accuracy:
    """The accuracy of a small mixture.

    A class scores four rows with one logarithm each and the classes of
    two groups are combined once, so a path has fewer than forty
    operations of four roundings. The parameters, their densities, and
    the conversions add as many.

    Parameters
    ----------
    backward : bool
        Whether the query is a gradient.

    Returns
    -------
    Accuracy
        The accuracy.
    """
    count = 80 * ELEMENTARY
    if backward:
        return Accuracy(domain="linear", operations=5 * count, condition=16.0)
    return Accuracy(domain="log", operations=count, magnitude=8.0)


def floats(values: Sequence[Fraction]) -> torch.Tensor:
    """Rationals as a float64 vector.

    Parameters
    ----------
    values : Sequence of Fraction
        The rationals.

    Returns
    -------
    torch.Tensor
        The vector.
    """
    return torch.tensor([float(value) for value in values], dtype=torch.float64)


def log_joint(name: str, prior: torch.Tensor) -> torch.Tensor:
    """The compiled log joint of a mixture program.

    Parameters
    ----------
    name : str
        ``"grouped"`` or ``"ungrouped"``.
    prior : torch.Tensor
        The distribution of a draw.

    Returns
    -------
    torch.Tensor
        The log joint of one run.
    """
    model = program(name).morphism
    assert isinstance(model, MonadicProgram)
    sites = {"prior": prior, "y": torch.tensor(ROWS, dtype=torch.bool)}
    if name == "grouped":
        sites["idx"] = torch.tensor(GROUPS)
    return model.log_joint(torch.zeros(1, 1), sites)


def groups_of(name: str) -> tuple[int, ...]:
    """The groups of a mixture's rows.

    Parameters
    ----------
    name : str
        ``"grouped"`` or ``"ungrouped"``.

    Returns
    -------
    tuple of int
        The groups.
    """
    return GROUPS if name == "grouped" else (0,) * len(ROWS)


def check_compiled(name: str) -> None:
    """Compare a compiled mixture with every draw enumerated.

    Parameters
    ----------
    name : str
        ``"grouped"`` or ``"ungrouped"``.
    """
    with double_precision():
        value = log_joint(name, floats(PRIOR))
    wanted = logarithm(
        finite_effects.evidence(partial(mixture, groups=groups_of(name)))
    )
    compare_tensor(
        value,
        torch.tensor([wanted], dtype=torch.float64),
        mixture_accuracy(),
        f"compiled {name}",
    )


def check_machine(name: str) -> None:
    """Compare a mixture on the reference machine with every draw enumerated.

    Parameters
    ----------
    name : str
        ``"grouped"`` or ``"ungrouped"``.
    """
    module = program(name).qiec
    assert module is not None
    data: dict[str, tuple[bool, ...] | tuple[int, ...]] = {
        "y": tuple(bool(row) for row in ROWS)
    }
    if name == "grouped":
        data["idx"] = GROUPS
    run = run_program(
        module,
        "mixture",
        data=data,
        sites={"prior": tuple(float(value) for value in PRIOR)},
    )
    assert isinstance(run.log_joint, float)
    compare_number(
        run.log_joint,
        logarithm(finite_effects.evidence(partial(mixture, groups=groups_of(name)))),
        mixture_accuracy(),
        f"machine {name}",
    )


def check_compiled_gradient(name: str) -> None:
    """Compare a mixture's gradient in the class weights with the exact one.

    A categorical reads its class weights up to a factor, so the function
    differentiated is the log weight at the weights divided by their sum.

    Parameters
    ----------
    name : str
        ``"grouped"`` or ``"ungrouped"``.
    """
    with double_precision():
        prior = floats(PRIOR).requires_grad_()
        (found,) = torch.autograd.grad(log_joint(name, prior).sum(), prior)
    groups = groups_of(name)

    def likelihood(weights: dict[str, torch.Tensor]) -> torch.Tensor:
        """The log likelihood as a function of the class probabilities.

        Parameters
        ----------
        weights : dict of str to torch.Tensor
            The probabilities.

        Returns
        -------
        torch.Tensor
            The log likelihood, summed over the draws of every group.
        """
        answer = torch.zeros((), dtype=torch.float64)
        scale = weights["p0"] + weights["p1"]
        for group in sorted(set(groups)):
            total = torch.zeros((), dtype=torch.float64)
            for value in (0, 1):
                product = weights[f"p{value}"] / scale
                for row, of in zip(ROWS, groups, strict=True):
                    if of == group:
                        chance = float(rate(value))
                        product = product * (chance if row else 1 - chance)
                total = total + product
            answer = answer + total.log()
        return answer

    reference = gradients.autograd(
        lambda weights: likelihood(dict(weights)),
        {f"p{value}": float(PRIOR[value]) for value in (0, 1)},
    )
    compare_gradients(
        {f"p{value}": found[value] for value in (0, 1)},
        reference,
        mixture_accuracy(backward=True),
        f"gradient of {name}",
    )


def chain_sites() -> dict[str, torch.Tensor]:
    """The parameters of the chain as tensors.

    Returns
    -------
    dict of str to torch.Tensor
        The sites.
    """
    return {
        "initial_row": floats(INITIAL),
        "transition_rows": torch.stack([floats(row) for row in TRANSITION]),
        "emission_rows": torch.stack([floats(row) for row in EMISSION]),
    }


def check_chain() -> None:
    """Compare a summed-out chain with every path enumerated."""
    model = program("hmm").morphism
    assert isinstance(model, MonadicProgram)
    with double_precision():
        value = model.log_joint(torch.tensor([SYMBOLS]), chain_sites())
    wanted = logarithm(finite_effects.evidence(chain)) + 2 * UNIFORM_THREE
    compare_tensor(
        value,
        torch.tensor([wanted], dtype=torch.float64),
        mixture_accuracy(),
        "chain",
    )


def check_chain_batch() -> None:
    """Compare a batch of chains with one exact recursion each."""
    model = program("hmm").morphism
    assert isinstance(model, MonadicProgram)
    rows = ((0, 2, 1), (1, 1, 1), (2, 0, 0))
    with double_precision():
        value = model.log_joint(torch.tensor(rows), chain_sites())
    wanted = [
        logarithm(exact.forward(INITIAL, TRANSITION, EMISSION, row))
        + 2 * UNIFORM_THREE
        for row in rows
    ]
    compare_tensor(
        value,
        torch.tensor(wanted, dtype=torch.float64),
        mixture_accuracy(),
        "chain batch",
    )


def check_chain_gradient() -> None:
    """Compare a chain's gradient with expected counts over enumerated paths.

    A categorical reads a row up to a factor, so the derivative in an
    entry is its expected count over the entry minus the expected count
    of the row.
    """
    model = program("hmm").morphism
    assert isinstance(model, MonadicProgram)
    with double_precision():
        sites = {name: value.requires_grad_() for name, value in chain_sites().items()}
        value = model.log_joint(torch.tensor([SYMBOLS]), sites).sum()
        found = torch.autograd.grad(value, list(sites.values()))
    mass = finite_effects.evidence(chain)
    uses: dict[str, Fraction] = {}
    for outcome in finite_effects.runs(chain):
        states = outcome.values
        names = [f"i{states[0]}"]
        for position, symbol in enumerate(SYMBOLS):
            names.append(f"t{states[position]}{states[position + 1]}")
            names.append(f"e{states[position + 1]}{symbol}")
        for name in names:
            uses[name] = uses.get(name, Fraction(0)) + outcome.weight / mass
    grads: dict[str, torch.Tensor | None] = {}
    wanted: dict[str, Fraction] = {}

    def row(label: str, values: Sequence[Fraction], entries: torch.Tensor) -> None:
        """Add the derivatives of one distribution.

        Parameters
        ----------
        label : str
            The distribution's name.
        values : Sequence of Fraction
            Its entries.
        entries : torch.Tensor
            The computed derivatives.
        """
        counts = [uses.get(f"{label}{k}", Fraction(0)) for k in range(len(values))]
        for k, value in enumerate(values):
            grads[f"{label}{k}"] = entries[k]
            wanted[f"{label}{k}"] = counts[k] / value - sum(counts, Fraction(0))

    row("i", INITIAL, found[0])
    for k in (0, 1):
        row(f"t{k}", TRANSITION[k], found[1][k])
        row(f"e{k}", EMISSION[k], found[2][k])
    compare_gradients(grads, wanted, mixture_accuracy(backward=True), "chain gradient")


def check_primitives() -> None:
    """Compare the two mixture primitives with sums of products.

    The grouped primitive scatters rows to groups, so a row of another
    group must not move a group's term.
    """
    classes = torch.stack(
        [
            floats([rate(k) if row else 1 - rate(k) for k in (0, 1)]).log()
            for row in ROWS
        ]
    )
    log_prior = floats(PRIOR).log()
    found = marginalize_grouped(classes, torch.tensor(GROUPS), log_prior, 2)
    wanted = logarithm(finite_effects.evidence(partial(mixture, groups=GROUPS)))
    compare_number(
        float(found), wanted, mixture_accuracy(), "grouped primitive"
    )
    single = marginalize_categorical(classes.sum(0, keepdim=True) + log_prior)
    alone = logarithm(
        finite_effects.evidence(partial(mixture, groups=(0,) * len(ROWS)))
    )
    compare_tensor(
        single,
        torch.tensor([alone], dtype=torch.float64),
        mixture_accuracy(),
        "ungrouped primitive",
    )


def check_empty_group() -> None:
    """Require a group without rows to contribute the mass of its prior."""
    classes = floats(PRIOR).log().expand(2, 2)
    found = marginalize_grouped(
        classes,
        torch.tensor([0, 0]),
        floats((Fraction(1, 8), Fraction(1, 2))).log(),
        2,
    )
    first = sum(
        (
            weight * chance * chance
            for weight, chance in zip(
                (Fraction(1, 8), Fraction(1, 2)), PRIOR, strict=True
            )
        ),
        Fraction(0),
    )
    compare_number(
        float(found),
        logarithm(first) + logarithm(Fraction(5, 8)),
        mixture_accuracy(),
        "a group without rows",
    )


def check_impossible_class() -> None:
    """Require a class of weight zero to leave a finite gradient elsewhere."""
    classes = torch.tensor(
        [[-math.inf, 0.0], [-math.inf, -1.0]], dtype=torch.float64, requires_grad=True
    )
    found = marginalize_grouped(
        classes, torch.tensor([0, 1]), torch.zeros(2, dtype=torch.float64), 2
    )
    (gradient,) = torch.autograd.grad(found, classes)
    compare_gradients(
        {f"{i}{k}": gradient[i, k] for i in (0, 1) for k in (0, 1)},
        {"00": 0.0, "01": 1.0, "10": 0.0, "11": 1.0},
        mixture_accuracy(backward=True),
        "an impossible class",
    )


def check_unknown_reduction() -> None:
    """Require an unknown reduction to be refused.

    Raises
    ------
    Mismatch
        If a number is returned.
    """
    try:
        value = marginalize_grouped(
            torch.zeros(2, 2), torch.tensor([0, 0]), torch.zeros(2), 1, reduction="median"
        )
    except ValueError as refusal:
        if "median" in str(refusal):
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"a median returned {float(value)}")


def check_scan() -> None:
    """Compare a scan's score of a trajectory with the sum of its steps."""
    inputs = Euclidean(name="Input", dim=1)
    hidden = Euclidean(name="Hidden", dim=1)
    with double_precision():
        torch.manual_seed(0)
        cell = ConditionalNormal(ProductSpace(components=(inputs, hidden)), hidden)
        scan = ScanMorphism(cell).double()
        x = torch.tensor([[[0.5], [-1.0], [2.0]]])
        states = torch.tensor([[[0.25], [1.5], [-0.75]]])
        with torch.no_grad():
            found = scan.log_prob(x, states)
            before = 0.0
            wanted = 0.0
            for position in range(3):
                raw = cell.param_source(
                    torch.tensor([[float(x[0, position, 0]), before]])
                )[0]
                location, log_scale = float(raw[0]), float(raw[1])
                at = float(states[0, position, 0])
                wanted += (
                    -0.5 * ((at - location) / math.exp(log_scale)) ** 2
                    - log_scale
                    - 0.5 * math.log(2 * math.pi)
                )
                before = at
    compare_tensor(
        found,
        torch.tensor([wanted], dtype=torch.float64),
        Accuracy(domain="log", operations=3 * 12 * ELEMENTARY, magnitude=64.0),
        "scan trajectory",
    )


def check_scan_refuses_final_state() -> None:
    """Require a scan to refuse to score a final state alone.

    Raises
    ------
    Mismatch
        If a number is returned.
    """
    inputs = Euclidean(name="Input", dim=1)
    hidden = Euclidean(name="Hidden", dim=1)
    scan = ScanMorphism(
        ConditionalNormal(ProductSpace(components=(inputs, hidden)), hidden)
    )
    try:
        value = scan.log_prob(torch.zeros(1, 3, 1), torch.zeros(1, 1))
    except ValueError:
        return
    raise Mismatch("refusal", math.inf, f"a final state scored {value.tolist()}")


def check_collapse() -> None:
    """Compare a collapsed normal pair with its closed-form marginal."""
    model = program("conjugate").morphism
    assert isinstance(model, MonadicProgram)
    value = 0.75
    with double_precision(), collapse({"t": "y"}):
        run = trace(
            model, torch.zeros(1, 1), observations={"y": torch.tensor([value])}
        )
    variance = 2.0**2 + 0.5**2
    wanted = -0.5 * (value - 0.5) ** 2 / variance - 0.5 * math.log(
        2 * math.pi * variance
    )
    assert run.log_joint is not None
    compare_tensor(
        run.log_joint,
        torch.tensor([wanted], dtype=torch.float64),
        Accuracy(domain="log", operations=12 * ELEMENTARY, magnitude=2.0),
        "collapsed pair",
    )


def check_collapse_refusal(case: str) -> None:
    """Require a pair without a registered marginal to be refused by name.

    Parameters
    ----------
    case : str
        ``"plate"`` for a plated child, ``"missing"`` for a child without
        an observation.

    Raises
    ------
    Mismatch
        If a trace is returned or the diagnostic differs.
    """
    if case == "plate":
        model = program("beta").morphism
        observations = {"y": torch.tensor(ROWS, dtype=torch.bool)}
        wanted = "draws from no registered family"
        pairs = {"theta": "y"}
    else:
        model = program("conjugate").morphism
        observations = {}
        wanted = "collapse needs an observation for child 'y'"
        pairs = {"t": "y"}
    assert isinstance(model, MonadicProgram)
    try:
        with collapse(pairs):
            run = trace(model, torch.zeros(1, 1), observations=observations)
    except Exception as refusal:
        if wanted in str(refusal):
            return
        raise Mismatch("refusal", math.inf, str(refusal)) from refusal
    raise Mismatch("refusal", math.inf, f"{case} returned {run.log_joint}")


def _register() -> None:
    """Register every check of this module."""
    for name in ("grouped", "ungrouped"):
        row = f"marginalize-{name}"
        add(f"mixture/compiled/{name}", row, "Q6", partial(check_compiled, name))
        add(f"mixture/machine/{name}", row, "Q1", partial(check_machine, name))
        add(f"mixture/gradient/{name}", row, "Q1g", partial(check_compiled_gradient, name))
    add("mixture/primitives", "marginalize-grouped", "Q6", check_primitives)
    add("mixture/empty_group", "marginalize-grouped", "Q6", check_empty_group)
    add("mixture/impossible_class", "marginalize-grouped", "Q1g", check_impossible_class)
    add("mixture/unknown_reduction", "marginalize-grouped", "Q24", check_unknown_reduction)
    add("mixture/chain", "marginalize-chain", "Q10", check_chain)
    add("mixture/chain_batch", "marginalize-chain", "Q10", check_chain_batch)
    add("mixture/chain_gradient", "marginalize-chain", "Q1g", check_chain_gradient)
    add("mixture/chain_counts", "marginalize-chain", "Q11", check_chain_gradient)
    add("mixture/scan", "scan", "Q10", check_scan)
    add("mixture/scan_final_state", "scan", "Q24", check_scan_refuses_final_state)
    add("mixture/collapse", "collapse", "Q6", check_collapse)
    for case in ("plate", "missing"):
        add(
            f"mixture/collapse_refuses/{case}",
            "collapse",
            "Q24",
            partial(check_collapse_refusal, case),
        )


_register()
