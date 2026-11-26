"""Checks of every inference consumer against exact log joints.

Each problem has a log joint written out from its definition and summed
over its discrete draws by enumeration. A consumer is checked at the
values it produces: the density it reports, or the objective it
minimizes, has to equal the one computed here. Consumers that only
produce draws are checked against an exact distribution function with a
bound that holds for every distribution.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from functools import partial
from typing import cast

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.effects.replay import replay
from quivers.inference import (
    ELBO,
    MCMC,
    SMC,
    SVI,
    AmortizedGuide,
    AutoDAIS,
    AutoDeltaGuide,
    AutoGuideList,
    AutoIAFGuide,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormalGuide,
    AutoMixtureGuide,
    AutoMultivariateNormalGuide,
    AutoNeuralSplineGuide,
    AutoNormalGuide,
    AutoStructured,
    GibbsKernel,
    Guide,
    HMCKernel,
    LatentRegistry,
    MCMCKernel,
    MetropolisKernel,
    NUTSKernel,
    Pathfinder,
    Predictive,
    PriorDraws,
    RandomWalkProposal,
    prior_draws,
    trace,
)
from quivers.inference.mcmc import PotentialFn
from quivers.inference.smc import TemperedPotential
from tests.evidence.checks._registry import add
from tests.evidence.checks.mixture import (
    EMISSION,
    INITIAL,
    PRIOR,
    ROWS,
    SYMBOLS,
    TRANSITION,
    chain,
    mixture,
    rate,
)
from tests.evidence.checks.span import double_precision
from tests.evidence.compare import (
    ELEMENTARY,
    Accuracy,
    Mismatch,
    compare_number,
)
from tests.evidence.oracles import closed_forms, derivations, finite_effects
from tests.evidence.oracles.derivations import Variable
from tests.evidence.sources import program

type Values = Mapping[str, torch.Tensor]
"""One draw of every latent, without batch axes."""

type Data = dict[str, torch.Tensor]
"""The observations of a problem, as the consumers annotate them."""

CORPUS = (("a", "b"), ("b", "a"))
"""The sentences of the grammar problem."""


class ConsumerProblem(ValueModel):
    """A model with observations and its exact log joint.

    Parameters
    ----------
    name : str
        The source's name.
    latents : tuple of str
        The latent sites.
    log_prior : Callable[[Values], torch.Tensor]
        The log prior of one draw.
    log_likelihood : Callable[[Values], torch.Tensor]
        The log likelihood of one draw.
    """

    name: str
    latents: tuple[str, ...]
    log_prior: Callable[[Values], torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_likelihood: Callable[[Values], torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    def model(self) -> MonadicProgram:
        """The compiled program.

        Returns
        -------
        MonadicProgram
            The program.
        """
        # A program that does not compile to steps is handed on as it is,
        # so that a consumer's own diagnostic is seen.
        return cast(MonadicProgram, program(self.name).morphism)

    def data(self) -> tuple[torch.Tensor, Data]:
        """The program's input and observations.

        Returns
        -------
        tuple of (torch.Tensor, Data)
            The input and the observations.
        """
        x = torch.zeros(1, 1)
        if self.name in ("beta", "ungrouped"):
            return x, {
                "y": torch.tensor(ROWS, dtype=torch.bool),
                "features": torch.tensor([[0.5, -1.0, 2.0]]),
            }
        if self.name == "hmm":
            return torch.tensor([SYMBOLS]), {}
        if self.name == "parsed":
            # The consumers read a corpus of words through an observation
            # whose annotation names tensors only.
            corpus = cast(torch.Tensor, [list(sentence) for sentence in CORPUS])
            return x, {"corpus": corpus}
        return x, {"y": torch.tensor([0.5])}

    def log_joint(self, values: Values) -> torch.Tensor:
        """The exact log joint of one draw.

        Parameters
        ----------
        values : Values
            The draw.

        Returns
        -------
        torch.Tensor
            The log joint.
        """
        return self.log_prior(values) + self.log_likelihood(values)


def _beta_prior(values: Values) -> torch.Tensor:
    """The log density of a Beta(2, 3) variable.

    Parameters
    ----------
    values : Values
        The draw.

    Returns
    -------
    torch.Tensor
        The log density, whose constant is ``log 12``.
    """
    theta = values["theta"].reshape(())
    return math.log(12) + theta.log() + 2 * (1 - theta).log()


def _beta_likelihood(values: Values) -> torch.Tensor:
    """The log likelihood of the rows.

    Parameters
    ----------
    values : Values
        The draw.

    Returns
    -------
    torch.Tensor
        The log likelihood.
    """
    theta = values["theta"].reshape(())
    ones = sum(ROWS)
    return ones * theta.log() + (len(ROWS) - ones) * (1 - theta).log()


def _nothing(values: Values) -> torch.Tensor:
    """The log density of a uniform variable over two categories.

    Parameters
    ----------
    values : Values
        The draw, unused.

    Returns
    -------
    torch.Tensor
        Zero.
    """
    del values
    return torch.zeros((), dtype=torch.float64)


def _mixture_likelihood(values: Values) -> torch.Tensor:
    """The log likelihood of one shared draw, summed over its values.

    Parameters
    ----------
    values : Values
        The draw of the class weights.

    Returns
    -------
    torch.Tensor
        The log likelihood.
    """
    weights = values["prior"].reshape(2)
    total = torch.zeros((), dtype=torch.float64)
    for value in (0, 1):
        chance = Fraction(1)
        for row in ROWS:
            chance *= rate(value) if row else 1 - rate(value)
        total = total + weights[value] * float(chance)
    return total.log()


def _chain_prior(values: Values) -> torch.Tensor:
    """The log density of two uniform rows over three categories.

    Parameters
    ----------
    values : Values
        The draw, unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del values
    return torch.tensor(2 * math.log(2), dtype=torch.float64)


def _chain_likelihood(values: Values) -> torch.Tensor:
    """The log likelihood of the symbols, summed over every path.

    Parameters
    ----------
    values : Values
        The draw of the rows.

    Returns
    -------
    torch.Tensor
        The log likelihood.
    """
    initial = values["initial_row"].reshape(2)
    transition = values["transition_rows"].reshape(2, 2)
    emission = values["emission_rows"].reshape(2, 3)
    total = torch.zeros((), dtype=torch.float64)
    for outcome in finite_effects.runs(chain):
        states = outcome.values
        product = initial[states[0]]
        for position, symbol in enumerate(SYMBOLS):
            product = product * transition[states[position], states[position + 1]]
            product = product * emission[states[position + 1], symbol]
        total = total + product
    return total.log()


def _parsed_prior(values: Values) -> torch.Tensor:
    """The log density of one uniform variable over three categories.

    Parameters
    ----------
    values : Values
        The draw, unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del values
    return torch.tensor(math.log(2), dtype=torch.float64)


def _parsed_likelihood(values: Values) -> torch.Tensor:
    """The log likelihood of the sentences, summed over every derivation.

    Parameters
    ----------
    values : Values
        The draw of the production probabilities.

    Returns
    -------
    torch.Tensor
        The log likelihood.
    """
    of_s = values["p_s"].reshape(3)
    of_v = values["p_v"].reshape(2)
    weights = {
        "join": of_s[0],
        "lift": of_s[1],
        "S:a": of_s[2],
        "V:b": of_v[0],
        "V:a": of_v[1],
    }
    answer = torch.zeros((), dtype=torch.float64)
    for sentence in CORPUS:
        axioms = [
            (f"{entry}", ("span", i, i + 1, entry[0]))
            for i, word in enumerate(sentence)
            for entry in ("S:a", "V:b", "V:a")
            if entry[2] == word
        ]
        i, k, j = Variable("i"), Variable("k"), Variable("j")
        trees = derivations.derivations(
            axioms,
            [
                (
                    "join",
                    (("span", i, k, "S"), ("span", k, j, "S")),
                    ("span", i, j, "S"),
                    "join",
                ),
                ("lift", (("span", i, j, "V"),), ("span", i, j, "S"), "lift"),
            ],
            ("span", 0, len(sentence), "S"),
        )
        answer = (
            answer
            + derivations.total(
                trees,
                weights,
                torch.zeros((), dtype=torch.float64),
                torch.ones((), dtype=torch.float64),
            ).log()
        )
    return answer


def _discrete_prior(values: Values) -> torch.Tensor:
    """The log probability of a coin with probability one quarter.

    Parameters
    ----------
    values : Values
        The draw.

    Returns
    -------
    torch.Tensor
        The log probability.
    """
    chance = 0.25 if bool(values["a"].reshape(())) else 0.75
    return torch.tensor(math.log(chance), dtype=torch.float64)


def _discrete_likelihood(values: Values) -> torch.Tensor:
    """The log density of the observation given the coin.

    Parameters
    ----------
    values : Values
        The draw.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    location = 2.0 * float(bool(values["a"].reshape(())))
    return torch.tensor(
        -0.5 * (0.5 - location) ** 2 - 0.5 * math.log(2 * math.pi), dtype=torch.float64
    )


def _recursive_prior(values: Values) -> torch.Tensor:
    """The log density of a standard normal draw.

    Parameters
    ----------
    values : Values
        The draw.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    t = values["t"].reshape(()).to(torch.float64)
    return -0.5 * t * t - 0.5 * math.log(2 * math.pi)


def _recursive_likelihood(values: Values) -> torch.Tensor:
    """The log density of the observation, whose mean a recursion shifts by three.

    Parameters
    ----------
    values : Values
        The draw.

    Returns
    -------
    torch.Tensor
        The log density of ``y = 1/2`` under a normal of mean ``t + 3``.
    """
    t = values["t"].reshape(()).to(torch.float64)
    return -0.5 * (0.5 - t - 3.0) ** 2 - 0.5 * math.log(2 * math.pi)


PROBLEMS: dict[str, ConsumerProblem] = {
    problem.name: problem
    for problem in (
        ConsumerProblem(
            name="beta",
            latents=("theta",),
            log_prior=_beta_prior,
            log_likelihood=_beta_likelihood,
        ),
        ConsumerProblem(
            name="ungrouped",
            latents=("prior",),
            log_prior=_nothing,
            log_likelihood=_mixture_likelihood,
        ),
        ConsumerProblem(
            name="hmm",
            latents=("initial_row", "transition_rows", "emission_rows"),
            log_prior=_chain_prior,
            log_likelihood=_chain_likelihood,
        ),
        ConsumerProblem(
            name="parsed",
            latents=("p_s", "p_v"),
            log_prior=_parsed_prior,
            log_likelihood=_parsed_likelihood,
        ),
        ConsumerProblem(
            name="discrete",
            latents=("a",),
            log_prior=_discrete_prior,
            log_likelihood=_discrete_likelihood,
        ),
        ConsumerProblem(
            name="recursive",
            latents=("t",),
            log_prior=_recursive_prior,
            log_likelihood=_recursive_likelihood,
        ),
    )
}
"""The consumer problems by name."""

CONTINUOUS = ("beta", "ungrouped", "hmm", "parsed", "recursive")
"""Problems whose latents are continuous."""

QUESTIONS = {
    "beta": "R1",
    "ungrouped": "R2",
    "hmm": "R3",
    "parsed": "R4",
    "discrete": "R5",
    "recursive": "R6",
}
"""The consumer column of each problem."""


def joint_accuracy(reference: float, condition: float = 1.0) -> Accuracy:
    """The accuracy of a log joint.

    A path through any of the problems has fewer than two hundred
    operations of four roundings. A coordinate near the edge of its
    support changes the log joint by its relative rounding times at most
    the number of observed rows, which is the condition number.

    Parameters
    ----------
    reference : float
        The log joint, whose size scales the tolerance.
    condition : float
        A further condition number.

    Returns
    -------
    Accuracy
        The accuracy.
    """
    return Accuracy(
        domain="log",
        operations=200 * ELEMENTARY,
        magnitude=max(16.0, abs(reference)),
        condition=8.0 * condition,
    )


def edge(values: Values) -> float:
    """The reciprocal of the smallest coordinate of a draw on its support.

    Parameters
    ----------
    values : Values
        A draw of probabilities.

    Returns
    -------
    float
        The condition number of taking logarithms of the coordinates.
    """
    sizes = [
        float(torch.minimum(value, 1 - value).abs().min()) for value in values.values()
    ]
    return 1.0 / max(min(sizes), 1e-12)


def check_trace(name: str) -> None:
    """Compare a traced log joint with the exact one at the traced values.

    Parameters
    ----------
    name : str
        The problem.
    """
    problem = PROBLEMS[name]
    with double_precision():
        torch.manual_seed(0)
        x, data = problem.data()
        run = trace(problem.model(), x, observations=data)
    values = {key: run.sites[key].value.detach() for key in problem.latents}
    assert run.log_joint is not None
    wanted = float(problem.log_joint(values))
    compare_number(
        float(run.log_joint.sum()), wanted, joint_accuracy(wanted), f"trace {name}"
    )


def check_replay(name: str) -> None:
    """Require a replayed trace to keep its values and its log joint.

    Parameters
    ----------
    name : str
        The problem.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS[name]
    with double_precision():
        torch.manual_seed(1)
        x, data = problem.data()
        first = trace(problem.model(), x, observations=data)
        with replay(first):
            second = trace(problem.model(), x, observations=data)
    for key in problem.latents:
        if not torch.equal(first.sites[key].value, second.sites[key].value):
            raise Mismatch("support", math.inf, f"replay changed {key!r}")
    values = {key: second.sites[key].value.detach() for key in problem.latents}
    assert second.log_joint is not None
    wanted = float(problem.log_joint(values))
    compare_number(
        float(second.log_joint.sum()), wanted, joint_accuracy(wanted), f"replay {name}"
    )


def registry_of(problem: ConsumerProblem) -> LatentRegistry:
    """The latent registry of a problem.

    Parameters
    ----------
    problem : ConsumerProblem
        The problem.

    Returns
    -------
    LatentRegistry
        The registry.
    """
    x, data = problem.data()
    model = problem.model()
    if model.steps_list_sites:
        return LatentRegistry.from_model(model, set(data))
    # A program whose sites a run reaches is read from a run, as the
    # registry's own refusal directs.
    return LatentRegistry.from_trace(model, trace(model, x, observations=data))


def constrained(
    registry: LatentRegistry, position: torch.Tensor
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The draw at a flat position and its log Jacobian.

    The Jacobian is the determinant of the derivative of the free
    coordinates of each event (all but the last of a simplex), computed
    here by automatic differentiation of the registry's map and not read
    from the registry.

    Parameters
    ----------
    registry : LatentRegistry
        The registry.
    position : torch.Tensor
        The flat position.

    Returns
    -------
    tuple of (dict, torch.Tensor)
        The draw and the log determinant.
    """

    def free(flat: torch.Tensor) -> torch.Tensor:
        """The free coordinates of the draw at a position.

        Parameters
        ----------
        flat : torch.Tensor
            The position.

        Returns
        -------
        torch.Tensor
            The coordinates.
        """
        parts = registry.unflatten_unconstrained(flat)
        values = []
        for site in registry.sites.values():
            value, _ = site.constrain(parts[site.name])
            unused = value.numel() - parts[site.name].numel()
            # A simplex has one coordinate more than its position per event.
            values.append((value[..., :-1] if unused else value).reshape(-1))
        return torch.cat(values)

    parts = registry.unflatten_unconstrained(position)
    draw = {
        site.name: site.constrain(parts[site.name])[0].detach()
        for site in registry.sites.values()
    }
    jacobian = torch.autograd.functional.jacobian(free, position)
    assert isinstance(jacobian, torch.Tensor)
    return draw, torch.linalg.slogdet(jacobian).logabsdet


def positions(registry: LatentRegistry) -> list[torch.Tensor]:
    """Flat positions to evaluate a registry at.

    Parameters
    ----------
    registry : LatentRegistry
        The registry.

    Returns
    -------
    list of torch.Tensor
        The origin and two other positions.
    """
    size = registry.total_unconstrained_dim
    steps = torch.arange(1, size + 1, dtype=torch.float64)
    return [torch.zeros(size, dtype=torch.float64), 0.25 * steps / size, -0.5 / steps]


def check_registry(name: str) -> None:
    """Compare a registry's coordinates and Jacobian with the exact ones.

    Parameters
    ----------
    name : str
        The problem.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS[name]
    with double_precision():
        registry = registry_of(problem)
        if set(registry.names) != set(problem.latents):
            raise Mismatch("support", math.inf, f"latents {registry.names}")
        for position in positions(registry):
            draw, log_det = constrained(registry, position)
            parts = registry.unflatten_unconstrained(position)
            reported = sum(
                float(site.constrain(parts[site.name])[1].sum())
                for site in registry.sites.values()
            )
            compare_number(
                reported,
                float(log_det),
                joint_accuracy(float(log_det)),
                f"log Jacobian of {name}",
            )
            for site in registry.sites.values():
                back, _ = site.unconstrain(draw[site.name])
                for found, wanted in zip(
                    back.reshape(-1).tolist(),
                    parts[site.name].reshape(-1).tolist(),
                    strict=True,
                ):
                    compare_number(
                        found,
                        wanted,
                        joint_accuracy(wanted),
                        f"inverse of {site.name}",
                    )


def check_potential(name: str, consumer: str) -> None:
    """Compare a potential's value and gradient with the exact density.

    Parameters
    ----------
    name : str
        The problem.
    consumer : str
        ``"potential"`` for the full density, ``"tempered"`` for the
        density with the likelihood raised to a power.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS[name]
    with double_precision():
        registry = registry_of(problem)
        x, data = problem.data()
        powers = (1.0,) if consumer == "potential" else (0.0, 0.25, 1.0)
        for power in powers:
            potential = (
                PotentialFn(problem.model(), registry, x, data)
                if consumer == "potential"
                else TemperedPotential(problem.model(), registry, x, data, power)
            )
            for position in positions(registry):
                draw, log_det = constrained(registry, position)
                wanted = float(
                    problem.log_prior(draw)
                    + power * problem.log_likelihood(draw)
                    + log_det
                )
                value, gradient = potential.value_and_grad(position)
                compare_number(
                    float(value), wanted, joint_accuracy(wanted), f"{consumer} {name}"
                )
                for k in range(position.numel()):
                    difference = _density_slope(problem, registry, position, k, power)
                    if abs(float(gradient[k]) - difference[0]) > difference[1]:
                        raise Mismatch(
                            "value",
                            abs(float(gradient[k]) - difference[0]),
                            f"{consumer} {name} gradient {k}",
                        )


def _density_slope(
    problem: ConsumerProblem,
    registry: LatentRegistry,
    position: torch.Tensor,
    coordinate: int,
    power: float,
) -> tuple[float, float]:
    """The slope of an exact log density by extrapolated differences.

    Parameters
    ----------
    problem : ConsumerProblem
        The problem.
    registry : LatentRegistry
        The registry.
    position : torch.Tensor
        The flat position.
    coordinate : int
        The coordinate.
    power : float
        The power of the likelihood.

    Returns
    -------
    tuple of (float, float)
        The slope and a bound on its error.
    """
    step = 2.0**-9

    def density(shift: float) -> float:
        """The log density at a shifted position.

        Parameters
        ----------
        shift : float
            The shift of the coordinate.

        Returns
        -------
        float
            The log density.
        """
        moved = position.clone()
        moved[coordinate] += shift
        draw, log_det = constrained(registry, moved)
        return float(
            problem.log_prior(draw) + power * problem.log_likelihood(draw) + log_det
        )

    quotients = []
    for width in (step, step / 2, step / 4):
        quotients.append((density(width) - density(-width)) / (2 * width))
    first = (4 * quotients[1] - quotients[0]) / 3
    second = (4 * quotients[2] - quotients[1]) / 3
    size = 64.0 * 2.0**-52 / (step / 4)
    return second, 4 * abs(second - first) + size


def uniform_bound(draws: int) -> float:
    """A bound on the distance of an empirical distribution function.

    By the inequality of Dvoretzky, Kiefer, and Wolfowitz with Massart's
    constant, the largest distance exceeds the bound with probability at
    most one in a thousand million, for every continuous distribution.

    Parameters
    ----------
    draws : int
        The number of independent draws.

    Returns
    -------
    float
        The bound.
    """
    return math.sqrt(math.log(2e9) / (2 * draws))


def compare_distribution(
    draws: torch.Tensor, distribution: Callable[[float], float], what: str
) -> None:
    """Compare draws with a continuous distribution function.

    Parameters
    ----------
    draws : torch.Tensor
        Independent draws.
    distribution : Callable[[float], float]
        The distribution function.
    what : str
        A description for the report.


    Raises
    ------
    Mismatch
        If the empirical distribution function is too far away.
    """
    ordered = sorted(draws.reshape(-1).tolist())
    count = len(ordered)
    distance = max(
        max(abs(distribution(x) - i / count), abs(distribution(x) - (i + 1) / count))
        for i, x in enumerate(ordered)
    )
    if not distance <= uniform_bound(count):
        raise Mismatch("value", distance, f"{what}: distance {distance:.4f}")


def beta_distribution(a: int, b: int) -> Callable[[float], float]:
    """The distribution function of a Beta variable with whole shapes.

    Parameters
    ----------
    a : int
        The first shape.
    b : int
        The second shape.

    Returns
    -------
    Callable[[float], float]
        The probability of at least ``a`` successes in ``a + b - 1`` trials.
    """

    def distribution(x: float) -> float:
        """The distribution function at a point.

        Parameters
        ----------
        x : float
            The point.

        Returns
        -------
        float
            The probability.
        """
        x = min(max(x, 0.0), 1.0)
        trials = a + b - 1
        return math.fsum(
            math.comb(trials, j) * x**j * (1 - x) ** (trials - j)
            for j in range(a, trials + 1)
        )

    return distribution


def check_prior(name: str, draws: int) -> None:
    """Compare ancestral prior draws with their distribution.

    Parameters
    ----------
    name : str
        The problem.
    draws : int
        The number of draws.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS[name]
    with double_precision():
        x, data = problem.data()
        found = prior_draws(problem.model(), x, data, num_samples=draws, seed=0)
    samples = found.samples
    if set(samples) != set(problem.latents):
        raise Mismatch("support", math.inf, f"prior draws of {sorted(samples)}")
    if name == "beta":
        compare_distribution(samples["theta"], beta_distribution(2, 3), "prior theta")
    elif name == "ungrouped":
        compare_distribution(samples["prior"][..., 0], lambda p: p, "prior weight")
    elif name == "hmm":
        compare_distribution(samples["initial_row"][..., 0], lambda p: p, "initial")
        compare_distribution(
            samples["transition_rows"][..., 0, 0], lambda p: p, "transition"
        )
        compare_distribution(
            samples["emission_rows"][..., 1, 2],
            lambda p: 1 - (1 - p) ** 2,
            "emission",
        )
    else:
        share = float(samples["a"].double().mean())
        if abs(share - 0.25) > uniform_bound(draws):
            raise Mismatch("value", abs(share - 0.25), f"share {share} of ones")


def check_predictive(name: str) -> None:
    """Compare predictive rows with the recorded draws they condition on.

    Parameters
    ----------
    name : str
        The problem.


    Raises
    ------
    Mismatch
        If a recorded draw changes or rows have another frequency.
    """
    problem = PROBLEMS[name]
    count = 60
    with double_precision():
        torch.manual_seed(2)
        x, data = problem.data()
        recorded = {
            key: value.expand(1, count, *value.shape).contiguous()
            for key, value in RECORDED[name].items()
        }
        draws = PriorDraws(samples=recorded, num_samples=count)
        context = {key: value for key, value in data.items() if key != "y"}
        found = Predictive(problem.model(), draws, num_samples=count)(x, context)
    for key, value in RECORDED[name].items():
        if not bool((found[key].reshape(count, -1) == value.reshape(1, -1)).all()):
            raise Mismatch("support", math.inf, f"predictive changed {key!r}")
    if name in ("beta", "ungrouped"):
        rows = found["y"].double().reshape(count, -1)[:, 0]
        if name == "beta":
            wanted = 0.8
        else:
            weight = float(RECORDED[name]["prior"][1])
            wanted = float(rate(0)) * (1 - weight) + float(rate(1)) * weight
        share = float(rows.mean())
        if abs(share - wanted) > uniform_bound(count):
            raise Mismatch("value", abs(share - wanted), f"share {share} of ones")


RECORDED: dict[str, dict[str, torch.Tensor]] = {
    "beta": {"theta": torch.tensor(0.8, dtype=torch.float64)},
    "ungrouped": {"prior": torch.tensor([0.25, 0.75], dtype=torch.float64)},
    "hmm": {
        "initial_row": torch.tensor([0.25, 0.75], dtype=torch.float64),
        "transition_rows": torch.tensor(
            [[0.5, 0.5], [0.125, 0.875]], dtype=torch.float64
        ),
        "emission_rows": torch.tensor(
            [[0.5, 0.25, 0.25], [0.125, 0.375, 0.5]], dtype=torch.float64
        ),
    },
    "parsed": {
        "p_s": torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64),
        "p_v": torch.tensor([0.5, 0.5], dtype=torch.float64),
    },
    "discrete": {"a": torch.tensor(1.0, dtype=torch.float64)},
}
"""One fixed draw of each problem."""


KERNELS: dict[str, Callable[[], MCMCKernel]] = {
    "hmc": lambda: HMCKernel(step_size=0.2, num_steps=4),
    "nuts": lambda: NUTSKernel(step_size=0.2, max_tree_depth=4),
    "metropolis": lambda: MetropolisKernel(RandomWalkProposal(0.5)),
}
"""The continuous transition kernels."""


def check_kernel(name: str, kernel: str) -> None:
    """Compare the density a kernel reports at each draw with the exact one.

    Parameters
    ----------
    name : str
        The problem.
    kernel : str
        The kernel's name.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS[name]
    with double_precision():
        registry = registry_of(problem)
        x, data = problem.data()
        result = MCMC(
            KERNELS[kernel](), num_warmup=4, num_samples=6, num_chains=1, seed=0
        ).run(problem.model(), x, data)
        if set(result.samples) != set(problem.latents):
            raise Mismatch("support", math.inf, f"draws of {sorted(result.samples)}")
        if result.log_densities.dtype != torch.float64:
            raise Mismatch(
                "dtype",
                math.inf,
                f"{kernel} recorded log densities in {result.log_densities.dtype}",
            )
        moved = set()
        for step in range(6):
            draw = {key: value[0, step] for key, value in result.samples.items()}
            moved.add(
                tuple(float(v) for value in draw.values() for v in value.reshape(-1))
            )
            parts = {
                site.name: site.unconstrain(draw[site.name])[0]
                for site in registry.sites.values()
            }
            position = registry.flatten_unconstrained(parts)
            _, log_det = constrained(registry, position)
            wanted = float(problem.log_joint(draw) + log_det)
            recorded = result.log_densities[0, step]
            compare_number(
                float(recorded),
                wanted,
                joint_accuracy(wanted, edge(draw)).with_(
                    dtype=str(recorded.dtype).removeprefix("torch.")
                ),
                f"{kernel} {name} draw {step}",
            )
        if len(moved) < 2:
            raise Mismatch("support", math.inf, f"{kernel} never moved")


def check_kernel_distribution(kernel: str) -> None:
    """Compare a long run's draws with the exact posterior.

    Parameters
    ----------
    kernel : str
        The kernel's name.


    Raises
    ------
    Mismatch
        If the mean differs by more than its error bound.
    """
    problem = PROBLEMS["beta"]
    with double_precision():
        x, data = problem.data()
        result = MCMC(
            KERNELS[kernel](), num_warmup=300, num_samples=1500, num_chains=2, seed=0
        ).run(problem.model(), x, data)
    ones = sum(ROWS)
    a, b = 2 + ones, 3 + len(ROWS) - ones
    draws = result.samples["theta"].reshape(-1)
    mean = a / (a + b)
    deviation = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    # Batch means over thirty batches estimate the error of a correlated
    # average; six of them bound it.
    batches = draws.reshape(30, -1).mean(1)
    error = 6 * float(batches.std()) / math.sqrt(30) + 1e-3 * deviation
    if abs(float(draws.mean()) - mean) > error:
        raise Mismatch(
            "value",
            abs(float(draws.mean()) - mean),
            f"{kernel} mean {float(draws.mean())}",
        )


def check_gibbs() -> None:
    """Compare a Gibbs block's draws of a coin with its exact posterior.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS["discrete"]
    count = 300
    with double_precision():
        x, data = problem.data()
        result = MCMC(
            GibbsKernel(["a"]), num_warmup=1, num_samples=count, num_chains=1, seed=0
        ).run(problem.model(), x, data)
    weights = [
        math.exp(float(problem.log_joint({"a": torch.tensor(bool(value))})))
        for value in (0, 1)
    ]
    wanted = weights[1] / math.fsum(weights)
    share = float(result.samples["a"].double().mean())
    if abs(share - wanted) > uniform_bound(count):
        raise Mismatch("value", abs(share - wanted), f"share {share} for {wanted}")


def check_smc(name: str, particles: int) -> None:
    """Compare a particle run's likelihoods, and for one problem its evidence.

    The one-step estimate of the evidence has relative variance
    ``E[L^2] / E[L]^2 - 1`` over the number of particles, which the Beta
    prior gives in closed form. Tempering in several steps does not
    increase it here, and six of its standard deviations bound the error.

    Parameters
    ----------
    name : str
        The problem.
    particles : int
        The number of particles.


    Raises
    ------
    Mismatch
        If a likelihood or the evidence differs.
    """
    problem = PROBLEMS[name]
    with double_precision():
        x, data = problem.data()
        result = SMC(
            MetropolisKernel(RandomWalkProposal(0.5)),
            num_particles=particles,
            num_mcmc_steps=2,
            num_pilot_steps=5,
            seed=0,
        ).run(problem.model(), x, data)
        for index in range(min(particles, 8)):
            draw = {key: value[index] for key, value in result.samples.items()}
            wanted = float(problem.log_likelihood(draw))
            compare_number(
                float(result.log_likelihoods[index]),
                wanted,
                joint_accuracy(wanted, edge(draw)),
                f"particle {index} of {name}",
            )
    if name == "beta":
        ones = sum(ROWS)
        zeros = len(ROWS) - ones
        first = closed_forms.beta_binomial(ones, zeros, Fraction(2), Fraction(3))
        second = closed_forms.beta_binomial(
            2 * ones, 2 * zeros, Fraction(2), Fraction(3)
        )
        spread = math.sqrt(float(second / first**2 - 1) / particles)
        found = math.exp(result.log_evidence) / float(first)
        if abs(found - 1) > 6 * spread:
            raise Mismatch("value", abs(found - 1), f"evidence ratio {found}")


def check_pathfinder(name: str) -> None:
    """Compare the densities of the retained draws with the exact ones.

    Parameters
    ----------
    name : str
        The problem.
    """
    problem = PROBLEMS[name]
    with double_precision():
        registry = registry_of(problem)
        x, data = problem.data()
        result = Pathfinder(
            num_paths=1, num_draws=8, num_psis_draws=8, num_elbo_draws=4, seed=0
        ).run(problem.model(), x, data)
        for index in range(8):
            draw = {key: value[0, index] for key, value in result.samples.items()}
            parts = {
                site.name: site.unconstrain(draw[site.name])[0]
                for site in registry.sites.values()
            }
            _, log_det = constrained(registry, registry.flatten_unconstrained(parts))
            wanted = float(problem.log_joint(draw) + log_det)
            compare_number(
                float(result.log_density[0, index]),
                wanted,
                joint_accuracy(wanted, edge(draw)),
                f"pathfinder {name} draw {index}",
            )


def _common(problem: ConsumerProblem) -> tuple[MonadicProgram, set[str]]:
    """The model and observed names a guide is built for.

    Parameters
    ----------
    problem : ConsumerProblem
        The problem.

    Returns
    -------
    tuple of (MonadicProgram, set of str)
        The model and the names.
    """
    _, data = problem.data()
    return problem.model(), set(data)


def _normal(problem: ConsumerProblem) -> AutoNormalGuide:
    """A factorized normal guide.

    Parameters
    ----------
    problem : ConsumerProblem
        The problem.

    Returns
    -------
    AutoNormalGuide
        The guide.
    """
    x, data = problem.data()
    model, names = _common(problem)
    return AutoNormalGuide(model, names, init_scale=0.4, x=x, observations=data)


def _amortized(problem: ConsumerProblem) -> AmortizedGuide:
    """A guide whose encoder reads a host datum.

    Parameters
    ----------
    problem : ConsumerProblem
        The problem, with one latent of one free coordinate.

    Returns
    -------
    AmortizedGuide
        The guide.
    """
    x, data = problem.data()
    return AmortizedGuide(
        problem.model(),
        set(data),
        {
            problem.latents[0]: (
                ConditionalNormal(
                    Euclidean(name="Features", dim=3), Euclidean(name="Code", dim=1)
                ),
                "features",
            )
        },
        x=x,
        observations=data,
    )


GUIDES: dict[str, Callable[[ConsumerProblem], Guide]] = {
    "normal": _normal,
    "mvn": lambda problem: AutoMultivariateNormalGuide(
        *_common(problem),
        init_scale=0.4,
        x=problem.data()[0],
        observations=problem.data()[1],
    ),
    "low-rank": lambda problem: AutoLowRankMultivariateNormalGuide(
        *_common(problem),
        rank=1,
        init_scale=0.4,
        x=problem.data()[0],
        observations=problem.data()[1],
    ),
    "iaf": lambda problem: AutoIAFGuide(
        *_common(problem),
        num_flows=2,
        hidden_dim=4,
        num_hidden_layers=1,
        x=problem.data()[0],
        observations=problem.data()[1],
    ),
    "spline": lambda problem: AutoNeuralSplineGuide(
        *_common(problem),
        num_flows=2,
        num_bins=4,
        hidden_dim=4,
        num_hidden_layers=1,
        x=problem.data()[0],
        observations=problem.data()[1],
    ),
    "structured": lambda problem: AutoStructured(
        *_common(problem),
        conditionals="normal",
        init_scale=0.4,
        x=problem.data()[0],
        observations=problem.data()[1],
    ),
    "mixture": lambda problem: AutoMixtureGuide([_normal(problem), _normal(problem)]),
    "delta": lambda problem: AutoDeltaGuide(
        *_common(problem), x=problem.data()[0], observations=problem.data()[1]
    ),
    "laplace": lambda problem: AutoLaplaceApproximation(
        *_common(problem), x=problem.data()[0], observations=problem.data()[1]
    ),
    "dais": lambda problem: AutoDAIS(
        _normal(problem),
        problem.model(),
        problem.data()[1],
        x=problem.data()[0],
        num_steps=2,
    ),
    "list": lambda problem: AutoGuideList({problem.latents[0]: _normal(problem)}),
    "amortized": _amortized,
}
"""The variational guides."""


def guide_applies(guide: str, name: str) -> bool:
    """Whether a guide is built for a problem.

    A flow with couplings needs two free coordinates, and an encoder is
    given the features of the problems that have them.

    Parameters
    ----------
    guide : str
        The guide's name.
    name : str
        The problem.

    Returns
    -------
    bool
        Whether the guide is built.
    """
    if guide in ("iaf", "spline"):
        return name in ("hmm", "parsed")
    if guide == "amortized":
        return name in ("beta", "ungrouped")
    return True


def check_objective(name: str, guide: str) -> None:
    """Compare an SVI step's loss with the exact log joint at its draw.

    The loss of one particle is the guide's log density minus the log
    joint at the drawn values, averaged over the guide's strata. The same
    random state gives the same draw.

    Parameters
    ----------
    name : str
        The problem.
    guide : str
        The guide's name.
    """
    problem = PROBLEMS[name]
    with double_precision():
        torch.manual_seed(3)
        built = GUIDES[guide](problem)
        x, data = problem.data()
        parameters = [p for p in built.parameters() if p.requires_grad]
        svi = SVI(
            problem.model(),
            built,
            torch.optim.SGD(parameters, lr=0.0),
            ELBO(),
        )
        torch.manual_seed(4)
        loss = svi.step(x, data)
        torch.manual_seed(4)
        wanted = 0.0
        smallest = 1.0
        for draw in built.rsample_strata(x):
            values = {key: value.detach() for key, value in draw.sites.items()}
            smallest = max(smallest, edge(values))
            wanted += math.exp(float(draw.log_weight.detach())) * (
                float(draw.log_density.detach().sum())
                - float(problem.log_joint(values))
            )
    compare_number(
        loss, wanted, joint_accuracy(wanted, smallest), f"objective {guide} {name}"
    )


def check_density(guide: str) -> None:
    """Require a guide's density to integrate to one and to describe its draws.

    The integral is taken by the trapezoid rule on the log odds of the
    one latent, at two spacings; the change between them bounds its
    error.

    Parameters
    ----------
    guide : str
        The guide's name.


    Raises
    ------
    Mismatch
        If the mass is not one.
    """
    problem = PROBLEMS["beta"]
    steps = 1200
    reach = 30.0
    with double_precision(), torch.no_grad():
        torch.manual_seed(5)
        built = GUIDES[guide](problem)
        grid = torch.linspace(-reach, reach, steps + 1)
        row = torch.zeros(1, 1)
        log_q = torch.stack(
            [
                built.log_prob(row, {"theta": torch.sigmoid(point).reshape(1)}).sum()
                for point in grid
            ]
        )
        slope = torch.nn.functional.logsigmoid(grid) + torch.nn.functional.logsigmoid(
            -grid
        )
        height = (log_q + slope).exp()
        fine = float(torch.trapezoid(height, grid))
        coarse = float(torch.trapezoid(height[::2], grid[::2]))
        bound = 4 * abs(fine - coarse) + 1e-9
        if not abs(fine - 1) <= bound:
            raise Mismatch("value", abs(fine - 1), f"{guide} has mass {fine}")
        cumulative = torch.cumulative_trapezoid(height, grid)
        torch.manual_seed(6)
        draws = torch.stack([built.rsample(row)["theta"].sum() for _ in range(400)])

    def distribution(x: float) -> float:
        """The distribution function of the guide's density.

        Parameters
        ----------
        x : float
            A probability.

        Returns
        -------
        float
            The mass below it.
        """
        odds = math.log(x) - math.log1p(-x)
        index = int((odds + reach) / (2 * reach) * steps)
        return float(cumulative[min(max(index, 0), steps - 1)])

    compare_distribution(draws, distribution, f"{guide} draws")


def check_rows_alone(guide: str) -> None:
    """Require a joint guide to score each of several rows as it scores it alone.

    Two rows with different values are scored together and one at a
    time. Each joint density must equal the density of its row alone,
    and the two rows' densities must differ, so a guide that scores
    the first row and repeats it fails.

    Parameters
    ----------
    guide : str
        The guide.

    Raises
    ------
    Mismatch
        If a row's joint density differs from its density alone, or the
        two rows receive the same density.
    """
    values = (0.5, 0.9)
    with double_precision(), torch.no_grad():
        torch.manual_seed(0)
        built = GUIDES[guide](PROBLEMS["beta"])
        joint = built.log_prob(torch.zeros(2, 1), {"theta": torch.tensor(values)})
        alone = [
            float(built.log_prob(torch.zeros(1, 1), {"theta": torch.tensor([value])}))
            for value in values
        ]
    for row, reference in enumerate(alone):
        compare_number(
            float(joint[row]),
            reference,
            joint_accuracy(reference),
            f"{guide}: row {row} scored with another row",
        )
    if alone[0] == alone[1]:
        raise Mismatch(
            "value", 0.0, f"{guide}: the two rows have the same density alone"
        )


def check_delta() -> None:
    """Require a point guide to place every draw at its point.

    Raises
    ------
    Mismatch
        If the result differs from its reference.
    """
    problem = PROBLEMS["beta"]
    with double_precision(), torch.no_grad():
        torch.manual_seed(7)
        built = GUIDES["delta"](problem)
        draws = built.rsample(torch.zeros(5, 1))["theta"].reshape(-1)
    if not bool((draws == draws[0]).all()):
        raise Mismatch("value", float(draws.std()), "a point guide moved")


def check_laplace() -> None:
    """Compare a Laplace approximation with its closed form.

    In the log odds the log posterior is ``a log s + b log (1 - s)`` with
    ``s`` the logistic function, whose curvature at any point is
    ``(a + b) s (1 - s)``.
    """
    problem = PROBLEMS["beta"]
    ones = sum(ROWS)
    a, b = 2 + ones, 3 + len(ROWS) - ones
    with double_precision():
        torch.manual_seed(8)
        built = GUIDES["laplace"](problem)
        assert isinstance(built, AutoLaplaceApproximation)
        x, data = problem.data()
        with torch.no_grad():
            built.map_z.fill_(math.log(a / b))
        built.fit_hessian(problem.model(), x, data, jitter=1e-300)
        theta = torch.sigmoid(torch.tensor([math.log(a / b) + 0.3]))
        found = float(built.log_prob(x, {"theta": theta}).detach())
    precision = (a + b) * (a / (a + b)) * (b / (a + b))
    at = float(theta)
    wanted = (
        0.5 * math.log(precision / (2 * math.pi))
        - 0.5 * precision * 0.3**2
        - math.log(at)
        - math.log1p(-at)
    )
    compare_number(found, wanted, joint_accuracy(wanted, 16.0), "Laplace density")


def check_dais() -> None:
    """Require a path estimate of the evidence to average to the evidence.


    Raises
    ------
    Mismatch
        If the average is farther away than six standard errors.
    """
    problem = PROBLEMS["beta"]
    ones = sum(ROWS)
    evidence = float(
        closed_forms.beta_binomial(ones, len(ROWS) - ones, Fraction(2), Fraction(3))
    )
    ratios = []
    with double_precision():
        torch.manual_seed(9)
        built = GUIDES["dais"](problem)
        x, _ = problem.data()
        for _ in range(60):
            for draw in built.rsample_strata(x):
                values = {key: value.detach() for key, value in draw.sites.items()}
                ratios.append(
                    math.exp(
                        float(problem.log_joint(values))
                        - float(draw.log_density.detach().sum())
                    )
                    / evidence
                )
    sample = torch.tensor(ratios, dtype=torch.float64)
    error = 6 * float(sample.std()) / math.sqrt(len(ratios))
    if abs(float(sample.mean()) - 1) > error:
        raise Mismatch(
            "value",
            abs(float(sample.mean()) - 1),
            f"evidence ratio {float(sample.mean())}",
        )


def consistency() -> None:
    """Require the mixture and chain parameters to be distributions.


    Raises
    ------
    Mismatch
        If a row does not sum to one.
    """
    for row in (PRIOR, INITIAL, *TRANSITION, *EMISSION):
        if sum(row, Fraction(0)) != 1:
            raise Mismatch("support", math.inf, f"{row} does not sum to one")
    assert finite_effects.evidence(partial(mixture, groups=(0,) * 4)) > 0


def _register() -> None:
    """Register every check of this module."""
    consistency()
    for name in PROBLEMS:
        add(
            f"consumer/trace/{name}",
            "trace",
            QUESTIONS[name],
            partial(check_trace, name),
        )
        add(
            f"consumer/replay/{name}",
            "replay",
            QUESTIONS[name],
            partial(check_replay, name),
        )
    for name in CONTINUOUS:
        question = QUESTIONS[name]
        add(
            f"consumer/registry/{name}",
            "latent-registry",
            question,
            partial(check_registry, name),
        )
        # A score step is a potential of the prior law, which a particle
        # run refuses to draw from; a checked program's joint states no
        # factor roles, which tempering reads.
        if name not in ("parsed", "recursive"):
            add(
                f"consumer/tempered/{name}",
                "smc",
                question,
                partial(check_potential, name, "tempered"),
            )
            add(f"consumer/smc/{name}", "smc", question, partial(check_smc, name, 16))
        add(
            f"consumer/pathfinder/{name}",
            "pathfinder",
            question,
            partial(check_pathfinder, name),
        )
        add(
            f"consumer/potential/{name}",
            "potential",
            question,
            partial(check_potential, name, "potential"),
        )
        for kernel in KERNELS:
            add(
                f"consumer/kernel/{kernel}/{name}",
                kernel,
                question,
                partial(check_kernel, name, kernel),
            )
        for guide in GUIDES:
            if not guide_applies(guide, name):
                continue
            add(
                f"consumer/objective/{guide}/{name}",
                f"svi-{guide}",
                question,
                partial(check_objective, name, guide),
            )
    for name in ("beta", "ungrouped", "hmm", "discrete"):
        add(
            f"consumer/prior/{name}",
            "prior",
            QUESTIONS[name],
            partial(check_prior, name, 200),
        )
        add(
            f"consumer/prior_long/{name}",
            "prior",
            QUESTIONS[name],
            partial(check_prior, name, 4000),
            slow=True,
        )
    # A checked program scores its observation as a datum it is given, so
    # it has no draw of the observation to predict.
    for name in (name for name in PROBLEMS if name != "recursive"):
        add(
            f"consumer/predictive/{name}",
            "predictive",
            QUESTIONS[name],
            partial(check_predictive, name),
        )
    add("consumer/gibbs/discrete", "gibbs", "R5", check_gibbs)
    for guide in GUIDES:
        if guide not in ("delta", "dais", "laplace") and guide_applies(guide, "beta"):
            add(
                f"consumer/density/{guide}",
                f"svi-{guide}",
                "R1",
                partial(check_density, guide),
            )
    for guide in ("mvn", "low-rank", "structured"):
        add(
            f"consumer/rows-alone/{guide}",
            f"svi-{guide}",
            "R1",
            partial(check_rows_alone, guide),
        )
    add("consumer/delta", "svi-delta", "R1", check_delta)
    add("consumer/laplace", "svi-laplace", "R1", check_laplace)
    add("consumer/dais", "svi-dais", "R1", check_dais)
    for kernel in KERNELS:
        add(
            f"consumer/distribution/{kernel}",
            kernel,
            "R1",
            partial(check_kernel_distribution, kernel),
            slow=True,
        )
    add("consumer/evidence", "smc", "R1", partial(check_smc, "beta", 512), slow=True)


_register()
