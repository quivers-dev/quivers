"""One representative differentiable run of every executor.

A run builds its inputs, executes, and returns the value together with
the gradients of its sum. Runs are repeated to show that a second call
does not depend on the first, and are executed under the counters to
show that counting changes no bit.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from quivers.categorical.traced import KleeneStarTrace
from quivers.continuous.families import ConditionalNormal
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.scan import ScanMorphism
from quivers.continuous.spaces import Euclidean, ProductSpace
from quivers.core.morphisms import observed
from quivers.core.objects import CoproductSet, FinSet
from quivers.core.elimination import plan_contraction, run_contraction
from quivers.core.wiring import EinsumWiring
from quivers.effects import collapse
from quivers.effects.replay import replay
from quivers.inference import (
    ELBO,
    MCMC,
    SMC,
    SVI,
    GibbsKernel,
    MCMCKernel,
    MetropolisKernel,
    Pathfinder,
    Predictive,
    PriorDraws,
    RandomWalkProposal,
    prior_draws,
    trace,
)
from quivers.inference.mcmc import PotentialFn
from quivers.continuous.tensor_backend import TensorBackend
from quivers.qiec.program_runtime import deduction_item, run_deduction, run_program
from quivers.stochastic.agenda import ChartView
from quivers.stochastic.semiring import LOG_PROB
from tests.evidence.checks import closure, consumers, contraction, deduction
from tests.evidence.checks import machine, mixture, span, terms
from tests.evidence.checks.span import double_precision
from tests.evidence import newton_controls, span_controls
from tests.evidence.compare import Mismatch
from tests.evidence.problems import DEDUCTIONS, logarithm
from tests.evidence.sources import program

type Result = list[torch.Tensor]
"""A value and the gradients of its sum."""

type Run = Callable[[], Result]
"""A representative run."""


def with_gradients(value: torch.Tensor, leaves: list[torch.Tensor]) -> Result:
    """A value and its gradients.

    Parameters
    ----------
    value : torch.Tensor
        The value.
    leaves : list of torch.Tensor
        The tensors differentiated by.

    Returns
    -------
    Result
        The value, then the gradients that exist.
    """
    found = torch.autograd.grad(value.sum(), leaves, allow_unused=True)
    return [value.detach(), *(gradient for gradient in found if gradient is not None)]


def _agenda(discipline: str) -> Run:
    """A run of the agenda engine.

    Parameters
    ----------
    discipline : str
        The agenda.

    Returns
    -------
    Run
        The run.
    """

    def run() -> Result:
        """Run the diamond problem.

        Returns
        -------
        Result
            The weight and its gradients.
        """
        problem = DEDUCTIONS["diamond"]
        leaves, parameters = deduction.leaves_of(problem, "logprob", requires_grad=True)
        result = deduction.run(problem, LOG_PROB, discipline, leaves)
        return with_gradients(
            ChartView(result).goal_weight(), list(parameters.values())
        )

    return run


def _number(value: bool | int | float) -> Result:
    """A number as a result.

    Parameters
    ----------
    value : bool or int or float
        The number.

    Returns
    -------
    Result
        The number as a float64 tensor.
    """
    return [torch.tensor(float(value), dtype=torch.float64)]


def _machine_lexical() -> Result:
    """Run a lexical deduction on the reference machine.

    Returns
    -------
    Result
        The weight and its gradients in the rule weights.
    """
    leaves = [
        torch.tensor(logarithm(weight), dtype=torch.float64, requires_grad=True)
        for weight in (machine.JOIN, machine.LIFT)
    ]
    weight = run_deduction(
        machine.module_of("lexical", SEMIRING="LogProb"),
        "G",
        tokens=["a", "b"],
        parameters={"G.rule.join:": leaves[0], "G.rule.lift:": leaves[1]},
    ).weight
    assert isinstance(weight, torch.Tensor)
    return with_gradients(weight, leaves)


def _machine_nonlexical() -> Result:
    """Run a chart deduction on the reference machine.

    Returns
    -------
    Result
        The weight and its gradients in the rule and axiom weights.
    """
    module = machine.module_of("nonlexical", SEMIRING="LogProb")
    rules = {
        name: torch.tensor(logarithm(value), dtype=torch.float64, requires_grad=True)
        for name, value in machine.NONLEXICAL_WEIGHTS.items()
    }
    axiom = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)
    weight = run_deduction(
        module,
        "W",
        axioms=[deduction_item(module, "W", "A")],
        axiom_weights=[axiom],
        parameters={f"W.rule.{name}:": leaf for name, leaf in rules.items()},
        fuel=200_000,
    ).weight
    assert isinstance(weight, torch.Tensor)
    return with_gradients(weight, [*rules.values(), axiom])


def _machine_chained() -> Result:
    """Run a Boolean cycle on the reference machine.

    Returns
    -------
    Result
        The truth value.
    """
    weight = run_deduction(
        machine.module_of("chained", SEMIRING="Boolean", TOLERANCE=""),
        "M",
        tokens=["a"],
    ).weight
    assert isinstance(weight, torch.Tensor) and weight.dtype == torch.bool
    return _number(bool(weight))


def _machine_program() -> Result:
    """Score a program that calls a recursive computation on the machine.

    Returns
    -------
    Result
        The log joint and its gradients in the site and the datum.
    """
    t = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    run = run_program(
        machine.module_of("recursive"),
        "recursive",
        data={"y": y},
        sites={"t": t},
        distributions=TensorBackend(),
    )
    return with_gradients(run.log_joint, [t, y])


def _star() -> Result:
    """Close a dense matrix.

    Returns
    -------
    Result
        The closure and its gradient.
    """
    matrix = closure.encode(closure.rational(closure.MATRICES["dense"]), True)
    matrix.requires_grad_()
    return with_gradients(closure.kleene_star(matrix, LOG_PROB), [matrix])


def _trace() -> Result:
    """Trace a feedback matrix.

    Returns
    -------
    Result
        The trace and its gradient.
    """
    entries = closure.encode(closure.rational(closure.FEEDBACK), False).requires_grad_()
    outer = FinSet(name="A", cardinality=2)
    loop = FinSet(name="U", cardinality=2)
    whole = CoproductSet(components=(outer, loop))
    algebra = closure.TRACES["markov"]
    morphism = observed(whole, whole, entries, algebra=algebra)
    value = KleeneStarTrace(algebra).trace(morphism, loop, outer, outer).tensor
    return with_gradients(value, [entries])


def _wiring() -> Result:
    """Contract a line of matrices.

    Returns
    -------
    Result
        The contraction and its gradients.
    """
    spec, _, _, tensors = contraction.inputs_of("chain", "logprob", zeros=False)
    leaves = [value.requires_grad_() for value in tensors]
    value = EinsumWiring(contraction.RULES["logprob"], spec).apply(*leaves)
    return with_gradients(value, leaves)


def _contraction_plan() -> Result:
    """Plan and run a line of matrices without a wiring.

    Returns
    -------
    Result
        The contraction and its gradients.
    """
    spec, sizes, _, tensors = contraction.inputs_of("chain", "logprob", zeros=False)
    letters, output = spec.split("->")
    algebra = contraction.RULES["logprob"]
    plan = plan_contraction(algebra, letters.split(","), output, sizes)
    leaves = [value.requires_grad_() for value in tensors]
    value, _ = run_contraction(plan, algebra, leaves)
    return with_gradients(value, leaves)


def _parser() -> Result:
    """Parse a sentence with a chart parser.

    Returns
    -------
    Result
        The score and its gradients.
    """
    with double_precision():
        parser = span.chart_parser(LOG_PROB, span.UNARY)
        value = parser(torch.tensor((0, 1, 1)))
        return with_gradients(value, list(parser.parameters()))


def _inside() -> Result:
    """Parse a sentence with the inside algorithm.

    Returns
    -------
    Result
        The score and its gradients.
    """
    with double_precision():
        algorithm, tensors = span.inside_algorithm(span.UNARY, requires_grad=True)
        value = algorithm(torch.tensor((0, 1, 1)))
        return with_gradients(value, list(tensors.values()))


def _mixture(name: str) -> Run:
    """A run of a mixture program.

    Parameters
    ----------
    name : str
        ``"grouped"`` or ``"ungrouped"``.

    Returns
    -------
    Run
        The run.
    """

    def run() -> Result:
        """Score the rows.

        Returns
        -------
        Result
            The log joint and its gradient.
        """
        with double_precision():
            prior = mixture.floats(mixture.PRIOR).requires_grad_()
            return with_gradients(mixture.log_joint(name, prior), [prior])

    return run


def _chain() -> Result:
    """Score a chain of states.

    Returns
    -------
    Result
        The log joint and its gradients.
    """
    model = program("hmm").morphism
    assert isinstance(model, MonadicProgram)
    with double_precision():
        sites = {k: v.requires_grad_() for k, v in mixture.chain_sites().items()}
        value = model.log_joint(torch.tensor([mixture.SYMBOLS]), sites)
        return with_gradients(value, list(sites.values()))


def _scan() -> Result:
    """Score a trajectory of a scan.

    Returns
    -------
    Result
        The log density and its gradient.
    """
    inputs = Euclidean(name="Input", dim=1)
    hidden = Euclidean(name="Hidden", dim=1)
    with double_precision():
        torch.manual_seed(1)
        scan = ScanMorphism(
            ConditionalNormal(ProductSpace(components=(inputs, hidden)), hidden)
        ).double()
        states = torch.tensor([[[0.25], [1.5]]], requires_grad=True)
        value = scan.log_prob(torch.tensor([[[0.5], [-1.0]]]), states)
        return with_gradients(value, [states])


def _collapse() -> Result:
    """Collapse a normal pair.

    Returns
    -------
    Result
        The log joint and its gradient in the observation.
    """
    model = program("conjugate").morphism
    assert isinstance(model, MonadicProgram)
    with double_precision(), collapse({"t": "y"}):
        value = torch.tensor([0.75], requires_grad=True)
        joint = trace(model, torch.zeros(1, 1), observations={"y": value}).log_joint
        assert joint is not None
        return with_gradients(joint, [value])


def _encoder() -> Result:
    """Fold a term.

    Returns
    -------
    Result
        The value and its gradients.
    """
    leaves = terms.parameters(requires_grad=True)
    value = terms.encoder(leaves)(terms.asymmetric(4))
    return with_gradients(value, list(leaves.values()))


def _decoder() -> Result:
    """Score a term.

    Returns
    -------
    Result
        The log probability and its gradient.
    """
    logits = torch.tensor([0.5, 1.0], dtype=torch.float64, requires_grad=True)
    value = terms.decoder(logits).log_prob(
        terms.node(terms.leaf(), terms.leaf()), torch.zeros(1, dtype=torch.float64)
    )
    return with_gradients(value, [logits])


def consume(row: str, name: str) -> Result:
    """Run an inference consumer on a problem.

    Parameters
    ----------
    row : str
        The consumer.
    name : str
        A consumer problem or another source.

    Returns
    -------
    Result
        What the consumer produced, as tensors.
    """
    with double_precision():
        torch.manual_seed(0)
        if name in consumers.PROBLEMS:
            problem = consumers.PROBLEMS[name]
        else:
            problem = consumers.ConsumerProblem(
                name=name,
                latents=("t",),
                log_prior=consumers.PROBLEMS["beta"].log_prior,
                log_likelihood=consumers.PROBLEMS["beta"].log_likelihood,
            )
        model = problem.model()
        x, data = problem.data()
        if row in ("trace", "replay"):
            first = trace(model, x, observations=data)
            if row == "replay":
                with replay(first):
                    first = trace(model, x, observations=data)
            assert first.log_joint is not None
            return [first.log_joint.detach()]
        if row == "latent-registry":
            registry = consumers.LatentRegistry.from_model(model, set(data))
            return [registry.flatten_unconstrained(registry.zero_unconstrained())]
        if row == "potential":
            registry = consumers.LatentRegistry.from_model(model, set(data))
            potential = PotentialFn(model, registry, x, data)
            size = registry.total_unconstrained_dim
            value, gradient = potential.value_and_grad(0.1 * torch.ones(size))
            return [value, gradient]
        if row == "prior":
            return list(prior_draws(model, x, data, num_samples=2).samples.values())
        if row == "predictive":
            recorded = consumers.RECORDED.get(name, {"t": torch.tensor(0.5)})
            draws = PriorDraws(
                samples={
                    key: value.expand(1, 2, *value.shape).contiguous()
                    for key, value in recorded.items()
                },
                num_samples=2,
            )
            context = {key: value for key, value in data.items() if key != "y"}
            return list(Predictive(model, draws, num_samples=2)(x, context).values())
        if row == "smc":
            found = SMC(
                MetropolisKernel(RandomWalkProposal(0.5)),
                num_particles=4,
                num_mcmc_steps=1,
                num_pilot_steps=1,
                seed=0,
            ).run(model, x, data)
            return [found.log_likelihoods]
        if row == "pathfinder":
            path = Pathfinder(
                num_paths=1, num_draws=2, num_psis_draws=2, num_elbo_draws=2, seed=0
            ).run(model, x, data)
            return [path.log_density]
        if row in consumers.KERNELS or row == "gibbs":
            kernel: MCMCKernel = (
                GibbsKernel(list(problem.latents))
                if row == "gibbs"
                else consumers.KERNELS[row]()
            )
            chain = MCMC(kernel, num_warmup=1, num_samples=2, num_chains=1, seed=0)
            return list(chain.run(model, x, data).samples.values())
        guide = consumers.GUIDES[row.removeprefix("svi-")](problem)
        parameters = [p for p in guide.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=0.0)
        SVI(model, guide, optimizer, ELBO()).step(x, data)
        return [p.grad.clone() for p in parameters if p.grad is not None]


CONSUMERS = (
    "trace",
    "replay",
    "latent-registry",
    "potential",
    "hmc",
    "nuts",
    "metropolis",
    "gibbs",
    "smc",
    "pathfinder",
    "prior",
    "predictive",
    *(f"svi-{guide}" for guide in consumers.GUIDES),
)
"""The inference consumers."""

RUNS: dict[str, Run] = {
    **{f"agenda-{name}": _agenda(name) for name in deduction.AGENDAS},
    "machine-lexical": _machine_lexical,
    "machine-nonlexical": _machine_nonlexical,
    "machine-chained": _machine_chained,
    "machine-program": _machine_program,
    "kleene-star": _star,
    "kleene-star-trace": _trace,
    "einsum-wiring": _wiring,
    "contraction-plan": _contraction_plan,
    "span-plan": span_controls.run,
    "newton-plan": newton_controls.run,
    "chart-parser": _parser,
    "inside": _inside,
    "marginalize-grouped": _mixture("grouped"),
    "marginalize-ungrouped": _mixture("ungrouped"),
    "marginalize-chain": _chain,
    "scan": _scan,
    "collapse": _collapse,
    "encoder": _encoder,
    "decoder": _decoder,
}
"""The representative run of every executor."""


def same_bits(left: Result, right: Result) -> bool:
    """Whether two results are equal bit by bit.

    Parameters
    ----------
    left : Result
        The first result.
    right : Result
        The second result.

    Returns
    -------
    bool
        Whether every tensor has the same shape, type, and bytes.
    """
    return len(left) == len(right) and all(
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.contiguous().numpy().tobytes() == b.contiguous().numpy().tobytes()
        for a, b in zip(left, right, strict=True)
    )


def check_repeat(name: str) -> None:
    """Require a second run of an executor to equal the first.

    Parameters
    ----------
    name : str
        The executor.


    Raises
    ------
    Mismatch
        If the two runs differ in a bit.
    """
    if not same_bits(RUNS[name](), RUNS[name]()):
        raise Mismatch("value", float("inf"), f"a second run of {name} differs")
