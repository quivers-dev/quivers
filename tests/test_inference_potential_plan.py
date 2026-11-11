"""The potential's prepared joint scores exactly as the reference machine.

`PotentialFn` scores its program through a `JointPlan` prepared once per
layout of the values (`quivers.effects.joint_plan`). The oracle is the
reference machine itself: a handler that installs nothing changes no
run, but its presence on the stack declines every plan, so the same
computation under it is the unprepared evaluation. Every comparison is
exact (``torch.equal``), never a tolerance.

Cover:

1. `value_and_grad` and the per-site densities of the plan equal the
   machine's at random positions, across the benchmark models and
   programs built by hand to reach every instruction the plan compiles
   (plates, gathers, lets, score steps, destructuring draws, calls,
   product inputs, learned parameters, marginalized blocks).
2. Fixed-seed chains are identical draw for draw, with identical
   evaluation counts, for HMC, NUTS, composite kernels whose Gibbs
   values the potential is conditioned on every transition, SMC on the
   tempered potential, and DAIS training.
3. The work per evaluation is bounded by counting: no reference-machine
   run and no preparation per call once a layout is prepared.
4. Invalidation: new observations, new conditioning (values and
   layouts), a program with a scan, whose sites depend on its input,
   handlers on the stack, an overridden ``log_joint``, and a plan asked
   to score a layout it was not prepared for.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
import torch
from torch.distributions import Normal

from quivers.continuous import Draw, Euclidean, MonadicProgram, make_fixed_normal
from quivers.continuous.inline import make_fixed_bernoulli
from quivers.continuous.program_steps import Call, Let, Score, reading
from quivers.core import FinSet, Unit
from quivers.dsl import loads
from quivers.effects import interpreter
from quivers.effects import joint_plan
from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.effects.joint_plan import JointPlan, JointPlans, prepare_joint
from quivers.inference import ELBO, SVI, AutoDAIS, AutoNormalGuide
from quivers.inference.mcmc import (
    MCMC,
    CycleKernel,
    GibbsKernel,
    HMCKernel,
    MetropolisKernel,
    MixtureKernel,
    NUTSKernel,
    PotentialFn,
    RandomWalkProposal,
)
from quivers.inference.registry import LatentRegistry
from quivers.inference.smc import SMC, TemperedPotential
from quivers.inference.trace import trace
from tests.benchmarks import datasets

_X = torch.zeros(1, 1)


class _Inert(EffectHandler):
    """A handler that installs nothing: a run under it is the run without
    it, but its presence declines every prepared joint."""

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        del run
        return ()


@contextmanager
def _counting(module: object, name: str) -> Iterator[list[int]]:
    """Count the calls of a module-level callable while the block runs."""
    original = getattr(module, name)
    calls = [0]

    def counted(*args: object, **kwargs: object) -> object:
        calls[0] += 1
        return original(*args, **kwargs)

    setattr(module, name, counted)
    try:
        yield calls
    finally:
        setattr(module, name, original)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

#: The input each model runs on; every other model takes ``_X``.
_INPUTS = {"product_input": torch.tensor([[0.5, -0.3]])}

_HIERARCHICAL = """
object Group : FinSet 4
object Obs : FinSet 12
program hier : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=5.0)
    sample tau <- HalfNormal(scale=2.0)
    sample sigma <- HalfNormal(scale=2.0)
    sample alpha : Group <- Normal(loc=mu, scale=tau)
    let theta = alpha[group]
    observe y : Obs <- Normal(loc=theta, scale=sigma)
    return y
export hier
"""

_FACTOR = """
object Step : FinSet 2
object Val : Real 1
program p : Val -> Val
    sample a <- Normal(loc=0.0, scale=1.0)
    let k = factor t : Step in a * (real(t) + 1.0)
    score s = sum(k)
    return a
export p
"""

_MIXTURE = """
object Resp : FinSet 5
object K : FinSet 2
object Val : Real 1
program p : Resp -> Val
    sample w <- Dirichlet(concentration=1.0) [over=K]
    sample m <- Normal(loc=0.0, scale=2.0) [over=K]
    marginalize c : Resp <- Categorical(probs=w) [reduction=logsumexp]
        let loc = m[c]
        observe y : Resp <- Normal(loc=loc, scale=1.0)
    return w
export p
"""

_LEARNED = """
object X : Real 1
morphism f : X -> X ~ Normal
program p : X -> X
    sample a <- Normal(loc=0.0, scale=1.0)
    sample b <- f(a)
    return b
export p
"""

_SCAN = """
object Driver : Real 2
object State : Real 3
morphism transition : Driver * State -> State ~ Normal
program step(u_t, s_prev) : Driver * State -> State
    sample s <- transition(u_t, s_prev)
    return s
define run_all = scan(step)
program ssm : Driver -> State
    sample h <- run_all
    return h
export ssm
"""

_REAL = Euclidean(name="R", dim=1)


def _switch_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = 3.0 * env["z"].reshape(()).to(env["mu"].dtype)
    return Normal(mean, 1.0).log_prob(env["mu"].reshape(())).reshape(1) + Normal(
        env["mu"].reshape(()), 1.0
    ).log_prob(torch.tensor(1.2)).reshape(1)


def _switch_program() -> MonadicProgram:
    """``z ~ Bernoulli(0.3)``, ``mu ~ N(0, 1)``, scored by a likelihood."""
    return MonadicProgram(
        Unit,
        Euclidean(name="out", dim=1),
        steps=[
            Draw(
                names=("z",),
                morphism=make_fixed_bernoulli(0.3, FinSet(name="B", cardinality=2)),
            ),
            Draw(names=("mu",), morphism=make_fixed_normal(0.0, 1.0, _REAL)),
            Score(name="lik", score=reading(_switch_likelihood, ("z", "mu"))),
        ],
        return_vars=(),
    )


def _shift(env: dict[str, torch.Tensor]) -> torch.Tensor:
    return env["a"] * 2.0 + 1.0


def _quadratic(env: dict[str, torch.Tensor]) -> torch.Tensor:
    return -(env["r"] ** 2).sum().reshape(1)


def _callee() -> MonadicProgram:
    """``b ~ N(0, 1)``; ``d = 2 a + 1``; returns ``d``, reading ``a``."""
    return MonadicProgram(
        _REAL,
        _REAL,
        steps=[
            Draw(names=("b",), morphism=make_fixed_normal(0.0, 1.0, _REAL)),
            Let(name="d", value=reading(_shift, ("a",))),
        ],
        return_vars=("d",),
    )


def _calling_program() -> MonadicProgram:
    """A draw, a call of a program in place, and a score on its result."""
    return MonadicProgram(
        Unit,
        _REAL,
        steps=[
            Draw(names=("a",), morphism=make_fixed_normal(0.0, 1.0, _REAL)),
            Call(name="r", program=_callee(), args=("a",)),
            Draw(names=("c",), morphism=make_fixed_normal(0.0, 2.0, _REAL), args=("r",)),
            Score(name="pen", score=reading(_quadratic, ("r",))),
        ],
        return_vars=("c",),
    )


_PRODUCT = """
object A : Real 1
object B : Real 1
program p(a_in, b_in) : A * B -> A
    sample m <- Normal(loc=a_in, scale=1.0)
    sample n <- Normal(loc=b_in, scale=2.0)
    return m
export p
"""


def _models() -> dict[str, tuple[MonadicProgram, dict[str, torch.Tensor]]]:
    """Every model the oracle runs, with its observations."""
    models: dict[str, tuple[MonadicProgram, dict[str, torch.Tensor]]] = {}
    for name in (
        "beta_bernoulli",
        "normal_normal",
        "normal_inverse_gamma",
        "gamma_exponential",
        "bayes_linear_regression",
        "eight_schools_centered",
        "eight_schools_noncentered",
        "correlated_regression",
        "neal_funnel",
        "ill_conditioned_mvn",
        "half_normal_scale",
        "truncated_normal_recovery",
    ):
        data = getattr(datasets, name)()
        assert isinstance(data.model, MonadicProgram)
        models[name] = (data.model, dict(data.observations))
    group = torch.arange(12) % 4
    models["hierarchical"] = (
        loads(_HIERARCHICAL).morphism,
        {"y": torch.linspace(-1.0, 2.0, 12), "group": group},
    )
    models["factor"] = (loads(_FACTOR).morphism, {})
    models["mixture"] = (
        loads(_MIXTURE).morphism,
        {"y": torch.tensor([-1.5, 0.2, 2.1, 1.8, -0.4])},
    )
    torch.manual_seed(0)
    models["learned"] = (loads(_LEARNED).morphism, {})
    models["call"] = (_calling_program(), {})
    models["product_input"] = (loads(_PRODUCT).morphism, {})
    return models


_MODELS = _models()


def _setup(
    model: MonadicProgram,
    observations: dict[str, torch.Tensor],
    x: torch.Tensor = _X,
) -> PotentialFn:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recorded = trace(model, x, observations=observations)
    registry = LatentRegistry.from_trace(model, recorded)
    return PotentialFn(model, registry, x, observations)


def _values(potential: PotentialFn, z: torch.Tensor) -> dict[str, torch.Tensor]:
    trace_side, _ = potential._trace_side(z)
    return {**trace_side, **potential._observations}


# ---------------------------------------------------------------------------
# 1. The plan is the machine, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_value_and_grad_equal_the_machine(name: str) -> None:
    model, observations = _MODELS[name]
    potential = _setup(model, observations, _INPUTS.get(name, _X))
    generator = torch.Generator().manual_seed(11)
    dim = potential.registry.total_unconstrained_dim
    for _ in range(4):
        z = torch.randn(dim, generator=generator)
        plan = potential._plans.plan(_values(potential, z))
        assert isinstance(plan, JointPlan), plan
        value, grad = potential.value_and_grad(z)
        with _Inert():
            expected_value, expected_grad = potential.value_and_grad(z)
        assert torch.equal(value, expected_value)
        assert torch.equal(grad, expected_grad)
        assert torch.isfinite(value)


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_densities_equal_the_trace(name: str) -> None:
    model, observations = _MODELS[name]
    x = _INPUTS.get(name, _X)
    potential = _setup(model, observations, x)
    z = torch.randn(
        potential.registry.total_unconstrained_dim,
        generator=torch.Generator().manual_seed(3),
    )
    values = _values(potential, z)
    plan = prepare_joint(model, x, values)
    assert isinstance(plan, JointPlan), plan
    densities, joint = plan.evaluate(x, values)
    recorded = trace(model, x, observations=values)
    assert list(densities) == list(recorded.sites)
    for label, site in recorded.sites.items():
        assert torch.equal(densities[label], site.log_prob), label
    assert recorded.log_joint is not None
    assert torch.equal(joint, recorded.log_joint)
    assert torch.equal(joint, model.log_joint(x, values))


def test_the_oracle_tells_positions_apart() -> None:
    """The comparison is not vacuous: the machine at another position, or
    the plan with an instruction dropped, differs from the plan."""
    model, observations = _MODELS["hierarchical"]
    potential = _setup(model, observations)
    dim = potential.registry.total_unconstrained_dim
    z = torch.randn(dim, generator=torch.Generator().manual_seed(1))
    value, _ = potential.value_and_grad(z)
    with _Inert():
        other, _ = potential.value_and_grad(z + 1e-3)
    assert not torch.equal(value, other)
    values = _values(potential, z)
    plan = prepare_joint(model, _X, values)
    assert isinstance(plan, JointPlan)
    instructions = plan._instructions
    sample = next(
        index
        for index, item in enumerate(instructions)
        if isinstance(item, joint_plan._Sample) and item.label == "y"
    )
    plan._instructions = instructions[:sample] + instructions[sample + 1 :]
    _, wrong = plan.evaluate(_X, values)
    assert not torch.equal(wrong, model.log_joint(_X, values))


def test_positions_outside_the_support_and_program_errors_agree() -> None:
    def scale_likelihood(env: dict[str, torch.Tensor]) -> torch.Tensor:
        return Normal(0.0, env["b"].reshape(()) + 1.0).log_prob(env["y"]).sum().reshape(1)

    def broken(env: dict[str, torch.Tensor]) -> torch.Tensor:
        if float(env["b"].reshape(())) > 50.0:
            raise RuntimeError("broken score")
        return env["b"].reshape(1)

    for score, reads in ((scale_likelihood, ("b", "y")), (broken, ("b",))):
        model = MonadicProgram(
            Unit,
            _REAL,
            steps=[
                Draw(names=("b",), morphism=make_fixed_normal(0.0, 1.0, _REAL)),
                Score(name="lik", score=reading(score, reads)),
            ],
            return_vars=(),
        )
        observations = {"y": torch.linspace(-1.0, 1.0, 6)} if "y" in reads else {}
        potential = _setup(model, observations)
        if score is scale_likelihood:
            outside = torch.tensor([-2.0])
            value, grad = potential.value_and_grad(outside)
            with _Inert():
                expected, expected_grad = potential.value_and_grad(outside)
            assert value == float("-inf") and expected == float("-inf")
            assert torch.equal(grad, expected_grad)
        else:
            with pytest.raises(RuntimeError, match="broken score"):
                potential.value_and_grad(torch.tensor([60.0]))
            with _Inert(), pytest.raises(RuntimeError, match="broken score"):
                potential.value_and_grad(torch.tensor([60.0]))


# ---------------------------------------------------------------------------
# 2. Fixed-seed chains are identical
# ---------------------------------------------------------------------------


def _same_result(run: Callable[[], object]) -> None:
    """Run twice, prepared and under the machine, and compare everything."""
    prepared = run()
    with _Inert():
        machine = run()
    samples_a = getattr(prepared, "samples")
    samples_b = getattr(machine, "samples")
    assert set(samples_a) == set(samples_b)
    for name in samples_a:
        assert torch.equal(samples_a[name], samples_b[name]), name
    for field in ("log_density", "num_evaluations", "num_gradient_evaluations"):
        if hasattr(prepared, field):
            assert torch.equal(
                torch.as_tensor(getattr(prepared, field)),
                torch.as_tensor(getattr(machine, field)),
            ), field


@pytest.mark.parametrize(
    "kernel",
    [
        lambda: HMCKernel(step_size=0.1, num_steps=5),
        lambda: NUTSKernel(max_tree_depth=5),
    ],
    ids=["hmc", "nuts"],
)
def test_gradient_chains_are_identical(kernel: Callable[[], object]) -> None:
    model, observations = _MODELS["hierarchical"]
    _same_result(
        lambda: MCMC(
            kernel(), num_warmup=15, num_samples=15, num_chains=2, seed=7
        ).run(model, _X, observations)
    )


@pytest.mark.parametrize("composite", ["cycle", "mixture"])
def test_composite_chains_with_gibbs_conditioning_are_identical(
    composite: str,
) -> None:
    def kernel() -> CycleKernel | MixtureKernel:
        components = {
            "gibbs": GibbsKernel(["z"]),
            "walk": MetropolisKernel(RandomWalkProposal(0.7)),
            "nuts": NUTSKernel(max_tree_depth=3),
        }
        if composite == "cycle":
            return CycleKernel(components)
        return MixtureKernel(components, probs=[0.4, 0.3, 0.3])

    _same_result(
        lambda: MCMC(kernel(), num_warmup=15, num_samples=25, num_chains=2, seed=3).run(
            _switch_program(), _X, {}
        )
    )


def test_smc_on_the_tempered_potential_is_identical() -> None:
    model, observations = _MODELS["hierarchical"]

    def run() -> object:
        return SMC(
            MetropolisKernel(RandomWalkProposal(0.5)),
            num_particles=40,
            num_mcmc_steps=2,
            num_pilot_steps=0,
            target_ess=0.6,
            seed=5,
        ).run(model, _X, observations)

    prepared = run()
    with _Inert():
        machine = run()
    for name in getattr(prepared, "samples"):
        assert torch.equal(
            getattr(prepared, "samples")[name], getattr(machine, "samples")[name]
        )
    assert getattr(prepared, "log_evidence") == getattr(machine, "log_evidence")
    assert torch.equal(
        getattr(prepared, "temperatures"), getattr(machine, "temperatures")
    )


def test_tempered_densities_are_identical() -> None:
    model, observations = _MODELS["hierarchical"]
    potential = _setup(model, observations)
    tempered = TemperedPotential(
        model, potential.registry, _X, observations, temperature=0.37
    )
    generator = torch.Generator().manual_seed(2)
    for _ in range(3):
        z = torch.randn(potential.registry.total_unconstrained_dim, generator=generator)
        parts = tempered.log_prior_and_likelihood(z)
        value, grad = tempered.value_and_grad(z)
        with _Inert():
            expected_parts = tempered.log_prior_and_likelihood(z)
            expected_value, expected_grad = tempered.value_and_grad(z)
        assert torch.equal(parts[0], expected_parts[0])
        assert torch.equal(parts[1], expected_parts[1])
        assert torch.equal(value, expected_value)
        assert torch.equal(grad, expected_grad)


def test_dais_training_is_identical() -> None:
    model, observations = _MODELS["hierarchical"]

    def train() -> list[torch.Tensor]:
        torch.manual_seed(4)
        base = AutoNormalGuide(model, observed_names={"y"})
        guide = AutoDAIS(base, model, observations=observations, num_steps=3)
        svi = SVI(model, guide, torch.optim.Adam(guide.parameters(), lr=1e-2), ELBO())
        for _ in range(4):
            svi.step(_X, observations)
        return [parameter.detach().clone() for parameter in guide.parameters()]

    prepared = train()
    with _Inert():
        machine = train()
    assert len(prepared) == len(machine)
    for a, b in zip(prepared, machine, strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 3. The work per evaluation, counted
# ---------------------------------------------------------------------------


def test_a_prepared_layout_runs_no_machine_and_prepares_once() -> None:
    model, observations = _MODELS["hierarchical"]
    potential = _setup(model, observations)
    dim = potential.registry.total_unconstrained_dim
    generator = torch.Generator().manual_seed(0)
    with (
        _counting(interpreter, "Evaluator") as machines,
        _counting(joint_plan, "prepare_joint") as preparations,
    ):
        for _ in range(50):
            potential.value_and_grad(torch.randn(dim, generator=generator))
            potential.log_density(torch.randn(dim, generator=generator))
        assert machines[0] == 0
        assert preparations[0] == 1
        # The counter sees the machine: the same calls under a handler
        # run it once per evaluation.
        with _Inert():
            for _ in range(5):
                potential.value_and_grad(torch.randn(dim, generator=generator))
        assert machines[0] == 5
        assert preparations[0] == 1
    assert potential.num_evaluations == 105
    assert potential.num_gradient_evaluations == 55


def test_conditioned_copies_share_the_prepared_layout() -> None:
    model = _switch_program()
    recorded = trace(model, _X, observations={})
    registry = LatentRegistry.from_trace(model, recorded, exclude={"z"})
    potential = PotentialFn(model, registry, _X, {})
    with (
        _counting(interpreter, "Evaluator") as machines,
        _counting(joint_plan, "prepare_joint") as preparations,
    ):
        for step in range(20):
            gibbs = {"z": torch.tensor([float(step % 2)])}
            potential.condition(gibbs).value_and_grad(torch.tensor([0.1 * step]))
    assert machines[0] == 0
    assert preparations[0] == 1


# ---------------------------------------------------------------------------
# 4. Invalidation
# ---------------------------------------------------------------------------


def test_new_observations_score_their_own_data() -> None:
    model, observations = _MODELS["hierarchical"]
    shifted = {**observations, "y": observations["y"] + 0.5}
    first = _setup(model, observations)
    second = _setup(model, shifted)
    z = torch.randn(
        first.registry.total_unconstrained_dim,
        generator=torch.Generator().manual_seed(6),
    )
    a, grad_a = first.value_and_grad(z)
    b, grad_b = second.value_and_grad(z)
    with _Inert():
        expected_a, expected_grad_a = first.value_and_grad(z)
        expected_b, expected_grad_b = second.value_and_grad(z)
    assert torch.equal(a, expected_a) and torch.equal(grad_a, expected_grad_a)
    assert torch.equal(b, expected_b) and torch.equal(grad_b, expected_grad_b)
    assert not torch.equal(a, b)


def test_new_conditioning_scores_its_own_values_and_layouts() -> None:
    """Conditioning on new values reuses the layout's plan and scores the
    new values; conditioning on values of a new layout (a plate of
    another length) prepares a plan of its own."""
    source = """
object Obs : FinSet 6
program p : Obs -> Obs
    sample mu <- Normal(loc=0.0, scale=1.0)
    observe y : Obs <- Normal(loc=mu, scale=1.0)
    return mu
export p
"""
    model = loads(source).morphism
    recorded = trace(model, _X, observations={"y": torch.zeros(6)})
    registry = LatentRegistry.from_trace(model, recorded, exclude={"y"})
    potential = PotentialFn(model, registry, _X, {})
    z = torch.tensor([0.3])
    results = []
    for y in (torch.zeros(6), torch.linspace(-1.0, 1.0, 6), torch.ones(4)):
        conditioned = potential.condition({"y": y})
        value, grad = conditioned.value_and_grad(z)
        with _Inert():
            expected, expected_grad = conditioned.value_and_grad(z)
        assert torch.equal(value, expected) and torch.equal(grad, expected_grad)
        results.append(value)
    assert len({float(value) for value in results}) == 3
    assert potential._plans.prepared == 2


def test_a_scan_is_scored_on_the_machine() -> None:
    """A scan's sites depend on its input's length and its recursion on the
    sequence, so the potential scores it the general way, and correctly at
    every length."""
    model = loads(_SCAN).morphism
    for length in (3, 5):
        u = torch.randn(1, length, 2, generator=torch.Generator().manual_seed(length))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            recorded = trace(model, u)
        registry = LatentRegistry.from_trace(model, recorded)
        assert registry.names == tuple(f"s@{t}" for t in range(length))
        potential = PotentialFn(model, registry, u, {})
        z = torch.randn(
            registry.total_unconstrained_dim, generator=torch.Generator().manual_seed(9)
        )
        trace_side, log_det = potential._trace_side(z)
        reason = potential._plans.plan(trace_side)
        assert isinstance(reason, str) and "scan" in reason
        with _counting(interpreter, "Evaluator") as machines:
            value = potential.log_density(z)
        assert machines[0] == 1
        assert torch.equal(value, model.log_joint(u, trace_side).sum() + log_det)


def test_handlers_and_overrides_decline_the_plan() -> None:
    model, observations = _MODELS["hierarchical"]
    plans = JointPlans(model, _X)
    potential = _setup(model, observations)
    values = _values(
        potential, torch.zeros(potential.registry.total_unconstrained_dim)
    )
    assert isinstance(plans.plan(values), JointPlan)
    with _Inert():
        assert plans.plan(values) == "handlers are on the stack"
    calls: list[int] = []
    original = model.log_joint

    def spy(x: torch.Tensor, intermediates: dict[str, torch.Tensor]) -> torch.Tensor:
        calls.append(1)
        return original(x, intermediates)

    model.log_joint = spy  # type: ignore[method-assign]
    try:
        assert plans.plan(values) == "the program's log_joint is overridden"
        plans.log_joint(values)
    finally:
        del model.log_joint
    assert calls == [1]
    assert isinstance(plans.plan(values), JointPlan)


def test_a_plan_refuses_a_layout_it_was_not_prepared_for() -> None:
    model, observations = _MODELS["hierarchical"]
    potential = _setup(model, observations)
    values = _values(
        potential, torch.zeros(potential.registry.total_unconstrained_dim)
    )
    plan = prepare_joint(model, _X, values)
    assert isinstance(plan, JointPlan)
    for changed in (
        {**values, "y": values["y"][:6]},
        {**values, "y": values["y"].to(torch.float64)},
        {key: value for key, value in values.items() if key != "group"},
    ):
        with pytest.raises(ValueError, match="layout"):
            plan.evaluate(_X, changed)
    with _Inert(), pytest.raises(ValueError, match="handlers are on the stack"):
        plan.evaluate(_X, values)


def test_a_copy_prepares_its_own_plans() -> None:
    model, observations = _MODELS["hierarchical"]
    potential = _setup(model, observations)
    z = torch.randn(
        potential.registry.total_unconstrained_dim,
        generator=torch.Generator().manual_seed(8),
    )
    value, grad = potential.value_and_grad(z)
    duplicate = copy.deepcopy(potential)
    assert duplicate._plans.prepared == 0
    copied_value, copied_grad = duplicate.value_and_grad(z)
    assert torch.equal(value, copied_value) and torch.equal(grad, copied_grad)
    assert duplicate._plans.prepared == 1
    assert math.isfinite(float(value))
