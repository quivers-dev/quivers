"""Tempering respects typed prior potentials and likelihood dependencies."""

import math

import pytest
import torch

from quivers.continuous import MonadicProgram
from quivers.continuous.program_steps import Call, Let, Scan, Score, reading
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference import SMC, TemperedPotential
from quivers.inference.mcmc import MetropolisKernel, RandomWalkProposal
from quivers.inference.mcmc.kernel import PotentialFn
from quivers.inference.prior import _ancestral_program
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace

SOURCE = """object Out:Real 1
program p:Out->Out
    sample mu <- Normal(loc=0.0,scale=1.0)
    observe y <- Normal(loc=mu,scale=1.0)
    return mu
export p
"""
X = torch.zeros(1, 1)
Y = torch.tensor(0.7, dtype=torch.float64)


def _model(*scores):
    original = loads(SOURCE).morphism
    return MonadicProgram(
        original.domain, original.codomain, (*original.steps, *scores), ("mu",)
    )


def _registry(model):
    return LatentRegistry.from_trace(model, trace(model, X, {"y": Y}))


@pytest.mark.parametrize("beta", [0.0, 0.25, 1.0])
def test_typed_factor_roles_match_independent_full_joint_and_every_derivative(beta):
    model = _model(
        Score(
            name="likelihood_sounding_prior",
            role="potential",
            score=reading(lambda env: -0.3 * env["mu"].square(), {"mu"}),
        ),
        Score(
            name="prior_sounding_likelihood",
            role="likelihood",
            score=reading(lambda env: -0.2 * (env["mu"] - 2).square(), {"mu"}),
        ),
    )
    registry = _registry(model)
    potential = TemperedPotential(model, registry, X, {"y": Y}, beta)
    z = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    prior, likelihood = potential.log_prior_and_likelihood(z)
    expected_prior = (
        torch.distributions.Normal(0.0, 1.0).log_prob(z).sum() - 0.3 * z.square().sum()
    )
    expected_likelihood = (
        torch.distributions.Normal(z, 1.0).log_prob(Y).sum()
        - 0.2 * (z - 2).square().sum()
    )
    torch.testing.assert_close(prior, expected_prior)
    torch.testing.assert_close(likelihood, expected_likelihood)
    actual = potential.log_density(z)
    expected = expected_prior + beta * expected_likelihood
    torch.testing.assert_close(actual, expected)
    first = torch.autograd.grad(actual, z, create_graph=True)[0]
    oracle_first = torch.autograd.grad(expected, z, create_graph=True)[0]
    torch.testing.assert_close(first, oracle_first)
    torch.testing.assert_close(
        torch.autograd.grad(first.sum(), z)[0],
        torch.autograd.grad(oracle_first.sum(), z)[0],
    )
    if beta == 1:
        torch.testing.assert_close(
            actual, PotentialFn(model, registry, X, {"y": Y}).log_density(z)
        )
    # Role partitioning shares one prepared execution, rather than projecting
    # and tracing a second program at every potential evaluation.
    assert potential._plans.prepared == 1


def test_nested_prior_potential_is_retained_in_temperature_zero():
    real = Euclidean(name="nested", dim=1)
    child = MonadicProgram(
        real,
        real,
        (
            Score(
                name="nested_factor",
                role="potential",
                score=reading(lambda env: -env["value"].square(), {"value"}),
            ),
        ),
        ("nested_factor",),
        params=("value",),
    )
    model = _model(Call(name="child", program=child, args=("mu",)))
    z = torch.tensor([0.4], dtype=torch.float64)
    prior, likelihood = TemperedPotential(
        model, _registry(model), X, {"y": Y}
    ).log_prior_and_likelihood(z)
    torch.testing.assert_close(
        prior, torch.distributions.Normal(0.0, 1.0).log_prob(z).sum() - z.square().sum()
    )
    torch.testing.assert_close(
        likelihood, torch.distributions.Normal(z, 1.0).log_prob(Y).sum()
    )


@pytest.mark.parametrize("nested", [False, True])
def test_smc_refuses_unknown_prior_normalizer_before_executing_any_score(nested):
    def invalid(env):
        raise AssertionError("prior normalization must be checked before execution")

    factor = Score(name="factor", role="potential", score=reading(invalid, ()))
    if nested:
        real = Euclidean(name="nested", dim=1)
        child = MonadicProgram(real, real, (factor,), ("factor",), params=("value",))
        model = _model(Call(name="child", program=child, args=("mu",)))
    else:
        model = _model(factor)
    sampler = SMC(MetropolisKernel(RandomWalkProposal(1.0)), num_particles=4)
    with pytest.raises(ValueError, match="cannot normalize explicit score potentials"):
        sampler.run(model, X, {"y": Y})


def test_initial_particles_use_projected_dependencies_without_likelihood_execution():
    def invalid(env):
        raise AssertionError(
            "an initial prior particle must not evaluate likelihood code"
        )

    model = _model(
        Score(name="unreachable", role="likelihood", score=reading(invalid, ()))
    )
    prior = _ancestral_program(model)
    registry = LatentRegistry.from_model(model, {"y"})
    sampler = SMC(MetropolisKernel(RandomWalkProposal(1.0)), num_particles=7)
    before = torch.random.get_rng_state()
    positions, gibbs = sampler._prior_particles(
        prior, X, {"y": Y}, registry, frozenset(), torch.Generator().manual_seed(3)
    )
    assert positions.shape == (7, 1) and gibbs == [{}] * 7
    assert torch.isfinite(positions).all()
    assert torch.equal(before, torch.random.get_rng_state())


def test_clamped_draw_still_belongs_to_likelihood_not_prior():
    model = loads(SOURCE.replace("observe y", "sample y")).morphism
    registry = _registry(model)
    z = torch.tensor([0.4], dtype=torch.float64)
    prior, likelihood = TemperedPotential(
        model, registry, X, {"y": Y}
    ).log_prior_and_likelihood(z)
    torch.testing.assert_close(
        prior, torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
    )
    torch.testing.assert_close(
        likelihood, torch.distributions.Normal(z, 1.0).log_prob(Y).sum()
    )


def test_likelihood_role_evidence_agrees_with_normal_normal_conjugacy():
    # A classified likelihood contributes the same exact normalized density as
    # an Observe; no inferred name such as 'lik' supplies its statistical role.
    original = loads(SOURCE).morphism
    likelihood = Score(
        name="unusual_name",
        role="likelihood",
        score=reading(
            lambda env: (
                torch.distributions.Normal(env["mu"], 1.0).log_prob(env["y"]).sum()
            ),
            {"mu", "y"},
        ),
    )
    model = MonadicProgram(
        original.domain, original.codomain, (original.steps[0], likelihood), ("mu",)
    )
    estimates = []
    for seed in range(4):
        estimates.append(
            SMC(
                MetropolisKernel(RandomWalkProposal(0.8)),
                num_particles=160,
                num_mcmc_steps=1,
                num_pilot_steps=0,
                seed=50 + seed,
            )
            .run(model, X, {"y": Y})
            .log_evidence
        )
    truth = torch.distributions.Normal(0.0, math.sqrt(2)).log_prob(Y)
    values = torch.tensor(estimates, dtype=torch.float64)
    assert (
        abs(values.mean() - truth) < 5 * values.std() / math.sqrt(len(values)) + 0.005
    )


def test_likelihood_binding_read_by_conditional_prior_retains_its_value():
    original = loads("""object Out:Real 1
program p(x:Real):Out->Out
    let shift=x+1.0
    sample mu <- Normal(loc=shift,scale=1.0)
    return mu
export p
""").morphism
    score = Score(
        name="shift", role="likelihood", score=reading(lambda env: env["x"] + 1, {"x"})
    )
    model = MonadicProgram(
        original.domain, original.codomain, (score, *original.steps[1:]), ("mu",)
    )
    prior = _ancestral_program(model)
    assert isinstance(prior.steps[0], Let)
    actual = trace(prior, X, {"x": torch.tensor(2.0), "mu": torch.tensor(0.0)})
    expected = torch.distributions.Normal(3.0, 1.0).log_prob(torch.tensor(0.0))
    torch.testing.assert_close(actual.log_joint.sum(), expected)
    sampler = SMC(MetropolisKernel(RandomWalkProposal(1.0)), num_particles=7)
    registry = LatentRegistry.from_trace(
        model, trace(model, X, {"x": torch.tensor(2.0)})
    )
    positions, _ = sampler._prior_particles(
        prior,
        X,
        {"x": torch.tensor(2.0)},
        registry,
        frozenset(),
        torch.Generator().manual_seed(9),
    )
    oracle = torch.randn(7, 1, generator=torch.Generator().manual_seed(9)) + 3
    torch.testing.assert_close(positions, oracle)


def test_tempered_potential_refuses_subsampled_scores():
    from quivers.effects import subsample

    model = _model()
    potential = TemperedPotential(model, _registry(model), X, {"y": Y})
    with subsample("UnusedPlate", torch.tensor([0])):
        with pytest.raises(ValueError, match="exact"):
            potential.log_prior_and_likelihood(torch.tensor([0.4]))
        with pytest.raises(ValueError, match="exact"):
            SMC(MetropolisKernel(RandomWalkProposal(1.0))).run(model, X, {"y": Y})


def test_role_partition_does_not_cancel_small_likelihood_against_huge_prior():
    model = _model(
        Score(
            name="constant_potential",
            role="potential",
            score=reading(lambda env: torch.tensor(-1e200, dtype=torch.float64), ()),
        )
    )
    z = torch.tensor([0.4], dtype=torch.float64)
    prior, likelihood = TemperedPotential(
        model, _registry(model), X, {"y": Y}
    ).log_prior_and_likelihood(z)
    assert prior < -1e199
    expected = torch.distributions.Normal(z, 1.0).log_prob(Y).sum()
    torch.testing.assert_close(likelihood, expected)


def test_scanned_prior_potential_uses_typed_role_at_every_occurrence():
    real = Euclidean(name="cell", dim=1)
    cell = MonadicProgram(
        real,
        real,
        (
            Score(
                name="cell_factor",
                role="potential",
                score=reading(lambda env: -0.2 * env["entry"].square(), {"entry"}),
            ),
        ),
        ("entry",),
        params=("entry", "state"),
    )
    model = _model(
        Let(name="sequence", value=reading(lambda env: env["seq"], {"seq"})),
        Scan(name="trajectory", step=cell, source="sequence", initial=torch.zeros(1)),
    )
    data = {"y": Y, "seq": torch.tensor([[[1.0], [2.0]]], dtype=torch.float64)}
    registry = LatentRegistry.from_trace(model, trace(model, X, data))
    z = torch.tensor([0.4], dtype=torch.float64)
    prior, likelihood = TemperedPotential(
        model, registry, X, data
    ).log_prior_and_likelihood(z)
    torch.testing.assert_close(
        prior, torch.distributions.Normal(0.0, 1.0).log_prob(z).sum() - 1
    )
    torch.testing.assert_close(
        likelihood, torch.distributions.Normal(z, 1.0).log_prob(Y).sum()
    )
    with pytest.raises(ValueError, match="cannot normalize explicit score potentials"):
        SMC(MetropolisKernel(RandomWalkProposal(1.0))).run(model, X, data)
