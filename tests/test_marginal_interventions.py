"""Interventions and subsampled plates on ``marginalize`` blocks.

An intervention on a response removes its conditional factor from the
block and keeps the class prior, the other responses, and everything that
reads the value. A subsampled plate scores whole units, scaled, and its
average over every subset is the full score. Every expected number is
enumerated here from torch densities.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator

import pytest
import torch
import torch.distributions as D

from quivers.continuous import unit_subsets
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.dsl.compiler import _marginal_blocks as blocks
from quivers.effects import clamp, do, scale, subsample
from quivers.effects.joint_plan import JointPlans, prepare_joint
from quivers.inference import ELBO, AutoNormalGuide, LatentRegistry
from quivers.inference._likelihood import (
    collect_log_likelihood,
    collect_unit_log_likelihood,
)
from quivers.inference.mcmc import PotentialFn
from quivers.inference.trace import Trace, trace

_TOLERANCE = 1e-10
_X = torch.zeros(1, 1, dtype=torch.long)

type Values = dict[str, torch.Tensor]


@pytest.fixture(autouse=True)
def _double() -> Iterator[None]:
    """Run each test in double precision with a fixed stream."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def _compiled(source: str) -> MonadicProgram:
    program = loads(source).morphism
    assert isinstance(program, MonadicProgram)
    return program


def _joint(program: MonadicProgram, values: Values) -> float:
    return float(program.log_joint(_X, values).sum())


def _run(program: MonadicProgram, values: Values) -> Trace:
    return trace(program, _X, values)


def _logsumexp(terms: list[torch.Tensor]) -> torch.Tensor:
    return torch.logsumexp(torch.stack(terms), 0)


# A block with two responses, the second of which reads the first, and a
# plain observation outside the block that reads the first too.
PAIR = """object Row : FinSet 4
object G : FinSet 2
object A : FinSet 2
object Val : Real 1

program pair : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        observe y : Row <- Normal(loc=mu[a], scale=1.0) [via=idx]
        observe w : Row <- Normal(loc=mu[a], scale=0.5) [via=jdx]
    let after = 2.0 * y
    observe v : Row <- Normal(loc=after, scale=1.0)
    return mu

export pair
"""

_LATENTS = ("mu", "pa")


def _pair() -> Values:
    return {
        "mu": torch.tensor([-0.6, 0.9]),
        "pa": torch.tensor([0.3, 0.7]),
        "idx": torch.tensor([0, 0, 1, 1]),
        "jdx": torch.tensor([1, 0, 1, 0]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5]),
        "w": torch.tensor([0.1, 0.7, -0.2, 0.9]),
        "v": torch.tensor([0.0, 0.5, -0.5, 1.0]),
    }


def _pair_units(values: Values, responses: tuple[str, ...]) -> torch.Tensor:
    """The units' weights with the factors of some responses only."""
    laws = {"y": ("idx", 1.0), "w": ("jdx", 0.5)}
    units: list[torch.Tensor] = []
    for unit in range(2):
        terms: list[torch.Tensor] = []
        for a in range(2):
            # The class weights are normalized.
            term = values["pa"][a].log() - values["pa"].sum().log()
            for response in responses:
                index, spread = laws[response]
                for row in range(4):
                    if int(values[index][row]) == unit:
                        term = term + D.Normal(values["mu"][a], spread).log_prob(
                            values[response][row]
                        )
            terms.append(term)
        units.append(_logsumexp(terms))
    return torch.stack(units)


def _pair_rest(values: Values, y: torch.Tensor) -> torch.Tensor:
    """The priors and the observation outside the block."""
    return (
        D.Normal(0.0, 1.0).log_prob(values["mu"]).sum()
        + D.HalfNormal(1.0).log_prob(values["pa"]).sum()
        + D.Normal(2.0 * y, 1.0).log_prob(values["v"]).sum()
    )


def _interventional(values: Values, fixed: Values) -> float:
    full = {**values, **fixed}
    kept = tuple(name for name in ("y", "w") if name not in fixed)
    return float(_pair_units(full, kept).sum() + _pair_rest(values, full["y"]))


_FIXES: list[tuple[str, ...]] = [("y",), ("w",), ("y", "w")]


@pytest.mark.parametrize("names", _FIXES)
def test_an_intervention_removes_the_factor_and_nothing_else(
    names: tuple[str, ...],
) -> None:
    program = _compiled(PAIR)
    values = _pair()
    fixed = {name: torch.tensor([2.0, -0.3, 0.8, -1.1]) for name in names}
    expected = _interventional(values, fixed)
    given = {name: value for name, value in values.items() if name not in names}
    with do(fixed):
        recorded = _run(program, given)
        # The value an intervention gives wins over an observed one.
        assert _joint(program, values) == pytest.approx(expected, abs=_TOLERANCE)
    assert float(recorded.log_joint.sum()) == pytest.approx(expected, abs=_TOLERANCE)
    for name in names:
        assert torch.equal(recorded.sites[name].value, fixed[name])
        assert recorded.sites[name].is_deterministic
    # Conditioning on the same values keeps the factor: another number.
    conditioned = _joint(program, {**values, **fixed})
    assert conditioned == pytest.approx(
        float(
            _pair_units({**values, **fixed}, ("y", "w")).sum()
            + _pair_rest(values, {**values, **fixed}["y"])
        ),
        abs=_TOLERANCE,
    )
    assert abs(conditioned - expected) > 0.1


def test_the_plan_reproduces_the_machine_under_an_intervention() -> None:
    program = _compiled(PAIR)
    values = _pair()
    fixed = {"y": torch.tensor([2.0, -0.3, 0.8, -1.1])}
    expected = _interventional(values, fixed)
    plans = JointPlans(program, _X)
    plain = float(plans.log_joint(values).sum())
    with do(fixed):
        plan = prepare_joint(program, _X, values)
        assert not isinstance(plan, str)
        densities, total = plan.evaluate(_X, values)
        recorded = _run(program, values)
        assert torch.equal(total, recorded.log_joint)
        for name, density in densities.items():
            assert torch.equal(density, recorded.sites[name].log_prob)
        assert float(plans.log_joint(values).sum()) == pytest.approx(
            expected, abs=_TOLERANCE
        )
    # The plan of the plain run is another plan, and gives its own number.
    assert plans.prepared == 2
    assert float(plans.log_joint(values).sum()) == plain
    assert abs(plain - expected) > 0.1
    with pytest.raises(ValueError, match="layout"):
        plan.evaluate(_X, values)
    # A handler that does more than fix sites is not the plan's run.
    with do(fixed), scale(2.0):
        assert isinstance(prepare_joint(program, _X, values), str)


def test_a_plan_that_scores_the_fixed_site_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the fixed label, the block keeps the factor and the check fails."""
    monkeypatch.setattr(
        blocks.GroupedObservation,
        "log_prob",
        lambda self, env, powers=True, conditions=None, _exact=(blocks.GroupedObservation.log_prob): (
            _exact(self, env, powers=powers)
        ),
    )
    with pytest.raises(AssertionError):
        test_the_plan_reproduces_the_machine_under_an_intervention()
    with pytest.raises(AssertionError):
        test_an_intervention_removes_the_factor_and_nothing_else(("y",))


def test_many_runs_and_positions_agree_with_each_alone() -> None:
    program = _compiled(PAIR)
    values = _pair()
    fixed = {"y": torch.tensor([2.0, -0.3, 0.8, -1.1])}
    means = torch.randn(3, 2)
    with do(fixed):
        together = program.log_joint(
            torch.zeros(3, 1, dtype=torch.long), {**values, "mu": means}
        )
        expected = torch.tensor(
            [_interventional({**values, "mu": row}, fixed) for row in means]
        )
        assert torch.allclose(together, expected, atol=_TOLERANCE)
    # One intervened value per run is one intervention per run.
    per_run = torch.randn(3, 4)
    with do({"y": per_run}):
        together = program.log_joint(torch.zeros(3, 1, dtype=torch.long), values)
    expected = torch.tensor([_interventional(values, {"y": row}) for row in per_run])
    assert torch.allclose(together, expected, atol=_TOLERANCE)
    # The batched density and gradient are each row's own.
    data = {name: value for name, value in values.items() if name not in _LATENTS}
    registry = LatentRegistry.from_model(program, set(data))
    potential = PotentialFn(program, registry, _X, data)
    positions = torch.randn(4, 4)
    plain = potential.log_density_batch(positions)
    with do(fixed):
        batched, gradients = potential.value_and_grad_batch(positions)
        for row in range(4):
            value, gradient = potential.value_and_grad(positions[row])
            assert float(batched[row]) == pytest.approx(float(value), abs=1e-9)
            assert torch.allclose(gradients[row], gradient, atol=1e-9)
            mean, unconstrained = positions[row][:2], positions[row][2:]
            bijector = registry.sites["pa"].bijector
            weight = bijector(unconstrained)
            exact = _interventional(
                {**values, "mu": mean, "pa": weight}, fixed
            ) + float(bijector.log_abs_det_jacobian(unconstrained, weight).sum())
            assert float(batched[row]) == pytest.approx(exact, abs=1e-9)
    assert float((plain - batched).abs().min()) > 1e-3


def test_released_responses_follow_the_interventional_law() -> None:
    """``w`` given ``do(y)`` ignores ``y``; given ``y`` observed it does not."""
    program = _compiled(PAIR)
    values = _pair()
    given = {
        name: value for name, value in values.items() if name not in ("y", "w", "v")
    }
    high = torch.full((4,), 4.0)
    with do({"y": high}):
        intervened = _run(program, given)
    observed = _run(program, {**given, "y": high})
    with clamp({"y": high}):
        clamped = _run(program, given)
    draws = 4000

    def mean(recorded: Trace) -> torch.Tensor:
        sampleable = recorded.sites["w"].sampleable
        assert sampleable is not None
        law = sampleable.raw_morphism
        return law.rsample(sampleable.input, torch.Size((draws,))).mean(0)

    prior = values["pa"]
    expected = float((prior * values["mu"]).sum())
    spread = math.sqrt(float((prior * (values["mu"] ** 2 + 0.25)).sum()) - expected**2)
    bound = 5.0 * spread / math.sqrt(draws)
    assert float((mean(intervened) - expected).abs().max()) < bound
    scores = D.Normal(values["mu"], 1.0).log_prob(high[0])
    posterior = (prior.log() + 2 * scores).softmax(-1)
    conditional = float((posterior * values["mu"]).sum())
    for recorded in (observed, clamped):
        assert float((mean(recorded) - conditional).abs().max()) < bound
    assert abs(conditional - expected) > 3 * bound
    # What is downstream reads the value either way.
    for recorded in (intervened, observed):
        assert torch.equal(recorded.sites["after"].value.reshape(-1), 2.0 * high)


def test_unit_scores_under_an_intervention_are_of_the_other_responses() -> None:
    program = _compiled(PAIR)
    values = _pair()
    fixed = {"y": torch.tensor([2.0, -0.3, 0.8, -1.1])}
    with do(fixed):
        recorded = _run(program, values)
    (units,) = recorded.sites["_marg_a"].units
    assert units.responses == ("w",)
    weights = _pair_units(values, ("w",))
    assert torch.allclose(units.log_weights(), weights, atol=_TOLERANCE)
    scored = collect_unit_log_likelihood([recorded], {"w"}, (1, 1))["w"]
    assert torch.allclose(scored.log_likelihood[0, 0], weights, atol=_TOLERANCE)
    with pytest.raises(TypeError, match="'y'"):
        collect_log_likelihood([recorded], {"y"}, (1, 1))


NEST = """object Row : FinSet 4
object G : FinSet 2
object A : FinSet 2
object B : FinSet 3
object Val : Real 1

program nest : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample nu : B <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    sample pb : B <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        marginalize b : B <- Categorical(probs=pb) [over=G]
            let loc_y = mu[a] + nu[b]
            observe y : Row <- Normal(loc=loc_y, scale=1.0) [via=idx, weights=power]
        observe w : Row <- Normal(loc=mu[a], scale=0.5) [via=jdx]
    return mu

export nest
"""


def _nest() -> Values:
    return {
        "mu": torch.tensor([-0.6, 0.9]),
        "nu": torch.tensor([0.5, -0.2, 1.1]),
        "pa": torch.tensor([0.3, 0.7]),
        "pb": torch.tensor([0.2, 0.5, 0.3]),
        "idx": torch.tensor([0, 0, 1, 1]),
        "jdx": torch.tensor([1, 0, 1, 0]),
        "power": torch.tensor([1.0, 0.5, 0.0, 2.0]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5]),
        "w": torch.tensor([0.1, 0.7, -0.2, 0.9]),
    }


def _nest_score(values: Values, responses: tuple[str, ...]) -> float:
    total = 0.0
    for unit in range(2):
        terms: list[torch.Tensor] = []
        for a in range(2):
            inner: list[torch.Tensor] = []
            for b in range(3):
                score = values["pb"][b].log()
                if "y" in responses:
                    for row in range(4):
                        if int(values["idx"][row]) == unit:
                            loc = values["mu"][a] + values["nu"][b]
                            score = score + values["power"][row] * D.Normal(
                                loc, 1.0
                            ).log_prob(values["y"][row])
                inner.append(score)
            term = values["pa"][a].log() + _logsumexp(inner)
            if "w" in responses:
                for row in range(4):
                    if int(values["jdx"][row]) == unit:
                        term = term + D.Normal(values["mu"][a], 0.5).log_prob(
                            values["w"][row]
                        )
            terms.append(term)
        total += float(_logsumexp(terms))
    return total


@pytest.mark.parametrize("names", _FIXES)
def test_nested_blocks_lose_the_factor_of_a_fixed_response(
    names: tuple[str, ...],
) -> None:
    program = _compiled(NEST)
    values = _nest()
    fixed = {name: torch.tensor([2.0, -0.3, 0.8, -1.1]) for name in names}
    kept = tuple(name for name in ("y", "w") if name not in names)
    with do(fixed):
        recorded = _run(program, values)
        plan = prepare_joint(program, _X, values)
        assert not isinstance(plan, str)
        assert torch.equal(plan.evaluate(_X, values)[1], recorded.log_joint)
    score = float(recorded.sites["_marg_a"].log_prob.sum())
    assert score == pytest.approx(_nest_score(values, kept), abs=_TOLERANCE)
    conditioned = _run(program, {**values, **fixed}).sites["_marg_a"]
    assert abs(float(conditioned.log_prob.sum()) - score) > 0.1


SHARED = """object Row : FinSet 4
object A : FinSet 3
object Val : Real 1

program shared : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa)
        observe y : Row <- Normal(loc=mu[a], scale=1.0)
    let after = 2.0 * y
    observe v : Row <- Normal(loc=after, scale=1.0)
    return mu

export shared
"""


def test_a_block_with_one_latent_loses_its_whole_factor() -> None:
    program = _compiled(SHARED)
    values = {
        "mu": torch.tensor([-0.6, 0.9, 0.1]),
        "pa": torch.tensor([0.3, 0.5, 0.2]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5]),
        "v": torch.tensor([0.0, 0.5, -0.5, 1.0]),
    }
    fixed = torch.tensor([2.0, -0.3, 0.8, -1.1])
    rest = (
        D.Normal(0.0, 1.0).log_prob(values["mu"]).sum()
        + D.HalfNormal(1.0).log_prob(values["pa"]).sum()
        + D.Normal(2.0 * fixed, 1.0).log_prob(values["v"]).sum()
    )
    with do({"y": fixed}):
        assert _joint(program, values) == pytest.approx(float(rest), abs=_TOLERANCE)
        plan = prepare_joint(program, _X, values)
        assert not isinstance(plan, str)
        assert float(plan.evaluate(_X, values)[1]) == pytest.approx(
            float(rest), abs=_TOLERANCE
        )
    block = torch.logsumexp(
        values["pa"].log()
        + D.Normal(values["mu"], 1.0).log_prob(fixed[:, None]).sum(0),
        -1,
    )
    conditioned = _joint(program, {**values, "y": fixed})
    assert conditioned == pytest.approx(float(rest + block), abs=_TOLERANCE)
    assert abs(float(block)) > 0.1


# --- subsampled plates ----------------------------------------------------


def _subsets(extent: int, size: int) -> list[torch.Tensor]:
    return [torch.tensor(rows) for rows in itertools.combinations(range(extent), size)]


def _average(
    score: Callable[[], torch.Tensor], plate: str, extent: int, size: int
) -> tuple[torch.Tensor, float]:
    """The mean of a score over every subset, and how much it varies."""
    found: list[torch.Tensor] = []
    for index in _subsets(extent, size):
        with subsample(plate, index):
            found.append(score())
    stacked = torch.stack(found)
    return stacked.mean(0), float((stacked.amax(0) - stacked.amin(0)).max())


BY_ROW = """object Row : FinSet 5
object A : FinSet 2
object Val : Real 1

program by_row : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=Row]
        observe y : Row <- Normal(loc=mu[a], scale=1.0) [via=idx]
    return mu

export by_row
"""

# The units are another object than the rows, which the map reads.
BY_SLOT = BY_ROW.replace(
    "object A : FinSet 2", "object A : FinSet 2\nobject Slot : FinSet 7"
).replace("[over=Row]", "[over=Slot]")

# Two rows in each unit.
BY_GROUP = (
    BY_ROW.replace("object Row : FinSet 5", "object Row : FinSet 6")
    .replace("object A : FinSet 2", "object A : FinSet 2\nobject G : FinSet 3")
    .replace("[over=Row]", "[over=G]")
)

INDEXED = """object Row : FinSet 5
object A : FinSet 2
object Val : Real 1

program indexed : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample pa : Row <- Beta(concentration1=1.0, concentration0=1.0)
    marginalize z : Row <- Bernoulli(probs=pa)
        let loc_y = mu[0] + real(z) * 1.5
        observe y : Row <- Normal(loc=loc_y, scale=1.0) [weights=power]
    return mu

export indexed
"""


def _flat() -> Values:
    return {
        "mu": torch.tensor([-0.6, 0.9]),
        "pa": torch.tensor([0.3, 0.7]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5, -0.4]),
        "idx": torch.arange(5),
    }


def _prior(values: Values) -> torch.Tensor:
    return (
        D.Normal(0.0, 1.0).log_prob(values["mu"]).sum()
        + D.HalfNormal(1.0).log_prob(values["pa"]).sum()
    )


def _by_unit(values: Values) -> torch.Tensor:
    """The weights of the units that hold rows, in unit order."""
    weights: list[torch.Tensor] = []
    for unit in sorted(set(values["idx"].tolist())):
        rows = [row for row, at in enumerate(values["idx"].tolist()) if at == unit]
        weights.append(
            torch.logsumexp(
                values["pa"].log()
                + sum(
                    D.Normal(values["mu"], 1.0).log_prob(values["y"][row])
                    for row in rows
                ),
                -1,
            )
        )
    return torch.stack(weights)


_CASES = [
    (BY_ROW, "Row", 5, torch.arange(5)),
    (BY_SLOT, "Row", 5, torch.tensor([6, 0, 3, 2, 5])),
    (BY_SLOT, "Slot", 7, torch.tensor([6, 0, 3, 2, 5])),
    (BY_GROUP, "G", 3, torch.tensor([0, 1, 2, 2, 0, 1])),
]


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize(("source", "plate", "extent", "index"), _CASES)
def test_the_average_over_every_subset_is_the_score_and_its_gradient(
    source: str, plate: str, extent: int, index: torch.Tensor, size: int
) -> None:
    program = _compiled(source)
    values = {**_flat(), "idx": index}
    if index.numel() == 6:
        values["y"] = torch.tensor([0.4, -1.0, 0.2, 1.5, -0.4, 0.7])
    assert _joint(program, values) == pytest.approx(
        float(_prior(values) + _by_unit(values).sum()), abs=_TOLERANCE
    )

    def score() -> torch.Tensor:
        latents = {name: values[name].clone().requires_grad_() for name in ("mu", "pa")}
        joint = program.log_joint(_X, {**values, **latents}).sum()
        gradients = torch.autograd.grad(joint, tuple(latents.values()))
        return torch.cat([joint.detach().reshape(1), *gradients])

    full = score()
    average, spread = _average(score, plate, extent, size)
    assert torch.allclose(average, full, atol=1e-9)
    assert spread > 0.1


def test_a_factor_of_one_and_a_scaled_prior_fail_the_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = unit_subsets.unit_coefficients
    wrong = [
        # No factor.
        lambda *given: (exact(*given) > 0).double(),
        # Twice the factor.
        lambda *given: 2.0 * exact(*given),
        # Every unit kept.
        lambda *given: torch.ones_like(exact(*given)),
    ]
    for variant in wrong:
        monkeypatch.setattr(blocks, "unit_coefficients", variant)
        with pytest.raises(AssertionError):
            test_the_average_over_every_subset_is_the_score_and_its_gradient(
                BY_GROUP, "G", 3, torch.tensor([0, 1, 2, 2, 0, 1]), 1
            )


def test_one_subset_is_the_scaled_weights_of_its_units() -> None:
    program = _compiled(BY_SLOT)
    values = {**_flat(), "idx": torch.tensor([6, 0, 3, 2, 5])}
    rows = torch.tensor([1, 4])
    scores = torch.logsumexp(
        values["pa"].log() + D.Normal(values["mu"], 1.0).log_prob(values["y"][:, None]),
        -1,
    )
    expected = float(_prior(values) + 2.5 * scores[rows].sum())
    with subsample("Row", rows):
        assert _joint(program, values) == pytest.approx(expected, abs=_TOLERANCE)
        # The response may be given at the rows kept alone.
        given = {**values, "y": values["y"][rows]}
        recorded = _run(program, given)
    assert float(recorded.log_joint.sum()) == pytest.approx(expected, abs=_TOLERANCE)
    (units,) = recorded.sites["_marg_a"].units
    assert float(units.log_weights().sum()) == pytest.approx(
        float(recorded.sites["_marg_a"].log_prob.sum()), abs=_TOLERANCE
    )
    with pytest.raises(TypeError, match="estimate the score"):
        collect_log_likelihood([recorded], {"y"}, (1, 1))


@pytest.mark.parametrize("size", [1, 3])
def test_an_indexed_latent_and_likelihood_powers_average_to_the_score(
    size: int,
) -> None:
    program = _compiled(INDEXED)
    values = {
        "mu": torch.tensor([-0.6, 0.9]),
        "pa": torch.tensor([0.2, 0.4, 0.6, 0.8, 0.5]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5, -0.4]),
        "power": torch.tensor([1.0, 0.5, 0.0, 2.0, 1.0]),
    }
    prior = torch.stack([1 - values["pa"], values["pa"]], -1).log()
    scores = D.Normal(values["mu"][0] + torch.tensor([0.0, 1.5]), 1.0).log_prob(
        values["y"][:, None]
    )
    weights = torch.logsumexp(prior + values["power"][:, None] * scores, -1)

    def score() -> torch.Tensor:
        site = _run(program, values).sites["_marg_z"]
        return site.log_prob.sum().detach()

    assert float(score()) == pytest.approx(float(weights.sum()), abs=_TOLERANCE)
    average, spread = _average(score, "Row", 5, size)
    assert float(average) == pytest.approx(float(weights.sum()), abs=_TOLERANCE)
    assert spread > 0.1
    rows = torch.tensor([0, 2, 3])
    with subsample("Row", rows):
        given = {**values, "y": values["y"][rows]}
        estimate = _run(program, given).sites["_marg_z"].log_prob.sum()
    assert float(estimate) == pytest.approx(
        float(weights[rows].sum() * 5 / 3), abs=_TOLERANCE
    )


def test_whole_units_of_rows_are_scaled_by_the_units() -> None:
    program = _compiled(BY_GROUP)
    values = {
        **_flat(),
        "idx": torch.tensor([0, 1, 2, 2, 0, 1]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5, -0.4, 0.7]),
    }
    full = _by_unit(values)
    estimates: list[float] = []
    for kept in itertools.combinations(range(3), 2):
        rows = [row for row in range(6) if int(values["idx"][row]) in kept]
        with subsample("Row", torch.tensor(rows)):
            site = _run(program, values).sites["_marg_a"]
        estimates.append(float(site.log_prob.sum()))
        assert estimates[-1] == pytest.approx(
            float(full[list(kept)].sum() * 1.5), abs=_TOLERANCE
        )
    assert sum(estimates) / 3 == pytest.approx(float(full.sum()), abs=_TOLERANCE)


def test_a_subset_that_splits_a_unit_is_refused() -> None:
    values = {
        **_flat(),
        "idx": torch.tensor([0, 1, 2, 2, 0, 1]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5, -0.4, 0.7]),
    }
    grouped = _compiled(BY_GROUP)
    with (
        pytest.raises(ValueError, match=r"splits a unit: the unit G=0 has rows"),
        subsample("Row", torch.tensor([0, 1, 5])),
    ):
        _joint(grouped, values)
    # A kept unit whose rows the subset of the rows leaves out.
    with (
        pytest.raises(ValueError, match=r"kept by the subset of \['G'\]"),
        subsample("G", torch.tensor([0])),
        subsample("Row", torch.tensor([0, 1, 5])),
    ):
        _joint(grouped, values)
    # The rows of the units kept select the data read, not a second factor.
    with subsample("G", torch.tensor([0])):
        one = _joint(grouped, values)
        with subsample("Row", torch.tensor([0, 2, 4])):
            assert _joint(grouped, values) == pytest.approx(one, abs=_TOLERANCE)
    shared = {
        "mu": torch.tensor([-0.6, 0.9, 0.1]),
        "pa": torch.tensor([0.3, 0.5, 0.2]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5]),
        "v": torch.tensor([0.0, 0.5, -0.5, 1.0]),
    }
    with (
        pytest.raises(ValueError, match="the one unit of the block over 'a'"),
        subsample("Row", torch.tensor([0, 2])),
    ):
        _joint(_compiled(SHARED), shared)
    # The whole plate is not a split.
    with subsample("Row", torch.arange(4)):
        assert _joint(_compiled(SHARED), shared) == pytest.approx(
            _joint(_compiled(SHARED), shared), abs=_TOLERANCE
        )


TWO_PLATES = """object Row : FinSet 3
object Col : FinSet 3
object G : FinSet 3
object A : FinSet 2
object Val : Real 1

program two : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        observe y : Row <- Normal(loc=mu[a], scale=1.0) [via=idx]
        observe w : Col <- Normal(loc=mu[a], scale=0.5) [via=jdx]
    return mu

export two
"""


def test_two_row_plates_that_select_the_same_units_are_refused() -> None:
    program = _compiled(TWO_PLATES)
    values = {
        **_flat(),
        "idx": torch.arange(3),
        "jdx": torch.arange(3),
        "y": torch.tensor([0.4, -1.0, 0.2]),
        "w": torch.tensor([0.1, 0.7, -0.2]),
    }
    index = torch.tensor([0, 2])
    with (
        pytest.raises(ValueError, match="each subset would select the block's units"),
        subsample("Row", index),
        subsample("Col", index),
    ):
        _joint(program, values)
    # One of them selects the units, with all the rows they hold.
    unit = torch.stack(
        [
            torch.logsumexp(
                values["pa"].log()
                + D.Normal(values["mu"], 1.0).log_prob(values["y"][row])
                + D.Normal(values["mu"], 0.5).log_prob(values["w"][row]),
                -1,
            )
            for row in range(3)
        ]
    )
    with subsample("Row", index):
        site = _run(program, values).sites["_marg_a"]
    assert float(site.log_prob.sum()) == pytest.approx(
        float(unit[index].sum() * 1.5), abs=_TOLERANCE
    )


NESTED_PLATES = """object Row : FinSet 4
object G : FinSet 2
object H : FinSet 2
object A : FinSet 2
object B : FinSet 2
object Val : Real 1

program nested_plates : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample nu : B <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    sample pb : B <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        marginalize b : B <- Categorical(probs=pb) [over=[G, H]]
            observe y : Row <- Normal(loc=nu[b], scale=1.0) [via=[g, h]]
        observe w : Row <- Normal(loc=mu[a], scale=0.5) [via=g]
    return mu

export nested_plates
"""


def test_an_inner_plate_is_inside_the_units_of_the_outer_block() -> None:
    program = _compiled(NESTED_PLATES)
    values = {
        "mu": torch.tensor([-0.6, 0.9]),
        "nu": torch.tensor([0.5, -0.2]),
        "pa": torch.tensor([0.3, 0.7]),
        "pb": torch.tensor([0.4, 0.6]),
        "g": torch.tensor([0, 0, 1, 1]),
        "h": torch.tensor([0, 1, 0, 1]),
        "y": torch.tensor([0.4, -1.0, 0.2, 1.5]),
        "w": torch.tensor([0.1, 0.7, -0.2, 0.9]),
    }
    with (
        pytest.raises(ValueError, match="splits a unit: the unit G=0"),
        subsample("H", torch.tensor([0])),
    ):
        _joint(program, values)
    full = float(_run(program, values).sites["_marg_a"].log_prob.sum())

    def score() -> torch.Tensor:
        return _run(program, values).sites["_marg_a"].log_prob.sum().detach()

    average, spread = _average(score, "G", 2, 1)
    assert float(average) == pytest.approx(full, abs=_TOLERANCE)
    assert spread > 0.01


def test_the_elbo_of_a_subsampled_block_is_unbiased() -> None:
    program = _compiled(BY_ROW)
    data = {"y": _flat()["y"], "idx": _flat()["idx"]}
    guide = AutoNormalGuide(program, set(data))
    objective = ELBO(num_particles=2)

    def loss() -> torch.Tensor:
        guide.zero_grad()
        torch.manual_seed(7)
        value = objective(program, guide, _X, data)
        value.backward()
        gradients = [
            parameter.grad.reshape(-1)
            for parameter in guide.parameters()
            if parameter.grad is not None
        ]
        return torch.cat([value.detach().reshape(1), *gradients])

    full = loss()
    average, spread = _average(loss, "Row", 5, 2)
    assert torch.allclose(average, full, atol=1e-9)
    assert spread > 0.1
