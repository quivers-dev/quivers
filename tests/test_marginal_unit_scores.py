"""Unit scores and response laws of ``marginalize`` blocks.

Every expected number is computed here from torch densities, by
enumerating the classes of a small model, or as a class-weighted mean, and
never through the code that scores the programs.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import pytest
import torch
import torch.distributions as D

from quivers.continuous import marginal as marginal_module
from quivers.continuous.measure import Independent, Mixture
from quivers.continuous.program_steps import Observe, Score
from quivers.continuous.programs import MonadicProgram
from quivers.dsl import loads
from quivers.dsl.compiler import _marginal_blocks as blocks
from quivers.effects import (
    SiteUnits,
    UnfactoredObservationError,
    observation_scoring,
    scale,
)
from quivers.effects import joint_plan
from quivers.effects.joint_plan import prepare_joint
from quivers.inference import (
    ELBO,
    SVI,
    AutoNormalGuide,
    LatentRegistry,
    MCMCResult,
    Predictive,
    posterior_expectations,
)
from quivers.inference import registry as registry_module
from quivers.inference._likelihood import (
    collect_log_likelihood,
    collect_unit_log_likelihood,
)
from quivers.inference.smc import TemperedPotential
from quivers.inference.trace import Trace, trace
from quivers.modeling import Model, ModelFit

_TOLERANCE = 1e-10

#: Five standard errors of a frequency over the draws a test takes.
_DRAWS = 4000
_BOUND = 5.0 * math.sqrt(0.25 / _DRAWS)


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


def _run(program: MonadicProgram, values: dict[str, torch.Tensor]) -> Trace:
    return trace(program, torch.zeros(1, 1, dtype=torch.long), values)


def _units(recorded: Trace, score: str) -> SiteUnits:
    (units,) = recorded.sites[score].units
    return units


def _joint(program: MonadicProgram, values: dict[str, torch.Tensor]) -> float:
    return float(program.log_joint(torch.zeros(1, 1, dtype=torch.long), values).sum())


def _grouped(rows: torch.Tensor, scores: torch.Tensor, units: int) -> torch.Tensor:
    """The sum of the rows' scores in each unit, one row at a time."""
    total = torch.zeros(units, *scores.shape[1:])
    for row, unit in enumerate(rows.tolist()):
        total[unit] = total[unit] + scores[row]
    return total


AROUND = """object Trial : FinSet 6
object Res : FinSet 3
object Val : Real 1

program around : Trial -> Val
    sample res_logits <- Normal(loc=0.0, scale=1.0) [over=Res]
    sample noise <- Beta(concentration1=1.0, concentration0=20.0)
    let res_probs = softmax(res_logits)
    marginalize r : Res <- Categorical(probs=res_probs) [reduction=logsumexp]
        let inside = real(dist <= halfwidth[r])
        let p = noise + (1.0 - 2.0 * noise) * inside
        observe accept : Trial <- Bernoulli(probs=p)
    return noise

export around
"""

AROUND_BY_ROW = (
    AROUND.replace("[reduction=logsumexp]", "[over=Trial, reduction=logsumexp]")
    .replace("Bernoulli(probs=p)", "Bernoulli(probs=p) [via=tidx]")
    .replace("program around :", "program around(tidx : Tensor[Int]([Trial])) :")
)


def _around() -> dict[str, torch.Tensor]:
    return {
        "dist": torch.tensor([0.5, 1.5, 2.5, 0.2, 1.2, 3.5]),
        "halfwidth": torch.tensor([1.0, 2.0, 3.0]),
        "accept": torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        "res_logits": torch.tensor([0.2, -0.1, 0.4]),
        "noise": torch.tensor(0.1),
    }


def _around_by_class() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The log prior of a reading, the rows' scores and success probabilities."""
    values = _around()
    prior = values["res_logits"].log_softmax(-1)
    inside = (values["dist"][:, None] <= values["halfwidth"][None]).double()
    probability = values["noise"] + (1.0 - 2.0 * values["noise"]) * inside
    return (
        prior,
        D.Bernoulli(probability).log_prob(values["accept"][:, None]),
        probability,
    )


def _around_prior() -> torch.Tensor:
    values = _around()
    return D.Normal(0.0, 1.0).log_prob(values["res_logits"]).sum() + D.Beta(
        1.0, 20.0
    ).log_prob(values["noise"])


def test_one_latent_for_all_rows_is_one_unit() -> None:
    prior, scores, _ = _around_by_class()
    block = torch.logsumexp(prior + scores.sum(0), -1)
    recorded = _run(_compiled(AROUND), _around())
    units = _units(recorded, "_marg_r")
    assert units.axes == () and units.sizes == ()
    assert units.row_units("accept").tolist() == [0] * 6
    assert float(units.log_weights()) == pytest.approx(float(block), abs=_TOLERANCE)
    joint = float(block + _around_prior())
    assert float(recorded.log_joint.sum()) == pytest.approx(joint, abs=_TOLERANCE)
    assert joint == pytest.approx(-5.647981643676758, abs=1e-5)
    # The other reading, one latent per row, is another number.
    by_row = float(torch.logsumexp(prior + scores, -1).sum() + _around_prior())
    assert by_row == pytest.approx(-5.0164475440979, abs=1e-5)
    assert abs(joint - by_row) > 0.6


def test_one_latent_for_all_rows_refuses_row_scores_and_says_why() -> None:
    recorded = _run(_compiled(AROUND), _around())
    with pytest.raises(UnfactoredObservationError) as caught:
        collect_log_likelihood([recorded], {"accept"}, (1, 1))
    message = str(caught.value)
    assert "all its rows (6 rows of 'accept') share one 'r'" in message
    assert "log of a sum of products" in message
    assert "[over=Trial]" in message and "[via=...]" in message
    assert isinstance(caught.value, TypeError)


@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4, 5), (3, 1, 5, 0, 2, 4)])
def test_one_latent_per_row_scores_each_row_through_its_map(
    order: tuple[int, ...],
) -> None:
    # A trial's own distance is a value of its unit, so a row reads the
    # distance of the unit its map names.
    prior, _, probability = _around_by_class()
    index = torch.tensor(order)
    accept = _around()["accept"]
    scores = D.Bernoulli(probability[index]).log_prob(accept[:, None])
    expected = torch.logsumexp(prior + scores, -1)
    recorded = _run(_compiled(AROUND_BY_ROW), {**_around(), "tidx": index})
    units = _units(recorded, "_marg_r")
    assert units.axes == ("Trial",) and units.sizes == (6,)
    by_unit = torch.zeros(6).index_copy(0, index, expected)
    assert torch.allclose(units.log_weights(), by_unit, atol=_TOLERANCE)
    collected = collect_log_likelihood([recorded], {"accept"}, (1, 1))["accept"]
    assert collected.shape == (1, 1, 6)
    assert torch.allclose(collected[0, 0], expected, atol=_TOLERANCE)
    score = recorded.sites["_marg_r"].log_prob.sum()
    assert abs(float(collected.sum() - score)) < _TOLERANCE
    assert float(recorded.log_joint.sum()) == pytest.approx(
        float(expected.sum() + _around_prior()), abs=_TOLERANCE
    )
    # Reading the units in order, without the map, is another vector unless
    # the map is the identity.
    if order != tuple(range(6)):
        assert not torch.allclose(units.log_weights(), expected, atol=1e-3)


def test_empty_units_contribute_nothing_and_rows_keep_their_scores() -> None:
    values = {name: item for name, item in _product().items() if name != "power"}
    source = (
        PRODUCT.replace("object G : FinSet 2", "object G : FinSet 3")
        .replace(", weights=power", "")
        .replace("object H : FinSet 2", "object H : FinSet 3")
    )
    values["g"] = torch.tensor([2, 0, 0, 1])
    values["h"] = torch.tensor([0, 2, 1, 1])
    scores = D.Normal(values["mu"], 1.0).log_prob(values["y"][:, None])
    expected = torch.logsumexp(values["pa"].log() + scores, -1)
    recorded = _run(_compiled(source), values)
    weights = _units(recorded, "_marg_a").log_weights().reshape(-1)
    flat = values["g"] * 3 + values["h"]
    assert torch.allclose(weights[flat], expected, atol=_TOLERANCE)
    empty = torch.ones(9, dtype=torch.bool).index_fill(0, flat, False)
    assert float(weights[empty].abs().max()) < _TOLERANCE
    rows = collect_log_likelihood([recorded], {"y"}, (1, 1))["y"]
    assert torch.allclose(rows[0, 0], expected, atol=_TOLERANCE)


def test_several_rows_of_a_unit_score_jointly() -> None:
    prior, _, probability = _around_by_class()
    index = torch.tensor([0, 0, 1, 1, 1, 5])
    accept = _around()["accept"]
    scores = D.Bernoulli(probability[index]).log_prob(accept[:, None])
    joint = torch.logsumexp(prior + _grouped(index, scores, 6), -1)
    recorded = _run(_compiled(AROUND_BY_ROW), {**_around(), "tidx": index})
    weights = _units(recorded, "_marg_r").log_weights()
    assert torch.allclose(weights, joint, atol=_TOLERANCE)
    # Reducing each row and then summing over a unit is another number.
    per_row = _grouped(index, torch.logsumexp(prior + scores, -1)[:, None], 6)[:, 0]
    assert not torch.allclose(weights, per_row, atol=1e-3)
    with pytest.raises(
        UnfactoredObservationError, match="Trial=0 holds 2 rows of 'accept'"
    ):
        collect_log_likelihood([recorded], {"accept"}, (1, 1))
    scored = collect_unit_log_likelihood([recorded], {"accept"}, (1, 1))["accept"]
    assert scored.axes == ("Trial",) and scored.responses == ("accept",)
    assert torch.allclose(scored.log_likelihood[0, 0], joint, atol=_TOLERANCE)
    assert torch.equal(scored.row_units["accept"], index)


NEST = """object Row : FinSet 5
object G : FinSet 2
object H : FinSet 3
object A : FinSet 2
object B : FinSet 3
object C : FinSet 2
object Val : Real 1

program nest : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample nu : B <- Normal(loc=0.0, scale=1.0)
    sample tau : C <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    sample pb : B <- HalfNormal(scale=1.0)
    sample pc : C <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        marginalize b : B <- Categorical(probs=pb) [over=G]
            let loc_y = mu[a] + nu[b]
            observe y : Row <- Normal(loc=loc_y, scale=1.0) [via=g]
        observe w : Row <- Normal(loc=mu[a], scale=0.5) [via=g2]
        marginalize c : C <- Categorical(probs=pc) [over=[G, H]]
            observe v : Row <- Normal(loc=tau[c], scale=2.0) [via=[g3, h]]
    return mu

export nest
"""


def _nest() -> dict[str, torch.Tensor]:
    return {
        "mu": torch.tensor([0.1, 0.7]),
        "nu": torch.tensor([-0.5, 0.2, 1.0]),
        "tau": torch.tensor([0.3, -0.4]),
        "pa": torch.tensor([0.3, 0.7]),
        "pb": torch.tensor([0.2, 0.3, 0.5]),
        "pc": torch.tensor([0.6, 0.4]),
        "y": torch.tensor([0.3, -0.2, 0.9, 1.4, 0.0]),
        "w": torch.tensor([0.0, 0.2, 0.4, 0.8, -0.3]),
        "v": torch.tensor([1.0, 0.2, 0.4, -0.8, 0.3]),
        "g": torch.tensor([0, 0, 1, 1, 0]),
        "h": torch.tensor([0, 0, 2, 1, 2]),
        "g2": torch.tensor([1, 0, 1, 1, 0]),
        "g3": torch.tensor([0, 1, 1, 0, 0]),
    }


def _nest_units(values: dict[str, torch.Tensor]) -> torch.Tensor:
    """The log weight of each outer unit, by enumerating every class."""
    weights = torch.zeros(2)
    for unit in range(2):
        terms: list[torch.Tensor] = []
        for a in range(2):
            term = values["pa"][a].log()
            inner: list[torch.Tensor] = []
            for b in range(3):
                score = values["pb"][b].log()
                for row in range(5):
                    if int(values["g"][row]) == unit:
                        loc = values["mu"][a] + values["nu"][b]
                        score = score + D.Normal(loc, 1.0).log_prob(values["y"][row])
                inner.append(score)
            term = term + torch.logsumexp(torch.stack(inner), 0)
            for row in range(5):
                if int(values["g2"][row]) == unit:
                    term = term + D.Normal(values["mu"][a], 0.5).log_prob(
                        values["w"][row]
                    )
            for position in range(3):
                sibling: list[torch.Tensor] = []
                for c in range(2):
                    score = values["pc"][c].log()
                    for row in range(5):
                        if (
                            int(values["g3"][row]) == unit
                            and int(values["h"][row]) == position
                        ):
                            score = score + D.Normal(values["tau"][c], 2.0).log_prob(
                                values["v"][row]
                            )
                    sibling.append(score)
                term = term + torch.logsumexp(torch.stack(sibling), 0)
            terms.append(term)
        weights[unit] = torch.logsumexp(torch.stack(terms), 0)
    return weights


def _nest_prior(values: dict[str, torch.Tensor]) -> torch.Tensor:
    normal = torch.cat([values["mu"], values["nu"], values["tau"]])
    half = torch.cat([values["pa"], values["pb"], values["pc"]])
    return (
        D.Normal(0.0, 1.0).log_prob(normal).sum()
        + D.HalfNormal(1.0).log_prob(half).sum()
    )


def test_nested_and_sibling_blocks_share_the_enclosing_class() -> None:
    values = _nest()
    recorded = _run(_compiled(NEST), values)
    units = _units(recorded, "_marg_a")
    expected = _nest_units(values)
    assert units.axes == ("G",)
    assert units.responses == ("y", "w", "v")
    assert torch.allclose(units.log_weights(), expected, atol=_TOLERANCE)
    score = recorded.sites["_marg_a"].log_prob.sum()
    assert abs(float(units.log_weights().sum() - score)) < _TOLERANCE
    assert float(recorded.log_joint.sum()) == pytest.approx(
        float(expected.sum() + _nest_prior(values)), abs=_TOLERANCE
    )
    # A row of an inner block belongs to the outer unit its group projects to.
    assert torch.equal(units.row_units("y"), values["g"])
    assert torch.equal(units.row_units("w"), values["g2"])
    assert torch.equal(units.row_units("v"), values["g3"])
    # Summing each inner block out on its own, as if the enclosing class were
    # not shared, gives another number.
    alone = {**values, "mu": torch.zeros(2)}
    assert not torch.allclose(_nest_units(alone), expected, atol=1e-3)


def test_the_checked_machine_gives_the_nested_joint() -> None:
    values = _nest()
    latents = ("mu", "nu", "tau", "pa", "pb", "pc")
    run = loads(NEST).run(
        "nest",
        data={
            name: tuple(value.tolist())
            for name, value in values.items()
            if name not in latents
        },
        sites={name: tuple(values[name].tolist()) for name in latents},
    )
    expected = float(_nest_units(values).sum() + _nest_prior(values))
    assert run.log_joint == pytest.approx(expected, abs=1e-9)


def test_a_row_map_into_a_unit_that_holds_other_responses_refuses() -> None:
    values = _nest()
    recorded = _run(_compiled(NEST), values)
    with pytest.raises(UnfactoredObservationError, match="rows of 'w'|of 'w' and"):
        collect_log_likelihood([recorded], {"w"}, (1, 1))
    scored = collect_unit_log_likelihood([recorded], {"w"}, (1, 1))
    assert set(scored) == {"y,w,v"}
    assert torch.allclose(
        scored["y,w,v"].log_likelihood[0, 0], _nest_units(values), atol=_TOLERANCE
    )


PRODUCT = """object Row : FinSet 4
object G : FinSet 2
object H : FinSet 2
object A : FinSet 3
object Val : Real 1

program product_plate : Row -> Val
    sample mu : A <- Normal(loc=0.0, scale=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=[G, H]]
        observe y : Row <- Normal(loc=mu[a], scale=1.0) [via=[g, h], weights=power]
    return mu

export product_plate
"""


def _product() -> dict[str, torch.Tensor]:
    return {
        "mu": torch.tensor([-1.0, 0.0, 2.0]),
        "pa": torch.tensor([0.2, 0.5, 0.3]),
        "y": torch.tensor([0.1, 1.5, -0.7, 0.4]),
        "g": torch.tensor([1, 0, 0, 1]),
        "h": torch.tensor([0, 1, 0, 1]),
        "power": torch.tensor([1.0, 0.5, 0.0, 2.0]),
    }


def test_likelihood_powers_weigh_the_score_and_not_the_row_scores() -> None:
    values = _product()
    recorded = _run(_compiled(PRODUCT), values)
    units = _units(recorded, "_marg_a")
    scores = D.Normal(values["mu"], 1.0).log_prob(values["y"][:, None])
    prior = values["pa"].log()
    flat = values["g"] * 2 + values["h"]
    powered = torch.logsumexp(
        prior + _grouped(flat, values["power"][:, None] * scores, 4), -1
    )
    raw = torch.logsumexp(prior + scores, -1)
    assert units.axes == ("G", "H") and units.sizes == (2, 2)
    assert torch.allclose(units.log_weights().reshape(-1), powered, atol=_TOLERANCE)
    score = recorded.sites["_marg_a"].log_prob.sum()
    assert abs(float(units.log_weights().sum() - score)) < _TOLERANCE
    rows = collect_log_likelihood([recorded], {"y"}, (1, 1))["y"][0, 0]
    assert torch.allclose(rows, raw, atol=_TOLERANCE)
    assert not torch.allclose(raw[2], powered[flat[2]], atol=1e-3)


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_a_reduction_that_is_not_a_sum_out_has_weights_and_no_density(
    reduction: str,
) -> None:
    values = _product()
    source = PRODUCT.replace("[over=[G, H]]", f"[over=[G, H], reduction={reduction}]")
    recorded = _run(_compiled(source), values)
    units = _units(recorded, "_marg_a")
    scores = D.Normal(values["mu"], 1.0).log_prob(values["y"][:, None])
    flat = values["g"] * 2 + values["h"]
    weighted = values["pa"].log() + _grouped(flat, values["power"][:, None] * scores, 4)
    expected = weighted.sum(-1) if reduction == "sum" else weighted.mean(-1)
    assert torch.allclose(units.log_weights().reshape(-1), expected, atol=_TOLERANCE)
    with pytest.raises(UnfactoredObservationError, match=f"by {reduction}"):
        units.log_likelihood()
    with pytest.raises(UnfactoredObservationError, match=f"by {reduction}"):
        collect_log_likelihood([recorded], {"y"}, (1, 1))
    released = {name: value for name, value in values.items() if name != "y"}
    with pytest.raises(ValueError, match="no law to draw from"):
        _run(_compiled(source), released)


INDEXED = """object Row : FinSet 4
object G : FinSet 2
object Val : Real 1

program indexed : Row -> Val
    sample eta <- Normal(loc=0.0, scale=1.0)
    marginalize z : Row <- Bernoulli(probs=p)
        let q = real(z) * 3.0 + 1.0
        observe y : Row <- Poisson(rate=q)
    return eta

export indexed
"""


def _indexed() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    y = torch.tensor([0, 2, 3, 1])
    scores = torch.stack(
        [D.Poisson(1.0).log_prob(y), D.Poisson(4.0).log_prob(y)], dim=-1
    )
    return {"eta": torch.tensor(0.0), "y": y, "p": torch.tensor(0.3)}, scores


def test_an_indexed_latent_is_one_unit_per_element_whatever_the_prior_layout() -> None:
    values, scores = _indexed()
    prior = torch.tensor([0.7, 0.3]).log()
    expected = torch.logsumexp(prior + scores, -1)
    source = INDEXED.replace("program indexed :", "program indexed(p : Real) :")
    recorded = _run(_compiled(INDEXED), values)
    units = _units(recorded, "_marg_z")
    assert units.axes == ("Row",)
    assert torch.allclose(units.log_weights(), expected, atol=_TOLERANCE)
    rows = collect_log_likelihood([recorded], {"y"}, (1, 1))["y"][0, 0]
    assert torch.allclose(rows, expected, atol=_TOLERANCE)
    del source
    # One latent for all the rows is another model, and another number.
    shared = INDEXED.replace("z : Row <-", "z <-")
    block = torch.logsumexp(prior + scores.sum(0), -1)
    recorded = _run(_compiled(shared), values)
    assert float(_units(recorded, "_marg_z").log_weights()) == pytest.approx(
        float(block), abs=_TOLERANCE
    )
    assert abs(float(block - expected.sum())) > 0.1
    with pytest.raises(UnfactoredObservationError, match="marginalize z : Row"):
        collect_log_likelihood([recorded], {"y"}, (1, 1))


def test_an_indexed_latent_reads_the_rows_of_its_element() -> None:
    values, scores = _indexed()
    source = INDEXED.replace("marginalize z : Row", "marginalize z : G").replace(
        "Poisson(rate=q)", "Poisson(rate=q) [via=idx]"
    )
    index = torch.tensor([1, 0, 1, 1])
    prior = torch.tensor([0.7, 0.3]).log()
    expected = torch.logsumexp(prior + _grouped(index, scores, 2), -1)
    recorded = _run(_compiled(source), {**values, "idx": index})
    units = _units(recorded, "_marg_z")
    assert units.axes == ("G",)
    assert torch.allclose(units.log_weights(), expected, atol=_TOLERANCE)
    released = {
        name: value for name, value in {**values, "idx": index}.items() if name != "y"
    }
    drawn = _run(_compiled(source), released)
    assert drawn.sites["y"].value.shape == (4,)
    assert drawn.sites["z"].value.numel() == 2


CHAIN = """object State : FinSet 2
object Symbol : FinSet 3

program cell(symbol, z_prev) : Symbol * State -> State
    marginalize z : State <- Categorical(probs=transition[z_prev])
        observe symbol <- Categorical(probs=emission[z])
    return z

define chain = scan(cell, init=start)

program prog : Symbol -> State
    sample start <- Dirichlet(concentration=1.0) [over=State]
    sample transition : State <- Dirichlet(concentration=1.0) [over=State, iid_over=State]
    sample emission : State <- Dirichlet(concentration=1.0) [over=Symbol, iid_over=State]
    sample path <- chain
    return start

export prog
"""


def test_the_positions_of_a_chain_are_one_unit() -> None:
    start = torch.tensor([0.6, 0.4])
    transition = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
    emission = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    symbols = torch.tensor([[2, 0, 1, 2]])
    total = torch.zeros(())
    for states in itertools.product(range(2), repeat=5):
        term = start[states[0]]
        for position, symbol in enumerate(symbols[0].tolist()):
            term = term * transition[states[position], states[position + 1]]
            term = term * emission[states[position + 1], symbol]
        total = total + term
    recorded = trace(
        _compiled(CHAIN),
        symbols,
        {"start": start, "transition": transition, "emission": emission},
    )
    (score,) = [site for site in recorded.sites.values() if site.units]
    (units,) = score.units
    assert units.axes == ()
    assert float(units.log_weights()) == pytest.approx(
        float(total.log()), abs=_TOLERANCE
    )
    assert float(score.log_prob.sum()) == pytest.approx(
        float(total.log()), abs=_TOLERANCE
    )
    (response,) = units.responses
    assert units.row_units(response).tolist() == [0, 0, 0, 0]
    with pytest.raises(UnfactoredObservationError, match="states of the chain"):
        units.row_log_likelihood(response)


def test_weights_of_a_run_transform_after_the_units_are_read() -> None:
    """The units are the step's, and a score transformer scales its site."""
    values = {**_around(), "tidx": torch.arange(6)}
    program = _compiled(AROUND_BY_ROW)
    with scale(0.5):
        recorded = _run(program, values)
    weights = _units(recorded, "_marg_r").log_weights()
    plain = _run(program, values)
    assert torch.equal(weights, _units(plain, "_marg_r").log_weights())
    assert float(recorded.sites["_marg_r"].log_prob.sum()) == pytest.approx(
        0.5 * float(weights.sum()), abs=_TOLERANCE
    )


def test_every_marginal_score_of_a_program_declares_its_units() -> None:
    for source in (AROUND, AROUND_BY_ROW, NEST, PRODUCT, INDEXED):
        steps = _compiled(source).steps
        scores = [step for step in steps if isinstance(step, Score)]
        assert scores and all(step.units is not None for step in scores)
        responses = {
            row.response
            for step in scores
            if step.units is not None
            for row in step.units.rows
        }
        drawn = {
            name
            for step in steps
            if isinstance(step, Observe) and step.law is not None
            for name in step.names
        }
        assert responses == drawn


# --- laws of released responses ----------------------------------------


def _draws(recorded: Trace, name: str, count: int = _DRAWS) -> torch.Tensor:
    """Joint draws of a released response at one run's values."""
    sampleable = recorded.sites[name].sampleable
    assert sampleable is not None
    return sampleable.raw_morphism.rsample(sampleable.input, torch.Size((count,)))


def _check_frequencies(
    draws: torch.Tensor,
    patterns: list[tuple[int, ...]],
    probabilities: torch.Tensor,
) -> None:
    """Compare how often each pattern is drawn with its probability."""
    positions = {pattern: index for index, pattern in enumerate(patterns)}
    counts = torch.zeros(len(patterns))
    for row in draws.long().tolist():
        counts[positions[tuple(row)]] += 1
    frequencies = counts / draws.shape[0]
    assert float((frequencies - probabilities).abs().max()) < _BOUND


SHARED_PAIR = """object Row : FinSet 3
object G : FinSet 2
object A : FinSet 2
object Val : Real 1

program pair : Row -> Val
    sample rate : A <- Beta(concentration1=1.0, concentration0=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        observe y : Row <- Bernoulli(probs=rate[a]) [via=idx]
        observe w : Row <- Bernoulli(probs=rate[a]) [via=jdx]
    return rate

export pair
"""


def _pair() -> dict[str, torch.Tensor]:
    return {
        "rate": torch.tensor([0.1, 0.85]),
        "pa": torch.tensor([0.4, 0.6]),
        "idx": torch.tensor([0, 0, 1]),
        "jdx": torch.tensor([0, 1, 1]),
    }


def _pair_probabilities(
    given: tuple[int, ...] | None = None,
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """The joint of ``(y, w)``, or of ``w`` given ``y``, by enumeration."""
    values = _pair()
    patterns = list(itertools.product((0, 1), repeat=6))
    joint = torch.zeros(len(patterns))
    for index, pattern in enumerate(patterns):
        total = 0.0
        for classes in itertools.product(range(2), repeat=2):
            term = math.prod(float(values["pa"][k]) for k in classes)
            for row in range(3):
                for value, unit in (
                    (pattern[row], int(values["idx"][row])),
                    (pattern[3 + row], int(values["jdx"][row])),
                ):
                    rate = float(values["rate"][classes[unit]])
                    term *= rate if value else 1.0 - rate
            total += term
        joint[index] = total
    if given is None:
        return patterns, joint
    kept = [index for index, pattern in enumerate(patterns) if pattern[:3] == given]
    conditional = joint[kept]
    return [patterns[index][3:] for index in kept], conditional / conditional.sum()


def _check_pair(program: MonadicProgram) -> None:
    """The first response by itself, and the second given each value of it."""
    recorded = _run(program, _pair())
    patterns, joint = _pair_probabilities()
    assert float(joint.sum()) == pytest.approx(1.0, abs=_TOLERANCE)
    first = list(itertools.product((0, 1), repeat=3))
    margin = torch.zeros(len(first))
    for index, pattern in enumerate(patterns):
        margin[first.index(pattern[:3])] += joint[index]
    _check_frequencies(_draws(recorded, "y"), first, margin)
    for given in first:
        released, conditional = _pair_probabilities(given)
        observed = {**_pair(), "y": torch.tensor(given, dtype=torch.double)}
        _check_frequencies(_draws(_run(program, observed), "w"), released, conditional)


def test_released_responses_of_a_unit_share_their_class() -> None:
    _check_pair(_compiled(SHARED_PAIR))


def test_a_class_per_row_is_refused_by_the_frequencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A law that ignores the earlier response and the sharing must fail."""
    exact = blocks.draw_classes

    def independent(
        levels: list[marginal_module.GroupLevel],
    ) -> tuple[torch.Tensor, ...]:
        """Forget every response bound before the draw."""
        return exact(
            [
                level.with_(
                    log_weights=torch.tensor([0.4, 0.6])
                    .log()
                    .expand_as(level.log_weights)
                )
                for level in levels
            ]
        )

    monkeypatch.setattr(blocks, "draw_classes", independent)
    with pytest.raises(AssertionError):
        _check_pair(_compiled(SHARED_PAIR))


def test_an_unshared_class_is_refused_by_the_frequencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draw with a class per row instead of per unit must fail."""

    def per_row(observation, env, classes):
        rows = observation.row_count(env)
        return exact(
            observation,
            env,
            lambda: torch.stack([classes()[row] for row in range(rows)]),
        )

    exact = blocks.GroupedObservation.draw
    monkeypatch.setattr(blocks.GroupedObservation, "draw", per_row)
    with pytest.raises(AssertionError):
        _check_pair(_compiled(SHARED_PAIR))


NEST_BINARY = """object Row : FinSet 3
object G : FinSet 2
object A : FinSet 2
object B : FinSet 2
object Val : Real 1

program deep : Row -> Val
    sample rate : A <- Beta(concentration1=1.0, concentration0=1.0)
    sample shift : B <- Beta(concentration1=1.0, concentration0=1.0)
    sample pa : A <- HalfNormal(scale=1.0)
    sample pb : B <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=G]
        marginalize b : B <- Categorical(probs=pb) [over=G]
            let q = rate[a] * shift[b]
            observe y : Row <- Bernoulli(probs=q) [via=g]
    return rate

export deep
"""


def test_nested_classes_are_drawn_jointly() -> None:
    values = {
        "rate": torch.tensor([0.98, 0.05]),
        "shift": torch.tensor([0.95, 0.1]),
        "pa": torch.tensor([0.5, 0.5]),
        "pb": torch.tensor([0.35, 0.65]),
        "g": torch.tensor([0, 0, 1]),
    }
    patterns = list(itertools.product((0, 1), repeat=3))
    expected = torch.zeros(len(patterns))
    for index, pattern in enumerate(patterns):
        for classes in itertools.product(range(2), repeat=4):
            a, b = classes[:2], classes[2:]
            term = math.prod(float(values["pa"][k]) for k in a)
            term *= math.prod(float(values["pb"][k]) for k in b)
            for row, value in enumerate(pattern):
                unit = int(values["g"][row])
                rate = float(values["rate"][a[unit]] * values["shift"][b[unit]])
                term *= rate if value else 1.0 - rate
            expected[index] += term
    program = _compiled(NEST_BINARY)
    recorded = _run(program, values)
    _check_frequencies(_draws(recorded, "y"), patterns, expected)
    # The rows' marginals alone, drawn independently, miss the joint.
    margins = torch.stack(
        [
            sum(
                expected[index]
                for index, pattern in enumerate(patterns)
                if pattern[row]
            )
            * torch.ones(())
            for row in range(3)
        ]
    )
    independent = torch.tensor(
        [
            math.prod(
                float(margins[row] if value else 1 - margins[row])
                for row, value in enumerate(pattern)
            )
            for pattern in patterns
        ]
    )
    assert float((independent - expected).abs().max()) > 1.5 * _BOUND
    law = recorded.sites["y"].sampleable
    assert law is not None
    distribution = law.raw_morphism.sampling_distribution(law.input)
    assert torch.allclose(distribution.mean, margins, atol=_TOLERANCE)


def test_classes_of_a_refining_plate_follow_their_enclosing_unit() -> None:
    """Two inner units per outer unit, with a class prior the outer class sets."""
    outer = torch.tensor([[0.3, 0.7], [0.6, 0.4]]).log()
    inner = torch.tensor(
        [
            [[0.9, 0.1], [0.2, 0.8]],
            [[0.5, 0.5], [0.1, 0.9]],
            [[0.3, 0.7], [0.6, 0.4]],
        ]
    ).log()
    # An inner unit's evidence need not sum to one over its classes.
    evidence = (
        inner + torch.tensor([[0.0, -1.0], [0.5, 0.0]])[:, None].repeat(2, 1, 1)[:3]
    )
    parent = torch.tensor([0, 1, 0])
    levels = [
        marginal_module.GroupLevel(log_weights=outer),
        marginal_module.GroupLevel(log_weights=evidence, parent=parent),
    ]
    patterns = list(itertools.product((0, 1), repeat=5))
    weights = torch.zeros(len(patterns))
    for index, pattern in enumerate(patterns):
        a, b = pattern[:2], pattern[2:]
        score = sum(float(outer[unit, a[unit]]) for unit in range(2))
        score += sum(
            float(evidence[unit, a[int(parent[unit])], b[unit]]) for unit in range(3)
        )
        weights[index] = math.exp(score)
    expected = weights / weights.sum()
    drawn = []
    for _ in range(_DRAWS):
        enclosing, own = marginal_module.draw_classes(levels)
        a = torch.stack([enclosing[0], enclosing[1]])
        assert int(enclosing[2]) == int(enclosing[0])
        drawn.append(torch.cat([a, own]))
    draws = torch.stack(drawn)
    _check_frequencies(draws, patterns, expected)
    # Without the evidence of the inner units the outer class is another law.
    marginal = torch.zeros(2)
    for index, pattern in enumerate(patterns):
        marginal[pattern[0]] += expected[index]
    assert float((marginal - outer[0].exp()).abs().max()) > 1.5 * _BOUND


def _posterior(draws: dict[str, torch.Tensor]) -> MCMCResult:
    """Recorded draws of one chain, as a sampler would record them."""
    count = next(iter(draws.values())).shape[0]
    return MCMCResult(
        samples={name: value.unsqueeze(0) for name, value in draws.items()},
        log_densities=torch.zeros(1, count),
        acceptance_rates=torch.ones(1),
        divergence_counts=torch.zeros(1, dtype=torch.long),
        num_warmup=0,
        num_samples=count,
    )


def test_expectations_are_class_weighted_means() -> None:
    prior, _, probability = _around_by_class()
    expected = (prior.exp() * probability).sum(-1)
    values = _around()
    draws = {
        "res_logits": values["res_logits"].expand(2, 3),
        "noise": values["noise"].reshape(1).expand(2),
    }
    data = {name: values[name] for name in ("dist", "halfwidth")}
    x = torch.zeros(1, 1, dtype=torch.long)
    for source, extra in ((AROUND, {}), (AROUND_BY_ROW, {"tidx": torch.arange(6)})):
        result = posterior_expectations(
            _compiled(source),
            _posterior(draws),
            x,
            {**data, **extra},
            sites=("accept",),
            seed=0,
        )["accept"]
        assert result.shape[:2] == (1, 2)
        for draw in result.reshape(2, 6):
            assert torch.allclose(draw, expected, atol=1e-7)


def test_categorical_expectations_are_class_weighted_probabilities() -> None:
    source = """object Row : FinSet 4
object A : FinSet 2
object W : FinSet 3
object Val : Real 1

program topics : Row -> Val
    sample phi : A <- Dirichlet(concentration=1.0) [over=W]
    sample pa : A <- HalfNormal(scale=1.0)
    marginalize a : A <- Categorical(probs=pa) [over=Row]
        observe w : Row <- Categorical(probs=phi[a]) [via=idx]
    return pa

export topics
"""
    phi = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    pa = torch.tensor([0.25, 0.75])
    expected = (pa[:, None] * phi).sum(0).expand(4, 3)
    result = posterior_expectations(
        _compiled(source),
        _posterior({"phi": phi.unsqueeze(0), "pa": pa.unsqueeze(0)}),
        torch.zeros(1, 1, dtype=torch.long),
        {"idx": torch.arange(4)},
        sites=("w",),
        seed=0,
    )["w"]
    assert torch.allclose(result[0, 0], expected, atol=1e-7)
    words = torch.tensor([2, 0, 1, 1])
    recorded = _run(
        _compiled(source), {"phi": phi, "pa": pa, "idx": torch.arange(4), "w": words}
    )
    rows = collect_log_likelihood([recorded], {"w"}, (1, 1))["w"][0, 0]
    assert torch.allclose(rows, (pa[:, None] * phi).sum(0).log()[words], atol=1e-7)


def test_the_marginal_law_is_a_mixture_over_the_classes() -> None:
    values = _nest()
    released = {name: value for name, value in values.items() if name != "y"}
    site = _run(_compiled(NEST), released).sites["y"]
    assert site.sampleable is not None and not site.is_observed
    law = site.sampleable.raw_morphism.sampling_distribution(site.sampleable.input)
    assert isinstance(law, Mixture)
    weights = values["pa"][:, None] * values["pb"][None]
    locations = values["mu"][:, None] + values["nu"][None]
    mean = float((weights * locations).sum())
    second = float((weights * (1.0 + locations.square())).sum())
    assert torch.allclose(law.mean, torch.full((5,), mean), atol=_TOLERANCE)
    assert torch.allclose(
        law.variance, torch.full((5,), second - mean**2), atol=_TOLERANCE
    )
    at = torch.tensor(0.4)
    density = torch.logsumexp(
        (weights.log() + D.Normal(locations, 1.0).log_prob(at)).reshape(-1), 0
    )
    assert torch.allclose(law.log_prob(at), density.expand(5), atol=_TOLERANCE)


def test_predictive_draws_use_the_live_response() -> None:
    _, _, probability = _around_by_class()
    prior, _, _ = _around_by_class()
    expected = (prior.exp() * probability).sum(-1)
    values = _around()
    draws = {
        "res_logits": values["res_logits"].expand(300, 3),
        "noise": values["noise"].reshape(1).expand(300),
    }
    program = _compiled(AROUND_BY_ROW)
    predicted = Predictive(program, _posterior(draws))(
        torch.zeros(1, 1, dtype=torch.long),
        {
            "dist": values["dist"],
            "halfwidth": values["halfwidth"],
            "tidx": torch.arange(6),
        },
    )["accept"]
    assert predicted.shape == (300, 6)
    bound = 5.0 * math.sqrt(0.25 / 300)
    assert float((predicted.mean(0) - expected).abs().max()) < bound


def test_a_released_response_scores_the_value_it_drew() -> None:
    """The joint of an ancestral run is the density at its own draw."""
    values = {name: value for name, value in _around().items() if name != "accept"}
    values["tidx"] = torch.arange(6)
    program = _compiled(AROUND_BY_ROW)
    recorded = _run(program, values)
    drawn = recorded.sites["accept"].value
    again = _joint(program, {**values, "accept": drawn})
    assert float(recorded.log_joint.sum()) == pytest.approx(again, abs=_TOLERANCE)


def test_truncated_components_are_mixed_after_truncation() -> None:
    source = PRODUCT.replace("weights=power", "trunc_lower=0.0")
    values = {
        name: value for name, value in _product().items() if name not in ("power", "y")
    }
    site = _run(_compiled(source), values).sites["y"]
    assert float(site.value.min()) >= 0.0
    assert site.sampleable is not None
    law = site.sampleable.raw_morphism.sampling_distribution(site.sampleable.input)
    normal = D.Normal(torch.zeros(()), torch.ones(()))
    alpha = -values["mu"]
    inverse_mills = normal.log_prob(alpha).exp() / (1.0 - normal.cdf(alpha))
    means = values["mu"] + inverse_mills
    expected = float((values["pa"] * means).sum())
    assert torch.allclose(law.mean, torch.full((4,), expected), atol=1e-8)


# --- scoring paths -----------------------------------------------------


def _check_plans() -> None:
    """Every plan against the machine, and its joint against the densities."""
    prior, _, probability = _around_by_class()
    index = torch.tensor([2, 0, 1, 5, 4, 3])
    scores = D.Bernoulli(probability[index]).log_prob(_around()["accept"][:, None])
    product = _product()
    powered = product["power"][:, None] * D.Normal(product["mu"], 1.0).log_prob(
        product["y"][:, None]
    )
    flat = product["g"] * 2 + product["h"]
    cases = (
        (
            AROUND_BY_ROW,
            {**_around(), "tidx": index},
            torch.logsumexp(prior + scores, -1).sum() + _around_prior(),
        ),
        (NEST, _nest(), _nest_units(_nest()).sum() + _nest_prior(_nest())),
        (
            PRODUCT,
            product,
            torch.logsumexp(product["pa"].log() + _grouped(flat, powered, 4), -1).sum()
            + D.Normal(0.0, 1.0).log_prob(product["mu"]).sum()
            + D.HalfNormal(1.0).log_prob(product["pa"]).sum(),
        ),
    )
    x = torch.zeros(1, 1, dtype=torch.long)
    for source, values, expected in cases:
        program = _compiled(source)
        plan = prepare_joint(program, x, values)
        assert not isinstance(plan, str)
        densities, total = plan.evaluate(x, values)
        recorded = _run(program, values)
        assert float(total) == pytest.approx(float(expected), abs=_TOLERANCE)
        assert torch.equal(total, recorded.log_joint)
        for name, density in densities.items():
            assert torch.equal(density, recorded.sites[name].log_prob)


def test_the_compiled_plan_and_the_machine_agree() -> None:
    _check_plans()


def test_a_plan_that_drops_or_doubles_the_score_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = joint_plan._ScoreAdd.execute

    def dropped(
        instruction: joint_plan._ScoreAdd, state: joint_plan._Evaluation
    ) -> None:
        state.slots[instruction.target] = None

    def doubled(
        instruction: joint_plan._ScoreAdd, state: joint_plan._Evaluation
    ) -> None:
        exact(instruction, state)
        exact(instruction, state)

    for wrong in (dropped, doubled):
        monkeypatch.setattr(joint_plan._ScoreAdd, "execute", wrong)
        with pytest.raises(AssertionError):
            _check_plans()


def test_many_runs_score_as_each_run_alone() -> None:
    program = _compiled(AROUND_BY_ROW)
    values = {**_around(), "tidx": torch.arange(6)}
    logits = torch.randn(5, 3)
    responses = torch.randint(0, 2, (5, 6)).double()
    per_run = {**values, "res_logits": logits, "accept": responses}
    x = torch.zeros(5, 1, dtype=torch.long)
    together = program.log_joint(x, per_run)
    alone = torch.tensor(
        [
            _joint(
                program, {**values, "res_logits": logits[run], "accept": responses[run]}
            )
            for run in range(5)
        ]
    )
    assert torch.allclose(together, alone, atol=_TOLERANCE)
    recorded = trace(program, x, per_run)
    units = recorded.sites["_marg_r"].units
    assert len(units) == 5
    for run in range(5):
        assert float(units[run].log_weights().sum()) == pytest.approx(
            float(recorded.sites["_marg_r"].log_prob[run]), abs=_TOLERANCE
        )
    with pytest.raises(ValueError, match="5 runs"):
        collect_log_likelihood([recorded], {"accept"}, (1, 1))


def test_scoring_policy_drops_the_score_and_keeps_the_raw_rows() -> None:
    values = {**_around(), "tidx": torch.arange(6)}
    program = _compiled(AROUND_BY_ROW)
    prior, scores, _ = _around_by_class()
    with observation_scoring(False):
        recorded = _run(program, values)
    assert float(recorded.log_joint.sum()) == pytest.approx(
        float(_around_prior()), abs=_TOLERANCE
    )
    rows = collect_log_likelihood([recorded], {"accept"}, (1, 1))["accept"][0, 0]
    assert torch.allclose(rows, torch.logsumexp(prior + scores, -1), atol=_TOLERANCE)


def test_tempering_puts_the_block_in_the_likelihood() -> None:
    values = {**_around(), "tidx": torch.arange(6)}
    program = _compiled(AROUND_BY_ROW)
    latents = {name: values.pop(name) for name in ("res_logits", "noise")}
    registry = LatentRegistry.from_model(program, set(values))
    assert registry.names == ("res_logits", "noise")
    potential = TemperedPotential(
        program, registry, torch.zeros(1, 1, dtype=torch.long), values
    )
    position = torch.cat(
        [latents["res_logits"], torch.logit(latents["noise"]).reshape(1)]
    )
    prior, scores, _ = _around_by_class()
    _, likelihood = potential.log_prior_and_likelihood(position)
    assert float(likelihood) == pytest.approx(
        float(torch.logsumexp(prior + scores, -1).sum()), abs=1e-9
    )


def _check_refusal() -> None:
    """A response with no value is refused, and one with a value is data."""
    values = {**_around(), "tidx": torch.arange(6)}
    program = _compiled(AROUND_BY_ROW)
    data = set(values) - {"res_logits", "noise"}
    assert LatentRegistry.from_model(program, data).names == ("res_logits", "noise")
    with pytest.raises(ValueError, match="data of the model rather than a latent"):
        LatentRegistry.from_model(program, data - {"accept"})
    released = {name: value for name, value in values.items() if name != "accept"}
    with pytest.raises(ValueError, match="data of the model rather than a latent"):
        LatentRegistry.from_trace(program, _run(program, released))


def test_a_response_is_never_a_latent() -> None:
    _check_refusal()


def test_a_registry_that_admits_the_response_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "MarginalizedFactor", Independent)
    with pytest.raises((pytest.fail.Exception, AssertionError, NotImplementedError)):
        _check_refusal()


def _check_objective() -> None:
    """The ELBO at seeded draws, and its gradient, against the densities."""
    values = {**_around(), "tidx": torch.arange(6)}
    program = _compiled(AROUND_BY_ROW)
    data = {
        name: value
        for name, value in values.items()
        if name not in ("res_logits", "noise")
    }
    guide = AutoNormalGuide(program, observed_names=set(data))
    x = torch.zeros(1, 1, dtype=torch.long)

    def gradient(loss: torch.Tensor) -> torch.Tensor:
        found = torch.autograd.grad(loss, tuple(guide.parameters()))
        return torch.cat([item.reshape(-1) for item in found])

    torch.manual_seed(11)
    loss = ELBO()(program, guide, x, data)
    torch.manual_seed(11)
    draws = guide.rsample(x)
    logits, noise = draws["res_logits"].reshape(3), draws["noise"].reshape(())
    inside = (data["dist"][:, None] <= data["halfwidth"][None]).double()
    scores = D.Bernoulli(noise + (1 - 2 * noise) * inside).log_prob(
        data["accept"][:, None]
    )
    joint = (
        torch.logsumexp(logits.log_softmax(-1) + scores, -1).sum()
        + D.Normal(0.0, 1.0).log_prob(logits).sum()
        + D.Beta(1.0, 20.0).log_prob(noise)
    )
    expected = guide.log_prob(x, draws).sum() - joint
    assert float(loss) == pytest.approx(float(expected), abs=1e-9)
    wanted = gradient(expected)
    assert torch.allclose(gradient(loss), wanted, atol=1e-9)
    optimizer = torch.optim.SGD(guide.parameters(), lr=1e-2)
    before = [item.detach().clone() for item in guide.parameters()]
    torch.manual_seed(11)
    SVI(program, guide, optimizer, ELBO()).step(x, data)
    moved = torch.cat(
        [
            (new - old).reshape(-1)
            for new, old in zip(guide.parameters(), before, strict=True)
        ]
    )
    for item, old in zip(guide.parameters(), before, strict=True):
        item.data.copy_(old)
    assert torch.allclose(moved, -1e-2 * wanted, atol=1e-9)


def test_a_variational_step_descends_the_marginal_objective() -> None:
    _check_objective()


def test_an_objective_without_the_class_sum_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = blocks.GroupedBlock.unit_weights
    for wrong in (
        # The block scores nothing.
        lambda *given, **options: exact(*given, **options) * 0.0,
        # The block scores the first unit for every unit.
        lambda *given, **options: exact(*given, **options)[:1].expand(6),
    ):
        monkeypatch.setattr(blocks.GroupedBlock, "unit_weights", wrong)
        with pytest.raises(AssertionError):
            _check_objective()


# --- fits ---------------------------------------------------------------


def _fit(source: str, index: torch.Tensor | None) -> ModelFit:
    """A bound model with recorded draws at two parameter values."""
    values = _around()
    data: dict[str, object] = {
        "dist": values["dist"],
        "halfwidth": values["halfwidth"],
        "accept": values["accept"].bool(),
    }
    if index is not None:
        data["tidx"] = index
    bound = Model(source).bind(data)
    logits = torch.stack([values["res_logits"], torch.tensor([-0.3, 0.6, 0.0])])
    return ModelFit(
        model=bound,
        posterior=_posterior({"res_logits": logits, "noise": torch.tensor([0.1, 0.2])}),
    )


def _fit_scores() -> torch.Tensor:
    """The row scores of the two draws, from torch densities."""
    values = _around()
    rows: list[torch.Tensor] = []
    for logits, noise in (
        (values["res_logits"], 0.1),
        (torch.tensor([-0.3, 0.6, 0.0]), 0.2),
    ):
        inside = (values["dist"][:, None] <= values["halfwidth"][None]).double()
        rows.append(
            torch.logsumexp(
                logits.log_softmax(-1)
                + D.Bernoulli(noise + (1 - 2 * noise) * inside).log_prob(
                    values["accept"][:, None]
                ),
                -1,
            )
        )
    return torch.stack(rows)


def test_a_fit_reads_row_scores_of_a_block_grouped_by_the_rows() -> None:
    fit = _fit(AROUND_BY_ROW, torch.arange(6))
    expected = _fit_scores()
    likelihood = fit.log_likelihood()["accept"]
    assert torch.allclose(likelihood[0], expected, atol=1e-6)
    analysis = fit.analysis()
    assert torch.allclose(analysis.log_likelihood["accept"][0], expected, atol=1e-6)
    lpd = torch.logsumexp(expected, 0) - math.log(2)
    variance = expected.var(dim=0, correction=1)
    result = analysis.waic()
    assert result.elpd_waic == pytest.approx(float((lpd - variance).sum()), abs=1e-5)
    units = fit.unit_log_likelihood()["accept"]
    assert units.axes == ("Trial",)
    assert torch.allclose(units.log_likelihood[0], expected, atol=1e-6)


def test_a_fit_of_one_shared_latent_reads_the_block_and_refuses_the_rows() -> None:
    fit = _fit(AROUND, None)
    with pytest.raises(UnfactoredObservationError, match="over=Trial"):
        fit.log_likelihood()
    with pytest.raises(UnfactoredObservationError, match="over=Trial"):
        fit.analysis()
    scored = fit.unit_log_likelihood()["accept"]
    assert scored.axes == () and scored.log_likelihood.shape == (1, 2, 1)
    values = _around()
    inside = (values["dist"][:, None] <= values["halfwidth"][None]).double()
    rows = D.Bernoulli(0.1 + 0.8 * inside).log_prob(values["accept"][:, None])
    block = torch.logsumexp(values["res_logits"].log_softmax(-1) + rows.sum(0), -1)
    assert float(scored.log_likelihood[0, 0, 0]) == pytest.approx(
        float(block), abs=1e-6
    )
    analysis = fit.analysis(units="factors")
    assert analysis.log_likelihood["accept"].shape == (1, 2, 1)
    assert "accept" not in analysis.observed_data


def test_a_fit_scores_the_groups_of_rows_that_share_a_latent() -> None:
    fit = _fit(AROUND_BY_ROW, torch.tensor([0, 0, 2, 3, 4, 5]))
    with pytest.raises(UnfactoredObservationError, match="Trial=0 holds 2 rows"):
        fit.analysis()
    analysis = fit.analysis(units="factors")
    assert analysis.log_likelihood["accept"].shape == (1, 2, 6)
    tree = analysis.to_datatree()
    assert tree["log_likelihood"]["accept"].dims[2:] == ("Trial",)
    released = fit.predict(num_samples=2)["accept"]
    assert released.shape == (2, 6)
