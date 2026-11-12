"""Latent network weights: ``[weights=latent, prior_scale=s]``.

A network kernel or a learned function declared with latent weights draws
every weight and bias of its network as a site with an independent
``Normal(0, s)`` prior, once per run, in each program without value
parameters that uses it. Each density claim is checked against a
hand-written torch joint at the same values: the compiled runtime (float32)
at ``rel=1e-5`` against the float64 oracle, the reference machine (float64)
at ``rel=1e-9``, and each paired with a wrong oracle the same tolerance
rejects. The posterior check compares NUTS with a quadrature of the exact
posterior, at four Monte Carlo standard errors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest
import torch
import torch.distributions as D

from quivers.dsl import CompileError, loads
from quivers.dsl.ast_nodes import ObjectDecl, ProgramDecl, SampleStep
from quivers.dsl.latent_weights import expand_latent_weights
from quivers.dsl.parser import parse
from quivers.dsl.program_elaboration import LATENT_WEIGHT_HEADS
from quivers.inference import MCMC, NUTSKernel, trace
from quivers.transpile import UnsupportedConstruct, available_targets, transpile

type Host = float | tuple["Host", ...]
"""A value as the reference machine takes it."""


def _normal(scale: float) -> D.Normal:
    """A zero-mean Normal with float64 parameters, for a float64 oracle.

    Parameters
    ----------
    scale : float
        Its standard deviation.

    Returns
    -------
    D.Normal
        The distribution; python-number parameters would make it float32.
    """
    return D.Normal(
        torch.tensor(0.0, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64)
    )


def _host(value: torch.Tensor) -> Host:
    """A tensor as nested tuples of floats, or a float for a scalar."""
    value = value.detach().double()
    if value.dim() == 0:
        return float(value)
    return tuple(_host(row) for row in value)


REGRESSION = """\
object Feat : Real 2
object Target : Real 1
object Row : FinSet 6
morphism net : Feat -> Target [param_source=mlp(3), weights=latent, prior_scale=0.5] ~ Normal
program reg : Row -> Target
    observe y : Row <- net(x)
    return y
export reg
"""

_SHAPES = {
    "net_param_layer0_weight": (3, 2),
    "net_param_layer0_bias": (3,),
    "net_param_weight": (2, 3),
    "net_param_bias": (2,),
}


def _regression_oracle(
    weights: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    scale: float = 0.5,
) -> float:
    """The regression's joint: weight priors and a Normal per row, float64."""
    w = {k: v.double() for k, v in weights.items()}
    prior = sum(float(_normal(scale).log_prob(v).sum()) for v in w.values())
    hidden = torch.tanh(
        x.double() @ w["net_param_layer0_weight"].T + w["net_param_layer0_bias"]
    )
    raw = hidden @ w["net_param_weight"].T + w["net_param_bias"]
    # The kernel's Normal reads its location from the first head and the
    # exponential of the second as its scale.
    likelihood = D.Normal(raw[:, 0], raw[:, 1].exp()).log_prob(y.double()).sum()
    return prior + float(likelihood)


def test_the_expansion_draws_each_weight_in_the_program_that_uses_it() -> None:
    expanded = expand_latent_weights(parse(REGRESSION), LATENT_WEIGHT_HEADS)
    objects = {
        name
        for statement in expanded.statements
        if isinstance(statement, ObjectDecl)
        for name in statement.names
    }
    assert {"net__w0_rows", "net__w0_cols", "net__w1_rows", "net__w1_cols"} <= objects
    program = next(s for s in expanded.statements if isinstance(s, ProgramDecl))
    drawn = [step.vars[0] for step in program.draws if isinstance(step, SampleStep)]
    assert drawn == list(_SHAPES)
    # Expanding twice changes nothing.
    assert expand_latent_weights(expanded, LATENT_WEIGHT_HEADS) == expanded


def test_a_latent_kernel_scores_alike_on_both_paths_and_by_hand() -> None:
    torch.manual_seed(0)
    program = loads(REGRESSION)
    model = program.morphism
    assert isinstance(model, torch.nn.Module)
    assert list(model.named_parameters()) == []
    x, y = torch.randn(6, 2), torch.randn(6)
    recorded = trace(model, torch.zeros(1, 1), observations={"x": x, "y": y})
    weights = {
        name: recorded.sites[name].value.reshape(shape)
        for name, shape in _SHAPES.items()
    }
    oracle = _regression_oracle(weights, x, y)
    assert float(recorded.log_joint[0]) == pytest.approx(oracle, rel=1e-5)
    run = program.run(
        "reg",
        data={"x": _host(x), "y": _host(y[:, None])},
        sites={name: _host(value) for name, value in weights.items()},
        seed=0,
    )
    assert run.log_joint == pytest.approx(oracle, rel=1e-9)
    # A prior at the default scale of one is told apart.
    wrong = _regression_oracle(weights, x, y, scale=1.0)
    assert abs(wrong - oracle) > 1e-5 * abs(oracle)


def test_the_checked_program_reads_the_weights_as_sites() -> None:
    program = loads(REGRESSION)
    entry = program.entry("reg")
    assert [(p.name, p.role) for p in entry.parameters] == [
        ("x", "data"),
        ("y", "observation"),
    ]


LET_CALL = """\
object Feat : Real 3
object Out : Real 2
morphism proj : Feat -> Out [role=function, param_source=mlp(4), weights=latent, prior_scale=0.5]
program p : Feat -> Out
    sample a <- Normal(loc=0.0, scale=1.0) [over=Feat]
    let z = proj(a)
    observe y <- Normal(loc=z, scale=0.5)
    return z
export p
"""


def test_a_latent_function_scores_alike_on_both_paths_and_by_hand() -> None:
    torch.manual_seed(1)
    program = loads(LET_CALL)
    recorded = trace(
        program.morphism, torch.zeros(1, 3), observations={"y": torch.randn(1, 2)}
    )
    shapes = {
        "proj_param_layer0_weight": (4, 3),
        "proj_param_layer0_bias": (4,),
        "proj_param_weight": (2, 4),
        "proj_param_bias": (2,),
    }
    w = {
        name: recorded.sites[name].value.reshape(s).double()
        for name, s in shapes.items()
    }
    a = recorded.sites["a"].value.reshape(3).double()
    y = recorded.sites["y"].value.reshape(2).double()

    def oracle(hidden_scale: float) -> float:
        hidden = torch.tanh(
            w["proj_param_layer0_weight"] @ a + w["proj_param_layer0_bias"]
        )
        z = w["proj_param_weight"] @ hidden + w["proj_param_bias"]
        total = sum(float(_normal(hidden_scale).log_prob(v).sum()) for v in w.values())
        total += float(_normal(1.0).log_prob(a).sum())
        return total + float(D.Normal(z, 0.5).log_prob(y).sum())

    expected = oracle(0.5)
    assert float(recorded.log_joint[0]) == pytest.approx(expected, rel=1e-5)
    run = program.run(
        "p",
        data={"feat": (0.0, 0.0, 0.0), "y": _host(y)},
        sites={"a": _host(a), **{k: _host(v) for k, v in w.items()}},
        seed=0,
    )
    assert run.log_joint == pytest.approx(expected, rel=1e-9)
    assert abs(oracle(1.0) - expected) > 1e-5 * abs(expected)


RECURRENT = """\
object Driver : Real 2
object Hidden : Real 3
object Obs : Real 1
morphism cell : Driver * Hidden -> Hidden [role=function, output=tanh, weights=latent]
morphism readout : Hidden -> Obs [weights=latent, prior_scale=0.5] ~ Normal
program step(u_t, h_prev) : Driver * Hidden -> Hidden
    let h = cell(u_t, h_prev)
    observe o <- readout(h)
    return h
define run_all = scan(step)
program rnn : Driver -> Hidden
    sample last <- run_all
    return last
export rnn
"""


def test_latent_networks_in_a_scan_cell_are_drawn_once_per_run() -> None:
    torch.manual_seed(2)
    program = loads(RECURRENT)
    drivers, observed = torch.randn(1, 4, 2), torch.randn(1, 4, 1)
    recorded = trace(program.morphism, drivers, observations={"o": observed})
    drawn = [
        name
        for name, site in recorded.sites.items()
        if not site.is_observed and not site.is_deterministic
    ]
    assert drawn == [
        "cell_param_weight",
        "cell_param_bias",
        "readout_param_weight",
        "readout_param_bias",
    ]
    w = {
        "cell_param_weight": recorded.sites["cell_param_weight"].value.reshape(3, 5),
        "cell_param_bias": recorded.sites["cell_param_bias"].value.reshape(3),
        "readout_param_weight": recorded.sites["readout_param_weight"].value.reshape(
            2, 3
        ),
        "readout_param_bias": recorded.sites["readout_param_bias"].value.reshape(2),
    }
    w64 = {k: v.double() for k, v in w.items()}

    def oracle(shift: float) -> float:
        total = float(_normal(1.0).log_prob(w64["cell_param_weight"]).sum())
        total += float(_normal(1.0).log_prob(w64["cell_param_bias"]).sum())
        total += float(_normal(0.5).log_prob(w64["readout_param_weight"]).sum())
        total += float(_normal(0.5).log_prob(w64["readout_param_bias"]).sum())
        h = torch.zeros(3, dtype=torch.float64)
        for t in range(4):
            joined = torch.cat([drivers[0, t].double(), h])
            h = torch.tanh(w64["cell_param_weight"] @ joined + w64["cell_param_bias"])
            raw = w64["readout_param_weight"] @ h + w64["readout_param_bias"]
            total += float(
                D.Normal(raw[0] + shift, raw[1].exp()).log_prob(
                    observed[0, t, 0].double()
                )
            )
        return total

    expected = oracle(0.0)
    assert float(recorded.log_joint[0]) == pytest.approx(expected, rel=1e-5)
    run = program.run(
        "rnn",
        data={"driver": _host(drivers[0]), "o": _host(observed[0])},
        sites={k: _host(v) for k, v in w.items()},
        seed=0,
    )
    assert run.log_joint == pytest.approx(expected, rel=1e-9)
    assert abs(oracle(0.01) - expected) > 1e-5 * abs(expected)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ("[weights=sampled] ~ Normal", "weights=sampled is neither"),
        ("[prior_scale=0.5] ~ Normal", "declare weights=latent to use it"),
        ("[weights=latent, prior_scale=-1.0] ~ Normal", "positive number"),
        ("[param_source=attention, weights=latent] ~ Normal", "'attention' source"),
        (
            "[weights=latent] ~ Horseshoe",
            "a family whose parameters a network supplies",
        ),
    ],
)
def test_malformed_latent_weight_options_are_refused(
    options: str, message: str
) -> None:
    source = REGRESSION.replace(
        "[param_source=mlp(3), weights=latent, prior_scale=0.5] ~ Normal", options
    )
    with pytest.raises(CompileError, match=message):
        loads(source)


def test_a_latent_network_composed_outside_a_scan_is_refused() -> None:
    source = REGRESSION.replace(
        "program reg", "define chain = net >> net2\nprogram reg"
    ).replace(
        "object Row : FinSet 6",
        "object Row : FinSet 6\nmorphism net2 : Target -> Target ~ Normal",
    )
    with pytest.raises(CompileError, match="composes 'net', whose weights are latent"):
        loads(source)


def test_a_latent_kernel_drawn_without_named_inputs_is_refused() -> None:
    source = REGRESSION.replace("observe y : Row <- net(x)", "observe y : Row <- net")
    with pytest.raises(CompileError, match="drawn from with its inputs named"):
        loads(source)


@pytest.mark.parametrize("network", ["", "param_source=mlp(3), "])
def test_every_target_refuses_latent_network_weights_by_name(network: str) -> None:
    source = REGRESSION.replace(
        "[param_source=mlp(3), weights=latent, prior_scale=0.5]",
        f"[{network}weights=latent]",
    )
    kind = "param-source:mlp:latent" if network else "param-source:linear:latent"
    for target in available_targets():
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(parse(source), target=target)
        reported = {k.partition(": ")[0] for k in caught.value.kinds}
        assert kind in reported, (target, caught.value.kinds)


POSTERIOR = """\
object Feat : Real 1
object Target : Real 1
object Row : FinSet 5
morphism net : Feat -> Target [weights=latent] ~ Normal
program reg : Row -> Target
    observe y : Row <- net(x)
    return y
export reg
"""


@pytest.mark.slow
def test_nuts_over_latent_weights_matches_a_quadrature_of_the_posterior() -> None:
    """Posterior means of the four weights against a grid integral.

    The model has four weights (location and log-scale heads, each a
    slope and an intercept) under unit Normal priors, so the posterior is
    a four-dimensional density a grid integrates directly. The tolerance
    for each weight is four Monte Carlo standard errors: the grid's
    posterior standard deviation over the square root of the chain's
    effective sample size.
    """
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(5, 1, generator=generator)
    y = 0.8 * x[:, 0] + 0.3 + 0.5 * torch.randn(5, generator=generator)
    program = loads(POSTERIOR)
    result = MCMC(
        NUTSKernel(max_tree_depth=6),
        num_warmup=200,
        num_samples=400,
        num_chains=1,
        seed=0,
    ).run(program.morphism, torch.zeros(1, 1), {"x": x, "y": y})

    axis = torch.linspace(-4.0, 4.0, 41, dtype=torch.float64)
    w_loc, w_scale, b_loc, b_scale = torch.meshgrid(
        axis, axis, axis, axis, indexing="ij"
    )
    xs, ys = x[:, 0].double(), y.double()
    log_density = (
        _normal(1.0).log_prob(torch.stack([w_loc, w_scale, b_loc, b_scale])).sum(0)
    )
    for i in range(5):
        loc = w_loc * xs[i] + b_loc
        scale = (w_scale * xs[i] + b_scale).exp()
        log_density = log_density + D.Normal(loc, scale).log_prob(ys[i])
    weights = torch.softmax(log_density.reshape(-1), dim=0)
    grids = {
        ("net_param_weight", 0): w_loc,
        ("net_param_weight", 1): w_scale,
        ("net_param_bias", 0): b_loc,
        ("net_param_bias", 1): b_scale,
    }
    rejected = 0
    for (site, k), grid in grids.items():
        values = grid.reshape(-1)
        mean = float((weights * values).sum())
        sd = math.sqrt(float((weights * (values - mean) ** 2).sum()))
        draws = result.samples[site].reshape(-1, 2)[:, k].double()
        effective = float(result.ess[site].reshape(-1)[k])
        tolerance = 4.0 * sd / math.sqrt(effective)
        assert abs(float(draws.mean()) - mean) <= tolerance, (site, k, mean, tolerance)
        # The prior mean of zero is told apart wherever the data move it.
        if abs(float(draws.mean())) > tolerance:
            rejected += 1
    assert rejected >= 1
