"""Numeric tier for [`IRMarginalize`][quivers.transpile.ir.IRMarginalize].

The composition-fixture tier [`test_numeric_equivalence`][tests.transpile.test_numeric_equivalence]
covers sample / observe / deterministic constructs but excludes
discrete-latent marginalisation. This file plugs that gap with two
self-contained fixtures whose marginal log-density sits in closed
form:

1. *Beta-Bernoulli mixture*: a 2-component Bernoulli mixture with a
   Dirichlet(1) mixing weight and per-component Beta priors. The
   `marginalize cls : Cls <- Categorical(probs=probs) [over=Obs]` form
   integrates out the per-observation class indicator analytically.
2. *Mixture of Normals*: a 2-component Normal mixture with
   per-component Normal priors on the location. Same marginalize
   shape, different observation family.

For each fixture, three log-density witnesses are constructed at the
same deterministic grid of (parameter, data) points:

* the closed-form analytic joint, built from the same
  [`torch.distributions`][torch.distributions] primitives the QVR
  trace accumulates (so the family-formula degree of freedom is
  eliminated);
* the QVR reference probe (in-process trace through the lowered
  [`MonadicProgram`][quivers.continuous.programs.MonadicProgram] that
  reduces the marginalize block via
  [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]);
* the Stan probe (in-container; the Stan renderer emits
  `log_sum_exp` per-group enumeration).

Pairwise constant-spread agreement is asserted between every
available pair. NumPyro / PyMC backends are skipped on these
fixtures: their renderers lower `marginalize` to a *sampled* discrete
latent and the in-container probes evaluate
[`log_density`][numpyro.infer.util.log_density] /
[`pymc.logp`][pymc.logp] at the clamped value, which is not the
marginalised joint QVR / Stan compute.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import pytest
import torch
import torch.distributions as D

from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.inference.trace import trace
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.probes._protocol import Point


def _observation_count_marg(points: list[Point]) -> int:
    """Sum of observed-data lengths across `points`; used to derive
    the per-fixture
    [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol].

    A marginalize fixture's observation count is the size of its
    ``y`` array (per point); the per-site `log_prob` round-off
    accumulates linearly across that array, so the tolerance scales
    with the count.
    """
    if not points:
        return 1
    pt = points[0]
    total = 0
    for value in pt.data.values():
        if isinstance(value, list):
            total += len(value)
        else:
            total += 1
    return max(total, 1)


_DOUBLE = torch.float64
"""Both QVR-probe and analytic evaluations use float64. The
constant-spread tolerance in
[`_equivalence._DEFAULT_ATOL`][tests.transpile._equivalence] is
5e-4; per-fixture spreads sit comfortably below that floor in
double precision when the model is in closed form."""


# ---------------------------------------------------------------------------
# Fixture sources. Both kept inline (rather than as `.qvr` files in the
# benchmark corpus) because they exercise marginalize-specific surface
# syntax that the inference benchmark suite does not consume.
# ---------------------------------------------------------------------------


_BETA_BERNOULLI_MIX_SOURCE = """object Cls : FinSet 2
object Obs : FinSet 30
program beta_bernoulli_mix : Obs -> Obs
    sample probs <- Dirichlet(concentration=1.0) [over=Cls]
    sample theta <- Dirichlet(concentration=2.0) [over=Cls]
    marginalize cls : Cls <- Categorical(probs=probs) [over=Obs, reduction=logsumexp]
        observe y : Obs <- Bernoulli(probs=theta[cls]) [via=idx]
    return probs
export beta_bernoulli_mix
"""


_NORMAL_MIX_SOURCE = """object Cls : FinSet 2
object Obs : FinSet 20
program normal_mix : Obs -> Obs
    sample probs <- Dirichlet(concentration=1.0) [over=Cls]
    sample mu_low <- Normal(loc=-2.0, scale=1.0)
    sample mu_diff <- HalfNormal(scale=1.0)
    let mu = factor c : Cls in mu_low + c * mu_diff
    let sigma = 0.5
    marginalize cls : Cls <- Categorical(probs=probs) [over=Obs, reduction=logsumexp]
        observe y : Obs <- Normal(loc=mu[cls], scale=sigma) [via=idx]
    return probs
export normal_mix
"""


_FIXTURE_SOURCES: dict[str, str] = {
    "beta_bernoulli_mix": _BETA_BERNOULLI_MIX_SOURCE,
    "normal_mix": _NORMAL_MIX_SOURCE,
}


# ---------------------------------------------------------------------------
# Closed-form joint log-densities.
#
# Each fixture's marginal joint factors as
#
#     log p(params, y, idx)
#       = log Dir(probs; 1)
#       + sum_k log Prior(theta_k or mu_k)
#       + sum_{g in groups} logsumexp_k
#               [ log probs[k]
#                 + sum_{n: idx[n] = g} log Obs(y_n; component_k) ]
#
# where ``idx`` is the per-observation index map into the marginalize
# block's grouping plate (here ``over=Obs`` so |G| = N and the
# identity index map sends each observation to its own group).
# ---------------------------------------------------------------------------


def _analytic_beta_bernoulli_mix(
    params: dict[str, float | list[float]],
    data: dict[str, list[float] | list[int]],
) -> float:
    probs = torch.tensor(params["probs"], dtype=_DOUBLE)
    theta = torch.tensor(params["theta"], dtype=_DOUBLE)

    y = torch.tensor(data["y"], dtype=_DOUBLE)
    idx = torch.tensor(data["idx"], dtype=torch.long)
    num_groups = int(idx.max().item()) + 1

    lp = D.Dirichlet(torch.ones(2, dtype=_DOUBLE)).log_prob(probs)
    lp = lp + D.Dirichlet(torch.full((2,), 2.0, dtype=_DOUBLE)).log_prob(theta)

    per_row_per_class = D.Bernoulli(probs=theta).log_prob(y.unsqueeze(-1))
    grouped = torch.zeros((num_groups, 2), dtype=_DOUBLE)
    grouped = grouped.index_add(0, idx, per_row_per_class)
    weighted = torch.log(probs).unsqueeze(0) + grouped
    lp = lp + torch.logsumexp(weighted, dim=-1).sum()
    return float(lp.item())


def _analytic_normal_mix(
    params: dict[str, float | list[float]],
    data: dict[str, list[float] | list[int]],
) -> float:
    probs = torch.tensor(params["probs"], dtype=_DOUBLE)
    mu_low = torch.tensor(params["mu_low"], dtype=_DOUBLE)
    mu_diff = torch.tensor(params["mu_diff"], dtype=_DOUBLE)
    mu = torch.stack([mu_low, mu_low + mu_diff])
    sigma = torch.tensor(0.5, dtype=_DOUBLE)

    y = torch.tensor(data["y"], dtype=_DOUBLE)
    idx = torch.tensor(data["idx"], dtype=torch.long)
    num_groups = int(idx.max().item()) + 1

    lp = D.Dirichlet(torch.ones(2, dtype=_DOUBLE)).log_prob(probs)
    lp = lp + D.Normal(
        torch.tensor(-2.0, dtype=_DOUBLE),
        torch.tensor(1.0, dtype=_DOUBLE),
    ).log_prob(mu_low)
    lp = lp + D.HalfNormal(torch.tensor(1.0, dtype=_DOUBLE)).log_prob(mu_diff)

    per_row_per_class = D.Normal(mu, sigma).log_prob(y.unsqueeze(-1))
    grouped = torch.zeros((num_groups, 2), dtype=_DOUBLE)
    grouped = grouped.index_add(0, idx, per_row_per_class)
    weighted = torch.log(probs).unsqueeze(0) + grouped
    lp = lp + torch.logsumexp(weighted, dim=-1).sum()
    return float(lp.item())


_AnalyticFn = Callable[
    [
        dict[str, float | list[float]],
        dict[str, list[float] | list[int]],
    ],
    float,
]


_ANALYTIC: dict[str, _AnalyticFn] = {
    "beta_bernoulli_mix": _analytic_beta_bernoulli_mix,
    "normal_mix": _analytic_normal_mix,
}


# ---------------------------------------------------------------------------
# Deterministic test points.
#
# Each fixture pins its observed data (`y`, `idx`) once; the
# parameter grid varies the latent (probs, theta_k or mu_k) so the
# constant-spread invariant is exercised across a non-trivial slice
# of parameter space. The grids are deliberately small (~6 points per
# fixture) so each Stan probe call stays inside the per-cell budget
# the harness allots.
# ---------------------------------------------------------------------------


def _beta_bernoulli_mix_points() -> list[Point]:
    n_obs = 30
    y = [1] * 15 + [0] * 15
    # Identity index map: each observation is its own group. This
    # exercises the per-row mixture form of `marginalize_grouped`
    # (`|G| = N`, scatter-add is a no-op, one logsumexp per row).
    # Every target reads the indices in QVR's zero-based origin; a
    # one-based host such as Stan rebases each subscript itself.
    idx_qvr = list(range(n_obs))
    # `theta` is a 2-simplex (sums to 1) so each `(theta[0],
    # theta[1])` pair is `(theta[0], 1 - theta[0])`; the two
    # mixture components are anti-correlated Bernoullis. The
    # mixture is still non-degenerate (the two component pmfs
    # differ at every `theta != 0.5`) and the marginal joint
    # exercises the same `marginalize_grouped` scatter-and-reduce
    # path as the independent-Beta parameterisation would.
    points: list[Point] = []
    grid = [
        ([0.5, 0.5], [0.3, 0.7]),
        ([0.5, 0.5], [0.2, 0.8]),
        ([0.3, 0.7], [0.4, 0.6]),
        ([0.7, 0.3], [0.2, 0.8]),
        ([0.4, 0.6], [0.5, 0.5]),
        ([0.6, 0.4], [0.35, 0.65]),
    ]
    for probs, theta in grid:
        points.append(
            Point(
                params={"probs": probs, "theta": theta},
                data={"y": y, "idx": idx_qvr},
            )
        )
    return points


def _normal_mix_points() -> list[Point]:
    n_obs = 20
    y = (
        [-2.0, -1.5, -2.5, -1.8, -2.2, -1.0, -2.7] * 2
        + [1.5, 2.0, 2.5, 1.8, 2.2, 1.0] * 1
    )[:n_obs]
    if len(y) < n_obs:
        y = y + [0.0] * (n_obs - len(y))
    idx_qvr = list(range(n_obs))
    # `mu_diff` is the separation between the two component
    # locations, constrained positive by the half-normal prior
    # (Stan's emit declares `real <lower = 0> mu_diff;`). The
    # closed-form analytic side uses
    # [`torch.distributions.HalfNormal`][torch.distributions.HalfNormal]
    # for the prior; Stan's `mu_diff ~ normal(0, 1)` plus the
    # lower-bound constraint with `jacobian=False` differs from the
    # half-normal density by the canonical `log 2` constant, which
    # is absorbed by the constant-spread tolerance.
    grid = [
        ([0.5, 0.5], -2.0, 4.0),
        ([0.3, 0.7], -1.5, 3.0),
        ([0.7, 0.3], -2.5, 5.0),
        ([0.4, 0.6], -1.0, 2.0),
        ([0.6, 0.4], -2.0, 4.0),
        ([0.5, 0.5], -1.0, 3.5),
    ]
    points: list[Point] = []
    for probs, m_low, m_diff in grid:
        points.append(
            Point(
                params={"probs": probs, "mu_low": m_low, "mu_diff": m_diff},
                data={"y": y, "idx": idx_qvr},
            )
        )
    return points


_PointsFn = Callable[[], list[Point]]


_POINTS: dict[str, _PointsFn] = {
    "beta_bernoulli_mix": _beta_bernoulli_mix_points,
    "normal_mix": _normal_mix_points,
}


_FIXTURES = tuple(_FIXTURE_SOURCES)


# Backends that natively handle `IRMarginalize`. NumPyro / PyMC are
# omitted from this set: their renderers lower the marginalize block
# to an *enumerated* discrete latent (`numpyro.sample("cls", ...)`),
# and the in-container probes' [`log_density`][numpyro.infer.util.log_density]
# / [`pymc.logp`][pymc.logp] call requires `cls` to be supplied as a
# value rather than integrated out, producing a different quantity
# than QVR's marginal joint.
_BACKENDS: dict[str, tuple[str, str, str]] = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
}


def _coerce(
    value: float | int | list[float] | list[int], *, integer: bool
) -> torch.Tensor:
    """Coerce a host-supplied value to the natural tensor shape for the QVR trace.

    Scalars land as 0-D tensors (so a `factor c : Cls in <expr>` body
    over a sample-site latent does not pick up a phantom trailing axis
    when the latent is scalar); lists land as 1-D tensors.
    Discrete-index-map inputs ride on the ``integer`` flag and are
    coerced to ``torch.long`` so
    [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]'s
    `index_add` sees a `LongTensor` group index.
    """
    if isinstance(value, list):
        return torch.tensor(
            [int(v) for v in value] if integer else [float(v) for v in value],
            dtype=torch.long if integer else _DOUBLE,
        )
    return torch.tensor(
        int(value) if integer else float(value),
        dtype=torch.long if integer else _DOUBLE,
    )


def _qvr_log_densities(
    source: str,
    fixture_name: str,
    points: list[Point],
    scratch: pathlib.Path,
) -> list[float]:
    """Run the QVR reference trace in-process under float64.

    Replaces [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] for
    marginalize fixtures: the shared probe lifts every scalar
    parameter to a 1-D tensor of shape ``(1,)``, but a sample-site
    scalar that feeds a `factor c : K in <body>` expression must
    arrive as a 0-D tensor so the resulting ``(K,)``-shaped factor
    tensor does not pick up a phantom trailing axis that
    [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]
    then rejects. The coercion is shape-aware (lists stay 1-D,
    scalars stay 0-D), and the integer-index-map ``idx`` rides on
    ``torch.long`` so `index_add` accepts it.
    """
    del scratch  # in-process trace; no scratch files needed.
    prior = torch.get_default_dtype()
    torch.set_default_dtype(_DOUBLE)
    try:
        module = parse(source)
        program = Compiler(module).compile()
        monadic = program._morphism
        if monadic is None:
            raise RuntimeError(
                f"qvr trace on {fixture_name!r}: lowered morphism is "
                "None (program likely declares scalar inputs; the "
                "marginalize numeric tier targets input-free fixtures)"
            )
        log_densities: list[float] = []
        for pt in points:
            obs_dict: dict[str, torch.Tensor] = {}
            for k, v in pt.params.items():
                obs_dict[k] = _coerce(v, integer=False)
            for k, v in pt.data.items():
                obs_dict[k] = _coerce(v, integer=(k == "idx"))
            x = torch.zeros(1, 1, dtype=_DOUBLE)
            tr = trace(monadic, x, observations=obs_dict)
            if tr.log_joint is None:
                raise RuntimeError(f"qvr trace on {fixture_name!r}: log_joint is None")
            log_densities.append(float(tr.log_joint.item()))
    finally:
        torch.set_default_dtype(prior)
    return log_densities


def _target_log_densities(
    backend: str,
    fixture_source: bytes,
    points: list[Point],
    scratch: pathlib.Path,
) -> list[float] | None:
    """Run the docker-backed probe for ``backend``; return ``None``
    when the image is absent so the caller can skip."""
    image, ext, script_name = _BACKENDS[backend]
    if not _docker.image_available(image):
        return None
    script_path = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    points_json = [{"params": pt.params, "data": pt.data} for pt in points]
    raw = _docker.run_probe(
        image=image,
        script=script_path,
        source=fixture_source,
        source_ext=ext,
        points=points_json,
        scratch=scratch / backend,
    )
    return [float(x) for x in raw["log_densities"]]


@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_marginalize_three_way_agreement(
    fixture_name: str,
    scratch: pathlib.Path,
) -> None:
    """For each fixture: QVR, Stan, and closed-form analytic agree on
    the marginal joint log-density up to a constant.

    Each pair is asserted independently so a failure pinpoints the
    leg in disagreement: a nonzero spread on (QVR vs analytic)
    isolates QVR's marginalize lowering; a nonzero spread on (Stan
    vs analytic) isolates the Stan renderer's `log_sum_exp`
    enumeration; a nonzero spread on (Stan vs QVR) isolates the
    transpile pipeline. The Stan leg is skipped (per-backend) when
    its Docker image is absent; the QVR / analytic leg runs
    unconditionally.
    """
    source = _FIXTURE_SOURCES[fixture_name]
    module = parse(source)

    qvr_points = _POINTS[fixture_name]()

    lp_analytic = [_ANALYTIC[fixture_name](pt.params, pt.data) for pt in qvr_points]
    lp_qvr = _qvr_log_densities(source, fixture_name, qvr_points, scratch)

    n_obs = _observation_count_marg(qvr_points)
    atol = _equivalence.adaptive_atol(n_obs=n_obs)
    _equivalence.assert_log_density_match(
        lp_qvr,
        lp_analytic,
        atol=atol,
        context=f"qvr@{fixture_name} vs analytic",
    )

    any_target_ran = False
    for backend in sorted(_BACKENDS):
        try:
            target_source = transpile(module, target=backend)
        except UnsupportedConstruct:
            continue
        lp_target = _target_log_densities(backend, target_source, qvr_points, scratch)
        if lp_target is None:
            continue
        any_target_ran = True
        _equivalence.assert_log_density_match(
            lp_target,
            lp_analytic,
            atol=atol,
            context=f"{backend}@{fixture_name} vs analytic",
        )
        _equivalence.assert_log_density_match(
            lp_target,
            lp_qvr,
            atol=atol,
            context=f"{backend}@{fixture_name} vs qvr",
        )

    if not any_target_ran:
        pytest.xfail(
            f"no marginalize-capable backend image available for "
            f"{fixture_name!r}; QVR vs analytic still passed"
        )


# ---------------------------------------------------------------------------
# Structural witness: the transpiled source must contain marginalize
# machinery, not a sampled-discrete-latent fallback. Without this the
# numeric agreement above could pass spuriously if the renderer
# silently lowered `marginalize` to a sample of the discrete latent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_stan_emits_log_sum_exp(fixture_name: str) -> None:
    """The Stan renderer must emit `log_sum_exp` enumeration for an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] node.

    This pins the *form* of the emit: a sampled-discrete-latent
    lowering would silently let the numeric tier pass by emitting a
    `categorical_rng`/sample-and-condition pair that
    [`Model.log_prob`][cmdstanpy.CmdStanModel.log_prob] reports the
    same joint for once the discrete value is fixed in the data
    block. We require the emit's `model {}` section to actually
    contain `log_sum_exp(...)`.
    """
    source = _FIXTURE_SOURCES[fixture_name]
    module = parse(source)
    emit = transpile(module, target="stan").decode("utf-8")
    if "log_sum_exp" not in emit:
        raise AssertionError(
            f"stan emit for {fixture_name!r} does not contain "
            f"`log_sum_exp` -- marginalize was not enumerated:\n{emit}"
        )
