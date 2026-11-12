"""Numeric tier for the `via` index map push-forward.

The companion `test_marginalize_numeric` file exercises
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] with an
identity index map (`|G| = N`, each observation in its own group).
This file specifically targets the non-identity case: an index map
:math:`g \\colon \\text{Obs} \\to \\text{Group}` re-indexes
observations into a strictly smaller grouping plate, and the
marginalize block's per-group accumulator must scatter-sum each
observation's per-class log-likelihood through `g` before the
log-sum-exp reduction. The shape under test is

```
marginalize cls : Cls <- Categorical(probs=probs) [over=Group]
    observe y : Obs <- Normal(loc=mu[cls], scale=sigma) [via=g]
```

with `|Obs| > |Group|` and `g[n]` not the identity. This stresses
the [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]
scatter-add (`grouped.index_add(0, g, ll)`) in QVR and the
corresponding `lps_cls[g[n_Obs], k] += <obs lpdf>` in the Stan emit.

Three log-density witnesses agree at every test point up to a
constant: the closed-form analytic joint, the QVR reference trace,
and the Stan probe (in-container). NumPyro / PyMC are excluded for
the same reason as in the marginalize tier: their renderers sample
the discrete latent rather than enumerate it, so their in-container
[`log_density`][numpyro.infer.util.log_density] / [`pymc.logp`][pymc.logp]
call requires a clamped `cls`, which is not the marginal joint Stan
and QVR compute.
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


def _observation_count_via(points: list[Point]) -> int:
    """Sum of observed-data lengths across `points`; used to derive
    the per-fixture
    [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol]."""
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


_SUBJECT_MIX_SOURCE = """object Cls : FinSet 2
object Group : FinSet 3
object Obs : FinSet 12
program subject_mix : Obs -> Obs
    sample probs <- Dirichlet(concentration=1.0) [over=Cls]
    sample mu_low <- Normal(loc=-2.0, scale=1.0)
    sample mu_diff <- HalfNormal(scale=1.0)
    let mu = factor c : Cls in mu_low + c * mu_diff
    let sigma = 0.5
    marginalize cls : Cls <- Categorical(probs=probs) [over=Group, reduction=logsumexp]
        observe y : Obs <- Normal(loc=mu[cls], scale=sigma) [via=g]
    return probs
export subject_mix
"""
"""Subject-level 2-component Normal mixture.

Twelve observations partitioned by a non-trivial index map `g :
Obs -> Group` into three groups of four. Each group has its own
mixture-class indicator `cls_g` (marginalised analytically); all
observations in group `g` share that group's class. The Stan
renderer emits the per-group scatter-and-reduce form

```
for (n in 1:N)
    for (k in 1:K)
        lps_cls[g[n], k] += normal_lpdf(y[n] | mu[k], sigma);
for (g in 1:G)
    target += log_sum_exp(lps_cls[g]);
```

which exactly mirrors
[`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]'s
`index_add` + `logsumexp` lowering on the QVR side.
"""


_FIXTURE_SOURCES: dict[str, str] = {
    "subject_mix": _SUBJECT_MIX_SOURCE,
}


# ---------------------------------------------------------------------------
# Closed-form joint log-density.
# ---------------------------------------------------------------------------


def _analytic_subject_mix(
    params: dict[str, float | list[float]],
    data: dict[str, list[float] | list[int]],
) -> float:
    """`log p(probs, mu_low, mu_diff, y, g)` for the subject-level mixture.

    Joint factorisation:

    .. math::

        \\log p
            = \\log \\mathrm{Dir}(\\text{probs}; 1)
            + \\log \\mathrm{Normal}(\\mu_{\\text{low}}; -2, 1)
            + \\log \\mathrm{HalfNormal}(\\mu_{\\text{diff}}; 1)
            + \\sum_{g} \\log \\sum_{k} \\big[
                \\text{probs}_k \\!\\cdot\\!
                \\prod_{n: g(n) = g}
                  \\mathrm{Normal}(y_n; \\mu_k, \\sigma)
              \\big],

    where :math:`\\mu = (\\mu_{\\text{low}}, \\mu_{\\text{low}} +
    \\mu_{\\text{diff}})` and :math:`\\sigma = 0.5`.
    """
    probs = torch.tensor(params["probs"], dtype=_DOUBLE)
    mu_low = torch.tensor(params["mu_low"], dtype=_DOUBLE)
    mu_diff = torch.tensor(params["mu_diff"], dtype=_DOUBLE)
    mu = torch.stack([mu_low, mu_low + mu_diff])
    sigma = torch.tensor(0.5, dtype=_DOUBLE)

    y = torch.tensor(data["y"], dtype=_DOUBLE)
    g_idx = torch.tensor(data["g"], dtype=torch.long)
    num_groups = int(g_idx.max().item()) + 1

    lp = D.Dirichlet(torch.ones(2, dtype=_DOUBLE)).log_prob(probs)
    lp = lp + D.Normal(
        torch.tensor(-2.0, dtype=_DOUBLE),
        torch.tensor(1.0, dtype=_DOUBLE),
    ).log_prob(mu_low)
    lp = lp + D.HalfNormal(torch.tensor(1.0, dtype=_DOUBLE)).log_prob(mu_diff)

    per_row_per_class = D.Normal(mu, sigma).log_prob(y.unsqueeze(-1))
    grouped = torch.zeros((num_groups, 2), dtype=_DOUBLE)
    grouped = grouped.index_add(0, g_idx, per_row_per_class)
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
    "subject_mix": _analytic_subject_mix,
}


# ---------------------------------------------------------------------------
# Deterministic test points.
# ---------------------------------------------------------------------------


def _subject_mix_points() -> list[Point]:
    """Six-point parameter grid with a fixed non-identity index map.

    The index map assigns four observations to each of three groups
    (`g = [0]*4 + [1]*4 + [2]*4`), so |Obs| = 12 strictly exceeds
    |Group| = 3 and the per-group scatter-add path is exercised in
    full. The within-group observation patterns are designed so the
    two mixture components are jointly identifiable: group 0 looks
    like component 0 (cluster of values near `mu_low`), group 1
    like component 1 (cluster near `mu_low + mu_diff`), group 2 is
    deliberately mixed so the per-group logsumexp does not collapse
    to a single class.
    """
    g_qvr = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    y = [
        -2.0,
        -1.8,
        -2.2,
        -1.5,
        1.5,
        1.0,
        2.0,
        1.8,
        -0.5,
        0.5,
        -1.0,
        1.0,
    ]
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
                data={"y": y, "g": g_qvr},
            )
        )
    return points


_PointsFn = Callable[[], list[Point]]


_POINTS: dict[str, _PointsFn] = {
    "subject_mix": _subject_mix_points,
}


_FIXTURES = tuple(_FIXTURE_SOURCES)


_BACKENDS: dict[str, tuple[str, str, str]] = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
}


# ---------------------------------------------------------------------------
# In-process QVR trace and Stan-side index shifting.
# ---------------------------------------------------------------------------


def _coerce(
    value: float | int | list[float] | list[int], *, integer: bool
) -> torch.Tensor:
    """Coerce a host-supplied value to the natural QVR tensor shape.

    Scalars land as 0-D tensors (so the
    `factor c : Cls in mu_low + c * mu_diff` body does not pick up
    a phantom trailing axis when `mu_low` / `mu_diff` are scalar);
    lists land as 1-D tensors. Integer index maps are coerced to
    `torch.long` so
    [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]'s
    `index_add` accepts the group-index argument.
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
) -> list[float]:
    """Run the QVR reference trace in-process under float64.

    The shared [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe]
    lifts every scalar parameter to a `(1,)`-shaped tensor; the
    factor expression in this fixture's `let mu = ...` requires
    its inputs to arrive as 0-D so the resulting `(|Cls|,)`-shaped
    mu vector does not pick up a phantom trailing axis. This
    helper does the shape-aware coercion locally.
    """
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
                "via-index-map numeric tier targets input-free fixtures)"
            )
        log_densities: list[float] = []
        for pt in points:
            obs_dict: dict[str, torch.Tensor] = {}
            for k, v in pt.params.items():
                obs_dict[k] = _coerce(v, integer=False)
            for k, v in pt.data.items():
                obs_dict[k] = _coerce(v, integer=(k == "g"))
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
def test_via_index_map_three_way_agreement(
    fixture_name: str,
    scratch: pathlib.Path,
) -> None:
    """QVR, Stan, and closed-form analytic agree on the joint
    log-density up to a constant for a non-identity `via` index map.

    A nonzero spread on (QVR vs analytic) isolates QVR's
    index map push-forward (the `index_add` lowering of
    [`marginalize_grouped`][quivers.continuous.plate.marginalize_grouped]);
    a nonzero spread on (Stan vs analytic) isolates the Stan
    renderer's `lps_cls[g[n_Obs], k] += <obs lpdf>` emit; a nonzero
    spread on (Stan vs QVR) isolates the transpile pipeline. The
    Stan leg is skipped per-backend when its Docker image is absent;
    the QVR / analytic leg runs unconditionally.
    """
    source = _FIXTURE_SOURCES[fixture_name]
    module = parse(source)

    qvr_points = _POINTS[fixture_name]()

    lp_analytic = [_ANALYTIC[fixture_name](pt.params, pt.data) for pt in qvr_points]
    lp_qvr = _qvr_log_densities(source, fixture_name, qvr_points)

    n_obs = _observation_count_via(qvr_points)
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
            f"no index-map-capable backend image available for "
            f"{fixture_name!r}; QVR vs analytic still passed"
        )


# ---------------------------------------------------------------------------
# Structural witness: the transpiled Stan source must scatter-add the
# per-row per-class log-likelihood through the supplied index map.
# Without this, a renderer that silently dropped the index map and
# scored each observation against a *fixed* group could still pass
# the numeric leg by coincidence for symmetric per-group data.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_stan_emit_scatter_through_index_map(fixture_name: str) -> None:
    """The Stan emit indexes `lps_cls` through the host-supplied
    index map variable, not through the row-plate index.

    Concretely the emit must contain a per-row update of the form
    `lps_cls[g[n_Obs], k] += <obs lpdf>(y[n_Obs] | mu[k], ...)`,
    where the leading index is the *index map* variable (`g`) and
    not the row index (`n_Obs`) itself. A renderer that dropped the
    index map and emitted `lps_cls[n_Obs, k] += ...` would silently
    convert the |Group|-shaped per-group accumulator into a
    |Obs|-shaped per-row one and yield a different marginal joint.
    """
    source = _FIXTURE_SOURCES[fixture_name]
    module = parse(source)
    emit = transpile(module, target="stan").decode("utf-8")
    needle = "lps_cls[g[n_Obs]"
    if needle not in emit:
        raise AssertionError(
            f"stan emit for {fixture_name!r} does not scatter through "
            f"the `g` index map (expected substring {needle!r}):\n{emit}"
        )
    # And the bound emit must still close with the per-group reduction.
    if "log_sum_exp(lps_cls[" not in emit:
        raise AssertionError(
            f"stan emit for {fixture_name!r} does not reduce the "
            f"per-group `lps_cls` accumulator with `log_sum_exp`:\n{emit}"
        )
