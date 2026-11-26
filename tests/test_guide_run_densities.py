"""Every guide scores each run of a batch alone.

An input of ``B`` rows is ``B`` independent runs, and `Guide.log_prob`
returns the ``B`` log densities of the guide's law, entry ``b`` the
density of run ``b``'s values alone. The tests check, for every guide
class, at ``B`` in ``{1, 2, 5}`` and at runs that all hold different
values:

1. The batch densities equal the densities of the runs scored one at a
   time, and equal a density written here by hand from torch
   primitives at the guide's current parameters, one run at a time.
2. The gradients of a weighted sum of the batch densities, with respect
   to the guide's parameters and to the site values, equal those of
   the same sum of the hand-written densities.
3. Values that the runs share (no run axis) enter every run's density
   in full, and a run axis of the wrong length is refused.
4. Mutations of the shared per-run operation are each rejected by the
   check of item 1: the first run repeated, the runs permuted, the
   bijector terms summed over the runs, the first run's bijector term
   given to every run, and a plate's terms read across runs.
5. Every concrete guide class in `quivers.inference` is covered, so a
   new guide cannot skip these checks.
"""

from __future__ import annotations

import importlib
import inspect
import math
import pkgutil
from collections.abc import Callable, Iterator

import didactic.api as dx
import pytest
import torch
import torch.nn.functional as F
from torch.distributions.transforms import StickBreakingTransform

import quivers.inference
from quivers._values import OPAQUE_BY_IDENTITY
from quivers.continuous import Euclidean, MonadicProgram
from quivers.continuous.families import ConditionalNormal
from quivers.dsl import loads
from quivers.inference import (
    AmortizedGuide,
    AutoDeltaGuide,
    AutoGuideList,
    AutoIAFGuide,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormalGuide,
    AutoMixtureGuide,
    AutoMultivariateNormalGuide,
    AutoNeuralSplineGuide,
    AutoNormalGuide,
    AutoNormalizingFlow,
    AutoStructured,
)
from quivers.inference.dais import AutoDAIS
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentSite
from quivers.inference.transforms import (
    MADE,
    BatchNormTransform,
    InverseAutoregressiveTransform,
    LULinearTransform,
    TransformModule,
)

# Tolerances of the float64 comparisons, fixed before the tests were run.
_RTOL = 1e-9
_ATOL = 1e-9

_RUN_COUNTS = (1, 2, 5)
_SUBJECTS = 3
_DATUM_WIDTH = 4


@pytest.fixture(autouse=True)
def _float64() -> Iterator[None]:
    """Run every test in double precision.

    Yields
    ------
    None
        Control, with float64 as the default dtype.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def _model() -> MonadicProgram:
    """A program with a bounded scalar, a simplex, and a plate of scales.

    Returns
    -------
    MonadicProgram
        ``theta`` on the unit interval, ``w`` on the 3-simplex, and
        ``tau``, three positive plate latents.
    """
    program = loads(
        f"object Subj : FinSet {_SUBJECTS}\n"
        "object Resp : FinSet 6\n"
        "program p : Resp -> Resp\n"
        "    sample theta <- Beta(concentration1=2.0, concentration0=3.0)\n"
        "    sample w <- Dirichlet(concentration=[1.0, 2.0, 3.0])\n"
        "    sample tau : Subj <- HalfNormal(scale=1.0)\n"
        "    observe r : Resp <- Bernoulli(probs=theta)\n"
        "    return theta\n"
        "export p\n"
    ).morphism
    assert isinstance(program, MonadicProgram)
    return program


def _responses() -> torch.Tensor:
    """The observed responses of one run.

    Returns
    -------
    torch.Tensor
        Six Bernoulli outcomes.
    """
    return torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])


def _tanh_shift(z: torch.Tensor) -> torch.Tensor:
    """A callable dependency from ``theta`` to the plate ``tau``.

    Parameters
    ----------
    z : torch.Tensor
        One draw of ``theta``'s flat coordinates, shape ``(1,)``.

    Returns
    -------
    torch.Tensor
        A shift of ``tau``'s flat location, shape ``(3,)``.
    """
    return torch.tanh(z) * torch.tensor([1.0, -0.5, 0.25])


def _encoder() -> ConditionalNormal:
    """A real-valued encoder from a datum row to one coordinate.

    Returns
    -------
    ConditionalNormal
        The encoder.
    """
    return ConditionalNormal(
        Euclidean(name="Y", dim=_DATUM_WIDTH), Euclidean(name="U", dim=1)
    )


# ---------------------------------------------------------------------------
# Runs, data, and their slices
# ---------------------------------------------------------------------------


class _Runs(dx.Model):
    """The values and data of ``B`` runs, every run different.

    Parameters
    ----------
    count : int
        The number of runs.
    sites : dict[str, torch.Tensor]
        Per-run values: ``theta`` ``(B,)``, ``w`` ``(B, 3)``, ``tau``
        ``(B, 3)``.
    data : dict[str, torch.Tensor]
        Per-run observations and data: ``r`` ``(6,)``, ``y`` ``(B, 4)``,
        ``u`` ``(B, 3, 4)``.
    """

    count: int
    sites: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    data: dict[str, torch.Tensor] = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def row(self, b: int) -> _Runs:
        """Run ``b`` alone, as a batch of one.

        Parameters
        ----------
        b : int
            The run.

        Returns
        -------
        _Runs
            The run's values and data with a run axis of one.
        """
        return _Runs(
            count=1,
            sites={name: value[b : b + 1] for name, value in self.sites.items()},
            data={
                "r": self.data["r"],
                "y": self.data["y"][b : b + 1],
                "u": self.data["u"][b : b + 1],
            },
        )

    def input(self) -> torch.Tensor:
        """The program input of the runs.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``.
        """
        return torch.zeros(self.count, 1)


def _runs(count: int, seed: int = 11) -> _Runs:
    """Draw ``count`` runs whose values all differ.

    Parameters
    ----------
    count : int
        The number of runs.
    seed : int
        The seed of the draw.

    Returns
    -------
    _Runs
        The runs.
    """
    generator = torch.Generator().manual_seed(seed + count)
    theta = torch.tensor([0.2, 0.7, 0.45, 0.9, 0.05])[:count]
    w = torch.softmax(torch.randn(count, 3, generator=generator), dim=-1)
    tau = torch.exp(torch.randn(count, _SUBJECTS, generator=generator))
    return _Runs(
        count=count,
        sites={"theta": theta, "w": w, "tau": tau},
        data={
            "r": _responses(),
            "y": torch.randn(count, _DATUM_WIDTH, generator=generator),
            "u": torch.randn(count, _SUBJECTS, _DATUM_WIDTH, generator=generator),
        },
    )


# ---------------------------------------------------------------------------
# The guides
# ---------------------------------------------------------------------------


def _perturbed(guide: Guide, seed: int) -> Guide:
    """Move every parameter of a guide off its initial value.

    Parameters
    ----------
    guide : Guide
        The guide.
    seed : int
        The seed of the move.

    Returns
    -------
    Guide
        The same guide.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in guide.parameters():
            parameter.add_(
                0.3 * torch.randn(parameter.shape, generator=generator).to(parameter)
            )
    return guide


def _fitted_laplace(model: MonadicProgram) -> AutoLaplaceApproximation:
    """A Laplace approximation with its Hessian fitted.

    Parameters
    ----------
    model : MonadicProgram
        The model.

    Returns
    -------
    AutoLaplaceApproximation
        The guide, in its Gaussian phase.
    """
    guide = AutoLaplaceApproximation(model, {"r"})
    _perturbed(guide, 3)
    guide.fit_hessian(model, torch.zeros(1, 1), {"r": _responses()})
    return guide


def _custom_flow(model: MonadicProgram) -> AutoNormalizingFlow:
    """A flow guide over a user-chosen stack of transforms.

    Parameters
    ----------
    model : MonadicProgram
        The model.

    Returns
    -------
    AutoNormalizingFlow
        The guide.
    """
    dimension = 1 + 2 + _SUBJECTS
    made = MADE(dim=dimension, n_per_dim=2, hidden=12, n_hidden_layers=1)
    stack: list[TransformModule] = [
        LULinearTransform(dimension),
        BatchNormTransform(dimension, momentum=0.5),
        InverseAutoregressiveTransform(made),
    ]
    guide = AutoNormalizingFlow(model, {"r"}, stack)
    _perturbed(guide, 8)
    guide.update_statistics(64)
    return guide


_BUILDERS: dict[str, Callable[[MonadicProgram, _Runs], Guide]] = {
    "normal": lambda m, runs: AutoNormalGuide(m, {"r"}),
    "delta": lambda m, runs: AutoDeltaGuide(m, {"r"}),
    "mvn": lambda m, runs: AutoMultivariateNormalGuide(m, {"r"}),
    "low-rank": lambda m, runs: AutoLowRankMultivariateNormalGuide(m, {"r"}, rank=2),
    "laplace-map": lambda m, runs: AutoLaplaceApproximation(m, {"r"}),
    "laplace": lambda m, runs: _fitted_laplace(m),
    "flow": lambda m, runs: _custom_flow(m),
    "iaf": lambda m, runs: AutoIAFGuide(m, {"r"}, num_flows=2),
    "spline": lambda m, runs: AutoNeuralSplineGuide(m, {"r"}, num_flows=2),
    "structured": lambda m, runs: AutoStructured(
        m,
        {"r"},
        conditionals={"theta": "mvn", "w": "normal", "tau": "mvn"},
        dependencies={
            "w": {"theta": "linear"},
            "tau": {"theta": _tanh_shift, "w": "linear"},
        },
    ),
    "structured-delta": lambda m, runs: AutoStructured(
        m,
        {"r"},
        conditionals={"theta": "delta", "w": "normal", "tau": "normal"},
        dependencies={"tau": {"w": "linear"}},
    ),
    "mixture": lambda m, runs: AutoMixtureGuide(
        [
            _perturbed(AutoNormalGuide(m, {"r"}), 5),
            _perturbed(AutoMultivariateNormalGuide(m, {"r"}), 6),
            _perturbed(AutoIAFGuide(m, {"r"}, num_flows=1), 7),
        ]
    ),
    "list": lambda m, runs: AutoGuideList(
        {
            "global": AutoMultivariateNormalGuide(m, {"r", "tau"}),
            "local": AutoNormalGuide(m, {"r", "theta", "w"}),
        }
    ),
    "amortized": lambda m, runs: AmortizedGuide(
        m,
        {"r"},
        {"theta": (_encoder(), "y"), "tau": (_encoder(), "u")},
        observations=dict(runs.data),
    ),
}

# Guides whose density does not depend on the bijector terms, so a
# mutation of those terms leaves it unchanged.
_CONSTANT_DENSITY = frozenset({"delta", "laplace-map"})


def _guide(kind: str, runs: _Runs) -> Guide:
    """Build and perturb one guide for the runs.

    Parameters
    ----------
    kind : str
        A key of `_BUILDERS`.
    runs : _Runs
        The runs, whose data an amortized guide observes.

    Returns
    -------
    Guide
        The guide.
    """
    torch.manual_seed(1)
    guide = _BUILDERS[kind](_model(), runs)
    if kind != "laplace":
        _perturbed(guide, 2)
    return guide


# ---------------------------------------------------------------------------
# Hand-written densities, one run at a time
# ---------------------------------------------------------------------------


def _unconstrain(name: str, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert one run's value of a site by hand.

    Parameters
    ----------
    name : str
        The site.
    value : torch.Tensor
        One run's value: ``()`` for ``theta``, ``(3,)`` for ``w`` and
        ``tau``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The flat coordinates of the value, and the log absolute
        Jacobian determinant of the inverse map at it, one entry per
        plate element (one entry for a site that is not plated).
    """
    if name == "theta":
        z = torch.log(value) - torch.log1p(-value)
        return z.reshape(1), (-torch.log(value) - torch.log1p(-value)).reshape(1)
    if name == "w":
        inverse = StickBreakingTransform().inv
        z = inverse(value)
        assert isinstance(z, torch.Tensor)
        return z, inverse.log_abs_det_jacobian(value, z).reshape(1)
    return torch.log(value), -torch.log(value)


def _coordinates(
    run: dict[str, torch.Tensor], names: tuple[str, ...]
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One run's flat coordinates per site and its total Jacobian term.

    Parameters
    ----------
    run : dict[str, torch.Tensor]
        One run's values, without a run axis.
    names : tuple[str, ...]
        The sites, in declaration order.

    Returns
    -------
    tuple[dict[str, torch.Tensor], torch.Tensor]
        The coordinates by site and the summed log Jacobian term.
    """
    coordinates: dict[str, torch.Tensor] = {}
    total = torch.zeros(())
    for name in names:
        coordinates[name], log_det = _unconstrain(name, run[name])
        total = total + log_det.sum()
    return coordinates, total


def _gaussian(z: torch.Tensor, loc: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """The log density of a Gaussian, written out.

    Parameters
    ----------
    z : torch.Tensor
        The point, shape ``(D,)``.
    loc : torch.Tensor
        The mean, shape ``(D,)``.
    cov : torch.Tensor
        The covariance, shape ``(D, D)``.

    Returns
    -------
    torch.Tensor
        :math:`-\\tfrac12 (D \\log 2\\pi + \\log|\\Sigma| + r^\\top
        \\Sigma^{-1} r)` with :math:`r = z - \\mu`.
    """
    chol = torch.linalg.cholesky(cov)
    solved = torch.linalg.solve_triangular(
        chol, (z - loc).unsqueeze(-1), upper=False
    ).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return -0.5 * (z.shape[0] * math.log(2 * math.pi) + log_det + (solved**2).sum())


def _normal(z: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The log density of independent Normals, written out and summed.

    Parameters
    ----------
    z, loc, scale : torch.Tensor
        Points, means, and scales of one shape.

    Returns
    -------
    torch.Tensor
        The summed log density.
    """
    return (
        -0.5 * ((z - loc) / scale) ** 2 - torch.log(scale) - 0.5 * math.log(2 * math.pi)
    ).sum()


def _cholesky(off: torch.Tensor, diag_raw: torch.Tensor) -> torch.Tensor:
    """A Cholesky factor from raw parameters, as the MVN guides build it.

    Parameters
    ----------
    off : torch.Tensor
        The raw square matrix whose strict lower triangle is kept.
    diag_raw : torch.Tensor
        The raw diagonal, mapped through softplus plus ``1e-6``.

    Returns
    -------
    torch.Tensor
        The lower-triangular factor.
    """
    return torch.tril(off, diagonal=-1) + torch.diag(F.softplus(diag_raw) + 1e-6)


def _oracle(guide: Guide, run: dict[str, torch.Tensor], data: _Runs) -> torch.Tensor:
    """The log density of one run under a guide, written by hand.

    Parameters
    ----------
    guide : Guide
        The guide, at its current parameters.
    run : dict[str, torch.Tensor]
        One run's values, without a run axis.
    data : _Runs
        The run alone, whose data an amortized guide reads.

    Returns
    -------
    torch.Tensor
        A scalar log density.
    """
    return _ORACLES[type(guide)](guide, run, data)


def _parameter(guide: Guide, name: str) -> torch.Tensor:
    """A guide's parameter or buffer by its dotted name.

    Parameters
    ----------
    guide : Guide
        The guide.
    name : str
        The name.

    Returns
    -------
    torch.Tensor
        The tensor, still in the guide's graph.
    """
    if name in dict(guide.named_buffers()):
        return guide.get_buffer(name)
    return guide.get_parameter(name)


def _oracle_normal(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """Independent Normals over every site's coordinates.

    Parameters
    ----------
    guide : Guide
        An `AutoNormalGuide`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del data
    names = tuple(guide.latent_names)
    coordinates, total = _coordinates(run, names)
    for name in names:
        loc = _parameter(guide, f"loc_{name}").reshape(-1)
        log_scale = _parameter(guide, f"log_scale_{name}").reshape(-1)
        scale = log_scale.exp().clamp(min=1e-6)
        total = total + _normal(coordinates[name].reshape(-1), loc, scale)
    return total


def _oracle_zero(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The density a point-mass guide reports by convention: zero.

    Parameters
    ----------
    guide : Guide
        An `AutoDeltaGuide`.
    run : dict[str, torch.Tensor]
        Unused.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        Zero.
    """
    del guide, run, data
    return torch.zeros(())


def _flat(
    guide: Guide, run: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """One run's flat point over a guide's sites and its Jacobian term.

    Parameters
    ----------
    guide : Guide
        The guide.
    run : dict[str, torch.Tensor]
        One run's values.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The flat point and the summed log Jacobian term.
    """
    names = tuple(guide.latent_names)
    coordinates, total = _coordinates(run, names)
    return torch.cat([coordinates[name].reshape(-1) for name in names]), total


def _oracle_mvn(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """A full-covariance Gaussian over the flat point.

    Parameters
    ----------
    guide : Guide
        An `AutoMultivariateNormalGuide`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del data
    z, total = _flat(guide, run)
    chol = _cholesky(
        _parameter(guide, "scale_offdiag"), _parameter(guide, "scale_diag_raw")
    )
    return total + _gaussian(z, _parameter(guide, "loc"), chol @ chol.T)


def _oracle_low_rank(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """A Gaussian with low-rank-plus-diagonal covariance, made dense.

    Parameters
    ----------
    guide : Guide
        An `AutoLowRankMultivariateNormalGuide`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del data
    z, total = _flat(guide, run)
    factor = _parameter(guide, "cov_factor")
    diagonal = F.softplus(_parameter(guide, "cov_diag_raw")) + 1e-6
    cov = factor @ factor.T + torch.diag(diagonal)
    return total + _gaussian(z, _parameter(guide, "loc"), cov)


def _oracle_laplace(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The Laplace Gaussian once fitted, zero before.

    Parameters
    ----------
    guide : Guide
        An `AutoLaplaceApproximation`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del data
    if not bool(_parameter(guide, "_hessian_fitted")):
        return torch.zeros(())
    z, total = _flat(guide, run)
    chol = _parameter(guide, "_scale_tril")
    return total + _gaussian(z, _parameter(guide, "map_z"), chol @ chol.T)


def _oracle_flow(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The flow's density, pulling one run's point back layer by layer.

    Parameters
    ----------
    guide : Guide
        An `AutoNormalizingFlow` or a subclass.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del data
    z, total = _flat(guide, run)
    layers = [
        layer
        for layer in guide.get_submodule("flow").children()
        if isinstance(layer, TransformModule)
    ]
    flow_log_det = torch.zeros(())
    for layer in reversed(layers):
        previous = layer.inv(z)
        assert isinstance(previous, torch.Tensor)
        flow_log_det = flow_log_det + layer.log_abs_det_jacobian(previous, z)
        z = previous
    base = _normal(z, torch.zeros_like(z), torch.ones_like(z))
    return total + base - flow_log_det


def _oracle_structured(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The structured guide's directed law over its sites.

    Parameters
    ----------
    guide : Guide
        An `AutoStructured`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        Unused.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    del data
    assert isinstance(guide, AutoStructured)
    names = tuple(guide.latent_names)
    coordinates, total = _coordinates(run, names)
    flats = {name: coordinates[name].reshape(-1) for name in names}
    joint_z: list[torch.Tensor] = []
    joint_loc: list[torch.Tensor] = []
    for name in names:
        kind = guide.conditionals[name]
        if kind == "delta":
            continue
        loc = _parameter(guide, f"loc_{name}").reshape(-1)
        for parent, dependency in guide.dependencies.get(name, {}).items():
            if isinstance(dependency, str):
                weight = _parameter(guide, f"deps.{name}.{parent}.weight")
                loc = loc + weight @ flats[parent]
            else:
                loc = loc + dependency(flats[parent])
        if kind == "normal":
            log_scale = _parameter(guide, f"log_scale_{name}").reshape(-1)
            total = total + _normal(flats[name], loc, log_scale.exp().clamp(min=1e-6))
        else:
            joint_z.append(flats[name])
            joint_loc.append(loc)
    if joint_z:
        chol = _cholesky(
            _parameter(guide, "mvn_scale_offdiag"),
            _parameter(guide, "mvn_scale_diag_raw"),
        )
        total = total + _gaussian(
            torch.cat(joint_z), torch.cat(joint_loc), chol @ chol.T
        )
    return total


def _components(guide: AutoMixtureGuide) -> list[Guide]:
    """The components of a mixture guide.

    Parameters
    ----------
    guide : AutoMixtureGuide
        The mixture.

    Returns
    -------
    list[Guide]
        The components, in order.
    """
    return [
        component
        for component in guide.get_submodule("components").children()
        if isinstance(component, Guide)
    ]


def _oracle_mixture(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The mixture of the components' hand-written densities.

    Parameters
    ----------
    guide : Guide
        An `AutoMixtureGuide`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        The run alone.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    assert isinstance(guide, AutoMixtureGuide)
    log_pi = torch.log_softmax(_parameter(guide, "mixture_logits"), dim=-1)
    components = torch.stack([_oracle(c, run, data) for c in _components(guide)])
    return torch.logsumexp(log_pi + components, dim=0)


def _oracle_list(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The sum of the parts' hand-written densities.

    Parameters
    ----------
    guide : Guide
        An `AutoGuideList`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        The run alone.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    assert isinstance(guide, AutoGuideList)
    total = torch.zeros(())
    for label in guide.part_labels:
        total = total + _oracle(guide.part(label), run, data)
    return total


def _oracle_amortized(
    guide: Guide, run: dict[str, torch.Tensor], data: _Runs
) -> torch.Tensor:
    """The encoders' densities at the run's own data rows.

    Parameters
    ----------
    guide : Guide
        The `AmortizedGuide` of `_BUILDERS`.
    run : dict[str, torch.Tensor]
        One run's values.
    data : _Runs
        The run alone, whose ``y`` and ``u`` the encoders read.

    Returns
    -------
    torch.Tensor
        The log density.
    """
    theta_encoder, tau_encoder = guide.get_submodule("encoders").children()
    assert isinstance(theta_encoder, ConditionalNormal)
    assert isinstance(tau_encoder, ConditionalNormal)
    total = torch.zeros(())
    mean, scale = theta_encoder._get_params(data.data["y"])
    z, log_det = _unconstrain("theta", run["theta"])
    total = total + _normal(z, mean[0], scale[0]) + log_det.sum()
    mean, scale = tau_encoder._get_params(data.data["u"][0])
    z, log_det = _unconstrain("tau", run["tau"])
    return total + _normal(z, mean[:, 0], scale[:, 0]) + log_det.sum()


_ORACLES: dict[
    type[Guide], Callable[[Guide, dict[str, torch.Tensor], _Runs], torch.Tensor]
] = {
    AutoNormalGuide: _oracle_normal,
    AutoDeltaGuide: _oracle_zero,
    AutoMultivariateNormalGuide: _oracle_mvn,
    AutoLowRankMultivariateNormalGuide: _oracle_low_rank,
    AutoLaplaceApproximation: _oracle_laplace,
    AutoNormalizingFlow: _oracle_flow,
    AutoIAFGuide: _oracle_flow,
    AutoNeuralSplineGuide: _oracle_flow,
    AutoStructured: _oracle_structured,
    AutoMixtureGuide: _oracle_mixture,
    AutoGuideList: _oracle_list,
    AmortizedGuide: _oracle_amortized,
}


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


def _gradients(
    total: torch.Tensor, leaves: list[torch.Tensor]
) -> tuple[torch.Tensor | None, ...]:
    """The gradients of a scalar with respect to leaves.

    Parameters
    ----------
    total : torch.Tensor
        The scalar.
    leaves : list[torch.Tensor]
        The leaves.

    Returns
    -------
    tuple[torch.Tensor | None, ...]
        One gradient per leaf, ``None`` for a leaf the scalar does not
        depend on, and for every leaf of a constant scalar.
    """
    if not total.requires_grad:
        return tuple(None for _ in leaves)
    return torch.autograd.grad(total, leaves, allow_unused=True)


def _check_runs(guide: Guide, runs: _Runs) -> None:
    """Check a guide's batch densities run by run.

    The batch densities must equal the densities of the runs scored one
    at a time and the hand-written densities, and the gradients of a
    weighted sum of them, with respect to the parameters and the site
    values, must equal those of the hand-written sum.

    Parameters
    ----------
    guide : Guide
        The guide.
    runs : _Runs
        The runs, every run's values different.

    Raises
    ------
    AssertionError
        If a density or gradient differs.
    """
    sites = {
        name: value.clone().requires_grad_(True) for name, value in runs.sites.items()
    }
    guide.observe(runs.data)
    batch = guide.log_prob(runs.input(), sites)
    assert batch.shape == (runs.count,)
    singles = []
    oracles = []
    for b in range(runs.count):
        alone = runs.row(b)
        guide.observe(alone.data)
        singles.append(guide.log_prob(alone.input(), alone.sites)[0])
        oracles.append(
            _oracle(guide, {name: value[b] for name, value in sites.items()}, alone)
        )
    torch.testing.assert_close(batch, torch.stack(singles), rtol=_RTOL, atol=_ATOL)
    expected = torch.stack(oracles)
    torch.testing.assert_close(batch, expected, rtol=_RTOL, atol=_ATOL)

    weights = torch.arange(1, runs.count + 1, dtype=batch.dtype)
    parameters = [p for p in guide.parameters() if p.requires_grad]
    leaves = [*parameters, *sites.values()]
    guide.observe(runs.data)
    found = _gradients((weights * guide.log_prob(runs.input(), sites)).sum(), leaves)
    reference = _gradients((weights * expected).sum(), leaves)
    for leaf, got, want in zip(leaves, found, reference, strict=True):
        torch.testing.assert_close(
            torch.zeros_like(leaf) if got is None else got,
            torch.zeros_like(leaf) if want is None else want,
            rtol=_RTOL,
            atol=_ATOL,
        )


# ---------------------------------------------------------------------------
# Every guide, every run count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", _RUN_COUNTS)
@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_batch_densities_are_the_densities_of_each_run_alone(
    kind: str, count: int
) -> None:
    runs = _runs(count)
    _check_runs(_guide(kind, runs), runs)


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_a_shared_value_enters_every_run_in_full(kind: str) -> None:
    runs = _runs(4)
    guide = _guide(kind, runs)
    guide.observe(runs.data)
    shared = {
        "theta": runs.sites["theta"][1],
        "w": runs.sites["w"],
        "tau": runs.sites["tau"][2],
    }
    batch = guide.log_prob(runs.input(), shared)
    expected = []
    for b in range(runs.count):
        alone = runs.row(b)
        guide.observe(alone.data)
        expected.append(
            guide.log_prob(
                alone.input(),
                {
                    "theta": shared["theta"].reshape(1),
                    "w": alone.sites["w"],
                    "tau": shared["tau"].reshape(1, _SUBJECTS),
                },
            )[0]
        )
    torch.testing.assert_close(batch, torch.stack(expected), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_a_run_axis_of_another_length_is_refused(kind: str) -> None:
    runs = _runs(2)
    guide = _guide(kind, runs)
    guide.observe(runs.data)
    for theta in (torch.tensor([0.3]), torch.tensor([0.3, 0.4, 0.5])):
        with pytest.raises(ValueError, match="run axis of"):
            guide.log_prob(runs.input(), {**runs.sites, "theta": theta})


@pytest.mark.parametrize("count", _RUN_COUNTS)
def test_dais_refuses_a_density_at_given_values(count: int) -> None:
    model = _model()
    runs = _runs(count)
    guide = AutoDAIS(
        AutoNormalGuide(model, {"r"}), model, {"r": _responses()}, num_steps=1
    )
    with pytest.raises(ValueError, match="no density at arbitrary sites"):
        guide.log_prob(runs.input(), runs.sites)
    (draw,) = guide.rsample_strata(runs.input())
    assert draw.log_density.shape == (count,)
    assert bool((draw.log_density == draw.log_density[0]).all())


@pytest.mark.parametrize("count", (2, 5))
def test_mixture_strata_score_a_mean_field_component_run_by_run(count: int) -> None:
    """The strata of a mixture carry each run's own mixture density.

    The mean-field component draws every run separately, so the joint
    components score values that differ across the runs.
    """
    model = _model()
    torch.manual_seed(4)
    guide = AutoMixtureGuide(
        [
            _perturbed(AutoNormalGuide(model, {"r"}), 1),
            _perturbed(AutoMultivariateNormalGuide(model, {"r"}), 2),
            _perturbed(AutoStructured(model, {"r"}, conditionals="mvn"), 3),
        ]
    )
    x = torch.zeros(count, 1)
    draws = guide.rsample_strata(x)
    assert len({float(v) for v in draws[0].sites["theta"].detach()}) == count
    for draw in draws:
        for b in range(count):
            alone = {
                name: (value[b : b + 1] if name != "tau" else value)
                for name, value in draw.sites.items()
            }
            single = guide.log_prob(torch.zeros(1, 1), alone)[0]
            torch.testing.assert_close(
                draw.log_density[b], single, rtol=_RTOL, atol=_ATOL
            )
            for component in _components(guide):
                torch.testing.assert_close(
                    component.log_prob(x, draw.sites)[b],
                    component.log_prob(torch.zeros(1, 1), alone)[0],
                    rtol=_RTOL,
                    atol=_ATOL,
                )


# ---------------------------------------------------------------------------
# Mutations of the shared per-run operation
# ---------------------------------------------------------------------------

_ALIGN = LatentSite.align_runs
_UNCONSTRAIN = LatentSite.unconstrain_runs


def _first_run_repeated(
    site: LatentSite, value: torch.Tensor, runs: int
) -> torch.Tensor:
    return _ALIGN(site, value, runs)[:1].expand(runs, *site.constrained_shape)


def _runs_permuted(site: LatentSite, value: torch.Tensor, runs: int) -> torch.Tensor:
    return _ALIGN(site, value, runs).roll(1, dims=0)


def _log_det_summed(
    site: LatentSite, value: torch.Tensor, runs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    z, log_det = _UNCONSTRAIN(site, value, runs)
    return z, log_det.sum(0, keepdim=True).expand_as(log_det)


def _log_det_of_first_run(
    site: LatentSite, value: torch.Tensor, runs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    z, log_det = _UNCONSTRAIN(site, value, runs)
    return z, log_det[:1].expand_as(log_det)


def _log_det_across_runs(
    site: LatentSite, value: torch.Tensor, runs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    z, log_det = _UNCONSTRAIN(site, value, runs)
    if log_det.dim() == 2:
        log_det = log_det.t().reshape(log_det.shape)
    return z, log_det


_MUTANTS: dict[
    str, tuple[str, Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]]
] = {
    "first run repeated": ("align_runs", _first_run_repeated),
    "runs permuted": ("align_runs", _runs_permuted),
    "bijector terms summed over runs": ("unconstrain_runs", _log_det_summed),
    "first run's bijector term to every run": (
        "unconstrain_runs",
        _log_det_of_first_run,
    ),
    "plate terms read across runs": ("unconstrain_runs", _log_det_across_runs),
}


@pytest.mark.parametrize("count", (2, 5))
@pytest.mark.parametrize("mutant", sorted(_MUTANTS))
@pytest.mark.parametrize("kind", sorted(set(_BUILDERS) - _CONSTANT_DENSITY))
def test_the_check_rejects_each_mutation(
    kind: str, mutant: str, count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = _runs(count)
    guide = _guide(kind, runs)
    _check_runs(guide, runs)
    method, replacement = _MUTANTS[mutant]
    monkeypatch.setattr(LatentSite, method, replacement)
    with pytest.raises(AssertionError):
        _check_runs(guide, runs)


def _batch_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The moments of a batch over its leading axes.

    Parameters
    ----------
    x : torch.Tensor
        The batch, shape ``(..., dim)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean and biased variance over every axis but the last; over
        every axis for a single point.
    """
    axes = tuple(range(x.dim() - 1)) or tuple(range(x.dim()))
    return x.mean(dim=axes), x.var(dim=axes, unbiased=False)


def _batch_log_det(
    layer: BatchNormTransform, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    del y
    _, var = _batch_moments(x)
    per_coord = layer._gamma - 0.5 * torch.log(var + 1e-5)
    return per_coord.sum().expand(x.shape[:-1])


def _batch_inverse(layer: BatchNormTransform, y: torch.Tensor) -> torch.Tensor:
    mean, var = _batch_moments(y)
    return (y - layer._beta) * (-layer._gamma).exp() * torch.sqrt(var + 1e-5) + mean


@pytest.mark.parametrize("count", (2, 5))
@pytest.mark.parametrize(
    ("method", "replacement"),
    [("log_abs_det_jacobian", _batch_log_det), ("_inverse", _batch_inverse)],
)
def test_the_check_rejects_batch_statistics_in_a_flow_density(
    method: str,
    replacement: Callable[..., torch.Tensor],
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _runs(count)
    guide = _guide("flow", runs)
    _check_runs(guide, runs)
    monkeypatch.setattr(BatchNormTransform, method, replacement)
    with pytest.raises(AssertionError):
        _check_runs(guide, runs)


def test_flow_statistics_update_from_the_inputs_each_layer_sees() -> None:
    """The flow folds each layer's inputs into its moments, in order.

    The density does not change with training mode, and only an
    explicit update moves the moments.
    """
    runs = _runs(3)
    guide = _guide("flow", runs)
    assert isinstance(guide, AutoNormalizingFlow)
    before = guide.log_prob(runs.input(), runs.sites)
    guide.train(False)
    torch.testing.assert_close(
        guide.log_prob(runs.input(), runs.sites), before, rtol=0, atol=0
    )
    guide.train(True)
    linear, norm, _ = guide.get_submodule("flow").children()
    assert isinstance(linear, LULinearTransform)
    assert isinstance(norm, BatchNormTransform)
    mean = norm.get_buffer("_running_mean").clone()
    var = norm.get_buffer("_running_var").clone()
    torch.manual_seed(21)
    base = torch.randn(32, 1 + 2 + _SUBJECTS)
    with torch.no_grad():
        inputs = linear(base)
    torch.manual_seed(21)
    guide.update_statistics(32)
    torch.testing.assert_close(
        norm.get_buffer("_running_mean"),
        0.5 * mean + 0.5 * inputs.mean(0),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        norm.get_buffer("_running_var"),
        0.5 * var + 0.5 * inputs.var(0, unbiased=False),
        rtol=1e-12,
        atol=1e-12,
    )
    assert not torch.allclose(guide.log_prob(runs.input(), runs.sites), before)
    with pytest.raises(ValueError, match="num_draws"):
        guide.update_statistics(1)


# ---------------------------------------------------------------------------
# Coverage of every guide class
# ---------------------------------------------------------------------------


def _guide_classes() -> set[type[Guide]]:
    """Every concrete guide class the inference package defines.

    Returns
    -------
    set[type[Guide]]
        The classes, after importing every module of the package.
    """
    for module in pkgutil.walk_packages(
        quivers.inference.__path__, prefix="quivers.inference."
    ):
        importlib.import_module(module.name)
    found: set[type[Guide]] = set()
    pending: list[type[Guide]] = [Guide]
    while pending:
        cls = pending.pop()
        for sub in cls.__subclasses__():
            pending.append(sub)
            if sub.__module__.startswith("quivers.") and not inspect.isabstract(sub):
                found.add(sub)
    return found


def test_every_guide_class_is_checked() -> None:
    classes = _guide_classes()
    built = {type(_guide(kind, _runs(1))) for kind in _BUILDERS}
    assert classes - {AutoDAIS} == built
    assert built == set(_ORACLES)
