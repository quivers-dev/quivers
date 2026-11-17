"""brms ``brm`` sampler and fitting controls over the shared inference drivers.

brms passes ``chains``, ``iter``, ``warmup``, ``thin``, ``init``,
``init_r``, ``control`` and ``algorithm`` to Stan. Their counterparts here
configure the existing `quivers.inference.MCMC` driver, its NUTS and HMC
kernels, and the SVI guides:

* ``control = list(adapt_delta, max_treedepth, stepsize, metric)`` becomes a
  `SamplerControl` setting the kernel's target acceptance, maximum tree
  depth, initial step size and mass matrix;
* ``thin`` keeps every ``thin``-th post-warmup transition;
* ``init = "random"`` (with ``init_r``), ``init = 0``, a list of per-chain
  values, or a function of the chain index select the driver's uniform,
  zero, or named starting values;
* ``algorithm`` selects NUTS (``"sampling"``), a mean-field or full-rank
  Gaussian guide (``"meanfield"``, ``"fullrank"``), a Laplace approximation
  (``"laplace"``), the fixed-parameter sampler (``"fixed_param"``), or
  Pathfinder (``"pathfinder"``), whose CmdStan arguments `PathfinderOptions`
  holds and whose path count is brms's ``chains``;
* ``file`` and ``file_refit`` cache a fit through the artifact store.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import didactic.api as dx
import narwhals as nw
from narwhals.typing import IntoDataFrame
import torch
from torch import nn

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.inference import (
    AutoLaplaceApproximation,
    AutoMultivariateNormalGuide,
    AutoNormalGuide,
    HMCKernel,
    NUTSKernel,
)
from quivers.inference.mcmc.driver import InitStrategy
from quivers.inference.mcmc.hmc import MassMatrixKind
from quivers.inference.mcmc.kernel import KernelState, MCMCKernel, PotentialFn
from quivers.inference.pathfinder import LBFGSSettings, Pathfinder
from quivers.inference.mcmc.parallel import ChainWorker
from quivers.inference.registry import LatentRegistry
import quivers.modeling.artifact as artifact_store

type Algorithm = Literal[
    "sampling", "meanfield", "fullrank", "laplace", "fixed_param", "pathfinder"
]
"""brms ``algorithm`` values."""

type InitValues = Mapping[str, torch.Tensor | float]
"""Natural-scale starting values keyed by latent site."""

type InitSpec = (
    Literal["random", "0", "jitter", "prior"]
    | float
    | InitValues
    | Sequence[InitValues]
    | Callable[[int], InitValues]
)
"""brms ``init``: a strategy name, ``0``, shared values, per-chain values,
or a function of the one-based chain index returning values."""

type FileRefit = Literal["never", "always", "on_change"]
"""brms ``file_refit``."""

type FitMethod = Literal["nuts", "hmc", "svi", "fixed_param", "pathfinder"]
"""The inference methods `fit` runs."""

type FitOption = (
    str
    | int
    | float
    | bool
    | None
    | Path
    | type
    | torch.Tensor
    | ValueModel
    | nn.Module
    | IntoDataFrame
    | Mapping[str, FitOption]
    | Sequence[FitOption]
    | Callable[[int], InitValues]
)
"""One canonical `fit` argument, as a refit recipe retains it."""


class SamplerControl(ValueModel):
    """brms ``control``: NUTS adaptation and trajectory settings.

    Parameters
    ----------
    adapt_delta : float
        Target mean acceptance statistic of step-size adaptation, in (0, 1).
    max_treedepth : int
        Maximum NUTS tree depth (at most ``2**max_treedepth`` leapfrog steps).
    stepsize : float or None
        Initial step size before adaptation; None keeps the kernel default.
    metric : {"diag_e", "dense_e", "unit_e"}
        Diagonal, dense, or unit Euclidean metric (mass matrix).
    """

    adapt_delta: float = 0.8
    max_treedepth: int = 10
    stepsize: float | None = None
    metric: Literal["diag_e", "dense_e", "unit_e"] = "diag_e"

    @dx.model_validator()
    def _validate(self) -> None:
        if not 0 < self.adapt_delta < 1:
            raise ValueError("control adapt_delta must lie strictly between 0 and 1")
        if (
            isinstance(self.max_treedepth, bool)
            or not isinstance(self.max_treedepth, int)
            or self.max_treedepth < 1
        ):
            raise ValueError("control max_treedepth must be a positive integer")
        if self.stepsize is not None and not self.stepsize > 0:
            raise ValueError("control stepsize must be positive")


class PathfinderOptions(ValueModel):
    """CmdStan ``pathfinder`` arguments, as brms passes them to cmdstanr.

    Parameters
    ----------
    draws : int
        Draws returned after Pareto-smoothed importance resampling (cmdstanr
        ``draws``, CmdStan ``num_psis_draws``).
    single_path_draws : int
        Draws per path (cmdstanr ``single_path_draws``, CmdStan
        ``num_draws``); with one path, the draws returned.
    history_size : int
        L-BFGS history size.
    max_lbfgs_iters : int
        Maximum L-BFGS iterations per path.
    num_elbo_draws : int
        Draws per ELBO estimate.
    psis_resample : bool
        Whether to importance resample across paths.
    calculate_lp : bool
        Whether to evaluate the log density at the returned draws; without
        it, no resampling takes place.
    save_single_paths : bool
        Whether to keep each path's draws.
    init_alpha, tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param : float
        L-BFGS first step length and convergence tolerances, as CmdStan's
        optimizer.
    """

    draws: int = 1000
    single_path_draws: int = 1000
    history_size: int = 5
    max_lbfgs_iters: int = 1000
    num_elbo_draws: int = 25
    psis_resample: bool = True
    calculate_lp: bool = True
    save_single_paths: bool = False
    init_alpha: float = 1e-3
    tol_obj: float = 1e-12
    tol_rel_obj: float = 1e4
    tol_grad: float = 1e-8
    tol_rel_grad: float = 1e7
    tol_param: float = 1e-8

    def lbfgs(self) -> LBFGSSettings:
        """Return the L-BFGS settings these arguments select.

        Returns
        -------
        LBFGSSettings
            History size, iteration budget, first step and tolerances.
        """
        return LBFGSSettings(
            history_size=self.history_size,
            max_iterations=self.max_lbfgs_iters,
            init_alpha=self.init_alpha,
            tol_obj=self.tol_obj,
            tol_rel_obj=self.tol_rel_obj,
            tol_grad=self.tol_grad,
            tol_rel_grad=self.tol_rel_grad,
            tol_param=self.tol_param,
        )


def pathfinder_options(
    options: PathfinderOptions | Mapping[str, float | int | bool] | None,
) -> PathfinderOptions | None:
    """Normalize cmdstanr ``pathfinder`` arguments.

    Parameters
    ----------
    options : PathfinderOptions or Mapping or None
        A typed record, or cmdstanr's argument names.

    Returns
    -------
    PathfinderOptions or None
        The typed record.

    Raises
    ------
    ValueError
        If a name is not a cmdstanr ``pathfinder`` argument or a value is
        invalid.
    """
    if options is None or isinstance(options, PathfinderOptions):
        return options
    known = {
        "draws",
        "single_path_draws",
        "history_size",
        "max_lbfgs_iters",
        "num_elbo_draws",
        "psis_resample",
        "calculate_lp",
        "save_single_paths",
        "init_alpha",
        "tol_obj",
        "tol_rel_obj",
        "tol_grad",
        "tol_rel_grad",
        "tol_param",
    }
    unknown = set(options) - known
    if unknown:
        raise ValueError(
            f"unknown pathfinder arguments {sorted(unknown)}; cmdstanr's "
            f"pathfinder takes {sorted(known)} (num_paths is chains)"
        )
    defaults = PathfinderOptions()

    def integer(name: str) -> int:
        """Read a count argument.

        Parameters
        ----------
        name : str
            Argument name.

        Returns
        -------
        int
            The given or default count.

        Raises
        ------
        ValueError
            If the value is not an integer.
        """
        value = options.get(name, getattr(defaults, name))
        if isinstance(value, bool) or not float(value).is_integer():
            raise ValueError(f"pathfinder {name} must be an integer")
        return int(value)

    def flag(name: str) -> bool:
        """Read a logical argument.

        Parameters
        ----------
        name : str
            Argument name.

        Returns
        -------
        bool
            The given or default flag.

        Raises
        ------
        ValueError
            If the value is not Boolean.
        """
        value = options.get(name, getattr(defaults, name))
        if not isinstance(value, bool):
            raise ValueError(f"pathfinder {name} must be True or False")
        return value

    def real(name: str) -> float:
        """Read a step length or tolerance.

        Parameters
        ----------
        name : str
            Argument name.

        Returns
        -------
        float
            The given or default value.
        """
        return float(options.get(name, getattr(defaults, name)))

    return PathfinderOptions(
        draws=integer("draws"),
        single_path_draws=integer("single_path_draws"),
        history_size=integer("history_size"),
        max_lbfgs_iters=integer("max_lbfgs_iters"),
        num_elbo_draws=integer("num_elbo_draws"),
        psis_resample=flag("psis_resample"),
        calculate_lp=flag("calculate_lp"),
        save_single_paths=flag("save_single_paths"),
        init_alpha=real("init_alpha"),
        tol_obj=real("tol_obj"),
        tol_rel_obj=real("tol_rel_obj"),
        tol_grad=real("tol_grad"),
        tol_rel_grad=real("tol_rel_grad"),
        tol_param=real("tol_param"),
    )


def checked_cores(cores: int) -> int:
    """Validate brms ``cores``.

    Parameters
    ----------
    cores : int
        Worker processes for the chains.

    Returns
    -------
    int
        The count.

    Raises
    ------
    ValueError
        If ``cores`` is not a positive integer.
    """
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ValueError("cores must be a positive integer")
    return cores


def pathfinder_driver(
    options: PathfinderOptions | None,
    *,
    num_paths: int,
    seed: int,
    init: InitSpec | None,
    init_r: float,
    starts: InitValues,
) -> Pathfinder:
    """Configure the Pathfinder driver for a brms fit.

    Parameters
    ----------
    options : PathfinderOptions or None
        cmdstanr arguments; None keeps CmdStan's defaults.
    num_paths : int
        brms ``chains``.
    seed : int
        Base seed.
    init : InitSpec or None
        brms ``init``: None or ``"random"`` draws uniform initial points on
        ``(-init_r, init_r)`` as CmdStan does, ``0`` starts at the origin,
        and mappings, per-path lists, or functions give named values.
    init_r : float
        Initial radius.
    starts : InitValues
        Formula starts and explicit initial values every path receives.

    Returns
    -------
    Pathfinder
        The driver.

    Raises
    ------
    ValueError
        If ``init`` is ``"jitter"`` or ``"prior"``, which Pathfinder does not
        take, or is otherwise invalid.
    """
    options = PathfinderOptions() if options is None else options
    if init in ("jitter", "prior"):
        raise ValueError(
            f"init={init!r} is an MCMC strategy; Pathfinder starts from "
            "'random' (uniform on (-init_r, init_r)), 0, or named values"
        )
    strategy, values = chain_initialization(
        "random" if init is None else init, starts, num_paths
    )
    return Pathfinder(
        num_paths=num_paths,
        num_draws=options.single_path_draws,
        num_psis_draws=options.draws,
        num_elbo_draws=options.num_elbo_draws,
        psis_resample=options.psis_resample,
        calculate_lp=options.calculate_lp,
        save_single_paths=options.save_single_paths,
        lbfgs=options.lbfgs(),
        init_radius=init_r,
        init_strategy="zero" if strategy == "zero" else "uniform",
        initial_values=values if isinstance(values, Mapping) else tuple(values),
        seed=seed,
    )


class InferenceControls(ValueModel):
    """The brms controls one fit passes to the shared inference drivers.

    Parameters
    ----------
    control : SamplerControl or None
        Kernel settings, or None for the kernel defaults.
    thin : int
        Keep every ``thin``-th post-warmup transition.
    init : InitSpec or None
        Starting values; None keeps the driver's seeded jitter.
    init_r : float
        Half-width of ``init = "random"``.
    pathfinder : PathfinderOptions or None
        cmdstanr ``pathfinder`` arguments, for ``algorithm = "pathfinder"``.
    cores : int
        brms ``cores``: worker processes for the MCMC chains.
    chain_worker : ChainWorker or None
        Recipe rebuilding the fit in a worker, required when ``cores > 1``.
    """

    control: SamplerControl | None = None
    thin: int = 1
    init: InitSpec | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    init_r: float = 2.0
    pathfinder: PathfinderOptions | None = None
    cores: int = 1
    chain_worker: ChainWorker | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _validate(self) -> None:
        if (
            isinstance(self.cores, bool)
            or not isinstance(self.cores, int)
            or self.cores < 1
        ):
            raise ValueError("cores must be a positive integer")
        if (
            isinstance(self.thin, bool)
            or not isinstance(self.thin, int)
            or self.thin < 1
        ):
            raise ValueError("thin must be a positive integer")
        if not self.init_r > 0:
            raise ValueError("init_r must be positive")


def sampler_control(
    control: SamplerControl | Mapping[str, float | int | str] | None,
) -> SamplerControl | None:
    """Normalize a brms-style control list.

    Parameters
    ----------
    control : SamplerControl or Mapping or None
        A typed record, or brms's names ``adapt_delta``, ``max_treedepth``,
        ``stepsize`` and ``metric``.

    Returns
    -------
    SamplerControl or None
        The typed record.

    Raises
    ------
    ValueError
        If a name is not a brms control.
    """
    if control is None or isinstance(control, SamplerControl):
        return control
    unknown = set(control) - {"adapt_delta", "max_treedepth", "stepsize", "metric"}
    if unknown:
        raise ValueError(
            f"unknown control settings {sorted(unknown)}; brms control takes "
            "adapt_delta, max_treedepth, stepsize, and metric"
        )
    metric = str(control.get("metric", "diag_e"))
    if metric not in {"diag_e", "dense_e", "unit_e"}:
        raise ValueError("control metric must be 'diag_e', 'dense_e', or 'unit_e'")
    stepsize = control.get("stepsize")
    return SamplerControl(
        adapt_delta=float(control.get("adapt_delta", 0.8)),
        max_treedepth=int(control.get("max_treedepth", 10)),
        stepsize=None if stepsize is None else float(stepsize),
        metric=cast(Literal["diag_e", "dense_e", "unit_e"], metric),
    )


def resolve_algorithm(
    algorithm: Algorithm | None,
    method: str,
    guide: type | None,
) -> tuple[FitMethod, type | None]:
    """Map brms ``algorithm`` onto a quivers method and guide.

    Parameters
    ----------
    algorithm : Algorithm or None
        brms algorithm, or None to keep ``method`` and ``guide``.
    method : str
        The ``method`` keyword.
    guide : type or None
        The ``guide`` keyword.

    Returns
    -------
    tuple
        The method and guide class.

    Raises
    ------
    ValueError
        If ``algorithm`` contradicts an explicit method or guide, or is
        unknown.
    """
    if algorithm is None:
        if method not in {"nuts", "hmc", "svi", "fixed_param", "pathfinder"}:
            raise ValueError(f"unknown inference method {method!r}")
        return cast(FitMethod, method), guide
    table: dict[str, tuple[FitMethod, type | None]] = {
        "sampling": ("nuts", None),
        "meanfield": ("svi", AutoNormalGuide),
        "fullrank": ("svi", AutoMultivariateNormalGuide),
        "laplace": ("svi", AutoLaplaceApproximation),
        "fixed_param": ("fixed_param", None),
        "pathfinder": ("pathfinder", None),
    }
    if algorithm not in table:
        raise ValueError(f"unknown algorithm {algorithm!r}")
    resolved, resolved_guide = table[algorithm]
    if method not in {"nuts", resolved}:
        raise ValueError(
            f"algorithm={algorithm!r} selects method={resolved!r}; omit method"
        )
    if guide is not None and guide is not resolved_guide:
        raise ValueError(f"algorithm={algorithm!r} selects its own guide; omit guide")
    return resolved, resolved_guide


def mcmc_kernel(
    method: Literal["nuts", "hmc", "fixed_param"], control: SamplerControl | None
) -> MCMCKernel:
    """Build the transition kernel for an MCMC method and brms control.

    Parameters
    ----------
    method : {"nuts", "hmc", "fixed_param"}
        Kernel family.
    control : SamplerControl or None
        brms control settings.

    Returns
    -------
    MCMCKernel
        The configured kernel.

    Raises
    ------
    ValueError
        If ``max_treedepth`` is given for HMC, or any control for the
        fixed-parameter sampler.
    """
    if method == "fixed_param":
        if control is not None:
            raise ValueError("the fixed-parameter sampler takes no control settings")
        return FixedParamKernel()
    if control is None:
        return NUTSKernel() if method == "nuts" else HMCKernel()
    kinds: dict[str, MassMatrixKind] = {
        "diag_e": "diagonal",
        "dense_e": "dense",
        "unit_e": "identity",
    }
    metric = kinds[control.metric]
    if method == "nuts":
        if control.stepsize is None:
            return NUTSKernel(
                max_tree_depth=control.max_treedepth,
                mass_matrix=metric,
                target_accept=control.adapt_delta,
            )
        return NUTSKernel(
            step_size=control.stepsize,
            max_tree_depth=control.max_treedepth,
            mass_matrix=metric,
            target_accept=control.adapt_delta,
        )
    if control.max_treedepth != 10:
        raise ValueError("max_treedepth applies to NUTS, not static HMC")
    if control.stepsize is None:
        return HMCKernel(mass_matrix=metric, target_accept=control.adapt_delta)
    return HMCKernel(
        step_size=control.stepsize,
        mass_matrix=metric,
        target_accept=control.adapt_delta,
    )


def chain_initialization(
    init: InitSpec | None,
    starts: InitValues,
    num_chains: int,
) -> tuple[InitStrategy, InitValues | tuple[InitValues, ...]]:
    """Resolve brms ``init`` into a driver strategy and named values.

    Parameters
    ----------
    init : InitSpec or None
        brms ``init``.
    starts : InitValues
        Support-aware formula starts and explicit ``initial_values``, which
        every chain receives unless ``init`` names the same site.
    num_chains : int
        Chain count.

    Returns
    -------
    tuple
        The strategy for unnamed coordinates, and shared or per-chain values.

    Raises
    ------
    ValueError
        If a numeric ``init`` is nonzero, a per-chain list has the wrong
        length, or a function returns something other than a mapping.
    """
    if init is None:
        return "jitter", starts
    if isinstance(init, str):
        strategy = {
            "random": "uniform",
            "0": "zero",
            "jitter": "jitter",
            "prior": "prior",
        }
        if init not in strategy:
            raise ValueError(f"unknown init {init!r}")
        return cast(InitStrategy, strategy[init]), starts
    if isinstance(init, (bool, int, float)):
        if isinstance(init, bool) or init != 0:
            raise ValueError("a numeric init must be 0, as in brms")
        return "zero", starts
    if isinstance(init, Mapping):
        return "uniform", {**starts, **init}
    if callable(init):
        values = []
        for chain in range(num_chains):
            result = init(chain + 1)
            if not isinstance(result, Mapping):
                raise ValueError("an init function must return a mapping of values")
            values.append({**starts, **result})
        return "uniform", tuple(values)
    chains = tuple(init)
    if len(chains) != num_chains or not all(isinstance(v, Mapping) for v in chains):
        raise ValueError(
            f"a per-chain init list needs one mapping per chain ({num_chains})"
        )
    return "uniform", tuple({**starts, **values} for values in chains)


class FixedParamKernel(MCMCKernel):
    """Stan's fixed-parameter sampler: every transition keeps the position.

    Each retained draw repeats the chain's initial values, so posterior
    replay generates predictions and derived quantities at fixed parameters,
    which is what brms's ``algorithm = "fixed_param"`` is for.
    """

    def init(
        self,
        registry: LatentRegistry,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        initial_position: torch.Tensor,
    ) -> KernelState:
        """Start at the given position.

        Parameters
        ----------
        registry : LatentRegistry
            Registry of the flat vector.
        model : MonadicProgram
            The model (unused).
        x : torch.Tensor
            The input (unused).
        observations : dict[str, torch.Tensor]
            The data (unused).
        initial_position : torch.Tensor
            Flat unconstrained start, shape ``(D,)``.

        Returns
        -------
        KernelState
            The starting state, evaluated by the first transition.

        Raises
        ------
        ValueError
            If the position has the wrong shape.
        """
        del model, x, observations
        dim = registry.total_unconstrained_dim
        if initial_position.shape != (dim,):
            raise ValueError(
                f"FixedParamKernel.init: initial_position must have shape ({dim},)"
            )
        self.is_adapting = False
        return KernelState(
            position=initial_position.clone(),
            log_density=torch.tensor(0.0),
            grad_log_density=torch.zeros(dim),
        )

    def step(self, state: KernelState, potential: PotentialFn) -> KernelState:
        """Keep the position and record its log density.

        Parameters
        ----------
        state : KernelState
            The current state.
        potential : PotentialFn
            The target.

        Returns
        -------
        KernelState
            The same position with ``accept_prob`` one.
        """
        state = self._evaluate_first(state, potential)
        return state.with_(
            step_count=state.step_count + 1,
            accept_count=state.accept_count + 1,
            diverged=False,
            stats={"accepted": 1.0, "accept_prob": 1.0},
        )


def needs_refit(
    path: Path, policy: FileRefit, options: Mapping[str, FitOption]
) -> bool:
    """Decide whether a cached fit must be recomputed, as brms ``file_refit``.

    Parameters
    ----------
    path : Path
        Artifact directory.
    policy : {"never", "always", "on_change"}
        brms ``file_refit``.
    options : Mapping[str, FitOption]
        The would-be fit's canonical arguments.

    Returns
    -------
    bool
        True when no artifact exists, the policy is ``always``, or under
        ``on_change`` the stored formula, data, priors, or inference
        settings differ from ``options``.

    Raises
    ------
    ValueError
        If ``policy`` is unknown.
    """
    if policy not in {"never", "always", "on_change"}:
        raise ValueError("file_refit must be 'never', 'always', or 'on_change'")
    if not path.exists() or policy == "always":
        return True
    return policy == "on_change" and not stored_options_match(path, options)


def stored_options_match(path: Path, options: Mapping[str, FitOption]) -> bool:
    """Compare a stored fit's recipe with a would-be fit's arguments.

    Parameters
    ----------
    path : Path
        Artifact directory.
    options : Mapping[str, FitOption]
        Canonical arguments, including ``data``.

    Returns
    -------
    bool
        Whether every argument except the caching keywords is equal.
    """
    stored = artifact_store.load_fit(path)
    recipe = stored.refit_recipe
    if recipe is None:
        return False
    method = recipe.method
    if not isinstance(method, Mapping):
        return False
    stored_options = cast(Mapping[str, FitOption], method)
    for name, value in options.items():
        if name in {"file", "file_refit", "snapshot_policy"}:
            continue
        if name == "data":
            if not _frames_equal(
                cast(IntoDataFrame, recipe._data), cast(IntoDataFrame, value)
            ):
                return False
            continue
        if name not in stored_options or not _equal(stored_options[name], value):
            return False
    return True


def _equal(left: FitOption, right: FitOption) -> bool:
    """Compare two fit arguments, tensors by value.

    Parameters
    ----------
    left, right : FitOption
        Arguments.

    Returns
    -------
    bool
        Equality.
    """
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    try:
        return bool(left == right)
    except TypeError, ValueError:
        return left is right


def _frames_equal(left: IntoDataFrame, right: IntoDataFrame) -> bool:
    """Compare two dataframes by value through Narwhals.

    Parameters
    ----------
    left, right : IntoDataFrame
        Dataframes.

    Returns
    -------
    bool
        Whether columns and values agree.
    """
    a = nw.from_native(left, eager_only=True).to_pandas()
    b = nw.from_native(right, eager_only=True).to_pandas()
    return list(a.columns) == list(b.columns) and a.equals(b)


__all__ = [
    "Algorithm",
    "FileRefit",
    "FixedParamKernel",
    "InferenceControls",
    "InitSpec",
    "PathfinderOptions",
    "SamplerControl",
]
