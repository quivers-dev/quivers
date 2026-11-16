"""Conversion from quivers fits to `xarray.DataTree` (the
ArviZ 1.x container).

The MCMC sampler hands back [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult]
records whose ``samples`` dict already follows the
``(num_chains, num_samples, *site_shape)`` shape convention ArviZ
expects.  This module repackages those tensors as numpy arrays under
the canonical ArviZ group names (``posterior``, ``sample_stats``,
``posterior_predictive``, ``log_likelihood``, ``observed_data``) plus
the user-supplied ``coords`` / ``dims`` for nice axis labels in
plots.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np

import arviz as az
import torch
import xarray as xr

from quivers.diagnostics._coordinates import axis_labels, event_dimensions
from quivers.diagnostics.calibration import SBCResult
from quivers.inference.mcmc.driver import MCMCResult

if TYPE_CHECKING:
    from quivers.diagnostics.posterior import PosteriorAnalysis


def _tensor_to_numpy(t: torch.Tensor):
    """Detach a tensor and expose CPU NumPy values for ArviZ.

    Parameters
    ----------
    t : torch.Tensor
        Tensor on any supported device, optionally carrying autograd history.

    Returns
    -------
    numpy.ndarray
        CPU values; BFloat16 uses exact Float32 values because NumPy lacks
        its dtype. The retained analysis converter records the original dtype.
    """
    values = t.detach().cpu()
    if values.dtype == torch.bfloat16:
        values = values.float()
    return values.numpy()


_ARVIZ_SAMPLE_STATS = {
    "diverged": "diverging",
    "tree_depth": "tree_depth",
    "num_steps": "n_steps",
    "energy": "energy",
    "step_size": "step_size",
    "accept_prob": "acceptance_rate",
}
"""ArviZ's ``sample_stats`` names for the kernels' per-draw statistics."""

_ARVIZ_INTEGER_STATS = frozenset({"tree_depth", "num_steps"})
_ARVIZ_BOOLEAN_STATS = frozenset({"diverged"})


def _groups_to_datatree(
    groups: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    coords: Mapping[str, list] | None = None,
    dims: Mapping[str, list[str]] | None = None,
    group_dims: Mapping[str, Mapping[str, list[str]]] | None = None,
) -> xr.DataTree:
    """Convert canonical tensor groups, sharing sampler-stat names and dtypes.

    Parameters
    ----------
    groups : Mapping
        Canonical group names and their tensor mappings.
    coords : Mapping or None
        Shared coordinate labels, including optional chain and draw labels.
    dims : Mapping or None
        Default event dimension names per variable.
    group_dims : Mapping or None
        Group-specific dimension overrides for observed versus reduced events.

    Returns
    -------
    xarray.DataTree
        ArviZ groups with validated dimensions and canonical sampler statistics.

    Raises
    ------
    ValueError
        If axes disagree, statistic aliases collide, or canonical count/Boolean
        statistics would lose information during their numeric conversion.
    """
    data = {}
    for group, tensors in groups.items():
        if not tensors:
            continue
        sampled = group not in {"observed_data", "constant_data"}
        selected_dims = (
            dims
            if group_dims is None
            else {
                **(dims or {}),
                **group_dims.get(group, {}),
            }
        )
        event_dimensions(
            {
                key: value.shape[2:] if sampled else value.shape
                for key, value in tensors.items()
            },
            coords=coords,
            dims=selected_dims,
        )
        if sampled:
            for value in tensors.values():
                if value.ndim >= 2:
                    axis_labels("chain", value.shape[0], coords)
                    axis_labels("draw", value.shape[1], coords)
        converted = {}
        for key, values in tensors.items():
            array = _tensor_to_numpy(values)
            name = key
            if group == "sample_stats":
                name = _ARVIZ_SAMPLE_STATS.get(key, key)
                if key in _ARVIZ_INTEGER_STATS:
                    if array.dtype.kind == "f":
                        counts = array.astype("float64")
                        invalid = (
                            not np.isfinite(counts).all()
                            or not np.equal(counts, np.rint(counts)).all()
                            or bool(((counts < 0) | (counts >= 2.0**63)).any())
                        )
                    else:
                        invalid = bool((array < 0).any()) or (
                            array.dtype.kind == "u"
                            and bool((array > np.uint64(2**63 - 1)).any())
                        )
                    if invalid:
                        raise ValueError(
                            f"sampler statistic {key!r} requires nonnegative finite integer counts within Int64 capacity"
                        )
                    array = array.astype("int64")
                elif key in _ARVIZ_BOOLEAN_STATS:
                    if not np.isin(array, (0, 1)).all():
                        raise ValueError(
                            f"sampler statistic {key!r} requires Boolean values"
                        )
                    array = array.astype(bool)
            if name in converted:
                raise ValueError(f"sampler statistic aliases collide at {name!r}")
            converted[name] = array
        data[group] = converted
    if group_dims:
        return xr.DataTree.from_dict(
            {
                group: az.from_dict(
                    {group: values},
                    coords=dict(coords) if coords is not None else None,
                    dims={**(dims or {}), **group_dims.get(group, {})},
                )[group].dataset
                for group, values in data.items()
            }
        )
    return az.from_dict(
        data,
        coords=dict(coords) if coords is not None else None,
        dims=dict(dims) if dims is not None else None,
    )


def _posterior_from_datatree(
    tree: xr.DataTree,
    *,
    kind: Literal["mcmc", "variational", "prior"] | None = None,
) -> PosteriorAnalysis:
    """Invert canonical tensor groups without inferring posterior provenance.

    Parameters
    ----------
    tree : xarray.DataTree
        Labeled posterior, likelihood, observation, and sampler-statistic groups.
    kind : str or None
        Explicit draw provenance for external trees; must agree with QVR metadata.

    Returns
    -------
    PosteriorAnalysis
        Owned CPU tensors and validated dtype, event, and sample-axis metadata.

    Raises
    ------
    ValueError, TypeError
        If metadata, provenance, sample alignment, real dtypes, or axis contracts
        are invalid. No parameter or likelihood evaluation is performed.
    """
    from quivers.diagnostics.posterior import PosteriorAnalysis

    if not isinstance(tree, xr.DataTree):
        raise TypeError("posterior restoration requires an xarray.DataTree")
    version = tree.attrs.get("quivers_posterior_format")
    if version is not None and version != "qvr-posterior/v1":
        raise ValueError(f"unsupported retained posterior format {version!r}")
    recorded_kind = tree.attrs.get("quivers_posterior_kind")
    if kind is not None and recorded_kind is not None and kind != recorded_kind:
        raise ValueError("explicit posterior kind disagrees with recorded provenance")
    provenance = recorded_kind if kind is None else kind
    if provenance not in {"mcmc", "variational", "prior"}:
        raise ValueError(
            "external posterior trees require explicit valid draw provenance"
        )
    groups: dict[str, dict[str, torch.Tensor]] = {}
    coords: dict[str, list] = {}
    dims: dict[str, list[str]] = {}
    group_dims: dict[str, dict[str, list[str]]] = {}
    reverse_stats = {value: key for key, value in _ARVIZ_SAMPLE_STATS.items()}
    recorded_names = tree.attrs.get("quivers_sample_stat_names")
    if recorded_names is not None:
        try:
            restored_names = json.loads(recorded_names)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "invalid retained sampler statistic name metadata"
            ) from error
        if not isinstance(restored_names, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in restored_names.items()
        ):
            raise ValueError(
                "sampler statistic name metadata must map strings to strings"
            )
        reverse_stats = restored_names
    stored_dtypes = json.loads(tree.attrs.get("quivers_tensor_dtypes", "{}"))
    if not isinstance(stored_dtypes, dict) or any(
        not isinstance(group, str)
        or not isinstance(values, dict)
        or any(
            not isinstance(name, str) or not isinstance(dtype, str)
            for name, dtype in values.items()
        )
        for group, values in stored_dtypes.items()
    ):
        raise ValueError("invalid retained tensor dtype metadata")
    supported = {
        str(dtype): dtype
        for dtype in (
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    }
    for group in ("posterior", "log_likelihood", "observed_data", "sample_stats"):
        values: dict[str, torch.Tensor] = {}
        groups[group] = values
        if group not in tree.children:
            continue
        dataset = tree[group].dataset
        group_dims[group] = {}
        for name, variable in dataset.data_vars.items():
            if not isinstance(name, str):
                raise TypeError("posterior variable names must be strings")
            sampled = group != "observed_data"
            axes: list[str] = []
            for axis in variable.dims:
                if not isinstance(axis, str):
                    raise TypeError("posterior dimension names must be strings")
                axes.append(axis)
            if sampled:
                if "chain" not in axes or "draw" not in axes:
                    raise ValueError(
                        "sampled variables require named chain and draw axes"
                    )
                event = [axis for axis in axes if axis not in {"chain", "draw"}]
                variable = variable.transpose("chain", "draw", *event)
            else:
                event = axes
            for axis in ["chain", "draw", *event] if sampled else event:
                labels = (
                    variable.coords[axis].values
                    if axis in variable.coords
                    else np.arange(variable.sizes[axis])
                )
                label_list = labels.tolist()
                if axis in coords and coords[axis] != label_list:
                    raise ValueError(
                        f"posterior groups disagree on labels of axis {axis!r}"
                    )
                coords[axis] = label_list
            if group != "sample_stats":
                group_dims[group][name] = event
                if group != "observed_data":
                    if name in dims and dims[name] != event:
                        raise ValueError(
                            f"sampled groups disagree on event axes of {name!r}"
                        )
                    dims[name] = event
            array = np.array(variable.values, copy=True)
            if array.dtype.kind not in "biuf":
                raise TypeError("retained analysis groups require real numeric tensors")
            tensor = torch.from_numpy(array)
            dtype_name = stored_dtypes.get(group, {}).get(name)
            if dtype_name is not None:
                if dtype_name not in supported:
                    raise ValueError(
                        f"unsupported retained tensor dtype {dtype_name!r}"
                    )
                if supported[dtype_name] != tensor.dtype:
                    converted = tensor.to(dtype=supported[dtype_name])
                    back = converted.to(dtype=tensor.dtype)
                    exact = torch.eq(back, tensor)
                    if tensor.is_floating_point():
                        exact = exact | (torch.isnan(back) & torch.isnan(tensor))
                    if not bool(exact.all()):
                        raise ValueError(
                            f"retained tensor {name!r} is not exact in its recorded dtype"
                        )
                    tensor = converted
            target = reverse_stats.get(name, name) if group == "sample_stats" else name
            if target in values:
                raise ValueError(f"sampler statistic aliases collide at {target!r}")
            values[target] = tensor
    result = PosteriorAnalysis(
        samples=groups["posterior"],
        log_likelihood=groups["log_likelihood"],
        observed_data=groups["observed_data"],
        sample_stats=groups["sample_stats"],
        kind=provenance,
        coords=coords,
        dims=dims,
        group_dims=group_dims,
    )
    result._validate_axes()
    return result


def to_datatree(
    posterior: MCMCResult,
    *,
    observed_data: Mapping[str, torch.Tensor] | None = None,
    posterior_predictive: Mapping[str, torch.Tensor] | None = None,
    log_likelihood: Mapping[str, torch.Tensor] | None = None,
    constant_data: Mapping[str, torch.Tensor] | None = None,
    coords: Mapping[str, list] | None = None,
    dims: Mapping[str, list[str]] | None = None,
) -> xr.DataTree:
    """Convert an `MCMCResult` into an ArviZ-style DataTree.

    Parameters
    ----------
    posterior : MCMCResult
        Sampler output.  ``posterior.samples[name]`` carries an
        ``(num_chains, num_samples, *site_shape)`` tensor; each
        becomes a posterior variable.  ``log_densities`` populates
        ``sample_stats/lp``, and each per-draw entry of
        `MCMCResult.sample_stats` a ``sample_stats`` variable under
        ArviZ's canonical name: ``diverged`` as ``diverging``
        (boolean), ``tree_depth`` (integer), ``num_steps`` as
        ``n_steps`` (integer), ``energy``, ``step_size``, and
        ``accept_prob`` as ``acceptance_rate``. A composite kernel's
        component-prefixed entries keep their names.
    observed_data : Mapping[str, torch.Tensor], optional
        Site name to observed tensor (the original data used at
        fit time).  Becomes the ``observed_data`` group.
    posterior_predictive : Mapping[str, torch.Tensor], optional
        Site name to posterior-predictive draws of shape
        ``(num_chains, num_samples, *site_shape)``.  Becomes the
        ``posterior_predictive`` group.
    log_likelihood : Mapping[str, torch.Tensor], optional
        Site name to per-observation log-likelihood of shape
        ``(num_chains, num_samples, *obs_shape)``.  Becomes the
        ``log_likelihood`` group; required for
        `arviz.loo` / `arviz.waic`.
    constant_data : Mapping[str, torch.Tensor], optional
        Site name to fixed covariate tensor (e.g. design matrix).
        Becomes the ``constant_data`` group.
    coords : Mapping[str, list], optional
        Coordinate values per named axis (e.g.
        ``{"Verb": ["eat", "drink", "run"]}``).  Forwarded to ArviZ.
    dims : Mapping[str, list[str]], optional
        Per-site axis names (e.g. ``{"beta": ["Verb"]}``).
        Forwarded to ArviZ.

    Returns
    -------
    xr.DataTree
        Canonical ArviZ DataTree consumable by every plotting and
        diagnostic function in the ArviZ 1.x API.
    """
    return _groups_to_datatree(
        {
            "posterior": posterior.samples,
            "sample_stats": {
                "lp": posterior.log_densities,
                **posterior.sample_stats,
            },
            "observed_data": observed_data or {},
            "posterior_predictive": posterior_predictive or {},
            "log_likelihood": log_likelihood or {},
            "constant_data": constant_data or {},
        },
        coords=coords,
        dims=dims,
    )


def to_datatree_from_svi(
    samples: Mapping[str, torch.Tensor],
    *,
    log_densities: torch.Tensor | None = None,
    observed_data: Mapping[str, torch.Tensor] | None = None,
    posterior_predictive: Mapping[str, torch.Tensor] | None = None,
    log_likelihood: Mapping[str, torch.Tensor] | None = None,
    constant_data: Mapping[str, torch.Tensor] | None = None,
    coords: Mapping[str, list] | None = None,
    dims: Mapping[str, list[str]] | None = None,
) -> xr.DataTree:
    """Convert a set of SVI or Predictive draws into an ArviZ-style
    DataTree with a single pseudo-chain.

    Each entry in `samples` is a `(num_samples, *site_shape)` tensor
    (the standard shape produced by `Predictive(model, guide,
    num_samples=...)`). A leading chain dimension of size 1 is
    inserted so the resulting DataTree matches the `(chain, draw,
    *event)` shape ArviZ expects. `log_densities`, when supplied,
    becomes `sample_stats/lp`; ArviZ diagnostic functions (`loo`,
    `waic`, `plot_trace`) work uniformly on the resulting object.

    Parameters
    ----------
    samples : Mapping[str, torch.Tensor]
        Site name to draws of shape ``(num_samples, *site_shape)``;
        becomes the ``posterior`` group.
    log_densities : torch.Tensor, optional
        Per-draw log densities; becomes ``sample_stats/lp``.
    observed_data : Mapping[str, torch.Tensor], optional
        Site name to observed tensor; becomes ``observed_data``.
    posterior_predictive : Mapping[str, torch.Tensor], optional
        Site name to posterior-predictive draws of shape
        ``(num_samples, *site_shape)``; becomes
        ``posterior_predictive``.
    log_likelihood : Mapping[str, torch.Tensor], optional
        Site name to per-observation log-likelihood of shape
        ``(num_samples, *obs_shape)``; becomes ``log_likelihood``.
    constant_data : Mapping[str, torch.Tensor], optional
        Site name to fixed covariate tensor; becomes ``constant_data``.
    coords : Mapping[str, list], optional
        Coordinate values per named axis, forwarded to ArviZ.
    dims : Mapping[str, list[str]], optional
        Per-site axis names, forwarded to ArviZ.

    Returns
    -------
    xr.DataTree
        The ArviZ DataTree, with a chain dimension of size 1.
    """
    # The public SVI adapter accepts draw-first tensors. Promote those axes
    # once; every downstream conversion consumes canonical chain/draw groups.
    posterior_group = {
        name: (value.reshape(1) if value.ndim == 0 else value).unsqueeze(0)
        for name, value in samples.items()
    }
    stats = {}
    if log_densities is not None:
        values = log_densities
        stats["lp"] = (values.reshape(1) if values.ndim == 0 else values).unsqueeze(0)
    return _groups_to_datatree(
        {
            "posterior": posterior_group,
            "sample_stats": stats,
            "observed_data": observed_data or {},
            "posterior_predictive": {
                name: value.unsqueeze(0)
                for name, value in (posterior_predictive or {}).items()
            },
            "log_likelihood": {
                name: value.unsqueeze(0)
                for name, value in (log_likelihood or {}).items()
            },
            "constant_data": constant_data or {},
        },
        coords=coords,
        dims=dims,
    )


def to_datatree_from_sbc(result: SBCResult) -> xr.DataTree:
    """Convert simulation-based calibration ranks for ArviZ's PIT plots.

    The ranks go into a ``prior_sbc`` group, the group
    ``arviz_plots.plot_ecdf_pit`` reads by default, with one variable
    per statistic laid out ``(chain=1, draw=num_simulations)``. Each
    rank ``r`` in ``{0, ..., L}`` is mapped to ``(r + 1/2) / (L + 1)``,
    the midpoint of its cell in an even partition of ``[0, 1]``: under
    calibration the values are uniform on those midpoints, so their
    ECDF meets the uniform CDF at every cell boundary, and the mapping
    is symmetric about one half, as the plot's coverage view assumes.

    Parameters
    ----------
    result : SBCResult
        The result of `quivers.diagnostics.calibration`.

    Returns
    -------
    xr.DataTree
        A DataTree with the ``prior_sbc`` group.
    """
    pit = (result.ranks.to(dtype=torch.float64) + 0.5) / (result.num_draws + 1)
    values = _tensor_to_numpy(pit)
    group = {
        name: values[None, :, column]
        for column, name in enumerate(result.statistic_names)
    }
    return az.from_dict({"prior_sbc": group})


def to_datatree_any(
    fit: MCMCResult
    | Mapping[str, torch.Tensor]
    | tuple[Mapping[str, torch.Tensor], torch.Tensor],
    *,
    observed_data: Mapping[str, torch.Tensor] | None = None,
    posterior_predictive: Mapping[str, torch.Tensor] | None = None,
    log_likelihood: Mapping[str, torch.Tensor] | None = None,
    constant_data: Mapping[str, torch.Tensor] | None = None,
    coords: Mapping[str, list] | None = None,
    dims: Mapping[str, list[str]] | None = None,
) -> xr.DataTree:
    """Dispatch a DataTree conversion on the fit type.

    Accepts:

    * an [`MCMCResult`][quivers.inference.MCMCResult], converted by
      `to_datatree` with its chain and draw semantics;
    * a mapping of site name to tensor (SVI or Predictive draws),
      converted by `to_datatree_from_svi` with a single pseudo-chain;
    * a tuple ``(samples, log_densities)``, converted like the mapping
      form with ``sample_stats/lp`` populated from the second entry.

    Consumers with a custom fit-container type can call the specific
    `to_datatree` or `to_datatree_from_svi` entry point directly.

    Parameters
    ----------
    fit : MCMCResult or Mapping[str, torch.Tensor] or tuple
        The fit to convert, in one of the forms above.
    observed_data : Mapping[str, torch.Tensor], optional
        Site name to observed tensor; becomes ``observed_data``.
    posterior_predictive : Mapping[str, torch.Tensor], optional
        Site name to posterior-predictive draws; becomes
        ``posterior_predictive``.
    log_likelihood : Mapping[str, torch.Tensor], optional
        Site name to per-observation log-likelihood; becomes
        ``log_likelihood``.
    constant_data : Mapping[str, torch.Tensor], optional
        Site name to fixed covariate tensor; becomes ``constant_data``.
    coords : Mapping[str, list], optional
        Coordinate values per named axis, forwarded to ArviZ.
    dims : Mapping[str, list[str]], optional
        Per-site axis names, forwarded to ArviZ.

    Returns
    -------
    xr.DataTree
        The ArviZ DataTree of the fit.

    Raises
    ------
    TypeError
        If ``fit`` is none of the accepted forms.
    """
    if isinstance(fit, MCMCResult):
        return to_datatree(
            fit,
            observed_data=observed_data,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            constant_data=constant_data,
            coords=coords,
            dims=dims,
        )
    if isinstance(fit, tuple) and len(fit) == 2:
        samples, log_densities = fit
        return to_datatree_from_svi(
            samples,
            log_densities=log_densities,
            observed_data=observed_data,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            constant_data=constant_data,
            coords=coords,
            dims=dims,
        )
    if isinstance(fit, Mapping):
        return to_datatree_from_svi(
            fit,
            observed_data=observed_data,
            posterior_predictive=posterior_predictive,
            log_likelihood=log_likelihood,
            constant_data=constant_data,
            coords=coords,
            dims=dims,
        )
    raise TypeError(
        f"to_datatree_any: cannot dispatch on fit type "
        f"{type(fit).__name__}; supported: MCMCResult, "
        "Mapping[str, Tensor] (SVI / Predictive), "
        "tuple[Mapping, Tensor] (samples + log-densities)."
    )


__all__ = [
    "to_datatree",
    "to_datatree_any",
    "to_datatree_from_svi",
    "to_datatree_from_sbc",
]
