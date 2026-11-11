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

from typing import Mapping

import arviz as az
import torch
import xarray as xr

from quivers.inference.mcmc.driver import MCMCResult


def _tensor_to_numpy(t: torch.Tensor):
    """Detach a tensor, move to CPU, and convert to numpy.  The
    detach is necessary because samples may carry an autograd
    graph (e.g. when produced by SVI's ``Predictive`` rather than
    NUTS).
    """
    return t.detach().cpu().numpy()


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
    data: dict[str, dict] = {}

    posterior_group: dict = {}
    for name, t in posterior.samples.items():
        posterior_group[name] = _tensor_to_numpy(t)
    data["posterior"] = posterior_group

    sample_stats_group: dict = {
        "lp": _tensor_to_numpy(posterior.log_densities),
    }
    # Per-draw sampler statistics, under ArviZ's canonical names where
    # one exists and under their own (component-prefixed) names
    # otherwise.
    for key, values in posterior.sample_stats.items():
        name = _ARVIZ_SAMPLE_STATS.get(key, key)
        array = _tensor_to_numpy(values)
        if key in _ARVIZ_INTEGER_STATS:
            array = array.astype("int64")
        elif key in _ARVIZ_BOOLEAN_STATS:
            array = array.astype(bool)
        sample_stats_group[name] = array
    data["sample_stats"] = sample_stats_group

    if observed_data:
        data["observed_data"] = {
            name: _tensor_to_numpy(t) for name, t in observed_data.items()
        }
    if posterior_predictive:
        data["posterior_predictive"] = {
            name: _tensor_to_numpy(t) for name, t in posterior_predictive.items()
        }
    if log_likelihood:
        data["log_likelihood"] = {
            name: _tensor_to_numpy(t) for name, t in log_likelihood.items()
        }
    if constant_data:
        data["constant_data"] = {
            name: _tensor_to_numpy(t) for name, t in constant_data.items()
        }

    return az.from_dict(
        data,
        coords=dict(coords) if coords is not None else None,
        dims=dict(dims) if dims is not None else None,
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
    data: dict[str, dict] = {}

    posterior_group: dict = {}
    for name, t in samples.items():
        arr = _tensor_to_numpy(t)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        posterior_group[name] = arr[None, ...]
    data["posterior"] = posterior_group

    if log_densities is not None:
        lp = _tensor_to_numpy(log_densities)
        if lp.ndim == 0:
            lp = lp.reshape(1)
        data["sample_stats"] = {"lp": lp[None, ...]}

    if observed_data:
        data["observed_data"] = {
            name: _tensor_to_numpy(t) for name, t in observed_data.items()
        }
    if posterior_predictive:
        data["posterior_predictive"] = {
            name: _tensor_to_numpy(t)[None, ...]
            for name, t in posterior_predictive.items()
        }
    if log_likelihood:
        data["log_likelihood"] = {
            name: _tensor_to_numpy(t)[None, ...] for name, t in log_likelihood.items()
        }
    if constant_data:
        data["constant_data"] = {
            name: _tensor_to_numpy(t) for name, t in constant_data.items()
        }

    return az.from_dict(
        data,
        coords=dict(coords) if coords is not None else None,
        dims=dict(dims) if dims is not None else None,
    )


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


__all__ = ["to_datatree", "to_datatree_any", "to_datatree_from_svi"]
