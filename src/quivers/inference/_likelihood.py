"""Collect raw conditional observation scores from posterior replay traces."""

from collections.abc import Iterable

import torch

from quivers.inference.trace import SampleSite, Trace
from quivers.inference._posterior import collect_site_draws


def observation_sites(
    recorded: Trace, names: set[str]
) -> dict[str, tuple[SampleSite, ...]]:
    """Group native observation occurrences in their trace visitation order.

    Parameters
    ----------
    recorded : Trace
        One replayed execution, including lexical observations.
    names : set[str]
        Requested source names or exact occurrence labels. An exact label
        takes precedence over its unsuffixed source name.

    Returns
    -------
    dict[str, tuple[SampleSite, ...]]
        Sampleable sites grouped under the selected name. Unavailable sites
        are omitted so each consumer can provide its own capability error.
    """
    current: dict[str, list[SampleSite]] = {}
    for name, site in recorded.sites.items():
        if site.sampleable is None:
            continue
        base, separator, position = name.rpartition("@")
        selected = (
            name
            if name in names
            else base
            if separator and position.isdigit()
            else name
        )
        if selected in names:
            current.setdefault(selected, []).append(site)
    return {name: tuple(sites) for name, sites in current.items()}


def collect_log_likelihood(
    traces: Iterable[Trace], observations: set[str], shape: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Collect one trace per chain/draw, retaining observation plate axes.

    Likelihood powers are excluded. A hidden observation requires a dedicated
    marginal-likelihood observer; this function refuses to substitute its
    conditional score. Every result is allocated once, after its first trace.
    """

    def values():
        for recorded in traces:
            current: dict[str, list[torch.Tensor]] = {}
            for name, sites in observation_sites(recorded, observations).items():
                for site in sites:
                    sampleable = site.sampleable
                    assert sampleable is not None
                    raw_pointwise = getattr(sampleable, "raw_log_prob_pointwise", None)
                    pointwise = getattr(sampleable.morphism, "log_prob_pointwise", None)
                    density = (
                        raw_pointwise(site.value)
                        if callable(raw_pointwise)
                        else pointwise(sampleable.input, site.value)
                        if callable(pointwise)
                        else sampleable.log_prob(site.value)
                    )
                    if not isinstance(density, torch.Tensor):
                        raise TypeError(
                            f"observation {name!r} has no native tensor log density"
                        )
                    current.setdefault(name, []).append(
                        density.reshape(1) if density.ndim == 0 else density
                    )
            if missing := observations - set(current):
                raise TypeError(
                    f"observations {sorted(missing)} require a marginal-likelihood observer"
                )
            yield {
                name: densities[0] if len(densities) == 1 else torch.stack(densities)
                for name, densities in current.items()
            }

    return collect_site_draws(values(), shape)
