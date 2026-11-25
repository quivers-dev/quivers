"""Collect raw observation scores from posterior replay traces.

An observation with a density of its own is scored at its site. The
density of a marginalized observation is in the score of its block, whose
units the trace holds: a row of it has a log likelihood when its unit
holds no other row, and the units have log likelihoods in every case.
"""

from collections.abc import Iterable, Iterator

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.morphisms import MarginalizedFactor
from quivers.effects.units import SiteUnits
from quivers.inference._posterior import collect_site_draws
from quivers.inference.trace import SampleSite, Trace
from quivers.inference.units import UnitLogLikelihood


def selected_name(label: str, names: set[str]) -> str:
    """The requested name an occurrence label is read under.

    Parameters
    ----------
    label : str
        A site's label, or ``"<name>@<n>"`` for an occurrence of a site a
        scan reaches once per position.
    names : set[str]
        The requested names.

    Returns
    -------
    str
        The label when it is requested itself, else the name without the
        occurrence.
    """
    if label in names:
        return label
    base, separator, position = label.rpartition("@")
    return base if separator and position.isdigit() else label


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
        selected = selected_name(name, names)
        if selected in names:
            current.setdefault(selected, []).append(site)
    return {name: tuple(sites) for name, sites in current.items()}


def _site_log_likelihood(site: SampleSite) -> torch.Tensor:
    """The log density of each row of an observation with its own density.

    Parameters
    ----------
    site : SampleSite
        The observation's site.

    Returns
    -------
    torch.Tensor
        Raw log densities on the observation's rows; one entry without a
        plate.
    """
    sampleable = site.sampleable
    assert sampleable is not None
    density = sampleable.raw_log_prob_pointwise(site.value)
    return density.reshape(1) if density.ndim == 0 else density


def _marginal(site: SampleSite) -> bool:
    """Whether a score step holds an observation's density.

    Parameters
    ----------
    site : SampleSite
        The observation's site.

    Returns
    -------
    bool
        ``True`` for the response of a marginalized block.
    """
    return isinstance(site.morphism, MarginalizedFactor)


def unit_sites(recorded: Trace, names: set[str]) -> tuple[SiteUnits, ...]:
    """The units of the score sites that hold requested observations.

    Parameters
    ----------
    recorded : Trace
        One replayed run.
    names : set[str]
        Requested source names or exact occurrence labels.

    Returns
    -------
    tuple[SiteUnits, ...]
        The units, in the order the run reached their scores.

    Raises
    ------
    ValueError
        If the trace holds several runs, whose scores are not one
        likelihood.
    """
    found: list[SiteUnits] = []
    for site in recorded.sites.values():
        if not any(
            selected_name(label, names) in names
            for units in site.units
            for label in units.responses
        ):
            continue
        if len(site.units) != 1:
            raise ValueError(
                f"score {site.name!r} holds {len(site.units)} runs; its units are "
                "read from a trace of one run"
            )
        found.extend(site.units)
    return tuple(found)


def _require_scored(observations: set[str], found: Iterable[str]) -> None:
    """Refuse requested observations that no site of a trace scores.

    Parameters
    ----------
    observations : set[str]
        The requested names.
    found : Iterable[str]
        The names scored.

    Raises
    ------
    TypeError
        If a name is missing.
    """
    if missing := observations - set(found):
        raise TypeError(
            f"observations {sorted(missing)} have no log likelihood in the replay: "
            "no observation site scores them, and no score step declares the units "
            "that hold their rows"
        )


def row_log_likelihood(
    recorded: Trace, observations: set[str]
) -> dict[str, torch.Tensor]:
    """The log likelihood of each row of the requested observations.

    Parameters
    ----------
    recorded : Trace
        One replayed run.
    observations : set[str]
        Requested source names or exact occurrence labels.

    Returns
    -------
    dict[str, torch.Tensor]
        Raw scores on each observation's rows, occurrences stacked in the
        order the run reached them. A row that several factors of the
        joint score has the sum of their scores.

    Raises
    ------
    UnfactoredObservationError
        If the rows of a marginalized observation share a latent.
    TypeError
        If nothing scores an observation.
    ValueError
        If the trace holds several runs of a score.
    """
    scored: dict[str, dict[str, torch.Tensor]] = {}

    def add(label: str, name: str, score: torch.Tensor) -> None:
        """Add one factor's scores of an occurrence's rows.

        Parameters
        ----------
        label : str
            The occurrence's label.
        name : str
            The requested name it is read under.
        score : torch.Tensor
            The scores.
        """
        own = scored.setdefault(name, {})
        own[label] = own[label] + score if label in own else score

    for name, sites in observation_sites(recorded, observations).items():
        for site in sites:
            if not _marginal(site):
                add(site.name, name, _site_log_likelihood(site))
    for units in unit_sites(recorded, observations):
        for label in dict.fromkeys(units.responses):
            name = selected_name(label, observations)
            if name in observations:
                add(label, name, units.row_log_likelihood(label))
    _require_scored(observations, scored)
    return {
        name: next(iter(own.values()))
        if len(own) == 1
        else torch.stack([*own.values()])
        for name, own in scored.items()
    }


def collect_log_likelihood(
    traces: Iterable[Trace], observations: set[str], shape: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Collect the row scores of one trace per chain and draw.

    Likelihood powers are excluded. Every result is allocated once, after
    its first trace.

    Parameters
    ----------
    traces : Iterable[Trace]
        One replayed run per retained draw, chains in order.
    observations : set[str]
        The observed names.
    shape : tuple[int, int]
        The number of chains and of draws in each.

    Returns
    -------
    dict[str, torch.Tensor]
        For each observation, ``(chains, draws, *rows)``.

    Raises
    ------
    UnfactoredObservationError
        If the rows of a marginalized observation share a latent. The
        message says which program gives each row its own.
    TypeError
        If nothing scores an observation.
    ValueError
        If the draws disagree on the sites or their shapes.
    """
    return collect_site_draws(
        (row_log_likelihood(recorded, observations) for recorded in traces), shape
    )


class _UnitScores(ValueModel):
    """The unit scores of one replayed run, under one key.

    Parameters
    ----------
    axes : tuple[str, ...] or None
        The objects that index the units.
    responses : tuple[str, ...]
        The responses whose rows the units hold.
    score : torch.Tensor
        The log likelihood of each unit.
    row_units : dict[str, torch.Tensor]
        The unit of each row of each response.
    """

    axes: tuple[str, ...] | None
    responses: tuple[str, ...]
    score: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    row_units: dict[str, torch.Tensor] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def _unit_scores(recorded: Trace, observations: set[str]) -> dict[str, _UnitScores]:
    """The unit scores of the requested observations in one run.

    Parameters
    ----------
    recorded : Trace
        One replayed run.
    observations : set[str]
        Requested source names or exact occurrence labels.

    Returns
    -------
    dict[str, _UnitScores]
        One entry per set of responses whose rows share units, keyed by
        their labels joined by commas.

    Raises
    ------
    UnfactoredObservationError
        If a score's unit weights are not log densities.
    TypeError
        If nothing scores an observation, or two factors of the joint
        score the same response in units of their own.
    ValueError
        If the trace holds several runs of a score.
    """
    result: dict[str, _UnitScores] = {}
    holders: dict[str, str] = {}

    def add(scores: _UnitScores) -> None:
        """Add the units of one factor of the joint.

        Parameters
        ----------
        scores : _UnitScores
            The factor's units.

        Raises
        ------
        TypeError
            If another factor scores one of its responses.
        """
        key = ",".join(scores.responses)
        for label in scores.responses:
            if label in holders:
                raise TypeError(
                    f"observation {label!r} is scored in the units of two factors "
                    "of the joint, so neither factor's units are independent of "
                    "the other's"
                )
            holders[label] = key
        result[key] = scores

    for name, sites in observation_sites(recorded, observations).items():
        for site in sites:
            if not _marginal(site):
                score = _site_log_likelihood(site)
                add(
                    _UnitScores(
                        axes=None,
                        responses=(site.name,),
                        score=score,
                        row_units={
                            site.name: torch.arange(score.numel(), device=score.device)
                        },
                    )
                )
    for units in unit_sites(recorded, observations):
        score = units.log_likelihood()
        add(
            _UnitScores(
                axes=units.axes,
                responses=tuple(dict.fromkeys(units.responses)),
                score=score.reshape(1) if score.ndim == 0 else score,
                row_units={label: units.row_units(label) for label in units.responses},
            )
        )
    _require_scored(
        observations, (selected_name(label, observations) for label in holders)
    )
    return result


def collect_unit_log_likelihood(
    traces: Iterable[Trace], observations: set[str], shape: tuple[int, int]
) -> dict[str, UnitLogLikelihood]:
    """Collect the unit scores of one trace per chain and draw.

    The units are the finest independent parts of the likelihood that the
    program declares: the rows of an observation with a density of its
    own, the units of a grouped block, and the whole of a block whose rows
    share one latent. Likelihood powers are excluded.

    Parameters
    ----------
    traces : Iterable[Trace]
        One replayed run per retained draw, chains in order.
    observations : set[str]
        The observed names. A block that holds one of them contributes
        with all its responses.
    shape : tuple[int, int]
        The number of chains and of draws in each.

    Returns
    -------
    dict[str, UnitLogLikelihood]
        One entry per set of responses whose rows share units, keyed by
        their labels joined by commas.

    Raises
    ------
    UnfactoredObservationError
        If a score's unit weights are not log densities.
    TypeError
        If nothing scores an observation, or two factors of the joint
        score the same response.
    ValueError
        If the draws disagree on the units or on which unit a row belongs
        to, so that the units are not the same parts of the data in each.
    """
    first: dict[str, _UnitScores] = {}

    def values() -> Iterator[dict[str, torch.Tensor]]:
        """Read each trace's unit scores, checking their rows.

        Yields
        ------
        dict[str, torch.Tensor]
            The scores of one draw.

        Raises
        ------
        ValueError
            If a row belongs to another unit than in the first draw.
        """
        for recorded in traces:
            current = _unit_scores(recorded, observations)
            for key, scores in current.items():
                known = first.setdefault(key, scores)
                for label, rows in scores.row_units.items():
                    if not torch.equal(rows, known.row_units[label]):
                        raise ValueError(
                            f"the rows of {label!r} belong to other units at some "
                            "draws, so the units are not the same parts of the "
                            "data at every draw"
                        )
            yield {key: scores.score for key, scores in current.items()}

    collected = collect_site_draws(values(), shape)
    return {
        key: UnitLogLikelihood(
            axes=first[key].axes,
            responses=first[key].responses,
            log_likelihood=values,
            row_units=first[key].row_units,
        )
        for key, values in collected.items()
    }
