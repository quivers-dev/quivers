"""Amortized guide: declared morphisms as the variational posterior.

In amortized inference the variational posterior of a latent site is a
kernel applied to the data, :math:`q_\\phi(z \\mid y) = e_\\phi(y)`:
the encoder of a variational autoencoder (Kingma and Welling 2014,
`doi:10.48550/arXiv.1312.6114
<https://doi.org/10.48550/arXiv.1312.6114>`_). `AmortizedGuide` runs
such kernels, any `ContinuousMorphism` with an exact density (a
declared morphism, or deterministic layers followed by one stochastic
kernel), as the guide of the sites it is bound to, so one set of
encoder weights serves every run and every new datum.

Each binding names the latent site, the encoder, and the datum the
encoder reads. The encoder is applied to every row of the datum: once
per run for a site drawn once per run (a datum with a leading run
axis), once per plate row for a plated local latent. Its draw is the
site's value when the encoder's support is the site's support, and is
pushed through the site's bijector when the encoder is real-valued and
the site is not, with the Jacobian in the density. Every binding is
checked against the registry: the site must be latent, the datum must
be supplied, and the encoder's event size and the rows it produces must
match the site.

The data reach the guide through `Guide.observe`, which every objective
calls with the observations of the step it evaluates.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
import torch.nn as nn
from torch.distributions import constraints as _constraints

from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    SampledComposition,
)
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import ContinuousSpace
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentSite


def _event_shape(space: AnySpace) -> tuple[int, ...]:
    """The event shape of a morphism's domain or codomain.

    Parameters
    ----------
    space : AnySpace
        The space; a `ContinuousSpace` has an event shape, a finite set
        is indexed by scalars.

    Returns
    -------
    tuple[int, ...]
        The event shape, ``()`` for a finite set.
    """
    if isinstance(space, ContinuousSpace):
        return tuple(space.event_shape)
    return ()


def _same_support(a: _constraints.Constraint, b: _constraints.Constraint) -> bool:
    """Whether two supports are the same set.

    Parameters
    ----------
    a, b : Constraint
        The supports.

    Returns
    -------
    bool
        ``True`` when they are the same constraint object, or of one
        class with equal bounds.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    for bound in ("lower_bound", "upper_bound"):
        if getattr(a, bound, None) != getattr(b, bound, None):
            return False
    return True


def _check_exact_density(encoder: ContinuousMorphism, inputs: torch.Tensor) -> None:
    """Refuse an encoder whose density is not exact.

    A composition's density is exact when every factor before the last
    is deterministic at the given inputs, or has a discrete codomain
    (summed out exactly); a stochastic continuous intermediate makes
    `SampledComposition.log_prob` score a reference path, not the
    endpoint's density, which a guide cannot use.

    Parameters
    ----------
    encoder : ContinuousMorphism
        The encoder.
    inputs : torch.Tensor
        Rows the encoder reads.

    Raises
    ------
    ValueError
        If a factor before the last is stochastic and continuous.
    """
    if not isinstance(encoder, SampledComposition):
        return
    current = inputs
    for factor in encoder.factors[:-1]:
        image = factor.point_mass_value(current)
        if image is not None:
            current = image
            continue
        if not isinstance(factor.codomain, ContinuousSpace):
            return
        raise ValueError(
            f"AmortizedGuide: the encoder's factor {type(factor).__name__} is "
            "stochastic and continuous, so the composition's log_prob scores a "
            "reference path rather than the density of its draws; compose "
            "deterministic layers before a single stochastic kernel"
        )


class _Binding:
    """One latent site's encoder, the datum it reads, and how its draws
    reach the site.

    Attributes
    ----------
    site : LatentSite
        The site.
    encoder : ContinuousMorphism
        The encoder.
    datum : str
        The observation or host datum the encoder reads.
    through_bijector : bool
        Whether the encoder draws unconstrained coordinates that the
        site's bijector maps onto its support.
    width : int
        The encoder's event size per row.
    """

    __slots__ = ("site", "encoder", "datum", "through_bijector", "width")

    def __init__(
        self,
        site: LatentSite,
        encoder: ContinuousMorphism,
        datum: str,
        through_bijector: bool,
        width: int,
    ) -> None:
        self.site = site
        self.encoder = encoder
        self.datum = datum
        self.through_bijector = through_bijector
        self.width = width


class AmortizedGuide(Guide):
    """Variational guide that runs encoders on the data.

    Parameters
    ----------
    model : MonadicProgram
        The model.
    observed_names : set[str]
        The observed site names, for a registry read off the model.
    encoders : Mapping[str, tuple[ContinuousMorphism, str]]
        For each latent site the guide covers, the encoder and the name
        of the datum (an observation or a host datum) it reads. Other
        latent sites are left to another guide, composed with this one
        in an `AutoGuideList`.
    x : torch.Tensor | None
        Keyword-only. A program input at which to build the registry
        from a run (`Guide.build_registry`).
    observations : dict[str, torch.Tensor] | None
        Keyword-only. The data of that run; when given, the guide also
        observes them (`observe`).

    Raises
    ------
    ValueError
        If ``encoders`` is empty, names a site that is not a latent of
        the registry, or binds an encoder whose support is neither the
        site's nor the real line (for a site with another support), or
        whose event size does not match the site's coordinates.
    TypeError
        If an encoder is not a `ContinuousMorphism`.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        encoders: Mapping[str, tuple[ContinuousMorphism, str]],
        *,
        x: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("AmortizedGuide: encoders must bind at least one site")
        self._registry = self.build_registry(
            model, observed_names, x=x, observations=observations
        )
        unknown = sorted(name for name in encoders if name not in self._registry)
        if unknown:
            raise ValueError(
                f"AmortizedGuide: {unknown} are not latent sites of the model; its "
                f"latent sites are {list(self._registry.names)}"
            )
        self._bindings: dict[str, _Binding] = {}
        modules: list[ContinuousMorphism] = []
        for name, (encoder, datum) in encoders.items():
            if not isinstance(encoder, ContinuousMorphism):
                raise TypeError(
                    f"AmortizedGuide: the encoder of {name!r} is a "
                    f"{type(encoder).__name__}, not a ContinuousMorphism"
                )
            self._bindings[name] = self._bind(self._registry[name], encoder, datum)
            modules.append(encoder)
        # The encoders' weights are the guide's variational parameters.
        self.encoders = nn.ModuleList(modules)
        self._data: dict[str, torch.Tensor] | None = None
        if observations is not None:
            self.observe(observations)

    @staticmethod
    def _bind(site: LatentSite, encoder: ContinuousMorphism, datum: str) -> _Binding:
        """Check one binding against its site.

        Parameters
        ----------
        site : LatentSite
            The site.
        encoder : ContinuousMorphism
            The encoder.
        datum : str
            The datum it reads.

        Returns
        -------
        _Binding
            The checked binding.

        Raises
        ------
        ValueError
            If the encoder's support or event size does not fit the site.
        """
        support = encoder.support
        width = math.prod(_event_shape(encoder.codomain)) or 1
        if _same_support(support, site.support):
            through_bijector = False
            expected = site.constrained_dim
            if site.support is _constraints.corr_cholesky:
                expected = site.constrained_dim * site.constrained_dim
        elif _same_support(support, _constraints.real) or _same_support(
            support, _constraints.real_vector
        ):
            through_bijector = True
            expected = site.unconstrained_dim
        else:
            raise ValueError(
                f"AmortizedGuide: the encoder of {site.name!r} has support "
                f"{support}, and the site's support is {site.support}; the encoder "
                "must have the site's support, or be real-valued so that its draws "
                "pass through the site's bijector"
            )
        if width != expected:
            raise ValueError(
                f"AmortizedGuide: the encoder of {site.name!r} draws {width} "
                f"coordinates per row; the site takes {expected} "
                f"({'unconstrained' if through_bijector else 'constrained'})"
            )
        return _Binding(site, encoder, datum, through_bijector, width)

    @property
    def latent_names(self) -> list[str]:
        """The sites the encoders cover."""
        return list(self._bindings)

    @property
    def uncovered_sites(self) -> tuple[str, ...]:
        """The model's latent sites no encoder covers; another guide in an
        `AutoGuideList` must cover them."""
        return tuple(n for n in self._registry.names if n not in self._bindings)

    def observe(self, observations: Mapping[str, torch.Tensor]) -> None:
        """Take the data the encoders read.

        Every objective calls this with the observations of the step it
        evaluates; call it directly before sampling the guide on other
        data, such as held-out data.

        Parameters
        ----------
        observations : Mapping[str, torch.Tensor]
            Observed values and host data.

        Raises
        ------
        ValueError
            If a datum an encoder reads is missing, or an encoder's
            density is not exact at it.
        """
        missing = sorted({b.datum for b in self._bindings.values()} - set(observations))
        if missing:
            raise ValueError(
                f"AmortizedGuide: the encoders read {missing}, which the "
                "observations do not supply"
            )
        data = {
            b.datum: torch.as_tensor(observations[b.datum])
            for b in self._bindings.values()
        }
        for binding in self._bindings.values():
            rows, _ = self._rows(binding, data[binding.datum])
            _check_exact_density(binding.encoder, rows[:1])
        self._data = data

    def _rows(
        self, binding: _Binding, datum: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The encoder's input rows and the leading shape they come from.

        Parameters
        ----------
        binding : _Binding
            The binding.
        datum : torch.Tensor
            The datum.

        Returns
        -------
        tuple[torch.Tensor, tuple[int, ...]]
            The rows, ``(n, *domain_event)``, and the datum's leading
            shape, whose product is ``n``.

        Raises
        ------
        ValueError
            If the datum's trailing shape is not the encoder's domain.
        """
        event = _event_shape(binding.encoder.domain)
        if tuple(datum.shape[datum.dim() - len(event) :]) != event or (
            datum.dim() < len(event)
        ):
            raise ValueError(
                f"AmortizedGuide: datum {binding.datum!r} has shape "
                f"{tuple(datum.shape)}; the encoder of {binding.site.name!r} reads "
                f"rows of shape {event}"
            )
        leading = tuple(datum.shape[: datum.dim() - len(event)])
        return datum.reshape(-1, *event), leading

    def _expected_leading(
        self, binding: _Binding, batch: int
    ) -> tuple[tuple[int, ...], ...]:
        """The leading shapes a site's draws may have.

        Parameters
        ----------
        binding : _Binding
            The binding.
        batch : int
            The number of runs.

        Returns
        -------
        tuple[tuple[int, ...], ...]
            ``((batch,),)`` for a site drawn once per run; the plate's
            shape (one value the runs share) or that shape behind the
            run axis for a plate site.
        """
        site = binding.site
        if not site.is_plate:
            return ((batch,),)
        plate = site.plate_shape or (site.plate_index_size,)
        return (tuple(plate), (batch, *plate))

    def _datum(
        self, binding: _Binding, batch: int
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """The encoder rows of a binding, checked against the site.

        Parameters
        ----------
        binding : _Binding
            The binding.
        batch : int
            The number of runs.

        Returns
        -------
        tuple[torch.Tensor, tuple[int, ...]]
            The rows and their leading shape.

        Raises
        ------
        RuntimeError
            If the guide has not observed any data.
        ValueError
            If the datum's rows do not line up with the site.
        """
        if self._data is None:
            raise RuntimeError(
                "AmortizedGuide: no data observed; objectives pass the step's "
                "observations, and direct sampling needs guide.observe(data) first"
            )
        rows, leading = self._rows(binding, self._data[binding.datum])
        allowed = self._expected_leading(binding, batch)
        if leading not in allowed:
            raise ValueError(
                f"AmortizedGuide: datum {binding.datum!r} has {leading} rows; site "
                f"{binding.site.name!r} takes {' or '.join(str(a) for a in allowed)} "
                f"for {batch} runs"
            )
        return rows, leading

    def _to_site(self, binding: _Binding, z: torch.Tensor) -> torch.Tensor:
        """Map encoder draws, ``(*leading, width)``, to site values.

        Parameters
        ----------
        binding : _Binding
            The binding.
        z : torch.Tensor
            The draws.

        Returns
        -------
        torch.Tensor
            Trace-side site values.
        """
        site = binding.site
        if binding.through_bijector:
            value = site.bijector(z)
            if not isinstance(value, torch.Tensor):
                raise TypeError("AmortizedGuide: a bijector returned a non-tensor")
        else:
            value = z
            if site.support is _constraints.corr_cholesky:
                d = site.constrained_dim
                value = value.reshape(*value.shape[:-1], d, d)
        if site.constrained_dim == 1 and value.dim() >= 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Draw every bound site from its encoder applied to its datum.

        Parameters
        ----------
        x : torch.Tensor
            The program input; its rows are the runs.

        Returns
        -------
        dict[str, torch.Tensor]
            Trace-side values: ``(B, *constrained_shape)`` for a site
            drawn once per run, the plate's rows (behind the run axis
            when the datum has one) for a plate site.
        """
        batch = int(x.shape[0])
        out: dict[str, torch.Tensor] = {}
        for name, binding in self._bindings.items():
            rows, leading = self._datum(binding, batch)
            draws = binding.encoder.rsample(rows).reshape(*leading, binding.width)
            out[name] = self._to_site(binding, draws)
        return out

    def log_prob(self, x: torch.Tensor, sites: dict[str, torch.Tensor]) -> torch.Tensor:
        """The guide's log-density of site values, per run.

        For a site reached through the bijector :math:`T`,
        :math:`\\log q(v) = \\log e(T^{-1}(v) \\mid y) + \\log
        |\\det J_{T^{-1}}(v)|`; otherwise :math:`\\log e(v \\mid y)`.
        A plate site's rows add up within a run.

        Parameters
        ----------
        x : torch.Tensor
            The program input.
        sites : dict[str, torch.Tensor]
            Trace-side values of the bound sites.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)``.

        Raises
        ------
        KeyError
            If a bound site is missing from ``sites``.
        """
        batch = int(x.shape[0])
        total = torch.zeros(batch, device=x.device)
        for name, binding in self._bindings.items():
            if name not in sites:
                raise KeyError(f"AmortizedGuide.log_prob: missing site {name!r}")
            rows, leading = self._datum(binding, batch)
            site = binding.site
            value = sites[name].reshape(*leading, *self._event(site))
            if binding.through_bijector:
                z = site.bijector.inv(value)
                log_det = site.bijector.inv.log_abs_det_jacobian(value, z)
                log_det = log_det.reshape(*leading, -1).sum(-1)
            else:
                z = value
                log_det = torch.zeros(leading, device=value.device)
            flat = z.reshape(-1, *_event_shape(binding.encoder.codomain))
            log_q = binding.encoder.log_prob(rows, flat).reshape(leading) + log_det
            if not site.is_plate:
                per_run = log_q
            elif len(leading) == len(self._expected_leading(binding, batch)[0]):
                # One value of the plate the runs share.
                per_run = log_q.sum().expand(batch)
            else:
                per_run = log_q.reshape(batch, -1).sum(-1)
            total = total + per_run
        return total

    @staticmethod
    def _event(site: LatentSite) -> tuple[int, ...]:
        """A site's event shape on its support, with the length-one axis
        kept.

        Parameters
        ----------
        site : LatentSite
            The site.

        Returns
        -------
        tuple[int, ...]
            ``(d, d)`` for a correlation Cholesky factor, ``(d,)``
            otherwise.
        """
        if site.support is _constraints.corr_cholesky:
            return (site.constrained_dim, site.constrained_dim)
        return (site.constrained_dim,)


__all__ = ["AmortizedGuide"]
