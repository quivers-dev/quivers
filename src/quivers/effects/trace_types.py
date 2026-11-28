"""Data types for execution traces.

`SampleSite` and `Trace` live here in a leaf module so both the
effect-handler interpreter and the thin `quivers.inference.trace`
wrapper import them without introducing a cycle. The effects
package produces traces; the inference package consumes them.
"""

from __future__ import annotations

import didactic.api as dx
import torch

from quivers._values import HeldValue, OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.effects.sites import TorchSampleable
from quivers.effects.units import SiteUnits
from quivers.qiec.structured_sampleables import FiniteSampleable

type SiteAddress = tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
"""The kernel's dynamic address of one occurrence of a site."""


class SampleSite(ValueModel):
    """Record of a single site in a program trace.

    Values and morphisms retain their runtime identity. Field updates
    use ``with_`` and preserve their autograd connections.

    Parameters
    ----------
    name : str
        Variable name bound at this site.
    morphism : ContinuousMorphism or None
        The distribution morphism (``None`` for let bindings and scores).
    value
        The sampled, observed, or computed tensor. A site drawn from a
        finite structured law keeps its one typed point as a value.
    log_prob : torch.Tensor
        The density the site contributed to the run's score, after every
        score transformer on the stack. Zero for let bindings and for
        intervened sites.
    is_observed : bool
        Whether this site was conditioned on an observed value.
    is_deterministic : bool
        Whether this is a let binding, a score step, or an intervened
        site.
    sampleable : TorchSampleable or FiniteSampleable or None
        The sampleable the site's request carried. A tensor law records
        its morphism and input. A structured law records its typed
        support. Let bindings and scores have no sampleable.
    address : SiteAddress or None
        The kernel's dynamic address of the site's request, which is one
        identity for the run's trace and for every handler that answered
        it.
    metadata : dict[str, HeldValue]
        Annotations handlers attached to the site.
    units : tuple[SiteUnits, ...]
        For a score step that declares a factorization, its units in each
        of the runs the trace holds; empty for any other site.
    """

    name: str
    morphism: ContinuousMorphism | None = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    value: object = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    log_prob: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    is_observed: bool = False
    is_deterministic: bool = False
    sampleable: TorchSampleable | FiniteSampleable | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    address: SiteAddress | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    metadata: dict[str, HeldValue] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    units: tuple[SiteUnits, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_runtime_fields(self) -> None:
        """Require the runtime objects a trace site exposes.

        Raises
        ------
        TypeError
            If a tensor, morphism, sampleable, or metadata field is malformed.
        """
        if self.morphism is not None and not isinstance(
            self.morphism, ContinuousMorphism
        ):
            raise TypeError("site morphism must be ContinuousMorphism or None")
        if isinstance(self.sampleable, FiniteSampleable):
            if self.morphism is not None:
                raise TypeError("a structured trace site carries no tensor morphism")
            self.sampleable._key(self.value)
            if not isinstance(self.log_prob, torch.Tensor):
                raise TypeError(
                    "a structured trace site's log_prob must be torch.Tensor"
                )
        else:
            if not isinstance(self.value, torch.Tensor) or not isinstance(
                self.log_prob, torch.Tensor
            ):
                raise TypeError("site value and log_prob must be torch.Tensor")
            if self.sampleable is not None and not isinstance(
                self.sampleable, TorchSampleable
            ):
                raise TypeError("site sampleable must be a typed sampleable or None")
        if self.address is not None:
            if not isinstance(self.address, tuple) or len(self.address) != 3:
                raise TypeError("site address must contain a site, frames, and path")
            static, frames, path = self.address
            if (
                not isinstance(static, str)
                or not isinstance(frames, tuple)
                or not all(
                    isinstance(frame, tuple)
                    and len(frame) == 2
                    and isinstance(frame[0], str)
                    and isinstance(frame[1], (str, int))
                    for frame in frames
                )
                or not isinstance(path, tuple)
                or not all(isinstance(index, int) for index in path)
            ):
                raise TypeError("site address must contain a site, frames, and path")
        if not isinstance(self.metadata, dict) or not all(
            isinstance(name, str) for name in self.metadata
        ):
            raise TypeError("site metadata must be a dictionary with string keys")

    @property
    def is_structured(self) -> bool:
        """Whether the site's value is a single point of a structured law.

        Returns
        -------
        bool
            True when the sampleable is a finite typed law.
        """
        return isinstance(self.sampleable, FiniteSampleable)


class Trace(ValueModel):
    """Complete execution trace of a monadic program.

    The ``sites`` mapping grows as the program executes. Finalizing a
    trace uses ``with_`` to attach its output and joint density.

    Parameters
    ----------
    sites : dict[str, SampleSite]
        All sites keyed by variable name.
    output : torch.Tensor or dict[str, torch.Tensor]
        The program's return value.
    log_joint : torch.Tensor
        Sum of log-densities across all sites. Shape ``(batch,)``.
    """

    sites: dict[str, SampleSite] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    output: torch.Tensor | dict[str, torch.Tensor] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    log_joint: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _validate_runtime_fields(self) -> None:
        """Require named sites and tensor-valued outputs and scores.

        Raises
        ------
        TypeError
            If the trace contains a malformed runtime field.
        """
        if not isinstance(self.sites, dict) or not all(
            isinstance(name, str) and isinstance(site, SampleSite)
            for name, site in self.sites.items()
        ):
            raise TypeError("trace sites must map strings to SampleSite values")
        if self.output is not None and not isinstance(self.output, torch.Tensor):
            if not isinstance(self.output, dict) or not all(
                isinstance(name, str) and isinstance(value, torch.Tensor)
                for name, value in self.output.items()
            ):
                raise TypeError(
                    "trace output must be a tensor, tensor dictionary, or None"
                )
        if self.log_joint is not None and not isinstance(self.log_joint, torch.Tensor):
            raise TypeError("trace log_joint must be torch.Tensor or None")

    @property
    def stochastic_sites(self) -> dict[str, SampleSite]:
        """The sites that are not deterministic.

        Returns
        -------
        dict[str, SampleSite]
            Every sampled or observed site.
        """
        return {k: v for k, v in self.sites.items() if not v.is_deterministic}

    @property
    def latent_sites(self) -> dict[str, SampleSite]:
        """The sites that are neither observed nor deterministic.

        Returns
        -------
        dict[str, SampleSite]
            Every sampled latent site.
        """
        return {
            k: v
            for k, v in self.sites.items()
            if not v.is_observed and not v.is_deterministic
        }

    @property
    def observed_sites(self) -> dict[str, SampleSite]:
        """The observed sites.

        Returns
        -------
        dict[str, SampleSite]
            Every site conditioned on an observation.
        """
        return {k: v for k, v in self.sites.items() if v.is_observed}


__all__ = ["SampleSite", "SiteAddress", "Trace"]
