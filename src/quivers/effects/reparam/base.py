"""Reparameterization strategies as effect handlers.

A `Reparam` strategy rewrites a sample site: given the site's sampleable
it produces the value the site takes and the density the site
contributes, according to a bijection that preserves the induced
distribution while reshaping the geometry downstream inference sees.
The canonical use case is HMC on funnel geometries: reparameterising
``y ~ Normal(0, exp(z))`` into ``y_raw ~ Normal(0, 1); y = exp(z) *
y_raw`` decouples the tight scale-location dependency that would
otherwise kill NUTS
([Betancourt and Girolami 2015](https://arxiv.org/abs/1312.0906)).

The `ReparamOrchestrator` handler of the program's ``random`` instance
answers each covered site with its strategy's value and emits the
strategy's density as the site's score; every other site forwards, its
sampleable remembered so a strategy of a later site may read it. A user
writes

    with reparam({"theta": LocScaleReparam(), "z": NeuTraReparam(guide)}):
        samples = nuts.run(model, x, observations)

and has per-site strategies apply in one pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.effects.sites import TorchSampleable
from quivers.qiec.builtins import (
    RANDOM,
    RANDOM_SAMPLE,
    SCORE,
    _emit_score,
    _handler_def,
    _site_and_sampleable,
)
from quivers.qiec.effects import EffectRow, ResumptionGrade, RowEntry
from quivers.qiec.evaluator import (
    ClauseContext,
    Forward,
    Resumption,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
)


class SiteRequest(ValueModel):
    """What a strategy is given for one sample site.

    Parameters
    ----------
    name
        The site's name.
    sampleable
        The site's distribution at its input.
    given
        A value the site is to take, when an enclosing strategy or a
        caller fixed one; ``None`` when the strategy draws.
    seen
        The sampleables of the sites the orchestrator saw before this
        one, by name, so a strategy may read a parent's parameters.
    observation
        The run's observation for the site, if any.
    """

    name: str
    sampleable: TorchSampleable = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    given: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    seen: Mapping[str, TorchSampleable] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observation: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @dx.model_validator()
    def _check_runtime_fields(self) -> None:
        """Validate sampleables, optional tensors, and the preceding-site mapping.

        Raises
        ------
        TypeError
            If a runtime field has an incompatible type.
        """
        if not isinstance(self.sampleable, TorchSampleable):
            raise TypeError("site sampleable must be a TorchSampleable")
        if any(
            value is not None and not isinstance(value, torch.Tensor)
            for value in (self.given, self.observation)
        ):
            raise TypeError("site given and observation must be tensors or None")
        if not isinstance(self.seen, Mapping) or not all(
            isinstance(name, str) and isinstance(value, TorchSampleable)
            for name, value in self.seen.items()
        ):
            raise TypeError("site seen must map names to TorchSampleable values")

    def with_value(self, value: torch.Tensor) -> SiteRequest:
        """The same request with a value fixed.

        Parameters
        ----------
        value : torch.Tensor
            The value the site is to take.

        Returns
        -------
        SiteRequest
            The request with ``given`` set.
        """
        return self.with_(given=value)


class Reparam(ABC):
    """Base class for site-level reparameterisation strategies.

    A strategy is a *policy*, not a handler. The `ReparamOrchestrator`
    handler dispatches each covered site to the matching strategy and
    installs the strategy's value and density on the site. Subclasses
    override :meth:`apply`.
    """

    @abstractmethod
    def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite one site.

        Parameters
        ----------
        site : SiteRequest
            The site, with a value fixed when one is given.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The value the site takes, in the site's original space so
            downstream sites see the intended value, and the density a
            downstream sampler should score the site with.
        """


class ReparamOrchestrator(EffectHandler):
    """Dispatch sample sites to per-site reparam strategies.

    Parameters
    ----------
    strategies : dict[str, Reparam]
        Site name to strategy.
    """

    def __init__(self, strategies: dict[str, Reparam]) -> None:
        self.strategies = dict(strategies)

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install the dispatching handler of the ``random`` instance.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The handler, which answers covered sites and forwards the
            rest.
        """
        kernel = run.kernel
        score_instance = kernel.score.entry.instance
        seen: dict[str, TorchSampleable] = {}
        definition = _handler_def(
            "Random.reparam",
            RANDOM,
            ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
            key=f"reparam-{id(self):x}",
            input_type=run.result_type,
            output_type=run.result_type,
            introduced=EffectRow(
                entries=(RowEntry(instance=score_instance, effect=SCORE),)
            ),
            total=False,
        )

        def sample(
            request: RuntimeRequest, resume: Resumption, context: ClauseContext
        ) -> object:
            """Answer a covered site with its strategy's value and density.

            Parameters
            ----------
            request : RuntimeRequest
                The request being answered and its arguments.
            resume : Resumption
                The continuation.
            context : ClauseContext
                Runtime services available to the clause.

            Returns
            -------
            object
                What the resumed computation produced, or a `Forward`
                for a site no strategy covers.
            """
            site, sampleable = _site_and_sampleable(request, definition.name)
            assert isinstance(site, str)
            assert isinstance(sampleable, TorchSampleable)
            strategy = self.strategies.get(site)
            if strategy is None:
                seen[site] = sampleable
                return Forward()
            value, log_prob = strategy.apply(
                SiteRequest(
                    name=site,
                    sampleable=sampleable,
                    seen=dict(seen),
                    observation=run.observations.get(site),
                )
            )
            seen[site] = sampleable
            _emit_score(
                context,
                request,
                score_instance,
                log_prob,
                run.validator,
                role="draw-score",
            )
            return resume(value)

        runtime = RuntimeHandler(
            definition,
            {
                RANDOM_SAMPLE: RuntimeClause(
                    invoke=sample, result_validator=run.validator
                )
            },
        )
        return (Installation(kernel.random, runtime),)


def reparam(strategies: dict[str, Reparam]) -> ReparamOrchestrator:
    """Return a `ReparamOrchestrator` for the given per-site strategies.

    Parameters
    ----------
    strategies : dict[str, Reparam]
        Site name to strategy.

    Returns
    -------
    ReparamOrchestrator
        The handler.
    """
    return ReparamOrchestrator(strategies)


__all__ = ["Reparam", "ReparamOrchestrator", "SiteRequest", "reparam"]
