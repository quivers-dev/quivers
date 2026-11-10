"""Trace handler: records every site the program visits.

`TraceHandler` installs an observing handler of the program's ``random``
instance and of its let and score steps at its position on the stack, so
it records each site's value as the handlers outside it answered it and
sees no site a handler inside it hid. A site's density is read after the
run from the contributions that reached the run's accumulator under the
site's provenance, after every score transformer on the stack. The thin
`quivers.inference.trace.trace` wrapper stacks this handler on and returns
its accumulated `Trace`.
"""

from __future__ import annotations

import torch

from quivers.continuous.program_steps import Let, Score
from quivers.effects.base import EffectHandler, Installation, RunContext
from quivers.effects.program_module import HostStep
from quivers.effects.sites import TorchSampleable
from quivers.effects.trace_types import SampleSite, Trace
from quivers.qiec.builtins import (
    COMPUTE_APPLY,
    RANDOM,
    RANDOM_SAMPLE,
    TraceEvent,
    TraceRecorder,
    trace_handler,
)


def _narrowest_shape(log_probs: list[torch.Tensor]) -> tuple[int, ...]:
    """The widest shape no site has to be replicated into.

    Shapes are right-aligned (a missing leading axis reads as extent
    1, exactly as broadcasting reads it) and the extent kept for each
    axis is the smallest any site carries there. Adding sites at that
    shape can only ever reduce a site, never repeat one.

    Parameters
    ----------
    log_probs : list of torch.Tensor
        Per-site log-densities that carry density. An empty list has
        no axis to agree on and gives the scalar shape.

    Returns
    -------
    tuple of int
        Target shape, right-aligned against every site.
    """
    if not log_probs:
        return ()
    rank = max(log_prob.dim() for log_prob in log_probs)
    extents: list[int] = []
    for axis in range(rank):
        smallest = None
        for log_prob in log_probs:
            offset = rank - log_prob.dim()
            here = 1 if axis < offset else int(log_prob.shape[axis - offset])
            smallest = here if smallest is None else min(smallest, here)
        assert smallest is not None
        extents.append(smallest)
    return tuple(extents)


def _reduce_to(log_prob: torch.Tensor, target: tuple[int, ...]) -> torch.Tensor:
    """Sum ``log_prob`` over every axis where it is wider than ``target``.

    The reduction keeps the axis at extent 1 rather than dropping it,
    so the result still lines up with ``target`` under the addition
    that follows and no term is repeated by that addition. An axis
    ``target`` does not reach at all belongs to a site of higher rank
    than any density-carrying one, and is summed away outright rather
    than kept, so it cannot widen the joint either.

    Parameters
    ----------
    log_prob : torch.Tensor
        One site's log-density.
    target : tuple of int
        Shape from
        [`_narrowest_shape`][quivers.effects.trace_handler._narrowest_shape],
        right-aligned against ``log_prob``.

    Returns
    -------
    torch.Tensor
        ``log_prob`` with its lane axes summed out.
    """
    rank = len(target)
    while log_prob.dim() > rank:
        log_prob = log_prob.sum(dim=0)
    offset = rank - log_prob.dim()
    axes = [
        axis - offset
        for axis in range(rank)
        if axis >= offset and int(log_prob.shape[axis - offset]) > target[axis]
    ]
    if not axes:
        return log_prob
    return log_prob.sum(dim=axes, keepdim=True)


class TraceHandler(EffectHandler):
    """Record every site visited during a program's execution.

    Produces a `Trace` whose ``sites`` dict is keyed by variable name,
    whose ``output`` is the program's return value, and whose
    ``log_joint`` is the sum of every site's ``log_prob``. A
    `TraceHandler` is single-use: run the program under one instance,
    read `trace`, then discard. The handler takes no configuration.
    """

    def __init__(self) -> None:
        self.trace: Trace = Trace()
        self._recorder = TraceRecorder()
        self._steps: dict[object, HostStep] = {}
        self._occurrences: dict[object, int] = {}

    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Install observers of the sites and of the let and score steps.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            One forwarding observer of the ``random`` instance and one of
            each let and score step's ``Compute`` instance.
        """
        kernel = run.kernel
        installations = [
            Installation(
                kernel.random,
                trace_handler(
                    RANDOM,
                    (RANDOM_SAMPLE,),
                    self._recorder,
                    validators={RANDOM_SAMPLE: run.validator},
                    answer_type=run.result_type,
                    key=f"trace-random-{id(self):x}",
                ),
            )
        ]
        for step in kernel.steps:
            if step.kind not in ("let", "score"):
                continue
            instance = next(
                item
                for item in kernel.module.instances
                if item.entry.instance == step.instance
            )
            self._steps[step.instance] = step
            installations.append(
                Installation(
                    instance,
                    trace_handler(
                        step.effect,
                        (COMPUTE_APPLY,),
                        self._recorder,
                        validators={COMPUTE_APPLY: run.validator},
                        answer_type=run.result_type,
                        key=f"trace-step-{instance.name}-{id(self):x}",
                    ),
                )
            )
        return tuple(installations)

    def finish(self, run: RunContext, output: object) -> None:
        """Build the trace from the recorded events and the run's contributions.

        Parameters
        ----------
        run : RunContext
            The run, with every contribution the accumulator received.
        output : object
            The program's output, which the trace does not read; the
            wrapper that stacks this handler sets ``trace.output`` from
            the value it returns.
        """
        del output
        for event in self._recorder.events:
            if event.operation == RANDOM_SAMPLE:
                self._record_site(run, event)
            else:
                self._record_step(run, event)

    def _record_site(self, run: RunContext, event: TraceEvent) -> None:
        """Record one sample site from its trace event.

        Parameters
        ----------
        run : RunContext
            The run.
        event : TraceEvent
            The recorded request and answer.
        """
        label, sampleable = event.arguments
        assert isinstance(label, str)
        assert isinstance(sampleable, TorchSampleable)
        value = event.result
        assert isinstance(value, torch.Tensor)
        contributions = [
            item for item in run.contributions if item.key == event.address
        ]
        observed = any(item.role == "condition-score" for item in contributions)
        intervened = event.address in run.interventions
        if contributions:
            log_prob = contributions[0].weight
            for item in contributions[1:]:
                log_prob = log_prob + item.weight
        else:
            log_prob = torch.zeros(
                value.shape[:1] if value.dim() > 0 else (1,), device=value.device
            )
        self.trace.sites[label] = SampleSite(
            name=label,
            morphism=sampleable.declared,
            value=value,
            log_prob=log_prob,
            is_observed=observed,
            is_deterministic=intervened,
            sampleable=sampleable,
            address=event.address,
            metadata=dict(run.annotations.get(label, {})),
        )

    def _record_step(self, run: RunContext, event: TraceEvent) -> None:
        """Record one let or score step from its trace event.

        Parameters
        ----------
        run : RunContext
            The run.
        event : TraceEvent
            The recorded request and answer.
        """
        step = self._steps[event.instance]
        assert isinstance(step.spec, Let | Score)
        name = step.spec.name
        if step.scanned:
            # A binding a scan's step makes once per position is
            # recorded per occurrence, as its sites are.
            occurrence = self._occurrences.get(event.instance, 0)
            self._occurrences[event.instance] = occurrence + 1
            name = f"{name}@{occurrence}"
        value = event.result
        if step.kind != "score" and isinstance(value, tuple):
            # A let binding a tuple holds a product of values rather
            # than one tensor; the trace records tensor-valued bindings,
            # and a tuple's components are bindings of their own once a
            # let projects them.
            return
        assert isinstance(value, torch.Tensor)
        if step.kind == "score":
            log_prob = self._step_contribution(run, step, event, value)
        else:
            log_prob = torch.zeros((), device=value.device)
        self.trace.sites[name] = SampleSite(
            name=name,
            morphism=None,
            value=value,
            log_prob=log_prob,
            is_deterministic=True,
            address=event.address,
        )

    def _step_contribution(
        self,
        run: RunContext,
        step: HostStep,
        event: TraceEvent,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """The contribution a score step's ``Score.add`` made, as accumulated.

        Parameters
        ----------
        run : RunContext
            The run.
        step : HostStep
            The score step's host part.
        event : TraceEvent
            The host part's recorded request, whose dynamic address the
            ``Score.add`` of the same occurrence shares.
        value : torch.Tensor
            The step's value, which shapes the zero returned when the
            contribution never reached the accumulator.

        Returns
        -------
        torch.Tensor
            The contribution after every transformer on the stack.
        """
        frames = event.address[1]
        total: torch.Tensor | None = None
        for item in run.contributions:
            if tuple(item.path[:3]) == ("steps", step.position, "score") and (
                item.key[1] == frames
            ):
                total = item.weight if total is None else total + item.weight
        if total is None:
            return torch.zeros_like(value)
        return total

    def total_log_joint(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sum every site's ``log_prob`` into the joint density.

        The accumulator uses the narrowest right-aligned shape shared
        by the nonzero site contributions. Wider axes are summed before
        addition so broadcasting cannot duplicate scalar terms. Zero
        contributions do not determine the target shape but are still
        added.

        Parameters
        ----------
        batch_size : int
            Leading extent of the program input. Not consulted: the
            joint's shape follows from the sites themselves, and a
            program whose sites reduce their own plate axes has a
            joint narrower than its input.
        device : torch.device
            Device the accumulator seeds on.

        Returns
        -------
        torch.Tensor
            The joint log-density.
        """
        del batch_size
        contributions = [site.log_prob for site in self.trace.sites.values()]
        target = _narrowest_shape(
            [lp for lp in contributions if not bool(torch.all(lp == 0))]
        )
        total = torch.zeros((), device=device)
        for log_prob in contributions:
            total = total + _reduce_to(log_prob, target)
        return total


__all__ = ["TraceHandler"]
