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

from quivers.continuous.plate import PlateDraw, VectorisedObserve
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


class TraceHandler(EffectHandler):
    """Record every site visited during a program's execution.

    Produces a `Trace` whose ``sites`` dict is keyed by variable name,
    whose ``output`` is the program's return value, and whose
    ``log_joint`` is the sum of every site's ``log_prob``. A call on an
    input of ``B`` rows makes ``B`` independent runs: the trace records
    each run's sites and lays them out along a leading run axis, so a
    site's ``log_prob`` holds one density per run, shape ``(B,)``, and
    its ``value`` holds one value per run (a plated site's rows after
    the run axis). A `TraceHandler` is single-use: run the program under
    one instance, read `trace`, then discard. The handler takes no
    configuration.
    """

    def __init__(self) -> None:
        self.trace: Trace = Trace()
        self._recorder = TraceRecorder()
        self._steps: dict[object, HostStep] = {}
        self._occurrences: dict[object, int] = {}
        self._run_sites: list[dict[str, SampleSite]] = []
        self._run_axes: dict[str, bool] = {}

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
        self._recorder = TraceRecorder()
        self._occurrences = {}
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
        sites: dict[str, SampleSite] = {}
        for event in self._recorder.events:
            if event.operation == RANDOM_SAMPLE:
                self._record_site(run, event, sites)
            else:
                self._record_step(run, event, sites)
        self._run_sites.append(sites)
        if run.run_index == run.runs - 1:
            self._merge_runs()

    def _merge_runs(self) -> None:
        """Lay the recorded runs' sites out along the run axis.

        A site's one-run value that leads with the run axis (any site
        not plated, and any let or score computed from such a value) is
        concatenated along it; any other one-run value gains the run
        axis by stacking. Each run's density of a site is summed to one
        number, so the site's ``log_prob`` is ``(B,)``.

        Raises
        ------
        RuntimeError
            If a value classified as leading with the run axis does not.
        """
        runs = self._run_sites
        for name, first in runs[0].items():
            per_run = [sites[name] for sites in runs]
            if len(per_run) == 1:
                value = first.value
            elif self._run_axes[name]:
                for site in per_run:
                    if site.value.dim() == 0 or int(site.value.shape[0]) != 1:
                        raise RuntimeError(
                            f"site {name!r} leads with the run axis but its value "
                            f"in one run has shape {tuple(site.value.shape)}"
                        )
                value = torch.cat([site.value for site in per_run], dim=0)
            else:
                value = torch.stack([site.value for site in per_run], dim=0)
            log_prob = torch.stack([site.log_prob.sum() for site in per_run])
            self.trace.sites[name] = SampleSite(
                name=name,
                morphism=first.morphism,
                value=value,
                log_prob=log_prob,
                is_observed=any(site.is_observed for site in per_run),
                is_deterministic=first.is_deterministic,
                sampleable=first.sampleable,
                address=first.address,
                metadata=dict(first.metadata),
            )

    def _record_site(
        self, run: RunContext, event: TraceEvent, sites: dict[str, SampleSite]
    ) -> None:
        """Record one sample site from its trace event.

        Parameters
        ----------
        run : RunContext
            The run.
        event : TraceEvent
            The recorded request and answer.
        sites : dict[str, SampleSite]
            The run's sites, which the site joins.
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
        self._run_axes[label] = not isinstance(
            sampleable.declared, PlateDraw | VectorisedObserve
        )
        sites[label] = SampleSite(
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

    def _record_step(
        self, run: RunContext, event: TraceEvent, sites: dict[str, SampleSite]
    ) -> None:
        """Record one let or score step from its trace event.

        Parameters
        ----------
        run : RunContext
            The run.
        event : TraceEvent
            The recorded request and answer.
        sites : dict[str, SampleSite]
            The run's sites, which the binding joins.
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
        self._run_axes[name] = run.carries_run_axis(step.program, step.spec.name)
        sites[name] = SampleSite(
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

    def total_log_joint(self, device: torch.device) -> torch.Tensor:
        """Sum every site's density into the joint, one per run.

        Parameters
        ----------
        device : torch.device
            Device the accumulator seeds on.

        Returns
        -------
        torch.Tensor
            The joint log-density of each run, shape ``(B,)``.
        """
        runs = len(self._run_sites)
        total = torch.zeros(runs, device=device)
        for site in self.trace.sites.values():
            total = total + site.log_prob
        return total


__all__ = ["TraceHandler"]
