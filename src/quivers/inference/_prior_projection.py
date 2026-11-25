"""Project native programs onto requested draws and their lexical dependencies."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from quivers.continuous.program_steps import (
    Call,
    Draw,
    Indexed,
    Let,
    Observe,
    Scan,
    Score,
    reads_of,
)
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean


def _labels(program: MonadicProgram) -> set[str]:
    """Collect draw labels recursively without executing the model.

    Parameters
    ----------
    program : MonadicProgram
        Native lexical program.

    Returns
    -------
    set[str]
        Base labels; scans add occurrence suffixes only during execution.
    """
    labels = set()
    for step in program.steps:
        if isinstance(step, Draw):
            labels.update(step.names)
        elif isinstance(step, Call) and isinstance(step.program, MonadicProgram):
            labels.update(_labels(step.program))
        elif isinstance(step, Scan) and isinstance(step.step, MonadicProgram):
            labels.update(_labels(step.step))
    return labels


def _score_steps(program: MonadicProgram) -> Iterator[Score]:
    """Read typed score specifications through lexical call and scan scopes.

    Parameters
    ----------
    program : MonadicProgram
        Native program whose score roles remain attached to their steps.

    Yields
    ------
    Score
        Each score specification in lexical execution order.
    """
    for step in program.steps:
        if isinstance(step, Score):
            yield step
        elif isinstance(step, Call) and isinstance(step.program, MonadicProgram):
            yield from _score_steps(step.program)
        elif isinstance(step, Scan) and isinstance(step.step, MonadicProgram):
            yield from _score_steps(step.step)


def _has_potential(program: MonadicProgram) -> bool:
    """Check nested scopes for explicit prior potentials.

    Parameters
    ----------
    program : MonadicProgram
        Native lexical program.

    Returns
    -------
    bool
        Whether projecting this scope must preserve a potential contribution.
    """
    return any(step.role == "potential" for step in _score_steps(program))


def _argument_names(arguments) -> set[str]:
    """Read names from ordinary and indexed step arguments.

    Parameters
    ----------
    arguments : Iterable or None
        Step arguments, including optional numeric constants.

    Returns
    -------
    set[str]
        Names needed before evaluating the arguments.
    """
    names = set()
    for argument in arguments or ():
        if isinstance(argument, str):
            names.add(argument)
        elif isinstance(argument, Indexed):
            names.update((argument.name, *argument.indices))
    return names


def project_prior_sites(
    model: MonadicProgram, sites: Iterable[str] | None = None
) -> MonadicProgram:
    """Retain requested draws, upstream values, and explicit prior potentials.

    This is a dependency projection, not a change to any conditional draw
    law. Observation scoring is controlled separately by the caller. An
    ancestral sampler must additionally refuse active global potentials
    when it cannot normalize them. Unknown callable dependencies retain
    all preceding bindings conservatively.

    Parameters
    ----------
    model : MonadicProgram
        Native program whose morphisms and input contracts remain shared.
    sites : Iterable[str] or None
        Requested latent labels; None selects every draw. Scan occurrence
        addresses retain their original matching scan executions. A caller
        requesting a particular occurrence must check it exists in the trace.

    Returns
    -------
    MonadicProgram
        A program returning an inert scalar and retaining each required step
        in its original order. Calls preserve used outputs and scans preserve
        their state feedback; unrelated downstream observations are omitted.

    Raises
    ------
    ValueError
        If a requested base label is absent from the native program.
    """
    available = _labels(model)
    selected = (
        available
        if sites is None
        else {
            name.rsplit("@", 1)[0] if name.rpartition("@")[2].isdigit() else name
            for name in sites
        }
    )
    if missing := selected - available:
        raise ValueError(f"unknown prior sites {sorted(missing)}")
    projected, _ = _project(model, selected, keep_return=False)
    return projected


def _project(model, selected, *, keep_return):
    """Compute one backward lexical dependency closure.

    Parameters
    ----------
    model : MonadicProgram
        One lexical scope.
    selected : set[str]
        Requested base labels across enclosing scopes.
    keep_return : bool
        Preserve the original output when a caller or scan uses it.

    Returns
    -------
    tuple[MonadicProgram, set[str]]
        Projected scope and the free bindings it still reads.
    """
    bound_before = []
    bound: set[str] = set(model._params or ())
    for step in model.steps:
        bound_before.append(set(bound))
        bound.update(step.names if isinstance(step, Draw | Observe) else (step.name,))
    needed: set[str] = set(model._return_vars) if keep_return else set()
    retained = []
    for step, preceding in reversed(tuple(zip(model.steps, bound_before, strict=True))):
        names = set(step.names) if isinstance(step, Draw | Observe) else {step.name}
        required = bool(names & needed)
        dependencies = set()
        if isinstance(step, Draw | Observe):
            required |= isinstance(step, Draw) and bool(names & selected)
            if required:
                dependencies.update(_argument_names(step.args))
                if isinstance(step, Observe):
                    dependencies.update(_argument_names((step.weights,)))
                    if step.event is not None:
                        dependencies.update(step.event.names())
                    # An observed value may be a preceding lexical binding.
                    dependencies.update(names & preceding)
        elif isinstance(step, Let | Score):
            required |= isinstance(step, Score) and step.role == "potential"
            if required:
                value = step.value if isinstance(step, Let) else step.score
                if isinstance(value, str):
                    dependencies.add(value)
                elif callable(value):
                    reads = reads_of(value)
                    dependencies.update(preceding if reads is None else reads)
                if isinstance(step, Score) and step.role == "likelihood":
                    # A conditional prior may consume the score's value even
                    # though its likelihood contribution is disabled.
                    step = Let(name=step.name, value=step.score, batched=step.batched)
        elif isinstance(step, Call):
            if isinstance(step.program, MonadicProgram):
                required |= bool(_labels(step.program) & selected) or _has_potential(
                    step.program
                )
                if required:
                    child, free = _project(
                        step.program, selected, keep_return=bool(names & needed)
                    )
                    step = step.with_(program=child)
                    dependencies.update(free)
                    dependencies.update(step.args)
        elif isinstance(step, Scan):
            if isinstance(step.step, MonadicProgram):
                required |= bool(_labels(step.step) & selected) or _has_potential(
                    step.step
                )
                if required:
                    child, free = _project(step.step, selected, keep_return=True)
                    step = step.with_(step=child)
                    dependencies.update(free)
                    dependencies.update(
                        name
                        for name in (step.source, step.initial_from)
                        if name is not None
                    )
        if required:
            retained.append(step)
            needed.difference_update(names)
            needed.update(dependencies)
    retained.reverse()
    returns = model._return_vars
    codomain = model.codomain
    if not keep_return:
        name = "__qvr_prior_projection_result"
        while name in bound:
            name += "_"
        retained.append(Let(name=name, value=0.0))
        returns = (name,)
        codomain = Euclidean(name="_prior_projection", dim=1)
    projected = MonadicProgram(
        model.domain,
        codomain,
        retained,
        returns,
        params=model._params,
        return_labels=model._return_labels if keep_return else None,
        effect_set=model.effect_set,
    )
    for attribute in (
        "host_data_types",
        "input_contract",
        "host_index_bounds",
        "host_data_shared",
        "host_data_notes",
        "site_plates",
    ):
        setattr(projected, attribute, getattr(model, attribute))
    active_inputs = needed | set(model._params or ())
    projected.host_data_types = {
        name: type_
        for name, type_ in model.host_data_types.items()
        if name in active_inputs
    }
    projected.host_index_bounds = {
        name: bounds
        for name, bounds in model.host_index_bounds.items()
        if name in active_inputs
    }
    return projected, needed - set(model._params or ())


__all__ = []
