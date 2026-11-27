"""Effect handlers for probabilistic programs, as lexical kernel handlers.

A `MonadicProgram` runs on the reference machine as the kernel
computation :mod:`quivers.effects.program_module` encodes it to, whose
sites are ``Random.sample`` requests on the program's ``random`` instance
and whose densities are ``Score.add`` requests on its ``score`` instance.
An `EffectHandler` is a description of a handler of one of those
instances: entering it as a context manager pushes it on a thread-local
stack, and `quivers.effects.interpreter.run_program` installs every
handler on the stack as a lexical handler around the program, in the
order the stack gives, so the composition rules are the kernel's own.
The design descends from the Pyro ``poutine`` and NumPyro ``handlers``
shape ([Pyro poutine docs](https://docs.pyro.ai/en/stable/poutine.html))
and from the operational account of handlers in
[Plotkin and Pretnar (2009)](https://doi.org/10.1007/978-3-642-00590-9_7);
its application to probabilistic programming follows
[Scibior et al. (2018)](https://doi.org/10.1145/3236778).

Two rules govern how a stack composes. A handler of the ``random``
instance sees a site's request before any handler outside it, and sees
the outer answer flow back through its resumption, so an inner trace
records what an outer clamp or intervention decided. The density a site
contributes is a ``Score.add`` the answering handler emits outward, so
every handler of the ``score`` instance on the stack transforms every
site's density on its way to the run's accumulator, and a trace reads a
site's density from what reached the accumulator under that site's
provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from functools import lru_cache
import threading
from typing import Self

import didactic.api as dx
import torch

from quivers._values import HeldValue, OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel

from quivers.continuous.program_steps import RunConditions
from quivers.continuous.programs import MonadicProgram
from quivers.effects.program_module import ProgramKernel
from quivers.qiec.effects import HandlerDef
from quivers.qiec.evaluator import RuntimeHandler, RuntimeRequest, RuntimeValidator
from quivers.qiec.identifiers import HandlerId
from quivers.qiec.module import NamedEffectInstance
from quivers.qiec.terms import Computation, Handle
from quivers.qiec.types import TypeExpr

_LOCAL = threading.local()


def _handler_stack() -> list[EffectHandler]:
    """Return the current thread's active handler stack.

    Returns
    -------
    list[EffectHandler]
        The stack, ordered outer-first; a fresh list on first access per
        thread. Every handler's `__enter__` appends to it and every
        `__exit__` pops it.
    """
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    return stack


type SiteKey = tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
"""The dynamic address of one occurrence of a site: its static identity,
its dynamic address frames, and its resumption path."""


class Contribution(QiecModel):
    """One score contribution that reached the run's accumulator.

    The address and path compare by value; the tensor retains identity.

    Parameters
    ----------
    key
        The address of the site the contribution was derived from.
    role
        What the contribution was for, from the emitting handler's
        provenance: a scored draw, a conditioned observation, a replay.
    weight
        The contribution as it arrived, after every transformer.
    path
        The structural path of the request's origin; a score step of the
        program has the path of its step, a derived contribution the
        path it was generated at.
    derived
        Whether a handler derived the contribution from a site (the
        density of a draw, an observation, a replay), rather than the
        program scoring it.
    """

    key: SiteKey = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    role: str
    weight: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    path: tuple[str | int, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    derived: bool = False


def _shared_value(label: str, value: torch.Tensor) -> torch.Tensor:
    """The value a run of a single-run call takes: the value itself.

    Parameters
    ----------
    label : str
        The site label, unused.
    value : torch.Tensor
        The value.

    Returns
    -------
    torch.Tensor
        ``value``.
    """
    del label
    return value


def _as_given(value: object, sampleable: object) -> object:
    """A value as it is given, already the run's.

    Parameters
    ----------
    value : object
        The value.
    sampleable : object
        The site's sampleable, unused.

    Returns
    -------
    object
        The value.
    """
    del sampleable
    return value


def _no_run_axis(program: MonadicProgram, name: str) -> bool:
    """The default run-axis classifier, which marks no binding.

    Parameters
    ----------
    program : MonadicProgram
        The program, unused.
    name : str
        The binding, unused.

    Returns
    -------
    bool
        ``False``.
    """
    del program, name
    return False


def _no_site_run_axis(label: str) -> bool:
    """The default site classifier, which marks no site.

    Parameters
    ----------
    label : str
        The site's label, unused.

    Returns
    -------
    bool
        ``False``.
    """
    del label
    return False


class RunContext(QiecModel):
    """What one run shares with the handlers installed on it.

    The checked kernel compares by value. Mutable per-run collections and
    executable callbacks retain identity, including the shared parameter store.

    Parameters
    ----------
    kernel
        The program's kernel encoding.
    validator
        The host value validator every handler's clauses check with.
    contributions
        The score contributions that reached the accumulator, in order.
    interventions
        The sites an intervention fixed, by address.
    fixed
        The labels of the same sites, an occurrence of a site a scan
        reaches once per position as ``"<label>@<n>"``.
    annotations
        Per-site annotations handlers attach, keyed by site label.
    parameters
        The learned parameters of the program's morphisms by qualified
        name, as the parameter store answers them.
    observations
        The run's observations at declared sites, by site label.
    supplied
        Values standing in for let bindings, by name, which their steps
        read rather than compute.
    occurrences
        How many times the run has reached each site a scan reaches once
        per position, by label.
    reached
        The addresses ``"<label>@<n>"`` of the occurrences reached.
    run_index
        The position of this run among the runs of one call: an input of
        ``B`` rows runs the program once per row, ``0`` to ``B - 1``.
    runs
        The number of runs the call makes, ``B``.
    slice_value
        Maps a site label and a value given for every run of the call to
        this run's value: the run's slice of a value laid out one per
        run, or the value itself when every run shares it.
    carries_run_axis
        Whether a binding of a program, by name, carries this run's
        leading axis of length one.
    site_carries_run_axis
        Whether a site's value in this run, by label, carries the run's
        leading axis of length one.
    vectorized
        Whether this execution carries all ``runs`` runs of the call at
        once, every value leading with the run axis of length ``runs``.
    select_run
        Given a value handed to a handler for every run of the call and the
        sampleable of the site it is used at, this run's value. A compiled
        run slices its values by `slice_value` before the run and takes
        each as it is here; a run that learns a site's shape only when it
        reaches the site slices here.
    subsamples
        The rows of each subsampled plate the run scores, by plate name,
        as the `SubsampleHandler` entries on the stack register them.
    """

    kernel: ProgramKernel = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    validator: RuntimeValidator = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    contributions: list[Contribution] = dx.field(
        default_factory=list, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    interventions: set[SiteKey] = dx.field(
        default_factory=set, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    fixed: set[str] = dx.field(
        default_factory=set, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    annotations: dict[str, dict[str, HeldValue]] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    parameters: dict[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    observations: dict[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    supplied: dict[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    occurrences: dict[str, int] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    reached: set[str] = dx.field(
        default_factory=set, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    run_index: int = 0
    runs: int = 1
    slice_value: Callable[[str, torch.Tensor], torch.Tensor] = dx.field(
        default=_shared_value, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    carries_run_axis: Callable[[MonadicProgram, str], bool] = dx.field(
        default=_no_run_axis, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    site_carries_run_axis: Callable[[str], bool] = dx.field(
        default=_no_site_run_axis, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    vectorized: bool = False
    select_run: Callable[[object, object], object] = dx.field(
        default=_as_given, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    subsamples: dict[str, torch.Tensor] = dx.field(
        default_factory=dict, opaque=True, extras=OPAQUE_BY_IDENTITY
    )

    @property
    def result_type(self) -> TypeExpr:
        """The type of the program's result.

        Returns
        -------
        TypeExpr
            The entry computation's result type, which every transparent
            handler answers with.
        """
        return self.kernel.result_type

    def intervene(self, request: RuntimeRequest) -> None:
        """Record that a site's own factor is not part of the joint.

        Parameters
        ----------
        request : RuntimeRequest
            The site's sample request, whose first argument is its label.
        """
        self.interventions.add(self.key_of(request))
        label = request.arguments[0]
        assert isinstance(label, str)
        self.fixed.add(label)

    def conditions(
        self, occurrence: Callable[[str], str] | None = None
    ) -> RunConditions:
        """What the run does to the factors of a score.

        Parameters
        ----------
        occurrence : Callable[[str], str] or None
            From a site's label to the address a step of a scan reads it
            at; ``None`` for a step outside any scan.

        Returns
        -------
        RunConditions
            The responses fixed, by the names a step reads them under, and
            the subsets of the subsampled plates.
        """
        if occurrence is None:
            fixed = frozenset(self.fixed)
        else:
            bases = {label.rpartition("@")[0] for label in self.fixed}
            fixed = frozenset(
                name for name in bases if name and occurrence(name) in self.fixed
            )
        return RunConditions(fixed=fixed, subsets=dict(self.subsamples))

    def key_of(self, request: RuntimeRequest) -> SiteKey:
        """The address of the site a request belongs to.

        Parameters
        ----------
        request : RuntimeRequest
            A sample request, or a score request derived from one.

        Returns
        -------
        SiteKey
            The request's own address for a sample request; for a derived
            request, the address of the site it was derived from.
        """
        static, dynamic, path = request.address
        parents = request.core.origin.parents
        if parents:
            # A derived request's declared dynamic path is the address of
            # the site it was derived from.
            static = str(parents[0])
            dynamic = tuple(
                (frame.scope, frame.key) for frame in request.core.origin.dynamic_path
            )
        return (static, dynamic, path)


@lru_cache(maxsize=256)
def _named_definition(definition: HandlerDef, name: str) -> HandlerDef:
    """Rename an immutable signature without retaining a runtime handler.

    Parameters
    ----------
    definition : HandlerDef
        The complete checked handler declaration.
    name : str
        Its position-dependent name in the run module.

    Returns
    -------
    HandlerDef
        The renamed declaration, shared across equivalent installations.
    """
    return definition.with_(id=HandlerId.derive("run", name), name=name)


class Installation(QiecModel):
    """One lexical handler a stack entry installs on a run.

    The effect instance compares by value; the executable handler retains
    identity.

    Parameters
    ----------
    instance
        The instance the handler handles: the program's ``random``,
        ``score``, or ``param`` instance.
    runtime
        The executable handler, whose definition joins the run's module.
    """

    instance: NamedEffectInstance = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    runtime: RuntimeHandler = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)

    def named(self, name: str) -> Installation:
        """The installation with its handler declared under a name.

        A module's handler names and identities are unique, while the
        prelude factories name every handler of a kind alike and derive
        its identity from the stack entry that made it; a run renames
        each installation it makes and derives its identity from the
        name, so two runs installing the same kinds of handler in the
        same order build the same module.

        Parameters
        ----------
        name : str
            The declaration's name, unique within the run.

        Returns
        -------
        Installation
            The installation with the renamed declaration, the rename
            carried into any per-installation factory the handler has.
        """
        definition = _named_definition(self.runtime.definition, name)
        runtime = self.runtime.with_(definition=definition)
        factory = self.runtime.context_factory
        if factory is not None:

            def context_factory() -> RuntimeHandler:
                """Build a fresh installation under the renamed declaration.

                Returns
                -------
                RuntimeHandler
                    The handler the original factory builds, renamed.
                """
                return factory().with_(definition=definition)

            runtime = runtime.with_(context_factory=context_factory)
        return Installation(instance=self.instance, runtime=runtime)


def handle_chain(installations: list[Installation], inner: Computation) -> Computation:
    """Nest handler installations around a computation, outermost first.

    Parameters
    ----------
    installations : list[Installation]
        The installations, outermost first.
    inner : Computation
        The computation handled.

    Returns
    -------
    Computation
        The nested ``Handle`` terms.
    """
    body = inner
    for installation in reversed(installations):
        body = Handle(
            instance=installation.instance.entry.instance,
            handler=installation.runtime.definition.id,
            computation=body,
            static_arguments=(),
        )
    return body


def fixed_site_values() -> dict[str, torch.Tensor] | None:
    """The sites the current stack fixes, when it does nothing else.

    Returns
    -------
    dict[str, torch.Tensor] or None
        The value of each fixed site, the innermost handler's where
        several fix one; ``None`` when a handler on the stack does
        anything besides fixing sites.
    """
    fixed: dict[str, torch.Tensor] = {}
    for handler in _handler_stack():
        values = handler.site_values
        if values is None:
            return None
        fixed.update(values)
    return fixed


class EffectHandler(ABC):
    """A handler of a program's canonical instances, stacked by ``with``.

    Subclasses implement :meth:`install`, which builds the lexical
    handlers the entry contributes to one run, and may implement
    :meth:`finish`, which runs after the program returns with the run's
    context. The handler activates by being used as a context manager:

        with clamp({"z": z_val}):
            samples = predictive.rsample(x)

    Nested ``with`` blocks stack handlers outer-first; the innermost
    handler of an instance is installed closest to the program.
    """

    @property
    def site_values(self) -> Mapping[str, torch.Tensor] | None:
        """The values the handler fixes sites to, when it does nothing else.

        Returns
        -------
        Mapping[str, torch.Tensor] or None
            The value of each site, by label, for a handler whose only
            effect is to answer those sites without scoring them; ``None``
            for any other handler.
        """
        return None

    def __enter__(self) -> Self:
        """Push the handler on the thread's stack.

        Returns
        -------
        Self
            The handler, so ``with handler as h`` binds it.
        """
        _handler_stack().append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Pop the handler off the thread's stack.

        Parameters
        ----------
        exc_type : type[BaseException] | None
            The exception type leaving the block, if any.
        exc : BaseException | None
            The exception, if any.
        tb : object
            Its traceback, if any.

        Raises
        ------
        RuntimeError
            If the handler is not the top of the stack, which means the
            blocks were exited out of order.
        """
        del exc_type, exc, tb
        stack = _handler_stack()
        if not stack or stack[-1] is not self:
            raise RuntimeError(
                "EffectHandler.__exit__: handler stack corruption; "
                "handlers must be entered / exited in strict LIFO order."
            )
        stack.pop()

    @abstractmethod
    def install(self, run: RunContext) -> tuple[Installation, ...]:
        """Build the lexical handlers this entry installs on a run.

        Parameters
        ----------
        run : RunContext
            The run being prepared.

        Returns
        -------
        tuple[Installation, ...]
            The handlers, each naming the instance it handles.
        """

    def finish(self, run: RunContext, output: object) -> None:
        """Observe the completed run.

        Parameters
        ----------
        run : RunContext
            The run, with every contribution the accumulator received.
        output : object
            The program's output.
        """
        del run, output


__all__ = [
    "Contribution",
    "EffectHandler",
    "Installation",
    "RunContext",
    "SiteKey",
]
