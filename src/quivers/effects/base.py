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
from dataclasses import dataclass, field, replace
import threading
from typing import Self

import torch

from quivers.effects.program_module import ProgramKernel
from quivers.qiec.evaluator import RuntimeHandler, RuntimeRequest, RuntimeValidator
from quivers.qiec.identifiers import HandlerId
from quivers.qiec.module import NamedEffectInstance
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


@dataclass(slots=True)
class Contribution:
    """One score contribution that reached the run's accumulator.

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
    """

    key: SiteKey
    role: str
    weight: torch.Tensor
    path: tuple[str | int, ...] = ()


@dataclass(slots=True)
class RunContext:
    """What one run shares with the handlers installed on it.

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
    """

    kernel: ProgramKernel
    validator: RuntimeValidator
    contributions: list[Contribution] = field(default_factory=list)
    interventions: set[SiteKey] = field(default_factory=set)
    annotations: dict[str, dict[str, object]] = field(default_factory=dict)
    parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    observations: dict[str, torch.Tensor] = field(default_factory=dict)
    supplied: dict[str, torch.Tensor] = field(default_factory=dict)
    occurrences: dict[str, int] = field(default_factory=dict)
    reached: set[str] = field(default_factory=set)

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


@dataclass(frozen=True, slots=True)
class Installation:
    """One lexical handler a stack entry installs on a run.

    Parameters
    ----------
    instance
        The instance the handler handles: the program's ``random``,
        ``score``, or ``param`` instance.
    runtime
        The executable handler, whose definition joins the run's module.
    """

    instance: NamedEffectInstance
    runtime: RuntimeHandler

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
        definition = replace(
            self.runtime.definition,
            id=HandlerId.derive("run", name),
            name=name,
        )
        runtime = replace(self.runtime, definition=definition)
        factory = self.runtime.context_factory
        if factory is not None:

            def context_factory() -> RuntimeHandler:
                """Build a fresh installation under the renamed declaration.

                Returns
                -------
                RuntimeHandler
                    The handler the original factory builds, renamed.
                """
                return replace(factory(), definition=definition)

            runtime.context_factory = context_factory
        return Installation(self.instance, runtime)


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
