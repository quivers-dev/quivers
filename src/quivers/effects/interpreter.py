"""Running a `MonadicProgram` on the reference machine under a handler stack.

`run_program` encodes the program as a kernel computation, installs the
handlers on the thread-local stack as lexical handlers of the program's
``random`` and ``score`` instances, serves the program's host steps
through ``Compute`` handlers over torch, and evaluates the whole on the
reference machine. The value it returns is the program's output;
handlers accumulate state on the side. A caller that wants a `Trace`
stacks a `TraceHandler` before `run_program` and reads its ``trace``
attribute afterwards; the thin `quivers.inference.trace.trace` wrapper
does exactly this.

The nesting of the installed handlers, outermost first, is fixed by the
kernel's rules rather than by the order of the stack alone: the run's
score accumulator, then every stack handler of the ``score`` instance in
stack order, then the default scoring draw and the run's observations,
then every stack handler of the ``random`` instance in stack order, then
the parameter store with every stack handler of the ``param`` instance
inside it, then the host steps. A density is a ``Score.add`` the answering
handler emits outward, so a score transformer anywhere on the stack
reaches every site, and an inner ``random`` handler sees the answer an
outer one gives.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from typing import TypeIs

import torch

from quivers.continuous.morphisms import ContinuousMorphism, MarginalizedFactor
from quivers.core.morphisms import extract_morphism
from quivers.continuous.plate import PlateDraw, VectorisedObserve
from quivers.continuous.spaces import ContinuousSpace
from quivers.continuous.program_steps import Call as CallStep
from quivers.continuous.program_steps import Draw, Let, Observe, Scan, Score
from quivers.continuous.programs import install_evaluator

from quivers.continuous.programs import MonadicProgram
from quivers.effects.base import (
    Contribution,
    EffectHandler,
    Installation,
    RunContext,
    _handler_stack,
)
from quivers.effects.program_module import (
    INPUT_NAME,
    SCAN_POSITION,
    SCAN_SEQUENCE,
    SCAN_STATES,
    HostStep,
    ProgramKernel,
    derived_bindings,
    once_sites,
    program_kernel,
    reversed_scan_sites,
    scanned_sites,
    site_label,
)
from quivers.effects.sites import TorchSampleable
from quivers.effects.subsample import site_plates, subsampled_rows
from quivers.effects.trace_handler import TraceHandler
from quivers.qiec.builtins import (
    PARAM_GET,
    ExtraValuePolicy,
    MissingValuePolicy,
    compute_handler,
    condition_handler,
    draw_scoring_handler,
    param_handler,
    score_handler,
)
from quivers.qiec.effects import EffectRequest, EffectRow, HandlerDef
from quivers.qiec.evaluator import (
    ClauseContext,
    Evaluator,
    RuntimeAttachments,
    RuntimeRequest,
)
from quivers.qiec.identifiers import SiteProvenance, SourceOrigin
from quivers.qiec.kinds import NAT
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.canonical import tensor_shape, tensor_type
from quivers.qiec.terms import (
    Call,
    Computation,
    Handle,
    LiteralValue,
    Local,
    Perform,
    Var,
)
from quivers.stochastic.agenda import ChartView, CorpusChart
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    IndexLiteral,
    IndexTerm,
    IndexVariable,
    TypeExpr,
    render_static,
)
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.effects import ComputationType
from quivers.qiec.module import KernelRegistry, validate_module
from quivers.effects.program_module import SOURCE_PROTOCOL

#: The name of the wrapper computation a run evaluates.
RUN = "__run"

type HostValue = (
    torch.Tensor
    | int
    | float
    | bool
    | str
    | TorchSampleable
    | tuple[HostValue, ...]
    | None
)
"""A value a host step produces or consumes: a tensor, a number, a
string, a sampleable, ``None``, or a tuple of such values."""

type RunValue = torch.Tensor | tuple[RunValue, ...]
"""A let or score value computed from one run's bindings: a tensor or a
tuple of them."""


def host_value(value: object) -> TypeIs[HostValue]:
    """Whether a value is one a host step may produce or consume.

    Parameters
    ----------
    value : object
        The value.

    Returns
    -------
    bool
        True for tensors, numbers, strings, sampleables, ``None``, and
        tuples of such values. The kernel checks the terms statically;
        the host convention is that every tensor carries the batch axis
        in front of the shape its type names.
    """
    if isinstance(value, tuple):
        return all(host_value(item) for item in value)
    return value is None or isinstance(
        value, torch.Tensor | int | float | bool | str | TorchSampleable
    )


class _Method(torch.nn.Module):
    """A module whose forward calls a named method of an inner module.

    ``torch.func.functional_call`` substitutes parameters for a forward
    pass only; routing a method through a forward lets a morphism's
    ``rsample`` or ``log_prob`` run under substituted parameters.

    Parameters
    ----------
    inner : ContinuousMorphism
        The morphism whose methods are called.
    """

    def __init__(self, inner: ContinuousMorphism) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, method: str, *arguments: torch.Tensor) -> torch.Tensor:
        """Call a method of the inner module.

        Parameters
        ----------
        method : str
            The method's name.
        *arguments : torch.Tensor
            Its arguments.

        Returns
        -------
        torch.Tensor
            What the method returned.
        """
        return getattr(self.inner, method)(*arguments)


class _ParameterizedMorphism(ContinuousMorphism):
    """A morphism run under parameters a ``Param`` handler supplied.

    Parameters
    ----------
    inner : ContinuousMorphism
        The morphism.
    parameters : Mapping[str, torch.Tensor]
        The values to run it under, by the morphism's own parameter
        names.
    """

    def __init__(
        self, inner: ContinuousMorphism, parameters: Mapping[str, torch.Tensor]
    ) -> None:
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner
        self._method = _Method(inner)
        self._values = {f"inner.{name}": value for name, value in parameters.items()}

    def _call(self, method: str, *arguments: torch.Tensor) -> torch.Tensor:
        """Run one method of the inner morphism under the supplied parameters.

        Parameters
        ----------
        method : str
            The method's name.
        *arguments : torch.Tensor
            Its arguments.

        Returns
        -------
        torch.Tensor
            What the method returned.
        """
        return torch.func.functional_call(
            self._method, self._values, (method, *arguments)
        )

    def rsample(  # type: ignore[override]
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw under the supplied parameters.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        sample_shape : torch.Size
            Extra sample dimensions, which the inner morphism interprets.

        Returns
        -------
        torch.Tensor
            The draw.
        """
        if sample_shape:
            return self._call("rsample", x, torch.as_tensor(tuple(sample_shape)))
        return self._call("rsample", x)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Score under the supplied parameters.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        y : torch.Tensor
            The value scored.

        Returns
        -------
        torch.Tensor
            The log density.
        """
        return self._call("log_prob", x, y)

    def log_prob_rows(
        self, x: torch.Tensor, y: torch.Tensor, *rows: torch.Tensor
    ) -> torch.Tensor:
        """Score a subset of a plate's rows under the supplied parameters.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        y : torch.Tensor
            The value scored.
        *rows : torch.Tensor
            One index per plate axis, as the inner morphism's
            ``log_prob_rows`` reads them.

        Returns
        -------
        torch.Tensor
            The log density of the kept rows.
        """
        return self._call("log_prob_rows", x, y, *rows)

    def __getattr__(self, name: str) -> torch.Tensor | torch.nn.Module:
        """Read a submodule, or any other attribute from the inner morphism.

        Parameters
        ----------
        name : str
            The attribute.

        Returns
        -------
        torch.Tensor | torch.nn.Module
            The registered submodule or parameter of that name, else the
            inner morphism's attribute, so a strategy reading its
            parameter methods sees them.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "_inner":
                raise
            return getattr(super().__getattr__("_inner"), name)


class _SubsampledMorphism(ContinuousMorphism):
    """A plated site's morphism scored at a subset of its plate rows, rescaled.

    Parameters
    ----------
    inner : ContinuousMorphism
        The site's morphism, run under any parameters a ``Param`` handler
        supplied; it scores a subset of rows through ``log_prob_rows``.
    rows : tuple[torch.Tensor, ...]
        One index per plate axis of the site, outermost first: the
        subset's rows along a subsampled axis, every row along another.
    factor : float
        The product, over the subsampled axes, of the axis's extent over
        the subset's size.
    """

    def __init__(
        self,
        inner: ContinuousMorphism,
        rows: tuple[torch.Tensor, ...],
        factor: float,
    ) -> None:
        super().__init__(inner.domain, inner.codomain)
        self._inner = inner
        self._rows = rows
        self._factor = factor

    def rsample(  # type: ignore[override]
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Draw every row of the plate, as the inner morphism does.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        sample_shape : torch.Size
            Extra sample dimensions.

        Returns
        -------
        torch.Tensor
            The draw.
        """
        return self._inner.rsample(x, sample_shape)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The subset's log density times the plate's factor.

        Parameters
        ----------
        x : torch.Tensor
            The input.
        y : torch.Tensor
            The site's value.

        Returns
        -------
        torch.Tensor
            ``factor`` times the summed log density of the kept rows.
        """
        log_prob_rows = getattr(self._inner, "log_prob_rows")
        return self._factor * log_prob_rows(x, y, *self._rows)


def _subsampled_rows(
    run: RunContext,
    step: HostStep,
    spec: Draw | Observe,
    morphism: ContinuousMorphism,
) -> tuple[tuple[torch.Tensor, ...], float] | None:
    """The rows a site is scored at under the run's subsampled plates.

    Parameters
    ----------
    run : RunContext
        The run, with the subsets its handlers registered.
    step : HostStep
        The site's host step.
    spec : Draw | Observe
        The site's step record.
    morphism : ContinuousMorphism
        The site's declared morphism.

    Returns
    -------
    tuple[tuple[torch.Tensor, ...], float] | None
        One index per plate axis of the site and the site's factor, or
        ``None`` for a site in no subsampled plate.

    Raises
    ------
    ValueError
        If the site is in a subsampled plate but its morphism has no
        per-row density, or the plate's extent at the site is not fixed.
    """
    if not run.subsamples:
        return None
    label = step.site
    assert label is not None
    found = subsampled_rows(label, site_plates(step.program, spec), run.subsamples)
    if found is not None and not isinstance(morphism, PlateDraw | VectorisedObserve):
        raise ValueError(
            f"site {label!r} ranges over a subsampled plate, but it draws from "
            f"{type(morphism).__name__}, which has no density per plate row; "
            "only a plate draw or a plated observation can be subsampled"
        )
    return found


def _split_observations(
    program: MonadicProgram, observations: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split observations into site values and host data.

    A site a scan reaches once per position takes its value at the
    ``n``-th occurrence under ``"<label>@<n>"``, of shape ``(batch,
    ...)``; or, for every position at once, under ``"<label>"``, with
    the positions on the axis after the batch, ``(batch, seq_len,
    ...)``, entry ``[:, t]`` being the value at input position ``t``:
    the ``t``-th occurrence, or for a site a reversed scan reaches,
    which visits the last position first, the ``(seq_len - 1 - t)``-th.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    observations : Mapping[str, torch.Tensor]
        Values keyed by name.

    Returns
    -------
    tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]
        The values at declared sites, keyed by site address; the values
        standing in for let bindings, by name, which their steps read
        rather than compute; and the remaining entries, which are host
        data the program's steps read by name.

    Raises
    ------
    ValueError
        If an observation names one variable of a destructuring draw,
        gives a scanned site's value both for every position and for
        one, lays a scanned site's values out with no position axis, or
        names a binding a scan or a call computes.
    """
    declared: dict[str, str] = {}
    computed: set[str] = set()
    for spec in program.steps:
        if isinstance(spec, Let | Score):
            computed.add(spec.name)
    for label in once_sites(program):
        for name in label.split(","):
            declared[name] = label
    scanned = scanned_sites(program)
    backward = reversed_scan_sites(program)
    derived, per_position = derived_bindings(program)
    sites: dict[str, torch.Tensor] = {}
    supplied: dict[str, torch.Tensor] = {}
    data: dict[str, torch.Tensor] = {}
    stacked: dict[str, torch.Tensor] = {}
    for name, value in observations.items():
        base, at, occurrence = name.rpartition("@")
        if name in computed:
            supplied[name] = value
        elif name in scanned:
            if value.dim() < 2:
                raise ValueError(
                    f"observation {name!r} is a site reached once per scan "
                    f"position, so its values are laid out (batch, seq_len, "
                    f"...); got shape {tuple(value.shape)}"
                )
            # A reversed scan reaches its last position first, so the
            # value at input position ``t`` is the occurrence that many
            # from the end.
            length = int(value.shape[1])
            for index, entry in enumerate(value.unbind(1)):
                occurrence = length - 1 - index if name in backward else index
                stacked[site_label(name, occurrence)] = entry
        elif at and occurrence.isdigit() and base in scanned:
            sites[name] = value
        elif name in derived or (at and occurrence.isdigit() and base in per_position):
            raise ValueError(
                f"{name!r} is computed by a scan or a call from the sites it "
                f"reaches, so it cannot be supplied; supply those sites instead"
            )
        else:
            label = declared.get(name)
            if label is None:
                data[name] = value
            elif label == name:
                sites[name] = value
            else:
                raise ValueError(
                    f"observation {name!r} names one variable of the destructuring "
                    f"draw {label!r}; observe the draw as a whole"
                )
    for address, value in stacked.items():
        if address in sites:
            raise ValueError(
                f"site {address!r} is given both on its own and in the sequence "
                f"of every position's values; give it once"
            )
        sites[address] = value
    return sites, supplied, data


def _host(argument: object) -> HostValue:
    """Narrow the argument a host function receives to a host value.

    Parameters
    ----------
    argument : object
        The argument the ``Compute`` handler passed.

    Returns
    -------
    HostValue
        The same argument.

    Raises
    ------
    TypeError
        If the argument is not a host value.
    """
    if not host_value(argument):
        raise TypeError(
            f"a host step received a {type(argument).__name__}, not a host value"
        )
    return argument


def _environment(step: HostStep, argument: object) -> dict[str, torch.Tensor]:
    """Rebuild the host environment a step reads from its argument tuple.

    Parameters
    ----------
    step : HostStep
        The step, naming the locals in order.
    argument : object
        The environment tuple the request carried.

    Returns
    -------
    dict[str, torch.Tensor]
        The locals by name.

    Raises
    ------
    TypeError
        If the argument is not a tuple of the step's arity.
    """
    if not isinstance(argument, tuple) or len(argument) != len(step.environment):
        raise TypeError(
            f"host step expects an environment of {len(step.environment)} values"
        )
    return dict(zip(step.environment, argument, strict=True))


def _check_sequence_layouts(
    kernel: ProgramKernel, observations: Mapping[str, torch.Tensor]
) -> None:
    """Check every whole-sequence value of a scanned site against its site.

    Parameters
    ----------
    kernel : ProgramKernel
        The program's encoding.
    observations : Mapping[str, torch.Tensor]
        The values given, by name.

    Raises
    ------
    ValueError
        If a value given under a scanned site's label is not laid out
        ``(batch, seq_len, *event)`` for the site's event shape, as a
        single position's value or a final state would not be.
    """
    for label, value in observations.items():
        if label not in kernel.scanned:
            continue
        event = kernel.site_events[label]
        if value.dim() < 2 or tuple(int(size) for size in value.shape[2:]) != event:
            layout = "".join(f", {size}" for size in event)
            raise ValueError(
                f"{label!r} names a site reached once per scan position, so a "
                f"value under that name is the sequence of every position's "
                f"value, laid out (batch, seq_len{layout}); got shape "
                f"{tuple(value.shape)}. Give one position's value as "
                f"{label + '@<n>'!r}"
            )


def _site_morphism(kernel: ProgramKernel, address: str) -> ContinuousMorphism | None:
    """The morphism a site address draws from.

    Parameters
    ----------
    kernel : ProgramKernel
        The program's encoding.
    address : str
        A site's label, or ``"<label>@<n>"`` for an occurrence of a site
        a scan reaches once per position.

    Returns
    -------
    ContinuousMorphism | None
        The morphism, or ``None`` for a label no sample request uses.
    """
    morphism = kernel.site_morphisms.get(address)
    if morphism is not None:
        return morphism
    label = address.rpartition("@")[0]
    return kernel.site_morphisms.get(label) if label in kernel.scanned else None


def _canonical(
    morphism: ContinuousMorphism | None, value: torch.Tensor
) -> torch.Tensor:
    """Restore a clamped plate latent's structured shape.

    Parameters
    ----------
    morphism : ContinuousMorphism | None
        The site's morphism, or ``None`` when no sample request uses
        the site.
    value : torch.Tensor
        The clamped value, which may arrive flattened.

    Returns
    -------
    torch.Tensor
        The value in the shape the morphism scores, when it has a
        canonical latent shape; the value unchanged otherwise.
    """
    canonical = getattr(morphism, "canonical_latent", None)
    if canonical is not None:
        return canonical(value)
    return value


def _sequence(step: Scan, value: torch.Tensor) -> torch.Tensor:
    """A scan's sequence with its positions on the axis after the batch.

    Parameters
    ----------
    step : Scan
        The scan.
    value : torch.Tensor
        The sequence as bound: ``(batch, seq_len)`` for a discrete
        entry, else ``(batch, seq_len, dim)`` or the folded
        ``(batch, seq_len * dim)``.

    Returns
    -------
    torch.Tensor
        ``(batch, seq_len)`` for a discrete entry, else
        ``(batch, seq_len, dim)``.

    Raises
    ------
    ValueError
        If the value has no position axis the step's entry width reads.
    """
    body = step.step
    assert isinstance(body, MonadicProgram)
    assert body._param_dims is not None and body._param_is_continuous is not None
    width = body._param_dims[0]
    if not body._param_is_continuous[0]:
        if value.dim() == 3 and int(value.shape[-1]) == 1:
            value = value.squeeze(-1)
        if value.dim() != 2:
            raise ValueError(
                f"scan {step.name!r}: a sequence of discrete entries is laid "
                f"out (batch, seq_len); got shape {tuple(value.shape)}"
            )
        return value
    if value.dim() == 3 and int(value.shape[-1]) == width:
        return value
    if value.dim() == 2 and int(value.shape[-1]) % width == 0:
        return value.reshape(value.shape[0], -1, width)
    raise ValueError(
        f"scan {step.name!r}: a sequence of {width}-wide entries is laid out "
        f"(batch, seq_len, {width}) or (batch, seq_len * {width}); got shape "
        f"{tuple(value.shape)}"
    )


def _plated(morphism: torch.nn.Module | None) -> bool:
    """Whether a site's morphism draws a plate's rows rather than one value.

    A plated site's value in one run is laid out along its plate's rows,
    with no leading run axis; every other site's value in one run leads
    with a run axis of length one.

    Parameters
    ----------
    morphism : torch.nn.Module | None
        The site's morphism, or ``None`` for a label no draw uses.

    Returns
    -------
    bool
        ``True`` for a `PlateDraw`, a `VectorisedObserve`, or the
        observation a ``marginalize`` block integrates over its plate,
        a `MarginalizedFactor`.
    """
    return isinstance(morphism, PlateDraw | VectorisedObserve | MarginalizedFactor)


def _event_numel(morphism: torch.nn.Module | None) -> int | None:
    """The number of elements a site that is not plated draws in one run.

    Parameters
    ----------
    morphism : torch.nn.Module | None
        The site's morphism, or ``None`` for a label no draw uses.

    Returns
    -------
    int | None
        The width of a continuous codomain, one for a discrete codomain,
        or ``None`` when the site has no morphism.
    """
    if morphism is None:
        return None
    codomain = getattr(morphism, "codomain", None)
    if isinstance(codomain, ContinuousSpace):
        return int(getattr(codomain, "dim"))
    return 1


def _plated_value(
    morphism: PlateDraw | VectorisedObserve, value: torch.Tensor
) -> torch.Tensor:
    """Lay one run's value of a plated site out along its plate's rows.

    Parameters
    ----------
    morphism : PlateDraw | VectorisedObserve
        The site's morphism.
    value : torch.Tensor
        One run's value, holding the plate's elements in any layout,
        such as one with a leading axis of length one.

    Returns
    -------
    torch.Tensor
        The value at the observation's response layout, or at the
        plate draw's latent layout.
    """
    if isinstance(morphism, VectorisedObserve):
        return value.reshape(morphism.response.shape)
    return morphism.canonical_latent(value)


def _plated_numel(morphism: PlateDraw | VectorisedObserve) -> int:
    """The number of elements a plated site's value holds in one run.

    Parameters
    ----------
    morphism : PlateDraw | VectorisedObserve
        The site's morphism.

    Returns
    -------
    int
        The plate's rows times each row's width.
    """
    if isinstance(morphism, VectorisedObserve):
        return int(morphism.response.numel())
    return int(getattr(morphism.codomain, "dim"))


class _RunAxis:
    """Which bindings of one run carry the leading run axis.

    Within one run, a value read from the program's input, from a site
    that is not plated, or from a let or score computed from such a
    value leads with a run axis of length one; a plated site's value,
    host data every run shares, and a let computed only from those do
    not. Draws, the input, and the parameters of called and scanned
    programs are classified from the program's structure; a let's or a
    score's flag is recorded when the run computes it.

    Parameters
    ----------
    program : MonadicProgram
        The program the run executes.
    per_run_data : Mapping[str, bool]
        For each host datum the run reads, whether it was given one
        value per run (and so carries the run axis).
    """

    def __init__(
        self, program: MonadicProgram, per_run_data: Mapping[str, bool]
    ) -> None:
        self._flags: dict[tuple[int, str], bool] = {}
        self._aliases: dict[tuple[int, str], tuple[int, str]] = {}
        self._parents: dict[int, int] = {}
        self._sites: dict[str, bool] = {}
        top = id(program)
        self._flags[(top, INPUT_NAME)] = True
        for name, per_run in per_run_data.items():
            self._flags[(top, name)] = per_run
        self._index(program, entry=True)

    def _index(self, program: MonadicProgram, *, entry: bool) -> None:
        """Classify a program's parameters and draws, and its callees'.

        Parameters
        ----------
        program : MonadicProgram
            The program.
        entry : bool
            Whether the program is the one the run executes, whose
            parameters split the run's input.
        """
        key = id(program)
        if entry and program._params is not None:
            for name in program._params:
                self._flags[(key, name)] = True
        for spec in program.steps:
            if isinstance(spec, Draw | Observe):
                morphism = program.step_module(spec)
                flag = not _plated(morphism)
                for name in (*spec.names, ",".join(spec.names)):
                    self._flags[(key, name)] = flag
                self._sites[",".join(spec.names)] = flag
            elif isinstance(spec, CallStep):
                callee = spec.program
                assert isinstance(callee, MonadicProgram)
                parameters = (
                    callee._params if callee._params is not None else (INPUT_NAME,)
                )
                for parameter, argument in zip(parameters, spec.args, strict=True):
                    self._aliases[(id(callee), parameter)] = (key, argument)
                self._flags[(id(callee), INPUT_NAME)] = True
                self._parents[id(callee)] = key
                self._aliases[(key, spec.name)] = (
                    id(callee),
                    callee._return_vars[0],
                )
                self._index(callee, entry=False)
            elif isinstance(spec, Scan):
                body = spec.step
                assert isinstance(body, MonadicProgram)
                self._flags[(key, spec.name)] = True
                self._parents[id(body)] = key
                self._flags[(id(body), INPUT_NAME)] = True
                for name in body._params or ():
                    self._flags[(id(body), name)] = True
                self._index(body, entry=False)

    def copy(self) -> _RunAxis:
        """An independent copy, as a fresh classification of the same run.

        A run records the flags of the bindings it computes; the copy
        starts from this classification's flags and records into its own.

        Returns
        -------
        _RunAxis
            The copy.
        """
        duplicate = _RunAxis.__new__(_RunAxis)
        duplicate._flags = dict(self._flags)
        duplicate._aliases = self._aliases
        duplicate._parents = self._parents
        duplicate._sites = dict(self._sites)
        return duplicate

    def given(self, program: MonadicProgram, label: str, flag: bool) -> None:
        """Record whether a site's given value carries the run axis.

        A value given for a site decides the flag by its own layout: an
        observation laid out along a plate's rows carries no run axis
        whatever morphism scores it.

        Parameters
        ----------
        program : MonadicProgram
            The program the run executes, whose site it is.
        label : str
            The site's label.
        flag : bool
            Whether the run's value leads with the run axis.
        """
        self._sites[label] = flag
        names = label.split(",")
        for name in (*names, label):
            self._flags[(id(program), name)] = flag

    def site(self, label: str) -> bool:
        """Whether a site's value in one run carries the run axis.

        Parameters
        ----------
        label : str
            The site's label, or ``"<label>@<n>"`` for an occurrence of
            a site a scan reaches once per position.

        Returns
        -------
        bool
            The site's flag; ``True`` for a site no step names.
        """
        base, at, occurrence = label.rpartition("@")
        key = base if at and occurrence.isdigit() else label
        return self._sites.get(key, True)

    def record(self, program: MonadicProgram, name: str, flag: bool) -> None:
        """Record whether a computed binding carries the run axis.

        Parameters
        ----------
        program : MonadicProgram
            The program the binding belongs to.
        name : str
            The binding.
        flag : bool
            Whether it carries the run axis.
        """
        self._flags[(id(program), name)] = flag

    def carries(self, program: MonadicProgram, name: str) -> bool:
        """Whether a binding of a program carries the run axis.

        Parameters
        ----------
        program : MonadicProgram
            The program the binding belongs to.
        name : str
            The binding.

        Returns
        -------
        bool
            Its flag.

        Raises
        ------
        RuntimeError
            If the binding is neither classified nor computed yet.
        """
        key = (id(program), name)
        seen: set[tuple[int, str]] = set()
        while key not in self._flags:
            if key in seen:
                raise RuntimeError(
                    f"the run cannot tell whether {name!r} carries the run axis"
                )
            seen.add(key)
            if key in self._aliases:
                key = self._aliases[key]
            elif key[0] in self._parents:
                # A called or scanned program reads the bindings of the
                # program it runs in that it does not bind itself.
                key = (self._parents[key[0]], key[1])
            else:
                raise RuntimeError(
                    f"the run cannot tell whether {name!r} carries the run axis"
                )
        return self._flags[key]


def _with_run_axis(value: RunValue) -> RunValue:
    """Give a value computed for one run its leading run axis back.

    Parameters
    ----------
    value : RunValue
        A tensor or a tuple of tensors computed from one run's values.

    Returns
    -------
    RunValue
        The value with a leading axis of length one on every tensor.

    Raises
    ------
    TypeError
        If the value is neither a tensor nor a tuple of tensors.
    """
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, tuple):
        return tuple(_with_run_axis(item) for item in value)
    raise TypeError(
        f"a binding computed from per-run values is a {type(value).__name__}, "
        "not a tensor or a tuple of tensors"
    )


class _HostSteps:
    """The host functions serving one program's ``Compute`` instances.

    Parameters
    ----------
    run : RunContext
        The run.
    x : torch.Tensor
        The program input.
    """

    def __init__(self, run: RunContext, x: torch.Tensor, axis: _RunAxis) -> None:
        self.run = run
        self.program = run.kernel.program
        self.x = x
        self.axis = axis

    def _one_run(
        self, step: HostStep, env: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Drop the run axis from the values of an environment that carry it.

        A let or score step is a function of one run's values: the run
        axis is removed before it runs and restored on its result, so a
        reduction, an index, or a collection it builds reads the
        binding's own axes and never the run's.

        Parameters
        ----------
        step : HostStep
            The step, whose program names the bindings.
        env : dict[str, torch.Tensor]
            The bindings the step reads.

        Returns
        -------
        tuple[dict[str, torch.Tensor], bool]
            The bindings at one run's shape, and whether any of them
            carried the run axis.

        Raises
        ------
        RuntimeError
            If a binding classified as carrying the run axis does not
            lead with an axis of length one.
        """
        stripped: dict[str, torch.Tensor] = {}
        carried = False
        for name, value in env.items():
            if isinstance(value, torch.Tensor) and self.axis.carries(
                step.program, name
            ):
                if value.dim() == 0 or int(value.shape[0]) != 1:
                    raise RuntimeError(
                        f"binding {name!r} carries the run axis but has shape "
                        f"{tuple(value.shape)}"
                    )
                stripped[name] = value[0]
                carried = True
            else:
                stripped[name] = value
        return stripped, carried

    def function(self, step: HostStep) -> Callable[[object, ClauseContext], object]:
        """The host function of a step.

        A draw reads its morphism's parameters by performing ``param``
        requests through the clause context; every other kind is a
        function of the environment tuple alone, `pure`.

        Parameters
        ----------
        step : HostStep
            The step.

        Returns
        -------
        Callable[[object, ClauseContext], object]
            The function the step's ``Compute`` handler calls with the
            environment tuple and the clause context.

        Raises
        ------
        ValueError
            If the step is of an unknown kind.
        """
        if step.kind == "draw":
            return lambda argument, context: self.draw(
                step,
                argument,
                lambda site, morphism: self._parameters(site, morphism, context),
            )
        pure = self.pure(step)
        return lambda argument, _context: pure(argument)

    def pure(self, step: HostStep) -> Callable[[object], object]:
        """The host function of a step that performs no effect.

        Parameters
        ----------
        step : HostStep
            The step, of any kind but ``"draw"``.

        Returns
        -------
        Callable[[object], object]
            The function of the environment tuple.

        Raises
        ------
        ValueError
            If the step is a draw, whose parameters are read through
            effects, or of an unknown kind.
        """
        if step.kind == "input":
            return self._split_input
        if step.kind == "draw":
            raise ValueError(
                "a draw step reads its parameters through effects; serve it "
                "with `draw`"
            )
        if step.kind == "apply":
            return lambda argument: self._apply(step, argument)
        if step.kind == "bind":
            return lambda argument: self._bind(step, argument)
        if step.kind == "let":
            return lambda argument: self._let(step, argument)
        if step.kind == "score":
            return lambda argument: self._score(step, argument)
        if step.kind == "pack":
            return lambda argument: self._pack(step, _host(argument))
        if step.kind == "site":
            return lambda _argument: self._site(step)
        if step.kind == "scan_length":
            return lambda argument: self._scan_length(step, _host(argument))
        if step.kind == "scan_initial":
            return lambda argument: self._scan_initial(step, _host(argument))
        if step.kind == "scan_entry":
            return lambda argument: self._scan_entry(step, _host(argument))
        if step.kind == "scan_state":
            return lambda argument: self._scan_state(step, _host(argument))
        if step.kind == "scan_states":
            return lambda argument: self._scan_states(step, _host(argument))
        if step.kind == "scan_collect":
            return lambda argument: self._scan_collect(step, _host(argument))
        raise ValueError(f"unknown host step kind {step.kind!r}")

    def _split_input(self, argument: object) -> object:
        """Split the program input into its named parameters.

        Parameters
        ----------
        argument : object
            The environment tuple, holding the input first.

        Returns
        -------
        object
            The parameters as a tuple, an index squeezed to a vector.
        """
        program = self.program
        assert program._params is not None
        assert program._param_dims is not None
        assert program._param_is_continuous is not None
        splits = torch.split(self.x, program._param_dims, dim=-1)
        parts: list[torch.Tensor] = []
        for chunk, continuous in zip(splits, program._param_is_continuous, strict=True):
            if not continuous and chunk.shape[-1] == 1:
                parts.append(chunk.squeeze(-1))
            else:
                parts.append(chunk)
        del argument
        return tuple(parts)

    def _input(
        self, spec: Draw | Observe, env: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """The input a draw step's morphism reads.

        Parameters
        ----------
        spec : Draw | Observe
            The step.
        env : dict[str, torch.Tensor]
            The bindings the step reads; the program's input among them
            when the step takes no arguments.

        Returns
        -------
        torch.Tensor
            The input of the program the step belongs to when the step
            takes no arguments; the run's input otherwise, which the
            arguments replace.
        """
        return env[INPUT_NAME] if spec.args is None else self.x

    def draw(
        self,
        step: HostStep,
        argument: object,
        parameters: Callable[[str, ContinuousMorphism], dict[str, torch.Tensor]],
    ) -> object:
        """Build the sampleable of a draw step.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.
        parameters : Callable[[str, ContinuousMorphism], dict[str, torch.Tensor]]
            Reads the parameters of the morphism drawing at a site, by
            the morphism's own names: on the reference machine, through
            the ``param`` instance.

        Returns
        -------
        object
            The morphism at its resolved input, under the parameters the
            ``param`` handlers answered.
        """
        spec = step.spec
        assert isinstance(spec, Draw | Observe)
        env = _environment(step, argument)
        morphism = step.program.step_module(spec)
        assert isinstance(morphism, ContinuousMorphism)
        inp = step.program._resolve_input(
            spec,
            self._input(spec, env),
            env,
            lambda name: self.axis.carries(step.program, name),
        )
        assert step.site is not None
        values = parameters(step.site, morphism)
        drawing: ContinuousMorphism = (
            _ParameterizedMorphism(morphism, values) if values else morphism
        )
        subsampled = _subsampled_rows(self.run, step, spec, morphism)
        if subsampled is not None:
            rows, factor = subsampled
            annotations = self.run.annotations.setdefault(step.site, {})
            annotations["subsample_scale"] = factor
            annotations["subsample_rows"] = rows
            drawing = _SubsampledMorphism(drawing, rows, factor)
        if drawing is morphism:
            return TorchSampleable(morphism, inp)
        return TorchSampleable(drawing, inp, declared=morphism)

    def _parameters(
        self, site: str, morphism: ContinuousMorphism, context: ClauseContext
    ) -> dict[str, torch.Tensor]:
        """Read a morphism's parameters through the ``param`` instance.

        Parameters
        ----------
        site : str
            The site the morphism draws at, which qualifies the names.
        morphism : ContinuousMorphism
            The morphism.
        context : ClauseContext
            The clause context the requests are performed through.

        Returns
        -------
        dict[str, torch.Tensor]
            The parameter values, by the morphism's own names.
        """
        kernel = self.run.kernel
        values: dict[str, torch.Tensor] = {}
        for name, parameter in morphism.named_parameters():
            qualified = f"{site}.{name}"
            shape = tuple(int(extent) for extent in parameter.shape)
            type_ = tensor_type(
                REAL, tuple(IndexLiteral(extent, NAT) for extent in shape)
            )
            request = EffectRequest(
                kernel.param.entry.instance,
                kernel.param.entry.effect,
                PARAM_GET,
                (type_,),
                (LiteralValue(qualified, STRING),),
                type_,
                SiteProvenance(
                    SourceOrigin(
                        kernel.module.module,
                        ("sites", site, "parameters", name),
                        "effect-request",
                        SOURCE_PROTOCOL,
                    )
                ),
            )
            value = context.evaluate(Perform(request))
            assert isinstance(value, torch.Tensor)
            values[name] = value
        return values

    def _apply(self, step: HostStep, argument: object) -> object:
        """Contract a categorical morphism's tensor against its input.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.

        Returns
        -------
        object
            The morphism's action on the batched input.
        """
        spec = step.spec
        assert isinstance(spec, Draw | Observe)
        env = _environment(step, argument)
        module = step.program.step_module(spec)
        assert module is not None
        categorical = extract_morphism(module)
        assert categorical is not None
        inp = step.program._resolve_input(
            spec,
            self._input(spec, env),
            env,
            lambda name: self.axis.carries(step.program, name),
        )
        return step.program._apply_categorical_morphism(
            categorical, inp, self.x.shape[0]
        )

    def _bind(self, step: HostStep, argument: object) -> object:
        """Split a destructuring draw into its variables.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple, holding the joint draw last.

        Returns
        -------
        object
            One tensor per bound variable.
        """
        spec = step.spec
        assert isinstance(spec, Draw | Observe)
        env = _environment(step, argument)
        result = env[",".join(spec.names)]
        scratch: dict[str, torch.Tensor] = {}
        step.program._bind_result(spec, result, scratch)
        return tuple(scratch[name] for name in spec.names)

    def _let(self, step: HostStep, argument: object) -> object:
        """Compute a let binding.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.

        Returns
        -------
        object
            The bound value: an alias, a closure's result, or a constant
            broadcast over the batch.
        """
        spec = step.spec
        assert isinstance(spec, Let)
        if step.program is self.program:
            supplied = self.run.supplied.get(spec.name)
            if supplied is not None:
                # A supplied value stands in for the computed one, so it
                # carries the run axis exactly when the computed value
                # would: when a binding the let reads carries it.
                _, carried = self._one_run(step, _environment(step, argument))
                if carried and (supplied.dim() == 0 or int(supplied.shape[0]) != 1):
                    supplied = supplied.unsqueeze(0)
                self.axis.record(step.program, spec.name, carried)
                return supplied
        env = _environment(step, argument)
        if isinstance(spec.value, str):
            self.axis.record(
                step.program, spec.name, self.axis.carries(step.program, spec.value)
            )
            return env[spec.value]
        if callable(spec.value):
            stripped, carried = self._one_run(step, env)
            value = spec.value(stripped)
            if isinstance(value, ChartView | CorpusChart):
                # The checked program binds a deduction's answer, the
                # inside weight of its goal.
                value = value.goal_weight()
            if carried:
                value = _with_run_axis(value)
            self.axis.record(step.program, spec.name, carried)
            return value
        self.axis.record(step.program, spec.name, True)
        return torch.full((self.x.shape[0],), float(spec.value), device=self.x.device)

    def _score(self, step: HostStep, argument: object) -> object:
        """Compute a score step's contribution.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : object
            The environment tuple.

        Returns
        -------
        object
            The contribution.
        """
        spec = step.spec
        assert isinstance(spec, Score)
        stripped, carried = self._one_run(step, _environment(step, argument))
        value = spec.score(stripped)
        if carried:
            value = _with_run_axis(value)
        self.axis.record(step.program, spec.name, carried)
        return value

    def _pack(self, step: HostStep, argument: HostValue) -> torch.Tensor:
        """Assemble a called program's input from its parameters.

        Parameters
        ----------
        step : HostStep
            The step, of the called program.
        argument : HostValue
            The environment tuple, holding the parameters in order.

        Returns
        -------
        torch.Tensor
            The parameters stacked along the feature axis, the layout
            the program's input split reads.
        """
        env = _environment(step, argument)
        params = step.program._params
        assert params is not None
        return step.program._stack_tensors([env[name] for name in params])

    def _site(self, step: HostStep) -> str:
        """Address the next occurrence of a site reached once per position.

        Parameters
        ----------
        step : HostStep
            The step, naming the site's label.

        Returns
        -------
        str
            ``"<label>@<n>"``, ``n`` counting the run's earlier
            occurrences of the label from zero.
        """
        assert step.site is not None
        occurrence = self.run.occurrences.get(step.site, 0)
        self.run.occurrences[step.site] = occurrence + 1
        address = site_label(step.site, occurrence)
        self.run.reached.add(address)
        return address

    def _scan_length(self, step: HostStep, argument: HostValue) -> int:
        """The number of positions a scan runs over.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : HostValue
            The environment tuple, holding the sequence.

        Returns
        -------
        int
            The sequence's length.
        """
        spec = step.spec
        assert isinstance(spec, Scan)
        (value,) = _environment(step, argument).values()
        return int(_sequence(spec, value).shape[1])

    def _scan_initial(self, step: HostStep, argument: HostValue) -> torch.Tensor:
        """A scan's initial state, one row per row of the sequence.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : HostValue
            The environment tuple, holding the sequence and, for a scan
            that starts from a binding, that binding.

        Returns
        -------
        torch.Tensor
            The initial state, ``(batch, dim H)``: the binding the scan
            starts from, else the scan's own state when it has one, else
            zero.

        Raises
        ------
        ValueError
            If the binding the scan starts from does not hold one state
            per row.
        """
        spec = step.spec
        assert isinstance(spec, Scan)
        environment = _environment(step, argument)
        value = environment[step.environment[0]]
        batch = int(value.shape[0])
        body = spec.step
        width = int(getattr(body.codomain, "dim"))
        if spec.initial_from is not None:
            start = environment[spec.initial_from]
            if start.dim() == 2 and tuple(start.shape) == (batch, width):
                return start
            if start.dim() == 1 and int(start.shape[0]) == width:
                return start.unsqueeze(0).expand(batch, -1)
            if start.numel() == batch * width:
                return start.reshape(batch, width)
            raise ValueError(
                f"scan {spec.name!r} starts from {spec.initial_from!r}, of shape "
                f"{tuple(start.shape)}, which is not one state of width {width} "
                f"per row of the batch of {batch}"
            )
        if spec.initial is not None:
            return spec.initial.unsqueeze(0).expand(batch, -1)
        dtype = value.dtype if value.is_floating_point() else torch.get_default_dtype()
        return torch.zeros(batch, width, device=value.device, dtype=dtype)

    def _scan_state(self, step: HostStep, argument: HostValue) -> torch.Tensor:
        """The state a scan's step returned, carried to the next position.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : HostValue
            The environment tuple, holding the returned state.

        Returns
        -------
        torch.Tensor
            The state, ``(batch, dim H)``.

        Raises
        ------
        ValueError
            If the step returned a value of another width.
        """
        spec = step.spec
        assert isinstance(spec, Scan)
        (value,) = _environment(step, argument).values()
        width = int(getattr(spec.step.codomain, "dim"))
        if value.dim() == 1 and width == 1:
            value = value.unsqueeze(-1)
        if value.dim() != 2 or int(value.shape[-1]) != width:
            raise ValueError(
                f"scan {spec.name!r}: the step returned a state of shape "
                f"{tuple(value.shape)}, not (batch, {width})"
            )
        return value

    def _scan_entry(self, step: HostStep, argument: HostValue) -> torch.Tensor:
        """The entry a scan reads at the next position it visits.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : HostValue
            The environment tuple, holding the sequence and the count of
            positions visited.

        Returns
        -------
        torch.Tensor
            ``(batch,)`` for a discrete entry, else ``(batch, dim)``: the
            entry at the count's position, or, for a reversed scan, at
            the position that many from the end.
        """
        spec = step.spec
        assert isinstance(spec, Scan)
        env = _environment(step, argument)
        visited = env[SCAN_POSITION]
        assert isinstance(visited, int)
        sequence = _sequence(spec, env[SCAN_SEQUENCE])
        position = int(sequence.shape[1]) - 1 - visited if spec.reverse else visited
        return sequence[:, position]

    def _scan_states(self, step: HostStep, argument: HostValue) -> torch.Tensor:
        """The states a scan returning every state has gathered at its start.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : HostValue
            The environment tuple, holding the sequence.

        Returns
        -------
        torch.Tensor
            No states, ``(batch, 0, dim H)``, on the sequence's device.
        """
        spec = step.spec
        assert isinstance(spec, Scan)
        (value,) = _environment(step, argument).values()
        width = int(getattr(spec.step.codomain, "dim"))
        dtype = value.dtype if value.is_floating_point() else torch.get_default_dtype()
        return torch.zeros(
            int(value.shape[0]), 0, width, device=value.device, dtype=dtype
        )

    def _scan_collect(self, step: HostStep, argument: HostValue) -> torch.Tensor:
        """Gather the state a scan's step returned among the scan's states.

        Parameters
        ----------
        step : HostStep
            The step.
        argument : HostValue
            The environment tuple, holding the states gathered so far and
            the state the step returned.

        Returns
        -------
        torch.Tensor
            The states, ``(batch, visited, dim H)``, in input order: the
            new state after the others, or, for a reversed scan, which
            visits the positions from the last, before them.
        """
        spec = step.spec
        assert isinstance(spec, Scan)
        env = _environment(step, argument)
        gathered = env[SCAN_STATES]
        state = env["__scan_next"].unsqueeze(1).to(gathered.dtype)
        parts = (state, gathered) if spec.reverse else (gathered, state)
        return torch.cat(parts, dim=1)


def _handle_chain(installations: list[Installation], inner: Computation) -> Computation:
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
            installation.instance.entry.instance,
            installation.runtime.definition.id,
            body,
            (),
        )
    return body


#: The kernel encodings and validated run modules of each program, keyed
#: by the program. An encoding depends on the program's steps and the
#: data names a run supplies; a run module also depends on the handlers
#: the run installs, whose definitions the module's identity carries. A
#: run rebuilds neither, so a program traced many times pays for its
#: encoding and its module's validation once.
_KERNELS: weakref.WeakKeyDictionary[
    MonadicProgram, dict[tuple[str, ...], ProgramKernel]
] = weakref.WeakKeyDictionary()
_REGISTRIES: weakref.WeakKeyDictionary[
    MonadicProgram, dict[tuple[tuple[str, ...], tuple[HandlerDef, ...]], KernelRegistry]
] = weakref.WeakKeyDictionary()


def _cached_kernel(program: MonadicProgram, data: tuple[str, ...]) -> ProgramKernel:
    """The kernel encoding of a program at the given data names.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    data : tuple[str, ...]
        The names of the host data a run supplies, sorted.

    Returns
    -------
    ProgramKernel
        The encoding, built once per program and data-name set.
    """
    kernels = _KERNELS.setdefault(program, {})
    kernel = kernels.get(data)
    if kernel is None:
        kernel = program_kernel(program, data)
        kernels[data] = kernel
    return kernel


def _checked_registry(
    program: MonadicProgram, kernel: ProgramKernel, module: QiecModule
) -> KernelRegistry:
    """The registry of a validated run module.

    Parameters
    ----------
    program : MonadicProgram
        The program the run module was built for.
    kernel : ProgramKernel
        The encoding the run module extends.
    module : QiecModule
        The run module: the kernel's declarations, the run's handlers,
        and the run's wrapper.

    Returns
    -------
    KernelRegistry
        The registry `validate_module` resolved, once per encoding and
        handler set: the module's declarations and wrapper are fixed by
        those, so two such runs build the same module.
    """
    registries = _REGISTRIES.setdefault(program, {})
    key = (kernel.data, tuple(module.handlers))
    registry = registries.get(key)
    if registry is None:
        registry = validate_module(module)
        registries[key] = registry
    return registry


def _element_admits(element: TypeExpr, value: torch.Tensor) -> bool:
    """Whether a tensor's dtype holds values of a checked element type.

    Parameters
    ----------
    element : TypeExpr
        The checked element type: ``Real``, ``Int``, or ``Bool``.
    value : torch.Tensor
        The supplied tensor.

    Returns
    -------
    bool
        True when the dtype is floating for ``Real``, a non-boolean
        integer for ``Int``, and boolean for ``Bool``; True for any other
        element type, whose host representation the run does not fix.
    """
    if element == REAL:
        return value.is_floating_point()
    if element == INT:
        return (
            not value.is_floating_point()
            and not value.is_complex()
            and (value.dtype != torch.bool)
        )
    if element == BOOL:
        return value.dtype == torch.bool
    return True


def check_host_data(
    program: MonadicProgram,
    data: Mapping[str, torch.Tensor | float | int],
    runs: int | None = None,
) -> dict[str, bool]:
    """Refuse host data that does not inhabit the program's checked types.

    The checked elaboration types every free name a program reads, and
    every target the module renders to reads the name at that type. A
    datum whose shape or element type differs would make the run compute
    a model the checked module does not describe, so it is refused. A
    dimension the checked type leaves as an extent variable takes the
    length of the first datum it shapes, and every later datum it shapes
    must agree. A run over an input batch replicates the whole program
    once per batch entry, so a datum may also carry one leading axis of
    the batch's length, one value of its checked type per run. A datum
    that fills a parameter of a plated draw directly, listed in the
    program's ``host_data_shared``, may also omit the plate axis its
    checked type leads with: it is then one value every row shares.

    Parameters
    ----------
    program : MonadicProgram
        The program, whose ``host_data_types`` give the checked types.
    data : Mapping[str, torch.Tensor | float | int]
        The supplied host data, by name.
    runs : int | None
        The length of the input batch, whose axis a datum may lead with;
        ``None`` admits no such axis.

    Returns
    -------
    dict[str, bool]
        For each checked datum supplied, whether it leads with the input
        batch's axis, one value per run, rather than one value every run
        shares.

    Raises
    ------
    ValueError
        If a datum's shape differs from the shape its checked type names,
        or an extent variable is fixed at two different lengths.
    TypeError
        If a datum's element type is not the one its checked type names.
    """
    extents: dict[str, tuple[int, str]] = {}
    per_run_data: dict[str, bool] = {}
    for name, expected in program.host_data_types.items():
        if name not in data:
            continue
        split = tensor_shape(expected)
        element, dimensions = (expected, ()) if split is None else split
        rendered = render_static(expected)
        if element == STRING:
            # Text has no tensor form: a sentence is a sequence of tokens,
            # a corpus a sequence of equally long sentences.
            supplied = _string_shape(name, data[name], rendered)
            value = None
        else:
            value = torch.as_tensor(data[name])
            supplied = tuple(value.shape)
        layouts = [tuple(dimensions)]
        if name in program.host_data_shared and dimensions:
            layouts.append(tuple(dimensions[1:]))
        per_run, layout = _matching_layout(supplied, layouts, runs)
        if layout is None:
            admissible = " or ".join(_rendered_shape(item) for item in layouts)
            shared = (
                ", the second a value every row of the plate shares"
                if len(layouts) > 1
                else ""
            )
            raise ValueError(
                f"host datum {name!r} has shape {supplied}, but the checked "
                f"program reads it at type {rendered}, of shape "
                f"{admissible}{shared}; supply data of that shape"
            )
        per_run_data[name] = per_run != supplied
        dimensions = layout
        for dimension, extent in zip(dimensions, per_run):
            if not isinstance(dimension, IndexVariable):
                continue
            fixed = extents.setdefault(dimension.name, (extent, name))
            if fixed[0] != extent:
                raise ValueError(
                    f"host datum {name!r} fixes extent {dimension.name!r} at "
                    f"{extent}, but {fixed[1]!r} fixes it at {fixed[0]}"
                )
        if value is not None and not _element_admits(element, value):
            raise TypeError(
                f"host datum {name!r} has dtype {value.dtype}, but the checked "
                f"program reads it at type {rendered}, whose elements are "
                f"{render_static(element)}"
            )
    return per_run_data


def _string_shape(name: str, value: object, rendered: str) -> tuple[int, ...]:
    """The shape of a nested sequence of tokens.

    Parameters
    ----------
    name : str
        The datum's name, for the diagnostic.
    value : object
        The datum: a string, or a sequence (list or tuple) of data of one
        shape.
    rendered : str
        The checked type, for the diagnostic.

    Returns
    -------
    tuple[int, ...]
        The lengths of its nesting levels; empty for one string.

    Raises
    ------
    TypeError
        If a leaf is not a string.
    ValueError
        If the sequences at one level differ in length.
    """
    if isinstance(value, str):
        return ()
    if not isinstance(value, list | tuple):
        raise TypeError(
            f"host datum {name!r} holds {type(value).__name__}, but the checked "
            f"program reads it at type {rendered}, whose elements are strings"
        )
    shapes = {_string_shape(name, item, rendered) for item in value}
    if len(shapes) > 1:
        raise ValueError(
            f"host datum {name!r} holds sequences of different lengths, but the "
            f"checked program reads it at type {rendered}; a corpus takes "
            "sentences of one length"
        )
    inner = shapes.pop() if shapes else ()
    return (len(value), *inner)


def _matching_layout(
    supplied: tuple[int, ...],
    layouts: list[tuple[IndexTerm, ...]],
    runs: int | None,
) -> tuple[tuple[int, ...], tuple[IndexTerm, ...] | None]:
    """The first admissible layout a supplied shape inhabits.

    Parameters
    ----------
    supplied : tuple[int, ...]
        The datum's shape.
    layouts : list[tuple[IndexTerm, ...]]
        The checked shapes the datum may take, in order of preference.
    runs : int | None
        The input batch's length, whose axis the datum may lead with.

    Returns
    -------
    tuple[tuple[int, ...], tuple[IndexTerm, ...] | None]
        The datum's shape for one run and the layout it inhabits, or the
        supplied shape and ``None`` when it inhabits none.
    """
    for layout in layouts:
        candidates = [supplied]
        if (
            runs is not None
            and len(supplied) == len(layout) + 1
            and supplied[0] == runs
        ):
            candidates.append(supplied[1:])
        for per_run in candidates:
            if len(per_run) == len(layout) and all(
                not isinstance(dimension, IndexLiteral) or dimension.value == extent
                for dimension, extent in zip(layout, per_run)
            ):
                return per_run, layout
    return supplied, None


def _rendered_shape(dimensions: tuple[IndexTerm, ...]) -> str:
    """Render a checked shape as a tuple of extents.

    Parameters
    ----------
    dimensions : tuple[IndexTerm, ...]
        The checked dimensions.

    Returns
    -------
    str
        The shape, as ``(3, 2)``, an extent variable by its name.
    """
    parts = [render_static(dimension) for dimension in dimensions]
    if len(parts) == 1:
        return f"({parts[0]},)"
    return f"({', '.join(parts)})"


def run_program(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Execute a program on the reference machine under the handler stack.

    An input of ``B`` rows is ``B`` independent runs of the whole program.
    Each run reads its slice of every value given one per run (a leading
    axis of length ``B``) and the whole of every value the runs share;
    the handlers on the stack are installed on every run, and the runs'
    outputs stack along a leading run axis.

    Parameters
    ----------
    program : MonadicProgram
        Program to execute.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    observations : Mapping[str, torch.Tensor] or None
        Values to condition observed sites on, keyed by site name.
        Undeclared keys are host data the program's steps read by name.

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        The program's return value: the single returned variable, or the
        returned variables keyed by their labels.

    Raises
    ------
    ValueError
        If an observation names one variable of a destructuring draw,
        or a value is given for an occurrence of a scanned site the run
        never reaches.
    """
    sites, supplied, data = _split_observations(program, observations or {})
    runs = int(x.shape[0]) if x.dim() else 1
    per_run_data = check_host_data(program, data, runs=runs if x.dim() else None)
    kernel = _cached_kernel(program, tuple(sorted(data)))
    _check_sequence_layouts(kernel, observations or {})
    if runs <= 1:
        run_sites = {
            label: _run_slice(kernel, label, value, 1, 0)
            for label, value in sites.items()
        }
        output, _ = _run_once(
            program, kernel, x, run_sites, supplied, data, per_run_data, 0, 1
        )
        return output
    vectorized = _vectorized_sites(program, kernel, sites, supplied, data, runs)
    if vectorized is not None:
        output, _ = _run_once(
            program,
            kernel,
            x,
            vectorized,
            supplied,
            data,
            per_run_data,
            runs - 1,
            runs,
            vectorized=True,
        )
        return output
    outputs: list[torch.Tensor | dict[str, torch.Tensor]] = []
    axis: _RunAxis | None = None
    for index in range(runs):
        run_sites = {
            label: _run_slice(kernel, label, value, runs, index)
            for label, value in sites.items()
        }
        run_supplied = {
            name: _per_run_value(name, value, runs, index)
            for name, value in supplied.items()
        }
        run_data = {
            name: value[index : index + 1] if per_run_data.get(name, False) else value
            for name, value in data.items()
        }
        output, axis = _run_once(
            program,
            kernel,
            x[index : index + 1],
            run_sites,
            run_supplied,
            run_data,
            per_run_data,
            index,
            runs,
        )
        outputs.append(output)
    assert axis is not None
    return _stack_runs(program, outputs, axis)


def _run_slice(
    kernel: ProgramKernel, label: str, value: torch.Tensor, runs: int, index: int
) -> torch.Tensor:
    """One run's value of a site given for every run of a call.

    A site's value for a call of ``runs`` runs is laid out either one
    value per run, along a leading axis of length ``runs`` ahead of the
    value one run takes, or as one value every run shares. A plated
    site's one-run value holds its plate's rows; any other site's leads
    with a run axis of length one.

    Parameters
    ----------
    kernel : ProgramKernel
        The program's encoding, which names each site's morphism.
    label : str
        The site's label or occurrence address.
    value : torch.Tensor
        The value given.
    runs : int
        The number of runs.
    index : int
        The run.

    Returns
    -------
    torch.Tensor
        The run's value.

    Raises
    ------
    ValueError
        If the value is laid out neither way.
    """
    morphism = _site_morphism(kernel, label)
    if isinstance(morphism, MarginalizedFactor):
        # The rows a marginalized block observes, which the block's
        # integrated score reads: one set per run along a leading axis
        # of length ``runs``, or one set every run shares.
        if runs > 1 and value.dim() >= 2 and int(value.shape[0]) == runs:
            return value[index]
        return value
    if isinstance(morphism, PlateDraw | VectorisedObserve):
        single = _plated_numel(morphism)
        if (
            runs > 1
            and value.dim() >= 1
            and int(value.shape[0]) == runs
            and value.numel() == runs * single
        ):
            return _plated_value(morphism, value[index])
        if value.numel() == single:
            return _plated_value(morphism, value)
        if runs == 1:
            return value
        layout = f"{single} elements, or {runs} runs of {single}"
    else:
        single = _event_numel(morphism)
        if single is not None and value.numel() == single:
            # One value every run shares, at one run's layout: led by
            # the run axis of length one.
            if value.dim() >= 1 and int(value.shape[0]) == 1:
                return value
            return value.unsqueeze(0)
        if runs == 1:
            return value
        if (
            value.dim() >= 1
            and int(value.shape[0]) == runs
            and (single is None or value.numel() == runs * single)
        ):
            return value[index : index + 1]
        layout = (
            f"{runs} runs of one value"
            if single is None
            else f"{single} elements, or {runs} runs of {single}"
        )
    raise ValueError(
        f"site {label!r} has a value of shape {tuple(value.shape)}, but an input "
        f"of {runs} rows runs the program {runs} times, so its value has "
        f"{layout}"
    )


def _vectorized_sites(
    program: MonadicProgram,
    kernel: ProgramKernel,
    sites: Mapping[str, torch.Tensor],
    supplied: Mapping[str, torch.Tensor],
    data: Mapping[str, torch.Tensor],
    runs: int,
) -> dict[str, torch.Tensor] | None:
    """The site values for executing every run of a call at once, if it can.

    Runs execute together when the program is a chain of single-variable
    draws and observations from morphisms that are not plated, with no
    let, score, call, or scan step and no host data: each such morphism
    scores and draws one value per row of its input, so a run axis
    leading every value keeps the runs apart exactly as running them one
    by one does. Every given value must be one draw per run or one draw
    every run shares, which is expanded across the runs.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    kernel : ProgramKernel
        Its encoding, which names each site's morphism.
    sites : Mapping[str, torch.Tensor]
        The values given at sites, for every run.
    supplied : Mapping[str, torch.Tensor]
        The values given for let bindings.
    data : Mapping[str, torch.Tensor]
        The host data.
    runs : int
        The number of runs.

    Returns
    -------
    dict[str, torch.Tensor] | None
        The site values, each leading with the run axis, or ``None``
        when the runs execute one by one.
    """
    if supplied or data:
        return None
    for spec in program.steps:
        if not isinstance(spec, Draw | Observe) or len(spec.names) != 1:
            return None
        morphism = program.step_module(spec)
        if (
            not isinstance(morphism, ContinuousMorphism)
            or isinstance(morphism, MonadicProgram)
            or _plated(morphism)
        ):
            return None
    together: dict[str, torch.Tensor] = {}
    for label, value in sites.items():
        morphism = _site_morphism(kernel, label)
        single = _event_numel(morphism)
        if single is None:
            return None
        if (
            value.dim() >= 1
            and int(value.shape[0]) == runs
            and value.numel() == runs * single
        ):
            rows = value
        elif value.numel() == single:
            one = _run_slice(kernel, label, value, 1, 0)
            rows = one.expand(runs, *one.shape[1:])
        else:
            return None
        codomain = getattr(morphism, "codomain", None)
        if isinstance(codomain, ContinuousSpace) and len(codomain.event_shape) == 1:
            # A continuous draw is one row of its event per run, so a
            # scalar's runs are a column, never a row the event reads.
            rows = rows.reshape(runs, single)
        together[label] = rows
    return together


def _site_carries(kernel: ProgramKernel, label: str, value: torch.Tensor) -> bool:
    """Whether one run's given value of a site leads with the run axis.

    Parameters
    ----------
    kernel : ProgramKernel
        The program's encoding, which names each site's morphism.
    label : str
        The site's label.
    value : torch.Tensor
        The run's value, as `_run_slice` lays it out.

    Returns
    -------
    bool
        ``False`` for a plated site and for an observation laid out
        along rows, whose element count is not one draw's; ``True`` for
        one draw led by the run axis.
    """
    morphism = _site_morphism(kernel, label)
    if _plated(morphism):
        return False
    single = _event_numel(morphism)
    if single is None:
        return value.dim() >= 1 and int(value.shape[0]) == 1
    return value.numel() == single and value.dim() >= 1 and int(value.shape[0]) == 1


def _per_run_value(
    name: str, value: torch.Tensor, runs: int, index: int
) -> torch.Tensor:
    """One run's value of a let binding supplied for every run of a call.

    Parameters
    ----------
    name : str
        The binding, unused.
    value : torch.Tensor
        The value given: one per run along a leading axis of length
        ``runs``, or one every run shares.
    runs : int
        The number of runs.
    index : int
        The run.

    Returns
    -------
    torch.Tensor
        The run's slice, leading with an axis of length one, of a value
        laid out one per run; the value itself otherwise.
    """
    del name
    if value.dim() >= 1 and int(value.shape[0]) == runs:
        return value[index : index + 1]
    return value


def _stack_runs(
    program: MonadicProgram,
    outputs: list[torch.Tensor | dict[str, torch.Tensor]],
    axis: _RunAxis,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Stack the outputs of a call's runs along the run axis.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    outputs : list[torch.Tensor | dict[str, torch.Tensor]]
        Each run's output.
    axis : _RunAxis
        The classification of the program's bindings, which says
        whether each returned variable leads with the run axis already.

    Returns
    -------
    torch.Tensor | dict[str, torch.Tensor]
        The outputs, one per run along the leading axis.
    """

    def combine(name: str, values: list[torch.Tensor]) -> torch.Tensor:
        """Join one returned variable's per-run values along the run axis.

        Parameters
        ----------
        name : str
            The returned variable.
        values : list[torch.Tensor]
            Its value in each run.

        Returns
        -------
        torch.Tensor
            The values concatenated when each already leads with the
            run axis, else stacked along a new one.
        """
        if axis.carries(program, name):
            return torch.cat(values, dim=0)
        return torch.stack(values, dim=0)

    if program._return_is_single:
        return combine(
            program._return_vars[0],
            [output for output in outputs if isinstance(output, torch.Tensor)],
        )
    keys = program._return_labels if program._return_labels else program._return_vars
    return {
        key: combine(name, [cast_dict(output)[key] for output in outputs])
        for key, name in zip(keys, program._return_vars, strict=True)
    }


def _run_once(
    program: MonadicProgram,
    kernel: ProgramKernel,
    x: torch.Tensor,
    sites: dict[str, torch.Tensor],
    supplied: dict[str, torch.Tensor],
    data: dict[str, torch.Tensor],
    per_run_data: Mapping[str, bool],
    run_index: int,
    runs: int,
    *,
    vectorized: bool = False,
) -> tuple[torch.Tensor | dict[str, torch.Tensor], _RunAxis]:
    """Execute one run of a program on the reference machine.

    With ``vectorized``, the call's ``runs`` runs execute together, each
    value leading with the run axis; see `_vectorized_sites` for the
    programs this applies to.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    kernel : ProgramKernel
        Its encoding at the run's host data.
    x : torch.Tensor
        The run's input, one row.
    sites : dict[str, torch.Tensor]
        The run's values at declared sites.
    supplied : dict[str, torch.Tensor]
        The run's values standing in for let bindings.
    data : dict[str, torch.Tensor]
        The run's host data.
    per_run_data : Mapping[str, bool]
        Whether each checked host datum was given one value per run; a
        datum the program does not type is one value every run shares.
    run_index : int
        The run's position among the call's runs.
    runs : int
        The number of runs the call makes.
    vectorized : bool
        Whether this execution carries every run of the call at once.

    Returns
    -------
    tuple[torch.Tensor | dict[str, torch.Tensor], _RunAxis]
        The run's output and the classification of its bindings.

    Raises
    ------
    ValueError
        If a value is given for an occurrence of a scanned site the run
        never reaches.
    """
    axis = _RunAxis(program, {name: per_run_data.get(name, False) for name in data})
    for label, value in sites.items():
        if "@" not in label:
            axis.given(program, label, _site_carries(kernel, label, value))
    run = RunContext(
        kernel,
        host_value,
        observations=dict(sites),
        supplied=supplied,
        run_index=run_index,
        runs=runs,
        slice_value=(
            (lambda label, value: value)
            if vectorized
            else (
                lambda label, value: _run_slice(kernel, label, value, runs, run_index)
            )
        ),
        carries_run_axis=axis.carries,
        site_carries_run_axis=axis.site,
        vectorized=vectorized,
    )
    stack = list(_handler_stack())
    steps = _HostSteps(run, x, axis)
    result_type = kernel.result_type

    def record(request: RuntimeRequest, weight: object) -> None:
        """Record a contribution reaching the accumulator.

        Parameters
        ----------
        request : RuntimeRequest
            The score request.
        weight : object
            The contribution.
        """
        assert isinstance(weight, torch.Tensor | float | int)
        run.contributions.append(
            Contribution(
                run.key_of(request),
                request.core.origin.origin.role,
                torch.as_tensor(weight),
                request.core.origin.origin.structural_path,
            )
        )

    accumulator, _ = score_handler(
        answer_type=result_type, observer=record, key="run-accumulator"
    )
    score_installations = [Installation(kernel.score, accumulator)]
    random_installations: list[Installation] = []
    param_installations: list[Installation] = []
    user_random: list[Installation] = []
    step_observers: list[Installation] = []
    step_instances = {item.entry.instance for item in kernel.module.instances}
    finishers: list[EffectHandler] = []
    for handler in stack:
        for installation in handler.install(run):
            if installation.instance is kernel.score:
                score_installations.append(installation)
            elif installation.instance is kernel.random:
                user_random.append(installation)
            elif installation.instance is kernel.param:
                param_installations.append(installation)
            elif installation.instance.entry.instance in step_instances:
                step_observers.append(installation)
            else:
                raise ValueError(
                    f"{type(handler).__name__} installs on an instance the run "
                    "does not serve"
                )
        finishers.append(handler)
    random_installations.append(
        Installation(
            kernel.random,
            draw_scoring_handler(
                score_instance=kernel.score.entry.instance,
                result_validator=host_value,
                answer_type=result_type,
                key="run-draw",
            ),
        )
    )
    if sites:
        clamped = {
            label: _canonical(_site_morphism(kernel, label), value)
            for label, value in sites.items()
        }
        random_installations.append(
            Installation(
                kernel.random,
                condition_handler(
                    clamped,
                    score_instance=kernel.score.entry.instance,
                    result_validator=host_value,
                    missing=MissingValuePolicy.FORWARD,
                    extra=ExtraValuePolicy.IGNORE,
                    answer_type=result_type,
                    key="run-observations",
                ),
            )
        )
    random_installations.extend(user_random)
    store: dict[str, torch.Tensor] = {}
    for step in kernel.steps:
        if step.kind != "draw" or step.site is None:
            continue
        assert isinstance(step.spec, Draw | Observe)
        morphism = step.program.step_module(step.spec)
        assert isinstance(morphism, ContinuousMorphism)
        for name, parameter in morphism.named_parameters():
            store[f"{step.site}.{name}"] = parameter
    run.parameters = store
    param_installations.insert(
        0,
        Installation(
            kernel.param,
            param_handler(
                lambda name, _request: store[name],
                result_validator=host_value,
                answer_type=result_type,
                key="run-parameters",
            ),
        ),
    )
    step_installations: list[Installation] = []
    for step in kernel.steps:
        instance = next(
            item
            for item in kernel.module.instances
            if item.entry.instance == step.instance
        )
        introduced = (
            EffectRow((kernel.param.entry,)) if step.kind == "draw" else EffectRow()
        )
        step_installations.append(
            Installation(
                instance,
                compute_handler(
                    steps.function(step),
                    effect=step.effect,
                    result_validator=host_value,
                    introduced=introduced,
                    answer_type=result_type,
                    key=f"run-step-{instance.name}",
                ),
            )
        )
    entry = kernel.module.computations[0]
    arguments = [x, *(data[name] for name in kernel.data)]
    parameters = [Local(local.name, local.type) for local in entry.parameters]
    call = Call(
        entry.id,
        entry.name,
        (),
        tuple(Var(parameter) for parameter in parameters),
        result_type,
        entry.type.effects,
        SourceOrigin(kernel.module.module, ("run", "call"), "call", SOURCE_PROTOCOL),
    )
    # A handler is installed only over an effect the entry performs:
    # a program with no draw performs neither Random nor Param, and
    # one with no draw and no score step performs no Score either.
    has_draws = any(step.kind == "draw" for step in kernel.steps)
    performs_score = entry.type.effects.lookup(kernel.score.entry.instance) is not None
    if not has_draws:
        random_installations = []
        param_installations = []
        if not performs_score:
            score_installations = []
    installations = [
        installation.named(f"{installation.runtime.definition.name}#{position}")
        for position, installation in enumerate(
            [
                *score_installations,
                *random_installations,
                *param_installations,
                *step_installations,
                *step_observers,
            ]
        )
    ]
    body = _handle_chain(installations, call)
    wrapper = NamedComputation(
        ComputationId.derive(kernel.module.module, "computation", RUN),
        RUN,
        (),
        tuple(parameters),
        body,
        ComputationType(EffectRow(), result_type),
        SourceOrigin(kernel.module.module, ("run",), "computation", SOURCE_PROTOCOL),
    )
    module = QiecModule(
        kernel.module.module,
        kernel.module.source_protocol,
        effects=kernel.module.effects,
        instances=kernel.module.instances,
        handlers=tuple(
            installation.runtime.definition for installation in installations
        ),
        computations=(*kernel.module.computations, wrapper),
    )
    registry = _checked_registry(program, kernel, module)
    attachments = RuntimeAttachments()
    for installation in installations:
        attachments.bind_handler(installation.runtime)
    evaluator = Evaluator(
        attachments, module=module, type_validator=lambda _type: host_value
    )
    # The run owns its handlers: a program a host step runs in turn (a
    # score closing over another program's joint, a sub-program drawn as
    # a step) is a separate invocation, whose sites this run's handlers
    # neither observe nor condition. The stack is cleared for the
    # evaluation and restored after it.
    active = _handler_stack()
    entered = active[:]
    active.clear()
    try:
        output = evaluator.evaluate_validated(
            body, registry, dict(zip(parameters, arguments, strict=True))
        )
    finally:
        active.extend(entered)
    unreached = sorted(
        address
        for address in sites
        if address.rpartition("@")[0] in kernel.scanned and address not in run.reached
    )
    if unreached:
        raise ValueError(
            f"the run never reached {', '.join(repr(item) for item in unreached)}: "
            f"a scan reaches its sites once per position of its sequence, and "
            f"the sequence has fewer positions than the values given"
        )
    for handler in finishers:
        handler.finish(run, output)
    if program._return_is_single:
        assert isinstance(output, torch.Tensor)
        return output, axis
    assert isinstance(output, tuple)
    keys = program._return_labels if program._return_labels else program._return_vars
    return {key: value for key, value in zip(keys, output, strict=True)}, axis


def sample_program(
    program: MonadicProgram,
    x: torch.Tensor,
    sample_shape: torch.Size = torch.Size(),
    observations: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Run a program forward under the handler stack, with sample dimensions.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    sample_shape : torch.Size
        Leading sample dimensions. Each element of the shape is one
        independent run of the program; the returned values stack the
        runs ahead of their batch axis.
    observations : Mapping[str, torch.Tensor] or None
        Values to condition observed sites on, keyed by site name, and
        host data the steps read by name.

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        The program's return value, with ``sample_shape`` in front.
    """
    if not sample_shape:
        return run_program(program, x, observations)
    count = 1
    for extent in sample_shape:
        count *= int(extent)
    runs = [run_program(program, x, observations) for _ in range(count)]
    first = runs[0]
    if isinstance(first, torch.Tensor):
        stacked = torch.stack([run for run in runs if isinstance(run, torch.Tensor)])
        return stacked.reshape(*sample_shape, *first.shape)
    assert isinstance(first, dict)
    return {
        key: torch.stack([cast_dict(run)[key] for run in runs]).reshape(
            *sample_shape, *first[key].shape
        )
        for key in first
    }


def cast_dict(value: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Read a program's dictionary return.

    Parameters
    ----------
    value : torch.Tensor | dict[str, torch.Tensor]
        A program's return value.

    Returns
    -------
    dict[str, torch.Tensor]
        The value, which is a dictionary.

    Raises
    ------
    TypeError
        If the value is a single tensor.
    """
    if isinstance(value, dict):
        return value
    raise TypeError("the program returns a single tensor, not named values")


def log_joint(
    program: MonadicProgram,
    x: torch.Tensor,
    intermediates: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """The joint log density of a program at values for all of its draws.

    The program runs on the reference machine under the handler stack
    with every draw conditioned on its given value. An input of ``B``
    rows is ``B`` independent runs of the whole program, and the joint
    is one number per run: the sum of that run's site densities. The
    joint of all the runs together is the sum of the result.

    Parameters
    ----------
    program : MonadicProgram
        The program.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    intermediates : Mapping[str, torch.Tensor]
        A value for every draw the program makes, keyed by variable
        name or by return label, together with any host data its steps
        read by name. A value is given either once per run, along a
        leading axis of length ``B``, or once for every run to share.

    Returns
    -------
    torch.Tensor
        The joint log density of each run, shaped ``(B,)``.

    Raises
    ------
    KeyError
        If a draw the program makes is given no value.
    """
    values = dict(intermediates)
    if program._return_labels:
        for label, name in zip(
            program._return_labels, program._return_vars, strict=True
        ):
            if label in values and name not in values:
                values[name] = values[label]
    with TraceHandler() as handler:
        run_program(program, x, values)
    for name, site in handler.trace.sites.items():
        if (
            site.sampleable is not None
            and not site.is_observed
            and not site.is_deterministic
        ):
            raise KeyError(name)
    return handler.total_log_joint(x.device)


class _ReferenceEvaluator:
    """The reference machine as the evaluator programs run through."""

    def sample_program(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        sample_shape: torch.Size,
        observations: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the program forward; see `sample_program`.

        Parameters
        ----------
        program : MonadicProgram
            The program.
        x : torch.Tensor
            Program input.
        sample_shape : torch.Size
            Leading sample dimensions.
        observations : Mapping[str, torch.Tensor] or None
            Values to clamp observed variables to, and host data.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            The program's return value.
        """
        return sample_program(program, x, sample_shape, observations)

    def log_joint(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        intermediates: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score the program; see `log_joint`.

        Parameters
        ----------
        program : MonadicProgram
            The program.
        x : torch.Tensor
            Program input.
        intermediates : Mapping[str, torch.Tensor]
            A value for every draw, and host data.

        Returns
        -------
        torch.Tensor
            The joint log density.
        """
        return log_joint(program, x, intermediates)


install_evaluator(_ReferenceEvaluator())

__all__ = [
    "run_program",
]
