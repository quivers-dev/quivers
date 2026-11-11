"""Monadic programs: sequenced probabilistic programs as ContinuousMorphisms.

A MonadicProgram defines a ContinuousMorphism via monadic sequencing
of draw steps. Each step samples from a named morphism, optionally
conditioned on previously drawn variables, and binds the result.
The program returns one or more of the bound variables as its output.

This corresponds to the Kleisli composition pattern used in
probabilistic programming languages like PDS (Grove & White),
where a sequence of ``let' x ~ D in ...`` bindings threads
probabilistic state through a generative model.

Features
--------
- Single and tuple returns: ``return x`` or ``return (x, y, z)``
- Named input parameters for product-domain sub-programs
- Multi-argument draw steps: ``draw z ~ f(x, y)``
- Destructuring draws from tuple-returning sub-programs:
  ``draw (a, b) ~ sub_prog(x)``

Example
-------
Given morphisms f : A -> B and g : B -> C, the monadic program::

    program p : A -> C
        draw x ~ f
        draw y ~ g(x)
        return y

is equivalent to the composition f >> g, but the program form
allows fan-out (using the input in multiple draws) and
non-linear variable dependency graphs.

PDS-style nested programs::

    program cg_update(y, z) : Belief * Belief -> Truth * Truth
        draw c ~ bern_c(y)
        draw d ~ bern_d(z)
        return (c, d)

    program factivityPrior : Entity -> Truth * Truth * Truth
        draw x ~ prior_x
        draw y ~ prior_y
        draw z ~ prior_z
        draw b ~ bern_b(x)
        draw (c, d) ~ cg_update(y, z)
        return (b, c, d)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

import torch
from torch import nn

from quivers.continuous.morphisms import (
    AnySpace,
    ContinuousMorphism,
    SampledComposition,
)
from quivers.continuous.program_steps import (
    Call,
    Draw,
    Indexed,
    Let,
    Observe,
    Scan,
    Score,
    Step,
    StepArgument,
)
from quivers.continuous.scan import ScanMorphism
from quivers.core.morphisms import as_torch_module
from quivers.qiec.types import TypeExpr


def _lookup_arg(
    env: Mapping[str, torch.Tensor],
    arg: StepArgument,
    carries: Callable[[str], bool] | None = None,
) -> torch.Tensor:
    """Resolve a draw or observe argument against the environment.

    Parameters
    ----------
    env : Mapping[str, torch.Tensor]
        The bindings so far.
    arg : StepArgument
        A binding's name, or an
        [`Indexed`][quivers.continuous.program_steps.Indexed] gather,
        which indexes the binding at each index binding in turn.
    carries : Callable[[str], bool] | None
        Whether a binding leads with the run axis of one run of the
        program. A gather is taken within the run: the run axis is
        dropped from the table and the indices before indexing and
        restored on the result, so an index never reads the run axis.
        ``None`` treats no binding as carrying it.

    Returns
    -------
    torch.Tensor
        The argument's value.

    Raises
    ------
    KeyError
        If a name the argument reads is unbound.
    """
    if isinstance(arg, Indexed):
        if arg.name not in env:
            raise KeyError(arg.name)
        tensor = env[arg.name]
        carried = carries is not None and carries(arg.name)
        if carried:
            tensor = tensor[0]
        for ix in arg.indices:
            if ix not in env:
                raise KeyError(ix)
            index = env[ix]
            if carries is not None and carries(ix):
                index = index[0]
                carried = True
            tensor = tensor[index]
        return tensor.unsqueeze(0) if carried else tensor
    if arg in env:
        return env[arg]
    raise KeyError(arg)


def _bound_names(step: Step) -> tuple[str, ...]:
    """The names a step binds.

    Parameters
    ----------
    step : Step
        A program step.

    Returns
    -------
    tuple[str, ...]
        A draw's or observe's names, or a let's or score's one name.
    """
    if isinstance(step, Draw | Observe):
        return step.names
    if isinstance(step, Let | Score | Call | Scan):
        return (step.name,)
    raise TypeError(f"{type(step).__name__} is not a program step")


def _runs_a_scan(morphism: object) -> bool:
    """Whether a morphism runs a scan.

    Parameters
    ----------
    morphism : object
        A draw's morphism.

    Returns
    -------
    bool
        ``True`` for a [`ScanMorphism`][quivers.continuous.ScanMorphism],
        or a composition with one among its factors.
    """
    if isinstance(morphism, ScanMorphism):
        return True
    if isinstance(morphism, SampledComposition):
        return any(_runs_a_scan(factor) for factor in morphism.factors)
    return False


def _check_call(step: Call) -> MonadicProgram:
    """Check a call step's callee against its arguments.

    Parameters
    ----------
    step : Call
        The step.

    Returns
    -------
    MonadicProgram
        The callee.

    Raises
    ------
    ValueError
        If the callee is not a program returning one value, or the
        arguments do not match its parameters.
    """
    callee = step.program
    if not isinstance(callee, MonadicProgram) or not callee._return_is_single:
        raise ValueError(
            f"call {step.name!r}: the callee must be a MonadicProgram returning "
            f"one value"
        )
    arity = len(callee._params) if callee._params is not None else 1
    if len(step.args) != arity:
        raise ValueError(
            f"call {step.name!r}: the callee takes {arity} argument(s), "
            f"got {len(step.args)}"
        )
    return callee


def _check_scan(step: Scan) -> MonadicProgram:
    """Check a scan step's step program and initial state.

    Parameters
    ----------
    step : Scan
        The step.

    Returns
    -------
    MonadicProgram
        The step program.

    Raises
    ------
    ValueError
        If the step program is not a program of an entry and a state
        returning one value, or the initial state is not a vector of
        the state's width.
    """
    body = step.step
    if (
        not isinstance(body, MonadicProgram)
        or body._params is None
        or len(body._params) != 2
        or not body._return_is_single
    ):
        raise ValueError(
            f"scan {step.name!r}: the step must be a MonadicProgram of two "
            f"parameters, the entry and the previous state, returning the "
            f"next state"
        )
    if step.initial is not None:
        width = getattr(body.codomain, "dim", None)
        if step.initial.dim() != 1 or (
            width is not None and int(step.initial.shape[0]) != int(width)
        ):
            raise ValueError(
                f"scan {step.name!r}: the initial state has shape "
                f"{tuple(step.initial.shape)}, not the ({width},) of the state"
            )
    return body


def _parameter_dtype(parts: Sequence[torch.Tensor]) -> torch.dtype:
    """The floating dtype stacked parameters are carried in.

    The widest floating dtype among the parts, so a float64 program
    stays float64; the default floating dtype when every part is an
    integer or boolean index.

    Parameters
    ----------
    parts : Sequence[torch.Tensor]
        The parameter tensors to stack.

    Returns
    -------
    torch.dtype
        The dtype every part is converted to.
    """
    floating = [t.dtype for t in parts if t.is_floating_point()]
    if not floating:
        return torch.get_default_dtype()
    dtype = floating[0]
    for other in floating[1:]:
        dtype = torch.promote_types(dtype, other)
    return dtype


def step_key(step: Draw | Observe) -> str:
    """The submodule name a program registers a draw's morphism under.

    Parameters
    ----------
    step : Draw | Observe
        A draw or observe step.

    Returns
    -------
    str
        ``_step_`` followed by the step's first bound name.
    """
    return f"_step_{step.names[0]}"


def _argument_text(arg: StepArgument) -> str:
    """Render a step argument as the source form it stands for.

    Parameters
    ----------
    arg : StepArgument
        A binding's name, or an indexed gather of one.

    Returns
    -------
    str
        ``name`` or ``name[index, ...]``.
    """
    if isinstance(arg, Indexed):
        return f"{arg.name}[{', '.join(arg.indices)}]"
    return arg


class ProgramEvaluator(Protocol):
    """What runs a program: the reference machine under the handler stack.

    The evaluator lives in `quivers.effects`, which builds on programs;
    it installs itself here when imported, so a program's own methods
    reach it without the program package depending on the effects
    package at import.
    """

    def sample_program(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        sample_shape: torch.Size,
        observations: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the program forward.

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
        ...

    def log_joint(
        self,
        program: MonadicProgram,
        x: torch.Tensor,
        intermediates: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score the program at values for all of its draws.

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
            The joint log density of each run of the input, shape
            ``(B,)``.
        """
        ...


_EVALUATOR: ProgramEvaluator | None = None


def install_evaluator(evaluator: ProgramEvaluator) -> None:
    """Install the evaluator every program's methods run through.

    Parameters
    ----------
    evaluator : ProgramEvaluator
        The evaluator.
    """
    global _EVALUATOR
    _EVALUATOR = evaluator


def _evaluator() -> ProgramEvaluator:
    """The installed evaluator.

    Returns
    -------
    ProgramEvaluator
        The evaluator `quivers.effects` installed.

    Raises
    ------
    RuntimeError
        If `quivers.effects` has not been imported.
    """
    if _EVALUATOR is None:
        raise RuntimeError(
            "no program evaluator is installed; import `quivers.effects`, "
            "whose interpreter runs programs on the reference machine"
        )
    return _EVALUATOR


class MonadicProgram(ContinuousMorphism):
    """A probabilistic program defined by monadic sequencing of draw steps.

    Each draw step samples from a ContinuousMorphism and binds the
    result to one or more named variables. Later steps can reference
    earlier bindings as their input. The program's output is the
    value(s) of the designated return variable(s).

    Parameters
    ----------
    domain : SetObject or ContinuousSpace
        The program's input space.
    codomain : SetObject or ContinuousSpace
        The program's output space.
    steps : Sequence[Step]
        The program body, in order: [`Draw`][quivers.continuous.Draw],
        [`Observe`][quivers.continuous.Observe],
        [`Let`][quivers.continuous.Let], and
        [`Score`][quivers.continuous.Score] records.
    return_vars : tuple[str, ...]
        Name(s) of the bound variable(s) whose value(s) are
        the program output.
    params : tuple[str, ...] or None
        Named input parameters for product-domain programs.
        When set, the program input is split along the feature
        dimension and each component is pre-bound in the env.
    return_labels : tuple[str, ...] or None
        Optional labels for tuple return fields. When set, the
        output dict uses these labels as keys instead of the
        variable names. Length must match return_vars.
    effect_set : frozenset[str] or None
        The declared effect row (``Sample``, ``Score``, ``Marginal``,
        ``Pure``), or ``None`` when unannotated.

    Attributes
    ----------
    host_data_types : dict[str, TypeExpr]
        The checked type of each host datum the program reads by name,
        keyed by name. The DSL compiler fills it from the program's
        checked elaboration; a run refuses a supplied datum whose shape
        or element type does not inhabit its entry. Empty for a program
        built directly, whose data carry no checked contract.
    host_data_shared : frozenset[str]
        The host data whose checked type leads with the plate of the
        draw they fill; each may also be supplied without that axis, as
        one value every row shares.

    Raises
    ------
    ValueError
        If a step binds no name, or two draws or observes bind the
        same name.
    """

    def __init__(
        self,
        domain: AnySpace,
        codomain: AnySpace,
        steps: Sequence[Step],
        return_vars: tuple[str, ...],
        params: tuple[str, ...] | None = None,
        return_labels: tuple[str, ...] | None = None,
        effect_set: frozenset[str] | None = None,
    ) -> None:
        super().__init__(domain, codomain)
        self._return_vars = return_vars
        self._return_is_single = len(return_vars) == 1
        self._params = params
        self._return_labels = return_labels
        # The effect-row annotation. None when unannotated;
        # otherwise carries the declared capability set
        # (Sample / Score / Marginal / Pure) for introspection by
        # downstream inference / dispatch code.
        self.effect_set: frozenset[str] | None = effect_set
        self.host_data_types: dict[str, TypeExpr] = {}
        self.host_data_shared: frozenset[str] = frozenset()
        # compute input component dimensions for param splitting
        if params is not None and len(params) > 1:
            self._param_dims = self._compute_component_dims(domain)
            self._param_is_continuous = self._compute_component_continuous(domain)

        else:
            self._param_dims = None
            self._param_is_continuous = None

        # Each draw's morphism registers as a named submodule so its
        # parameters are visible to optimizers.
        for step in steps:
            if not _bound_names(step):
                raise ValueError(f"{step.kind} step binds no name")
            if isinstance(step, Draw | Observe):
                key = step_key(step)
                if key in self._modules:
                    raise ValueError(
                        f"two {step.kind} steps bind {step.names[0]!r}; each "
                        f"draw or observe binds a name no other one does"
                    )
                if _runs_a_scan(step.morphism):
                    # Drawn as one opaque sample, the scan's per-position
                    # sites would be neither traced nor scored.
                    raise ValueError(
                        f"{step.kind} {step.names[0]!r} draws from a scan; "
                        f"bind it with a Scan step, which runs the step's "
                        f"sites as sites of this program"
                    )
                # A backend-agnostic `Morphism` registers through its
                # parameter container, which keeps the morphism under
                # ``_morphism`` for `extract_morphism`.
                self.add_module(key, as_torch_module(step.morphism))
            elif isinstance(step, Call):
                self.add_module(f"_call_{step.name}", _check_call(step))
            elif isinstance(step, Scan):
                self.add_module(f"_scan_{step.name}", _check_scan(step))
                if isinstance(step.initial, nn.Parameter):
                    self.register_parameter(f"_scan_{step.name}_init", step.initial)
        self._steps: tuple[Step, ...] = tuple(steps)

    @property
    def steps(self) -> tuple[Step, ...]:
        """The program body, in order.

        Returns
        -------
        tuple[Step, ...]
            The records the program was built from.
        """
        return self._steps

    def step_module(self, step: Draw | Observe) -> nn.Module:
        """The registered module a draw or observe step applies.

        For a [`ContinuousMorphism`][quivers.continuous.ContinuousMorphism]
        this is the morphism itself; for a backend-agnostic
        [`Morphism`][quivers.core.Morphism] it is the parameter container
        [`as_torch_module`][quivers.core.as_torch_module] wraps it in.

        Parameters
        ----------
        step : Draw | Observe
            One of the program's steps.

        Returns
        -------
        nn.Module
            The submodule the program registered for the step.

        Raises
        ------
        KeyError
            If the program registered no module for the step.
        """
        module = self._modules[step_key(step)]
        if module is None:
            raise KeyError(step_key(step))
        return module

    @staticmethod
    def _compute_component_dims(space: AnySpace) -> list[int]:
        """Compute per-component feature dimensions for a product space.

        Parameters
        ----------
        space : AnySpace
            A product space (ProductSet or similar).

        Returns
        -------
        list[int]
            Feature dimensions for each component.
        """
        from quivers.core.objects import ProductSet
        from quivers.continuous.spaces import ContinuousSpace, ProductSpace

        if isinstance(space, (ProductSet, ProductSpace)):
            dims = []

            for c in space.components:
                if isinstance(c, ContinuousSpace):
                    dims.append(c.dim)

                else:
                    # discrete FinSet component: 1 dimension (index)
                    dims.append(1)

            return dims

        # non-product: single component
        if isinstance(space, ContinuousSpace):
            return [int(space.dim)]

        return [1]

    @staticmethod
    def _compute_component_continuous(space: AnySpace) -> list[bool]:
        """Determine which components of a product space are continuous.

        Parameters
        ----------
        space : AnySpace
            A product space.

        Returns
        -------
        list[bool]
            True for continuous components, False for discrete.
        """
        from quivers.core.objects import ProductSet
        from quivers.continuous.spaces import ContinuousSpace, ProductSpace

        if isinstance(space, (ProductSet, ProductSpace)):
            return [isinstance(c, ContinuousSpace) for c in space.components]

        return [isinstance(space, ContinuousSpace)]

    def _resolve_input(
        self,
        spec: Draw | Observe,
        x: torch.Tensor,
        env: dict[str, torch.Tensor],
        carries: Callable[[str], bool] | None = None,
    ) -> torch.Tensor:
        """Resolve the input tensor for a draw step.

        Parameters
        ----------
        spec : Draw | Observe
            The step.
        x : torch.Tensor
            The raw program input.
        env : dict[str, torch.Tensor]
            Current variable environment.
        carries : Callable[[str], bool] | None
            Whether a binding leads with the run axis, which a gathered
            argument indexes past.

        Returns
        -------
        torch.Tensor
            Input tensor for the morphism.
        """
        if spec.args is None:
            return x

        # The arguments stack along the feature dimension. When the
        # morphism declares per-parameter event ranks and dims (an
        # inline distribution), the stacking interprets each argument by
        # its declared role rather than guessing from tensor rank: a
        # rank-0 position is a per-row scalar; a rank->=1 position is a
        # vector feature of its declared dim, arriving either as a
        # shared ``(D,)`` vector or a per-row ``(N, D)`` block.
        # Any other single argument passes through, promoted to a column
        # when it is a rank-1 float.
        parts = [_lookup_arg(env, a, carries) for a in spec.args]
        morph = self._modules.get(step_key(spec))
        event_ranks = getattr(morph, "_param_event_ranks", None)
        param_spec = getattr(morph, "_param_spec", None)
        if event_ranks is not None and param_spec is not None:
            var_ranks = tuple(
                event_ranks[i] for i, (k, _) in enumerate(param_spec) if k == "var"
            )
            var_dims = tuple(
                None if v is None else int(v) for k, v in param_spec if k == "var"
            )
            if len(var_ranks) == len(parts):
                return self._stack_params(parts, var_ranks, var_dims)
        if len(parts) == 1:
            return self._promote_rank(parts[0])
        return self._stack_tensors(parts)

    @staticmethod
    def _stack_params(
        parts: list[torch.Tensor],
        event_ranks: tuple[int, ...],
        dims: tuple[int | None, ...],
    ) -> torch.Tensor:
        """Stack inline-distribution parameters into a single input.

        Each parameter is reshaped to a per-row ``(batch, dim)`` block
        using its declared event rank and dim, and the blocks are
        concatenated along the feature axis. A shared parameter (no
        leading batch: a scalar ``()``/``(1,)`` or a vector ``(dim,)``)
        reshapes to a batch-1 block and broadcasts against any per-row
        parameters. When every parameter is shared, the result is a
        single batch-1 row; the response batch is supplied downstream by
        the observed value.
        """
        dtype = _parameter_dtype(parts)
        shaped: list[torch.Tensor] = []
        for t, rank, dim in zip(parts, event_ranks, dims, strict=True):
            tf = t.to(dtype)
            if rank >= 1:
                # Vector feature of size ``dim``. A tensor holding exactly
                # one such vector (a ``(dim,)`` vector, or a plate of
                # ``dim`` scalar rows) is shared by every row.
                if tf.dim() <= 1 or (dim is not None and dim > 1 and tf.numel() == dim):
                    # Shared vector ``(dim,)`` (or a scalar broadcast to
                    # the vector width): one row.
                    shaped.append(tf.reshape(1, -1))
                else:
                    shaped.append(tf.reshape(tf.shape[0], -1))
            elif dim is not None:
                # A parameter of declared per-row width: its entries are
                # rows of that width, however many leading axes carry
                # them (a value drawn under the program's batch axis
                # arrives as ``(1, N)`` for ``N`` rows of one column).
                if tf.numel() % dim != 0:
                    raise RuntimeError(
                        f"_stack_params: a parameter of declared width {dim} "
                        f"holds {tf.numel()} entries, which are no whole "
                        f"number of rows of that width"
                    )
                shaped.append(tf.reshape(-1, dim))
            else:
                # Per-row scalar.
                if tf.dim() == 0:
                    shaped.append(tf.reshape(1, 1))
                elif tf.dim() == 1:
                    shaped.append(tf.unsqueeze(-1))
                else:
                    shaped.append(tf.reshape(tf.shape[0], -1))
        batch = max((t.shape[0] for t in shaped), default=1)
        broadcast: list[torch.Tensor] = []
        for t in shaped:
            if t.shape[0] == batch:
                broadcast.append(t)
            elif t.shape[0] == 1:
                broadcast.append(t.expand(batch, *t.shape[1:]))
            else:
                raise RuntimeError(
                    "_stack_params: incompatible batch sizes "
                    f"{[t.shape[0] for t in shaped]}; expected each to be "
                    f"1 (shared) or {batch} (per-row)"
                )
        return torch.cat(broadcast, dim=-1)

    @staticmethod
    def _promote_rank(t: torch.Tensor) -> torch.Tensor:
        """Promote a rank-1 continuous tensor to ``(batch, 1)``.

        Integer tensors (which feed ``nn.Embedding`` lookups in
        ``_LookupSource``) are returned unchanged so discrete program
        inputs continue to index correctly.

        Parameters
        ----------
        t : torch.Tensor
            Tensor to promote.

        Returns
        -------
        torch.Tensor
            ``t.unsqueeze(-1)`` if ``t`` is a 1-D floating-point tensor,
            otherwise ``t`` unchanged.
        """
        if t.dtype in (
            torch.long,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        ):
            return t
        if t.dim() == 1:
            return t.unsqueeze(-1)
        return t

    @staticmethod
    def _stack_tensors(parts: list[torch.Tensor]) -> torch.Tensor:
        """Stack tensors along feature dimension.

        Handles 1D (discrete, shape ``(batch,)``) and 2D (continuous,
        shape ``(batch, d)``) tensors by unsqueezing 1D tensors before
        concatenation, and broadcasts scalar / size-1 leading dims
        against the maximum batch size present in the input list.
        The size-1 broadcast covers the hierarchical-model case where
        a scalar latent (e.g. ``sigma``) is concatenated with a
        per-row tensor (e.g. ``mu``) ahead of an indexed observe.

        Parameters
        ----------
        parts : list[torch.Tensor]
            Tensors to stack.

        Returns
        -------
        torch.Tensor
            Concatenated tensor along dim=-1.
        """
        dtype = _parameter_dtype(parts)
        expanded = []
        for p in parts:
            if p.dim() == 1:
                expanded.append(p.unsqueeze(-1).to(dtype))
            else:
                expanded.append(p.to(dtype))

        batch = max(t.shape[0] for t in expanded)
        broadcast = []
        for t in expanded:
            if t.shape[0] == batch:
                broadcast.append(t)
            elif t.shape[0] == 1:
                broadcast.append(t.expand(batch, *t.shape[1:]))
            else:
                raise RuntimeError(
                    f"_stack_tensors: incompatible batch sizes "
                    f"{[t.shape[0] for t in expanded]}; expected each to be "
                    f"1 (scalar) or {batch} (per-row)"
                )

        return torch.cat(broadcast, dim=-1)

    def _bind_result(
        self,
        spec: Draw | Observe,
        result: torch.Tensor | dict[str, torch.Tensor],
        env: dict[str, torch.Tensor],
    ) -> None:
        """Bind the morphism result to variable(s) in the env.

        Parameters
        ----------
        spec : Draw | Observe
            The step.
        result : torch.Tensor or dict[str, torch.Tensor]
            The morphism output. A dict for tuple-returning
            sub-programs.
        env : dict[str, torch.Tensor]
            Variable environment (mutated in place).
        """
        if len(spec.names) == 1:
            # simple binding
            if isinstance(result, dict):
                # sub-program returned dict but we're binding to single var
                # — shouldn't happen if types are correct
                env[spec.names[0]] = result  # type: ignore[assignment]

            else:
                env[spec.names[0]] = result

        else:
            # destructuring: unpack dict from sub-program
            if isinstance(result, dict):
                for var_name in spec.names:
                    env[var_name] = result[var_name]

            else:
                # tensor result from product-codomain morphism: split
                # along feature dim
                morph = self._modules[step_key(spec)]
                assert morph is not None
                morph_cm = cast(ContinuousMorphism, morph)
                dims = self._compute_component_dims(morph_cm.codomain)
                splits = torch.split(result, dims, dim=-1)

                for var_name, chunk in zip(spec.names, splits):
                    env[var_name] = chunk.squeeze(-1) if chunk.shape[-1] == 1 else chunk

    @property
    def observed_names(self) -> set[str]:
        """The names the program's observations are supplied under.

        Returns
        -------
        set[str]
            The names of the program's ``observe`` steps, with those of
            the programs its [`Call`][quivers.continuous.Call] and
            [`Scan`][quivers.continuous.Scan] steps run in place.
        """
        names: set[str] = set()
        for spec in self._steps:
            if isinstance(spec, Observe):
                names.update(spec.names)
            elif isinstance(spec, Call):
                names |= _check_call(spec).observed_names
            elif isinstance(spec, Scan):
                names |= _check_scan(spec).observed_names
        return names

    def rsample(  # type: ignore[override]
        self,
        x: torch.Tensor,
        sample_shape: torch.Size = torch.Size(),
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run the program forward, returning the designated output(s).

        The program runs on the reference machine under the active
        handler stack, so a `clamp`, `do`, or `trace` in scope applies
        to the draws the run makes.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Its ``B`` rows are ``B`` independent runs of
            the program, whose outputs stack along a leading run axis
            (ahead of a plated value's rows).
        sample_shape : torch.Size
            Additional leading sample dimensions; each element is one
            independent run of the program.
        observations : dict[str, torch.Tensor] or None
            Values to clamp observed variables to, keyed by variable
            name, together with any host data the steps read by name.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            The value of the return variable(s). Returns a tensor
            for single-variable returns, or a dict keyed by variable
            name for tuple returns.
        """
        return _evaluator().sample_program(self, x, sample_shape, observations)

    def _apply_categorical_morphism(
        self,
        morph,
        inp: torch.Tensor,
        batch: int,
    ) -> torch.Tensor:
        """Apply a V-enriched `Morphism` to a batched input
        tensor as a deterministic step.

        The morphism's tensor has shape ``(*dom.shape, *cod.shape)``
        with values in the algebra's lattice. For a batched input
        ``inp`` of shape ``(batch, *dom.shape)``, the V-enriched
        action is the algebra tensor product followed by join over
        the domain axes — equivalent to a matrix-vector contraction
        when the algebra is product-fuzzy / Markov / boolean. We
        delegate that contraction to the active algebra's
        ``composition_kernel`` so a single deterministic-bind path
        handles every supported algebra uniformly.
        """
        m_tensor = morph.tensor
        # Broadcast inp to (batch, *dom.shape). For a one-hot input
        # of shape (batch,) into a finite domain, treat the values
        # as integer indices and gather the corresponding rows.
        dom_shape = tuple(morph.domain.shape)
        cod_shape = tuple(morph.codomain.shape)
        if inp.dim() == 1 or (inp.dim() == 2 and inp.shape[-1] == 1):
            idx = inp.reshape(-1).to(torch.long)
            if idx.numel() != batch:
                idx = idx[:batch] if idx.numel() > batch else idx.expand(batch)
            # m_tensor has shape (*dom_shape, *cod_shape). Index
            # selecting the first len(dom_shape) axes by `idx` gives
            # (batch, *cod_shape). For a single-axis domain this is
            # m_tensor[idx]; for multi-axis we treat each component
            # of idx separately (and require dom_shape to be
            # 1-axis for the simple gather path).
            if len(dom_shape) == 1:
                return m_tensor[idx]
            # Multi-axis: contract by einsum below.
            raise RuntimeError(
                "deterministic V-Cat step: integer-indexed input "
                f"only supported for 1-axis domains; got {dom_shape}"
            )
        # General contraction via einsum: input has shape
        # (batch, *dom.shape), m has shape (*dom.shape, *cod.shape).
        in_letters = "".join(chr(ord("a") + i) for i in range(len(dom_shape)))
        out_letters = "".join(
            chr(ord("a") + len(dom_shape) + j) for j in range(len(cod_shape))
        )
        eq = f"...{in_letters},{in_letters}{out_letters}->...{out_letters}"
        return torch.einsum(eq, inp, m_tensor)

    def has_conditional_density(self) -> bool:
        """No: a program's density at its output marginalizes its draws.

        [`log_prob`][quivers.continuous.programs.MonadicProgram.log_prob]
        raises for exactly this reason, and a caller choosing between
        constructions reads the answer here rather than by catching
        that raise.
        [`log_joint`][quivers.continuous.programs.MonadicProgram.log_joint]
        is the exact route: it scores the draws themselves, given
        values for all of them.
        """
        return False

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability is not supported for monadic programs.

        Computing log p(y | x) for a monadic program requires
        marginalizing over all intermediate variables, which is
        intractable in general. Use ``rsample`` for forward sampling
        and condition via score function estimators or variational
        methods.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "log_prob is not supported for monadic programs; "
            "computing p(y | x) requires marginalizing over all "
            "intermediate draws, which is intractable in general. "
            "use rsample() for forward sampling."
        )

    def log_joint(
        self,
        x: torch.Tensor,
        intermediates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Joint log-density given all intermediate values.

        When all intermediate variables are observed (e.g. during
        inference with HMC/NUTS), computes the joint log-density:

            log p(x_1, ..., x_n | input) = sum_i log p(x_i | pa(x_i))

        where pa(x_i) is the parent variable of step i (either the
        program input or a previously drawn variable). The program
        runs on the reference machine with every draw conditioned on
        its given value, under the active handler stack, so the joint
        is the one [`trace`][quivers.inference.trace.trace] reports.

        An input of ``B`` rows is ``B`` independent runs of the whole
        program, and the joint is one number per run; the joint of all
        the runs is the sum of the result. The rows the program
        observes come from its data and its plates, not from the input,
        so a program whose observations live in a plate is scored on a
        one-row input.

        Parameters
        ----------
        x : torch.Tensor
            Program input, one row per run.
        intermediates : dict[str, torch.Tensor]
            Values for ALL bound variables (keyed by variable name
            or by return label if labels are set), together with any
            host data the steps read by name. A value is given either
            once per run, along a leading axis of length ``B``, or once
            for every run to share.

        Returns
        -------
        torch.Tensor
            The joint log-density of each run, shape ``(B,)``.

        Raises
        ------
        KeyError
            If a draw the program makes is given no value.
        """
        return _evaluator().log_joint(self, x, intermediates)

    def __repr__(self) -> str:
        parts = []

        for s in self._steps:
            if isinstance(s, Score):
                parts.append(f"score {s.name} = {s.score}")
            elif isinstance(s, Call):
                parts.append(f"let {s.name} <- call({', '.join(s.args)})")
            elif isinstance(s, Scan):
                source = s.source if s.source is not None else "input"
                parts.append(f"let {s.name} = scan({source})")
            elif isinstance(s, Let):
                parts.append(f"let {s.name} = {s.value}")
            elif isinstance(s, Draw | Observe):
                names: tuple[str, ...] = s.names
                lhs = f"({','.join(names)})" if len(names) > 1 else names[0]
                rhs = step_key(s).removeprefix("_step_")
                if s.args:
                    rhs += f"({', '.join(_argument_text(a) for a in s.args)})"
                parts.append(f"{s.kind} {lhs} ~ {rhs}")

        steps = ", ".join(parts)
        if self._return_labels and not self._return_is_single:
            ret_parts = [
                f"{l}: {v}" for l, v in zip(self._return_labels, self._return_vars)
            ]
            ret = f"({', '.join(ret_parts)})"

        elif not self._return_is_single:
            ret = f"({', '.join(self._return_vars)})"

        else:
            ret = self._return_vars[0]
        params = f"({', '.join(self._params)})" if self._params else ""
        return (
            f"MonadicProgram{params}({self.domain!r} -> {self.codomain!r}, "
            f"[{steps}] -> {ret})"
        )


__all__ = [
    "MonadicProgram",
    "step_key",
]
