"""Evaluating a computation written for one position at many positions.

A log density is written for one position: every value in it has its
own axes and no other, and a shape, an element count, or a rank means
what it says. Evaluating it at ``B`` positions adds one axis to every
value that depends on the position. `map_positions` makes that axis part
of the value's type rather than of its shape: the computation runs once,
under [`torch.func.vmap`][torch.func.vmap], on values that carry the
position axis at a declared mapping level, where no shape, element
count, or index of the computation can see it. Two axes of equal length
are therefore never confused, a value that does not depend on the
position carries no axis and is computed once, and the computation's own
layout rules hold unchanged.

Three things a computation may do have no meaning for ``B`` positions at
once, and each is stated rather than inferred.

Refusing a position. A check that raises when its operand is outside a
domain refuses the whole call. `reject_unless` states such a check for
the position: over many positions it marks the positions that fail and
lets the others continue. `demand` states a check whose failure is a
defect of the program or its data rather than of the position, so it
raises over many positions as over one. The validators of
`torch.distributions` and the factorizations of `torch.linalg` that
raise are read the same way by `_Refusals`.

Branching on values. An iteration that stops when every entry has
converged, or a branch that skips work no entry needs, asks `everywhere`
or `anywhere` about the array of all positions, and `largest` and
`entry_count` size such a computation. `branch` takes one of two
computations by a scalar of the position. `elementwise` declares a routine
that maps its arguments entry by entry, selecting entries by value: it
is called with the position axis as an explicit leading axis.

Reading the axis itself. `lift` calls a function written for an explicit
leading axis, a step's callable declared ``batched`` for instance, with
that axis leading every value that has one.

Drawing. The mapping draws independently at every position, but a draw
that fills a tensor in place, as torch's distributions draw, cannot. `draw`
calls a drawing function beneath the mapping level, with the position
axis leading every tensor, so its in-place draws fill one entry per
position.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from types import CodeType, FrameType
from typing import ParamSpec, Protocol, TypeVar, cast

import didactic.api as dx
import torch
from torch._C._functorch import (
    _add_batch_dim,
    _unwrap_batched,
    current_level,
    is_batchedtensor,
    maybe_get_level,
)
from torch._functorch.pyfunctorch import temporarily_pop_interpreter_stack
from torch.autograd.function import FunctionCtx
from torch.distributions import constraints as _constraints
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.geometric import Geometric
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.wishart import Wishart
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel

P = ParamSpec("P")
Result = TypeVar("Result")
Reduced = TypeVar("Reduced", bool, float)
Values = TypeVar("Values")
Positions = TypeVar("Positions")

type Leaf = torch.Tensor | torch.Size | torch.dtype | torch.device | str | float | None
"""A leaf of an operation's arguments or of its result."""

type Tree = Leaf | tuple[Tree, ...] | list[Tree] | dict[str, Tree]
"""An operation's arguments or its result: nested containers of leaves."""

_ROW_MARK = "_quivers_leads_with_positions"
"""The attribute marking a tensor whose leading axis is the position axis."""


class PositionLayoutError(ValueError):
    """A value of a batched callable does not lead with the position axis.

    A callable declared ``batched`` keeps the position axis leading every
    value that depends on the position. A result that does not, or a
    validator that refuses a value whose leading axis is not the position
    axis, cannot be read per position.
    """


class UnreadRefusalError(RuntimeError):
    """A refusal torch makes of a whole call that no position accounts for.

    A position is refused alone. When a check inside torch refuses values
    of many positions without saying which, or reduces them to one answer,
    reading it as a refusal of every position would let one position decide
    for the others. The evaluation stops instead.

    Parameters
    ----------
    subject : str
        What refused.
    """

    def __init__(self, subject: str) -> None:
        super().__init__(
            f"{subject}, and which positions it refuses cannot be read under torch "
            f"{torch.__version__}"
        )


class PositionValues(ValueModel):
    """The values of a computation at many positions.

    Attributes
    ----------
    values : torch.Tensor
        The computation's value at each position, the position axis
        leading.
    accepted : torch.Tensor
        For each position, whether every check the computation makes of
        it holds. The value at a position that is not accepted has no
        meaning.
    """

    values: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    accepted: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


class _Check(ValueModel):
    """A check a computation made of its positions.

    Attributes
    ----------
    valid : torch.Tensor
        Whether the check holds, one value per position at the frame's
        mapping level.
    defect : Exception | None
        The error a failure raises, or ``None`` when a failure refuses
        the position.
    """

    valid: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    defect: Exception | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


class _Frame:
    """The checks of one evaluation at many positions.

    Parameters
    ----------
    count : int
        The number of positions.
    """

    __slots__ = ("count", "level", "checks", "explicit", "refused", "shared")

    def __init__(self, count: int) -> None:
        self.count = count
        # The mapping level of the evaluation, set once it has started.
        self.level = 0
        self.checks: list[_Check] = []
        # The checks of each enclosing call of `lift`, whose values lead
        # with the position axis explicitly.
        self.explicit: list[list[_Check]] = []
        # Whether a position has failed a check so far, at the mapping
        # level.
        self.refused: torch.Tensor | None = None
        # Whether a check of a value every position shares has failed, which
        # refuses them all.
        self.shared = False

    def add(self, check: _Check) -> None:
        """Add a check at the mapping level.

        Parameters
        ----------
        check : _Check
            The check.
        """
        self.checks.append(check)
        failed = ~check.valid
        self.refused = failed if self.refused is None else self.refused | failed

    def record(self, valid: torch.Tensor, defect: Exception | None) -> None:
        """Record a check of values that depend on the position.

        Parameters
        ----------
        valid : torch.Tensor
            Whether each entry passes.
        defect : Exception | None
            The error a failure raises, or ``None`` when it refuses the
            position.

        Raises
        ------
        PositionLayoutError
            If the check is made inside `lift` and ``valid`` does not
            lead with the position axis.
        """
        if not self.explicit:
            self.add(_Check(valid=valid.all(), defect=defect))
            return
        if valid.dim() == 0 or int(valid.shape[0]) != self.count:
            raise PositionLayoutError(
                f"a check inside a batched callable reads values of shape "
                f"{tuple(valid.shape)}, which do not lead with the "
                f"{self.count} positions"
            )
        self.explicit[-1].append(
            _Check(valid=valid.reshape(self.count, -1).all(-1), defect=defect)
        )


_FRAME: ContextVar[_Frame | None] = ContextVar("quivers_position_frame", default=None)


def _mapped(value: torch.Tensor, frame: _Frame) -> bool:
    """Whether a tensor carries the position axis at the frame's level.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.
    frame : _Frame
        The evaluation.

    Returns
    -------
    bool
        ``True`` for a value that depends on the position.
    """
    return is_batchedtensor(value) and maybe_get_level(value) == frame.level


def _positional(value: torch.Tensor, frame: _Frame) -> bool:
    """Whether a tensor depends on the position, wherever it is read.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.
    frame : _Frame
        The evaluation.

    Returns
    -------
    bool
        ``True`` for a value at the frame's mapping level, or one derived
        from an explicit leading position axis inside `lift`.
    """
    if frame.explicit:
        return bool(getattr(value, _ROW_MARK, False))
    return _mapped(value, frame)


def _unread() -> PositionLayoutError:
    """The error of a refusal, inside `lift`, no position can be read from.

    Returns
    -------
    PositionLayoutError
        The error.
    """
    return PositionLayoutError(
        "a check inside a batched callable refuses a value that does not lead with "
        "the positions, so which positions it refuses cannot be read; return -inf at "
        "the positions outside the support"
    )


def reject_unless(valid: torch.Tensor, error: Callable[[], Exception]) -> None:
    """Refuse a position unless a check of it holds.

    Parameters
    ----------
    valid : torch.Tensor
        Whether each entry is inside the domain.
    error : Callable[[], Exception]
        The error an evaluation at one position raises. A sampler reads
        it as a position of zero density.

    Raises
    ------
    Exception
        ``error()``, when an entry fails and ``valid`` does not depend
        on the positions of an evaluation at many, where the positions
        that fail are marked instead.
    """
    frame = _FRAME.get()
    if frame is not None and _positional(valid, frame):
        frame.record(valid, None)
        return
    if not bool(valid.all()):
        if frame is not None:
            if frame.explicit:
                raise _unread()
            frame.shared = True
        raise error()


def rejects_position(error: BaseException) -> bool:
    """Whether an error raised while scoring a position means the
    position lies outside the model's support.

    A ``ValueError`` raised by a Torch distribution validator marks a
    constraint violation, as does whatever `reject_unless` raises: the
    inline parameter validators, the truncation-mass and
    finite-parameter checks of recorded events, the COM-Poisson
    modal-capacity and normalizer-budget refusals, which brms's Stan code
    also rejects (``nu is too close to zero``, ``cannot handle y >
    10000``), and the incomplete Beta continued fraction exhausting its
    budget at extreme count shapes.
    The code object of the innermost traceback frame identifies the
    validator, independently of the error message.
    A `torch.linalg.LinAlgError` is a factorization of a matrix that
    the position makes numerically non-positive-definite. Every other
    error is a defect of the program or its data.

    Parameters
    ----------
    error : BaseException
        The error raised by ``log_joint``.

    Returns
    -------
    bool
        ``True`` when the position should be rejected with a
        log-density of ``-inf``; ``False`` when the error must
        propagate.
    """
    if isinstance(error, PositionLayoutError | UnreadRefusalError):
        return False
    if isinstance(error, torch.linalg.LinAlgError):
        return True
    if not isinstance(error, ValueError | RuntimeError):
        return False
    frame = error.__traceback__
    if frame is None:
        return False
    while frame.tb_next is not None:
        frame = frame.tb_next
    return frame.tb_frame.f_code in _REFUSING_CODE


_REFUSING_CODE = frozenset(
    {
        Distribution.__init__.__code__,
        Distribution._validate_sample.__code__,
        reject_unless.__code__,
    }
)
"""The raising frames that refuse a position."""


def demand(valid: torch.Tensor, error: Callable[[], Exception]) -> None:
    """Raise unless a check the program must pass at every position holds.

    Parameters
    ----------
    valid : torch.Tensor
        Whether each entry passes.
    error : Callable[[], Exception]
        The error a failure raises.

    Raises
    ------
    Exception
        ``error()``, when an entry fails. Over many positions it is
        raised at the end of the evaluation, unless an earlier check
        refused every position that fails.
    """
    frame = _FRAME.get()
    if frame is not None and _positional(valid, frame):
        frame.record(valid, error())
        return
    if not bool(valid.all()):
        raise error()


def depends_on_position(value: torch.Tensor) -> bool:
    """Whether a value depends on the positions of an evaluation at many.

    Parameters
    ----------
    value : torch.Tensor
        The value.

    Returns
    -------
    bool
        ``False`` outside `map_positions`, and for a value every position
        shares.
    """
    frame = _FRAME.get()
    return frame is not None and _positional(value, frame)


def evaluating() -> bool:
    """Whether an evaluation at many positions is under way.

    Returns
    -------
    bool
        ``True`` inside `map_positions`, whose values may carry a
        position axis.
    """
    return _FRAME.get() is not None


class _Mapping(Protocol):
    """What `torch.func.vmap` tells a function's own rule of the mapping."""

    @property
    def batch_size(self) -> int:
        """The number of positions.

        Returns
        -------
        int
            The number.
        """
        ...


class _Lifted(torch.autograd.Function):
    """Call a function with the position axis leading its tensors.

    The vectorization rule is the function itself on the tensors beneath
    the mapping level, so its operations are recorded for
    differentiation as they run.
    """

    generate_vmap_rule = False

    @staticmethod
    def forward(
        function: Callable[..., tuple[torch.Tensor, ...]],
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Refuse a call with no position axis.

        Parameters
        ----------
        function : Callable[..., tuple[torch.Tensor, ...]]
            The function.
        *tensors : torch.Tensor
            Its tensors.

        Raises
        ------
        RuntimeError
            Always: `lift` applies the function only to tensors at a
            mapping level.
        """
        del function, tensors
        raise RuntimeError("a lifted call needs a tensor with a position axis")

    @staticmethod
    def setup_context(
        ctx: FunctionCtx,
        inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> None:
        """Save nothing for the backward pass.

        Parameters
        ----------
        ctx : FunctionCtx
            The context.
        inputs : tuple[torch.Tensor, ...]
            The inputs.
        output : tuple[torch.Tensor, ...]
            The outputs.
        """

    @staticmethod
    def vmap(
        info: _Mapping,
        in_dims: tuple[int | None, ...],
        *args: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        """Call the function with explicit position axes.

        Parameters
        ----------
        info : _Mapping
            The mapping, unused.
        in_dims : tuple[int | None, ...]
            The position axis of each argument.
        *args : torch.Tensor
            The function, then its tensors beneath the mapping level.

        Returns
        -------
        tuple[tuple[torch.Tensor, ...], tuple[int, ...]]
            The results, each leading with the position axis.
        """
        del info
        function = cast("Callable[..., tuple[torch.Tensor, ...]]", args[0])
        tensors = args[1:]
        axes = in_dims[1:]
        leading = [
            tensor if axis is None else tensor.movedim(axis, 0)
            for tensor, axis in zip(tensors, axes, strict=True)
        ]
        results = function(*leading, carried=tuple(axis is not None for axis in axes))
        return results, (0,) * len(results)


def _standing_in(value: torch.Tensor, refused: torch.Tensor) -> torch.Tensor:
    """A value whose refused positions stand in for an accepted one.

    What a refused position computes has no meaning, and is never read. Its
    entries are those of a position still accepted, so that a routine over
    the array of all positions meets no entry it would raise at or iterate
    over on account of a refused one.

    Parameters
    ----------
    value : torch.Tensor
        The value, the position axis leading.
    refused : torch.Tensor
        Whether each position is refused.

    Returns
    -------
    torch.Tensor
        The value, unchanged at the positions accepted, and at every
        position when none is.
    """
    accepted = (~refused).nonzero()
    if not accepted.numel() or not bool(refused.any()):
        return value.view_as(value)
    own = value[accepted[0, 0]].detach()
    return torch.where(
        refused.reshape(-1, *((1,) * (value.dim() - 1))), own.unsqueeze(0), value
    )


def lift(
    function: Callable[P, Result], subject: str, *args: P.args, **kwargs: P.kwargs
) -> Result:
    """Call a function with the position axis leading its tensors.

    Outside `map_positions`, or when no tensor among the arguments
    depends on the position, the function is called as it is.

    Parameters
    ----------
    function : Callable[P, Result]
        The function. It receives every tensor that depends on the
        position with the position axis leading, and every other value
        as it is, and returns tensors that lead with the position axis.
    subject : str
        What the function is, for the diagnostic.
    *args
        Its positional arguments: tensors, and nested tuples, lists, and
        dictionaries of them, among other values.
    **kwargs
        Its keyword arguments.

    Returns
    -------
    Result
        The results, with the position axis at the mapping level again.

    Raises
    ------
    PositionLayoutError
        If a result is not a tensor that leads with the position axis.
    """
    frame = _FRAME.get()
    leaves, structure = pytree.tree_flatten((args, kwargs))
    slots = [
        index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)
    ]
    if frame is None or not any(_mapped(leaves[index], frame) for index in slots):
        return function(*args, **kwargs)
    returned: list[pytree.TreeSpec] = []
    recorded: list[_Check] = []
    refused = frame.refused

    def explicit(
        *tensors: torch.Tensor, carried: tuple[bool, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Run the function on explicit axes and gather its checks.

        Parameters
        ----------
        *tensors : torch.Tensor
            The tensors, the position axis leading those that carry it.
        carried : tuple[bool, ...]
            Which tensors carry it.

        Returns
        -------
        tuple[torch.Tensor, ...]
            The function's results, then the checks it made.

        Raises
        ------
        PositionLayoutError
            If a result is not a tensor that leads with the position
            axis.
        """
        filled = list(leaves)
        rows = None if refused is None else tensors[0]
        count = len(slots)
        for index, tensor, positional in zip(
            slots, tensors[-count:], carried[-count:], strict=True
        ):
            if positional:
                # A value of its own, so that marking it marks no other
                # tensor.
                tensor = (
                    tensor.view_as(tensor)
                    if rows is None
                    else _standing_in(tensor, rows)
                )
                setattr(tensor, _ROW_MARK, True)
            filled[index] = tensor
        frame.explicit.append(recorded)
        try:
            # The call itself is an operation the enclosing mode handles,
            # with the mode off, so the function's operations install it
            # again.
            with _Refusals(frame):
                own_args, own_kwargs = pytree.tree_unflatten(filled, structure)
                call = cast("Callable[..., Result]", function)
                results, spec = pytree.tree_flatten(call(*own_args, **own_kwargs))
        finally:
            frame.explicit.pop()
        returned.append(spec)
        for result in results:
            if not isinstance(result, torch.Tensor):
                raise PositionLayoutError(
                    f"{subject} returns a {type(result).__name__}, not a tensor, "
                    f"for {frame.count} positions"
                )
            if result.dim() == 0 or int(result.shape[0]) != frame.count:
                raise PositionLayoutError(
                    f"{subject} returns a value of shape {tuple(result.shape)} for "
                    f"{frame.count} positions; a batched callable returns one value "
                    f"per position along a leading axis of length {frame.count}"
                )
        return (*results, *(check.valid for check in recorded))

    outputs = cast(
        "tuple[torch.Tensor, ...]",
        _Lifted.apply(
            explicit,
            *(() if refused is None else (refused,)),
            *(leaves[index] for index in slots),
        ),
    )
    (spec,) = returned
    count = spec.num_leaves
    for check, valid in zip(recorded, outputs[count:], strict=True):
        frame.add(_Check(valid=valid, defect=check.defect))
    return cast("Result", pytree.tree_unflatten(list(outputs[:count]), spec))


def draw(
    function: Callable[[tuple[torch.Tensor, ...], int | None], torch.Tensor],
    tensors: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Draw independently at every position.

    Outside `map_positions` the function is called as it is, with no
    position count. Inside it, the function runs beneath the mapping
    level: every tensor is given with the position axis leading, a value
    every position shares expanded along it, and the function returns a
    draw that leads with the position axis, which is mapped again. Every
    position thus draws its own value however the function draws, in place
    or not, and the draw keeps the gradient of every tensor.

    Parameters
    ----------
    function : Callable[[tuple[torch.Tensor, ...], int | None], torch.Tensor]
        The drawing function: it receives the tensors and the number of
        positions, ``None`` outside an evaluation at many positions, and
        returns a draw that leads with the position axis when there is one.
    tensors : tuple[torch.Tensor, ...]
        Its tensors, each of the shape it has at one position.

    Returns
    -------
    torch.Tensor
        The draw at the position.

    Raises
    ------
    PositionLayoutError
        If the draw does not lead with the position axis.
    """
    frame = _FRAME.get()
    if frame is None or current_level() != frame.level:
        return function(tensors, None)
    level = frame.level
    leading: list[torch.Tensor] = []
    for tensor in tensors:
        value, axis = _unwrap_batched(tensor, level)
        leading.append(
            value.expand(frame.count, *value.shape)
            if axis is None
            else value.movedim(axis, 0)
        )
    with temporarily_pop_interpreter_stack():
        drawn = function(tuple(leading), frame.count)
    if drawn.dim() == 0 or int(drawn.shape[0]) != frame.count:
        raise PositionLayoutError(
            f"a draw at {frame.count} positions has shape {tuple(drawn.shape)}"
        )
    return _add_batch_dim(drawn, 0, level)


def branch(
    condition: torch.Tensor,
    when_true: Callable[[Values], Result],
    when_false: Callable[[Values], Result],
    values: Values,
) -> Result:
    """Take one of two computations by a scalar condition of the position.

    Only the branch taken is computed. Over many positions each branch is
    computed at the positions that take it and at no other, so a branch
    makes no check of, and lends no derivative to, a position that does
    not take it.

    Parameters
    ----------
    condition : torch.Tensor
        A Boolean of one element.
    when_true : Callable
        The computation of ``values`` where the condition holds.
    when_false : Callable
        The computation where it does not.
    values : Values
        What the branches read: tensors, and nested tuples, lists, and
        dictionaries of them, among other values.

    Returns
    -------
    Result
        The value of the branch taken.

    Raises
    ------
    PositionLayoutError
        If the branches taken by different positions return values of
        different structures or shapes, which no one array holds.
    """
    frame = _FRAME.get()
    if frame is None or frame.explicit or not _mapped(condition, frame):
        return when_true(values) if bool(condition) else when_false(values)
    leaves, structure = pytree.tree_flatten(values)
    slots = [
        index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)
    ]
    carried = [_mapped(leaves[index], frame) for index in slots]

    def select(chosen: torch.Tensor, *tensors: torch.Tensor) -> list[torch.Tensor]:
        """Compute each branch at the positions that take it.

        Parameters
        ----------
        chosen : torch.Tensor
            The condition at each position.
        *tensors : torch.Tensor
            The tensors among the values, with the position axis explicit.

        Returns
        -------
        list[torch.Tensor]
            The leaves of the value, with the position axis explicit.

        Raises
        ------
        PositionLayoutError
            If the branches return values no one array holds.
        """
        taken = chosen.reshape(-1)
        parts: list[list[torch.Tensor]] = []
        rows: list[torch.Tensor] = []
        refusals: list[tuple[torch.Tensor, Exception]] = []
        for function, mask in ((when_true, taken), (when_false, ~taken)):
            index = mask.nonzero().squeeze(-1)
            if not index.numel():
                continue

            def at(
                *local: torch.Tensor,
                function: Callable[[Values], Result] = function,
            ) -> list[torch.Tensor]:
                """The branch at one position.

                Parameters
                ----------
                *local : torch.Tensor
                    The position's tensors.
                function : Callable
                    The branch.

                Returns
                -------
                list[torch.Tensor]
                    The leaves of its value.
                """
                filled = list(leaves)
                for slot, tensor in zip(slots, local, strict=True):
                    filled[slot] = tensor
                result, spec = pytree.tree_flatten(
                    function(pytree.tree_unflatten(filled, structure))
                )
                specs.append(spec)
                return [torch.as_tensor(leaf) for leaf in result]

            local = [
                tensor.index_select(0, index) if positional else tensor
                for tensor, positional in zip(tensors, carried, strict=True)
            ]
            in_dims = [0 if positional else None for positional in carried]
            part = _run_mapped(at, local, in_dims, index, frame)
            if isinstance(part, Exception):
                refusals.append((index, part))
                continue
            parts.append(part)
            rows.append(index)
        if not parts:
            # Every position takes a branch that refuses a value they share.
            frame.shared = True
            raise refusals[0][1]
        for index, _ in refusals:
            count = int(index.numel())
            parts.append(
                [leaf.new_zeros((count, *leaf.shape[1:])) for leaf in parts[0]]
            )
            rows.append(index)
        if (
            any(spec != specs[0] for spec in specs)
            or len({tuple(tuple(leaf.shape[1:]) for leaf in part) for part in parts})
            > 1
        ):
            raise PositionLayoutError(
                "the branches of a conditional return values of different shapes at "
                "different positions"
            )
        order = torch.cat(rows).argsort()
        return [
            torch.cat(list(column)).index_select(0, order)
            for column in zip(*parts, strict=True)
        ]

    specs: list[pytree.TreeSpec] = []
    results = lift(
        select, "a conditional", condition, *(leaves[index] for index in slots)
    )
    return cast("Result", pytree.tree_unflatten(results, specs[0]))


def _account(frame: _Frame, error: Exception) -> None:
    """Refuse to read a refusal of every position that none accounts for.

    Parameters
    ----------
    frame : _Frame
        The evaluation.
    error : Exception
        The error that ended it.

    Raises
    ------
    UnreadRefusalError
        If the error refuses a position, and no check of a value the
        positions share has failed.
    """
    if rejects_position(error) and not frame.shared:
        raise UnreadRefusalError(
            f"a check raises for every position: {error}"
        ) from error


def _run_mapped(
    function: Callable[..., list[torch.Tensor]],
    tensors: list[torch.Tensor],
    in_dims: list[int | None],
    index: torch.Tensor,
    outer: _Frame,
) -> list[torch.Tensor] | Exception:
    """Run a function at a subset of the positions of an evaluation.

    Parameters
    ----------
    function : Callable[..., list[torch.Tensor]]
        The function of one position's tensors.
    tensors : list[torch.Tensor]
        The tensors, the subset leading those that depend on the position.
    in_dims : list[int | None]
        The position axis of each.
    index : torch.Tensor
        The positions of the subset.
    outer : _Frame
        The evaluation, inside a call of `lift`, whose checks the
        subset's join.

    Returns
    -------
    list[torch.Tensor] | Exception
        The function's results, the subset leading each; or the refusal of
        a value the subset shares, which refuses the subset and no other
        position.
    """
    inner = _Frame(int(index.numel()))

    def at(*local: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The function and the checks it makes, at one position.

        Parameters
        ----------
        *local : torch.Tensor
            The position's tensors.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor]]
            The results, and each check's outcome.
        """
        inner.level = current_level()
        return function(*local), [check.valid for check in inner.checks]

    token = _FRAME.set(inner)
    try:
        with _Refusals(inner):
            results, outcomes = torch.func.vmap(
                at, in_dims=tuple(in_dims), randomness="different"
            )(*tensors)
    except (ValueError, RuntimeError) as error:
        _account(inner, error)
        if not rejects_position(error):
            raise
        outer.explicit[-1].append(
            _Check(
                valid=torch.ones(outer.count, dtype=torch.bool)
                .index_fill(0, index.cpu(), False)
                .to(index.device)
            )
        )
        return error
    finally:
        _FRAME.reset(token)
    for check, valid in zip(inner.checks, outcomes, strict=True):
        everyone = torch.ones(outer.count, dtype=torch.bool, device=valid.device)
        outer.explicit[-1].append(
            _Check(valid=everyone.index_copy(0, index, valid), defect=check.defect)
        )
    return results


def elementwise(function: Callable[P, Result]) -> Callable[P, Result]:
    """Declare a routine that maps the entries of its arguments.

    Every tensor argument broadcasts against the others, and each entry
    of a result is a function of the corresponding entries alone, so the
    routine may branch on its arguments as arrays. Over many positions it
    is called once with the position axis leading, ahead of the axes the
    arguments have at one position.

    Parameters
    ----------
    function : Callable
        The routine, returning a tensor or a tuple of tensors of the
        broadcast shape.

    Returns
    -------
    Callable
        The routine, callable at one position or at many.
    """

    @functools.wraps(function)
    def mapped(*args: P.args, **kwargs: P.kwargs) -> Result:
        """Call the routine, aligning the position axis ahead of the others.

        Parameters
        ----------
        *args
            The routine's positional arguments.
        **kwargs
            Its keyword arguments.

        Returns
        -------
        Result
            The routine's result.
        """
        frame = _FRAME.get()
        tensors = [
            leaf
            for leaf in pytree.tree_leaves((args, kwargs))
            if isinstance(leaf, torch.Tensor)
        ]
        if frame is None or not any(_mapped(leaf, frame) for leaf in tensors):
            return function(*args, **kwargs)
        rank = max(leaf.dim() for leaf in tensors)

        def aligned(leaf: Leaf) -> Leaf:
            """Pad a value so that the position axis leads every other.

            Parameters
            ----------
            leaf : Leaf
                A tensor or another leaf of the arguments.

            Returns
            -------
            Leaf
                A positional tensor at the arguments' common rank;
                another leaf unchanged.
            """
            if isinstance(leaf, torch.Tensor) and _mapped(leaf, frame):
                return leaf.reshape((1,) * (rank - leaf.dim()) + tuple(leaf.shape))
            return leaf

        padded_args, padded_kwargs = pytree.tree_map(aligned, (args, kwargs))
        name = getattr(function, "__name__", "an elementwise routine")
        return lift(function, name, *padded_args, **padded_kwargs)

    return mapped


def _over_positions(
    value: torch.Tensor, reduce: Callable[[torch.Tensor], Reduced]
) -> Reduced:
    """Reduce a value over every axis, the positions' included.

    Parameters
    ----------
    value : torch.Tensor
        The value.
    reduce : Callable[[torch.Tensor], Reduced]
        The reduction of the value with the position axis explicit.

    Returns
    -------
    Reduced
        What the reduction returns, which no position has of its own.
    """
    found: list[Reduced] = []

    def explicit(entries: torch.Tensor) -> torch.Tensor:
        """Reduce the value and pass it on.

        Parameters
        ----------
        entries : torch.Tensor
            The value, with the position axis explicit.

        Returns
        -------
        torch.Tensor
            The value.
        """
        found.append(reduce(entries))
        return entries

    lift(explicit, "a reduction over the positions", value)
    return found[0]


def everywhere(condition: torch.Tensor) -> bool:
    """Whether a condition holds at every entry and every position.

    An iteration that stops once each entry has converged, and a branch
    that only skips work no entry needs, is taken for the array as a
    whole: the result at an entry does not depend on which way it goes.
    Over many positions the array holds every position that has passed
    the checks so far, since the entries of the others have no meaning.

    Parameters
    ----------
    condition : torch.Tensor
        The condition at each entry.

    Returns
    -------
    bool
        Whether it holds throughout.
    """
    frame = _FRAME.get()
    held = condition.all()
    if frame is not None and not frame.explicit and frame.refused is not None:
        held = held | frame.refused
    return _over_positions(held, lambda entries: bool(entries.all()))


def anywhere(condition: torch.Tensor) -> bool:
    """Whether a condition holds at some entry of some position.

    Parameters
    ----------
    condition : torch.Tensor
        The condition at each entry, read as `everywhere` reads one.

    Returns
    -------
    bool
        Whether it holds at any.
    """
    return not everywhere(~condition)


def largest(value: torch.Tensor, default: float) -> float:
    """The largest entry of a value over every position.

    It sizes a computation, the number of terms of a series for instance,
    that a larger size leaves unchanged.

    Parameters
    ----------
    value : torch.Tensor
        The value.
    default : float
        The size when the value has no entry, or no position has passed
        the checks so far.

    Returns
    -------
    float
        Its largest entry at any position that has passed the checks so
        far.
    """
    if not value.numel():
        return default
    frame = _FRAME.get()
    top = value.detach().amax()
    if frame is not None and not frame.explicit and frame.refused is not None:
        top = torch.where(frame.refused, -torch.inf, top)
    found = _over_positions(top, lambda entries: float(entries.max()))
    return default if found == float("-inf") else found


def entry_count(value: torch.Tensor) -> int:
    """The number of entries a value has over every position.

    Parameters
    ----------
    value : torch.Tensor
        The value.

    Returns
    -------
    int
        Its entries at one position, times the number of positions when
        it depends on the position.
    """
    frame = _FRAME.get()
    if frame is not None and not frame.explicit and _mapped(value, frame):
        return value.numel() * frame.count
    return value.numel()


def _wishart_degrees(frame: FrameType) -> torch.Tensor:
    """Whether a Wishart under construction has enough degrees of freedom.

    Parameters
    ----------
    frame : FrameType
        The frame of `Wishart.__init__`.

    Returns
    -------
    torch.Tensor
        Whether each exceeds the matrix size less one. The constructor
        also asks whether any is below the size itself, to warn.
    """
    return frame.f_locals["self"].df > frame.f_locals["event_shape"][-1] - 1


class _Verdict(ValueModel):
    """How a validator's question about many positions is answered.

    Attributes
    ----------
    accepts : bool
        The truth value under which the validator accepts its operand.
    valid : Callable[[FrameType], torch.Tensor] | None
        Reads the check from the validator's frame, when its operand is
        not the check itself.
    defect : str | None
        The message of the error the validator raises when a refusal is a
        defect of the program, or ``None`` when it refuses the position.
    refuses : bool
        Whether the validator raises when it does not accept.
    """

    accepts: bool
    valid: Callable[[FrameType], torch.Tensor] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    defect: str | None = None
    refuses: bool = False


_VALIDATORS: Mapping[CodeType, _Verdict] = {
    Distribution.__init__.__code__: _Verdict(accepts=True, refuses=True),
    Distribution._validate_sample.__code__: _Verdict(accepts=True, refuses=True),
    type(_constraints.positive_semidefinite).check.__code__: _Verdict(accepts=True),
    type(_constraints.positive_definite).check.__code__: _Verdict(accepts=True),
    Geometric.__init__.__code__: _Verdict(
        accepts=True, defect="Geometric: expected parameter probs to be positive"
    ),
    ContinuousBernoulli.__init__.__code__: _Verdict(
        accepts=True, defect="The parameter probs has invalid values"
    ),
    Wishart.__init__.__code__: _Verdict(
        accepts=False,
        valid=_wishart_degrees,
        defect="Wishart: expected df to be greater than the matrix size less one",
    ),
}
"""The validators that branch on the values they are given.

A validator asks one question, `torch._is_all_true` or the truth of a
tensor, from its own frame or from a helper of `torch.distributions` it
calls, and goes on when the answer is the verdict's. `_check_contract`
holds torch to this.
"""

_MERGING: frozenset[Callable[..., torch.Tensor]] = frozenset(
    {torch._is_all_true, torch._is_any_true}
)
"""The reductions whose mapping rule answers once for every position."""


def _verdict_of(caller: FrameType | None) -> tuple[_Verdict, FrameType] | None:
    """The verdict of the validator a question comes from.

    Parameters
    ----------
    caller : FrameType | None
        The frame that asks.

    Returns
    -------
    tuple[_Verdict, FrameType] | None
        The verdict and the validator's frame: the nearest validator among
        the frames of `torch.distributions` that led to the question.
    """
    while caller is not None and str(caller.f_globals.get("__name__", "")).startswith(
        "torch.distributions"
    ):
        verdict = _VALIDATORS.get(caller.f_code)
        if verdict is not None:
            return verdict, caller
        caller = caller.f_back
    return None


type _Operation = Callable[..., tuple[torch.Tensor, torch.Tensor]]
"""A factorization returning its result and an error code per matrix."""

_FACTORIZATIONS: Mapping[Callable[..., torch.Tensor], _Operation] = {
    torch.linalg.cholesky: torch.linalg.cholesky_ex,
    torch.linalg.inv: torch.linalg.inv_ex,
    torch.inverse: torch.linalg.inv_ex,
    torch.Tensor.inverse: torch.linalg.inv_ex,
    torch.linalg.solve: torch.linalg.solve_ex,
}
"""The factorizations that raise, and the form that reports instead."""


_DECOMPOSITIONS: frozenset[Callable[..., Tree]] = frozenset(
    {
        torch.linalg.eigh,
        torch.linalg.eigvalsh,
        torch.linalg.eig,
        torch.linalg.eigvals,
        torch.linalg.svd,
        torch.linalg.svdvals,
    }
)
"""The decompositions whose backend may raise for every matrix when one is
not finite. Such a matrix has a decomposition of NaN."""

_ENTRYWISE: frozenset[Callable[..., torch.Tensor]] = frozenset(
    {torch.special.log_ndtr, torch.isclose}
)
"""The entrywise operations `torch.func.vmap` has no rule for."""


def _tensors(values: Tree) -> list[torch.Tensor]:
    """The tensors among nested values.

    Parameters
    ----------
    values : Tree
        The values.

    Returns
    -------
    list[torch.Tensor]
        The tensors.
    """
    return [leaf for leaf in pytree.tree_leaves(values) if torch.is_tensor(leaf)]


class _Refusals(TorchFunctionMode):
    """Read what raises at one position as a check of many.

    Parameters
    ----------
    frame : _Frame
        The evaluation.
    """

    def __init__(self, frame: _Frame) -> None:
        super().__init__()
        self._frame = frame

    def __torch_function__(
        self,
        func: Callable[..., Tree],
        types: Sequence[type],
        args: tuple[Tree, ...] = (),
        kwargs: dict[str, Tree] | None = None,
    ) -> Tree:
        """Answer a validator or factorization for many positions.

        Parameters
        ----------
        func : Callable
            The operation.
        types : Sequence[type]
            The types overriding it, unused.
        args : tuple[Tree, ...]
            Its positional arguments.
        kwargs : dict[str, Tree] | None
            Its keyword arguments.

        Returns
        -------
        Tree
            The operation's result.
        """
        del types
        frame = self._frame
        named = kwargs or {}
        if func in _MERGING or func is torch.Tensor.__bool__:
            operand = cast("torch.Tensor", args[0])
            found = _verdict_of(sys._getframe(1))
            positional = _positional(operand, frame)
            # A validator asking the truth of a tensor has reduced it over
            # every axis, so inside `lift` its refusal names no position.
            reduced = func is torch.Tensor.__bool__ and bool(frame.explicit)
            if found is not None and positional and not reduced:
                verdict, validator = found
                frame.record(
                    operand if verdict.valid is None else verdict.valid(validator),
                    None if verdict.defect is None else ValueError(verdict.defect),
                )
                return verdict.accepts
            if found is not None and (frame.explicit or not positional):
                verdict, _ = found
                answer = bool(func(*args, **named))
                if answer is not verdict.accepts and frame.explicit:
                    raise _unread()
                if answer is not verdict.accepts and verdict.refuses:
                    frame.shared = True
                return answer
            if func in _MERGING and positional and not frame.explicit:
                raise UnreadRefusalError(
                    f"{getattr(func, '__name__', 'a reduction')} answers once for "
                    "every position"
                )
        elif func in _FACTORIZATIONS:
            reporting = _FACTORIZATIONS[cast("Callable[..., torch.Tensor]", func)]
            result, info = reporting(*args, **named)
            valid = info == 0
            self._mark(_tensors((args, named)), (result, valid))
            reject_unless(
                valid,
                lambda: torch.linalg.LinAlgError(
                    f"{getattr(func, '__name__', 'factorization')}: the matrix is "
                    "singular or not positive-definite"
                ),
            )
            return result
        elif func in _DECOMPOSITIONS:
            matrix = cast("torch.Tensor", args[0])
            finite = torch.isfinite(matrix).all(-1).all(-1)
            result = func(
                torch.where(finite[..., None, None], matrix, 0.0), *args[1:], **named
            )
            kept = pytree.tree_map(
                lambda part: torch.where(
                    finite.reshape(
                        *finite.shape, *((1,) * (part.dim() - finite.dim()))
                    ),
                    part,
                    torch.nan,
                ),
                result,
            )
            self._mark([matrix], _tensors(kept))
            return kept
        elif func in _ENTRYWISE and not frame.explicit:
            return elementwise(func)(*args, **named)
        if not frame.explicit:
            return func(*args, **named)
        operands = _tensors((args, named))
        versions = {id(operand): operand._version for operand in operands}
        result = func(*args, **named)
        # An operation in place returns nothing or its operand, whose version
        # it changes; an operand returned as it was is no new value.
        self._mark(
            operands,
            [
                tensor
                for tensor in _tensors((result, args[:1]))
                if versions.get(id(tensor)) != tensor._version
            ],
        )
        return result

    def _mark(
        self, operands: list[torch.Tensor], results: Sequence[torch.Tensor]
    ) -> None:
        """Mark what an operation computes from values led by the positions.

        Inside `lift`, a value computed from one that leads with the
        position axis depends on the position, and leads with it too.

        Parameters
        ----------
        operands : list[torch.Tensor]
            The operation's operands.
        results : Sequence[torch.Tensor]
            What it computed.
        """
        if any(getattr(operand, _ROW_MARK, False) for operand in operands):
            for result in results:
                setattr(result, _ROW_MARK, True)


def map_positions(
    function: Callable[[Positions], torch.Tensor], positions: Positions
) -> PositionValues:
    """Evaluate a function of one position at many.

    Parameters
    ----------
    function : Callable[[Positions], torch.Tensor]
        The function of one position, returning a scalar. It runs once.
    positions : Positions
        The positions: a tensor, or nested tuples and dictionaries of
        them, the position axis leading each.

    Returns
    -------
    PositionValues
        The value at each position, and whether each position passed
        every check the function makes of it.

    Raises
    ------
    Exception
        Any error the function raises that is not a check of a position,
        and the defect of the first failing check that positions still
        accepted fail.
    """
    _check_contract()
    return _evaluate(function, positions)


def _evaluate(
    function: Callable[[Positions], torch.Tensor], positions: Positions
) -> PositionValues:
    """Evaluate a function of one position at many, as `map_positions` does.

    Parameters
    ----------
    function : Callable[[Positions], torch.Tensor]
        The function of one position, returning a scalar.
    positions : Positions
        The positions, the position axis leading each tensor.

    Returns
    -------
    PositionValues
        The value at each position, and whether each is accepted.

    Raises
    ------
    Exception
        As `map_positions` raises it.
    """
    leading = pytree.tree_leaves(positions)[0]
    count = int(leading.shape[0])
    frame = _Frame(count)

    def at(position: Positions) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """The function and the checks it makes, at one position.

        Parameters
        ----------
        position : Positions
            The position.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, ...]]
            The value, and each check's outcome.
        """
        frame.level = current_level()
        value = function(position)
        return value, tuple(check.valid for check in frame.checks)

    token = _FRAME.set(frame)
    try:
        with _Refusals(frame):
            values, outcomes = torch.func.vmap(at, randomness="different")(positions)
    except (ValueError, RuntimeError) as error:
        _account(frame, error)
        raise
    finally:
        _FRAME.reset(token)
    accepted = torch.ones(count, dtype=torch.bool, device=leading.device)
    for check, valid in zip(frame.checks, outcomes, strict=True):
        if check.defect is not None and bool((accepted & ~valid).any()):
            raise check.defect
        accepted = accepted & valid
    return PositionValues(values=values, accepted=accepted)


_CONTRACT: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...] = (
    (
        "a parameter validator",
        lambda z: Normal(0.0, z, validate_args=True).log_prob(torch.zeros_like(z)),
    ),
    (
        "a value validator",
        lambda z: Gamma(
            z.new_tensor(2.0), z.new_tensor(1.0), validate_args=True
        ).log_prob(z),
    ),
    (
        "a matrix validator",
        lambda z: MultivariateNormal(
            z.new_zeros(2),
            covariance_matrix=z * torch.eye(2, dtype=z.dtype),
            validate_args=True,
        ).log_prob(z.new_ones(2)),
    ),
    (
        "a factorization",
        lambda z: torch.linalg.cholesky(z * torch.eye(2, dtype=z.dtype)).sum(),
    ),
    (
        "a validator inside a batched callable",
        lambda z: lift(
            lambda scale: Normal(0.0, scale, validate_args=True).log_prob(
                torch.zeros_like(scale)
            ),
            "the contract",
            z,
        ),
    ),
)
"""Functions of a scalar that refuse it exactly when it is negative."""


@functools.cache
def _check_contract() -> None:
    """Check once that torch refuses positions the way this module reads it.

    `_Refusals` and `lift` rest on how torch asks its validators'
    questions, on which operations raise, and on the mapping levels of
    `torch.func.vmap`, none of which torch promises. Each is exercised here
    on positions of which some are refused and some are not: a version that
    asks differently is named in an error, rather than refusing every
    position or none.

    Raises
    ------
    UnreadRefusalError
        If a function of `_CONTRACT` refuses other positions than those it
        refuses alone, or gives an accepted position another value.
    """
    rows = torch.tensor([1.0, -1.0, 2.0, -3.0, 0.5], dtype=torch.float64)
    expected = rows > 0
    for subject, function in _CONTRACT:
        try:
            found = _evaluate(function, rows)
        except Exception as error:
            raise UnreadRefusalError(f"{subject} stops: {error}") from error
        if not torch.equal(found.accepted, expected):
            raise UnreadRefusalError(
                f"{subject} refuses positions {(~found.accepted).nonzero().flatten().tolist()} "
                f"where it refuses {(~expected).nonzero().flatten().tolist()} one at a time"
            )
        alone = torch.stack([function(row) for row in rows[expected]])
        if not torch.allclose(found.values[expected], alone, rtol=1e-12, atol=0.0):
            raise UnreadRefusalError(
                f"{subject} gives an accepted position another value"
            )
