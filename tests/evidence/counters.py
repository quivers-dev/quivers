"""Counters of the work a computation does.

Work is counted inside tensor operations: an elementwise operation
counts its elements, a reduction counts the elements it reads, a matrix
product counts its scalar products. Memory is read from the allocator's
own events. Calls of Python functions are counted by the interpreter's
monitoring events, so no function is replaced. A counter forwards every
operation unchanged, which `tests/evidence/test_counters_do_not_interfere`
checks bit by bit.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import CodeType, FunctionType
from typing import Protocol

import didactic.api as dx
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from quivers.stochastic.semiring import ChartSemiring


class WorkReport(dx.Model):
    """The work of one computation.

    Parameters
    ----------
    sums : int
        Scalar additions and comparisons of a semiring sum.
    products : int
        Scalar multiplications of a semiring product.
    divisions : int
        Scalar divisions, and reciprocals.
    elementary : int
        Scalar exponentials and logarithms.
    solves : int
        Scalar operations of linear solves and eigenvalue problems.
    moved : int
        Elements copied, selected, or compared without arithmetic.
    allocated : int
        Elements of new tensors.
    kernels : int
        Tensor operations dispatched.
    saved : int
        Elements saved for the backward pass.
    allocated_bytes : int
        Bytes requested from the allocator.
    peak_bytes : int
        The largest number of bytes live at once.
    calls : dict of str to int
        Calls of watched Python functions.
    """

    sums: int = 0
    products: int = 0
    divisions: int = 0
    elementary: int = 0
    solves: int = 0
    moved: int = 0
    allocated: int = 0
    kernels: int = 0
    saved: int = 0
    allocated_bytes: int = 0
    peak_bytes: int = 0
    calls: dict[str, int] = {}

    @property
    def scalar(self) -> int:
        """All scalar arithmetic.

        Returns
        -------
        int
            The sum of the arithmetic counters.
        """
        return (
            self.sums + self.products + self.divisions + self.elementary + self.solves
        )


#: Operations by the first part of their dispatched name.
ADDITIVE = frozenset(
    {
        "add",
        "add_",
        "sub",
        "sub_",
        "rsub",
        "sum",
        "cumsum",
        "maximum",
        "minimum",
        "max",
        "min",
        "amax",
        "amin",
        "scatter_add",
        "scatter_add_",
        "index_add",
        "index_add_",
        "_index_put_impl_",
    }
)
MULTIPLICATIVE = frozenset({"mul", "mul_", "prod"})
DIVISIVE = frozenset({"div", "div_", "reciprocal"})
ELEMENTARY = frozenset(
    {
        "exp",
        "log",
        "log1p",
        "expm1",
        "logsumexp",
        "logaddexp",
        "log_softmax",
        "_log_softmax",
        "softmax",
        "_softmax",
        "log_sigmoid_forward",
        "sigmoid",
    }
)
PRODUCTS = frozenset({"mm", "bmm", "matmul", "mv", "dot", "addmm"})
SOLVES = frozenset(
    {
        "linalg_solve_ex",
        "_linalg_solve_ex",
        "linalg_eigvals",
        "_linalg_eigvals",
        "linalg_eig",
        "linalg_inv_ex",
        "linalg_lu_factor_ex",
        "linalg_lu",
        "triangular_solve",
        "linalg_solve_triangular",
        "linalg_lu_solve",
        "_linalg_det",
        "_linalg_slogdet",
    }
)


type DispatchValue = (
    torch.Tensor
    | int
    | float
    | bool
    | str
    | None
    | torch.dtype
    | torch.device
    | Sequence[DispatchValue]
    | Mapping[str, DispatchValue]
)
"""An argument or a result of a dispatched operation."""


class Operation(Protocol):
    """A dispatched tensor operation."""

    __name__: str

    def __call__(self, *args: DispatchValue, **kwargs: DispatchValue) -> DispatchValue:
        """Run the operation.

        Parameters
        ----------
        *args : DispatchValue
            The positional arguments.
        **kwargs : DispatchValue
            The keyword arguments.

        Returns
        -------
        DispatchValue
            The result.
        """
        ...


def tensors(value: DispatchValue) -> Iterator[torch.Tensor]:
    """The tensors inside a dispatched value.

    Parameters
    ----------
    value : DispatchValue
        The value.

    Yields
    ------
    torch.Tensor
        Each tensor.
    """
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


class TensorCounter(TorchDispatchMode):
    """A dispatch mode that counts scalar work and forwards every operation."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = dict.fromkeys(
            (
                "sums",
                "products",
                "divisions",
                "elementary",
                "solves",
                "moved",
                "allocated",
                "kernels",
            ),
            0,
        )
        self.largest = 0

    def __torch_dispatch__(
        self,
        func: Operation,
        types: Sequence[type],
        args: Sequence[DispatchValue] = (),
        kwargs: Mapping[str, DispatchValue] | None = None,
    ) -> DispatchValue:
        """Forward one operation and count it.

        Parameters
        ----------
        func : Operation
            The operation.
        types : Sequence of type
            The tensor subclasses involved, unused.
        args : Sequence of DispatchValue
            The positional arguments.
        kwargs : Mapping of str to DispatchValue, optional
            The keyword arguments.

        Returns
        -------
        DispatchValue
            The operation's own result.
        """
        del types
        result = func(*args, **(kwargs or {}))
        name = func.__name__.split(".")[0]
        read = max((t.numel() for t in tensors(tuple(args))), default=0)
        made = sum(t.numel() for t in tensors(result))
        wide = max(read, made)
        self.counts["kernels"] += 1
        if name in ADDITIVE:
            self.counts["sums"] += wide
        elif name in MULTIPLICATIVE:
            self.counts["products"] += wide
        elif name in DIVISIVE:
            self.counts["divisions"] += wide
        elif name in ELEMENTARY:
            self.counts["elementary"] += wide
        elif name in PRODUCTS:
            inner = next(tensors(tuple(args))).shape[-1]
            self.counts["products"] += max(made, 1) * inner
            self.counts["sums"] += max(made, 1) * inner
        elif name in SOLVES:
            side = max(
                (t.shape[-1] for t in tensors(tuple(args)) if t.dim()), default=0
            )
            self.counts["solves"] += side**3
        else:
            self.counts["moved"] += made
        owners = {t.untyped_storage()._cdata for t in tensors(tuple(args))}
        fresh = [t for t in tensors(result) if t.untyped_storage()._cdata not in owners]
        self.counts["allocated"] += sum(t.numel() for t in fresh)
        self.largest = max([self.largest, *(t.numel() for t in fresh)])
        return result


class CallCounter:
    """Counts calls of Python functions through interpreter events.

    Parameters
    ----------
    functions : Mapping[str, FunctionType]
        The functions to watch, by the name to report.
    """

    def __init__(self, functions: Mapping[str, FunctionType]) -> None:
        self.counts = dict.fromkeys(functions, 0)
        self._names: dict[CodeType, str] = {
            function.__code__: name for name, function in functions.items()
        }
        self._tool = 0

    def __enter__(self) -> CallCounter:
        """Start counting.

        Returns
        -------
        CallCounter
            The counter.
        """
        monitoring = sys.monitoring
        self._tool = next(
            tool for tool in range(6) if monitoring.get_tool(tool) is None
        )
        monitoring.use_tool_id(self._tool, "evidence")
        monitoring.register_callback(
            self._tool, monitoring.events.PY_START, self._count
        )
        for code in self._names:
            monitoring.set_local_events(self._tool, code, monitoring.events.PY_START)
        return self

    def __exit__(self, *exception: type | BaseException | None) -> None:
        """Stop counting.

        Parameters
        ----------
        *exception : type or BaseException or None
            The exception in flight, which is not handled.
        """
        for code in self._names:
            sys.monitoring.set_local_events(self._tool, code, 0)
        sys.monitoring.free_tool_id(self._tool)

    def _count(self, code: CodeType, offset: int) -> None:
        """Count one call.

        Parameters
        ----------
        code : CodeType
            The code entered.
        offset : int
            The instruction, unused.
        """
        del offset
        name = self._names.get(code)
        if name is not None:
            self.counts[name] += 1


@contextmanager
def saved_elements() -> Iterator[list[int]]:
    """Count the elements saved for the backward pass.

    Yields
    ------
    list of int
        One entry per saved tensor.
    """
    sizes: list[int] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Record a saved tensor and keep it.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        sizes.append(tensor.numel())
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        """Return a saved tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield sizes


def measure(
    body: Callable[[], torch.Tensor | None],
    functions: Mapping[str, FunctionType] | None = None,
) -> WorkReport:
    """Run a computation and report its scalar work and calls.

    Parameters
    ----------
    body : Callable[[], torch.Tensor or None]
        The computation.
    functions : Mapping[str, FunctionType] or None
        Python functions whose calls are counted.

    Returns
    -------
    WorkReport
        The work.
    """
    counter = TensorCounter()
    calls = CallCounter(functions or {})
    with calls, saved_elements() as sizes, counter:
        body()
    return WorkReport(**counter.counts, saved=sum(sizes), calls=calls.counts)


def memory(body: Callable[[], torch.Tensor | None]) -> WorkReport:
    """Run a computation and report its allocations.

    The figures come from the allocator's events, so they include the
    buffers of kernels and of the backward graph.

    Parameters
    ----------
    body : Callable[[], torch.Tensor or None]
        The computation.

    Returns
    -------
    WorkReport
        The allocated bytes and the peak of live bytes.
    """
    with profile(
        activities=[ProfilerActivity.CPU], profile_memory=True, acc_events=True
    ) as session:
        result = body()
    del result
    live = 0
    peak = 0
    total = 0
    recorder = session.profiler
    assert recorder is not None
    events = recorder.kineto_results.events()
    for event in sorted(
        (e for e in events if e.name() == "[memory]"), key=lambda e: e.start_ns()
    ):
        size = event.nbytes()
        live += size
        total += max(size, 0)
        peak = max(peak, live)
    return WorkReport(allocated_bytes=total, peak_bytes=peak)


def forwarding(base: ChartSemiring) -> tuple[ChartSemiring, dict[str, int]]:
    """A semiring that counts its operations and forwards them.

    The result is an instance of a subclass of the base's class, so code
    that asks for the class treats it the same way.

    Parameters
    ----------
    base : ChartSemiring
        The semiring.

    Returns
    -------
    tuple of (ChartSemiring, dict of str to int)
        The counting semiring, and its counts of scalar sums and products.
    """
    counts = {"sums": 0, "products": 0}
    parent = type(base)

    def times(self: ChartSemiring, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Count and forward a product.

        Parameters
        ----------
        a : torch.Tensor
            The first factor.
        b : torch.Tensor
            The second factor.

        Returns
        -------
        torch.Tensor
            The base's product.
        """
        result = parent.times(self, a, b)
        counts["products"] += result.numel()
        return result

    def plus(self: ChartSemiring, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Count and forward a reduction.

        Parameters
        ----------
        scores : torch.Tensor
            The terms.
        dim : int
            The axis.

        Returns
        -------
        torch.Tensor
            The base's reduction.
        """
        result = parent.plus(self, scores, dim)
        counts["sums"] += scores.numel() - result.numel()
        return result

    def plus_pair(
        self: ChartSemiring, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Count and forward a binary sum.

        Parameters
        ----------
        a : torch.Tensor
            The first term.
        b : torch.Tensor
            The second term.

        Returns
        -------
        torch.Tensor
            The base's sum.
        """
        result = parent.plus_pair(self, a, b)
        counts["sums"] += result.numel()
        return result

    counting = type(
        f"Counted{parent.__name__}",
        (parent,),
        {"times": times, "plus": plus, "plus_pair": plus_pair},
    )
    instance = counting()
    assert isinstance(instance, ChartSemiring)
    return instance, counts
