"""The numbers of the reference machine, as tensors.

Every value of a numeric type that the machine computes is a
`torch.Tensor`: a ``Real`` or a ``LogWeight`` is a zero-dimensional
floating tensor of the run's real dtype, an ``Int`` a zero-dimensional
``int64`` tensor, a ``Bool`` a zero-dimensional ``bool`` tensor, and a
``Tensor`` of one of those element types a tensor of its shape. A value of
any other type keeps its host form: a string, a constructor, a product as a
tuple, a tensor of structured elements as nested tuples.

So the machine keeps the gradient of everything it computes, and a number
leaves the tensor representation in exactly three places, each of which
reads a discrete value and none of which needs a derivative:

* a *decision*, the condition of a branch (`decide`), which selects the
  code that runs;
* a *host index*, an ``Int`` that selects an entry of a tuple of
  structured values or keys an item (`host_index`, `discrete_key`), and a
  number a report states (`host_real`);
* the *host boundary*, where values enter from or return to Python
  (`machine_value`, `host_value`).

Over many positions at once (`quivers._positions.map_positions`) a value
may depend on the position. A decision or a host index read from such a
value has one answer per position, which no single evaluation can follow;
reading one raises `PositionDependentControl`, and the caller evaluates the
positions one at a time instead.
"""

from __future__ import annotations

from typing import Literal

import torch

from quivers._positions import depends_on_position
from quivers.qiec._models import QiecModel
from quivers.qiec.canonical import LOG_WEIGHT, tensor_shape
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    IndexLiteral,
    TypeApplication,
    TypeExpr,
)

type NumberKind = Literal["real", "integer", "truth"]
"""What kind of number a numeric type holds."""

type RealDtype = Literal["float16", "bfloat16", "float32", "float64"]
"""The floating dtypes a run may carry its real numbers in."""

type Number = torch.Tensor
"""A number of the machine."""

_KINDS: dict[TypeExpr, NumberKind] = {
    REAL: "real",
    LOG_WEIGHT: "real",
    INT: "integer",
    BOOL: "truth",
}
"""The numeric scalar types and the kind of number each holds."""


class PositionDependentControl(RuntimeError):
    """A discrete value read for control depends on the position.

    Raised over many positions at once when a branch condition, a host
    index, or an item key differs between positions: one evaluation cannot
    take every branch, so the positions must be evaluated one at a time.

    Parameters
    ----------
    what : str
        What read the value.
    """

    def __init__(self, what: str) -> None:
        super().__init__(f"{what} depends on the position")
        self.what = what


class NumberDomain(QiecModel):
    """The dtype and device a run carries its numbers in.

    Parameters
    ----------
    real
        The floating dtype of ``Real`` and ``LogWeight`` values.
    device
        The device of every number.
    """

    real: RealDtype = "float64"
    device: str = "cpu"

    @classmethod
    def of(cls, tensor: torch.Tensor) -> NumberDomain:
        """The domain of a floating tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            A tensor whose dtype and device the run takes.

        Returns
        -------
        NumberDomain
            The domain; a tensor that is not floating gives the default
            floating dtype.
        """
        dtype = tensor.dtype if tensor.is_floating_point() else torch.get_default_dtype()
        return cls(real=_dtype_name(dtype), device=str(tensor.device))

    @classmethod
    def default(cls) -> NumberDomain:
        """The domain of torch's default floating dtype on the CPU.

        Returns
        -------
        NumberDomain
            The domain.
        """
        return cls(real=_dtype_name(torch.get_default_dtype()))

    @property
    def real_dtype(self) -> torch.dtype:
        """The floating dtype.

        Returns
        -------
        torch.dtype
            The dtype ``real`` names.
        """
        return getattr(torch, self.real)

    @property
    def torch_device(self) -> torch.device:
        """The device.

        Returns
        -------
        torch.device
            The device ``device`` names.
        """
        return torch.device(self.device)

    def dtype(self, kind: NumberKind) -> torch.dtype:
        """The dtype of one kind of number.

        Parameters
        ----------
        kind : NumberKind
            The kind.

        Returns
        -------
        torch.dtype
            The real dtype, ``int64`` or ``bool``.
        """
        if kind == "real":
            return self.real_dtype
        if kind == "integer":
            return torch.int64
        return torch.bool

    def number(self, kind: NumberKind, value: bool | int | float) -> torch.Tensor:
        """A host scalar as a number of the machine.

        Parameters
        ----------
        kind : NumberKind
            The kind of number.
        value : bool | int | float
            The scalar.

        Returns
        -------
        torch.Tensor
            A zero-dimensional tensor of the kind's dtype.
        """
        return torch.tensor(value, dtype=self.dtype(kind), device=self.torch_device)

    def zero(self) -> torch.Tensor:
        """The real zero, which is the log weight of a certain event.

        Returns
        -------
        torch.Tensor
            ``0`` in the real dtype.
        """
        return self.number("real", 0.0)


def _dtype_name(dtype: torch.dtype) -> RealDtype:
    """The name of a floating dtype.

    Parameters
    ----------
    dtype : torch.dtype
        The dtype.

    Returns
    -------
    RealDtype
        Its name.

    Raises
    ------
    TypeError
        If the dtype is not one a run carries reals in.
    """
    names: dict[torch.dtype, RealDtype] = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    if dtype not in names:
        raise TypeError(f"{dtype} is not a real dtype of the machine")
    return names[dtype]


def number_kind(type_: TypeExpr) -> NumberKind | None:
    """The kind of number a type's values are, if they are numbers.

    Parameters
    ----------
    type_ : TypeExpr
        A type.

    Returns
    -------
    NumberKind | None
        The kind for a numeric scalar type or a tensor of one, ``None``
        for every other type.
    """
    kind = _KINDS.get(type_)
    if kind is not None:
        return kind
    shape = tensor_shape(type_)
    if shape is not None:
        return _KINDS.get(shape[0])
    return None


def static_shape(type_: TypeExpr) -> tuple[int, ...] | None:
    """The literal shape of a tensor type.

    Parameters
    ----------
    type_ : TypeExpr
        A type.

    Returns
    -------
    tuple[int, ...] | None
        The dimensions when every one is a literal; ``()`` for a scalar
        type; ``None`` otherwise.
    """
    if type_ in _KINDS:
        return ()
    shape = tensor_shape(type_)
    if shape is None:
        return None
    dimensions: list[int] = []
    for dimension in shape[1]:
        if not isinstance(dimension, IndexLiteral) or not isinstance(
            dimension.value, int
        ):
            return None
        dimensions.append(dimension.value)
    return tuple(dimensions)


def _kind_admits(kind: NumberKind, dtype: torch.dtype) -> bool:
    """Whether a tensor's dtype holds numbers of a kind exactly.

    Parameters
    ----------
    kind : NumberKind
        The kind.
    dtype : torch.dtype
        The tensor's dtype.

    Returns
    -------
    bool
        Floating or integer dtypes for reals, integer dtypes for
        integers, ``bool`` for truth values.
    """
    if dtype == torch.bool:
        return kind == "truth"
    if dtype.is_floating_point:
        return kind == "real"
    if dtype.is_complex:
        return False
    return kind in ("real", "integer")


def _host_admits(kind: NumberKind, value: object) -> bool:
    """Whether a host scalar is a number of a kind.

    Parameters
    ----------
    kind : NumberKind
        The kind.
    value : object
        The scalar.

    Returns
    -------
    bool
        A ``bool`` for truth values, an ``int`` that is no ``bool`` for
        integers, an ``int`` or ``float`` that is no ``bool`` for reals.
    """
    if isinstance(value, bool):
        return kind == "truth"
    if isinstance(value, int):
        return kind in ("real", "integer")
    if isinstance(value, float):
        return kind == "real"
    return False


def _host_entries(value: object, kind: NumberKind) -> bool:
    """Whether nested host sequences hold only numbers of a kind.

    Parameters
    ----------
    value : object
        Nested tuples or lists, or a scalar.
    kind : NumberKind
        The kind.

    Returns
    -------
    bool
        Whether every leaf is a host number of the kind.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, tuple | list):
            pending.extend(item)
        elif not _host_admits(kind, item):
            return False
    return True


def machine_value(type_: TypeExpr, value: object, domain: NumberDomain) -> object:
    """A host value as the machine holds it, by its checked type.

    The value is converted where it enters the machine: a run's argument,
    an attachment, the environment of a host evaluation. A tensor keeps its
    gradient; only its dtype and device are made the run's.

    Parameters
    ----------
    type_ : TypeExpr
        The value's checked type.
    value : object
        The host value: a number, nested tuples or lists of numbers, a
        tensor, or any value of a non-numeric type.
    domain : NumberDomain
        The run's numbers.

    Returns
    -------
    object
        A tensor for a value of a numeric type whose host form is one;
        components converted for a product; the value itself otherwise,
        so a value of the wrong form reaches the type's validator, which
        names it.
    """
    kind = number_kind(type_)
    if kind is not None:
        return _number(type_, kind, value, domain)
    if (
        isinstance(type_, TypeApplication)
        and type_.constructor.name.startswith("Product")
        and isinstance(value, tuple | list)
        and len(value) == len(type_.arguments)
    ):
        return tuple(
            machine_value(item, component, domain)  # type: ignore[arg-type]
            for item, component in zip(type_.arguments, value, strict=True)
        )
    shape = tensor_shape(type_)
    if shape is not None and isinstance(value, tuple | list):
        element = shape[0]
        rank = len(shape[1])
        return _nested_elements(value, element, rank, domain)
    return value


def _nested_elements(
    value: tuple[object, ...] | list[object],
    element: TypeExpr,
    rank: int,
    domain: NumberDomain,
) -> tuple[object, ...]:
    """A host tensor of structured elements as nested tuples.

    Parameters
    ----------
    value : tuple or list
        The entries along the outermost axis.
    element : TypeExpr
        The element type.
    rank : int
        The axes left, at least one.
    domain : NumberDomain
        The run's numbers.

    Returns
    -------
    tuple[object, ...]
        Nested tuples, each element converted by its type.
    """
    if rank <= 1:
        return tuple(machine_value(element, item, domain) for item in value)
    return tuple(
        _nested_elements(item, element, rank - 1, domain)
        if isinstance(item, tuple | list)
        else item
        for item in value
    )


def _number(
    type_: TypeExpr, kind: NumberKind, value: object, domain: NumberDomain
) -> object:
    """A host number or tensor of numbers as the machine's tensor.

    Parameters
    ----------
    type_ : TypeExpr
        The numeric type.
    kind : NumberKind
        The kind of its numbers.
    value : object
        The host value.
    domain : NumberDomain
        The run's numbers.

    Returns
    -------
    object
        The tensor, or the value unchanged when it is no number of the
        kind.
    """
    dtype = domain.dtype(kind)
    if isinstance(value, torch.Tensor):
        if not _kind_admits(kind, value.dtype):
            return value
        if kind == "integer" and value.dtype != torch.int64:
            return value.to(dtype=torch.int64, device=domain.torch_device)
        if value.dtype == dtype and value.device == domain.torch_device:
            return value
        return value.to(dtype=dtype, device=domain.torch_device)
    if not _host_entries(value, kind):
        return value
    tensor = torch.tensor(value, dtype=dtype, device=domain.torch_device)
    shape = static_shape(type_)
    if shape is not None and tensor.numel() == 0 and len(shape) > tensor.dim():
        tensor = tensor.reshape(shape)
    return tensor


def host_value(value: object) -> object:
    """A machine value as plain Python data, for the host to print or store.

    Parameters
    ----------
    value : object
        A machine value.

    Returns
    -------
    object
        A number for a zero-dimensional tensor, nested lists of numbers for
        a tensor of higher rank, the components of a tuple converted, and
        any other value itself.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, tuple):
        return tuple(host_value(item) for item in value)
    return value


def decide(value: object, what: str) -> bool:
    """Read a truth value that selects the code a run takes.

    Parameters
    ----------
    value : object
        The machine's truth value.
    what : str
        What decides, for the diagnostics.

    Returns
    -------
    bool
        The truth value.

    Raises
    ------
    TypeError
        If the value is not a scalar truth value.
    PositionDependentControl
        If, over many positions, it depends on the position.
    """
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or value.dim() != 0
    ):
        raise TypeError(f"{what} did not evaluate to a scalar Boolean")
    if depends_on_position(value):
        raise PositionDependentControl(what)
    return bool(value)


def host_index(value: object, what: str) -> int:
    """Read an integer that selects among host values.

    Parameters
    ----------
    value : object
        The machine's integer.
    what : str
        What reads it, for the diagnostics.

    Returns
    -------
    int
        The integer.

    Raises
    ------
    TypeError
        If the value is not a scalar integer.
    PositionDependentControl
        If, over many positions, it depends on the position.
    """
    if (
        not isinstance(value, torch.Tensor)
        or value.dim() != 0
        or value.dtype.is_floating_point
        or value.dtype == torch.bool
    ):
        raise TypeError(f"{what} did not evaluate to a scalar Int")
    if depends_on_position(value):
        raise PositionDependentControl(what)
    return int(value)


def host_real(value: torch.Tensor, what: str) -> float:
    """Read a real number a report or a diagnostic states.

    Parameters
    ----------
    value : torch.Tensor
        A scalar real of the machine.
    what : str
        What reads it, for the diagnostics.

    Returns
    -------
    float
        The number; the reading keeps no gradient, since a report is not
        part of the computation.

    Raises
    ------
    PositionDependentControl
        If, over many positions, it depends on the position.
    """
    if depends_on_position(value):
        raise PositionDependentControl(what)
    return float(value.detach())


def discrete_key(value: torch.Tensor) -> tuple[str, tuple[int, ...], tuple[object, ...]]:
    """The value of a tensor of integers or truth values, as a hashable key.

    Parameters
    ----------
    value : torch.Tensor
        A tensor of a dtype that is not floating.

    Returns
    -------
    tuple[str, tuple[int, ...], tuple[object, ...]]
        Its dtype, its shape and its entries in row-major order.

    Raises
    ------
    PositionDependentControl
        If, over many positions, it depends on the position.
    """
    if depends_on_position(value):
        raise PositionDependentControl("an item key")
    return (str(value.dtype), tuple(value.shape), tuple(value.reshape(-1).tolist()))


__all__ = [
    "Number",
    "NumberDomain",
    "NumberKind",
    "PositionDependentControl",
    "RealDtype",
    "decide",
    "discrete_key",
    "host_index",
    "host_real",
    "host_value",
    "machine_value",
    "number_kind",
    "static_shape",
]
