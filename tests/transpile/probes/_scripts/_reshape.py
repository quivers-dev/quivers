"""Shared in-container reshape helper for probe scripts.

The host harness sends each (param, data) value as a flat
row-major Python list under `points.json`. The model the probe
loads, however, typically declares the same name with a multi-
dimensional declared shape (e.g. Stan `array[16] int obs` or PyMC
`obs.shape == (16, 4)`). Each probe script imports
:func:`reshape_point` and rebuilds the multi-dimensional native
container the underlying runtime expects.

The reshape table arrives in `/io/shapes.json`:

    {"obs": [16], "rule_weights": [16], "logits": [3, 4]}

The dtype table (optional) arrives in `/io/dtypes.json`:

    {"obs": "int", "logits": "float"}

A name absent from either table falls through to its raw
list value. The float / int cast applies even for scalars so
backends that distinguish `int` and `real` (Stan, JAGS) get the
right type.

The export channel rides alongside. `/io/export_names.json` holds
the QVR program's return-variable names, in declaration order:

    ["phi"]

A probe that finds the file must read the exported value out of the
emitted program's own return surface (the model function's `return`,
a Stan generated quantity, a Gen `assess` return value) and report
one entry per name per point under the result's `exports` key. The
file's absence means the caller did not ask for the export channel,
and the probe reports only log-densities.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

# The probe payload is strictly numeric: scalars (Python int or
# float) and arbitrarily nested lists of the same. Aliases live
# under `TYPE_CHECKING` so the helper imports on every in-container
# Python (3.9 in JAGS / BUGS, 3.11 in Edward2 / Node, 3.12 in the
# rest); `from __future__ import annotations` defers annotation
# evaluation so the recursive `NestedNumber` reference resolves
# lazily for type checkers and never executes at runtime. `Any`
# and `object` are intentionally avoided.
if TYPE_CHECKING:
    from typing import Protocol

    Number = int | float
    NestedNumber = Number | list["NestedNumber"]
    PointSection = dict[str, NestedNumber]
    Point = dict[str, PointSection]

    class ArrayLike(Protocol):
        """Any runtime array exposing numpy's `tolist`.

        Every probe runtime that carries an exported value back to
        the harness (numpy, jax, torch, tensorflow) satisfies this,
        so the conversion needs neither a runtime import of those
        libraries nor a bare `object` annotation.
        """

        def tolist(self) -> NestedNumber: ...

    ExportValue = NestedNumber | ArrayLike | tuple["ExportValue", ...]


def load_tables(
    io: pathlib.Path,
) -> tuple[
    dict[str, list[int]],
    dict[str, str],
]:
    """Read `/io/shapes.json` and `/io/dtypes.json`. Each absent
    file resolves to an empty dict so the helper is a no-op in
    legacy probe runs that don't ship the tables."""
    shapes_path = io / "shapes.json"
    dtypes_path = io / "dtypes.json"
    shapes = json.loads(shapes_path.read_text()) if shapes_path.exists() else {}
    dtypes = json.loads(dtypes_path.read_text()) if dtypes_path.exists() else {}
    return shapes, dtypes


def _flat_to_nested(
    flat: Sequence[Number],
    shape: list[int],
) -> NestedNumber:
    """Reshape a flat row-major list into a nested list with the
    given shape. Returns the scalar element when `shape == []`."""
    if not shape:
        if len(flat) != 1:
            raise ValueError(f"scalar shape but len(flat)={len(flat)}")
        return flat[0]
    expected = 1
    for d in shape:
        expected *= d
    if len(flat) != expected:
        raise ValueError(
            f"flat length {len(flat)} does not match shape "
            f"{shape} (expected {expected})"
        )
    if len(shape) == 1:
        return list(flat)
    stride = expected // shape[0]
    return [
        _flat_to_nested(flat[i * stride : (i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


def reshape_value(
    name: str,
    value: NestedNumber,
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
) -> NestedNumber:
    """Reshape one (name, value) pair using the loaded tables.

    - When `name` has a shape entry, treat `value` as flat row-major
      and rebuild a nested list of the declared shape (scalar shape
      `[]` collapses a length-1 list to its element).
    - When `name` has a dtype entry, recursively cast every leaf to
      `int` or `float`.
    - When neither table mentions `name`, return the value unchanged
      (preserves the existing 7-fixture probe contract).
    """
    if name in shapes:
        if isinstance(value, (int, float)):
            value = _flat_to_nested([value], shapes[name])
        elif isinstance(value, list):
            value = _flat_to_nested(
                cast("Sequence[Number]", value),
                shapes[name],
            )
    if name in dtypes:
        value = _cast_leaves(value, dtypes[name])
    return value


def _cast_leaves(value: NestedNumber, dtype: str) -> NestedNumber:
    """Recursively cast every leaf of a nested list to the target
    dtype (`"int"` -> Python int, anything else -> float)."""
    if isinstance(value, list):
        return [_cast_leaves(v, dtype) for v in value]
    if dtype == "int":
        return int(value)
    return float(value)


def reshape_point(
    point: Point,
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
) -> Point:
    """Reshape every entry under `params` / `data` in one point."""
    out: Point = {}
    for section in ("params", "data"):
        out[section] = {
            name: reshape_value(name, value, shapes, dtypes)
            for name, value in point.get(section, {}).items()
        }
    return out


def load_export_names(io: pathlib.Path) -> list[str]:
    """Read `/io/export_names.json`.

    Returns the QVR program's return-variable names in declaration
    order, or an empty list when the caller did not ship the file.
    An empty list means "do not report the export channel"; it never
    means "the program exports nothing", because a program with no
    return clause is never scheduled through this channel in the
    first place.
    """
    path = io / "export_names.json"
    if not path.exists():
        return []
    return [str(name) for name in json.loads(path.read_text())]


def as_nested(value: ExportValue) -> NestedNumber:
    """Convert one runtime array / scalar into JSON-ready numbers.

    Every probe runtime that can carry an exported value back to the
    harness (numpy, jax, torch, tensorflow) exposes numpy's `tolist`,
    so the conversion dispatches on that method rather than on the
    concrete array class. A Python tuple (the shape a multi-name
    return takes in every Python target) recurses elementwise; a bare
    scalar widens to `float`.

    A boolean leaf stays an `int`: a Bernoulli-supported export is
    integer-valued, and JSON's `true` would compare unequal against
    the reference's `1.0` under a numeric tolerance.

    Three attribute probes cover every runtime the images carry, in
    order of directness: `tolist` (numpy, jax, torch), `numpy` (a
    TensorFlow eager tensor, which has no `tolist`), and `value`
    (an Edward2 `RandomVariable`, which wraps the tensor its export
    denotes).
    """
    to_list = getattr(value, "tolist", None)
    if to_list is not None:
        return to_list()
    to_numpy = getattr(value, "numpy", None)
    if to_numpy is not None:
        return as_nested(to_numpy())
    wrapped = getattr(value, "value", None)
    if wrapped is not None:
        return as_nested(wrapped)
    if isinstance(value, tuple):
        return [as_nested(item) for item in value]
    if isinstance(value, list):
        return [as_nested(item) for item in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return float(value)


def export_payload(
    names: list[str],
    returned: ExportValue | None,
) -> list[NestedNumber]:
    """Split one model return value into one entry per export name.

    A program returning a single name hands back a bare value; a
    program returning several hands back a tuple, in the order the
    `return` clause declares. The arity is checked rather than
    assumed: a return whose arity disagrees with the requested names
    is a renderer defect (a dropped or duplicated export), and it has
    to surface as a probe failure rather than as a silently truncated
    comparison.
    """
    if not names:
        raise ValueError(
            "export_payload called with no export names; the caller "
            "did not ship /io/export_names.json and the probe must "
            "not report an export channel."
        )
    if returned is None:
        raise ValueError(
            f"the emitted model returns nothing where the QVR program "
            f"exports {names}. A transpilation that drops the return "
            f"clause emits a program denoting the right joint and the "
            f"wrong kernel."
        )
    if len(names) == 1:
        return [as_nested(returned)]
    if not isinstance(returned, tuple):
        raise ValueError(
            f"the emitted model returns a single value where the QVR "
            f"program exports {len(names)} ({names}). The renderer "
            f"dropped part of the program's return clause."
        )
    if len(returned) != len(names):
        raise ValueError(
            f"the emitted model returns {len(returned)} value(s) "
            f"where the QVR program exports {len(names)} ({names})."
        )
    return [as_nested(item) for item in returned]


__all__ = [
    "as_nested",
    "export_payload",
    "load_export_names",
    "load_tables",
    "reshape_point",
    "reshape_value",
]
