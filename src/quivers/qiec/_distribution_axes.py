"""Retained batch/event proofs of compositional distribution values."""

from quivers._independent import independent_axes
from quivers.qiec.canonical import sampled_element, tensor_shape
from quivers.qiec.families import FAMILIES
from quivers.qiec.terms import (
    Comprehension,
    DistributionValue,
    LiteralValue,
    PlateAxis,
    TensorValue,
    Value,
    Var,
)


def distribution_axes(
    value: Value,
) -> tuple[tuple[PlateAxis, ...], tuple[PlateAxis, ...]]:
    """Read independent batch axes separately from sampled tensor coordinates.

    Parameters
    ----------
    value : Value
        A checked distribution construction or a tensor of constructions.

    Returns
    -------
    tuple[tuple[PlateAxis, ...], tuple[PlateAxis, ...]]
        Batch and event axes. A typed sampleable reference without a batch
        proof retains its tensor coordinates as event axes.

    Raises
    ------
    TypeError, ValueError
        If Independent's count is not static or consumes unavailable axes.
    """
    if isinstance(value, Comprehension):
        # The enclosing construction's explicit plate owns this tensor axis.
        return distribution_axes(value.body)
    if isinstance(value, TensorValue) and value.items:
        axes = distribution_axes(value.items[0])
        if any(distribution_axes(item) != axes for item in value.items[1:]):
            raise ValueError(
                "component distributions have incompatible batch/event axes"
            )
        return axes
    if isinstance(value, DistributionValue):
        record = FAMILIES[value.name]
        arguments = dict(value.arguments)
        if record.compositional:
            base = arguments.get("base", arguments.get("component"))
            if base is not None:
                batch, event = distribution_axes(base)
                if value.name == "Independent":
                    count = arguments.get("reinterpreted_batch_ndims")
                    if not isinstance(count, LiteralValue):
                        raise ValueError(
                            "Independent requires a static integer reinterpreted_batch_ndims"
                        )
                    batch, event = independent_axes(batch, event, count.value)
                return (*value.plate.batch, *batch), (*value.plate.event, *event)
        sampled = sampled_element(value.result_type)
        shape = tensor_shape(sampled) if sampled is not None else None
        dimensions = shape[1] if shape is not None else ()
        natural = dimensions[len(value.plate.batch) + len(value.plate.event) :]
        return value.plate.batch, (
            *value.plate.event,
            *(
                PlateAxis(name=f"__native_event_{i}", size=dim)
                for i, dim in enumerate(natural)
            ),
        )
    if isinstance(value, Var):
        sampled = sampled_element(value.local.type)
        shape = tensor_shape(sampled) if sampled is not None else None
        if shape is not None:
            return (), tuple(
                PlateAxis(name=f"__reference_event_{i}", size=dim)
                for i, dim in enumerate(shape[1])
            )
    return (), ()
