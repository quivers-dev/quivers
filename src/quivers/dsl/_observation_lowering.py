"""Recorded events lower to lazy density or mass queries on one sampling law."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from quivers._observation import EVENT_OPTIONS
from quivers.dsl.ast_nodes import ObserveStep, OptionName, OptionNumber
from quivers.dsl.program_templates import static_program_parameters
from quivers.qiec.canonical import (
    LOG_WEIGHT,
    sampled_element,
    tensor_shape,
    tensor_type,
)
from quivers.qiec.families import FAMILIES
from quivers.qiec.terms import (
    Comprehension,
    ConditionalValue,
    DistributionValue,
    Gather,
    LiteralValue,
    Local,
    LogDensity,
    LogMass,
    PlateShape,
    Value,
    Var,
)
from quivers.qiec.types import BOOL, INT, REAL

if TYPE_CHECKING:
    from quivers.dsl.program_elaboration import _Elaborator, _ProgramState, _Scope


def recorded_density(
    elaborator: _Elaborator,
    distribution: DistributionValue,
    observed: Value,
    step: ObserveStep,
    scope: _Scope,
    state: _ProgramState,
) -> Value:
    """Retain independent rows while scoring only their active event branch.

    Parameters
    ----------
    elaborator : _Elaborator
        Existing checked program elaborator and primitive constructor.
    distribution : DistributionValue
        Checked response law, before any recording or truncation.
    observed : Value
        Exact outcomes or first censoring endpoints.
    step : ObserveStep
        Source options naming retained response inputs.
    scope : _Scope
        Runtime and static scope at the observation.
    state : _ProgramState
        Shared input-contract and program construction state.

    Returns
    -------
    Value
        Unweighted LogWeight scalar or tensor retaining response batch axes.

    Raises
    ------
    QiecDiagnosticError
        If options, response events, carriers, or batch shapes are invalid.
    """
    e = elaborator
    entries = {
        entry.key: entry.value for entry in step.options if entry.key in EVENT_OPTIONS
    }
    if not entries:
        return LogDensity(
            sampleable=distribution,
            value=observed,
            origin=distribution.origin,
            batch=distribution.plate.batch,
        )
    if len(entries) != sum(entry.key in EVENT_OPTIONS for entry in step.options):
        e._fail(step, "observation event options must occur once", code="qiec-program")
    if "cens_upper" in entries and "cens" not in entries:
        e._fail(step, "cens_upper requires a cens status", code="qiec-program")
    if distribution.plate.event or FAMILIES[distribution.name].event_rank:
        e._fail(
            step, "recorded events require scalar response events", code="qiec-program"
        )
    dimensions = tuple(axis.size for axis in distribution.plate.batch)
    path = ("programs", state.declaration.name, "sites", step.vars[0], "event")
    values: dict[str, Value] = {}
    for key, option in entries.items():
        carrier = INT if EVENT_OPTIONS[key][1] == "Int" else REAL
        expected = tensor_type(carrier, dimensions) if dimensions else carrier
        if isinstance(option, OptionNumber):
            number = option.value
            if carrier == INT:
                if number not in {-1, 0, 1, 2}:
                    e._fail(
                        step,
                        "cens requires integer codes -1, 0, 1, or 2",
                        code="qiec-program",
                    )
                number = int(number)
            elif math.isnan(number):
                e._fail(step, f"{key} requires non-NaN endpoints", code="qiec-program")
            if carrier == REAL and math.isinf(number):
                negative = e._primitive(
                    "log_probability",
                    (LiteralValue(value=0.0, type=REAL),),
                    step,
                    (*path, key, "negative-infinity"),
                )
                values[key] = (
                    negative
                    if number < 0
                    else e._primitive(
                        "neg_real", (negative,), step, (*path, key, "positive-infinity")
                    )
                )
            else:
                values[key] = LiteralValue(
                    value=number if carrier == INT else float(number), type=carrier
                )
        elif isinstance(option, OptionName) and option.value not in {"true", "false"}:
            local = scope.lookup(option.value)
            if local is None:
                static_names = {
                    str(parameter.name)
                    for parameter in static_program_parameters(state.declaration)
                }
                if option.value in e._program_objects or option.value in static_names:
                    e._fail(
                        step,
                        f"{key} requires a runtime value; {option.value!r} is static",
                        code="qiec-program",
                    )
                local = e._declare_parameter(
                    option.value,
                    e._pinned_types.get(option.value, expected),
                    "data",
                    scope,
                    state,
                    step,
                    axes=tuple(axis.name for axis in distribution.plate.batch),
                )
            if local.type not in (carrier, expected):
                e._fail(
                    step,
                    f"{key} requires {e._render(carrier)} or {e._render(expected)}, got {e._render(local.type)}",
                    code="qiec-program",
                )
            values[key] = Var(local=local)
        else:
            e._fail(
                step,
                f"{key} requires a runtime value name or number",
                code="qiec-program",
            )

    indices = tuple(
        Local(name=f"__event_row_{index}", type=INT) for index in range(len(dimensions))
    )
    scalar_scope = scope.with_(
        locals={**scope.locals, **{index.name: index for index in indices}}
    )

    def scalar(value: Value, intrinsic: int = 0) -> Value:
        if isinstance(value, DistributionValue):
            record = FAMILIES[value.name]
            ranks = {parameter.name: parameter.rank for parameter in record.parameters}
            arguments = tuple(
                (name, scalar(argument, ranks.get(name, 0)))
                for name, argument in value.arguments
            )
            # Nested constructions keep intrinsic component/event axes; only
            # the enclosing response replication is discharged here.
            batch = tuple(
                axis
                for axis in value.plate.batch
                if axis not in distribution.plate.batch
            )
            provisional = value.with_(
                arguments=arguments,
                plate=PlateShape(batch=batch, event=value.plate.event),
            )
            return provisional.with_(
                result_type=e._plated_type(provisional, scalar_scope, step)
            )
        actual = e._value_type(value, scalar_scope.context(), step)
        shape = tensor_shape(actual)
        if shape is None:
            return value
        element, sizes = shape
        count = max(0, len(sizes) - intrinsic)
        if count > len(indices):
            e._fail(
                step,
                "event parameters introduce undeclared response batch axes",
                code="qiec-program",
            )
        result = value
        for index in indices[len(indices) - count :]:
            sizes = sizes[1:]
            result = Gather(
                value=result,
                index=Var(local=index),
                result_type=tensor_type(element, sizes) if sizes else element,
            )
        return result

    base = scalar(distribution)
    assert isinstance(base, DistributionValue)
    point = scalar(observed)
    rows = {name: scalar(value) for name, value in values.items()}
    sampled = sampled_element(base.result_type)
    assert sampled is not None

    def restrict(
        law: DistributionValue, low: Value, high: Value, *, closed: bool = True
    ) -> DistributionValue:
        record = FAMILIES["Restrict"]
        return DistributionValue(
            family=record.id,
            name=record.name,
            arguments=(
                ("base", law),
                ("low", low),
                ("high", high),
                ("lower_closed", LiteralValue(value=closed, type=BOOL)),
                ("upper_closed", LiteralValue(value=True, type=BOOL)),
            ),
            result_type=law.result_type,
            origin=distribution.origin,
        )

    # Canonical JSON carries finite literals. Infinite endpoints are values
    # of existing closed primitives, rather than nonportable JSON numbers.
    negative = e._primitive(
        "log_probability",
        (LiteralValue(value=0.0, type=REAL),),
        step,
        (*path, "negative-infinity"),
    )
    infinity = e._primitive("neg_real", (negative,), step, (*path, "positive-infinity"))
    if "trunc_lower" in rows or "trunc_upper" in rows:
        restricted = restrict(
            base, rows.get("trunc_lower", negative), rows.get("trunc_upper", infinity)
        )
        record = FAMILIES["Normalize"]
        base = DistributionValue(
            family=record.id,
            name=record.name,
            arguments=(("base", restricted),),
            result_type=base.result_type,
            origin=distribution.origin,
        )
    result: Value = LogDensity(sampleable=base, value=point, origin=distribution.origin)
    if "cens" in rows:
        status = e._primitive(
            "observation_status", (rows["cens"],), step, (*path, "status")
        )
        endpoint = (
            e._primitive(
                "bool_to_real" if sampled == BOOL else "int_to_real",
                (point,),
                step,
                (*path, "endpoint"),
            )
            if sampled in (BOOL, INT)
            else point
        )
        upper = e._primitive(
            "interval_upper",
            (endpoint, rows["cens_upper"])
            if "cens_upper" in rows
            else (infinity, LiteralValue(value=0.0, type=REAL)),
            step,
            (*path, "upper"),
        )

        def mass(low: Value, high: Value, *, closed: bool = True) -> Value:
            return LogMass(
                sampleable=restrict(base, low, high, closed=closed),
                origin=distribution.origin,
            )

        result = ConditionalValue(
            condition=e._primitive(
                "eq_int",
                (status, LiteralValue(value=0, type=INT)),
                step,
                (*path, "exact"),
            ),
            then=result,
            otherwise=ConditionalValue(
                condition=e._primitive(
                    "eq_int",
                    (status, LiteralValue(value=-1, type=INT)),
                    step,
                    (*path, "left"),
                ),
                then=mass(negative, endpoint),
                otherwise=ConditionalValue(
                    condition=e._primitive(
                        "eq_int",
                        (status, LiteralValue(value=1, type=INT)),
                        step,
                        (*path, "right"),
                    ),
                    then=mass(endpoint, infinity, closed=False),
                    otherwise=mass(endpoint, upper, closed=False),
                    result_type=LOG_WEIGHT,
                ),
                result_type=LOG_WEIGHT,
            ),
            result_type=LOG_WEIGHT,
        )
    for position in reversed(range(len(indices))):
        result = Comprehension(
            binder=indices[position],
            extent=dimensions[position],
            body=result,
            result_type=tensor_type(LOG_WEIGHT, dimensions[position:]),
        )
    return result
