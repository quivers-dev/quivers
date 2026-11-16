"""Dependent shape signatures shared by distribution checking and elaboration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.qiec._models import validate_shape
from quivers.qiec.index_arithmetic import infer_index_extents, solve_index_equation
from quivers.qiec.kinds import NAT, IndexBinder
from quivers.qiec.substitution import StaticSubstitution, substitute_index
from quivers.qiec.types import (
    IndexLiteral,
    IndexSum,
    IndexTerm,
    IndexVariable,
    add_indices,
    shift_index,
)


def _variables(term: IndexTerm) -> tuple[IndexVariable, ...]:
    """Collect nominal variables from a canonical Nat dimension.

    Parameters
    ----------
    term : IndexTerm
        Natural-number dimension expression.

    Returns
    -------
    tuple[IndexVariable, ...]
        Its variable summands, retaining repeated terms and rigid identities.
    """
    canonical = add_indices(term)
    if isinstance(canonical, IndexVariable):
        return (canonical,)
    return canonical.terms if isinstance(canonical, IndexSum) else ()


class FamilyShapeSignature(ValueModel):
    """A Nat telescope relating parameter dimensions and one sample event.

    Parameters
    ----------
    telescope : tuple[IndexBinder, ...]
        Local Nat variables, freshened for every distribution application.
    parameter_shapes : tuple[tuple[str, tuple[IndexTerm, ...]], ...]
        Each parameter's own dimensions, excluding leading plate batch axes.
    event_shape : tuple[IndexTerm, ...]
        One sample's event dimensions. An empty tuple denotes a scalar event.

    Notes
    -----
    All dimension dependencies use the ordinary checked Nat language. Scheme
    variables are local to this signature; they cannot bind a caller's rigid
    indices or identify unrelated caller axes by coincident cardinality.
    """

    telescope: tuple[IndexBinder, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    parameter_shapes: tuple[tuple[str, tuple[IndexTerm, ...]], ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    event_shape: tuple[IndexTerm, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_signature(self) -> None:
        """Check local Nat scope and unique parameter declarations.

        Raises
        ------
        TypeError
            If a field has the wrong structural type or an index is not Nat.
        ValueError
            If names repeat, binders are not Nat, or a dimension escapes scope.
        """
        validate_shape(self.telescope, tuple[IndexBinder, ...], path="telescope")
        validate_shape(
            self.parameter_shapes,
            tuple[tuple[str, tuple[IndexTerm, ...]], ...],
            path="parameter_shapes",
        )
        validate_shape(self.event_shape, tuple[IndexTerm, ...], path="event_shape")
        if any(not isinstance(binder, IndexBinder) for binder in self.telescope):
            raise TypeError("family shape telescope contains a non-index binder")
        names = [binder.name for binder in self.telescope]
        if len(names) != len(set(names)) or any(b.sort != NAT for b in self.telescope):
            raise ValueError("family shape telescope requires distinct Nat binders")
        parameters = [name for name, _ in self.parameter_shapes]
        if len(parameters) != len(set(parameters)):
            raise ValueError("family shape signature repeats a parameter name")
        for shape in (*[shape for _, shape in self.parameter_shapes], self.event_shape):
            for term in shape:
                for variable in _variables(term):
                    if variable.identity is not None or variable.name not in names:
                        raise ValueError(
                            "family shape signature refers to an unbound or rigid index"
                        )

    def parameter_shape(self, name: str) -> tuple[IndexTerm, ...]:
        """Return one parameter's intrinsic dimension pattern.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        tuple[IndexTerm, ...]
            Its dimensions under this signature's telescope.

        Raises
        ------
        KeyError
            If the signature contains no parameter of this name.
        """
        for parameter, shape in self.parameter_shapes:
            if parameter == name:
                return shape
        raise KeyError(name)


def legacy_family_shape_signature(
    parameters: Sequence[tuple[str, int, tuple[int | None, ...]]],
    event_rank: int,
    event_sources: Sequence[str],
    event_offsets: tuple[int, ...] = (),
) -> FamilyShapeSignature:
    """Translate existing event-dimension shorthand into one dependent contract.

    Parameters
    ----------
    parameters : Sequence[tuple[str, int, tuple[int | None, ...]]]
        Parameter name, rank, and legacy event-dimension correspondence.
    event_rank : int
        Native sample event rank.
    event_sources : Sequence[str]
        Alternative parameters fixing the sample's trailing native dimensions.
    event_offsets : tuple[int, ...]
        Existing nonnegative parameter-to-sample offsets.

    Returns
    -------
    FamilyShapeSignature
        A canonical contract consumed by the same checking path as explicitly
        authored shape signatures.

    Raises
    ------
    ValueError
        If legacy ranks, sources, or event-dimension correspondences conflict.
    """
    if type(event_rank) is not int or event_rank < 0:
        raise ValueError("native event rank must be a nonnegative integer")
    if event_offsets and (
        len(event_offsets) != event_rank
        or any(type(v) is not int or v < 0 for v in event_offsets)
    ):
        raise ValueError(
            "event offsets require one nonnegative integer per event dimension"
        )
    ranks = {name: rank for name, rank, _ in parameters}
    for source in event_sources:
        if source not in ranks:
            raise ValueError(f"event source {source!r} is not a parameter")
        if ranks[source] < event_rank:
            raise ValueError(
                f"event source {source!r} has rank below the family's event rank"
            )
    for name, rank, correspondence in parameters:
        if type(rank) is not int or rank < 0:
            raise ValueError(f"parameter {name!r} rank must be a nonnegative integer")
        if correspondence and (
            len(correspondence) != rank
            or any(
                v is not None and (type(v) is not int or not 0 <= v < event_rank)
                for v in correspondence
            )
        ):
            raise ValueError(f"parameter {name!r} has incompatible event dimensions")
    binders = []

    def variable(name):
        """Declare one fresh local Nat axis for translated metadata.

        Parameters
        ----------
        name : str
            Local scheme name.

        Returns
        -------
        IndexVariable
            The variable declared by the appended telescope binder.
        """
        binders.append(IndexBinder(name=name, sort=NAT))
        return IndexVariable(name=name, sort=NAT)

    event = tuple(variable(f"event_{i}") for i in range(event_rank))
    patterns = []
    for name, rank, correspondence in parameters:
        dimensions = []
        for i in range(rank):
            dimension = correspondence[i] if correspondence else None
            dimensions.append(
                event[dimension]
                if dimension is not None
                else variable(f"parameter_{name}_{i}")
            )
        shape = tuple(dimensions)
        patterns.append((name, shape))
    if event_sources and event_rank:
        source = dict(patterns)[event_sources[0]]
        event = tuple(
            shift_index(size, offset)
            for size, offset in zip(
                source[-event_rank:], event_offsets or (0,) * event_rank, strict=True
            )
        )
    return FamilyShapeSignature(
        telescope=tuple(binders), parameter_shapes=tuple(patterns), event_shape=event
    )


def instantiate_family_shapes(
    signature: FamilyShapeSignature, indices: Mapping[str, IndexTerm]
) -> tuple[dict[str, tuple[IndexTerm, ...]], tuple[IndexTerm, ...]]:
    """Instantiate shape patterns using the ordinary capture-safe substitution.

    Parameters
    ----------
    signature : FamilyShapeSignature
        The local dependent shape contract.
    indices : Mapping[str, IndexTerm]
        One replacement for every local Nat binder.

    Returns
    -------
    tuple[dict[str, tuple[IndexTerm, ...]], tuple[IndexTerm, ...]]
        Parameter patterns and sample event pattern after substitution.

    Raises
    ------
    ValueError
        If substitutions do not supply exactly the local telescope.
    """
    if set(indices) != {b.name for b in signature.telescope}:
        raise ValueError(
            "family shape substitution must instantiate its exact telescope"
        )
    for term in indices.values():
        add_indices(term)
    replacement = StaticSubstitution(
        types=(), indices=tuple(indices.items()), effects=()
    )
    return (
        {
            name: tuple(substitute_index(t, replacement) for t in shape)
            for name, shape in signature.parameter_shapes
        },
        tuple(substitute_index(t, replacement) for t in signature.event_shape),
    )


def family_shape_equations(
    parameter_shapes: Mapping[str, tuple[IndexTerm, ...]],
    event_shape: tuple[IndexTerm, ...],
    actual_shapes: Mapping[str, tuple[IndexTerm, ...]],
    event: tuple[IndexTerm, ...] | None = None,
) -> tuple[tuple[IndexTerm, IndexTerm], ...]:
    """Collect the same exact dimension equations for every checking frontend.

    Parameters
    ----------
    parameter_shapes : Mapping[str, tuple[IndexTerm, ...]]
        Instantiated intrinsic shape patterns.
    event_shape : tuple[IndexTerm, ...]
        Instantiated sample event pattern.
    actual_shapes : Mapping[str, tuple[IndexTerm, ...]]
        Supplied intrinsic parameter dimensions; scalar broadcasts omit shape
        evidence rather than inventing dimensions.
    event : tuple[IndexTerm, ...] or None
        Declared native event dimensions, when available.

    Returns
    -------
    tuple[tuple[IndexTerm, IndexTerm], ...]
        Pattern/actual equations, without mutating either scope.

    Raises
    ------
    ValueError
        If a supplied intrinsic parameter or event has the wrong rank.
    """
    pairs = []
    for name, actual in actual_shapes.items():
        expected = parameter_shapes[name]
        if len(actual) != len(expected):
            raise ValueError(
                f"parameter {name!r} has rank {len(actual)}, expected {len(expected)}"
            )
        pairs.extend(zip(expected, actual, strict=True))
    if event is not None:
        if len(event) != len(event_shape):
            raise ValueError("distribution event has the wrong rank")
        pairs.extend(zip(event_shape, event, strict=True))
    return tuple(pairs)


def match_family_shapes(
    signature: FamilyShapeSignature,
    actual_shapes: Mapping[str, tuple[IndexTerm, ...]],
    event: tuple[IndexTerm, ...] | None = None,
) -> tuple[IndexTerm, ...]:
    """Check symbolic shapes by solving only freshly scoped scheme variables.

    Parameters
    ----------
    signature : FamilyShapeSignature
        Dependent family contract.
    actual_shapes : Mapping[str, tuple[IndexTerm, ...]]
        Intrinsic supplied parameter shapes.
    event : tuple[IndexTerm, ...] or None
        Native event dimensions supplied by a plate or result type.

    Returns
    -------
    tuple[IndexTerm, ...]
        The uniquely instantiated native event shape.

    Raises
    ------
    ValueError
        If exact shape equations disagree or cannot determine the event.
    """
    used = {
        v.name
        for shape in (*actual_shapes.values(), event or ())
        for t in shape
        for v in _variables(t)
    }
    indices = {}
    for position, binder in enumerate(signature.telescope):
        name = f"__qvr_family_shape_{position}"
        while name in used:
            name += "_"
        used.add(name)
        indices[binder.name] = IndexVariable(name=name, sort=NAT)
    parameters, expected_event = instantiate_family_shapes(signature, indices)
    equations = list(
        family_shape_equations(parameters, expected_event, actual_shapes, event)
    )
    unknowns = set(indices.values())
    ground = [
        (left, int(right.value))
        for left, right in equations
        if isinstance(right, IndexLiteral) and right.sort == NAT
    ]
    inferred = infer_index_extents(ground)
    replacement = StaticSubstitution(
        types=(),
        indices=tuple(
            (name, IndexLiteral(value=value, sort=NAT))
            for name, value in inferred.items()
        ),
        effects=(),
    )
    unknowns = {v for v in unknowns if v.name not in inferred}
    while True:
        changed = False
        for left, right in equations:
            left, right = (
                substitute_index(left, replacement),
                substitute_index(right, replacement),
            )
            if left == right:
                continue
            for unknown in sorted(unknowns, key=lambda variable: variable.name):
                solution = solve_index_equation(left, right, unknown)
                if solution is None:
                    continue
                fixed = StaticSubstitution(
                    types=(), indices=((unknown.name, solution),), effects=()
                )
                prior = tuple(
                    (name, substitute_index(term, fixed))
                    for name, term in replacement.indices
                )
                replacement = StaticSubstitution(
                    types=(), indices=(*prior, (unknown.name, solution)), effects=()
                )
                unknowns.remove(unknown)
                changed = True
                break
        if not changed:
            break
    if any(
        substitute_index(a, replacement) != substitute_index(b, replacement)
        for a, b in equations
    ):
        raise ValueError("incompatible dependent family parameter/event extents")
    result = tuple(substitute_index(t, replacement) for t in expected_event)
    if any(v in unknowns for t in result for v in _variables(t)):
        raise ValueError("family event shape requires further extent evidence")
    return result


def validate_family_shape_dimensions(
    signature: FamilyShapeSignature, shapes: Mapping[str, Sequence[int]]
) -> dict[str, int]:
    """Validate concrete host dimensions with the shared exact Nat solver.

    Parameters
    ----------
    signature : FamilyShapeSignature
        The same contract used by source and kernel checking.
    shapes : Mapping[str, Sequence[int]]
        Supplied intrinsic tensor dimensions.

    Returns
    -------
    dict[str, int]
        Uniquely inferred scheme extents.

    Raises
    ------
    ValueError
        If a parameter has the wrong rank or exact shape evidence conflicts.
    """
    actual = {}
    for name, dimensions in shapes.items():
        if any(
            isinstance(size, bool) or not isinstance(size, int) for size in dimensions
        ):
            raise TypeError("observed shape extents must be integers")
        actual[name] = tuple(IndexLiteral(value=size, sort=NAT) for size in dimensions)
    equations = family_shape_equations(
        dict(signature.parameter_shapes), signature.event_shape, actual
    )
    inferred = infer_index_extents(
        [
            (pattern, int(cast(IndexLiteral, value).value))
            for pattern, value in equations
        ]
    )
    replacement = StaticSubstitution(
        types=(),
        indices=tuple(
            (name, IndexLiteral(value=value, sort=NAT))
            for name, value in inferred.items()
        ),
        effects=(),
    )
    if any(
        substitute_index(pattern, replacement) != value for pattern, value in equations
    ):
        raise ValueError("family shape contract requires further extent evidence")
    return inferred


__all__ = [
    "FamilyShapeSignature",
    "family_shape_equations",
    "instantiate_family_shapes",
    "legacy_family_shape_signature",
    "match_family_shapes",
    "validate_family_shape_dimensions",
]
