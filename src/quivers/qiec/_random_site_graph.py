"""Check the first-order source scope shared by the exact site interpreters."""

from __future__ import annotations

from collections.abc import Callable
from graphlib import CycleError, TopologicalSorter

from quivers.qiec.builtins import RANDOM
from quivers.qiec.checking import KernelRegistry
from quivers.qiec.differentiation import JvpUnsupported
from quivers.qiec.effects import EMPTY_ROW, EffectRequest
from quivers.qiec.identifiers import ComputationId, ConstructorId, EffectInstanceId
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.terms import (
    Bind,
    Call,
    CallAddress,
    Case,
    Computation,
    If,
    Perform,
    Return,
    WithTraceAddresses,
    _CALL_ADDRESS,
)
from quivers.qiec.types import BOOL, REAL, UNIT, TypeApplication, TypeExpr

type _Address = tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]


def _random_computation(
    module: QiecModule,
    computation: ComputationId,
    fragment: str,
) -> tuple[KernelRegistry, NamedComputation]:
    """Resolve a validated scalar source with a single nominal Random row.

    Parameters
    ----------
    module : QiecModule
        The source supplying the actual computations and their rows.
    computation : ComputationId
        The selected unspecialized entry.
    fragment : str
        The exact estimator fragment used in refusal diagnostics.
    Returns
    -------
    tuple[KernelRegistry, NamedComputation]
        The validated registry and actual source declaration.

    Raises
    ------
    ValueError
        If there is no computation of the given identity.
    JvpUnsupported
        If the selected entry leaves the fragment.
    """
    registry = validate_module(module)
    owner = next((item for item in module.computations if item.id == computation), None)
    if owner is None:
        raise ValueError(f"unknown computation for {fragment} plan {computation}")
    row = owner.type.effects
    if (
        owner.telescope
        or owner.type.result != REAL
        or any(parameter.type != REAL for parameter in owner.parameters)
        or row.tail is not None
        or len(row.entries) != 1
        or row.entries[0].effect != RANDOM
    ):
        raise JvpUnsupported(
            f"a {fragment} plan needs a scalar real entry with one nominal Random instance"
        )
    return registry, owner


def _check_first_order_graph(
    module: QiecModule,
    owner: NamedComputation,
    fragment: str,
    check_request: Callable[[EffectRequest, EffectInstanceId], None],
    *,
    require_acyclic: bool = False,
    allow_closed_cases: bool = False,
) -> None:
    """Check the reachable graph of a first-order single-instance entry.

    Parameters
    ----------
    module : QiecModule
        The validated module providing the actual calls.
    owner : NamedComputation
        The selected first-order scalar source with a single nominal effect.
    fragment : str
        The fragment named in refusal diagnostics.
    check_request : Callable
        The fragment's check of an operation on the chosen nominal instance.
    require_acyclic : bool
        Require the reachable source declarations to form an acyclic call graph.
    allow_closed_cases : bool
        Admit monomorphic constructor data with scalar fields and the
        linear call-address scopes of handler discharge.

    Raises
    ------
    JvpUnsupported
        If a reachable declaration, term or operation leaves the fragment.
    """
    declarations = {item.id: item for item in module.computations}
    row = owner.type.effects
    instance = row.entries[0].instance
    types: set[TypeExpr] = {REAL, BOOL, UNIT}
    case_constructors: set[ConstructorId] = set()
    if allow_closed_cases:
        scalar_constructors = {
            constructor.id
            for constructor in module.constructors
            if not constructor.telescope
            and all(field.type in types for field in constructor.fields)
        }
        for family in module.families:
            if (
                family.closed
                and not family.parameters
                and not family.indices
                and scalar_constructors.issuperset(family.constructors)
            ):
                types.add(TypeApplication(constructor=family.type_constructor))
                case_constructors.update(family.constructors)
    seen = {owner.id}
    pending: list[tuple[ComputationId, Computation]] = [(owner.id, owner.body)]
    edges: dict[ComputationId, set[ComputationId]] = {}
    while pending:
        caller, term = pending.pop()
        if isinstance(term, Bind):
            pending.extend(((caller, term.then), (caller, term.first)))
        elif isinstance(term, If):
            pending.extend(((caller, term.otherwise), (caller, term.then)))
        elif allow_closed_cases and isinstance(term, Case):
            if (
                term.motive.indices
                or term.motive.result_type not in types
                or any(
                    branch.static_arguments
                    or branch.constructor not in case_constructors
                    for branch in term.branches
                )
            ):
                raise JvpUnsupported(
                    f"{fragment} cases need a closed monomorphic family with scalar fields"
                )
            pending.extend((caller, branch.body) for branch in term.branches)
        elif allow_closed_cases and isinstance(term, WithTraceAddresses):
            if (
                term.previous is not None
                or term.resumptions
                or term.clauses
                or any(
                    address.local.type != _CALL_ADDRESS for address in term.addresses
                )
            ):
                raise JvpUnsupported(
                    f"{fragment} can only restore linear call addresses"
                )
            pending.append((caller, term.body))
        elif (
            allow_closed_cases
            and isinstance(term, CallAddress)
            and term.previous is None
        ):
            continue
        elif isinstance(term, Perform):
            check_request(term.request, instance)
        elif isinstance(term, Call):
            target = declarations[term.callee]
            if (
                target.telescope
                or target.type.effects not in (EMPTY_ROW, row)
                or target.type.result not in types
                or any(parameter.type not in types for parameter in target.parameters)
            ):
                raise JvpUnsupported(f"{fragment} cannot enter call `{term.name}`")
            if require_acyclic:
                edges.setdefault(caller, set()).add(target.id)
            if target.id not in seen:
                seen.add(target.id)
                pending.append((target.id, target.body))
        elif not isinstance(term, Return):
            raise JvpUnsupported(
                f"{type(term).__name__} is outside the {fragment} fragment"
            )
    if require_acyclic:
        try:
            TopologicalSorter(edges).prepare()
        except CycleError as error:
            raise JvpUnsupported(
                f"{fragment} requires an acyclic call graph"
            ) from error
