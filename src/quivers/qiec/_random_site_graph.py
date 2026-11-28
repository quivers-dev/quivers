"""Check the first-order source scope shared by the exact site interpreters."""

from __future__ import annotations

from collections.abc import Callable

from quivers.qiec.builtins import RANDOM
from quivers.qiec.checking import KernelRegistry
from quivers.qiec.differentiation import JvpUnsupported
from quivers.qiec.effects import EMPTY_ROW, EffectRequest
from quivers.qiec.identifiers import ComputationId, EffectInstanceId
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.terms import Bind, Call, Computation, If, Perform, Return
from quivers.qiec.types import BOOL, REAL, UNIT

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

    Raises
    ------
    JvpUnsupported
        If a reachable declaration, term or operation leaves the fragment.
    """
    declarations = {item.id: item for item in module.computations}
    row = owner.type.effects
    instance = row.entries[0].instance
    seen = {owner.id}
    pending: list[Computation] = [owner.body]
    while pending:
        term = pending.pop()
        if isinstance(term, Bind):
            pending.extend((term.then, term.first))
        elif isinstance(term, If):
            pending.extend((term.otherwise, term.then))
        elif isinstance(term, Perform):
            check_request(term.request, instance)
        elif isinstance(term, Call):
            target = declarations[term.callee]
            if (
                target.telescope
                or target.type.effects not in (EMPTY_ROW, row)
                or target.type.result not in (REAL, BOOL, UNIT)
                or any(
                    parameter.type not in (REAL, BOOL, UNIT)
                    for parameter in target.parameters
                )
            ):
                raise JvpUnsupported(f"{fragment} cannot enter call `{term.name}`")
            if target.id not in seen:
                seen.add(target.id)
                pending.append(target.body)
        elif not isinstance(term, Return):
            raise JvpUnsupported(
                f"{type(term).__name__} is outside the {fragment} fragment"
            )
