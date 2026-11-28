"""Trace the checked projection and grouping of typed lexical rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.core.algebras import LOG_PROB
from quivers.qiec._lexical_outcome_plan import (
    NORMALIZATION_ASSUMPTION,
    normalized_row_complexity,
)
from quivers.qiec.contracts import (
    WORD_RAM_EXPECTED_HASHING,
    AbsoluteLogError,
    AccuracyContract,
    Acyclic,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    DerivativeSemantics,
    InterpretationId,
    NoRetention,
    NormalizedWeightRows,
    Numerics,
    Outcome,
    Phase,
    QuerySpec,
    Target,
    Unresolved,
)
from quivers.qiec.coverage import constructor_skolems
from quivers.qiec.effects import EMPTY_ROW
from quivers.qiec.foreign import DeductionOutcomes, DeductionProofs
from quivers.qiec.module import _first_handled_call
from quivers.qiec.rules import RulePresentation
from quivers.qiec.terms import (
    Bind,
    Call,
    Case,
    ConstructorValue,
    Handle,
    Perform,
    Return,
    Var,
)
from quivers.qiec.types import BOOL
from quivers.stochastic.planning import Alternative, PlanRecord

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping
    from typing import Literal

    import torch

    from quivers.qiec.declarations import ConstructorDecl
    from quivers.qiec.effects import HandlerDef
    from quivers.qiec.evaluator import ClauseContext, RuntimeRequest
    from quivers.qiec.foreign import DeductionOutcomeGrouping
    from quivers.qiec.identifiers import ConstructorId
    from quivers.qiec.module import NamedComputation, QiecModule
    from quivers.qiec.types import TypeExpr


GROUPING_ALGORITHM = "hash-grouped-structural-keys"
"""The checked weighted list reducer's schedule."""

_RESOURCES: tuple[Literal["time"], Literal["space"]] = ("time", "space")
_GROUPING_CONDITIONS = (
    "key calls return exact discrete data; their own work and workspace are T and M",
    "expected constant hash-table overhead after charging word visits for key hashing "
    "and equal-key comparisons; adversarial collision work can reach O(N (N + S))",
    "fixed scalar number precision and constant work per scalar log-semiring operation",
    "the reverse pass covers only the live input weights of the grouping stage at fixed "
    "structural keys and support; it excludes the input chart and downstream program",
    "the bounds exclude optional trace inspection and serialization",
)
_WORK = ComplexityClaim(
    query="raw log mass per exact named key of weighted lexical item rows",
    model=WORD_RAM_EXPECTED_HASHING,
    parameters=tuple(
        ComplexityParameter(name=name, meaning=meaning)
        for name, meaning in (
            ("N", "input item rows processed by the checked mapper"),
            ("D", "distinct checked keys in the reducer"),
            ("S", "total words in the keys, counted with their structural occurrences"),
            ("T", "total work of the N calls to the checked named key"),
            ("M", "maximum live workspace of one key call apart from retained results"),
        )
    ),
    bounds=tuple(
        ComplexityBound(
            phase=phase,
            resource=resource,
            expression=expression,
            status=Unresolved(
                obligation="no matching lower bound for all key computations"
            ),
        )
        for phase, time, space in (
            (Phase.CONSTRUCTION, "N + T", "N + S + M"),
            (Phase.EXECUTION, "N + S", "N + S + D"),
            (Phase.DIFFERENTIATION, "N + D", "N + D"),
            (Phase.REFIT, "N + S + T + D", "N + S + M + D"),
        )
        for resource, expression in zip(_RESOURCES, (time, space), strict=True)
    ),
    conditions=(
        "the checked mapper calls the named key once on each item and forwards its live weight",
        *_GROUPING_CONDITIONS,
    ),
)
_PROOF_SPACE = {
    Phase.CONSTRUCTION: "P + N + S + M + C",
    Phase.EXECUTION: "P + N + S + D + C",
    Phase.DIFFERENTIATION: "N + D + C",
    Phase.REFIT: "P + N + S + M + D + C",
}
_PROOF_WORK = _WORK.with_(
    query="raw log mass per exact named key of weighted lexical proof rows",
    parameters=(
        ComplexityParameter(
            name="N", meaning="supplied proof rows processed by the mapper"
        ),
        *(parameter for parameter in _WORK.parameters if parameter.name in ("D", "S")),
        ComplexityParameter(
            name="T",
            meaning="total work of branch and key specialization and the N key calls",
        ),
        ComplexityParameter(
            name="M",
            meaning="largest live workspace of one branch and key call apart from retained results",
        ),
        ComplexityParameter(
            name="P",
            meaning="words in the supplied input proof values and their static indices",
        ),
        ComplexityParameter(
            name="C",
            meaning="words retained in the bounded branch and computation specialization caches",
        ),
    ),
    bounds=tuple(
        bound.with_(expression=_PROOF_SPACE[bound.phase])
        if bound.resource == "space"
        else bound
        for bound in _WORK.bounds
    ),
    conditions=(
        "the mapper opens each proof row's exact indices and calls the key on the unmodified proof",
        *_GROUPING_CONDITIONS,
        "T includes lookup, hashing and specialization of the checked branch and call at actual indices",
        "C includes retained static graphs and indices from this and earlier calls; "
        "live tensors and proof weights do not enter those specialization caches",
        "the forward space bound includes the proof inputs held during mapping; "
        "upstream proof generation and its differentiation tape have a separate plan",
        "the number of positive output rows is at most D, and D is at most N",
    ),
)


def _as_row(body: object, constructor: ConstructorId, fields: tuple[Var, ...]) -> bool:
    """Check a branch returns a row with these unmodified fields.

    Parameters
    ----------
    body
        The candidate checked branch body.
    constructor
        The exact row constructor identity.
    fields
        The original locals in their expected order.

    Returns
    -------
    bool
        Whether the body returns that ordinary nominal row.
    """
    return (
        isinstance(body, Return)
        and isinstance(body.value, ConstructorValue)
        and body.value.constructor == constructor
        and body.value.static_arguments == ()
        and body.value.fields == fields
    )


def _key_from_mapper(
    mapper: NamedComputation,
    chart: DeductionOutcomes,
    grouping: DeductionOutcomeGrouping,
    input_type: TypeExpr,
    proof_entry: ConstructorDecl | None,
) -> Call | None:
    """Identify the checked key call by the exact row conversion it performs.

    Parameters
    ----------
    mapper
        The computation called to convert lexical item or proof rows.
    chart, grouping
        The raw and keyed nominal row constructors.
    input_type
        The raw row type passed by the wrapper.
    proof_entry
        The checked existential proof row declaration, or None for item rows.

    Returns
    -------
    Call or None
        The key call made once per row, when its checked scope, fields,
        recursive call, and weight forwarding have the expected shape.
    """
    if (
        mapper.telescope
        or mapper.type.effects != EMPTY_ROW
        or len(mapper.parameters) != 1
        or mapper.parameters[0].type != input_type
        or not isinstance(case := mapper.body, Case)
        or case.scrutinee != Var(local=mapper.parameters[0])
        or len(case.branches) != 2
    ):
        return None
    empty, entry = case.branches
    indices = (
        () if proof_entry is None else constructor_skolems(proof_entry, entry.scope)
    )
    if (
        empty.constructor != chart.empty
        or empty.static_arguments
        or empty.fields
        or not _as_row(empty.body, grouping.empty, ())
        or entry.constructor != chart.entry
        or entry.static_arguments != indices
        or len(entry.fields) != 3
        or not isinstance(convert := entry.body, Bind)
        or not isinstance(key_call := convert.first, Call)
        or not isinstance(rest := convert.then, Bind)
        or not isinstance(tail_call := rest.first, Call)
    ):
        return None
    item, weight, tail = entry.fields
    if (
        key_call.static_arguments != indices
        or key_call.arguments != (Var(local=item),)
        or tail_call.callee != mapper.id
        or tail_call.static_arguments
        or tail_call.arguments != (Var(local=tail),)
        or not _as_row(
            rest.then,
            grouping.entry,
            (Var(local=convert.binder), Var(local=weight), Var(local=rest.binder)),
        )
    ):
        return None
    return key_call


def key_plan_trace(
    module: QiecModule,
    definition: HandlerDef,
    grouping: DeductionOutcomeGrouping,
    request: RuntimeRequest,
    context: ClauseContext,
    grouped: Mapping[Hashable, list[torch.Tensor]],
    normalized: bool,
) -> dict[str, object] | None:
    """Describe a completed native projection and grouping if recognized.

    Parameters
    ----------
    module
        The checked executable containing the handler and all named calls.
    definition, grouping
        The installed handler and its checked row representation.
    request
        The actual reducer request, with its complete checked term and site.
    context
        The native handler context, with the active nominal call.
    grouped
        Weights grouped by their exact structural keys in the successful run.
    normalized
        Whether the returned positive rows share a finite log normalizer.

    Returns
    -------
    dict of str to object or None
        The typed plan report, or no event for a standalone or different reducer.
    """
    from quivers.qiec.numbers import NumberDomain
    from quivers.qiec.serialization import dumps

    identity = context._enclosing_computation()
    by_id = {computation.id: computation for computation in module.computations}
    wrapper = by_id.get(identity) if identity is not None else None
    if (
        wrapper is None
        or wrapper.telescope
        or wrapper.type.effects != EMPTY_ROW
        or len(wrapper.parameters) != 2
        or not isinstance(body := wrapper.body, Bind)
        or not isinstance(project := body.first, Call)
        or not isinstance(handle := body.then, Handle)
        or not isinstance(perform := handle.computation, Perform)
        # An equal call instantiation may reuse a body loaded in another run.
        or perform.request != request.core
        or handle.handler != definition.id
        or handle.instance != request.instance
        or handle.arguments
        or handle.static_arguments
    ):
        return None
    rows, normalize = wrapper.parameters
    if (
        normalize.type != BOOL
        or project.static_arguments
        or project.arguments != (Var(local=rows),)
        or request.core.arguments != (Var(local=body.binder), Var(local=normalize))
    ):
        return None
    owners = tuple(
        candidate
        for candidate in module.handlers
        if candidate.output_type == rows.type
        and isinstance(candidate.foreign, DeductionOutcomes | DeductionProofs)
    )
    mapper = by_id.get(project.callee)
    if mapper is None or len(owners) != 1:
        return None
    (owner,) = owners
    descriptor = owner.foreign
    assert isinstance(descriptor, DeductionOutcomes | DeductionProofs)
    chart = descriptor.chart if isinstance(descriptor, DeductionProofs) else descriptor
    presentation = chart.presentation
    assert isinstance(presentation, RulePresentation)
    if presentation.conditions:
        return None
    proof_entry = None
    if isinstance(descriptor, DeductionProofs):
        proof_entry = next(
            (entry for entry in module.constructors if entry.id == chart.entry), None
        )
        if proof_entry is None:
            return None
    key_call = _key_from_mapper(mapper, chart, grouping, rows.type, proof_entry)
    if key_call is None:
        return None
    key_id = key_call.callee
    key = by_id.get(key_id)
    if (
        key is None
        or key.type.effects != EMPTY_ROW
        or key.type.result != grouping.key_type
        or _first_handled_call(key_id, by_id, set()) is not None
        or (
            proof_entry is None
            and (
                key.telescope
                or tuple(parameter.type for parameter in key.parameters)
                != (chart.item_type,)
            )
        )
    ):
        return None

    numerics = context.attachments.numbers
    inputs = {
        (weight.dtype, weight.device): weight
        for weights in grouped.values()
        for weight in weights
    }
    if len(inputs) > 1:
        return None
    if inputs:
        numerics = NumberDomain.of(next(iter(inputs.values())))
    retention = NoRetention()
    source = Outcome(key=key_id)
    mass = "normalized" if normalized else "raw"
    quantity = f"{mass} log mass per exact named key"
    if proof_entry is not None:
        quantity += " of supplied proof rows"
    work = _WORK if proof_entry is None else _PROOF_WORK
    key_assumptions = (
        (
            "the actual checked mapper applies this pure key to each supplied item row",
            "the key depends only on the item; item-weight aggregation can precede the map",
        )
        if proof_entry is None
        else (
            "each proof is passed to the pure key at the exact indices bound by its row",
            "the mapper forwards its live weight and can distinguish proofs of the same item",
        )
    )
    pooling = (
        Alternative(
            algorithm="individual proof enumeration before key application",
            reason="the key factors through the item, so finite semiring sums can be taken once per item",
        )
        if proof_entry is None
        else Alternative(
            algorithm="item-weight aggregation before key application",
            reason="requires the proof key to factor through its conclusion; the checked call does not supply that law",
        )
    )
    record = PlanRecord(
        query=QuerySpec(
            presentation=presentation.id,
            interpretation=InterpretationId.derive(presentation.id, "weight"),
            observable=NormalizedWeightRows(source=source) if normalized else source,
            input_types=(rows.type, normalize.type),
            retention=retention,
        ),
        recursion=Acyclic(),
        accuracy=AccuracyContract(
            numerics=Numerics(dtype=numerics.real, domain="log"),
            target=Target(quantity=quantity, metric=AbsoluteLogError()),
        ),
        retention=retention,
        algebra=LOG_PROB.subject,
        claims=LOG_PROB.claims(),
        algorithm=GROUPING_ALGORITHM,
        assumptions=(
            *key_assumptions,
            "checked structural equality compares the key's nominal constructors and ordered fields",
            (
                "the chart has its own plan and the grouping stage performs no rule joins"
                if proof_entry is None
                else "the grouping stage receives proofs as inputs; the upstream provider "
                "has its own structural row and optional node budget"
            ),
            "a key call can consume evaluation fuel; no general host compute or memory cap applies",
            *((NORMALIZATION_ASSUMPTION,) if normalized else ()),
        ),
        structure=presentation.analysis,
        complexity=normalized_row_complexity(work, quantity) if normalized else work,
        alternatives=(
            pooling,
            Alternative(
                algorithm="dense reduction over the complete key carrier",
                reason="requires an explicit finite enumeration of the key carrier, "
                "which this reducer's structural-equality interface does not supply",
            ),
        ),
        backward="autograd",
    )
    return {
        "handler": str(definition.id),
        "operation": str(request.operation),
        "site": str(request.core.origin.origin.site_id()),
        "presentation": str(presentation.id),
        "wrapper": str(wrapper.id),
        "mapper": str(mapper.id),
        "key": str(key.id),
        **(
            {
                "proof_row_owner": str(owner.id),
                "proof_row_entry": str(proof_entry.id),
                "key_site": str(key_call.origin.site_id()),
            }
            if proof_entry is not None
            else {}
        ),
        "key_type": dumps(grouping.key_type),
        "output_type": dumps(definition.output_type),
        "normalized": normalized,
        **({"positive_total_mass": True} if normalized else {}),
        "device": numerics.device,
        "temporary_provenance": "structural-key-table-and-weight-lists",
        "derivative": dumps(DerivativeSemantics()),
        "derivative_inputs": "finite input row log weights at fixed keys and support",
        "plan": dumps(record),
    }


__all__: list[str] = []
