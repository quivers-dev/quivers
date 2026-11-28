"""Describe the two-lane inside fold for one supplied Boolean table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from quivers.core.algebras import LOG_PROB
from quivers.qiec._lexical_outcome_plan import (
    NORMALIZATION_ASSUMPTION,
    _CHART_SPACE,
    _GROUNDING_CONDITIONS,
    _GROUNDING_PARAMETERS,
    _REVERSE,
    _SETUP,
    _lexical_input_types,
    normalized_row_complexity,
)
from quivers.qiec.canonical import tensor_type
from quivers.qiec.contracts import (
    WORD_RAM_EXPECTED_HASHING,
    AbsoluteLogError,
    AccuracyContract,
    Acyclic,
    BooleanProofWeightRows,
    ComplexityBound,
    ComplexityClaim,
    ComplexityParameter,
    DerivativeSemantics,
    InterpretationId,
    NoRetention,
    Numerics,
    Phase,
    QuerySpec,
    Target,
    Unresolved,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.rules import RulePresentation
from quivers.qiec.types import BOOL, IndexLiteral
from quivers.stochastic.planning import Alternative, PlanRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch

    from quivers.qiec.effects import HandlerDef
    from quivers.qiec.foreign import DeductionBooleanOutcomes
    from quivers.qiec.identifiers import ConstructorId, OperationId


BOOLEAN_ALGORITHM = "indexed-packed-forest-boolean-fold"
_CONSTRUCT = f"{_SETUP} + F"
_EXECUTE = "V + A + K + L + M"
_SPACE = f"{_CHART_SPACE} + F + L + M"
_BACKWARD = f"{_REVERSE} + L + M"
_RESOURCES: tuple[Literal["time"], Literal["space"]] = ("time", "space")
_WORK = ComplexityClaim(
    query="raw log mass of each positive Boolean proof meaning",
    model=WORD_RAM_EXPECTED_HASHING,
    parameters=(
        *_GROUNDING_PARAMETERS,
        *(
            ComplexityParameter(name=name, meaning=meaning)
            for name, meaning in (
                ("F", "entries of the supplied lexical and ordered rule truth tables"),
                (
                    "K",
                    "word and term visits to hash and compare items in the goal subgraph and fold",
                ),
                (
                    "L",
                    "contribution lanes across all useful axiom edges and ordered rule inputs",
                ),
                ("M", "premise weight products over all useful rule lanes"),
            )
        ),
    ),
    bounds=tuple(
        ComplexityBound(
            phase=phase,
            resource=resource,
            expression=expression,
            status=Unresolved(
                obligation="no matching lower bound for the complete phase"
            ),
        )
        for phase, time, space in (
            (Phase.CONSTRUCTION, _CONSTRUCT, f"{_CHART_SPACE} + F"),
            (Phase.EXECUTION, _EXECUTE, _SPACE),
            (Phase.DIFFERENTIATION, _BACKWARD, _BACKWARD),
            (
                Phase.REFIT,
                f"{_CONSTRUCT} + {_EXECUTE} + {_BACKWARD}",
                f"{_SPACE} + {_BACKWARD}",
            ),
        )
        for resource, expression in zip(_RESOURCES, (time, space), strict=True)
    ),
    conditions=(
        *_GROUNDING_CONDITIONS,
        "the provider converts and slices the supplied table in F entries; "
        "the caller's truth-table operation evaluation and device scheduling are excluded",
        "the fold keeps two log masses per useful item and visits each goal edge "
        "with its premises in their original order",
        "a useful axiom contributes one lane and a useful rule of arity k contributes "
        "2^k lanes with k products each; k is at most two, so L <= 4 E and M <= 8 E",
        "the reverse pass charges scalar products and sums and allows a dense "
        "W-entry adjoint for each useful input weight selection",
        "the truth table and support stay fixed during that reverse pass",
        "a refit rebuilds the chart and the live weight graph; optional trace "
        "inspection and serialization are excluded",
    ),
)


def boolean_plan_trace(
    descriptor: DeductionBooleanOutcomes,
    definition: HandlerDef,
    operation: OperationId,
    tokens: int,
    terms: int,
    weights: torch.Tensor,
    table: torch.Tensor,
    lexical: Mapping[str, bool],
    rules: Mapping[ConstructorId, tuple[bool, ...]],
    normalized: bool,
    work: Mapping[str, int],
) -> dict[str, object]:
    """Report the identity, table and counts of a completed native fold.

    Parameters
    ----------
    descriptor
        The actual checked chart and its ordered Boolean table layout.
    definition, operation
        The installed handler and its executed operation identity.
    tokens, terms
        The actual sentence and lexical payload extents.
    weights, table
        The live log weights and the evaluated truth table supplied to the provider.
    lexical, rules
        The exact lexical entries and rule slices used to select meaning lanes.
    normalized
        Whether the output rows share a finite log normalizer.
    work
        The measured item, lane, product and reduction counts.

    Returns
    -------
    dict of str to object
        The typed plan, immutable truth table and measured execution counters.
    """
    from quivers.qiec.numbers import NumberDomain
    from quivers.qiec.serialization import dumps

    chart = descriptor.chart
    presentation = chart.presentation
    assert isinstance(presentation, RulePresentation)
    numerics = NumberDomain.of(weights)
    mode = "normalized" if normalized else "raw"
    quantity = f"{mode} log mass of each positive Boolean proof meaning"
    retention = NoRetention()
    record = PlanRecord(
        query=QuerySpec(
            presentation=presentation.id,
            interpretation=InterpretationId.derive(
                presentation.id, "boolean-proof-table", operation
            ),
            observable=BooleanProofWeightRows(normalized=normalized),
            input_types=(
                *_lexical_input_types(chart, tokens, terms, weights),
                tensor_type(BOOL, (IndexLiteral(value=table.numel(), sort=NAT),)),
                BOOL,
            ),
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
        algorithm=BOOLEAN_ALGORITHM,
        assumptions=(
            "finite contiguous spans with acyclic equal-width dependencies",
            "rules are unconditional and have at most two premises and no value fields",
            "the supplied total Boolean tables select one output lane per ordered input",
            "finite distributivity aggregates disjoint proof choices within each meaning lane",
            "negative infinity denotes structural zero and the span range check applies to finite log weights",
            "the result carries only weighted Boolean rows; the packed forest and lane sums are temporary",
            "the host fold exposes no compute or memory cap",
            *((NORMALIZATION_ASSUMPTION,) if normalized else ()),
        ),
        structure=presentation.analysis,
        complexity=normalized_row_complexity(_WORK, quantity) if normalized else _WORK,
        alternatives=(
            Alternative(
                algorithm="budgeted proof enumeration followed by a named key",
                reason="preserves proof choices under a row cap but unfolds and interprets each proof",
            ),
            Alternative(
                algorithm="one scalar weight per item",
                reason="loses the premise truth values required by these tables unless a further factorization is proved",
            ),
        ),
        backward="autograd",
        joins=chart._item_outcome_joins,
    )
    return {
        "handler": str(definition.id),
        "operation": str(operation),
        "presentation": str(presentation.id),
        "output_type": dumps(definition.output_type),
        "normalized": normalized,
        **({"positive_total_mass": True} if normalized else {}),
        "sentence_length": tokens,
        "start": chart.start,
        "device": numerics.device,
        "truth_table_device": str(table.device),
        "truth_table_entries": table.numel(),
        "truth_table": dumps((tuple(lexical.items()), tuple(rules.items()))),
        "temporary_provenance": "packed-forest-and-boolean-lanes",
        "derivative": dumps(DerivativeSemantics()),
        "derivative_inputs": "ordered finite log weights at fixed truth tables and support",
        **work,
        "plan": dumps(record),
    }


__all__: list[str] = []
