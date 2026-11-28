"""Requirements of the fixed unguarded raw lexical item query."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from quivers.core.algebras import LOG_PROB
from quivers.qiec.canonical import tensor_type
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
    Numerics,
    Phase,
    QuerySpec,
    Target,
    Unresolved,
    Weight,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.rules import RulePresentation
from quivers.qiec.types import BOOL, REAL, STRING, IndexLiteral
from quivers.stochastic.planning import Alternative, PlanRecord

if TYPE_CHECKING:
    import torch

    from quivers.qiec.effects import HandlerDef
    from quivers.qiec.foreign import DeductionOutcomes
    from quivers.qiec.identifiers import OperationId


OUTCOME_ALGORITHM = "indexed-packed-forest-topological-fold"
"""The schedule run by the checked lexical outcome provider."""

_SETUP = "(m + 1)^3 P^2 + S (T + 1) + B^3 + W + n + J + V + E + R"
_CHART_SPACE = "(m + 1)^2 P + S + T + B^2 + W + n + I + V + E + R"
_FOLD = "V + A + S + K"
_REVERSE = "(A + 1) (W + 1) + V"
_PHASES = (
    (Phase.CONSTRUCTION, _SETUP, _CHART_SPACE),
    (Phase.EXECUTION, _FOLD, f"{_CHART_SPACE} + K"),
    (Phase.DIFFERENTIATION, _REVERSE, _REVERSE),
    (
        Phase.REFIT,
        f"{_SETUP} + {_FOLD} + {_REVERSE}",
        f"{_CHART_SPACE} + K + {_REVERSE}",
    ),
)

_RESOURCES: tuple[Literal["time"], Literal["space"]] = ("time", "space")

_WORK = ComplexityClaim(
    query="raw log mass of each distinct full-span item",
    model=WORD_RAM_EXPECTED_HASHING,
    parameters=tuple(
        ComplexityParameter(name=name, meaning=meaning)
        for name, meaning in (
            ("P", "presentation and index-pattern nodes and binders"),
            ("m", "maximum number of premise positions in a rule"),
            ("S", "constructor symbols of the closed index sort"),
            ("T", "nodes in all lexical category and form payloads"),
            ("B", "maximum bit width of integers used by the exact range check"),
            ("W", "scalar entries in the supplied log-weight tensor"),
            ("n", "length of the input sentence"),
            (
                "J",
                "word and term visits in lexical lookup, building indexes, matching "
                "candidates, instantiating conclusions, and hashing and comparing keys",
            ),
            ("I", "words held by chart indexes, grounded terms and bindings"),
            ("V", "ground items in the constructed forest"),
            ("E", "input or grounded rule edges in that forest"),
            ("R", "ordered premise references in that forest"),
            ("A", "edges and ordered premise references among goal ancestors"),
            (
                "K",
                "word and term visits to hash and compare output keys and reify rows",
            ),
        )
    ),
    bounds=tuple(
        ComplexityBound(
            phase=phase,
            resource=resource,
            expression=expression,
            status=Unresolved(obligation="no matching lower bound for this phase"),
        )
        for phase, time, space in _PHASES
        for resource, expression in zip(_RESOURCES, (time, space), strict=True)
    ),
    conditions=(
        "fixed dtype, constant work per scalar log semiring operation",
        "table access has expected constant overhead after charging key visits",
        "preparing keyed fallback trigger orders considers at most m + 1 "
        "triggers and m rounds of at most m patterns per rule",
        "lexical decoding checks private constructor membership and arity "
        "at every node, each with at most S names to scan",
        "the range check charges exact integer and rational work, including "
        "nullable proof size, with a conservative cubic bound in operand bits",
        "the fold scans the original item order, but evaluates only goal ancestors",
        "fixed support and a single reverse pass from the returned weights",
        "there are at most A live weight selections and each reverse from "
        "a tensor selection may allocate a dense W-entry adjoint",
        "one refit rebuilds the lexical chart and its live autograd graph",
        "the bounds concern provider work, including output reification, "
        "excluding optional trace metadata and the caller's program",
    ),
)


def raw_item_plan_trace(
    descriptor: DeductionOutcomes,
    definition: HandlerDef,
    operation: OperationId,
    tokens: int,
    terms: int,
    weights: torch.Tensor,
) -> dict[str, object]:
    """Describe the raw item operation that just returned its outcome rows.

    Parameters
    ----------
    descriptor
        The original checked unguarded lexical declaration.
    definition, operation
        The installed handler and its executed operation identity.
    tokens, terms
        The actual sentence and lexical payload extents.
    weights
        The actual unbatched log-weight input.

    Returns
    -------
    dict of str to object
        The checked plan record and the concrete request and result identities.
    """
    from quivers.qiec.numbers import NumberDomain
    from quivers.qiec.serialization import dumps

    presentation = descriptor.presentation
    assert isinstance(presentation, RulePresentation)
    numerics = NumberDomain.of(weights)
    retention = NoRetention()
    record = PlanRecord(
        query=QuerySpec(
            presentation=presentation.id,
            interpretation=InterpretationId.derive(presentation.id, "weight"),
            observable=Weight(),
            input_types=(
                tensor_type(STRING, (IndexLiteral(value=tokens, sort=NAT),)),
                tensor_type(
                    descriptor.item_type, (IndexLiteral(value=terms, sort=NAT),)
                ),
                tensor_type(REAL, (IndexLiteral(value=weights.numel(), sort=NAT),)),
                STRING,
                BOOL,
            ),
            retention=retention,
        ),
        recursion=Acyclic(),
        accuracy=AccuracyContract(
            numerics=Numerics(dtype=numerics.real, domain="log"),
            target=Target(
                quantity="raw log mass per positive full-span item",
                metric=AbsoluteLogError(),
            ),
        ),
        retention=retention,
        algebra=LOG_PROB.subject,
        claims=LOG_PROB.claims(),
        algorithm=OUTCOME_ALGORITHM,
        assumptions=(
            "finite contiguous span presentation with acyclic equal-width dependencies",
            "full item identity is the key; distinct lexical occurrence weights sum",
            "the declared span range check applies to each finite log weight; "
            "negative infinity denotes structural zero",
            "the result holds typed rows only; occurrence edges and ordered premise "
            "references are retained inside the provider until the fold returns",
            "the host fold has no compute or memory cap; QIEC step fuel does not "
            "charge each chart or reification operation",
        ),
        structure=presentation.analysis,
        complexity=_WORK,
        alternatives=(
            Alternative(
                algorithm="proof enumeration",
                reason="it preserves the finite mass but requires work in the "
                "unfolded proof sizes and an explicit proof budget",
            ),
            Alternative(
                algorithm="two-pass weight evaluation",
                reason="eligible acyclic weight runs may avoid keeping grounded "
                "edges; this operation uses the checked lexical forest schedule",
            ),
        ),
        backward="autograd",
        joins=descriptor._item_outcome_joins,
    )
    return {
        "handler": str(definition.id),
        "operation": str(operation),
        "presentation": str(presentation.id),
        "output_type": dumps(definition.output_type),
        "selector": "item",
        "normalized": False,
        "sentence_length": tokens,
        "start": descriptor.start,
        "device": numerics.device,
        "span_order": (
            ("equal_width_height", descriptor.span_order.height),
            ("maximum_arity", descriptor.span_order.maximum_arity),
            ("nullable", descriptor.span_order.nullable),
        ),
        "temporary_provenance": "packed-forest",
        "derivative": dumps(DerivativeSemantics()),
        "derivative_inputs": "ordered finite log weights at fixed support",
        "plan": dumps(record),
    }


__all__: list[str] = []
