"""Requirements of the fixed checked lexical item and proof queries."""

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
    Control,
    DerivativeSemantics,
    InterpretationId,
    NoRetention,
    NormalizedWeightRows,
    Numerics,
    Phase,
    QuerySpec,
    Rational,
    Target,
    Unresolved,
    Weight,
    WeightedProofRows,
)
from quivers.qiec.kinds import NAT
from quivers.qiec.rules import RulePresentation
from quivers.qiec.types import BOOL, INT, REAL, STRING, IndexLiteral, TypeApplication
from quivers.stochastic.planning import Alternative, PlanRecord

if TYPE_CHECKING:
    import torch

    from quivers.qiec.effects import HandlerDef
    from quivers.qiec.foreign import DeductionOutcomes, DeductionProofs
    from quivers.qiec.identifiers import OperationId


OUTCOME_ALGORITHM = "indexed-packed-forest-topological-fold"
"""The schedule run by the checked lexical outcome provider."""

PROOF_ALGORITHM = "indexed-packed-forest-budgeted-enumeration"
"""The schedule run by the checked lexical proof provider."""

NORMALIZATION_ASSUMPTION = (
    "the common log normalizer is finite, so its mathematical mass is positive; "
    "a zero or nonfinite total is refused"
)
"""The runtime gate every successful normalized-row report has passed."""

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

_GROUNDING_PARAMETERS = tuple(
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
    )
)

_GROUNDING_CONDITIONS = (
    "fixed dtype, constant work per scalar log semiring operation",
    "table access has expected constant overhead after charging key visits",
    "preparing keyed fallback trigger orders considers at most m + 1 "
    "triggers and m rounds of at most m patterns per rule",
    "lexical decoding checks private constructor membership and arity "
    "at every node, each with at most S names to scan",
    "the range check charges exact integer and rational work, including "
    "nullable proof size, with a conservative cubic bound in operand bits",
)

_WORK = ComplexityClaim(
    query="raw log mass of each distinct full-span item",
    model=WORD_RAM_EXPECTED_HASHING,
    parameters=(
        *_GROUNDING_PARAMETERS,
        ComplexityParameter(
            name="K",
            meaning="word and term visits to hash and compare output keys and reify rows",
        ),
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
        *_GROUNDING_CONDITIONS,
        "the fold scans the original item order, but evaluates only goal ancestors",
        "fixed support and a single reverse pass from the returned weights",
        "there are at most A live weight selections and each reverse from "
        "a tensor selection may allocate a dense W-entry adjoint",
        "one refit rebuilds the lexical chart and its live autograd graph",
        "the bounds concern provider work, including output reification, "
        "excluding optional trace metadata and the caller's program",
    ),
)


def normalized_row_complexity(work: ComplexityClaim, quantity: str) -> ComplexityClaim:
    """Add one finite log normalizer, its broadcast subtraction, and their reverse.

    Parameters
    ----------
    work
        The executed raw row fold's time and space claims.
    quantity
        The quantity to which the additional normalization applies.

    Returns
    -------
    ComplexityClaim
        The same construction bound, with one linear pass in the positive
        row count added to work and live storage for the other phases.
    """
    return work.with_(
        query=quantity,
        parameters=(
            *work.parameters,
            ComplexityParameter(
                name="H", meaning="positive rows sharing one log normalizer"
            ),
        ),
        bounds=tuple(
            bound
            if bound.phase == Phase.CONSTRUCTION
            else bound.with_(expression=f"{bound.expression} + H")
            for bound in work.bounds
        ),
        conditions=(
            *work.conditions,
            "stacking, summing, checking, subtracting and differentiating the common "
            "log normalizer each visit at most H positive rows",
        ),
    )


def lexical_item_plan_trace(
    descriptor: DeductionOutcomes,
    definition: HandlerDef,
    operation: OperationId,
    tokens: int,
    terms: int,
    weights: torch.Tensor,
    normalized: bool,
) -> dict[str, object]:
    """Describe the checked item operation that just returned its outcome rows.

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
    normalized
        Whether the returned positive rows share a finite log normalizer.

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
    source = Weight()
    quantity = (
        "normalized log mass per positive full-span item"
        if normalized
        else "raw log mass per positive full-span item"
    )
    record = PlanRecord(
        query=QuerySpec(
            presentation=presentation.id,
            interpretation=InterpretationId.derive(presentation.id, "weight"),
            observable=NormalizedWeightRows(source=source) if normalized else source,
            input_types=(
                *_lexical_input_types(descriptor, tokens, terms, weights),
                STRING,
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
            *((NORMALIZATION_ASSUMPTION,) if normalized else ()),
        ),
        structure=presentation.analysis,
        complexity=normalized_row_complexity(_WORK, quantity) if normalized else _WORK,
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
        "normalized": normalized,
        **({"positive_total_mass": True} if normalized else {}),
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


_PROOF_EXECUTION = "P + V + A C^2 + (m + 1) Q + U + G"
_PROOF_SPACE = f"{_CHART_SPACE} + V C + (m + 1) Q + U + G"
_PROOF_REVERSE = f"Q + {_REVERSE}"
_PROOF_PHASES = (
    (Phase.CONSTRUCTION, _SETUP, _CHART_SPACE),
    (Phase.EXECUTION, _PROOF_EXECUTION, _PROOF_SPACE),
    (Phase.DIFFERENTIATION, _PROOF_REVERSE, _PROOF_REVERSE),
    (
        Phase.REFIT,
        f"{_SETUP} + {_PROOF_EXECUTION} + {_PROOF_REVERSE}",
        f"{_PROOF_SPACE} + {_PROOF_REVERSE}",
    ),
)
_PROOF_WORK = ComplexityClaim(
    query="positive-weight structural proof rows of all requested full-span items",
    model=WORD_RAM_EXPECTED_HASHING,
    parameters=(
        *_GROUNDING_PARAMETERS,
        ComplexityParameter(
            name="C",
            meaning="maximum bit width of proof caps and sums of saturated root counts",
        ),
        ComplexityParameter(
            name="Q",
            meaning="unfolded structural proof nodes across all full-span root derivations",
        ),
        ComplexityParameter(name="U", meaning="words in the materialized typed rows"),
        ComplexityParameter(
            name="G",
            meaning="word and index visits in proof-value lifting, certification and reification",
        ),
    ),
    bounds=tuple(
        ComplexityBound(
            phase=phase,
            resource=resource,
            expression=expression,
            status=Unresolved(obligation="no matching lower bound for this phase"),
        )
        for phase, time, space in _PROOF_PHASES
        for resource, expression in zip(_RESOURCES, (time, space), strict=True)
    ),
    conditions=(
        *_GROUNDING_CONDITIONS,
        "saturated counts per goal ancestor, computed "
        "with constant many C-bit operations per edge or premise reference",
        "a successful preflight makes each useful subderivation extend to a "
        "distinct node position in a requested complete proof for some fixed "
        "enclosing context of that item",
        "ordered Cartesian prefixes can copy their premises up to m times; "
        "the derivation list is linear in Q at fixed maximum arity",
        "only positive-weight rows reach typed constructor reification; "
        "the structural cap check precedes proof enumeration and edge weight assignment",
        "fixed support and a single reverse pass from the returned weights",
        "indexed lexical reads may each produce a dense W-entry adjoint",
        "one refit regrounds, rechecks the caps, reifies proof values, and rebuilds "
        "the autograd tape",
        "the bounds exclude optional trace metadata and the caller's program",
    ),
)


def _lexical_input_types(
    descriptor: DeductionOutcomes, tokens: int, terms: int, weights: torch.Tensor
) -> tuple[TypeApplication, ...]:
    """Give the canonical types at the actual extents of one checked request.

    Parameters
    ----------
    descriptor
        The checked lexical declaration.
    tokens, terms
        The actual sentence and lexical payload extents.
    weights
        The actual unbatched log-weight input.

    Returns
    -------
    tuple of TypeApplication
        The sentence, lexical values, and weights tensor types, in that order.
    """
    return (
        tensor_type(STRING, (IndexLiteral(value=tokens, sort=NAT),)),
        tensor_type(descriptor.item_type, (IndexLiteral(value=terms, sort=NAT),)),
        tensor_type(REAL, (IndexLiteral(value=weights.numel(), sort=NAT),)),
    )


def proof_rows_plan_trace(
    descriptor: DeductionProofs,
    definition: HandlerDef,
    operation: OperationId,
    tokens: int,
    terms: int,
    weights: torch.Tensor,
    max_proofs: int,
    max_nodes: int | None,
    normalized: bool,
) -> dict[str, object]:
    """Describe the checked lexical proof enumeration that produced the rows.

    Parameters
    ----------
    descriptor
        The original checked lexical and proof declarations.
    definition, operation
        The installed handler and its executed operation identity.
    tokens, terms
        The actual sentence and lexical payload extents.
    weights
        The actual unbatched log-weight input.
    max_proofs, max_nodes
        The structural preflight caps. A missing node cap permits any
        unfolded size at the declared row count.
    normalized
        Whether the returned weights share a positive normalizer.

    Returns
    -------
    dict of str to object
        The checked plan record and the concrete request and result identities.
    """
    from quivers.qiec.numbers import NumberDomain
    from quivers.qiec.serialization import dumps

    chart = descriptor.chart
    presentation = chart.presentation
    assert isinstance(presentation, RulePresentation)
    numerics = NumberDomain.of(weights)
    retention = NoRetention()
    controls = (Control(name="max_proofs", value=Rational(numerator=max_proofs)),)
    if max_nodes is not None:
        controls += (Control(name="proof_nodes", value=Rational(numerator=max_nodes)),)
    record = PlanRecord(
        query=QuerySpec(
            presentation=presentation.id,
            interpretation=InterpretationId.derive(presentation.id, "weight"),
            observable=WeightedProofRows(normalized=normalized),
            input_types=(
                *_lexical_input_types(chart, tokens, terms, weights),
                INT,
                BOOL,
            ),
            retention=retention,
        ),
        recursion=Acyclic(),
        accuracy=AccuracyContract(
            numerics=Numerics(dtype=numerics.real, domain="log"),
            target=Target(
                quantity=(
                    "normalized log mass per positive full-span derivation"
                    if normalized
                    else "raw log mass per positive full-span derivation"
                ),
                metric=AbsoluteLogError(),
            ),
            controls=controls,
        ),
        retention=retention,
        algebra=LOG_PROB.subject,
        claims=LOG_PROB.claims(),
        algorithm=PROOF_ALGORITHM,
        assumptions=(
            "finite contiguous span presentation with acyclic equal-width dependencies",
            "typed proof rows preserve each occurrence and the original premise order",
            "caps count structural root derivations and unfolded nodes across distinct "
            "full-span goals before any weight-dependent omission",
            "exceeding either cap refuses the entire request before proof enumeration",
            "normalization requires positive total mass",
            "the host fold has no general compute or memory budget; its controls cap "
            "the structural proof list and, when requested, its unfolded nodes",
        ),
        structure=presentation.analysis,
        complexity=_PROOF_WORK,
        alternatives=(
            Alternative(
                algorithm=OUTCOME_ALGORITHM,
                reason="one total per item merges the distinct derivations required "
                "by the observable",
            ),
            Alternative(
                algorithm="return the packed forest",
                reason="the forest supports later unfolding, while this request "
                "returns the checked weighted proof values themselves",
            ),
        ),
        backward="autograd",
        joins=chart._item_outcome_joins,
    )
    return {
        "handler": str(definition.id),
        "operation": str(operation),
        "presentation": str(presentation.id),
        "proof_presentation": str(descriptor.proof_presentation.id),
        "output_type": dumps(definition.output_type),
        "normalized": normalized,
        "sentence_length": tokens,
        "start": chart.start,
        "device": numerics.device,
        "span_order": (
            ("equal_width_height", chart.span_order.height),
            ("maximum_arity", chart.span_order.maximum_arity),
            ("nullable", chart.span_order.nullable),
        ),
        "temporary_provenance": "packed-forest-and-shared-derivations",
        "derivative": dumps(DerivativeSemantics()),
        "derivative_inputs": "ordered finite log weights at fixed support",
        "plan": dumps(record),
    }


__all__: list[str] = []
