"""Fold Boolean proof meanings over an acyclic packed lexical forest."""

from __future__ import annotations

from itertools import product

import torch

from quivers.core.algebras import LOG_PROB
from quivers.qiec._boolean_outcome_plan import BOOLEAN_ALGORITHM, boolean_plan_trace
from quivers.qiec.deduction_outcomes import (
    _lexical_forest,
    _materialize_rows,
    _outcome_masses,
    _packed_trace,
)
from quivers.qiec.effects import HandlerDef
from quivers.qiec.evaluator import (
    ClauseContext,
    InvalidHandlerError,
    Resumption,
    RuntimeClause,
    RuntimeConstructor,
    RuntimeHandler,
    RuntimeRequest,
    TailResume,
)
from quivers.qiec.foreign import DeductionBooleanOutcomes
from quivers.qiec.identifiers import OperationId
from quivers.qiec.numbers import decide
from quivers.qiec.rules import Item
from quivers.stochastic.forest import edge_weight


def deduction_boolean_outcomes_handler(
    definition: HandlerDef, operation: OperationId, semantics: DeductionBooleanOutcomes
) -> RuntimeHandler:
    """Provide the packed interpretation's checked query.

    Parameters
    ----------
    definition, operation
        The typed handler and its single linear operation.
    semantics
        The lexical presentation and ordered Boolean table layout.

    Returns
    -------
    RuntimeHandler
        The native handler, with no proof enumeration path.
    """

    def query(
        request: RuntimeRequest, _resume: Resumption, context: ClauseContext
    ) -> TailResume:
        """Ground once, then fold one pair of log masses per useful item.

        Parameters
        ----------
        request
            Sentence, lexical terms, live log weights, truth table and mode.
        _resume
            The continuation supplied by the checked handler.
        context
            The run's number domain and optional trace observer.

        Returns
        -------
        TailResume
            The positive-mass Boolean rows in false-then-true order.

        Raises
        ------
        InvalidHandlerError
            If the request fails its carrier or finite acyclic contract.
        EvaluationError
            If probability normalization or numerical evaluation fails.
        """
        tokens, terms, weights, table, normalize = request.arguments
        if not (
            isinstance(tokens, tuple)
            and isinstance(terms, tuple)
            and isinstance(weights, torch.Tensor)
            and isinstance(table, torch.Tensor)
            and table.shape == (semantics.table_size,)
            and table.dtype == torch.bool
        ):
            raise InvalidHandlerError("malformed packed Boolean deduction query")

        truths = tuple(decide(entry, "Boolean proof truth table") for entry in table)
        end = len(semantics.lexical_words)
        lexical = dict(zip(semantics.lexical_words, truths[:end], strict=True))
        forest, inputs, goals = _lexical_forest(semantics.chart, tokens, terms, weights)
        rule_tables = {}
        for rule in forest.presentation.rules:
            next_end = end + (1 << rule.arity)
            rule_tables[rule.id] = truths[end:next_end]
            end = next_end

        useful = forest.below_all(goals)
        order = useful.order
        if order is None:
            raise InvalidHandlerError(
                "packed Boolean proof outcomes require an acyclic forest"
            )
        zero = LOG_PROB.zero_like((), weights)
        masses: dict[Item, tuple[torch.Tensor, torch.Tensor]] = {}
        lanes = products = reduced_entries = 0
        for item in order:
            false: list[torch.Tensor] = []
            true: list[torch.Tensor] = []
            for edge in useful.edges[item]:
                weight = edge_weight(forest, edge, LOG_PROB, inputs, like=weights)
                if edge.occurrence is not None:
                    (word,) = edge.values
                    assert isinstance(word, str)
                    (true if lexical[word] else false).append(weight)
                    lanes += 1
                    continue
                choices = product((0, 1), repeat=len(edge.premises))
                for outcome, selection in zip(
                    rule_tables[edge.constructor], choices, strict=True
                ):
                    contribution = weight
                    for premise, branch in zip(edge.premises, selection, strict=True):
                        contribution = LOG_PROB.tensor_op(
                            contribution, masses[premise][branch]
                        )
                        products += 1
                    (true if outcome else false).append(contribution)
                    lanes += 1
            reduced_entries += len(false) + len(true)
            masses[item] = (
                LOG_PROB.join(torch.stack(false), 0) if false else zero,
                LOG_PROB.join(torch.stack(true), 0) if true else zero,
            )

        normalized = decide(normalize, "Boolean proof outcome normalization")
        rows = _outcome_masses(
            {
                outcome: [masses[goal][index] for goal in goals]
                for index, outcome in enumerate((False, True))
                if goals
            },
            normalized,
        )
        context.emit_trace(
            "deduction.boolean_outcomes",
            lambda: {
                **_packed_trace(forest, normalized, len(goals), len(rows)),
                "algorithm": BOOLEAN_ALGORITHM,
                "observer": "weighted-boolean-proof-meanings",
                "evaluated_items": len(order),
                "truth_table_entries": len(truths),
                "edge_meaning_lanes": lanes,
                "weight_products": products,
                "reduced_entries": reduced_entries,
                "derivative": "autograd-on-finite-log-weights-at-fixed-truth-tables",
                "unmeasured": (
                    "truth_table_operation_cost",
                    "backward_arithmetic",
                    "elements_saved_for_backward",
                    "peak_bytes",
                ),
            },
        )
        result = _materialize_rows(
            definition.output_type,
            semantics.chart.empty,
            semantics.chart.entry,
            (
                (context.attachments.numbers.number("truth", outcome), mass)
                for outcome, mass in rows.items()
            ),
        )
        context.emit_trace(
            "deduction.boolean_outcomes.plan",
            lambda: boolean_plan_trace(
                semantics,
                definition,
                operation,
                len(tokens),
                len(terms),
                weights,
                table,
                lexical,
                rule_tables,
                normalized,
                {
                    "evaluated_items": len(order),
                    "edge_meaning_lanes": lanes,
                    "weight_products": products,
                    "reduced_entries": reduced_entries,
                },
            ),
        )
        return TailResume(value=result)

    return RuntimeHandler(
        definition=definition,
        clauses={
            operation: RuntimeClause(
                invoke=query,
                result_validator=lambda result: (
                    isinstance(result, RuntimeConstructor)
                    and result.result_type == definition.output_type
                ),
            )
        },
        duplicable_context=True,
    )


__all__: list[str] = []
