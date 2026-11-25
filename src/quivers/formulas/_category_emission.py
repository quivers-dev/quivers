"""Emit the per-category predictors of categorical and simplex families.

A model with ``mu<c>`` or ``sigma<c>`` formulas has one ordinary scalar
predictor per free category, stacked along the axis of the free coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from quivers.dsl.ast_nodes import (
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
    Statement,
)
from quivers.formulas._ast import _add, _factor, _index, _let, _var
from quivers.formulas._category_formulas import (
    check_scale_formulas,
    free_categories,
    free_scales,
    is_category_parameter,
    is_scale_parameter,
)
from quivers.formulas._term_emission import ProgramEmission, emit_predictor
from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    from quivers.formulas.compile import FormulaToQVRModule

__all__: list[str] = []


def append_category_predictors(
    lens: FormulaToQVRModule,
    formula: Formula,
    statements: list[Statement],
    steps: list[ProgramStep],
    correlated: Mapping[tuple[str, str, str], LetExprNode],
) -> bool:
    """Emit one ordinary predictor per free category and stack the free logits.

    Each ``mu<c>`` is emitted with the ``dpar_mu<c>_`` prefix and bound as
    ``mu<c>``; ``eta_nonbaseline`` is their ``(Resp, K - 1)`` stack. Scoring
    costs O(n sum_c p_c) for ``p_c`` columns of category ``c``.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Lens with the completed per-category formulas.
    formula : Formula
        Main formula with the response.
    statements : list[Statement]
        Module declarations.
    steps : list[ProgramStep]
        Program steps receiving the predictors.
    correlated : Mapping
        Correlated group coefficients from the shared correlation pass.

    Returns
    -------
    bool
        True when per-category predictors were emitted, False when the
        shared-coefficient emission applies.

    Raises
    ------
    ValueError
        If the family has an explicit all-category parameterization or a
        nonlinear location.
    """
    family = lens._family
    if family.response is None or not any(
        is_category_parameter(family, name) for name in lens._distributional
    ):
        return False
    labels = lens._component_labels(formula)
    reference = family.response.reference_index(labels)
    if reference is None:
        raise ValueError(
            "category-specific formulas require a reference category (refcat)"
        )
    if lens._predictor_graph is not None and lens._predictor_graph.roots:
        raise ValueError("category-specific formulas require linear predictors")
    values = []
    categories = free_categories(family, labels)
    missing = [name for _, name in categories if name not in lens._distributional]
    if missing:
        raise ValueError(
            f"category-specific formulas require every free category: missing {missing}"
        )
    for _, name in categories:
        eta = emit_predictor(
            lens._predictor(f"dpar_{name}_"),
            ProgramEmission(lens, statements, correlated),
            steps,
        )
        steps.append(_let(f"dpar_{name}_eta", eta))
        steps.append(_let(name, _var(f"dpar_{name}_eta")))
        values.append(_var(f"dpar_{name}_eta"))
    steps.append(
        _let(
            "category_row_zero",
            _factor((("response", "Resp"),), body=LetExprLiteral(value=0.0)),
        )
    )
    steps.append(
        _let(
            "eta_by_category",
            _factor(
                (("category_logit", "FormulaCategoryLogit"),),
                cases=tuple(
                    (index, _add(_var("category_row_zero"), value))
                    for index, value in enumerate(values)
                ),
            ),
        )
    )
    steps.append(
        _let(
            "eta_nonbaseline",
            _factor(
                (("response", "Resp"), ("category_logit", "FormulaCategoryLogit")),
                body=_index(
                    "eta_by_category", _var("category_logit"), _var("response")
                ),
            ),
        )
    )
    lens._append_component_response(
        steps,
        parameter="mu",
        prefix="",
        reference=reference,
        coordinates=tuple(index for index, _ in categories),
        category_index="FormulaCategory",
    )
    return True


def append_scale_predictors(
    lens: FormulaToQVRModule,
    formula: Formula,
    statements: list[Statement],
    steps: list[ProgramStep],
    correlated: Mapping[tuple[str, str, str], LetExprNode],
    prior: str,
    link: Callable[[list[ProgramStep], str, LetExprNode], None],
) -> bool:
    """Emit the per-category logistic-normal scales and stack them by row.

    A category with a ``sigma<c>`` formula gets an ordinary scalar predictor
    with the ``dpar_sigma<c>_`` prefix under the scale link, a category with
    a fixed ``sigma<c>`` gets that constant, and any other category gets a
    scalar positive site ``sigma<c>``. ``sigma`` is their
    ``(Resp, K - 1)`` stack, the value the row-varying scale of the
    logistic-normal law reads. Scoring costs O(n sum_c p_c) for ``p_c``
    columns of category ``c``.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Lens with the authored per-category scale formulas.
    formula : Formula
        Main formula with the response.
    steps : list[ProgramStep]
        Program steps receiving the scales.
    correlated : Mapping
        Correlated group coefficients from the shared correlation pass.
    prior : str
        Default prior of a scalar category scale.
    link : Callable
        Appends the steps binding a named scale to the inverse link of its
        linear predictor.

    Returns
    -------
    bool
        True when per-category scales were emitted, False when no
        ``sigma<c>`` formula or fixed value was given.

    Raises
    ------
    ValueError
        If a ``sigma<c>`` names no free category or ``sigma`` is also given.
    """
    family = lens._family
    scale = family.component_scale
    if scale is None or not any(
        is_scale_parameter(family, name)
        for name in lens._distributional.keys() | lens._fixed_parameters.keys()
    ):
        return False
    labels = lens._component_labels(formula)
    check_scale_formulas(
        family, labels, lens._distributional.keys() | lens._fixed_parameters.keys()
    )
    steps.append(
        _let(
            "scale_row_zero",
            _factor((("response", "Resp"),), body=LetExprLiteral(value=0.0)),
        )
    )
    values: list[LetExprNode] = []
    for _, name in free_scales(family, labels):
        if name in lens._fixed_parameters:
            steps.append(
                _let(name, LetExprLiteral(value=float(lens._fixed_parameters[name])))
            )
            values.append(_add(_var("scale_row_zero"), _var(name)))
            continue
        if name not in lens._distributional:
            steps.append(lens._prior_draw(name, lens._user_priors.get(name, prior)))
            values.append(_add(_var("scale_row_zero"), _var(name)))
            continue
        eta = emit_predictor(
            lens._predictor(f"dpar_{name}_"),
            ProgramEmission(lens, statements, correlated),
            steps,
        )
        steps.append(_let(f"dpar_{name}_eta", eta))
        link(steps, name, _var(f"dpar_{name}_eta"))
        values.append(_add(_var("scale_row_zero"), _var(name)))
    steps.append(
        _let(
            f"{scale.name}_by_category",
            _factor(
                (("category_logit", "FormulaCategoryLogit"),),
                cases=tuple(enumerate(values)),
            ),
        )
    )
    steps.append(
        _let(
            scale.name,
            _factor(
                (("response", "Resp"), ("category_logit", "FormulaCategoryLogit")),
                body=_index(
                    f"{scale.name}_by_category",
                    _var("category_logit"),
                    _var("response"),
                ),
            ),
        )
    )
    return True
