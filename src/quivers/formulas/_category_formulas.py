"""Category-specific linear predictors of categorical and simplex families.

brms gives every non-reference category ``c`` of a categorical, multinomial,
Dirichlet, Dirichlet-multinomial or logistic-normal response its own
distributional parameter ``mu<c>`` (``muB``, ``muC`` for levels ``A``, ``B``,
``C`` with reference ``A``). The main formula predicts every ``mu<c>`` that
has no formula of its own, and an explicit ``mu<c> ~ ...`` replaces it for
that category. Each category then has its own coefficients, group effects and
priors (``dpar = "muB"``); the reference category's logit is zero.

The shared-coefficient emission (one coefficient vector over the
``K - 1`` free categories) is the same model when every category has the
same formula; per-category formulas are emitted as ordinary scalar
distributional predictors ``dpar_mu<c>_`` and stacked into the free logits.

The logistic-normal family also has one scale ``sigma<c>`` per free category:
the standard deviation of that category's log ratio to the reference. A
``sigma<c> ~ ...`` formula predicts it through the log link with its own
coefficients, and a category without a formula keeps one scalar scale.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
import re
from typing import TYPE_CHECKING, TypeVar

from quivers.dsl.ast_nodes import LetExprLiteral, LetExprNode, ProgramStep, Statement
from quivers.formulas._ast import _add, _factor, _index, _let, _var
from quivers.formulas.family import Family
from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    from quivers.formulas.compile import FormulaToQVRModule

_Value = TypeVar("_Value")


def category_parameter(label: str) -> str:
    """Name the location parameter of one response category.

    Parameters
    ----------
    label : str
        Category or component label.

    Returns
    -------
    str
        ``mu`` followed by the label with characters other than letters,
        digits and underscores removed, as brms forms ``mu<cat>``.

    Raises
    ------
    ValueError
        If the label has no letter, digit or underscore.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", str(label))
    if not cleaned:
        raise ValueError(f"category label {label!r} cannot name a parameter")
    return f"mu{cleaned}"


def is_category_parameter(family: Family, name: str) -> bool:
    """Decide whether a distributional name is a per-category location.

    Parameters
    ----------
    family : Family
        Response family.
    name : str
        Distributional parameter name.

    Returns
    -------
    bool
        True for ``mu<c>`` names of a categorical or simplex family.
    """
    return (
        family.response is not None
        and name.startswith("mu")
        and name != "mu"
        and name not in {aux.name for aux in family.aux_params}
    )


def scale_parameter(scale: str, label: str) -> str:
    """Name the scale parameter of one response category.

    Parameters
    ----------
    scale : str
        Name of the per-component scale parameter.
    label : str
        Category or component label.

    Returns
    -------
    str
        The scale name followed by the label with characters other than letters,
        digits and underscores removed, as brms forms ``sigma<cat>``.

    Raises
    ------
    ValueError
        If the label has no letter, digit or underscore.
    """
    return scale + category_parameter(label)[2:]


def is_scale_parameter(family: Family, name: str) -> bool:
    """Decide whether a distributional name is a per-category scale.

    Parameters
    ----------
    family : Family
        Response family.
    name : str
        Distributional parameter name.

    Returns
    -------
    bool
        True for a name extending that of the parameter a family declares
        `AuxParam.per_component`, as ``sigma<c>`` of the logistic-normal
        family.
    """
    scale = family.component_scale
    return scale is not None and name.startswith(scale.name) and name != scale.name


def free_scales(family: Family, labels: tuple[str, ...]) -> tuple[tuple[int, str], ...]:
    """List the free categories and their scale parameter names.

    Parameters
    ----------
    family : Family
        Logistic-normal family.
    labels : tuple[str, ...]
        Retained component labels in response order.

    Returns
    -------
    tuple[tuple[int, str], ...]
        Coordinate positions and ``sigma<c>`` names in retained order.
    """
    assert family.response is not None and family.component_scale is not None
    scale = family.component_scale.name
    return tuple(
        (index, scale_parameter(scale, labels[index]))
        for index in family.response.coordinate_indices(labels)
    )


def check_scale_formulas(
    family: Family, labels: tuple[str, ...], names: Collection[str]
) -> None:
    """Check per-category scale formulas against the free categories.

    Parameters
    ----------
    family : Family
        Response family.
    labels : tuple[str, ...]
        Retained component labels of the response.
    names : Collection[str]
        Names of the distributional and fixed parameters.

    Raises
    ------
    ValueError
        If a ``sigma<c>`` names no free category (the reference category has
        no scale), or if ``sigma`` is given beside a ``sigma<c>``.
    """
    requested = {name for name in names if is_scale_parameter(family, name)}
    if not requested or family.component_scale is None:
        return
    scale = family.component_scale.name
    known = {name for _, name in free_scales(family, labels)}
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"unknown scale parameters {sorted(unknown)}; the free categories "
            f"have {sorted(known)} (the reference category has no scale)"
        )
    if scale in names:
        raise ValueError(
            f"{family.name} takes either {scale!r}, which predicts every "
            f"category's scale, or per-category scales {sorted(known)}, not both"
        )


def free_categories(
    family: Family, labels: tuple[str, ...]
) -> tuple[tuple[int, str], ...]:
    """List the free categories and their location parameter names.

    Parameters
    ----------
    family : Family
        Categorical or simplex family.
    labels : tuple[str, ...]
        Retained category or component labels in response order.

    Returns
    -------
    tuple[tuple[int, str], ...]
        Coordinate positions and ``mu<c>`` names in retained order.
    """
    assert family.response is not None
    return tuple(
        (index, category_parameter(labels[index]))
        for index in family.response.coordinate_indices(labels)
    )


def category_distributional(
    family: Family,
    labels: tuple[str, ...],
    authored: str,
    distributional: Mapping[str, _Value] | None,
    *,
    every_category: bool = False,
) -> dict[str, _Value | str] | None:
    """Copy the main formula to every free category without a formula.

    Parameters
    ----------
    family : Family
        Response family.
    labels : tuple[str, ...]
        Retained category or component labels of the response.
    authored : str
        Authored main formula.
    distributional : Mapping or None
        Authored distributional formulas.
    every_category : bool
        Whether the main formula holds terms with scalar coefficients only
        (see :func:`requires_category_predictors`), so that every free
        category takes its own copy even without an authored ``mu<c>``.

    Returns
    -------
    dict or None
        The distributional formulas, completed with every ``mu<c>`` when any
        per-category formula was given or ``every_category`` holds; otherwise
        unchanged.

    Raises
    ------
    ValueError
        If a ``mu<c>`` names no free category (the reference category has no
        predictor).
    """
    if distributional is None and not every_category:
        return None
    result: dict[str, _Value | str] = dict(distributional or {})
    requested = {name for name in result if is_category_parameter(family, name)}
    if not requested and not every_category:
        return result
    names = {name for _, name in free_categories(family, labels)}
    unknown = requested - names
    if unknown:
        raise ValueError(
            f"unknown category parameters {sorted(unknown)}; the free categories "
            f"are {sorted(names)} (the reference category has no predictor)"
        )
    rhs = authored.split("~", 1)[1]
    for _, name in free_categories(family, labels):
        result.setdefault(name, f"{name} ~ {rhs}")
    return result


def requires_category_predictors(formula: Formula) -> bool:
    """Decide whether a predictor has terms with scalar coefficients only.

    brms gives each category its own monotonic simplexes, Gaussian process
    hyperparameters, multi-membership coefficients and ``gr()`` options, so a
    predictor holding one of these is emitted once per free category through
    the ordinary scalar predictor rather than with one coefficient vector over
    the categories.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.

    Returns
    -------
    bool
        True when the predictor has an ``mo()``, ``gp()`` or ``mm()`` term or a
        ``gr()`` option ``by``, ``dist`` or ``pw``.
    """
    return bool(
        formula.monotonic_terms
        or any(term.gp is not None for term in formula.design_terms)
        or formula.grouping.memberships
        or formula.grouping.options
    )


def component_valued_prefixes(
    family: Family, distributional: Mapping[str, _Value]
) -> frozenset[str]:
    """Name the predictors emitted with one coefficient per response category.

    Parameters
    ----------
    family : Family
        Response family.
    distributional : Mapping
        Distributional predictors by parameter name.

    Returns
    -------
    frozenset[str]
        Coefficient prefixes of the component-valued predictors: the location
        when no category has its own formula, and the logistic-normal scale
        predictor. Empty for a scalar response.
    """
    if family.response is None:
        return frozenset()
    prefixes = set()
    if not any(is_category_parameter(family, name) for name in distributional):
        prefixes.add("")
    scale = family.component_scale
    if scale is not None and scale.name in distributional:
        prefixes.add(f"dpar_{scale.name}_")
    return frozenset(prefixes)


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
        predictor = lens._distributional[name]
        eta = lens._append_predictor(
            predictor, statements, steps, prefix=f"dpar_{name}_", correlated=correlated
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
        predictor = lens._distributional.get(name)
        if name in lens._fixed_parameters:
            steps.append(
                _let(name, LetExprLiteral(value=float(lens._fixed_parameters[name])))
            )
            values.append(_add(_var("scale_row_zero"), _var(name)))
            continue
        if predictor is None:
            steps.append(lens._prior_draw(name, lens._user_priors.get(name, prior)))
            values.append(_add(_var("scale_row_zero"), _var(name)))
            continue
        eta = lens._append_predictor(
            predictor, statements, steps, prefix=f"dpar_{name}_", correlated=correlated
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


def main_without_terms(formula: Formula) -> Formula:
    """Strip the main formula's predictor terms when every category has its own.

    Parameters
    ----------
    formula : Formula
        Main formula.

    Returns
    -------
    Formula
        The response and modifiers without coefficients or group effects.
    """
    return formula.with_(
        fixed_columns=(),
        design_terms=(),
        random_terms=(),
        offsets=(),
        monotonic_terms=(),
    )


__all__ = [
    "append_category_predictors",
    "append_scale_predictors",
    "check_scale_formulas",
    "category_distributional",
    "category_parameter",
    "component_valued_prefixes",
    "free_categories",
    "free_scales",
    "is_category_parameter",
    "is_scale_parameter",
    "main_without_terms",
    "requires_category_predictors",
    "scale_parameter",
]
