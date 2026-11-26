"""brms autocorrelation terms: parsing, validation, binding, and prediction.

A formula such as ``y ~ x + ar(time = t, gr = g, p = 2)`` keeps its ordinary
predictor and replaces the Gaussian or Student response with one
`ResidualResponse` law over all rows. The term's coefficients enter the
existing auxiliary-parameter path, so priors, `get_prior`, fitting, replay,
and artifacts use the shared formula machinery. Supported terms are
``ar``, ``ma``, ``arma`` (with ``cov``), ``cosy``, ``unstr``, ``sar``,
``fcor``, and ``car``. A conditional ARMA term combines with ``sar`` or
``fcor``, the one pair of residual terms brms admits; ``se(..., sigma =
TRUE)`` combines with a time-series covariance; ``cens()``, ``trunc()``, and
weights combine with the conditional ARMA formulation.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
import io
import tokenize
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
from formulae.matrices import DesignMatrices
import narwhals as nw
from narwhals.typing import IntoDataFrame
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution, Independent

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._custom_types import CustomFamilySpec
from quivers.formulas._residual_dependence import (
    ResidualKind,
    ResidualLayout,
    ResidualResponse,
    TruncatedResidualRows,
    sar_weights,
)
from quivers.dsl.ast_nodes import ProgramStep, Statement
from quivers.formulas._latent_effects import (
    LatentTerm,
    latent_car_term,
)
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas._response_roles import bind_roles
from quivers.formulas.family import AuxParam, Family, links
from quivers.formulas.formula import Formula, _formula_and_design, _qvr_name
from quivers.inference._posterior import collect_site_draws
from quivers.inference._response import response_distribution
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas.compile import FormulaToQVRModule
    from quivers.formulas.formula import Formula, ResponseModifiers
    from quivers.inference.trace import Trace

type AutocorMatrix = np.ndarray | pd.DataFrame
"""A ``data2`` matrix: SAR weights or an FCOR covariance over response rows."""

AUTOCOR_CALLS = frozenset({"ar", "ma", "arma", "cosy", "unstr", "sar", "car", "fcor"})
"""brms autocorrelation term constructors recognized in a formula."""

_SIGNATURES: Mapping[str, tuple[str, ...]] = {
    "arma": ("time", "gr", "p", "q", "cov"),
    "ar": ("time", "gr", "p", "cov"),
    "ma": ("time", "gr", "q", "cov"),
    "cosy": ("time", "gr"),
    "unstr": ("time", "gr"),
    "sar": ("M", "type"),
    "car": ("M", "gr", "type"),
    "fcor": ("M",),
}

_R_CONSTANTS: Mapping[str, bool | None] = {
    "TRUE": True,
    "FALSE": False,
    "T": True,
    "F": False,
    "True": True,
    "False": False,
    "NA": None,
}


class AutocorTerm(ValueModel):
    """One parsed brms autocorrelation term.

    Parameters
    ----------
    kind : {"arma", "cosy", "unstr", "sar", "car", "fcor"}
        Term class; ``ar`` and ``ma`` are ARMA terms with one zero order.
    time, group : str or None
        Time-ordering and grouping columns; None uses data order and one series.
    p, q : int
        Autoregressive and moving-average orders.
    cov : bool
        Covariance formulation (brms ``cov = TRUE``).
    sar_type : {"lag", "error"} or None
        Simultaneous autoregression type.
    car_type : {"escar", "esicar", "icar", "bym2"} or None
        Conditional autoregression type.
    matrix : str or None
        Name of the ``data2`` matrix for ``sar``, ``car``, and ``fcor``.
    label : str
        Authored call text.
    """

    kind: Literal["arma", "cosy", "unstr", "sar", "car", "fcor"]
    time: str | None = None
    group: str | None = None
    p: int = 0
    q: int = 0
    cov: bool = False
    sar_type: Literal["lag", "error"] | None = None
    car_type: Literal["escar", "esicar", "icar", "bym2"] | None = None
    matrix: str | None = None
    label: str = ""

    @property
    def dimension(self) -> Literal["time", "space", "fixed"]:
        """Name brms's dimension of this term.

        Returns
        -------
        {"time", "space", "fixed"}
            Time series, spatial, or fixed-covariance term.
        """
        if self.kind in {"arma", "cosy", "unstr"}:
            return "time"
        return "space" if self.kind in {"sar", "car"} else "fixed"

    @property
    def residual_kind(self) -> ResidualKind:
        """Name the `ResidualLayout` kind scoring this term.

        Returns
        -------
        ResidualKind
            Residual layout kind.

        Raises
        ------
        ValueError
            For a CAR term, which is not a residual structure.
        """
        if self.kind == "sar":
            return "sar_lag" if self.sar_type == "lag" else "sar_error"
        if self.kind == "car":
            raise ValueError("a CAR term is a latent effect, not a residual structure")
        return self.kind


def _call_text(rhs: str, start: int) -> int:
    """Find the closing parenthesis of a call whose ``(`` is at ``start``.

    Parameters
    ----------
    rhs : str
        Formula text.
    start : int
        Index of the opening parenthesis.

    Returns
    -------
    int
        Index one past the matching closing parenthesis.

    Raises
    ------
    ValueError
        If the parentheses do not balance.
    """
    depth, quote = 0, None
    for position in range(start, len(rhs)):
        char = rhs[position]
        if quote is not None:
            quote = None if char == quote else quote
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if not depth:
                return position + 1
    raise ValueError("unbalanced parentheses in an autocorrelation term")


def _argument(node: ast.expr, name: str, label: str) -> str | int | bool | None:
    """Read one literal or bare-name argument of an autocorrelation call.

    Parameters
    ----------
    node : ast.expr
        Parsed argument.
    name : str
        Formal parameter name.
    label : str
        Authored call, for errors.

    Returns
    -------
    str or int or bool or None
        Variable or matrix name, order, flag, or type string.

    Raises
    ------
    ValueError
        If the argument is neither a name nor a literal.
    """
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return _argument(ast.Constant(value=-node.operand.value), name, label)
    if isinstance(node, ast.Name):
        if node.id in _R_CONSTANTS:
            return _R_CONSTANTS[node.id]
        return node.id
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, int, bool, float)
    ):
        value = node.value
        if isinstance(value, float) and value.is_integer() and name in {"p", "q"}:
            return int(value)
        if isinstance(value, float):
            raise ValueError(f"{label}: argument {name!r} must be a whole number")
        return value
    raise ValueError(
        f"{label}: argument {name!r} must be a column name or a literal value"
    )


def parse_autocor_call(text: str) -> AutocorTerm:
    """Parse one brms autocorrelation call with brms's argument validation.

    Parameters
    ----------
    text : str
        Call text such as ``arma(time = t, gr = g, p = 1, q = 1)``.

    Returns
    -------
    AutocorTerm
        The validated term.

    Raises
    ------
    ValueError
        If arguments are unknown, duplicated, of the wrong kind, or violate
        brms's order and covariance restrictions.
    """
    try:
        call = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as error:
        raise ValueError(f"invalid autocorrelation term {text!r}") from error
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"invalid autocorrelation term {text!r}")
    callee = call.func.id
    label = text.strip()
    signature = _SIGNATURES[callee]
    if len(call.args) > len(signature):
        raise ValueError(f"{label}: too many positional arguments")
    values: dict[str, str | int | bool | None] = {}
    for name, node in zip(signature, call.args, strict=False):
        values[name] = _argument(node, name, label)
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg not in signature:
            raise ValueError(f"{label}: unknown argument {keyword.arg!r}")
        if keyword.arg in values:
            raise ValueError(f"{label}: argument {keyword.arg!r} given twice")
        values[keyword.arg] = _argument(keyword.value, keyword.arg, label)

    def text_value(name: str, *, required: bool = False) -> str | None:
        value = values.get(name)
        if value is None:
            if required:
                raise ValueError(f"{label}: argument {name!r} is missing")
            return None
        if not isinstance(value, str) or isinstance(value, bool):
            raise ValueError(f"{label}: argument {name!r} must name one variable")
        return value

    def order(name: str, default: int) -> int:
        value = values.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "Autoregressive order must be a non-negative integer."
                if name == "p"
                else "Moving-average order must be a non-negative integer."
            )
        return value

    if callee in {"arma", "ar", "ma"}:
        p = order("p", 1) if callee != "ma" else 0
        q = order("q", 1) if callee != "ar" else 0
        if not p + q:
            raise ValueError("At least one of 'p' and 'q' should be greater zero.")
        cov = values.get("cov", False)
        if not isinstance(cov, bool):
            raise ValueError(f"{label}: argument 'cov' must be TRUE or FALSE")
        if cov and (p > 1 or q > 1):
            raise ValueError(
                "Covariance formulation of ARMA structures is only possible for "
                "effects of maximal order one."
            )
        return AutocorTerm(
            kind="arma",
            time=text_value("time"),
            group=text_value("gr"),
            p=p,
            q=q,
            cov=cov,
            label=label,
        )
    if callee in {"cosy", "unstr"}:
        required = callee == "unstr"
        return AutocorTerm(
            kind="cosy" if callee == "cosy" else "unstr",
            time=text_value("time", required=required),
            group=text_value("gr", required=required),
            cov=True,
            label=label,
        )
    matrix = text_value("M")
    if matrix is None:
        raise ValueError(f"Argument 'M' is missing in {callee}().")
    if callee == "sar":
        kind = values.get("type", "lag")
        if kind not in {"lag", "error"}:
            raise ValueError(f"{label}: type must be 'lag' or 'error'")
        return AutocorTerm(
            kind="sar",
            matrix=matrix,
            sar_type="lag" if kind == "lag" else "error",
            cov=True,
            label=label,
        )
    if callee == "car":
        kind = values.get("type", "escar")
        car_types: Mapping[str, Literal["escar", "esicar", "icar", "bym2"]] = {
            "escar": "escar",
            "esicar": "esicar",
            "icar": "icar",
            "bym2": "bym2",
        }
        if not isinstance(kind, str) or kind not in car_types:
            raise ValueError(
                f"{label}: type must be 'escar', 'esicar', 'icar', or 'bym2'"
            )
        return AutocorTerm(
            kind="car",
            matrix=matrix,
            group=text_value("gr"),
            car_type=car_types[kind],
            label=label,
        )
    return AutocorTerm(kind="fcor", matrix=matrix, cov=True, label=label)


def split_autocor_terms(formula: str) -> tuple[str, tuple[AutocorTerm, ...]]:
    """Remove standalone autocorrelation terms from a formula's right side.

    Parameters
    ----------
    formula : str
        Authored formula.

    Returns
    -------
    tuple[str, tuple[AutocorTerm, ...]]
        The formula without autocorrelation terms (``1`` when none remain)
        and the parsed terms in authored order.

    Raises
    ------
    ValueError
        If a term is not a standalone additive term, or the combination
        violates brms's one-term-per-class rules.
    """
    if not any(name in formula for name in AUTOCOR_CALLS):
        return formula, ()
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(formula).readline)
        if token.type not in {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}
    ]
    tilde = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.type == tokenize.OP and token.string == "~"
        ),
        None,
    )
    if tilde is None:
        return formula, ()
    offset = len(formula.splitlines(keepends=True)[0])
    if "\n" in formula:
        raise ValueError("autocorrelation formulas must be written on one line")
    del offset
    calls: list[tuple[int, int]] = []
    depth = 0
    for index, token in enumerate(tokens):
        if token.type == tokenize.OP and token.string in "([":
            depth += 1
        elif token.type == tokenize.OP and token.string in ")]":
            depth -= 1
        if (
            index > tilde
            and token.type == tokenize.NAME
            and token.string in AUTOCOR_CALLS
            and index + 1 < len(tokens)
            and tokens[index + 1].string == "("
        ):
            previous = tokens[index - 1]
            end = _call_text(formula, tokens[index + 1].start[1])
            following = next((item for item in tokens if item.start[1] >= end), None)
            standalone = (
                depth == 0
                and (index - 1 == tilde or previous.string == "+")
                and (following is None or following.string == "+")
            )
            if not standalone:
                raise ValueError(
                    f"autocorrelation term {token.string}(...) must be a standalone "
                    "additive term of the response formula"
                )
            calls.append((token.start[1], end))
    if not calls:
        return formula, ()
    terms = tuple(parse_autocor_call(formula[start:end]) for start, end in calls)
    kinds = [term.kind for term in terms]
    if len(set(kinds)) != len(kinds):
        raise ValueError("Can only model one term per autocorrelation class.")
    if sum(term.dimension == "time" for term in terms) > 1:
        raise ValueError("Can only model one time-series term.")
    if sum(term.dimension == "space" for term in terms) > 1:
        raise ValueError("Can only model one spatial term.")
    if sum(term.cov for term in terms) > 1:
        raise ValueError("Can only model one covariance matrix of natural residuals.")
    stripped = formula
    for start, end in reversed(calls):
        left = stripped[:start].rstrip()
        right = stripped[end:].lstrip()
        if left.endswith("+"):
            left = left[:-1].rstrip()
        elif right.startswith("+"):
            right = right[1:].lstrip()
        stripped = f"{left} {right}".strip()
    lhs, _, rhs = stripped.partition("~")
    if not rhs.strip():
        stripped = f"{lhs.rstrip()} ~ 1"
    return stripped, terms


def _residual_record(
    name: str, coefficients: Sequence[str], *, student: bool, unstructured: bool
) -> DistributionFamily:
    """Declare the checked signature of one residual response law.

    Parameters
    ----------
    name : str
        Catalog family name.
    coefficients : Sequence[str]
        Scalar structural parameter names.
    student : bool
        Include Student degrees of freedom.
    unstructured : bool
        Include the Cholesky correlation factor over time levels.

    Returns
    -------
    DistributionFamily
        Scalar-event family with one batch coordinate per response row.
    """
    parameters = [
        FamilyParameter(name="loc", constraint="real", rank=0),
        FamilyParameter(name="scale", constraint="positive", rank=0),
    ]
    if student:
        parameters.append(FamilyParameter(name="nu", constraint="positive", rank=0))
    parameters.extend(
        FamilyParameter(name=coefficient, constraint="real", rank=0)
        for coefficient in coefficients
    )
    if unstructured:
        parameters.append(
            FamilyParameter(
                name="factor",
                constraint="lower_cholesky",
                rank=2,
                scalar_broadcast=False,
            )
        )
    return DistributionFamily(
        name=name,
        parameters=tuple(parameters),
        support="real",
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=False,
        compositional=False,
    )


class ResidualFactory(ValueModel):
    """Build the residual response law of one retained layout.

    Parameters
    ----------
    layout : ResidualLayout
        Training or prediction row layout.
    coefficients : tuple[str, ...]
        Scalar structural parameter names.
    """

    layout: ResidualLayout = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    coefficients: tuple[str, ...] = ()

    def __call__(self, parameters: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct the law at natural-scale parameter values.

        Parameters
        ----------
        parameters : Mapping[str, torch.Tensor]
            ``loc``, ``scale``, optional ``nu``, structural coefficients, and
            an optional ``factor``.

        Returns
        -------
        Distribution
            A `ResidualResponse` over the layout's rows.
        """
        return ResidualResponse(
            self.layout,
            parameters["loc"],
            parameters["scale"],
            {name: parameters[name] for name in self.coefficients},
            parameters.get("nu"),
            parameters.get("factor"),
        )


class AutocorBinding(ValueModel):
    """Retained autocorrelation configuration of one fitted response.

    Parameters
    ----------
    term : AutocorTerm
        Parsed term that selects the residual law: the covariance, SAR, or
        fixed-covariance term, or the conditional ARMA term alone.
    arma : AutocorTerm or None
        Conditional ARMA term combined with a SAR or fixed-covariance
        ``term``.
    se : str or int or float or None
        Authored source of known standard errors added to a time-series
        covariance, read again on prediction rows.
    layout : ResidualLayout
        Training row layout.
    record : DistributionFamily
        Checked residual-law signature.
    coefficients : tuple[str, ...]
        Scalar structural parameter names.
    bounds : tuple[tuple[str, float | None, float | None], ...]
        Admissible interval of each structural parameter.
    time_levels : tuple[str, ...]
        Unstructured correlations only: training time levels.
    response : str
        Response column, read as prediction history when present.
    """

    term: AutocorTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    arma: AutocorTerm | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    se: str | int | float | None = None
    layout: ResidualLayout = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    record: DistributionFamily = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    coefficients: tuple[str, ...] = ()
    bounds: tuple[tuple[str, float | None, float | None], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    time_levels: tuple[str, ...] = ()
    response: str = ""


_BINDINGS: dict[int, tuple[FamilyCatalog, AutocorBinding]] = {}


def autocor_binding(family: Family) -> AutocorBinding | None:
    """Return the autocorrelation binding of a family, if it has one.

    Parameters
    ----------
    family : Family
        Fitted or prepared response family.

    Returns
    -------
    AutocorBinding or None
        The retained binding, or None for an ordinary family.
    """
    if family.custom is None:
        return None
    entry = _BINDINGS.get(id(family.custom.catalog))
    return entry[1] if entry is not None and entry[0] is family.custom.catalog else None


def _catalog(binding: AutocorBinding, layout: ResidualLayout) -> FamilyCatalog:
    """Register a residual law over one layout in a fresh catalog.

    Parameters
    ----------
    binding : AutocorBinding
        Retained signature and coefficient names.
    layout : ResidualLayout
        Rows the law covers.

    Returns
    -------
    FamilyCatalog
        Compilation-local catalog.
    """
    return FamilyCatalog().with_family(
        binding.record,
        ResidualFactory(layout=layout, coefficients=binding.coefficients),
    )


def _time_level(value: float | int | str) -> str:
    """Canonicalize one time value so integer and float spellings agree.

    Parameters
    ----------
    value : float or int or str
        Time value.

    Returns
    -------
    str
        ``repr`` of the float value for numbers, the text otherwise.
    """
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return repr(float(value))
    return str(value)


def _series_layout(
    term: AutocorTerm,
    frame: pd.DataFrame,
    *,
    student: bool,
    history: np.ndarray | None,
    time_levels: tuple[str, ...] | None,
    caller: str,
) -> tuple[ResidualLayout, tuple[str, ...]]:
    """Sort rows by group and time and code each series as brms does.

    Parameters
    ----------
    term : AutocorTerm
        Time-series term.
    frame : pandas.DataFrame
        Included rows in original order.
    student : bool
        Student innovations.
    history : np.ndarray or None
        Response history for the conditional ARMA predictor.
    time_levels : tuple[str, ...] or None
        Retained unstructured time levels, or None to derive them.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[ResidualLayout, tuple[str, ...]]
        Layout and the unstructured time levels.

    Raises
    ------
    ValueError
        If a time or group column is absent or missing values, time points
        repeat within a group, or prediction rows use new UNSTR time points.
    """
    rows = len(frame)
    for role, column in (("time", term.time), ("gr", term.group)):
        if column is not None:
            if column not in frame.columns:
                raise ValueError(
                    f"{caller}: {term.label} names {role} column {column!r}, which is absent"
                )
            if bool(frame[column].isna().to_numpy().any()):
                raise ValueError(
                    f"{caller}: {term.label} {role} column {column!r} has missing values"
                )
    group = (
        frame[term.group].to_numpy()
        if term.group is not None
        else np.zeros(rows, dtype=np.int64)
    )
    time = (
        frame[term.time].to_numpy()
        if term.time is not None
        else np.arange(rows, dtype=np.int64)
    )
    keys = pd.DataFrame({"group": group, "time": time})
    if keys.duplicated().any():
        raise ValueError("Time points within groups must be unique.")
    order = tuple(
        int(index)
        for index in keys.sort_values(
            ["group", "time"], kind="mergesort"
        ).index.to_numpy()
    )
    codes = pd.factorize(keys["group"].to_numpy()[list(order)])[0]
    starts = tuple(int(i) for i in np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]))
    lengths = tuple(int(b - a) for a, b in zip(starts, (*starts[1:], rows)))
    levels: tuple[str, ...] = ()
    time_index: tuple[int, ...] = ()
    if term.kind == "unstr":
        labels = [_time_level(value) for value in keys["time"]]
        derived = tuple(
            _time_level(value) for value in sorted(keys["time"].unique().tolist())
        )
        levels = derived if time_levels is None else time_levels
        if set(derived) - set(levels):
            raise ValueError("Cannot handle new time points in UNSTR models.")
        position = {level: index for index, level in enumerate(levels)}
        time_index = tuple(position[labels[row]] for row in order)
    layout = ResidualLayout(
        kind=term.residual_kind,
        rows=rows,
        order=order,
        starts=starts,
        lengths=lengths,
        time_index=time_index,
        times=len(levels),
        p=term.p,
        q=term.q,
        stationary=term.cov,
        student=student,
        history=history if term.kind == "arma" and not term.cov else None,
    )
    return layout, levels


def _data_matrix(
    term: AutocorTerm, data2: Mapping[str, AutocorMatrix] | None, rows: int, caller: str
) -> np.ndarray:
    """Read and check the ``data2`` matrix named by a SAR or FCOR term.

    Parameters
    ----------
    term : AutocorTerm
        Term naming the matrix.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    rows : int
        Included response rows.
    caller : str
        Entry point named in errors.

    Returns
    -------
    np.ndarray
        Dense float matrix of shape ``(rows, rows)``; an FCOR vector becomes a
        diagonal matrix.

    Raises
    ------
    ValueError
        If the matrix is missing, nonnumeric, or has the wrong dimensions, or
        an FCOR matrix is not symmetric positive definite.
    """
    assert term.matrix is not None
    if data2 is None or term.matrix not in data2:
        raise ValueError(f"{caller}: {term.label} requires data2[{term.matrix!r}]")
    value = data2[term.matrix]
    matrix = np.asarray(
        value.to_numpy() if isinstance(value, pd.DataFrame) else value,
        dtype=np.float64,
    )
    if term.kind == "fcor" and matrix.ndim == 1:
        matrix = np.diag(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"'M' for {term.kind.upper()} terms must be a matrix.")
    if matrix.shape != (rows, rows):
        raise ValueError(
            f"Dimensions of 'M' for {term.kind.upper()} terms must be equal to "
            "the number of observations."
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"'M' for {term.kind.upper()} terms must be finite.")
    if term.kind == "fcor":
        if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12 * np.abs(matrix).max()):
            raise ValueError("'M' for FCOR terms must be symmetric.")
        if np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError("'M' for FCOR terms must be positive definite.")
    return matrix


def _known_variance(
    frame: pd.DataFrame, source: str | int | float | None
) -> np.ndarray | None:
    """Read squared known standard errors of the rows of one frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Training or prediction rows.
    source : str or int or float or None
        Authored ``se()`` source, or None when the formula has none.

    Returns
    -------
    np.ndarray or None
        One squared standard error per row, or None.

    Raises
    ------
    ValueError
        If the standard errors are absent, nonnumeric, or negative.
    """
    if source is None:
        return None
    _, bound = bind_roles(frame, {"standard_error": source, "se_sigma": True})
    error = bound.get("standard_error")
    if error is None:
        raise ValueError("se() bound no standard errors")
    return np.square(np.asarray(error.column.data, dtype=np.float64))


def _structure_layout(
    term: AutocorTerm,
    arma: AutocorTerm | None,
    frame: pd.DataFrame,
    *,
    student: bool,
    history: np.ndarray | None,
    time_levels: tuple[str, ...] | None,
    data2: Mapping[str, AutocorMatrix] | None,
    known_variance: np.ndarray | None,
    caller: str,
) -> tuple[ResidualLayout, tuple[str, ...]]:
    """Code the rows of one frame for a residual term or admitted pair.

    Parameters
    ----------
    term : AutocorTerm
        Term selecting the residual law.
    arma : AutocorTerm or None
        Conditional ARMA term combined with a SAR or fixed covariance.
    frame : pandas.DataFrame
        Included rows in original order, with a fresh range index.
    student : bool
        Student innovations.
    history : np.ndarray or None
        Response history of a conditional ARMA recursion.
    time_levels : tuple[str, ...] or None
        Retained unstructured time levels, or None to derive them.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices over the rows of ``frame``.
    known_variance : np.ndarray or None
        Squared known standard errors of a time-series covariance.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[ResidualLayout, tuple[str, ...]]
        Layout and the unstructured time levels.

    Raises
    ------
    ValueError
        If the rows violate a term's time, group, or matrix contract.
    """
    rows = len(frame)
    if term.dimension == "time":
        layout, levels = _series_layout(
            term,
            frame,
            student=student,
            history=history,
            time_levels=time_levels,
            caller=caller,
        )
        if known_variance is not None:
            layout = layout.with_(known_variance=known_variance)
        return layout, levels
    matrix = _data_matrix(term, data2, rows, caller)
    order, starts, lengths = tuple(range(rows)), (0,), (rows,)
    if term.kind == "fcor":
        layout = ResidualLayout(
            kind="fcor",
            rows=rows,
            order=order,
            starts=starts,
            lengths=lengths,
            student=student,
            matrix=matrix,
            matrix_factor=np.linalg.cholesky(matrix),
        )
    else:
        weights, eigenvalues = sar_weights(matrix)
        layout = ResidualLayout(
            kind=term.residual_kind,
            rows=rows,
            order=order,
            starts=starts,
            lengths=lengths,
            student=student,
            weights=weights,
            eigenvalues=eigenvalues,
        )
    if arma is not None:
        series, _ = _series_layout(
            arma, frame, student=student, history=None, time_levels=None, caller=caller
        )
        layout = layout.with_(
            p=arma.p,
            q=arma.q,
            mean_order=series.order,
            mean_starts=series.starts,
            mean_lengths=series.lengths,
            history=history,
        )
    return layout, ()


def _addition_rules(
    term: AutocorTerm,
    arma: AutocorTerm | None,
    modifiers: ResponseModifiers,
    caller: str,
) -> bool:
    """Apply brms's rules for response terms beside residual structures.

    brms scores a time-series covariance and a fixed covariance with one
    multivariate density, which admits no weights, censoring, or truncation
    (``stan_log_lik_gaussian_time`` and ``_fcor`` in ``R/stan-likelihood.R``),
    and adds ``se()`` to a time-series covariance only, with ``sigma =
    TRUE`` (``no_sigma`` in ``R/families.R``); the conditional ARMA
    formulation refuses ``se()`` (``stan_ac`` in ``R/stan-predictor.R``) and
    admits the other terms row by row. For a SAR term brms writes these
    terms into a Stan program that does not compile, because
    ``normal_lagsar_lpdf`` and its siblings take the whole response vector
    and one scalar ``sigma``, so no brms model exists to match.

    Parameters
    ----------
    term : AutocorTerm
        Term selecting the residual law.
    arma : AutocorTerm or None
        Conditional ARMA term combined with it.
    modifiers : ResponseModifiers
        Response-side roles of the formula.
    caller : str
        Entry point named in errors.

    Returns
    -------
    bool
        Whether known standard errors enter a time-series covariance.

    Raises
    ------
    ValueError
        With brms's message for each combination brms refuses, and a located
        message for the SAR combinations brms cannot compile.
    """
    roles = dict(modifiers.sources)
    conditional = arma is not None or (term.kind == "arma" and not term.cov)
    adjusted = [
        name
        for name, present in (
            ("weights()", "weights" in roles),
            ("cens()", modifiers.censoring is not None),
            ("trunc()", modifiers.truncation is not None),
        )
        if present
    ]
    error = modifiers.standard_error
    if term.kind == "sar" and (adjusted or error is not None):
        present = ", ".join([*adjusted, *(["se()"] if error is not None else [])])
        raise ValueError(
            f"{caller}: {term.label} with {present} has no brms model: brms writes "
            "these terms into a call of its SAR density, which takes the whole "
            "response vector and one scalar sigma, and the Stan program does not "
            "compile"
        )
    if error is not None:
        if conditional:
            raise ValueError(
                "Please set cov = TRUE in ARMA structures when including known "
                "standard errors."
            )
        if term.kind == "fcor":
            raise ValueError("Invalid addition arguments for this model.")
        if not error.sigma:
            raise ValueError(
                "Please set argument 'sigma' of function 'se' to TRUE when "
                "modeling time-series covariance matrices."
            )
    if adjusted and (term.kind != "arma" or term.cov):
        raise ValueError("Invalid addition arguments for this model.")
    return error is not None


def autocor_family(
    terms: tuple[AutocorTerm, ...],
    family: Family,
    *,
    data: IntoDataFrame,
    response: np.ndarray,
    response_name: str,
    data2: Mapping[str, AutocorMatrix] | None,
    distributional: Mapping[str, str | NonlinearExpression] | None,
    modifiers: ResponseModifiers,
    caller: str,
) -> Family:
    """Bind residual autocorrelation terms to an elliptical family.

    Parameters
    ----------
    terms : tuple[AutocorTerm, ...]
        One residual term, or a conditional ARMA term with a ``sar`` or
        ``fcor`` term.
    family : Family
        Base response family.
    data : IntoDataFrame
        Included training rows in original order.
    response : np.ndarray
        Evaluated response values on those rows.
    response_name : str
        Response column name.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices for ``sar`` and ``fcor``.
    distributional : Mapping[str, str or NonlinearExpression] or None
        Auxiliary-parameter formulas.
    modifiers : ResponseModifiers
        Response-side roles of the formula.
    caller : str
        Entry point named in errors.

    Returns
    -------
    Family
        A custom family whose native law is a `ResidualResponse` over the
        training rows, sharing the base family's link and auxiliary priors.

    Raises
    ------
    ValueError
        If the family, terms, auxiliary regressions, or response roles fall
        outside the brms formulations.
    """
    conditional = next(
        (term for term in terms if term.kind == "arma" and not term.cov), None
    )
    structural = [term for term in terms if term is not conditional]
    if len(structural) > 1:
        raise ValueError("Can only model one covariance matrix of natural residuals.")
    term = structural[0] if structural else conditional
    assert term is not None
    arma = conditional if structural else None
    label = " + ".join(item.label for item in terms)
    elliptical = family.elliptical
    if elliptical is None:
        raise ValueError(
            f"{caller}: {label} requires a family declaring an elliptical "
            f"residual law (Family.elliptical), such as 'gaussian' or 'student_t'; "
            f"{family.name!r} declares none"
        )
    if not family.location_link.is_identity:
        raise ValueError(
            f"{caller}: {label} requires the identity link of the location"
        )
    regressed = set(distributional or {})
    if elliptical.df in regressed:
        raise ValueError(f"{caller}: {label} requires a scalar {elliptical.df!r}")
    if elliptical.scale in regressed and term.kind == "sar":
        raise ValueError(
            "SAR models are not implemented when predicting "
            f"{elliptical.scale!r} or 'nu'."
        )
    known = _addition_rules(term, arma, modifiers, caller)
    student = elliptical.df is not None
    frame = nw.from_native(data, eager_only=True).to_pandas().reset_index(drop=True)
    rows = len(frame)
    history = np.asarray(response, dtype=np.float64).reshape(rows)
    se_source = dict(modifiers.sources).get("standard_error") if known else None
    if isinstance(se_source, bool):
        raise ValueError(f"{caller}: se() requires a column or a numeric value")
    layout, time_levels = _structure_layout(
        term,
        arma,
        frame,
        student=student,
        history=history,
        time_levels=None,
        data2=data2,
        known_variance=_known_variance(frame, se_source),
        caller=caller,
    )
    coefficients: list[str] = []
    bounds: list[tuple[str, float | None, float | None]] = []
    auxiliary: list[AuxParam] = []
    for item in (arma, term):
        if item is None:
            continue
        if item.kind == "arma":
            names = [f"ar_{lag}" for lag in range(1, item.p + 1)]
            names += [f"ma_{lag}" for lag in range(1, item.q + 1)]
            for name in names:
                if item.cov:
                    bounds.append((name, -1.0, 1.0))
                    prior = "Uniform(-1.0, 1.0)"
                else:
                    bounds.append((name, None, None))
                    prior = "Normal(0.0, 1.0)"
                auxiliary.append(
                    AuxParam(name=name, prior=prior, link=links["identity"])
                )
            coefficients += names
        elif item.kind == "cosy":
            coefficients.append("cosy")
            bounds.append(("cosy", 0.0, 1.0))
            auxiliary.append(
                AuxParam(name="cosy", prior="Uniform(0.0, 1.0)", link=links["identity"])
            )
        elif item.kind == "sar":
            name = "lagsar" if item.sar_type == "lag" else "errorsar"
            assert layout.eigenvalues is not None
            low, high = float(layout.eigenvalues[0]), float(layout.eigenvalues[-1])
            coefficients.append(name)
            bounds.append((name, low, high))
            auxiliary.append(
                AuxParam(
                    name=name,
                    prior=f"Uniform({low!r}, {high!r})",
                    link=links["identity"],
                )
            )
    clash = regressed & set(coefficients)
    if clash:
        raise ValueError(
            f"{caller}: autocorrelation parameters {sorted(clash)} cannot be regressed"
        )
    # One family per response, so two responses of a joint model keep their
    # own row layouts in one catalog.
    name = (
        f"{'ResidualStudentT' if student else 'ResidualNormal'}_"
        f"{_qvr_name(response_name)}"
    )
    record = _residual_record(
        name, coefficients, student=student, unstructured=term.kind == "unstr"
    )
    binding = AutocorBinding(
        term=term,
        arma=arma,
        se=se_source,
        layout=layout,
        record=record,
        coefficients=tuple(coefficients),
        bounds=tuple(bounds),
        time_levels=time_levels,
        response=response_name,
    )
    catalog = _catalog(binding, layout)
    _BINDINGS[id(catalog)] = (catalog, binding)
    bindings = [("loc", "mu"), ("scale", elliptical.scale)]
    if elliptical.df is not None:
        bindings.append(("nu", elliptical.df))
    bindings.extend((coefficient, coefficient) for coefficient in coefficients)
    if term.kind == "unstr":
        bindings.append(("factor", "formula_cortime_factor"))
    order = {parameter.name: index for index, parameter in enumerate(record.parameters)}
    return Family(
        name=name,
        location_link=family.location_link,
        observe_family=name,
        aux_params=(*family.aux_params, *auxiliary),
        recorded_events=family.recorded_events,
        custom=CustomFamilySpec(
            catalog=catalog,
            bindings=tuple(sorted(bindings, key=lambda item: order[item[0]])),
            data=family.custom.data if family.custom is not None else (),
        ),
    )


def latent_residual_term(
    term: AutocorTerm, *, data: IntoDataFrame, caller: str
) -> LatentTerm:
    """Bind brms's latent residuals for a family without natural residuals.

    brms adds ``err`` to the linear predictor with ``err = sderr * zerr``
    and the term's structure: a conditional AR term adds
    ``sum ar_k err_(n-k)`` (MA terms need ``cov = TRUE``), and a covariance
    term correlates ``err`` within each series. The ARMA coefficients are
    bounded by ``[-1, 1]`` and ``cosy`` by ``[0, 1]``, as brms bounds them
    with latent residuals.

    Parameters
    ----------
    term : AutocorTerm
        Residual term.
    data : IntoDataFrame
        Included training rows.
    caller : str
        Entry point named in errors.

    Returns
    -------
    LatentTerm
        The latent residual term.

    Raises
    ------
    ValueError
        With brms's messages for MA terms without ``cov = TRUE``, and SAR or
        FCOR terms.
    """
    if term.kind == "sar":
        raise ValueError("SAR terms are not implemented for this family.")
    if term.kind == "fcor":
        raise ValueError("FCOR terms are not implemented for this family.")
    if term.kind == "arma" and not term.cov and term.q > 0:
        raise ValueError(
            "Please set cov = TRUE when modeling MA structures for this family."
        )
    frame = nw.from_native(data, eager_only=True).to_pandas().reset_index(drop=True)
    layout, levels = _series_layout(
        term, frame, student=False, history=None, time_levels=None, caller=caller
    )
    coefficients: list[tuple[str, str, float, float]] = []
    if term.kind == "arma":
        coefficients += [(f"ar_{lag}", "ar", -1.0, 1.0) for lag in range(1, term.p + 1)]
        coefficients += [(f"ma_{lag}", "ma", -1.0, 1.0) for lag in range(1, term.q + 1)]
        if not term.cov:
            # u = (I + sum ar_k L^k) err: the moving average of iid innovations.
            layout = layout.with_(p=0, q=term.p)
    elif term.kind == "cosy":
        coefficients.append(("cosy", "cosy", 0.0, 1.0))
    return LatentTerm(
        name="err",
        kind="residual",
        group=term.group,
        layout=layout,
        coefficients=tuple(coefficients),
        codes=tuple(range(len(frame))),
        time_levels=levels,
        label=term.label,
    )


def rebound_residual(
    term: LatentTerm, binding: AutocorTerm, frame: pd.DataFrame, *, include: bool
) -> LatentTerm:
    """Bind latent residuals to fresh rows, as brms predicts them.

    Fresh rows draw new residuals from the fitted structure at each posterior
    draw (brms ``cov = TRUE``); the conditional formulation cannot, as brms
    states.

    Parameters
    ----------
    term : LatentTerm
        Fitted residuals.
    binding : AutocorTerm
        Their authored structure.
    frame : pandas.DataFrame
        Fresh rows.
    include : bool
        Include autocorrelation terms (brms ``incl_autocor``).

    Returns
    -------
    LatentTerm
        Residuals of the fresh rows.

    Raises
    ------
    ValueError
        For a conditional structure, or rows its series cannot hold.
    """
    assert term.layout is not None
    if term.layout.kind == "arma" and not term.layout.stationary and include:
        raise ValueError(
            "Cannot predict new latent residuals when using cov = FALSE in "
            "autocor terms."
        )
    rows = frame.reset_index(drop=True)
    if not include and binding.kind != "unstr" and binding.time is not None:
        # An excluded term adds nothing to the predictor (brms
        # ``incl_autocor = FALSE``), so its rows need no time order:
        # prediction grids repeat time points within a group.
        rows = rows.assign(**{binding.time: range(len(rows))})
    layout, _ = _series_layout(
        binding,
        rows,
        student=False,
        history=None,
        time_levels=_unstr_levels(term),
        caller="new_data",
    )
    if binding.kind == "arma" and not binding.cov:
        layout = layout.with_(p=0, q=binding.p)
    return term.with_(layout=layout, codes=tuple(range(len(frame))))


def _unstr_levels(term: LatentTerm) -> tuple[str, ...] | None:
    """Return a latent residual term's retained UNSTR time levels.

    Parameters
    ----------
    term : LatentTerm
        Latent residual term.

    Returns
    -------
    tuple[str, ...] or None
        The levels, or None for other structures.
    """
    assert term.layout is not None
    if term.layout.kind != "unstr":
        return None
    return term.time_levels


def bind_autocor_terms(
    terms: tuple[AutocorTerm, ...],
    family: Family,
    formula: Formula,
    *,
    data: IntoDataFrame,
    data2: Mapping[str, AutocorMatrix] | None,
    distributional: Mapping[str, str | NonlinearExpression] | None,
    nonlinear: bool,
    caller: str,
) -> tuple[Family, Formula]:
    """Bind every parsed autocorrelation term of one formula.

    A residual term replaces the response law (`autocor_family`); a ``car``
    term adds a latent spatial effect to the location predictor
    (`latent_car_term`).

    Parameters
    ----------
    terms : tuple[AutocorTerm, ...]
        Parsed terms.
    family : Family
        Base response family.
    formula : Formula
        Evaluated location formula.
    data : IntoDataFrame
        Included training rows.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    distributional : Mapping or None
        Auxiliary-parameter formulas.
    nonlinear : bool
        Whether the location is a nonlinear formula.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[Family, Formula]
        The bound family and the formula with its latent terms.

    Raises
    ------
    ValueError
        If a term is incompatible with the family or formula, as in brms.
    """
    residual = tuple(term for term in terms if term.kind != "car")
    if family.mixture is not None:
        if residual:
            raise ValueError(
                "Autocorrelation terms cannot be applied in mixture models."
            )
        # The component locations hold the spatial fields.
        return family, formula
    if nonlinear and any(term.kind == "car" for term in terms):
        raise ValueError("CAR terms are not implemented for non-linear models.")
    if residual and nonlinear and family.elliptical is None:
        raise ValueError("Latent residuals are not implemented for non-linear models.")
    if residual and family.elliptical is None:
        # A SAR or fixed-covariance term raises brms's refusal before any
        # second latent term could be added.
        ordered = sorted(residual, key=lambda term: term.dimension == "time")
        formula = formula.with_(
            latent_terms=(
                *formula.latent_terms,
                *(
                    latent_residual_term(term, data=data, caller=caller)
                    for term in ordered
                ),
            )
        )
        residual = ()
    if residual:
        family = autocor_family(
            residual,
            family,
            data=data,
            response=formula.response_values,
            response_name=formula.response_name,
            data2=data2,
            distributional=distributional,
            modifiers=formula.modifiers,
            caller=caller,
        )
        binding = autocor_binding(family)
        if binding is not None and binding.se is not None:
            # The squared errors live in the residual layout, on the diagonal
            # of each series covariance, so no scale role reaches the program.
            formula = formula.with_(
                modifiers=formula.modifiers.with_(
                    standard_error=None,
                    sources=tuple(
                        (role, source)
                        for role, source in formula.modifiers.sources
                        if role not in {"standard_error", "se_sigma"}
                    ),
                )
            )
    return family, bind_car_terms(terms, formula, data=data, data2=data2, caller=caller)


def bind_car_terms(
    terms: tuple[AutocorTerm, ...],
    formula: Formula,
    *,
    data: IntoDataFrame,
    data2: Mapping[str, AutocorMatrix] | None,
    caller: str,
) -> Formula:
    """Add the spatial fields of one predictor to its formula.

    Parameters
    ----------
    terms : tuple[AutocorTerm, ...]
        Parsed terms of the predictor.
    formula : Formula
        Evaluated predictor.
    data : IntoDataFrame
        Included training rows.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    caller : str
        Entry point named in errors.

    Returns
    -------
    Formula
        The formula with one latent term per ``car`` term.

    Raises
    ------
    ValueError
        If a matrix is missing from ``data2``.
    """
    for term in terms:
        if term.kind != "car":
            continue
        assert term.matrix is not None and term.car_type is not None
        if data2 is None or term.matrix not in data2:
            raise ValueError(f"{caller}: {term.label} requires data2[{term.matrix!r}]")
        frame = nw.from_native(data, eager_only=True).to_pandas().reset_index(drop=True)
        formula = formula.with_(
            latent_terms=(
                *formula.latent_terms,
                latent_car_term(
                    term.label, term.car_type, term.group, data2[term.matrix], frame
                ),
            )
        )
    return formula


def fields_and_design(
    text: str,
    *,
    data: IntoDataFrame,
    data2: Mapping[str, AutocorMatrix] | None,
    caller: str,
) -> tuple[Formula, DesignMatrices]:
    """Evaluate one auxiliary predictor with its structured fields.

    Parameters
    ----------
    text : str
        Formula of the predictor.
    data : IntoDataFrame
        Included training rows.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices.
    caller : str
        Entry point named in errors.

    Returns
    -------
    tuple[Formula, DesignMatrices]
        The evaluated predictor and its retained design. Residual structures
        of a predictor other than the location are latent residuals.
    """
    stripped, terms = split_autocor_terms(text)
    formula, design = _formula_and_design(stripped, data)
    residual = sorted(
        (term for term in terms if term.kind != "car"),
        key=lambda term: term.dimension == "time",
    )
    formula = formula.with_(
        formula=text,
        latent_terms=tuple(
            latent_residual_term(term, data=data, caller=caller) for term in residual
        ),
    )
    return (
        bind_car_terms(terms, formula, data=data, data2=data2, caller=caller),
        design,
    )


def autocor_prior_target(
    family: Family, name: str
) -> tuple[str, str | None, float | None, float | None] | None:
    """Give the brms prior class, coefficient, and bounds of a structural site.

    Parameters
    ----------
    family : Family
        Prepared response family.
    name : str
        Auxiliary site name.

    Returns
    -------
    tuple[str, str or None, float or None, float or None] or None
        Class, coefficient label, and interval of an autocorrelation
        parameter, or None for any other site.
    """
    binding = autocor_binding(family)
    if binding is None:
        return None
    for site, lower, upper in binding.bounds:
        if site == name:
            role, _, lag = name.partition("_")
            return role, lag or None, lower, upper
    return None


def autocor_matrix_targets(family: Family) -> tuple[tuple[str, str], ...]:
    """List correlation-matrix prior sites of an autocorrelation term.

    Parameters
    ----------
    family : Family
        Prepared response family.

    Returns
    -------
    tuple[tuple[str, str], ...]
        ``(site, class)`` pairs: the unstructured ``Lcortime`` factor.
    """
    binding = autocor_binding(family)
    if binding is None or binding.term.kind != "unstr":
        return ()
    return (("Lcortime", "Lcortime"),)


def append_autocor_steps(
    lens: FormulaToQVRModule, statements: list[Statement], steps: list[ProgramStep]
) -> None:
    """Emit structural sites that are not scalar auxiliary parameters.

    The unstructured term draws an LKJ Cholesky factor over its retained time
    levels through the compiler's existing correlation emitter.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler holding the prepared family and correlation emitter.
    statements : list
        Module statements receiving the time-level axis.
    steps : list
        Program steps receiving the factor.
    """
    binding = autocor_binding(lens._family)
    if binding is None or binding.term.kind != "unstr":
        return
    axis = "FormulaAutocorTime"
    lens._append_finite_object(statements, axis, binding.layout.times)
    lens._append_residual_correlation(
        steps,
        axis,
        binding.layout.times,
        result="formula_cortime_factor",
        site="Lcortime",
    )


def _prediction_history(
    fit: BayesianFit, frame: pd.DataFrame, binding: AutocorBinding
) -> np.ndarray:
    """Read fresh response history, NaN where it is absent.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula with its response evaluator.
    frame : pandas.DataFrame
        Prediction rows.
    binding : AutocorBinding
        Retained term.

    Returns
    -------
    np.ndarray
        Response history in row order.
    """
    rows = len(frame)
    if binding.response not in frame.columns:
        return np.full(rows, np.nan)
    plan = fit.prediction_plan
    assert plan is not None
    evaluator = plan.designs["mu"].response
    assert evaluator is not None
    filled = frame.copy()
    missing = filled[binding.response].isna().to_numpy()
    if missing.any():
        donor = float(np.nanmean(np.asarray(fit.formula.response_values)))
        filled[binding.response] = filled[binding.response].fillna(donor)
    values = np.array(
        evaluator.term.term.eval_new_data(filled), dtype=np.float64, copy=True
    ).reshape(rows)
    values[missing] = np.nan
    return values


def autocor_catalog(
    fit: BayesianFit,
    frame: pd.DataFrame | None,
    *,
    include: bool,
    data2: Mapping[str, AutocorMatrix] | None = None,
) -> FamilyCatalog | None:
    """Rebuild the residual law for prediction rows, as brms's prediction does.

    New rows form their own series: ARMA history comes from the response
    column of the new rows when present (NaN rows are predicted), never from
    the training rows. A SAR or fixed-covariance term reads its matrix over
    the new rows from ``data2`` (brms ``newdata2``), and known standard
    errors are read from the new rows. ``include=False`` drops the
    structure, leaving independent Gaussian or Student rows (brms
    ``incl_autocor = FALSE``) whose scale keeps any known standard errors.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula.
    frame : pandas.DataFrame or None
        Prediction rows, or None for the training rows.
    include : bool
        Include the autocorrelation structure.
    data2 : Mapping[str, AutocorMatrix] or None
        Named matrices over the prediction rows.

    Returns
    -------
    FamilyCatalog or None
        The prediction catalog, or None when the fit has no autocorrelation
        term.

    Raises
    ------
    ValueError
        If new rows lack a SAR or FCOR matrix of their own, or violate a
        term's time, group, or matrix contract.
    """
    binding = autocor_binding(fit.family)
    if binding is None:
        return None
    if frame is None and include:
        assert fit.family.custom is not None
        return fit.family.custom.catalog
    rows = binding.layout.rows if frame is None else len(frame)
    fresh = None if frame is None else frame.reset_index(drop=True)
    known = (
        binding.layout.known_variance
        if fresh is None
        else _known_variance(fresh, binding.se)
    )
    if not include:
        layout = ResidualLayout(
            kind="independent",
            rows=rows,
            order=tuple(range(rows)),
            starts=(0,),
            lengths=(rows,),
            student=binding.layout.student,
            known_variance=known,
        )
        return _catalog(binding, layout)
    assert fresh is not None
    if binding.term.dimension != "time" and (
        data2 is None or binding.term.matrix not in data2
    ):
        raise ValueError(
            f"{binding.term.label}: prediction on new_data requires "
            f"data2[{binding.term.matrix!r}] over the new rows; predict on the "
            "training rows or pass incl_autocor=False"
        )
    conditional = binding.arma is not None or (
        binding.term.kind == "arma" and not binding.term.cov
    )
    layout, _ = _structure_layout(
        binding.term,
        binding.arma,
        fresh,
        student=binding.layout.student,
        history=_prediction_history(fit, fresh, binding) if conditional else None,
        time_levels=binding.time_levels,
        data2=data2,
        known_variance=known,
        caller="new_data",
    )
    return _catalog(binding, layout)


def autocor_joint(family: Family) -> bool:
    """Report whether a family's residual structure is non-factorized.

    Parameters
    ----------
    family : Family
        Fitted response family.

    Returns
    -------
    bool
        True for covariance and SAR structures, whose pointwise likelihood is
        brms's leave-one-out conditional; False otherwise, including the
        conditional ARMA formulation, whose rows factorize.
    """
    binding = autocor_binding(family)
    return binding is not None and not (
        binding.term.kind == "arma" and not binding.term.cov
    )


def _residual_law(law: Distribution) -> ResidualResponse:
    """Unwrap event reinterpretations around a replayed residual law.

    Parameters
    ----------
    law : Distribution
        Replayed response distribution.

    Returns
    -------
    ResidualResponse
        The residual law.

    Raises
    ------
    TypeError
        If the response is not a residual dependence law.
    """
    while isinstance(law, Independent):
        law = law.base_dist
    if isinstance(law, TruncatedResidualRows):
        law = law.residual
    if not isinstance(law, ResidualResponse):
        raise TypeError("the response is not a residual dependence law")
    return law


def autocor_log_likelihood(
    traces: Iterable[Trace], response: str, shape: tuple[int, int]
) -> dict[str, torch.Tensor]:
    """Collect brms's pointwise log likelihood of a residual structure.

    Parameters
    ----------
    traces : Iterable[Trace]
        Replayed executions, one per chain and draw.
    response : str
        Response site name.
    shape : tuple[int, int]
        Chain and draw counts.

    Returns
    -------
    dict[str, torch.Tensor]
        Response name to scores of shape ``(chain, draw, row)``.
    """

    def values():
        for recorded in traces:
            law = _residual_law(response_distribution(recorded, response))
            value = recorded.sites[response].value
            yield {response: law.loo_log_prob(value.reshape(law.loc.shape))}

    return collect_site_draws(values(), shape)


def autocor_linear_predictor(recorded: Trace, response: str) -> torch.Tensor:
    """Read brms's ``mu`` including conditional ARMA terms from a replay.

    Parameters
    ----------
    recorded : Trace
        One replayed execution.
    response : str
        Response site name.

    Returns
    -------
    torch.Tensor
        Linear predictor of each row.
    """
    return _residual_law(response_distribution(recorded, response)).linear_predictor


__all__ = ["AutocorMatrix", "AutocorTerm"]
