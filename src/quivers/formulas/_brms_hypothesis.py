"""brms ``hypothesis()``: hypothesis strings evaluated into brms's table.

A brms hypothesis is a string ``"left (= | < | >) right"`` whose sides are R
expressions over parameter names. brms prefixes each name with the parameter
class (``b_``, or ``sd_<group>__`` with a group), evaluates
``(left) - (right)`` on the posterior draws, and reports the estimate, its
error, a credible interval, the evidence ratio, the posterior probability
and a star. This module reads the same strings, evaluates them through the
shared contrast evaluator over the brms-named draws of a fit, and builds
the same table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import CubicSpline
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.diagnostics.contrast import (
    ContrastExpression,
    _contrast_plan,
    _evaluate_contrast,
)
from quivers.formulas._r_expression import HYPOTHESIS_NAME, translate_r_expression

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit
    from quivers.formulas._joint_fit import MultivariateFit

type HypothesisInput = str | Sequence[str] | Mapping[str, str]
"""One brms hypothesis string, several, or several keyed by their names."""

_SIGNS = re.compile(r"=|<|>")
_SIDES = re.compile(r"[^=<>]+")
_SPACE = re.compile(r"[ \t\r\n]")

#: Grid size of R's ``density()`` in brms's ``density_ratio``.
_DENSITY_POINTS = 4096


class HypothesisRow(ValueModel):
    """One row of brms's hypothesis table.

    Parameters
    ----------
    hypothesis : str
        The hypothesis name, by default ``"(left)-(right) > 0"``.
    estimate, est_error : float
        Mean and standard deviation of the contrast draws, or median and
        scaled median absolute deviation when robust.
    ci_lower, ci_upper : float
        Quantiles of the contrast: ``alpha / 2`` and ``1 - alpha / 2`` for a
        two-sided hypothesis, ``alpha`` and ``1 - alpha`` for a one-sided one.
    evid_ratio : float
        Posterior odds of a one-sided hypothesis, or the ratio of posterior
        to prior density at zero for a point hypothesis (NaN without prior
        draws, brms's ``NA``).
    post_prob : float
        ``evid_ratio / (1 + evid_ratio)``, one for an infinite ratio.
    star : str
        ``"*"`` when a one-sided posterior probability exceeds
        ``1 - alpha`` or a two-sided interval excludes zero, else ``""``.
    group : str or None
        Level of the grouping factor for the ``ranef`` and ``coef`` scopes.
    """

    hypothesis: str
    estimate: float
    est_error: float
    ci_lower: float
    ci_upper: float
    evid_ratio: float
    post_prob: float
    star: str
    group: str | None = None


class HypothesisTable(ValueModel):
    """The value of brms ``hypothesis()``.

    Parameters
    ----------
    rows : tuple[HypothesisRow, ...]
        One row per hypothesis (and per level for the per-level scopes).
    samples : torch.Tensor
        Contrast draws ``(draw, row)``, chains concatenated, brms's
        ``samples`` with columns ``H1``, ``H2``, ....
    prior_samples : torch.Tensor or None
        Contrast draws under the prior, when prior draws were available for
        every row.
    class_ : str
        Parameter class prefix without its trailing underscores.
    alpha : float
        The level of the intervals and stars.
    """

    rows: tuple[HypothesisRow, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    samples: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    prior_samples: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    class_: str = ""
    alpha: float = 0.05

    def to_frame(self) -> pd.DataFrame:
        """Return the table with brms's column names.

        Returns
        -------
        pandas.DataFrame
            Columns ``Hypothesis``, ``Estimate``, ``Est.Error``,
            ``CI.Lower``, ``CI.Upper``, ``Evid.Ratio``, ``Post.Prob`` and
            ``Star``, preceded by ``Group`` for the per-level scopes.
        """
        frame = pd.DataFrame(
            {
                "Hypothesis": [row.hypothesis for row in self.rows],
                "Estimate": [row.estimate for row in self.rows],
                "Est.Error": [row.est_error for row in self.rows],
                "CI.Lower": [row.ci_lower for row in self.rows],
                "CI.Upper": [row.ci_upper for row in self.rows],
                "Evid.Ratio": [row.evid_ratio for row in self.rows],
                "Post.Prob": [row.post_prob for row in self.rows],
                "Star": [row.star for row in self.rows],
            }
        )
        if any(row.group is not None for row in self.rows):
            frame.insert(0, "Group", [row.group for row in self.rows])
        return frame

    def __str__(self) -> str:
        """Render the table as brms prints it.

        Returns
        -------
        str
            The header, the table rounded to two digits, and the legend.
        """
        one, two = (1 - self.alpha * 2) * 100, (1 - self.alpha) * 100
        return (
            f"Hypothesis Tests for class {self.class_}:\n"
            f"{self.to_frame().round(2).to_string()}\n---\n"
            f"'CI': {one:g}%-CI for one-sided and {two:g}%-CI for two-sided "
            "hypotheses.\n"
            "'*': For one-sided hypotheses, the posterior probability exceeds "
            f"{two:g}%;\nfor two-sided hypotheses, the value tested against lies "
            f"outside the {two:g}%-CI.\n"
            "Posterior probabilities of point hypotheses assume equal prior "
            "probabilities."
        )


class ParsedHypothesis(ValueModel):
    """One brms hypothesis string split as brms splits it.

    Parameters
    ----------
    contrast : str
        ``"(left)"`` or ``"(left)-(right)"``, white space removed.
    sign : {"=", "<", ">"}
        The relation tested against zero.
    """

    contrast: str
    sign: Literal["=", "<", ">"]


def is_brms_hypothesis(expression: object) -> bool:
    """Report whether a hypothesis argument is in brms's string form.

    Parameters
    ----------
    expression : object
        The first argument of a ``hypothesis`` method.

    Returns
    -------
    bool
        True for a sequence or mapping of strings, and for one string with
        ``=``, ``<`` or ``>`` outside every parenthesis. A QVR contrast
        expression has no relation at that depth, since a contrast returns a
        number.
    """
    if isinstance(expression, Mapping):
        return True
    if isinstance(expression, Sequence) and not isinstance(expression, str):
        return True
    if not isinstance(expression, str):
        return False
    depth = 0
    for character in expression:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
        elif depth == 0 and character in "=<>":
            return True
    return False


def parse_hypothesis(text: str) -> ParsedHypothesis:
    """Split one hypothesis string into its contrast and relation.

    Parameters
    ----------
    text : str
        ``"left (= | < | >) right"``.

    Returns
    -------
    ParsedHypothesis
        ``(left)-(right)``, or ``(left)`` when the right side is ``0``, and
        the relation, as brms's ``eval_hypothesis`` forms them.

    Raises
    ------
    ValueError
        If the string does not hold exactly one relation between two sides.
    """
    stripped = _SPACE.sub("", text)
    signs = _SIGNS.findall(stripped)
    sides = _SIDES.findall(stripped)
    if len(signs) != 1 or len(sides) != 2:
        raise ValueError(
            f"hypothesis {text!r}: every hypothesis must be of the form "
            "'left (= OR < OR >) right'"
        )
    contrast = f"({sides[0]})" + ("" if sides[1] == "0" else f"-({sides[1]})")
    sign: Literal["=", "<", ">"] = (
        "=" if signs[0] == "=" else "<" if signs[0] == "<" else ">"
    )
    return ParsedHypothesis(contrast=contrast, sign=sign)


def r_density_at(values: npt.NDArray[np.float64], point: float) -> float:
    """Evaluate R's kernel density estimate of a sample at one point.

    This follows brms's ``density_ratio``: ``density(x, from, to, n = 4096)``
    with the Gaussian kernel and the ``bw.nrd0`` bandwidth, evaluated at
    ``point`` by cubic spline interpolation. The range is the sample range,
    extended to ``point`` by a quarter standard deviation when the sample
    lies on one side of it.

    Parameters
    ----------
    values : numpy.ndarray
        Sample, at least two values.
    point : float
        Evaluation point.

    Returns
    -------
    float
        The density estimate.

    Raises
    ------
    ValueError
        If the sample has fewer than two values or no spread.

    Notes
    -----
    R bins the sample linearly on ``2 n`` grid points, convolves the bins
    with the kernel by a fast Fourier transform, and interpolates linearly
    to the output grid; this function reproduces those steps. R then uses a
    Forsythe, Malcolm and Moler cubic spline, and this function a
    not-a-knot cubic spline. The two agree to the interpolation error of a
    smooth density on a grid of 4096 points. Cost is O(m + n log n) for
    ``m`` draws.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    count = x.size
    if count < 2:
        raise ValueError("a density estimate requires at least two draws")
    deviation = float(x.std(ddof=1))
    lower, upper = float(x.min()), float(x.max())
    if lower > point:
        lower = point - deviation / 4
    elif upper < point:
        upper = point + deviation / 4
    quartiles = np.quantile(x, [0.25, 0.75])
    spread = min(deviation, float(quartiles[1] - quartiles[0]) / 1.34)
    if spread == 0:
        spread = deviation or abs(float(x[0])) or 1.0
    bandwidth = 0.9 * spread * count ** (-0.2)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("a density estimate requires draws with a finite spread")
    n = _DENSITY_POINTS
    low, high = lower - 4 * bandwidth, upper + 4 * bandwidth
    # R's BinDist: linear binning of unit mass onto the first n grid points.
    delta = (high - low) / (n - 1)
    position = (x - low) / delta
    index = np.floor(position).astype(np.int64)
    fraction = position - index
    mass = np.zeros(2 * n)
    inside = (index >= 0) & (index <= n - 2)
    np.add.at(mass, index[inside], (1 - fraction[inside]) / count)
    np.add.at(mass, index[inside] + 1, fraction[inside] / count)
    below = index == -1
    mass[0] += fraction[below].sum() / count
    above = index == n - 1
    np.add.at(mass, index[above], (1 - fraction[above]) / count)
    lags = np.linspace(0, 2 * (high - low), 2 * n)
    lags[n + 1 :] = -lags[n - 1 : 0 : -1]
    kernel = np.exp(-0.5 * (lags / bandwidth) ** 2) / (
        bandwidth * math.sqrt(2 * math.pi)
    )
    smoothed = np.fft.ifft(np.fft.fft(mass) * np.conj(np.fft.fft(kernel))).real
    density = np.maximum(0.0, smoothed[:n])
    grid = np.linspace(lower, upper, n)
    ordinates = np.interp(grid, np.linspace(low, high, n), density)
    return float(CubicSpline(grid, ordinates)(point))


def _summary_row(
    name: str,
    sign: Literal["=", "<", ">"],
    draws: npt.NDArray[np.float64],
    prior: npt.NDArray[np.float64] | None,
    alpha: float,
    robust: bool,
    group: str | None,
) -> HypothesisRow:
    """Summarize the draws of one contrast as brms's ``eval_hypothesis``.

    Parameters
    ----------
    name : str
        Hypothesis name.
    sign : {"=", "<", ">"}
        Relation tested against zero.
    draws : numpy.ndarray
        Posterior contrast draws.
    prior : numpy.ndarray or None
        Prior contrast draws.
    alpha : float
        Level of the interval and star.
    robust : bool
        Median and scaled median absolute deviation instead of mean and
        standard deviation.
    group : str or None
        Level label for a per-level scope.

    Returns
    -------
    HypothesisRow
        The table row.
    """
    probabilities = (alpha / 2, 1 - alpha / 2) if sign == "=" else (alpha, 1 - alpha)
    if robust:
        estimate = float(np.median(draws))
        error = 1.4826 * float(np.median(np.abs(draws - estimate)))
    else:
        estimate = float(draws.mean())
        error = float(draws.std(ddof=1)) if draws.size > 1 else math.nan
    lower, upper = (float(value) for value in np.quantile(draws, probabilities))
    if sign == "=":
        ratio = (
            math.nan
            if prior is None
            else r_density_at(draws, 0.0) / r_density_at(prior, 0.0)
        )
    else:
        favourable = int((draws < 0).sum() if sign == "<" else (draws > 0).sum())
        against = draws.size - favourable
        ratio = math.inf if against == 0 else favourable / against
    probability = 1.0 if math.isinf(ratio) else ratio / (1 + ratio)
    if sign == "=":
        star = not (lower <= 0 <= upper)
    else:
        star = probability > 1 - alpha
    return HypothesisRow(
        hypothesis=name,
        estimate=estimate,
        est_error=error,
        ci_lower=lower,
        ci_upper=upper,
        evid_ratio=ratio,
        post_prob=probability,
        star="*" if star else "",
        group=group,
    )


def _contrast_draws(
    contrast: str,
    prefix: str,
    samples: Mapping[str, torch.Tensor],
    prior: Mapping[str, torch.Tensor] | None,
    context: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Evaluate one contrast on posterior and prior draws.

    Parameters
    ----------
    contrast : str
        ``(left)-(right)`` over unprefixed parameter names.
    prefix : str
        Class prefix joined to every name.
    samples : Mapping[str, torch.Tensor]
        Named posterior draws ``(chain, draw)``.
    prior : Mapping[str, torch.Tensor] or None
        Named prior draws.
    context : str
        Hypothesis named in errors.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray or None]
        Flattened posterior draws and, when every variable has prior draws,
        flattened prior draws.

    Raises
    ------
    ValueError
        If the expression is malformed or names a parameter that is not
        among the draws; the message lists the parameters of the class.
    """
    aliases: dict[str, str] = {}

    def alias(name: str) -> str:
        """Give one authored name a QVR identifier.

        Parameters
        ----------
        name : str
            Authored parameter name.

        Returns
        -------
        str
            The identifier bound to the prefixed name.
        """
        return aliases.setdefault(name, f"hypothesis_variable_{len(aliases)}")

    translated = translate_r_expression(
        contrast, context=context, name_pattern=HYPOTHESIS_NAME, rename=alias
    )
    missing = sorted(prefix + name for name in aliases if prefix + name not in samples)
    if missing:
        available = sorted(name for name in samples if name.startswith(prefix))
        raise ValueError(
            f"{context}: some parameters cannot be found in the model: "
            f"{missing!r}; the parameters of this class are {available!r}"
        )
    expression = ContrastExpression(
        expression=translated.expression,
        bindings={identifier: prefix + name for name, identifier in aliases.items()},
    )
    plan = _contrast_plan(expression, samples, None, None)
    posterior = _evaluate_contrast(plan, samples).reshape(-1).numpy()
    if prior is None or any(prefix + name not in prior for name in aliases):
        return posterior, None
    return posterior, _evaluate_contrast(plan, prior).reshape(-1).numpy()


def _named(hypothesis: HypothesisInput) -> list[tuple[str | None, str]]:
    """Normalize the hypothesis argument to optional names and strings.

    Parameters
    ----------
    hypothesis : str or Sequence[str] or Mapping[str, str]
        brms hypothesis strings.

    Returns
    -------
    list[tuple[str or None, str]]
        Name (None for the default) and hypothesis string, in order.

    Raises
    ------
    TypeError
        If an entry is not a string, or no hypothesis is given.
    """
    if isinstance(hypothesis, str):
        items: list[tuple[str | None, str]] = [(None, hypothesis)]
    elif isinstance(hypothesis, Mapping):
        items = [(str(name) or None, text) for name, text in hypothesis.items()]
    else:
        items = [(None, text) for text in hypothesis]
    if not items or not all(isinstance(text, str) for _, text in items):
        raise TypeError("argument 'hypothesis' must be one or more strings")
    return items


def hypothesis_table(
    samples: Mapping[str, torch.Tensor],
    hypothesis: HypothesisInput,
    *,
    prefix: str = "",
    alpha: float = 0.05,
    robust: bool = False,
    prior: Mapping[str, torch.Tensor] | None = None,
    group_label: str | None = None,
) -> HypothesisTable:
    """Evaluate brms hypothesis strings on named draws.

    Parameters
    ----------
    samples : Mapping[str, torch.Tensor]
        Draws ``(chain, draw)`` keyed by parameter name.
    hypothesis : str or Sequence[str] or Mapping[str, str]
        Hypothesis strings; a mapping gives each its name.
    prefix : str
        Class prefix joined to every name of a hypothesis, such as ``"b_"``
        or ``"sd_patient__"``.
    alpha : float
        Level of the intervals and stars, in [0, 1].
    robust : bool
        Median and scaled median absolute deviation instead of mean and
        standard deviation.
    prior : Mapping[str, torch.Tensor] or None
        Prior draws under the same names, for the evidence ratio of a point
        hypothesis.
    group_label : str or None
        Level label recorded on every row.

    Returns
    -------
    HypothesisTable
        brms's table, with the contrast draws.

    Raises
    ------
    ValueError
        If ``alpha`` is outside [0, 1], a string is not a brms hypothesis, or
        a name is not a parameter.
    TypeError
        If ``robust`` is not Boolean or an entry is not a string.

    Notes
    -----
    Cost is one pass over the draws per hypothesis, plus one density
    estimate per point hypothesis with prior draws.
    """
    if isinstance(alpha, bool) or not 0 <= alpha <= 1:
        raise ValueError("argument 'alpha' must be a single value in [0,1]")
    if type(robust) is not bool:
        raise TypeError("robust must be Boolean")
    rows, columns, prior_columns = [], [], []
    for name, text in _named(hypothesis):
        parsed = parse_hypothesis(text)
        draws, prior_draws = _contrast_draws(
            parsed.contrast, prefix, samples, prior, f"hypothesis {text!r}"
        )
        rows.append(
            _summary_row(
                name or f"{parsed.contrast} {parsed.sign} 0",
                parsed.sign,
                draws,
                prior_draws,
                alpha,
                robust,
                group_label,
            )
        )
        columns.append(torch.as_tensor(draws))
        prior_columns.append(prior_draws)
    return HypothesisTable(
        rows=tuple(rows),
        samples=torch.stack(columns, dim=1),
        prior_samples=torch.stack(
            [torch.as_tensor(column) for column in prior_columns if column is not None],
            dim=1,
        )
        if prior_columns and all(column is not None for column in prior_columns)
        else None,
        class_=prefix.rstrip("_"),
        alpha=alpha,
    )


def fit_hypothesis(
    fit: BayesianFit | MultivariateFit,
    hypothesis: HypothesisInput,
    *,
    class_: str | None,
    group: str | None,
    scope: Literal["standard", "ranef", "coef"],
    alpha: float,
    robust: bool,
    prior: bool,
    num_samples: int | None,
    num_prior_samples: int | None,
    seed: int,
) -> HypothesisTable:
    """Evaluate brms hypothesis strings on a fitted formula model.

    Parameters
    ----------
    fit : BayesianFit or MultivariateFit
        Fitted model.
    hypothesis : str or Sequence[str] or Mapping[str, str]
        brms hypothesis strings.
    class_ : str or None
        Parameter class joined to every name: ``"b"`` reads ``x`` as
        ``b_x``. None or ``""`` reads names as they are in `variables`.
    group : str or None
        Grouping factor. In the standard scope it extends the prefix to
        ``<class>_<group>__``. The per-level scopes require it.
    scope : {"standard", "ranef", "coef"}
        Parameters by name, or the per-level draws of ``ranef`` or ``coef``
        for ``group``, one row per level and hypothesis.
    alpha : float
        Level of the intervals and stars.
    robust : bool
        Median and scaled median absolute deviation.
    prior : bool
        Draw from the prior for the evidence ratio of point hypotheses.
        Prior draws retained by ``sample_prior`` are used either way.
    num_samples, num_prior_samples : int or None
        Posterior and prior draw counts.
    seed : int
        Seed for variational and prior draws.

    Returns
    -------
    HypothesisTable
        brms's table.

    Raises
    ------
    ValueError
        If the scope is unknown, a per-level scope lacks a known group, or a
        hypothesis is malformed or names an unknown parameter.
    """
    if scope not in {"standard", "ranef", "coef"}:
        raise ValueError("hypothesis scope must be standard, ranef, or coef")
    if scope == "standard":
        prefix = class_ or ""
        if group:
            prefix = f"{prefix}_{group}__"
        elif prefix:
            prefix = f"{prefix}_"
        view = (
            fit.prior_view(num_samples=num_prior_samples, seed=(seed + 1) % 2**64)
            if prior or fit.prior is not None
            else None
        )
        return hypothesis_table(
            fit.variables(num_samples=num_samples, seed=seed),
            hypothesis,
            prefix=prefix,
            alpha=alpha,
            robust=robust,
            prior=None if view is None else view.variables(),
        )
    arrays = (
        fit.ranef(summary=False, num_samples=num_samples, seed=seed)
        if scope == "ranef"
        else fit.coef(summary=False, num_samples=num_samples, seed=seed)
    )
    if not group or group not in arrays:
        raise ValueError(f"'group' should be one of {sorted(arrays)!r}")
    labels, values = arrays[group]
    assert isinstance(values, torch.Tensor)
    levels = fit.group_levels[group]
    tables = [
        hypothesis_table(
            {
                str(label): values[:, :, position, column]
                for column, label in enumerate(labels)
            },
            hypothesis,
            alpha=alpha,
            robust=robust,
            group_label=str(level),
        )
        for position, level in enumerate(levels)
    ]
    # brms orders the rows by hypothesis, then by level.
    count = len(tables[0].rows)
    order = [(row, level) for row in range(count) for level in range(len(tables))]
    return HypothesisTable(
        rows=tuple(tables[level].rows[row] for row, level in order),
        samples=torch.stack(
            [tables[level].samples[:, row] for row, level in order], dim=1
        ),
        class_="",
        alpha=alpha,
    )


__all__: list[str] = []
