"""brms ``gp()`` terms through the Hilbert-space approximation.

A Gaussian process (GP) term adds ``f(x)`` to a linear predictor with
``f ~ GP(0, k)`` for a stationary covariance kernel ``k`` with marginal
standard deviation ``sdgp`` and length scale ``lscale``
[@rasmussen2006gp]. brms's approximate GP (``k = m``) expands the kernel in
the eigenfunctions of the Laplacian on ``[-L, L]`` [@solin2020hilbert;
@riutort2023practical]:

``f(x) = sum_j sqrt(S(sqrt(lambda_j))) phi_j(x) z_j``, ``z_j ~ N(0, 1)``,

with ``phi_j`` the sinusoidal eigenfunctions, ``lambda_j`` their
eigenvalues, and ``S`` the kernel's spectral density. The basis matrix
``phi_j(x_n)`` depends only on the data, so the term is one Gaussian design
block whose per-coordinate scale ``sqrt(S)`` is an expression of the sampled
``sdgp`` and ``lscale``. The block shares the design path of penalized
smooths (`quivers.formulas.formula.GaussianDesignTerm`): its basis is a
formulae stateful transform, retained for fresh rows, and its innovations,
hyperparameters, and priors are ordinary latent sites.

Scoring costs O(n m) for ``n`` rows and ``m = k^D`` basis functions, and
prediction at ``r`` fresh rows O(r m) per draw.

brms's exact GP (no ``k``, or ``k = NA``) evaluates the latent function at
the ``u`` distinct scaled inputs in brms's non-centered form

``f = cholesky(K(sdgp, lscale) + 1e-12 I) zgp``, ``zgp ~ N(0, I)``,

with the QVR builtin ``cholesky``, and adds ``f`` at each row's input. The
row-to-input map and ``by`` multipliers form the block's design matrix. Each
log-density evaluation costs O(u^3) for the factor and O(n u) for the
contraction. A prediction at ``m`` distinct fresh inputs draws ``f`` there
from its Gaussian law given the fitted ``f`` under the same draw's
hyperparameters, as brms's ``.predictor_gp_new`` does, at O(u^3 + u^2 m +
u m^2 + m^3) per draw.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as optimize
import scipy.special as special
import torch
from torch.distributions import Distribution, MultivariateNormal, constraints
from torch.types import _size

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value
from quivers.dsl.ast_nodes import (
    DrawArgNamed,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
    LetExprLiteral,
    LetExprNode,
    ProgramStep,
    SampleStep,
    TypeName,
    atom_to_draw_arg,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import (
    _add,
    _call,
    _div,
    _let,
    _mul,
    _sub,
    _var,
)
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

if TYPE_CHECKING:
    from formulae.matrices import DesignMatrices

    from quivers.formulas.compile import FormulaToQVRModule
    from quivers.formulas.family import Family
    from quivers.formulas.formula import Formula

type ColumnInput = pd.Series | npt.NDArray[np.generic]
type GpKernel = Literal["exp_quad", "matern32", "matern52", "exponential"]

_KERNELS: tuple[GpKernel, ...] = ("exp_quad", "matern52", "matern32", "exponential")
_SMOOTHNESS: dict[GpKernel, float] = {
    "exponential": 0.5,
    "matern32": 1.5,
    "matern52": 2.5,
}
_RENAME = (
    (" ", ""),
    ("(", ""),
    (")", ""),
    ("[", ""),
    ("]", ""),
    (",", ""),
    ('"', ""),
    ("'", ""),
    ("?", ""),
    ("+", "P"),
    ("-", "M"),
    ("*", "MU"),
    ("/", "D"),
    ("^", "E"),
    ("=", "EQ"),
    ("$", "USD"),
)
_TAIL_PROBABILITY = 0.01
#: Diagonal brms's ``gp_*`` Stan functions add to the training kernel.
EXACT_JITTER = 1e-12
#: Diagonal brms adds to the conditional covariance at new inputs
#: (``prepare_predictions_gp``, ``nug`` for new data).
PREDICTION_NUGGET = 1e-8


def _frozen(value: npt.ArrayLike) -> np.ndarray:
    """Copy a numeric array once and clear its writeable flag.

    Parameters
    ----------
    value : array_like
        Known numeric values.

    Returns
    -------
    numpy.ndarray
        Read-only float64 copy.
    """
    frozen = np.array(value, dtype=np.float64, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


def _frozen_or_none(value: npt.ArrayLike | None) -> np.ndarray | None:
    """Freeze an optional numeric array.

    Parameters
    ----------
    value : array_like or None
        Known numeric values, or None.

    Returns
    -------
    numpy.ndarray or None
        Read-only float64 copy, or None.
    """
    return None if value is None else _frozen(value)


def brms_rename(text: str) -> str:
    """Apply brms's default renaming of special characters in labels.

    Parameters
    ----------
    text : str
        Covariate or variable text.

    Returns
    -------
    str
        The label brms uses in parameter names.
    """
    for pattern, replacement in _RENAME:
        text = text.replace(pattern, replacement)
    return text


def _qvr(text: str) -> str:
    """Normalize a brms label into a QVR identifier fragment.

    Parameters
    ----------
    text : str
        brms label.

    Returns
    -------
    str
        Identifier characters only.
    """
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return cleaned.strip("_") or "gp"


class GaussianProcessBlock(ValueModel):
    """One approximate GP (one ``by`` contrast) as a Gaussian design block.

    Attributes
    ----------
    label : str
        brms ``sdgp`` coefficient label, for instance ``gpx`` or ``gpxga``.
    lscale_labels : tuple[str, ...]
        brms ``lscale`` coefficient labels: one for an isotropic GP, one per
        covariate otherwise.
    kernel : GpKernel
        Covariance kernel.
    frequencies : numpy.ndarray
        ``(m, D)`` square roots of the Laplacian eigenvalues, brms's
        ``slambda``, read-only.
    lscale_priors : tuple[str, ...]
        Default prior of each length scale, in ``lscale_labels`` order.
    inputs : numpy.ndarray or None
        For an exact GP, the ``(u, D)`` distinct scaled training inputs,
        brms's ``Xgp``, in order of first appearance (every active row when
        ``gr`` is false); None for the approximation, whose ``frequencies``
        then hold the basis.
    gr : bool
        Whether rows sharing an input share one latent value.
    dmax : float
        Divisor applied to the covariates (one when ``scale = FALSE``).
    prediction_inputs : numpy.ndarray or None
        For an exact GP specialized to prediction rows, their ``(m, D)``
        distinct scaled inputs, at which the latent function is drawn given
        its training values.
    """

    label: str
    lscale_labels: tuple[str, ...]
    kernel: GpKernel
    frequencies: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    lscale_priors: tuple[str, ...]
    inputs: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_or_none)
    )
    gr: bool = True
    dmax: float = 1.0
    prediction_inputs: np.ndarray | None = dx.field(
        default=None, opaque=True, extras=opaque_by_value(_frozen_or_none)
    )

    @property
    def exact(self) -> bool:
        """Report whether the block is brms's exact GP.

        Returns
        -------
        bool
            True when the block retains its training inputs.
        """
        return self.inputs is not None

    @property
    def dimension(self) -> int:
        """Count the GP's covariates.

        Returns
        -------
        int
            Input dimension ``D``.
        """
        return int(self.frequencies.shape[1])

    @property
    def isotropic(self) -> bool:
        """Report whether one length scale serves every covariate.

        Returns
        -------
        bool
            True for a single length scale.
        """
        return len(self.lscale_labels) == 1

    def sites(self, prefix: str) -> tuple[str, tuple[str, ...]]:
        """Name the block's ``sdgp`` and ``lscale`` latent sites.

        Parameters
        ----------
        prefix : str
            Owning predictor namespace.

        Returns
        -------
        tuple[str, tuple[str, ...]]
            The ``sdgp`` site and the ``lscale`` sites.
        """
        return (
            f"{prefix}sdgp_{_qvr(self.label)}",
            tuple(f"{prefix}lscale_{_qvr(label)}" for label in self.lscale_labels),
        )


class GaussianProcessSubterm(ValueModel):
    """The retained scaling, centering, and basis of one sub-GP.

    Attributes
    ----------
    block : GaussianProcessBlock
        Emission metadata.
    level : str or None
        Factor level whose indicator selects the rows, or None for every row.
    dmax : float
        Divisor applied to the covariates (one when ``scale = FALSE``).
    centers : numpy.ndarray
        Training means of the scaled covariates, brms's ``cmeans``.
    boundary : numpy.ndarray
        Boundary ``L`` of each covariate's domain ``[-L, L]``.
    orders : numpy.ndarray
        ``(m, D)`` integer eigenfunction orders, brms's ``expand.grid``.
    """

    block: GaussianProcessBlock = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    level: str | None
    dmax: float
    centers: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    boundary: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    orders: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))

    def basis(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the eigenfunctions at covariate rows, as brms does.

        Costs O(rows times m times D).

        Parameters
        ----------
        x : numpy.ndarray
            ``(rows, D)`` covariates on their original scale.

        Returns
        -------
        numpy.ndarray
            ``(rows, m)`` basis values.
        """
        centered = x / self.dmax - self.centers
        result = np.ones((x.shape[0], self.orders.shape[0]))
        for d, half_width in enumerate(self.boundary):
            angle = (
                self.orders[:, d][None, :]
                * math.pi
                / (2 * half_width)
                * (centered[:, d][:, None] + half_width)
            )
            result = result * np.sin(angle) / math.sqrt(half_width)
        return result


class GaussianProcessState(ValueModel):
    """Everything a fitted ``gp()`` term retains for fresh rows.

    Attributes
    ----------
    covariates : tuple[str, ...]
        Covariate labels.
    by : str or None
        ``by`` variable label.
    by_kind : {"none", "numeric", "factor"}
        Role of the ``by`` variable.
    levels : tuple[str, ...]
        Training levels of a factor ``by``.
    subterms : tuple[GaussianProcessSubterm, ...]
        One sub-GP per ``by`` contrast, or one in total.
    """

    covariates: tuple[str, ...]
    by: str | None
    by_kind: Literal["none", "numeric", "factor"]
    levels: tuple[str, ...] = ()
    subterms: tuple[GaussianProcessSubterm, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )


def _numeric(value: ColumnInput) -> npt.NDArray[np.float64]:
    """Read one numeric GP covariate.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated covariate.

    Returns
    -------
    numpy.ndarray
        Float values.

    Raises
    ------
    ValueError
        If the column is not a finite numeric vector.
    """
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iuf" or array.dtype == bool:
        raise ValueError("Predictors of Gaussian processes should be numeric vectors.")
    result = array.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Predictors of Gaussian processes must be finite.")
    return result


def _factor(value: ColumnInput) -> tuple[npt.NDArray[np.str_], tuple[str, ...]] | None:
    """Read a factor ``by`` variable, or None for a numeric one.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated ``by`` column.

    Returns
    -------
    tuple or None
        Row labels and levels in R's factor order.

    Raises
    ------
    ValueError
        For an ordered factor, whose R contrasts are polynomial.
    """
    series = pd.Series(np.asarray(value)) if not isinstance(value, pd.Series) else value
    if isinstance(series.dtype, pd.CategoricalDtype):
        if series.dtype.ordered:
            raise ValueError(
                "gp(): an ordered factor 'by' variable takes polynomial contrasts in R; "
                "convert it to an unordered factor"
            )
        levels = tuple(str(level) for level in series.cat.categories)
    elif series.dtype.kind in "OUSb":
        levels = tuple(sorted({str(label) for label in series.tolist()}))
    else:
        return None
    return np.asarray([str(label) for label in series.tolist()]), levels


def _label(value: ColumnInput, position: int) -> str:
    """Name an evaluated covariate.

    Parameters
    ----------
    value : pandas.Series or numpy.ndarray
        Evaluated column.
    position : int
        Position among the covariates.

    Returns
    -------
    str
        The column name, or a positional name for an unnamed array.
    """
    name = getattr(value, "name", None)
    return str(name) if name is not None else f"x{position + 1}"


def _squared_distances(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pairwise squared Euclidean distances, brms's ``diff_quad``.

    Parameters
    ----------
    x : numpy.ndarray
        ``(rows, D)`` points.

    Returns
    -------
    numpy.ndarray
        ``(rows, rows)`` squared distances summed over columns.
    """
    return ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)


def lscale_prior(points: npt.NDArray[np.float64]) -> str:
    """Solve brms's inverse-gamma length-scale prior from training points.

    brms (``def_lscale_prior``) places one percent of prior mass below the
    smallest positive pairwise distance ``lb`` and one percent above the
    largest ``ub``, with ``lb >= 0.01 ub``, solving for the inverse-gamma
    shape and scale on the log scale and rounding them to six decimals. A
    failed solve leaves brms's ``normal(0, 0.5)`` on the positive half line.
    Costs O(rows^2 D) once at preparation.

    Parameters
    ----------
    points : numpy.ndarray
        ``(rows, D)`` scaled training covariates.

    Returns
    -------
    str
        A QVR prior expression.

    Raises
    ------
    ValueError
        If the covariates do not take two distinct values.
    """
    squared = _squared_distances(points)
    positive = squared[squared > 0]
    if positive.size == 0:
        raise ValueError(
            "gp(): the covariates take one value, so no length-scale prior exists"
        )
    upper = math.sqrt(float(squared.max()))
    lower = max(math.sqrt(float(positive.min())), 0.01 * upper)

    def equations(log_parameters: npt.NDArray[np.float64]) -> list[float]:
        shape, scale = np.exp(log_parameters)
        below = special.gammaincc(shape, scale / lower)
        above = special.gammainc(shape, scale / upper)
        return [
            math.log(max(below, 1e-300)) - math.log(_TAIL_PROBABILITY),
            math.log(max(above, 1e-300)) - math.log(_TAIL_PROBABILITY),
        ]

    solution = optimize.root(equations, np.zeros(2), method="hybr")
    if not solution.success or not np.allclose(equations(solution.x), 0, atol=1e-8):
        return "HalfNormal(0.5)"
    shape, scale = np.round(np.exp(solution.x), 6)
    return f"InverseGamma({float(shape)!r}, {float(scale)!r})"


def sdgp_prior(response: npt.NDArray[np.generic], family: Family) -> str:
    """Give brms's default ``sdgp`` prior, its ``def_scale_prior``.

    brms uses a half Student t with three degrees of freedom and scale
    ``max(2.5, round(mad(link(y)), 1))`` for the identity, log, inverse,
    square-root, and inverse-square links on a non-categorical response,
    and scale 2.5 otherwise. Log-scale families read the response through
    the log link.

    Parameters
    ----------
    response : numpy.ndarray
        Training response values.
    family : Family
        Response family.

    Returns
    -------
    str
        A QVR prior expression.
    """
    link = family.location_link.name
    if family.name in {"lognormal", "shifted_lognormal", "hurdle_lognormal"}:
        link = "log"
    scale = 2.5
    values = np.asarray(response)
    if (
        family.response is None
        and values.ndim == 1
        and values.dtype.kind in "iuf"
        and link in {"identity", "log", "inverse", "sqrt", "inverse_square"}
    ):
        y = values.astype(np.float64)
        if link in {"log", "inverse", "inverse_square"}:
            y = np.where(y == 0, y + 0.1, y)
        with np.errstate(all="ignore"):
            transformed = {
                "identity": y,
                "log": np.log(y),
                "inverse": 1 / y,
                "sqrt": np.sqrt(y),
                "inverse_square": 1 / y**2,
            }[link]
            spread = round(
                1.4826 * float(np.median(np.abs(transformed - np.median(transformed)))),
                1,
            )
        if math.isfinite(spread):
            scale = max(scale, spread)
    return f"HalfStudentT(3.0, {float(scale)!r})"


def _boundary(centered: npt.NDArray[np.float64], c: tuple[float, ...]) -> np.ndarray:
    """brms's ``choose_L``: the boundary factor times the covariate range.

    Parameters
    ----------
    centered : numpy.ndarray
        Scaled and centered training covariates.
    c : tuple[float, ...]
        Boundary factor per covariate.

    Returns
    -------
    numpy.ndarray
        ``L`` per covariate.
    """
    spread = max(1.0, float(centered.max() - centered.min())) if centered.size else 1.0
    return np.asarray(c, dtype=np.float64) * spread


class GaussianProcess:
    """The brms ``gp()`` term as a formulae stateful transform."""

    __stateful_transform__ = True

    def __init__(self) -> None:
        """Start without retained training state."""
        self.state: GaussianProcessState | None = None

    def __call__(
        self,
        *covariates: ColumnInput,
        by: ColumnInput | None = None,
        k: int | None = None,
        cov: str = "exp_quad",
        iso: bool = True,
        gr: bool = True,
        cmc: bool = True,
        scale: bool = True,
        c: float | tuple[float, ...] = 5 / 4,
        **unsupported: str | int | float | bool | None,
    ) -> npt.NDArray[np.float64]:
        """Construct the basis on the first call and evaluate it after.

        Parameters
        ----------
        *covariates : pandas.Series or numpy.ndarray
            Numeric GP covariates.
        by : pandas.Series, numpy.ndarray, or None
            Numeric multiplier, or a factor giving one GP per contrast.
        k : int or None
            Basis functions per covariate, ``m = k^D`` in total; None
            (brms's ``NA``) for the exact GP.
        cov : str
            ``exp_quad``, ``matern32``, ``matern52``, or ``exponential``
            (alias ``matern12``).
        iso : bool
            One length scale for every covariate (True) or one each.
        gr : bool
            Center on the distinct covariate rows, as brms's grouping does.
        cmc : bool
            Cell-mean coding of a factor ``by`` (one GP per level). False
            fits a GP on every row plus treatment-contrast GPs.
        scale : bool
            Divide the covariates by their largest pairwise distance.
        c : float or tuple of float
            Boundary factor per covariate of the approximation; the exact GP
            ignores it, as brms does.
        **unsupported : str or int or float or bool or None
            Rejected arguments.

        Returns
        -------
        numpy.ndarray
            For the approximation, ``(rows, sum of m)`` basis columns of every
            sub-GP, each set to zero outside its rows and multiplied by its
            contrast value. For the exact GP, ``1 + D`` columns per sub-GP:
            the contrast value of each row (zero outside its rows) and the
            covariates, from which `gp_rows` reads the inputs.

        Raises
        ------
        ValueError
            For an invalid argument, or covariates that differ in role from
            training.
        """
        if unsupported:
            raise ValueError(
                f"gp() arguments {sorted(unsupported)!r} are not supported; supported "
                "arguments are by, k, cov, iso, gr, cmc, scale, and c"
            )
        if not covariates:
            raise ValueError("gp() requires at least one covariate")
        x = np.column_stack([_numeric(column) for column in covariates])
        if self.state is None:
            self.state = self._fit(covariates, x, by, k, cov, iso, gr, cmc, scale, c)
        return self._evaluate(x, by)

    def _fit(
        self,
        covariates: tuple[ColumnInput, ...],
        x: npt.NDArray[np.float64],
        by: ColumnInput | None,
        k: int | None,
        cov: str,
        iso: bool,
        gr: bool,
        cmc: bool,
        scale: bool,
        c: float | tuple[float, ...],
    ) -> GaussianProcessState:
        """Validate arguments and retain each sub-GP's training state.

        Parameters
        ----------
        covariates : tuple
            Evaluated covariate columns.
        x : numpy.ndarray
            ``(rows, D)`` covariates.
        by, k, cov, iso, gr, cmc, scale, c
            As `GaussianProcess.__call__`.

        Returns
        -------
        GaussianProcessState
            Retained state.

        Raises
        ------
        ValueError
            As brms does for invalid arguments.
        """
        if not isinstance(cov, str):
            raise ValueError("gp() cov must be a kernel name string")
        kernel = "exponential" if cov == "matern12" else cov
        if kernel not in _KERNELS:
            raise ValueError(
                f"'{cov}' is not a valid GP covariance kernel. Valid kernels are: "
                + ", ".join(_KERNELS)
            )
        for name, flag in (("iso", iso), ("gr", gr), ("cmc", cmc), ("scale", scale)):
            if type(flag) is not bool:
                raise ValueError(f"gp() {name} must be TRUE or FALSE")
        dimension = x.shape[1]
        # brms's k = NA (here None) is the exact GP, which ignores c.
        exact = k is None
        boundary: tuple[float, ...] = ()
        if not exact:
            if isinstance(k, bool) or not isinstance(k, (int, float)) or k != int(k):
                raise ValueError("gp() k must be one integer")
            if int(k) < 1:
                raise ValueError("'k' must be positive.")
            boundary = (c,) if isinstance(c, (int, float)) else tuple(c)
            if any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in boundary
            ):
                raise ValueError("gp() c must be numeric")
            if len(boundary) == 1:
                boundary = boundary * dimension
            if len(boundary) != dimension:
                raise ValueError(
                    "'c' must be of the same length as the number of covariates."
                )
            if any(value <= 0 for value in boundary):
                raise ValueError("'c' must be positive.")
        isotropic = iso or dimension == 1
        names = tuple(_label(column, i) for i, column in enumerate(covariates))
        label = "gp" + brms_rename("".join(names))
        by_kind: Literal["none", "numeric", "factor"] = "none"
        by_name = None
        levels: tuple[str, ...] = ()
        contrasts: list[tuple[str, str | None, npt.NDArray[np.float64]]] = []
        rows = x.shape[0]
        if by is None:
            contrasts.append(("", None, np.ones(rows)))
        else:
            by_name = _label(by, 0)
            label += brms_rename(by_name)
            factor = _factor(by)
            if factor is None:
                by_kind = "numeric"
                contrasts.append(("", None, np.ones(rows)))
            else:
                by_kind = "factor"
                labels, levels = factor
                selected = levels if cmc else levels[1:]
                if not cmc:
                    contrasts.append(("Intercept", None, np.ones(rows)))
                for level in selected:
                    contrasts.append(
                        (level, level, (labels == level).astype(np.float64))
                    )
        orders = (
            np.zeros((0, dimension))
            if k is None
            else np.array(
                [
                    [(m // int(k) ** d) % int(k) + 1 for d in range(dimension)]
                    for m in range(int(k) ** dimension)
                ],
                dtype=np.float64,
            )
        )
        subterms = []
        for suffix, level, weights in contrasts:
            subset = x[weights != 0]
            if subset.shape[0] == 0:
                raise ValueError(f"gp(): the 'by' level {level!r} has no training rows")
            distinct = np.unique(subset, axis=0) if gr else subset
            dmax = 1.0
            if scale:
                dmax = math.sqrt(float(_squared_distances(distinct).max()))
                if not dmax > 0:
                    raise ValueError(
                        "Could not scale GP covariates. Please set 'scale' to FALSE in 'gp'."
                    )
            scaled = (_first_occurrences(subset) if gr else subset) / dmax
            if exact:
                centers = np.zeros(0)
                half_widths = np.zeros(0)
                frequencies = np.zeros((0, dimension))
            else:
                centers = scaled.mean(0)
                half_widths = _boundary(scaled - centers, boundary)
                frequencies = orders * math.pi / (2 * half_widths[None, :])
            sub_label = label + brms_rename(suffix)
            if isotropic:
                lscale_labels = (sub_label,)
                priors = (lscale_prior(scaled),)
            else:
                lscale_labels = tuple(sub_label + brms_rename(name) for name in names)
                priors = tuple(
                    lscale_prior(scaled[:, d : d + 1]) for d in range(dimension)
                )
            subterms.append(
                GaussianProcessSubterm(
                    block=GaussianProcessBlock(
                        label=sub_label,
                        lscale_labels=lscale_labels,
                        kernel=kernel,
                        frequencies=frequencies,
                        lscale_priors=priors,
                        inputs=scaled if exact else None,
                        gr=gr,
                        dmax=dmax,
                    ),
                    level=level,
                    dmax=dmax,
                    centers=centers,
                    boundary=half_widths,
                    orders=orders,
                )
            )
        return GaussianProcessState(
            covariates=names,
            by=by_name,
            by_kind=by_kind,
            levels=levels,
            subterms=tuple(subterms),
        )

    def _evaluate(
        self, x: npt.NDArray[np.float64], by: ColumnInput | None
    ) -> npt.NDArray[np.float64]:
        """Evaluate every sub-GP's retained basis at rows.

        Parameters
        ----------
        x : numpy.ndarray
            ``(rows, D)`` covariates.
        by : pandas.Series, numpy.ndarray, or None
            ``by`` column of the rows.

        Returns
        -------
        numpy.ndarray
            Concatenated basis columns.

        Raises
        ------
        ValueError
            If the covariates or ``by`` variable changed role or levels.
        """
        state = self.state
        assert state is not None
        if x.shape[1] != state.subterms[0].orders.shape[1]:
            raise ValueError("gp() covariates changed between training and evaluation")
        if (by is None) != (state.by_kind == "none"):
            raise ValueError("gp() by variable changed between training and evaluation")
        multiplier = np.ones(x.shape[0])
        labels = None
        if state.by_kind == "numeric":
            assert by is not None
            if _factor(by) is not None:
                raise ValueError("gp() numeric by variable became a factor")
            multiplier = _numeric(by)
        elif state.by_kind == "factor":
            assert by is not None
            factor = _factor(by)
            if factor is None:
                raise ValueError("gp() factor by variable became numeric")
            labels = factor[0]
            unknown = set(labels.tolist()) - set(state.levels)
            if unknown:
                raise ValueError(
                    f"gp() by variable has unseen levels {sorted(unknown)!r}"
                )
        columns = []
        for subterm in state.subterms:
            weights = multiplier
            if subterm.level is not None:
                assert labels is not None
                weights = (labels == subterm.level).astype(np.float64)
            if subterm.block.exact:
                columns.append(np.column_stack((weights, x)))
                continue
            values = np.zeros((x.shape[0], subterm.orders.shape[0]))
            active = weights != 0
            values[active] = subterm.basis(x[active]) * weights[active, None]
            columns.append(values)
        return np.column_stack(columns)


def _first_occurrences(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Keep the first row of each distinct covariate combination, in order.

    Parameters
    ----------
    x : numpy.ndarray
        ``(rows, D)`` covariates.

    Returns
    -------
    numpy.ndarray
        Distinct rows in order of first appearance, as brms's ``Jgp``.
    """
    _, first = np.unique(x, axis=0, return_index=True)
    return x[np.sort(first)]


def gp_namespace() -> dict[str, type]:
    """Return the formula callable constructing ``gp()`` terms.

    Returns
    -------
    dict[str, type]
        ``gp`` bound to `GaussianProcess`.
    """
    return {"gp": GaussianProcess}


def _log_spectral_density(
    block: GaussianProcessBlock,
    log_sdgp: LetExprNode,
    lscales: tuple[LetExprNode, ...],
    frequencies: tuple[LetExprNode, ...],
) -> LetExprNode:
    """Build the log spectral density at the block's frequencies.

    The densities are brms's ``spd_gp_*`` functions with per-covariate
    length scales ``l_d`` and squared frequencies ``w_d^2``:
    ``exp_quad`` has ``2 log sdgp + D/2 log(2 pi) + sum log l_d
    - 1/2 sum l_d^2 w_d^2``, and a Matérn kernel of smoothness ``nu`` has
    ``2 log sdgp + log C + sum log l_d - (nu + D/2) log(2 nu + sum l_d^2 w_d^2)``
    with ``C = 2^D pi^(D/2) Gamma(nu + D/2) (2 nu)^nu / Gamma(nu)``.

    Parameters
    ----------
    block : GaussianProcessBlock
        Kernel and dimension.
    log_sdgp : LetExprNode
        ``log(sdgp)``.
    lscales : tuple[LetExprNode, ...]
        Length scale per covariate.
    frequencies : tuple[LetExprNode, ...]
        Squared frequency vectors per covariate.

    Returns
    -------
    LetExprNode
        Log density, one entry per basis function.
    """
    dimension = block.dimension
    weighted = _add(
        *(
            _mul(_mul(lscale, lscale), frequency)
            for lscale, frequency in zip(lscales, frequencies, strict=True)
        )
    )
    log_lscales = _add(*(_call("log", lscale) for lscale in lscales))
    two = LetExprLiteral(value=2.0)
    if block.kernel == "exp_quad":
        constant = 0.5 * dimension * math.log(2 * math.pi)
        return _sub(
            _add(_mul(two, log_sdgp), LetExprLiteral(value=constant), log_lscales),
            _mul(LetExprLiteral(value=0.5), weighted),
        )
    nu = _SMOOTHNESS[block.kernel]
    constant = (
        dimension * math.log(2.0)
        + 0.5 * dimension * math.log(math.pi)
        + math.lgamma(nu + 0.5 * dimension)
        + nu * math.log(2 * nu)
        - math.lgamma(nu)
    )
    return _sub(
        _add(_mul(two, log_sdgp), LetExprLiteral(value=constant), log_lscales),
        _mul(
            LetExprLiteral(value=nu + 0.5 * dimension),
            _call("log", _add(LetExprLiteral(value=2 * nu), weighted)),
        ),
    )


def gp_block_steps(
    lens: FormulaToQVRModule,
    block: GaussianProcessBlock,
    name: str,
    prefix: str,
    sdgp_default: str,
) -> list[ProgramStep]:
    """Emit one GP's hyperparameters, innovations, and coefficients.

    For the approximation the coefficients are brms's
    ``rgp = sqrt(spd(slambda; sdgp, lscale)) .* zgp``, computed as
    ``exp(log(spd) / 2)`` so a vanishing density keeps finite gradients. For
    the exact GP they are the latent function at the distinct inputs (see
    `_exact_steps`). ``name_coefficients`` then enters the shared design
    contraction.

    Parameters
    ----------
    lens : FormulaToQVRModule
        Compiler with the shared prior emitter.
    block : GaussianProcessBlock
        Retained block.
    name : str
        Prefixed design identifier.
    prefix : str
        Owning predictor namespace.
    sdgp_default : str
        Default ``sdgp`` prior.

    Returns
    -------
    list[ProgramStep]
        Prior draws, innovations, and coefficient bindings.
    """
    sdgp, lscales = block.sites(prefix)
    steps: list[ProgramStep] = [lens._prior_draw(sdgp, sdgp_default)]
    for site, default in zip(lscales, block.lscale_priors, strict=True):
        steps.append(lens._prior_draw(site, default))
    if block.exact:
        return steps + _exact_steps(block, name, sdgp, lscales)
    raw = name + "_z"
    steps.append(
        SampleStep(
            vars=(raw,),
            morphism="Normal",
            args=(
                DrawArgNamed(parameter="loc", value=atom_to_draw_arg(0.0)),
                DrawArgNamed(parameter="scale", value=atom_to_draw_arg(1.0)),
            ),
            index=TypeName(name=name + "_Coordinate"),
        )
    )
    frequency_names = [
        _var(f"{name}_frequency_sq_{d + 1}") for d in range(block.dimension)
    ]
    lscale_values = (
        tuple(_var(lscales[0]) for _ in range(block.dimension))
        if block.isotropic
        else tuple(_var(site) for site in lscales)
    )
    log_spd = name + "_log_spd"
    steps.append(
        _let(
            log_spd,
            _log_spectral_density(
                block,
                _call("log", _var(sdgp)),
                lscale_values,
                tuple(frequency_names),
            ),
        )
    )
    steps.append(
        _let(
            name + "_coefficients",
            _mul(
                _call("exp", _mul(LetExprLiteral(value=0.5), _var(log_spd))),
                _var(raw),
            ),
        )
    )
    return steps


def gp_runtime_parameters(
    name: str, block: GaussianProcessBlock | None
) -> tuple[QiecValueParameter, ...]:
    """Declare a block's data inputs.

    Parameters
    ----------
    name : str
        Prefixed design identifier.
    block : GaussianProcessBlock or None
        Retained block, or None for a block that is not a GP.

    Returns
    -------
    tuple[QiecValueParameter, ...]
        ``name_frequency_sq_d`` vectors over the block's coordinate axis, or
        for an exact GP its distance and jitter matrices over the training
        inputs and, for prediction, the anchor over the fresh inputs.
    """
    if block is None:
        return ()
    if block.exact:
        axis = _input_axis(name, block)
        parameters = [
            QiecValueParameter(name=matrix, type_expr=_real_tensor(axis, axis))
            for matrix in (*_distance_names(name, block), name + "_jitter")
        ]
        if block.prediction_inputs is not None:
            parameters.append(
                QiecValueParameter(
                    name=name + "_anchor",
                    type_expr=_real_tensor(name + "_Coordinate"),
                )
            )
        return tuple(parameters)
    return tuple(
        QiecValueParameter(
            name=f"{name}_frequency_sq_{d + 1}",
            type_expr=QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(QiecTypeName(name="Real"),),
                indices=(
                    QiecShapeIndex(
                        dimensions=(QiecIndexName(name=name + "_Coordinate"),)
                    ),
                ),
            ),
        )
        for d in range(block.dimension)
    )


def gp_observations(
    name: str, block: GaussianProcessBlock | None
) -> dict[str, torch.Tensor]:
    """Bind a block's retained data inputs in double precision.

    The approximation binds its squared frequencies, and the exact GP the
    pairwise distances of its training inputs and brms's ``1e-12`` jitter.
    Both are retained training structure, so fresh rows bind the same values.

    Parameters
    ----------
    name : str
        Prefixed design identifier.
    block : GaussianProcessBlock or None
        Retained block, or None for a block that is not a GP.

    Returns
    -------
    dict[str, torch.Tensor]
        Inputs by name.
    """
    if block is None:
        return {}
    if block.exact:
        assert block.inputs is not None
        values = {
            matrix: torch.as_tensor(distance, dtype=torch.float64)
            for matrix, distance in zip(
                _distance_names(name, block),
                _distances(block, block.inputs, block.inputs),
                strict=True,
            )
        }
        values[name + "_jitter"] = EXACT_JITTER * torch.eye(
            len(block.inputs), dtype=torch.float64
        )
        return values
    return {
        f"{name}_frequency_sq_{d + 1}": torch.as_tensor(
            block.frequencies[:, d] ** 2, dtype=torch.float64
        )
        for d in range(block.dimension)
    }


def _real_tensor(*axes: str) -> QiecTypeApplication:
    """Spell a real tensor type over named finite axes.

    Parameters
    ----------
    *axes : str
        Axis names, outermost first.

    Returns
    -------
    QiecTypeApplication
        ``Tensor[Real]([axes])``.
    """
    return QiecTypeApplication(
        constructor="Tensor",
        static_arguments=(QiecTypeName(name="Real"),),
        indices=(
            QiecShapeIndex(dimensions=tuple(QiecIndexName(name=a) for a in axes)),
        ),
    )


def _input_axis(name: str, block: GaussianProcessBlock) -> str:
    """Name the axis of an exact GP's training inputs.

    The training inputs are the design coordinates while fitting. A block
    specialized to prediction rows has the fresh inputs as coordinates and
    keeps its training inputs on their own axis.

    Parameters
    ----------
    name : str
        Prefixed design identifier.
    block : GaussianProcessBlock
        Exact block.

    Returns
    -------
    str
        QVR axis name.
    """
    if block.prediction_inputs is None:
        return name + "_Coordinate"
    return name + "_Input"


def _distance_names(name: str, block: GaussianProcessBlock) -> tuple[str, ...]:
    """Name an exact block's distance matrices.

    Parameters
    ----------
    name : str
        Prefixed design identifier.
    block : GaussianProcessBlock
        Exact block.

    Returns
    -------
    tuple[str, ...]
        One Euclidean distance for an isotropic GP, one per covariate
        otherwise.
    """
    count = 1 if block.isotropic else block.dimension
    return tuple(f"{name}_distance_{d + 1}" for d in range(count))


def _distances(
    block: GaussianProcessBlock,
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Pairwise distances between scaled inputs, as brms's ``cov_gp`` reads them.

    Costs O(rows times columns times D).

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block, which fixes isotropy.
    left, right : numpy.ndarray
        ``(rows, D)`` and ``(columns, D)`` scaled inputs.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The Euclidean distance over all covariates for an isotropic GP, or
        the absolute difference of each covariate otherwise.
    """
    differences = left[:, None, :] - right[None, :, :]
    if block.isotropic:
        return (np.sqrt((differences**2).sum(-1)),)
    return tuple(np.abs(differences[:, :, d]) for d in range(differences.shape[-1]))


def _correlation_expression(
    kernel: GpKernel, distance: LetExprNode, lscale: LetExprNode
) -> LetExprNode:
    """Build brms's unit-variance kernel of one distance matrix.

    ``exp_quad`` is ``exp(-r^2 / (2 l^2))``, ``exponential``
    ``exp(-r / l)``, ``matern32`` ``(1 + sqrt(3) r / l) exp(-sqrt(3) r / l)``,
    and ``matern52`` ``(1 + sqrt(5) r / l + 5 r^2 / (3 l^2)) exp(-sqrt(5) r / l)``,
    as in brms's ``cov_gp_*`` functions and Stan's ``gp_*_cov``.

    Parameters
    ----------
    kernel : GpKernel
        Kernel name.
    distance : LetExprNode
        Distance matrix ``r``.
    lscale : LetExprNode
        Length scale ``l``.

    Returns
    -------
    LetExprNode
        Elementwise kernel values.
    """
    ratio = _div(distance, lscale)
    if kernel == "exp_quad":
        return _call("exp", _mul(LetExprLiteral(value=-0.5), _mul(ratio, ratio)))
    if kernel == "exponential":
        return _call("exp", _mul(LetExprLiteral(value=-1.0), ratio))
    root = math.sqrt(3.0) if kernel == "matern32" else math.sqrt(5.0)
    polynomial = _add(
        LetExprLiteral(value=1.0), _mul(LetExprLiteral(value=root), ratio)
    )
    if kernel == "matern52":
        polynomial = _add(
            polynomial, _mul(LetExprLiteral(value=5.0 / 3.0), _mul(ratio, ratio))
        )
    return _mul(polynomial, _call("exp", _mul(LetExprLiteral(value=-root), ratio)))


def _exact_steps(
    block: GaussianProcessBlock, name: str, sdgp: str, lscales: tuple[str, ...]
) -> list[ProgramStep]:
    """Emit brms's exact GP from its sampled hyperparameters.

    The steps are brms's ``gp_<cov>`` Stan function: the kernel matrix
    ``sdgp^2 prod_d k_d(r_d / l_d)`` of the training inputs (one factor of
    the Euclidean distance for an isotropic GP), plus ``1e-12`` on the
    diagonal, its lower Cholesky factor, and ``f = L zgp`` with
    ``zgp ~ N(0, I)``. A block specialized to prediction rows then draws
    ``f`` at the fresh inputs from `GaussianProcessConditional`.

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block.
    name : str
        Prefixed design identifier.
    sdgp : str
        Marginal standard deviation site.
    lscales : tuple[str, ...]
        Length-scale sites.

    Returns
    -------
    list[ProgramStep]
        Innovations, kernel, factor, and ``name_coefficients``.
    """
    raw = name + "_z"
    steps: list[ProgramStep] = [
        SampleStep(
            vars=(raw,),
            morphism="Normal",
            args=(
                DrawArgNamed(parameter="loc", value=atom_to_draw_arg(0.0)),
                DrawArgNamed(parameter="scale", value=atom_to_draw_arg(1.0)),
            ),
            index=TypeName(name=_input_axis(name, block)),
        )
    ]
    correlation: LetExprNode | None = None
    for distance, lscale in zip(
        _distance_names(name, block),
        lscales if not block.isotropic else lscales[:1],
        strict=True,
    ):
        factor = _correlation_expression(block.kernel, _var(distance), _var(lscale))
        correlation = factor if correlation is None else _mul(correlation, factor)
    assert correlation is not None
    steps.append(
        _let(
            name + "_kernel",
            _add(
                _mul(_mul(_var(sdgp), _var(sdgp)), correlation),
                _var(name + "_jitter"),
            ),
        )
    )
    steps.append(_let(name + "_factor", _call("cholesky", _var(name + "_kernel"))))
    if block.prediction_inputs is None:
        steps.append(
            _let(
                name + "_coefficients",
                _call("matmul", _var(name + "_factor"), _var(raw)),
            )
        )
        return steps
    steps.append(
        _let(name + "_values", _call("matmul", _var(name + "_factor"), _var(raw)))
    )
    arguments = {"values": name + "_values", "scale": sdgp}
    arguments.update(
        {
            f"lscale_{d + 1}": site
            for d, site in enumerate(lscales if not block.isotropic else lscales[:1])
        }
    )
    arguments["anchor"] = name + "_anchor"
    steps.append(
        SampleStep(
            vars=(name + "_coefficients",),
            morphism=conditional_family_name(name),
            args=tuple(
                DrawArgNamed(parameter=key, value=atom_to_draw_arg(value))
                for key, value in arguments.items()
            ),
        )
    )
    return steps


def gp_axes(
    name: str, block: GaussianProcessBlock | None
) -> tuple[tuple[str, int], ...]:
    """List the finite axes a block declares beside its design coordinates.

    Parameters
    ----------
    name : str
        Prefixed design identifier.
    block : GaussianProcessBlock or None
        Retained block, or None for a block that is not a GP.

    Returns
    -------
    tuple[tuple[str, int], ...]
        The training-input axis of an exact block specialized to prediction
        rows, and nothing otherwise.
    """
    if block is None or block.prediction_inputs is None:
        return ()
    assert block.inputs is not None
    return ((name + "_Input", len(block.inputs)),)


def gp_rows(
    raw: npt.NDArray[np.float64], block: GaussianProcessBlock
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Read an exact sub-GP's distinct inputs and row design, as brms's ``data_gp``.

    The active rows are those with a nonzero contrast value (brms's
    ``Igp``). With ``gr``, rows sharing covariate values share one input, in
    order of first appearance (brms's ``Jgp``); otherwise each active row is
    its own input. Inputs are divided by the retained ``dmax``. The design
    holds each active row's contrast value (brms's ``Cgp``) in the column of
    its input. Costs O(rows times D log rows).

    Parameters
    ----------
    raw : numpy.ndarray
        ``(rows, 1 + D)`` contrast values and covariates, as the exact
        `GaussianProcess` evaluates them.
    block : GaussianProcessBlock
        Exact block with its grouping and scaling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(u, D)`` scaled distinct inputs and the ``(rows, u)`` design.
    """
    weights = raw[:, 0]
    active = np.flatnonzero(weights != 0)
    selected = raw[active, 1:]
    if block.gr and len(active):
        _, first, inverse = np.unique(
            selected, axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        points = selected[first[order]]
        codes = rank[np.asarray(inverse).reshape(-1)]
    else:
        points = selected
        codes = np.arange(len(active))
    design = np.zeros((raw.shape[0], len(points)))
    design[active, codes] = weights[active]
    return points / block.dmax, design


def _correlation(
    kernel: GpKernel, distance: torch.Tensor, lscale: torch.Tensor
) -> torch.Tensor:
    """Evaluate brms's unit-variance kernel natively.

    Parameters
    ----------
    kernel : GpKernel
        Kernel name.
    distance : torch.Tensor
        Distances with trailing matrix axes.
    lscale : torch.Tensor
        Length scales broadcast against the distances.

    Returns
    -------
    torch.Tensor
        Kernel values, as `_correlation_expression` spells them.
    """
    ratio = distance / lscale
    if kernel == "exp_quad":
        return torch.exp(-0.5 * ratio * ratio)
    if kernel == "exponential":
        return torch.exp(-ratio)
    root = math.sqrt(3.0) if kernel == "matern32" else math.sqrt(5.0)
    polynomial = 1 + root * ratio
    if kernel == "matern52":
        polynomial = polynomial + 5.0 / 3.0 * ratio * ratio
    return polynomial * torch.exp(-root * ratio)


def exact_covariance(
    block: GaussianProcessBlock,
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
    sdgp: torch.Tensor,
    lscales: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Evaluate brms's ``cov_gp`` between two sets of scaled inputs.

    Costs O(rows times columns times D) per batch element.

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block: kernel and isotropy.
    left, right : numpy.ndarray
        ``(rows, D)`` and ``(columns, D)`` scaled inputs.
    sdgp : torch.Tensor
        Marginal standard deviations with any batch shape.
    lscales : tuple[torch.Tensor, ...]
        One length scale for an isotropic GP, one per covariate otherwise,
        each with the batch shape.

    Returns
    -------
    torch.Tensor
        Covariances with the batch shape followed by ``(rows, columns)``.
    """
    result = (sdgp.double() ** 2)[..., None, None]
    for distance, lscale in zip(_distances(block, left, right), lscales, strict=True):
        result = result * _correlation(
            block.kernel,
            torch.as_tensor(distance, dtype=torch.float64),
            lscale.double()[..., None, None],
        )
    return result


def _covariance_factor(matrix: torch.Tensor, nugget: float) -> torch.Tensor:
    """Factor covariance matrices, refusing as brms's ``try_nug`` does.

    Parameters
    ----------
    matrix : torch.Tensor
        Symmetric matrices with leading batch axes.
    nugget : float
        Diagonal already added, named in the error.

    Returns
    -------
    torch.Tensor
        Lower Cholesky factors.

    Raises
    ------
    ValueError
        If a matrix is not numerically positive definite.
    """
    factor, info = torch.linalg.cholesky_ex(matrix)
    if bool((info != 0).any()):
        raise ValueError(
            "The Gaussian process covariance matrix is not positive definite. "
            f"This occurs for numerical reasons. A diagonal of {nugget:g} was added."
        )
    return factor


def exact_values(
    block: GaussianProcessBlock,
    sdgp: torch.Tensor,
    lscales: tuple[torch.Tensor, ...],
    innovations: torch.Tensor,
) -> torch.Tensor:
    """Compute the latent function at the training inputs, brms's ``.predictor_gp_old``.

    Costs O(u^3) per batch element.

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block.
    sdgp : torch.Tensor
        Marginal standard deviations with any batch shape.
    lscales : tuple[torch.Tensor, ...]
        Length scales with the batch shape.
    innovations : torch.Tensor
        ``zgp`` with the batch shape followed by ``u``.

    Returns
    -------
    torch.Tensor
        ``cholesky(K + 1e-12 I) zgp`` with the shape of ``innovations``.
    """
    assert block.inputs is not None
    covariance = exact_covariance(block, block.inputs, block.inputs, sdgp, lscales)
    covariance = covariance + EXACT_JITTER * torch.eye(
        len(block.inputs), dtype=torch.float64
    )
    factor = _covariance_factor(covariance, EXACT_JITTER)
    return (factor @ innovations.double()[..., None])[..., 0]


def conditional_moments(
    block: GaussianProcessBlock,
    values: torch.Tensor,
    sdgp: torch.Tensor,
    lscales: tuple[torch.Tensor, ...],
    points: npt.NDArray[np.float64],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give the Gaussian law of the latent function at fresh inputs.

    brms's ``.predictor_gp_new``: with ``K = K(X, X) + 1e-12 I = L L^T``,
    ``K_* = K(X, X_*)``, and ``V = L^{-1} K_*``, the mean is
    ``V^T L^{-1} f`` and the covariance ``K(X_*, X_*) - V^T V + 1e-8 I``.
    Costs O(u^3 + u^2 m + u m^2 + m^3) per batch element for ``u`` training
    and ``m`` fresh inputs.

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block with its training inputs.
    values : torch.Tensor
        Latent function at the training inputs, batch shape then ``u``.
    sdgp : torch.Tensor
        Marginal standard deviations.
    lscales : tuple[torch.Tensor, ...]
        Length scales.
    points : numpy.ndarray
        ``(m, D)`` scaled fresh inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean, batch shape then ``m``, and the lower Cholesky factor of the
        covariance, batch shape then ``(m, m)``.

    Raises
    ------
    ValueError
        If a covariance matrix is not numerically positive definite.
    """
    assert block.inputs is not None
    training = exact_covariance(block, block.inputs, block.inputs, sdgp, lscales)
    training = training + EXACT_JITTER * torch.eye(
        len(block.inputs), dtype=torch.float64
    )
    factor = _covariance_factor(training, EXACT_JITTER)
    cross = exact_covariance(block, block.inputs, points, sdgp, lscales)
    solved = torch.linalg.solve_triangular(factor, cross, upper=False)
    whitened = torch.linalg.solve_triangular(
        factor, values.double()[..., None], upper=False
    )
    mean = (solved.transpose(-1, -2) @ whitened)[..., 0]
    covariance = (
        exact_covariance(block, points, points, sdgp, lscales)
        - solved.transpose(-1, -2) @ solved
        + PREDICTION_NUGGET * torch.eye(len(points), dtype=torch.float64)
    )
    return mean, _covariance_factor(covariance, PREDICTION_NUGGET)


class GaussianProcessConditional(Distribution):
    """The latent function of an exact GP at fresh inputs given its training values.

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block specialized to prediction rows.
    values : torch.Tensor
        Latent function at the training inputs.
    sdgp : torch.Tensor
        Marginal standard deviation.
    lscales : tuple[torch.Tensor, ...]
        Length scales.
    """

    has_rsample = True

    def __init__(
        self,
        block: GaussianProcessBlock,
        values: torch.Tensor,
        sdgp: torch.Tensor,
        lscales: tuple[torch.Tensor, ...],
    ) -> None:
        assert block.prediction_inputs is not None
        batch = torch.broadcast_shapes(
            values.shape[:-1], sdgp.shape, *(lscale.shape for lscale in lscales)
        )
        self._loc, self._scale_tril = conditional_moments(
            block,
            values.expand(*batch, values.shape[-1]),
            sdgp.expand(batch),
            tuple(lscale.expand(batch) for lscale in lscales),
            np.asarray(block.prediction_inputs),
        )
        super().__init__(
            batch,
            torch.Size((len(block.prediction_inputs),)),
            validate_args=False,
        )

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Declare no torch-validated arguments.

        Returns
        -------
        dict[str, constraints.Constraint]
            An empty mapping.
        """
        return {}

    @property
    def support(self) -> constraints.Constraint:
        """Give the real vectors.

        Returns
        -------
        constraints.Constraint
            Support of the law.
        """
        return constraints.real_vector

    @property
    def mean(self) -> torch.Tensor:
        """Give the conditional mean.

        Returns
        -------
        torch.Tensor
            Mean at the fresh inputs.
        """
        return self._loc

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw the latent function at the fresh inputs.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Draws with the sample, batch, and event shapes.
        """
        shape = self._extended_shape(sample_shape)
        noise = torch.randn(shape, dtype=self._loc.dtype)
        return self._loc + (self._scale_tril @ noise[..., None])[..., 0]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score latent values at the fresh inputs.

        Parameters
        ----------
        value : torch.Tensor
            Values at the fresh inputs.

        Returns
        -------
        torch.Tensor
            Multivariate Normal log density.
        """
        return MultivariateNormal(self._loc, scale_tril=self._scale_tril).log_prob(
            value.double()
        )


class _ConditionalFactory(ValueModel):
    """Build `GaussianProcessConditional` laws of one specialized block.

    Parameters
    ----------
    block : GaussianProcessBlock
        Exact block specialized to prediction rows.
    """

    block: GaussianProcessBlock = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    def __call__(self, parameters: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct the conditional law from the program's parameters.

        Parameters
        ----------
        parameters : Mapping[str, torch.Tensor]
            ``values``, ``scale`` (``sdgp``), ``lscale_d``, and ``anchor``.

        Returns
        -------
        Distribution
            The conditional law.
        """
        count = 1 if self.block.isotropic else self.block.dimension
        return GaussianProcessConditional(
            self.block,
            parameters["values"],
            parameters["scale"],
            tuple(parameters[f"lscale_{d + 1}"] for d in range(count)),
        )


def conditional_family_name(name: str) -> str:
    """Name the conditional law of one exact block.

    Parameters
    ----------
    name : str
        Prefixed design identifier.

    Returns
    -------
    str
        Constructor name, unique per block.
    """
    return "GaussianProcessConditional_" + name


def gp_conditional_catalog(
    blocks: Iterable[tuple[str, GaussianProcessBlock]],
) -> FamilyCatalog | None:
    """Register the conditional law of every exact block specialized to prediction.

    Parameters
    ----------
    blocks : Iterable[tuple[str, GaussianProcessBlock]]
        Prefixed design identifiers and their blocks.

    Returns
    -------
    FamilyCatalog or None
        The laws, or None when no block is specialized.
    """
    catalog: FamilyCatalog | None = None
    for name, block in blocks:
        if block.prediction_inputs is None:
            continue
        count = 1 if block.isotropic else block.dimension
        record = DistributionFamily(
            name=conditional_family_name(name),
            parameters=(
                FamilyParameter(
                    name="values",
                    constraint="real",
                    rank=1,
                    event_dims=(None,),
                    scalar_broadcast=False,
                ),
                FamilyParameter(name="scale", constraint="positive", rank=0),
                *(
                    FamilyParameter(
                        name=f"lscale_{d + 1}", constraint="positive", rank=0
                    )
                    for d in range(count)
                ),
                FamilyParameter(
                    name="anchor",
                    constraint="real",
                    rank=1,
                    event_dims=(0,),
                    scalar_broadcast=False,
                ),
            ),
            support="real",
            element=REAL,
            event_rank=1,
            event_sources=("anchor",),
            discrete=False,
            reparameterizable=True,
            compositional=False,
        )
        entry = FamilyCatalog().with_family(record, _ConditionalFactory(block=block))
        catalog = entry if catalog is None else catalog.with_catalog(entry)
    return catalog


def gp_prediction(
    items: Iterable[tuple[str, str, Formula]],
    designs: Mapping[str, DesignMatrices],
    frame: pd.DataFrame,
) -> tuple[dict[str, Formula], dict[str, torch.Tensor], tuple[str, ...]]:
    """Specialize every exact GP block to prediction rows, as brms predicts them.

    Each exact block reads its fresh inputs and design from the rows
    (`gp_rows`) and draws the latent function there from
    `GaussianProcessConditional`, given the replayed training values, under
    each posterior draw. A sub-GP with no active row keeps one zero design
    column at its first training input, so it adds nothing.

    Parameters
    ----------
    items : Iterable[tuple[str, str, Formula]]
        Fitted predictor name, input prefix, and formula of each predictor.
    designs : Mapping[str, DesignMatrices]
        Retained evaluators by predictor name.
    frame : pandas.DataFrame
        Prediction rows.

    Returns
    -------
    tuple[dict[str, Formula], dict[str, torch.Tensor], tuple[str, ...]]
        The formulas with specialized blocks, the fresh designs and anchors,
        and the latent sites to draw afresh; empty when no exact block exists.
    """
    formulas: dict[str, Formula] = {}
    values: dict[str, torch.Tensor] = {}
    fresh: list[str] = []
    for parameter, prefix, formula in items:
        if not any(
            term.gp is not None and term.gp.exact for term in formula.design_terms
        ):
            formulas[parameter] = formula
            continue
        design = designs[parameter]
        assert design.common is not None
        evaluated = design.common.evaluate_new_data(frame)
        assert evaluated.design_matrix is not None
        terms = []
        for term in formula.design_terms:
            block = term.gp
            if block is None or not block.exact:
                terms.append(term)
                continue
            assert block.inputs is not None
            columns = evaluated.design_matrix[:, design.common.slices[term.term]]
            raw = np.asarray(columns, dtype=np.float64)[:, list(term.column_indices)]
            points, rows = gp_rows(raw, block)
            if not len(points):
                points, rows = block.inputs[:1], np.zeros((len(frame), 1))
            name = prefix + term.qvr_name
            terms.append(
                term.with_(data=rows, gp=block.with_(prediction_inputs=points))
            )
            values[name] = torch.as_tensor(rows, dtype=torch.float64)
            values[name + "_anchor"] = torch.zeros(len(points), dtype=torch.float64)
            fresh.append(name + "_coefficients")
        formulas[parameter] = formula.with_(design_terms=tuple(terms))
    return formulas, values, tuple(fresh)


def exact_contribution(
    draws: Mapping[str, torch.Tensor],
    block: GaussianProcessBlock,
    name: str,
    prefix: str,
    raw: npt.NDArray[np.float64],
    generator: torch.Generator,
) -> torch.Tensor:
    """Evaluate one exact sub-GP's contribution at rows for posterior draws.

    The latent function is drawn at the rows' distinct inputs from its law
    given the training values of each draw, as brms's ``conditional_effects``
    does through ``.predictor_gp_new``.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained draws with chain and draw axes.
    block : GaussianProcessBlock
        Exact block.
    name : str
        Prefixed design identifier.
    prefix : str
        Owning predictor namespace.
    raw : numpy.ndarray
        ``(rows, 1 + D)`` contrast values and covariates.
    generator : torch.Generator
        Seeded source of the conditional draws.

    Returns
    -------
    torch.Tensor
        Contributions with shape ``(chain, draw, rows)``.
    """
    points, rows = gp_rows(raw, block)
    sdgp_site, lscale_sites = block.sites(prefix)
    sdgp = draws[sdgp_site].double()
    lscales = tuple(
        draws[site].double()
        for site in (lscale_sites[:1] if block.isotropic else lscale_sites)
    )
    if not len(points):
        return sdgp.new_zeros((*sdgp.shape, raw.shape[0]))
    values = exact_values(block, sdgp, lscales, draws[name + "_z"])
    mean, factor = conditional_moments(block, values, sdgp, lscales, points)
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    latent = mean + (factor @ noise[..., None])[..., 0]
    return latent @ torch.as_tensor(rows, dtype=torch.float64).T


def gp_prior_sites(
    block: GaussianProcessBlock, prefix: str
) -> tuple[tuple[str, str, str], ...]:
    """List ``(site, class, coef)`` of a block's hyperparameters.

    Parameters
    ----------
    block : GaussianProcessBlock
        Retained block.
    prefix : str
        Owning predictor namespace.

    Returns
    -------
    tuple[tuple[str, str, str], ...]
        brms classes ``sdgp`` and ``lscale`` with their coefficient labels.
    """
    sdgp, lscales = block.sites(prefix)
    return ((sdgp, "sdgp", block.label),) + tuple(
        (site, "lscale", label)
        for site, label in zip(lscales, block.lscale_labels, strict=True)
    )


def spectral_density(
    block: GaussianProcessBlock, sdgp: torch.Tensor, lscales: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Evaluate brms's spectral density at the block's frequencies.

    Parameters
    ----------
    block : GaussianProcessBlock
        Kernel and frequencies.
    sdgp : torch.Tensor
        Marginal standard deviations with any batch shape.
    lscales : tuple[torch.Tensor, ...]
        One length scale per covariate (repeated for an isotropic GP).

    Returns
    -------
    torch.Tensor
        Densities with the batch shape followed by the basis axis.
    """
    squared = torch.as_tensor(block.frequencies**2, dtype=sdgp.dtype)
    dimension = block.dimension
    weighted = sum(
        lscale[..., None] ** 2 * squared[:, d] for d, lscale in enumerate(lscales)
    )
    log_lscales = sum(lscale.log() for lscale in lscales)
    assert isinstance(weighted, torch.Tensor) and isinstance(log_lscales, torch.Tensor)
    if block.kernel == "exp_quad":
        log_spd = (
            2 * sdgp.log()[..., None]
            + 0.5 * dimension * math.log(2 * math.pi)
            + log_lscales[..., None]
            - 0.5 * weighted
        )
    else:
        nu = _SMOOTHNESS[block.kernel]
        constant = (
            dimension * math.log(2.0)
            + 0.5 * dimension * math.log(math.pi)
            + math.lgamma(nu + 0.5 * dimension)
            + nu * math.log(2 * nu)
            - math.lgamma(nu)
        )
        log_spd = (
            2 * sdgp.log()[..., None]
            + constant
            + log_lscales[..., None]
            - (nu + 0.5 * dimension) * torch.log(2 * nu + weighted)
        )
    return log_spd.exp()


def gp_block_coefficients(
    draws: dict[str, torch.Tensor], block: GaussianProcessBlock, name: str, prefix: str
) -> torch.Tensor:
    """Assemble ``sqrt(spd) * z`` for posterior draws of one block.

    Parameters
    ----------
    draws : dict[str, torch.Tensor]
        Constrained draws with chain and draw axes.
    block : GaussianProcessBlock
        Retained block.
    name : str
        Prefixed design identifier.
    prefix : str
        Owning predictor namespace.

    Returns
    -------
    torch.Tensor
        Coefficients with shape ``(chain, draw, m)``.
    """
    sdgp, lscales = block.sites(prefix)
    lengths = tuple(
        draws[lscales[0] if block.isotropic else lscales[d]].double()
        for d in range(block.dimension)
    )
    spd = spectral_density(block, draws[sdgp].double(), lengths)
    return spd.sqrt() * draws[name + "_z"].double()


__all__: list[str] = []
