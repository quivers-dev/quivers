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
from functools import reduce
import math
from typing import TYPE_CHECKING, Literal, Self

import didactic.api as dx
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.optimize as optimize
import scipy.special as special
import torch
from torch.distributions import Distribution, MultivariateNormal, constraints
from torch.types import _size

from quivers._positions import demand
from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value
from quivers.dsl.ast_nodes import (
    DrawArgNamed,
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
    _draw,
    _factor,
    _index,
    _let,
    _mul,
    _sub,
    _var,
)
from quivers.formulas._term import (
    COORDINATE,
    AxisDecl,
    EffectSource,
    PopulationCoefficient,
    VariableDecl,
    coordinate_scopes,
    ColumnName,
    Contribution,
    Emission,
    InputDecl,
    OutputAxis,
    SiteDecl,
    at_coordinate,
    design_declarations,
    flat_index,
    over_output,
    read_only,
)
from quivers.formulas.formula_parts import FormulaPart
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

if TYPE_CHECKING:
    from quivers.formulas._name_scope import NameScope
    from quivers.formulas._row_source import PredictorRows


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

    def specific(self) -> GaussianProcessBlock:
        """Return the block of one process per ordinal transition.

        Returns
        -------
        GaussianProcessBlock
            The same block, with labels distinct from those of a block
            shared by the transitions.
        """
        return self.with_(
            label="cs" + self.label,
            lscale_labels=tuple("cs" + label for label in self.lscale_labels),
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


def _by_factor(
    value: ColumnInput,
) -> tuple[npt.NDArray[np.str_], tuple[str, ...]] | None:
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
            factor = _by_factor(by)
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
            if _by_factor(by) is not None:
                raise ValueError("gp() numeric by variable became a factor")
            multiplier = _numeric(by)
        elif state.by_kind == "factor":
            assert by is not None
            factor = _by_factor(by)
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
    demand(
        info == 0,
        lambda: ValueError(
            "The Gaussian process covariance matrix is not positive definite. "
            f"This occurs for numerical reasons. A diagonal of {nugget:g} was added."
        ),
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


class _ProcessTerm(ValueModel):
    """The hyperparameters and design of one Gaussian process block.

    Parameters
    ----------
    label : str
        Authored term.
    name : str
        Prefixed design identifier.
    prefix : str
        Owning predictor namespace.
    block : GaussianProcessBlock
        Retained block.
    matrix : np.ndarray
        Rows by coordinates; read-only.
    columns : tuple[int, ...]
        Columns of the evaluated term the block reads.
    scale_prior : str
        Default ``sdgp`` prior.
    output : OutputAxis or None
        Axis of the coefficient vectors.
    """

    label: str
    name: str
    prefix: str
    block: GaussianProcessBlock = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    matrix: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(read_only))
    columns: tuple[int, ...]
    scale_prior: str
    output: OutputAxis | None = None

    def column_names(self) -> tuple[ColumnName, ...]:
        """List the data columns the term reads under their own names.

        Returns
        -------
        tuple[ColumnName, ...]
            Empty: the inputs are named after the term.
        """
        return ()

    def variables(self, scope: NameScope) -> tuple[VariableDecl, ...]:
        """Name the term's draws as brms does.

        Parameters
        ----------
        scope : NameScope
            Names of the term's predictor.

        Returns
        -------
        tuple[VariableDecl, ...]
            The hyperparameters and the standardized innovations.
        """
        sdgp, lscales = self.block.sites(self.prefix)
        return tuple(
            declaration
            for index, part in coordinate_scopes(self.output, scope)
            for declaration in (
                VariableDecl(
                    name=f"sdgp_{part.infix}{brms_rename(self.block.label)}",
                    site=sdgp,
                    kind="special",
                    index=index,
                    parameter=part.position,
                ),
                *(
                    VariableDecl(
                        name=f"lscale_{part.infix}{brms_rename(label)}",
                        site=site,
                        kind="special",
                        index=index,
                        parameter=part.position,
                    )
                    for site, label in zip(
                        lscales, self.block.lscale_labels, strict=True
                    )
                ),
                VariableDecl(
                    name=f"zgp_{part.infix}{self.block.label}",
                    site=self.name + "_z",
                    kind="special",
                    index=index,
                    parameter=part.position,
                    flat=self.output.size if self.output is not None else None,
                ),
            )
        )

    def population(self) -> tuple[PopulationCoefficient, ...]:
        """List the term's population-level coefficients.

        Returns
        -------
        tuple[PopulationCoefficient, ...]
            Empty.
        """
        return ()

    def smooth_part(self, label: str) -> Self | None:
        """Return the term's share of one smooth or Gaussian process.

        Parameters
        ----------
        label : str
            Authored smooth term.

        Returns
        -------
        Self or None
            The process.
        """
        return self if self.label == label else None

    def effects(self) -> tuple[EffectSource, ...]:
        """List the design terms whose variables brms plots.

        Returns
        -------
        tuple[EffectSource, ...]
            The term.
        """
        return (EffectSource(label=self.label, kind="gp"),)

    def contributions(self) -> tuple[Contribution, ...]:
        """List the bound summands of the term.

        Returns
        -------
        tuple[Contribution, ...]
            The row contribution.
        """
        return (
            Contribution(
                name=self.name + "_per_row",
                rows=True,
                coordinates=self.output is not None,
                part=FormulaPart(
                    field="design_terms", key=self.name.removeprefix(self.prefix)
                ),
            ),
        )

    def hyperparameters(self) -> tuple[SiteDecl, ...]:
        """Declare the ``sdgp`` and ``lscale`` sites.

        Returns
        -------
        tuple[SiteDecl, ...]
            The marginal standard deviation, then each length scale, per
            coordinate of the output axis.
        """
        sdgp, lscales = self.block.sites(self.prefix)
        output = self.output
        axes = (output.name,) if output is not None else ()
        labels = output.labels if output is not None else ()
        parameters = output.parameters if output is not None else ()
        return (
            SiteDecl(
                name=sdgp,
                class_="sdgp",
                default=self.scale_prior,
                coef=self.block.label,
                axes=axes,
                component_labels=labels,
                parameters=parameters,
                lower=0.0,
            ),
            *(
                SiteDecl(
                    name=site,
                    class_="lscale",
                    default=default,
                    coef=label,
                    axes=axes,
                    component_labels=labels,
                    parameters=parameters,
                    lower=0.0,
                )
                for site, default, label in zip(
                    lscales,
                    self.block.lscale_priors,
                    self.block.lscale_labels,
                    strict=True,
                )
            ),
        )

    def _innovation_axis(self, elements: AxisDecl) -> AxisDecl:
        """Declare the axis of the innovations.

        Parameters
        ----------
        elements : AxisDecl
            Axis of the elements with one innovation per output coordinate.

        Returns
        -------
        AxisDecl
            The axis itself for a scalar process; otherwise a flat axis that
            holds the coordinates of each element consecutively.
        """
        if self.output is None:
            return elements
        return AxisDecl(
            name=self.name + "_InnovationCoordinate",
            size=elements.size * self.output.size,
        )

    def _innovation_site(self, elements: AxisDecl) -> SiteDecl:
        """Declare the innovations, whose law the term fixes.

        Parameters
        ----------
        elements : AxisDecl
            Axis of the elements with one innovation per output coordinate.

        Returns
        -------
        SiteDecl
            The site.
        """
        return SiteDecl(
            name=self.name + "_z",
            class_="zgp",
            axes=(self._innovation_axis(elements).name,),
            overridable=False,
        )

    def _innovations(self, elements: AxisDecl) -> tuple[ProgramStep, ...]:
        """Draw the standard-normal innovations.

        Parameters
        ----------
        elements : AxisDecl
            Axis of the elements with one innovation per output coordinate.

        Returns
        -------
        tuple[ProgramStep, ...]
            The draw and, for a vector of processes, the binding of its
            innovations by coordinate and element.
        """
        raw, output = self.name + "_z", self.output
        draw = _draw(
            raw,
            "Normal",
            (0.0, 1.0),
            index=TypeName(name=self._innovation_axis(elements).name),
        )
        if output is None:
            return (draw,)
        element = "process_element"
        return (
            draw,
            _let(
                raw + "_by_coordinate",
                over_output(
                    output,
                    _index(raw, flat_index(output, _var(element))),
                    (element, elements.name),
                ),
            ),
        )

    def _innovation(self) -> LetExprNode:
        """Read the innovations at the bound output coordinate.

        Returns
        -------
        LetExprNode
            Vector over the elements.
        """
        raw = self.name + "_z"
        if self.output is None:
            return _var(raw)
        return at_coordinate(raw + "_by_coordinate", self.output)

    def _lengths(self) -> tuple[LetExprNode, ...]:
        """Read the length scales at the bound output coordinate.

        Returns
        -------
        tuple[LetExprNode, ...]
            One per covariate; the shared one of an isotropic process.
        """
        _, lscales = self.block.sites(self.prefix)
        return tuple(
            at_coordinate(
                lscales[0] if self.block.isotropic else lscales[d], self.output
            )
            for d in range(self.block.dimension)
        )

    def _drawn(
        self, emission: Emission, coefficients: tuple[ProgramStep, ...]
    ) -> tuple[ProgramStep, ...]:
        """Wrap the coefficients in the hyperparameters and the contribution.

        Parameters
        ----------
        emission : Emission
            Program being emitted.
        coefficients : tuple[ProgramStep, ...]
            Steps binding ``name_coefficients``.

        Returns
        -------
        tuple[ProgramStep, ...]
            Every step of the term.
        """
        return (
            *(emission.draw(site) for site in self.hyperparameters()),
            *coefficients,
            _let(
                self.name + "_per_row",
                over_output(
                    self.output,
                    _call(
                        "matmul",
                        _var(self.name),
                        at_coordinate(self.name + "_coefficients", self.output),
                    ),
                ),
            ),
        )

    def _parameters(
        self, draws: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        """Read the hyperparameters and innovations in one batch shape.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]
            Marginal standard deviation, length scale of each covariate (one
            for an isotropic process), and innovations. The batch shape is
            the draws, then the output axis; the innovations add the
            elements.
        """
        sdgp, lscales = self.block.sites(self.prefix)
        raw = draws[self.name + "_z"].double()
        if self.output is not None:
            raw = raw.unflatten(-1, (-1, self.output.size)).transpose(-1, -2)
        return (
            draws[sdgp].double(),
            tuple(
                draws[site].double()
                for site in (lscales[:1] if self.block.isotropic else lscales)
            ),
            raw,
        )

    def _by_element(self, values: torch.Tensor) -> torch.Tensor:
        """Put the output axis of a batch of values last.

        Parameters
        ----------
        values : torch.Tensor
            Values with the batch shape of `_parameters`, then the elements.

        Returns
        -------
        torch.Tensor
            Values by draw and element, then output coordinate.
        """
        return values if self.output is None else values.transpose(-1, -2)


class SpectralTerm(_ProcessTerm):
    """An approximate Gaussian process on brms's spectral basis.

    The coefficients are ``sqrt(spd(frequencies; sdgp, lscale)) * z``.
    """

    def at(self, rows: PredictorRows) -> SpectralTerm:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        SpectralTerm
            The same coefficients over the spectral basis at the rows.
        """
        if rows.frame is None:
            return self
        return self.with_(matrix=rows.matrix(self.label)[:, list(self.columns)])

    def coefficients(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the coefficients of the term for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        torch.Tensor
            Coefficients by spectral basis function, then output coordinate.
        """
        sdgp, lscales, innovations = self._parameters(draws)
        spd = spectral_density(
            self.block, sdgp, lscales * (self.block.dimension // len(lscales))
        )
        return self._by_element(spd.sqrt() * innovations)

    def effect(
        self, draws: Mapping[str, torch.Tensor], generator: torch.Generator
    ) -> torch.Tensor:
        """Evaluate the contribution for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.
        generator : torch.Generator
            Unused: the basis is known.

        Returns
        -------
        torch.Tensor
            Contributions with shape ``(chain, draw, rows)``, followed by the
            output axis.
        """
        design = torch.tensor(self.matrix)
        return torch.einsum("cdj...,nj->cdn...", self.coefficients(draws), design)

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            The coordinate axis and, for a vector of processes, the flat axis
            of the innovations.
        """
        axis = design_declarations(self.name, self.matrix)[0]
        return tuple(dict.fromkeys((axis, self._innovation_axis(axis))))

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs the term reads.

        Returns
        -------
        tuple[InputDecl, ...]
            The squared frequencies of each covariate and the basis matrix.
        """
        return (
            *(
                InputDecl(
                    name=f"{self.name}_frequency_sq_{d + 1}",
                    element="Real",
                    axes=(self.name + "_Coordinate",),
                    declared=True,
                    values=torch.as_tensor(
                        self.block.frequencies[:, d] ** 2, dtype=torch.float64
                    ),
                )
                for d in range(self.block.dimension)
            ),
            design_declarations(self.name, self.matrix)[1],
        )

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            The hyperparameters and the innovations.
        """
        axis = design_declarations(self.name, self.matrix)[0]
        return (*self.hyperparameters(), self._innovation_site(axis))

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            None.
        """
        return None

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Draw the term's sites and bind its contribution.

        The coefficients are brms's ``rgp = sqrt(spd(slambda; sdgp, lscale))
        .* zgp``, computed as ``exp(log(spd) / 2)`` so a vanishing density
        keeps finite gradients.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            Hyperparameters, innovations, coefficients, and contribution.
        """
        block, name, output = self.block, self.name, self.output
        sdgp, _ = block.sites(self.prefix)
        log_spd = _log_spectral_density(
            block,
            _call("log", at_coordinate(sdgp, output)),
            self._lengths(),
            tuple(_var(f"{name}_frequency_sq_{d + 1}") for d in range(block.dimension)),
        )
        return self._drawn(
            emission,
            (
                *self._innovations(design_declarations(name, self.matrix)[0]),
                _let(name + "_log_spd", over_output(output, log_spd)),
                _let(
                    name + "_coefficients",
                    over_output(
                        output,
                        _mul(
                            _call(
                                "exp",
                                _mul(
                                    LetExprLiteral(value=0.5),
                                    at_coordinate(name + "_log_spd", output),
                                ),
                            ),
                            self._innovation(),
                        ),
                    ),
                ),
            ),
        )


class ExactProcessTerm(_ProcessTerm):
    """An exact Gaussian process at its distinct inputs.

    The coefficients are the latent function at the inputs. A block
    specialized to prediction rows draws them at the fresh inputs from
    `GaussianProcessConditional`, given the function at the training inputs.
    """

    def _training_axis(self) -> AxisDecl:
        """Declare the axis of the training inputs.

        Returns
        -------
        AxisDecl
            The design coordinates while fitting, and an axis of its own for
            a block specialized to prediction rows.
        """
        assert self.block.inputs is not None
        return AxisDecl(
            name=_input_axis(self.name, self.block),
            size=len(self.block.inputs),
        )

    def at(self, rows: PredictorRows) -> ExactProcessTerm:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        ExactProcessTerm
            The function drawn at the rows' distinct inputs given its training
            values. A block with no active row keeps one zero design column at
            its first training input, so it adds nothing.
        """
        if rows.frame is None:
            return self
        block = self.block
        assert block.inputs is not None
        raw = np.asarray(rows.matrix(self.label), dtype=np.float64)
        points, design = gp_rows(raw[:, list(self.columns)], block)
        if not len(points):
            points, design = block.inputs[:1], np.zeros((len(raw), 1))
        return self.with_(matrix=design, block=block.with_(prediction_inputs=points))

    def coefficients(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the coefficients of the term for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        torch.Tensor
            Values of the function at the training inputs, then output
            coordinates.
        """
        sdgp, lscales, innovations = self._parameters(draws)
        return self._by_element(exact_values(self.block, sdgp, lscales, innovations))

    def effect(
        self, draws: Mapping[str, torch.Tensor], generator: torch.Generator
    ) -> torch.Tensor:
        """Evaluate the contribution for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.
        generator : torch.Generator
            Source of the draws of the function at the rows' inputs.

        Returns
        -------
        torch.Tensor
            Contributions with shape ``(chain, draw, rows)``, followed by the
            output axis.
        """
        block = self.block
        values = self.coefficients(draws)
        design = torch.tensor(self.matrix)
        if block.prediction_inputs is not None:
            if not design.any():
                return values.new_zeros(
                    (*values.shape[:2], len(design), *values.shape[3:])
                )
            sdgp, lscales, innovations = self._parameters(draws)
            mean, factor = conditional_moments(
                block,
                exact_values(block, sdgp, lscales, innovations),
                sdgp,
                lscales,
                block.prediction_inputs,
            )
            noise = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
            values = self._by_element(mean + (factor @ noise[..., None])[..., 0])
        return torch.einsum("cdj...,nj->cdn...", values, design)

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            The coordinate axis, the axis of the training inputs of a
            specialized block, and the flat axis of the innovations of a
            vector of processes.
        """
        coordinates, _ = design_declarations(self.name, self.matrix)
        training = self._training_axis()
        return tuple(
            dict.fromkeys((coordinates, training, self._innovation_axis(training)))
        )

    def _fresh_sites(self) -> tuple[SiteDecl, ...]:
        """Declare the function at the fresh inputs of a specialized block.

        Returns
        -------
        tuple[SiteDecl, ...]
            One vector per coordinate of the output axis; empty for a block
            over its training inputs.
        """
        if self.block.prediction_inputs is None:
            return ()
        name = self.name + "_coefficients"
        return tuple(
            SiteDecl(
                name=site,
                class_="zgp",
                event_axis=self.name + "_Coordinate",
                overridable=False,
                fresh=True,
            )
            for site in (
                (name,)
                if self.output is None
                else tuple(f"{name}_{index}" for index in range(self.output.size))
            )
        )

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs the term reads.

        Returns
        -------
        tuple[InputDecl, ...]
            The distances between the training inputs, brms's ``1e-12``
            jitter, the anchor over the fresh inputs of a specialized block,
            and the row design.
        """
        block = self.block
        assert block.inputs is not None
        axis = self._training_axis().name
        matrices = [
            *zip(
                _distance_names(self.name, block),
                _distances(block, block.inputs, block.inputs),
                strict=True,
            ),
            (self.name + "_jitter", EXACT_JITTER * np.eye(len(block.inputs))),
        ]
        anchor = (
            (
                InputDecl(
                    name=self.name + "_anchor",
                    element="Real",
                    axes=(self.name + "_Coordinate",),
                    declared=True,
                    values=torch.zeros(
                        len(block.prediction_inputs), dtype=torch.float64
                    ),
                ),
            )
            if block.prediction_inputs is not None
            else ()
        )
        return (
            *(
                InputDecl(
                    name=name,
                    element="Real",
                    axes=(axis, axis),
                    declared=True,
                    values=torch.as_tensor(values, dtype=torch.float64),
                )
                for name, values in matrices
            ),
            *anchor,
            design_declarations(self.name, self.matrix)[1],
        )

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            The hyperparameters, the innovations at the training inputs, and
            the function at the fresh inputs of a specialized block.
        """
        return (
            *self.hyperparameters(),
            self._innovation_site(self._training_axis()),
            *self._fresh_sites(),
        )

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            The conditional law of the function at the fresh inputs of a
            specialized block, or None.
        """
        return gp_conditional_catalog(((self.name, self.block),))

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Draw the term's sites and bind its contribution.

        The steps are brms's ``gp_<cov>`` Stan function: the kernel matrix
        ``sdgp^2 prod_d k_d(r_d / l_d)`` of the training inputs (one factor of
        the Euclidean distance for an isotropic GP), plus ``1e-12`` on the
        diagonal, its lower Cholesky factor, and ``f = L zgp`` with
        ``zgp ~ N(0, I)``. A block specialized to prediction rows then draws
        ``f`` at the fresh inputs from `GaussianProcessConditional`.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            Hyperparameters, innovations, kernel, factor, coefficients, and
            contribution.
        """
        block, name, output = self.block, self.name, self.output
        sdgp, lscales = block.sites(self.prefix)
        scale = at_coordinate(sdgp, output)
        correlation = reduce(
            _mul,
            (
                _correlation_expression(block.kernel, _var(distance), lscale)
                for distance, lscale in zip(
                    _distance_names(name, block),
                    self._lengths()[:1] if block.isotropic else self._lengths(),
                    strict=True,
                )
            ),
        )
        kernel = _add(_mul(_mul(scale, scale), correlation), _var(name + "_jitter"))
        values = _call(
            "matmul", at_coordinate(name + "_factor", output), self._innovation()
        )
        steps: list[ProgramStep] = [
            *self._innovations(self._training_axis()),
            _let(name + "_kernel", over_output(output, kernel)),
            _let(
                name + "_factor",
                over_output(
                    output, _call("cholesky", at_coordinate(name + "_kernel", output))
                ),
            ),
        ]
        fresh = self._fresh_sites()
        if not fresh:
            steps.append(_let(name + "_coefficients", over_output(output, values)))
            return self._drawn(emission, tuple(steps))
        steps.append(_let(name + "_values", over_output(output, values)))
        arguments = {
            "values": name + "_values",
            "scale": sdgp,
            **{
                f"lscale_{d + 1}": site
                for d, site in enumerate(lscales[:1] if block.isotropic else lscales)
            },
        }
        for index, site in enumerate(fresh):
            bound = {}
            for key, value in arguments.items():
                bound[key] = value if output is None else f"{value}_{index}"
                if output is not None:
                    steps.append(
                        _let(
                            bound[key],
                            _index(value, LetExprLiteral(value=index, integral=True)),
                        )
                    )
            steps.append(
                SampleStep(
                    vars=(site.name,),
                    morphism=conditional_family_name(name),
                    args=tuple(
                        DrawArgNamed(parameter=key, value=atom_to_draw_arg(value))
                        for key, value in (bound | {"anchor": name + "_anchor"}).items()
                    ),
                )
            )
        if output is not None:
            steps.append(
                _let(
                    name + "_coefficients",
                    _factor(
                        ((COORDINATE, output.name),),
                        cases=tuple(
                            (index, _var(site.name)) for index, site in enumerate(fresh)
                        ),
                    ),
                )
            )
        return self._drawn(emission, tuple(steps))


def process_term(
    *,
    label: str,
    name: str,
    prefix: str,
    block: GaussianProcessBlock,
    matrix: np.ndarray,
    columns: tuple[int, ...],
    scale_prior: str,
    output: OutputAxis | None,
) -> SpectralTerm | ExactProcessTerm:
    """Build the term of one Gaussian process block.

    Parameters
    ----------
    label : str
        Authored term.
    name : str
        Prefixed design identifier.
    prefix : str
        Owning predictor namespace.
    block : GaussianProcessBlock
        Retained block.
    matrix : np.ndarray
        Rows by coordinates; read-only.
    columns : tuple[int, ...]
        Columns of the evaluated term the block reads.
    scale_prior : str
        Default ``sdgp`` prior.
    output : OutputAxis or None
        Axis of the predictor.

    Returns
    -------
    SpectralTerm or ExactProcessTerm
        The approximate or the exact process.
    """
    variant = ExactProcessTerm if block.exact else SpectralTerm
    return variant(
        label=label,
        name=name,
        prefix=prefix,
        block=block,
        matrix=matrix,
        columns=columns,
        scale_prior=scale_prior,
        output=output,
    )


__all__: list[str] = []
