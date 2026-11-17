"""The brms prior surface over the formula compiler's prior catalogue.

This module holds four pieces that share the typed prior targets of
`quivers.formulas._prior_emission`:

* the improper flat prior (brms's empty prior string), represented by two
  compilation-local native laws, ``ImproperFlat`` on the real line and
  ``ImproperHalfFlat`` on the positive half line, each the Lebesgue measure
  scaled by ``exp(log_density)``. Fitting scores them; every request for a
  draw outside a structure-only recording raises;
* constant priors (brms ``constant()``), which replace a latent site by a
  deterministic binding;
* the brms default prior policy, reproducing ``brms::default_prior`` for the
  parameter classes both packages share;
* prior tables with brms's columns (``default_prior``, ``validate_prior``,
  ``prior_summary``).

Derived sites (constants, the original-scale intercept of a centered
predictor, coefficients recovered from QR coordinates) are deterministic
bindings over the sampled coordinates; `derived_site_draws` evaluates them
over posterior draws so every consumer of ``draws()`` reads them by their
ordinary names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Literal

import didactic.api as dx
import numpy as np
import pandas as pd
import torch
from torch.distributions import Distribution, constraints
from torch.types import _size

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    DrawArgNamed,
    DrawArgScalar,
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
    LetStep,
    Module,
    ProgramDecl,
    SampleStep,
    TypeName,
)
from quivers.dsl.emit import draw_call_to_source
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _factor, finite_object_sizes
from quivers.formulas._priors import (
    ConstantPrior,
    ImproperFlat,
    PriorSet,
    PriorSpec,
    PriorTarget,
    R2D2,
    RegularizedHorseshoe,
    resolve_priors,
)
from quivers.inference._latent_recording import recording_latent_structure
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

FLAT_FAMILY = "ImproperFlat"
"""Native name of the improper flat law on the real line."""

HALF_FLAT_FAMILY = "ImproperHalfFlat"
"""Native name of the improper flat law on the positive half line."""

FLAT_FAMILIES = frozenset({FLAT_FAMILY, HALF_FLAT_FAMILY})
"""Every native improper flat law."""

SHRINKAGE_SCALE_CLASSES = frozenset({"sd", "sds", "sdgp", "sderr", "sdcar"})
"""Standard-deviation classes a brms shrinkage prior replaces by its scale."""

DefaultPriorPolicy = Literal["quivers", "brms"]
"""Which default priors fill the sites no rule selects: the proper, autoscaled
quivers defaults, or the defaults of ``brms::default_prior``."""


class _ImproperFlatLaw(Distribution):
    """The Lebesgue measure on a half line or the real line, scaled by a constant.

    The log density is ``log_density`` at every point of the support. The
    measure has infinite mass, so it has no draws: `rsample` answers only a
    structure-only recording (which discards the value) and raises otherwise.

    Parameters
    ----------
    log_density : torch.Tensor
        Constant log density, broadcast over the batch.
    positive : bool
        Support the positive half line rather than the real line.
    """

    has_rsample = True

    def __init__(self, log_density: torch.Tensor, *, positive: bool) -> None:
        self.log_density = log_density
        self._positive = positive
        super().__init__(batch_shape=log_density.shape, validate_args=False)

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Return the parameter constraints.

        Returns
        -------
        dict[str, torch.distributions.constraints.Constraint]
            ``log_density`` is any real number.
        """
        return {"log_density": constraints.real}

    @property
    def support(self) -> constraints.Constraint:
        """Return the declared support.

        Returns
        -------
        torch.distributions.constraints.Constraint
            Positive half line or real line.
        """
        return constraints.positive if self._positive else constraints.real

    def expand(
        self, batch_shape: _size, _instance: _ImproperFlatLaw | None = None
    ) -> _ImproperFlatLaw:
        """Broadcast the constant density over a larger batch.

        Parameters
        ----------
        batch_shape : torch.Size or tuple[int, ...]
            Target batch shape.
        _instance : _ImproperFlatLaw or None
            Unused torch extension argument.

        Returns
        -------
        _ImproperFlatLaw
            The same measure over the expanded batch.
        """
        del _instance
        return _ImproperFlatLaw(
            self.log_density.expand(torch.Size(batch_shape)), positive=self._positive
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score a value: the constant inside the support, minus infinity outside.

        Parameters
        ----------
        value : torch.Tensor
            Points, broadcast against the batch.

        Returns
        -------
        torch.Tensor
            Constant log density, or ``-inf`` outside the positive half line.
        """
        density = self.log_density.to(value.dtype).expand(
            torch.broadcast_shapes(value.shape, self.batch_shape)
        )
        if self._positive:
            return torch.where(value > 0, density, torch.full_like(density, -math.inf))
        return density + torch.zeros_like(value)

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Return a placeholder inside a structure-only recording, and refuse otherwise.

        Parameters
        ----------
        sample_shape : torch.Size or tuple[int, ...]
            Leading sample axes.

        Returns
        -------
        torch.Tensor
            Ones (half line) or zeros (real line) with the batch shape, only
            while an inference driver records latent structure.

        Raises
        ------
        ValueError
            Outside such a recording: an improper flat prior has no draws.
        """
        if not recording_latent_structure():
            raise ValueError(
                "an improper flat prior (brms's default for class b) has no "
                "draws: prior draws, sample_prior, prior initialization, and "
                "marginal-likelihood estimates require a proper prior on every "
                "site; give the flat sites a proper prior"
            )
        shape = torch.Size(sample_shape) + self.batch_shape
        fill = 1.0 if self._positive else 0.0
        return torch.full(shape, fill, dtype=self.log_density.dtype)

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Delegate to `rsample` without gradients.

        Parameters
        ----------
        sample_shape : torch.Size or tuple[int, ...]
            Leading sample axes.

        Returns
        -------
        torch.Tensor
            See `rsample`.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)


def _flat_record(name: str, support: str) -> DistributionFamily:
    """Build the semantic contract of one improper flat law.

    Parameters
    ----------
    name : str
        Native family name.
    support : str
        ``"real"`` or ``"positive"``.

    Returns
    -------
    DistributionFamily
        Scalar continuous family with one real ``log_density`` parameter.
    """
    return DistributionFamily(
        name=name,
        parameters=(FamilyParameter(name="log_density", constraint="real", rank=0),),
        support=support,
        element=REAL,
        event_rank=0,
        event_sources=(),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )


def _real_flat(parameters: Mapping[str, torch.Tensor]) -> Distribution:
    """Build the real-line flat law.

    Parameters
    ----------
    parameters : Mapping[str, torch.Tensor]
        The ``log_density`` tensor.

    Returns
    -------
    Distribution
        The improper law.
    """
    return _ImproperFlatLaw(parameters["log_density"], positive=False)


def _half_flat(parameters: Mapping[str, torch.Tensor]) -> Distribution:
    """Build the positive half-line flat law.

    Parameters
    ----------
    parameters : Mapping[str, torch.Tensor]
        The ``log_density`` tensor.

    Returns
    -------
    Distribution
        The improper law.
    """
    return _ImproperFlatLaw(parameters["log_density"], positive=True)


FLAT_CATALOG: FamilyCatalog = (
    FamilyCatalog()
    .with_family(_flat_record(FLAT_FAMILY, "real"), _real_flat)
    .with_family(
        _flat_record(HALF_FLAT_FAMILY, "positive"),
        _half_flat,
        support=constraints.positive,
    )
)
"""The compilation-local catalog of both improper flat laws."""


def flat_prior_step(
    site: str, *, lower: float | None, upper: float | None, index: TypeName | None
) -> SampleStep:
    """Emit brms's flat prior on a site's admissible interval.

    Parameters
    ----------
    site : str
        Latent site binder.
    lower, upper : float or None
        Intersected bounds of the target and the rule.
    index : TypeName or None
        The site's nominal coefficient axis.

    Returns
    -------
    SampleStep
        ``Uniform(lower, upper)`` when both bounds are finite (the flat density
        normalized on a bounded interval), ``ImproperHalfFlat`` for a lower
        bound of zero, and ``ImproperFlat`` without bounds.

    Raises
    ------
    ValueError
        For a single bound other than a zero lower bound, which has no native
        improper law.
    """
    if lower is not None and upper is not None:
        if not lower < upper:
            raise ValueError(f"flat prior bounds for {site!r} are empty")
        return SampleStep(
            vars=(site,),
            morphism="Uniform",
            args=(
                DrawArgNamed(parameter="low", value=DrawArgScalar(value=float(lower))),
                DrawArgNamed(parameter="high", value=DrawArgScalar(value=float(upper))),
            ),
            index=index,
        )
    if upper is not None or (lower is not None and lower != 0.0):
        raise ValueError(
            f"a flat prior on {site!r} supports no bound, a zero lower bound, or "
            "two finite bounds; give a proper prior for other half lines"
        )
    return SampleStep(
        vars=(site,),
        morphism=HALF_FLAT_FAMILY if lower == 0.0 else FLAT_FAMILY,
        args=(DrawArgNamed(parameter="log_density", value=DrawArgScalar(value=0.0)),),
        index=index,
    )


def improper_sites(module: Module) -> tuple[str, ...]:
    """List the sites of a module that carry an improper flat prior.

    Parameters
    ----------
    module : Module
        Emitted formula module.

    Returns
    -------
    tuple[str, ...]
        Site binders in program order.
    """
    return tuple(
        step.vars[0]
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, SampleStep) and step.morphism in FLAT_FAMILIES
    )


def require_proper(module: Module, purpose: str) -> None:
    """Refuse an operation that needs a normalized prior on every site.

    Parameters
    ----------
    module : Module
        Emitted formula module.
    purpose : str
        The refused operation, named in the error.

    Raises
    ------
    ValueError
        If any site carries an improper flat prior; the error names them.
    """
    sites = improper_sites(module)
    if sites:
        raise ValueError(
            f"{purpose} requires a proper prior on every site, but "
            f"{', '.join(repr(site) for site in sites)} carry the improper flat "
            "prior; give them a proper prior (for instance "
            "PriorSpec(distribution='Normal(0, 5)', class_='b'))"
        )


def with_flat_catalog(
    catalog: FamilyCatalog | None, module: Module
) -> FamilyCatalog | None:
    """Join the improper flat laws to a catalog when the module uses them.

    Parameters
    ----------
    catalog : FamilyCatalog or None
        Existing compilation catalog.
    module : Module
        The module about to be compiled or checked.

    Returns
    -------
    FamilyCatalog or None
        ``catalog`` unchanged when no site is flat, otherwise the joined catalog.
    """
    if not improper_sites(module):
        return catalog
    return FLAT_CATALOG if catalog is None else catalog.with_catalog(FLAT_CATALOG)


def flat_event_rank(morphism: str) -> int | None:
    """Return the event rank of an improper flat law, or None for other families.

    Parameters
    ----------
    morphism : str
        Draw family name.

    Returns
    -------
    int or None
        Zero for the flat laws.
    """
    return 0 if morphism in FLAT_FAMILIES else None


def _constant_factor(
    target: PriorTarget, prior: ConstantPrior, axis: str | None, size: int | None
) -> LetExprNode:
    """Build the constant Cholesky factor of a correlation-matrix site.

    brms's ``constant()`` on a correlation class (``cor``, ``rescor``,
    ``cortime``) fixes the Cholesky factor ``L`` of the correlation matrix,
    the parameter brms samples. The value is thus the ``K x K`` lower
    triangular factor in row-major order, whose rows have unit norm.

    Parameters
    ----------
    target : PriorTarget
        Selected site whose native event is a matrix.
    prior : ConstantPrior
        The factor's ``K * K`` entries with ``broadcast=False``.
    axis : str or None
        The site's declared coordinate axis.
    size : int or None
        Its cardinality ``K``.

    Returns
    -------
    LetExprNode
        A row factor of column factors over the coordinate axis.

    Raises
    ------
    ValueError
        If the site declares no finite axis, a single value is broadcast,
        the entry count is not ``K * K``, or the entries are not the lower
        Cholesky factor of a correlation matrix.
    """
    if axis is None or size is None:
        raise ValueError(
            f"constant prior on correlation site {target.site!r} requires a "
            "declared finite coordinate axis"
        )
    values = prior.value if isinstance(prior.value, tuple) else (prior.value,)
    if prior.broadcast or len(values) != size * size:
        raise ValueError(
            f"constant prior on correlation site {target.site!r} fixes the "
            f"{size} x {size} Cholesky factor of the correlation matrix, as "
            f"brms's constant() on a correlation class does; give its "
            f"{size * size} entries in row-major order with broadcast=False "
            "(the identity fixes every correlation at zero)"
        )
    rows = [values[row * size : (row + 1) * size] for row in range(size)]
    for row, entries in enumerate(rows):
        if (
            any(entry != 0.0 for entry in entries[row + 1 :])
            or entries[row] <= 0.0
            or abs(sum(entry * entry for entry in entries) - 1.0) > 1e-8
        ):
            raise ValueError(
                f"constant prior on correlation site {target.site!r} is not the "
                "lower Cholesky factor of a correlation matrix: every row "
                "needs unit norm, a positive diagonal entry, and zeros above it"
            )
    return _factor(
        (("constant_row", axis),),
        cases=tuple(
            (
                row,
                _factor(
                    (("constant_column", axis),),
                    cases=tuple(
                        (column, LetExprLiteral(value=float(entry)))
                        for column, entry in enumerate(entries)
                    ),
                ),
            )
            for row, entries in enumerate(rows)
        ),
    )


def _constant_value(
    target: PriorTarget,
    prior: ConstantPrior,
    axis: str | None = None,
    size: int | None = None,
) -> LetExprNode:
    """Build the deterministic value of a constant prior over the target's axes.

    Parameters
    ----------
    target : PriorTarget
        Selected site with its nominal coordinate axes.
    prior : ConstantPrior
        Value and broadcast policy.
    axis : str or None
        Declared coordinate axis of the replaced sample site.
    size : int or None
        Cardinality of that axis.

    Returns
    -------
    LetExprNode
        A literal, a factor over the target's single axis, or the Cholesky
        factor of a correlation-matrix site.

    Raises
    ------
    ValueError
        If the value count disagrees with the target width, a vector is
        given to a scalar site, the target has several axes, or a
        correlation-matrix site is not given a valid Cholesky factor.
    """
    if target.event_rank == 2:
        return _constant_factor(target, prior, axis, size)
    values = prior.value if isinstance(prior.value, tuple) else (prior.value,)
    if not target.axes:
        if len(values) != 1:
            raise ValueError(
                f"constant prior on scalar {target.site!r} takes one value, got {len(values)}"
            )
        return LetExprLiteral(value=float(values[0]))
    if len(target.axes) != 1:
        raise ValueError(
            f"constant prior on {target.site!r} supports one coefficient axis"
        )
    width = target.width
    if len(values) == 1 and prior.broadcast:
        return _factor(
            (("constant_coordinate", target.axes[0]),),
            body=LetExprLiteral(value=float(values[0])),
        )
    if len(values) != width:
        raise ValueError(
            f"constant prior on {target.site!r} needs {width} values, got {len(values)}"
        )
    return _factor(
        (("constant_coordinate", target.axes[0]),),
        cases=tuple(
            (index, LetExprLiteral(value=float(value)))
            for index, value in enumerate(values)
        ),
    )


def apply_fixed_rules(
    module: Module,
    targets: Sequence[PriorTarget],
    selected: Mapping[str, PriorSpec],
) -> tuple[Module, dict[str, PriorSpec]]:
    """Emit flat and constant rules and return the remaining hierarchy rules.

    Parameters
    ----------
    module : Module
        Emitted module with every selected site still a latent draw.
    targets : Sequence[PriorTarget]
        Role metadata of the selected sites.
    selected : Mapping[str, PriorSpec]
        Resolved non-string rules.

    Returns
    -------
    tuple[Module, dict[str, PriorSpec]]
        The rewritten module and the shrinkage rules left for the hierarchy
        emitter.

    Raises
    ------
    ValueError
        If a constant has the wrong width, a flat rule has unsupported bounds,
        or a constant names a site another prior depends on as a shrinkage
        member.
    """
    by_site = {target.site: target for target in targets}
    flat: dict[str, tuple[float | None, float | None]] = {}
    constants: dict[str, ConstantPrior] = {}
    remaining: dict[str, PriorSpec] = {}
    for site, rule in selected.items():
        distribution = rule.distribution
        target = by_site[site]
        if isinstance(distribution, ImproperFlat):
            lows = [v for v in (target.lower, rule.lower) if v is not None]
            highs = [v for v in (target.upper, rule.upper) if v is not None]
            flat[site] = (max(lows) if lows else None, min(highs) if highs else None)
        elif isinstance(distribution, ConstantPrior):
            constants[site] = distribution
        else:
            remaining[site] = rule
    if not flat and not constants:
        return module, remaining
    sizes = finite_object_sizes(module)
    statements = []
    for statement in module.statements:
        if isinstance(statement, ProgramDecl):
            draws = []
            for step in statement.draws:
                name = step.vars[0] if isinstance(step, SampleStep) else None
                if name is not None and name in constants:
                    axis = step.index.name if isinstance(step.index, TypeName) else None
                    draws.append(
                        LetStep(
                            name=name,
                            value=_constant_value(
                                by_site[name],
                                constants[name],
                                axis,
                                sizes.get(axis) if axis is not None else None,
                            ),
                        )
                    )
                elif name is not None and name in flat:
                    lower, upper = flat[name]
                    draws.append(
                        flat_prior_step(
                            name,
                            lower=lower,
                            upper=upper,
                            index=step.index
                            if isinstance(step.index, TypeName)
                            else None,
                        )
                    )
                else:
                    draws.append(step)
            statement = statement.with_(draws=tuple(draws))
        statements.append(statement)
    return module.with_(statements=tuple(statements)), remaining


def _let_steps(module: Module) -> dict[str, LetExprNode]:
    """Collect the program's deterministic bindings by name.

    Parameters
    ----------
    module : Module
        Emitted formula module.

    Returns
    -------
    dict[str, LetExprNode]
        Let names and their expressions.
    """
    return {
        step.name: step.value
        for statement in module.statements
        if isinstance(statement, ProgramDecl)
        for step in statement.draws
        if isinstance(step, LetStep)
    }


_ELEMENTWISE = {
    "exp": torch.exp,
    "log": torch.log,
    "log1p": torch.log1p,
    "sqrt": torch.sqrt,
    "softplus": torch.nn.functional.softplus,
}
"""Elementwise builtins a derived parameter binding may apply."""


def derived_site_draws(
    module: Module, draws: Mapping[str, torch.Tensor], names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Evaluate deterministic parameter sites over posterior draws.

    A constant prior, the original-scale intercept of a centered predictor,
    and coefficients recovered from QR coordinates are deterministic bindings
    of sampled sites. Each is evaluated draw by draw from the same expression
    the native program executes, at cost O(S k) for S draws and k terms.

    Parameters
    ----------
    module : Module
        The fitted module.
    draws : Mapping[str, torch.Tensor]
        Constrained draws with leading chain and draw axes.
    names : Sequence[str]
        Reported parameter names to add when they are deterministic bindings.

    Returns
    -------
    dict[str, torch.Tensor]
        ``draws`` extended with every requested name bound by a let.

    Raises
    ------
    ValueError
        If a requested binding depends on data or uses an expression outside
        affine arithmetic over parameters, literals, and finite factors.
    """
    lets = _let_steps(module)
    requested = [name for name in names if name not in draws and name in lets]
    if not requested:
        return dict(draws)
    leading = next(iter(draws.values())).shape[:2] if draws else torch.Size((1, 1))
    sizes = finite_object_sizes(module)
    values: dict[str, torch.Tensor] = dict(draws)

    def evaluate(node: LetExprNode, scope: Mapping[str, int]) -> torch.Tensor:
        if isinstance(node, LetExprLiteral):
            return torch.tensor(float(node.value), dtype=torch.float64)
        if isinstance(node, LetExprVar):
            if node.name in scope:
                return torch.tensor(scope[node.name], dtype=torch.long)
            return resolve(node.name)
        if isinstance(node, LetExprUnaryOp) and node.op == "-":
            return -evaluate(node.operand, scope)
        if isinstance(node, LetExprBinOp) and node.op in {"+", "-", "*", "/"}:
            left = evaluate(node.left, scope)
            right = evaluate(node.right, scope)
            if node.op == "+":
                return left + right
            if node.op == "-":
                return left - right
            if node.op == "*":
                return left * right
            return left / right
        if isinstance(node, LetExprCall) and node.func in _ELEMENTWISE:
            arguments = [evaluate(argument, scope) for argument in node.args]
            return _ELEMENTWISE[node.func](*arguments)
        if isinstance(node, LetExprIndex) and isinstance(node.array, LetExprVar):
            array = resolve(node.array.name)
            positions = tuple(int(evaluate(item, scope)) for item in node.indices)
            return array[(..., *positions)]
        if isinstance(node, LetExprList):
            return torch.stack(
                [
                    evaluate(item, scope).expand(*leading)
                    if evaluate(item, scope).ndim == 0
                    else evaluate(item, scope)
                    for item in node.items
                ],
                dim=-1,
            )
        if isinstance(node, LetExprFactor) and len(node.binders) == 1:
            binder = node.binders[0]
            axis = binder.index.name if isinstance(binder.index, TypeName) else None
            if axis is None or axis not in sizes:
                raise ValueError("a derived factor needs a declared finite axis")
            columns = []
            cases = {case.label: case.value for case in node.cases}
            for coordinate in range(sizes[axis]):
                body = node.body if node.body is not None else cases[coordinate]
                value = evaluate(body, {**scope, binder.var: coordinate})
                columns.append(value.expand(*leading) if value.ndim == 0 else value)
            # The binder is the first event axis; a nested factor adds the
            # axes of its body after it.
            return torch.stack(columns, dim=len(leading))
        raise ValueError(
            "a derived parameter site uses an expression outside affine "
            f"parameter arithmetic: {type(node).__name__}"
        )

    def resolve(name: str) -> torch.Tensor:
        if name in values:
            return values[name]
        if name not in lets:
            raise ValueError(
                f"derived parameter depends on {name!r}, which is neither a "
                "posterior draw nor a parameter binding"
            )
        value = evaluate(lets[name], {})
        if value.ndim == 0:
            value = value.expand(*leading)
        values[name] = value
        return value

    for name in requested:
        resolve(name)
    dtype = next(iter(draws.values())).dtype if draws else torch.get_default_dtype()
    return {**dict(draws), **{name: values[name].to(dtype) for name in requested}}


def r_round(value: float, digits: int = 1) -> float:
    """Round as R's ``round`` does for a finite double.

    R (``fround`` in ``src/nmath/fround.c``) takes the two candidates
    ``floor(10^d x) / 10^d`` and ``ceil(10^d x) / 10^d`` of ``|x|`` and keeps
    the one at the smaller distance computed in double precision, the even
    one on a tie. This differs from Python's `round`, which rounds the exact
    decimal value of the double: R rounds 0.45 to 0.4, Python to 0.5.

    Parameters
    ----------
    value : float
        Finite value.
    digits : int
        Decimal places.

    Returns
    -------
    float
        Rounded value.
    """
    # Exact R port: Python's round disagrees at 0.45.
    sign = -1.0 if value < 0 else 1.0
    magnitude = abs(float(value))
    scale = 10.0**digits
    scaled = scale * magnitude
    lower = math.floor(scaled)
    down, up = lower / scale, math.ceil(scaled) / scale
    above, below = up - magnitude, magnitude - down
    return sign * (up if above < below or (above == below and lower % 2 == 1) else down)


def r_mad(values: np.ndarray) -> float:
    """Compute R's ``mad``: 1.4826 times the median absolute deviation.

    Parameters
    ----------
    values : np.ndarray
        Sample, possibly containing NaN.

    Returns
    -------
    float
        The scaled median absolute deviation; NaN if any value is NaN, as R's
        default ``na.rm = FALSE`` gives.
    """
    if np.isnan(values).any():
        return math.nan
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center)))


_TRANSFORMABLE_LINKS = frozenset(
    {"identity", "log", "inverse", "sqrt", "inverse_square"}
)
"""Links brms applies to the response when it scales default priors."""

_LOGSCALE_FAMILIES = frozenset({"lognormal", "shifted_lognormal", "hurdle_lognormal"})
"""Families brms scales on the log response (``has_logscale``)."""


def _linked_response(response: np.ndarray, link: str) -> np.ndarray:
    """Apply a link to the response as brms's ``def_scale_prior`` does.

    Parameters
    ----------
    response : np.ndarray
        Training responses.
    link : str
        A link in `_TRANSFORMABLE_LINKS`.

    Returns
    -------
    np.ndarray
        Linked responses; zeros are moved to 0.1 for the log and inverse
        links, and nonpositive values give NaN or infinity as in R.
    """
    y = np.asarray(response, dtype=np.float64)
    if link in {"log", "inverse", "inverse_square"}:
        y = np.where(y == 0, y + 0.1, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        if link == "log":
            return np.log(y)
        if link == "inverse":
            return 1.0 / y
        if link == "inverse_square":
            return 1.0 / y**2
        if link == "sqrt":
            return np.sqrt(y)
    return y


def scale_prior_arguments(
    response: np.ndarray | None,
    link: str,
    *,
    center: bool,
    offset_mean: float = 0.0,
) -> tuple[float, float]:
    """Return the location and scale of brms's ``def_scale_prior``.

    Parameters
    ----------
    response : np.ndarray or None
        Scalar numeric training responses, or None for factor and vector
        responses, which keep the fixed location 0 and scale 2.5.
    link : str
        Link applied before scaling.
    center : bool
        Keep location zero (scale priors) or use the rounded linked median
        minus the mean offset (location intercepts).
    offset_mean : float
        Mean of the predictor's offsets.

    Returns
    -------
    tuple[float, float]
        ``(location, scale)`` with ``scale = max(2.5, round(mad, 1))``.
    """
    location, scale = 0.0, 2.5
    if response is None or link not in _TRANSFORMABLE_LINKS:
        return location, scale
    linked = _linked_response(response, link)
    spread = r_mad(linked)
    if math.isfinite(spread):
        rounded = r_round(spread)
        if math.isfinite(rounded):
            scale = max(scale, rounded)
    if not center:
        middle = float(np.median(linked)) if not np.isnan(linked).any() else math.nan
        if math.isfinite(middle):
            location = r_round(middle)
        if math.isfinite(offset_mean):
            location = location - offset_mean
    return location, scale


def _number(value: float) -> str:
    """Render a number in canonical QVR literal form.

    Parameters
    ----------
    value : float
        Finite value.

    Returns
    -------
    str
        ``repr`` of the float.
    """
    return repr(float(value))


_IDENTITY_DPAR_PRIORS: Mapping[str, str] = {
    "shape": "Gamma(0.01, 0.01)",
    "nu": "Gamma(2.0, 0.1)",
    "phi": "Gamma(0.01, 0.01)",
    "kappa": "Gamma(2.0, 0.01)",
    "beta": "Gamma(1.0, 0.1)",
    "zi": "Beta(1.0, 1.0)",
    "hu": "Beta(1.0, 1.0)",
    "zoi": "Beta(1.0, 1.0)",
    "coi": "Beta(1.0, 1.0)",
    "bs": "Gamma(1.0, 1.0)",
    "bias": "Beta(1.0, 1.0)",
    "quantile": "Beta(1.0, 1.0)",
    "xi": "Normal(0.0, 2.5)",
    "alpha": "Normal(0.0, 4.0)",
    "disc": "LogNormal(0.0, 1.0)",
}
"""``def_dpar_prior`` for an estimated (not predicted) auxiliary parameter."""

_LINKED_DPAR_PRIORS: Mapping[str, str] = {
    "shape": "StudentT(3.0, 0.0, 2.5)",
    "nu": "Normal(2.7, 0.8)",
    "phi": "StudentT(3.0, 0.0, 2.5)",
    "kappa": "Normal(5.0, 0.8)",
    "beta": "Normal(1.7, 1.3)",
    "zi": "Logistic(0.0, 1.0)",
    "hu": "Logistic(0.0, 1.0)",
    "zoi": "Logistic(0.0, 1.0)",
    "coi": "Logistic(0.0, 1.0)",
    "bs": "Normal(-0.6, 1.3)",
    "bias": "Logistic(0.0, 1.0)",
    "quantile": "Logistic(0.0, 1.0)",
    "xi": "Normal(0.0, 4.0)",
    "alpha": "Normal(0.0, 4.0)",
    "disc": "Normal(0.0, 1.0)",
}
"""``def_dpar_prior`` for the centered intercept of a predicted parameter."""

_NEGBINOMIAL_SHAPE = frozenset(
    {"negative_binomial", "zero_inflated_negbinomial", "hurdle_negbinomial"}
)
"""Families whose estimated shape has the pinned ``inv_gamma(0.4, 0.3)`` default."""

_BOUNDED_FLAT_CLASSES = frozenset({"ar", "ma", "cosy", "car", "lagsar", "errorsar"})
"""Structural classes brms gives a flat prior over their bounded domain."""

_UNBOUNDED_FLAT_CLASSES = frozenset({"ar", "ma", "meanme", "sdme"})
"""Classes brms gives a flat prior over their whole domain: conditional ARMA
coefficients (no bounds, R/priors.R ``prior_ac``), and the latent means and
standard deviations of ``me()`` terms (``prior_Xme``; ``sdme`` has
``lb = 0``)."""


def _brms_dpar_class(family: str, name: str) -> str:
    """Map a quivers auxiliary name to its brms parameter class.

    Parameters
    ----------
    family : str
        quivers family name.
    name : str
        Auxiliary parameter name.

    Returns
    -------
    str
        brms class name (``disp`` of ``negative_binomial`` is brms ``shape``,
        and a per-category scale ``sigma<c>`` of ``logistic_normal`` takes the
        defaults of ``sigma``).
    """
    if family == "logistic_normal" and name.startswith("sigma"):
        return "sigma"
    return "shape" if family == "negative_binomial" and name == "disp" else name


def brms_default_rule(
    target: PriorTarget,
    *,
    family: str,
    location_link: str,
    dpar_links: Mapping[str, str],
    response: np.ndarray | None,
    offset_means: Mapping[str, float],
    centered: frozenset[str],
) -> PriorSpec | None:
    """Return brms's default prior for one target, or None to keep quivers's.

    Parameters
    ----------
    target : PriorTarget
        Overrideable site.
    family : str
        quivers family name.
    location_link : str
        Link of the location predictor.
    dpar_links : Mapping[str, str]
        Links of the auxiliary parameters.
    response : np.ndarray or None
        Scalar numeric training responses, or None.
    offset_means : Mapping[str, float]
        Mean offset of each predictor, keyed by predicted parameter.
    centered : frozenset[str]
        Sites of centered intercepts (brms class ``Intercept``).

    Returns
    -------
    PriorSpec or None
        A site rule, or None where brms has no counterpart (ordinal thresholds,
        mixture weights, baseline hazards, structured latent terms) or where
        the quivers default already is brms's (``simo``, ``lscale``).
        Population, ``mo()`` and ``mi()`` coefficients (class ``b``),
        conditional ARMA coefficients, ``meanme`` and ``sdme`` take brms's
        improper flat prior.
    """
    role = target.class_
    if family in _LOGSCALE_FAMILIES:
        # brms has_logscale: scale defaults on the log response.
        location_link = "log"

    def rule(distribution: str | ImproperFlat) -> PriorSpec:
        return PriorSpec(distribution=distribution, class_=None, site=target.site)

    _, scale = scale_prior_arguments(response, location_link, center=True)
    if role == "b":
        return rule(ImproperFlat())
    if role in SHRINKAGE_SCALE_CLASSES:
        return rule(f"HalfStudentT(3.0, {_number(scale)})")
    if role in {"cor", "rescor"}:
        return rule("LKJCholesky(1.0)")
    if role in _BOUNDED_FLAT_CLASSES and (
        target.lower is not None and target.upper is not None
    ):
        return rule(f"Uniform({_number(target.lower)}, {_number(target.upper)})")
    # Flat on the whole domain: the flat law takes the target's zero lower
    # bound for sdme.
    if role in _UNBOUNDED_FLAT_CLASSES:
        return rule(ImproperFlat())
    if role in _BOUNDED_FLAT_CLASSES:
        return None
    if role == "Intercept":
        if target.site not in centered:
            return None
        dpar = target.dpar or "mu"
        if dpar == "mu":
            if family == "von_mises" and location_link == "tan_half":
                return rule("StudentT(1.0, 0.0, 1.0)")
            location, intercept_scale = scale_prior_arguments(
                response,
                location_link,
                center=False,
                offset_mean=offset_means.get("mu", 0.0),
            )
            return rule(
                f"StudentT(3.0, {_number(location)}, {_number(intercept_scale)})"
            )
        link = dpar_links.get(dpar, "identity")
        brms_class = _brms_dpar_class(family, dpar)
        if brms_class == "sigma":
            linked = "log" if family in _LOGSCALE_FAMILIES else link
            _, sigma_scale = scale_prior_arguments(response, linked, center=True)
            return rule(f"StudentT(3.0, 0.0, {_number(sigma_scale)})")
        table = _IDENTITY_DPAR_PRIORS if link == "identity" else _LINKED_DPAR_PRIORS
        text = table.get(brms_class)
        return None if text is None else rule(text)
    if target.dpar is not None and role == target.dpar:
        brms_class = _brms_dpar_class(family, role)
        if brms_class == "sigma":
            if family == "negbinomial2":
                return rule("Gamma(0.4, 0.3)")
            linked = "log" if family in _LOGSCALE_FAMILIES else "identity"
            _, sigma_scale = scale_prior_arguments(response, linked, center=True)
            return rule(f"HalfStudentT(3.0, {_number(sigma_scale)})")
        if brms_class == "shape" and family in _NEGBINOMIAL_SHAPE:
            return rule("InverseGamma(0.4, 0.3)")
        if brms_class == "ndt" and response is not None:
            return rule(f"Uniform(0.0, {_number(float(np.min(response)))})")
        if brms_class == "kappa" and family == "xbeta":
            return rule("Gamma(0.01, 0.01)")
        text = _IDENTITY_DPAR_PRIORS.get(brms_class)
        return None if text is None else rule(text)
    return None


class PriorRow(ValueModel):
    """One row of a prior table, in brms's ``brmsprior`` columns.

    Parameters
    ----------
    site : str
        Generated latent site the row governs.
    prior : str
        Effective prior: a canonical QVR distribution call, ``flat`` for the
        improper flat prior, or ``constant(v)``.
    class_ : str
        Parameter class.
    coef, group, response, dpar, nlpar : str or None
        brms ``coef``, ``group``, ``resp``, ``dpar`` and ``nlpar`` columns.
    lower, upper : float or None
        brms ``lb`` and ``ub``.
    tag : str or None
        brms ``tag`` of the rule that set the prior.
    source : {"user", "default"}
        Whether a user rule or the default policy set the prior.
    """

    site: str
    prior: str
    class_: str
    coef: str | None = None
    group: str | None = None
    response: str | None = None
    dpar: str | None = None
    nlpar: str | None = None
    lower: float | None = None
    upper: float | None = None
    tag: str | None = None
    source: Literal["user", "default"] = "default"


class PriorTable(ValueModel):
    """The effective prior of every overrideable site, as brms tabulates it.

    Parameters
    ----------
    rows : tuple[PriorRow, ...]
        One row per site, in emission order.
    policy : {"quivers", "brms"}
        Default policy that filled sites no user rule selects.
    """

    rows: tuple[PriorRow, ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    policy: DefaultPriorPolicy = "quivers"

    def to_pandas(self) -> pd.DataFrame:
        """Return the table with brms's column names.

        Returns
        -------
        pandas.DataFrame
            Columns ``prior, class, coef, group, resp, dpar, nlpar, lb, ub,
            tag, source, site``; absent labels are empty strings and absent
            bounds are NaN, as in brms.
        """
        return pd.DataFrame(
            {
                "prior": [row.prior for row in self.rows],
                "class": [row.class_ for row in self.rows],
                "coef": [row.coef or "" for row in self.rows],
                "group": [row.group or "" for row in self.rows],
                "resp": [row.response or "" for row in self.rows],
                "dpar": [
                    "" if row.dpar in (None, "mu") else row.dpar for row in self.rows
                ],
                "nlpar": [row.nlpar or "" for row in self.rows],
                "lb": [
                    math.nan if row.lower is None else row.lower for row in self.rows
                ],
                "ub": [
                    math.nan if row.upper is None else row.upper for row in self.rows
                ],
                "tag": [row.tag or "" for row in self.rows],
                "source": [row.source for row in self.rows],
                "site": [row.site for row in self.rows],
            }
        )

    def row(self, site: str) -> PriorRow:
        """Return the row of one site.

        Parameters
        ----------
        site : str
            Generated site name.

        Returns
        -------
        PriorRow
            Its row.

        Raises
        ------
        KeyError
            If the table has no such site.
        """
        for item in self.rows:
            if item.site == site:
                return item
        raise KeyError(site)


def _step_prior(step: SampleStep | LetStep) -> str:
    """Render one site's effective prior for a table row.

    Parameters
    ----------
    step : SampleStep or LetStep
        The site's emitted step.

    Returns
    -------
    str
        ``flat``, ``constant(...)``, or the canonical distribution call.
    """
    if isinstance(step, LetStep):
        value = step.value
        if isinstance(value, LetExprLiteral):
            return f"constant({_number(float(value.value))})"
        if isinstance(value, LetExprFactor):
            if value.body is not None and isinstance(value.body, LetExprLiteral):
                return f"constant({_number(float(value.body.value))})"
            return (
                "constant(("
                + ", ".join(
                    _number(float(case.value.value))
                    for case in value.cases
                    if isinstance(case.value, LetExprLiteral)
                )
                + "))"
            )
        if isinstance(value, LetExprVar):
            return f"= {value.name}"
        return "deterministic"
    if step.morphism in FLAT_FAMILIES:
        return "flat" if step.morphism == FLAT_FAMILY else "flat on (0, inf)"
    return draw_call_to_source(step.morphism, step.args, static_args=step.static_args)


def prior_table(
    module: Module,
    targets: Sequence[PriorTarget],
    user: Mapping[str, PriorSpec],
    *,
    policy: DefaultPriorPolicy,
) -> PriorTable:
    """Tabulate the effective prior of every target of an emitted module.

    Parameters
    ----------
    module : Module
        Final emitted module, after every prior rewrite.
    targets : Sequence[PriorTarget]
        Overrideable sites.
    user : Mapping[str, PriorSpec]
        Winning user rule of each site a user rule selects.
    policy : {"quivers", "brms"}
        Default policy of the remaining sites.

    Returns
    -------
    PriorTable
        One row per target present in the module.
    """
    steps: dict[str, SampleStep | LetStep] = {}
    for statement in module.statements:
        if isinstance(statement, ProgramDecl):
            for step in statement.draws:
                if isinstance(step, SampleStep):
                    steps[step.vars[0]] = step
                elif isinstance(step, LetStep):
                    steps.setdefault(step.name, step)
    rows = []
    for target in targets:
        step = steps.get(target.site)
        if step is None:
            continue
        rule = user.get(target.site)
        lower, upper = target.lower, target.upper
        if rule is not None:
            if rule.lower is not None:
                lower = rule.lower if lower is None else max(lower, rule.lower)
            if rule.upper is not None:
                upper = rule.upper if upper is None else min(upper, rule.upper)
        rows.append(
            PriorRow(
                site=target.site,
                prior=_step_prior(step),
                class_=target.class_,
                coef=target.coef,
                group=target.group,
                response=target.response,
                dpar=target.dpar,
                nlpar=target.nlpar,
                lower=lower,
                upper=upper,
                tag=None if rule is None else rule.tag,
                source="default" if rule is None else "user",
            )
        )
    return PriorTable(rows=tuple(rows), policy=policy)


def user_selection(
    priors: PriorSet, targets: Sequence[PriorTarget]
) -> dict[str, PriorSpec]:
    """Resolve user rules against targets for table attribution.

    Parameters
    ----------
    priors : PriorSet
        Authored rules.
    targets : Sequence[PriorTarget]
        Sites they were validated against.

    Returns
    -------
    dict[str, PriorSpec]
        Winning rule per selected site.
    """
    return resolve_priors(priors, targets) if priors.rules else {}


def join_main_rules(
    selected: Mapping[str, PriorSpec], targets: Sequence[PriorTarget]
) -> dict[str, PriorSpec]:
    """Join shrinkage rules of one predictor under the rule marked main.

    brms joins every shrinkage prior of one predictor (one response,
    distributional and nonlinear parameter) into a single hierarchy and takes
    its arguments from the rule marked ``main = TRUE``; with one rule, that
    rule is main.

    Parameters
    ----------
    selected : Mapping[str, PriorSpec]
        Resolved rules by site.
    targets : Sequence[PriorTarget]
        Role metadata.

    Returns
    -------
    dict[str, PriorSpec]
        Rules whose hierarchies take the main rule's arguments and share one
        block per predictor when any rule of that predictor is main.

    Raises
    ------
    ValueError
        If a predictor marks several rules main, mixes horseshoe and R2D2,
        or combines main with an explicit block.
    """
    by_site = {target.site: target for target in targets}
    scopes: dict[tuple[str | None, str | None, str | None], list[str]] = {}
    for site, rule in selected.items():
        if isinstance(rule.distribution, (RegularizedHorseshoe, R2D2)):
            target = by_site[site]
            key = (target.response, target.dpar or "mu", target.nlpar)
            scopes.setdefault(key, []).append(site)
    result = dict(selected)
    for key, sites in scopes.items():
        rules = [selected[site] for site in sites]
        hierarchies = [rule.distribution for rule in rules]
        main_rules: list[PriorSpec] = []
        for rule in rules:
            hierarchy = rule.distribution
            if (
                isinstance(hierarchy, (RegularizedHorseshoe, R2D2))
                and hierarchy.main
                and rule not in main_rules
            ):
                main_rules.append(rule)
        if not main_rules:
            continue
        if len(main_rules) > 1:
            raise ValueError(
                "exactly one shrinkage rule per predictor may be marked main"
            )
        main = main_rules[0].distribution
        if any(type(h) is not type(main) for h in hierarchies):
            raise ValueError(
                "one predictor cannot combine horseshoe and R2D2 shrinkage"
            )
        if any(rule.block is not None for rule in rules):
            raise ValueError("main = True joins a predictor's classes; omit block")
        label = "main_" + "_".join(
            "".join(c if c.isalnum() else "_" for c in (part or "x")) for part in key
        )
        for site in sites:
            result[site] = selected[site].with_(distribution=main, block=label)
    return result
