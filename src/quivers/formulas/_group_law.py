"""The law of one block of group-level coefficients, and its planning.

A block is a matrix ``B`` of coefficients over the levels of one grouping
factor (rows) and the block's coefficient coordinates (columns). Its law is
the product of independent factors:

- the level covariance ``V`` between grouping levels, the identity or a known
  matrix on its support;
- the coefficient correlation ``R`` between coordinates, the identity, one
  LKJ factor, or one LKJ factor per level of a ``by`` variable;
- the scale ``sd`` of each coordinate, optionally indexed by the ``by`` level
  of a grouping level and by a component axis;
- a per-level Student-t mixing multiplier ``dfm``;
- a per-level weight on the standard-normal prior of the innovations;
- the row map carrying a level-indexed coefficient to the response rows.

With standard-normal innovations ``Z``, the coefficients of level ``j`` are

    b_j = mean + dfm_j A_j g_j,    A_j = diag(sd[by_j]) L_R[by_j],

for the rows ``g_j`` of ``G = L_V Z``. Noncentered raw sites carry ``Z`` and
the program applies every factor. Centered raw sites carry the product of the
factors `GroupBlockLaw.carried` lists, and the program applies the rest.

The emitter, the coordinate layout read by prediction, the coefficient
reconstruction, and the per-factor structure read by postprocessing are all
functions of these laws.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._group_covariance import GroupCovarianceBasis
from quivers.formulas._group_layout import GroupBlockLayout, GroupMemberLayout

__all__: list[str] = []


class LevelCovariance(dx.TaggedUnion, ValueModel, discriminator="kind"):
    """The covariance between the levels of a grouping factor."""


class IdentityCovariance(LevelCovariance):
    """Independent, identically scaled grouping levels."""

    kind: Literal["identity"] = "identity"


class KnownCovariance(LevelCovariance):
    """A known covariance between grouping levels, on its support.

    Parameters
    ----------
    basis : GroupCovarianceBasis
        Projection from support coordinates to the physical levels and the
        factored support covariance.
    """

    basis: GroupCovarianceBasis = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    kind: Literal["known"] = "known"


class ByLevels(ValueModel):
    """The ``by`` variable partitioning the levels of a grouping factor.

    Parameters
    ----------
    variable : str
        Authored ``by`` variable.
    levels : tuple[str, ...]
        Its levels, in scale-site order.
    codes : tuple[int, ...]
        The ``by`` level of each grouping level, in level order.
    axis : str
        Finite axis over the ``by`` levels.
    input : str
        Runtime input holding ``codes`` over the levels in use.
    """

    variable: str
    levels: tuple[str, ...]
    codes: tuple[int, ...]
    axis: str
    input: str


class StudentMixing(ValueModel):
    """The scale mixture giving a grouping factor Student-t coefficients.

    Parameters
    ----------
    df_site : str
        Degrees-of-freedom site.
    mixing_site : str
        Per-level inverse-chi-square mixing site.
    multiplier : str
        Binding of the per-level multiplier ``sqrt(df * mixing)``.
    prior : str
        Default prior of the degrees of freedom.
    lower : float
        Lower bound restricting the degrees of freedom.
    """

    df_site: str
    mixing_site: str
    multiplier: str
    prior: str = "Gamma(2.0, 0.1)"
    lower: float = 1.0


class PriorWeight(ValueModel):
    """Per-level weights on the standard-normal prior of the innovations.

    Parameters
    ----------
    input : str
        Runtime input holding one weight per level in use.
    weights : tuple[float, ...]
        Training weights, in level order.
    """

    input: str
    weights: tuple[float, ...]


#: Fixed matrices of a known level covariance read by a program: the map from
#: centered and from noncentered support coordinates to the levels, the factor
#: of the support coordinates' own covariance, and the inverse of that factor.
type BasisMatrix = Literal["projection", "factor", "row_factor", "whitening"]


class GroupLevelLaw(ValueModel):
    """The factors of a block's law that index the grouping levels.

    Parameters
    ----------
    group : str
        Authored grouping factor.
    axis : str
        Finite axis over its levels.
    size : int
        Number of levels.
    covariance : LevelCovariance
        Covariance between the levels.
    by : ByLevels or None
        Partition of the levels selecting scales and correlations.
    mixing : StudentMixing or None
        Student-t scale mixture; None for Gaussian coefficients.
    prior_weight : PriorWeight or None
        Weights on the innovations' prior; None for unit weights.
    """

    group: str
    axis: str
    size: int
    covariance: LevelCovariance = dx.field(
        default_factory=IdentityCovariance, opaque=True, extras=OPAQUE_BY_VALUE
    )
    by: ByLevels | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_VALUE)
    mixing: StudentMixing | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    prior_weight: PriorWeight | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @property
    def basis(self) -> GroupCovarianceBasis | None:
        """Return the support basis of a known level covariance.

        Returns
        -------
        GroupCovarianceBasis or None
            The basis, or None for independent levels.
        """
        covariance = self.covariance
        return covariance.basis if isinstance(covariance, KnownCovariance) else None

    @property
    def support_axis(self) -> str:
        """Return the axis over the support of a known level covariance.

        Returns
        -------
        str
            Finite axis name.
        """
        return f"FormulaGroupSupport_{self.axis}"

    def basis_input(self, matrix: BasisMatrix) -> str:
        """Name the runtime input holding one matrix of the support basis.

        Parameters
        ----------
        matrix : {"projection", "factor", "row_factor", "whitening"}
            Matrix of the basis.

        Returns
        -------
        str
            Runtime input name.
        """
        return f"__formula_level_basis_{self.axis}_{matrix}"


class CoefficientCorrelation(dx.TaggedUnion, discriminator="kind"):
    """The correlation between the coefficient coordinates of a block."""

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the Cholesky-factor sites of the correlation.

        Returns
        -------
        tuple[str, ...]
            Sample sites, empty for independent coordinates.
        """
        return ()


class IndependentCoordinates(CoefficientCorrelation):
    """Uncorrelated coefficient coordinates."""

    kind: Literal["independent"] = "independent"


class LKJCorrelation(CoefficientCorrelation):
    """One LKJ-distributed correlation shared by every grouping level.

    Parameters
    ----------
    site : str
        Cholesky-factor site.
    """

    site: str
    kind: Literal["lkj"] = "lkj"

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the Cholesky-factor site.

        Returns
        -------
        tuple[str, ...]
            The one sample site.
        """
        return (self.site,)


class ByLKJCorrelation(CoefficientCorrelation):
    """One LKJ-distributed correlation per level of the ``by`` variable.

    Parameters
    ----------
    by_sites : tuple[str, ...]
        Cholesky-factor sites in ``by``-level order.
    """

    by_sites: tuple[str, ...]
    kind: Literal["by_lkj"] = "by_lkj"

    @property
    def sites(self) -> tuple[str, ...]:
        """Return the Cholesky-factor sites.

        Returns
        -------
        tuple[str, ...]
            One sample site per ``by`` level.
        """
        return self.by_sites


class GroupScale(ValueModel):
    """The standard deviation of one member, over two optional axes.

    Parameters
    ----------
    site : str
        Standard-deviation site.
    by_axis : str or None
        Axis over the ``by`` levels, when each has its own scale.
    component_axis : str or None
        Axis over the member's components, when each has its own scale.
    cell_axis : str or None
        Flat axis of a scale with both axes, holding each ``by`` level's
        components consecutively.
    prior : str or None
        Default prior; None for the configured group-level scale prior.
    lower : float or None
        Lower bound imposed on an explicitly configured prior; None leaves
        a configured prior as written.
    """

    site: str
    by_axis: str | None = None
    component_axis: str | None = None
    cell_axis: str | None = None
    prior: str | None = None
    lower: float | None = None

    @property
    def axis(self) -> str | None:
        """Return the one finite axis the site is drawn over.

        Returns
        -------
        str or None
            The flat axis of a scale with both axes, its single axis
            otherwise, or None for a scalar scale.
        """
        return self.cell_axis or self.component_axis or self.by_axis


class GroupMean(ValueModel):
    """The affine mean of one member over its components.

    Parameters
    ----------
    site : str
        Mean site.
    prior : str
        Default prior.
    """

    site: str
    prior: str


class RowMap(dx.TaggedUnion, discriminator="kind"):
    """The map from a level-indexed coefficient to the response rows."""


class SingleLevelRows(RowMap):
    """Each row reads the coefficient of its one grouping level.

    Parameters
    ----------
    index : str
        Runtime input holding each row's level.
    covariate : str or None
        Input or binding multiplying the coefficient; None for an intercept.
    """

    index: str
    covariate: str | None = None
    kind: Literal["single"] = "single"


class MembershipRows(RowMap):
    """Each row sums weighted coefficients over its member levels.

    Parameters
    ----------
    index : str
        Runtime input of the factor's single-level row index, which the
        member indices replace.
    indices : tuple[str, ...]
        Runtime inputs holding each row's level for every member.
    covariates : tuple[str, ...]
        Runtime inputs holding the product of each member's weight and
        covariate.
    covariate : str or None
        Program binding multiplying the sum, for a covariate that every
        member shares and the program computes.
    """

    index: str
    indices: tuple[str, ...]
    covariates: tuple[str, ...]
    covariate: str | None = None
    kind: Literal["membership"] = "membership"


class GroupMemberLaw(ValueModel):
    """One named coefficient of a block.

    Parameters
    ----------
    prefix, slope : str
        Predictor namespace and coefficient label.
    axis : str or None
        Component axis; None for a scalar coefficient.
    width, offset : int
        Component count and first coordinate within the block.
    coefficient : str
        Binding of the coefficient over the grouping levels.
    raw_site : str or None
        Sample site carrying the member's coordinates; None for a block
        whose level covariance has rank zero.
    scale : GroupScale
        Its standard deviation.
    rows : RowMap
        Its map to the response rows.
    mean : GroupMean or None
        Affine mean of the coefficient; None for a zero mean.
    cell_axis : str or None
        Flat axis over the levels and components, level-major, indexing the
        raw site of a member with a component axis that owns its site.
    """

    prefix: str
    slope: str
    axis: str | None
    width: int
    offset: int
    coefficient: str
    raw_site: str | None
    scale: GroupScale = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rows: RowMap = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    mean: GroupMean | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_VALUE)
    cell_axis: str | None = None

    @property
    def mean_site(self) -> str | None:
        """Return the site of the member's affine mean.

        Returns
        -------
        str or None
            Mean site, or None for a zero mean.
        """
        return self.mean.site if self.mean is not None else None

    @property
    def key(self) -> tuple[str, str]:
        """Return the member's identity within its grouping factor.

        Returns
        -------
        tuple[str, str]
            Predictor namespace and coefficient label.
        """
        return self.prefix, self.slope


class LevelSite(ValueModel):
    """A sample site holding values for every level of a grouping factor.

    Given the sites without a level axis, the values of distinct levels are
    independent, so a new level draws its values from the site's own prior.

    Parameters
    ----------
    group : str
        Authored grouping factor.
    site : str
        Sample site, with the levels on its first axis.
    width : int
        Number of values per level.
    standard_normal : bool
        Whether the values are independent standard-normal innovations.
    """

    group: str
    site: str
    width: int
    standard_normal: bool = False


#: A factor a raw site may carry: the product of the level covariance and the
#: innovations is always carried or mapped by the block's basis.
type CarriedFactor = Literal["correlation", "scale", "mixing", "mean"]


class GroupBlockLaw(ValueModel):
    """The joint law of one block of group-level coefficients.

    Parameters
    ----------
    identity : str
        Block identity, unique within the grouping factor.
    level : GroupLevelLaw
        Factors over the grouping levels.
    members : tuple[GroupMemberLaw, ...]
        Coefficients in coordinate order.
    correlation : CoefficientCorrelation
        Correlation between the coordinates.
    parameterization : {"centered", "noncentered"}
        Whether the raw sites carry coefficients or standard-normal
        innovations.
    placement : {"joint", "inline"}
        Whether the block is emitted ahead of the predictors or where its
        one member's predictor is emitted.
    coordinate_axis : str or None
        Axis over the block's coordinates; None for one scalar coefficient.
    support_site : str or None
        Name of the raw site in the support coordinates of a known level
        covariance, which the members carry when the covariance of the
        levels in use has positive rank; None for independent levels.
    """

    identity: str
    level: GroupLevelLaw = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    members: tuple[GroupMemberLaw, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    correlation: CoefficientCorrelation = dx.field(
        default_factory=IndependentCoordinates, opaque=True, extras=OPAQUE_BY_VALUE
    )
    parameterization: Literal["centered", "noncentered"] = "noncentered"
    placement: Literal["joint", "inline"] = "joint"
    coordinate_axis: str | None = None
    support_site: str | None = None

    @property
    def group(self) -> str:
        """Return the authored grouping factor.

        Returns
        -------
        str
            Grouping-factor name.
        """
        return self.level.group

    @property
    def width(self) -> int:
        """Return the number of coefficient coordinates.

        Returns
        -------
        int
            Sum of the member widths.
        """
        return sum(member.width for member in self.members)

    @property
    def carried(self) -> frozenset[CarriedFactor]:
        """Return the factors the raw sites' own law includes.

        Returns
        -------
        frozenset[CarriedFactor]
            Nothing for noncentered innovations. Centered coordinates carry
            the coefficients, hence every factor, and centered support
            coordinates of a known level covariance carry the Gaussian
            deviations before mixing and mean.
        """
        if self.parameterization == "noncentered":
            return frozenset()
        if self.level.basis is not None:
            return frozenset({"correlation", "scale"})
        return frozenset({"correlation", "scale", "mixing", "mean"})

    @property
    def joint_site(self) -> bool:
        """Return whether one raw site carries every coordinate of the block.

        Returns
        -------
        bool
            True for a known level covariance and for a carried correlation;
            otherwise each member owns its raw site.
        """
        return self.level.basis is not None or (
            "correlation" in self.carried
            and not isinstance(self.correlation, IndependentCoordinates)
        )

    @property
    def level_column_factors(self) -> bool:
        """Return whether the carried column factor differs between levels.

        Returns
        -------
        bool
            True when the raw sites carry scales of a ``by`` partition.
        """
        return "scale" in self.carried and self.level.by is not None

    @property
    def level_sites(self) -> tuple[LevelSite, ...]:
        """Return the sites indexed by the grouping levels.

        Returns
        -------
        tuple[LevelSite, ...]
            The raw sites of independent levels, in member order, and the
            mixing site, which the blocks of a factor share. The support
            coordinates of a known level covariance index no level.
        """
        widths: dict[str, int] = {}
        if self.level.basis is None:
            for member in self.members:
                if member.raw_site is not None:
                    widths[member.raw_site] = (
                        widths.get(member.raw_site, 0) + member.width
                    )
        sites = [
            LevelSite(
                group=self.group,
                site=site,
                width=width,
                standard_normal=not self.carried,
            )
            for site, width in widths.items()
        ]
        if self.level.mixing is not None:
            sites.append(
                LevelSite(group=self.group, site=self.level.mixing.mixing_site, width=1)
            )
        return tuple(sites)

    @property
    def layout(self) -> GroupBlockLayout:
        """Return the coordinate layout read by prediction and postprocessing.

        Returns
        -------
        GroupBlockLayout
            Raw sites, scale sites, and coordinate slices of the block.
        """
        correlation = self.correlation
        return GroupBlockLayout(
            group=self.group,
            identity=self.identity,
            coordinate_axis=self.coordinate_axis,
            parameterization=self.parameterization,
            members=tuple(
                GroupMemberLayout(
                    prefix=member.prefix,
                    slope=member.slope,
                    axis=member.axis,
                    width=member.width,
                    offset=member.offset,
                    coefficient=member.coefficient,
                    raw_site=member.raw_site,
                    scale_site=member.scale.site,
                    mean_site=member.mean_site,
                )
                for member in self.members
            ),
            correlation_site=correlation.site
            if isinstance(correlation, LKJCorrelation)
            else None,
            level_covariance=self.level.basis,
        )


def _rename(names: Mapping[str, str], value: str | None) -> str | None:
    """Rename an optional name, keeping names outside the mapping.

    Parameters
    ----------
    names : Mapping[str, str]
        Local names and their replacements.
    value : str or None
        Optional local name.

    Returns
    -------
    str or None
        The replacement, the name itself when it has none, or None.
    """
    return None if value is None else names.get(value, value)


def renamed_law(
    law: GroupBlockLaw,
    names: Mapping[str, str],
    objects: Mapping[str, str],
    prefix: Callable[[str], str],
    identity: str,
) -> GroupBlockLaw:
    """Carry a law into another namespace of bindings and objects.

    Parameters
    ----------
    law : GroupBlockLaw
        Law in its local namespace.
    names : Mapping[str, str]
        Local sites, bindings and inputs and their replacements.
    objects : Mapping[str, str]
        Local axes and their replacements.
    prefix : Callable[[str], str]
        Map from a local predictor namespace to its replacement.
    identity : str
        Block identity in the new namespace.

    Returns
    -------
    GroupBlockLaw
        The same law with every identity, axis, site and input renamed.
    """
    level = law.level
    by, mixing, weight = level.by, level.mixing, level.prior_weight
    correlation = law.correlation
    if isinstance(correlation, LKJCorrelation):
        correlation = correlation.with_(
            site=names.get(correlation.site, correlation.site)
        )
    elif isinstance(correlation, ByLKJCorrelation):
        correlation = correlation.with_(
            by_sites=tuple(names.get(site, site) for site in correlation.by_sites)
        )
    members = []
    for member in law.members:
        rows = member.rows
        if isinstance(rows, SingleLevelRows):
            rows = rows.with_(
                index=names.get(rows.index, rows.index),
                covariate=_rename(names, rows.covariate),
            )
        elif isinstance(rows, MembershipRows):
            rows = rows.with_(
                index=names.get(rows.index, rows.index),
                indices=tuple(names.get(name, name) for name in rows.indices),
                covariates=tuple(names.get(name, name) for name in rows.covariates),
                covariate=_rename(names, rows.covariate),
            )
        members.append(
            member.with_(
                prefix=prefix(member.prefix),
                axis=_rename(objects, member.axis),
                coefficient=names.get(member.coefficient, member.coefficient),
                raw_site=_rename(names, member.raw_site),
                mean=None
                if member.mean is None
                else member.mean.with_(
                    site=names.get(member.mean.site, member.mean.site)
                ),
                cell_axis=_rename(objects, member.cell_axis),
                scale=member.scale.with_(
                    site=names.get(member.scale.site, member.scale.site),
                    by_axis=_rename(objects, member.scale.by_axis),
                    component_axis=_rename(objects, member.scale.component_axis),
                    cell_axis=_rename(objects, member.scale.cell_axis),
                ),
                rows=rows,
            )
        )
    return law.with_(
        identity=identity,
        coordinate_axis=_rename(objects, law.coordinate_axis),
        support_site=_rename(names, law.support_site),
        correlation=correlation,
        members=tuple(members),
        level=level.with_(
            axis=objects.get(level.axis, level.axis),
            by=None
            if by is None
            else by.with_(
                axis=objects.get(by.axis, by.axis), input=names.get(by.input, by.input)
            ),
            mixing=None
            if mixing is None
            else mixing.with_(
                df_site=names.get(mixing.df_site, mixing.df_site),
                mixing_site=names.get(mixing.mixing_site, mixing.mixing_site),
                multiplier=names.get(mixing.multiplier, mixing.multiplier),
            ),
            prior_weight=None
            if weight is None
            else weight.with_(input=names.get(weight.input, weight.input)),
        ),
    )


class LawInput(ValueModel):
    """One runtime input the blocks of a program read.

    Parameters
    ----------
    name : str
        Input name.
    element : {"Int", "Real"}
        Element carrier.
    axes : tuple[str, ...]
        Finite axes of the input.
    value : torch.Tensor or None
        Training value the law retains; None for row data the design holds.
    """

    name: str
    element: Literal["Int", "Real"]
    axes: tuple[str, ...]
    value: torch.Tensor | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )


def law_inputs(laws: Iterable[GroupBlockLaw]) -> tuple[LawInput, ...]:
    """List the runtime inputs that the blocks of a program read.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Laws of the blocks.

    Returns
    -------
    tuple[LawInput, ...]
        Each input once: the member indices and weighted covariates of
        multi-membership rows, then per grouping factor the ``by`` codes, the
        prior weights, and the matrices of a known level covariance.
    """
    laws = tuple(laws)
    inputs: dict[str, LawInput] = {}

    def add(
        name: str,
        element: Literal["Int", "Real"],
        axes: tuple[str, ...],
        value: torch.Tensor | None = None,
    ) -> None:
        inputs.setdefault(
            name, LawInput(name=name, element=element, axes=axes, value=value)
        )

    for law in laws:
        for member in law.members:
            rows = member.rows
            if isinstance(rows, MembershipRows):
                for index, covariate in zip(rows.indices, rows.covariates, strict=True):
                    add(index, "Int", ("Resp",))
                    add(covariate, "Real", ("Resp",))
    for law in laws:
        level = law.level
        if level.by is not None:
            add(
                level.by.input,
                "Int",
                (level.axis,),
                torch.tensor(level.by.codes, dtype=torch.long),
            )
        if level.prior_weight is not None:
            add(
                level.prior_weight.input,
                "Real",
                (level.axis,),
                torch.tensor(level.prior_weight.weights, dtype=torch.float64),
            )
        basis = level.basis
        if basis is None or not basis.rank:
            continue
        support = level.support_axis
        projection, _, factor = basis.tensors()
        coordinate = torch.tensor(basis.coordinate_factor, dtype=torch.float64)
        if "scale" in law.carried:
            add(
                level.basis_input("projection"),
                "Real",
                (level.axis, support),
                projection,
            )
            add(level.basis_input("row_factor"), "Real", (support, support), coordinate)
        else:
            add(level.basis_input("factor"), "Real", (level.axis, support), factor)
            add(
                level.basis_input("row_factor"),
                "Real",
                (support, support),
                torch.eye(basis.rank, dtype=torch.float64),
            )
        if level.prior_weight is not None and "scale" in law.carried:
            add(
                level.basis_input("whitening"),
                "Real",
                (support, level.axis),
                torch.linalg.solve_triangular(
                    coordinate,
                    torch.eye(basis.rank, dtype=torch.float64),
                    upper=False,
                ),
            )
    return tuple(inputs.values())


def level_names(laws: Iterable[GroupBlockLaw]) -> frozenset[str]:
    """Name what the blocks of one grouping factor share in a program.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Laws of the blocks.

    Returns
    -------
    frozenset[str]
        Level axes, row indices, ``by`` axes and inputs, Student-t sites and
        multipliers, prior-weight inputs, and the axes and inputs of known
        level covariances.
    """
    names: set[str] = set()
    for law in laws:
        level = law.level
        names.add(level.axis)
        for member in law.members:
            rows = member.rows
            if isinstance(rows, SingleLevelRows):
                names.add(rows.index)
            elif isinstance(rows, MembershipRows):
                names.update((rows.index, *rows.indices))
        if level.by is not None:
            names.update((level.by.axis, level.by.input))
        if level.mixing is not None:
            names.update(
                (
                    level.mixing.df_site,
                    level.mixing.mixing_site,
                    level.mixing.multiplier,
                )
            )
        if level.prior_weight is not None:
            names.add(level.prior_weight.input)
        if level.basis is not None:
            names.add(level.support_axis)
            names.update(
                level.basis_input(matrix)
                for matrix in ("projection", "factor", "row_factor", "whitening")
            )
    return frozenset(names)


def option_prior_sites(
    laws: Iterable[GroupBlockLaw],
) -> tuple[tuple[str, str, str, str | None, float | None], ...]:
    """List the overrideable sites the level factors of the blocks add.

    Parameters
    ----------
    laws : Iterable[GroupBlockLaw]
        Emitted group blocks.

    Returns
    -------
    tuple[tuple[str, str, str, str or None, float or None], ...]
        Site, prior class (``df`` or ``cor``), group, coefficient label (the
        ``by`` level of a correlation), and lower bound: each Student-t
        degrees-of-freedom site once, then every per-``by``-level
        correlation.
    """
    laws = tuple(laws)
    degrees: dict[str, tuple[str, str, str, str | None, float | None]] = {}
    for law in laws:
        mixing = law.level.mixing
        if mixing is not None:
            degrees.setdefault(
                mixing.df_site, (mixing.df_site, "df", law.group, None, mixing.lower)
            )
    sites = list(degrees.values())
    for law in laws:
        correlation, by = law.correlation, law.level.by
        if isinstance(correlation, ByLKJCorrelation) and by is not None:
            for label, site in zip(by.levels, correlation.by_sites, strict=True):
                sites.append((site, "cor", law.group, label, None))
    return tuple(sites)


def _by_codes(law: GroupBlockLaw, codes: torch.Tensor | None) -> torch.Tensor | None:
    """Read the ``by`` level of each grouping level a computation ranges over.

    Parameters
    ----------
    law : GroupBlockLaw
        Block law.
    codes : torch.Tensor or None
        ``by`` level of each level in use; None for the training levels.

    Returns
    -------
    torch.Tensor or None
        Integer codes, or None for a block without a ``by`` partition.
    """
    if law.level.by is None:
        return None
    if codes is None:
        return torch.tensor(law.level.by.codes, dtype=torch.long)
    return codes.to(dtype=torch.long)


def level_scales(
    draws: Mapping[str, torch.Tensor],
    law: GroupBlockLaw,
    *,
    sample_ndims: int,
    codes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Read the standard deviation of every coordinate at every level.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with ``sample_ndims`` leading sample axes.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.
    codes : torch.Tensor or None
        ``by`` level of each level in use; None for the training levels.

    Returns
    -------
    torch.Tensor
        Sample axes, levels (one for scales without a ``by`` axis),
        coordinates.

    Raises
    ------
    ValueError
        If a scale has a ``by`` axis and the levels have no ``by`` codes.
    """
    codes = _by_codes(law, codes)
    scales = []
    for member in law.members:
        scale = draws[member.scale.site]
        scale = scale.reshape(*scale.shape[:sample_ndims], -1, member.width)
        if member.scale.by_axis is not None:
            if codes is None:
                raise ValueError(
                    f"scale {member.scale.site!r} is indexed by a by level and "
                    "the grouping levels have no by codes"
                )
            scale = scale.index_select(sample_ndims, codes.to(scale.device))
        scales.append(scale)
    count = max(scale.shape[-2] for scale in scales)
    return torch.cat(
        [scale.expand(*scale.shape[:-2], count, scale.shape[-1]) for scale in scales],
        -1,
    )


def block_mean(
    draws: Mapping[str, torch.Tensor], law: GroupBlockLaw, *, like: torch.Tensor
) -> torch.Tensor:
    """Read the affine mean of every coordinate.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites.
    law : GroupBlockLaw
        Block law.
    like : torch.Tensor
        Values shaped sample axes, levels, coordinates.

    Returns
    -------
    torch.Tensor
        Sample axes, one, coordinates; zero for a member without a mean.
    """
    shape = like.shape[:-2]
    return torch.cat(
        [
            draws[member.mean.site].to(like).reshape(*shape, 1, member.width)
            if member.mean is not None
            else like.new_zeros(*shape, 1, member.width)
            for member in law.members
        ],
        -1,
    )


def level_multiplier(
    draws: Mapping[str, torch.Tensor], law: GroupBlockLaw, *, like: torch.Tensor
) -> torch.Tensor | None:
    """Read the Student-t mixing multiplier of every level.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites.
    law : GroupBlockLaw
        Block law.
    like : torch.Tensor
        Values shaped sample axes, levels, coordinates.

    Returns
    -------
    torch.Tensor or None
        ``sqrt(df * mixing)`` shaped sample axes, levels, one; None for
        Gaussian coefficients.
    """
    mixing = law.level.mixing
    if mixing is None:
        return None
    shape = like.shape[:-2]
    df = draws[mixing.df_site].to(like).reshape(*shape, 1, 1)
    return (df * draws[mixing.mixing_site].to(like).reshape(*shape, -1, 1)).sqrt()


def _raw_rows(
    draws: Mapping[str, torch.Tensor], law: GroupBlockLaw, *, sample_ndims: int
) -> torch.Tensor:
    """Read the block's raw sites as one array over rows and coordinates.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with ``sample_ndims`` leading sample axes.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.

    Returns
    -------
    torch.Tensor
        Sample axes, grouping levels or support coordinates, coordinates.

    Raises
    ------
    ValueError
        If a member of a block with a level covariance of positive rank has
        no raw site.
    """
    basis = law.level.basis
    if basis is not None and not basis.rank:
        scale = draws[law.members[0].scale.site]
        return scale.new_zeros(*scale.shape[:sample_ndims], 0, law.width)
    widths: dict[str, int] = {}
    for member in law.members:
        if member.raw_site is None:
            raise ValueError(f"group block {law.identity!r} has no retained raw site")
        widths[member.raw_site] = widths.get(member.raw_site, 0) + member.width
    return torch.cat(
        [
            draws[site].reshape(*draws[site].shape[:sample_ndims], -1, width)
            for site, width in widths.items()
        ],
        -1,
    )


def _level_rows(
    draws: Mapping[str, torch.Tensor], law: GroupBlockLaw, *, sample_ndims: int
) -> torch.Tensor:
    """Map the raw sites to the grouping levels.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with ``sample_ndims`` leading sample axes, on the
        training levels.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.

    Returns
    -------
    torch.Tensor
        Sample axes, levels, coordinates, with the level covariance between
        levels and the factors the block carries within a level. Values of
        a known covariance are computed in double precision.
    """
    raw = _raw_rows(draws, law, sample_ndims=sample_ndims)
    basis = law.level.basis
    if basis is None:
        return raw
    raw = raw.to(dtype=torch.promote_types(raw.dtype, torch.float64))
    projection, _, factor = basis.tensors(like=raw)
    return (projection if "scale" in law.carried else factor) @ raw


def _correlate(
    rows: torch.Tensor,
    draws: Mapping[str, torch.Tensor],
    law: GroupBlockLaw,
    *,
    sample_ndims: int,
    codes: torch.Tensor | None,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or remove the correlation factor of each level's coordinates.

    Parameters
    ----------
    rows : torch.Tensor
        Sample axes, levels, coordinates.
    draws : Mapping[str, torch.Tensor]
        Constrained sites holding the Cholesky factors.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.
    codes : torch.Tensor or None
        ``by`` level of each level in use; None for the training levels.
    inverse : bool
        Whether to solve by the factor instead of multiplying by it.

    Returns
    -------
    torch.Tensor
        Row ``j`` multiplied, or solved, by the factor of its ``by`` level.

    Raises
    ------
    ValueError
        If the correlation has one factor per ``by`` level and the levels
        have no ``by`` codes.
    """
    correlation = law.correlation
    if isinstance(correlation, LKJCorrelation):
        factor = draws[correlation.site].to(rows)
        if inverse:
            return torch.linalg.solve_triangular(
                factor, rows.transpose(-1, -2), upper=False
            ).transpose(-1, -2)
        return rows @ factor.transpose(-1, -2)
    if isinstance(correlation, ByLKJCorrelation):
        codes = _by_codes(law, codes)
        if codes is None:
            raise ValueError(
                "per-by-level correlations require the by codes of the levels"
            )
        factors = torch.stack(
            [draws[site].to(rows) for site in correlation.by_sites], dim=sample_ndims
        ).index_select(sample_ndims, codes.to(rows.device))
        column = rows.unsqueeze(-1)
        if inverse:
            return torch.linalg.solve_triangular(factors, column, upper=False).squeeze(
                -1
            )
        return (factors @ column).squeeze(-1)
    return rows


def column_deviations(
    gaussian: torch.Tensor,
    draws: Mapping[str, torch.Tensor],
    law: GroupBlockLaw,
    *,
    sample_ndims: int,
    codes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a block's column factor to rows with identity column covariance.

    Parameters
    ----------
    gaussian : torch.Tensor
        Sample axes, levels, coordinates, with the level covariance between
        rows and independent standard coordinates within a row.
    draws : Mapping[str, torch.Tensor]
        Constrained hyperparameter sites.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.
    codes : torch.Tensor or None
        ``by`` level of each row; None for the training levels.

    Returns
    -------
    torch.Tensor
        Rows ``diag(sd[by_j]) L[by_j] g_j``: the deviation of level ``j``'s
        coefficients from the block mean, before any Student-t mixing.
    """
    scale = level_scales(draws, law, sample_ndims=sample_ndims, codes=codes)
    correlated = _correlate(
        gaussian, draws, law, sample_ndims=sample_ndims, codes=codes
    )
    return correlated * scale.to(gaussian)


def block_gaussian(
    draws: Mapping[str, torch.Tensor], law: GroupBlockLaw, *, sample_ndims: int
) -> torch.Tensor:
    """Read a block before its column factor, mixing and mean.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with ``sample_ndims`` leading sample axes, on the
        training levels.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.

    Returns
    -------
    torch.Tensor
        Sample axes, levels, coordinates: rows with the level covariance
        between levels and independent standard coordinates, ``L_V Z`` for
        the standard-normal innovations ``Z``.
    """
    rows = _level_rows(draws, law, sample_ndims=sample_ndims)
    if "mean" in law.carried:
        rows = rows - block_mean(draws, law, like=rows)
    multiplier = level_multiplier(draws, law, like=rows)
    if "mixing" in law.carried and multiplier is not None:
        rows = rows / multiplier
    if "scale" in law.carried:
        rows = rows / level_scales(draws, law, sample_ndims=sample_ndims).to(rows)
    if "correlation" in law.carried:
        rows = _correlate(
            rows, draws, law, sample_ndims=sample_ndims, codes=None, inverse=True
        )
    return rows


def block_coefficients(
    draws: Mapping[str, torch.Tensor], law: GroupBlockLaw, *, sample_ndims: int
) -> torch.Tensor:
    """Decode the coefficients of a block from its raw sites.

    Every factor the raw sites do not carry is applied: the column factor of
    each level's ``by`` level, the Student-t mixing multiplier, and the mean.
    Cost is O(levels x width^2) per draw, plus O(levels x rank x width) for a
    known level covariance.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with ``sample_ndims`` leading sample axes, on the
        training levels.
    law : GroupBlockLaw
        Block law.
    sample_ndims : int
        Number of leading sample axes.

    Returns
    -------
    torch.Tensor
        Sample axes, levels, coordinates.
    """
    rows = _level_rows(draws, law, sample_ndims=sample_ndims)
    if "scale" not in law.carried:
        rows = column_deviations(rows, draws, law, sample_ndims=sample_ndims)
    multiplier = level_multiplier(draws, law, like=rows)
    if "mixing" not in law.carried and multiplier is not None:
        rows = rows * multiplier
    if "mean" not in law.carried and any(
        member.mean is not None for member in law.members
    ):
        rows = rows + block_mean(draws, law, like=rows)
    return rows


def encode_support(
    gaussian: torch.Tensor,
    basis: GroupCovarianceBasis,
    draws: Mapping[str, torch.Tensor],
    law: GroupBlockLaw,
    *,
    parameterization: Literal["centered", "noncentered"],
    codes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Encode one draw of a block in the support coordinates of a basis.

    Parameters
    ----------
    gaussian : torch.Tensor
        Levels of the basis by coordinates, with the level covariance
        between rows and independent standard coordinates within a row.
    basis : GroupCovarianceBasis
        Support of the levels being encoded.
    draws : Mapping[str, torch.Tensor]
        Constrained hyperparameter sites of one draw.
    law : GroupBlockLaw
        Block law.
    parameterization : {"centered", "noncentered"}
        Whether the coordinates carry deviations with the block's column
        factor or standardized innovations.
    codes : torch.Tensor or None
        ``by`` level of each row; None for the training levels.

    Returns
    -------
    torch.Tensor
        Support by coordinates, empty for a basis of rank zero.
    """
    projection, _, _ = basis.tensors(like=gaussian)
    if parameterization == "centered" or not basis.rank:
        return projection.T @ column_deviations(
            gaussian, draws, law, sample_ndims=0, codes=codes
        )
    coordinate_factor = torch.tensor(
        basis.coordinate_factor, dtype=gaussian.dtype, device=gaussian.device
    ).reshape(basis.rank, basis.rank)
    return torch.linalg.solve_triangular(
        coordinate_factor, projection.T @ gaussian, upper=False
    )


def group_coefficients(
    draws: Mapping[str, torch.Tensor],
    laws: Sequence[GroupBlockLaw],
    levels: Mapping[str, Sequence[str]],
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Reconstruct the coefficients of every block from its retained sites.

    Parameters
    ----------
    draws : Mapping[str, torch.Tensor]
        Constrained sites with chain and draw axes.
    laws : Sequence[GroupBlockLaw]
        Laws of the emitted blocks.
    levels : Mapping[str, Sequence[str]]
        Authored group labels in retained level order.

    Returns
    -------
    dict[tuple[str, str, str], torch.Tensor]
        Prefix, group, and slope to arrays shaped chain, draw, level and,
        for a member with a component axis, component.

    Raises
    ------
    ValueError
        If the retained levels of a factor disagree with its ``by`` codes or
        with the levels its raw sites index.
    """
    result = {}
    for law in laws:
        count = len(levels[law.group])
        by = law.level.by
        if by is not None and len(by.codes) != count:
            raise ValueError(
                f"grouping factor {law.group!r} has {count} levels but "
                f"{len(by.codes)} by-level codes"
            )
        coefficients = block_coefficients(draws, law, sample_ndims=2)
        if coefficients.shape[-2] != count:
            raise ValueError(
                f"group block {law.identity!r} retains {coefficients.shape[-2]} "
                f"levels of {law.group!r}, which has {count}"
            )
        for member in law.members:
            value = coefficients[..., member.offset : member.offset + member.width]
            result[(member.prefix, law.group, member.slope)] = (
                value if member.axis is not None else value.squeeze(-1)
            )
    return result
