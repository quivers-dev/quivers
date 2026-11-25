"""Structured latent coordinates added to a formula's linear predictor.

A latent term draws one vector of coordinates from a structured law and adds
``value[index]`` to the location predictor, where ``index`` maps each response
row to a coordinate. brms's conditional autoregressive terms (``car``) use
spatial locations as coordinates, and its latent residuals use response rows.
The coordinates are ordinary latent sites, so the shared inference drivers,
posterior replay, prior catalogue, and artifact codecs treat them like group
effects.

The intrinsic CAR laws live on the subspace that sums to zero within every
connected component of the adjacency graph. `ComponentSumToZero` names that
support, and its registered bijection maps the free coordinates (one fewer
than each component's size) onto it.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
import torch
from torch.distributions import Distribution, Independent, Transform, constraints
from torch.distributions.constraint_registry import biject_to
from torch.types import _size

from quivers._values import OPAQUE_BY_VALUE, ValueModel, opaque_by_value
from quivers.dsl.ast_nodes import (
    DrawArgNamed,
    LetExprIndex,
    LetExprLiteral,
    ProgramStep,
    SampleStep,
    TypeName,
    atom_to_draw_arg,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._residual_dependence import ResidualLayout, ResidualResponse
from quivers.formulas._ast import _add, _call, _let, _mul, _sub, _var
from quivers.formulas._term import (
    AxisDecl,
    Contribution,
    Emission,
    InputDecl,
    OutputAxis,
    SiteDecl,
)
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

if TYPE_CHECKING:
    from quivers.formulas.formula import Formula

CarType = Literal["escar", "esicar", "icar", "bym2"]
"""brms's conditional autoregressive types."""


def _frozen(value: np.ndarray) -> np.ndarray:
    """Copy a numeric array once and clear its writeable flag.

    Parameters
    ----------
    value : np.ndarray
        Known array.

    Returns
    -------
    np.ndarray
        Read-only copy.
    """
    frozen = np.array(value, copy=True, order="C")
    frozen.setflags(write=False)
    return frozen


class ComponentSumToZero(constraints.Constraint):
    """Vectors summing to zero within each listed component.

    Parameters
    ----------
    components : tuple[tuple[int, ...], ...]
        Coordinate indices of each component, partitioning ``range(size)``.
    size : int
        Vector length.
    """

    event_dim = 1

    def __init__(self, components: tuple[tuple[int, ...], ...], size: int) -> None:
        self.components = components
        self.size = size
        super().__init__()

    def unconstrained_dim(self, constrained_dim: int) -> int:
        """Count the free coordinates: the size less one per component.

        Parameters
        ----------
        constrained_dim : int
            Vector length.

        Returns
        -------
        int
            Number of free coordinates.
        """
        return constrained_dim - len(self.components)

    def check(self, value: torch.Tensor) -> torch.Tensor:
        """Test every component sum against a rounding tolerance.

        Parameters
        ----------
        value : torch.Tensor
            Candidate vectors.

        Returns
        -------
        torch.Tensor
            Boolean result per vector.
        """
        result = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
        tolerance = 1e3 * torch.finfo(value.dtype).eps
        for members in self.components:
            part = value[..., list(members)]
            bound = tolerance * (1 + part.abs().sum(-1))
            result = result & (part.sum(-1).abs() <= bound)
        return result


class ComponentSumToZeroTransform(Transform):
    """Fill each component's last coordinate with minus the sum of the others.

    The map is linear and injective. A law on this support scores the free
    coordinates, so the transform contributes no Jacobian term.

    Parameters
    ----------
    constraint : ComponentSumToZero
        Target support.
    """

    domain = constraints.real_vector
    bijective = True

    def __init__(self, constraint: ComponentSumToZero) -> None:
        super().__init__(cache_size=0)
        self._constraint = constraint
        self.codomain = constraint
        free: list[int] = []
        last: list[int] = []
        owner: list[int] = []
        for index, members in enumerate(constraint.components):
            free.extend(members[:-1])
            owner.extend([index] * (len(members) - 1))
            last.append(members[-1])
        self._free = torch.as_tensor(free, dtype=torch.long)
        self._last = torch.as_tensor(last, dtype=torch.long)
        self._owner = torch.as_tensor(owner, dtype=torch.long)

    def __eq__(self, other: object) -> bool:
        """Compare transforms by their target support.

        Parameters
        ----------
        other : object
            Another transform.

        Returns
        -------
        bool
            Whether both fill the same components.
        """
        return (
            isinstance(other, ComponentSumToZeroTransform)
            and other._constraint.components == self._constraint.components
        )

    def __hash__(self) -> int:
        """Hash by the component partition.

        Returns
        -------
        int
            Hash of the components.
        """
        return hash(self._constraint.components)

    def forward_shape(self, shape: _size) -> torch.Size:
        """Map free-coordinate shapes to vector shapes.

        Parameters
        ----------
        shape : torch.Size
            Shape with trailing free coordinates.

        Returns
        -------
        torch.Size
            Shape with trailing vector length.
        """
        return torch.Size((*shape[:-1], self._constraint.size))

    def inverse_shape(self, shape: _size) -> torch.Size:
        """Map vector shapes to free-coordinate shapes.

        Parameters
        ----------
        shape : torch.Size
            Shape with trailing vector length.

        Returns
        -------
        torch.Size
            Shape with trailing free coordinates.
        """
        return torch.Size((*shape[:-1], len(self._free)))

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """Expand free coordinates into a component-centered vector.

        Parameters
        ----------
        x : torch.Tensor
            Free coordinates.

        Returns
        -------
        torch.Tensor
            Vector summing to zero within each component.
        """
        size = self._constraint.size
        full = x.new_zeros((*x.shape[:-1], size))
        full = full.index_copy(-1, self._free.to(x.device), x)
        sums = x.new_zeros((*x.shape[:-1], len(self._constraint.components)))
        sums = sums.index_add(-1, self._owner.to(x.device), x)
        return full.index_copy(-1, self._last.to(x.device), -sums)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Read the free coordinates of a component-centered vector.

        Parameters
        ----------
        y : torch.Tensor
            Constrained vector.

        Returns
        -------
        torch.Tensor
            Free coordinates.
        """
        return y[..., self._free.to(y.device)]

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Contribute nothing: laws on this support score free coordinates.

        Parameters
        ----------
        x : torch.Tensor
            Free coordinates.
        y : torch.Tensor
            Constrained vectors.

        Returns
        -------
        torch.Tensor
            Zeros over the batch shape.
        """
        return x.new_zeros(x.shape[:-1])


@biject_to.register(ComponentSumToZero)
def _component_sum_to_zero(constraint: ComponentSumToZero) -> Transform:
    """Register the free-coordinate bijection of a component-centered support.

    Parameters
    ----------
    constraint : ComponentSumToZero
        Target support.

    Returns
    -------
    Transform
        The expansion transform.
    """
    return ComponentSumToZeroTransform(constraint)


class CarGraph(ValueModel):
    """A validated adjacency graph over retained spatial locations.

    Parameters
    ----------
    locations : tuple[str, ...]
        Location labels in coordinate order.
    edges : np.ndarray
        ``(2, E)`` endpoints of each undirected edge, ``i < j``.
    neighbors : np.ndarray
        Number of neighbors of each location.
    components : tuple[tuple[int, ...], ...]
        Connected components, isolated locations included.
    eigenvalues : np.ndarray
        Eigenvalues of ``D^(-1/2) W D^(-1/2)``, empty when a location is isolated.
    log_pdet : float
        Sum over components of ``log(m_C pdet(L_C))`` for the graph Laplacian
        ``L = D - W``, the normalizing term of the intrinsic CAR law.
    inverse_scale : np.ndarray
        Per location, the reciprocal of its component's BYM2 scaling factor,
        and zero for an isolated location.
    eigenvectors : tuple[np.ndarray, ...]
        Per component, the Laplacian eigenvectors with positive eigenvalue,
        scaled by their inverse square roots, for intrinsic CAR sampling.
    """

    locations: tuple[str, ...]
    edges: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    neighbors: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    components: tuple[tuple[int, ...], ...]
    eigenvalues: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    log_pdet: float
    inverse_scale: np.ndarray = dx.field(opaque=True, extras=opaque_by_value(_frozen))
    eigenvectors: tuple[np.ndarray, ...] = dx.field(
        opaque=True, extras=opaque_by_value(lambda value: tuple(map(_frozen, value)))
    )

    @property
    def rank(self) -> int:
        """Give the intrinsic CAR rank, locations less components.

        Returns
        -------
        int
            Rank of the graph Laplacian.
        """
        return len(self.locations) - len(self.components)


def car_graph(
    matrix: np.ndarray, locations: tuple[str, ...], car_type: CarType
) -> CarGraph:
    """Validate an adjacency matrix as brms does and prepare its CAR summaries.

    Every nonzero entry marks two adjacent locations (brms converts nonzero
    values to one). The matrix must be square and symmetric with a zero
    diagonal. ``escar`` and ``esicar`` require every location to have a
    neighbor, as brms requires. The BYM2 scaling factor of a component is the
    geometric mean of the diagonal of the generalized inverse of its
    Laplacian, which is the marginal variance of the intrinsic CAR under the
    sum-to-zero constraint [@riebler2016intuitive; @morris2019spatial]. The
    dense eigendecompositions here run once at preparation, never in scoring.

    Parameters
    ----------
    matrix : np.ndarray
        Adjacency over ``locations``, in that order.
    locations : tuple[str, ...]
        Location labels.
    car_type : CarType
        brms CAR type.

    Returns
    -------
    CarGraph
        Edges, neighbor counts, components, and normalizing summaries.

    Raises
    ------
    ValueError
        If the matrix is not square, finite, symmetric, and zero on the
        diagonal, or an exact sparse type has an isolated location.
    """
    adjacency = np.asarray(matrix, dtype=np.float64)
    count = len(locations)
    if adjacency.shape != (count, count):
        raise ValueError(
            "Dimensions of 'M' for CAR terms must be equal to the number of locations."
        )
    if not np.isfinite(adjacency).all():
        raise ValueError("'M' for CAR terms must be finite.")
    if not np.array_equal(adjacency != 0, (adjacency != 0).T):
        raise ValueError("'M' for CAR terms must be symmetric.")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("'M' for CAR terms must have a zero diagonal.")
    binary = (adjacency != 0).astype(np.float64)
    neighbors = binary.sum(1)
    if car_type in {"escar", "esicar"} and np.any(neighbors == 0):
        raise ValueError(
            "For exact sparse CAR, all locations should have at least one neighbor "
            "within the provided data set. Consider using type = 'icar' instead."
        )
    rows, columns = np.nonzero(np.triu(binary, 1))
    _, labels = csgraph.connected_components(sparse.csr_matrix(binary), directed=False)
    components = tuple(
        tuple(int(i) for i in np.flatnonzero(labels == label))
        for label in range(int(labels.max()) + 1)
    )
    laplacian = np.diag(neighbors) - binary
    log_pdet = 0.0
    inverse_scale = np.zeros(count)
    vectors: list[np.ndarray] = []
    for members in components:
        size = len(members)
        if size == 1:
            vectors.append(np.zeros((1, 0)))
            continue
        values, basis = np.linalg.eigh(laplacian[np.ix_(members, members)])
        positive = values[1:]
        log_pdet += math.log(size) + float(np.log(positive).sum())
        generalized = (basis[:, 1:] / positive) @ basis[:, 1:].T
        scale = math.exp(float(np.mean(np.log(np.diag(generalized)))))
        inverse_scale[list(members)] = 1.0 / scale
        vectors.append(basis[:, 1:] / np.sqrt(positive))
    eigenvalues = (
        np.linalg.eigvalsh(binary / np.sqrt(np.outer(neighbors, neighbors)))
        if np.all(neighbors > 0)
        else np.zeros(0)
    )
    return CarGraph(
        locations=locations,
        edges=np.stack((rows, columns)).astype(np.int64),
        neighbors=neighbors,
        components=components,
        eigenvalues=eigenvalues,
        log_pdet=log_pdet,
        inverse_scale=inverse_scale,
        eigenvectors=tuple(vectors),
    )


class SparseCAR(Distribution):
    """The proper and intrinsic CAR laws of brms, scored by sparse precision.

    The proper law (``escar``) is ``N(0, [tau (D - alpha W)]^(-1))`` and costs
    O(n + E) per evaluation after the one eigendecomposition in `car_graph`:
    ``log det(D - alpha W) = sum log d_i + sum log(1 - alpha lambda_i)``. The
    intrinsic law has precision ``tau (D - W)`` on the subspace summing to zero
    within each connected component and is scored as a density of the free
    coordinates, also in O(n + E).

    Parameters
    ----------
    graph : CarGraph
        Prepared adjacency.
    scale : torch.Tensor
        Positive scale ``tau^(-1/2)``.
    alpha : torch.Tensor or None
        Proper CAR dependence in ``[0, 1)``, or None for the intrinsic law.
    """

    has_rsample = True

    def __init__(
        self, graph: CarGraph, scale: torch.Tensor, alpha: torch.Tensor | None
    ) -> None:
        self.graph = graph
        self.scale = scale
        self.alpha = alpha
        batch = (
            scale.shape
            if alpha is None
            else torch.broadcast_shapes(scale.shape, alpha.shape)
        )
        super().__init__(
            batch, torch.Size((len(graph.locations),)), validate_args=False
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
        """Give the real vectors, or the component-centered subspace.

        Returns
        -------
        constraints.Constraint
            Support of the law.
        """
        if self.alpha is not None:
            return constraints.real_vector
        return ComponentSumToZero(self.graph.components, len(self.graph.locations))

    @property
    def mean(self) -> torch.Tensor:
        """Give the zero mean.

        Returns
        -------
        torch.Tensor
            Zeros with the batch and event shape.
        """
        return self.scale.new_zeros((*self.batch_shape, *self.event_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score a location vector by its sparse precision.

        Parameters
        ----------
        value : torch.Tensor
            Location effects.

        Returns
        -------
        torch.Tensor
            Log density, minus infinity outside the parameter domain.
        """
        graph = self.graph
        first = torch.as_tensor(np.array(graph.edges[0]), device=value.device)
        second = torch.as_tensor(np.array(graph.edges[1]), device=value.device)
        precision = self.scale.square().reciprocal()
        count = len(graph.locations)
        if self.alpha is None:
            quadratic = (value[..., first] - value[..., second]).square().sum(-1)
            rank = graph.rank
            return (
                0.5 * rank * precision.log()
                + 0.5 * graph.log_pdet
                - 0.5 * rank * math.log(2 * math.pi)
                - 0.5 * precision * quadratic
            )
        neighbors = torch.as_tensor(
            graph.neighbors, dtype=value.dtype, device=value.device
        )
        eigenvalues = torch.as_tensor(
            graph.eigenvalues, dtype=value.dtype, device=value.device
        )
        alpha = self.alpha
        inside = (alpha >= 0) & (alpha < 1)
        safe = torch.where(inside, alpha, torch.zeros_like(alpha))
        cross = 2 * (value[..., first] * value[..., second]).sum(-1)
        quadratic = (neighbors * value.square()).sum(-1) - safe * cross
        log_det = (
            count * precision.log()
            + neighbors.log().sum()
            + torch.log1p(-safe[..., None] * eigenvalues).sum(-1)
        )
        score = (
            0.5 * log_det
            - 0.5 * precision * quadratic
            - 0.5 * count * math.log(2 * math.pi)
        )
        return torch.where(inside, score, torch.full_like(score, -torch.inf))

    def rsample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw location vectors for prior simulation.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Draws, each centered within components for the intrinsic law.
        """
        shape = torch.Size(sample_shape)
        graph = self.graph
        count = len(graph.locations)
        dtype = self.scale.dtype
        if self.alpha is None:
            result = self.scale.new_zeros((*shape, *self.batch_shape, count))
            for members, vectors in zip(
                graph.components, graph.eigenvectors, strict=True
            ):
                if not vectors.shape[1]:
                    continue
                basis = torch.as_tensor(vectors, dtype=dtype)
                noise = torch.randn(
                    (*shape, *self.batch_shape, vectors.shape[1]), dtype=dtype
                )
                draw = (noise @ basis.T) * self.scale[..., None]
                result = result.index_copy(-1, torch.as_tensor(members), draw)
            return result
        neighbors = np.asarray(graph.neighbors)
        adjacency = np.zeros((count, count))
        adjacency[graph.edges[0], graph.edges[1]] = 1.0
        adjacency = adjacency + adjacency.T
        alpha = self.alpha.detach().expand(self.batch_shape).reshape(-1)
        draws = []
        for a in alpha.tolist():
            factor = np.linalg.cholesky(np.diag(neighbors) - a * adjacency)
            noise = np.random.default_rng(int(torch.randint(2**31, ())))
            draws.append(
                np.linalg.solve(
                    factor.T, noise.standard_normal((math.prod(shape) or 1, count)).T
                ).T
            )
        stacked = torch.as_tensor(np.stack(draws, 1), dtype=dtype)
        stacked = stacked.reshape(*shape, *self.batch_shape, count)
        return stacked * self.scale[..., None]


def car_catalog(graph: CarGraph) -> FamilyCatalog:
    """Register the proper and intrinsic CAR laws over one graph.

    Parameters
    ----------
    graph : CarGraph
        Prepared adjacency.

    Returns
    -------
    FamilyCatalog
        Catalog with ``SparseCAR(scale, alpha, anchor)`` and
        ``SparseICAR(scale, anchor)``; ``anchor`` is a zero vector over the
        locations that fixes the event extent.
    """

    def record(name: str, alpha: bool) -> DistributionFamily:
        parameters = [FamilyParameter(name="scale", constraint="positive", rank=0)]
        if alpha:
            parameters.append(FamilyParameter(name="alpha", constraint="real", rank=0))
        parameters.append(
            FamilyParameter(
                name="anchor",
                constraint="real",
                rank=1,
                event_dims=(0,),
                scalar_broadcast=False,
            )
        )
        return DistributionFamily(
            name=name,
            parameters=tuple(parameters),
            support="real",
            element=REAL,
            event_rank=1,
            event_sources=("anchor",),
            discrete=False,
            reparameterizable=True,
            compositional=False,
        )

    def proper(parameters: Mapping[str, torch.Tensor]) -> Distribution:
        return SparseCAR(graph, parameters["scale"], parameters["alpha"])

    def intrinsic(parameters: Mapping[str, torch.Tensor]) -> Distribution:
        return SparseCAR(graph, parameters["scale"], None)

    return (
        FamilyCatalog()
        .with_family(record("SparseCAR", True), proper)
        .with_family(
            record("SparseICAR", False),
            intrinsic,
            support=ComponentSumToZero(graph.components, len(graph.locations)),
        )
    )


class _ResidualLatentFactory(ValueModel):
    """Build the latent residual law of one row layout.

    Parameters
    ----------
    layout : ResidualLayout
        Series layout; a conditional AR structure is stored as the moving
        average of the latent innovations with the AR coefficients.
    coefficients : tuple[str, ...]
        Structural coefficient sites in brms's names.
    """

    layout: ResidualLayout = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    coefficients: tuple[str, ...] = ()

    def __call__(self, parameters: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct the zero-mean residual law over all rows.

        Parameters
        ----------
        parameters : Mapping[str, torch.Tensor]
            ``scale`` (``sderr``), structural coefficients, ``anchor``, and an
            optional ``factor``.

        Returns
        -------
        Distribution
            The residual law with one event over the rows.
        """
        anchor = parameters["anchor"]
        scale = parameters["scale"]
        scale = (
            scale[..., None]
            if scale.ndim and scale.shape[-1] != anchor.shape[-1]
            else scale
        )
        loc = torch.zeros_like(anchor)
        coefficients = {
            name.replace("ar_", "ma_")
            if self.layout.kind == "arma" and not self.layout.stationary
            else name: parameters[name]
            for name in self.coefficients
        }
        law = ResidualResponse(
            self.layout,
            loc,
            scale.expand_as(loc) if scale.ndim else scale.expand(loc.shape),
            coefficients,
            None,
            parameters.get("factor"),
        )
        return Independent(law, 1)


def residual_latent_catalog(
    layout: ResidualLayout, coefficients: tuple[str, ...]
) -> FamilyCatalog:
    """Register brms's latent residual law over one row layout.

    Parameters
    ----------
    layout : ResidualLayout
        Series layout of the rows.
    coefficients : tuple[str, ...]
        Structural coefficient sites.

    Returns
    -------
    FamilyCatalog
        Catalog with ``LatentResidual(scale, coefficients..., [factor], anchor)``.
    """
    parameters = [FamilyParameter(name="scale", constraint="positive", rank=0)]
    parameters.extend(
        FamilyParameter(name=name, constraint="real", rank=0) for name in coefficients
    )
    if layout.kind == "unstr":
        parameters.append(
            FamilyParameter(
                name="factor",
                constraint="lower_cholesky",
                rank=2,
                scalar_broadcast=False,
            )
        )
    parameters.append(
        FamilyParameter(
            name="anchor",
            constraint="real",
            rank=1,
            event_dims=(0,),
            scalar_broadcast=False,
        )
    )
    record = DistributionFamily(
        name="LatentResidual",
        parameters=tuple(parameters),
        support="real",
        element=REAL,
        event_rank=1,
        event_sources=("anchor",),
        discrete=False,
        reparameterizable=True,
        compositional=False,
    )
    return FamilyCatalog().with_family(
        record, _ResidualLatentFactory(layout=layout, coefficients=coefficients)
    )


class LatentTerm(ValueModel):
    """A structured latent vector added to the location predictor.

    Parameters
    ----------
    name : str
        Site prefix: ``car`` or ``err``.
    kind : {"car", "residual"}
        brms CAR effect or brms latent residuals.
    car_type : CarType or None
        brms CAR type.
    group : str or None
        Location column, or None when each row is its own location.
    graph : CarGraph or None
        Prepared adjacency over fitted locations.
    layout : ResidualLayout or None
        Latent residuals only: the series layout of the rows.
    coefficients : tuple[tuple[str, str, float, float], ...]
        Latent residuals only: ``(site, class, lower, upper)`` of each
        structural coefficient, in brms's names.
    codes : tuple[int, ...]
        Coordinate of each fitted row.
    time_levels : tuple[str, ...]
        Latent unstructured residuals only: training time levels.
    label : str
        Authored term.
    """

    name: str
    kind: Literal["car", "residual"]
    car_type: CarType | None = None
    group: str | None = None
    graph: CarGraph | None = dx.field(default=None, opaque=True, extras=OPAQUE_BY_VALUE)
    layout: ResidualLayout | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    coefficients: tuple[tuple[str, str, float, float], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    codes: tuple[int, ...] = ()
    time_levels: tuple[str, ...] = ()
    label: str = ""

    @property
    def size(self) -> int:
        """Count the latent coordinates.

        Returns
        -------
        int
            Locations for CAR, rows for latent residuals.
        """
        if self.graph is not None:
            return len(self.graph.locations)
        assert self.layout is not None
        return self.layout.rows

    @property
    def axis(self) -> str:
        """Name the finite location axis.

        Returns
        -------
        str
            QVR object name.
        """
        return f"Formula{self.name.capitalize()}Coordinate"

    @property
    def catalog(self) -> FamilyCatalog:
        """Build the compilation-local law catalog.

        Returns
        -------
        FamilyCatalog
            The CAR laws over this term's graph, or the latent residual law.
        """
        if self.graph is not None:
            return car_catalog(self.graph)
        assert self.layout is not None
        return residual_latent_catalog(
            self.layout, tuple(site for site, *_ in self.coefficients)
        )

    def sites(self, scale_prior: str, correlation_prior: str) -> tuple[SiteDecl, ...]:
        """Declare the field's hyperparameters and coordinates.

        Parameters
        ----------
        scale_prior : str
            Default prior of the scale.
        correlation_prior : str
            Default prior of an unstructured correlation.

        Returns
        -------
        tuple[SiteDecl, ...]
            brms classes ``sdcar``, ``car``, and ``rhocar``, or ``sderr``, the
            structural coefficients and the correlation factor of latent
            residuals, each followed by the coordinates they govern.
        """

        def hyper(
            name: str,
            class_: str,
            default: str,
            lower: float | None = None,
            upper: float | None = None,
            event_axis: str | None = None,
        ) -> SiteDecl:
            return SiteDecl(
                name=name,
                class_=class_,
                default=default,
                lower=lower,
                upper=upper,
                event_axis=event_axis,
            )

        def coordinates(name: str, *, event: bool = True) -> SiteDecl:
            return SiteDecl(
                name=name,
                class_="field",
                axes=() if event else (self.axis,),
                event_axis=self.axis if event else None,
                overridable=False,
            )

        name = self.name
        scale = hyper(f"sd{name}", f"sd{name}", scale_prior, 0.0)
        if self.layout is not None:
            unstructured = self.layout.kind == "unstr" and self.layout.times > 1
            return (
                scale,
                *(
                    hyper(site, class_, f"Uniform({lower!r}, {upper!r})", lower, upper)
                    for site, class_, lower, upper in self.coefficients
                ),
                *(
                    (
                        hyper(
                            "Lcortime",
                            "Lcortime",
                            correlation_prior,
                            event_axis="FormulaAutocorTime",
                        ),
                    )
                    if unstructured
                    else ()
                ),
                coordinates(name),
            )
        if self.car_type == "escar":
            return (
                scale,
                hyper(name, name, "Uniform(0.0, 1.0)", 0.0, 1.0),
                coordinates(f"r{name}"),
            )
        if self.car_type == "esicar":
            return (scale, coordinates(f"r{name}"))
        if self.car_type == "icar":
            return (scale, coordinates(f"z{name}"))
        return (
            scale,
            coordinates(f"z{name}"),
            hyper(f"rho{name}", f"rho{name}", "Beta(1.0, 1.0)", 0.0, 1.0),
            coordinates(f"nsz{name}", event=False),
        )

    def observations(
        self, codes: tuple[int, ...], *, include: bool = True
    ) -> dict[str, torch.Tensor]:
        """Bind the row-to-coordinate index and the data inputs of the term.

        Parameters
        ----------
        codes : tuple[int, ...]
            Coordinate of each row.
        include : bool
            Add the term to the predictor; False is brms's
            ``incl_autocor = FALSE``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{name}_idx``, ``{name}_anchor``, ``{name}_include``, and for BYM2
            ``{name}_inverse_scale``.
        """
        values = {
            f"{self.name}_idx": torch.as_tensor(codes, dtype=torch.long),
            f"{self.name}_anchor": torch.zeros(self.size),
            f"{self.name}_include": torch.tensor(1.0 if include else 0.0),
        }
        if self.car_type == "bym2" and self.graph is not None:
            values[f"{self.name}_inverse_scale"] = torch.as_tensor(
                np.array(self.graph.inverse_scale), dtype=torch.float32
            )
        return values


def latent_car_term(
    term_label: str,
    car_type: CarType,
    group: str | None,
    matrix: np.ndarray | pd.DataFrame,
    frame: pd.DataFrame,
) -> LatentTerm:
    """Bind a brms ``car`` term to its locations and adjacency.

    With ``gr``, the fitted locations are the sorted levels of that column and
    the matrix rows must be labeled by them (a labeled DataFrame); brms takes
    ``M[locations, locations]``. Without ``gr``, every row is a location and
    the matrix covers the rows in order.

    Parameters
    ----------
    term_label : str
        Authored call.
    car_type : CarType
        brms CAR type.
    group : str or None
        Location column.
    matrix : np.ndarray or pandas.DataFrame
        Adjacency matrix.
    frame : pandas.DataFrame
        Included training rows.

    Returns
    -------
    LatentTerm
        Retained term.

    Raises
    ------
    ValueError
        If labels are missing or do not cover the locations, or the matrix is
        invalid.
    """
    if group is None:
        locations = tuple(str(i) for i in range(len(frame)))
        values = np.asarray(
            matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else matrix,
            dtype=np.float64,
        )
        codes = tuple(range(len(frame)))
    else:
        if group not in frame.columns:
            raise ValueError(f"{term_label}: location column {group!r} is absent")
        if not isinstance(matrix, pd.DataFrame):
            raise ValueError("Row names are required for 'M' in CAR terms.")
        column = frame[group]
        if bool(column.isna().to_numpy().any()):
            raise ValueError(
                f"{term_label}: location column {group!r} has missing values"
            )
        locations = tuple(str(value) for value in sorted(column.unique().tolist()))
        labels = [str(label) for label in matrix.index]
        if not set(locations) <= set(labels):
            raise ValueError(
                "Row names of 'M' for CAR terms do not match the names of the grouping levels."
            )
        if [str(label) for label in matrix.columns] != labels:
            raise ValueError(
                "'M' for CAR terms must have matching row and column labels."
            )
        position = {label: index for index, label in enumerate(labels)}
        order = [position[location] for location in locations]
        values = matrix.to_numpy(dtype=np.float64)[np.ix_(order, order)]
        index = {location: code for code, location in enumerate(locations)}
        codes = tuple(index[str(value)] for value in column.tolist())
    return LatentTerm(
        name="car",
        kind="car",
        car_type=car_type,
        group=group,
        graph=car_graph(values, locations, car_type),
        codes=codes,
        label=term_label,
    )


def latent_prediction_codes(term: LatentTerm, frame: pd.DataFrame) -> tuple[int, ...]:
    """Map prediction rows to fitted locations, as brms requires.

    Parameters
    ----------
    term : LatentTerm
        Retained term.
    frame : pandas.DataFrame
        Prediction rows.

    Returns
    -------
    tuple[int, ...]
        Location coordinate of each row.

    Raises
    ------
    ValueError
        Without a grouping factor, or for locations that were not fitted.
    """
    if term.group is None:
        raise ValueError("Without a grouping factor, CAR models cannot handle newdata.")
    if term.group not in frame.columns:
        raise ValueError(f"new_data requires the CAR location column {term.group!r}")
    assert term.graph is not None
    index = {location: code for code, location in enumerate(term.graph.locations)}
    labels = [str(value) for value in frame[term.group].tolist()]
    if any(label not in index for label in labels):
        raise ValueError("Cannot handle new locations in CAR models.")
    return tuple(index[label] for label in labels)


def _named_draw(
    name: str,
    family: str,
    arguments: Mapping[str, str | float],
    *,
    index: str | None = None,
) -> SampleStep:
    """Build one named-argument latent draw.

    Parameters
    ----------
    name : str
        Latent site.
    family : str
        Distribution constructor.
    arguments : Mapping[str, str or float]
        Parameter bindings by name.
    index : str or None
        Optional plate axis.

    Returns
    -------
    SampleStep
        The draw.
    """
    return SampleStep(
        vars=(name,),
        morphism=family,
        args=tuple(
            DrawArgNamed(parameter=key, value=atom_to_draw_arg(value))
            for key, value in arguments.items()
        ),
        index=None if index is None else TypeName(name=index),
    )


class FieldTerm(ValueModel):
    """A structured field indexed by a coordinate of each row.

    Parameters
    ----------
    term : LatentTerm
        Retained term.
    scale_prior : str
        Default prior of the field's scale.
    correlation_prior : str
        Default prior of an unstructured correlation.
    include : bool
        Add the field to the predictor; False is brms's
        ``incl_autocor = FALSE``.
    output : OutputAxis or None
        None; a field has one scalar value per coordinate.
    """

    term: LatentTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    scale_prior: str
    correlation_prior: str
    include: bool = True
    output: OutputAxis | None = None

    @property
    def label(self) -> str:
        """Return the authored term.

        Returns
        -------
        str
            Label of the term in the formula.
        """
        return self.term.label

    def contributions(self) -> tuple[Contribution, ...]:
        """List the bound summands of the term.

        Returns
        -------
        tuple[Contribution, ...]
            The row contribution.
        """
        return (
            Contribution(
                name=f"{self.term.name}_per_row",
                rows=True,
                coordinates=self.output is not None,
            ),
        )

    def axes(self) -> tuple[AxisDecl, ...]:
        """List the finite axes the term uses.

        Returns
        -------
        tuple[AxisDecl, ...]
            The coordinate axis and, for unstructured residuals, the time
            axis of their correlation.
        """
        term = self.term
        axes = [AxisDecl(name=term.axis, size=term.size)]
        if term.layout is not None and term.layout.kind == "unstr":
            axes.append(
                AxisDecl(
                    name="FormulaAutocorTime",
                    size=term.layout.times,
                )
            )
        return tuple(axes)

    def inputs(self) -> tuple[InputDecl, ...]:
        """List the data inputs the term reads.

        Returns
        -------
        tuple[InputDecl, ...]
            The row index into the coordinates, the zero anchor fixing the
            event extent, the inclusion flag, and for BYM2 the per-location
            inverse scaling factor.
        """
        term = self.term
        axes: dict[str, tuple[Literal["Real", "Int"], tuple[str, ...]]] = {
            f"{term.name}_idx": ("Int", ("Resp",)),
            f"{term.name}_include": ("Real", ()),
        }
        return tuple(
            InputDecl(
                name=name,
                element=axes.get(name, ("Real", ()))[0],
                axes=axes.get(name, ("Real", (term.axis,)))[1],
                declared=True,
                values=value,
            )
            for name, value in term.observations(
                term.codes, include=self.include
            ).items()
        )

    def sites(self) -> tuple[SiteDecl, ...]:
        """List the latent sites the term owns.

        Returns
        -------
        tuple[SiteDecl, ...]
            The scale, the structural parameters, and the coordinates.
        """
        return self.term.sites(self.scale_prior, self.correlation_prior)

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            The laws of the field's coordinates.
        """
        return self.term.catalog

    def steps(self, emission: Emission) -> tuple[ProgramStep, ...]:
        """Draw the field and bind its contribution.

        The emitted parameterization follows brms's Stan code: ``escar`` and
        ``esicar`` draw ``rcar`` directly with precision ``1 / sdcar^2``;
        ``icar`` draws unit-precision ``zcar`` and sets
        ``rcar = sdcar * zcar``; ``bym2`` sets
        ``rcar = (sqrt(1 - rhocar) nszcar + sqrt(rhocar / s) zcar) sdcar``.
        Latent residuals draw ``err`` from their structured law with scale
        ``sderr``. The contribution is multiplied by the data scalar
        ``{name}_include``, zero for brms's ``incl_autocor = FALSE``.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[ProgramStep, ...]
            Hyperparameters, coordinates, and contribution.
        """
        name = self.term.name
        effect, steps = (
            self._residual_steps(emission, self.axes()[1:])
            if self.term.layout is not None
            else self._spatial_steps(emission)
        )
        contribution = _mul(
            _var(f"{name}_include"),
            LetExprIndex(array=_var(effect), indices=(_var(f"{name}_idx"),)),
        )
        return (*steps, _let(f"{name}_per_row", contribution))

    def _residual_steps(
        self, emission: Emission, axes: tuple[AxisDecl, ...]
    ) -> tuple[str, tuple[ProgramStep, ...]]:
        """Draw latent residuals from their structured law.

        Parameters
        ----------
        emission : Emission
            Program being emitted.
        axes : tuple[AxisDecl, ...]
            The time axis of an unstructured correlation.

        Returns
        -------
        tuple[str, tuple[ProgramStep, ...]]
            The residual binding and its steps.
        """
        name = self.term.name
        declared = {site.name: site for site in self.sites()}
        scale = f"sd{name}"
        coefficients = tuple(site for site, *_ in self.term.coefficients)
        steps: list[ProgramStep] = [
            emission.draw(declared[site]) for site in (scale, *coefficients)
        ]
        arguments: dict[str, str | float] = {
            "scale": scale,
            **{site: site for site in coefficients},
        }
        for axis in axes:
            factor = "formula_cortime_factor"
            steps.extend(emission.correlation("Lcortime", axis, factor))
            arguments["factor"] = factor
        arguments["anchor"] = f"{name}_anchor"
        steps.append(_named_draw(name, "LatentResidual", arguments))
        return name, tuple(steps)

    def _spatial_steps(self, emission: Emission) -> tuple[str, tuple[ProgramStep, ...]]:
        """Draw a conditional autoregressive field over its locations.

        Parameters
        ----------
        emission : Emission
            Program being emitted.

        Returns
        -------
        tuple[str, tuple[ProgramStep, ...]]
            The binding of the location effects and its steps.
        """
        term, name = self.term, self.term.name
        declared = {site.name: site for site in self.sites()}
        scale, effect, anchor = f"sd{name}", f"r{name}", f"{name}_anchor"
        steps: list[ProgramStep] = [emission.draw(declared[scale])]
        if term.car_type == "escar":
            steps.append(emission.draw(declared[name]))
            steps.append(
                _named_draw(
                    effect,
                    "SparseCAR",
                    {"scale": scale, "alpha": name, "anchor": anchor},
                )
            )
            return effect, tuple(steps)
        if term.car_type == "esicar":
            steps.append(
                _named_draw(effect, "SparseICAR", {"scale": scale, "anchor": anchor})
            )
            return effect, tuple(steps)
        standard = f"z{name}"
        steps.append(
            _named_draw(standard, "SparseICAR", {"scale": 1.0, "anchor": anchor})
        )
        if term.car_type == "icar":
            steps.append(_let(effect, _mul(_var(standard), _var(scale))))
            return effect, tuple(steps)
        steps.append(emission.draw(declared[f"rho{name}"]))
        steps.append(
            _named_draw(
                f"nsz{name}", "Normal", {"loc": 0.0, "scale": 1.0}, index=term.axis
            )
        )
        rho = _var(f"rho{name}")
        spatial = _mul(
            _call("sqrt", _mul(rho, _var(f"{name}_inverse_scale"))), _var(standard)
        )
        unstructured = _mul(
            _call("sqrt", _sub(LetExprLiteral(value=1.0), rho)), _var(f"nsz{name}")
        )
        steps.append(_let(effect, _mul(_add(unstructured, spatial), _var(scale))))
        return effect, tuple(steps)


def field_terms(
    formula: Formula,
    *,
    scale_prior: str = "",
    correlation_prior: str = "",
    include: bool = True,
) -> tuple[FieldTerm, ...]:
    """Build the structured fields of one predictor.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    scale_prior, correlation_prior : str
        Default priors; empty when only the inputs of the terms are read.
    include : bool
        Add the fields to the predictor.

    Returns
    -------
    tuple[FieldTerm, ...]
        One term per field.
    """
    return tuple(
        FieldTerm(
            term=term,
            scale_prior=scale_prior,
            correlation_prior=correlation_prior,
            include=include,
        )
        for term in formula.latent_terms
    )
