"""Residual correlation across the responses of a joint model.

Under ``set_rescor(TRUE)`` the scalar residuals of several responses of one row
form one vector with an elliptical law: a multivariate Normal, or a multivariate
Student-t with one shared degrees-of-freedom parameter. An elliptical family is
closed under taking the law of some coordinates, so a reader of one response
(its pointwise score, its conditional moments) reads the marginal of the
coordinates it selects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import didactic.api as dx
import torch
from torch.distributions import Distribution, MultivariateNormal, constraints
from torch.types import _size

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.continuous._multivariate_student_t import MultivariateStudentT
from quivers.formulas._residual_dependence import ResidualLayout, ResidualResponse
from quivers.qiec.families import DistributionFamily, FamilyParameter
from quivers.qiec.types import REAL

type JointLaw = MultivariateNormal | MultivariateStudentT
"""The law of the residual vector of one row."""


class RecursiveResiduals(Distribution):
    """Correlated residuals of responses whose locations follow their own past.

    Row ``t`` of response ``k`` is ``y_tk = m_tk + e_tk``, where ``m_tk`` is
    the location plus the conditional ARMA recursion of response ``k`` over
    its own recorded responses, and the vector ``e_t`` of row ``t`` has an
    elliptical law with scale factor ``L_t``. The map from ``y`` to ``e`` is
    lower triangular with unit diagonal in each response's series order,
    so its Jacobian is one and the density is ``prod_t f(e_t; L_t)``: the
    recursion moves the locations and leaves the law of each row's vector
    as it is. A response without a recursion has ``m_tk`` its location.

    Parameters
    ----------
    loc : torch.Tensor
        Locations before the recursions, ``(..., rows, responses)``.
    scale_tril : torch.Tensor
        Lower scale factor of each row, ``(..., rows, responses, responses)``.
    coordinates : Sequence[ResidualResponse or None]
        Each response's residual law, read for its recursion and recorded
        responses, or None for a response without one.
    df : torch.Tensor or None
        Shared Student degrees of freedom, or None for a Normal vector.
    completion : torch.Tensor or None
        Responses of every row with the missing ones completed,
        ``(..., rows, responses)``; None scores the value given. With a
        completion the law is the joint law of the recorded and the missing
        responses, written over the recorded rows: the recursions read the
        completed responses of earlier rows, and each row's factor is the
        density of its completed vector.
    """

    has_rsample = False

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Declare no argument constraints; the factors are checked when built.

        Returns
        -------
        dict[str, constraints.Constraint]
            An empty mapping.
        """
        return {}

    @property
    def support(self) -> constraints.Constraint:
        """Give the support: real vectors.

        Returns
        -------
        constraints.Constraint
            Real values with one event axis.
        """
        return constraints.independent(constraints.real, 1)

    def __init__(
        self,
        loc: torch.Tensor,
        scale_tril: torch.Tensor,
        coordinates: Sequence[ResidualResponse | None],
        df: torch.Tensor | None = None,
        completion: torch.Tensor | None = None,
    ) -> None:
        if loc.ndim < 2 or loc.shape[-1] != len(coordinates):
            raise ValueError("one location per row and response is required")
        self.loc = loc
        self.scale_tril = scale_tril.expand(*loc.shape, loc.shape[-1])
        self.df = df
        self.completion = completion
        self.coordinates = tuple(coordinates)
        super().__init__(loc.shape[:-1], loc.shape[-1:], validate_args=False)

    def _row_law(self, means: torch.Tensor) -> JointLaw:
        """Return the elliptical law of each row's vector at given means.

        Parameters
        ----------
        means : torch.Tensor
            Conditional means, ``(..., rows, responses)``.

        Returns
        -------
        JointLaw
            Normal or Student vectors with the rows' scale factors.
        """
        if self.df is None:
            return MultivariateNormal(means, scale_tril=self.scale_tril)
        return MultivariateStudentT(
            self.df.expand(means.shape[:-1]), means, scale_tril=self.scale_tril
        )

    def conditional_means(
        self, values: torch.Tensor, innovations: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Give each response's location plus its recursion over ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Responses, ``(..., rows, responses)``, NaN where missing.
        innovations : torch.Tensor or None
            Innovations completing missing responses.

        Returns
        -------
        torch.Tensor
            Conditional means of the rows.
        """
        columns = []
        for index, law in enumerate(self.coordinates):
            if law is None:
                columns.append(self.loc[..., index])
                continue
            columns.append(
                law.conditional_means(
                    values[..., index],
                    None if innovations is None else innovations[..., index],
                )
            )
        return torch.stack(columns, -1)

    def history(self) -> torch.Tensor:
        """Return the recorded responses of the rows, NaN where absent.

        Returns
        -------
        torch.Tensor
            Recorded responses, ``(rows, responses)``; a response without a
            recursion has none.
        """
        columns = [
            torch.full(self.loc.shape[-2:-1], torch.nan, dtype=self.loc.dtype)
            if law is None
            else law.history.reshape(-1)
            for law in self.coordinates
        ]
        return torch.stack(columns, -1).to(self.loc.device)

    def rows(self) -> JointLaw:
        """Return the law of each row's vector given the recorded responses.

        Missing recorded responses are completed by one joint draw.

        Returns
        -------
        JointLaw
            Normal or Student vectors at the conditional means.
        """
        return self._row_law(
            self.conditional_means(self.history(), self._innovations(()))
        )

    def _innovations(self, sample_shape: _size) -> torch.Tensor:
        """Draw innovation vectors of every row.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Innovations of shape ``(*sample_shape, *batch, responses)``.
        """
        return self._row_law(torch.zeros_like(self.loc)).sample(
            torch.Size(sample_shape)
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score each row's factor of the likelihood.

        Parameters
        ----------
        value : torch.Tensor
            Responses, ``(..., rows, responses)``; with a completion, only
            their shape is read.

        Returns
        -------
        torch.Tensor
            Log density of each row's innovation vector.
        """
        values = value if self.completion is None else self.completion.expand_as(value)
        return self._row_law(self.conditional_means(values)).log_prob(values)

    @property
    def mean(self) -> torch.Tensor:
        """Give the conditional mean of each row given the recorded responses.

        Returns
        -------
        torch.Tensor
            Means, ``(..., rows, responses)``.
        """
        return self.rows().mean

    @property
    def variance(self) -> torch.Tensor:
        """Give each coordinate's conditional variance.

        Returns
        -------
        torch.Tensor
            Variances, ``(..., rows, responses)``.
        """
        return self._row_law(self.loc).variance

    def sample(self, sample_shape: _size = torch.Size()) -> torch.Tensor:
        """Draw responses from the joint law given the recorded responses.

        Innovation vectors are drawn jointly for every row; each response is
        then its conditional mean over its recorded responses plus its
        innovation, and a missing recorded response is completed by the
        same innovation, so the draw follows the joint law exactly.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample shape.

        Returns
        -------
        torch.Tensor
            Draws, ``(*sample_shape, *batch, responses)``.
        """
        shape = torch.Size(sample_shape)
        history = self.history()
        draws = []
        for innovation in self._innovations(shape).reshape(-1, *self.loc.shape):
            draws.append(self.conditional_means(history, innovation) + innovation)
        return torch.stack(draws).reshape(*shape, *self.loc.shape)


class RecursiveResidualFactory(ValueModel):
    """Build the law of correlated residuals with recursions.

    Parameters
    ----------
    layouts : tuple[ResidualLayout or None, ...]
        Row coding of each response's residual law, or None for a response
        without one.
    coefficients : tuple[tuple[tuple[str, str], ...], ...]
        For each response, its structural parameters as ``(kernel
        parameter, name in the response's law)`` pairs.
    student : bool
        Student vectors with shared degrees of freedom.
    """

    layouts: tuple[ResidualLayout | None, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_VALUE
    )
    coefficients: tuple[tuple[tuple[str, str], ...], ...] = dx.field(
        default=(), opaque=True, extras=OPAQUE_BY_VALUE
    )
    student: bool = False

    def __call__(self, parameters: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct the law at natural-scale parameter values.

        Parameters
        ----------
        parameters : Mapping[str, torch.Tensor]
            ``loc``, ``scale_tril``, optional ``df`` and ``completion``, and
            each response's structural parameters.

        Returns
        -------
        Distribution
            A `RecursiveResiduals` law.
        """
        loc, factor = parameters["loc"], parameters["scale_tril"]
        df = parameters.get("df")
        scale = factor.square().sum(-1).sqrt().expand_as(loc)
        coordinates = tuple(
            None
            if layout is None
            else ResidualResponse(
                layout,
                loc[..., index],
                scale[..., index],
                {local: parameters[name] for name, local in names},
                df,
            )
            for index, (layout, names) in enumerate(
                zip(self.layouts, self.coefficients, strict=True)
            )
        )
        return RecursiveResiduals(
            loc, factor, coordinates, df, parameters.get("completion")
        )


def recursive_residual_record(
    name: str, coefficients: Sequence[str], *, student: bool, completion: bool = False
) -> DistributionFamily:
    """Declare the checked signature of correlated residuals with recursions.

    Parameters
    ----------
    name : str
        Catalog family name.
    coefficients : Sequence[str]
        Kernel names of the responses' structural parameters.
    student : bool
        Include the shared Student degrees of freedom.
    completion : bool
        Include the completed responses of every row, so missing responses
        are scored as coordinates of the law.

    Returns
    -------
    DistributionFamily
        A vector-event family over the responses, one batch row per row.
    """
    parameters = [
        *(
            (FamilyParameter(name="df", constraint="positive", rank=0),)
            if student
            else ()
        ),
        FamilyParameter(name="loc", constraint="real", rank=1, event_dims=(0,)),
        FamilyParameter(
            name="scale_tril", constraint="lower_cholesky", rank=2, event_dims=(0, 0)
        ),
        *(
            FamilyParameter(name=coefficient, constraint="real", rank=0)
            for coefficient in coefficients
        ),
        *(
            (
                FamilyParameter(
                    name="completion", constraint="real", rank=1, event_dims=(0,)
                ),
            )
            if completion
            else ()
        ),
    ]
    return DistributionFamily(
        name=name,
        parameters=tuple(parameters),
        support="real",
        element=REAL,
        event_rank=1,
        event_sources=("loc",),
        discrete=False,
        reparameterizable=False,
        compositional=False,
    )


def marginal(law: torch.distributions.Distribution, keep: Sequence[int]) -> JointLaw:
    """Return the law of some coordinates of an elliptical residual vector.

    The coordinates ``keep`` of a Normal vector with location ``m`` and scale
    matrix ``S`` are Normal with location ``m[keep]`` and scale matrix
    ``S[keep, keep]``; a Student-t vector keeps its degrees of freedom as
    well, since it is a Normal vector divided by one shared chi variable.

    Parameters
    ----------
    law : torch.distributions.Distribution
        Law of the whole residual vector, with event axis over responses; a
        `RecursiveResiduals` law is read at its recorded responses.
    keep : Sequence[int]
        Coordinates to keep, in the order of the result.

    Returns
    -------
    JointLaw
        The law of the kept coordinates.

    Raises
    ------
    TypeError
        If ``law`` is not a multivariate Normal or Student-t.
    """
    if isinstance(law, RecursiveResiduals):
        # Each response's recursion reads only its own recorded responses,
        # so the kept coordinates are elliptical at their conditional means.
        law = law.rows()
    if not isinstance(law, MultivariateNormal | MultivariateStudentT):
        raise TypeError("a residual-correlation event must be elliptical")
    selector = torch.tensor(list(keep), device=law.loc.device)
    loc = law.loc.index_select(-1, selector)
    factor = law.scale_tril
    shape = (
        (factor @ factor.transpose(-1, -2))
        .index_select(-1, selector)
        .index_select(-2, selector)
    )
    if isinstance(law, MultivariateStudentT):
        return MultivariateStudentT(
            law.df, loc, scale_tril=torch.linalg.cholesky(shape)
        )
    return MultivariateNormal(loc, scale_tril=torch.linalg.cholesky(shape))


def coordinate(
    law: torch.distributions.Distribution, index: int
) -> torch.distributions.Normal | torch.distributions.StudentT:
    """Return the scalar law of one coordinate of an elliptical residual vector.

    Parameters
    ----------
    law : torch.distributions.Distribution
        Law of the whole residual vector.
    index : int
        Coordinate.

    Returns
    -------
    torch.distributions.Normal or torch.distributions.StudentT
        The marginal law of the coordinate, with the vector's degrees of
        freedom for a Student-t vector.
    """
    kept = marginal(law, (index,))
    loc = kept.loc[..., 0]
    scale = kept.scale_tril[..., 0, 0]
    if isinstance(kept, MultivariateStudentT):
        return torch.distributions.StudentT(kept.df, loc, scale)
    return torch.distributions.Normal(loc, scale)


__all__: list[str] = []
