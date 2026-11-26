"""Residual correlation across the responses of a joint model.

Under ``set_rescor(TRUE)`` the scalar residuals of several responses of one row
form one vector with an elliptical law: a multivariate Normal, or a multivariate
Student-t with one shared degrees-of-freedom parameter. An elliptical family is
closed under taking the law of some coordinates, so a reader of one response
(its pointwise score, its conditional moments) reads the marginal of the
coordinates it selects.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.distributions import MultivariateNormal

from quivers.continuous._multivariate_student_t import MultivariateStudentT

type JointLaw = MultivariateNormal | MultivariateStudentT
"""The law of the residual vector of one row."""


def marginal(law: torch.distributions.Distribution, keep: Sequence[int]) -> JointLaw:
    """Return the law of some coordinates of an elliptical residual vector.

    The coordinates ``keep`` of a Normal vector with location ``m`` and scale
    matrix ``S`` are Normal with location ``m[keep]`` and scale matrix
    ``S[keep, keep]``; a Student-t vector keeps its degrees of freedom as
    well, since it is a Normal vector divided by one shared chi variable.

    Parameters
    ----------
    law : torch.distributions.Distribution
        Law of the whole residual vector, with event axis over responses.
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
