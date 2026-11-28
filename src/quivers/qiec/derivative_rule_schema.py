"""Portable names of authored QIEC derivative actions."""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import ComputationId


class RationalDerivativeDeclaration(QiecModel):
    """Name the authored directional actions of a checked scalar computation.

    Parameters
    ----------
    primal : ComputationId
        The computation whose derivative is declared.
    jvp : ComputationId or None
        A forward computation taking the primal coordinates and their tangents.
    vjp : tuple[ComputationId, ...] or None
        A scalar reverse action for each primal coordinate, in input order.
        They each take the primal coordinates followed by the output seed.
    """

    primal: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    jvp: ComputationId | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    vjp: tuple[ComputationId, ...] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require at least one declared action.

        Raises
        ------
        ValueError
            If the declaration supplies neither action.
        """
        self._validate_recursive_fields()
        if self.jvp is None and self.vjp is None:
            raise ValueError("a rational derivative declaration needs a JVP or VJP")


__all__ = ["RationalDerivativeDeclaration"]
