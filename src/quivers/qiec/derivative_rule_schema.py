"""Portable names of authored QIEC derivative actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import ComputationId

if TYPE_CHECKING:
    from quivers.qiec.derivative_rules import RationalDerivativeDecision
    from quivers.qiec.module import QiecModule


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

    def check(
        self, module: QiecModule, *, max_work: int = 65_536, fuel: int = 65_536
    ) -> RationalDerivativeDecision:
        """Resolve and certify these actions against the supplied checked source.

        Parameters
        ----------
        module : QiecModule
            The module that defines the named computations.
        max_work : int
            The rational-arithmetic and counterexample budget.
        fuel : int
            Reference machine steps allowed for each computation.

        Returns
        -------
        RationalDerivativeDecision
            The exact decidable result with domain and any counterexample.

        Raises
        ------
        RationalLawUnsupported
            If a named declaration is absent or an action is unsupported.
        RationalLawLimit
            If certification cannot complete under the supplied limits.
        """
        from quivers.qiec.derivative_rules import check_rational_derivative_rule
        from quivers.qiec.rational_laws import RationalLawUnsupported

        names = {
            computation.id: computation.name for computation in module.computations
        }
        try:
            primal = names[self.primal]
            jvp = None if self.jvp is None else names[self.jvp]
            vjp = None if self.vjp is None else tuple(names[rule] for rule in self.vjp)
        except KeyError as error:
            raise RationalLawUnsupported(
                "a rational derivative declaration names an unknown computation"
            ) from error
        return check_rational_derivative_rule(
            module, primal, jvp=jvp, vjp=vjp, max_work=max_work, fuel=fuel
        )


__all__ = ["RationalDerivativeDeclaration"]
