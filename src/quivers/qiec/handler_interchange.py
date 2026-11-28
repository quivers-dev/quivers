"""Check the finite value observation of a declared handler interchange."""

from __future__ import annotations

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.identifiers import ComputationId
from quivers.qiec.laws import FiniteLawDecision, FiniteLawUnsupported, check_finite_law
from quivers.qiec.module import QiecModule
from quivers.qiec.terms import Call, Computation, Handle


class FiniteHandlerInterchange(QiecModel):
    """Declare the two source computations of a finite interchange check.

    Parameters
    ----------
    left, right : ComputationId
        Computations with the same finite inputs that install the same two
        distinct effect instances in opposite orders around one common call.
        Each installation retains its handler and initial arguments.
    """

    left: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    right: ComputationId = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)

    @dx.model_validator()
    def _validate_record(self) -> None:
        """Require two distinct nominal computations.

        Raises
        ------
        ValueError
            If both sides name the same computation.
        """
        self._validate_recursive_fields()
        if self.left == self.right:
            raise ValueError("a handler interchange requires two distinct computations")


def _installations(body: Computation) -> tuple[Handle, Handle, Call]:
    """Read the outer and inner installations and their common named call.

    Parameters
    ----------
    body : Computation
        A checked computation supplied as one side of an interchange.

    Returns
    -------
    tuple[Handle, Handle, Call]
        The outer installation, inner installation, and invoked computation.

    Raises
    ------
    FiniteLawUnsupported
        If the two installations around a call are absent.
    """
    if isinstance(body, Handle) and isinstance(inner := body.computation, Handle):
        if isinstance(call := inner.computation, Call):
            return body, inner, call
    raise FiniteLawUnsupported("an interchange needs two handlers around a named call")


def check_finite_handler_interchange(
    module: QiecModule,
    statement: FiniteHandlerInterchange,
    *,
    max_assignments: int = 65_536,
    fuel: int = 65_536,
) -> FiniteLawDecision:
    """Decide equality of returned values for one finite handler interchange.

    The two sides call the same checked body with identical arguments.
    Both orders must type check, and the existing finite law evaluator
    checks every input assignment or returns a counterexample. The
    result is scoped to these bodies and returned values. It supplies
    no rewrite permission for another body or for trace observations.

    Parameters
    ----------
    module : QiecModule
        The source defining the handlers and the two computations.
    statement : FiniteHandlerInterchange
        The pair of nominal computations to compare.
    max_assignments : int
        The largest finite carrier product to enumerate.
    fuel : int
        The reference machine's fuel per side and assignment.

    Returns
    -------
    FiniteLawDecision
        The complete finite comparison or a separating assignment.

    Raises
    ------
    FiniteLawUnsupported
        If the pair is outside the finite fragment or changes more than
        the order of the two handler installations.
    FiniteLawLimit
        If the resource limits prevent a decision.
    """
    computations = {item.id: item for item in module.computations}
    lhs, rhs = computations.get(statement.left), computations.get(statement.right)
    if lhs is None or rhs is None:
        raise FiniteLawUnsupported("an interchange names an unknown computation")
    if lhs.parameters != rhs.parameters:
        raise FiniteLawUnsupported("interchange sides need identical named inputs")
    outer, inner, left_call = _installations(lhs.body)
    swapped_inner, swapped_outer, right_call = _installations(rhs.body)
    if outer.instance == inner.instance:
        raise FiniteLawUnsupported("interchange requires distinct effect instances")
    if outer != swapped_outer.with_(
        computation=outer.computation
    ) or inner != swapped_inner.with_(computation=inner.computation):
        raise FiniteLawUnsupported("interchange sides change a handler installation")
    if left_call != right_call.with_(origin=left_call.origin):
        raise FiniteLawUnsupported("interchange sides need the same named call")
    return check_finite_law(
        module, lhs.name, rhs.name, max_assignments=max_assignments, fuel=fuel
    )


__all__ = ["FiniteHandlerInterchange", "check_finite_handler_interchange"]
