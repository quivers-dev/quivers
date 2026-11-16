"""Declared runtime design columns share the existing checked source contracts.

A symbolic column supplies an expression rather than numeric placeholder data.
Its parameter record retains the nominal row axis and the canonical Tensor type;
coefficient emission and prior selection retain their ordinary authored labels.
"""

from __future__ import annotations

import math

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import LetExprNode
from quivers.qiec import IndexLiteral, ProgramParameter, REAL, tensor_shape

__all__: list[str] = []


class SymbolicColumn(ValueModel):
    """One scalar-per-row design column supplied by a checked value expression.

    Parameters
    ----------
    term : str
        Authored term used by the ordinary prior catalog and coefficient labels.
    name : str
        Authored scalar design-column label.
    parameter : ProgramParameter
        Existing canonical Tensor[Real] parameter contract with one declared
        nominal row axis. The joint emitter substitutes its value expression
        and removes this local input from the external telescope.
    expression : LetExprNode
        Canonical QVR expression depending on retained latent or runtime bindings.
        Its type is checked in the complete generated program, not guessed from extents.
    category_specific : bool
        Use the existing transition-indexed coefficient layout when requested.
    prior_multiplier : float
        Explicit multiplier for eligible default coefficient prior scales.
        One preserves the authored default; no empirical RMS is computed from
        unavailable latent values. Explicit user priors remain unchanged.
    """

    term: str
    name: str
    parameter: ProgramParameter = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    expression: LetExprNode = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    category_specific: bool = False
    prior_multiplier: float = 1.0

    @dx.model_validator()
    def _validate(self) -> None:
        """Require the canonical scalar-per-row Real contract and a proper scale.

        Raises
        ------
        ValueError
            If names, row geometry, carrier, role, or scale policy are invalid.
        """
        shape = tensor_shape(self.parameter.type)
        if (
            not self.term
            or not self.name
            or not self.parameter.name
            or self.parameter.role != "data"
            or shape is None
            or shape[0] != REAL
            or len(shape[1]) != 1
            or len(self.parameter.axes) != 1
            or not self.parameter.axes[0]
        ):
            raise ValueError(
                "a symbolic design column requires a named Tensor[Real] input with one nominal row axis"
            )
        if (
            isinstance(self.prior_multiplier, bool)
            or not math.isfinite(self.prior_multiplier)
            or self.prior_multiplier <= 0
        ):
            raise ValueError(
                "a symbolic column prior multiplier must be finite and positive"
            )

    @property
    def qvr_name(self) -> str:
        """Read the checked local input name used by every coefficient emitter.

        Returns
        -------
        str
            Canonical parameter name, without an independently generated alias.
        """
        return self.parameter.name

    @property
    def is_intercept(self) -> bool:
        """A latent-dependent runtime column is not the ordinary constant intercept.

        Returns
        -------
        bool
            False; intercept coordinates retain their existing representation.
        """
        return False

    def validate_rows(self, axis: str, size: int) -> None:
        """Check a column against the consuming predictor's declared row scope.

        Parameters
        ----------
        axis : str
            Nominal local row binder retained by the formula compiler.
        size : int
            Checked cardinality of that row binder.

        Raises
        ------
        ValueError
            If the supplied nominal scope or its closed extent differs. Equal
            cardinality alone does not identify two separately named row scopes.
        """
        shape = tensor_shape(self.parameter.type)
        assert shape is not None
        dimension = shape[1][0]
        if (
            self.parameter.axes != (axis,)
            or not isinstance(dimension, IndexLiteral)
            or dimension.value != size
        ):
            raise ValueError(
                "symbolic design column must retain its predictor's nominal row axis and extent"
            )
