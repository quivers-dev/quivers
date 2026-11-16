"""Immutable response-indexed specifications for one joint formula program."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas.family import Family, families
from quivers.formulas.formula import _qvr_name


class ResponseModel(ValueModel):
    """Specify one response using the ordinary formula preparation contract.

    Parameters
    ----------
    formula : str
        Authored response formula, including ordinary response modifiers.
    family : str or Family
        Registered family or a configured family value.
    distributional, nonlinear : dict or None
        Existing distributional predictors and nonlinear dependency graph.
    fixed_parameters : dict or None
        Auxiliary parameters fixed on their natural scale.
    weights, decision : str or None
        Ordinary likelihood-weight and decision column shortcuts.
    binomial_trials : int or str or None
        Finite count total or trial column, following the ordinary family API.
    thresholds_by : str or None
        Retained grouping factor for ordinal thresholds.
    mixture_components : int
        Number of components for a finite mixture response.

    Notes
    -----
    Data, priors, inference options, and group covariance declarations belong
    to the joint fit. Each response retains its own component and category
    axes. A response label does not replace either coordinate system.
    """

    formula: str
    family: str | Family = dx.field(
        default="gaussian", opaque=True, extras=OPAQUE_BY_VALUE
    )
    distributional: dict[str, str | NonlinearExpression] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    nonlinear: dict[str, str | NonlinearExpression] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    fixed_parameters: dict[str, float] | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    weights: str | None = None
    decision: str | None = None
    binomial_trials: int | str | None = None
    thresholds_by: str | None = None
    mixture_components: int = 2

    @dx.model_validator()
    def _validate(self) -> None:
        """Check the formula, family, column shortcuts, and predictor maps.

        Raises
        ------
        ValueError
            If the formula lacks a response, the family name is unknown, or a
            column shortcut is not a nonempty column name.
        TypeError
            If the family is neither a name nor a Family, or a predictor map
            does not map parameter names to formulas or expressions.
        """
        if not self.formula.strip() or "~" not in self.formula:
            raise ValueError("a response model requires a response formula with '~'")
        if isinstance(self.family, str) and self.family not in families:
            raise ValueError(f"unknown response family {self.family!r}")
        if not isinstance(self.family, (str, Family)):
            raise TypeError("a response family must be a registered name or Family")
        for name in ("weights", "decision", "thresholds_by"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must name a data column")
        for name in ("distributional", "nonlinear"):
            values = getattr(self, name)
            if values is not None and not all(
                isinstance(key, str)
                and key
                and isinstance(value, (str, NonlinearExpression))
                for key, value in values.items()
            ):
                raise TypeError(
                    f"{name} must map parameter names to formulas or expressions"
                )

    @property
    def resolved_family(self) -> Family:
        """Resolve a family name without changing its configured links.

        Returns
        -------
        Family
            Existing family metadata used by ordinary preparation.
        """
        return families[self.family] if isinstance(self.family, str) else self.family


class MultivariateFormula(ValueModel):
    """Combine named response graphs into one probabilistic program.

    Parameters
    ----------
    responses : dict[str, ResponseModel]
        At least two explicitly named responses, in authored coordinate order.
        Names select priors and result views. Formula left-hand sides identify
        their observed data columns.
    rescor : bool
        Estimate residual correlation between aligned scalar Gaussian or
        Student responses. False permits heterogeneous response families.
        Shared group-effect identifiers apply in either mode.
    response_names : tuple[str, ...] or None
        Explicit response coordinate order. None retains mapping insertion
        order. The order is part of equality, hashing, and serialization.
    """

    responses: dict[str, ResponseModel] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rescor: bool = False
    response_names: tuple[str, ...]

    def __init__(
        self,
        *,
        responses: Mapping[str, ResponseModel],
        rescor: bool = False,
        response_names: tuple[str, ...] | None = None,
    ) -> None:
        """Retain the responses with an explicit coordinate order.

        Parameters
        ----------
        responses : Mapping[str, ResponseModel]
            Named responses, copied into a plain dictionary.
        rescor : bool
            Estimate residual correlation between aligned responses.
        response_names : tuple[str, ...] or None
            Explicit response order, or None for the mapping's insertion order.
        """
        construct = cast(Callable[..., None], super().__init__)
        construct(
            responses=dict(responses),
            rescor=rescor,
            response_names=tuple(responses)
            if response_names is None
            else response_names,
        )

    @dx.model_validator()
    def _validate(self) -> None:
        """Check response labels, coordinate order, and the residual policy.

        Raises
        ------
        ValueError
            If fewer than two responses are given, the order does not list
            each response exactly once, a label is empty or collides after QVR
            name normalization, or residual correlation joins families other
            than one common Gaussian or Student family.
        TypeError
            If ``rescor`` is not Boolean or a response is not a ResponseModel.
        """
        if len(self.responses) < 2:
            raise ValueError("a multivariate formula requires at least two responses")
        if type(self.rescor) is not bool:
            raise TypeError("rescor must be Boolean")
        if len(self.response_names) != len(self.responses) or set(
            self.response_names
        ) != set(self.responses):
            raise ValueError("response_names must list every response exactly once")
        normalized = set()
        for name, response in self.responses.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("response labels must be nonempty text")
            token = _qvr_name(name)
            if token in normalized:
                raise ValueError("response labels collide after QVR name normalization")
            normalized.add(token)
            if not isinstance(response, ResponseModel):
                raise TypeError("responses must map labels to ResponseModel values")
        if self.rescor:
            names = {
                response.resolved_family.name for response in self.responses.values()
            }
            if len(names) != 1 or not names <= {"gaussian", "student_t"}:
                raise ValueError(
                    "rescor requires homogeneous Gaussian or Student responses"
                )


__all__: list[str] = []
