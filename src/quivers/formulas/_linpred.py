"""Read fitted linear predictors from native replay and formula metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from quivers.formulas._mixture import mixture_component_count
from quivers.formulas._posterior import LinpredBinding
from quivers.formulas._predictor import Predictor, response_axis
from quivers.formulas._residual_law import linear_predictor
from quivers.formulas.family import LOCATION
from quivers.formulas.formula import Formula, _qvr_name

if TYPE_CHECKING:
    from formulae.matrices import DesignMatrices
    from quivers.formulas._predictor_graph import PredictorGraph
    from quivers.formulas._residual_law import ResidualLaw
    from quivers.formulas.family import Family
    from quivers.inference.trace import Trace


def population_coefficients(
    predictors: tuple[Predictor, ...],
    draws: Mapping[str, torch.Tensor],
    prefix: str,
    namespace: str = "",
) -> dict[str, torch.Tensor]:
    """Read the population coefficients named by fitted terms.

    Parameters
    ----------
    predictors : tuple[Predictor, ...]
        Retained predictors in the response's local names.
    draws : Mapping[str, torch.Tensor]
        Native draws on chain and draw axes.
    prefix : str
        Local prefix of the selected predictor.
    namespace : str
        Prefix applied to the response's bindings in the native draws.

    Returns
    -------
    dict[str, torch.Tensor]
        Coefficients by fitted design-column label.
    """
    return {
        coefficient.label: draws[namespace + coefficient.site]
        for predictor in predictors
        if predictor.prefix == prefix
        for term in predictor.terms
        for coefficient in term.population()
    }


@dataclass(frozen=True, slots=True)
class PredictorReadout:
    """Fitted response metadata used to read and label one predictor."""

    formula: Formula
    family: Family
    designs: Mapping[str, DesignMatrices] | None
    distributional: Mapping[str, Formula]
    predictor_graph: PredictorGraph | None
    predictors: tuple[Predictor, ...]
    fixed_parameters: Mapping[str, float]
    category_labels: tuple[int | str, ...]
    location: str | None
    residual: ResidualLaw | None

    def predictor_labels(self, parameter: str = "mu") -> tuple[int | str, ...]:
        """Label the last axis returned by posterior_linpred for one graph output.

        Parameters
        ----------
        parameter : str
            Family parameter or named nonlinear intermediate.

        Returns
        -------
        tuple[int or str, ...]
            Output coordinate order; empty for scalar-row values. Family
            coordinates use fitted response order, with the ALR reference
            omitted. Intermediate vectors retain authored label order.

        Raises
        ------
        ValueError
            If the parameter has no retained predictor or graph binding.
        """
        from quivers.formulas._predictor_graph import ComponentExpression

        graph = self.predictor_graph
        if graph is not None and parameter in graph.expressions:
            value = graph.expressions[parameter]
            return value.components if isinstance(value, ComponentExpression) else ()
        if graph is not None and parameter in graph.leaves:
            return ()
        response = self.family.response
        # mixtures: mu stacks the component locations mu1, ..., muK.
        if parameter == "mu" and self.family.mixture is not None:
            return tuple(
                f"mu{index}"
                for index in range(1, mixture_component_count(self.family) + 1)
            )
        if parameter == "mu":
            if response is None:
                return ()
            labels = self.category_labels
            if response.kind == "logistic_normal":
                return tuple(
                    labels[index]
                    for index in response.coordinate_indices(
                        tuple(str(label) for label in labels)
                    )
                )
            return labels
        if graph is not None and parameter in graph.roots:
            value = graph.roots[parameter]
            if not isinstance(value, ComponentExpression):
                return ()
            assert response is not None
            labels = self.category_labels
            return tuple(
                labels[index]
                for index in response.coordinate_indices(
                    tuple(str(label) for label in labels)
                )
            )
        if self._constant_parameter(parameter):
            return ()
        self._formula_predictor(parameter)
        return (
            self.coefficient_labels
            if self.family.component_scale is not None
            and parameter == self.family.component_scale.name
            else ()
        )

    def _predictor_width(self, parameter: str) -> int:
        """Read the declared coordinate width of a fitted predictor.

        Parameters
        ----------
        parameter : str
            Location or auxiliary parameter name.

        Returns
        -------
        int
            Component width, or zero for a scalar predictor.
        """
        if self.predictor_graph is not None:
            from quivers.formulas._predictor_graph import ComponentExpression

            value = self.predictor_graph.expressions.get(parameter)
            if value is None:
                value = self.predictor_graph.roots.get(parameter)
            if parameter != "mu" and parameter in (
                self.predictor_graph.leaves.keys()
                | self.predictor_graph.expressions.keys()
                | self.predictor_graph.roots.keys()
            ):
                return (
                    len(value.components)
                    if isinstance(value, ComponentExpression)
                    else 0
                )
        response = self.family.response
        # mixtures: mu stacks the component locations mu1, ..., muK.
        if parameter == "mu" and self.family.mixture is not None:
            return mixture_component_count(self.family)
        if response is None:
            return 0
        if parameter == "mu":
            return len(self.category_labels) - int(response.kind == "logistic_normal")
        return (
            len(self.category_labels) - 1
            if self.family.component_scale is not None
            and parameter == self.family.component_scale.name
            else 0
        )

    @property
    def coefficient_labels(self) -> tuple[int | str, ...]:
        """Label free component coefficients in their retained coordinate order.

        Returns
        -------
        tuple[int | str, ...]
            Nonreference component labels, or all labels without a reference.
        """
        response = self.family.response
        if response is None:
            return ()
        labels = self.category_labels
        return tuple(
            labels[index]
            for index in response.coordinate_indices(tuple(str(v) for v in labels))
        )

    def _constant_parameter(self, parameter: str) -> bool:
        """Report whether an auxiliary parameter has no predictor of its own.

        Parameters
        ----------
        parameter : str
            Parameter name.

        Returns
        -------
        bool
            True for an auxiliary parameter of the family that neither a
            distributional formula nor the predictor graph predicts, so its
            value is constant over the rows.
        """
        graph = self.predictor_graph
        return (
            parameter != "mu"
            and parameter not in self.distributional
            and not (
                graph is not None
                and parameter
                in (graph.leaves.keys() | graph.expressions.keys() | graph.roots.keys())
            )
            and any(aux.name == parameter for aux in self.family.aux_params)
        )

    def _vector_coordinate(self, parameter: str) -> tuple[Predictor, int] | None:
        """Locate a parameter that is one coordinate of a vector predictor.

        Parameters
        ----------
        parameter : str
            Parameter name, for instance ``muB``.

        Returns
        -------
        tuple[Predictor, int] or None
            The predictor and the coordinate; None for any other parameter.
        """
        for predictor in self.predictors:
            output = predictor.output
            if (
                output is not None
                and not predictor.nonlinear
                and parameter in output.parameters
            ):
                return predictor, output.parameters.index(parameter)
        return None

    def _row_reader(
        self,
        parameter: str,
        binding: LinpredBinding,
        recorded: Trace,
        site: Callable[[str], str],
        location: Callable[[], torch.Tensor] | None = None,
    ) -> Callable[[str], torch.Tensor]:
        """Read the row bindings of a linear predictor from one replay.

        brms's ``mu`` includes the conditional ARMA recursion of a residual
        law, which moves the location by lagged recorded residuals; the
        replayed residual law states that location.

        Parameters
        ----------
        parameter : str
            Predictor name.
        binding : LinpredBinding
            Binding holding the predictor.
        recorded : Trace
            One replayed execution.
        site : Callable[[str], str]
            Program name of a binding of this response: the identity for a
            single fit, the response namespace in a joint program.
        location : Callable[[], torch.Tensor] or None
            The response law's location, recursion included; None reads the
            law of the response's own observation.

        Returns
        -------
        Callable[[str], torch.Tensor]
            The value of a named binding of this response.
        """
        response = site(_qvr_name(self.formula.response_name))

        def read(name: str) -> torch.Tensor:
            if name == binding.site and parameter == "mu" and self.residual is not None:
                if location is not None:
                    return location()
                return linear_predictor(recorded, response)
            return recorded.sites[site(name)].value

        return read

    def _linpred_binding(self, parameter: str, transform: bool) -> LinpredBinding:
        """Choose the row binding holding a linear predictor.

        Parameters
        ----------
        parameter : str
            Predictor name.
        transform : bool
            Request the inverse-link binding.

        Returns
        -------
        LinpredBinding
            The binding, with how its values are read.

        Raises
        ------
        ValueError
            If the parameter has no predictor or an ordinal inverse link is
            requested.
        """
        if parameter == LOCATION and self.location is None:
            if transform and parameter in self.fixed_parameters:
                return LinpredBinding(site=parameter)
            raise ValueError(
                f"the response kernel {self.family.name!r} has no location {parameter!r} "
                "with a fitted predictor; "
                "name one of its parameters with a formula: "
                f"{[aux.name for aux in self.family.aux_params if aux.name not in self.fixed_parameters]}"
            )
        coordinate = self._vector_coordinate(parameter)
        if coordinate is not None and coordinate[0].prefix:
            predictor, index = coordinate
            assert predictor.output is not None
            return LinpredBinding(
                site=predictor.parameter
                if transform
                else f"{predictor.prefix}eta_nonbaseline",
                width=predictor.output.size,
                coordinate=index,
            )
        axis = (
            response_axis(
                self.family, tuple(str(label) for label in self.category_labels)
            )
            if self.family.response is not None
            else None
        )
        if axis is not None and parameter in axis.parameters:
            return LinpredBinding(
                site="eta_nonbaseline",
                width=axis.size,
                coordinate=axis.parameters.index(parameter),
            )
        graph = self.predictor_graph
        graph_parameter = graph is not None and parameter in (
            graph.leaves.keys() | graph.expressions.keys() | graph.roots.keys()
        )
        # brms: an auxiliary parameter without a formula is constant over the
        # rows, and its draws are returned on the parameter's own scale.
        if self._constant_parameter(parameter):
            return LinpredBinding(site=parameter)
        # mixtures: parameter="mu" returns every component location.
        if not graph_parameter and not (
            parameter == "mu" and self.family.mixture is not None
        ):
            self._formula_predictor(parameter)
        ordinal = self.family.ordinal
        if transform and parameter == "mu" and ordinal is not None:
            raise ValueError(
                "ordinal probabilities depend on thresholds; use posterior_epred"
            )
        retained_log_rate = self.family.retains_log_location and parameter == "mu"
        site = (
            ("mu" if transform and not retained_log_rate else "eta")
            if parameter == "mu"
            else (parameter if transform else f"dpar_{parameter}_eta")
        )
        if graph is not None and parameter in (
            graph.leaves.keys() | graph.expressions.keys()
        ):
            site = f"nlpar_{parameter}"
        specific = parameter == "mu" and (self.formula.category_specific)
        width = self._predictor_width(parameter)
        if specific:
            width = (
                len(self.category_labels)
                - 1
                - int(ordinal is not None and ordinal.kind == "hurdle_cumulative")
            )
        return LinpredBinding(
            site=site,
            width=width,
            exponentiate=retained_log_rate and transform,
            specific="eta_nonbaseline" if specific else None,
        )

    def _formula_predictor(self, parameter: str) -> tuple[Formula, str]:
        """Read retained predictor metadata without re-evaluating its formula.

        Parameters
        ----------
        parameter : str
            The location parameter mu or an auxiliary with a retained formula.

        Returns
        -------
        tuple[Formula, str]
            The evaluated predictor and its native coefficient-name prefix.

        Raises
        ------
        ValueError
            If the parameter has no retained distributional formula.
        """
        if (
            self.predictor_graph is not None
            and parameter in self.predictor_graph.leaves
        ):
            return self.predictor_graph.leaves[parameter], f"nlpar_{parameter}_"
        # mixtures: coefficients belong to the component locations mu<k>.
        if parameter == "mu" and self.family.mixture is not None:
            raise ValueError(
                "a mixture has no single location predictor; select a component "
                f"location {self.predictor_labels('mu')}"
            )
        if parameter == "mu":
            return self.formula, ""
        coordinate = self._vector_coordinate(parameter)
        if coordinate is not None:
            return self._formula_predictor(coordinate[0].parameter)
        if parameter not in self.distributional:
            raise ValueError(
                f"parameter {parameter!r} has no fitted distributional predictor"
            )
        return self.distributional[parameter], f"dpar_{parameter}_"
