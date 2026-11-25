"""Structured fields as terms of a predictor.

A field is a latent vector with a structured law, read at a coordinate of each
row: a conditional autoregressive effect of the row's location, or a latent
residual of the row itself.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprIndex,
    LetExprLiteral,
    ProgramStep,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _add, _call, _let, _mul, _sub, _var
from quivers.formulas._autocor import (
    AutocorTerm,
    rebound_residual,
    split_autocor_terms,
)
from quivers.formulas._latent_effects import (
    LatentTerm,
    _named_draw,
    latent_prediction_codes,
)
from quivers.formulas._row_source import PredictorRows
from quivers.formulas._term import (
    AxisDecl,
    ColumnName,
    Contribution,
    Emission,
    InputDecl,
    OutputAxis,
    SiteDecl,
)
from quivers.formulas.formula import Formula

__all__: list[str] = []


class FieldTerm(ValueModel):
    """A structured field indexed by a coordinate of each row.

    Parameters
    ----------
    term : LatentTerm
        Retained term.
    binding : AutocorTerm or None
        Authored residual structure, which fresh rows are bound to; None for
        a spatial field.
    scale_prior : str
        Default prior of the field's scale.
    correlation_prior : str
        Default prior of an unstructured correlation.
    include : bool
        Add the field to the predictor; False is brms's
        ``incl_autocor = FALSE``.
    fresh : bool
        Whether the coordinates are those of fresh rows.
    output : OutputAxis or None
        None; a field has one scalar value per coordinate.
    """

    term: LatentTerm = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    binding: AutocorTerm | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )
    scale_prior: str
    correlation_prior: str
    include: bool = True
    fresh: bool = False
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

    def at(self, rows: PredictorRows) -> FieldTerm:
        """Return the term on other rows.

        Parameters
        ----------
        rows : PredictorRows
            Rows of one request, read through the term's predictor.

        Returns
        -------
        FieldTerm
            The fitted field at the rows' locations, or fresh residuals of
            the rows with the fitted structure.

        Raises
        ------
        ValueError
            For new locations, spatial fields without a location column, or
            fresh residuals of a conditional structure.
        """
        include = rows.source.include
        if rows.frame is None:
            return self.with_(include=include)
        if self.binding is None:
            codes = latent_prediction_codes(self.term, rows.frame)
            return self.with_(term=self.term.with_(codes=codes), include=include)
        return self.with_(
            term=rebound_residual(self.term, self.binding, rows.frame, include=include),
            include=include,
            fresh=True,
        )

    def column_names(self) -> tuple[ColumnName, ...]:
        """List the data columns the term reads under their own names.

        Returns
        -------
        tuple[ColumnName, ...]
            Empty: the inputs are named after the term.
        """
        return ()

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
        coordinates = self.term.name
        return tuple(
            site.with_(fresh=True) if self.fresh and site.name == coordinates else site
            for site in self.term.sites(self.scale_prior, self.correlation_prior)
        )

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
    parsed = (
        {
            item.label: item
            for item in (split_autocor_terms(formula.formula)[1])
            if item.kind != "car"
        }
        if formula.latent_terms
        else {}
    )
    return tuple(
        FieldTerm(
            term=term,
            binding=parsed.get(term.label),
            scale_prior=scale_prior,
            correlation_prior=correlation_prior,
            include=include,
        )
        for term in formula.latent_terms
    )
