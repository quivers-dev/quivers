"""Structured fields as terms of a predictor.

A field is a latent vector with a structured law, read at a coordinate of each
row: a conditional autoregressive effect of the row's location, or a latent
residual of the row itself.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    LetExprNode,
    LetExprIndex,
    LetExprLiteral,
    ProgramStep,
)
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._ast import _add, _call, _factor, _index, _let, _mul, _sub, _var
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
    COORDINATE,
    coordinate_scopes,
    AxisDecl,
    EffectSource,
    FormulaPart,
    PopulationCoefficient,
    VariableDecl,
    SmoothPart,
    ColumnName,
    Contribution,
    Emission,
    InputDecl,
    OutputAxis,
    SiteDecl,
)
from quivers.formulas._variable_catalogue import CHOLESKY_CLASSES, time_label
from quivers.formulas.formula import Formula

if TYPE_CHECKING:
    from quivers.formulas._name_scope import NameScope

__all__: list[str] = []


class FieldTerm(ValueModel):
    """A structured field indexed by a coordinate of each row.

    Parameters
    ----------
    prefix : str
        Predictor namespace.
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
        Axis of the predictor, each of whose coordinates has a field of its
        own with its own hyperparameters.
    """

    prefix: str
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

    @property
    def _coordinates(self) -> tuple[tuple[str, LetExprNode | None], ...]:
        """List the fields of the term.

        Returns
        -------
        tuple[tuple[str, LetExprNode or None], ...]
            Suffix of the sites of each field, and its output coordinate.
        """
        if self.output is None:
            return (("", None),)
        return tuple(
            (f"_{index}", LetExprLiteral(value=index, integral=True))
            for index in range(self.output.size)
        )

    def _local_sites(self) -> tuple[SiteDecl, ...]:
        """Declare the sites of one field under their own names.

        Returns
        -------
        tuple[SiteDecl, ...]
            The scale, the structural parameters, and the coordinates.
        """
        return self.term.sites(self.scale_prior, self.correlation_prior)

    def variables(self, scope: NameScope) -> tuple[VariableDecl, ...]:
        """Name the term's draws as brms does.

        Parameters
        ----------
        scope : NameScope
            Names of the term's predictor.

        Returns
        -------
        tuple[VariableDecl, ...]
            Scale, structural parameters, and coordinates of each field.
        """
        declarations = []
        for (suffix, coordinate), (index, part) in zip(
            self._coordinates, coordinate_scopes(self.output, scope), strict=True
        ):
            owner = "_" + part.infix.rstrip("_") if part.infix else ""
            for site in self._local_sites():
                vector = site.event_axis is not None
                if not site.overridable:
                    # A field's own coordinates have no hyperparameter axis.
                    declarations.append(
                        VariableDecl(
                            name=f"{site.name}{owner}",
                            site=f"{self.prefix}{site.name}{suffix}",
                            kind="latent",
                            parameter=part.position,
                        )
                    )
                    continue
                lag = site.name.removeprefix(f"{site.class_}_")
                name = (
                    f"{site.class_}{owner}[{lag}]"
                    if lag != site.name
                    else CHOLESKY_CLASSES.get(site.class_, site.name) + owner
                )
                declarations.append(
                    VariableDecl(
                        name=name,
                        site=f"{self.prefix}{site.name}{suffix if vector else ''}",
                        kind="special",
                        index=() if vector else index,
                        pairs=tuple(
                            time_label(level) for level in self.term.time_levels
                        )
                        if vector
                        else (),
                        parameter=part.position,
                    )
                )
        return tuple(declarations)

    def coefficients(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the coefficients of the term for posterior draws.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Constrained draws with chain and draw axes.

        Returns
        -------
        torch.Tensor
            Field by coordinate, then output coordinate.
        """
        values = [
            self._field_values(
                {
                    site.name: draws[f"{self.prefix}{site.name}{suffix}"]
                    if site.event_axis is not None or not site.overridable
                    else draws[self.prefix + site.name]
                    if self.output is None
                    else draws[self.prefix + site.name][..., index]
                    for site in self._local_sites()
                }
            )
            for index, (suffix, _) in enumerate(self._coordinates)
        ]
        return values[0] if self.output is None else torch.stack(values, dim=-1)

    def _field_values(self, draws: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate one field from the draws of its sites.

        Parameters
        ----------
        draws : Mapping[str, torch.Tensor]
            Draws under the sites' own names.

        Returns
        -------
        torch.Tensor
            Field by coordinate.
        """
        term, name = self.term, self.term.name
        if term.layout is not None:
            return draws[name]
        if term.car_type in {"escar", "esicar"}:
            return draws[f"r{name}"]
        scale = draws[f"sd{name}"][..., None]
        if term.car_type == "icar":
            return scale * draws[f"z{name}"]
        assert term.graph is not None
        mixing = draws[f"rho{name}"][..., None]
        inverse = torch.tensor(term.graph.inverse_scale, dtype=scale.dtype)
        return scale * (
            (1 - mixing).sqrt() * draws[f"nsz{name}"]
            + (mixing * inverse).sqrt() * draws[f"z{name}"]
        )

    def population(self) -> tuple[PopulationCoefficient, ...]:
        """List the term's population-level coefficients.

        Returns
        -------
        tuple[PopulationCoefficient, ...]
            Empty.
        """
        return ()

    def smooth_part(self, label: str) -> SmoothPart | None:
        """Return the term's share of one smooth or Gaussian process.

        Parameters
        ----------
        label : str
            Authored smooth term.

        Returns
        -------
        SmoothPart or None
            None.
        """
        return None

    def effects(self) -> tuple[EffectSource, ...]:
        """List the design terms whose variables brms plots.

        Returns
        -------
        tuple[EffectSource, ...]
            Empty: brms plots no structured fields.
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
                name=f"{self.prefix}{self.term.name}_per_row",
                rows=True,
                coordinates=self.output is not None,
                part=FormulaPart(field="latent_terms", key=self.term.name),
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
        axes = [AxisDecl(name=self.prefix + term.axis, size=term.size)]
        if term.layout is not None and term.layout.kind == "unstr":
            axes.append(
                AxisDecl(
                    name=self.prefix + "FormulaAutocorTime",
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
                name=self.prefix + name,
                element=axes.get(name, ("Real", ()))[0],
                axes=axes.get(name, ("Real", (self.prefix + term.axis,)))[1],
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
            The scale, the structural parameters, and the coordinates of each
            field.
        """
        output, sites = self.output, []
        for site in self._local_sites():
            fresh = self.fresh and site.name == self.term.name
            name = self.prefix + site.name
            if site.event_axis is None and site.overridable:
                sites.append(
                    site.with_(
                        name=name,
                        axes=(output.name,) if output is not None else (),
                        component_labels=output.labels if output is not None else (),
                        parameters=output.parameters if output is not None else (),
                    )
                )
                continue
            sites.extend(
                site.with_(
                    name=name + suffix,
                    fresh=fresh,
                    axes=tuple(self.prefix + axis for axis in site.axes),
                    event_axis=None
                    if site.event_axis is None
                    else self.prefix + site.event_axis,
                )
                for suffix, _ in self._coordinates
            )
        return tuple(sites)

    def catalog(self) -> FamilyCatalog | None:
        """Return the laws only the term's compilation knows.

        Returns
        -------
        FamilyCatalog or None
            The laws of the field's coordinates.
        """
        catalog = self.term.catalog
        return catalog.renamed(
            {
                name: self._law(name)
                for name in (item.name for item in catalog.definitions)
            }
            if self.prefix
            else {}
        )

    def _law(self, name: str) -> str:
        """Name a law of the term in the program.

        Parameters
        ----------
        name : str
            Law of the term's own catalog.

        Returns
        -------
        str
            Its name, qualified by the predictor.
        """
        return f"{name}_{self.prefix.rstrip('_')}" if self.prefix else name

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
        name, output = self.prefix + self.term.name, self.output
        hyper = tuple(
            site
            for site in self.sites()
            if site.event_axis is None and site.overridable
        )
        steps: list[ProgramStep] = []
        effects = []
        for position, (suffix, coordinate) in enumerate(self._coordinates):
            names = {
                site.name.removeprefix(self.prefix): site.name + suffix
                for site in hyper
            }
            drawn = {site.name.removeprefix(self.prefix): site for site in hyper}

            def draw(local: str, at: int = position) -> tuple[ProgramStep, ...]:
                """Draw a hyperparameter once, and read it for this field."""
                site = drawn[local]
                draws = (emission.draw(site),) if not at else ()
                if output is None or coordinate is None:
                    return draws
                return (
                    *draws,
                    _let(names[local], _index(site.name, coordinate)),
                )

            effect, field = (
                self._residual_steps(emission, suffix, names, draw)
                if self.term.layout is not None
                else self._spatial_steps(suffix, names, draw)
            )
            steps.extend(field)
            effects.append(effect)
        contribution = (
            _mul(
                _var(f"{name}_include"),
                LetExprIndex(array=_var(effects[0]), indices=(_var(f"{name}_idx"),)),
            )
            if output is None
            else _factor(
                ((COORDINATE, output.name),),
                cases=tuple(
                    (
                        position,
                        _mul(
                            _var(f"{name}_include"),
                            LetExprIndex(
                                array=_var(effect), indices=(_var(f"{name}_idx"),)
                            ),
                        ),
                    )
                    for position, effect in enumerate(effects)
                ),
            )
        )
        return (*steps, _let(f"{name}_per_row", contribution))

    def _residual_steps(
        self,
        emission: Emission,
        suffix: str,
        names: Mapping[str, str],
        draw: Callable[[str], tuple[ProgramStep, ...]],
    ) -> tuple[str, tuple[ProgramStep, ...]]:
        """Draw latent residuals from their structured law.

        Parameters
        ----------
        emission : Emission
            Program being emitted.
        suffix : str
            Suffix of the field's sites.
        names : Mapping[str, str]
            Binding of each hyperparameter of the field.
        draw : Callable
            Draws one hyperparameter and reads it for the field.

        Returns
        -------
        tuple[str, tuple[ProgramStep, ...]]
            The residual binding and its steps.
        """
        name = self.prefix + self.term.name
        scale = f"sd{self.term.name}"
        coefficients = tuple(site for site, *_ in self.term.coefficients)
        steps: list[ProgramStep] = [
            step for site in (scale, *coefficients) for step in draw(site)
        ]
        arguments: dict[str, str | float] = {
            "scale": names[scale],
            **{site: names[site] for site in coefficients},
        }
        for axis in self.axes()[1:]:
            factor = f"{self.prefix}formula_cortime_factor{suffix}"
            steps.extend(
                emission.correlation(f"{self.prefix}Lcortime{suffix}", axis, factor)
            )
            arguments["factor"] = factor
        arguments["anchor"] = f"{name}_anchor"
        steps.append(_named_draw(name + suffix, self._law("LatentResidual"), arguments))
        return name + suffix, tuple(steps)

    def _spatial_steps(
        self,
        suffix: str,
        names: Mapping[str, str],
        draw: Callable[[str], tuple[ProgramStep, ...]],
    ) -> tuple[str, tuple[ProgramStep, ...]]:
        """Draw a conditional autoregressive field over its locations.

        Parameters
        ----------
        suffix : str
            Suffix of the field's sites.
        names : Mapping[str, str]
            Binding of each hyperparameter of the field.
        draw : Callable
            Draws one hyperparameter and reads it for the field.

        Returns
        -------
        tuple[str, tuple[ProgramStep, ...]]
            The binding of the location effects and its steps.
        """
        term, name = self.term, self.term.name
        local = self.prefix + name
        scale, anchor = f"sd{name}", f"{local}_anchor"
        effect = f"{self.prefix}r{name}{suffix}"
        steps: list[ProgramStep] = [*draw(scale)]
        if term.car_type == "escar":
            steps.extend(draw(name))
            steps.append(
                _named_draw(
                    effect,
                    self._law("SparseCAR"),
                    {"scale": names[scale], "alpha": names[name], "anchor": anchor},
                )
            )
            return effect, tuple(steps)
        if term.car_type == "esicar":
            steps.append(
                _named_draw(
                    effect,
                    self._law("SparseICAR"),
                    {"scale": names[scale], "anchor": anchor},
                )
            )
            return effect, tuple(steps)
        standard = f"{self.prefix}z{name}{suffix}"
        steps.append(
            _named_draw(
                standard, self._law("SparseICAR"), {"scale": 1.0, "anchor": anchor}
            )
        )
        if term.car_type == "icar":
            steps.append(_let(effect, _mul(_var(standard), _var(names[scale]))))
            return effect, tuple(steps)
        steps.extend(draw(f"rho{name}"))
        unscaled = f"{self.prefix}nsz{name}{suffix}"
        steps.append(
            _named_draw(
                unscaled,
                "Normal",
                {"loc": 0.0, "scale": 1.0},
                index=self.prefix + term.axis,
            )
        )
        rho = _var(names[f"rho{name}"])
        spatial = _mul(
            _call("sqrt", _mul(rho, _var(f"{local}_inverse_scale"))), _var(standard)
        )
        unstructured = _mul(
            _call("sqrt", _sub(LetExprLiteral(value=1.0), rho)), _var(unscaled)
        )
        steps.append(
            _let(effect, _mul(_add(unstructured, spatial), _var(names[scale])))
        )
        return effect, tuple(steps)


def field_terms(
    formula: Formula,
    prefix: str = "",
    *,
    output: OutputAxis | None = None,
    scale_prior: str = "",
    correlation_prior: str = "",
    include: bool = True,
) -> tuple[FieldTerm, ...]:
    """Build the structured fields of one predictor.

    Parameters
    ----------
    formula : Formula
        Evaluated predictor.
    prefix : str
        Its namespace.
    output : OutputAxis or None
        Its axis.
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
            prefix=prefix,
            output=output,
            term=term,
            binding=parsed.get(term.label),
            scale_prior=scale_prior,
            correlation_prior=correlation_prior,
            include=include,
        )
        for term in formula.latent_terms
    )
