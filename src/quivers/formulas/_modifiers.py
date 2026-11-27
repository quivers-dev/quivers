"""The modifiers of a response observation, composed by one fold.

A response term that changes how a response is scored is a `Modifier`:
``rate()`` exposure, ``se()`` known errors, the residual law correlated
across the rows, ``cens()`` and ``trunc()`` recorded events, and
``weights()``. `response_modifiers` lists them in the fixed order of the
fold, and each states what it requires of the declared properties of the
family and the residual law (`Modifier.admit`). The fold changes the native
law and its arguments first (`Modifier.apply`), then the written
observation (`Modifier.observe`).

Every refusal a modifier states is forced: a family that declares no
exposure law, no error scale, or no recorded events has no law for the term;
rows that are not independent given the parameters have no row factor for a
weight to raise or for a censored row to integrate; and brms refuses known
errors beside a conditional recursion or a fixed covariance, and has no model
for a SAR term beside any of these terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import didactic.api as dx
import numpy as np

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE, ValueModel
from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgName,
    DrawArgNamed,
    LetExprLiteral,
    LetExprNode,
    LetStep,
    ObserveStep,
    ProgramStep,
)
from quivers.dsl.let_expr_traversal import substitute_let_expr
from quivers.dsl.program_templates import draw_argument_expression
from quivers.formulas._ast import _add, _bind, _call, _div, _let, _mul, _var
from quivers.formulas._kernel import KernelArguments, KernelEmission
from quivers.formulas._observation import WEIGHTS, Observation
from quivers.formulas._residual_law import ResidualLaw
from quivers.formulas._response_roles import event_option_columns
from quivers.formulas.family import Binding, Family, families
from quivers.formulas.formula import FixedColumn, ResponseModifiers
from quivers.qiec.families import family as semantic_family

__all__: list[str] = []

KNOWN_SCALE = "formula_sigma"
"""Binding of the response scale under known errors."""


def combine_error_scales(
    error: LetExprNode, residual: LetExprNode, reference: LetExprNode
) -> LetExprNode:
    """Combine independent error scales relative to their larger scale.

    Parameters
    ----------
    error, residual : LetExprNode
        Known standard error and residual scale.
    reference : LetExprNode
        Elementwise maximum of the two scales.

    Returns
    -------
    LetExprNode
        The combined scale, without squaring potentially large scales.
    """
    return _mul(
        reference,
        _call(
            "sqrt",
            _add(
                _call("pow", _div(error, reference), LetExprLiteral(value=2.0)),
                _call("pow", _div(residual, reference), LetExprLiteral(value=2.0)),
            ),
        ),
    )


def observation_argument(
    observation: ObserveStep, parameter: str
) -> tuple[int, DrawArg] | None:
    """Find the argument passed to a named native constructor parameter.

    Parameters
    ----------
    observation : ObserveStep
        The native observation.
    parameter : str
        Name of the parameter in the native constructor.

    Returns
    -------
    tuple[int, DrawArg] or None
        Position in the authored arguments and the value passed there.
    """
    schema = semantic_family(observation.morphism).parameters
    for position, argument in enumerate(observation.args or ()):
        if isinstance(argument, DrawArgNamed):
            if argument.parameter == parameter:
                return position, argument.value
        elif position < len(schema) and schema[position].name == parameter:
            return position, argument
    return None


class Admission(ValueModel):
    """What the modifiers of a response are checked against.

    Parameters
    ----------
    family : Family
        Response family and its declared properties.
    modifiers : ResponseModifiers
        Response terms of the formula.
    residual : ResidualLaw or None
        Residual law correlated across the rows.
    scale_sources : frozenset[str]
        Parameters given a formula, a fixed value, or a prior.
    rows : int
        Number of response rows.
    """

    family: Family = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    modifiers: ResponseModifiers = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    residual: ResidualLaw | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    scale_sources: frozenset[str] = frozenset()
    rows: int = 0


class Modifier(dx.TaggedUnion, discriminator="kind"):
    """A response term that changes how a response is scored."""

    @property
    def changes_arguments(self) -> bool:
        """Whether the modifier changes the law before observation."""
        return False

    @property
    def admission_rank(self) -> int:
        """Rank of the modifier's check among those of a response.

        The residual law is checked first, since brms's rules for terms
        beside a residual structure name the structure; the exposure law
        last, since a family without one refuses every ``rate()`` alike.

        Returns
        -------
        int
            Smaller ranks are checked first.
        """
        raise NotImplementedError

    def admit(self, admission: Admission) -> None:
        """Check what the term requires of the family and the residual law.

        Parameters
        ----------
        admission : Admission
            Family, response terms, and residual law.
        """
        return

    def apply(
        self, arguments: KernelArguments, emission: KernelEmission
    ) -> KernelArguments:
        """Change the native law or its arguments.

        Parameters
        ----------
        arguments : KernelArguments
            Native law, its arguments, and the bindings written before it.
        emission : KernelEmission
            Program being emitted.

        Returns
        -------
        KernelArguments
            The changed law; unchanged by a term that does not change it.
        """
        return arguments

    def apply_to_step(
        self, observation: ObserveStep, family: Family
    ) -> tuple[ObserveStep, tuple[ProgramStep, ...]]:
        """Apply a modifier to the arguments of an authored observation.

        Parameters
        ----------
        observation : ObserveStep
            The native observation in its program.
        family : Family
            Declared response law.

        Returns
        -------
        tuple[ObserveStep, tuple[ProgramStep, ...]]
            The observation and bindings it needs in the same scope.

        Raises
        ------
        ValueError
            If the modifier requires a law that is unavailable for the step.
        """
        if self.changes_arguments:
            raise ValueError(
                f"the program {family.name!r} scores its response by its own "
                f"observation; the response term {self.kind!r} changes that law, "
                "so the program must state it itself"
            )
        return observation, ()

    def observe(self, observation: Observation) -> Observation:
        """Change the written observation.

        Parameters
        ----------
        observation : Observation
            The observation of the native law.

        Returns
        -------
        Observation
            The changed observation; unchanged by a term that does not change
            it.
        """
        return observation


@dataclass(frozen=True, slots=True)
class _NativeExposure:
    """The law of an accumulated count in native constructor arguments.

    Parameters
    ----------
    parameter : str
        Constructor parameter this conversion reads.
    arguments : tuple[Binding, ...]
        Constructor arguments in terms of that parameter and ``exposure``.
    kernel : str or None
        Constructor of the accumulated count when it changes.
    """

    parameter: str
    arguments: tuple[Binding, ...]
    kernel: str | None = None


_NATIVE_EXPOSURES: Mapping[str, tuple[_NativeExposure, ...]] = {
    "Poisson": (
        _NativeExposure("rate", (Binding(name="rate", expr="rate * exposure"),)),
    ),
    "NegativeBinomial": (
        _NativeExposure(
            "total_count", (Binding(name="total_count", expr="total_count * exposure"),)
        ),
    ),
    "Geometric": (
        _NativeExposure(
            "probs",
            (
                Binding(name="total_count", expr="exposure"),
                Binding(name="probs", expr="1.0 - probs"),
            ),
            kernel="NegativeBinomial",
        ),
        _NativeExposure(
            "logits",
            (
                Binding(name="total_count", expr="exposure"),
                Binding(name="logits", expr="0.0 - logits"),
            ),
            kernel="NegativeBinomial",
        ),
    ),
}


class RateExposure(Modifier):
    """A ``rate()`` denominator: the count of ``denominator`` exposure units.

    The mean becomes ``mu * denom`` for every link, and the negative binomial
    shape is multiplied by ``denom`` as well, so the count is a sum of
    ``denom`` independent unit-exposure counts. brms writes these as
    ``poisson_log_lpmf(Y | mu + log_denom)`` and
    ``neg_binomial_2_log_lpmf(Y | mu + log_denom, shape * denom)``; the
    geometric family becomes the negative binomial with shape ``denom``.
    Expected values include the denominator; linear predictors and ``mu`` do
    not.

    Parameters
    ----------
    denominator : str
        Binding of the denominators.
    """

    kind: Literal["exposure"] = "exposure"
    denominator: str

    @property
    def changes_arguments(self) -> bool:
        """The modifier changes the law before observation."""
        return True

    @property
    def admission_rank(self) -> int:
        """Rank of the check: 4.

        Returns
        -------
        int
            The rank.
        """
        return 4

    def admit(self, admission: Admission) -> None:
        """Require a law, or the law of every mixture component, under exposure.

        Parameters
        ----------
        admission : Admission
            Family and response terms.

        Raises
        ------
        ValueError
            If the family, or a component of a mixture, declares no exposure
            law.
        """
        family = admission.family
        if family.kernel_law is not None and family.observe_family in _NATIVE_EXPOSURES:
            return
        # The exposure scales the law of each component of a mixture.
        laws = family.mixture.components if family.mixture is not None else (family,)
        for law in laws:
            if law.exposure is None:
                admitted = sorted(
                    name
                    for name, entry in families.items()
                    if entry.exposure is not None
                )
                raise ValueError(
                    f"rate() is not supported for the {law.name} family; "
                    f"it applies to {', '.join(admitted)}"
                )

    def apply(
        self, arguments: KernelArguments, emission: KernelEmission
    ) -> KernelArguments:
        """Pass the law of the count over its exposure.

        Scoring costs O(n) beyond the unscaled law.

        Parameters
        ----------
        arguments : KernelArguments
            Native law of a unit-exposure count.
        emission : KernelEmission
            Program being emitted.

        Returns
        -------
        KernelArguments
            The declared exposure law and its arguments.
        """
        exposure = emission.inputs.family.exposure
        assert exposure is not None
        denominator = _var(self.denominator)
        return arguments.with_(
            kernel=exposure.observe_family or arguments.kernel,
            values=tuple(
                self.denominator if argument == "exposure" else argument
                for argument in exposure.observe_args
            ),
            lets=(
                *arguments.lets,
                *(
                    _let(binding.name, _bind(binding.expr, exposure=denominator))
                    for binding in exposure.observe_lets
                ),
            ),
        )

    def apply_to_step(
        self, observation: ObserveStep, family: Family
    ) -> tuple[ObserveStep, tuple[ProgramStep, ...]]:
        """Pass a native law its arguments over the exposure units.

        Parameters
        ----------
        observation : ObserveStep
            Authored observation of the unit-exposure count.
        family : Family
            Response law declaring the unit-exposure constructor.

        Returns
        -------
        tuple[ObserveStep, tuple[ProgramStep, ...]]
            The accumulated-count observation and bindings for its arguments.

        Raises
        ------
        ValueError
            If the constructor has no exposure law for the supplied arguments.
        """
        for rule in _NATIVE_EXPOSURES.get(observation.morphism, ()):
            found = observation_argument(observation, rule.parameter)
            if found is None:
                continue
            position, original = found
            arguments = list(observation.args or ())
            values = {
                "exposure": _var(self.denominator),
                rule.parameter: draw_argument_expression(original),
            }
            lets: list[ProgramStep] = []
            replacements: list[DrawArgNamed] = []
            for binding in rule.arguments:
                name = f"FormulaProgramExposure_{binding.name}"
                lets.append(_let(name, _bind(binding.expr, **values)))
                replacements.append(
                    DrawArgNamed(parameter=binding.name, value=DrawArgName(text=name))
                )
            if rule.kernel is None:
                (replacement,) = replacements
                arguments[position] = (
                    replacement
                    if isinstance(arguments[position], DrawArgNamed)
                    else replacement.value
                )
            else:
                arguments = list(replacements)
            return (
                observation.with_(
                    morphism=rule.kernel or observation.morphism, args=tuple(arguments)
                ),
                tuple(lets),
            )
        raise ValueError(
            f"rate() has no declared exposure law for the supplied "
            f"{observation.morphism!r} arguments in {family.name!r}"
        )


class KnownError(Modifier):
    """``se()`` known standard errors of the responses.

    With ``sigma = FALSE`` the response scale is the known error; with
    ``sigma = TRUE`` it is ``sqrt(se**2 + sigma**2)``. Errors a time-series
    covariance takes on its diagonal belong to the residual law instead.

    Parameters
    ----------
    sigma : bool
        Whether a residual scale is estimated beside the known errors.
    """

    kind: Literal["known_error"] = "known_error"
    sigma: bool

    @property
    def changes_arguments(self) -> bool:
        """The modifier changes the law before observation."""
        return True

    @property
    def admission_rank(self) -> int:
        """Rank of the check: 1.

        Returns
        -------
        int
            The rank.
        """
        return 1

    def admit(self, admission: Admission) -> None:
        """Require a declared error scale that nothing else gives.

        Parameters
        ----------
        admission : Admission
            Family, response terms, and residual law.

        Raises
        ------
        ValueError
            If the family declares no error scale, or the scale is given a
            formula, value, or prior while the known errors replace it.
        """
        family = admission.family
        law = family.kernel_law
        residual = family.error_scale
        if (
            residual is None
            and law is not None
            and law.error_scale_argument is not None
        ):
            residual = law.error_scale_binding
        elif residual is None:
            admitted = sorted(
                name
                for name, entry in families.items()
                if entry.error_scale is not None
            )
            raise ValueError(
                f"response term se() is not supported for family "
                f"{family.name!r}, which declares no error scale "
                f"(Family.error_scale); families admitting se() are {admitted}"
            )
        if not self.sigma and residual in admission.scale_sources:
            raise ValueError(
                f"se() without sigma=True excludes a separate residual {residual}; "
                "remove its formula, fixed value, or prior, or write "
                "se(error, sigma=True)"
            )

    def apply(
        self, arguments: KernelArguments, emission: KernelEmission
    ) -> KernelArguments:
        """Pass the total scale wherever the family names its residual scale.

        Parameters
        ----------
        arguments : KernelArguments
            Native law over the formula's residual scale.
        emission : KernelEmission
            Program being emitted.

        Returns
        -------
        KernelArguments
            The law over the total scale.
        """
        scale = emission.inputs.family.error_scale
        assert scale is not None
        steps: list[ProgramStep] = []
        emission.host.known_error_scale(emission.formula, steps, emission)
        emission.names[scale] = KNOWN_SCALE
        total: Mapping[str, LetExprNode] = {scale: _var(KNOWN_SCALE)}
        return arguments.with_(
            values=tuple(
                KNOWN_SCALE if value == scale else value for value in arguments.values
            ),
            lets=(
                *steps,
                *(
                    step.with_(value=substitute_let_expr(step.value, total))
                    if isinstance(step, LetStep)
                    else step
                    for step in arguments.lets
                ),
            ),
        )

    def apply_to_step(
        self, observation: ObserveStep, family: Family
    ) -> tuple[ObserveStep, tuple[ProgramStep, ...]]:
        """Replace the native constructor's scale with the known-error scale.

        Parameters
        ----------
        observation : ObserveStep
            The observation in its program.
        family : Family
            Declared response law and its scale parameter.

        Returns
        -------
        tuple[ObserveStep, tuple[ProgramStep, ...]]
            The observation and bindings for its total scale.
        """
        law = family.kernel_law
        assert law is not None and law.error_scale_argument is not None
        parameter = law.error_scale_argument
        found = observation_argument(observation, parameter)
        if found is None:
            raise ValueError(
                f"the observation under {observation.morphism!r} does not pass "
                f"the declared error scale {parameter!r}"
            )
        position, scale = found
        arguments = list(observation.args or ())
        original = arguments[position]
        error = _var("formula_standard_error")
        value = error
        lets: list[ProgramStep] = []
        if self.sigma:
            reference_name = "formula_error_reference"
            residual = draw_argument_expression(scale)
            lets.append(_let(reference_name, _call("max", error, residual)))
            value = combine_error_scales(error, residual, _var(reference_name))
        lets.append(_let(KNOWN_SCALE, value))
        replaced = DrawArgName(text=KNOWN_SCALE)
        arguments[position] = (
            original.with_(value=replaced)
            if isinstance(original, DrawArgNamed)
            else replaced
        )
        return observation.with_(args=tuple(arguments)), tuple(lets)


class ResidualDependence(Modifier):
    """Natural residuals correlated across the rows."""

    kind: Literal["residual_dependence"] = "residual_dependence"

    @property
    def changes_arguments(self) -> bool:
        """The modifier changes the law before observation."""
        return True

    @property
    def admission_rank(self) -> int:
        """Rank of the check: 0.

        Returns
        -------
        int
            The rank.
        """
        return 0

    def admit(self, admission: Admission) -> None:
        """Apply brms's rules for response terms beside residual structures.

        brms scores a time-series covariance and a fixed covariance with one
        multivariate density, which admits no weights, censoring, or
        truncation (``stan_log_lik_gaussian_time`` and ``_fcor`` in
        ``R/stan-likelihood.R``): the rows have no factor of their own for a
        weight to raise, and a censored row would need an orthant
        probability. It adds ``se()`` to a time-series covariance only, with
        ``sigma = TRUE`` (``no_sigma`` in ``R/families.R``); the conditional
        ARMA formulation refuses ``se()`` (``stan_ac`` in
        ``R/stan-predictor.R``) and admits the other terms row by row. For a
        SAR term brms writes these terms into a Stan program that does not
        compile, because ``normal_lagsar_lpdf`` and its siblings take the
        whole response vector and one scalar ``sigma``, so no brms model
        exists to match.

        Parameters
        ----------
        admission : Admission
            Response terms and residual law.

        Raises
        ------
        ValueError
            With brms's message for each combination brms refuses, and a
            located message for the SAR combinations brms cannot compile.
        """
        law = admission.residual
        assert law is not None
        term, modifiers = law.term, admission.modifiers
        roles = dict(modifiers.sources)
        adjusted = [
            name
            for name, present in (
                ("weights()", "weights" in roles),
                ("cens()", modifiers.censoring is not None),
                ("trunc()", modifiers.truncation is not None),
            )
            if present
        ]
        error = modifiers.standard_error
        if term.kind == "sar" and (adjusted or error is not None):
            present = ", ".join([*adjusted, *(["se()"] if error is not None else [])])
            raise ValueError(
                f"{law.caller}: {term.label} with {present} has no brms model: brms "
                "writes these terms into a call of its SAR density, which takes the "
                "whole response vector and one scalar sigma, and the Stan program "
                "does not compile"
            )
        if error is not None:
            if law.conditional:
                raise ValueError(
                    "Please set cov = TRUE in ARMA structures when including known "
                    "standard errors."
                )
            if term.kind == "fcor":
                raise ValueError("Invalid addition arguments for this model.")
            if not error.sigma:
                raise ValueError(
                    "Please set argument 'sigma' of function 'se' to TRUE when "
                    "modeling time-series covariance matrices."
                )
        if adjusted and not law.factorized:
            raise ValueError("Invalid addition arguments for this model.")

    def apply(
        self, arguments: KernelArguments, emission: KernelEmission
    ) -> KernelArguments:
        """Observe the response through the residual law.

        Parameters
        ----------
        arguments : KernelArguments
            Native law of independent rows.
        emission : KernelEmission
            Program being emitted.

        Returns
        -------
        KernelArguments
            The law's kernel over the location and the (total) scale, after
            its structural sites that are not scalar.
        """
        law = emission.inputs.residual
        assert law is not None
        for axis in law.axes():
            emission.declare(axis)
        scale = law.elliptical.scale
        return arguments.with_(
            kernel=law.name,
            values=law.arguments("mu", emission.names.get(scale, scale)),
            lets=(*arguments.lets, *law.steps(emission.terms)),
            factorized=law.factorized,
        )


class RecordedEvents(Modifier):
    """``cens()`` and ``trunc()``: responses recorded as bounds or truncated.

    Parameters
    ----------
    options : tuple[str, ...]
        Recorded-event options, in option order.
    columns : tuple[str, ...]
        The binding of each option.
    """

    kind: Literal["recorded_events"] = "recorded_events"
    options: tuple[str, ...]
    columns: tuple[str, ...]

    @property
    def admission_rank(self) -> int:
        """Rank of the check: 2.

        Returns
        -------
        int
            The rank.
        """
        return 2

    def admit(self, admission: Admission) -> None:
        """Require a family that declares recorded events.

        Parameters
        ----------
        admission : Admission
            Family and response terms.

        Raises
        ------
        ValueError
            If the family does not admit a requested term, naming the family,
            the term, the missing declaration, and the families that admit it.
        """
        modifiers = admission.modifiers
        terms = [
            name
            for name, bound in (
                ("cens", modifiers.censoring),
                ("trunc", modifiers.truncation),
            )
            if bound is not None
        ]
        if not terms or admission.family.recorded_events:
            return
        admitted = sorted(
            name for name, entry in families.items() if entry.recorded_events
        )
        raise ValueError(
            f"response term {' and '.join(f'{term}()' for term in terms)} is not "
            f"supported for family {admission.family.name!r}: the family does not "
            "declare recorded events (Family.recorded_events); families admitting "
            f"cens() and trunc() are {admitted}"
        )

    def observe(self, observation: Observation) -> Observation:
        """Attach the recorded-event columns to the site.

        Parameters
        ----------
        observation : Observation
            Observation of rows that factorize.

        Returns
        -------
        Observation
            The observation with the events.
        """
        return observation.with_recorded_events(
            tuple(zip(self.options, self.columns, strict=True))
        )


class RowWeights(Modifier):
    """``weights()``: each row's likelihood factor raised to its weight."""

    kind: Literal["row_weights"] = "row_weights"

    @property
    def admission_rank(self) -> int:
        """Rank of the check: 3.

        Returns
        -------
        int
            The rank.
        """
        return 3

    def admit(self, admission: Admission) -> None:
        """Require one finite nonnegative weight per row.

        Parameters
        ----------
        admission : Admission
            Response terms and the number of rows.

        Raises
        ------
        ValueError
            If the weights do not align with the rows or are not finite
            nonnegative reals.
        """
        weights = admission.modifiers.weights
        assert weights is not None
        data = weights.data
        if data.shape != (admission.rows,):
            raise ValueError("weights must align with every evaluated response row")
        if (
            data.dtype.kind not in "iuf"
            or not np.isfinite(data).all()
            or np.any(data < 0)
        ):
            raise ValueError("weights must be finite nonnegative real values")

    def observe(self, observation: Observation) -> Observation:
        """Weight the rows of the site.

        Parameters
        ----------
        observation : Observation
            Observation of rows that factorize.

        Returns
        -------
        Observation
            The weighted observation.
        """
        return observation.with_(weights=WEIGHTS)


def response_modifiers(
    modifiers: ResponseModifiers, residual: ResidualLaw | None
) -> tuple[Modifier, ...]:
    """List the modifiers of a response in the order of the fold.

    The order is exposure, known errors, residual law, recorded events,
    weights. Known errors that the residual law takes on its covariance
    (`ResidualLaw.absorbed_roles`) are part of the residual law and not a
    modifier of their own.

    Parameters
    ----------
    modifiers : ResponseModifiers
        Response terms of the formula.
    residual : ResidualLaw or None
        Residual law correlated across the rows.

    Returns
    -------
    tuple[Modifier, ...]
        The modifiers present.
    """
    events = tuple(
        (option, column.qvr_name) for option, column in event_option_columns(modifiers)
    )
    error = modifiers.standard_error
    return (
        *(
            (RateExposure(denominator=modifiers.rate.qvr_name),)
            if modifiers.rate is not None
            else ()
        ),
        *(
            (KnownError(sigma=error.sigma),)
            if error is not None
            and (residual is None or "standard_error" not in residual.absorbed_roles)
            else ()
        ),
        *((ResidualDependence(),) if residual is not None else ()),
        *(
            (
                RecordedEvents(
                    options=tuple(option for option, _ in events),
                    columns=tuple(column for _, column in events),
                ),
            )
            if events
            else ()
        ),
        *((RowWeights(),) if modifiers.weights is not None else ()),
    )


def admit_modifiers(modifiers: tuple[Modifier, ...], admission: Admission) -> None:
    """Check every modifier of a response.

    Each modifier is checked in the order of its `Modifier.admission_rank`.

    Parameters
    ----------
    modifiers : tuple[Modifier, ...]
        Modifiers of the response.
    admission : Admission
        Family, response terms, and residual law.
    """
    for modifier in sorted(modifiers, key=lambda item: item.admission_rank):
        modifier.admit(admission)


def response_columns(
    modifiers: ResponseModifiers, residual: ResidualLaw | None
) -> tuple[tuple[str, FixedColumn], ...]:
    """Name the response columns a response program reads.

    Every column of the response terms is an input of the program except
    those the residual law takes into its covariance.

    Parameters
    ----------
    modifiers : ResponseModifiers
        Response terms of the formula.
    residual : ResidualLaw or None
        Residual law correlated across the rows.

    Returns
    -------
    tuple[tuple[str, FixedColumn], ...]
        Role and column of each input, in canonical input order.
    """
    absorbed = frozenset() if residual is None else residual.absorbed_roles
    return tuple(
        (role, column) for role, column in modifiers.columns() if role not in absorbed
    )
