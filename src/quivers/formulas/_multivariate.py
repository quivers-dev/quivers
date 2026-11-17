"""Response-indexed specifications for one joint (multivariate) formula program.

The Python surface follows brms. `bf` builds one response, and
``bf(...) + bf(...)`` combines responses into a `MultivariateFormula`.
A left-hand side ``mvbind(y1, y2)`` repeats one right-hand side for several
responses. `set_rescor` adds the residual-correlation setting, and `mvbf`
combines any number of responses at once.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Literal, cast

import didactic.api as dx

from quivers._values import OPAQUE_BY_VALUE, ValueModel
from quivers.formulas._predictor_graph import NonlinearExpression
from quivers.formulas.family import Family, families
from quivers.formulas._category_formulas import (
    is_category_parameter,
    is_scale_parameter,
)
from quivers.formulas._mixture import expand_mixture

#: Response families admitting a residual correlation, following brms.
RESCOR_FAMILIES: frozenset[str] = frozenset({"gaussian", "student_t"})

_MVBIND = re.compile(r"^\s*mvbind\s*\((?P<names>[^()]*)\)\s*(?P<rest>[|~].*)$", re.S)
_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


def response_label(formula: str) -> str:
    """Derive the brms response name from a response formula.

    brms names a response by its left-hand-side variable with every
    character other than a letter or digit removed, so ``y_1 ~ x`` and
    ``y.1 ~ x`` both name the response ``y1``.

    Parameters
    ----------
    formula : str
        Response formula, including optional response modifiers after ``|``.

    Returns
    -------
    str
        The response label used in parameter names such as ``b_y1_x``.

    Raises
    ------
    ValueError
        If the formula has no left-hand side or the label is empty.
    """
    if "~" not in formula:
        raise ValueError(f"response formula {formula!r} requires '~'")
    left = formula.split("~", 1)[0].split("|", 1)[0]
    label = "".join(character for character in left if character.isalnum())
    if not label or not label[0].isalpha():
        raise ValueError(
            f"response formula {formula!r} has no response name beginning with a letter"
        )
    return label


def uses_missing_response(formula: str) -> bool:
    """Report whether a response formula declares a ``mi()`` response modifier.

    Parameters
    ----------
    formula : str
        Response formula.

    Returns
    -------
    bool
        True when the response side carries ``mi(...)``.
    """
    left = formula.split("~", 1)[0]
    return "|" in left and re.search(r"\bmi\s*\(", left.split("|", 1)[1]) is not None


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
    mecor : bool
        Correlate the latent values of the response's ``me()`` terms that
        share a grouping, the brms ``set_mecor`` default.
    center : bool or None
        brms ``bf(center = )`` for this response's location intercept.
    decomp : {"none", "QR"}
        brms ``bf(decomp = )`` for this response's location coefficients.
    default_priors : {"quivers", "brms"}
        Default prior policy of this response, set from ``fit``.
    cmc, sparse : bool
        brms ``bf(cmc = , sparse = )``.
    unused : tuple[str, ...]
        brms ``bf(unused = )``: further columns joining row retention.

    Notes
    -----
    Data, priors, inference options, and group covariance declarations belong
    to the joint fit. Each response retains its own component and category
    axes. A response label does not replace either coordinate system.
    Adding two responses with ``+`` returns a `MultivariateFormula`, as
    ``bf(...) + bf(...)`` does in brms.
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
    mecor: bool = True
    # brms population-coordinate options and prior policy.
    center: bool | None = None
    decomp: Literal["none", "QR"] = "none"
    default_priors: Literal["quivers", "brms"] = "quivers"
    cmc: bool = True
    sparse: bool = False
    unused: tuple[str, ...] = ()

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
        if _MVBIND.match(self.formula):
            raise ValueError(
                "mvbind() names several responses; build it with bf() or mvbf()"
            )
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

    @property
    def label(self) -> str:
        """Return the brms response name derived from the left-hand side.

        Returns
        -------
        str
            Left-hand-side variable with non-alphanumeric characters removed.
        """
        return response_label(self.formula)

    def __add__(
        self, other: ResponseModel | MultivariateFormula | RescorSetting | MecorSetting
    ) -> MultivariateFormula | ResponseModel:
        """Combine this response with another response or a setting.

        Parameters
        ----------
        other : ResponseModel or MultivariateFormula or RescorSetting or MecorSetting
            Further responses, the residual-correlation setting, or the
            ``me()`` correlation setting of this response.

        Returns
        -------
        MultivariateFormula or ResponseModel
            The joint specification, responses in authored order, or this
            response with its ``set_mecor`` setting.

        Raises
        ------
        TypeError
            If the operand is not a response, joint formula, or setting.
        ValueError
            If a setting is added to a single response.
        """
        if isinstance(other, MecorSetting):
            return self.with_(mecor=other.mecor)
        if isinstance(other, RescorSetting):
            raise ValueError(
                "set_rescor() applies to a multivariate formula of at least two responses"
            )
        return mvbf(self, other)


class RescorSetting(ValueModel):
    """The residual-correlation setting added to a joint formula.

    Parameters
    ----------
    rescor : bool
        Estimate residual correlations between the responses.
    """

    rescor: bool

    @dx.model_validator()
    def _validate(self) -> None:
        """Require a Boolean setting.

        Raises
        ------
        TypeError
            If ``rescor`` is not Boolean.
        """
        if type(self.rescor) is not bool:
            raise TypeError("set_rescor requires a Boolean")


class MecorSetting(ValueModel):
    """The ``me()`` latent-correlation setting added to a response formula.

    Parameters
    ----------
    mecor : bool
        Correlate the latent values of ``me()`` terms sharing a grouping.
    """

    mecor: bool

    @dx.model_validator()
    def _validate(self) -> None:
        """Require a Boolean setting.

        Raises
        ------
        TypeError
            If ``mecor`` is not Boolean.
        """
        if type(self.mecor) is not bool:
            raise TypeError("set_mecor requires a Boolean")


def set_mecor(mecor: bool = True) -> MecorSetting:
    """Return the ``me()`` correlation setting, spelled as in brms.

    Parameters
    ----------
    mecor : bool
        Correlate the latent values of a response's ``me()`` terms.

    Returns
    -------
    MecorSetting
        A value added to a response or joint formula with ``+``.
    """
    return MecorSetting(mecor=mecor)


def set_rescor(rescor: bool = True) -> RescorSetting:
    """Return the residual-correlation setting, spelled as in brms.

    Parameters
    ----------
    rescor : bool
        Estimate residual correlations between Gaussian or Student responses.

    Returns
    -------
    RescorSetting
        A value added to a joint formula with ``+``.
    """
    return RescorSetting(rescor=rescor)


def default_rescor(responses: Mapping[str, ResponseModel]) -> bool:
    """Resolve an unset residual-correlation policy as brms does.

    brms estimates residual correlations by default exactly when every
    response uses one common family that admits them (Gaussian or Student)
    and no response is a missing-value ``mi()`` response.

    Parameters
    ----------
    responses : Mapping[str, ResponseModel]
        The joint responses.

    Returns
    -------
    bool
        The brms default.
    """
    names = {response.resolved_family.name for response in responses.values()}
    return (
        len(names) == 1
        and names <= RESCOR_FAMILIES
        and not any(
            uses_missing_response(response.formula) for response in responses.values()
        )
    )


class MultivariateFormula(ValueModel):
    """Combine named response graphs into one probabilistic program.

    Parameters
    ----------
    responses : dict[str, ResponseModel]
        At least two explicitly named responses, in authored coordinate order.
        Names select priors and result views and form brms parameter names
        such as ``b_<resp>_<coef>``. They contain only letters and digits.
    rescor : bool or None
        Estimate residual correlation between aligned scalar Gaussian or
        Student responses. None applies the brms default (`default_rescor`).
        Shared group-effect identifiers apply in either mode.
    response_names : tuple[str, ...] or None
        Explicit response coordinate order. None retains mapping insertion
        order. The order is part of equality, hashing, and serialization.
    rescor_explicit : bool or None
        Whether the setting was given rather than defaulted. A defaulted
        setting is resolved again when responses are added, as brms resolves
        an unset ``rescor`` only for the complete model. None records
        whether ``rescor`` was given.
    """

    responses: dict[str, ResponseModel] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    rescor: bool = False
    response_names: tuple[str, ...]
    rescor_explicit: bool = False

    def __init__(
        self,
        *,
        responses: Mapping[str, ResponseModel],
        rescor: bool | None = None,
        response_names: tuple[str, ...] | None = None,
        rescor_explicit: bool | None = None,
    ) -> None:
        """Retain the responses with an explicit coordinate order.

        Parameters
        ----------
        responses : Mapping[str, ResponseModel]
            Named responses, copied into a plain dictionary.
        rescor : bool or None
            Estimate residual correlation between aligned responses, or None
            for the brms default.
        response_names : tuple[str, ...] or None
            Explicit response order, or None for the mapping's insertion order.
        rescor_explicit : bool or None
            Whether the setting was given, or None to record whether
            ``rescor`` is None.
        """
        construct = cast(Callable[..., None], super().__init__)
        construct(
            responses=dict(responses),
            rescor=default_rescor(responses)
            if rescor is None and len(responses) > 1
            else bool(rescor),
            response_names=tuple(responses)
            if response_names is None
            else response_names,
            rescor_explicit=rescor is not None
            if rescor_explicit is None
            else rescor_explicit,
        )

    @dx.model_validator()
    def _validate(self) -> None:
        """Check response labels, coordinate order, and the residual policy.

        Raises
        ------
        ValueError
            If no response is given, the order does not list
            each response exactly once, a label is not alphanumeric, or
            residual correlation joins families other than one common
            Gaussian or Student family.
        TypeError
            If ``rescor`` is not Boolean or a response is not a ResponseModel.
        """
        if not self.responses:
            raise ValueError("a joint formula requires at least one response")
        if type(self.rescor) is not bool:
            raise TypeError("rescor must be Boolean")
        if len(self.response_names) != len(self.responses) or set(
            self.response_names
        ) != set(self.responses):
            raise ValueError("response_names must list every response exactly once")
        for name, response in self.responses.items():
            if not isinstance(name, str) or not _LABEL.match(name):
                raise ValueError(
                    f"response label {name!r} must start with a letter and contain "
                    "only letters and digits, as brms response names do"
                )
            if not isinstance(response, ResponseModel):
                raise TypeError("responses must map labels to ResponseModel values")
        if self.rescor:
            names = {
                response.resolved_family.name for response in self.responses.values()
            }
            if len(names) != 1 or not names <= RESCOR_FAMILIES:
                raise ValueError(
                    "rescor requires homogeneous Gaussian or Student responses; "
                    f"got families {sorted(names)!r}"
                )

    def __add__(
        self, other: ResponseModel | MultivariateFormula | RescorSetting | MecorSetting
    ) -> MultivariateFormula:
        """Add responses, the residual-correlation, or the ``me()`` setting.

        Parameters
        ----------
        other : ResponseModel or MultivariateFormula or RescorSetting or MecorSetting
            Further responses, `set_rescor`'s setting, or `set_mecor`'s
            setting applied to every response.

        Returns
        -------
        MultivariateFormula
            The extended specification.
        """
        if isinstance(other, MecorSetting):
            return MultivariateFormula(
                responses={
                    name: response.with_(mecor=other.mecor)
                    for name, response in self.responses.items()
                },
                rescor=self.rescor,
                response_names=self.response_names,
                rescor_explicit=self.rescor_explicit,
            )
        if isinstance(other, RescorSetting):
            return MultivariateFormula(
                responses=self.responses,
                rescor=other.rescor,
                response_names=self.response_names,
            )
        return mvbf(self, other)

    def response(self, name: str) -> ResponseModel:
        """Return one named response.

        Parameters
        ----------
        name : str
            Response label.

        Returns
        -------
        ResponseModel
            The response specification.

        Raises
        ------
        ValueError
            If no response has this label.
        """
        if name not in self.responses:
            raise ValueError(
                f"unknown response {name!r}; expected one of {self.response_names!r}"
            )
        return self.responses[name]

    def with_default_priors(
        self, policy: Literal["quivers", "brms"]
    ) -> MultivariateFormula:
        """Apply one default prior policy to every response.

        Parameters
        ----------
        policy : {"quivers", "brms"}
            Default prior policy, as `fit` keyword ``default_priors``.

        Returns
        -------
        MultivariateFormula
            The same specification, each response under ``policy``.
        """
        return MultivariateFormula(
            responses={
                name: response.with_(default_priors=policy)
                for name, response in self.responses.items()
            },
            rescor=self.rescor,
            response_names=self.response_names,
            rescor_explicit=self.rescor_explicit,
        )


def mvbf(
    *parts: ResponseModel | MultivariateFormula, rescor: bool | None = None
) -> MultivariateFormula:
    """Combine responses into one joint formula, as brms ``mvbf`` does.

    Parameters
    ----------
    *parts : ResponseModel or MultivariateFormula
        Responses, or joint formulas whose responses are appended in order.
    rescor : bool or None
        Residual-correlation setting. None keeps a setting carried by a
        joint part, and otherwise applies the brms default.

    Returns
    -------
    MultivariateFormula
        Responses keyed by their brms names.

    Raises
    ------
    ValueError
        If two responses share a name.
    TypeError
        If a part is neither a response nor a joint formula.
    """
    responses: dict[str, ResponseModel] = {}
    carried: bool | None = None
    for part in parts:
        if isinstance(part, MultivariateFormula):
            items = [(name, part.responses[name]) for name in part.response_names]
            if part.rescor_explicit and carried is None:
                carried = part.rescor
        elif isinstance(part, ResponseModel):
            items = [(part.label, part)]
        else:
            raise TypeError(
                "mvbf combines ResponseModel and MultivariateFormula values"
            )
        for name, response in items:
            if name in responses:
                raise ValueError(
                    f"response {name!r} appears twice; brms forbids using the same "
                    "response variable twice in one model"
                )
            responses[name] = response
    return MultivariateFormula(
        responses=responses, rescor=carried if rescor is None else rescor
    )


def bf(
    formula: str,
    *parameters: str,
    family: str | Family = "gaussian",
    nl: bool = False,
    fixed_parameters: Mapping[str, float] | None = None,
    weights: str | None = None,
    decision: str | None = None,
    binomial_trials: int | str | None = None,
    thresholds_by: str | None = None,
    mixture_components: int = 2,
    center: bool | None = None,
    decomp: Literal["none", "QR"] = "none",
    cmc: bool = True,
    sparse: bool = False,
    unused: tuple[str, ...] = (),
) -> ResponseModel | MultivariateFormula:
    """Build a response formula with brms ``bf()`` syntax.

    Parameters
    ----------
    formula : str
        Response formula. A left-hand side ``mvbind(y1, y2, ...)`` repeats
        the right-hand side and every other argument for each named response.
    *parameters : str
        Formulas for distributional parameters (``"sigma ~ x"``) or, with
        ``nl=True``, for nonlinear parameters (``"a ~ 1 + (1 | g)"``).
        A left-hand side naming an auxiliary parameter of the family is
        distributional; any other name is nonlinear.
    family : str or Family
        Response family of every response built by this call.
    nl : bool
        Read the main formula's right-hand side as a nonlinear expression.
    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters fixed on their natural scale.
    weights, decision, binomial_trials, thresholds_by : str or int or None
        Ordinary response-role shortcuts.
    mixture_components : int
        Component count of a finite mixture response.
    center : bool or None
        brms ``center``: sample the location intercept on the centered
        design; None centers under the brms default prior policy only.
    decomp : {"none", "QR"}
        brms ``decomp``: sample the location coefficients in QR coordinates.
    cmc : bool
        brms ``cmc``: False keeps treatment coding without an intercept.
    sparse : bool
        brms ``sparse``: the same model without centering.
    unused : tuple[str, ...]
        brms ``unused``: further columns kept with the model rows.

    Returns
    -------
    ResponseModel or MultivariateFormula
        One response; a joint formula for an ``mvbind`` left-hand side,
        whose residual-correlation setting follows the brms default; or a
        one-response joint formula when ``mi()``, ``me()``, or ``index()``
        terms appear, since those compile through the joint program.

    Raises
    ------
    ValueError
        If a parameter formula lacks a left-hand side, names a nonlinear
        parameter without ``nl=True``, or ``mvbind`` names fewer than two
        responses.
    """
    resolved = families[family] if isinstance(family, str) else family
    resolved = expand_mixture(resolved, mixture_components)
    auxiliary = {parameter.name for parameter in resolved.aux_params}
    distributional: dict[str, str | NonlinearExpression] = {}
    nonlinear: dict[str, str | NonlinearExpression] = {}
    for text in parameters:
        if "~" not in text or not text.split("~", 1)[0].strip():
            raise ValueError(f"parameter formula {text!r} requires a named left side")
        name = text.split("~", 1)[0].strip()
        # brms mu<c> category locations and mixture component parameters.
        if (
            name in auxiliary
            or is_category_parameter(resolved, name)
            or is_scale_parameter(resolved, name)
        ):
            distributional[name] = text
        elif nl:
            nonlinear[name] = "~" + text.split("~", 1)[1]
        else:
            raise ValueError(
                f"parameter {name!r} is not an auxiliary parameter of family "
                f"{resolved.name!r}; nonlinear parameters require nl=True"
            )

    def response(text: str) -> ResponseModel:
        """Build one response with this call's shared options.

        Parameters
        ----------
        text : str
            One response formula.

        Returns
        -------
        ResponseModel
            The response specification.
        """
        return ResponseModel(
            formula=text,
            family=family,
            distributional=distributional or None,
            nonlinear=nonlinear if nl else None,
            fixed_parameters=dict(fixed_parameters) if fixed_parameters else None,
            weights=weights,
            decision=decision,
            binomial_trials=binomial_trials,
            thresholds_by=thresholds_by,
            mixture_components=mixture_components,
            center=center,
            decomp=decomp,
            cmc=cmc,
            sparse=sparse,
            unused=tuple(unused),
        )

    bound = _MVBIND.match(formula)
    if bound is None:
        single = response(formula)
        # mi() and me() terms compile through the joint program, so one
        # response with them is a one-response joint formula.
        if uses_missing_syntax(formula, *parameters):
            return MultivariateFormula(responses={single.label: single})
        return single
    names = [name.strip() for name in bound.group("names").split(",") if name.strip()]
    if len(names) < 2:
        raise ValueError("mvbind() requires at least two response variables")
    return mvbf(*(response(name + " " + bound.group("rest")) for name in names))


def uses_missing_syntax(*formulas: str) -> bool:
    """Report whether formulas use ``mi()``, ``me()``, or ``index()`` terms.

    Parameters
    ----------
    *formulas : str
        Formula texts.

    Returns
    -------
    bool
        True when any formula uses one of these terms.
    """
    return any(
        re.search(r"\b(mi|me|index)\s*\(", formula) is not None for formula in formulas
    )


def is_mvbind(formula: str) -> bool:
    """Report whether a formula's left-hand side is ``mvbind(...)``.

    Parameters
    ----------
    formula : str
        Authored formula text.

    Returns
    -------
    bool
        True for an ``mvbind`` left-hand side.
    """
    return _MVBIND.match(formula) is not None


__all__: list[str] = []
