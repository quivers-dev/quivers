"""Bidirectional `didactic.api.Lens` from `Formula` to a
QVR [`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] AST.

The compilation from a formula to a QVR program is a panproto-style
*lens* whose complement is the strict subset of `Formula`
fields that are not recoverable from the emitted
`Module`, packaged as `FormulaData`. The forward
direction produces ``(module, formula_data)`` where the structural
fields of the formula (which columns exist, intercept flag, random
effect group / slope structure, response identifier) are encoded as
QVR latent / let / observe shape and the un-encodable fields (per-row
data, original identifier names lost to `_qvr_name`'s
non-alphanumeric → underscore substitution, the ``term`` / ``name``
presentation labels, the original formula string) ride in
`FormulaData`. The backward direction recovers the structure
by calling `_decode_module` on the target and fuses it with the
complement.

The lens target is a `Module`
that the existing [`quivers.dsl.compiler.Compiler`][quivers.dsl.compiler.Compiler] consumes
directly, identical in shape to one produced by
[`quivers.dsl.parser.parse`][quivers.dsl.parser.parse].
Authored prior strings enter through that parser once per distinct expression
within a forward preparation; emission and specialization retain their source AST.

Emitted structure (one named scalar coefficient per design-matrix
*column*, matching the brms / lme4 canonical layout in which
``poly(x, 2)`` produces two coefficients ``poly(x, 2)_1`` and
``poly(x, 2)_2``):

* One ``object Resp : N`` declaration per response plate.
* One ``object G : K`` declaration per random-effect grouping
  factor (with ``K`` levels).
* For each fixed-design column ``c``: one scalar latent draw
  inside the program body, ``beta_c <- Normal(0, fixed_prior)``,
  whose scale is autoscaled by ``1 / rms(c)`` so the nominal
  value reads in contribution space and means the same thing on
  a raw predictor and on an orthonormal ``poly`` column. An
  explicit per-name prior is emitted as written. The per-row
  covariate values for ``c`` flow in as a free variable via the
  host-data channel (``observations[c]``).
* For each random-effect group ``(slope | g)``: a
  `HalfNormal` scale latent plus a per-level plate draw,
  with the per-row contribution as a plate-gather
  ``alpha_g[g_idx]`` (or ``beta_g_slope[g_idx] * slope`` for a
  random slope). Single-bar blocks couple coefficients through an LKJ
  Cholesky factor; double bars retain independent effects. Explicit shared
  IDs can couple location and auxiliary coefficients.
* One ``observe`` step closes the program with the family's
  observation kernel applied to the inverse-link of the linear
  predictor.
* Categorical responses use category zero as an identified reference
  and lift every fixed and random term over the remaining ``K - 1``
  logits before applying softmax.
* Finite mixtures compose one ordinary emission per component into a
  native ``Mixture`` observation (`quivers.formulas._mixture`).

GetPut holds for every `Formula` ``f``: ``backward(*forward(f)) == f``.
"""

from __future__ import annotations


from quivers._count_capacity import _MAX_EXACT_COUNT

import math
import re
from collections.abc import Collection, Iterable, Sequence
from typing import Literal, Mapping

import didactic.api as dx
import numpy as np
import torch

from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprIdent,
    LetExprBinOp,
    LetExprCall,
    LetExprLiteral,
    LetExprNode,
    LetExprVar,
    LetStep,
    DiscreteConstructor,
    DrawArgNamed,
    DrawArg,
    DrawArgList,
    DrawArgDist,
    DrawArgScalar,
    Module,
    ObserveStep,
    OptionEntry,
    OptionName,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    atom_to_draw_arg as _to_draw_arg,
    Statement,
    ObjectDecl,
    TypeFromExpr,
    TypeName,
    QiecIndexName,
    QiecShapeIndex,
    QiecTypeApplication,
    QiecTypeName,
    QiecValueParameter,
)
from quivers.dsl.emit import draw_call_to_source
from quivers.dsl.draw_args import bind_family_arguments
from quivers.dsl.family_schemas import family_parameter_names
from quivers.family_catalog import FamilyCatalog
from quivers.formulas._design import SymbolicColumn
from quivers.formulas.family import Domain, Family, Link, families, links
from quivers.formulas._recorded_event_families import check_recorded_event_family

# brms mu<c>: category-specific predictors of categorical and simplex families.
from quivers.formulas._category_formulas import (
    append_category_predictors,
    append_scale_predictors,
    component_valued_prefixes,
    is_category_parameter,
    is_scale_parameter,
    main_without_terms,
)

# brms rate() response denominators.
from quivers.formulas._rate import rate_arguments
from quivers.formulas._threshold_groups import (
    append_group_log_masses,
    append_group_thresholds,
    threshold_inputs,
    threshold_kinds,
)

# mixtures: brms mixture() families compose ordinary component emissions.
from quivers.formulas._mixture import (
    emit_mixture,
    expand_mixture,
    validate_mixture_options,
)
from quivers.formulas._group_layout import GroupBlockLayout
from quivers.formulas._group_law import GroupBlockLaw, LevelSite, law_inputs
from quivers.formulas._group_emission import bind_rows, emit_group_block
from quivers.formulas._group_plan import (
    coefficient_binding,
    plan_group_blocks,
    row_map,
    plan_threshold_block,
)
from quivers.formulas._group_covariance import (
    GroupCovariance,
    GroupCovarianceBasis,
)
from quivers.formulas._ast import (
    _bind,
    _draw,
    _let,
    _var,
    _add,
    _mul,
    _sub,
    _div,
    _call,
    _index,
    _factor,
    finite_object_sizes,
)
from quivers.formulas._predictor_graph import (
    PredictorGraph,
    ComponentExpression,
    predictor_items,
)
from quivers.dsl.let_expr_traversal import substitute_let_expr
from quivers.formulas._priors import (
    PriorInput,
    PriorSet,
    PriorSpec,
    PriorTarget,
    normalize_priors,
    resolve_priors,
)
from quivers.formulas._prior_emission import (
    build_prior_targets,
    ordinary_prior_step,
    apply_prior_plan,
    order_prior_dependencies,
)

# brms prior surface, default policy, and population-coordinate options.
from quivers.formulas._brms_priors import (
    DefaultPriorPolicy,
    brms_default_rule,
    sdgp_prior,
    with_flat_catalog,
)
from quivers.formulas._population_coordinates import (
    CENTERED_SUFFIX,
    PopulationCoordinates,
    emit_population,
    population_coordinates,
)

from quivers.formulas._autocor import append_autocor_steps
from quivers.formulas._gp import gp_block_steps
from quivers.formulas._monotonic_emission import append_monotonic_terms
from quivers.formulas._latent_effects import append_latent_terms
from quivers.formulas._predictor import (
    Predictor,
    PredictorOptions,
    plan_predictors,
    response_axis,
    transition_axis,
)
from quivers.formulas._term import (
    OutputAxis,
    Term,
    bind_inputs,
    declare_axes,
    declare_inputs,
    join_catalogs,
)
from quivers.formulas.formula import (
    FixedColumn,
    Formula,
    FormulaData,
    RandomTerm,
    _qvr_name,
)


#: Prior families whose default scalar scale is divided by the column RMS.
_AUTOSCALED_PRIOR_FAMILIES: frozenset[str] = frozenset(
    {"Normal", "Cauchy", "Laplace", "StudentT", "Logistic"}
)


def _autoscaled_prior_step(
    step: SampleStep,
    column: "np.ndarray",
) -> SampleStep:
    """Rescale a default coefficient prior for its predictor column.

    For a term ``beta * column``, a coefficient scale ``s`` induces contribution
    scale ``s * rms(column)``. If the prior family has a scalar scale
    argument, this function divides that argument by the column RMS. It leaves
    unsupported families, nonnumeric scales, and columns with nonpositive or
    nonfinite RMS unchanged. This follows the default autoscaling convention
    used by `rstanarm`.

    Parameters
    ----------
    step : SampleStep
        Parsed default coefficient prior.
    column : np.ndarray
        Predictor column entering the coefficient's contribution.

    Returns
    -------
    SampleStep
        The prior with only an eligible default scalar scale rescaled.
    """
    scale = _default_scalar_scale(step)
    if scale is None:
        return step
    rms = float(np.sqrt(np.mean(np.square(np.asarray(column, dtype=np.float64)))))
    if not np.isfinite(rms) or rms <= 0.0:
        return step
    return _with_scalar_scale(step, scale, scale.value / rms)


def _default_scalar_scale(step: SampleStep) -> DrawArgScalar | None:
    """Read the scalar scale an eligible default prior can be rescaled by.

    Parameters
    ----------
    step : SampleStep
        Parsed default coefficient prior.

    Returns
    -------
    DrawArgScalar or None
        The named scalar ``scale`` argument, or None when the family is not
        autoscaled, an argument is a nested distribution or a list, or the
        scale is not a scalar literal.
    """
    if step.morphism not in _AUTOSCALED_PRIOR_FAMILIES or any(
        isinstance(arg, DrawArgNamed)
        and isinstance(arg.value, DrawArgDist | DrawArgList)
        for arg in step.args or ()
    ):
        return None
    return next(
        (
            argument.value
            for argument in step.args or ()
            if isinstance(argument, DrawArgNamed)
            and argument.parameter == "scale"
            and isinstance(argument.value, DrawArgScalar)
        ),
        None,
    )


def _with_scalar_scale(
    step: SampleStep, scale: DrawArgScalar, value: float
) -> SampleStep:
    """Replace the scalar scale of a prior step.

    Parameters
    ----------
    step : SampleStep
        Parsed default coefficient prior.
    scale : DrawArgScalar
        Its scalar ``scale`` argument.
    value : float
        The scale the returned step carries.

    Returns
    -------
    SampleStep
        The prior with only its ``scale`` argument replaced.
    """
    return step.with_(
        args=tuple(
            argument.with_(value=scale.with_(value=value))
            if isinstance(argument, DrawArgNamed) and argument.parameter == "scale"
            else argument
            for argument in step.args or ()
        )
    )


def _scaled_prior_step(step: SampleStep, multiplier: float) -> SampleStep:
    """Apply one explicit contribution-scale policy to eligible default priors.

    Parameters
    ----------
    step : SampleStep
        Parsed default coefficient prior.
    multiplier : float
        Checked positive scale multiplier, computed from an evaluated column or
        declared for a symbolic column whose values are unavailable.

    Returns
    -------
    SampleStep
        Prior with an eligible scalar scale multiplied once.
    """
    scale = _default_scalar_scale(step)
    if scale is None:
        return step
    return _with_scalar_scale(step, scale, scale.value * multiplier)


def _named_prior_arguments(
    family: str, arguments: tuple[DrawArg, ...]
) -> tuple[DrawArg, ...]:
    """Bind prior shorthand positions through the existing constructor schemas.

    Parameters
    ----------
    family : str
        Authored family name, retaining semantic aliases.
    arguments : tuple[DrawArg, ...]
        Canonically parsed positional or named arguments.

    Returns
    -------
    tuple[DrawArg, ...]
        Canonical named arguments, including recursively bound constructors.

    Raises
    ------
    ValueError
        If a constructor is unknown, mixes argument styles, or fails its schema.
    """
    parameters = family_parameter_names(family)
    if parameters is None:
        raise ValueError(f"prior family {family!r} has no QVR parameter schema")
    if any(isinstance(arg, DrawArgNamed) for arg in arguments) and not all(
        isinstance(arg, DrawArgNamed) for arg in arguments
    ):
        raise ValueError(
            f"prior family {family!r} mixes positional and named arguments"
        )

    def nested(argument: DrawArg) -> DrawArg:
        """Retain each canonical argument while binding its nested constructors.

        Parameters
        ----------
        argument : DrawArg
            Parsed argument, list, or distribution value.

        Returns
        -------
        DrawArg
            The same canonical node with nested constructor names bound.

        Raises
        ------
        ValueError
            If a nested constructor does not match a complete schema.
        """
        if isinstance(argument, DrawArgDist):
            return argument.with_(
                args=_named_prior_arguments(argument.family, argument.args)
            )
        if isinstance(argument, DrawArgList):
            return argument.with_(items=tuple(nested(item) for item in argument.items))
        return argument

    named = []
    for position, argument in enumerate(arguments):
        if isinstance(argument, DrawArgNamed):
            name, value = argument.parameter, argument.value
        else:
            if position >= len(parameters):
                raise ValueError(
                    f"prior family {family!r} has no positional parameter {position}"
                )
            name, value = parameters[position], argument
        named.append(DrawArgNamed(parameter=name, value=nested(value)))
    try:
        bound = bind_family_arguments(
            family, tuple(named), parameters, allow_defaults=False
        )
    except TypeError as error:
        raise ValueError(str(error)) from error
    return tuple(DrawArgNamed(parameter=name, value=value) for name, value in bound)


def _auxiliary_draw(
    var: str,
    prior: str,
    *,
    lower: float | None = None,
    templates: dict[str, SampleStep] | None = None,
) -> SampleStep:
    """Parse one prior through QVR and retain its canonical argument AST.

    Parameters
    ----------
    var : str
        Actual sampled site binder.
    prior : str
        Authored distribution expression, including positional shorthand.
    lower : float or None
        Existing auxiliary-domain lower-bound policy.
    templates : dict[str, SampleStep] or None
        Optional preparation-local cache of immutable, unbound templates.

    Returns
    -------
    SampleStep
        The prior with its actual binder and any required domain restriction.

    Raises
    ------
    ValueError
        If the text is not one complete distribution call or violates the
        auxiliary lower-bound policy.
    """
    template = None if templates is None else templates.get(prior)
    if template is None:
        from quivers.dsl.parser import ParseError, parse

        expression = prior.strip().replace("\n", "\n    ")
        try:
            module = parse(
                "program prior : Input -> Output\n"
                f"    sample prior_value <- {expression}\n"
                "    return prior_value\n"
            )
        except ParseError as error:
            raise ValueError(
                f"invalid prior distribution {prior!r}: {error}"
            ) from error
        if len(module.statements) != 1:
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )
        program = module.statements[0]
        if not isinstance(program, ProgramDecl) or len(program.draws) != 1:
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )
        template = program.draws[0]
        if not isinstance(template, SampleStep) or template.vars != ("prior_value",):
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )
        if template.static_args or template.index is not None or template.options:
            raise ValueError(
                "an auxiliary prior must contain a single distribution call"
            )
        template = template.with_(
            args=_named_prior_arguments(template.morphism, template.args or ())
        )
        if templates is not None:
            templates[prior] = template
    step = template.with_(vars=(var,))
    return _prior_lower_bound(step, lower)


def _prior_lower_bound(step: SampleStep, lower: float | None) -> SampleStep:
    """Retain the existing auxiliary-domain restriction on a parsed prior.

    Parameters
    ----------
    step : SampleStep
        Bound prior, retaining its canonical arguments.
    lower : float or None
        Required auxiliary-domain lower bound.

    Returns
    -------
    SampleStep
        The same prior or its existing lower-bounded restriction.

    Raises
    ------
    ValueError
        If an explicit restriction does not ensure the required lower bound.
    """
    var = step.vars[0]
    if lower is not None:
        if step.morphism == "Restrict":
            cuts = [
                arg.value
                for arg in step.args or ()
                if isinstance(arg, DrawArgNamed) and arg.parameter == "low"
            ]
            if (
                len(cuts) != 1
                or not isinstance(cuts[0], DrawArgScalar)
                or cuts[0].value < lower
            ):
                raise ValueError(
                    f"prior for {var!r} must have a literal lower bound at least {lower}"
                )
        else:
            step = SampleStep(
                vars=(var,),
                morphism="Restrict",
                args=(
                    DrawArgNamed(
                        parameter="base",
                        value=DrawArgDist(family=step.morphism, args=step.args or ()),
                    ),
                    DrawArgNamed(parameter="low", value=DrawArgScalar(value=lower)),
                ),
            )
    return step


def _append_link(
    steps: list[ProgramStep], name: str, eta: LetExprNode, link: Link
) -> None:
    """Bind a parameter to its declared inverse link of a linear predictor.

    Parameters
    ----------
    steps : list[ProgramStep]
        Program body receiving the link's intermediates and the parameter.
    name : str
        Parameter being defined.
    eta : LetExprNode
        Linear predictor.
    link : Link
        Inverse link, emitted exactly as it declares itself.
    """
    for binding in link.intermediates:
        steps.append(
            _let(
                binding.name.format(name=name),
                _bind(binding.expr.format(eta="eta", name=name), eta=eta),
            )
        )
    steps.append(
        _let(name, _bind(link.inverse_expr.format(eta="eta", name=name), eta=eta))
    )


def _decode_module(module: Module, layouts: tuple[GroupBlockLayout, ...] = ()) -> dict:
    """Recover the structural part of a `Formula` from the
    emitted `Module`.

    Returns a dict carrying the fields the lens forward can deterministically
    produce from a `Formula`:

    * ``n_obs``: the ``Resp`` cardinality.
    * ``group_cardinalities``: ``{qvr_group_name: K}`` for each
      ``object G : K`` declaration preceding the program block.
    * ``fixed_qvr_names``: list of ``(qvr_name, is_intercept)`` in
      emission order, recovered from the latent ``intercept`` /
      ``beta_<qvr_name>`` declarations.
    * ``random_terms_qvr``: list of ``(qvr_group_name, slope_qvr_or_intercept)``
      in emission order, recovered from the
      ``sigma_<g>_<slope>`` + ``alpha_<g>``/``beta_<g>_<slope>`` pattern.
    * ``response_qvr_name``: the QVR-legal identifier of the
      observe step's target.
    * ``observe_family``: the family name on the observe step.

    The decoder is intentionally narrow: it knows the canonical
    emission shape of `FormulaToQVRModule` and recognises that
    shape only. It is not a general QVR parser.
    """
    sizes = finite_object_sizes(module)
    n_obs = sizes.get("Resp")
    group_cardinalities = {
        name: size
        for name, size in sizes.items()
        if name != "Resp" and not name.startswith("Formula")
    }
    program = next(
        (
            statement
            for statement in module.statements
            if isinstance(statement, ProgramDecl)
        ),
        None,
    )

    if program is None or n_obs is None:
        raise ValueError(
            "_decode_module: module is not a formula-emitted program "
            "(missing Resp object or program declaration)"
        )

    # Walk the program steps in emission order. The lens emits, per
    # the layout in `_build_module`, the following pattern:
    #
    #   * One SampleStep per fixed coefficient (`intercept` or
    #     `beta_<qvr_name>`).
    #   * For non-intercepts, a LetStep `<beta_name>_per_row =
    #     beta_name * <qvr_name>` immediately after the SampleStep.
    #   * One SampleStep `sigma_<g>_<slope>` per random-effect entry,
    #     followed by the plate draw (`alpha_<g>` or
    #     `beta_<g>_<slope>`) and a LetStep `<latent_var>_per_row`.
    #   * Aux-family SampleSteps (one per aux param).
    #   * `let eta = ...`, `let mu = ...`.
    #   * A final ObserveStep indexed by `Resp` bearing the response
    #     qvr_name and the observe-family.
    fixed_qvr_names: list[tuple[str, bool]] = []
    random_terms_qvr: list[tuple[str, str]] = []
    response_qvr_name: str = ""
    observe_family: str = ""

    seen_letnames: set[str] = set()
    for step in program.draws:
        if isinstance(step, LetStep):
            seen_letnames.add(step.name)
            continue
        if isinstance(step, ObserveStep):
            if step.index is not None:
                response_qvr_name = step.vars[0]
                observe_family = step.morphism
            continue
        if not isinstance(step, SampleStep):
            continue
        if len(step.vars) != 1:
            continue
        var = step.vars[0]
        # Fixed effects: `intercept` or `beta_<qvr_name>` with no
        # plate index. Random-effect sigmas are also unindexed, but
        # are followed by an indexed plate draw; distinguish by
        # checking the variable name pattern.
        if step.index is None:
            if var == "intercept":
                fixed_qvr_names.append(("", True))
            elif var.startswith("beta_") and not var.startswith("beta_"):
                # unreachable; placeholder to clarify next branch
                pass
            elif var.startswith("sigma_"):
                # random-effect scale; defer to the plate draw that
                # follows, which carries (group, slope) in its name.
                continue
            elif var.startswith("beta_"):
                qvr_name = var.removeprefix("beta_")
                fixed_qvr_names.append((qvr_name, False))
            else:
                # aux family parameter (e.g. `sigma`, `phi`, ...).
                continue
        else:
            # Indexed plate draw: this is the random-effect latent.
            # Centered emission:    `alpha_<g>` or `beta_<g>_<slope>`
            # Non-centered emission: `z_<g>_<slope>`, the unit-scale
            # standard-Normal that the let-expression scales by sigma.
            # `alpha_<g>` ↔ intercept slope; `beta_<g>_<slope>` ↔
            # named slope.
            if not isinstance(step.index, TypeName):
                continue
            qgroup = step.index.name
            if qgroup in {"FormulaCategoryLogit", "FormulaCutpoint"}:
                if var == "intercept":
                    fixed_qvr_names.append(("", True))
                elif var.startswith("beta_"):
                    fixed_qvr_names.append((var.removeprefix("beta_"), False))
                continue
            if qgroup.startswith("FormulaRandomCell_") and var.startswith("z_"):
                for source_group in sorted(group_cardinalities, key=len, reverse=True):
                    prefix = f"z_{source_group}_"
                    if not var.startswith(prefix):
                        continue
                    slope_qvr = var.removeprefix(prefix)
                    random_terms_qvr.append((source_group, slope_qvr))
                    break
                continue
            if var.startswith("alpha_"):
                random_terms_qvr.append((qgroup, "Intercept"))
            elif var.startswith("beta_"):
                # Strip `beta_<qgroup>_` to isolate the slope qvr_name.
                prefix = f"beta_{qgroup}_"
                if var.startswith(prefix):
                    slope_qvr = var.removeprefix(prefix)
                    random_terms_qvr.append((qgroup, slope_qvr))
            elif var.startswith("z_"):
                # Non-centered form: `z_<qgroup>_<slope>`.
                prefix = f"z_{qgroup}_"
                if var.startswith(prefix):
                    slope_qvr = var.removeprefix(prefix)
                    if slope_qvr == "Intercept":
                        random_terms_qvr.append((qgroup, "Intercept"))
                    else:
                        random_terms_qvr.append((qgroup, slope_qvr))

    raw_names = {
        step.vars[0]
        for step in program.draws
        if isinstance(step, SampleStep) and len(step.vars) == 1
    }
    for layout in layouts:
        if not set(layout.raw_sites) <= raw_names:
            continue
        for member in layout.members:
            term = (_qvr_name(layout.group), _qvr_name(member.slope))
            if (
                not member.prefix
                and member.coefficient in seen_letnames
                and term not in random_terms_qvr
            ):
                random_terms_qvr.append(term)

    return {
        "n_obs": n_obs,
        "group_cardinalities": group_cardinalities,
        "fixed_qvr_names": fixed_qvr_names,
        "random_terms_qvr": random_terms_qvr,
        "response_qvr_name": response_qvr_name,
        "observe_family": observe_family,
    }


class FormulaToQVRModule(dx.Lens[Formula, Module, FormulaData]):
    """Translate a `Formula` to a QVR `Module` AST.

    A typed `didactic.api.Lens` whose complement is a
    `FormulaData` carrier containing the fields of the source
    `Formula` that are *not* recoverable from the emitted
    `Module`. The per-row data arrays, the original (pre-
    `_qvr_name`) identifiers, the per-column ``term`` / ``name``
    presentation labels, and the original formula string travel in
    the complement; everything else (which columns there are, the
    intercept / random-term structure, the family choice) is decoded
    back out of the Module by `_decode_module`.

    Parameters
    ----------
    family : Family
        Response family from `quivers.formulas.families`.
    fixed_prior : str
        Default prior for fixed-effect coefficients, in the surface
        form ``"Family(arg, arg, ...)"``; numeric args become floats,
        identifier args stay as variable references in the emitted
        program.
    random_scale_prior : str
        Default prior for random-effect scale parameters.
    correlation_prior : str
        Default LKJ Cholesky prior for correlated coefficient blocks.
    group_covariances : Mapping[str, GroupCovariance] or None
        Known covariance between labeled levels of each grouping factor.
        Coefficient correlations retain their existing LKJ hierarchy.
    user_priors : PriorInput or None
        Exact-name overrides or typed role selectors. More specific selectors
        refine broader rules; ambiguous and unused rules raise an error.
    reparameterize : {"centered", "noncentered"}
        Conditional Gaussian coordinates for scalar, category-specific and
        component-valued group effects, and grouped threshold log slacks.
    binomial_trials : int or str
        Nonnegative common trial count or a per-row trial-count column.
    thresholds_by : str or None
        Grouping factor whose levels receive partially pooled ordinal
        cutpoint spacings.
    mixture_components : int
        Number of components for ``family="mixture"``; must be at least two.
    predictor_name : str or None
        Host-data name for an external differentiable predictor.
    distributional : Mapping[str, Formula] or None
        Auxiliary-parameter formulas on the same response rows. Each
        predictor receives its own coefficients, covariates, and group effects;
        a ``dpar_<parameter>_`` prefix distinguishes their names. Group
        coding must agree across predictors.

    fixed_parameters : Mapping[str, float] or None
        Auxiliary parameters bound to constants on their natural scale, without
        latent sites or priors. Cannot overlap distributional formulas.
    default_priors : {"quivers", "brms"}
        Default policy for sites no prior rule selects: the proper, autoscaled
        quivers defaults, or ``brms::default_prior`` (flat ``b``, Student-t
        intercept and scale priors scaled by the response).
    center : bool or None
        brms ``center``: sample the location intercept on the column-centered
        design and report it on the original scale. None centers under the
        brms policy and not under the quivers policy. Distributional
        predictors are centered under the brms policy or when True.
    decomp : {"none", "QR"}
        brms ``decomp``: sample the location coefficients in scaled QR
        coordinates and report them on the original scale.

    Raises
    ------
    ValueError
        If an option is out of range or does not apply to ``family``:
        ``thresholds_by`` outside the cumulative family,
        ``binomial_trials`` for a family without trials or outside the
        exact finite-trial range from zero through ``2**53``,
        ``mixture_components`` below two or outside the mixture
        family, or an unknown ``reparameterize`` value.

    Notes
    -----
    GetPut: `backward` ``(forward(f))`` ``=`` ``f`` for every
    `Formula` ``f``. PutGet holds on pairs ``(t, c)`` for
    which ``t`` is in the image of `forward` and ``c`` is the
    corresponding `FormulaData`.
    """

    def __init__(
        self,
        family: Family,
        *,
        fixed_prior: str = "Normal(0.0, 5.0)",
        random_scale_prior: str = "HalfNormal(1.0)",
        user_priors: PriorInput | None = None,
        reparameterize: Literal["centered", "noncentered"] = "noncentered",
        binomial_trials: int | str | None = None,
        thresholds_by: str | None = None,
        mixture_components: int = 2,
        predictor_name: str | None = None,
        distributional: Mapping[str, Formula] | None = None,
        predictor_graph: PredictorGraph | None = None,
        correlation_prior: str = "LKJCholesky(1.0)",
        group_covariances: Mapping[str, GroupCovariance] | None = None,
        fixed_parameters: Mapping[str, float] | None = None,
        default_priors: DefaultPriorPolicy = "quivers",
        center: bool | None = None,
        decomp: Literal["none", "QR"] = "none",
    ) -> None:
        # mixtures: the registered "mixture" family is mixture(gaussian, nmix=K).
        family = expand_mixture(family, mixture_components)
        self._mixture_targets: tuple[PriorTarget, ...] = ()
        self._family = family
        # brms default prior policy and population-coordinate options.
        if default_priors not in ("quivers", "brms"):
            raise ValueError("default_priors must be 'quivers' or 'brms'")
        if center is not None and type(center) is not bool:
            raise TypeError("center must be True, False, or None")
        if decomp not in ("none", "QR"):
            raise ValueError("decomp must be 'none' or 'QR'")
        coordinates_supported = (
            family.ordinal is None
            and family.response is None
            and family.mixture is None
        )
        if (center or decomp == "QR") and not coordinates_supported:
            raise ValueError(
                f"center=True and decomp='QR' are not supported for the "
                f"{family.name} family, whose intercepts are thresholds, "
                "category logits, or mixture components; use center=False"
            )
        self._default_priors: DefaultPriorPolicy = default_priors
        # Recorded rows of an mi() response, the rows brms's default scales
        # read (median and mad with na.rm = TRUE; fit-followups).
        self._recorded_rows: np.ndarray | None = None
        self._center = center
        self._decomp: Literal["none", "QR"] = decomp
        self._coordinates_supported = coordinates_supported
        self._group_covariances = dict(group_covariances or {})
        if any(
            not isinstance(value, GroupCovariance)
            for value in self._group_covariances.values()
        ):
            raise TypeError(
                "group_covariances values must be GroupCovariance declarations"
            )
        for aux in family.aux_params:
            # The native law checks its own constraint, not a narrower domain.
            if aux.domain.lower and not (
                aux.link.image is not None and aux.link.image.within(aux.domain)
            ):
                raise ValueError(
                    f"parameter {aux.name!r} of family {family.name!r} must "
                    f"{aux.domain.requirement()}, and its link {aux.link.name!r} "
                    "does not map into that range (Link.image)"
                )
        self._fixed_parameters = dict(fixed_parameters or {})
        self._correlation_prior = correlation_prior
        self._distributional = dict(distributional or {})
        self._predictor_graph = predictor_graph
        auxiliary_names = {aux.name for aux in family.aux_params}
        expression_roots = (
            set(predictor_graph.roots) - {"mu"}
            if predictor_graph is not None
            else set()
        )
        if expression_roots - auxiliary_names:
            raise ValueError(
                f"unknown distributional graph roots {sorted(expression_roots - auxiliary_names)}"
            )
        if expression_roots & (
            self._distributional.keys() | self._fixed_parameters.keys()
        ):
            raise ValueError(
                "a family parameter cannot have multiple predictor definitions"
            )
        unknown_fixed = {
            name
            for name in self._fixed_parameters.keys() - auxiliary_names
            if not is_scale_parameter(family, name)
        }
        if unknown_fixed:
            raise ValueError(
                f"unknown fixed auxiliary parameters {sorted(unknown_fixed)}"
            )
        conflicting_fixed = self._fixed_parameters.keys() & self._distributional.keys()
        if conflicting_fixed:
            raise ValueError(
                f"fixed parameters cannot have distributional formulas: {sorted(conflicting_fixed)}"
            )
        for name, value in self._fixed_parameters.items():
            # mixtures: each component validates its own fixed values.
            if family.mixture is not None:
                break
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"fixed parameter {name!r} must be a finite real number"
                )
            domain = next(
                (aux.domain for aux in family.aux_params if aux.name == name),
                Domain(),
            )
            if not domain.contains(value):
                raise ValueError(
                    f"fixed parameter {name!r} must {domain.requirement()}"
                )
        unknown = self._distributional.keys() - auxiliary_names
        # Category-specific locations mu<c> are checked against the labels.
        unknown = {
            name
            for name in unknown
            if not is_category_parameter(family, name)
            and not is_scale_parameter(family, name)
        }
        if unknown:
            raise ValueError(
                f"FormulaToQVRModule: unknown distributional parameters {sorted(unknown)}; "
                f"choices are {sorted(auxiliary_names)}"
            )
        self._fixed_prior = fixed_prior
        self._random_scale_prior = random_scale_prior
        self._set_prior_configuration(user_priors)
        self._emitted_groups: list[GroupBlockLaw] = []
        self._baseline_level_sites: tuple[LevelSite, ...] = ()
        # Level hyperparameters another program draws, and those drawn so far.
        self._supplied_sites: frozenset[str] = frozenset()
        self._drawn_sites: frozenset[str] = frozenset()
        self._group_plan: dict[tuple[str, str, str], GroupBlockLaw] = {}
        self._predictors: tuple[Predictor, ...] = ()
        self._term_laws: FamilyCatalog | None = None
        conflicting = (
            self._distributional.keys() | expression_roots
        ) & self._named_prior_inputs.keys()
        if conflicting:
            raise ValueError(
                f"priors on distributionally regressed parameters {sorted(conflicting)} must name their coefficients, such as 'dpar_sigma_intercept'"
            )
        if thresholds_by is not None and (
            family.ordinal is None
            or family.ordinal.kind not in {"cumulative", "hurdle_cumulative"}
        ):
            raise ValueError(
                "FormulaToQVRModule: thresholds_by is only valid for the "
                "cumulative family or hurdle_cumulative family"
            )
        from quivers.formulas._response import resolve_trials

        binomial_trials = resolve_trials(family, binomial_trials)
        if family.trials_parameter is None and binomial_trials != 1:
            raise ValueError(
                "FormulaToQVRModule: binomial_trials is only valid for the "
                "families with a trial-count parameter"
            )
        if binomial_trials is not None and (
            not isinstance(binomial_trials, (int, str))
            or isinstance(binomial_trials, bool)
        ):
            raise ValueError(
                "FormulaToQVRModule: binomial_trials must be a nonnegative integer "
                "or a data-column name"
            )
        if isinstance(binomial_trials, int) and binomial_trials < 0:
            raise ValueError("FormulaToQVRModule: binomial_trials must be nonnegative")
        if isinstance(binomial_trials, int) and binomial_trials > _MAX_EXACT_COUNT:
            raise ValueError(
                "FormulaToQVRModule: binomial_trials must be at most 2**53"
            )
        if (
            not isinstance(mixture_components, int)
            or isinstance(mixture_components, bool)
            or mixture_components < 2
        ):
            raise ValueError(
                "FormulaToQVRModule: mixture_components must be an integer "
                "of at least two"
            )
        if family.mixture is None and mixture_components != 2:
            raise ValueError(
                "FormulaToQVRModule: mixture_components is only valid for the "
                "mixture family"
            )
        self._binomial_trials = binomial_trials
        self._thresholds_by = thresholds_by
        self._mixture_components = mixture_components
        if family.mixture is not None:
            validate_mixture_options(
                family,
                self._distributional,
                self._fixed_parameters,
                predictor_name=predictor_name,
                nonlinear=predictor_graph is not None and bool(predictor_graph.leaves),
                thresholds_by=thresholds_by,
            )
        self._predictor_name = (
            _qvr_name(predictor_name) if predictor_name is not None else None
        )
        if reparameterize not in ("centered", "noncentered"):
            raise ValueError(
                f"FormulaToQVRModule: reparameterize must be 'centered' "
                f"or 'noncentered', got {reparameterize!r}"
            )
        self._reparameterize: Literal["centered", "noncentered"] = reparameterize

    def _set_prior_configuration(self, user_priors: PriorInput | None) -> None:
        """Initialize one canonical prior plan for this response emission.

        Parameters
        ----------
        user_priors : PriorInput or None
            Existing exact-name or typed prior rules.
        """
        self._prior_set = normalize_priors(user_priors)
        self._named_prior_inputs = {
            rule.site: rule.distribution
            for rule in self._prior_set.rules
            if rule.site is not None
            and rule.selector == {"site": rule.site}
            and isinstance(rule.distribution, str)
        }
        self._initial_user_priors = {
            name: text
            for name, text in self._named_prior_inputs.items()
            if "(" not in text.partition("(")[2] and "[" not in text
        }
        self._user_priors: Mapping[str, str] = dict(self._initial_user_priors)
        self._prior_steps: dict[str, SampleStep] = {}
        self._prior_templates: dict[str, SampleStep] = {}
        self._prior_targets: tuple[PriorTarget, ...] = ()
        self._prior_module: Module | None = None
        self._hierarchy_sites: frozenset[str] = frozenset()

    @property
    def group_covariances(self) -> dict[str, GroupCovariance]:
        """Read the known level covariances of this emission's grouping factors.

        Returns
        -------
        dict[str, GroupCovariance]
            Covariances given directly and those read from ``gr(cov = M)``.
        """
        return dict(self._group_covariances)

    @property
    def group_layouts(self) -> tuple[GroupBlockLayout, ...]:
        """Read the raw-coordinate layouts emitted by the latest forward pass.

        Returns
        -------
        tuple[GroupBlockLayout, ...]
            Immutable group blocks shared with retained prediction plans.
        """
        return tuple(law.layout for law in self._emitted_groups)

    @property
    def level_sites(self) -> tuple[LevelSite, ...]:
        """Return the sites of the latest forward pass indexed by levels.

        Returns
        -------
        tuple[LevelSite, ...]
            Every level-indexed site once, of the group blocks and of a
            grouped baseline hazard.
        """
        sites = {
            site.site: site for law in self._emitted_groups for site in law.level_sites
        }
        return (*sites.values(), *self._baseline_level_sites)

    @property
    def predictors(self) -> tuple[Predictor, ...]:
        """Return the linear predictors of the latest forward pass.

        Returns
        -------
        tuple[Predictor, ...]
            The location, the distributional parameters, and the nonlinear
            parameters. Empty for a mixture, whose components have their own.
        """
        return self._predictors

    @property
    def group_laws(self) -> tuple[GroupBlockLaw, ...]:
        """Return the law of each group block of the latest forward pass.

        Returns
        -------
        tuple[GroupBlockLaw, ...]
            Laws in emission order.
        """
        return tuple(self._emitted_groups)

    @property
    def term_laws(self) -> FamilyCatalog | None:
        """Return the laws the terms of the latest forward pass draw from.

        Returns
        -------
        FamilyCatalog or None
            Compilation-local laws, or None when every law is built in.
        """
        return self._term_laws

    def families(self, response: FamilyCatalog | None = None) -> FamilyCatalog | None:
        """Return every compilation-local law of the latest forward pass.

        Parameters
        ----------
        response : FamilyCatalog or None
            Laws of the response on the rows in use; None for those of the
            fitted family.

        Returns
        -------
        FamilyCatalog or None
            The response laws joined with those of the terms.
        """
        if response is None and self._family.custom is not None:
            response = self._family.custom.catalog
        if response is None or self._term_laws is None:
            return response if response is not None else self._term_laws
        return response.with_catalog(self._term_laws)

    def terms(self, formula: Formula) -> tuple[Term, ...]:
        """List the terms of every linear predictor.

        Parameters
        ----------
        formula : Formula
            Location predictor and response.

        Returns
        -------
        tuple[Term, ...]
            Terms in predictor order.
        """
        return tuple(
            term for predictor in self._plan(formula, {}) for term in predictor.terms
        )

    def _observe_argument(
        self, argument: str | float, scale: Mapping[str, LetExprNode]
    ) -> str | float:
        """Resolve one declared native constructor argument.

        Parameters
        ----------
        argument : str or float
            A declared `Family.observe_args` entry.
        scale : Mapping[str, LetExprNode]
            Names standing in for formula parameters under known errors.

        Returns
        -------
        str or float
            The bound program name or literal constant.
        """
        if argument == "trials":
            if isinstance(self._binomial_trials, str):
                return _qvr_name(self._binomial_trials)
            assert self._binomial_trials is not None
            return float(self._binomial_trials)
        if isinstance(argument, str) and argument in scale:
            replacement = scale[argument]
            assert isinstance(replacement, LetExprVar)
            return replacement.name
        return argument

    def _append_known_error_scale(
        self, formula: Formula, program_steps: list[ProgramStep]
    ) -> None:
        """Bind ``formula_sigma``, the response scale under known errors.

        With ``se(e)`` (brms ``sigma = FALSE``) the scale is the known error;
        with ``se(e, sigma=True)`` it is ``sqrt(e**2 + sigma**2)``, formed
        relative to the larger term so neither square overflows.

        Parameters
        ----------
        formula : Formula
            Formula whose response roles include known standard errors.
        program_steps : list[ProgramStep]
            Program body receiving the scale bindings.

        Raises
        ------
        ValueError
            If the formula has no known standard errors.
        """
        error = formula.modifiers.standard_error
        residual = self._family.error_scale
        if error is None or residual is None:
            raise ValueError(
                "a known-error scale requires an se() response term and a family "
                "declaring Family.error_scale"
            )
        if not error.sigma:
            program_steps.append(_let("formula_sigma", _var("formula_standard_error")))
        else:
            row = _var("response")
            se = _index("formula_standard_error", row)
            sigma = (
                _index(residual, row)
                if residual in self._distributional
                or self._predictor_graph is not None
                and residual in self._predictor_graph.roots
                else _var(residual)
            )
            program_steps.append(
                _let(
                    "formula_error_reference",
                    _factor((("response", "Resp"),), body=_call("max", se, sigma)),
                )
            )
            scale = _index("formula_error_reference", row)
            combined = _mul(
                scale,
                _call(
                    "sqrt",
                    _add(
                        _call("pow", _div(se, scale), LetExprLiteral(value=2.0)),
                        _call("pow", _div(sigma, scale), LetExprLiteral(value=2.0)),
                    ),
                ),
            )
            program_steps.append(
                _let("formula_sigma", _factor((("response", "Resp"),), body=combined))
            )

    def forward(self, formula: Formula, /) -> tuple[Module, FormulaData]:
        """Compile one formula through the shared response emitter.

        Parameters
        ----------
        formula : Formula
            Evaluated response and design metadata.

        Returns
        -------
        tuple[Module, FormulaData]
            Existing QVR program and its retained formula complement.
        """
        return self._forward(formula)

    def _forward(
        self,
        formula: Formula,
        /,
        *,
        supplied_groups: Mapping[tuple[str, str, str], LetExprNode] | None = None,
        supplied_sites: Collection[str] = (),
        check: bool = True,
    ) -> tuple[Module, FormulaData]:
        """Compile a formula to a QVR module and its complement.

        Parameters
        ----------
        formula : Formula
            The parsed formula with its design data.
        supplied_groups : Mapping or None
            Group coefficients another program emits, gathered at the
            response rows' levels.
        supplied_sites : Collection[str]
            Level hyperparameters of grouping factors that another program
            draws and this program reads.
        check : bool
            Whether to check the emitted module.

        Returns
        -------
        tuple[Module, FormulaData]
            The emitted QVR module AST, and the fields of ``formula``
            the module does not encode.

        Raises
        ------
        ValueError
            If the response is incompatible with the family (for
            instance non-integer or non-contiguous category labels
            for a categorical or cumulative family), or a prior does
            not parse as ``"Family(arg, ...)"``.
        """
        self._user_priors = dict(self._initial_user_priors)
        self._prior_steps = {}
        self._prior_templates = {}
        self._prior_targets = ()
        self._prior_module = None
        self._hierarchy_sites = frozenset()
        # A typed prior explicitly activates an auxiliary whose default is fixed.
        for auxiliary in self._family.aux_params:
            if (
                auxiliary.fixed_default is None
                or auxiliary.name in self._fixed_parameters
                or auxiliary.name in self._distributional
            ):
                continue
            target = PriorTarget(
                site=auxiliary.name, class_=auxiliary.name, dpar=auxiliary.name
            )
            if any(
                all(
                    getattr(target, key) == value
                    for key, value in rule.selector.items()
                )
                for rule in self._prior_set.rules
            ):
                self._user_priors = dict(self._user_priors) | {
                    auxiliary.name: auxiliary.prior
                }
        self._emitted_groups = []
        self._supplied_sites = frozenset(supplied_sites)
        self._drawn_sites = self._supplied_sites
        known_groups = {
            group
            for _, _, predictor in self._predictor_items(formula)
            for group in predictor.group_levels
        }
        unknown_covariances = self._group_covariances.keys() - known_groups
        if unknown_covariances:
            raise ValueError(
                f"group_covariances names unknown grouping factors {sorted(unknown_covariances)!r}"
            )
        for _, _, predictor in self._predictor_items(formula):
            for group, covariance in self._group_covariances.items():
                if group in predictor.group_levels:
                    covariance.subset(predictor.group_levels[group])
        if self._family.vector_response:
            if not formula.response_layout.event_shape:
                raise ValueError(
                    f"{self._family.name} requires an explicit cbind vector response"
                )
            if self._family.trials_parameter is not None:
                from quivers.formulas._response import count_totals

                totals = count_totals(formula)
                if isinstance(self._binomial_trials, int) and not bool(
                    (totals == self._binomial_trials).all()
                ):
                    raise ValueError(
                        "vector response row totals disagree with supplied trials"
                    )
            else:
                from quivers.formulas._response import validate_simplex

                validate_simplex(formula)
        if formula.response_values.shape[1:] != formula.response_layout.event_shape:
            raise ValueError("response values do not match their declared event layout")
        if formula.response_layout.category_levels:
            levels = self._family.coded_levels
            if not self._family.category_response and levels is None:
                raise ValueError(
                    f"{self._family.name} requires a numeric response, not factor category codes"
                )
            if (
                not self._family.category_response
                and len(formula.response_layout.category_levels) != levels
            ):
                count = "two" if levels == 2 else levels
                raise ValueError(
                    f"{self._family.name} requires exactly {count} factor response levels"
                )
        if formula.response_layout.event_shape and not self._family.vector_response:
            raise ValueError(f"{self._family.name} requires a scalar response event")
        category_columns = tuple(
            col for col in formula.fixed_columns if col.category_specific
        )
        category_terms = tuple(
            term
            for term in formula.random_terms
            if term.column is not None and term.column.category_specific
        )
        if category_columns or category_terms:
            if self._family.ordinal is None:
                raise ValueError("cs terms require an ordinal response family")
            if (
                self._family.ordinal.kind in {"cumulative", "hurdle_cumulative"}
                and any(column.name == "cs(1)" for column in category_columns)
                and any(column.is_intercept for column in formula.fixed_columns)
            ):
                raise ValueError(
                    "fixed cs(1) duplicates the freely estimated cumulative "
                    "threshold profile and proportional intercept; use the ordinary "
                    "intercept and thresholds instead, or use category-specific "
                    "predictors that vary across rows"
                )
            if self._thresholds_by is not None:
                duplicate = next(
                    (
                        term
                        for term in category_terms
                        if term.group == self._thresholds_by
                        and term.column is not None
                        and term.column.name == "cs(1)"
                    ),
                    None,
                )
                if duplicate is not None:
                    raise ValueError(
                        f"category-specific intercept {duplicate.slope!r} for "
                        f"{duplicate.group!r} duplicates the full thresholds_by "
                        "profile; use the threshold profile and its ordinary "
                        "group intercept instead"
                    )
            if (
                self._family.ordinal.kind == "hurdle_cumulative"
                and self._category_count(formula, family=self._family.name) < 3
            ):
                raise ValueError(
                    "cs terms require at least two positive ordinal categories"
                )
        for name, predictor in self._distributional.items():
            # An ordinal mixture component checks its own location formula.
            if (
                self._family.mixture is not None
                and self._family.response_ordinal() is not None
                and re.fullmatch(r"mu\d+", name)
            ):
                continue
            if any(col.category_specific for col in predictor.fixed_columns) or any(
                term.column is not None and term.column.category_specific
                for term in predictor.random_terms
            ):
                raise ValueError(f"cs terms are only valid for mu, not {name!r}")
        interval = self._family.response_interval
        if interval is not None:
            response = np.asarray(formula.response_values, dtype=np.float64)
            # A missing response is NaN and is not an observed value.
            observed = response[~np.isnan(response)]
            if np.isinf(observed).any() or not all(
                interval.contains(value) for value in observed.tolist()
            ):
                raise ValueError(
                    f"{self._family.name} response values must be finite and "
                    f"{interval.requirement()}"
                )
            for aux in self._family.aux_params:
                shift = self._fixed_parameters.get(aux.name)
                if (
                    aux.below_response
                    and shift is not None
                    and (observed <= shift).any()
                ):
                    raise ValueError(
                        f"{self._family.name} responses must exceed fixed {aux.name}"
                    )
        if formula.modifiers.decision is not None and (
            self._family.decision_event is None
        ):
            raise ValueError(
                "a decision response modifier requires a family declaring "
                "Family.decision_event, such as wiener"
            )
        if self._family.decision_event is not None:
            decision = formula.modifiers.decision
            response = np.asarray(formula.response_values)
            if decision is None:
                raise ValueError(
                    f"{self._family.name} requires a decision column: "
                    "response | dec(column) or decision=column"
                )
            if (
                decision.data.shape != response.shape
                or decision.data.dtype.kind not in "iu"
                or not np.isin(decision.data, [0, 1]).all()
            ):
                raise ValueError(
                    "decisions must be checked integer 0/1 values aligned with response rows"
                )
            aliases = {"formula_response", "formula_decision"}
            if aliases & {
                _qvr_name(formula.response_name),
                self._predictor_name,
                *(column.qvr_name for column in formula.fixed_columns),
            }:
                raise ValueError(
                    "formula_response and formula_decision are reserved response input aliases"
                )
        if formula.modifiers.standard_error is not None:
            error = formula.modifiers.standard_error
            residual = self._family.error_scale
            if residual is None:
                admitted = sorted(
                    name
                    for name, entry in families.items()
                    if entry.error_scale is not None
                )
                raise ValueError(
                    f"response term se() is not supported for family "
                    f"{self._family.name!r}, which declares no error scale "
                    f"(Family.error_scale); families admitting se() are {admitted}"
                )
            if not error.sigma and (
                residual in self._distributional
                or residual in self._fixed_parameters
                or residual in self._user_priors
            ):
                raise ValueError(
                    f"se() without sigma=True excludes a separate residual {residual}; remove its formula, fixed value, or prior, or write se(error, sigma=True)"
                )
        check_recorded_event_family(
            self._family,
            censoring=formula.modifiers.censoring,
            truncation=formula.modifiers.truncation,
        )
        if (
            formula.modifiers.trials is not None
            and self._family.trials_parameter is None
        ):
            raise ValueError(
                "trials requires a response family with finite trial counts"
            )
        role_aliases = {
            column.qvr_name
            for role, column in formula.modifiers.columns()
            if role not in {"weights", "decision"}
        }
        if formula.modifiers.standard_error is not None:
            role_aliases.update({"formula_sigma", "formula_error_reference"})
        collisions = role_aliases & {
            _qvr_name(formula.response_name),
            self._predictor_name,
            *(column.qvr_name for column in formula.fixed_columns),
        }
        if collisions:
            raise ValueError(
                f"{sorted(collisions)} are reserved response input aliases"
            )
        if formula.modifiers.weights is not None:
            weights = formula.modifiers.weights.data
            if weights.shape != formula.response_values.shape[:1]:
                raise ValueError("weights must align with every evaluated response row")
            if (
                weights.dtype.kind not in "iuf"
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
            ):
                raise ValueError("weights must be finite nonnegative real values")
            if (
                _qvr_name(formula.response_name) == "formula_weights"
                or self._predictor_name == "formula_weights"
                or any(
                    column.qvr_name == "formula_weights"
                    for column in formula.fixed_columns
                )
            ):
                raise ValueError(
                    "formula_weights is reserved for response likelihood weights"
                )
        group_codings: dict[str, tuple[str, tuple[str, ...], tuple[int, ...]]] = {}
        for name, _, predictor in self._predictor_items(formula):
            if predictor.response_values.shape != formula.response_values.shape:
                raise ValueError(
                    f"distributional parameter {name!r} has a different row count"
                )
            reserved_offsets = {
                f"formula_offset_{index}" for index, _ in enumerate(predictor.offsets)
            }
            collisions = reserved_offsets & {
                column.qvr_name for column in predictor.fixed_columns
            }
            if collisions:
                raise ValueError(
                    f"offset input names are reserved: {sorted(collisions)}"
                )
            for group, levels in predictor.group_levels.items():
                normalized = _qvr_name(group)
                coding = (group, levels, predictor.group_indices[group])
                existing = group_codings.setdefault(normalized, coding)
                if coding != existing:
                    raise ValueError(
                        f"distributional parameter {name!r} has incompatible coding for {group!r}"
                    )
        if self._distributional and any(
            col.qvr_name.startswith("dpar_") for col in formula.fixed_columns
        ):
            raise ValueError(
                "the 'dpar_' prefix is reserved for distributional coefficients and inputs"
            )
        module = self._build_module(formula, supplied_groups=supplied_groups)
        targets = self._catalogue_prior_targets(formula, module)
        # brms default policy: one rule for every site no user rule selects.
        defaults = self._brms_defaults(formula, targets)
        if self._prior_set.rules or defaults:
            missing = self._named_prior_inputs.keys() - {
                target.site for target in targets
            }
            if missing:
                raise ValueError(
                    f"unused prior overrides {sorted(missing)}; available parameters are {sorted(target.site for target in targets)}"
                )
            selected = {**defaults, **resolve_priors(self._prior_set, targets)}
            ordinary = {
                site: rule
                for site, rule in selected.items()
                if isinstance(rule.distribution, str)
            }
            by_site = {target.site: target for target in targets}
            steps = {
                site: ordinary_prior_step(by_site[site], rule, draw=self._parse_prior)
                for site, rule in ordinary.items()
            }
            resolved = {
                site: draw_call_to_source(
                    step.morphism, step.args, static_args=step.static_args
                )
                for site, step in steps.items()
            }
            current = {
                step.vars[0]: step
                for statement in module.statements
                if isinstance(statement, ProgramDecl)
                for step in statement.draws
                if isinstance(step, SampleStep)
            }
            legacy_only = all(
                rule.site is not None
                and rule.selector == {"site": rule.site}
                and rule.site in self._initial_user_priors
                for rule in self._prior_set.rules
            )
            same = all(
                site in current
                and (
                    (step.morphism, step.args)
                    == (current[site].morphism, current[site].args)
                    or legacy_only
                    and current[site].morphism == "TruncatedNormal"
                    and step.morphism == "Normal"
                )
                for site, step in steps.items()
            )
            if not same:
                self._user_priors = resolved
                self._prior_steps = steps
                self._emitted_groups = []
                self._drawn_sites = self._supplied_sites
                module = self._build_module(formula, supplied_groups=supplied_groups)
                targets = self._catalogue_prior_targets(formula, module)
            special = {
                site: rule
                for site, rule in selected.items()
                if not isinstance(rule.distribution, str)
            }
            if special:
                before = {
                    step.vars[0]
                    for statement in module.statements
                    if isinstance(statement, ProgramDecl)
                    for step in statement.draws
                    if isinstance(step, SampleStep)
                }
                residual = self._family.residual_scale
                scalar_residual = residual is not None and (
                    any(
                        target.site == residual.name and not target.axes
                        for target in targets
                    )
                    or residual.name in self._fixed_parameters
                )
                module = apply_prior_plan(
                    module,
                    targets,
                    special,
                    draw=self._parse_prior,
                    training_rows=int(formula.response_values.shape[0]),
                    sigma=residual.name
                    if residual is not None and scalar_residual
                    else None,
                )
                after = {
                    step.vars[0]
                    for statement in module.statements
                    if isinstance(statement, ProgramDecl)
                    for step in statement.draws
                    if isinstance(step, SampleStep)
                }
                self._hierarchy_sites = frozenset(after - before)
            else:
                module = order_prior_dependencies(module)
        self._prior_targets = targets
        self._prior_module = module
        if check and (
            self._predictor_graph is not None or self._prior_set.rules or defaults
        ):
            from quivers.dsl.qiec_lowering import lower_qvr_to_qiec

            lower_qvr_to_qiec(
                module,
                # Improper flat priors are compilation-local native laws.
                families=with_flat_catalog(self.families(), module),
            )
        complement = FormulaData(
            formula=formula.formula,
            response_name=formula.response_name,
            response_values=formula.response_values,
            response_layout=formula.response_layout,
            fixed_column_names={
                col.qvr_name: (col.term, col.name) for col in formula.fixed_columns
            },
            fixed_column_data={
                col.qvr_name: col.data
                for col in formula.fixed_columns
                if isinstance(col, FixedColumn) and not col.is_intercept
            },
            symbolic_columns={
                col.qvr_name: col
                for col in formula.fixed_columns
                if isinstance(col, SymbolicColumn)
            },
            group_original_names={_qvr_name(g): g for g in formula.group_levels.keys()},
            group_levels=dict(formula.group_levels),
            group_indices=dict(formula.group_indices),
            group_laws=self.group_laws,
            level_sites=self.level_sites,
            prior_targets=self.prior_targets,
            random_terms=formula.random_terms,
            fixed_column_order=tuple(col.qvr_name for col in formula.fixed_columns),
            design_terms=formula.design_terms,
            category_specific_columns=tuple(
                col.qvr_name for col in formula.fixed_columns if col.category_specific
            ),
            offsets=formula.offsets,
            modifiers=formula.modifiers,
            monotonic_terms=formula.monotonic_terms,
            grouping=formula.grouping,
        )
        return module, complement

    @property
    def prior_targets(self) -> tuple[PriorTarget, ...]:
        """Read immutable role and axis metadata from the latest emitted model."""
        return self._prior_targets

    @property
    def prior_set(self) -> PriorSet:
        """Read the canonical authored prior rules retained for replay."""
        return self._prior_set

    def _brms_defaults(
        self, formula: Formula, targets: tuple[PriorTarget, ...]
    ) -> dict[str, PriorSpec]:
        """Return the brms default rule of every target under the brms policy.

        Parameters
        ----------
        formula : Formula
            Evaluated location predictor and responses.
        targets : tuple[PriorTarget, ...]
            Overrideable sites of the emitted module.

        Returns
        -------
        dict[str, PriorSpec]
            Site rules; empty under the quivers policy.
        """
        if self._default_priors != "brms":
            return {}
        response = (
            None
            if formula.response_layout.event_shape
            or formula.response_layout.category_levels
            else np.asarray(formula.response_values, dtype=np.float64)
        )
        if response is not None and self._recorded_rows is not None:
            response = response[self._recorded_rows]
        offset_means = {
            name: float(np.mean(sum(np.asarray(o.data) for o in predictor.offsets)))
            for name, _, predictor in self._predictor_items(formula)
            if predictor.offsets
        }
        centered = frozenset(
            target.site
            for target in targets
            if target.site.endswith("intercept" + CENTERED_SUFFIX)
        )
        rules = {}
        for target in targets:
            rule = brms_default_rule(
                target,
                family=self._family,
                response=response,
                offset_means=offset_means,
                centered=centered,
            )
            if rule is not None:
                rules[target.site] = rule
        return rules

    def _catalogue_prior_targets(
        self, formula: Formula, module: Module
    ) -> tuple[PriorTarget, ...]:
        # mixtures: the composed component catalogue.
        if self._family.mixture is not None:
            return self._mixture_targets
        value = self._threshold_location_formula(formula)
        targets = build_prior_targets(
            module,
            value,
            self._family,
            self._predictors,
            self.group_laws,
            draw=self._parse_prior,
            thresholds_by=self._thresholds_by,
        )
        # brms center = FALSE puts the location intercept in class b.
        if self._center is False:
            targets = tuple(
                target.with_(class_="b")
                if target.class_ == "Intercept"
                and target.coef == "Intercept"
                and target.dpar == "mu"
                else target
                for target in targets
            )
        return targets

    def get_prior(self, formula: Formula, /) -> dict[str, str]:
        """Inspect the effective priors without compiling or fitting a model.

        Parameters
        ----------
        formula : Formula
            Evaluated response and predictor columns.

        Returns
        -------
        dict[str, str]
            Overrideable parameter names and their canonical distribution
            calls, including the actual scales after column autoscaling.
            Sampled shrinkage includes hyperprior calls and conditional scale
            references; retain ``prior_set`` to reconstruct that specification.
            Conditional group draws and standard-normal innovations are
            excluded: their laws define the group parameterization.

        Raises
        ------
        ValueError
            If a prior override does not name an available parameter.
        """
        module, _ = self.forward(formula)
        return self._prior_calls(formula, module)

    def _prior_calls(self, formula: Formula, module: Module) -> dict[str, str]:
        """Collect overrideable priors from the same AST used for fitting.

        Parameters
        ----------
        formula : Formula
            Evaluated predictor metadata, combined with auxiliary predictors.
        module : Module
            Emitted native model declarations and program steps.

        Returns
        -------
        dict[str, str]
            Overrideable site names and effective canonical distribution calls.
        """
        targets = (
            self._prior_targets
            if self._prior_module is module
            else self._catalogue_prior_targets(formula, module)
        )
        names = {target.site for target in targets} | self._hierarchy_sites
        calls: dict[str, str] = {}
        for statement in module.statements:
            if not isinstance(statement, ProgramDecl):
                continue
            for step in statement.draws:
                if not isinstance(step, SampleStep):
                    continue
                name = step.vars[0]
                if name in names:
                    calls[name] = draw_call_to_source(
                        step.morphism, step.args, static_args=step.static_args
                    )
        return calls

    def backward(self, target: Module, complement: FormulaData, /) -> Formula:
        """Recover a formula from a QVR module and its complement.

        The column, intercept, and random-term structure is decoded
        from ``target``; the data arrays and presentation labels come
        from ``complement``.

        Parameters
        ----------
        target : Module
            A QVR module in the image of `forward`.
        complement : FormulaData
            The complement `forward` returned with that module.

        Returns
        -------
        Formula
            The reconstructed formula.
        """
        decoded = _decode_module(target, complement.group_layouts)
        fixed_column_names = complement.fixed_column_names
        fixed_column_data = complement.fixed_column_data

        fixed_columns: list[FixedColumn | SymbolicColumn] = []
        n_obs = decoded["n_obs"]
        ones = np.ones(n_obs, dtype=float)
        for qvr_name, is_intercept in decoded["fixed_qvr_names"]:
            if is_intercept:
                term, name = fixed_column_names.get(
                    "intercept", ("Intercept", "Intercept")
                )
                fixed_columns.append(
                    FixedColumn(
                        term=term,
                        name=name,
                        qvr_name="intercept",
                        is_intercept=True,
                        data=ones,
                    )
                )
            elif qvr_name in complement.symbolic_columns:
                fixed_columns.append(complement.symbolic_columns[qvr_name])
            else:
                term, name = fixed_column_names.get(qvr_name, (qvr_name, qvr_name))
                fixed_columns.append(
                    FixedColumn(
                        term=term,
                        name=name,
                        qvr_name=qvr_name,
                        is_intercept=False,
                        category_specific=qvr_name
                        in complement.category_specific_columns,
                        data=fixed_column_data[qvr_name],
                    )
                )

        if complement.fixed_column_order:
            by_name = {column.qvr_name: column for column in fixed_columns}
            fixed_columns = [
                by_name.pop(name)
                for name in complement.fixed_column_order
                if name in by_name
            ]
            fixed_columns.extend(by_name.values())

        random_terms: list[RandomTerm] = []
        for qgroup, slope_qvr in decoded["random_terms_qvr"]:
            group = (
                complement.group_original_names[qgroup]
                if qgroup in complement.group_original_names
                else qgroup
            )
            if slope_qvr == "Intercept":
                random_terms.append(RandomTerm(slope="Intercept", group=group))
            else:
                # Recover the original slope name by inverting via the
                # presentation map when available; otherwise pass the
                # qvr-name through.
                term_name_pairs = {
                    qvr: (term, name)
                    for qvr, (term, name) in fixed_column_names.items()
                }
                _, original_slope = term_name_pairs.get(
                    slope_qvr, (slope_qvr, slope_qvr)
                )
                random_terms.append(RandomTerm(slope=original_slope, group=group))

        original_terms = {
            (_qvr_name(term.group), _qvr_name(term.slope)): term
            for term in complement.random_terms
        }
        random_terms = [
            original_terms.get((_qvr_name(term.group), _qvr_name(term.slope)), term)
            for term in random_terms
        ]
        by_term = {
            (_qvr_name(term.group), _qvr_name(term.slope)): term
            for term in random_terms
        }
        random_terms = [
            by_term.pop(key) for key in original_terms if key in by_term
        ] + list(by_term.values())
        if self._thresholds_by is not None:
            random_terms = [
                term
                for term in random_terms
                if not (
                    term.group == self._thresholds_by
                    and term.slope == "Intercept"
                    and (_qvr_name(term.group), "Intercept") not in original_terms
                )
            ]
        return Formula(
            formula=complement.formula,
            response_name=complement.response_name,
            fixed_columns=tuple(fixed_columns),
            design_terms=complement.design_terms,
            monotonic_terms=complement.monotonic_terms,
            grouping=complement.grouping,
            random_terms=tuple(random_terms),
            offsets=complement.offsets,
            modifiers=complement.modifiers,
            response_values=complement.response_values,
            response_layout=complement.response_layout,
            group_levels=complement.group_levels,
            group_indices=complement.group_indices,
        )

    def symbolic_column_expressions(self, formula: Formula) -> dict[str, LetExprNode]:
        """Read declared latent-dependent inputs using the ordinary predictor layout.

        Parameters
        ----------
        formula : Formula
            Primary predictor, together with configured auxiliary and graph leaves.

        Returns
        -------
        dict[str, LetExprNode]
            Actual local input names and canonical expressions. Fixed and random
            uses of one column retain their existing, separately declared aliases.

        Raises
        ------
        ValueError
            If two uses assign different expressions to the same local input.
        """
        expressions: dict[str, LetExprNode] = {}
        for name, column in self._symbolic_columns(formula):
            previous = expressions.setdefault(name, column.expression)
            if previous != column.expression:
                raise ValueError(f"symbolic input {name!r} has conflicting expressions")
        return expressions

    def _symbolic_columns(
        self, formula: Formula
    ) -> tuple[tuple[str, SymbolicColumn], ...]:
        """Resolve symbolic inputs once through the common predictor enumeration.

        Parameters
        ----------
        formula : Formula
            Primary predictor and its retained auxiliary or nonlinear leaves.

        Returns
        -------
        tuple[tuple[str, SymbolicColumn], ...]
            Checked runtime names paired with their explicit column contracts.

        Raises
        ------
        ValueError
            If a column belongs to another nominal row scope or extent.
        """
        columns: list[tuple[str, SymbolicColumn]] = []
        for _, prefix, predictor in self._predictor_items(formula):
            start = len(columns)
            columns.extend(
                (prefix + column.qvr_name, column)
                for column in predictor.fixed_columns
                if isinstance(column, SymbolicColumn)
            )
            columns.extend(
                (prefix + "random_" + term.column.qvr_name, term.column)
                for term in predictor.random_terms
                if isinstance(term.column, SymbolicColumn)
            )
            for _, column in columns[start:]:
                column.validate_rows("Resp", len(predictor.response_values))
        return tuple(columns)

    def fixed_column_observations(self, formula: Formula) -> dict[str, torch.Tensor]:
        """Bind retained covariates, offsets, and response likelihood weights.

        Parameters
        ----------
        formula : Formula
            The formula whose fixed-effect columns are bound.

        Returns
        -------
        dict[str, torch.Tensor]
            Numeric predictor and offset columns as ``float32`` tensors,
            keyed by their QVR input names. Auxiliary columns use the prefix
            ``dpar_<parameter>_``. Response weights retain their stored
            ``float64`` precision under the ``formula_weights`` key.
        """
        obs: dict[str, torch.Tensor] = {}
        obs.update(bind_inputs(law_inputs(self.group_laws)))
        if (
            formula.response_layout.event_shape
            and self._family.trials_parameter is not None
        ):
            from quivers.formulas._response import count_totals

            obs["formula_trials"] = count_totals(formula)
        if formula.modifiers.weights is not None:
            obs["formula_weights"] = torch.as_tensor(
                formula.modifiers.weights.data.copy()
            )
        if formula.modifiers.decision is not None:
            obs["formula_decision"] = torch.as_tensor(
                formula.modifiers.decision.data.copy(), dtype=torch.bool
            )
        for role, column in formula.modifiers.columns():
            if role not in {"weights", "decision"}:
                obs[column.qvr_name] = torch.as_tensor(column.data.copy())
        if formula.modifiers.threshold_groups is not None and threshold_kinds(
            self._family
        ):
            obs.update(
                threshold_inputs(
                    formula.modifiers.threshold_groups, threshold_kinds(self._family)
                )
            )
        obs.update(
            bind_inputs(item for term in self.terms(formula) for item in term.inputs())
        )
        if self._predictor_graph is not None:
            from quivers.formulas._predictor_graph import graph_input_bindings

            obs.update(
                graph_input_bindings(
                    self._predictor_graph,
                    {
                        name: column.data
                        for name, column in self._predictor_graph.inputs.items()
                    },
                )
            )
        return obs

    @staticmethod
    def _ordinal_category_count(formula: Formula) -> int:
        return FormulaToQVRModule._category_count(formula, family="cumulative")

    @staticmethod
    def _component_labels(formula: Formula) -> tuple[str, ...]:
        """Read retained event components or scalar factor category labels.

        Parameters
        ----------
        formula : Formula
            Formula carrying the response coordinate layout.

        Returns
        -------
        tuple[str, ...]
            Authored vector labels, factor levels, or numeric category codes.
        """
        if formula.response_layout.component_names:
            return formula.response_layout.component_names
        if formula.response_layout.category_levels:
            return formula.response_layout.category_levels
        return tuple(
            str(index)
            for index in range(
                FormulaToQVRModule._category_count(formula, family="categorical")
            )
        )

    @staticmethod
    def _category_count(formula: Formula, *, family: str) -> int:
        if formula.response_layout.component_names:
            return len(formula.response_layout.component_names)
        values = np.asarray(formula.response_values, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{family} family requires a nonempty, finite response")
        if np.any(values < -(2**63)) or np.any(values >= 2**63):
            raise ValueError(
                f"{family} family requires integer category labels in the Int64 range"
            )
        integers = values.astype(np.int64)
        if not np.equal(values, integers).all():
            raise ValueError(f"{family} family requires integer category labels")
        if formula.response_layout.category_levels:
            count = len(formula.response_layout.category_levels)
            if np.any(integers < 0) or np.any(integers >= count):
                raise ValueError(
                    f"{family} factor response codes exceed the retained levels"
                )
            if count < 2:
                raise ValueError(f"{family} family requires at least two categories")
            return count
        levels = np.unique(integers)
        if levels[0] != 0 or levels[-1] != levels.size - 1:
            raise ValueError(
                f"{family} family requires contiguous category labels "
                f"0..K-1; got {levels.tolist()!r}"
            )
        if levels.size < 2:
            raise ValueError(f"{family} family requires at least two categories")
        return int(levels.size)

    @staticmethod
    def _append_finite_object(
        statements: list[Statement], name: str, cardinality: int
    ) -> None:
        """Create or verify a shared compiler-owned finite axis.

        Parameters
        ----------
        statements : list[Statement]
            Module declarations receiving the finite object if absent.
        name : str
            Shared axis identity.
        cardinality : int
            Exact extent required by the predictor metadata.

        Raises
        ------
        ValueError
            If a previous declaration gives this axis another extent.
        """
        existing = next(
            (
                statement
                for statement in statements
                if isinstance(statement, ObjectDecl) and name in statement.names
            ),
            None,
        )
        if existing is not None:
            if existing.init == TypeFromExpr(
                expr=DiscreteConstructor(constructor="FinSet", args=(str(cardinality),))
            ):
                return
            raise ValueError(
                f"incompatible dimensions for shared formula axis {name!r}"
            )
        statements.append(
            ObjectDecl(
                names=(name,),
                init=TypeFromExpr(
                    expr=DiscreteConstructor(
                        constructor="FinSet", args=(str(cardinality),)
                    )
                ),
            )
        )

    def _append_ordinal_cutpoints(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> str:
        """Emit identified thresholds and return their variable name.

        Cutpoints are represented by a zero origin followed by cumulative
        positive spacings and are centered before use. Centering separates the
        threshold location from the formula intercept. With ``thresholds_by``
        the log spacings receive hierarchical group-level deviations, so every
        group gets an ordered scale without introducing an unidentified random
        threshold shift alongside its random intercept.

        Parameters
        ----------
        formula : Formula
            Response categories, category-specific design and group coding.
        statements : list[Statement]
            Declarations receiving the threshold and profile axes.
        program_steps : list[ProgramStep]
            Steps receiving priors and threshold transformations.

        Returns
        -------
        str
            Shared or rowwise threshold binding; empty for a binary hurdle
            law without a positive-category boundary.

        Raises
        ------
        ValueError
            If a group is absent or a prior call is malformed.
        """
        n_categories = self._category_count(formula, family=self._family.name)
        spec = self._family.ordinal
        assert spec is not None
        if spec.kind == "hurdle_cumulative":
            n_categories -= 1
        n_cutpoints = n_categories - 1
        if n_cutpoints == 0:
            return ""

        ordered = spec.kind in {"cumulative", "hurdle_cumulative"}
        constrained_cs = ordered and (
            any(column.category_specific for column in formula.fixed_columns)
            or any(
                term.column is not None and term.column.category_specific
                for term in formula.random_terms
            )
        )
        cutpoint_index = "FormulaCutpoint"
        self._append_finite_object(statements, cutpoint_index, n_cutpoints)

        # brms thres(n, gr): one unpooled threshold vector per group.
        threshold_groups = formula.modifiers.threshold_groups
        if threshold_groups is not None:
            if self._thresholds_by is not None:
                raise ValueError(
                    "thres(gr = ...) gives each group its own thresholds and "
                    "cannot be combined with thresholds_by"
                )
            if spec.thresholds != "flexible":
                raise ValueError(
                    "thres(gr = ...) is supported for the ordinal families with "
                    "flexible thresholds"
                )
            if any(column.category_specific for column in formula.fixed_columns) or any(
                term.column is not None and term.column.category_specific
                for term in formula.random_terms
            ):
                raise ValueError(
                    "Cannot use category specific effects in models with "
                    "multiple thresholds."
                )
            return append_group_thresholds(
                formula,
                threshold_groups,
                ordered=ordered,
                cutpoint_axis=cutpoint_index,
                statements=statements,
                steps=program_steps,
                declare=self._append_finite_object,
                draw=lambda site, default, index: self._prior_draw(
                    site, default, index=index
                ),
                location_prior=self._fixed_prior,
            )

        if n_cutpoints == 1:
            if self._thresholds_by is not None:
                program_steps.extend(
                    (
                        _let(
                            "cutpoint_matrix",
                            _factor(
                                (
                                    ("threshold_group", _qvr_name(self._thresholds_by)),
                                    ("cutpoint", cutpoint_index),
                                ),
                                body=LetExprLiteral(value=0.0),
                            ),
                        ),
                        _let(
                            "row_cutpoints",
                            _index(
                                "cutpoint_matrix",
                                _var(f"{_qvr_name(self._thresholds_by)}_idx"),
                            ),
                        ),
                    )
                )
                return "row_cutpoints"
            program_steps.append(
                _let(
                    "cutpoints",
                    _factor(
                        (("cutpoint", cutpoint_index),),
                        cases=((0, LetExprLiteral(value=0.0)),),
                    ),
                )
            )
            return "cutpoints"

        if spec.thresholds == "equidistant" and self._thresholds_by is None:
            if ordered:
                program_steps.append(
                    self._prior_draw(
                        "cutpoint_spacing",
                        "LogNormal(0.0, 1.0)",
                        lower=0.0,
                    )
                )
            else:
                program_steps.append(
                    self._prior_draw("cutpoint_spacing", "Normal(0.0, 1.0)")
                )
            equidistant_spacing: LetExprNode = _var("cutpoint_spacing")
            if constrained_cs:
                self._append_ordinal_spacing_bounds(
                    formula, statements, program_steps, n_cutpoints - 1
                )
                equidistant_spacing = _add(
                    _call("max", _var("ordinal_spacing_bound")), equidistant_spacing
                )
            program_steps.append(
                _let(
                    "cutpoints",
                    _factor(
                        (("cutpoint", cutpoint_index),),
                        body=_mul(
                            _sub(
                                _var("cutpoint"),
                                LetExprLiteral(value=(n_cutpoints - 1) / 2),
                            ),
                            equidistant_spacing,
                        ),
                    ),
                )
            )
            return "cutpoints"

        if not ordered:
            if spec.thresholds == "sum_to_zero":
                index = cutpoint_index
                latent = "cutpoints_raw"
                n_free = n_cutpoints
            else:
                index = "FormulaThresholdContrast"
                latent = "cutpoint_offset"
                n_free = n_cutpoints - 1
                self._append_finite_object(statements, index, n_free)
            program_steps.append(
                self._prior_draw(latent, "Normal(0.0, 1.0)", index=TypeName(name=index))
            )
            if spec.thresholds == "sum_to_zero":
                raw = _var(latent)
            else:
                raw = _factor(
                    (("cutpoint", cutpoint_index),),
                    cases=(
                        (0, LetExprLiteral(value=0.0)),
                        *(
                            (
                                position,
                                _index(
                                    latent,
                                    LetExprLiteral(value=position - 1, integral=True),
                                ),
                            )
                            for position in range(1, n_cutpoints)
                        ),
                    ),
                )
            program_steps.append(_let("cutpoints_uncentered", raw))
            program_steps.append(
                _let(
                    "cutpoints",
                    _sub(
                        _var("cutpoints_uncentered"),
                        _call("mean", _var("cutpoints_uncentered")),
                    ),
                )
            )
            return "cutpoints"

        spacing_index = "FormulaSpacing"
        n_spacings = n_cutpoints - 1
        self._append_finite_object(statements, spacing_index, n_spacings)

        if self._thresholds_by is None:
            program_steps.append(
                self._prior_draw(
                    "cutpoint_log_spacing",
                    "Normal(0.0, 1.0)",
                    index=TypeName(name=spacing_index),
                )
            )
            spacing: LetExprNode = _call("exp", _var("cutpoint_log_spacing"))
            if constrained_cs:
                self._append_ordinal_spacing_bounds(
                    formula, statements, program_steps, n_spacings
                )
                spacing = _add(_var("ordinal_spacing_bound"), spacing)
            program_steps.append(
                _let(
                    "cutpoint_cumulative",
                    _call("cumsum", spacing),
                )
            )
            cases = ((0, LetExprLiteral(value=0.0)),) + tuple(
                (
                    cutpoint,
                    _index(
                        "cutpoint_cumulative",
                        LetExprLiteral(value=float(cutpoint - 1), integral=True),
                    ),
                )
                for cutpoint in range(1, n_cutpoints)
            )
            program_steps.append(
                _let(
                    "cutpoints_uncentered",
                    _factor((("cutpoint", cutpoint_index),), cases=cases),
                )
            )
            program_steps.append(
                _let(
                    "cutpoints",
                    _sub(
                        _var("cutpoints_uncentered"),
                        _call("mean", _var("cutpoints_uncentered")),
                    ),
                )
            )
            return "cutpoints"

        group = self._thresholds_by
        if group not in formula.group_levels:
            raise ValueError(
                f"thresholds_by={group!r} must name a grouping factor in "
                f"the formula; choices are {sorted(formula.group_levels)!r}"
            )
        qgroup = _qvr_name(group)
        n_groups = len(formula.group_levels[group])
        # Free shape coordinates can be narrower than the physical gap axis.
        n_shape = 1 if spec.thresholds == "equidistant" else n_spacings
        shape_index = spacing_index
        if n_shape != n_spacings:
            shape_index = "FormulaThresholdShape"
            self._append_finite_object(statements, shape_index, n_shape)
        cell_index = "FormulaThresholdCell"
        self._append_finite_object(statements, cell_index, n_groups * n_shape)

        emit_group_block(
            self,
            plan_threshold_block(
                group,
                n_groups,
                shape_index,
                n_shape,
                self._level_basis(group, formula.group_levels)
                if group in self._group_covariances
                else None,
                cell_axis=cell_index,
                parameterization=self._reparameterize,
            ),
            statements,
            program_steps,
        )
        spacing_matrix: LetExprNode = _call("exp", _var("cutpoint_log_spacing_matrix"))
        if spec.thresholds == "equidistant":
            # The one gap must dominate all adjacent CS contrasts. A proper
            # hierarchical log-slack prior remains on its positive excess.
            gap: LetExprNode = _call(
                "exp",
                _index(
                    "cutpoint_log_spacing_matrix",
                    _var("threshold_group"),
                    LetExprLiteral(value=0, integral=True),
                ),
            )
            if constrained_cs:
                self._append_ordinal_spacing_bounds(
                    formula, statements, program_steps, n_spacings
                )
                program_steps.append(
                    _let(
                        "ordinal_group_spacing_bound",
                        _factor(
                            (("threshold_group", qgroup),),
                            body=_call(
                                "max",
                                _index(
                                    "ordinal_spacing_bound", _var("threshold_group")
                                ),
                            ),
                        ),
                    )
                )
                gap = _add(
                    _index("ordinal_group_spacing_bound", _var("threshold_group")),
                    gap,
                )
            spacing_matrix = _factor(
                (("threshold_group", qgroup), ("spacing", spacing_index)),
                body=gap,
            )
        elif constrained_cs:
            self._append_ordinal_spacing_bounds(
                formula, statements, program_steps, n_spacings
            )
            spacing_matrix = _add(_var("ordinal_spacing_bound"), spacing_matrix)
        program_steps.append(
            _let(
                "cutpoint_cumulative_matrix",
                _call("cumsum", spacing_matrix),
            )
        )
        program_steps.append(
            _let(
                "cutpoint_zero_by_group",
                _factor(
                    (("threshold_group", qgroup),),
                    body=LetExprLiteral(value=0.0),
                ),
            )
        )
        position_cases: tuple[tuple[int, LetExprNode], ...] = (
            (0, _var("cutpoint_zero_by_group")),
        ) + tuple(
            (
                cutpoint,
                _factor(
                    (("threshold_group", qgroup),),
                    body=_index(
                        "cutpoint_cumulative_matrix",
                        _var("threshold_group"),
                        LetExprLiteral(value=float(cutpoint - 1), integral=True),
                    ),
                ),
            )
            for cutpoint in range(1, n_cutpoints)
        )
        program_steps.append(
            _let(
                "cutpoints_by_position",
                _factor((("cutpoint", cutpoint_index),), cases=position_cases),
            )
        )
        program_steps.append(
            _let(
                "cutpoint_matrix_uncentered",
                _factor(
                    (("threshold_group", qgroup), ("cutpoint", cutpoint_index)),
                    body=_index(
                        "cutpoints_by_position",
                        _var("cutpoint"),
                        _var("threshold_group"),
                    ),
                ),
            )
        )
        centered_value = _sub(
            _index(
                "cutpoint_matrix_uncentered",
                _var("threshold_group"),
                _var("cutpoint"),
            ),
            _call(
                "mean",
                _index("cutpoint_matrix_uncentered", _var("threshold_group")),
            ),
        )
        program_steps.append(
            _let(
                "cutpoint_matrix",
                _factor(
                    (("threshold_group", qgroup), ("cutpoint", cutpoint_index)),
                    body=centered_value,
                ),
            )
        )
        program_steps.append(
            _let(
                "row_cutpoints",
                _index("cutpoint_matrix", _var(f"{qgroup}_idx")),
            )
        )
        return "row_cutpoints"

    def _threshold_location_formula(self, formula: Formula) -> Formula:
        """Reuse or introduce the one group location of a full threshold profile.

        Parameters
        ----------
        formula : Formula
            Formula whose threshold grouping factor has retained coding.

        Returns
        -------
        Formula
            Formula with exactly one ordinary intercept for the threshold group.

        Raises
        ------
        ValueError
            If the requested threshold group has no evaluated coding.
        """
        group = self._thresholds_by
        if group is None:
            return formula
        if group not in formula.group_levels:
            raise ValueError(
                f"thresholds_by={group!r} requires evaluated group coding; "
                "include the group in the formula or use the dataframe fitting adapter"
            )
        if any(
            term.group == group and term.slope == "Intercept"
            for term in formula.random_terms
        ):
            return formula
        return formula.with_(
            random_terms=(
                *formula.random_terms,
                RandomTerm(group=group, slope="Intercept"),
            )
        )

    def _append_ordinal_spacing_bounds(
        self,
        formula: Formula,
        statements: list[Statement],
        steps: list[ProgramStep],
        n_spacings: int,
    ) -> None:
        """Bound adjacent linear category effects on the retained design.

        Parameters
        ----------
        formula : Formula
            Evaluated design whose rows define the fitting domain.
        statements : list[Statement]
            Module declarations receiving the spacing axis when necessary.
        steps : list[ProgramStep]
            Program steps receiving one linear-time reduction per spacing.
        n_spacings : int
            Number of gaps between adjacent cumulative thresholds.

        Notes
        -----
        The emitted bound is ``max(0, max_i(v[i,j+1]-v[i,j]))``.
        Adding a positive slack gives every strictly feasible threshold gap
        for the fixed linear effects. A prior on the log slack is a proper
        conditional prior, distinct from independently specified threshold
        priors conditioned on the feasible domain.
        """
        axis = "FormulaSpacing"
        if not any(
            isinstance(statement, ObjectDecl) and statement.names == (axis,)
            for statement in statements
        ):
            self._append_finite_object(statements, axis, n_spacings)
        gap = _var("constraint_spacing")

        def difference(row: LetExprNode) -> LetExprNode:
            """Form adjacent transition-effect differences.

            Parameters
            ----------
            row : LetExprNode
                Original fitting row index.

            Returns
            -------
            LetExprNode
                Adjacent effect difference at the bound spacing position.
            """
            return _sub(
                _index(
                    "eta_nonbaseline",
                    row,
                    _add(gap, LetExprLiteral(value=1, integral=True)),
                ),
                _index("eta_nonbaseline", row, gap),
            )

        def reduce_rows(row_axis: str, row: LetExprNode) -> LetExprNode:
            """Reduce a nonempty row partition to nonnegative gap bounds.

            Parameters
            ----------
            row_axis : str
                Finite partition axis.
            row : LetExprNode
                Original row selected at each partition position.

            Returns
            -------
            LetExprNode
                One maximum lower bound for each threshold gap.
            """
            return _factor(
                (("constraint_spacing", axis),),
                body=_call(
                    "max",
                    LetExprLiteral(value=0.0),
                    _call(
                        "max",
                        _factor((("constraint_row", row_axis),), body=difference(row)),
                    ),
                ),
            )

        if self._thresholds_by is not None:
            group = self._thresholds_by
            cases: list[tuple[int, LetExprNode]] = []
            for code in range(len(formula.group_levels[group])):
                positions = tuple(
                    i
                    for i, value in enumerate(formula.group_indices[group])
                    if value == code
                )
                if not positions:
                    bound = _factor(
                        (("constraint_spacing", axis),), body=LetExprLiteral(value=0.0)
                    )
                else:
                    row_axis = f"FormulaConstraintRows_{code}"
                    rows_name = f"ordinal_constraint_rows_{code}"
                    self._append_finite_object(statements, row_axis, len(positions))
                    steps.append(
                        _let(
                            rows_name,
                            _factor(
                                (("constraint_member", row_axis),),
                                cases=tuple(
                                    (i, LetExprLiteral(value=position, integral=True))
                                    for i, position in enumerate(positions)
                                ),
                            ),
                        )
                    )
                    bound = reduce_rows(
                        row_axis, _index(rows_name, _var("constraint_row"))
                    )
                cases.append((code, bound))
            value = _factor(
                (("constraint_group", _qvr_name(group)),), cases=tuple(cases)
            )
        else:
            value = reduce_rows("Resp", _var("constraint_row"))
        steps.append(
            _let(
                "ordinal_spacing_bound",
                value,
            )
        )

    @staticmethod
    def _row_predictor(
        formula: Formula, name: str, row: LetExprNode, *, neural: bool = False
    ) -> LetExprNode:
        """Index predictors whose declared design includes a row axis.

        Parameters
        ----------
        formula : Formula
            Proportional predictor metadata.
        name : str
            Predictor binding.
        row : LetExprNode
            Row index.
        neural : bool
            Whether an external predictor supplies a row axis.

        Returns
        -------
        LetExprNode
            Row-indexed predictor or its shared scalar binding.
        """
        rowwise = (
            neural
            or bool(formula.design_terms)
            or bool(formula.offsets)
            or any(
                term.column is None or not term.column.category_specific
                for term in formula.random_terms
            )
            or any(
                not column.is_intercept and not column.category_specific
                for column in formula.fixed_columns
            )
        )
        return _index(name, row) if rowwise else _var(name)

    def _append_ordinal_contrasts(
        self, formula: Formula, cutpoints: str, program_steps: list[ProgramStep]
    ) -> None:
        """Materialize and validate each row/transition contrast once.

        Parameters
        ----------
        formula : Formula
            Proportional and category-specific predictor metadata.
        cutpoints : str
            Shared or rowwise threshold binding.
        program_steps : list[ProgramStep]
            Steps receiving checked discrimination, contrasts and cumulative
            row-order validation.
        """
        row, transition = _var("ordinal_row"), _var("ordinal_transition")
        eta = self._row_predictor(
            formula, "mu", row, neural=self._predictor_name is not None
        )
        category_specific = any(
            column.category_specific for column in formula.fixed_columns
        ) or any(
            term.column is not None and term.column.category_specific
            for term in formula.random_terms
        )
        if category_specific:
            eta = _add(eta, _index("eta_nonbaseline", row, transition))
        disc_formula = self._distributional.get("disc")
        program_steps.append(
            _let("ordinal_disc", _call("positive_parameter", _var("disc")))
        )
        disc = (
            self._row_predictor(disc_formula, "ordinal_disc", row)
            if disc_formula is not None
            else _var("ordinal_disc")
        )
        cuts = (
            _index(cutpoints, row, transition)
            if self._thresholds_by is not None
            or formula.modifiers.threshold_groups is not None
            else _index(cutpoints, transition)
        )
        program_steps.append(
            _let(
                "ordinal_contrasts",
                _factor(
                    (
                        ("ordinal_row", "Resp"),
                        ("ordinal_transition", "FormulaCutpoint"),
                    ),
                    body=_mul(disc, _sub(cuts, eta)),
                ),
            )
        )
        spec = self._family.ordinal
        if (
            category_specific
            and spec is not None
            and spec.kind in {"cumulative", "hurdle_cumulative"}
            and self._category_count(formula, family=self._family.name)
            - int(spec.kind == "hurdle_cumulative")
            > 2
        ):
            gap = _var("ordinal_spacing")
            differences = _factor(
                (("ordinal_row", "Resp"), ("ordinal_spacing", "FormulaSpacing")),
                body=_sub(
                    _index(
                        "ordinal_contrasts",
                        row,
                        _add(gap, LetExprLiteral(value=1, integral=True)),
                    ),
                    _index("ordinal_contrasts", row, gap),
                ),
            )
            program_steps.append(
                _let(
                    "ordinal_effective_spacing",
                    _call("positive_parameter", differences),
                )
            )

    @staticmethod
    def _append_ordinal_cdf(
        argument: LetExprNode, link: str, prefix: str, steps: list[ProgramStep]
    ) -> tuple[str, str]:
        """Bind a shared pair of stable log CDF/survival expressions.

        Parameters
        ----------
        argument : LetExprNode
            Finite ordinal contrasts.
        link : str
            Static inverse link.
        prefix : str
            Namespace of intermediate bindings.
        steps : list[ProgramStep]
            Steps receiving stable probability expressions.

        Returns
        -------
        tuple of str
            Log-CDF and log-survival binding names.

        Raises
        ------
        ValueError
            If the inverse link is unsupported.
        """

        def bind(suffix: str, expression: LetExprNode) -> LetExprVar:
            """Append an intermediate probability binding.

            Parameters
            ----------
            suffix : str
                Local binding label.
            expression : LetExprNode
                Expression to evaluate once.

            Returns
            -------
            LetExprVar
                Reference to the intermediate.
            """
            name = prefix + suffix
            steps.append(_let(name, expression))
            return _var(name)

        def choose(
            condition: LetExprNode, yes: LetExprNode, no: LetExprNode
        ) -> LetExprCall:
            """Select safe elementwise alternatives.

            Parameters
            ----------
            condition : LetExprNode
                Boolean predicate.
            yes : LetExprNode
                Value where the predicate holds.
            no : LetExprNode
                Value elsewhere.

            Returns
            -------
            LetExprCall
                Selection expression.
            """
            return _call("select", condition, yes, no)

        zero, one = LetExprLiteral(value=0.0), LetExprLiteral(value=1.0)
        z = bind("argument", argument)

        def minus(x: LetExprNode) -> LetExprNode:
            """Negate a real expression.

            Parameters
            ----------
            x : LetExprNode
                Expression to negate.

            Returns
            -------
            LetExprNode
                Its additive inverse.
            """
            return _mul(LetExprLiteral(value=-1.0), x)

        if link in {"logit", "probit_approx"}:
            if link == "probit_approx":
                z = bind(
                    "approximation",
                    _add(
                        _mul(
                            LetExprLiteral(value=0.07056),
                            _call("pow", z, LetExprLiteral(value=3.0)),
                        ),
                        _mul(LetExprLiteral(value=1.5976), z),
                    ),
                )
            cdf, sf = _call("logsigmoid", z), _call("logsigmoid", minus(z))
        elif link == "probit":
            cdf, sf = _call("log_ndtr", z), _call("log_ndtr", minus(z))
        elif link == "softit":
            positive = bind("positive", _call("softplus", z))
            small = LetExprBinOp(op="<", left=z, right=LetExprLiteral(value=-35.0))
            safe = choose(small, zero, z)
            log_positive = bind(
                "log_positive", choose(small, z, _call("log", _call("softplus", safe)))
            )
            denominator = bind("denominator", _call("log1p", positive))
            cdf, sf = _sub(log_positive, denominator), minus(denominator)
        elif link == "cauchit":
            magnitude = bind("magnitude", _call("abs", z))
            tail = LetExprBinOp(op=">", left=magnitude, right=one)
            safe = choose(tail, magnitude, one)
            small = bind(
                "small",
                _sub(
                    _call("log", _call("atan", _div(one, safe))),
                    LetExprLiteral(value=math.log(math.pi)),
                ),
            )
            large = bind("large", _call("log1mexp", small))
            negative = LetExprBinOp(op="<", left=z, right=zero)
            regular = _div(
                _call("atan", choose(tail, zero, z)), LetExprLiteral(value=math.pi)
            )
            cdf = choose(
                tail,
                choose(negative, small, large),
                _call("log_probability", _add(LetExprLiteral(value=0.5), regular)),
            )
            sf = choose(
                tail,
                choose(negative, large, small),
                _call("log_probability", _sub(LetExprLiteral(value=0.5), regular)),
            )
        elif link == "cloglog":
            small = LetExprBinOp(op="<", left=z, right=LetExprLiteral(value=-35.0))
            large = LetExprBinOp(op=">", left=z, right=LetExprLiteral(value=80.0))
            safe_cdf = choose(small, zero, choose(large, zero, z))
            regular = _call("log1mexp", minus(_call("exp", safe_cdf)))
            cdf = choose(small, z, choose(large, zero, regular))
            overflow = LetExprBinOp(op=">", left=z, right=LetExprLiteral(value=709.0))
            sf = choose(
                overflow,
                _call("log_probability", zero),
                minus(_call("exp", choose(overflow, zero, z))),
            )
        else:
            raise ValueError(f"unsupported ordinal link {link!r}")
        bind("cdf", cdf)
        bind("sf", sf)
        return prefix + "cdf", prefix + "sf"

    def _append_ordinal_logits(
        self,
        formula: Formula,
        cutpoints: str,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> None:
        """Lower ordinal log masses with shared transitions and one prefix scan.

        Parameters
        ----------
        formula : Formula
            Response and predictor metadata.
        cutpoints : str
            Shared or row-indexed threshold binding.
        statements : list[Statement]
            Declarations receiving the response-category axis.
        program_steps : list[ProgramStep]
            Steps receiving normalized category log masses.
        """
        spec = self._family.ordinal
        assert spec is not None
        categories = self._category_count(formula, family=self._family.name)
        hurdle = spec.kind == "hurdle_cumulative"
        base_categories = categories - int(hurdle)
        self._append_finite_object(statements, "FormulaOrdinalCategory", categories)
        row = _var("ordinal_row")
        cdf_name: str | None = None
        sf_name: str | None = None
        if base_categories > 1:
            self._append_ordinal_contrasts(formula, cutpoints, program_steps)
            argument = _var("ordinal_contrasts")
            if spec.kind in {"acat", "cratio"}:
                argument = _mul(LetExprLiteral(value=-1.0), argument)
            cdf_name, sf_name = self._append_ordinal_cdf(
                argument, spec.link, "ordinal_log_", program_steps
            )
            if spec.kind == "acat":
                program_steps.append(
                    _let(
                        "ordinal_prefix",
                        _call("cumsum", _sub(_var(cdf_name), _var(sf_name))),
                    )
                )
            elif spec.kind in {"sratio", "cratio"}:
                continue_name = sf_name if spec.kind == "sratio" else cdf_name
                program_steps.append(
                    _let("ordinal_prefix", _call("cumsum", _var(continue_name)))
                )

        def cell(name: str, position: int) -> LetExprNode:
            """Select one transition at the current response row.

            Parameters
            ----------
            name : str
                Transition binding.
            position : int
                Zero-based transition position.

            Returns
            -------
            LetExprNode
                Selected transition value.
            """
            return _index(name, row, LetExprLiteral(value=position, integral=True))

        def base_mass(category: int) -> LetExprNode:
            """Construct a category log mass before a hurdle component.

            Parameters
            ----------
            category : int
                Zero-based positive-law category.

            Returns
            -------
            LetExprNode
                Stable category log mass.
            """
            if base_categories == 1:
                return LetExprLiteral(value=0.0)
            assert cdf_name is not None and sf_name is not None
            if spec.kind == "acat":
                return (
                    LetExprLiteral(value=0.0)
                    if category == 0
                    else cell("ordinal_prefix", category - 1)
                )
            if spec.kind in {"sratio", "cratio"}:
                previous = (
                    LetExprLiteral(value=0.0)
                    if category == 0
                    else cell("ordinal_prefix", category - 1)
                )
                stop_name = cdf_name if spec.kind == "sratio" else sf_name
                return (
                    previous
                    if category == base_categories - 1
                    else _add(previous, cell(stop_name, category))
                )
            if category == 0:
                return cell(cdf_name, 0)
            if category == base_categories - 1:
                return cell(sf_name, category - 1)
            if spec.link == "logit":
                lower, upper = (
                    cell("ordinal_contrasts", category - 1),
                    cell("ordinal_contrasts", category),
                )
                return _add(
                    cell(cdf_name, category),
                    cell(sf_name, category - 1),
                    _call("log1mexp", _sub(lower, upper)),
                )
            mirror = LetExprBinOp(
                op=">",
                left=cell(cdf_name, category - 1),
                right=LetExprLiteral(value=-math.log(2)),
            )
            high = _call(
                "select", mirror, cell(sf_name, category - 1), cell(cdf_name, category)
            )
            low = _call(
                "select", mirror, cell(sf_name, category), cell(cdf_name, category - 1)
            )
            return _add(high, _call("log1mexp", _sub(low, high)))

        hu: LetExprNode | None = None
        if hurdle:
            hu_formula = self._distributional.get("hu")
            hu = (
                self._row_predictor(hu_formula, "hu", row)
                if hu_formula is not None
                else _var("hu")
            )
        cases: list[tuple[int, LetExprNode]] = []
        for category in range(categories):
            if hurdle:
                assert hu is not None
                value = (
                    _call("log_probability", hu)
                    if category == 0
                    else _add(
                        _call("log_probability", _sub(LetExprLiteral(value=1.0), hu)),
                        base_mass(category - 1),
                    )
                )
            else:
                value = base_mass(category)
            cases.append((category, _factor((("ordinal_row", "Resp"),), body=value)))
        program_steps.append(
            _let(
                "ordinal_logits_by_category",
                _factor(
                    (("ordinal_category", "FormulaOrdinalCategory"),),
                    cases=tuple(cases),
                ),
            )
        )
        threshold_groups = formula.modifiers.threshold_groups
        program_steps.append(
            _let(
                "ordinal_logits"
                if threshold_groups is None
                else "ordinal_padded_logits",
                _factor(
                    (
                        ("ordinal_row", "Resp"),
                        ("ordinal_category", "FormulaOrdinalCategory"),
                    ),
                    body=_index(
                        "ordinal_logits_by_category", _var("ordinal_category"), row
                    ),
                ),
            )
        )
        if threshold_groups is not None:
            # brms thres(n, gr): keep each row's own categories.
            append_group_log_masses(
                threshold_groups,
                spec.kind,
                "ordinal_padded_logits",
                "ordinal_logits",
                row_axis="Resp",
                category_axis="FormulaOrdinalCategory",
                steps=program_steps,
            )

    def _parse_prior(self, name: str, text: str) -> SampleStep:
        """Bind a canonical prior template in this forward preparation's cache.

        Parameters
        ----------
        name : str
            Existing latent-site binder.
        text : str
            Authored prior expression.

        Returns
        -------
        SampleStep
            The canonical prior with its actual site name.

        Raises
        ------
        ValueError
            If the prior is not one valid distribution construction.
        """
        return _auxiliary_draw(name, text, templates=self._prior_templates)

    def _prior_draw(
        self,
        name: str,
        default: str,
        *,
        column: FixedColumn | SymbolicColumn | None = None,
        index=None,
        lower: float | None = None,
    ) -> SampleStep:
        """Emit ordinary priors through one parser, retaining explicit overrides.

        Parameters
        ----------
        name, default : str
            Existing site binder and its default source prior.
        column : FixedColumn or None
            Autoscale only an unoverridden non-intercept default coefficient.
        index : ObjectExpr or None
            Original nominal coefficient axes.
        lower : float or None
            Existing auxiliary-domain lower-bound policy.

        Returns
        -------
        SampleStep
            Proper source prior with unchanged site name and index metadata.
        """
        if name in self._prior_steps:
            result = self._prior_steps[name]
        else:
            explicit = name in self._user_priors
            text = self._user_priors.get(name, default)
            result = self._parse_prior(name, text)
            if not explicit and column is not None and not column.is_intercept:
                result = (
                    _scaled_prior_step(result, column.prior_multiplier)
                    if isinstance(column, SymbolicColumn)
                    else _autoscaled_prior_step(result, column.data)
                )
        return _prior_lower_bound(result, lower).with_(index=index)

    def _append_categorical_predictor(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
        *,
        ordinal_transitions: bool = False,
        correlated: Mapping[tuple[str, str, str], LetExprNode] | None = None,
        parameter: str = "mu",
        component_count: int | None = None,
        reference: int | None = 0,
        prefix: str = "",
    ) -> None:
        """Emit one shared scalar-component predictor in retained label order.

        Parameters
        ----------
        formula : Formula
            Evaluated design columns and group identities.
        statements : list[Statement]
            Module objects receiving component and innovation axes.
        program_steps : list[ProgramStep]
            Priors, deterministic predictors, and group transformations.
        ordinal_transitions : bool
            Retain transition coordinates for the ordinal likelihood emitter.
        correlated : Mapping or None
            Coefficients already transformed by joint correlation blocks.
        parameter : str
            Location or coordinate-valued auxiliary parameter name.
        component_count : int or None
            Explicit coordinate extent, or the retained response category count.
        reference : int or None
            Zero-predictor component; None estimates every coordinate.
        prefix : str
            Native namespacing for auxiliary designs and coefficients.

        Raises
        ------
        ValueError
            If the response categories or retained component axis are invalid.
        """
        n_categories = component_count or self._category_count(
            formula, family=self._family.name
        )
        if (
            not ordinal_transitions
            and self._family.response is not None
            and parameter == "mu"
        ):
            labels = self._component_labels(formula)
            reference = self._family.response.reference_index(labels)
        coordinates = tuple(
            index for index in range(n_categories) if index != reference
        )
        if (
            ordinal_transitions
            and self._family.ordinal is not None
            and self._family.ordinal.kind == "hurdle_cumulative"
        ):
            n_categories -= 1
        n_logits = n_categories - int(reference is not None)
        category_index = f"{prefix}FormulaCategory"
        logit_index = (
            "FormulaCutpoint"
            if ordinal_transitions
            else f"{prefix}FormulaCategoryLogit"
        )
        if not ordinal_transitions:
            self._append_finite_object(statements, category_index, n_categories)
            self._append_finite_object(statements, logit_index, n_logits)

        cell_terms: list[LetExprNode] = []
        response_var = _var("response")
        logit_var = _var("category_logit")

        for col in formula.fixed_columns:
            beta_name = prefix + (
                "intercept" if col.is_intercept else f"beta_{col.qvr_name}"
            )
            program_steps.append(
                self._prior_draw(
                    beta_name,
                    self._fixed_prior,
                    column=col,
                    index=TypeName(name=logit_index),
                )
            )
            coefficient = _index(beta_name, logit_var)
            if col.is_intercept:
                cell_terms.append(coefficient)
            else:
                cell_terms.append(
                    _mul(coefficient, _index(prefix + col.qvr_name, response_var))
                )

        cell_terms.extend(
            _index(f"{prefix}formula_offset_{index}", response_var)
            for index, _ in enumerate(formula.offsets)
        )

        if (
            self._predictor_name is not None
            and not ordinal_transitions
            and parameter == "mu"
        ):
            predictor_index = _add(
                _mul(
                    response_var,
                    LetExprLiteral(value=float(n_logits), integral=True),
                ),
                logit_var,
            )
            cell_terms.append(_index(self._predictor_name, predictor_index))

        cell_terms.extend(
            _index(name, logit_var, response_var)
            for name in self._append_group_terms(
                formula,
                statements,
                program_steps,
                prefix=prefix,
                correlated=correlated,
                output_axis=logit_index,
            )
        )

        for contribution in self._append_design_terms(
            formula,
            program_steps,
            prefix=prefix,
            component_axis=logit_index,
            components=n_logits,
            statements=statements,
        ):
            cell_terms.append(_index(contribution.name, response_var, logit_var))

        program_steps.append(
            _let(
                f"{prefix}eta_nonbaseline",
                _factor(
                    (("response", "Resp"), ("category_logit", logit_index)),
                    body=_add(*cell_terms),
                ),
            )
        )
        if ordinal_transitions:
            return
        self._append_component_response(
            program_steps,
            parameter=parameter,
            prefix=prefix,
            reference=reference,
            coordinates=coordinates,
            category_index=category_index,
        )

    def _append_component_response(
        self,
        program_steps,
        *,
        parameter,
        prefix,
        reference,
        coordinates,
        category_index,
    ):
        """Complete shared labelled predictors with the retained reference and family link.

        Parameters
        ----------
        program_steps : list
            Steps already binding the free coordinate predictor matrix.
        parameter, prefix : str
            Family output and its established namespace.
        reference : int or None
            Identified reference index, or all-coordinate parameterization.
        coordinates : tuple[int, ...]
            Free response coordinates in retained order.
        category_index : str
            Full response component axis.
        """
        response_var = _var("response")
        if (
            self._family.response is not None
            and self._family.response.kind == "logistic_normal"
            and parameter == "mu"
        ):
            program_steps.append(_let("eta", _var(f"{prefix}eta_nonbaseline")))
            program_steps.append(_let("mu", _var("eta")))
            return
        program_steps.append(
            _let(
                f"{prefix}categorical_zero_logit",
                _factor(
                    (("response", "Resp"),),
                    body=LetExprLiteral(value=0.0),
                ),
            )
        )
        category_cases: tuple[tuple[int, LetExprNode], ...] = (
            ()
            if reference is None
            else ((reference, _var(f"{prefix}categorical_zero_logit")),)
        ) + tuple(
            (
                category,
                _factor(
                    (("response", "Resp"),),
                    body=_index(
                        f"{prefix}eta_nonbaseline",
                        response_var,
                        LetExprLiteral(value=float(coordinate), integral=True),
                    ),
                ),
            )
            for coordinate, category in enumerate(coordinates)
        )
        program_steps.append(
            _let(
                f"{prefix}categorical_logits_by_category",
                _factor((("category", category_index),), cases=category_cases),
            )
        )
        program_steps.append(
            _let(
                f"{prefix}eta",
                _factor(
                    (("response", "Resp"), ("category", category_index)),
                    body=_index(
                        f"{prefix}categorical_logits_by_category",
                        _var("category"),
                        response_var,
                    ),
                ),
            )
        )
        link = (
            self._family.location_link
            if parameter == "mu"
            else next(
                aux.link for aux in self._family.aux_params if aux.name == parameter
            )
        )
        _append_link(program_steps, parameter, _var(f"{prefix}eta"), link)

    def _level_basis(
        self, group: str, levels: Mapping[str, Sequence[str]]
    ) -> GroupCovarianceBasis:
        """Read the support basis of a grouping factor's known covariance.

        Parameters
        ----------
        group : str
            Grouping factor with a known level covariance.
        levels : Mapping[str, Sequence[str]]
            Labels of each grouping factor in level-axis order.

        Returns
        -------
        GroupCovarianceBasis
            The basis of an emitted block of the factor, or a new
            factorization over its labels.
        """
        for law in self._emitted_groups:
            if law.group == group and law.level.basis is not None:
                return law.level.basis
        return self._group_covariances[group].basis(levels[group])

    def _output_axes(self, formula: Formula) -> dict[str, OutputAxis]:
        """Read the axis of each predictor with one coefficient per component.

        Parameters
        ----------
        formula : Formula
            Response layout and categories.

        Returns
        -------
        dict[str, OutputAxis]
            Axes by predictor namespace; empty for a scalar response.
        """
        prefixes = component_valued_prefixes(self._family, self._distributional)
        if not prefixes:
            return {}
        axis = response_axis(self._family, self._component_labels(formula))
        assert axis is not None
        return {prefix: axis.with_(name=prefix + axis.name) for prefix in prefixes}

    def _transition_axis(self, formula: Formula) -> OutputAxis | None:
        """Read the axis of category-specific coefficients.

        Parameters
        ----------
        formula : Formula
            Response categories and predictors.

        Returns
        -------
        OutputAxis or None
            The transition axis, or None when no predictor has a
            category-specific coefficient.
        """
        if not any(
            column is not None and column.category_specific
            for _, _, predictor in self._predictor_items(formula)
            for column in (
                *predictor.fixed_columns,
                *(term.column for term in predictor.random_terms),
            )
        ):
            return None
        return transition_axis(
            self._family, self._category_count(formula, family=self._family.name)
        )

    def _plan(
        self, formula: Formula, laws: Mapping[tuple[str, str, str], GroupBlockLaw]
    ) -> tuple[Predictor, ...]:
        """Plan every linear predictor of the model.

        Parameters
        ----------
        formula : Formula
            Location predictor and response.
        laws : Mapping[tuple[str, str, str], GroupBlockLaw]
            Laws of the group blocks the program emits, by coefficient.

        Returns
        -------
        tuple[Predictor, ...]
            The planned predictors.
        """
        return plan_predictors(
            formula,
            self._family,
            distributional=self._distributional,
            graph=self._predictor_graph,
            options=PredictorOptions(
                coefficient_prior=self._fixed_prior,
                scale_prior=self._random_scale_prior,
                correlation_prior=self._correlation_prior,
                default_priors=self._default_priors,
                center=self._center,
                decomp=self._decomp,
                coordinates=self._coordinates_supported,
                host=self._predictor_name,
            ),
            outputs=self._output_axes(formula),
            transitions=self._transition_axis(formula),
            laws=laws,
        )

    def _plan_groups(
        self, formula: Formula, supplied: Collection[tuple[str, str, str]]
    ) -> tuple[GroupBlockLaw, ...]:
        """Plan the law of every group block the program emits.

        Parameters
        ----------
        formula : Formula
            Location predictor, combined with configured auxiliary predictors.
        supplied : Collection[tuple[str, str, str]]
            Coefficients, as (prefix, group, slope), another program emits.

        Returns
        -------
        tuple[GroupBlockLaw, ...]
            Laws of the joint and the independent blocks.
        """
        # brms mu<c>: per-category formulas replace the main predictor terms.
        if any(
            is_category_parameter(self._family, name) for name in self._distributional
        ):
            formula = main_without_terms(formula)
        transitions = self._transition_axis(formula)
        laws = plan_group_blocks(
            self._predictor_items(formula),
            levels={
                group: levels
                for _, _, predictor in self._predictor_items(formula)
                for group, levels in predictor.group_levels.items()
            },
            covariances=self._group_covariances,
            parameterization=self._reparameterize,
            component_axes={
                prefix: (axis.name, axis.size)
                for prefix, axis in self._output_axes(formula).items()
            },
            transition_axis=(transitions.name, transitions.size)
            if transitions is not None
            else None,
            supplied=supplied,
        )
        self._group_plan = {
            (member.prefix, law.group, member.slope): law
            for law in laws
            for member in law.members
        }
        return laws

    def _append_correlations(
        self,
        laws: Iterable[GroupBlockLaw],
        statements: list[Statement],
        steps: list[ProgramStep],
        supplied: Mapping[tuple[str, str, str], LetExprNode],
    ) -> dict[tuple[str, str, str], LetExprNode]:
        """Emit the joint group blocks ahead of the predictors.

        Parameters
        ----------
        laws : Iterable[GroupBlockLaw]
            Planned laws.
        statements : list[Statement]
            Module statements receiving correlation-dimension objects.
        steps : list[ProgramStep]
            Program steps receiving priors and group transformations.
        supplied : Mapping
            Coefficients another program emits, gathered at the rows' levels.

        Returns
        -------
        dict[tuple[str, str, str], LetExprNode]
            Coefficients keyed by predictor prefix, group, and slope.

        Raises
        ------
        ValueError
            If a correlation prior uses a different distribution family.
        """
        result = dict(supplied)
        for law in laws:
            if law.placement == "joint":
                result.update(emit_group_block(self, law, statements, steps))
        return result

    def _append_group_terms(
        self,
        formula: Formula,
        statements: list[Statement],
        steps: list[ProgramStep],
        *,
        prefix: str,
        correlated: Mapping[tuple[str, str, str], LetExprNode] | None,
        output_axis: str | None,
    ) -> list[str]:
        """Emit the row contribution of every group coefficient of a predictor.

        Each coefficient is gathered at the rows' levels, from its joint
        block or from its own block emitted here, and carried to the rows by
        its row map.

        Parameters
        ----------
        formula : Formula
            Evaluated group terms and their row maps.
        statements : list[Statement]
            Module declarations receiving the axes of group blocks.
        steps : list[ProgramStep]
            Steps receiving the blocks and the row contributions.
        prefix : str
            Namespace of the predictor's coefficients and covariates.
        correlated : Mapping or None
            Coefficients of blocks emitted ahead of the predictor, gathered
            at the rows' levels.
        output_axis : str or None
            Component axis of the predictor's coefficients; None for scalar
            coefficients.

        Returns
        -------
        list[str]
            The binding of each group coefficient's contribution: a row
            vector for scalar coefficients, otherwise an array over the
            output axis and the rows.
        """
        emitted = dict(correlated or {})
        names: list[str] = []
        for term in formula.random_terms:
            key = (prefix, term.group, term.slope)
            if key not in emitted:
                emitted.update(
                    emit_group_block(self, self._group_plan[key], statements, steps)
                )
            name = f"{coefficient_binding(prefix, term.group, term.slope)}_per_row"
            bind_rows(
                name, row_map(formula, term, prefix), emitted[key], output_axis, steps
            )
            names.append(name)
        return names

    def _append_predictor(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
        *,
        prefix: str = "",
        predictor_name: str | None = None,
        correlated: Mapping[tuple[str, str, str], LetExprNode] | None = None,
    ) -> LetExprNode:
        """Emit one independently named predictor, sharing only grouping axes.

        Parameters
        ----------
        formula : Formula
            Evaluated design columns and group coding.
        statements : list[Statement]
            Module declarations receiving the axes of group blocks.
        program_steps : list[ProgramStep]
            Steps to append coefficient draws and row contributions to.
        prefix : str
            Namespace for this predictor's coefficients and covariates.
        predictor_name : str or None
            Optional host-computed contribution on the predictor scale.
        correlated : Mapping[tuple[str, str, str], LetExprNode] or None
            Row-wise coefficients keyed by predictor prefix, group, and slope.

        Returns
        -------
        LetExprNode
            Sum of the predictor contributions, or scalar zero.
        """
        linear_terms: list[LetExprNode] = [
            _var(f"{prefix}formula_offset_{index}")
            for index, _ in enumerate(formula.offsets)
        ]
        linear_terms.extend(
            self._append_design_terms(formula, program_steps, prefix=prefix)
        )
        linear_terms.extend(
            append_monotonic_terms(self, formula, program_steps, prefix=prefix)
        )

        # brms centering and QR coordinates replace the ordinary coefficient loop.
        coordinates = self._population_plan(formula, prefix)
        if coordinates is not None:
            linear_terms.extend(
                emit_population(
                    formula,
                    program_steps,
                    prefix,
                    coordinates,
                    lambda name, column: self._prior_draw(
                        name, self._fixed_prior, column=column
                    ),
                )
            )
        # Fixed effects: one scalar latent per design-matrix column.
        for col in formula.fixed_columns if coordinates is None else ():
            beta_name = prefix + (
                "intercept" if col.is_intercept else f"beta_{col.qvr_name}"
            )
            program_steps.append(
                self._prior_draw(beta_name, self._fixed_prior, column=col)
            )
            if col.is_intercept:
                linear_terms.append(_var(beta_name))
            else:
                contrib_name = f"{beta_name}_per_row"
                program_steps.append(
                    _let(
                        contrib_name, _mul(_var(beta_name), _var(prefix + col.qvr_name))
                    )
                )
                linear_terms.append(_var(contrib_name))

        if predictor_name is not None:
            # A host-attached differentiable predictor contributes on the
            # linear-predictor scale. The free name is supplied by ``fit`` at
            # every optimization step, so gradients remain connected to the
            # external module rather than being frozen into the data frame.
            linear_terms.append(_var(predictor_name))

        linear_terms.extend(
            _var(name)
            for name in self._append_group_terms(
                formula,
                statements,
                program_steps,
                prefix=prefix,
                correlated=correlated,
                output_axis=None,
            )
        )

        return _add(*linear_terms) if linear_terms else LetExprLiteral(value=0.0)

    def _population_plan(
        self, formula: Formula, prefix: str
    ) -> PopulationCoordinates | None:
        """Decide brms centering and QR coordinates for one predictor.

        Parameters
        ----------
        formula : Formula
            Evaluated predictor.
        prefix : str
            Its namespace: location, ``dpar_<name>_``, or ``nlpar_<name>_``.

        Returns
        -------
        PopulationCoordinates or None
            The coordinate plan, or None for the ordinary emission.
        """
        if not self._coordinates_supported or prefix.startswith("nlpar_"):
            return None
        if prefix:
            center = self._default_priors == "brms" or self._center is True
            decomp = "none"
        else:
            center = (
                self._center
                if self._center is not None
                else self._default_priors == "brms"
            )
            decomp = self._decomp
        return population_coordinates(formula, prefix, center=center, decomp=decomp)

    def _append_design_terms(
        self,
        formula: Formula,
        steps: list[ProgramStep],
        *,
        prefix: str = "",
        component_axis: str | None = None,
        components: int | None = None,
        statements: list[Statement] | None = None,
    ) -> list[LetExprVar]:
        """Emit coupled Gaussian coordinates through the shared source priors.

        Parameters
        ----------
        formula : Formula
            Retained row designs and coupled coefficient blocks.
        steps : list[ProgramStep]
            Existing source program receiving priors and contractions.
        prefix : str
            Owning predictor namespace.
        component_axis : str or None
            Existing nominal free-response or transition coordinate axis.
        components : int or None
            Extent of that axis, when a component-valued predictor is required.
        statements : list[Statement] or None
            Existing module declarations receiving joint innovation axes.

        Returns
        -------
        list[LetExprVar]
            Scalar-row or row-by-component additive contribution bindings.
        """
        values = []
        shared_scales: set[str] = set()
        for term in formula.design_terms:
            name = prefix + term.qvr_name
            scale, raw, coefficients = (
                prefix + (term.scale_group or term.qvr_name) + "_scale",
                name + "_z",
                name + "_coefficients",
            )
            axis = name + "_Coordinate"
            # brms approximate gp(): coefficients sqrt(spd(sdgp, lscale)) * z.
            if term.gp is not None:
                if component_axis is not None:
                    raise ValueError(
                        f"{term.term}: gp() terms are not supported in predictors "
                        "with one coefficient per response category"
                    )
                steps.extend(
                    gp_block_steps(
                        self,
                        term.gp,
                        name,
                        prefix,
                        sdgp_prior(formula.response_values, self._family),
                    )
                )
                contribution = name + "_per_row"
                steps.append(
                    _let(contribution, _call("matmul", _var(name), _var(coefficients)))
                )
                values.append(_var(contribution))
                continue
            # Smooths sharing an mgcv id share one sampled smoothing scale.
            if scale not in shared_scales:
                shared_scales.add(scale)
                steps.append(
                    self._prior_draw(
                        scale,
                        self._random_scale_prior,
                        index=TypeName(name=component_axis)
                        if component_axis is not None
                        else None,
                    )
                )
            if component_axis is None:
                steps.append(
                    _draw(raw, "Normal", (0.0, 1.0), index=TypeName(name=axis))
                )
                steps.append(_let(coefficients, _mul(_var(scale), _var(raw))))
            else:
                assert components is not None and statements is not None
                innovation_axis = name + "_InnovationCoordinate"
                self._append_finite_object(
                    statements, innovation_axis, term.data.shape[1] * components
                )
                steps.append(
                    _draw(
                        raw, "Normal", (0.0, 1.0), index=TypeName(name=innovation_axis)
                    )
                )
                basis, component = _var("design_basis"), _var("design_component")
                flat = _add(
                    _mul(basis, LetExprLiteral(value=components, integral=True)),
                    component,
                )
                steps.append(
                    _let(
                        coefficients,
                        _factor(
                            (
                                ("design_basis", axis),
                                ("design_component", component_axis),
                            ),
                            body=_mul(_index(scale, component), _index(raw, flat)),
                        ),
                    )
                )
            contribution = name + "_per_row"
            steps.append(
                _let(contribution, _call("matmul", _var(name), _var(coefficients)))
            )
            values.append(_var(contribution))
        return values

    def _runtime_parameters(self, formula: Formula) -> tuple[QiecValueParameter, ...]:
        """Declare checked group-index and response-weight input tensors.

        Parameters
        ----------
        formula : Formula
            Primary predictor, combined with configured auxiliary predictors.

        Returns
        -------
        tuple[QiecValueParameter, ...]
            Checked integer group indices and an optional Real row-weight tensor.
        """
        groups = dict.fromkeys(
            group
            for _, _, predictor in self._predictor_items(formula)
            for group in predictor.group_indices
        )
        index_type = QiecTypeApplication(
            constructor="Tensor",
            static_arguments=(QiecTypeName(name="Int"),),
            indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
        )
        parameters = tuple(
            QiecValueParameter(name=f"{_qvr_name(group)}_idx", type_expr=index_type)
            for group in groups
        )
        parameters += declare_inputs(
            (
                *law_inputs(self.group_laws),
                *(
                    item
                    for predictor in self._predictors
                    for term in predictor.terms
                    for item in term.inputs()
                ),
            )
        )
        if formula.modifiers.decision is not None:
            decision_type = index_type.with_(
                static_arguments=(QiecTypeName(name="Bool"),)
            )
            parameters += (
                QiecValueParameter(name="formula_decision", type_expr=decision_type),
            )
            response_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(QiecTypeName(name="Real"),),
                indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
            )
            parameters += (
                QiecValueParameter(name="formula_response", type_expr=response_type),
            )
        if formula.response_layout.event_shape:
            response_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(
                    QiecTypeName(
                        name="Int"
                        if self._family.trials_parameter is not None
                        else "Real"
                    ),
                ),
                indices=(
                    QiecShapeIndex(
                        dimensions=(
                            QiecIndexName(name="Resp"),
                            QiecIndexName(name="FormulaCategory"),
                        )
                    ),
                ),
            )
            parameters += (
                QiecValueParameter(
                    name=_qvr_name(formula.response_name), type_expr=response_type
                ),
            )
            if self._family.trials_parameter is not None:
                parameters += (
                    QiecValueParameter(name="formula_trials", type_expr=index_type),
                )
        if formula.modifiers.weights is not None:
            weight_type = QiecTypeApplication(
                constructor="Tensor",
                static_arguments=(QiecTypeName(name="Real"),),
                indices=(QiecShapeIndex(dimensions=(QiecIndexName(name="Resp"),)),),
            )
            parameters += (
                QiecValueParameter(name="formula_weights", type_expr=weight_type),
            )
        if self._predictor_graph is not None:
            for column in self._predictor_graph.inputs.values():
                carrier = (
                    "Bool"
                    if column.data.dtype.kind == "b"
                    else "Int"
                    if column.data.dtype.kind in "iu"
                    else "Real"
                )
                parameters += (
                    QiecValueParameter(
                        name=column.qvr_name,
                        type_expr=index_type.with_(
                            static_arguments=(QiecTypeName(name=carrier),)
                        ),
                    ),
                )
        existing = {parameter.name for parameter in parameters}
        for role, column in formula.modifiers.columns():
            if column.qvr_name in existing:
                continue
            carrier = (
                "Int" if role in {"censoring", "trials", "threshold_group"} else "Real"
            )
            role_type = index_type.with_(static_arguments=(QiecTypeName(name=carrier),))
            parameters += (
                QiecValueParameter(name=column.qvr_name, type_expr=role_type),
            )
            existing.add(column.qvr_name)
        return parameters

    def _append_vector_observation(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
    ) -> None:
        """Lower one named response event using shared component predictors.

        Parameters
        ----------
        formula : Formula
            Vector response layout and evaluated predictors.
        statements : list[Statement]
            Module objects receiving covariance coordinates.
        program_steps : list[ProgramStep]
            Shared priors, predictors, and final response factor.

        Raises
        ------
        ValueError
            If a residual-correlation prior is not an LKJ Cholesky law.
        """
        assert self._family.response is not None
        kind = self._family.response.kind
        if kind == "multinomial":
            args = ("formula_trials", "eta")
        elif kind in {"dirichlet", "dirichlet_multinomial"}:
            phi = (
                _index("phi", _var("response"))
                if "phi" in self._distributional
                or self._predictor_graph is not None
                and "phi" in self._predictor_graph.roots
                else _var("phi")
            )
            concentration = _factor(
                (("response", "Resp"), ("component", "FormulaCategory")),
                body=_mul(_index("mu", _var("response"), _var("component")), phi),
            )
            program_steps.append(_let("formula_concentration", concentration))
            args = (
                ("formula_trials", "formula_concentration")
                if kind == "dirichlet_multinomial"
                else ("formula_concentration",)
            )
        elif kind == "dirichlet2":
            args = ("mu",)
        else:
            width = len(self._component_labels(formula)) - 1
            self._append_residual_correlation(
                program_steps,
                "FormulaCategoryLogit",
                width,
                result="formula_alr_correlation",
            )
            component = self._family.component_scale
            assert component is not None
            name = component.name
            sigma = (
                _index(name, _var("response"), _var("i"))
                if name in self._distributional
                or any(
                    is_scale_parameter(self._family, parameter)
                    for parameter in self._distributional.keys()
                    | self._fixed_parameters.keys()
                )
                or self._predictor_graph is not None
                and isinstance(
                    self._predictor_graph.roots.get(name), ComponentExpression
                )
                else _index(name, _var("response"))
                if self._predictor_graph is not None
                and name in self._predictor_graph.roots
                else _var(name)
                if name in self._fixed_parameters
                else _index(name, _var("i"))
            )
            program_steps.append(
                _let(
                    "formula_alr_scale",
                    _factor(
                        (
                            ("response", "Resp"),
                            ("i", "FormulaCategoryLogit"),
                            ("j", "FormulaCategoryLogit"),
                        ),
                        body=_mul(
                            sigma,
                            _index("formula_alr_correlation", _var("i"), _var("j")),
                        ),
                    ),
                )
            )
            reference = self._family.response.reference_index(
                self._component_labels(formula)
            )
            assert reference is not None
            args = ("mu", "formula_alr_scale", reference)
        observation = _draw(
            _qvr_name(formula.response_name),
            self._family.observe_family,
            args,
            index=TypeName(name="Resp"),
            mode="score",
            weights="formula_weights"
            if formula.modifiers.weights is not None
            else None,
        )
        if kind == "multinomial":
            observation = observation.with_(
                args=(
                    DrawArgNamed(
                        parameter="total_count", value=_to_draw_arg("formula_trials")
                    ),
                    DrawArgNamed(parameter="logits", value=_to_draw_arg("eta")),
                )
            )
        observation = observation.with_(
            options=(
                *observation.options,
                OptionEntry(key="over", value=OptionName(value="FormulaCategory")),
            )
        )
        program_steps.append(observation)

    def _append_residual_correlation(
        self,
        steps: list[ProgramStep],
        axis: str,
        width: int,
        *,
        result: str,
        site: str = "rescor",
    ) -> None:
        """Emit the shared proper correlation factor for an explicit event axis.

        Parameters
        ----------
        steps : list[ProgramStep]
            Existing program steps, extended in dependency order.
        axis : str
            Nominal response or log-ratio coordinate axis.
        width : int
            Number of coordinates on that axis.
        result : str
            Deterministic factor binding consumed by the residual scale.
        site : str
            Authored correlation prior site.
        """
        if width < 1:
            raise ValueError("residual correlation requires a nonempty event axis")
        if width == 1:
            steps.append(
                _let(
                    result,
                    _factor((("i", axis), ("j", axis)), body=LetExprLiteral(value=1.0)),
                )
            )
            return
        prior = self._prior_draw(site, self._correlation_prior)
        if prior.morphism != "LKJCholesky":
            raise ValueError("residual correlations require an LKJCholesky prior")
        steps.append(prior.with_(index=TypeName(name=axis)))
        steps.append(_let(result, _var(site)))

    def _predictor_items(self, formula: Formula):
        """Enumerate additive designs once across ordinary and nonlinear parameters.

        Parameters
        ----------
        formula : Formula
            Primary response and ordinary location metadata.

        Returns
        -------
        tuple
            Parameter labels, stable prefixes, and evaluated formulas.
        """
        return predictor_items(formula, self._distributional, self._predictor_graph)

    def _graph_expression_at(self, expression, row, component=None):
        """Project nominal graph inputs to one row and explicitly labelled coordinate.

        Parameters
        ----------
        expression : LetExprNode
            Canonical scalar expression at one result coordinate.
        row : LetExprNode
            Observation-row index.
        component : str or None
            Requested coordinate label, required for vector dependencies.

        Returns
        -------
        LetExprNode
            Capture-safe row expression over existing QVR bindings.

        Raises
        ------
        ValueError
            If a vector dependency has no compatible labelled coordinate.
        """
        graph = self._predictor_graph
        assert graph is not None
        substitution = {
            column.qvr_name: _index(column.qvr_name, row)
            for column in graph.inputs.values()
        }
        for name in (*graph.leaves, *graph.expressions):
            value = graph.expressions.get(name)
            if isinstance(value, ComponentExpression):
                if component is None:
                    # Only reject an actually used vector, not unrelated nodes.
                    from quivers.dsl.let_expr_traversal import free_let_names

                    if f"nlpar_{name}" in free_let_names(expression):
                        raise ValueError(
                            f"nonlinear vector {name!r} requires explicit component expressions"
                        )
                    continue
                if component not in value.components:
                    from quivers.dsl.let_expr_traversal import free_let_names

                    if f"nlpar_{name}" in free_let_names(expression):
                        raise ValueError(
                            f"nonlinear vector {name!r} has no component {component!r}"
                        )
                    continue
                substitution[f"nlpar_{name}"] = _index(
                    f"nlpar_{name}",
                    row,
                    LetExprLiteral(
                        value=value.components.index(component), integral=True
                    ),
                )
            else:
                substitution[f"nlpar_{name}"] = _index(f"nlpar_{name}", row)
        return substitute_let_expr(expression, substitution)

    def _append_graph_value(
        self, name, value, statements, steps, *, axis=None, components=None
    ):
        """Materialize a graph node with explicit row and optional nominal component axes.

        Parameters
        ----------
        name : str
            Existing QVR binding name.
        value : LetExprNode or ComponentExpression
            Scalar-row or explicitly labelled expression.
        statements, steps : list
            Module axes and deterministic program statements.
        axis : str or None
            Shared coordinate axis for a family root, otherwise node-local.
        components : tuple[str, ...] or None
            Required family coordinate labels in canonical retained order.

        Raises
        ------
        ValueError
            If result labels or scalar/vector carriers do not match the root.
        """
        row = _var("graph_row")
        if not isinstance(value, ComponentExpression):
            if components is not None:
                raise ValueError(
                    f"nonlinear {name!r} requires explicit component expressions for {components}"
                )
            steps.append(
                _let(
                    name,
                    _factor(
                        (("graph_row", "Resp"),),
                        body=self._graph_expression_at(value, row),
                    ),
                )
            )
            return
        labels = value.components if components is None else components
        if set(labels) != set(value.components):
            raise ValueError(
                f"nonlinear {name!r} component labels {value.components} do not match expected {labels}"
            )
        axis = axis or f"FormulaGraphCoordinate_{name}"
        self._append_finite_object(statements, axis, len(labels))
        by_coordinate = f"{name}_by_coordinate"
        cases = tuple(
            (
                index,
                _factor(
                    (("graph_row", "Resp"),),
                    body=self._graph_expression_at(
                        value.values[value.components.index(label)], row, label
                    ),
                ),
            )
            for index, label in enumerate(labels)
        )
        steps.append(
            _let(by_coordinate, _factor((("graph_coordinate", axis),), cases=cases))
        )
        steps.append(
            _let(
                name,
                _factor(
                    (("graph_row", "Resp"), ("graph_coordinate", axis)),
                    body=_index(by_coordinate, _var("graph_coordinate"), row),
                ),
            )
        )

    def _append_graph_parameters(self, formula, statements, steps, correlated):
        """Emit nonlinear leaves through existing coefficient and correlation paths.

        Parameters
        ----------
        formula : Formula
            Primary response metadata.
        statements, steps : list
            Existing axes and program statements.
        correlated : Mapping
            Shared group coefficients across all additive parameter leaves.
        """
        graph = self._predictor_graph
        if graph is None:
            return
        for name, leaf in graph.leaves.items():
            value = self._append_predictor(
                leaf, statements, steps, prefix=f"nlpar_{name}_", correlated=correlated
            )
            raw = f"nlpar_{name}_raw"
            steps.append(_let(raw, value))
            steps.append(
                _let(
                    f"nlpar_{name}",
                    _factor(
                        (("graph_row", "Resp"),),
                        body=self._row_predictor(leaf, raw, _var("graph_row")),
                    ),
                )
            )
        for name, expression in graph.expressions.items():
            self._append_graph_value(f"nlpar_{name}", expression, statements, steps)

    def _append_auxiliaries(
        self,
        formula: Formula,
        statements: list[Statement],
        program_steps: list[ProgramStep],
        correlated: Mapping[tuple[str, str, str], LetExprNode],
    ) -> None:
        """Lower shared scalar or coordinate-valued auxiliary parameters.

        Parameters
        ----------
        formula : Formula
            Response metadata and primary design.
        statements : list[Statement]
            Module objects receiving coordinate axes.
        program_steps : list[ProgramStep]
            Program steps receiving priors and predictors.
        correlated : Mapping[tuple[str,str,str],LetExprNode]
            Shared correlated group contributions.
        """
        for aux in self._family.aux_params:
            if (
                aux.name == self._family.error_scale
                and formula.modifiers.standard_error is not None
                and not formula.modifiers.standard_error.sigma
            ):
                continue
            if aux.name in self._fixed_parameters:
                program_steps.append(
                    _let(
                        aux.name,
                        LetExprLiteral(value=float(self._fixed_parameters[aux.name])),
                    )
                )
                continue
            if (
                self._predictor_graph is not None
                and aux.name in self._predictor_graph.roots
            ):
                eta_name = f"dpar_{aux.name}_eta"
                value = self._predictor_graph.roots[aux.name]
                labels = None
                if isinstance(value, ComponentExpression):
                    if not aux.per_component:
                        raise ValueError(
                            f"family parameter {aux.name!r} requires a scalar nonlinear root"
                        )
                    assert self._family.response is not None
                    labels = tuple(
                        self._component_labels(formula)[index]
                        for index in self._family.response.coordinate_indices(
                            self._component_labels(formula)
                        )
                    )
                self._append_graph_value(
                    eta_name,
                    value,
                    statements,
                    program_steps,
                    axis=f"dpar_{aux.name}_FormulaCategoryLogit",
                    components=labels,
                )
                _append_link(program_steps, aux.name, _var(eta_name), aux.link)
                continue
            predictor = self._distributional.get(aux.name)
            # brms sigma<c>: one scale formula or scalar scale per free category.
            if aux.per_component and append_scale_predictors(
                self,
                formula,
                statements,
                program_steps,
                correlated,
                aux.prior,
                lambda steps, name, eta, link=aux.link: _append_link(
                    steps, name, eta, link
                ),
            ):
                continue
            if aux.per_component and predictor is not None:
                self._append_categorical_predictor(
                    predictor,
                    statements,
                    program_steps,
                    parameter=aux.name,
                    component_count=len(self._component_labels(formula)) - 1,
                    reference=None,
                    prefix=f"dpar_{aux.name}_",
                    correlated=correlated,
                )
                continue
            if predictor is None:
                if aux.fixed_default is not None and aux.name not in self._user_priors:
                    program_steps.append(
                        _let(aux.name, LetExprLiteral(value=aux.fixed_default))
                    )
                    continue
                prior = (
                    self._user_priors[aux.name]
                    if aux.name in self._user_priors
                    else aux.with_response_minimum(
                        aux.prior, np.asarray(formula.response_values)
                    )
                )
                draw = self._prior_draw(
                    aux.name,
                    prior,
                    lower=aux.domain.lower if aux.domain.lower else None,
                )
                if aux.per_component:
                    draw = draw.with_(index=TypeName(name="FormulaCategoryLogit"))
                program_steps.append(draw)
            else:
                linear_predictor = self._append_predictor(
                    predictor,
                    statements,
                    program_steps,
                    prefix=f"dpar_{aux.name}_",
                    correlated=correlated,
                )
                eta_name = f"dpar_{aux.name}_eta"
                program_steps.append(_let(eta_name, linear_predictor))
                _append_link(program_steps, aux.name, _var(eta_name), aux.link)

    def _build_module(
        self,
        formula: Formula,
        *,
        supplied_groups: Mapping[tuple[str, str, str], LetExprNode] | None = None,
    ) -> Module:
        # mixtures: one ordinary emission per component inside Mixture(...).
        if self._family.mixture is not None:
            emission = emit_mixture(self, formula, supplied_groups)
            self._mixture_targets = emission.prior_targets
            self._emitted_groups.extend(emission.group_laws)
            self._term_laws = emission.families
            return emission.module
        formula = self._threshold_location_formula(formula)
        statements: list[Statement] = []
        n_obs = formula.response_values.shape[0]
        statements.append(
            ObjectDecl(
                names=("Resp",),
                init=TypeFromExpr(
                    expr=DiscreteConstructor(
                        constructor="FinSet",
                        args=(str(int(n_obs)),),
                    )
                ),
            )
        )
        if self._family.baseline is not None and self._family.baseline.by is not None:
            group = self._family.baseline.by
            if group not in formula.group_levels:
                raise ValueError(
                    f"baseline group {group!r} requires retained group coding; "
                    "supply formula_from_data(..., groups=(group,))"
                )
        group_sizes = {
            _qvr_name(group): len(levels)
            for _, _, predictor in self._predictor_items(formula)
            for group, levels in predictor.group_levels.items()
        }
        for qgroup, size in group_sizes.items():
            self._append_finite_object(statements, qgroup, size)
        laws = self._plan_groups(formula, supplied_groups or {})
        self._predictors = self._plan(
            formula,
            {
                (member.prefix, law.group, member.slope): law
                for law in laws
                for member in law.members
            },
        )
        terms = tuple(
            term for predictor in self._predictors for term in predictor.terms
        )
        self._term_laws = join_catalogs(None, terms)
        for axis in declare_axes(terms):
            self._append_finite_object(statements, axis.name, axis.size)
        # Grouped thresholds read one level per row, which a row with several
        # member levels does not have.
        for _, prefix, predictor in self._predictor_items(formula):
            for membership in predictor.grouping.memberships:
                if membership.group == self._thresholds_by:
                    raise ValueError(
                        f"multi-membership factor {membership.group!r} cannot group "
                        "thresholds: grouped thresholds read one level per row, "
                        "and a row of an mm() factor has several"
                    )

        program_steps: list[ProgramStep] = []
        if self._family.baseline is not None:
            from quivers.formulas._survival import baseline_level_sites, baseline_steps

            basis = self._family.baseline.resolve(formula)
            self._baseline_level_sites = baseline_level_sites(
                basis, self._family.baseline.by
            )
            baseline_objects, baseline_draws = baseline_steps(
                basis,
                prior=self._prior_draw("sbhaz", self._user_priors["sbhaz"])
                if "sbhaz" in self._user_priors
                else None,
                group=_qvr_name(self._family.baseline.by)
                if self._family.baseline.by is not None
                else None,
            )
            statements.extend(baseline_objects)
            program_steps.extend(baseline_draws)
        if self._family.response is not None:
            count = self._category_count(formula, family=self._family.name)
            axis = response_axis(self._family, self._component_labels(formula))
            assert axis is not None
            self._append_finite_object(statements, "FormulaCategory", count)
            self._append_finite_object(statements, axis.name, axis.size)
            for predictor in self._predictors:
                if predictor.output is not None:
                    self._append_finite_object(
                        statements, predictor.output.name, predictor.output.size
                    )
            correlated = self._append_correlations(
                laws, statements, program_steps, supplied_groups or {}
            )
            self._append_graph_parameters(
                formula, statements, program_steps, correlated
            )
            if (
                self._predictor_graph is not None
                and "mu" in self._predictor_graph.roots
            ):
                labels = self._component_labels(formula)
                response = self._family.response
                assert response is not None
                coordinates = response.coordinate_indices(labels)
                expected = tuple(labels[index] for index in coordinates)
                self._append_graph_value(
                    "eta_nonbaseline",
                    self._predictor_graph.roots["mu"],
                    statements,
                    program_steps,
                    axis="FormulaCategoryLogit",
                    components=expected,
                )
                self._append_component_response(
                    program_steps,
                    parameter="mu",
                    prefix="",
                    reference=response.reference_index(labels),
                    coordinates=coordinates,
                    category_index="FormulaCategory",
                )
            elif not append_category_predictors(
                self, formula, statements, program_steps, correlated
            ):
                self._append_categorical_predictor(
                    formula, statements, program_steps, correlated=correlated
                )
            if self._family.vector_response:
                self._append_auxiliaries(formula, statements, program_steps, correlated)
                self._append_vector_observation(formula, statements, program_steps)
            else:
                program_steps.append(
                    _draw(
                        _qvr_name(formula.response_name),
                        "Categorical",
                        ("mu",),
                        index=TypeName(name="Resp"),
                        mode="score",
                        weights="formula_weights"
                        if formula.modifiers.weights is not None
                        else None,
                    )
                )
            statements.append(
                ProgramDecl(
                    name="model",
                    params=None,
                    value_parameters=self._runtime_parameters(formula),
                    domain=TypeName(name="Resp"),
                    codomain=TypeName(name="Resp"),
                    draws=tuple(program_steps),
                    return_vars=(_qvr_name(formula.response_name),),
                )
            )
            statements.append(ExportDecl(expr=ExprIdent(name="model")))
            return Module(statements=tuple(statements))

        proportional = formula.with_(
            fixed_columns=tuple(
                col for col in formula.fixed_columns if not col.category_specific
            ),
            random_terms=tuple(
                term
                for term in formula.random_terms
                if term.column is None or not term.column.category_specific
            ),
        )
        correlated = self._append_correlations(
            laws, statements, program_steps, supplied_groups or {}
        )
        self._append_graph_parameters(formula, statements, program_steps, correlated)
        eta_expr = self._append_predictor(
            proportional,
            statements,
            program_steps,
            predictor_name=self._predictor_name,
            correlated=correlated,
        )

        if self._predictor_graph is not None and "mu" in self._predictor_graph.roots:
            root = self._predictor_graph.roots["mu"]
            if isinstance(root, ComponentExpression):
                raise ValueError(
                    "a scalar response requires a scalar nonlinear mu expression"
                )
            self._append_graph_value("graph_mu_eta", root, statements, program_steps)
            eta_expr = _var("graph_mu_eta")
            if self._predictor_name is not None:
                eta_expr = _add(eta_expr, _var(self._predictor_name))
        self._append_auxiliaries(formula, statements, program_steps, correlated)

        cutpoints_name: str | None = None
        cutpoint_steps: list[ProgramStep] = []
        if self._family.ordinal is not None:
            cutpoints_name = self._append_ordinal_cutpoints(
                formula, statements, cutpoint_steps
            )

        category_columns = tuple(
            col for col in formula.fixed_columns if col.category_specific
        )
        category_terms = tuple(
            term
            for term in formula.random_terms
            if term.column is not None and term.column.category_specific
        )
        if category_columns or category_terms:
            self._append_categorical_predictor(
                formula.with_(
                    fixed_columns=category_columns,
                    design_terms=(),
                    random_terms=category_terms,
                    offsets=(),
                ),
                statements,
                program_steps,
                ordinal_transitions=True,
                correlated=correlated,
            )

        program_steps.extend(cutpoint_steps)

        # Structured latent terms (brms car) add coordinates to the predictor.
        eta_expr = append_latent_terms(
            self, formula, statements, program_steps, eta_expr
        )
        program_steps.append(_let("eta", eta_expr))
        # A log-scale location remains useful even when exp(eta) overflows.
        # Its inverse-link prediction is evaluated on request, not as an
        # unused part of every density or native response-mean computation.
        retained_log_location = self._family.retains_log_location
        if not retained_log_location:
            _append_link(
                program_steps,
                "mu",
                _var("eta"),
                links["identity"]
                if self._family.ordinal is not None
                else self._family.location_link,
            )

        scale: dict[str, LetExprNode] = {}
        if self._family.ordinal is not None:
            assert cutpoints_name is not None
            if self._family.scores_cutpoints:
                # Preserve the existing cumulative source and constructor when
                # discrimination is its fixed unit default.
                disc_is_unit = (
                    self._fixed_parameters.get("disc", 1.0) == 1.0
                    and "disc" not in self._distributional
                    and "disc" not in self._user_priors
                    and not category_columns
                    and not category_terms
                )
                if not disc_is_unit:
                    self._append_ordinal_contrasts(
                        formula, cutpoints_name, program_steps
                    )
                    program_steps.append(
                        _let("ordinal_scaled_eta", LetExprLiteral(value=0.0))
                    )
                    obs_args: tuple[str | float, ...] = (
                        "ordinal_scaled_eta",
                        "ordinal_contrasts",
                    )
                else:
                    obs_args = ("mu", cutpoints_name)
            else:
                self._append_ordinal_logits(
                    formula, cutpoints_name, statements, program_steps
                )
                obs_args = ("ordinal_logits",)
        else:
            # The family states its own map to the native constructor. Under
            # known errors the total scale stands wherever it names its own.
            if formula.modifiers.standard_error is not None:
                assert self._family.error_scale is not None
                self._append_known_error_scale(formula, program_steps)
                scale[self._family.error_scale] = _var("formula_sigma")
            if self._family.decision_event is not None:
                program_steps.append(
                    _let(
                        _qvr_name(formula.response_name),
                        _bind(self._family.decision_event),
                    )
                )
            program_steps.extend(
                _let(binding.name, _bind(binding.expr, **scale))
                for binding in self._family.observe_lets
            )
            obs_args = tuple(
                self._observe_argument(argument, scale)
                for argument in self._family.observe_args
            )
        if self._family.custom is not None:
            # Autocorrelation terms: structural matrices beyond scalar auxiliaries.
            append_autocor_steps(self, statements, program_steps)
            obs_args = tuple(
                "formula_sigma" if binding in scale else binding
                for _, binding in self._family.custom.bindings
            )
        # brms rate(): count denominators scale the mean and NB shape.
        observe_family, obs_args = rate_arguments(
            self._family,
            self._family.observe_family,
            obs_args,
            program_steps,
            formula.modifiers,
        )
        observation = _draw(
            _qvr_name(formula.response_name),
            observe_family,
            obs_args,
            index=TypeName(name="Resp"),
            mode="score",
            family_catalog=self._family.custom.catalog
            if self._family.custom is not None
            else None,
            weights="formula_weights"
            if formula.modifiers.weights is not None
            else None,
        )
        if retained_log_location:
            assert self._family.log_location_argument is not None
            location = _to_draw_arg("mu")
            observation = observation.with_(
                args=tuple(
                    DrawArgNamed(
                        parameter=self._family.log_location_argument,
                        value=_to_draw_arg("eta"),
                    )
                    if isinstance(argument, DrawArgNamed) and argument.value == location
                    else argument
                    for argument in observation.args or ()
                )
            )
        if self._family.ordinal is not None and not self._family.scores_cutpoints:
            observation = observation.with_(
                args=(
                    DrawArgNamed(
                        parameter="logits", value=_to_draw_arg("ordinal_logits")
                    ),
                )
            )
        from quivers.formulas._response_roles import event_option_columns

        observation = observation.with_(
            options=(
                *observation.options,
                *(
                    OptionEntry(key=key, value=OptionName(value=column.qvr_name))
                    for key, column in event_option_columns(formula.modifiers)
                ),
            )
        )
        program_steps.append(observation)

        program_decl = ProgramDecl(
            name="model",
            params=None,
            value_parameters=self._runtime_parameters(formula),
            domain=TypeName(name="Resp"),
            codomain=TypeName(name="Resp"),
            draws=tuple(program_steps),
            return_vars=(_qvr_name(formula.response_name),),
        )
        statements.append(program_decl)
        statements.append(ExportDecl(expr=ExprIdent(name="model")))
        return Module(statements=tuple(statements))


__all__ = [
    "FormulaToQVRModule",
]
