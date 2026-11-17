"""One source-AST prior plan over the formula compiler's existing sites."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from graphlib import CycleError, TopologicalSorter
import math

from quivers.dsl.ast_nodes import (
    DrawArgDist,
    DrawArgNamed,
    DrawArgScalar,
    LetExprLiteral,
    LetStep,
    Module,
    ObjectDecl,
    ProgramDecl,
    ProgramStep,
    SampleStep,
    ObserveStep,
    TypeName,
    TypeFromExpr,
    DiscreteConstructor,
)
from quivers.dsl.step_resolution import _step_value_names
from quivers.formulas._ast import (
    _add,
    _call,
    _factor,
    _index,
    _let,
    _mul,
    _sub,
    _var,
    finite_object_sizes,
)
from quivers.formulas._autocor import autocor_matrix_targets, autocor_prior_target
from quivers.formulas._category_formulas import scale_parameter
from quivers.formulas._gp import gp_prior_sites
from quivers.formulas._group_law import LKJCorrelation, option_prior_sites
from quivers.formulas._monotonic_emission import monotonic_prior_sites
from quivers.formulas._priors import PriorTarget, PriorSpec, RegularizedHorseshoe, R2D2

# brms prior surface: flat and constant rules, main = TRUE, sampled coordinates.
from quivers.formulas._brms_priors import (
    SHRINKAGE_SCALE_CLASSES,
    apply_fixed_rules,
    flat_event_rank,
    join_main_rules,
)
from quivers.formulas._population_coordinates import sampled_site


def _ordered_steps(steps: Sequence[ProgramStep]) -> tuple[ProgramStep, ...]:
    """Order prior dependencies with the existing exhaustive source-name walker.

    Parameters
    ----------
    steps : Sequence[ProgramStep]
        Formula-only draws and deterministic bindings.

    Returns
    -------
    tuple[ProgramStep, ...]
        Stable topological order; unchanged latent binders retain RNG addresses.

    Raises
    ------
    ValueError
        If a conditional prior creates a cycle among program bindings.
    """
    bindings = {}
    for index, step in enumerate(steps):
        if isinstance(step, (SampleStep, ObserveStep)):
            bindings.update((name, index) for name in step.vars)
        elif isinstance(step, LetStep):
            bindings[step.name] = index
    dependencies = {
        i: {bindings[name] for name in _step_value_names(step) if name in bindings}
        for i, step in enumerate(steps)
    }
    try:
        order = TopologicalSorter(dependencies).static_order()
        return tuple(steps[i] for i in order)
    except CycleError as error:
        raise ValueError("conditional prior dependency cycle") from error


def _literal(value, *, integral=False):
    """Build one ordinary real source literal."""
    return LetExprLiteral(value=float(value), integral=integral)


def order_prior_dependencies(module: Module) -> Module:
    """Topologically order the existing program without changing its bindings."""
    return module.with_(
        statements=tuple(
            item.with_(draws=_ordered_steps(item.draws))
            if isinstance(item, ProgramDecl)
            else item
            for item in module.statements
        )
    )


def apply_prior_plan(
    module: Module,
    targets: Sequence[PriorTarget],
    selected: Mapping[str, PriorSpec],
    *,
    draw: Callable[[str, str], SampleStep],
    training_rows: int | Mapping[str, int],
    sigma: str | Mapping[str, str] | None = None,
) -> Module:
    """Rewrite existing prior sites and emit shared proper shrinkage hyperpriors.

    Parameters
    ----------
    module : Module
        Existing formula-emitted program and nominal axes.
    targets : Sequence[PriorTarget]
        Role metadata for its overrideable sample sites.
    selected : Mapping[str, PriorSpec]
        Resolved rules from the same catalogue.
    draw : Callable
        Existing compositional QVR prior parser, taking a binder and source.
    training_rows : int or Mapping[str, int]
        Retained training count, or counts by selected site for response
        graphs with different observation sets. Shared shrinkage calibrates
        each coefficient to its own response count.
    sigma : str or Mapping[str, str] or None
        Existing scalar residual scale, or one scale per selected site in a
        joint response graph. Absent when autoscaling is invalid.

    Returns
    -------
    Module
        Same response program and coefficient binders with replaced prior laws.

    Raises
    ------
    ValueError
        If a hierarchy targets an incompatible class, has inconsistent block
        settings, or supplies a mismatched allocation-concentration vector.
    """
    # Flat and constant rules rewrite their sites; main = TRUE joins classes.
    module, selected = apply_fixed_rules(module, targets, selected)
    selected = join_main_rules(selected, targets)
    by_site = {target.site: target for target in targets}
    statements = list(module.statements)
    program = next(item for item in statements if isinstance(item, ProgramDecl))
    original = {
        step.vars[0]: step for step in program.draws if isinstance(step, SampleStep)
    }
    blocks = {}
    for site, rule in selected.items():
        if isinstance(rule.distribution, str):
            continue
        target = by_site[site]
        if target.class_ not in {"b", *SHRINKAGE_SCALE_CLASSES}:
            raise ValueError(
                f"shrinkage prior on {site!r} requires b or sd coordinates "
                "(classes b, sd, sds, sdgp, sderr, sdcar)"
            )
        # brms: a shrunken standard deviation is the shrinkage scale itself.
        if target.class_ in SHRINKAGE_SCALE_CLASSES and (
            rule.lower is not None or rule.upper is not None
        ):
            raise ValueError(
                f"shrinkage on standard deviation {site!r} takes no bounds"
            )
        key = ("block", rule.block) if rule.block is not None else ("rule", rule)
        blocks.setdefault(key, []).append((target, rule))
    replacements = {}
    hyper = []
    occupied = set(original) | {
        name
        for item in statements
        if isinstance(item, ObjectDecl)
        for name in item.names
    }
    for block_index, members in enumerate(blocks.values()):
        hierarchy = members[0][1].distribution
        if any(rule.distribution != hierarchy for _, rule in members):
            raise ValueError(
                "shared shrinkage block has inconsistent hierarchy settings"
            )
        label = members[0][1].block or str(block_index)
        if not all(c.isalnum() or c == "_" for c in label) or not label:
            raise ValueError(
                "shrinkage block names must contain only letters, digits, or underscores"
            )
        prefix = "prior_" + label
        axis = "FormulaPrior_" + label
        generated = {
            axis,
            *(
                prefix + suffix
                for suffix in ("_global", "_slab", "_local", "_R2", "_allocation")
            ),
            *(prefix + "_scale_" + target.site for target, _ in members),
        }
        if occupied & generated:
            raise ValueError(
                f"shrinkage block {label!r} collides with an existing prior binding"
            )
        occupied.update(generated)
        width = sum(target.width for target, _ in members)
        statements.insert(
            0,
            ObjectDecl(
                names=(axis,),
                init=TypeFromExpr(
                    expr=DiscreteConstructor(constructor="FinSet", args=(str(width),))
                ),
            ),
        )
        counts: dict[str, int] = {}
        common_count: int | None = None
        if isinstance(hierarchy, RegularizedHorseshoe):
            counts = {
                target.site: training_rows[target.site]
                if isinstance(training_rows, Mapping)
                else training_rows
                for target, _ in members
            }
            common_counts = set(counts.values())
            common_count = (
                next(iter(common_counts)) if len(common_counts) == 1 else None
            )
            global_scale = (
                hierarchy.scale_global
                if hierarchy.par_ratio is None
                else hierarchy.par_ratio / math.sqrt(common_count)
                if common_count is not None
                else hierarchy.par_ratio
            )
            hyper.extend(
                (
                    draw(
                        prefix + "_global",
                        f"HalfStudentT({hierarchy.df_global!r},{global_scale!r})",
                    ),
                    draw(
                        prefix + "_slab",
                        f"InverseGamma({hierarchy.df_slab / 2!r},{hierarchy.df_slab / 2!r})",
                    ),
                )
            )
            local = draw(
                prefix + "_local", f"HalfStudentT({hierarchy.df!r},1.0)"
            ).with_(index=TypeName(name=axis))
            hyper.append(local)
        else:
            assert isinstance(hierarchy, R2D2)
            hyper.append(
                draw(
                    prefix + "_R2",
                    f"Beta({hierarchy.mean_R2 * hierarchy.prec_R2!r},{(1 - hierarchy.mean_R2) * hierarchy.prec_R2!r})",
                )
            )
            concentration = hierarchy.cons_D2
            if isinstance(concentration, tuple):
                if len(concentration) != width:
                    raise ValueError(
                        f"R2D2 cons_D2 must have length {width}, got {len(concentration)}"
                    )
            else:
                concentration = (concentration,) * width
            if width == 1:
                hyper.append(
                    _let(
                        prefix + "_allocation",
                        _factor((("allocation_coordinate", axis),), body=_literal(1)),
                    )
                )
            else:
                allocation = draw(
                    prefix + "_allocation",
                    "Dirichlet([" + ",".join(repr(c) for c in concentration) + "])",
                )
                hyper.append(allocation.with_(index=TypeName(name=axis)))
        offset = 0
        for target, rule in members:
            target_sigma = (
                sigma.get(target.site) if isinstance(sigma, Mapping) else sigma
            )
            coordinate = _literal(offset, integral=True)
            binders = tuple(
                (f"prior_coordinate_{i}", name) for i, name in enumerate(target.axes)
            )
            for index, (_name, _axis) in enumerate(binders):
                stride = math.prod(target.shape[index + 1 :])
                coordinate = _add(
                    coordinate,
                    _mul(_var(binders[index][0]), _literal(stride, integral=True)),
                )
            autoscale = (
                hierarchy.autoscale
                and target_sigma is not None
                and target.dpar in (None, "mu")
            )
            if isinstance(hierarchy, RegularizedHorseshoe):
                log_global = _call("log", _var(prefix + "_global"))
                if hierarchy.par_ratio is not None and common_count is None:
                    log_global = _sub(
                        log_global, _literal(math.log(counts[target.site]) / 2)
                    )
                if autoscale and target_sigma is not None:
                    log_global = _add(log_global, _call("log", _var(target_sigma)))
                log_product = _add(
                    log_global, _call("log", _index(prefix + "_local", coordinate))
                )
                log_slab = _add(
                    _literal(math.log(hierarchy.scale_slab)),
                    _mul(_literal(0.5), _call("log", _var(prefix + "_slab"))),
                )
                log_scale = _sub(
                    log_product,
                    _mul(
                        _literal(0.5),
                        _call(
                            "softplus", _mul(_literal(2), _sub(log_product, log_slab))
                        ),
                    ),
                )
            else:
                fraction = _var(prefix + "_R2")
                log_scale = _mul(
                    _literal(0.5),
                    _sub(
                        _add(
                            _call("log", _index(prefix + "_allocation", coordinate)),
                            _call("log", fraction),
                        ),
                        _call("log1p", _mul(_literal(-1), fraction)),
                    ),
                )
                if autoscale and target_sigma is not None:
                    log_scale = _add(log_scale, _call("log", _var(target_sigma)))
            scale = _call("exp", log_scale)
            if binders:
                scale = _factor(binders, body=scale)
            scale_name = prefix + "_scale_" + target.site
            prior = draw(
                target.site,
                ("HalfNormal" if target.class_ == "sd" else "Normal")
                + "("
                + ("" if target.class_ == "sd" else "0.0,")
                + scale_name
                + ")",
            )
            base_step = original[target.site]
            if base_step.morphism == "TruncatedNormal":
                replacement_args = {
                    argument.parameter: argument
                    for argument in prior.args or ()
                    if isinstance(argument, DrawArgNamed)
                }
                if target.class_ == "sd":
                    replacement_args["loc"] = DrawArgNamed(
                        parameter="loc", value=DrawArgScalar(value=0.0)
                    )
                args = tuple(
                    replacement_args.get(argument.parameter, argument)
                    if isinstance(argument, DrawArgNamed)
                    else argument
                    for argument in base_step.args or ()
                )
                replacement = base_step.with_(args=args)
            else:
                replacement = base_step.with_(morphism=prior.morphism, args=prior.args)
            replacements[target.site] = (
                _let(scale_name, scale),
                # brms (R2D2M2, horseshoe on sd): the standard deviation is
                # the shrinkage scale; coefficients keep their Normal kernel.
                _let(target.site, _var(scale_name))
                if target.class_ in SHRINKAGE_SCALE_CLASSES
                else restrict_prior_step(target, rule, replacement),
            )
            offset += target.width
    for site, rule in selected.items():
        if not isinstance(rule.distribution, str):
            continue
        prior = ordinary_prior_step(by_site[site], rule, draw=draw)
        replacements[site] = (
            original[site].with_(morphism=prior.morphism, args=prior.args),
        )
    draws = []
    for step in program.draws:
        if isinstance(step, SampleStep) and step.vars[0] in replacements:
            draws.extend(replacements[step.vars[0]])
        else:
            draws.append(step)
    if selected:
        draws = list(_ordered_steps(hyper + draws))
    updated = program.with_(draws=tuple(draws))
    return module.with_(
        statements=tuple(updated if item is program else item for item in statements)
    )


__all__ = []


def build_prior_targets(
    module,
    formula,
    family,
    predictors,
    laws,
    *,
    nonlinear_names=(),
    draw,
    thresholds_by=None,
    predictor_scopes=None,
    predictor_component_labels=None,
):
    """Combine explicit formula roles, group-block laws, and actual sample axes.

    Parameters
    ----------
    module : Module
        Existing emitted program before prior replacement.
    formula : Formula
        Response layout and retained authored labels.
    family : Family
        Response and auxiliary parameter metadata.
    predictors : Sequence
        Canonical predictor_items output with labels, prefixes, and designs.
    laws : Sequence[GroupBlockLaw]
        Laws of the group blocks of the same emission.
    nonlinear_names : Collection[str]
        Names of retained nonlinear parameter designs.
    draw : Callable
        Existing compositional prior parser.
    thresholds_by : str or None
        Original ordinal threshold grouping factor.
    predictor_scopes : Mapping or None
        Explicit response/dpar/nlpar coordinates for a joint program's
        existing predictor prefixes. Ordinary programs derive their scopes.
    predictor_component_labels : Mapping or None
        Free component labels for each explicitly indexed joint predictor.

    Returns
    -------
    tuple[PriorTarget, ...]
        One canonical record per actual overrideable sample site.
    """
    from quivers.formulas.formula import _qvr_name
    from quivers.qiec.families import family as semantic_family

    program = next(item for item in module.statements if isinstance(item, ProgramDecl))
    samples = {
        step.vars[0]: step for step in program.draws if isinstance(step, SampleStep)
    }
    sizes = finite_object_sizes(module)
    targets = {}
    scopes = {
        prefix: ({"nlpar": name} if name in nonlinear_names else {"dpar": name})
        for name, prefix, _ in predictors
    }
    scopes.update(predictor_scopes or {})
    labels = ()
    if family.response is not None:
        names = tuple(
            formula.response_layout.component_names
            or formula.response_layout.category_levels
        )
        if not names:
            names = tuple(str(i) for i in sorted(set(formula.response_values.tolist())))
        labels = tuple(str(names[i]) for i in family.response.coordinate_indices(names))

    def add(site, role, *, lower=None, upper=None, component_labels=None, **metadata):
        if site not in samples or site in targets:
            return
        step = samples[site]
        axes = ()
        if step.index is not None:
            if not isinstance(step.index, TypeName) or step.index.name not in sizes:
                raise ValueError(
                    f"prior target {site!r} has no retained nominal finite axis"
                )
            axes = (step.index.name,)
        # Improper flat laws live in a compilation-local catalog.
        event_rank = flat_event_rank(step.morphism)
        if event_rank is None:
            event_rank = semantic_family(step.morphism).event_rank
        targets[site] = PriorTarget(
            site=site,
            class_=role,
            axes=axes,
            shape=tuple(sizes[axis] for axis in axes),
            lower=lower,
            upper=upper,
            event_rank=event_rank,
            component_labels=(labels if component_labels is None else component_labels)
            if axes and role in {"b", "Intercept", "sd", "sds"}
            else (),
            **metadata,
        )

    for _name, prefix, predictor in predictors:
        for term in predictor.design_terms:
            # brms gp(): classes sdgp and lscale in place of one sds.
            if term.gp is not None:
                for site, role, coef in gp_prior_sites(term.gp, prefix):
                    add(
                        site,
                        role,
                        coef=coef,
                        lower=0.0,
                        component_labels=(predictor_component_labels or {}).get(prefix),
                        **scopes[prefix],
                    )
                continue
            add(
                prefix + (term.scale_group or term.qvr_name) + "_scale",
                "sds",
                coef=term.term,
                lower=0.0,
                component_labels=(predictor_component_labels or {}).get(prefix),
                **scopes[prefix],
            )
        for column in predictor.fixed_columns:
            # Centered and QR predictors sample transformed coordinates.
            site = sampled_site(
                prefix
                + ("intercept" if column.is_intercept else "beta_" + column.qvr_name),
                samples,
            )
            add(
                site,
                "Intercept" if column.is_intercept else "b",
                coef=column.name,
                component_labels=(predictor_component_labels or {}).get(prefix),
                **scopes[prefix],
            )
        # brms mo(): class b for bsp, class simo for each simplex.
        for site, role, coef in monotonic_prior_sites(predictor, prefix):
            add(
                site,
                role,
                coef=coef,
                component_labels=(predictor_component_labels or {}).get(prefix),
                **scopes[prefix],
            )
        for term in predictor.random_terms:
            site = f"{prefix}sigma_{_qvr_name(term.group)}_{_qvr_name(term.slope)}"
            add(
                site,
                "sd",
                coef=term.slope,
                group=term.group,
                lower=0.0,
                component_labels=(predictor_component_labels or {}).get(prefix),
                **scopes[prefix],
            )
    # brms gr(dist = "student") df and gr(by =) per-level correlations.
    for site, role, group, coef, lower in option_prior_sites(laws):
        add(site, role, group=group, coef=coef, lower=lower)
    for law in laws:
        correlation = law.correlation
        if not isinstance(correlation, LKJCorrelation):
            continue
        common = {tuple(scopes[member.prefix].items()) for member in law.members}
        scope = dict(next(iter(common))) if len(common) == 1 else {}
        add(correlation.site, "cor", group=law.group, **scope)
    for auxiliary in family.aux_params:
        default = draw(auxiliary.name, auxiliary.prior)
        support = semantic_family(default.morphism).support
        lower = 0.0 if support in {"positive", "nonnegative", "unit_interval"} else None
        upper = 1.0 if support == "unit_interval" else None
        # A positive parameter lies where its domain and its link's image do.
        ends = (
            lower,
            auxiliary.domain.lower or None,
            auxiliary.link.image.lower
            if lower is not None and auxiliary.link.image is not None
            else None,
        )
        lower = max((end for end in ends if end is not None), default=None)
        # Autocorrelation parameters take brms classes (ar, ma, cosy, ...).
        structural = autocor_prior_target(family, auxiliary.name)
        if structural is not None:
            role, lag, lower, upper = structural
            add(auxiliary.name, role, lower=lower, upper=upper, coef=lag)
            continue
        add(
            auxiliary.name,
            auxiliary.name,
            lower=lower,
            upper=upper,
            dpar=auxiliary.name,
        )
    if family.ordinal is not None:
        for site in (
            "cutpoint_log_spacing",
            "cutpoint_spacing",
            "cutpoint_offset",
            "cutpoints_raw",
            "cutpoint_log_spacing_mean",
            "cutpoint_log_spacing_scale",
            "threshold_location",
        ):
            add(
                site,
                "Intercept" if site == "cutpoints_raw" else site,
                group=thresholds_by,
            )
    if family.baseline is not None:
        add("sbhaz", "sbhaz")
    component_scale = family.component_scale
    if component_scale is not None:
        add("rescor", "rescor")
        # brms sigma<c>: the scalar scale of a category without a formula.
        for label in labels:
            name = scale_parameter(component_scale.name, label)
            add(name, name, lower=0.0, dpar=name)
    for site, role in autocor_matrix_targets(family):
        add(site, role)
    for latent in formula.latent_terms:
        for site, role, lower, upper in latent.prior_sites():
            add(site, role, lower=lower, upper=upper)
    return tuple(targets.values())


def ordinary_prior_step(target: PriorTarget, rule: PriorSpec, *, draw) -> SampleStep:
    """Parse a compositional prior and intersect its requested and target bounds.

    Parameters
    ----------
    target : PriorTarget
        Semantic role and admissible coefficient bounds.
    rule : PriorSpec
        Ordinary source prior and optional narrower bounds.
    draw : Callable
        Existing shared QVR distribution-call parser.

    Returns
    -------
    SampleStep
        Unplated source draw; callers retain the original nominal axes.

    Raises
    ------
    ValueError
        If bounds have empty intersection or the rule is a hierarchy.
    """
    if not isinstance(rule.distribution, str):
        raise ValueError("ordinary_prior_step requires a QVR distribution expression")
    return restrict_prior_step(target, rule, draw(target.site, rule.distribution))


def restrict_prior_step(
    target: PriorTarget, rule: PriorSpec, prior: SampleStep
) -> SampleStep:
    """Normalize a conditional prior on the intersection of its requested bounds.

    Parameters
    ----------
    target : PriorTarget
        Existing nominal coordinates and admissible coefficient domain.
    rule : PriorSpec
        Ordinary or hierarchical prior with optional scalar bounds.
    prior : SampleStep
        Existing parsed conditional kernel, retaining its original binders.

    Returns
    -------
    SampleStep
        The same kernel or its normalized measure restriction. Hyperprior
        marginals remain unchanged when the kernel depends on sampled heads.

    Raises
    ------
    ValueError
        If the requested bounds have empty intersection with the target domain.
    """
    lows = [v for v in (target.lower, rule.lower) if v is not None]
    highs = [v for v in (target.upper, rule.upper) if v is not None]
    lower = max(lows) if lows else None
    upper = min(highs) if highs else None
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f"prior bounds for {target.site!r} have empty intersection with its domain"
        )
    from quivers.qiec.families import family as semantic_family

    support = semantic_family(prior.morphism).support
    prior_lower = (
        0.0 if support in {"positive", "nonnegative", "unit_interval"} else None
    )
    prior_upper = 1.0 if support == "unit_interval" else None
    if lower is not None and prior_lower is not None and prior_lower >= lower:
        lower = None
    if upper is not None and prior_upper is not None and prior_upper <= upper:
        upper = None
    if lower is not None or upper is not None:
        arguments = [
            DrawArgNamed(
                parameter="base",
                value=DrawArgDist(family=prior.morphism, args=prior.args or ()),
            )
        ]
        arguments.extend(
            DrawArgNamed(parameter=name, value=DrawArgScalar(value=value))
            for name, value in [("low", lower), ("high", upper)]
            if value is not None
        )
        prior = prior.with_(morphism="Restrict", args=tuple(arguments))
    return prior
