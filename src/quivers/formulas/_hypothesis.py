"""Original formula-coordinate selection for shared retained-draw contrasts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quivers.diagnostics.contrast import _contrast_hypothesis
from quivers.formulas._group_law import group_coefficients

# Pathfinder draws report variational provenance.
from quivers.inference.pathfinder import draw_provenance

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def _formula_contrast_inputs(
    fit: BayesianFit, draws, *, class_, parameter, group, scope
):
    """Select authored labels using actual site and Gaussian block metadata.

    Parameters
    ----------
    fit : BayesianFit
        Fitted predictor, response, and emission layouts.
    draws : Mapping
        One jointly aligned posterior or prior stream.
    class_ : str or None
        Parameter role; None selects exact latent site names.
    parameter : str
        Existing distributional or nonlinear predictor label.
    group : str or None
        Original grouping factor, required for per-level scopes.
    scope : str
        Standard parameter, random adjustment, or combined coefficient scope.

    Returns
    -------
    tuple
        Selected tensors, nominal event dimensions, and original labels.

    Raises
    ------
    ValueError
        If selector metadata is missing, empty, incompatible, or ambiguous.
    """
    plan = fit.prediction_plan
    if plan is None or not plan.prior_targets:
        raise ValueError(
            "formula hypotheses require retained parameter-role metadata; use PosteriorAnalysis for explicit latent contrasts"
        )
    if scope not in {"standard", "ranef", "coef"}:
        raise ValueError("hypothesis scope must be standard, ranef, or coef")
    if not isinstance(parameter, str) or not parameter:
        raise ValueError("hypothesis parameter must name a fitted predictor")
    targets = plan.prior_targets
    samples, dimensions, coordinates = {}, {}, {}
    original_components = {str(label): label for label in fit.category_labels}

    def component_labels(target):
        """Read response components or ordinal transitions from fitted roles.

        Parameters
        ----------
        target : PriorTarget
            Original predictor and coordinate metadata.

        Returns
        -------
        tuple
            Original category labels or adjacent category pairs.
        """
        if target.component_labels:
            return tuple(
                original_components.get(label, label)
                for label in target.component_labels
            )
        for name, _, predictor in fit._predictor_items():
            if name != (target.nlpar or target.dpar):
                continue
            columns = (
                (
                    term.column
                    for term in predictor.random_terms
                    if term.group == target.group and term.slope == target.coef
                )
                if target.group is not None
                else (
                    column
                    for column in predictor.fixed_columns
                    if column.name == target.coef
                )
            )
            if any(
                column is not None and column.category_specific for column in columns
            ):
                categories = fit.category_labels
                if (
                    fit.family.ordinal is not None
                    and fit.family.ordinal.kind == "hurdle_cumulative"
                ):
                    categories = categories[1:]
                return tuple(zip(categories[:-1], categories[1:], strict=True))
        return ()

    def add(name, value, axes=(), labels=()):
        """Add one uniquely named event with its complete nominal layout.

        Parameters
        ----------
        name : str
            Original coefficient or parameter label.
        value : torch.Tensor
            Retained draws with leading chain and draw axes.
        axes, labels : tuple
            Complete nominal event axes and their original coordinate labels.

        Raises
        ------
        ValueError
            If names, event rank, or coordinate labels conflict.
        """
        if name in samples:
            raise ValueError(
                f"ambiguous hypothesis variable {name!r}; select a grouping factor or an exact latent scope"
            )
        if len(axes) != value.ndim - 2:
            raise ValueError(
                f"hypothesis variable {name!r} has no complete retained nominal event layout"
            )
        samples[name], dimensions[name] = value, tuple(axes)
        for axis, labels_ in zip(axes, labels, strict=True):
            labels_ = tuple(labels_)
            if axis in coordinates and coordinates[axis] != labels_:
                raise ValueError(
                    f"hypothesis axis {axis!r} has incompatible original labels"
                )
            coordinates[axis] = labels_

    def target_axes(target, value):
        """Complete an actual emitted site's batch and intrinsic event axes.

        Parameters
        ----------
        target : PriorTarget
            Original emitted parameter role and independent axes.
        value : torch.Tensor
            Retained constrained site values on explicit sample axes.

        Returns
        -------
        tuple
            Complete nominal axis names and original coordinate labels.
        """
        axes = list(target.axes)
        labels = [
            component_labels(target) or tuple(range(size)) for size in target.shape
        ]
        for i in range(len(axes), value.ndim - 2):
            axes.append(f"{target.site}_event_{i - len(target.axes)}")
            labels.append(tuple(range(value.shape[i + 2])))
        return tuple(axes), tuple(labels)

    if scope != "standard":
        if class_ != "b" or group is None:
            raise ValueError(
                "per-level coef and ranef hypotheses require class_='b' and group"
            )
        predictor, prefix = fit._formula_predictor(parameter)
        if group not in predictor.group_levels:
            raise ValueError(
                f"unknown grouping factor {group!r} for predictor {parameter!r}"
            )
        coefficients = group_coefficients(
            draws,
            tuple(
                law
                for law in plan.group_laws
                if law.group == group
                and any(member.prefix == prefix for member in law.members)
            ),
            fit.group_levels,
        )
        varying = {
            term.slope: coefficients[(prefix, group, term.slope)]
            for term in predictor.random_terms
            if term.group == group
        }
        fixed = fit._fixed_effects(draws, predictor, prefix) if scope == "coef" else {}
        population_targets = {
            target.coef: target
            for target in targets
            if target.class_ in {"b", "Intercept"}
            and (target.dpar == parameter or target.nlpar == parameter)
        }
        group_axis = f"group:{group}"
        for label in dict.fromkeys((*varying, *fixed)):
            adjustment = varying.get(label)
            population = fixed.get(label)
            if adjustment is None:
                assert population is not None
                value = population.unsqueeze(2).expand(
                    *population.shape[:2],
                    len(predictor.group_levels[group]),
                    *population.shape[2:],
                )
            else:
                value = (
                    adjustment
                    if population is None
                    else adjustment + population.unsqueeze(2)
                )
            axes, labels = (group_axis,), (predictor.group_levels[group],)
            if value.ndim > 3:
                target = population_targets.get(label)
                if adjustment is None:
                    assert target is not None
                    component_axis = target.axes[0] if target.axes else None
                else:
                    member = next(
                        member
                        for layout in plan.group_layouts
                        if layout.group == group
                        for member in layout.members
                        if member.prefix == prefix and member.slope == label
                    )
                    component_axis = member.axis
                    if target is None:
                        target = next(
                            target
                            for target in targets
                            if target.site == member.scale_site
                        )
                if component_axis is None:
                    raise ValueError(
                        "vector group hypotheses require retained component axes"
                    )
                axes += (component_axis,)
                labels += (component_labels(target),)
            add(label, value, axes, labels)
        return samples, dimensions, coordinates

    if class_ is None:
        if group is not None:
            raise ValueError("exact latent hypotheses do not accept a group filter")
        by_site = {target.site: target for target in targets}
        for name, value in draws.items():
            if name in by_site:
                axes, labels = target_axes(by_site[name], value)
            else:
                axes = tuple(f"{name}_event_{i}" for i in range(value.ndim - 2))
                labels = tuple(tuple(range(size)) for size in value.shape[2:])
            add(name, value, axes, labels)
        return samples, dimensions, coordinates

    for target in targets:
        if target.class_ not in ({"b", "Intercept"} if class_ == "b" else {class_}):
            continue
        if group is not None and target.group != group:
            continue
        if (
            class_ in {"b", "Intercept", "sd"}
            and (target.nlpar or target.dpar) != parameter
        ):
            continue
        if class_ == "cor" and (target.nlpar or target.dpar) not in (None, parameter):
            continue
        if target.site not in draws:
            continue
        value = draws[target.site]
        if class_ == "cor":
            layout = next(
                layout
                for layout in plan.group_layouts
                if layout.correlation_site == target.site
            )
            value = value @ value.transpose(-1, -2)
            parameters = {prefix: name for name, prefix, _ in fit._predictor_items()}
            by_site = {item.site: item for item in targets}
            coordinate_labels = []
            for member in layout.members:
                if member.axis is not None:
                    assert member.scale_site is not None
                    components = component_labels(by_site[member.scale_site])
                else:
                    components = (None,)
                coordinate_labels.extend(
                    (parameters[member.prefix], member.slope, component)
                    for component in components
                )
            labels = tuple(coordinate_labels)
            axis = f"correlation:{layout.identity}"
            first = layout.members[0]
            predictor = next(
                predictor
                for _, prefix, predictor in fit._predictor_items()
                if prefix == first.prefix
            )
            authored = next(
                term.correlation_id
                for term in predictor.random_terms
                if term.group == layout.group and term.slope == first.slope
            )
            assert authored is not None
            add(authored, value, (axis + ":row", axis + ":column"), (labels, labels))
        else:
            axes, labels = target_axes(target, value)
            add(target.coef or target.class_, value, axes, labels)
    if not samples:
        raise ValueError(
            f"hypothesis selector class_={class_!r}, parameter={parameter!r}, group={group!r} matches no retained parameters"
        )
    return samples, dimensions, coordinates


def _formula_hypothesis(
    fit: BayesianFit,
    expression,
    *,
    class_,
    parameter,
    group,
    scope,
    prior,
    num_samples,
    num_prior_samples,
    seed,
    **summary,
):
    """Collect each joint stream once, then use the canonical contrast evaluator.

    Parameters
    ----------
    fit : BayesianFit
        Fitted formula and retained metadata.
    expression : str or ContrastExpression
        Existing QVR expression over selected authored labels.
    class_, parameter, group, scope : str or None
        Parameter-role and per-level selectors.
    prior : bool
        Include corresponding joint prior draws and region evidence.
    num_samples, num_prior_samples : int or None
        Posterior and prior stream draw counts.
    seed : int
        Isolated retained-draw seed.
    **summary
        Existing hypothesis summary options.

    Returns
    -------
    HypothesisSummary
        Shared strict-region posterior and optional encompassing-prior evidence.

    Raises
    ------
    ValueError, TypeError
        If selectors, stream counts, or the shared contrast contract disagree.
    """
    if not isinstance(prior, bool):
        raise TypeError("prior must be Boolean")
    if summary.get("point") is not None and not prior:
        raise ValueError("point-density evidence requires prior=True")
    if num_prior_samples is not None and not prior:
        raise ValueError("num_prior_samples requires prior=True")
    options = dict(class_=class_, parameter=parameter, group=group, scope=scope)
    samples, dims, coords = _formula_contrast_inputs(
        fit, fit.draws(num_samples=num_samples, seed=seed), **options
    )
    prior_samples = None
    if prior:
        view = fit.prior_view(num_samples=num_prior_samples, seed=(seed + 1) % 2**64)
        prior_samples, prior_dims, prior_coords = _formula_contrast_inputs(
            view, view.draws(), **options
        )
        if prior_dims != dims or prior_coords != coords:
            raise ValueError("prior and posterior formula coordinates disagree")
    kind = draw_provenance(fit.posterior)
    return _contrast_hypothesis(
        samples,
        expression,
        prior=prior_samples,
        dims=dims,
        coords=coords,
        kind=kind,
        **summary,
    )


__all__: list[str] = []
