"""Joint (multivariate) formula artifacts over the shared fitted-artifact storage.

A joint artifact encodes the original rows, the canonical `fit` arguments
(including the `MultivariateFormula` as a declarative record), the trained
program state, and the posterior. Restoration rebuilds every response design
through the same `_prepare_joint` used for fitting, checks the regenerated
source and bindings against the saved ones, and runs no inference.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import cast

import torch
from narwhals.typing import IntoDataFrame

from quivers.dsl.emit import module_to_source
from quivers.formulas._fit import _InferenceResult, fit
from quivers.formulas._joint_fit import MultivariateFit, _prepare_joint
from quivers.formulas._data_options import Knots, formula_data_options
from quivers.formulas._multivariate import MultivariateFormula
from quivers.formulas._priors import normalize_priors
from quivers.formulas._refit import capture_recipe
from quivers.inference import MCMCResult, PriorDraws
from quivers.modeling._artifact_codec import (
    ArtifactContext,
    _Codec,
    _module_state,
    _restore_module,
)
from quivers.modeling.refit import RefitRecipe
from quivers.modeling.artifact import (
    _posterior_state,
    _restore_posterior,
    _validate_recorded,
)

__all__: list[str] = []

#: Manifest kind of a joint formula artifact.
JOINT_KIND = "formula_joint"


def joint_state(fitted: MultivariateFit, codec: _Codec) -> dict:
    """Capture a joint fit's original inputs and trained state.

    Parameters
    ----------
    fitted : MultivariateFit
        Trained joint fit with its canonical recipe.
    codec : _Codec
        Declarative encoder with formula records enabled.

    Returns
    -------
    dict
        Manifest fields: data, options, source, program state, observations,
        posterior, prior, and losses.

    Raises
    ------
    ValueError
        If the fit retains no canonical recipe or original-row snapshot.
    """
    recipe = fitted.refit_recipe
    if (
        recipe is None
        or recipe.provenance.get("kind") != "formula"
        or recipe.provenance.get("schema") != "qvr-refit/v1"
    ):
        raise ValueError(
            "joint formula artifacts require the canonical original-data refit recipe"
        )
    if fitted.preparation_data is None:
        raise ValueError("joint formula artifacts require the retained training rows")
    return {
        "kind": JOINT_KIND,
        "data": codec.encode(fitted.preparation_data),
        "recipe_data": codec.encode(recipe.data)
        if recipe.snapshot_policy == "reference"
        else None,
        "options": codec.encode(dict(recipe.method)),
        "ownership": recipe.snapshot_policy,
        "source": fitted.qvr_source,
        "program": codec.encode(_module_state(fitted.program)),
        "observations": codec.encode(dict(fitted.observations)),
        "posterior": _posterior_state(fitted.posterior, codec),
        "prior": codec.encode(fitted.prior),
        "sample_prior": fitted.sample_prior,
        "losses": codec.encode(fitted.losses),
        "criteria": codec.encode(dict(fitted.criteria)),
    }


def restore_joint(manifest: dict, codec: _Codec) -> MultivariateFit:
    """Rebuild a joint fit without inference.

    Cost: one preparation and compilation of the joint program, plus one
    strict replay of the restored posterior.

    Parameters
    ----------
    manifest : dict
        Verified joint artifact manifest.
    codec : _Codec
        Checked tensor and declarative decoder.

    Returns
    -------
    MultivariateFit
        Rebuilt designs, checked program, and restored posterior.

    Raises
    ------
    TypeError
        If the saved formula is not a joint formula or bindings are not tensors.
    ValueError
        If the regenerated source or a binding disagrees with the saved one.
    """
    data = codec.decode(manifest["data"])
    options = codec.decode(manifest["options"])
    bound = inspect.signature(fit).bind(
        data=data, snapshot_policy=manifest["ownership"], **options
    )
    bound.apply_defaults()
    specification = bound.arguments["formula"]
    if not isinstance(specification, MultivariateFormula):
        raise TypeError("a joint artifact must retain a MultivariateFormula")
    recipe = cast(
        RefitRecipe[IntoDataFrame, Mapping[str, object], MultivariateFit],
        capture_recipe(
            fit,
            data=codec.decode(manifest["recipe_data"])
            if manifest["ownership"] == "reference"
            else data,
            arguments=bound.arguments,
            snapshot_policy=manifest["ownership"],
        ),
    )
    preparation = recipe._data if manifest["ownership"] == "clone" else data
    arguments = bound.arguments
    # brms knots of the saved fit.
    with formula_data_options(
        cast(Knots | None, arguments.get("knots")),
        cast(bool, arguments.get("drop_unused_levels", True)),
    ):
        composition, program = _prepare_joint(
            specification,
            preparation,
            fixed_prior=arguments["fixed_prior"],
            random_scale_prior=arguments["random_scale_prior"],
            correlation_prior=arguments["correlation_prior"],
            group_covariances=arguments["group_covariances"],
            priors=arguments["priors"],
            reparameterize=arguments["reparameterize"],
            data2=arguments.get("data2"),
            caller="load_fit",
        )
    prepared, emission = composition.prepared, composition.emission
    observations = composition.observations
    if module_to_source(emission.module) != manifest["source"]:
        raise ValueError("artifact joint preparation changed the checked source")
    saved = codec.decode(manifest["observations"])
    if not isinstance(saved, dict) or any(
        not isinstance(value, torch.Tensor) for value in saved.values()
    ):
        raise TypeError("artifact joint observations require named tensors")
    if saved.keys() != observations.keys():
        raise ValueError("artifact joint binding changed its observation names")
    for name, value in saved.items():
        if not torch.equal(observations[name].to(value), value):
            raise ValueError(f"artifact joint preparation changed binding {name!r}")
    _restore_module(program, codec.decode(manifest["program"]))
    context = ArtifactContext(
        program, torch.zeros(1, 1, dtype=torch.long), observations
    )
    posterior = _restore_posterior(manifest["posterior"], codec, context)
    prior = codec.decode(manifest["prior"])
    if prior is not None:
        if not isinstance(prior, MCMCResult | PriorDraws):
            raise TypeError("artifact retained prior must be a recorded result")
        _validate_recorded(prior, context)
    result = _InferenceResult(
        posterior, codec.decode(manifest["losses"]), prior, manifest["sample_prior"]
    )
    return MultivariateFit(
        specification=specification,
        program=program,
        posterior=result.posterior,
        observations=observations,
        emission=emission,
        prepared=prepared,
        losses=result.losses,
        prior=result.prior,
        sample_prior=result.sample_prior,
        priors=normalize_priors(arguments["priors"]),
        fixed_prior=arguments["fixed_prior"],
        random_scale_prior=arguments["random_scale_prior"],
        correlation_prior=arguments["correlation_prior"],
        group_covariances=dict(arguments["group_covariances"] or {}),
        reparameterize=arguments["reparameterize"],
        preparation_data=preparation,
        refit_recipe=recipe,
        missing=composition.missing,
        missing_specs=composition.specs,
        data2=dict(arguments.get("data2") or {}),
        criteria=codec.decode(manifest["criteria"]),
    )
