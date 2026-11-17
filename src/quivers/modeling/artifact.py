"""Durable fitted models reconstructed from checked specifications and tensor state.

Artifacts contain no live program, dataframe evaluator, or executable pickle.
Loading rebuilds checked source or formula preparation and restores its state;
it never fits a model or draws a replacement posterior. Registered factories are explicit
caller-provided reconstruction code, not imports selected by the file.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quivers.formulas import BayesianFit as _BayesianFit
    from quivers.formulas import MultivariateFit as _MultivariateFit
else:
    _BayesianFit = object
    _MultivariateFit = object


import torch

from quivers._atomic import atomic_output
from quivers.inference import AutoNormalGuide, Guide, MCMCResult, PriorDraws
from quivers.inference.mcmc.checkpoint import tensor_digest
from quivers.modeling._artifact_codec import (
    ArtifactContext,
    ArtifactFactory,
    _Codec,
    _module_state,
    _restore_module,
)
from quivers.modeling.model import Model, ModelFit

__all__ = ["ArtifactContext", "ArtifactFactory", "save_fit", "load_fit"]

_FORMAT = "qvr-fitted-artifact/v1"


def _site_layout(registry) -> dict[str, object]:
    """Describe checked latent geometry independently of its tensor values.

    Parameters
    ----------
    registry : LatentRegistry
        Canonical latent layout used by the guide.

    Returns
    -------
    dict[str, object]
        Named support and event geometry used to verify reconstruction.
    """
    return {
        name: {
            "constrained": site.constrained_shape,
            "unconstrained": site.unconstrained_shape,
            "event": site.event_shape,
            "plate": site.plate_shape,
            "support": str(site.support),
        }
        for name, site in registry.sites.items()
    }


def _posterior_state(
    posterior: MCMCResult | PriorDraws | Guide, codec: _Codec
) -> dict[str, object]:
    """Capture retained posterior state without drawing from a guide.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws or Guide
        Fitted posterior object.
    codec : _Codec
        Declarative value encoder with explicit extension factories.

    Returns
    -------
    dict[str, object]
        Ordinary result fields, or guide parameters and deterministic anchors.
    """
    if type(posterior) is AutoNormalGuide:
        registry = posterior.registry
        anchors = registry.trace_values(
            {
                name: site.constrain(posterior._loc(name).detach())[0]
                for name, site in registry.sites.items()
            },
            1,
        )
        return {
            "kind": "normal",
            "state": codec.encode(_module_state(posterior)),
            "anchors": codec.encode(anchors),
            "layout": codec.encode(_site_layout(registry)),
            "observed": sorted(registry.observed_names),
        }
    result = {"kind": "value", "value": codec.encode(posterior)}
    if isinstance(posterior, Guide):
        result["layout"] = codec.encode(_site_layout(posterior.registry))
        result["observed"] = sorted(posterior.registry.observed_names)
    return result


def _restore_posterior(payload: dict, codec: _Codec, context: ArtifactContext):
    """Rebuild a guide through strict fixed-value replay or a result constructor.

    Parameters
    ----------
    payload : dict
        Versioned posterior specification.
    codec : _Codec
        Checked decoder.
    context : ArtifactContext
        Recompiled model and original bindings.

    Returns
    -------
    MCMCResult or PriorDraws or Guide
        Original posterior family with restored state.

    Raises
    ------
    ValueError, TypeError
        If reconstruction changes latent identities or their event geometry.
    """
    if payload["kind"] == "value":
        value = codec.decode(payload["value"], context)
        if not isinstance(value, MCMCResult | PriorDraws | Guide):
            raise TypeError(
                "artifact posterior must reconstruct a supported posterior class"
            )
        if isinstance(value, MCMCResult | PriorDraws):
            _validate_recorded(value, context)
        elif (
            value.registry.model is not context.program
            or _site_layout(value.registry) != codec.decode(payload["layout"])
            or value.registry.observed_names != frozenset(payload["observed"])
        ):
            raise ValueError(
                "artifact guide factory changed the checked model or latent layout"
            )
        return value
    if payload["kind"] != "normal":
        raise ValueError("unknown artifact posterior contract")
    from quivers.inference.registry import LatentRegistry

    anchors = codec.decode(payload["anchors"])
    recorded = _strict_trace(context, anchors)
    found = set(anchors)
    available = {
        name
        for name, site in recorded.sites.items()
        if site.morphism is not None and not site.is_deterministic
    }
    if not found <= available:
        raise ValueError(
            "artifact latent addresses were not reached during strict replay"
        )
    recorded = recorded.with_(
        sites={
            name: site.with_(is_observed=False) if name in found else site
            for name, site in recorded.sites.items()
        }
    )
    registry = LatentRegistry.from_trace(context.program, recorded)
    # A static guide may list checked external inputs as observed bindings;
    # trace registries list only scored random sites. Preserve that original
    # public registry contract, but never introduce an unbound observation.
    saved_observed = frozenset(payload["observed"])
    if not registry.observed_names <= saved_observed or not (
        saved_observed - registry.observed_names
    ) <= set(context.observations):
        raise ValueError("artifact observation roles changed during strict replay")
    registry = LatentRegistry(registry.sites, saved_observed, context.program)
    if _site_layout(registry) != codec.decode(
        payload["layout"]
    ) or registry.observed_names != frozenset(payload["observed"]):
        raise ValueError(
            "artifact latent support, observation roles, or event geometry changed"
        )
    guide = AutoNormalGuide.from_registry(registry)
    _restore_module(guide, codec.decode(payload["state"]))
    return guide


def _strict_trace(context: ArtifactContext, anchors: dict[str, torch.Tensor]):
    """Replay one fully specified execution without stochastic fallback.

    Parameters
    ----------
    context : ArtifactContext
        Checked native program and original data binding.
    anchors : dict[str, torch.Tensor]
        A fixed value for every reached latent address.

    Returns
    -------
    Trace
        Canonical trace with original observed and latent site roles.

    Raises
    ------
    ValueError
        If an anchor is not reached or a random request has no saved value.
    """
    from quivers.effects.clamp import ClampHandler
    from quivers.inference import trace
    from quivers.inference.predictive import _require_replayed_sites
    from quivers.qiec.builtins import MissingValuePolicy

    observed = context.program.observed_names
    if set(anchors) & set(context.observations) or any(
        name in observed or name.rpartition("@")[0] in observed for name in anchors
    ):
        raise ValueError(
            "artifact posterior labels overlap checked observations or external inputs"
        )
    with torch.no_grad(), ClampHandler(anchors, missing=MissingValuePolicy.ERROR):
        recorded = trace(context.program, context.input, dict(context.observations))
    _require_replayed_sites(recorded, anchors)
    if any(
        recorded.sites[name].is_deterministic or recorded.sites[name].morphism is None
        for name in anchors
    ):
        raise ValueError("artifact posterior labels name deterministic bindings")
    return recorded


def _validate_recorded(
    posterior: MCMCResult | PriorDraws, context: ArtifactContext
) -> None:
    """Reuse retained-axis validation and a strict replay for recorded draws.

    Parameters
    ----------
    posterior : MCMCResult or PriorDraws
        Restored constrained tensors, including discrete carriers.
    context : ArtifactContext
        Checked model and observations used by those tensors.

    Raises
    ------
    ValueError
        If axes, sample names, or reached sites disagree with the model.
    """
    from quivers.inference._posterior import constrained_draws
    from quivers.inference.registry import LatentRegistry

    try:
        registry = LatentRegistry.from_model(context.program, set(context.observations))
    except ValueError, NotImplementedError:
        registry = None
    samples = constrained_draws(
        posterior, context.input, num_samples=1, registry=registry
    )
    single = {name: value[0, 0] for name, value in samples.items()}
    anchors = registry.trace_values(single, 1) if registry is not None else single
    _strict_trace(context, anchors)


def save_fit(
    fitted: ModelFit | _BayesianFit | _MultivariateFit,
    path: str | Path,
    *,
    factories: Mapping[str, ArtifactFactory] | None = None,
) -> None:
    """Atomically save a fitted model using JSON and safe tensor payloads.

    Parameters
    ----------
    fitted : ModelFit or BayesianFit or MultivariateFit
        Original checked model, posterior, deterministic module state, and recipe.
    path : str or pathlib.Path
        Destination checkpoint. An existing file is replaced only after success.
    factories : Mapping[str, ArtifactFactory] or None
        Explicit contracts for external modules, custom guides, or method objects.
        No arbitrary Python object or import path is serialized.

    Raises
    ------
    TypeError, ValueError
        If the fit or one of its configurations has no portable contract.
    """
    from quivers import __version__

    if isinstance(fitted, ModelFit):
        codec = _Codec(factories)
        manifest = _source_state(fitted, codec)
    else:
        if not type(fitted).__module__.startswith("quivers.formulas."):
            raise TypeError(
                "save_fit requires a source ModelFit or formula BayesianFit"
            )
        from quivers.formulas._fit import BayesianFit
        from quivers.formulas._artifact import _formula_state

        # Joint (multivariate) formula fits share this storage format.
        from quivers.formulas._joint_fit import MultivariateFit
        from quivers.formulas._joint_artifact import joint_state

        if isinstance(fitted, MultivariateFit):
            codec = _Codec(factories, formula=True)
            manifest = joint_state(fitted, codec)
        elif not isinstance(fitted, BayesianFit):
            raise TypeError(
                "save_fit requires a source ModelFit or formula BayesianFit"
            )
        else:
            codec = _Codec(factories, formula=True)
            manifest = _formula_state(fitted, codec)
    manifest.update(
        format=_FORMAT, producer={"package": "quivers", "version": __version__}
    )
    text = json.dumps(manifest, sort_keys=True, allow_nan=False)
    payload = {
        "manifest": text,
        "manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tensors": codec.tensors,
        "digests": {
            name: tensor_digest(value) for name, value in codec.tensors.items()
        },
    }
    with atomic_output(Path(path)) as temporary:
        torch.save(payload, temporary)


def _source_state(fitted: ModelFit, codec: _Codec) -> dict:
    """Capture a source fit through its original checked binding and recipe.

    Parameters
    ----------
    fitted : ModelFit
        Bound source result.
    codec : _Codec
        Closed declarative encoder.

    Returns
    -------
    dict
        Source-specific manifest fields, without live executable objects.
    """
    bound = fitted.model
    recipe = fitted.refit_recipe
    canonical_recipe = (
        recipe is not None
        and recipe.provenance.get("kind") == "source"
        and recipe.provenance.get("schema") == "qvr-refit/v1"
    )
    recipe_state = None
    if recipe is not None:
        recipe_state = (
            {
                "kind": "canonical",
                "data": None
                if recipe.snapshot_policy == "reference"
                else codec.encode(recipe.data),
                "method": codec.encode(recipe.method),
                "ownership": recipe.snapshot_policy,
                "provenance": dict(recipe.provenance),
            }
            if canonical_recipe
            else {"kind": "factory", "value": codec.encode(recipe)}
        )
    source, entry = bound.specification.source, bound.specification.entry
    if canonical_recipe:
        assert recipe is not None
        source, entry = recipe.provenance["source"], recipe.provenance["entry"]
    return {
        "kind": "source",
        "source": source,
        "families": codec.encode(bound.specification.families),
        "entry": entry,
        "data": codec.encode(dict(bound.data)),
        "input": codec.encode(bound.input),
        "static": codec.encode(dict(bound.template_arguments)),
        "seed": bound.compilation_seed,
        "program": codec.encode(_module_state(bound.require_tensor_program())),
        "posterior": _posterior_state(fitted.posterior, codec),
        "prior": codec.encode(fitted.prior),
        "sample_prior": fitted.sample_prior,
        "losses": codec.encode(fitted.losses),
        "recipe": recipe_state,
    }


def load_fit(
    path: str | Path,
    *,
    factories: Mapping[str, ArtifactFactory] | None = None,
    map_location: str | torch.device = "cpu",
) -> ModelFit | _BayesianFit | _MultivariateFit:
    """Reconstruct a fit without optimization or replacement posterior draws.

    Parameters
    ----------
    path : str or pathlib.Path
        Fitted artifact produced by save_fit.
    factories : Mapping[str, ArtifactFactory] or None
        Explicit trusted reconstructors required by the artifact.
    map_location : str or torch.device
        Tensor destination, passed to torch.load with weights_only=True.

    Returns
    -------
    ModelFit or BayesianFit or MultivariateFit
        Independent checked binding, posterior, and reconstructed live recipe.
        Loading a reference-owned recipe shares its newly loaded data, not the
        original process's memory. Saving always captures values at write time.

    Raises
    ------
    ValueError, TypeError, RuntimeError
        If the format, tensor integrity, checked source, or factory contract fails.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "manifest",
        "manifest_sha256",
        "tensors",
        "digests",
    }:
        raise ValueError("not a fitted artifact payload")
    text = payload["manifest"]
    if (
        not isinstance(text, str)
        or hashlib.sha256(text.encode()).hexdigest() != payload["manifest_sha256"]
    ):
        raise ValueError("artifact manifest integrity check failed")
    manifest = json.loads(text)
    if manifest.get("format") != _FORMAT or manifest.get("kind") not in {
        "source",
        "formula",
        "formula_joint",
    }:
        raise ValueError("unsupported fitted artifact format or kind")
    tensors, digests = payload["tensors"], payload["digests"]
    if (
        not isinstance(tensors, dict)
        or not isinstance(digests, dict)
        or set(tensors) != set(digests)
    ):
        raise ValueError("artifact tensor index is malformed")
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or tensor_digest(value) != digests[name]:
            raise ValueError(f"artifact tensor integrity check failed for {name!r}")
    codec = _Codec(
        factories, tensors, formula=manifest["kind"] in {"formula", "formula_joint"}
    )
    from quivers._rng import seeded_rng

    with seeded_rng(0, devices=[value.device for value in tensors.values()]):
        if manifest["kind"] == "source":
            return _restore_source(manifest, codec)
        if manifest["kind"] == "formula_joint":
            from quivers.formulas._joint_artifact import restore_joint

            return restore_joint(manifest, codec)
        from quivers.formulas._artifact import _restore_formula

        return _restore_formula(manifest, codec)


def _restore_source(manifest: dict, codec: _Codec) -> ModelFit:
    """Rebuild checked source state, posterior, and its declarative refit recipe.

    Parameters
    ----------
    manifest : dict
        Verified format manifest.
    codec : _Codec
        Tensor and declarative configuration decoder.

    Returns
    -------
    ModelFit
        Fresh independent runtime with the saved posterior family and state.

    Raises
    ------
    TypeError, ValueError
        If a retained posterior, prior, or recipe reconstruction is incompatible.
    """
    bound = Model(
        manifest["source"],
        entry=manifest["entry"],
        families=codec.decode(manifest.get("families")),
    ).bind(
        codec.decode(manifest["data"]),
        parameters=codec.decode(manifest["static"]),
        input=codec.decode(manifest["input"]),
        seed=manifest["seed"],
    )
    _restore_module(bound.require_tensor_program(), codec.decode(manifest["program"]))
    context = ArtifactContext(
        bound.require_tensor_program(), bound.input, bound.tensor_data()
    )
    posterior = _restore_posterior(manifest["posterior"], codec, context)
    prior = codec.decode(manifest["prior"])
    if prior is not None:
        if not isinstance(prior, MCMCResult | PriorDraws):
            raise TypeError("artifact retained prior must be a recorded result")
        _validate_recorded(prior, context)
    recipe = None
    if manifest["recipe"] is not None:
        from quivers.modeling._refit import source_recipe
        from quivers.modeling.refit import RefitRecipe

        spec = manifest["recipe"]
        if spec["kind"] == "factory":
            recipe = codec.decode(spec["value"], context)
            if not isinstance(recipe, RefitRecipe):
                raise TypeError("artifact recipe factory must reconstruct RefitRecipe")
        elif spec["kind"] == "canonical":
            provenance = spec["provenance"]
            retained = (
                bound
                if spec["ownership"] == "reference"
                else bound.with_(data=codec.decode(spec["data"]))
            )
            recipe = source_recipe(
                retained,
                codec.decode(spec["method"]),
                snapshot_policy=spec["ownership"],
                sample_prior=manifest["sample_prior"],
                num_prior_samples=provenance["num_prior_samples"],
                prior_seed=provenance["prior_seed"],
            )
        else:
            raise ValueError("unsupported artifact refit recipe contract")
    return ModelFit(
        model=bound,
        posterior=posterior,
        sample_prior=manifest["sample_prior"],
        prior=prior,
        losses=codec.decode(manifest["losses"]),
        refit_recipe=recipe,
    )
