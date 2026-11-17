"""Formula refits reuse the canonical fitting signature and retained inputs."""

from collections.abc import Callable, Mapping
import inspect
from types import MappingProxyType
from typing import TYPE_CHECKING

from narwhals.typing import IntoDataFrame

from quivers.modeling.refit import RefitRecipe, SnapshotPolicy, _snapshot

if TYPE_CHECKING:
    from quivers.formulas._fit import BayesianFit


def capture_recipe(
    fitter: Callable[..., BayesianFit],
    *,
    data: IntoDataFrame,
    arguments: Mapping[str, object],
    snapshot_policy: SnapshotPolicy,
) -> RefitRecipe[IntoDataFrame, Mapping[str, object], BayesianFit]:
    """Snapshot original rows and every canonical fitting argument before training."""
    signature = inspect.signature(fitter)
    options = _snapshot(
        {
            name: arguments[name]
            for name in signature.parameters
            if name not in {"data", "snapshot_policy"}
        }
    )

    def run(
        frame: IntoDataFrame, settings: Mapping[str, object], seed: int | None
    ) -> BayesianFit:
        # Module state and initial values are copied before every fit, so a
        # refit cannot train the retained initial predictor or another result.
        values = _snapshot(dict(settings))
        values["data"] = frame
        values.setdefault("snapshot_policy", snapshot_policy)
        if seed is not None:
            values["seed"] = seed
        bound = signature.bind(**values)
        return fitter(*bound.args, **bound.kwargs)

    return RefitRecipe(
        data,
        MappingProxyType(options),
        run,
        snapshot_policy=snapshot_policy,
        provenance={
            "kind": "formula",
            "schema": "qvr-refit/v1",
            "formula": str(options["formula"]),
        },
    )


def update_fit(
    fitted: BayesianFit, data: IntoDataFrame | None, overrides: Mapping[str, object]
) -> BayesianFit:
    """Merge explicit overrides into a recipe without another fitting path."""
    recipe = fitted.refit_recipe
    if recipe is None:
        raise ValueError("this fit has no retained refit recipe")
    options = {**recipe.method, **overrides}
    if "method" in overrides and overrides["method"] != recipe.method["method"]:
        inactive = (
            ("num_warmup", "num_samples", "num_chains")
            if overrides["method"] == "svi"
            else ("num_steps",)
        )
        for name in inactive:
            if name not in overrides:
                options[name] = None
    return recipe.run(data, method=options)
