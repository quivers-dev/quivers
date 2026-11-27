"""Formula refits reuse the canonical fitting signature and retained inputs."""

from collections.abc import Callable, Mapping
import inspect
from types import MappingProxyType

import narwhals as nw
from narwhals.typing import IntoDataFrame

from quivers.modeling.refit import RefitRecipe, SnapshotPolicy, _snapshot
from quivers.formulas._formula_options import expand_dot_formula
from quivers.formulas._ordinal_coding import zero_coded_columns


def capture_recipe[Result](
    fitter: Callable[..., Result],
    *,
    data: IntoDataFrame,
    arguments: Mapping[str, object],
    snapshot_policy: SnapshotPolicy,
) -> RefitRecipe[IntoDataFrame, Mapping[str, object], Result]:
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
    ) -> Result:
        # Module state and initial values are copied before every fit, so a
        # refit cannot train the retained initial predictor or another result.
        values = _snapshot(dict(settings))
        values["data"] = frame
        values.setdefault("snapshot_policy", snapshot_policy)
        if seed is not None:
            values["seed"] = seed
        bound = signature.bind(**values)
        # An integer response keeps the category coding of the original rows.
        with zero_coded_columns(nw.from_native(data, eager_only=True).to_pandas()):
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


def update_fit[Result](
    recipe: RefitRecipe[IntoDataFrame, Mapping[str, object], Result] | None,
    data: IntoDataFrame | None,
    overrides: Mapping[str, object],
) -> Result:
    """Merge explicit overrides into a recipe without another fitting path."""
    if recipe is None:
        raise ValueError("this fit has no retained refit recipe")
    # R update.formula: "." stands for the fitted sides.
    if isinstance(overrides.get("formula"), str) and isinstance(
        recipe.method.get("formula"), str
    ):
        overrides = {
            **overrides,
            "formula": expand_dot_formula(
                str(recipe.method["formula"]), str(overrides["formula"])
            ),
        }
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
    # Pathfinder runs no MCMC or SVI iterations, and its arguments apply to
    # it alone.
    switched = {overrides.get("method"), overrides.get("algorithm")} - {None}
    if "pathfinder" in switched:
        for name in ("num_warmup", "num_samples", "num_steps", "control"):
            if name not in overrides:
                options[name] = None
        if "thin" not in overrides:
            options["thin"] = 1
    elif switched and "pathfinder" not in overrides:
        options["pathfinder"] = None
    return recipe.run(data, method=options)
