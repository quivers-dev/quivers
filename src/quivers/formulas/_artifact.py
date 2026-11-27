"""Formula artifacts reuse retained configuration and the canonical native binding.

Only declarative dataframe columns and formula options are encoded. Live design
evaluators are rebuilt by the same preparation used for fitting and prediction.
"""

from collections.abc import Mapping
import inspect
from typing import cast
from narwhals.typing import IntoDataFrame

import narwhals as nw
import pandas as pd
import torch

from quivers.continuous.programs import MonadicProgram
from quivers.dsl.emit import module_to_source
from quivers.formulas._fit_core import FormulaFit
from quivers.formulas._program_kernel import program_arguments
from quivers.formulas._fit import (
    BayesianFit,
    _InferenceResult,
    _assemble_fit,
    _prepare_fit,
    _prepare_formula,
    fit,
)
from quivers.formulas._refit import capture_recipe
from quivers.inference import MCMCResult, PriorDraws
from quivers.modeling._artifact_codec import (
    ArtifactContext,
    _Codec,
    _module_state,
    _restore_module,
)
from quivers.modeling.artifact import (
    _posterior_state,
    _restore_posterior,
    _validate_recorded,
)
from quivers.modeling.refit import RefitRecipe

__all__: list[str] = []


def _column(series: pd.Series) -> dict:
    """Encode a scalar column with its exact categorical or numeric dtype.

    Parameters
    ----------
    series : pandas.Series
        One named training column, including nullable or categorical values.

    Returns
    -------
    dict
        Values and declarative dtype metadata.

    Raises
    ------
    TypeError
        If the column has no supported scalar dtype contract."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return {
            "values": series.cat.codes.tolist(),
            "categories": series.cat.categories.tolist(),
            "ordered": dtype.ordered,
        }
    if getattr(dtype, "kind", None) not in {"b", "i", "u", "f", "O", "U"}:
        raise TypeError(
            f"no artifact dataframe dtype contract for {dtype}; register an explicit ArtifactFactory"
        )
    values = series.tolist()
    # Nullable scalar sentinels are data, not executable pandas objects.
    values = [None if value is pd.NA or value is pd.NaT else value for value in values]
    return {"values": values, "dtype": str(dtype)}


def _encode_frame(value: object) -> dict:
    """Use the same Narwhals dataframe conversion as formula preparation.

    Parameters
    ----------
    value : object
        Native pandas, Polars, or Arrow dataframe accepted by formula fitting.

    Returns
    -------
    dict
        Ordered scalar columns, categorical levels, and original row index.

    Raises
    ------
    TypeError, ValueError
        If the frame contains duplicate columns or unsupported scalar values."""
    frame = nw.from_native(cast(IntoDataFrame, value), eager_only=True).to_pandas()
    if not frame.columns.is_unique:
        raise ValueError("artifact dataframe columns must be unique")
    return {
        "columns": list(frame.columns),
        "data": [_column(cast(pd.Series, frame[name])) for name in frame.columns],
        "index": (
            {
                "kind": "multi",
                "levels": [_column(pd.Series(level)) for level in frame.index.levels],
                "codes": [codes.tolist() for codes in frame.index.codes],
                "names": list(frame.index.names),
            }
            if isinstance(frame.index, pd.MultiIndex)
            else {
                "kind": "index",
                "values": _column(pd.Series(frame.index)),
                "name": frame.index.name,
            }
        ),
    }


def _decode_frame(spec: dict) -> pd.DataFrame:
    """Rebuild safe scalar columns without checkpoint-selected Python imports.

    Parameters
    ----------
    spec : dict
        Declarative column and row-index specification.

    Returns
    -------
    pandas.DataFrame
        Independent values preserving categorical order and exact integers.

    Raises
    ------
    ValueError, TypeError
        If the column schema or dtype is unsupported."""
    if len(spec["columns"]) != len(spec["data"]) or len(set(spec["columns"])) != len(
        spec["columns"]
    ):
        raise ValueError("artifact dataframe column schema is malformed")
    frame = pd.DataFrame(
        {
            name: _decode_column(column)
            for name, column in zip(spec["columns"], spec["data"], strict=True)
        }
    )
    index = spec["index"]
    if index["kind"] == "multi":
        frame.index = pd.MultiIndex(
            levels=[pd.Index(_decode_column(level)) for level in index["levels"]],
            codes=index["codes"],
            names=index["names"],
        )
    elif index["kind"] == "index":
        frame.index = pd.Index(_decode_column(index["values"]), name=index["name"])
    else:
        raise ValueError("unsupported artifact dataframe index contract")
    return frame


def _decode_column(column: dict) -> pd.Series:
    """Rebuild one checked scalar column for dataframe values or index levels.

    Parameters
    ----------
    column : dict
        Declarative values with numeric, nullable, or categorical dtype metadata.

    Returns
    -------
    pandas.Series
        Independent scalar values preserving the original dtype.

    Raises
    ------
    ValueError, TypeError
        If the dtype or categorical codes are invalid.
    """
    if "categories" in column:
        return pd.Series(
            pd.Categorical.from_codes(
                column["values"], column["categories"], ordered=column["ordered"]
            )
        )
    dtype = pd.api.types.pandas_dtype(column["dtype"])
    if getattr(dtype, "kind", None) not in {"b", "i", "u", "f", "O", "U"}:
        raise ValueError("unsupported artifact dataframe dtype")
    return pd.Series(column["values"], dtype=dtype)


def _arguments(function, options: Mapping[str, object]) -> dict:
    """Project canonical settings onto an existing preparation or assembly signature.

    Parameters
    ----------
    function : Callable
        Existing shared preparation or result constructor.
    options : Mapping[str, object]
        Validated public fitting arguments.

    Returns
    -------
    dict
        Exactly the options accepted by this shared operation."""
    return {
        name: value
        for name, value in options.items()
        if name in inspect.signature(function).parameters
    }


def _stored_fit_state[Result](
    fitted: FormulaFit,
    codec: _Codec,
    *,
    kind: str,
    data: IntoDataFrame,
    recipe: RefitRecipe[IntoDataFrame, Mapping[str, object], Result],
    source: str,
    modules: Mapping[str, torch.nn.Module | None] | None = None,
) -> dict:
    """Encode a formula outcome and the rows and recipe that reproduce it.

    Parameters
    ----------
    fitted : FormulaFit
        Native fitted state shared by all response models.
    codec : _Codec
        Declarative checkpoint encoder.
    kind : str
        Kind of checkpoint to reconstruct.
    data : IntoDataFrame
        Original training rows retained for preparation.
    recipe : RefitRecipe
        Canonical fit arguments and ownership of their original rows.
    source : str
        Canonical source of the prepared formula model.
    modules : Mapping or None
        Additional trained modules of this checkpoint kind.

    Returns
    -------
    dict
        Encoded fit state and its reconstruction inputs.
    """
    return {
        "kind": kind,
        "data": codec.encode(data),
        "recipe_data": codec.encode(recipe.data)
        if recipe.snapshot_policy == "reference"
        else None,
        "options": codec.encode(dict(recipe.method)),
        "ownership": recipe.snapshot_policy,
        "source": source,
        "program": codec.encode(_module_state(fitted.program)),
        **{name: codec.encode(module) for name, module in (modules or {}).items()},
        "observations": codec.encode(dict(fitted.observations)),
        "posterior": _posterior_state(fitted.posterior, codec),
        "prior": codec.encode(fitted.prior),
        "sample_prior": fitted.sample_prior,
        "losses": codec.encode(fitted.losses),
        "criteria": codec.encode(dict(fitted.criteria)),
    }


def _restore_fit_state(
    manifest: dict,
    codec: _Codec,
    program: MonadicProgram,
    observations: Mapping[str, torch.Tensor],
) -> _InferenceResult:
    """Restore the fitted state of a checked model and its observation bindings.

    Parameters
    ----------
    manifest : dict
        Verified formula artifact specification.
    codec : _Codec
        Checked tensor and declarative option decoder.
    program : MonadicProgram
        The recompiled formula model.
    observations : Mapping[str, torch.Tensor]
        Checked observations and host-data bindings of that model.

    Returns
    -------
    _InferenceResult
        The recorded posterior, prior, prior mode and loss history.

    Raises
    ------
    TypeError
        If a retained prior is present without recorded draws.
    """
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
    return _InferenceResult(
        posterior, codec.decode(manifest["losses"]), prior, manifest["sample_prior"]
    )


def _formula_state(fitted: BayesianFit, codec: _Codec) -> dict:
    """Capture original preparation inputs and fitted state without live evaluators.

    Parameters
    ----------
    fitted : BayesianFit
        Trained formula result and its canonical original-data recipe.
    codec : _Codec
        Closed tensor and option encoder with explicit extension factories.

    Returns
    -------
    dict
        Original data and options, trained states, and checked source identity.

    Raises
    ------
    ValueError, TypeError
        If reconstruction inputs or explicit object factories are absent."""
    recipe = fitted.refit_recipe
    if (
        recipe is None
        or recipe.provenance.get("kind") != "formula"
        or recipe.provenance.get("schema") != "qvr-refit/v1"
    ):
        raise ValueError(
            "formula artifacts require the canonical original-data refit recipe"
        )
    if (
        fitted.prediction_plan is None
        or fitted.prediction_plan.preparation_data is None
    ):
        raise ValueError(
            "formula artifact requires its retained checked prediction plan"
        )
    return _stored_fit_state(
        fitted,
        codec,
        kind="formula",
        data=fitted.prediction_plan.preparation_data,
        recipe=recipe,
        source=module_to_source(fitted.prediction_plan.module),
        modules={"predictor": fitted.predictor},
    )


def _restore_formula(manifest: dict, codec: _Codec) -> BayesianFit:
    """Rebuild formula designs and native state without inference or predictor replay.

    Parameters
    ----------
    manifest : dict
        Verified formula artifact specification.
    codec : _Codec
        Checked tensor and declarative option decoder.

    Returns
    -------
    BayesianFit
        Rebuilt designs, checked native program, and restored posterior state.

    Raises
    ------
    TypeError, ValueError, RuntimeError
        If options, generated source, bindings, or retained state disagree."""
    data = codec.decode(manifest["data"])
    options = codec.decode(manifest["options"])
    # Bind the public signature to reject stale or extra options rather than
    # silently changing a saved model as the implementation evolves.
    bound = inspect.signature(fit).bind(
        data=data, snapshot_policy=manifest["ownership"], **options
    )
    bound.apply_defaults()
    recipe = capture_recipe(
        fit,
        data=codec.decode(manifest["recipe_data"])
        if manifest["ownership"] == "reference"
        else data,
        arguments=bound.arguments,
        snapshot_policy=manifest["ownership"],
    )
    predictor = codec.decode(manifest["predictor"])
    observations = codec.decode(manifest["observations"])
    predictor_name = "neural_eta" if predictor is not None else None
    if not isinstance(observations, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in observations.items()
    ):
        raise TypeError("artifact formula observations require named tensors")
    if predictor_name is not None and predictor_name not in observations:
        raise ValueError("artifact is missing its saved predictor output")
    # A program fit stores the program's text and formulas, as it was called.
    arguments = bound.arguments
    formula, family, distributional = program_arguments(
        arguments["formula"],
        program=arguments["program"],
        entry=arguments["entry"],
        static=arguments["static"],
        parameters=arguments["parameters"],
        response=arguments["response"],
        family=arguments["family"],
        distributional=arguments["distributional"],
        nonlinear=arguments["nonlinear"],
        fixed_parameters=arguments["fixed_parameters"],
        caller="load_fit",
    )
    prepared = _prepare_formula(
        **_arguments(
            _prepare_formula,
            {
                **options,
                "formula": formula,
                "family": family,
                "distributional": distributional,
                "data": data,
                "predictor_name": predictor_name,
                "caller": "load_fit",
                "allow_missing": True,
            },
        )
    )
    native = _prepare_fit(
        prepared,
        predictor=predictor,
        predictor_data=options["predictor_data"],
        method=options["method"],
        predictor_value=observations.get(predictor_name),
        preparation_data=recipe._data if manifest["ownership"] == "clone" else data,
    )
    if module_to_source(native.module) != manifest["source"]:
        raise ValueError("artifact formula preparation changed the checked source")
    if native.observations.keys() != observations.keys() or any(
        native.observations[name].shape != value.shape
        for name, value in observations.items()
    ):
        raise ValueError(
            "artifact formula binding changed its checked observation geometry"
        )
    for name, value in observations.items():
        if name == predictor_name:
            continue
        fresh = native.observations[name].to(value)
        if not torch.equal(fresh, value):
            raise ValueError(
                f"artifact formula preparation changed checked binding {name!r}"
            )
    native.observations.update(observations)
    result = _restore_fit_state(manifest, codec, native.program, native.observations)
    return _assemble_fit(
        native, result, refit_recipe=recipe, **_arguments(_assemble_fit, options)
    ).with_(criteria=codec.decode(manifest["criteria"]))
