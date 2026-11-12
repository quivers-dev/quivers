"""In-container PyMC probe.

The transpiled source defines a top-level ``build_model(<data>=None)``
function: the renderer wraps the ``pymc.Model`` block in a builder so
the per-call data inputs land in the function signature. The probe
instantiates one [`pymc.Model`][pymc.Model] per point with that
point's data, then evaluates the joint log-density in
constrained-space to match the QVR reference probe's convention.

Constrained-space evaluation rationale:
[`Model.compile_logp`][pymc.Model.compile_logp] expects values in
the unconstrained (transformed) space and adds the inverse-transform
Jacobian by default; the QVR reference computes log-densities in the
constrained space (the natural support, e.g. ``[0, 1]`` for
``Beta``) without a Jacobian term. The probe thus sums
[`pymc.logp`][pymc.logp] over every free and observed RV directly,
substituting each free RV with its constrained-space value from
``Point.params`` and each observed RV with its data from
``Point.data``. The constant-spread tolerance in the harness then
absorbs the per-fixture normalisation offset between QVR's and
PyMC's internal conventions.

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. ``build_model`` hands back
the model rather than a value, so PyMC's export surface is
[`pymc.Deterministic`][pymc.Deterministic]: the renderer registers
each returned name under ``<name>_value`` and the probe evaluates
that model variable with every free RV pinned to the point.
"""

import json
import pathlib

import numpy as np
import pymc
import pytensor
import pytensor.tensor as pt

from _reshape import (
    as_nested,
    load_export_names,
    load_tables,
    reshape_point,
)


def _arr(value):
    return np.asarray(value)


def _build_model(source: str, data_kw: dict) -> pymc.Model:
    """Execute the transpiled source and call ``build_model(**data)``.

    The renderer emits ``def build_model(<input1>=None, ...): ...``
    with one keyword argument per [`IRDataInput`][quivers.transpile.ir.IRDataInput].
    The probe locates that function in the executed namespace and
    invokes it with the per-point data dict, returning the constructed
    [`pymc.Model`][pymc.Model] instance.
    """
    ns = {"pymc": pymc, "np": np, "pt": pt}
    exec(source, ns)  # noqa: S102
    builder = ns.get("build_model")
    if builder is None:
        msg = (
            "transpiled PyMC source missing top-level `build_model` "
            "function; available names: "
            + ", ".join(sorted(k for k in ns if not k.startswith("_")))
        )
        raise RuntimeError(msg)
    model = builder(**data_kw)
    if not isinstance(model, pymc.Model):
        msg = (
            "transpiled `build_model` did not return a `pymc.Model`; "
            f"got {type(model).__name__}"
        )
        raise RuntimeError(msg)
    return model


def _joint_logp_constrained(
    model: pymc.Model,
    params: dict,
) -> float:
    """Sum ``pymc.logp(rv, value)`` over every free and observed RV,
    plus every potential.

    Free RVs receive their constrained-space value from ``params``;
    observed RVs use their attached observation tensor. Free-RV
    references inside observed-RV expressions are substituted with
    their constrained values via
    [`graph_replace`][pytensor.graph.replace.graph_replace] so the
    resulting graph has no free symbolic inputs.

    A free RV whose prior log-density is a constant (``Beta(1, 1)``,
    ``Uniform(0, 1)``) and whose value no observed site reads leaves
    no node in the summed graph at all, because PyTensor constant-folds
    the term away. Its substitution is then unused, which
    ``graph_replace`` reports as an error under its default strict
    mode. That is a property of the model, not a harness fault, so the
    replacement runs non-strict. The hard error for a free RV *missing*
    from ``params`` above is what keeps an unclamped latent from
    slipping through.
    """
    substitutions: dict[object, object] = {}
    logp_terms: list[object] = []

    for rv in model.free_RVs:
        if rv.name not in params:
            msg = (
                f"pymc probe: missing param for free RV {rv.name!r}; "
                f"available: {sorted(params)}"
            )
            raise RuntimeError(msg)
        value = pt.as_tensor(params[rv.name]).astype(rv.dtype)
        logp_terms.append(pymc.logp(rv, value).sum())
        substitutions[rv] = value

    for rv in model.observed_RVs:
        obs_value = model.rvs_to_values[rv]
        logp_terms.append(pymc.logp(rv, obs_value).sum())

    # A `score` step and a helper's scored weight enter the model as
    # potentials, terms of the joint in their own right.
    for potential in model.potentials:
        logp_terms.append(potential.sum())

    total = logp_terms[0]
    for term in logp_terms[1:]:
        total = total + term

    if substitutions:
        total = pytensor.graph.replace.graph_replace(
            total,
            substitutions,
            strict=False,
        )

    return float(total.eval())


def _exported_values(
    model: pymc.Model,
    params: dict,
    export_names: list,
) -> list:
    """Evaluate each `<name>_value` deterministic at the point.

    The PyMC renderer registers one
    [`pymc.Deterministic`][pymc.Deterministic] per returned name, so a
    missing entry means the emit dropped the program's return clause
    and the model denotes the right joint under the wrong kernel. That
    is a renderer defect, so it raises here rather than reporting a
    shorter export vector the comparison would silently skip.

    Every free RV is replaced by its constrained-space value from the
    point before evaluation, exactly as
    :func:`_joint_logp_constrained` does, so the returned value is a
    deterministic function of the point rather than of whatever the
    graph would draw.
    """
    substitutions: dict[object, object] = {}
    for rv in model.free_RVs:
        if rv.name not in params:
            msg = (
                f"pymc probe: missing param for free RV {rv.name!r}; "
                f"available: {sorted(params)}"
            )
            raise RuntimeError(msg)
        substitutions[rv] = pt.as_tensor(params[rv.name]).astype(rv.dtype)

    values = []
    for name in export_names:
        alias = f"{name}_value"
        variable = model.named_vars.get(alias)
        if variable is None:
            msg = (
                f"pymc probe: the emitted model registers no "
                f"{alias!r} deterministic, so it exposes nothing for "
                f"the QVR program's exported {name!r}. Available "
                f"model variables: {sorted(model.named_vars)}"
            )
            raise RuntimeError(msg)
        if substitutions:
            variable = pytensor.graph.replace.graph_replace(
                variable,
                substitutions,
                strict=False,
            )
        values.append(as_nested(np.asarray(variable.eval())))
    return values


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    export_names = load_export_names(io)

    log_densities = []
    exports = []
    for pt_record in points:
        reshaped = reshape_point(pt_record, shapes, dtypes)
        data_kw = {k: _arr(v) for k, v in reshaped.get("data", {}).items()}
        params = {k: _arr(v) for k, v in reshaped.get("params", {}).items()}
        model = _build_model(source, data_kw)
        log_densities.append(_joint_logp_constrained(model, params))
        if export_names:
            exports.append(_exported_values(model, params, export_names))

    result = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
