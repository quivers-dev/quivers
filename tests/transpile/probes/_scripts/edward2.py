"""In-container Edward2 probe.

Uses the standalone ``edward2`` package (``google/edward2``); the
TFP-bundled ``tensorflow_probability.edward2`` submodule was retired.
The trace API is ``ed.tape`` and the value-override API is
``ed.condition`` (replacement for the old ``ed.interception``).

The probe computes joint log-density by tracing the model once
(under any conditioning whose value-shape matches the
``RandomVariable``'s ``value.shape``) and then evaluating
``rv.distribution.log_prob(target_value)`` against the user-supplied
value tensor. Observed variables whose target value shape exceeds
the RV's ``value.shape`` (the canonical case: a ``Normal(loc=mu)``
where ``mu`` is itself a vector-shaped deterministic input expression)
are paired with the distribution and scored directly: the model's
RV value is not used, so its sample shape is irrelevant.

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. Edward2's export surface is
the model function's own `return`, so the value comes from the
conditioned second trace: every latent is pinned there, which makes
the returned quantity a deterministic function of the point rather
than of whatever the RV happened to draw.
"""

import json
import pathlib

import tensorflow as tf
import edward2 as ed

from _reshape import (
    export_payload,
    load_export_names,
    load_tables,
    reshape_point,
)


def _is_factor(rv) -> bool:
    """Whether a traced variable is a scored weight rather than a draw.

    Parameters
    ----------
    rv
        A random variable read off the tape.

    Returns
    -------
    bool
        True for the factor distribution generated Edward2 modules
        trace a `score` step or a scored call through.
    """
    return type(rv.distribution).__name__ == "_QvrFactorDistribution"


def _tensor(value):
    """Cast a JSON-decoded payload into a TF tensor.

    Defaults float values to ``tf.float32`` to match TFP's default
    distribution dtype; lists of ints become ``tf.int32`` (matching
    ``Bernoulli`` / ``Categorical`` integer support). Mixing dtypes
    across the model graph (e.g. a ``float32`` sampled RV multiplied
    by a ``float64`` design-matrix input) raises in TF's strict
    type-checked ops.
    """
    if isinstance(value, list):
        if value and all(isinstance(v, bool) for v in value):
            return tf.constant(value, dtype=tf.int32)
        if value and all(isinstance(v, int) for v in value):
            return tf.constant(value, dtype=tf.int32)
        return tf.constant(value, dtype=tf.float32)
    if isinstance(value, bool):
        return tf.constant(value, dtype=tf.int32)
    if isinstance(value, int):
        return tf.constant(value, dtype=tf.int32)
    if isinstance(value, float):
        return tf.constant(value, dtype=tf.float32)
    return tf.constant(value)


def _cast_for_rv(rv, raw):
    """Cast ``raw`` to a tensor matching the random variable's dtype."""
    return tf.constant(raw, dtype=rv.distribution.dtype)


def _matches_value_shape(rv, raw_tensor) -> bool:
    """True iff ``raw_tensor.shape`` matches ``rv.value.shape``.

    The Edward2 ``condition`` tracer overwrites the RV's value with
    the conditioned tensor before constructing the RV, but the
    ``RandomVariable`` constructor validates the value's shape against
    ``sample_shape + batch_shape + event_shape``. A mismatch raises;
    we conservatively detect it here so the second trace can either
    condition only the matching RVs (and score the rest separately)
    or bail out of conditioning entirely for that name.
    """
    rv_shape = tuple(int(d) for d in rv.value.shape)
    raw_shape = tuple(int(d) for d in raw_tensor.shape)
    return rv_shape == raw_shape


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    export_names = load_export_names(io)

    ns = {"edward2": ed, "ed": ed, "tf": tf}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    exports = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        params = reshaped.get("params", {})
        data = reshaped.get("data", {})
        data_kw = {k: _tensor(v) for k, v in data.items()}

        # Pass 1: trace the model with no value overrides. This gives
        # the per-name `RandomVariable` so we can read each
        # distribution's dtype and `value.shape`.
        with ed.tape() as probe_tape:
            model(**data_kw)

        target_values: dict[str, tf.Tensor] = {}
        for name, raw in {**params, **data}.items():
            if name not in probe_tape:
                continue
            target_values[name] = _cast_for_rv(probe_tape[name], raw)

        # Conditioning a name whose target value shape does not match
        # the RV's `value.shape` would raise in the RV constructor;
        # restrict the second trace's overrides to the safe set, and
        # score the rest by direct `dist.log_prob(value)` evaluation
        # against the first-trace distribution.
        safe_overrides: dict[str, tf.Tensor] = {}
        direct_scores: dict[str, tf.Tensor] = {}
        for name, target in target_values.items():
            if _matches_value_shape(probe_tape[name], target):
                safe_overrides[name] = target
            else:
                direct_scores[name] = target

        with ed.condition(**safe_overrides):
            with ed.tape() as recorded:
                returned = model(**data_kw)
        if export_names:
            # An exported name that reached `direct_scores` was not
            # conditioned in this trace, so the `RandomVariable` the
            # model returned still carries whatever it drew. Comparing
            # that against the reference would compare two independent
            # draws, which passes or fails by chance; refuse instead.
            drawn = sorted(n for n in export_names if n in direct_scores)
            if drawn:
                msg = (
                    "edward2 probe: exported name(s) "
                    f"{drawn} could not be conditioned in this trace "
                    "(their point value's shape disagrees with the "
                    "random variable's), so the returned value is a "
                    "fresh draw rather than a function of the point."
                )
                raise RuntimeError(msg)
            exports.append(export_payload(export_names, returned))

        # Every random variable on the tape must be pinned, either by
        # a value override in the second trace or by a direct score
        # against the user-supplied value. One that is neither would be
        # scored at whatever it happened to draw, which turns the probe
        # into a random number generator: the returned log-density then
        # varies run to run and the constant-spread contract measures
        # sampling noise rather than the model's measure. Fail instead,
        # matching the pymc probe's hard error on a free RV with no
        # supplied value. A factor variable is a one-point distribution
        # whose log density is a scored weight; it has nothing to pin.
        unclamped = sorted(
            name
            for name, rv in recorded.items()
            if name not in safe_overrides
            and name not in direct_scores
            and not _is_factor(rv)
        )
        if unclamped:
            msg = (
                "edward2 probe: unclamped random variable(s) "
                f"{unclamped}; every traced site must be supplied "
                "through the point's params or data. available params: "
                f"{sorted(params)}; available data: {sorted(data)}"
            )
            raise RuntimeError(msg)

        # Preserve small log terms when a plate or the joint contains
        # cancelling factors. Widen before either reduction; widening
        # an already-rounded float32 sum cannot recover the lost terms.
        log_terms = []
        for name, rv in recorded.items():
            value = direct_scores[name] if name in direct_scores else rv.value
            log_prob = rv.distribution.log_prob(value)
            log_terms.append(tf.reduce_sum(tf.cast(log_prob, tf.float64)))
        total = tf.add_n(log_terms)
        log_densities.append(float(total.numpy()))

    result = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
