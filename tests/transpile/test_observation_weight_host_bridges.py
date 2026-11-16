"""Genuine optional Python host backends validate and differentiate weight primitives.

This module imports no QVR code, so it also runs under an isolated host Python
version with pytest --noconftest when a backend does not support QVR's Python.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
from pathlib import Path
from typing import cast

import numpy as np
import pytest

ROOT = Path(__file__).parents[2] / "src/quivers/transpile"


def _bridge(target: str, **bindings: object) -> Callable:
    namespace = dict(bindings)
    exec((ROOT / "runtime_qiec.py").read_text(), namespace)
    spec = importlib.util.spec_from_file_location(
        "qvr_runtime_sources", ROOT / "_runtime_sources.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exec(module.python_runtime_source(ROOT / f"runtime_qiec_{target}.py"), namespace)
    return cast(Callable, namespace["_qvr_qiec_primitive"])


def test_pytensor_symbolic_assertion_and_absorbing_zero_gradients() -> None:
    pytensor = pytest.importorskip("pytensor")
    pt = pytest.importorskip("pytensor.tensor")
    from pytensor.compile.mode import Mode

    primitive = _bridge("pymc", np=np)
    weights, density = pt.dvector("weights"), pt.dvector("density")
    checked = primitive("observation_weight", (weights,))
    result = primitive("scale_weight", (checked, density))
    gradients = pt.grad(result.sum(), (weights, density))
    function = pytensor.function(
        [weights, density],
        [checked, result, *gradients],
        mode=Mode(linker="py", optimizer="fast_compile"),
    )
    values = function([0.0, 2.0, 0.5], [-np.inf, -3.0, -4.0])
    for actual, expected in zip(
        values,
        [[0.0, 2.0, 0.5], [0.0, -6.0, -2.0], [0.0, -3.0, -4.0], [0.0, 2.0, 0.5]],
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for invalid in (-1.0, np.inf, -np.inf, np.nan):
        with pytest.raises(AssertionError, match="finite and nonnegative"):
            function([0.0, invalid, 1.0], [-np.inf, -3.0, -4.0])
    for dtype in ("bool", "int64"):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            primitive("observation_weight", (pt.vector(dtype=dtype),))


def test_jax_host_validation_survives_jit_grad_vmap_and_unused_result() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    primitive = _bridge("numpyro", jnp=jnp)

    def weighted(weights, density):
        checked = primitive("observation_weight", (weights,))
        return primitive("scale_weight", (checked, density))

    weights = jnp.asarray([0.0, 2.0, 0.5])
    density = jnp.asarray([-jnp.inf, -3.0, -4.0])
    np.testing.assert_array_equal(
        jax.jit(weighted)(weights, density), [0.0, -6.0, -2.0]
    )
    gradient = jax.jit(jax.grad(lambda w: jnp.sum(weighted(w, density))))
    np.testing.assert_array_equal(gradient(weights), [0.0, -3.0, -4.0])
    mapped = jax.jit(jax.vmap(lambda w: primitive("observation_weight", (w,))))
    np.testing.assert_array_equal(
        mapped(jnp.stack([weights, weights])), np.stack([weights, weights])
    )
    for invalid in (-1.0, np.inf, -np.inf, np.nan):
        with pytest.raises(Exception, match="finite and nonnegative"):
            jax.jit(weighted)(
                jnp.asarray([0.0, invalid, 1.0]), density
            ).block_until_ready()
    for value in (jnp.asarray([True]), jnp.asarray([1])):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            primitive("observation_weight", (value,))

    def unused(weights):
        primitive("observation_weight", (weights,))
        return jnp.asarray(0.0)

    with pytest.raises(Exception, match="finite and nonnegative"):
        jax.jit(unused)(jnp.asarray([-1.0])).block_until_ready()


def test_tensorflow_validation_survives_function_and_gradient_tape() -> None:
    tf = pytest.importorskip("tensorflow")
    primitive = _bridge("edward2", tf=tf)

    def weighted(weights, density):
        checked = primitive("observation_weight", (weights,))
        return primitive("scale_weight", (checked, density))

    weights = tf.Variable([0.0, 2.0, 0.5], dtype=tf.float64)
    density = tf.Variable([-np.inf, -3.0, -4.0], dtype=tf.float64)
    with tf.GradientTape() as tape:
        result = weighted(weights, density)
        total = tf.reduce_sum(result)
    np.testing.assert_array_equal(result.numpy(), [0.0, -6.0, -2.0])
    gradients = tape.gradient(total, (weights, density))
    np.testing.assert_array_equal(gradients[0].numpy(), [0.0, -3.0, -4.0])
    np.testing.assert_array_equal(gradients[1].numpy(), [0.0, 2.0, 0.5])
    compiled = tf.function(weighted)
    np.testing.assert_array_equal(compiled(weights, density).numpy(), [0.0, -6.0, -2.0])
    for invalid in (-1.0, np.inf, -np.inf, np.nan):
        with pytest.raises(
            tf.errors.InvalidArgumentError, match="finite and nonnegative"
        ):
            compiled(tf.constant([0.0, invalid, 1.0], dtype=tf.float64), density)
    for value in (tf.constant([True]), tf.constant([1])):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            primitive("observation_weight", (value,))
