"""QIEC distribution bridge for generated NumPyro modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a floating JAX array; a log density is the distribution's own
``log_prob`` at the scored point, which keeps an integral dtype so a
discrete family can index its probabilities with it.
"""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# The bridge is grafted after the shared runtime into a generated module,
# which binds the host library names and the runtime's helpers before the
# bridge is read.

import jax as _qvr_qiec_jax
import jax.scipy.special as _qvr_qiec_jax_special


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, jnp.ndarray):
        return value
    return jnp.asarray(_qvr_qiec_nested(value), dtype=jnp.result_type(float))


def _qvr_qiec_point(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, jnp.ndarray):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, tuple):
        return jnp.asarray(_qvr_qiec_nested(value))
    return value


def _qvr_qiec_log_density(distribution, value):
    return distribution.log_prob(_qvr_qiec_point(value))


def _qvr_qiec_site_names():
    # Site labels a helper's draws and scores carry, counted so the n-th
    # occurrence of a label in one run of the model is "<label>@<n>", as
    # the reference machine replays them.
    occurrences = {}

    def name(label):
        count = occurrences.get(label, 0)
        occurrences[label] = count + 1
        return label if count == 0 else label + "@" + str(count)

    return name


def _qvr_qiec_native_operations(
    random_instance, sample_operation, score_instance, add_operation
):
    # The program's canonical instances handled by the host's own
    # primitives, so a computation the model calls draws and scores as
    # the model does.
    name = _qvr_qiec_site_names()

    def sample(request):
        label, distribution = request["arguments"]
        return numpyro.sample(name(label), distribution)

    def add(request):
        (weight,) = request["arguments"]
        numpyro.factor(name("score"), _qvr_qiec_array(weight))
        return None

    return {
        (random_instance, sample_operation): sample,
        (score_instance, add_operation): add,
    }


_qvr_qiec_host_math.update(
    {
        "min_real": jnp.minimum,
        "max_real": jnp.maximum,
        "pow_real": jnp.power,
        "exp": jnp.exp,
        "log": jnp.log,
        "sqrt": jnp.sqrt,
        "expm1": jnp.expm1,
        "log1p": jnp.log1p,
        "log2": jnp.log2,
        "log10": jnp.log10,
        "rsqrt": lambda value: 1.0 / jnp.sqrt(value),
        "sign": jnp.sign,
        "reciprocal": jnp.reciprocal,
        "sin": jnp.sin,
        "cos": jnp.cos,
        "tan": jnp.tan,
        "asin": jnp.arcsin,
        "acos": jnp.arccos,
        "atan": jnp.arctan,
        "sinh": jnp.sinh,
        "cosh": jnp.cosh,
        "tanh": jnp.tanh,
        "asinh": jnp.arcsinh,
        "acosh": jnp.arccosh,
        "atanh": jnp.arctanh,
        "floor": jnp.floor,
        "ceil": jnp.ceil,
        "round": jnp.round,
        "trunc": jnp.trunc,
        "real_to_int": lambda value: jnp.trunc(value).astype(jnp.int32),
        "select_real": jnp.where,
        "select_int": jnp.where,
        "bool_to_real": lambda value: jnp.asarray(value).astype(jnp.result_type(float)),
        "bool_to_int": lambda value: jnp.asarray(value).astype(jnp.int32),
        "erf": _qvr_qiec_jax_special.erf,
        "erfc": _qvr_qiec_jax_special.erfc,
        "erfinv": _qvr_qiec_jax_special.erfinv,
        "lgamma": _qvr_qiec_jax_special.gammaln,
        "digamma": _qvr_qiec_jax_special.digamma,
        "sigmoid": _qvr_qiec_jax.nn.sigmoid,
        "relu": _qvr_qiec_jax.nn.relu,
        "relu6": _qvr_qiec_jax.nn.relu6,
        "elu": _qvr_qiec_jax.nn.elu,
        "selu": _qvr_qiec_jax.nn.selu,
        "gelu": lambda value: _qvr_qiec_jax.nn.gelu(value, approximate=False),
        "silu": _qvr_qiec_jax.nn.silu,
        "mish": lambda value: value * jnp.tanh(_qvr_qiec_jax.nn.softplus(value)),
        "softplus": _qvr_qiec_jax.nn.softplus,
        "logsigmoid": _qvr_qiec_jax.nn.log_sigmoid,
        "softsign": _qvr_qiec_jax.nn.soft_sign,
    }
)
