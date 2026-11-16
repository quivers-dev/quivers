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
from jax.experimental import io_callback as _qvr_qiec_io_callback


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


def _qvr_qiec_log_mass(distribution, retained=0, shape=()):
    normalizer = getattr(distribution, "log_normalizer", None)
    mass = jnp.asarray(normalizer() if callable(normalizer) else 0.0)
    shape = tuple(shape) or tuple(distribution.batch_shape)
    mass = jnp.broadcast_to(mass, shape)
    if (
        isinstance(retained, bool)
        or not isinstance(retained, int)
        or not 0 <= retained <= mass.ndim
    ):
        raise ValueError("log_mass retained axes must be within the batch rank")
    return (
        jnp.sum(mass, axis=tuple(range(retained, mass.ndim)))
        if mass.ndim > retained
        else mass
    )


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


def _qvr_qiec_jax_observation_weight(value):
    tensor = jnp.asarray(value)
    if not jnp.issubdtype(tensor.dtype, jnp.floating):
        raise ValueError(
            "observation weights must be finite and nonnegative real values"
        )
    valid = jnp.all(jnp.isfinite(tensor) & (tensor >= 0))

    def require_valid(flag):
        if not bool(flag):
            raise ValueError("observation weights must be finite and nonnegative")

    if isinstance(tensor, _qvr_qiec_jax.core.Tracer):
        # Validation is an IO effect rather than an elidable debug callback.
        # Only a stopped Boolean crosses the host boundary, leaving the
        # checked tensor's identity derivative intact. Independent checks
        # need no ordering, which also permits vmap over the computation.
        _qvr_qiec_io_callback(
            require_valid, None, _qvr_qiec_jax.lax.stop_gradient(valid), ordered=False
        )
    else:
        require_valid(valid)
    return tensor


def _qvr_qiec_jax_scale_weight(coefficient, density):
    coefficient, density = jnp.asarray(coefficient), jnp.asarray(density)
    safe = jnp.where(coefficient == 0, jnp.zeros_like(density), density)
    return coefficient * safe


_qvr_qiec_host_math.update(
    {
        "observation_weight": _qvr_qiec_jax_observation_weight,
        "scale_weight": _qvr_qiec_jax_scale_weight,
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


def _qvr_probability_check(value, valid, name):
    import jax
    from jax.experimental import io_callback

    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise ValueError(name + " requires real values")

    def require(flag):
        if not bool(flag):
            raise ValueError(name + " received a value outside its domain")

    if isinstance(value, jax.core.Tracer):
        io_callback(require, None, jax.lax.stop_gradient(jnp.all(valid)), ordered=False)
    else:
        require(jnp.all(valid))
    return value


def _qvr_log_probability(value):
    value = jnp.asarray(value)
    value = _qvr_probability_check(
        value, jnp.isfinite(value) & (value >= 0) & (value <= 1), "log_probability"
    )
    safe = jnp.where(value > 0, value, jnp.ones_like(value))
    return jnp.where(value > 0, jnp.log(safe), -jnp.inf)


def _qvr_log1mexp(value):
    value = jnp.asarray(value)
    value = _qvr_probability_check(value, ~jnp.isnan(value) & (value <= 0), "log1mexp")
    safe = jnp.where(jnp.isfinite(value) & (value < 0), value, -jnp.ones_like(value))
    near = safe > -0.6931471805599453
    a = jnp.where(near, safe, -jnp.ones_like(value))
    b = jnp.where(near, -jnp.ones_like(value), safe)
    result = jnp.where(near, jnp.log(-jnp.expm1(a)), jnp.log1p(-jnp.exp(b)))
    return jnp.where(value == 0, -jnp.inf, jnp.where(jnp.isneginf(value), 0.0, result))


def _qvr_log_ndtr(value):
    import jax.scipy.special as special

    value = jnp.asarray(value)
    value = _qvr_probability_check(value, ~jnp.isnan(value), "log_ndtr")
    safe = jnp.where(jnp.isfinite(value), value, jnp.zeros_like(value))
    result = special.log_ndtr(safe)
    return jnp.where(
        jnp.isneginf(value), -jnp.inf, jnp.where(jnp.isposinf(value), 0.0, result)
    )


_qvr_qiec_host_math.update(
    {
        "log_probability": _qvr_log_probability,
        "log1mexp": _qvr_log1mexp,
        "log_ndtr": _qvr_log_ndtr,
    }
)


def _qvr_positive_parameter(value):
    value = jnp.asarray(value)
    return _qvr_probability_check(
        value, jnp.isfinite(value) & (value > 0), "positive_parameter"
    )


_qvr_qiec_host_math["positive_parameter"] = _qvr_positive_parameter
