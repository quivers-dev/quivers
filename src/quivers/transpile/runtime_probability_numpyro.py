"""Shared probability primitives for standalone numpyro runtime generation."""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false


def _qvr_probability_check(value, valid, name, integer=False):
    import jax
    from jax.experimental import io_callback

    expected = jnp.integer if integer else jnp.floating
    if not jnp.issubdtype(value.dtype, expected):
        raise ValueError(
            name + (" requires integer values" if integer else " requires real values")
        )

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


def _qvr_positive_parameter(value):
    value = jnp.asarray(value)
    return _qvr_probability_check(
        value, jnp.isfinite(value) & (value > 0), "positive_parameter"
    )


def _qvr_observation_status(value):
    """Validate integer recorded-event codes without changing their carrier.

    Parameters
    ----------
    value
        Scalar or lifted integer code tensor.

    Returns
    -------
    integer tensor
        Codes minus one through two, unchanged.

    Raises
    ------
    ValueError
        If the carrier is not integer or an active code is invalid.
    """
    value = jnp.asarray(value)
    return _qvr_probability_check(
        value, (value >= -1) & (value <= 2), "observation_status", integer=True
    )


def _qvr_interval_upper(lower, upper):
    """Check an active interval and retain its upper-endpoint derivatives.

    Parameters
    ----------
    lower, upper
        Real scalar or lifted endpoints; lower may be infinite.

    Returns
    -------
    real tensor
        Finite upper endpoint, strictly above lower.

    Raises
    ------
    ValueError
        If the endpoint carrier or active interval is invalid.
    """
    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    if not jnp.issubdtype(lower.dtype, jnp.floating):
        raise ValueError("interval_upper requires real endpoints")
    upper = jnp.broadcast_to(upper, jnp.broadcast_shapes(lower.shape, upper.shape))
    return _qvr_probability_check(
        upper,
        ~jnp.isnan(lower) & jnp.isfinite(upper) & (upper > lower),
        "interval_upper",
    )
