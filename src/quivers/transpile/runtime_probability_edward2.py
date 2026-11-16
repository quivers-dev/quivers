"""Shared probability primitives for standalone edward2 runtime generation."""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false


def _qvr_probability_check(value, valid, name, integer=False):
    valid_type = value.dtype.is_integer if integer else value.dtype.is_floating
    if not valid_type:
        raise ValueError(
            name + (" requires integer values" if integer else " requires real values")
        )
    checked = tf.debugging.assert_equal(
        tf.reduce_all(valid),
        True,
        message=name + " received a value outside its domain",
    )
    with tf.control_dependencies([] if checked is None else [checked]):
        return tf.identity(value)


def _qvr_log_probability(value):
    value = tf.convert_to_tensor(value)
    value = _qvr_probability_check(
        value, tf.math.is_finite(value) & (value >= 0) & (value <= 1), "log_probability"
    )
    safe = tf.where(value > 0, value, tf.ones_like(value))
    return tf.where(value > 0, tf.math.log(safe), tf.cast(-float("inf"), value.dtype))


def _qvr_log1mexp(value):
    value = tf.convert_to_tensor(value)
    value = _qvr_probability_check(
        value, ~tf.math.is_nan(value) & (value <= 0), "log1mexp"
    )
    safe = tf.where(tf.math.is_finite(value) & (value < 0), value, -tf.ones_like(value))
    near = safe > -0.6931471805599453
    a = tf.where(near, safe, -tf.ones_like(value))
    b = tf.where(near, -tf.ones_like(value), safe)
    result = tf.where(near, tf.math.log(-tf.math.expm1(a)), tf.math.log1p(-tf.exp(b)))
    return tf.where(
        value == 0,
        tf.cast(-float("inf"), value.dtype),
        tf.where(tf.math.is_inf(value), tf.zeros_like(value), result),
    )


def _qvr_log_ndtr(value):
    value = tf.convert_to_tensor(value)
    value = _qvr_probability_check(value, ~tf.math.is_nan(value), "log_ndtr")
    safe = tf.where(tf.math.is_finite(value), value, tf.zeros_like(value))
    result = tfp.distributions.Normal(
        tf.zeros_like(value), tf.ones_like(value)
    ).log_cdf(safe)
    return tf.where(
        tf.math.is_inf(value),
        tf.where(value < 0, tf.cast(-float("inf"), value.dtype), tf.zeros_like(value)),
        result,
    )


def _qvr_positive_parameter(value):
    value = tf.convert_to_tensor(value)
    return _qvr_probability_check(
        value, tf.math.is_finite(value) & (value > 0), "positive_parameter"
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
    value = tf.convert_to_tensor(value)
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
    lower, upper = tf.convert_to_tensor(lower), tf.convert_to_tensor(upper)
    if not lower.dtype.is_floating:
        raise ValueError("interval_upper requires real endpoints")
    upper = tf.broadcast_to(
        upper, tf.broadcast_dynamic_shape(tf.shape(lower), tf.shape(upper))
    )
    return _qvr_probability_check(
        upper,
        ~tf.math.is_nan(lower) & tf.math.is_finite(upper) & (upper > lower),
        "interval_upper",
    )
