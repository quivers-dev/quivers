"""A log-density factor for generated Edward2 modules.

Edward2 traces random variables and reads a joint as the sum of each
variable's log density at its value. A factor is a random variable of
a one-point distribution whose log density is the factor's weight, so
the tape carries the weight the way the other hosts' factor
primitives do.
"""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# The helper is grafted into a generated module, which binds the host
# library names before the helper is read.


class _QvrFactorDistribution(tfp.distributions.Distribution):
    # A distribution over the single point zero whose log density at
    # that point is the weight it was built with.

    def __init__(
        self, log_factor, validate_args=False, allow_nan_stats=True, name="QvrFactor"
    ):
        parameters = dict(locals())
        self._log_factor = tf.convert_to_tensor(log_factor, dtype_hint=tf.float32)
        super().__init__(
            dtype=self._log_factor.dtype,
            reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name,
        )

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return {"log_factor": tfp.util.ParameterProperties()}

    @property
    def log_factor(self):
        return self._log_factor

    def _event_shape(self):
        return tf.TensorShape([])

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _batch_shape(self):
        return self._log_factor.shape

    def _batch_shape_tensor(self):
        return tf.shape(self._log_factor)

    def _sample_n(self, n, seed=None):
        return tf.zeros(
            tf.concat([[n], tf.shape(self._log_factor)], axis=0), dtype=self.dtype
        )

    def _log_prob(self, value):
        return self._log_factor


def _qvr_qiec_factor(name, weight):
    return edward2.make_random_variable(_QvrFactorDistribution)(
        log_factor=weight, name=name
    )


def _qvr_probability_check(value, valid, name):
    if not value.dtype.is_floating:
        raise ValueError(name + " requires real values")
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
