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


# qvr-shared-probability:edward2:start
# qvr-shared-probability:edward2:end
