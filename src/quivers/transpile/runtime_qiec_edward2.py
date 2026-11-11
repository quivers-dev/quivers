"""QIEC distribution bridge for generated Edward2 modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a ``float32`` TensorFlow tensor; a log density is the
TensorFlow Probability distribution's own ``log_prob`` at the converted
value.
"""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# The bridge is grafted after the shared runtime into a generated module,
# which binds the host library names and the runtime's helpers before the
# bridge is read.


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, tf.Tensor):
        return value
    return tf.convert_to_tensor(_qvr_qiec_nested(value), dtype=tf.float32)


def _qvr_qiec_log_density(distribution, value):
    return distribution.log_prob(_qvr_qiec_array(value))


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
    # primitives: a draw is a traced random variable of the
    # distribution's family, so an interceptor and the tape see it under
    # its name, and a score is a traced factor variable.
    name = _qvr_qiec_site_names()

    def sample(request):
        label, distribution = request["arguments"]
        constructor = edward2.make_random_variable(type(distribution))
        parameters = {
            key: value
            for key, value in distribution.parameters.items()
            if key != "name"
        }
        return constructor(name=name(label), **parameters)

    def add(request):
        (weight,) = request["arguments"]
        _qvr_qiec_factor(name("score"), _qvr_qiec_array(weight))
        return None

    return {
        (random_instance, sample_operation): sample,
        (score_instance, add_operation): add,
    }


def _qvr_qiec_tensors(values):
    # The values a host function is applied to, each a tensor even
    # when it arrived as a Python number.
    return tuple(tf.convert_to_tensor(value, dtype_hint=tf.float32) for value in values)


def _qvr_qiec_tf(spelling):
    # A TensorFlow function named by its attribute path on `tf`, which
    # the generated module reads from its namespace at the call, so
    # the module loads without TensorFlow.
    def apply(*values):
        function = tf
        for attribute in spelling.split("."):
            function = getattr(function, attribute)
        return function(*_qvr_qiec_tensors(values))

    return apply


def _qvr_qiec_tf_trunc(value):
    (tensor,) = _qvr_qiec_tensors((value,))
    return tf.math.sign(tensor) * tf.floor(tf.abs(tensor))


_qvr_qiec_host_math.update(
    {
        "min_real": _qvr_qiec_tf("minimum"),
        "max_real": _qvr_qiec_tf("maximum"),
        "pow_real": _qvr_qiec_tf("pow"),
        "exp": _qvr_qiec_tf("exp"),
        "log": _qvr_qiec_tf("math.log"),
        "sqrt": _qvr_qiec_tf("sqrt"),
        "expm1": _qvr_qiec_tf("math.expm1"),
        "log1p": _qvr_qiec_tf("math.log1p"),
        "log2": lambda value: _qvr_qiec_tf("math.log")(value) / tf.math.log(2.0),
        "log10": lambda value: _qvr_qiec_tf("math.log")(value) / tf.math.log(10.0),
        "rsqrt": _qvr_qiec_tf("math.rsqrt"),
        "sign": _qvr_qiec_tf("sign"),
        "reciprocal": _qvr_qiec_tf("math.reciprocal"),
        "sin": _qvr_qiec_tf("sin"),
        "cos": _qvr_qiec_tf("cos"),
        "tan": _qvr_qiec_tf("tan"),
        "asin": _qvr_qiec_tf("asin"),
        "acos": _qvr_qiec_tf("acos"),
        "atan": _qvr_qiec_tf("atan"),
        "sinh": _qvr_qiec_tf("sinh"),
        "cosh": _qvr_qiec_tf("cosh"),
        "tanh": _qvr_qiec_tf("tanh"),
        "asinh": _qvr_qiec_tf("asinh"),
        "acosh": _qvr_qiec_tf("acosh"),
        "atanh": _qvr_qiec_tf("atanh"),
        "floor": _qvr_qiec_tf("floor"),
        "ceil": _qvr_qiec_tf("math.ceil"),
        "round": _qvr_qiec_tf("round"),
        "trunc": _qvr_qiec_tf_trunc,
        "real_to_int": lambda value: tf.cast(_qvr_qiec_tf_trunc(value), tf.int32),
        "select_real": tf.where,
        "select_int": tf.where,
        "bool_to_real": lambda value: tf.cast(value, tf.float32),
        "bool_to_int": lambda value: tf.cast(value, tf.int32),
        "erf": _qvr_qiec_tf("math.erf"),
        "erfc": _qvr_qiec_tf("math.erfc"),
        "erfinv": _qvr_qiec_tf("math.erfinv"),
        "lgamma": _qvr_qiec_tf("math.lgamma"),
        "digamma": _qvr_qiec_tf("math.digamma"),
        "sigmoid": _qvr_qiec_tf("math.sigmoid"),
        "relu": _qvr_qiec_tf("nn.relu"),
        "relu6": _qvr_qiec_tf("nn.relu6"),
        "elu": _qvr_qiec_tf("nn.elu"),
        "selu": _qvr_qiec_tf("nn.selu"),
        "gelu": lambda value: tf.nn.gelu(
            *_qvr_qiec_tensors((value,)), approximate=False
        ),
        "silu": _qvr_qiec_tf("nn.silu"),
        "mish": lambda value: (
            _qvr_qiec_tf("identity")(value)
            * tf.tanh(tf.nn.softplus(*_qvr_qiec_tensors((value,))))
        ),
        "softplus": _qvr_qiec_tf("nn.softplus"),
        "logsigmoid": _qvr_qiec_tf("math.log_sigmoid"),
        "softsign": _qvr_qiec_tf("nn.softsign"),
    }
)
