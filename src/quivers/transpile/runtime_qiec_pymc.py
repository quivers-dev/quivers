"""QIEC distribution bridge for generated PyMC modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a NumPy array; a log density is ``pymc.logp`` of the
unregistered distribution at the converted value, a PyTensor expression
the enclosing model evaluates. A family whose PyMC support is shifted
against the QIEC convention is wrapped with its offset, and the offset
is added to the scored value.
"""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# The bridge is grafted after the shared runtime into a generated module,
# which binds the host library names and the runtime's helpers before the
# bridge is read.

import pytensor.tensor as _qvr_qiec_pt


class _QvrQiecShifted:
    def __init__(self, distribution, offset):
        self.distribution = distribution
        self.offset = offset


def _qvr_qiec_shifted(distribution, offset):
    return _QvrQiecShifted(distribution, offset)


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, tuple):
        return np.asarray(_qvr_qiec_nested(value), dtype="float64")
    return _qvr_qiec_nested(value)


def _qvr_qiec_log_density(distribution, value):
    point = _qvr_qiec_array(value)
    if isinstance(distribution, _QvrQiecShifted):
        return pymc.logp(distribution.distribution, point + distribution.offset)
    return pymc.logp(distribution, point)


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
    # the model does: a draw registers the unregistered distribution
    # as a named variable of the enclosing model.
    name = _qvr_qiec_site_names()

    def sample(request):
        label, distribution = request["arguments"]
        model = pymc.Model.get_context()
        if isinstance(distribution, _QvrQiecShifted):
            # The variable is drawn on PyMC's support and read back on
            # QIEC's, so the site holds the host's convention.
            drawn = model.register_rv(distribution.distribution, name(label))
            return drawn - distribution.offset
        return model.register_rv(distribution, name(label))

    def add(request):
        (weight,) = request["arguments"]
        pymc.Potential(name("score"), weight)
        return None

    return {
        (random_instance, sample_operation): sample,
        (score_instance, add_operation): add,
    }


_qvr_qiec_host_math.update(
    {
        "min_real": _qvr_qiec_pt.minimum,
        "max_real": _qvr_qiec_pt.maximum,
        "pow_real": _qvr_qiec_pt.power,
        "exp": _qvr_qiec_pt.exp,
        "log": _qvr_qiec_pt.log,
        "sqrt": _qvr_qiec_pt.sqrt,
        "expm1": _qvr_qiec_pt.expm1,
        "log1p": _qvr_qiec_pt.log1p,
        "log2": _qvr_qiec_pt.log2,
        "log10": _qvr_qiec_pt.log10,
        "rsqrt": lambda value: 1.0 / _qvr_qiec_pt.sqrt(value),
        "sign": _qvr_qiec_pt.sign,
        "reciprocal": _qvr_qiec_pt.reciprocal,
        "sin": _qvr_qiec_pt.sin,
        "cos": _qvr_qiec_pt.cos,
        "tan": _qvr_qiec_pt.tan,
        "asin": _qvr_qiec_pt.arcsin,
        "acos": _qvr_qiec_pt.arccos,
        "atan": _qvr_qiec_pt.arctan,
        "sinh": _qvr_qiec_pt.sinh,
        "cosh": _qvr_qiec_pt.cosh,
        "tanh": _qvr_qiec_pt.tanh,
        "asinh": _qvr_qiec_pt.arcsinh,
        "acosh": _qvr_qiec_pt.arccosh,
        "atanh": _qvr_qiec_pt.arctanh,
        "floor": _qvr_qiec_pt.floor,
        "ceil": _qvr_qiec_pt.ceil,
        "round": _qvr_qiec_pt.round_half_to_even,
        "trunc": _qvr_qiec_pt.trunc,
        "real_to_int": lambda value: _qvr_qiec_pt.trunc(value).astype("int64"),
        "erf": _qvr_qiec_pt.erf,
        "erfc": _qvr_qiec_pt.erfc,
        "erfinv": _qvr_qiec_pt.erfinv,
        "lgamma": _qvr_qiec_pt.gammaln,
        "digamma": _qvr_qiec_pt.psi,
        "sigmoid": _qvr_qiec_pt.sigmoid,
        "relu": lambda value: _qvr_qiec_pt.maximum(value, 0.0),
        "relu6": lambda value: _qvr_qiec_pt.clip(value, 0.0, 6.0),
        "elu": lambda value: _qvr_qiec_pt.switch(
            value > 0.0, value, _qvr_qiec_pt.expm1(value)
        ),
        "selu": lambda value: (
            1.0507009873554805
            * _qvr_qiec_pt.switch(
                value > 0.0, value, 1.6732632423543772 * _qvr_qiec_pt.expm1(value)
            )
        ),
        "gelu": lambda value: (
            0.5 * value * (1.0 + _qvr_qiec_pt.erf(value / _qvr_qiec_pt.sqrt(2.0)))
        ),
        "silu": lambda value: value * _qvr_qiec_pt.sigmoid(value),
        "mish": lambda value: value * _qvr_qiec_pt.tanh(_qvr_qiec_pt.softplus(value)),
        "softplus": _qvr_qiec_pt.softplus,
        "logsigmoid": lambda value: -_qvr_qiec_pt.softplus(-value),
        "softsign": lambda value: value / (1.0 + _qvr_qiec_pt.abs(value)),
    }
)
