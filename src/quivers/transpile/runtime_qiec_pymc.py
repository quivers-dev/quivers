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
from pytensor.raise_op import Assert as _qvr_qiec_assert


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


def _qvr_qiec_log_mass(distribution, retained=0, shape=()):
    if isinstance(distribution, _QvrQiecShifted):
        distribution = distribution.distribution
    normalizer = getattr(distribution, "log_normalizer", None)
    if callable(normalizer):
        mass = _qvr_qiec_pt.as_tensor_variable(normalizer())
    else:
        rank = distribution.owner.op.ndim_supp
        mass = _qvr_qiec_pt.zeros(distribution.shape[: distribution.ndim - rank])
    if shape:
        mass = _qvr_qiec_pt.broadcast_to(mass, shape)
    if (
        isinstance(retained, bool)
        or not isinstance(retained, int)
        or not 0 <= retained <= mass.ndim
    ):
        raise ValueError("log_mass retained axes must be within the batch rank")
    return (
        mass.sum(axis=tuple(range(retained, mass.ndim)))
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


def _qvr_qiec_pt_observation_weight(value):
    tensor = _qvr_qiec_pt.as_tensor_variable(value)
    if not tensor.dtype.startswith("float"):
        raise ValueError(
            "observation weights must be finite and nonnegative real values"
        )
    is_nan = _qvr_qiec_pt.isnan(tensor)
    is_infinite = _qvr_qiec_pt.isinf(tensor)
    # PyTensor's overloads also describe non-tensor inputs; this path has
    # already constructed a floating TensorVariable, so both are predicates.
    assert is_nan is not None and is_infinite is not None
    valid = _qvr_qiec_pt.all(~is_nan & ~is_infinite & (tensor >= 0))
    return _qvr_qiec_assert("observation weights must be finite and nonnegative")(
        tensor, valid
    )


def _qvr_qiec_pt_scale_weight(coefficient, density):
    coefficient = _qvr_qiec_pt.as_tensor_variable(coefficient)
    density = _qvr_qiec_pt.as_tensor_variable(density)
    safe = _qvr_qiec_pt.switch(
        _qvr_qiec_pt.eq(coefficient, 0), _qvr_qiec_pt.zeros_like(density), density
    )
    return coefficient * safe


_qvr_qiec_host_math.update(
    {
        "observation_weight": _qvr_qiec_pt_observation_weight,
        "scale_weight": _qvr_qiec_pt_scale_weight,
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
        "select_real": _qvr_qiec_pt.switch,
        "select_int": _qvr_qiec_pt.switch,
        "bool_to_real": lambda value: _qvr_qiec_pt.as_tensor_variable(value).astype(
            "float64"
        ),
        "bool_to_int": lambda value: _qvr_qiec_pt.as_tensor_variable(value).astype(
            "int64"
        ),
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


# qvr-shared-probability:pymc:start
# qvr-shared-probability:pymc:end


_qvr_qiec_host_math.update(
    {
        "log_probability": _qvr_log_probability,
        "log1mexp": _qvr_log1mexp,
        "log_ndtr": _qvr_log_ndtr,
    }
)


def _qvr_select(condition, then, otherwise):
    pt = pymc.pytensorf.pt
    if isinstance(condition, bool):
        condition = pt.constant(condition, dtype="bool")
    return pt.switch(condition, then, otherwise)


_qvr_qiec_host_math.update({"select_real": _qvr_select, "select_int": _qvr_select})


_qvr_qiec_host_math["positive_parameter"] = _qvr_positive_parameter


_qvr_qiec_host_math.update(
    {
        "observation_status": _qvr_observation_status,
        "interval_upper": _qvr_interval_upper,
    }
)
