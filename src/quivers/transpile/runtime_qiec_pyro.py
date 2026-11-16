"""QIEC distribution bridge for generated Pyro modules.

A QIEC tensor is a nested tuple until it reaches a distribution, where
it becomes a tensor of the host's default floating dtype; a log density
is the distribution's own ``log_prob`` at the converted value.
"""

# ruff: noqa: F821
# pyright: reportUndefinedVariable=false
# The bridge is grafted after the shared runtime into a generated module,
# which binds the host library names and the runtime's helpers before the
# bridge is read.

import torch as _qvr_qiec_torch_module


def _qvr_qiec_nested(value):
    if isinstance(value, tuple):
        return [_qvr_qiec_nested(item) for item in value]
    if isinstance(value, bool):
        return float(value)
    return value


def _qvr_qiec_array(value):
    value = _qvr_qiec_value(value)
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(_qvr_qiec_nested(value), dtype=torch.get_default_dtype())


def _qvr_qiec_log_density(distribution, value):
    return distribution.log_prob(_qvr_qiec_array(value))


def _qvr_qiec_log_mass(distribution, retained=0, shape=()):
    # Query raw measures through their own normalizer. Host probability
    # distributions are normalized by definition; never estimate at a point.
    normalizer = getattr(distribution, "log_normalizer", None)
    if callable(normalizer):
        mass = torch.as_tensor(normalizer())
    else:
        parameters = [
            value
            for name in distribution.arg_constraints
            if isinstance(value := distribution.__dict__.get(name), torch.Tensor)
        ]
        anchor = sum(
            torch.where(torch.isfinite(point), point, 0) * 0
            for value in parameters
            for point in (value[(0,) * value.ndim] if value.numel() else value.sum(),)
        )
        mass = torch.as_tensor(anchor)
    shape = tuple(shape) or tuple(distribution.batch_shape)
    mass = mass.expand(shape)
    if (
        isinstance(retained, bool)
        or not isinstance(retained, int)
        or not 0 <= retained <= mass.ndim
    ):
        raise ValueError("log_mass retained axes must be within the batch rank")
    return mass.sum(tuple(range(retained, mass.ndim))) if mass.ndim > retained else mass


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
        return pyro.sample(name(label), distribution)

    def add(request):
        (weight,) = request["arguments"]
        pyro.factor(name("score"), _qvr_qiec_array(weight))
        return None

    return {
        (random_instance, sample_operation): sample,
        (score_instance, add_operation): add,
    }


def _qvr_qiec_torch(function):
    # A torch function over a value that may still be a Python number.
    return lambda *values: function(
        *(_qvr_qiec_torch_module.as_tensor(value) for value in values)
    )


def _qvr_qiec_torch_observation_weight(value):
    tensor = _qvr_qiec_torch_module.as_tensor(value)
    if not tensor.is_floating_point() or not bool(
        (_qvr_qiec_torch_module.isfinite(tensor) & (tensor >= 0)).all()
    ):
        raise ValueError(
            "observation weights must be finite and nonnegative real values"
        )
    return tensor


def _qvr_qiec_torch_weight_array(value, device):
    if isinstance(value, tuple):
        if not value:
            return _qvr_qiec_torch_module.empty((0,), device=device)
        # as_tensor(tuple_of_live_tensors) copies their scalar values and
        # severs autograd. Stack the checked tensor positions instead.
        return _qvr_qiec_torch_module.stack(
            tuple(_qvr_qiec_torch_weight_array(item, device) for item in value)
        )
    return _qvr_qiec_torch_module.as_tensor(value, device=device)


def _qvr_qiec_torch_scale_weight(coefficient, density):
    device = (
        density.device
        if isinstance(density, _qvr_qiec_torch_module.Tensor)
        else coefficient.device
        if isinstance(coefficient, _qvr_qiec_torch_module.Tensor)
        else None
    )
    coefficient = _qvr_qiec_torch_weight_array(coefficient, device)
    density = _qvr_qiec_torch_weight_array(density, device)
    safe = _qvr_qiec_torch_module.where(
        coefficient == 0, _qvr_qiec_torch_module.zeros_like(density), density
    )
    return coefficient * safe


_qvr_qiec_host_math.update(
    {
        "observation_weight": _qvr_qiec_torch_observation_weight,
        "scale_weight": _qvr_qiec_torch_scale_weight,
        "min_real": _qvr_qiec_torch(_qvr_qiec_torch_module.minimum),
        "max_real": _qvr_qiec_torch(_qvr_qiec_torch_module.maximum),
        "pow_real": _qvr_qiec_torch(_qvr_qiec_torch_module.pow),
        "exp": _qvr_qiec_torch(_qvr_qiec_torch_module.exp),
        "log": _qvr_qiec_torch(_qvr_qiec_torch_module.log),
        "sqrt": _qvr_qiec_torch(_qvr_qiec_torch_module.sqrt),
        "expm1": _qvr_qiec_torch(_qvr_qiec_torch_module.expm1),
        "log1p": _qvr_qiec_torch(_qvr_qiec_torch_module.log1p),
        "log2": _qvr_qiec_torch(_qvr_qiec_torch_module.log2),
        "log10": _qvr_qiec_torch(_qvr_qiec_torch_module.log10),
        "rsqrt": _qvr_qiec_torch(_qvr_qiec_torch_module.rsqrt),
        "sign": _qvr_qiec_torch(_qvr_qiec_torch_module.sign),
        "reciprocal": _qvr_qiec_torch(_qvr_qiec_torch_module.reciprocal),
        "sin": _qvr_qiec_torch(_qvr_qiec_torch_module.sin),
        "cos": _qvr_qiec_torch(_qvr_qiec_torch_module.cos),
        "tan": _qvr_qiec_torch(_qvr_qiec_torch_module.tan),
        "asin": _qvr_qiec_torch(_qvr_qiec_torch_module.asin),
        "acos": _qvr_qiec_torch(_qvr_qiec_torch_module.acos),
        "atan": _qvr_qiec_torch(_qvr_qiec_torch_module.atan),
        "sinh": _qvr_qiec_torch(_qvr_qiec_torch_module.sinh),
        "cosh": _qvr_qiec_torch(_qvr_qiec_torch_module.cosh),
        "tanh": _qvr_qiec_torch(_qvr_qiec_torch_module.tanh),
        "asinh": _qvr_qiec_torch(_qvr_qiec_torch_module.asinh),
        "acosh": _qvr_qiec_torch(_qvr_qiec_torch_module.acosh),
        "atanh": _qvr_qiec_torch(_qvr_qiec_torch_module.atanh),
        "floor": _qvr_qiec_torch(_qvr_qiec_torch_module.floor),
        "ceil": _qvr_qiec_torch(_qvr_qiec_torch_module.ceil),
        "round": _qvr_qiec_torch(_qvr_qiec_torch_module.round),
        "trunc": _qvr_qiec_torch(_qvr_qiec_torch_module.trunc),
        "select_real": _qvr_qiec_torch(_qvr_qiec_torch_module.where),
        "select_int": _qvr_qiec_torch(_qvr_qiec_torch_module.where),
        "bool_to_real": _qvr_qiec_torch(
            lambda value: value.to(_qvr_qiec_torch_module.get_default_dtype())
        ),
        "bool_to_int": _qvr_qiec_torch(
            lambda value: value.to(_qvr_qiec_torch_module.int64)
        ),
        "real_to_int": _qvr_qiec_torch(
            lambda value: _qvr_qiec_torch_module.trunc(value).to(
                _qvr_qiec_torch_module.int64
            )
        ),
        "erf": _qvr_qiec_torch(_qvr_qiec_torch_module.erf),
        "erfc": _qvr_qiec_torch(_qvr_qiec_torch_module.erfc),
        "erfinv": _qvr_qiec_torch(_qvr_qiec_torch_module.erfinv),
        "lgamma": _qvr_qiec_torch(_qvr_qiec_torch_module.lgamma),
        "digamma": _qvr_qiec_torch(_qvr_qiec_torch_module.digamma),
        "sigmoid": _qvr_qiec_torch(_qvr_qiec_torch_module.sigmoid),
        "relu": _qvr_qiec_torch(_qvr_qiec_torch_module.relu),
        "relu6": _qvr_qiec_torch(
            lambda value: _qvr_qiec_torch_module.clamp(value, 0.0, 6.0)
        ),
        "elu": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.elu),
        "selu": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.selu),
        "gelu": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.gelu),
        "silu": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.silu),
        "mish": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.mish),
        "softplus": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.softplus),
        "logsigmoid": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.logsigmoid),
        "softsign": _qvr_qiec_torch(_qvr_qiec_torch_module.nn.functional.softsign),
    }
)


def _qvr_log_probability(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or not bool(
        (torch.isfinite(value) & (value >= 0) & (value <= 1)).all()
    ):
        raise ValueError("log_probability requires finite real values in [0, 1]")
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.log(), -torch.inf)


def _qvr_log1mexp(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or not bool(
        (~torch.isnan(value) & (value <= 0)).all()
    ):
        raise ValueError("log1mexp requires nonpositive real values")
    interior = torch.isfinite(value) & (value < 0)
    safe = torch.where(interior, value, -torch.ones_like(value))
    near = safe > -0.6931471805599453
    near_arg = torch.where(near, safe, -torch.ones_like(safe))
    far_arg = torch.where(near, -torch.ones_like(safe), safe)
    result = torch.where(
        near, torch.log(-torch.expm1(near_arg)), torch.log1p(-far_arg.exp())
    )
    return torch.where(
        value == 0, -torch.inf, torch.where(value.isneginf(), 0.0, result)
    )


def _qvr_log_ndtr(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or bool(torch.isnan(value).any()):
        raise ValueError("log_ndtr requires real values other than NaN")
    safe = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    result = torch.special.log_ndtr(safe)
    return torch.where(
        value.isneginf(), -torch.inf, torch.where(value.isposinf(), 0.0, result)
    )


_qvr_qiec_host_math.update(
    {
        "log_probability": _qvr_log_probability,
        "log1mexp": _qvr_log1mexp,
        "log_ndtr": _qvr_log_ndtr,
    }
)


def _qvr_positive_parameter(value):
    value = torch.as_tensor(value)
    if not value.is_floating_point() or not bool(
        (torch.isfinite(value) & (value > 0)).all()
    ):
        raise ValueError(
            "positive_parameter requires finite strictly positive real values"
        )
    return value


_qvr_qiec_host_math["positive_parameter"] = _qvr_positive_parameter
