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


_qvr_qiec_host_math.update(
    {
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
