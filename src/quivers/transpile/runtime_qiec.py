"""Target-side QIEC runtime embedded in generated Python modules.

Generated functions carry the checked structural ABI with every request and
handler delimiter. Process-local callbacks supply code only; they cannot
override checked coverage, grades, or types.
"""

import contextvars as _qvr_qiec_contextvars
import math as _qvr_qiec_math


def _qvr_qiec_pure(value):
    return ("pure", value)


def _qvr_qiec_call(thunk, frames=(), tail=False):
    # A deferred computation. Calls are forced by the trampolines in
    # _qvr_qiec_run and the handler walk rather than when they are built, so
    # a recursive QIEC computation does not consume host stack per call.
    # ``frames`` are the dynamic address frames the result runs under, and
    # ``tail`` records that the call retires the frame of the computation
    # that made it.
    return ("call", thunk, tuple(frames), tail)


def _qvr_qiec_join_frames(outer, inner, tail):
    # A tail call replaces the innermost enclosing call frame, so a
    # tail-recursive loop names its requests under one frame at a time; a
    # local instance beneath the call is not a call frame and stays.
    if tail and outer and outer[-1][0] == "call":
        return (*outer[:-1], *inner)
    return (*outer, *inner)


def _qvr_qiec_scoped(computation, frames):
    if not frames or computation[0] == "pure":
        return computation
    if computation[0] == "call":
        _, thunk, inner, tail = computation
        return ("call", thunk, _qvr_qiec_join_frames(frames, inner, tail), False)
    if computation[0] == "bind":
        _, inner, continuation, captures = computation
        return (
            "bind",
            _qvr_qiec_scoped(inner, frames),
            lambda value: _qvr_qiec_scoped(continuation(value), frames),
            captures,
        )
    request = dict(computation[1])
    static, dynamic, resumptions = request["address"]
    # Outer scopes stamp their frames after inner ones did, so they go in
    # front to keep the address outermost first.
    request["address"] = (static, (*frames, *dynamic), resumptions)
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_scoped(computation[2](value), frames),
    )


def _qvr_qiec_force(computation, pending=()):
    # The trampoline. Deferred calls are forced and deferred binds are
    # unfolded onto an explicit continuation stack, so however deep a
    # recursion is, the host stack stays flat. A request surfacing beneath
    # pending binds carries them in its continuation, and their captures,
    # so a multi-shot resumption still sees everything it copies.
    stack = list(pending)
    current = computation
    while True:
        tag = current[0]
        if tag == "call":
            _, thunk, frames, _ = current
            current = _qvr_qiec_scoped(thunk(), frames)
        elif tag == "bind":
            _, inner, continuation, captures = current
            stack.append((continuation, captures))
            current = inner
        elif tag == "pure":
            if not stack:
                return current
            continuation, _ = stack.pop()
            current = continuation(current[1])
        else:
            if not stack:
                return current
            request = dict(current[1])
            request["captures"] = (
                *request.get("captures", ()),
                *(capture for _, captures in stack for capture in captures),
            )
            rest = tuple(stack)
            resume_effect = current[2]
            return (
                "effect",
                request,
                lambda value: _qvr_qiec_force(resume_effect(value), rest),
            )


_qvr_qiec_serials = {"call": 0, "instance": 0}


def _qvr_qiec_enter_call(name, thunk, tail):
    _qvr_qiec_serials["call"] += 1
    serial = _qvr_qiec_serials["call"]
    return _qvr_qiec_call(thunk, (("call", name + "#" + str(serial)),), tail)


def _qvr_qiec_instance(thunk):
    _qvr_qiec_serials["instance"] += 1
    return _qvr_qiec_call(thunk, (("instance", _qvr_qiec_serials["instance"]),))


def _qvr_qiec_resume(resume, value):
    return _qvr_qiec_as_computation(resume(value))


def _qvr_qiec_if(condition, then, otherwise):
    condition = _qvr_qiec_value(condition)
    if _qvr_qiec_is_host(condition):
        # A host's Boolean array decides the branch as the host reads
        # it; a host that traces rather than evaluates raises here, as
        # it does for any data-dependent branch.
        condition = bool(condition)
    if not isinstance(condition, bool):
        raise TypeError("QIEC if condition is not a Boolean")
    return then() if condition else otherwise()


def _qvr_qiec_is_host(value):
    # A value of the host library rather than of Python: a tensor or a
    # symbolic variable, which the host's own functions operate on.
    return value is not None and not isinstance(
        value, (bool, int, float, str, tuple, list, dict, bytes)
    )


# Host spellings of the primitives, which the target's bridge fills so a
# primitive applied to the host's array stays on the host's array and
# keeps its gradient; a primitive applied to plain Python values uses the
# Python table.
_qvr_qiec_host_math = {}


def _qvr_qiec_div_int(a, b):
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def _qvr_qiec_mod_int(a, b):
    return a - b * _qvr_qiec_div_int(a, b)


# The closed primitive table. Names and semantics mirror the kernel
# registry; integer division and remainder truncate toward zero on every
# host, which is why they are spelled out rather than borrowed.
_qvr_qiec_primitives = {
    "add_int": lambda a, b: a + b,
    "sub_int": lambda a, b: a - b,
    "mul_int": lambda a, b: a * b,
    "div_int": _qvr_qiec_div_int,
    "mod_int": _qvr_qiec_mod_int,
    "neg_int": lambda a: -a,
    "abs_int": abs,
    "min_int": min,
    "max_int": max,
    "add_real": lambda a, b: a + b,
    "sub_real": lambda a, b: a - b,
    "mul_real": lambda a, b: a * b,
    "div_real": lambda a, b: a / b,
    "neg_real": lambda a: -a,
    "abs_real": abs,
    "min_real": min,
    "max_real": max,
    "pow_real": _qvr_qiec_math.pow,
    "exp": _qvr_qiec_math.exp,
    "log": _qvr_qiec_math.log,
    "sqrt": _qvr_qiec_math.sqrt,
    "eq_int": lambda a, b: a == b,
    "ne_int": lambda a, b: a != b,
    "lt_int": lambda a, b: a < b,
    "le_int": lambda a, b: a <= b,
    "gt_int": lambda a, b: a > b,
    "ge_int": lambda a, b: a >= b,
    "eq_real": lambda a, b: a == b,
    "ne_real": lambda a, b: a != b,
    "lt_real": lambda a, b: a < b,
    "le_real": lambda a, b: a <= b,
    "gt_real": lambda a, b: a > b,
    "ge_real": lambda a, b: a >= b,
    "eq_bool": lambda a, b: a == b,
    "ne_bool": lambda a, b: a != b,
    "eq_string": lambda a, b: a == b,
    "ne_string": lambda a, b: a != b,
    "and": lambda a, b: a and b,
    "or": lambda a, b: a or b,
    "not": lambda a: not a,
    "concat": lambda a, b: a + b,
    "int_to_real": float,
    "int_to_string": str,
    "real_to_int": _qvr_qiec_math.trunc,
    "bool_to_real": float,
    "bool_to_int": int,
    "select_real": lambda c, a, b: a if c else b,
    "select_int": lambda c, a, b: a if c else b,
    "expm1": _qvr_qiec_math.expm1,
    "log1p": _qvr_qiec_math.log1p,
    "log2": _qvr_qiec_math.log2,
    "log10": _qvr_qiec_math.log10,
    "rsqrt": lambda a: 1.0 / _qvr_qiec_math.sqrt(a),
    "square": lambda a: a * a,
    "sign": lambda a: float((a > 0.0) - (a < 0.0)),
    "reciprocal": lambda a: 1.0 / a,
    "sin": _qvr_qiec_math.sin,
    "cos": _qvr_qiec_math.cos,
    "tan": _qvr_qiec_math.tan,
    "asin": _qvr_qiec_math.asin,
    "acos": _qvr_qiec_math.acos,
    "atan": _qvr_qiec_math.atan,
    "sinh": _qvr_qiec_math.sinh,
    "cosh": _qvr_qiec_math.cosh,
    "tanh": _qvr_qiec_math.tanh,
    "asinh": _qvr_qiec_math.asinh,
    "acosh": _qvr_qiec_math.acosh,
    "atanh": _qvr_qiec_math.atanh,
    "floor": lambda a: float(_qvr_qiec_math.floor(a)),
    "ceil": lambda a: float(_qvr_qiec_math.ceil(a)),
    "round": lambda a: float(round(a)),
    "trunc": lambda a: float(_qvr_qiec_math.trunc(a)),
    "erf": _qvr_qiec_math.erf,
    "erfc": _qvr_qiec_math.erfc,
    "erfinv": lambda a: _qvr_qiec_erfinv(a),
    "lgamma": _qvr_qiec_math.lgamma,
    "digamma": lambda a: _qvr_qiec_digamma(a),
    "sigmoid": lambda a: _qvr_qiec_sigmoid(a),
    "relu": lambda a: max(a, 0.0),
    "relu6": lambda a: min(max(a, 0.0), 6.0),
    "elu": lambda a: a if a > 0.0 else _qvr_qiec_math.expm1(a),
    "selu": lambda a: (
        1.0507009873554805
        * (a if a > 0.0 else 1.6732632423543772 * (_qvr_qiec_math.exp(a) - 1.0))
    ),
    "gelu": lambda a: (
        0.5 * a * (1.0 + _qvr_qiec_math.erf(a / _qvr_qiec_math.sqrt(2.0)))
    ),
    "silu": lambda a: a * _qvr_qiec_sigmoid(a),
    "mish": lambda a: a * _qvr_qiec_math.tanh(_qvr_qiec_softplus(a)),
    "softplus": lambda a: _qvr_qiec_softplus(a),
    "logsigmoid": lambda a: -_qvr_qiec_softplus(-a),
    "softsign": lambda a: a / (1.0 + abs(a)),
    "as_weight": lambda a: a,
    "weight_value": lambda a: a,
    "add_weight": lambda a, b: a + b,
    "scale_weight": lambda a, b: a * b,
}


def _qvr_qiec_sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + _qvr_qiec_math.exp(-value))
    exponent = _qvr_qiec_math.exp(value)
    return exponent / (1.0 + exponent)


def _qvr_qiec_softplus(value):
    return max(value, 0.0) + _qvr_qiec_math.log1p(_qvr_qiec_math.exp(-abs(value)))


def _qvr_qiec_erfinv(value):
    if value <= -1.0 or value >= 1.0:
        if value == 1.0:
            return _qvr_qiec_math.inf
        if value == -1.0:
            return -_qvr_qiec_math.inf
        raise ValueError("erfinv is defined on [-1, 1]")
    if value == 0.0:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    magnitude = abs(value)
    if magnitude < 0.7:
        square = magnitude * magnitude
        estimate = (
            magnitude
            * (
                ((-0.140543331 * square + 0.914624893) * square - 1.645349621) * square
                + 0.886226899
            )
            / (
                (
                    ((0.012229801 * square - 0.329097515) * square + 1.442710462)
                    * square
                    - 2.118377725
                )
                * square
                + 1.0
            )
        )
    else:
        tail = _qvr_qiec_math.sqrt(-_qvr_qiec_math.log((1.0 - magnitude) / 2.0))
        estimate = (
            ((1.641345311 * tail + 3.429567803) * tail - 1.62490649) * tail
            - 1.970840454
        ) / ((1.637067800 * tail + 3.543889200) * tail + 1.0)
    for _ in range(3):
        error = _qvr_qiec_math.erf(estimate) - magnitude
        estimate -= error / (
            2.0
            / _qvr_qiec_math.sqrt(_qvr_qiec_math.pi)
            * _qvr_qiec_math.exp(-estimate * estimate)
        )
    return sign * estimate


def _qvr_qiec_digamma(value):
    if value <= 0.0 and value == _qvr_qiec_math.floor(value):
        raise ValueError("digamma has a pole at nonpositive integers")
    if value < 0.0:
        return _qvr_qiec_digamma(1.0 - value) - _qvr_qiec_math.pi / _qvr_qiec_math.tan(
            _qvr_qiec_math.pi * value
        )
    result = 0.0
    while value < 6.0:
        result -= 1.0 / value
        value += 1.0
    inverse = 1.0 / value
    square = inverse * inverse
    return (
        result
        + _qvr_qiec_math.log(value)
        - 0.5 * inverse
        - square
        * (
            1.0 / 12.0
            - square
            * (
                1.0 / 120.0
                - square * (1.0 / 252.0 - square * (1.0 / 240.0 - square / 132.0))
            )
        )
    )


def _qvr_qiec_broadcast(implementation, arguments):
    tensors = [argument for argument in arguments if isinstance(argument, tuple)]
    if not tensors:
        return implementation(*arguments)
    length = len(tensors[0])
    if any(len(tensor) != length for tensor in tensors):
        raise ValueError("QIEC primitive applied to tensors of differing shapes")
    return tuple(
        _qvr_qiec_broadcast(
            implementation,
            tuple(
                argument[index] if isinstance(argument, tuple) else argument
                for argument in arguments
            ),
        )
        for index in range(length)
    )


def _qvr_qiec_primitive(name, arguments):
    implementation = _qvr_qiec_primitives.get(name)
    if implementation is None:
        raise KeyError("unknown QIEC primitive " + name)
    values = tuple(_qvr_qiec_value(argument) for argument in arguments)
    host = _qvr_qiec_host_math.get(name)
    if host is not None and any(_qvr_qiec_is_host(value) for value in values):
        return _qvr_qiec_broadcast(host, values)
    return _qvr_qiec_broadcast(implementation, values)


def _qvr_qiec_gather(value, index):
    value = _qvr_qiec_value(value)
    index = _qvr_qiec_value(index)
    if not isinstance(value, tuple):
        raise TypeError("QIEC gather from a non-tensor runtime value")
    if isinstance(index, tuple):
        return tuple(_qvr_qiec_gather(value, item) for item in index)
    return value[index]


def _qvr_qiec_flat(value):
    if isinstance(value, tuple):
        return [entry for item in value for entry in _qvr_qiec_flat(item)]
    return [value]


def _qvr_qiec_reduce(operator, value):
    entries = _qvr_qiec_flat(_qvr_qiec_value(value))
    if operator == "sum":
        return sum(entries)
    if operator == "mean":
        return sum(entries) / len(entries)
    if operator == "max":
        return max(entries)
    if operator == "min":
        return min(entries)
    if operator == "prod":
        result = 1
        for entry in entries:
            result *= entry
        return result
    peak = max(entries)
    return peak + _qvr_qiec_math.log(
        sum(_qvr_qiec_math.exp(entry - peak) for entry in entries)
    )


def _qvr_qiec_rowwise(operator, value):
    value = _qvr_qiec_value(value)
    if value and isinstance(value[0], tuple):
        return tuple(_qvr_qiec_rowwise(operator, item) for item in value)
    row = [float(item) for item in value]
    if operator == "softmax":
        peak = max(row)
        weights = [_qvr_qiec_math.exp(item - peak) for item in row]
        total = sum(weights)
        return tuple(weight / total for weight in weights)
    if operator == "log_softmax":
        peak = max(row)
        normalizer = peak + _qvr_qiec_math.log(
            sum(_qvr_qiec_math.exp(item - peak) for item in row)
        )
        return tuple(item - normalizer for item in row)
    if operator == "cumsum":
        running = 0.0
        out = []
        for item in row:
            running += item
            out.append(running)
        return tuple(out)
    if operator == "sort":
        return tuple(sorted(row))
    total = sum(row)
    return tuple(item / total for item in row)


def _qvr_qiec_weight_sum(value):
    return sum(_qvr_qiec_flat(_qvr_qiec_value(value)))


def _qvr_qiec_segment_sum(value, index, groups):
    totals = [0.0] * groups
    for weight, group in zip(_qvr_qiec_value(value), _qvr_qiec_value(index)):
        totals[group] += weight
    return tuple(totals)


def _qvr_qiec_project(value, position):
    value = _qvr_qiec_value(value)
    if not isinstance(value, tuple) or position >= len(value):
        raise TypeError("QIEC projection from a non-product runtime value")
    return value[position]


def _qvr_qiec_static_kind(argument):
    kind = argument.get("kind") if isinstance(argument, dict) else None
    if kind in {
        "type-variable",
        "type-application",
        "function-type",
        "equality-type",
    }:
        return "type"
    if kind in {"effect-variable", "effect-ref"}:
        return "effect"
    return "index"


def _qvr_qiec_static_environment(telescope, arguments):
    if arguments is None:
        arguments = []
    if isinstance(arguments, dict):
        expected = {binder["name"] for binder in telescope}
        if set(arguments) != expected:
            raise TypeError("QIEC static arguments do not match the checked telescope")
        ordered = [arguments[binder["name"]] for binder in telescope]
    else:
        ordered = list(arguments)
    if len(ordered) != len(telescope):
        raise TypeError(
            "QIEC static argument arity does not match the checked telescope"
        )
    environment = {}
    for binder, argument in zip(telescope, ordered, strict=True):
        if _qvr_qiec_static_kind(argument) != binder["kind"]:
            raise TypeError(
                "QIEC static argument kind does not match the checked telescope"
            )
        environment[binder["name"]] = argument
    return environment


def _qvr_qiec_specialize(value, environment):
    if isinstance(value, list):
        return [_qvr_qiec_specialize(item, environment) for item in value]
    if isinstance(value, tuple):
        return tuple(_qvr_qiec_specialize(item, environment) for item in value)
    if not isinstance(value, dict):
        return value
    if value.get("kind") in {
        "type-variable",
        "index-variable",
        "effect-variable",
    }:
        key = value.get("identity") or value.get("name")
        if key in environment:
            return environment[key]
    return {key: _qvr_qiec_specialize(item, environment) for key, item in value.items()}


_qvr_qiec_ambient_handlers = _qvr_qiec_contextvars.ContextVar(
    "qvr_qiec_ambient_handlers", default=()
)


def _qvr_qiec_unique(items):
    result = []
    for item in items:
        if not any(candidate is item for candidate in result):
            result.append(item)
    return tuple(result)


def _qvr_qiec_with_ambient(controller, thunk):
    token = _qvr_qiec_ambient_handlers.set(
        (*_qvr_qiec_ambient_handlers.get(), controller)
    )
    try:
        return thunk()
    finally:
        _qvr_qiec_ambient_handlers.reset(token)


def _qvr_qiec_effect(request, static_environment):
    request = _qvr_qiec_specialize(request, static_environment)
    request.setdefault("captures", ())
    controllers = _qvr_qiec_unique(
        (*request.get("handler_captures", ()), *_qvr_qiec_ambient_handlers.get())
    )
    request["handler_captures"] = controllers
    request["handler_states"] = tuple(
        controller["capture"]() for controller in controllers
    )
    return ("effect", request, _qvr_qiec_pure)


def _qvr_qiec_bind(computation, continuation, captures=()):
    if computation[0] == "pure":
        return continuation(computation[1])
    if computation[0] in ("call", "bind"):
        return ("bind", computation, continuation, tuple(captures))
    request = dict(computation[1])
    request["captures"] = (*request.get("captures", ()), *captures)
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_bind(computation[2](value), continuation, captures),
    )


def _qvr_qiec_binding(value):
    return (
        isinstance(value, dict) and value.get("qiec") == "binding" and "value" in value
    )


def _qvr_qiec_value(value):
    return value["value"] if _qvr_qiec_binding(value) else value


def _qvr_qiec_constructor(
    constructor, static_arguments, fields, result_type, static_environment
):
    return {
        "qiec": "constructor",
        "constructor": constructor,
        "static_arguments": tuple(
            _qvr_qiec_specialize(static_arguments, static_environment)
        ),
        "fields": tuple(fields),
        "result_type": _qvr_qiec_specialize(result_type, static_environment),
    }


def _qvr_qiec_evidence(evidence, static_environment):
    return {
        "qiec": "evidence",
        "evidence": _qvr_qiec_specialize(evidence, static_environment),
    }


def _qvr_qiec_transport(evidence, value, target_type, static_environment):
    # Equality evidence is proof irrelevant at runtime, but retaining both
    # endpoints in the generated call prevents an erasing backend boundary.
    _qvr_qiec_specialize(evidence, static_environment)
    _qvr_qiec_specialize(target_type, static_environment)
    return value


def _qvr_qiec_attachment(attachments, attachment, expected_type, static_environment):
    if attachment not in attachments:
        raise KeyError("missing QIEC attachment " + attachment)
    binding = attachments[attachment]
    if not _qvr_qiec_binding(binding) or "type" not in binding:
        raise TypeError("QIEC attachment must be a typed binding descriptor")
    expected_type = _qvr_qiec_specialize(expected_type, static_environment)
    if binding["type"] != expected_type:
        raise TypeError("QIEC attachment type disagrees with checked IR")
    return _qvr_qiec_value(
        _qvr_qiec_validate(
            binding.get("validator"), binding, "attachment " + attachment
        )
    )


def _qvr_qiec_case(value, branches, metadata, static_environment):
    metadata = _qvr_qiec_specialize(metadata, static_environment)
    if not isinstance(value, dict) or value.get("qiec") != "constructor":
        raise TypeError("QIEC case scrutinee is not a constructor value")
    constructor = value.get("constructor")
    if constructor not in metadata["branches"]:
        raise ValueError("constructor is not admitted by the checked QIEC case")
    if constructor not in branches:
        raise ValueError("no QIEC case branch for constructor " + str(constructor))
    branch = metadata["branches"][constructor]
    canonical = branch.get("static_arguments", ())
    actual = value.get("static_arguments", ())
    if len(actual) < len(canonical):
        raise TypeError("QIEC constructor static arguments do not match case branch")
    branch_environment = dict(static_environment)
    matched = actual[-len(canonical) :] if canonical else ()
    for variable, argument in zip(canonical, matched, strict=True):
        key = variable.get("identity") or variable.get("name")
        if key is None:
            raise TypeError("QIEC case branch has an unbindable static variable")
        branch_environment[key] = argument
    return branches[constructor](branch_environment, *value.get("fields", ()))


def _qvr_qiec_member(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _qvr_qiec_as_computation(value):
    if (
        isinstance(value, tuple)
        and value
        and value[0] in ("pure", "effect", "call", "bind")
    ):
        return value
    return _qvr_qiec_pure(value)


def _qvr_qiec_validate(validator, value, label):
    if validator is None:
        return value
    valid = validator(_qvr_qiec_value(value))
    if valid is False:
        raise TypeError("QIEC runtime value failed " + label)
    return value


def _qvr_qiec_validate_computation(computation, validator, label):
    return _qvr_qiec_bind(
        _qvr_qiec_as_computation(computation),
        lambda value: _qvr_qiec_pure(_qvr_qiec_validate(validator, value, label)),
    )


def _qvr_qiec_capture(locals_, attachment_ids, attachments):
    captured = [
        {"kind": "local", "label": name, "binding": binding}
        for name, binding in locals_.items()
    ]
    for attachment in attachment_ids:
        if attachment not in attachments:
            raise KeyError("missing QIEC attachment " + attachment)
        captured.append(
            {
                "kind": "attachment",
                "label": attachment,
                "binding": attachments[attachment],
            }
        )
    return tuple(captured)


def _qvr_qiec_duplicable(value):
    if _qvr_qiec_binding(value):
        return bool(value.get("duplicable", False))
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return True
    if isinstance(value, tuple):
        return all(_qvr_qiec_duplicable(item) for item in value)
    if isinstance(value, dict) and value.get("qiec") == "constructor":
        return all(_qvr_qiec_duplicable(item) for item in value.get("fields", ()))
    if isinstance(value, dict) and value.get("qiec") == "evidence":
        return True
    return False


def _qvr_qiec_validate_capture(request, current_handler):
    for capture in request.get("captures", ()):
        binding = capture["binding"]
        if not _qvr_qiec_duplicable(binding):
            raise RuntimeError(
                "QIEC unrestricted resumption captures nonduplicable "
                + capture["kind"]
                + " "
                + capture["label"]
            )
        if _qvr_qiec_binding(binding):
            capture["seed"] = binding["value"]
            if binding.get("mutable", False) and not callable(binding.get("fork")):
                raise RuntimeError(
                    "QIEC mutable duplicable binding requires fork " + capture["label"]
                )
    controllers = _qvr_qiec_unique(
        (*request.get("handler_captures", ()), current_handler)
    )
    for controller in controllers:
        controller["snapshot"]()
        handler = controller["get"]()
        if not _qvr_qiec_member(handler, "duplicable_context", False):
            raise RuntimeError(
                "QIEC unrestricted resumption captures nonduplicable handler context"
            )
        if _qvr_qiec_member(handler, "mutable_context", False) and not callable(
            _qvr_qiec_member(handler, "fork_context")
        ):
            raise RuntimeError(
                "QIEC mutable duplicable handler context requires fork_context"
            )


def _qvr_qiec_fork_capture(request, current_handler, shot):
    del shot
    for capture in request.get("captures", ()):
        binding = capture["binding"]
        if _qvr_qiec_binding(binding) and callable(binding.get("fork")):
            binding["value"] = binding["fork"](capture["seed"])
    controllers = _qvr_qiec_unique(
        (*request.get("handler_captures", ()), current_handler)
    )
    for controller in controllers:
        controller["install_fork"]()


def _qvr_qiec_readdress(computation, shot):
    computation = _qvr_qiec_force(computation)
    if computation[0] == "pure":
        return computation
    request = dict(computation[1])
    static, dynamic, resumptions = request["address"]
    request["address"] = (static, dynamic, (*resumptions, shot))
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_readdress(computation[2](value), shot),
    )


def _qvr_qiec_lifecycle(handler):
    return {"handler": handler, "finalized": False}


def _qvr_qiec_lifecycle_exit(lifecycle):
    if lifecycle["finalized"]:
        return
    lifecycle["finalized"] = True
    callback = _qvr_qiec_member(lifecycle["handler"], "on_exit")
    if callable(callback):
        callback()


def _qvr_qiec_lifecycle_drop(lifecycle):
    if lifecycle["finalized"]:
        return
    lifecycle["finalized"] = True
    callback = _qvr_qiec_member(lifecycle["handler"], "on_drop")
    if callable(callback):
        callback()


def _qvr_qiec_lifecycle_enter(handler):
    lifecycle = _qvr_qiec_lifecycle(handler)
    try:
        callback = _qvr_qiec_member(handler, "on_enter")
        if callable(callback):
            callback()
    except BaseException:
        _qvr_qiec_lifecycle_drop(lifecycle)
        raise
    return lifecycle


def _qvr_qiec_drop_request(request):
    for lifecycle in request.get("lifecycles", ()):
        _qvr_qiec_lifecycle_drop(lifecycle)


def _qvr_qiec_finalize(computation, lifecycle):
    current = _qvr_qiec_force(_qvr_qiec_as_computation(computation))
    if current[0] == "pure":
        _qvr_qiec_lifecycle_exit(lifecycle)
        return current
    request = dict(current[1])
    request["lifecycles"] = (*request.get("lifecycles", ()), lifecycle)
    return (
        "effect",
        request,
        lambda value: _qvr_qiec_finalize(current[2](value), lifecycle),
    )


def _qvr_qiec_handler_controller(handler_ref, lifecycle_ref, manifest):
    seed = [handler_ref[0]]

    def install_fork():
        prototype = seed[0]
        fork = _qvr_qiec_member(prototype, "fork_context")
        if not callable(fork):
            return
        clone = fork(prototype)
        try:
            _qvr_qiec_validate_handler(clone, manifest)
        except BaseException:
            callback = _qvr_qiec_member(clone, "on_drop")
            if callable(callback):
                callback()
            raise
        lifecycle = _qvr_qiec_lifecycle_enter(clone)
        handler_ref[0] = clone
        lifecycle_ref[0] = lifecycle

    return {
        "get": lambda: handler_ref[0],
        "seed": lambda: seed[0],
        "snapshot": lambda: seed.__setitem__(0, handler_ref[0]),
        "install_fork": install_fork,
        "capture": lambda: (handler_ref[0], lifecycle_ref[0]),
        "restore": lambda state: (
            handler_ref.__setitem__(0, state[0]),
            lifecycle_ref.__setitem__(0, state[1]),
        ),
    }


def _qvr_qiec_handler_operations(manifest):
    return {clause["operation"] for clause in manifest["clauses"]}


def _qvr_qiec_validate_handler(handler, manifest):
    declared = _qvr_qiec_member(handler, "definition")
    if declared is not None and declared != manifest:
        raise ValueError("QIEC runtime handler definition disagrees with checked IR")
    structural = _qvr_qiec_handler_operations(manifest)
    executable = set(_qvr_qiec_member(handler, "operations", {}))
    if executable != structural:
        raise ValueError("QIEC runtime handler clauses disagree with checked IR")
    if callable(_qvr_qiec_member(handler, "fork_context")) and not _qvr_qiec_member(
        handler, "duplicable_context", False
    ):
        raise ValueError("QIEC handler fork_context requires duplicable_context")
    if (
        _qvr_qiec_member(handler, "mutable_context", False)
        and _qvr_qiec_member(handler, "duplicable_context", False)
        and not callable(_qvr_qiec_member(handler, "fork_context"))
    ):
        raise ValueError("QIEC mutable duplicable handler requires fork_context")


def _qvr_qiec_invoke(entry, *arguments):
    invoke = _qvr_qiec_member(entry, "invoke", entry)
    if not callable(invoke):
        raise TypeError("QIEC runtime entry is not callable")
    return invoke(*arguments)


_qvr_qiec_authored = {}


def _qvr_qiec_handle(
    computation,
    instance,
    manifest,
    static_arguments,
    computation_static_environment,
    handlers,
    attachments=None,
    operations=None,
):
    static_arguments = _qvr_qiec_specialize(
        static_arguments, computation_static_environment
    )
    handler_static_environment = _qvr_qiec_static_environment(
        manifest["telescope"], static_arguments
    )
    manifest = _qvr_qiec_specialize(manifest, handler_static_environment)
    manifest["telescope"] = []
    handler_id = manifest["id"]
    if handler_id in handlers:
        prototype = handlers[handler_id]
    elif handler_id in _qvr_qiec_authored:
        # An authored handler's clauses are generated code beside this
        # runtime; an attachment under the same identity is an explicit
        # foreign stand-in and takes precedence.
        prototype = _qvr_qiec_authored[handler_id]
    else:
        raise KeyError("missing QIEC handler attachment " + handler_id)
    handler = prototype
    factory = _qvr_qiec_member(prototype, "context_factory")
    if callable(factory):
        handler = factory()
    if handler is prototype and _qvr_qiec_member(prototype, "mutable_context", False):
        raise ValueError("QIEC mutable handler requires context_factory")
    handler_ref = [handler]
    _qvr_qiec_validate_handler(handler_ref[0], manifest)
    lifecycle_ref = [_qvr_qiec_lifecycle_enter(handler)]
    root_lifecycle = lifecycle_ref[0]
    controller = _qvr_qiec_handler_controller(handler_ref, lifecycle_ref, manifest)
    grades = {clause["operation"]: clause["grade"] for clause in manifest["clauses"]}
    context = {
        "handler": handler_id,
        "definition": manifest,
        "static_arguments": static_arguments,
        "static": handler_static_environment,
        "attachments": {} if attachments is None else attachments,
        "handlers": handlers,
        "operations": {} if operations is None else operations,
        "resumption_uses": 0,
    }

    def walk(current):
        handler = handler_ref[0]
        current = _qvr_qiec_force(current)
        if current[0] == "pure":
            value = _qvr_qiec_validate(
                _qvr_qiec_member(handler, "input_validator"),
                current[1],
                "handler input type",
            )
            return_clause = _qvr_qiec_member(handler, "return")
            result = (
                value
                if return_clause is None
                else _qvr_qiec_invoke(return_clause, value, context)
            )
            return _qvr_qiec_finalize(
                _qvr_qiec_validate_computation(
                    result,
                    _qvr_qiec_member(handler, "output_validator"),
                    "handler output type",
                ),
                lifecycle_ref[0],
            )
        request = current[1]
        for captured_controller, state in zip(
            request.get("handler_captures", ()),
            request.get("handler_states", ()),
            strict=True,
        ):
            if captured_controller is controller:
                controller["restore"](state)
                handler = handler_ref[0]
                break

        def continue_forwarded(value):
            handler_ref[0] = forwarded_handler
            lifecycle_ref[0] = forwarded_lifecycle
            return _qvr_qiec_with_ambient(controller, lambda: walk(current[2](value)))

        if request["instance"] != instance:
            forwarded_handler = handler_ref[0]
            forwarded_lifecycle = lifecycle_ref[0]
            forwarded = dict(request)
            forwarded["handler_captures"] = _qvr_qiec_unique(
                (*request.get("handler_captures", ()), controller)
            )
            forwarded["handler_states"] = tuple(
                captured["capture"]() for captured in forwarded["handler_captures"]
            )
            forwarded["lifecycles"] = (
                *request.get("lifecycles", ()),
                lifecycle_ref[0],
            )
            return (
                "effect",
                forwarded,
                continue_forwarded,
            )
        clauses = _qvr_qiec_member(handler, "operations", {})
        operation = request["operation"]
        if request["effect"] != manifest["effect"]:
            raise TypeError("QIEC request and installed handler effect disagree")
        if operation not in clauses:
            if not manifest["total"] or manifest["forwards_unknown"]:
                forwarded_handler = handler_ref[0]
                forwarded_lifecycle = lifecycle_ref[0]
                forwarded = dict(request)
                forwarded["handler_captures"] = _qvr_qiec_unique(
                    (*request.get("handler_captures", ()), controller)
                )
                forwarded["handler_states"] = tuple(
                    captured["capture"]() for captured in forwarded["handler_captures"]
                )
                forwarded["lifecycles"] = (
                    *request.get("lifecycles", ()),
                    lifecycle_ref[0],
                )
                return (
                    "effect",
                    forwarded,
                    continue_forwarded,
                )
            raise KeyError("missing QIEC handler clause " + operation)
        uses = [0]
        grade = grades[operation]
        if grade == "omega":
            _qvr_qiec_validate_capture(request, controller)

        def resume(value):
            shot = uses[0]
            uses[0] += 1
            context["resumption_uses"] = uses[0]
            if grade == "0":
                raise RuntimeError("QIEC zero-grade clause resumed")
            if grade in ("aff", "1") and uses[0] > 1:
                raise RuntimeError("QIEC resumption exceeds grade " + grade)
            if grade == "omega":
                _qvr_qiec_fork_capture(request, controller, shot)
            value = _qvr_qiec_validate(
                _qvr_qiec_member(clauses[operation], "result_validator"),
                value,
                "handler operation result type",
            )
            resumed = _qvr_qiec_with_ambient(
                controller,
                lambda: walk(_qvr_qiec_readdress(current[2](value), shot)),
            )
            return resumed[1] if resumed[0] == "pure" else resumed

        clause_lifecycle = lifecycle_ref[0]
        try:
            result = _qvr_qiec_invoke(clauses[operation], request, resume, context)
        except BaseException:
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            raise
        if grade == "1" and uses[0] != 1:
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            raise RuntimeError("QIEC linear clause must resume exactly once")
        if uses[0] == 0:
            _qvr_qiec_drop_request(request)
        if grade == "0" and uses[0] != 0:
            raise RuntimeError("QIEC zero-grade clause cannot resume")
        return _qvr_qiec_finalize(
            _qvr_qiec_validate_computation(
                result,
                _qvr_qiec_member(handler, "output_validator"),
                "handler output type",
            ),
            clause_lifecycle,
        )

    try:
        computation = _qvr_qiec_with_ambient(
            controller, lambda: _qvr_qiec_force(computation())
        )
        return walk(computation)
    except BaseException:
        _qvr_qiec_lifecycle_drop(lifecycle_ref[0])
        _qvr_qiec_lifecycle_drop(root_lifecycle)
        raise


def _qvr_qiec_operation(operations, request):
    key = (request["instance"], request["operation"])
    entry = operations.get(key)
    if entry is None:
        by_instance = operations.get(request["instance"], {})
        entry = _qvr_qiec_member(by_instance, request["operation"])
    if entry is None:
        raise RuntimeError(
            "unhandled QIEC operation "
            + request["operation"]
            + " on "
            + request["instance"]
        )
    result = _qvr_qiec_invoke(entry, request)
    return _qvr_qiec_validate(
        _qvr_qiec_member(entry, "result_validator"),
        result,
        "operation result type",
    )


def _qvr_qiec_run(build, operations):
    # The entry builds its body after the serials reset, so every call and
    # allocation in a run is numbered from one and two identical runs
    # address their requests identically.
    _qvr_qiec_serials["call"] = 0
    _qvr_qiec_serials["instance"] = 0
    current = _qvr_qiec_force(build())
    while current[0] == "effect":
        try:
            current = _qvr_qiec_force(
                current[2](_qvr_qiec_operation(operations, current[1]))
            )
        except BaseException:
            _qvr_qiec_drop_request(current[1])
            raise
    return _qvr_qiec_value(current[1])
