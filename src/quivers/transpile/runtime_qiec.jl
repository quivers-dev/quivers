abstract type _QvrQiecComputation end
struct _QvrQiecPure <: _QvrQiecComputation
    value
end
struct _QvrQiecEffect <: _QvrQiecComputation
    request
    continuation
end
# A deferred computation. Calls are forced by the trampolines in _qvr_qiec_run and
# the handler walk rather than when they are built, so a recursive QIEC computation
# does not consume host stack per call. `frames` are the dynamic address frames the
# result runs under, and `tail` records that the call retires the frame of the
# computation that made it.
struct _QvrQiecCall <: _QvrQiecComputation
    thunk
    frames
    tail
end
struct _QvrQiecBind <: _QvrQiecComputation
    inner
    continuation
    captures
end
_qvr_qiec_pure(value) = _QvrQiecPure(value)
_qvr_qiec_call(thunk, frames=Any[], tail=false) = _QvrQiecCall(thunk, Any[frames...], tail)
function _qvr_qiec_join_frames(outer, inner, tail)
    if tail && !isempty(outer) && outer[end][1] == "call"
        return Any[outer[1:end-1]..., inner...]
    end
    return Any[outer..., inner...]
end
function _qvr_qiec_scoped(comp, frames)
    (isempty(frames) || comp isa _QvrQiecPure) && return comp
    comp isa _QvrQiecCall && return _QvrQiecCall(comp.thunk, _qvr_qiec_join_frames(frames, comp.frames, comp.tail), false)
    comp isa _QvrQiecBind && return _QvrQiecBind(_qvr_qiec_scoped(comp.inner, frames), value -> _qvr_qiec_scoped(comp.continuation(value), frames), comp.captures)
    request = copy(comp.request)
    request["address"] = Any[request["address"][1], Any[frames..., request["address"][2]...], request["address"][3]]
    return _QvrQiecEffect(request, value -> _qvr_qiec_scoped(comp.continuation(value), frames))
end
# The trampoline. Deferred calls are forced and deferred binds are unfolded onto an
# explicit continuation stack, so however deep a recursion is, the host stack stays
# flat. A request surfacing beneath pending binds carries them in its continuation,
# and their captures, so a multi-shot resumption still sees everything it copies.
function _qvr_qiec_force(comp, pending=Any[])
    stack = Any[pending...]
    current = comp
    while true
        if current isa _QvrQiecCall
            current = _qvr_qiec_scoped(current.thunk(), current.frames)
        elseif current isa _QvrQiecBind
            push!(stack, (current.continuation, current.captures))
            current = current.inner
        elseif current isa _QvrQiecPure
            isempty(stack) && return current
            continuation, _ = pop!(stack)
            current = continuation(current.value)
        else
            isempty(stack) && return current
            request = copy(current.request)
            request["captures"] = Any[get(request, "captures", Any[])..., (capture for (_, captures) in stack for capture in captures)...]
            rest = Any[stack...]
            resume_effect = current.continuation
            return _QvrQiecEffect(request, value -> _qvr_qiec_force(resume_effect(value), rest))
        end
    end
end
const _qvr_qiec_serials = Dict("call" => 0, "instance" => 0)
function _qvr_qiec_enter_call(name, thunk, tail)
    _qvr_qiec_serials["call"] += 1
    return _qvr_qiec_call(thunk, Any[Any["call", name * "#" * string(_qvr_qiec_serials["call"])]], tail)
end
function _qvr_qiec_instance(thunk)
    _qvr_qiec_serials["instance"] += 1
    return _qvr_qiec_call(thunk, Any[Any["instance", _qvr_qiec_serials["instance"]]], false)
end
_qvr_qiec_resume(resume, value) = _qvr_qiec_as_computation(resume(value))
function _qvr_qiec_if(condition, then, otherwise)
    condition = _qvr_qiec_value(condition)
    condition isa Bool || error("QIEC if condition is not a Boolean")
    return condition ? then() : otherwise()
end
# The closed primitive table. Names and semantics mirror the kernel registry;
# integer division and remainder truncate toward zero on every host.
const _qvr_qiec_primitives = Dict{String, Any}(
    "add_int" => (a, b) -> a + b,
    "sub_int" => (a, b) -> a - b,
    "mul_int" => (a, b) -> a * b,
    "div_int" => (a, b) -> div(a, b),
    "mod_int" => (a, b) -> rem(a, b),
    "neg_int" => a -> -a,
    "abs_int" => abs,
    "min_int" => min,
    "max_int" => max,
    "add_real" => (a, b) -> a + b,
    "sub_real" => (a, b) -> a - b,
    "mul_real" => (a, b) -> a * b,
    "div_real" => (a, b) -> a / b,
    "neg_real" => a -> -a,
    "abs_real" => abs,
    "min_real" => min,
    "max_real" => max,
    "pow_real" => (a, b) -> Float64(a) ^ Float64(b),
    "exp" => exp,
    "log" => log,
    "sqrt" => sqrt,
    "eq_int" => (a, b) -> a == b,
    "ne_int" => (a, b) -> a != b,
    "lt_int" => (a, b) -> a < b,
    "le_int" => (a, b) -> a <= b,
    "gt_int" => (a, b) -> a > b,
    "ge_int" => (a, b) -> a >= b,
    "eq_real" => (a, b) -> a == b,
    "ne_real" => (a, b) -> a != b,
    "lt_real" => (a, b) -> a < b,
    "le_real" => (a, b) -> a <= b,
    "gt_real" => (a, b) -> a > b,
    "ge_real" => (a, b) -> a >= b,
    "eq_bool" => (a, b) -> a == b,
    "ne_bool" => (a, b) -> a != b,
    "eq_string" => (a, b) -> a == b,
    "ne_string" => (a, b) -> a != b,
    "and" => (a, b) -> a && b,
    "or" => (a, b) -> a || b,
    "not" => a -> !a,
    "concat" => (a, b) -> a * b,
    "int_to_real" => a -> Float64(a),
    "int_to_string" => a -> string(a),
    "real_to_int" => a -> Int(trunc(a)),
    "bool_to_real" => a -> a ? 1.0 : 0.0,
    "bool_to_int" => a -> a ? 1 : 0,
    "select_real" => (c, a, b) -> ifelse(c, a, b),
    "select_int" => (c, a, b) -> ifelse(c, a, b),
    "expm1" => expm1,
    "log1p" => log1p,
    "log2" => log2,
    "log10" => log10,
    "rsqrt" => a -> 1.0 / sqrt(a),
    "square" => a -> a * a,
    "sign" => a -> Float64(sign(a)),
    "reciprocal" => a -> 1.0 / a,
    "sin" => sin,
    "cos" => cos,
    "tan" => tan,
    "asin" => asin,
    "acos" => acos,
    "atan" => atan,
    "sinh" => sinh,
    "cosh" => cosh,
    "tanh" => tanh,
    "asinh" => asinh,
    "acosh" => acosh,
    "atanh" => atanh,
    "floor" => a -> floor(Float64(a)),
    "ceil" => a -> ceil(Float64(a)),
    "round" => a -> round(Float64(a)),
    "trunc" => a -> trunc(Float64(a)),
    "erf" => a -> _qvr_qiec_erf(Float64(a)),
    "erfc" => a -> 1.0 - _qvr_qiec_erf(Float64(a)),
    "erfinv" => a -> _qvr_qiec_erfinv(Float64(a)),
    "lgamma" => a -> _qvr_qiec_lgamma(Float64(a)),
    "digamma" => a -> _qvr_qiec_digamma(Float64(a)),
    "sigmoid" => a -> _qvr_qiec_sigmoid(Float64(a)),
    "relu" => a -> max(a, 0.0),
    "relu6" => a -> min(max(a, 0.0), 6.0),
    "elu" => a -> a > 0.0 ? a : expm1(a),
    "selu" => a -> 1.0507009873554805 * (a > 0.0 ? a : 1.6732632423543772 * (exp(a) - 1.0)),
    "gelu" => a -> 0.5 * a * (1.0 + _qvr_qiec_erf(a / sqrt(2.0))),
    "silu" => a -> a * _qvr_qiec_sigmoid(Float64(a)),
    "mish" => a -> a * tanh(_qvr_qiec_softplus(Float64(a))),
    "softplus" => a -> _qvr_qiec_softplus(Float64(a)),
    "logsigmoid" => a -> -_qvr_qiec_softplus(-Float64(a)),
    "softsign" => a -> a / (1.0 + abs(a)),
    "as_weight" => a -> Float64(a),
    "weight_value" => a -> Float64(a),
    "add_weight" => (a, b) -> a + b,
    "scale_weight" => (a, b) -> a * b,
)
_qvr_qiec_sigmoid(value) = value >= 0.0 ? 1.0 / (1.0 + exp(-value)) : exp(value) / (1.0 + exp(value))
_qvr_qiec_softplus(value) = max(value, 0.0) + log1p(exp(-abs(value)))
# The error function by its Taylor series near the origin and by the
# continued fraction of the complementary function in the tails, which
# together reach double precision.
function _qvr_qiec_erf(value)
    sign_ = value < 0.0 ? -1.0 : 1.0
    x = abs(value)
    if x < 2.5
        term = x
        total = x
        n = 0
        while abs(term) > 1e-17 * abs(total) && n < 200
            n += 1
            term *= -x * x / n
            total += term / (2n + 1)
        end
        return sign_ * 2.0 / sqrt(pi) * total
    end
    fraction = x
    for k in 60:-1:1
        fraction = x + (k / 2.0) / fraction
    end
    return sign_ * (1.0 - exp(-x * x) / sqrt(pi) / fraction)
end
function _qvr_qiec_erfinv(value)
    (value < -1.0 || value > 1.0) && error("erfinv is defined on [-1, 1]")
    value == 1.0 && return Inf
    value == -1.0 && return -Inf
    value == 0.0 && return 0.0
    sign_ = value > 0.0 ? 1.0 : -1.0
    magnitude = abs(value)
    if magnitude < 0.7
        square = magnitude * magnitude
        estimate = magnitude * (((-0.140543331 * square + 0.914624893) * square - 1.645349621) * square + 0.886226899) /
            ((((0.012229801 * square - 0.329097515) * square + 1.442710462) * square - 2.118377725) * square + 1.0)
    else
        tail = sqrt(-log((1.0 - magnitude) / 2.0))
        estimate = (((1.641345311 * tail + 3.429567803) * tail - 1.62490649) * tail - 1.970840454) / ((1.637067800 * tail + 3.543889200) * tail + 1.0)
    end
    for _ in 1:3
        estimate -= (_qvr_qiec_erf(estimate) - magnitude) / (2.0 / sqrt(pi) * exp(-estimate * estimate))
    end
    return sign_ * estimate
end
# Lanczos approximation of log gamma.
function _qvr_qiec_lgamma(value)
    if value < 0.5
        return log(pi / abs(sin(pi * value))) - _qvr_qiec_lgamma(1.0 - value)
    end
    coefficients = (676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    x = value - 1.0
    total = 0.99999999999980993
    for (index, coefficient) in enumerate(coefficients)
        total += coefficient / (x + index)
    end
    t = x + 7.5
    return 0.5 * log(2.0 * pi) + (x + 0.5) * log(t) - t + log(total)
end
function _qvr_qiec_digamma(value)
    (value <= 0.0 && value == floor(value)) && error("digamma has a pole at nonpositive integers")
    value < 0.0 && return _qvr_qiec_digamma(1.0 - value) - pi / tan(pi * value)
    result = 0.0
    while value < 6.0
        result -= 1.0 / value
        value += 1.0
    end
    inverse = 1.0 / value
    square = inverse * inverse
    return result + log(value) - 0.5 * inverse - square * (1.0 / 12.0 - square * (1.0 / 120.0 - square * (1.0 / 252.0 - square * (1.0 / 240.0 - square / 132.0))))
end
function _qvr_qiec_broadcast(implementation, arguments)
    tensors = [argument for argument in arguments if argument isa Tuple]
    isempty(tensors) && return implementation(arguments...)
    length_ = length(tensors[1])
    any(length(tensor) != length_ for tensor in tensors) && error("QIEC primitive applied to tensors of differing shapes")
    return Tuple(_qvr_qiec_broadcast(implementation, Any[argument isa Tuple ? argument[index] : argument for argument in arguments]) for index in 1:length_)
end
function _qvr_qiec_primitive(name, arguments)
    haskey(_qvr_qiec_primitives, name) || error("unknown QIEC primitive " * name)
    return _qvr_qiec_broadcast(_qvr_qiec_primitives[name], Any[_qvr_qiec_value(argument) for argument in arguments])
end
function _qvr_qiec_gather(value, index)
    value = _qvr_qiec_value(value)
    index = _qvr_qiec_value(index)
    value isa Tuple || error("QIEC gather from a non-tensor runtime value")
    index isa Tuple && return Tuple(_qvr_qiec_gather(value, item) for item in index)
    return value[index + 1]
end
_qvr_qiec_flat(value) = value isa Tuple ? [entry for item in value for entry in _qvr_qiec_flat(item)] : Any[value]
function _qvr_qiec_reduce(operator, value)
    entries = _qvr_qiec_flat(_qvr_qiec_value(value))
    operator == "sum" && return sum(entries)
    operator == "mean" && return sum(entries) / length(entries)
    operator == "max" && return maximum(entries)
    operator == "min" && return minimum(entries)
    operator == "prod" && return prod(entries)
    peak = maximum(entries)
    return peak + log(sum(exp(entry - peak) for entry in entries))
end
function _qvr_qiec_rowwise(operator, value)
    value = _qvr_qiec_value(value)
    (!isempty(value) && value[1] isa Tuple) && return Tuple(_qvr_qiec_rowwise(operator, item) for item in value)
    row = [Float64(item) for item in value]
    if operator == "softmax"
        weights = exp.(row .- maximum(row))
        return Tuple(weights ./ sum(weights))
    elseif operator == "log_softmax"
        peak = maximum(row)
        return Tuple(row .- (peak + log(sum(exp.(row .- peak)))))
    elseif operator == "cumsum"
        return Tuple(cumsum(row))
    elseif operator == "sort"
        return Tuple(sort(row))
    end
    return Tuple(row ./ sum(row))
end
_qvr_qiec_weight_sum(value) = sum(_qvr_qiec_flat(_qvr_qiec_value(value)))
function _qvr_qiec_segment_sum(value, index, groups)
    totals = zeros(Float64, groups)
    for (weight, group) in zip(_qvr_qiec_value(value), _qvr_qiec_value(index))
        totals[group + 1] += weight
    end
    return Tuple(totals)
end
function _qvr_qiec_project(value, position)
    value = _qvr_qiec_value(value)
    (value isa Tuple && position < length(value)) || error("QIEC projection from a non-product runtime value")
    return value[position + 1]
end
const _qvr_qiec_authored = Dict{String, Any}()
function _qvr_qiec_static_kind(argument)
    kind = argument isa AbstractDict ? get(argument, "kind", nothing) : nothing
    kind in ("type-variable", "type-application", "function-type", "equality-type") && return "type"
    kind in ("effect-variable", "effect-ref") && return "effect"
    return "index"
end
function _qvr_qiec_static_environment(telescope, arguments)
    arguments === nothing && (arguments = Any[])
    if arguments isa AbstractDict
        Set(keys(arguments)) == Set(binder["name"] for binder in telescope) || error("QIEC static arguments do not match the checked telescope")
        ordered = Any[arguments[binder["name"]] for binder in telescope]
    else
        ordered = collect(arguments)
    end
    length(ordered) == length(telescope) || error("QIEC static argument arity does not match the checked telescope")
    environment = Dict()
    for (binder, argument) in zip(telescope, ordered)
        _qvr_qiec_static_kind(argument) == binder["kind"] || error("QIEC static argument kind does not match the checked telescope")
        environment[binder["name"]] = argument
    end
    return environment
end
function _qvr_qiec_specialize(value, environment)
    value isa AbstractVector && return Any[_qvr_qiec_specialize(item, environment) for item in value]
    value isa Tuple && return tuple((_qvr_qiec_specialize(item, environment) for item in value)...)
    value isa AbstractDict || return value
    if get(value, "kind", nothing) in ("type-variable", "index-variable", "effect-variable")
        identity = get(value, "identity", nothing)
        static_key = identity === nothing ? get(value, "name", nothing) : identity
        haskey(environment, static_key) && return environment[static_key]
    end
    return Dict(key => _qvr_qiec_specialize(item, environment) for (key, item) in value)
end
const _qvr_qiec_ambient_key = :qvr_qiec_ambient_handlers
_qvr_qiec_ambient_handlers() = get(task_local_storage(), _qvr_qiec_ambient_key, Any[])
function _qvr_qiec_unique(items)
    result = Any[]
    for item in items
        any(candidate -> candidate === item, result) || push!(result, item)
    end
    return result
end
function _qvr_qiec_with_ambient(controller, thunk)
    storage = task_local_storage()
    had_previous = haskey(storage, _qvr_qiec_ambient_key)
    previous = get(storage, _qvr_qiec_ambient_key, Any[])
    storage[_qvr_qiec_ambient_key] = Any[previous..., controller]
    try
        return thunk()
    finally
        had_previous ? (storage[_qvr_qiec_ambient_key] = previous) : delete!(storage, _qvr_qiec_ambient_key)
    end
end
function _qvr_qiec_effect(request, static_environment)
    copy_request = _qvr_qiec_specialize(request, static_environment)
    get!(copy_request, "captures", Any[])
    controllers = _qvr_qiec_unique(Any[get(copy_request, "handler_captures", Any[])..., _qvr_qiec_ambient_handlers()...])
    copy_request["handler_captures"] = controllers
    copy_request["handler_states"] = Any[controller["capture"]() for controller in controllers]
    return _QvrQiecEffect(copy_request, _qvr_qiec_pure)
end
function _qvr_qiec_bind(comp, continuation, captures=Any[])
    comp isa _QvrQiecPure && return continuation(comp.value)
    (comp isa _QvrQiecCall || comp isa _QvrQiecBind) && return _QvrQiecBind(comp, continuation, Any[captures...])
    request = copy(comp.request)
    request["captures"] = Any[get(request, "captures", Any[])..., captures...]
    return _QvrQiecEffect(request, value -> _qvr_qiec_bind(comp.continuation(value), continuation, captures))
end
_qvr_qiec_is_binding(value) = value isa AbstractDict && get(value, "qiec", nothing) == "binding" && haskey(value, "value")
_qvr_qiec_value(value) = _qvr_qiec_is_binding(value) ? value["value"] : value
_qvr_qiec_constructor(constructor, static_arguments, fields, result_type, static_environment) = Dict("qiec" => "constructor", "constructor" => constructor, "static_arguments" => _qvr_qiec_specialize(static_arguments, static_environment), "fields" => fields, "result_type" => _qvr_qiec_specialize(result_type, static_environment))
_qvr_qiec_evidence(evidence, static_environment) = Dict("qiec" => "evidence", "evidence" => _qvr_qiec_specialize(evidence, static_environment))
function _qvr_qiec_transport(evidence, value, target_type, static_environment)
    _qvr_qiec_specialize(evidence, static_environment)
    _qvr_qiec_specialize(target_type, static_environment)
    return value
end
function _qvr_qiec_attachment(attachments, attachment, expected_type, static_environment)
    haskey(attachments, attachment) || error("missing QIEC attachment " * attachment)
    binding = attachments[attachment]
    _qvr_qiec_is_binding(binding) && haskey(binding, "type") || error("QIEC attachment must be a typed binding descriptor")
    expected_type = _qvr_qiec_specialize(expected_type, static_environment)
    binding["type"] == expected_type || error("QIEC attachment type disagrees with checked IR")
    return _qvr_qiec_value(_qvr_qiec_validate(get(binding, "validator", nothing), binding, "attachment " * attachment))
end
function _qvr_qiec_case(value, branches, metadata, static_environment)
    metadata = _qvr_qiec_specialize(metadata, static_environment)
    value isa AbstractDict && get(value, "qiec", nothing) == "constructor" || error("QIEC case scrutinee is not a constructor value")
    constructor = value["constructor"]
    haskey(metadata["branches"], constructor) || error("constructor is not admitted by the checked QIEC case")
    haskey(branches, constructor) || error("no QIEC case branch for " * constructor)
    branch = metadata["branches"][constructor]
    canonical = get(branch, "static_arguments", Any[])
    actual = get(value, "static_arguments", Any[])
    length(actual) >= length(canonical) || error("QIEC constructor static arguments do not match case branch")
    branch_environment = copy(static_environment)
    for (index, variable) in enumerate(canonical)
        identity = get(variable, "identity", nothing)
        static_key = identity === nothing ? get(variable, "name", nothing) : identity
        static_key === nothing && error("QIEC case branch has an unbindable static variable")
        branch_environment[static_key] = actual[length(actual) - length(canonical) + index]
    end
    return branches[constructor](branch_environment, value["fields"]...)
end
_qvr_qiec_as_computation(value) = value isa _QvrQiecComputation ? value : _qvr_qiec_pure(value)
function _qvr_qiec_validate(validator, value, label)
    validator === nothing && return value
    validator(_qvr_qiec_value(value)) === false && error("QIEC runtime value failed " * label)
    return value
end
_qvr_qiec_validate_computation(comp, validator, label) = _qvr_qiec_bind(_qvr_qiec_as_computation(comp), value -> _qvr_qiec_pure(_qvr_qiec_validate(validator, value, label)))
function _qvr_qiec_capture(locals, attachment_ids, attachments)
    captured = Any[Dict("kind" => "local", "label" => name, "binding" => binding) for (name, binding) in locals]
    for attachment in attachment_ids
        haskey(attachments, attachment) || error("missing QIEC attachment " * attachment)
        push!(captured, Dict("kind" => "attachment", "label" => attachment, "binding" => attachments[attachment]))
    end
    return captured
end
function _qvr_qiec_duplicable(value)
    _qvr_qiec_is_binding(value) && return get(value, "duplicable", false)
    (value === nothing || value isa Bool || value isa Number || value isa AbstractString || value isa Tuple) && return !(value isa Tuple) || all(_qvr_qiec_duplicable, value)
    value isa AbstractDict && get(value, "qiec", nothing) == "constructor" && return all(_qvr_qiec_duplicable, value["fields"])
    return value isa AbstractDict && get(value, "qiec", nothing) == "evidence"
end
function _qvr_qiec_validate_capture(request, current_handler)
    for capture in get(request, "captures", Any[])
        binding = capture["binding"]
        _qvr_qiec_duplicable(binding) || error("QIEC unrestricted resumption captures nonduplicable " * capture["kind"] * " " * capture["label"])
        if _qvr_qiec_is_binding(binding)
            capture["seed"] = binding["value"]
            get(binding, "mutable", false) && !(get(binding, "fork", nothing) isa Function) && error("QIEC mutable duplicable binding requires fork " * capture["label"])
        end
    end
    for controller in _qvr_qiec_unique(Any[get(request, "handler_captures", Any[])..., current_handler])
        controller["snapshot"]()
        handler = controller["get"]()
        get(handler, "duplicable_context", false) || error("QIEC unrestricted resumption captures nonduplicable handler context")
        get(handler, "mutable_context", false) && !(get(handler, "fork_context", nothing) isa Function) && error("QIEC mutable duplicable handler context requires fork_context")
    end
end
function _qvr_qiec_fork_capture(request, current_handler, shot)
    for capture in get(request, "captures", Any[])
        binding = capture["binding"]
        if _qvr_qiec_is_binding(binding) && get(binding, "fork", nothing) isa Function
            binding["value"] = binding["fork"](capture["seed"])
        end
    end
    for controller in _qvr_qiec_unique(Any[get(request, "handler_captures", Any[])..., current_handler])
        controller["install_fork"]()
    end
end
function _qvr_qiec_readdress(comp, shot)
    comp = _qvr_qiec_force(comp)
    comp isa _QvrQiecPure && return comp
    request = copy(comp.request)
    request["address"] = Any[request["address"][1], request["address"][2], Any[request["address"][3]..., shot]]
    return _QvrQiecEffect(request, value -> _qvr_qiec_readdress(comp.continuation(value), shot))
end
function _qvr_qiec_validate_handler(handler, manifest)
    haskey(handler, "definition") && handler["definition"] != manifest && error("QIEC runtime handler definition disagrees with checked IR")
    structural = Set(clause["operation"] for clause in manifest["clauses"])
    Set(keys(get(handler, "operations", Dict()))) == structural || error("QIEC runtime handler clauses disagree with checked IR")
    get(handler, "fork_context", nothing) isa Function && !get(handler, "duplicable_context", false) && error("QIEC handler fork_context requires duplicable_context")
    get(handler, "mutable_context", false) && get(handler, "duplicable_context", false) && !(get(handler, "fork_context", nothing) isa Function) && error("QIEC mutable duplicable handler requires fork_context")
end
_qvr_qiec_lifecycle(handler) = Dict("handler" => handler, "finalized" => false)
function _qvr_qiec_lifecycle_exit(lifecycle)
    lifecycle["finalized"] && return
    lifecycle["finalized"] = true
    callback = get(lifecycle["handler"], "on_exit", nothing)
    callback isa Function && callback()
end
function _qvr_qiec_lifecycle_drop(lifecycle)
    lifecycle["finalized"] && return
    lifecycle["finalized"] = true
    callback = get(lifecycle["handler"], "on_drop", nothing)
    callback isa Function && callback()
end
function _qvr_qiec_lifecycle_enter(handler)
    lifecycle = _qvr_qiec_lifecycle(handler)
    try
        callback = get(handler, "on_enter", nothing)
        callback isa Function && callback()
    catch
        _qvr_qiec_lifecycle_drop(lifecycle)
        rethrow()
    end
    return lifecycle
end
function _qvr_qiec_drop_request(request)
    foreach(_qvr_qiec_lifecycle_drop, get(request, "lifecycles", Any[]))
end
function _qvr_qiec_finalize(comp, lifecycle)
    current = _qvr_qiec_force(_qvr_qiec_as_computation(comp))
    if current isa _QvrQiecPure
        _qvr_qiec_lifecycle_exit(lifecycle)
        return current
    end
    request = copy(current.request)
    request["lifecycles"] = Any[get(request, "lifecycles", Any[])..., lifecycle]
    return _QvrQiecEffect(request, value -> _qvr_qiec_finalize(current.continuation(value), lifecycle))
end
_qvr_qiec_invoke(entry, arguments...) = (entry isa AbstractDict ? entry["invoke"] : entry)(arguments...)
function _qvr_qiec_handle(comp, instance, manifest, static_arguments, computation_static_environment, handlers, attachments=Dict(), operations=Dict())
    static_arguments = _qvr_qiec_specialize(static_arguments, computation_static_environment)
    handler_static_environment = _qvr_qiec_static_environment(manifest["telescope"], static_arguments)
    manifest = _qvr_qiec_specialize(manifest, handler_static_environment)
    manifest["telescope"] = Any[]
    handler_id = manifest["id"]
    prototype = if haskey(handlers, handler_id)
        handlers[handler_id]
    elseif haskey(_qvr_qiec_authored, handler_id)
        _qvr_qiec_authored[handler_id]
    else
        error("missing QIEC handler attachment " * handler_id)
    end
    handler = prototype
    get(prototype, "context_factory", nothing) isa Function && (handler = prototype["context_factory"]())
    handler === prototype && get(prototype, "mutable_context", false) && error("QIEC mutable handler requires context_factory")
    handler_ref = Any[handler]
    _qvr_qiec_validate_handler(handler_ref[1], manifest)
    lifecycle_ref = Any[_qvr_qiec_lifecycle_enter(handler)]
    root_lifecycle = lifecycle_ref[1]
    seed_ref = Any[handler]
    controller = Dict(
        "get" => (() -> handler_ref[1]),
        "seed" => (() -> seed_ref[1]),
        "snapshot" => (() -> (seed_ref[1] = handler_ref[1])),
        "capture" => (() -> Any[handler_ref[1], lifecycle_ref[1]]),
        "restore" => (state -> begin
            handler_ref[1] = state[1]
            lifecycle_ref[1] = state[2]
        end),
        "install_fork" => (() -> begin
            seed = seed_ref[1]
            fork = get(seed, "fork_context", nothing)
            fork isa Function || return
            clone = fork(seed)
            try
                _qvr_qiec_validate_handler(clone, manifest)
            catch
                callback = get(clone, "on_drop", nothing)
                callback isa Function && callback()
                rethrow()
            end
            handler_ref[1] = clone
            lifecycle_ref[1] = _qvr_qiec_lifecycle_enter(clone)
        end),
    )
    grades = Dict(clause["operation"] => clause["grade"] for clause in manifest["clauses"])
    context = Dict("handler" => handler_id, "definition" => manifest, "static_arguments" => static_arguments, "static" => handler_static_environment, "attachments" => attachments, "handlers" => handlers, "operations" => operations, "resumption_uses" => 0)
    function walk(current)
        handler = handler_ref[1]
        current = _qvr_qiec_force(current)
        if current isa _QvrQiecPure
            value = _qvr_qiec_validate(get(handler, "input_validator", nothing), current.value, "handler input type")
            return_clause = get(handler, "return", nothing)
            result = return_clause === nothing ? value : _qvr_qiec_invoke(return_clause, value, context)
            return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, get(handler, "output_validator", nothing), "handler output type"), lifecycle_ref[1])
        end
        request = current.request
        for (captured_controller, state) in zip(get(request, "handler_captures", Any[]), get(request, "handler_states", Any[]))
            if captured_controller === controller
                controller["restore"](state)
                handler = handler_ref[1]
            end
        end
        function forward()
            forwarded_handler = handler_ref[1]
            forwarded_lifecycle = lifecycle_ref[1]
            forwarded = copy(request)
            forwarded["handler_captures"] = _qvr_qiec_unique(Any[get(request, "handler_captures", Any[])..., controller])
            forwarded["handler_states"] = Any[captured["capture"]() for captured in forwarded["handler_captures"]]
            forwarded["lifecycles"] = Any[get(request, "lifecycles", Any[])..., lifecycle_ref[1]]
            return _QvrQiecEffect(forwarded, value -> begin
                handler_ref[1] = forwarded_handler
                lifecycle_ref[1] = forwarded_lifecycle
                _qvr_qiec_with_ambient(controller, () -> walk(current.continuation(value)))
            end)
        end
        request["instance"] != instance && return forward()
        clauses = get(handler, "operations", Dict())
        operation = request["operation"]
        request["effect"] == manifest["effect"] || error("QIEC request and installed handler effect disagree")
        if !haskey(clauses, operation)
            (!manifest["total"] || manifest["forwards_unknown"]) && return forward()
            error("missing QIEC handler clause " * operation)
        end
        uses = Ref(0)
        grade = grades[operation]
        grade == "omega" && _qvr_qiec_validate_capture(request, controller)
        function resume(value)
            shot = uses[]
            uses[] += 1
            context["resumption_uses"] = uses[]
            grade == "0" && error("QIEC zero-grade clause resumed")
            grade in ("aff", "1") && uses[] > 1 && error("QIEC resumption exceeds grade " * grade)
            grade == "omega" && _qvr_qiec_fork_capture(request, controller, shot)
            validator = clause isa AbstractDict ? get(clause, "result_validator", nothing) : nothing
            value = _qvr_qiec_validate(validator, value, "handler operation result type")
            resumed = _qvr_qiec_with_ambient(controller, () -> walk(_qvr_qiec_readdress(current.continuation(value), shot)))
            return resumed isa _QvrQiecPure ? resumed.value : resumed
        end
        clause = clauses[operation]
        clause_lifecycle = lifecycle_ref[1]
        result = try
            _qvr_qiec_invoke(clause, request, resume, context)
        catch
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            rethrow()
        end
        if grade == "1" && uses[] != 1
            _qvr_qiec_drop_request(request)
            _qvr_qiec_lifecycle_drop(clause_lifecycle)
            error("QIEC linear clause must resume exactly once")
        end
        uses[] == 0 && _qvr_qiec_drop_request(request)
        return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, get(handler, "output_validator", nothing), "handler output type"), clause_lifecycle)
    end
    try
        comp = _qvr_qiec_with_ambient(controller, () -> _qvr_qiec_force(comp()))
        return walk(comp)
    catch
        _qvr_qiec_lifecycle_drop(lifecycle_ref[1])
        _qvr_qiec_lifecycle_drop(root_lifecycle)
        rethrow()
    end
end
function _qvr_qiec_run(build, operations)
    _qvr_qiec_serials["call"] = 0
    _qvr_qiec_serials["instance"] = 0
    current = _qvr_qiec_force(build())
    while current isa _QvrQiecEffect
        request = current.request
        key = (request["instance"], request["operation"])
        entry = get(operations, key, nothing)
        entry === nothing && haskey(operations, request["instance"]) && (entry = get(operations[request["instance"]], request["operation"], nothing))
        entry === nothing && error("unhandled QIEC operation " * request["operation"] * " on " * request["instance"])
        try
            result = _qvr_qiec_invoke(entry, request)
            validator = entry isa AbstractDict ? get(entry, "result_validator", nothing) : nothing
            current = _qvr_qiec_force(current.continuation(_qvr_qiec_validate(validator, result, "operation result type")))
        catch
            _qvr_qiec_drop_request(request)
            rethrow()
        end
    end
    return _qvr_qiec_value(current.value)
end
