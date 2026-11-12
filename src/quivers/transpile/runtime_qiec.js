// Target-side QIEC runtime embedded in generated WebPPL programs.
//
// Generated functions carry the checked structural ABI with every request and
// handler delimiter. Process-local callbacks supply code only; they cannot
// override checked coverage, grades, or types.
//
// The runtime is written in WebPPL's functional subset: a name is bound once,
// loops are recursion, and the only mutable state is the global store, which
// holds the call and instance serials, the ambient handler stack, and one
// cell per handler installation. WebPPL trampolines every call, so the
// recursion over a computation's binds and resumptions consumes no host
// stack however deep it goes.
var _qvr_qiec_next = function(counter) {
  var value = (globalStore[counter] || 0) + 1;
  globalStore[counter] = value;
  return value;
};
// A mutable cell in the global store, so handler state survives the
// single-assignment discipline and follows WebPPL's own backtracking.
var _qvr_qiec_cell = function(value) {
  var id = "_qvr_qiec_cell_" + _qvr_qiec_next("_qvr_qiec_cell_serial");
  globalStore[id] = value;
  return id;
};
var _qvr_qiec_cell_get = function(id) { return globalStore[id]; };
var _qvr_qiec_cell_set = function(id, value) {
  globalStore[id] = value;
  return value;
};
var _qvr_qiec_pure = function(value) { return { tag: "pure", value: value }; };
// A deferred computation. Calls are forced by the drivers in _qvr_qiec_run and
// the handler walk rather than when they are built, so the frames a recursive
// QIEC computation runs under are joined as it unfolds. `frames` are the
// dynamic address frames the result runs under, and `tail` records that the
// call retires the frame of the computation that made it.
var _qvr_qiec_call = function(thunk, frames, tail) {
  return { tag: "call", thunk: thunk, frames: frames || [], tail: tail === true };
};
var _qvr_qiec_join_frames = function(outer, inner, tail) {
  if (tail && outer.length > 0 && outer[outer.length - 1][0] === "call") {
    return outer.slice(0, -1).concat(inner);
  }
  return outer.concat(inner);
};
var _qvr_qiec_scoped = function(comp, frames) {
  if (frames.length === 0 || comp.tag === "pure") { return comp; }
  if (comp.tag === "call") {
    return {
      tag: "call",
      thunk: comp.thunk,
      frames: _qvr_qiec_join_frames(frames, comp.frames, comp.tail),
      tail: false
    };
  }
  if (comp.tag === "bind") {
    return {
      tag: "bind",
      inner: _qvr_qiec_scoped(comp.inner, frames),
      continuation: function(value) { return _qvr_qiec_scoped(apply(comp.continuation, [value]), frames); },
      captures: comp.captures
    };
  }
  var address = comp.request.address;
  var request = Object.assign({}, comp.request, {
    address: [address[0], frames.concat(address[1]), address[2]]
  });
  return {
    tag: "effect",
    request: request,
    continuation: function(value) { return _qvr_qiec_scoped(apply(comp.continuation, [value]), frames); }
  };
};
// The driver. Deferred calls are forced and deferred binds are unfolded onto
// an explicit continuation stack. A request surfacing beneath pending binds
// carries them in its continuation, and their captures, so a multi-shot
// resumption still sees everything it copies.
var _qvr_qiec_force = function(comp, pending) {
  var stack = pending || [];
  if (comp.tag === "call") {
    return _qvr_qiec_force(_qvr_qiec_scoped(apply(comp.thunk, []), comp.frames), stack);
  }
  if (comp.tag === "bind") {
    return _qvr_qiec_force(comp.inner, stack.concat([[comp.continuation, comp.captures]]));
  }
  if (comp.tag === "pure") {
    if (stack.length === 0) { return comp; }
    var top = stack[stack.length - 1];
    return _qvr_qiec_force(apply(top[0], [comp.value]), stack.slice(0, -1));
  }
  if (stack.length === 0) { return comp; }
  var captures = reduce(
    function(entry, accumulated) { return accumulated.concat(entry[1]); },
    comp.request.captures || [],
    stack
  );
  var request = Object.assign({}, comp.request, { captures: captures });
  var resumeEffect = comp.continuation;
  return {
    tag: "effect",
    request: request,
    continuation: function(value) { return _qvr_qiec_force(apply(resumeEffect, [value]), stack); }
  };
};
var _qvr_qiec_enter_call = function(name, thunk, tail) {
  var serial = _qvr_qiec_next("_qvr_qiec_call_serial");
  return _qvr_qiec_call(thunk, [["call", name + "#" + serial]], tail);
};
var _qvr_qiec_instance = function(thunk) {
  var serial = _qvr_qiec_next("_qvr_qiec_instance_serial");
  return _qvr_qiec_call(thunk, [["instance", serial]], false);
};
var _qvr_qiec_resume = function(resume, value) { return _qvr_qiec_as_computation(apply(resume, [value])); };
var _qvr_qiec_if = function(condition0, then, otherwise) {
  var condition = _qvr_qiec_value(condition0);
  if (typeof condition !== "boolean") { error("QIEC if condition is not a Boolean"); }
  return condition ? apply(then, []) : apply(otherwise, []);
};
var _qvr_qiec_div_int = function(a, b) {
  if (b === 0) { error("QIEC integer division by zero"); }
  return Math.trunc(a / b);
};
var _qvr_qiec_sigmoid = function(value) {
  if (value >= 0) { return 1 / (1 + Math.exp(-value)); }
  var exponent = Math.exp(value);
  return exponent / (1 + exponent);
};
var _qvr_qiec_softplus = function(value) {
  return Math.max(value, 0) + Math.log1p(Math.exp(-Math.abs(value)));
};
// The error function by its Taylor series near the origin and by the
// continued fraction of the complementary function in the tails.
var _qvr_qiec_erf_series = function(x, term, total, n) {
  if (Math.abs(term) <= 1e-17 * Math.abs(total) || n >= 200) { return total; }
  var next = term * (-x * x / (n + 1));
  return _qvr_qiec_erf_series(x, next, total + next / (2 * (n + 1) + 1), n + 1);
};
var _qvr_qiec_erf_fraction = function(x, k, fraction) {
  if (k < 1) { return fraction; }
  return _qvr_qiec_erf_fraction(x, k - 1, x + (k / 2) / fraction);
};
var _qvr_qiec_erf = function(value) {
  var sign = value < 0 ? -1 : 1;
  var x = Math.abs(value);
  if (x < 2.5) {
    return sign * 2 / Math.sqrt(Math.PI) * _qvr_qiec_erf_series(x, x, x, 0);
  }
  return sign * (1 - Math.exp(-x * x) / Math.sqrt(Math.PI) / _qvr_qiec_erf_fraction(x, 60, x));
};
var _qvr_qiec_erfinv_refine = function(estimate, magnitude, steps) {
  if (steps === 0) { return estimate; }
  var refined = estimate - (_qvr_qiec_erf(estimate) - magnitude) / (2 / Math.sqrt(Math.PI) * Math.exp(-estimate * estimate));
  return _qvr_qiec_erfinv_refine(refined, magnitude, steps - 1);
};
var _qvr_qiec_erfinv = function(value) {
  if (value < -1 || value > 1) { error("erfinv is defined on [-1, 1]"); }
  if (value === 1) { return Infinity; }
  if (value === -1) { return -Infinity; }
  if (value === 0) { return 0; }
  var sign = value > 0 ? 1 : -1;
  var magnitude = Math.abs(value);
  var square = magnitude * magnitude;
  var tail = Math.sqrt(-Math.log((1 - magnitude) / 2));
  var estimate = magnitude < 0.7
    ? magnitude * (((-0.140543331 * square + 0.914624893) * square - 1.645349621) * square + 0.886226899) /
      ((((0.012229801 * square - 0.329097515) * square + 1.442710462) * square - 2.118377725) * square + 1)
    : (((1.641345311 * tail + 3.429567803) * tail - 1.62490649) * tail - 1.970840454) / ((1.637067800 * tail + 3.543889200) * tail + 1);
  return sign * _qvr_qiec_erfinv_refine(estimate, magnitude, 3);
};
// Lanczos approximation of log gamma.
var _qvr_qiec_lgamma = function(value) {
  if (value < 0.5) {
    return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * value))) - _qvr_qiec_lgamma(1 - value);
  }
  var coefficients = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  var x = value - 1;
  var total = 0.99999999999980993 + sum(mapIndexed(function(i, coefficient) { return coefficient / (x + i + 1); }, coefficients));
  var t = x + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(total);
};
var _qvr_qiec_digamma_shift = function(value, result) {
  if (value >= 6) { return [value, result]; }
  return _qvr_qiec_digamma_shift(value + 1, result - 1 / value);
};
var _qvr_qiec_digamma = function(value) {
  if (value <= 0 && value === Math.floor(value)) { error("digamma has a pole at nonpositive integers"); }
  if (value < 0) { return _qvr_qiec_digamma(1 - value) - Math.PI / Math.tan(Math.PI * value); }
  var shifted = _qvr_qiec_digamma_shift(value, 0);
  var inverse = 1 / shifted[0];
  var square = inverse * inverse;
  return shifted[1] + Math.log(shifted[0]) - 0.5 * inverse - square * (1 / 12 - square * (1 / 120 - square * (1 / 252 - square * (1 / 240 - square / 132))));
};
// The closed primitive table. Names and semantics mirror the kernel registry;
// integer division and remainder truncate toward zero on every host.
var _qvr_qiec_primitives = {
  add_int: function(a, b) { return a + b; },
  sub_int: function(a, b) { return a - b; },
  mul_int: function(a, b) { return a * b; },
  div_int: _qvr_qiec_div_int,
  mod_int: function(a, b) { return a - b * _qvr_qiec_div_int(a, b); },
  neg_int: function(a) { return -a; },
  abs_int: function(a) { return Math.abs(a); },
  min_int: function(a, b) { return Math.min(a, b); },
  max_int: function(a, b) { return Math.max(a, b); },
  add_real: function(a, b) { return a + b; },
  sub_real: function(a, b) { return a - b; },
  mul_real: function(a, b) { return a * b; },
  div_real: function(a, b) { return a / b; },
  neg_real: function(a) { return -a; },
  abs_real: function(a) { return Math.abs(a); },
  min_real: function(a, b) { return Math.min(a, b); },
  max_real: function(a, b) { return Math.max(a, b); },
  pow_real: function(a, b) { return Math.pow(a, b); },
  exp: function(a) { return Math.exp(a); },
  log: function(a) { return Math.log(a); },
  sqrt: function(a) { return Math.sqrt(a); },
  eq_int: function(a, b) { return a === b; },
  ne_int: function(a, b) { return a !== b; },
  lt_int: function(a, b) { return a < b; },
  le_int: function(a, b) { return a <= b; },
  gt_int: function(a, b) { return a > b; },
  ge_int: function(a, b) { return a >= b; },
  eq_real: function(a, b) { return a === b; },
  ne_real: function(a, b) { return a !== b; },
  lt_real: function(a, b) { return a < b; },
  le_real: function(a, b) { return a <= b; },
  gt_real: function(a, b) { return a > b; },
  ge_real: function(a, b) { return a >= b; },
  eq_bool: function(a, b) { return a === b; },
  ne_bool: function(a, b) { return a !== b; },
  eq_string: function(a, b) { return a === b; },
  ne_string: function(a, b) { return a !== b; },
  and: function(a, b) { return a && b; },
  or: function(a, b) { return a || b; },
  not: function(a) { return !a; },
  concat: function(a, b) { return a + b; },
  int_to_real: function(a) { return a; },
  int_to_string: function(a) { return a.toString(); },
  real_to_int: function(a) { return Math.trunc(a); },
  bool_to_real: function(a) { return a ? 1.0 : 0.0; },
  bool_to_int: function(a) { return a ? 1 : 0; },
  select_real: function(c, a, b) { return c ? a : b; },
  select_int: function(c, a, b) { return c ? a : b; },
  expm1: function(a) { return Math.expm1(a); },
  log1p: function(a) { return Math.log1p(a); },
  log2: function(a) { return Math.log2(a); },
  log10: function(a) { return Math.log10(a); },
  rsqrt: function(a) { return 1 / Math.sqrt(a); },
  square: function(a) { return a * a; },
  sign: function(a) { return Math.sign(a); },
  reciprocal: function(a) { return 1 / a; },
  sin: function(a) { return Math.sin(a); },
  cos: function(a) { return Math.cos(a); },
  tan: function(a) { return Math.tan(a); },
  asin: function(a) { return Math.asin(a); },
  acos: function(a) { return Math.acos(a); },
  atan: function(a) { return Math.atan(a); },
  sinh: function(a) { return Math.sinh(a); },
  cosh: function(a) { return Math.cosh(a); },
  tanh: function(a) { return Math.tanh(a); },
  asinh: function(a) { return Math.asinh(a); },
  acosh: function(a) { return Math.acosh(a); },
  atanh: function(a) { return Math.atanh(a); },
  floor: function(a) { return Math.floor(a); },
  ceil: function(a) { return Math.ceil(a); },
  // A tie rounds to even, as the QVR reference does; `Math.round`
  // rounds it up.
  round: function(a) {
    var r = Math.floor(a + 0.5);
    return (a + 0.5 === r && r % 2 !== 0) ? r - 1 : r;
  },
  trunc: function(a) { return Math.trunc(a); },
  erf: function(a) { return _qvr_qiec_erf(a); },
  erfc: function(a) { return 1 - _qvr_qiec_erf(a); },
  erfinv: function(a) { return _qvr_qiec_erfinv(a); },
  lgamma: function(a) { return _qvr_qiec_lgamma(a); },
  digamma: function(a) { return _qvr_qiec_digamma(a); },
  sigmoid: function(a) { return _qvr_qiec_sigmoid(a); },
  relu: function(a) { return Math.max(a, 0); },
  relu6: function(a) { return Math.min(Math.max(a, 0), 6); },
  elu: function(a) { return a > 0 ? a : Math.expm1(a); },
  selu: function(a) { return 1.0507009873554805 * (a > 0 ? a : 1.6732632423543772 * (Math.exp(a) - 1)); },
  gelu: function(a) { return 0.5 * a * (1 + _qvr_qiec_erf(a / Math.sqrt(2))); },
  silu: function(a) { return a * _qvr_qiec_sigmoid(a); },
  mish: function(a) { return a * Math.tanh(_qvr_qiec_softplus(a)); },
  softplus: function(a) { return _qvr_qiec_softplus(a); },
  logsigmoid: function(a) { return -_qvr_qiec_softplus(-a); },
  softsign: function(a) { return a / (1 + Math.abs(a)); },
  as_weight: function(a) { return a; },
  weight_value: function(a) { return a; },
  add_weight: function(a, b) { return a + b; },
  scale_weight: function(a, b) { return a * b; }
};
var _qvr_qiec_broadcast = function(implementation, args) {
  var tensors = filter(function(argument) { return Array.isArray(argument); }, args);
  if (tensors.length === 0) { return apply(implementation, args); }
  var length = tensors[0].length;
  if (any(function(tensor) { return tensor.length !== length; }, tensors)) {
    error("QIEC primitive applied to tensors of differing shapes");
  }
  return Object.freeze(mapN(function(index) {
    return _qvr_qiec_broadcast(implementation, map(function(argument) { return Array.isArray(argument) ? argument[index] : argument; }, args));
  }, length));
};
var _qvr_qiec_primitive = function(name, args) {
  if (!_.has(_qvr_qiec_primitives, name)) { error("unknown QIEC primitive " + name); }
  return _qvr_qiec_broadcast(_qvr_qiec_primitives[name], map(_qvr_qiec_value, args));
};
var _qvr_qiec_gather = function(value0, index0) {
  var value = _qvr_qiec_value(value0);
  var index = _qvr_qiec_value(index0);
  if (!Array.isArray(value)) { error("QIEC gather from a non-tensor runtime value"); }
  if (Array.isArray(index)) {
    return Object.freeze(map(function(item) { return _qvr_qiec_gather(value, item); }, index));
  }
  return value[index];
};
var _qvr_qiec_flat = function(value) {
  if (!Array.isArray(value)) { return [value]; }
  return reduce(function(item, accumulated) { return accumulated.concat(_qvr_qiec_flat(item)); }, [], value);
};
var _qvr_qiec_total = function(entries) {
  return reduce(function(a, b) { return a + b; }, 0, entries);
};
var _qvr_qiec_reduce = function(operator, value) {
  var entries = _qvr_qiec_flat(_qvr_qiec_value(value));
  var total = _qvr_qiec_total(entries);
  if (operator === "sum") { return total; }
  if (operator === "mean") { return total / entries.length; }
  if (operator === "max") { return Math.max.apply(null, entries); }
  if (operator === "min") { return Math.min.apply(null, entries); }
  if (operator === "prod") { return reduce(function(a, b) { return a * b; }, 1, entries); }
  var peak = Math.max.apply(null, entries);
  return peak + Math.log(_qvr_qiec_total(map(function(b) { return Math.exp(b - peak); }, entries)));
};
var _qvr_qiec_cumsum = function(row, position, running) {
  if (position === row.length) { return []; }
  var next = running + row[position];
  return [next].concat(_qvr_qiec_cumsum(row, position + 1, next));
};
var _qvr_qiec_rowwise = function(operator, value0) {
  var value = _qvr_qiec_value(value0);
  if (value.length > 0 && Array.isArray(value[0])) {
    return Object.freeze(map(function(item) { return _qvr_qiec_rowwise(operator, item); }, value));
  }
  var row = map(function(item) { return +item; }, value);
  if (operator === "softmax") {
    var peak = Math.max.apply(null, row);
    var weights = map(function(item) { return Math.exp(item - peak); }, row);
    var total = _qvr_qiec_total(weights);
    return Object.freeze(map(function(weight) { return weight / total; }, weights));
  }
  if (operator === "log_softmax") {
    var top = Math.max.apply(null, row);
    var normalizer = top + Math.log(_qvr_qiec_total(map(function(b) { return Math.exp(b - top); }, row)));
    return Object.freeze(map(function(item) { return item - normalizer; }, row));
  }
  if (operator === "cumsum") { return Object.freeze(_qvr_qiec_cumsum(row, 0, 0)); }
  if (operator === "sort") { return Object.freeze(sort(row)); }
  var rowTotal = _qvr_qiec_total(row);
  return Object.freeze(map(function(item) { return item / rowTotal; }, row));
};
var _qvr_qiec_weight_sum = function(value) {
  return _qvr_qiec_total(_qvr_qiec_flat(_qvr_qiec_value(value)));
};
var _qvr_qiec_segment_sum = function(value, index, groups) {
  var weights = _qvr_qiec_value(value);
  var members = _qvr_qiec_value(index);
  return Object.freeze(mapN(function(group) {
    return _qvr_qiec_total(mapIndexed(function(position, weight) { return members[position] === group ? weight : 0; }, weights));
  }, groups));
};
var _qvr_qiec_project = function(value0, position) {
  var value = _qvr_qiec_value(value0);
  if (!Array.isArray(value) || position >= value.length) { error("QIEC projection from a non-product runtime value"); }
  return value[position];
};
var _qvr_qiec_static_kind = function(argument) {
  var kind = argument && argument.kind;
  if (["type-variable", "type-application", "function-type", "equality-type"].indexOf(kind) >= 0) { return "type"; }
  if (["effect-variable", "effect-ref"].indexOf(kind) >= 0) { return "effect"; }
  return "index";
};
var _qvr_qiec_ordered_statics = function(names, args) {
  if (Array.isArray(args)) { return args; }
  if (JSON.stringify(Object.keys(args).sort()) !== JSON.stringify(names.slice().sort())) {
    error("QIEC static arguments do not match the checked telescope");
  }
  return map(function(name) { return args[name]; }, names);
};
var _qvr_qiec_static_environment = function(telescope, args) {
  var names = map(function(binder) { return binder.name; }, telescope);
  var ordered = _qvr_qiec_ordered_statics(names, args || []);
  if (ordered.length !== telescope.length) { error("QIEC static argument arity does not match the checked telescope"); }
  if (any(function(index) { return _qvr_qiec_static_kind(ordered[index]) !== telescope[index].kind; }, _.range(telescope.length))) {
    error("QIEC static argument kind does not match the checked telescope");
  }
  return _.zipObject(names, ordered);
};
// Only plain data is specialized: a distribution or other host object
// passes through with its prototype intact.
var _qvr_qiec_plain = function(value) {
  var prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
};
var _qvr_qiec_specialize = function(value, environment) {
  if (Array.isArray(value)) { return map(function(item) { return _qvr_qiec_specialize(item, environment); }, value); }
  if (!value || typeof value !== "object" || !_qvr_qiec_plain(value)) { return value; }
  if (["type-variable", "index-variable", "effect-variable"].indexOf(value.kind) >= 0) {
    var staticKey = value.identity || value.name;
    if (_.has(environment, staticKey)) { return environment[staticKey]; }
  }
  var keys = Object.keys(value);
  return _.zipObject(keys, map(function(key) { return _qvr_qiec_specialize(value[key], environment); }, keys));
};
var _qvr_qiec_unique = function(items) { return _.uniq(items); };
var _qvr_qiec_ambient = function() { return globalStore._qvr_qiec_ambient || []; };
var _qvr_qiec_with_ambient = function(controller, thunk) {
  var previous = _qvr_qiec_ambient();
  globalStore._qvr_qiec_ambient = previous.concat([controller]);
  var result = apply(thunk, []);
  globalStore._qvr_qiec_ambient = previous;
  return result;
};
var _qvr_qiec_effect = function(request0, staticEnvironment) {
  var request = _qvr_qiec_specialize(request0, staticEnvironment);
  var handlerCaptures = _qvr_qiec_unique((request.handler_captures || []).concat(_qvr_qiec_ambient()));
  var completed = Object.assign({ captures: [] }, request, {
    handler_captures: handlerCaptures,
    handler_states: map(_qvr_qiec_controller_capture, handlerCaptures)
  });
  return { tag: "effect", request: completed, continuation: _qvr_qiec_pure };
};
var _qvr_qiec_bind = function(comp, continuation, captures0) {
  var captures = captures0 || [];
  if (comp.tag === "pure") { return apply(continuation, [comp.value]); }
  if (comp.tag === "call" || comp.tag === "bind") {
    return { tag: "bind", inner: comp, continuation: continuation, captures: captures };
  }
  var request = Object.assign({}, comp.request, { captures: (comp.request.captures || []).concat(captures) });
  return {
    tag: "effect",
    request: request,
    continuation: function(value) { return _qvr_qiec_bind(apply(comp.continuation, [value]), continuation, captures); }
  };
};
// A binding descriptor holds its value directly, or in a global-store cell
// when the host may replace it between shots of a resumption.
var _qvr_qiec_is_binding = function(value) {
  return !!value && value.qiec === "binding" && (_.has(value, "value") || _.has(value, "cell"));
};
var _qvr_qiec_value = function(value) {
  if (!_qvr_qiec_is_binding(value)) { return value; }
  return _.has(value, "cell") ? globalStore[value.cell] : value.value;
};
var _qvr_qiec_constructor = function(constructor, staticArguments, fields, resultType, staticEnvironment) {
  return Object.freeze({
    qiec: "constructor",
    constructor: constructor,
    static_arguments: _qvr_qiec_specialize(staticArguments, staticEnvironment),
    fields: Object.freeze(fields),
    result_type: _qvr_qiec_specialize(resultType, staticEnvironment)
  });
};
var _qvr_qiec_evidence = function(evidence, staticEnvironment) {
  return Object.freeze({ qiec: "evidence", evidence: _qvr_qiec_specialize(evidence, staticEnvironment) });
};
var _qvr_qiec_transport = function(evidence, value, targetType, staticEnvironment) {
  _qvr_qiec_specialize(evidence, staticEnvironment);
  _qvr_qiec_specialize(targetType, staticEnvironment);
  return value;
};
var _qvr_qiec_attachment = function(attachments, attachment, expectedType0, staticEnvironment) {
  if (!_.has(attachments, attachment)) { error("missing QIEC attachment " + attachment); }
  var binding = attachments[attachment];
  if (!_qvr_qiec_is_binding(binding) || !_.has(binding, "type")) { error("QIEC attachment must be a typed binding descriptor"); }
  var expectedType = _qvr_qiec_specialize(expectedType0, staticEnvironment);
  if (JSON.stringify(binding.type) !== JSON.stringify(expectedType)) { error("QIEC attachment type disagrees with checked IR"); }
  return _qvr_qiec_value(_qvr_qiec_validate(binding.validator, binding, "attachment " + attachment));
};
var _qvr_qiec_case = function(value, branches, metadata0, staticEnvironment) {
  var metadata = _qvr_qiec_specialize(metadata0, staticEnvironment);
  if (!value || value.qiec !== "constructor") { error("QIEC case scrutinee is not a constructor value"); }
  if (!_.has(metadata.branches, value.constructor)) { error("constructor is not admitted by the checked QIEC case"); }
  if (!_.has(branches, value.constructor)) { error("no QIEC case branch for " + value.constructor); }
  var branch = metadata.branches[value.constructor];
  var canonical = branch.static_arguments || [];
  var actual = value.static_arguments || [];
  if (actual.length < canonical.length) { error("QIEC constructor static arguments do not match case branch"); }
  var keys = map(function(variable) {
    var staticKey = variable.identity || variable.name;
    if (!staticKey) { error("QIEC case branch has an unbindable static variable"); }
    return staticKey;
  }, canonical);
  var bound = actual.slice(actual.length - canonical.length);
  var branchEnvironment = Object.assign({}, staticEnvironment, _.zipObject(keys, bound));
  return apply(branches[value.constructor], [branchEnvironment].concat(value.fields));
};
var _qvr_qiec_member = function(container, key, fallback) {
  return container && _.has(container, key) ? container[key] : fallback;
};
var _qvr_qiec_as_computation = function(value) {
  return value && (value.tag === "pure" || value.tag === "effect" || value.tag === "call" || value.tag === "bind") ? value : _qvr_qiec_pure(value);
};
var _qvr_qiec_validate = function(validator, value, label) {
  if (!validator) { return value; }
  if (apply(validator, [_qvr_qiec_value(value)]) === false) { error("QIEC runtime value failed " + label); }
  return value;
};
var _qvr_qiec_validate_computation = function(comp, validator, label) {
  return _qvr_qiec_bind(_qvr_qiec_as_computation(comp), function(value) {
    return _qvr_qiec_pure(_qvr_qiec_validate(validator, value, label));
  });
};
var _qvr_qiec_capture = function(locals, attachmentIds, attachments) {
  var captured = map(function(name) { return { kind: "local", label: name, binding: locals[name] }; }, Object.keys(locals));
  var attached = map(function(attachment) {
    if (!_.has(attachments, attachment)) { error("missing QIEC attachment " + attachment); }
    return { kind: "attachment", label: attachment, binding: attachments[attachment] };
  }, attachmentIds);
  return captured.concat(attached);
};
var _qvr_qiec_duplicable = function(value) {
  if (_qvr_qiec_is_binding(value)) { return value.duplicable === true; }
  if (value === null || ["boolean", "number", "string", "undefined"].indexOf(typeof value) >= 0) { return true; }
  if (Array.isArray(value)) { return Object.isFrozen(value) && all(_qvr_qiec_duplicable, value); }
  if (value && value.qiec === "constructor") { return all(_qvr_qiec_duplicable, value.fields); }
  return !!value && value.qiec === "evidence";
};
// Handler installations. Each controller is a global-store key whose cells
// hold the installed handler, its lifecycle, and the seed a fork copies.
var _qvr_qiec_controller_new = function(handler, manifest, lifecycle) {
  var id = "_qvr_qiec_controller_" + _qvr_qiec_next("_qvr_qiec_controller_serial");
  globalStore[id + ":handler"] = handler;
  globalStore[id + ":lifecycle"] = lifecycle;
  globalStore[id + ":seed"] = handler;
  globalStore[id + ":manifest"] = manifest;
  return id;
};
var _qvr_qiec_controller_handler = function(id) { return globalStore[id + ":handler"]; };
var _qvr_qiec_controller_lifecycle = function(id) { return globalStore[id + ":lifecycle"]; };
var _qvr_qiec_controller_snapshot = function(id) {
  globalStore[id + ":seed"] = globalStore[id + ":handler"];
  return null;
};
var _qvr_qiec_controller_capture = function(id) {
  return [globalStore[id + ":handler"], globalStore[id + ":lifecycle"]];
};
var _qvr_qiec_controller_restore = function(id, state) {
  globalStore[id + ":handler"] = state[0];
  globalStore[id + ":lifecycle"] = state[1];
  return null;
};
var _qvr_qiec_controller_install_fork = function(id) {
  var seed = globalStore[id + ":seed"];
  if (typeof seed.fork_context !== "function") { return null; }
  var clone = apply(seed.fork_context, [seed]);
  _qvr_qiec_validate_handler(clone, globalStore[id + ":manifest"]);
  globalStore[id + ":handler"] = clone;
  globalStore[id + ":lifecycle"] = _qvr_qiec_lifecycle_enter(clone);
  return null;
};
var _qvr_qiec_validate_capture = function(request, currentHandler) {
  var captures = map(function(capture) {
    var binding = capture.binding;
    if (!_qvr_qiec_duplicable(binding)) {
      error("QIEC unrestricted resumption captures nonduplicable " + capture.kind + " " + capture.label);
    }
    if (!_qvr_qiec_is_binding(binding)) { return capture; }
    if (binding.mutable && typeof binding.fork !== "function") {
      error("QIEC mutable duplicable binding requires fork " + capture.label);
    }
    return Object.assign({}, capture, { seed: _qvr_qiec_value(binding) });
  }, request.captures || []);
  map(function(controller) {
    _qvr_qiec_controller_snapshot(controller);
    var handler = _qvr_qiec_controller_handler(controller);
    if (!handler.duplicable_context) { error("QIEC unrestricted resumption captures nonduplicable handler context"); }
    if (handler.mutable_context && typeof handler.fork_context !== "function") {
      error("QIEC mutable duplicable handler context requires fork_context");
    }
    return null;
  }, _qvr_qiec_unique((request.handler_captures || []).concat([currentHandler])));
  return Object.assign({}, request, { captures: captures });
};
var _qvr_qiec_fork_capture = function(request, currentHandler, shot) {
  map(function(capture) {
    var binding = capture.binding;
    if (_qvr_qiec_is_binding(binding) && typeof binding.fork === "function") {
      if (!_.has(binding, "cell")) { error("QIEC mutable duplicable binding needs a cell " + capture.label); }
      globalStore[binding.cell] = apply(binding.fork, [capture.seed]);
    }
    return null;
  }, request.captures || []);
  map(_qvr_qiec_controller_install_fork, _qvr_qiec_unique((request.handler_captures || []).concat([currentHandler])));
  return null;
};
var _qvr_qiec_readdress = function(comp0, shot) {
  var comp = _qvr_qiec_force(comp0);
  if (comp.tag === "pure") { return comp; }
  var address = comp.request.address;
  var request = Object.assign({}, comp.request, { address: [address[0], address[1], address[2].concat([shot])] });
  return {
    tag: "effect",
    request: request,
    continuation: function(value) { return _qvr_qiec_readdress(apply(comp.continuation, [value]), shot); }
  };
};
var _qvr_qiec_validate_handler = function(handler, manifest) {
  if (handler.definition && JSON.stringify(handler.definition) !== JSON.stringify(manifest)) {
    error("QIEC runtime handler definition disagrees with checked IR");
  }
  var structural = sort(map(function(clause) { return clause.operation; }, manifest.clauses));
  var executable = sort(Object.keys(handler.operations || {}));
  if (JSON.stringify(structural) !== JSON.stringify(executable)) { error("QIEC runtime handler clauses disagree with checked IR"); }
  if (typeof handler.fork_context === "function" && !handler.duplicable_context) { error("QIEC handler fork_context requires duplicable_context"); }
  if (handler.mutable_context && handler.duplicable_context && typeof handler.fork_context !== "function") {
    error("QIEC mutable duplicable handler requires fork_context");
  }
  return null;
};
// A lifecycle is a cell holding the handler and whether it has finalized.
var _qvr_qiec_lifecycle_exit = function(lifecycle) {
  var state = _qvr_qiec_cell_get(lifecycle);
  if (state.finalized) { return null; }
  _qvr_qiec_cell_set(lifecycle, { handler: state.handler, finalized: true });
  if (typeof state.handler.on_exit === "function") { apply(state.handler.on_exit, []); }
  return null;
};
var _qvr_qiec_lifecycle_drop = function(lifecycle) {
  var state = _qvr_qiec_cell_get(lifecycle);
  if (state.finalized) { return null; }
  _qvr_qiec_cell_set(lifecycle, { handler: state.handler, finalized: true });
  if (typeof state.handler.on_drop === "function") { apply(state.handler.on_drop, []); }
  return null;
};
var _qvr_qiec_lifecycle_enter = function(handler) {
  var lifecycle = _qvr_qiec_cell({ handler: handler, finalized: false });
  if (typeof handler.on_enter === "function") { apply(handler.on_enter, []); }
  return lifecycle;
};
var _qvr_qiec_drop_request = function(request) {
  map(_qvr_qiec_lifecycle_drop, request.lifecycles || []);
  return null;
};
var _qvr_qiec_finalize = function(comp, lifecycle) {
  var current = _qvr_qiec_force(_qvr_qiec_as_computation(comp));
  if (current.tag === "pure") {
    _qvr_qiec_lifecycle_exit(lifecycle);
    return current;
  }
  var request = Object.assign({}, current.request, { lifecycles: (current.request.lifecycles || []).concat([lifecycle]) });
  return {
    tag: "effect",
    request: request,
    continuation: function(value) { return _qvr_qiec_finalize(apply(current.continuation, [value]), lifecycle); }
  };
};
var _qvr_qiec_invoke = function(entry, args) { return apply(entry.invoke || entry, args); };
var _qvr_qiec_handle = function(comp, instance, rawManifest, staticArguments0, computationStaticEnvironment, handlers, attachments0, operations0) {
  var staticArguments = _qvr_qiec_specialize(staticArguments0, computationStaticEnvironment);
  var handlerStaticEnvironment = _qvr_qiec_static_environment(rawManifest.telescope, staticArguments);
  var manifest = Object.assign({}, _qvr_qiec_specialize(rawManifest, handlerStaticEnvironment), { telescope: [] });
  var handlerId = manifest.id;
  if (!_.has(handlers, handlerId) && !_.has(_qvr_qiec_authored, handlerId)) { error("missing QIEC handler attachment " + handlerId); }
  var prototype = _.has(handlers, handlerId) ? handlers[handlerId] : _qvr_qiec_authored[handlerId];
  var handler = typeof prototype.context_factory === "function" ? apply(prototype.context_factory, []) : prototype;
  if (handler === prototype && prototype.mutable_context) { error("QIEC mutable handler requires context_factory"); }
  _qvr_qiec_validate_handler(handler, manifest);
  var controller = _qvr_qiec_controller_new(handler, manifest, _qvr_qiec_lifecycle_enter(handler));
  var grades = _.zipObject(
    map(function(clause) { return clause.operation; }, manifest.clauses),
    map(function(clause) { return clause.grade; }, manifest.clauses)
  );
  var context = {
    handler: handlerId,
    definition: manifest,
    static_arguments: staticArguments,
    "static": handlerStaticEnvironment,
    attachments: attachments0 || {},
    handlers: handlers,
    operations: operations0 || {}
  };
  var walk = function(current0) {
    var current = _qvr_qiec_force(current0);
    if (current.tag === "pure") {
      var returning = _qvr_qiec_controller_handler(controller);
      var value = _qvr_qiec_validate(returning.input_validator, current.value, "handler input type");
      var returned = returning["return"] ? _qvr_qiec_invoke(returning["return"], [value, context]) : value;
      return _qvr_qiec_finalize(_qvr_qiec_validate_computation(returned, returning.output_validator, "handler output type"), _qvr_qiec_controller_lifecycle(controller));
    }
    var request = current.request;
    mapIndexed(function(index, captured) {
      if (captured === controller) { _qvr_qiec_controller_restore(controller, request.handler_states[index]); }
      return null;
    }, request.handler_captures || []);
    var handler = _qvr_qiec_controller_handler(controller);
    var forward = function() {
      var forwardedState = _qvr_qiec_controller_capture(controller);
      var handlerCaptures = _qvr_qiec_unique((request.handler_captures || []).concat([controller]));
      var forwarded = Object.assign({}, request, {
        handler_captures: handlerCaptures,
        handler_states: map(_qvr_qiec_controller_capture, handlerCaptures),
        lifecycles: (request.lifecycles || []).concat([forwardedState[1]])
      });
      return {
        tag: "effect",
        request: forwarded,
        continuation: function(value) {
          _qvr_qiec_controller_restore(controller, forwardedState);
          return _qvr_qiec_with_ambient(controller, function() { return walk(apply(current.continuation, [value])); });
        }
      };
    };
    if (request.instance !== instance) { return forward(); }
    var clause = _qvr_qiec_member(handler.operations || {}, request.operation, null);
    if (JSON.stringify(request.effect) !== JSON.stringify(manifest.effect)) { error("QIEC request and installed handler effect disagree"); }
    if (!clause) {
      if (!manifest.total || manifest.forwards_unknown) { return forward(); }
      error("missing QIEC handler clause " + request.operation);
    }
    var grade = grades[request.operation];
    var seeded = grade === "omega" ? _qvr_qiec_validate_capture(request, controller) : request;
    var uses = _qvr_qiec_cell(0);
    var resume = function(value0) {
      var shot = _qvr_qiec_cell_get(uses);
      _qvr_qiec_cell_set(uses, shot + 1);
      if (grade === "0") { error("QIEC zero-grade clause resumed"); }
      if ((grade === "aff" || grade === "1") && shot > 0) { error("QIEC resumption exceeds grade " + grade); }
      if (grade === "omega") { _qvr_qiec_fork_capture(seeded, controller, shot); }
      var value = _qvr_qiec_validate(clause.result_validator, value0, "handler operation result type");
      var resumed = _qvr_qiec_with_ambient(controller, function() { return walk(_qvr_qiec_readdress(apply(current.continuation, [value]), shot)); });
      return resumed.tag === "pure" ? resumed.value : resumed;
    };
    var clauseLifecycle = _qvr_qiec_controller_lifecycle(controller);
    var result = _qvr_qiec_invoke(clause, [seeded, resume, context]);
    var used = _qvr_qiec_cell_get(uses);
    if (grade === "1" && used !== 1) {
      _qvr_qiec_drop_request(seeded);
      _qvr_qiec_lifecycle_drop(clauseLifecycle);
      error("QIEC linear clause must resume exactly once");
    }
    if (used === 0) { _qvr_qiec_drop_request(seeded); }
    return _qvr_qiec_finalize(_qvr_qiec_validate_computation(result, handler.output_validator, "handler output type"), clauseLifecycle);
  };
  var started = _qvr_qiec_with_ambient(controller, function() { return _qvr_qiec_force(apply(comp, [])); });
  return walk(started);
};
var _qvr_qiec_operation_entry = function(operations, request) {
  var flat = _qvr_qiec_member(operations, request.instance + "|" + request.operation, null);
  if (flat) { return flat; }
  var byInstance = _qvr_qiec_member(operations, request.instance, null);
  var nested = byInstance ? _qvr_qiec_member(byInstance, request.operation, null) : null;
  if (!nested) { error("unhandled QIEC operation " + request.operation + " on " + request.instance); }
  return nested;
};
var _qvr_qiec_drive = function(current, operations) {
  if (current.tag !== "effect") { return _qvr_qiec_value(current.value); }
  var request = current.request;
  var entry = _qvr_qiec_operation_entry(operations, request);
  var result = _qvr_qiec_invoke(entry, [request]);
  var validated = _qvr_qiec_validate(entry.result_validator, result, "operation result type");
  return _qvr_qiec_drive(_qvr_qiec_force(apply(current.continuation, [validated])), operations);
};
var _qvr_qiec_run = function(build, operations) {
  globalStore._qvr_qiec_call_serial = 0;
  globalStore._qvr_qiec_instance_serial = 0;
  return _qvr_qiec_drive(_qvr_qiec_force(apply(build, [])), operations);
};
