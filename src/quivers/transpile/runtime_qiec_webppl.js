// QIEC distribution bridge for generated WebPPL programs. A QIEC tensor is
// a frozen nested array until it reaches a distribution, where it becomes
// a plain nested array; a log density is the distribution's own `score`
// at the converted value. Half-line families fold a symmetric base onto
// the nonnegative reals, doubling its density there. The bridge is
// written in WebPPL's functional subset, like the runtime before it.
var _qvr_qiec_array = function(value0) {
  var value = _qvr_qiec_value(value0);
  if (Array.isArray(value)) { return map(_qvr_qiec_array, value); }
  if (typeof value === "boolean") { return value ? 1 : 0; }
  return value;
};
// A distribution the prelude defines carries `qvr: true` and a `score`
// written in WebPPL, which is reached through a bound name so the call
// stays inside the transform; a built-in distribution's `score` reads
// `this` and keeps the member-call form.
var _qvr_qiec_score = function(distribution, value) {
  if (distribution.qvr === true) {
    var scoreFn = distribution.score;
    return scoreFn(value);
  }
  return distribution.score(value);
};
var _qvr_qiec_log_density = function(distribution, value) {
  return _qvr_qiec_score(distribution, _qvr_qiec_array(value));
};
var _qvr_qiec_log_mass = function(distribution, retained, shape) {
  if (typeof retained !== "number" || retained < 0 || retained > shape.length || Math.floor(retained) !== retained) {
    error("log_mass retained axes must be within the batch rank");
  }
  var normalizer = distribution.log_normalizer;
  var mass = normalizer === undefined ? 0 : normalizer();
  var rank = function(value) { return Array.isArray(value) ? 1 + (value.length === 0 ? 0 : rank(value[0])) : 0; };
  var entry = function(value, axis, index) {
    if (rank(value) !== shape.length - axis) { return value; }
    if (value.length !== 1 && value.length !== shape[axis]) { error("log_mass normalizer shape differs from its declared axes"); }
    return value[value.length === 1 ? 0 : index];
  };
  var reduce = function(axis, value) {
    return axis === shape.length ? value : sum(map(function(i) { return reduce(axis + 1, entry(value, axis, i)); }, _.range(shape[axis])));
  };
  var build = function(axis, value) {
    return axis === retained ? reduce(axis, value) : map(function(i) { return build(axis + 1, entry(value, axis, i)); }, _.range(shape[axis]));
  };
  return build(0, mass);
};
var _qvr_qiec_half = function(base) {
  return {
    qvr: true,
    sample: function() { return Math.abs(sample(base)); },
    score: function(x) { return x < 0 ? -Infinity : Math.log(2) + _qvr_qiec_score(base, x); },
    support: function() { return { lower: 0, upper: Infinity }; }
  };
};
var _qvr_qiec_categorical = function(ps) {
  return Categorical({ ps: ps, vs: _qvr_support(ps) });
};
var _qvr_qiec_mixture_normal = function(weights, loc, scale) {
  var components = mapIndexed(function(i, m) { return Gaussian({ mu: m, sigma: scale[i] }); }, loc);
  return Mixture({ dists: components, ps: weights });
};
// Site labels a helper's draws and scores carry, counted so the n-th
// occurrence of a label in one run of the model is "<label>@<n>", as the
// reference machine replays them. The counts live in a global-store cell.
var _qvr_qiec_site_names = function() {
  var cell = _qvr_qiec_cell({});
  return function(label) {
    var occurrences = _qvr_qiec_cell_get(cell);
    var count = _qvr_qiec_member(occurrences, label, 0);
    _qvr_qiec_cell_set(cell, Object.assign({}, occurrences, _.zipObject([label], [count + 1])));
    return count === 0 ? label : label + "@" + count;
  };
};
// A helper's draw and scored weight under WebPPL's own `sample` and
// `factor`, each carrying its site name so a driver that clamps sites by
// name can replace these two alone.
var _qvr_qiec_draw = function(label, distribution) { return sample(distribution); };
var _qvr_qiec_add = function(label, weight) { factor(weight); return null; };
// The program's canonical instances handled by WebPPL's own primitives,
// so a computation the model calls draws and scores as the model does.
var _qvr_qiec_native_operations = function(randomInstance, sampleOperation, scoreInstance, addOperation) {
  var name = _qvr_qiec_site_names();
  // WebPPL renames the identifier , so the request's field
  // is read under its string key.
  var sampleEntry = function(request) {
    var label = request["arguments"][0];
    var distribution = request["arguments"][1];
    return _qvr_qiec_draw(name(label), distribution);
  };
  var addEntry = function(request) {
    _qvr_qiec_add(name("score"), _qvr_qiec_array(request["arguments"][0]));
    return null;
  };
  return _.zipObject(
    [randomInstance + "|" + sampleOperation, scoreInstance + "|" + addOperation],
    [sampleEntry, addEntry]
  );
};
