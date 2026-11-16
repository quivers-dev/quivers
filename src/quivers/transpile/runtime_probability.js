// Shared checked scalar probability helpers for JavaScript and WebPPL.
var _qvr_log_probability_scalar = function(x) {
  if (typeof x !== "number" || x !== x || x < 0 || x > 1) { error("log_probability requires a finite real in [0,1]"); }
  return x === 0 ? -Infinity : Math.log(x);
};
var _qvr_log1mexp_scalar = function(x) {
  if (typeof x !== "number" || x !== x || x > 0) { error("log1mexp requires a nonpositive real"); }
  return x === 0 ? -Infinity : (x > -Math.LN2 ? Math.log(-Math.expm1(x)) : Math.log1p(-Math.exp(x)));
};
var _qvr_normal_fraction = function(x, k, denominator) {
  return k === 0 ? denominator : _qvr_normal_fraction(x, k - 1, x + k / denominator);
};
var _qvr_log_ndtr_with_erf = function(x, erf) {
  if (typeof x !== "number" || x !== x) { error("log_ndtr requires a real other than NaN"); }
  if (x === Infinity) { return 0; }
  if (x === -Infinity) { return -Infinity; }
  if (x < -2.5) { return -0.5 * x * x - 0.9189385332046727 - Math.log(_qvr_normal_fraction(-x, 96, -x)); }
  if (x > 2.5) { return Math.log1p(-Math.exp(-0.5 * x * x - 0.9189385332046727) / _qvr_normal_fraction(x, 96, x)); }
  return Math.log(0.5 * (1 + erf(x / Math.sqrt(2))));
};


var _qvr_positive_parameter_scalar = function(x) {
  if (typeof x !== "number" || x !== x || x === Infinity || x <= 0) { error("positive_parameter requires a finite strictly positive real"); }
  return x;
};
