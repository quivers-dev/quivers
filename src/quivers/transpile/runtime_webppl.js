// WebPPL runtime helpers: distribution constructors for QVR families
// that WebPPL's built-in `dists` module does not ship.
//
// Each helper is a plain JavaScript function that takes a `params`
// object and returns a distribution object with `sample`, `score`,
// and `support` methods, marked `qvr: true` so a reader can tell it
// from a built-in distribution and score it through the transform.
// WebPPL's `sample(dist)` and `observe(dist, value)` accept any
// object exposing this triple.
//
// WebPPL compiles its source through a CPS transform that rejects
// `while` / `for` loops, variable reassignment, and in-place array
// mutation. Every helper below is therefore written in WebPPL's
// functional subset: single-assignment `var`, recursion, and the
// `map` / `mapN` / `sum` / `reduce` combinators. The densities are
// identical to the standard closed forms.
//
// Mathematical references for each density:
//
//   Logistic              : Balakrishnan 1991
//                           https://doi.org/10.1201/9781482277098
//   BetaBinomial          : Skellam 1948
//                           https://www.jstor.org/stable/2983694
//   HalfStudentT          : Gelman 2006
//                           https://doi.org/10.1214/06-BA117A
//   Kumaraswamy           : Kumaraswamy 1980
//                           https://doi.org/10.1016/0022-1694(80)90036-0
//   LKJCholesky           : Lewandowski, Kurowicka, Joe 2009
//                           https://doi.org/10.1016/j.jmva.2009.04.008
//   ContinuousBernoulli   : Loaiza-Ganem and Cunningham 2019
//                           https://arxiv.org/abs/1907.06845
//   MatrixNormal          : Dawid 1981
//                           https://doi.org/10.1093/biomet/68.1.265

var _lgamma = function(z) {
  // Lanczos approximation. Domain: z > 0.
  var g = 7;
  var c = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - _lgamma(1 - z);
  }
  var zz = z - 1;
  var x = c[0] + sum(mapN(function(k) {
    var i = k + 1;
    return c[i] / (zz + i);
  }, g + 1));
  var t = zz + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (zz + 0.5) * Math.log(t) - t + Math.log(x);
};

var _lbeta = function(a, b) {
  return _lgamma(a) + _lgamma(b) - _lgamma(a + b);
};

var _gaussian_sample = function(mu, sigma) {
  // Box-Muller.
  var u1 = Math.random();
  var u2 = Math.random();
  var z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + sigma * z;
};

var _gamma_sample = function(shape, scale) {
  // Marsaglia and Tsang 2000 for shape >= 1; for shape < 1 boost.
  if (shape < 1) {
    var u = Math.random();
    return _gamma_sample(shape + 1, scale) * Math.pow(u, 1 / shape);
  }
  var d = shape - 1 / 3;
  var c = 1 / Math.sqrt(9 * d);
  var attempt = function() {
    var x = _gaussian_sample(0, 1);
    var v0 = 1 + c * x;
    if (v0 <= 0) { return attempt(); }
    var v = v0 * v0 * v0;
    var u2 = Math.random();
    if (u2 < 1 - 0.0331 * x * x * x * x) { return d * v * scale; }
    if (Math.log(u2) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
      return d * v * scale;
    }
    return attempt();
  };
  return attempt();
};

var _beta_sample = function(a, b) {
  var x = _gamma_sample(a, 1);
  var y = _gamma_sample(b, 1);
  return x / (x + y);
};

var _binomial_sample = function(n, p) {
  return sum(mapN(function(i) {
    return Math.random() < p ? 1 : 0;
  }, n));
};

var _qvr_bcast = function(op, a, b) {
  // Broadcasting binary arithmetic over scalars and (possibly nested)
  // arrays. WebPPL's `+`, `-`, `*`, `/` operators are scalar-only, so a
  // deterministic `let` combining an array-valued prior with another
  // value must route through this helper to stay elementwise instead of
  // coercing arrays to NaN. Scalar-with-array broadcasts the scalar
  // across every element; array-with-array applies elementwise and
  // recurses so rank-2 operands broadcast correctly.
  var a_arr = Array.isArray(a);
  var b_arr = Array.isArray(b);
  if (!a_arr && !b_arr) {
    return op === "+" ? a + b
      : op === "-" ? a - b
      : op === "*" ? a * b
      : op === "/" ? a / b
      : op === "^" ? Math.pow(a, b)
      : op === "max" ? Math.max(a, b)
      : op === "min" ? Math.min(a, b)
      : op === "quot" ? Math.trunc(a / b)
      : op === "rem" ? a % b
      : NaN;
  }
  var n = a_arr ? a.length : b.length;
  return mapN(function(i) {
    var ai = a_arr ? a[i] : a;
    var bi = b_arr ? b[i] : b;
    return _qvr_bcast(op, ai, bi);
  }, n);
};

var _qvr_reduce_last = function(fold, x) {
  // Reduce the innermost axis of a (possibly nested) array, leaving
  // every outer axis intact. WebPPL's own `sum` flattens a nested
  // array to a single scalar, which is a different quantity from the
  // per-row reduction a QVR `let mu = sum(z_row * w_row)` denotes:
  // the binding's own plate carries the outer axis and only the
  // operands' event axis is collapsed. Recursion walks by index
  // through `mapN` rather than passing this function to `map`, which
  // WebPPL's CPS transform mishandles for a self-referential
  // callback.
  if (!Array.isArray(x)) { return x; }
  if (x.length > 0 && Array.isArray(x[0])) {
    return mapN(function(i) {
      return _qvr_reduce_last(fold, x[i]);
    }, x.length);
  }
  return fold(x);
};
var _qvr_sum_last = function(x) {
  return _qvr_reduce_last(function(row) { return sum(row); }, x);
};
var _qvr_mean_last = function(x) {
  return _qvr_reduce_last(function(row) {
    return sum(row) / row.length;
  }, x);
};
var _qvr_prod_last = function(x) {
  return _qvr_reduce_last(function(row) {
    return reduce(function(v, acc) { return acc * v; }, 1, row);
  }, x);
};
var _qvr_max_last = function(x) {
  return _qvr_reduce_last(function(row) {
    return reduce(function(v, acc) { return v > acc ? v : acc; },
                  -Infinity, row);
  }, x);
};
var _qvr_min_last = function(x) {
  return _qvr_reduce_last(function(row) {
    return reduce(function(v, acc) { return v < acc ? v : acc; },
                  Infinity, row);
  }, x);
};
var _qvr_total = function(x) {
  // Sum every leaf of a (possibly nested) array. The marginalize
  // lowering reduces its atoms elementwise over whatever rows the
  // scope's plates carry, then folds the whole thing into a single
  // `factor` increment.
  if (!Array.isArray(x)) { return x; }
  return sum(mapN(function(i) { return _qvr_total(x[i]); }, x.length));
};
var _qvr_take_last = function(x, k) {
  // Slice index `k` off the innermost axis of a (possibly nested)
  // array, leaving every outer axis intact. A `Categorical` atom set
  // reads its log-weights off the class axis, which is the innermost
  // axis of the probability tensor however many grouping axes sit
  // above it.
  if (!Array.isArray(x)) { return x; }
  if (x.length > 0 && Array.isArray(x[0])) {
    return mapN(function(i) { return _qvr_take_last(x[i], k); }, x.length);
  }
  return x[k];
};
var _qvr_gather_log = function(rows, idx) {
  // Gather one row of `rows` per entry of `idx` and take its
  // elementwise logarithm. A grouped `marginalize` reads its per-row
  // prior through a fibration, so the weights are the grouping
  // plate's rows selected once per observation. JavaScript indexes an
  // array with one integer and yields `undefined` for an array of
  // them, so the gather is a map rather than a subscript.
  return map(function(i) { return log(rows[i]); }, idx);
};
var _qvr_group_sums = function(rows, via, extent) {
  // Sum the per-row entries of `rows` within each of the `extent`
  // groups the fibration `via` sends the rows to. A grouped
  // `marginalize` keys its accumulator by group, so one atom's
  // per-row log-likelihoods are pooled per group before the
  // reduction over the atoms runs.
  return mapN(function(g) {
    return sum(mapN(function(i) {
      return via[i] === g ? rows[i] : 0;
    }, rows.length));
  }, extent);
};
var _qvr_concat = function(rows, i) {
  // Concatenate `rows` (an array of arrays) from index `i` onward.
  // WebPPL's `reduce` is a right fold, which would reverse the
  // factor order an affine map's conditioning vector depends on, so
  // the walk is written as an explicit recursion by index.
  if (i >= rows.length) { return []; }
  return rows[i].concat(_qvr_concat(rows, i + 1));
};
var _qvr_affine = function(weight, bias, sources, rowOffset, rows, link) {
  // One head's row block of the affine parameter map `W x + b`, where
  // `x` is the concatenation of `sources` in declaration order.
  // WebPPL has no matrix product and no numeric array type, so the
  // contraction is written as a `mapN` over the codomain axis with an
  // inner `sum` over the domain; `link` is the head's elementwise
  // transform, the identity or `exp`.
  var x = _qvr_concat(sources, 0);
  return mapN(function(i) {
    var row = weight[rowOffset + i];
    var z = sum(mapN(function(j) {
      return row[j] * x[j];
    }, x.length)) + bias[rowOffset + i];
    return link === "exp" ? Math.exp(z) : z;
  }, rows);
};
var _qvr_logsumexp = function(terms) {
  // Elementwise `logsumexp` across a list of same-shaped terms, one
  // per atom of a marginalized latent's finite support. Shifting by
  // the running maximum keeps the exponentials in range; an all
  // `-Infinity` row stays `-Infinity` rather than becoming `NaN`.
  if (terms.length === 0) { return -Infinity; }
  if (Array.isArray(terms[0])) {
    return mapN(function(i) {
      return _qvr_logsumexp(mapN(function(k) {
        return terms[k][i];
      }, terms.length));
    }, terms[0].length);
  }
  var m = reduce(function(v, acc) { return v > acc ? v : acc; },
                 -Infinity, terms);
  if (m === -Infinity) { return -Infinity; }
  return m + Math.log(sum(map(function(t) {
    return Math.exp(t - m);
  }, terms)));
};
var _qvr_poisson_score = function(params, value) {
  // log Poisson pmf, defined at the boundary rate 0. WebPPL's
  // built-in `Poisson` rejects `mu = 0` outright, but a marginalize
  // atom that pins a zero-inflation indicator to 0 gates the rate to
  // exactly that boundary, where the distribution is the point mass
  // at 0 and the QVR reference scores it as such.
  var mu = params.mu;
  if (mu === 0) { return value === 0 ? 0 : -Infinity; }
  return value * Math.log(mu) - mu - _lgamma(value + 1);
};
var _qvr_score = function(dist, value) {
  // Score a value under a distribution declared in this prelude.
  // WebPPL compiles `dist.score(value)` as a direct JavaScript member
  // call, outside its CPS transform, so a `score` body that reaches
  // for a WebPPL combinator (`sum`, `map`, `mapN`) returns a
  // trampoline thunk instead of a number. Calling through a top-level
  // helper keeps the body inside the transform. Built-in WebPPL
  // distributions read `this` in their own `score` and must keep the
  // member-call form instead.
  var scoreFn = dist.score;
  return scoreFn(value);
};
// Logistic sigmoid, mapped over scalars and (possibly nested) arrays.
// The QVR `sigmoid` math primitive has no WebPPL stdlib counterpart;
// the deterministic `let mu = sigmoid(eta)` binding drives an
// elementwise transform when `eta` is an array-valued prior gather.
var sigmoid = function(x) {
  return Array.isArray(x) ? map(sigmoid, x) : 1 / (1 + Math.exp(-x));
};

// Elementwise math primitives, mapped over scalars and (possibly
// nested) arrays. The QVR `exp` / `log` / `sqrt` / `abs` math
// primitives have no WebPPL stdlib globals; a deterministic
// `let scale = exp(eta)` binding against an array-valued gather needs
// the elementwise form.
var exp = function(x) {
  return Array.isArray(x) ? map(exp, x) : Math.exp(x);
};

var log = function(x) {
  return Array.isArray(x) ? map(log, x) : Math.log(x);
};

var sqrt = function(x) {
  return Array.isArray(x) ? map(sqrt, x) : Math.sqrt(x);
};

var abs = function(x) {
  return Array.isArray(x) ? map(abs, x) : Math.abs(x);
};

// QVR `max` / `min` reduce the innermost axis of one argument and are
// pointwise over two, broadcasting a scalar across an array; `pow` is
// pointwise. WebPPL has no globals of these names.
var max = function(a, b) {
  return b === undefined ? _qvr_max_last(a) : _qvr_bcast("max", a, b);
};

var min = function(a, b) {
  return b === undefined ? _qvr_min_last(a) : _qvr_bcast("min", a, b);
};

var pow = function(a, b) {
  return _qvr_bcast("^", a, b);
};

// Integer division and remainder truncate toward zero, pointwise.
var quot = function(a, b) {
  return _qvr_bcast("quot", a, b);
};

var rem = function(a, b) {
  return _qvr_bcast("rem", a, b);
};

var Logistic = function(params) {
  var loc = params.loc;
  var scale = params.scale;
  return {
    qvr: true,
    sample: function() {
      var u = Math.random();
      return loc + scale * Math.log(u / (1 - u));
    },
    score: function(x) {
      var z = (x - loc) / scale;
      return -z - Math.log(scale) - 2 * Math.log(1 + Math.exp(-z));
    },
    support: function() {
      return { lower: -Infinity, upper: Infinity };
    }
  };
};

var BetaBinomial = function(params) {
  var n = params.total_count;
  var a = params.concentration1;
  var b = params.concentration0;
  return {
    qvr: true,
    sample: function() {
      var p = _beta_sample(a, b);
      return _binomial_sample(n, p);
    },
    score: function(k) {
      var log_comb = _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1);
      var log_beta_post = _lbeta(a + k, b + n - k);
      var log_beta_prior = _lbeta(a, b);
      return log_comb + log_beta_post - log_beta_prior;
    },
    support: function() {
      return { lower: 0, upper: n };
    }
  };
};

var HalfStudentT = function(params) {
  var df = params.df;
  var scale = params.scale;
  return {
    qvr: true,
    sample: function() {
      var z = _gaussian_sample(0, 1);
      var u = _gamma_sample(df / 2, 2);
      var t = z / Math.sqrt(u / df);
      return Math.abs(t) * scale;
    },
    score: function(x) {
      if (x < 0) { return -Infinity; }
      var z = x / scale;
      var log_t = _lgamma((df + 1) / 2) - _lgamma(df / 2)
                  - 0.5 * Math.log(df * Math.PI)
                  - ((df + 1) / 2) * Math.log(1 + (z * z) / df);
      return log_t - Math.log(scale) + Math.log(2);
    },
    support: function() {
      return { lower: 0, upper: Infinity };
    }
  };
};

var Kumaraswamy = function(params) {
  var a = params.concentration1;
  var b = params.concentration0;
  return {
    qvr: true,
    sample: function() {
      var u = Math.random();
      return Math.pow(1 - Math.pow(1 - u, 1 / b), 1 / a);
    },
    score: function(x) {
      if (x <= 0 || x >= 1) { return -Infinity; }
      return Math.log(a) + Math.log(b)
             + (a - 1) * Math.log(x)
             + (b - 1) * Math.log(1 - Math.pow(x, a));
    },
    support: function() {
      return { lower: 0, upper: 1 };
    }
  };
};

var LKJCholesky = function(params) {
  var dim = params.dim;
  var eta = params.concentration;
  return {
    qvr: true,
    sample: function() {
      // Onion method: draw beta-distributed partial correlations
      // and stitch them into a Cholesky factor. Returns a square
      // lower-triangular matrix as an array of arrays. Each row k is
      // built from its own random draws and is independent of the
      // other rows, so the rows map independently.
      return mapN(function(k) {
        if (k === 0) {
          return mapN(function(j) { return j === 0 ? 1 : 0; }, dim);
        }
        var alpha = eta + (dim - 1 - k) / 2;
        var r2 = _beta_sample(k / 2, alpha);
        var r = Math.sqrt(r2);
        // Sample a unit vector in R^k via independent normals.
        var u = mapN(function(m) { return _gaussian_sample(0, 1); }, k);
        var sumsq = sum(map(function(g) { return g * g; }, u));
        var norm = Math.sqrt(sumsq);
        return mapN(function(t) {
          return t < k ? r * u[t] / norm
            : (t === k ? Math.sqrt(1 - r2) : 0);
        }, dim);
      }, dim);
    },
    score: function(L) {
      // Unnormalised LKJ density on the Cholesky factor:
      //   log p(L) = sum_{i=2}^{dim} (dim - i + 2 * eta - 2) * log L[i][i]
      // plus the LKJ normalising constant. For the syntactic check
      // we return only the data-dependent part; the constant cancels
      // in MCMC ratios.
      return sum(mapN(function(k) {
        var i = k + 1;
        return (dim - i - 1 + 2 * eta - 2) * Math.log(L[i][i]);
      }, dim - 1));
    },
    support: function() {
      return { lower: -1, upper: 1 };
    }
  };
};

var ContinuousBernoulli = function(params) {
  var p = params.probs;
  var log_norm = (Math.abs(p - 0.5) < 1e-4)
    ? (Math.log(2) + 2 * (p - 0.5) * (p - 0.5)
       + (4 / 3) * Math.pow(p - 0.5, 4))
    : (Math.log(Math.abs(2 * Math.atanh(1 - 2 * p)))
       - Math.log(Math.abs(1 - 2 * p)));
  return {
    qvr: true,
    sample: function() {
      var u = Math.random();
      if (Math.abs(p - 0.5) < 1e-4) { return u; }
      return Math.log1p((2 * p - 1) * u / (1 - p)) / Math.log(p / (1 - p));
    },
    score: function(x) {
      if (x < 0 || x > 1) { return -Infinity; }
      return x * Math.log(p) + (1 - x) * Math.log(1 - p) + log_norm;
    },
    support: function() {
      return { lower: 0, upper: 1 };
    }
  };
};

var _qvr_rbf_kernel = function(x, length_scale, jitter) {
  var n = x.length;
  return mapN(function(i) {
    return mapN(function(j) {
      var d = x[i] - x[j];
      var v = Math.exp(-0.5 * d * d / (length_scale * length_scale));
      return i === j ? v + jitter : v;
    }, n);
  }, n);
};

var _qvr_zeros = function(n) {
  return mapN(function(i) { return 0; }, n);
};

// The support of a Categorical over the classes its probability
// vector indexes. WebPPL's `Categorical` takes the values it ranges
// over beside their probabilities and has no default for them, while
// QVR's `Categorical` ranges over the positions of its own vector.
// Derived from `ps` rather than from a class count carried down to
// the call site, so it is the right length wherever the vector came
// from.
var _qvr_support = function(ps) {
  return mapN(function(i) { return i; }, ps.length);
};


// Dense matrix utilities. Matrices are arrays-of-arrays in row-major
// order (so `A[i][j]` is row i, column j). The helpers cover only what
// MatrixNormal needs: element-wise add / sub, transpose, multiply,
// Cholesky decomposition (lower-triangular), Cholesky-based solve, log
// determinant, and trace.
var _mat_add = function(A, B) {
  return mapN(function(i) {
    return mapN(function(j) { return A[i][j] + B[i][j]; }, A[0].length);
  }, A.length);
};

var _mat_sub = function(A, B) {
  return mapN(function(i) {
    return mapN(function(j) { return A[i][j] - B[i][j]; }, A[0].length);
  }, A.length);
};

var _mat_transpose = function(A) {
  return mapN(function(i) {
    return mapN(function(j) { return A[j][i]; }, A.length);
  }, A[0].length);
};

var _mat_mul = function(A, B) {
  var k = A[0].length;
  return mapN(function(i) {
    return mapN(function(j) {
      return sum(mapN(function(t) { return A[i][t] * B[t][j]; }, k));
    }, B[0].length);
  }, A.length);
};

var _mat_chol = function(A) {
  // Standard Cholesky decomposition for SPD A; returns lower
  // triangular L with L L' = A. Rows are built in order because row i
  // depends on rows 0..i-1; columns within a row are built in order
  // because column j depends on columns 0..j-1 of the same row.
  var n = A.length;
  var buildRow = function(i, Lprev) {
    var buildCol = function(j, rowAcc) {
      if (j > i) {
        return rowAcc.concat(mapN(function(m) { return 0; }, n - j));
      }
      var Lj = j === i ? rowAcc : Lprev[j];
      var s = sum(mapN(function(k) { return rowAcc[k] * Lj[k]; }, j));
      var val = i === j
        ? Math.sqrt(A[i][i] - s)
        : (A[i][j] - s) / Lprev[j][j];
      return buildCol(j + 1, rowAcc.concat([val]));
    };
    return buildCol(0, []);
  };
  var buildAll = function(i, acc) {
    if (i >= n) { return acc; }
    return buildAll(i + 1, acc.concat([buildRow(i, acc)]));
  };
  return buildAll(0, []);
};

var _mat_solve_chol = function(L, B) {
  // Solve A X = B given the Cholesky factor L of A (A = L L').
  // Forward substitution L Y = B, then backward L' X = Y.
  var n = L.length;
  var m = B[0].length;
  var solveForward = function(i, Yacc) {
    if (i >= n) { return Yacc; }
    var row = mapN(function(col) {
      var s = B[i][col]
        - sum(mapN(function(k) { return L[i][k] * Yacc[k][col]; }, i));
      return s / L[i][i];
    }, m);
    return solveForward(i + 1, Yacc.concat([row]));
  };
  var Y = solveForward(0, []);
  var solveBackward = function(ib, Xacc) {
    if (ib < 0) { return Xacc; }
    var row = mapN(function(col) {
      var s = Y[ib][col]
        - sum(mapN(function(kk) {
            var k = ib + 1 + kk;
            return L[k][ib] * Xacc[kk][col];
          }, n - 1 - ib));
      return s / L[ib][ib];
    }, m);
    return solveBackward(ib - 1, [row].concat(Xacc));
  };
  return solveBackward(n - 1, []);
};

var _mat_logdet_chol = function(L) {
  // log|A| = 2 sum_i log L[i][i] for the Cholesky factor of A.
  return 2 * sum(mapN(function(i) { return Math.log(L[i][i]); }, L.length));
};

var _mat_trace = function(A) {
  return sum(mapN(function(i) { return A[i][i]; }, A.length));
};

var MatrixNormal = function(params) {
  // X in R^{p x n} with mean loc, row covariance U (p x p, SPD), and
  // column covariance V (n x n, SPD). Equivalent to
  //   vec(X) ~ MultiNormal(vec(loc), V (x) U).
  // log p(X | loc, U, V) =
  //   -0.5 * ( n p log(2 pi) + n log|U| + p log|V|
  //          + tr(V^{-1} (X - loc)' U^{-1} (X - loc)) )
  var loc = params.loc;
  var U = params.row_covariance;
  var V = params.col_covariance;
  return {
    qvr: true,
    sample: function() {
      // X = loc + L_U Z L_V' with Z iid standard normal, L_U L_U' = U,
      // L_V L_V' = V.
      var p = loc.length;
      var n = loc[0].length;
      var L_U = _mat_chol(U);
      var L_V = _mat_chol(V);
      var Z = mapN(function(i) {
        return mapN(function(j) { return _gaussian_sample(0, 1); }, n);
      }, p);
      return _mat_add(loc, _mat_mul(_mat_mul(L_U, Z), _mat_transpose(L_V)));
    },
    score: function(X) {
      var p = X.length;
      var n = X[0].length;
      var D = _mat_sub(X, loc);
      var L_U = _mat_chol(U);
      var L_V = _mat_chol(V);
      var U_inv_D = _mat_solve_chol(L_U, D);
      var quad = _mat_mul(_mat_transpose(D), U_inv_D);
      var V_inv_quad = _mat_solve_chol(L_V, quad);
      var trace_term = _mat_trace(V_inv_quad);
      var log_det_U = _mat_logdet_chol(L_U);
      var log_det_V = _mat_logdet_chol(L_V);
      return -0.5 * (n * p * Math.log(2 * Math.PI)
                     + n * log_det_U
                     + p * log_det_V
                     + trace_term);
    },
    support: function() {
      return { lower: -Infinity, upper: Infinity };
    }
  };
};

var _poisson_sample = function(lambda) {
  // Knuth's multiplication method. Adequate for the moderate rates
  // that arise in the gamma-Poisson mixture used by NegativeBinomial.
  var L = Math.exp(-lambda);
  var step = function(k, prod) {
    var p = prod * Math.random();
    return p <= L ? k : step(k + 1, p);
  };
  return step(0, 1);
};

var LogNormal = function(params) {
  // torch LogNormal(loc, scale): X = exp(loc + scale * Z), Z ~ N(0, 1).
  var loc = params.loc;
  var scale = params.scale;
  return {
    qvr: true,
    sample: function() {
      return Math.exp(_gaussian_sample(loc, scale));
    },
    score: function(x) {
      if (x <= 0) { return -Infinity; }
      var lx = Math.log(x);
      var z = (lx - loc) / scale;
      return -lx - Math.log(scale) - 0.5 * Math.log(2 * Math.PI)
             - 0.5 * z * z;
    },
    support: function() {
      return { lower: 0, upper: Infinity };
    }
  };
};

var StudentT = function(params) {
  // torch StudentT(df, loc, scale): location-scale Student-t.
  var df = params.df;
  var loc = params.loc;
  var scale = params.scale;
  return {
    qvr: true,
    sample: function() {
      var z = _gaussian_sample(0, 1);
      var u = _gamma_sample(df / 2, 2);
      var t = z / Math.sqrt(u / df);
      return loc + scale * t;
    },
    score: function(x) {
      var z = (x - loc) / scale;
      var log_t = _lgamma((df + 1) / 2) - _lgamma(df / 2)
                  - 0.5 * Math.log(df * Math.PI)
                  - ((df + 1) / 2) * Math.log(1 + (z * z) / df);
      return log_t - Math.log(scale);
    },
    support: function() {
      return { lower: -Infinity, upper: Infinity };
    }
  };
};

var Weibull = function(params) {
  // torch Weibull(scale, concentration): density
  //   (k / lam) * (x / lam)^(k - 1) * exp(-(x / lam)^k), x >= 0,
  // with lam = scale and k = concentration.
  var lam = params.scale;
  var k = params.concentration;
  return {
    qvr: true,
    sample: function() {
      var u = Math.random();
      return lam * Math.pow(-Math.log(1 - u), 1 / k);
    },
    score: function(x) {
      if (x <= 0) { return -Infinity; }
      var z = x / lam;
      return Math.log(k) - Math.log(lam)
             + (k - 1) * Math.log(z)
             - Math.pow(z, k);
    },
    support: function() {
      return { lower: 0, upper: Infinity };
    }
  };
};

var NegativeBinomial = function(params) {
  // torch NegativeBinomial(total_count, probs): pmf proportional to
  //   C(k + r - 1, k) * (1 - probs)^r * probs^k, mean r * probs / (1 - probs).
  // Sampled as a gamma-Poisson mixture with gamma scale probs / (1 - probs).
  var r = params.total_count;
  var p = params.probs;
  return {
    qvr: true,
    sample: function() {
      var lambda = _gamma_sample(r, p / (1 - p));
      return _poisson_sample(lambda);
    },
    score: function(k) {
      if (k < 0) { return -Infinity; }
      return _lgamma(k + r) - _lgamma(k + 1) - _lgamma(r)
             + r * Math.log(1 - p) + k * Math.log(p);
    },
    support: function() {
      return { lower: 0, upper: Infinity };
    }
  };
};
