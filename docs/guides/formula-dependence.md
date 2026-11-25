# Residual dependence structures

brms adds residual dependence to a formula with autocorrelation terms
[@burkner2017brms]. QVR parses the same terms, fits them with the existing
SVI, HMC, and NUTS drivers, and replays them through the shared prediction,
likelihood, refit, and artifact paths. The term changes the response law. It
does not add a predictor column.

| Term | Structure | Parameters (prior class) | Default prior |
|---|---|---|---|
| `ar(time, gr, p)`, `ma(time, gr, q)`, `arma(time, gr, p, q)` | conditional ARMA of order (p, q) | `ar_1`, ..., `ma_1`, ... (`ar`, `ma`) | `Normal(0, 1)` |
| the same with `cov = TRUE` | stationary ARMA covariance, orders at most one | `ar_1`, `ma_1` (`ar`, `ma`) | `Uniform(-1, 1)` |
| `cosy(time, gr)` | compound symmetry | `cosy` | `Uniform(0, 1)` |
| `unstr(time, gr)` | unstructured correlation over time levels | `Lcortime` (`Lcortime`) | `LKJCholesky(1)` |
| `sar(M, type = "lag")`, `sar(M, type = "error")` | simultaneous autoregression | `lagsar`, `errorsar` | uniform between the extreme eigenvalues of `M` |
| `fcor(M)` | fixed covariance `M` scaled by `sigma` | none | none |
| `car(M, gr, type)` with `type` one of `escar`, `esicar`, `icar`, `bym2` | conditional autoregressive location effect | `sdcar`, `car` (`escar`), `rhocar` (`bym2`) | `random_scale_prior`, `Uniform(0, 1)`, `Beta(1, 1)` |

For `family="gaussian"` and `family="student_t"`, the residual terms change
the response law, as brms's natural-residual formulation does. For every
other family with one location predictor, brms adds latent residuals to the
linear predictor instead, and QVR does the same (see below). A `car` term
adds a spatial effect to the linear predictor of any such family. Matrices
named in `sar`, `fcor`, and `car` come from the `data2` argument, as in brms.

As in brms, a formula may combine one time-series term, one spatial term,
and at most one covariance of natural residuals. A conditional ARMA term
holds no covariance, so it combines with `sar` or with `fcor`, and a `car`
term combines with any residual term. Every other pair of residual terms is
refused with brms's message.

## A first-order autoregression

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import PriorSpec, fit, get_prior

rows = []
for series in ("a", "b"):
    noise = 0.0
    for time in range(8):
        noise = 0.6 * noise + 0.3 * np.sin(3.0 * time + len(series))
        rows.append({"g": series, "t": time, "x": time / 8, "y": 0.5 * time / 8 + noise})
data = pd.DataFrame(rows)

assert "ar_1" in get_prior("y ~ x + ar(time = t, gr = g)", data=data)
model = fit(
    "y ~ x + ar(time = t, gr = g)",
    data=data,
    method="svi",
    num_steps=50,
    seed=1,
    priors=PriorSpec(distribution="Normal(0.0, 0.5)", class_="ar"),
)
forecast = pd.DataFrame(
    {"g": "a", "t": [8, 9, 10], "x": [1.0, 1.125, 1.25], "y": [0.4, np.nan, np.nan]}
)
draws = model.posterior_predict(new_data=forecast, num_samples=20, seed=2)
assert draws.shape == (1, 20, 3)
```

## Time ordering and groups

As in brms, rows are ordered by the grouping factor `gr` and then by `time`
within each group. Without `time`, the data order defines time. Without `gr`,
all rows form one series. Time points must be unique within a group. Groups
may have different lengths. The fitted rows keep their original order in
every prediction and likelihood result.

The conditional ARMA formulation adds lagged residuals to the linear
predictor, so its rows need no positions beyond their order. The covariance
formulations (`cov = TRUE`, `cosy`) assign positions one to n within each
group, and every group uses the leading block of one correlation matrix.
`unstr` instead indexes its correlation matrix by the observed time values,
so a group missing a time level uses the matching sub-block.

## Student responses

A conditional ARMA model with `family="student_t"` gives every row an
independent Student innovation. A covariance formulation or a SAR term uses
one [multivariate Student law](https://en.wikipedia.org/wiki/Multivariate_t-distribution)
per correlated series, so all rows of a series share one mixing scale. This
follows brms's `student_t_time` and `student_t_lagsar` likelihoods.

## Prediction

`posterior_epred`, `posterior_predict`, `posterior_linpred`, and
`log_likelihood` follow brms.

- For the conditional ARMA formulation, the expected response and `mu`
  include the ARMA terms computed from response history. Predictions on the
  fitted rows condition on the observed responses, one step ahead.
- New rows form their own series, as brms's `newdata` does. Rows whose
  response is present serve as history. A row with a missing or NaN response
  is predicted, and a draw from its predictive law becomes the history of the
  following rows. brms's `oos` argument corresponds to setting those rows'
  responses to NaN. Each returned draw is a separate draw from the row's
  predictive law, as in brms.
- Covariance formulations predict each series jointly. Their expected
  response is the linear predictor.
- A lag SAR expectation is $(I-\rho W)^{-1}\mu$. SAR and `fcor` predictions
  on new rows read a matrix over those rows from the `data2` argument of the
  prediction method, brms's `newdata2`.
- `incl_autocor=False` also removes `car` effects and latent residuals.
- `incl_autocor=False` removes the structure: rows become independent with
  the fitted residual scale.
- `unstr` refuses time values that were not fitted.

`log_likelihood` returns brms's pointwise values. A conditional ARMA row
scores its one-step conditional density. A covariance or SAR row scores its
leave-one-out predictive density given all other rows of its series
[@burkner2021efficient], the value that PSIS-LOO requires for these
non-factorized models. Those values do not sum to the joint density. The
fitted target always uses the exact joint density.

## Combining terms

### A conditional ARMA term with `sar` or `fcor`

brms adds the conditional ARMA terms to the linear predictor and then scores
the whole response vector under the SAR or fixed-covariance law. With
$\mu_n=\eta_n+\sum_k\theta_k e_{n-k}+\sum_k\phi_k e_{n-k}$ and
$e_n=y_n-\eta_n-\sum_k\theta_k e_{n-k}$ computed along each series, a lag SAR
scores $(I-\rho W)y-\mu$, an error SAR scores $(I-\rho W)(y-\mu)$, and `fcor`
scores $y-\mu$ under covariance $\sigma^2 M$. The lag SAR expectation is
$(I-\rho W)^{-1}\mu$, and `log_likelihood` returns the leave-one-out
densities of the SAR or fixed-covariance law around $\mu$.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import fit

rows = [
    {"g": series, "t": time, "x": np.cos(1.7 * time + shift)}
    for shift, series in enumerate(("a", "b"))
    for time in range(5)
]
data = pd.DataFrame(rows)
data["y"] = 0.4 * data.x + 0.3 * np.sin(2.0 * np.arange(10))
W = np.zeros((10, 10))
for i in range(9):
    W[i, i + 1] = W[i + 1, i] = 1.0
W = W / W.sum(axis=1, keepdims=True)

model = fit(
    "y ~ x + ar(time = t, gr = g) + sar(W)",
    data=data,
    data2={"W": W},
    method="svi",
    num_steps=20,
    seed=1,
)
assert {"ar_1", "lagsar"} <= set(model.draws(num_samples=1))
fresh = data.iloc[:5]
expected = model.posterior_epred(
    new_data=fresh, data2={"W": W[:5, :5] / W[:5, :5].sum(axis=1, keepdims=True)},
    num_samples=4,
)
assert expected.shape == (1, 4, 5)
```

Every row's response must be present when these terms are combined. brms
draws a missing response from the joint law of all rows, which defines no
draw for one row, so QVR refuses missing history here.

### Known standard errors with a time-series covariance

`y | se(s, sigma = TRUE) ~ ... + cosy(time, gr)`, and the same with
`unstr` or an ARMA term with `cov = TRUE`, adds the squared standard errors
to the diagonal of each series covariance,
$\Sigma_g=\operatorname{diag}(\sigma)\,R\,\operatorname{diag}(\sigma)+\operatorname{diag}(s^2)$,
brms's `normal_time_hom_se` and `student_t_time_hom_se` likelihoods. The
measurement errors thus stay independent across the rows of a series. brms
requires `sigma = TRUE` here, and so does QVR. Each series then has its own
covariance factor, O(n_g^3) for a series of n_g rows. New rows read their
standard errors from the prediction frame, and `incl_autocor=False` keeps
them in the scale $\sqrt{\sigma^2+s^2}$ of independent rows.

<!-- python: run -->
```python
data["s"] = 0.1 + 0.05 * data.t
known = fit(
    "y | se(s, sigma = TRUE) ~ x + cosy(time = t, gr = g)",
    data=data,
    method="svi",
    num_steps=20,
    seed=1,
)
assert known.log_likelihood(num_samples=2)["y"].shape == (1, 2, 10)
```

### Censoring, truncation, and weights with a conditional ARMA term

The rows of the conditional ARMA formulation are conditionally independent
given the earlier responses, so brms adjusts each row. The residuals $e_n$
use the recorded response of every row, which for a censored row is its
recorded endpoint (the lower endpoint of an interval). A censored row then
contributes a tail or interval mass of its conditional law
$F(\cdot\mid\mu_n,\sigma)$, a truncated row divides by the mass of its
truncation interval, and a weight multiplies the row's log density.

<!-- python: run -->
```python
data["status"] = np.where(data.y > 0.3, "right", "none")
censored = fit(
    "y | cens(status) + trunc(lb = -5, ub = 5) ~ x + ar(time = t, gr = g)",
    data=data,
    method="svi",
    num_steps=20,
    seed=1,
)
draws = censored.posterior_predict(num_samples=3, seed=2)
assert bool(((draws >= -5) & (draws <= 5)).all())
```

Prediction draws each row from its truncated conditional law. A missing
response among new rows is drawn from that truncated law before it enters
the history of later rows, as in brms.

## Refits and cross-validation

`update` and `refit_callback` reuse the retained recipe, including `data2`.
Fold refits recompute the series from the training rows of each fold, and
the held-out rows form their own series when scored, as brms's `kfold` does
through its `newdata` path. brms places no restriction on the folds of these
models. A fold that removes rows from inside a series joins the remaining
rows as if they were consecutive, because the ARMA terms use row order
rather than time values. Folds that hold out whole series by `gr` avoid
this, and they are the folds QVR's tests use for time-series terms. A SAR or
`fcor` matrix covers all fitted rows, so partitioned refits of those terms
fail with the dimension error that brms also raises. `save_fit` stores the `data2` matrices with the other fitting
options, and `load_fit` rebuilds the same response law.

## Cost

Scoring the conditional ARMA formulation costs O(n (p + q)) for n rows. The
stationary ARMA and compound-symmetry formulations factor one correlation of
the longest series, O(m^3) for m positions, and then solve each series. A
SAR term computes the eigenvalues of `M` once when the model is prepared.
Each evaluation then costs O(nnz) sparse products plus an O(n) log
determinant $\sum_i \log(1-\rho\lambda_i)$ [@ord1975estimation], and never
forms the dense covariance. Lag SAR prediction solves
$(I-\rho W)x=b$ by sparse LU.

## Conditional autoregressive effects

`car(M, gr = loc, type = ...)` adds `rcar[loc]` to the linear predictor
[@morris2019spatial]. The predictor may be a location, a distributional or
nonlinear parameter, or the location of a mixture component, and each
predictor has a field of its own, named after its parameter as in brms
(`sdcar_sigma`, `rcar_sigma[1]`). The location of a categorical or other
per-category response has one independent field per free category
(`sdcar_muB`, `sdcar_muC`). `M` is an adjacency matrix over locations, passed as a
pandas DataFrame whose index and columns are the location labels, which must
cover the levels of `gr`. Without `gr`, every row is a location and `M` is
indexed by row. Every nonzero entry marks two neighbors. `M` must be square,
symmetric, and zero on the diagonal, and `escar` and `esicar` require every
location to have a neighbor, as brms requires.

- `escar` draws `rcar` from the proper CAR law with precision
  $\tau(D-\alpha W)$, where $D$ holds neighbor counts, $\tau=1/\mathrm{sdcar}^2$,
  and $\alpha$ is `car`.
- `esicar` and `icar` use the intrinsic law with precision $\tau(D-W)$.
- `bym2` sets
  $\mathrm{rcar}=\mathrm{sdcar}\,(\sqrt{1-\rho}\,\theta+\sqrt{\rho/s}\,\phi)$
  with $\theta\sim\mathcal N(0,I)$, an intrinsic $\phi$, $\rho$ = `rhocar`,
  and the scaling factor $s$ [@riebler2016intuitive].

The intrinsic effects sum to zero within every connected component of the
graph, exactly: the sampler moves the free coordinates and the last location
of each component is minus the sum of the others. The scaling factor $s$ is
the geometric mean of the diagonal of the generalized inverse of the graph
[Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix), which is the
marginal variance of the intrinsic law under that constraint. Scoring uses
the sparse edge list, O(n + E) for n locations and E edges. The eigenvalues
needed by `escar` and the scaling factors are computed once, when the model
is prepared.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
from quivers.formulas import fit

labels = [f"L{i}" for i in range(5)]
adjacency = np.zeros((5, 5))
for i in range(4):
    adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
M = pd.DataFrame(adjacency, index=labels, columns=labels)
data = pd.DataFrame({"loc": labels * 2, "x": np.linspace(-1, 1, 10)})
data["y"] = [0, 1, 2, 1, 3, 1, 0, 2, 4, 2]
model = fit(
    "y ~ x + car(M, gr = loc, type = 'bym2')",
    data=data,
    data2={"M": M},
    family="poisson",
    method="svi",
    num_steps=20,
)
rates = model.posterior_epred(
    new_data=pd.DataFrame({"loc": ["L4"], "x": [0.5]}), num_samples=5
)
assert rates.shape == (1, 5, 1)
```

Prediction maps rows to fitted locations. New locations, and new rows of a
`car` term without `gr`, are refused with brms's messages.

## Latent residuals for other families

For a family without natural residuals, such as `poisson` or `bernoulli`,
brms adds a latent vector `err` with scale `sderr` to the linear predictor.
A conditional `ar` term draws `err` independently and adds
$\sum_k \phi_k\,\mathrm{err}_{t-k}$, so `ma` terms there need `cov = TRUE`. A
covariance term (`cov = TRUE`, `cosy`, `unstr`) correlates `err` within each
series. The ARMA coefficients are bounded by $[-1, 1]$ and `cosy` by
$[0, 1]$, as brms bounds them, and `sderr` takes `random_scale_prior`. `sar`
and `fcor` need natural residuals and are refused, as in brms. The fitted
`err` values are replayed on the training rows. New rows of a covariance
term draw fresh residuals from the structure at each posterior draw, as
brms does, and new rows of the conditional formulation are refused with
brms's message.

## Where QVR differs from brms

1. brms gives the conditional ARMA coefficients an improper flat prior, and
   so does `default_priors="brms"`. The default quivers policy uses the
   proper `Normal(0, 1)`, which keeps the same unbounded support and leaves
   prior draws, `sample_prior`, and SMC evidence available; these refuse a
   fit with flat priors.
2. brms's SAR likelihoods omit the constants $-\tfrac{n}{2}\log(2\pi)$ and
   $-\tfrac{n}{2}\log\pi$. QVR scores the normalized densities, so their
   values differ by a constant.
3. A `sar` or `fcor` matrix indexes the rows of `data` as given. brms sorts
   the rows of a model with a time-series term by group and time and leaves
   the matrix as given, so in brms the matrix rows of a conditional ARMA term
   beside `sar` or `fcor` are the sorted rows. The two agree for data sorted
   by `gr` and then `time`. QVR keeps the row identities of the data, because
   a matrix row then names the same observation whatever the row order.
4. brms's intrinsic CAR (`icar`, `bym2`) uses a soft sum-to-zero over all
   locations, and its `esicar` and BYM2 scaling assume one connected graph.
   QVR constrains each connected component exactly and scales each component
   by its own factor, following @riebler2016intuitive. On a connected graph
   the two agree, up to brms's diagonal jitter of order 1e-8. An isolated
   location then has no intrinsic effect.
5. Every refusal of `se()`, likelihood weights, censoring, or truncation
   beside a residual term follows brms. Its time-covariance and FCOR
   likelihoods reject weights, censoring, and truncation
   (`R/stan-likelihood.R`, `stan_log_lik_gaussian_time` and
   `stan_log_lik_gaussian_fcor`), FCOR rejects `se()`, the conditional ARMA
   formulation rejects `se()` (`R/stan-predictor.R`, `stan_ac`), and a
   time-series covariance requires `se(..., sigma = TRUE)`. For a `sar` term
   brms writes each of these response terms into a Stan program that does
   not compile, because its SAR densities take the whole response vector and
   one scalar `sigma`. No brms model exists to match, and QVR refuses those
   combinations with a message that says so.
6. brms predicts new rows of a `sar` or `fcor` term with the fitted matrix
   when `newdata2` is absent and the row counts agree. QVR requires a matrix
   over the new rows in `data2`, because an equal row count does not make the
   new rows the fitted observations.
7. With a conditional ARMA term beside `sar` or `fcor`, a missing response
   among the prediction rows is refused, for the reason given above.
8. A row that is both censored and truncated scores the mass of its
   censored event inside the truncation interval, as every censored and
   truncated response does in QVR. brms divides the whole tail mass by the
   mass of the truncation interval. The two agree for rows that are only
   censored or only truncated.
