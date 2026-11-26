# Missing data and measurement error

A formula model can treat missing values and noisy measurements as latent
quantities, as brms does [@burkner2017brms]. A response declared with
`mi()` keeps its family law: its recorded values are observed, and each
missing value is a latent coordinate drawn from that law. A predictor term
`mi(x)` uses those completed values. A predictor `me(x, sdx)` replaces a
noisy covariate by a latent true value with a known measurement error.
These terms compile through the joint program of
[multivariate formulas](formula-multivariate.md), so a model with them is
built with `bf` and fitted by the ordinary [`fit`](../api/formulas/fit.md).
A formula string passed directly to `fit` with these terms is refused with
a pointer to `bf`.

## Imputation during fitting with `mi()`

`bf("x | mi() ~ z")` models the response `x` from `z`, and
`bf("y ~ mi(x)")` uses the completed `x` in the model for `y`. The joint
density is brms's: with recorded values $x_o$ and latent missing values
$x_m$, the response law is evaluated on the completed vector, so the latent
values receive exactly the density of the model for `x`. Missing values of
`x` are thus imputed from that model and from their role in the model for
`y`, jointly in each posterior draw. An `mi()` term enters the predictor
linearly, also in interactions such as `mi(x):z`. A nonlinear use such as
`I(mi(x)^2)` is refused. A family admits `mi()` responses when its response
is one continuous value whose law has a density and no point masses, as in
brms: Gaussian, Student, skew Normal, log-Normal, ex-Gaussian, Gamma,
Weibull, Fréchet, exponential, inverse Gaussian, Beta, von Mises,
asymmetric Laplace, generalized extreme value, and any custom family with
such a law. Discrete, categorical, and vector responses, laws with point
masses (the hurdle and inflated families), mixtures, and the Wiener, Cox,
and shifted log-Normal families refuse them, each with the reason.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import bf, fit, set_rescor

data = pd.DataFrame(
    {
        "bmi": [22.0, 24.5, 21.0, 27.0, 25.5, 23.0, 26.0, 20.5],
        "chl": [180.0, np.nan, 170.0, 230.0, np.nan, 190.0, 215.0, np.nan],
        "age": [1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 1.0],
    }
)
data["chl"] = data["chl"] / 100
spec = (
    bf("bmi ~ age * mi(chl)")
    + bf("chl | mi() ~ age")
    + set_rescor(False)
)
fitted = fit(spec, data=data, method="svi", num_steps=50, seed=1)
names = set(fitted.variables(num_samples=4))
assert {"bsp_bmi_michl", "bsp_bmi_age:michl", "Ymi_chl[2]"} <= names
scores = fitted.log_likelihood(resp="chl", num_samples=4)["chl"]
assert bool(scores[..., 1].isnan().all())
```

The model has no residual correlation, which is the brms default when a
response uses `mi()`. Parameter names follow brms: `bsp_<resp>_mi<x>` for
`mi()` coefficients and `Ymi_<resp>[<row>]` for imputed values.

With `set_rescor(True)`, Gaussian or Student responses share one
[multivariate Normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
or
[multivariate Student](https://en.wikipedia.org/wiki/Multivariate_t-distribution)
law per row, and that law scores the completed vector of the row, brms's
`multi_normal_cholesky_lpdf(Yl | Mu, LSigma)`. A missing value is then
informed by its own model, by every model that uses it through `mi()`, and
by the recorded responses of its row through the residual correlation. As
in brms, the missing values have no prior of their own, so they carry the
improper flat law. Operations that need a proper prior on every site
(`sample_prior`, prior draws, and SMC evidence) thus refuse such a model.
A response of such a model may also carry a conditional ARMA term. The
recursion then reads the completed responses of earlier rows, and the
density is the exact joint law of the recorded and missing responses (see
[residual correlations across responses](formula-dependence.md#residual-correlations-across-responses)).

<!-- python: run -->
```python
correlated = fit(
    bf("bmi ~ age + mi(chl)") + bf("chl | mi() ~ age") + set_rescor(True),
    data=data,
    method="svi",
    num_steps=50,
    seed=1,
)
names = set(correlated.variables(num_samples=4))
assert {"rescor__bmi__chl", "Ymi_chl[2]", "bsp_bmi_michl"} <= names
joint = correlated.log_likelihood(num_samples=4)["bmi__chl"]
assert bool(joint.isfinite().all())
drawn = correlated.posterior_predict(num_samples=4)
assert drawn["bmi"].shape == drawn["chl"].shape == (1, 4, 8)
```

The pointwise unit of such a model is the row's vector event. On a
training row where a selected response has no recorded value, the score is
the log density of the row's recorded values under the joint law given the
completed value, and a row with no recorded value scores zero; brms reports
`NA` for such rows. `resp` selects the responses whose exact marginal is
scored. `posterior_predict` draws each row's vector jointly at
the completed predictor values, and `posterior_epred` returns the locations
at those values.

An `mi()` term may also be the slope of a group-level term, as in
`bmi ~ mi(chl) + (mi(chl) | clinic)`. The varying coefficient of each
group then multiplies the completed values, brms's
`(bsp + r[J[n]]) * Yl[n]`, and its names use the `mi` spelling:
`sd_clinic__bmi_michl` and `r_clinic__bmi[<level>,michl]`. The same term must
appear among the population-level terms. Without it brms declares the
varying slope and leaves it out of the predictor, so QVR refuses the formula.

<!-- python: run -->
```python
data["clinic"] = ["a", "b"] * 4
varying = fit(
    bf("bmi ~ mi(chl) + (mi(chl) | clinic)")
    + bf("chl | mi() ~ age")
    + set_rescor(False),
    data=data,
    method="svi",
    num_steps=20,
    seed=1,
)
assert "sd_clinic__bmi_michl" in varying.variables(num_samples=2)
```

## Known measurement error on a response

`bf("x | mi(sdx) ~ z")` declares known measurement standard deviations in
column `sdx`. Every true value is then latent, $x^\ast_i$ drawn from the
response law, and each recorded value is a Normal measurement
$x_i \sim \mathcal N(x^\ast_i, \mathrm{sd}_i)$. A missing value is a row
without a measurement. brms names the true values `Yl_<resp>[<row>]`.

## Measurement-error predictors with `me()`

`me(x, sdx)` adds a latent true covariate with brms's noncentered
parameterization $X_{\mathrm{me}} = \mu_{\mathrm{me}} +
\sigma_{\mathrm{me}} z$, $z \sim \mathcal N(0, 1)$, and each recorded
value is $x_i \sim \mathcal N(X_{\mathrm{me},i}, \mathrm{sd}_i)$. With
`gr = g`, rows of one level share a latent value, and their recorded
values and errors must agree, as in brms. The `me()` terms of a response
that share a grouping are correlated through one
[LKJ](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution)
Cholesky factor [@lewandowski2009lkj] unless `set_mecor(False)` is added.
Their hyperparameters have the brms prior classes `meanme`, `sdme`, and
`corme`, which `PriorSpec` selects.

<!-- python: run -->
```python
import pandas as pd

from quivers.formulas import PriorSpec, bf, fit, set_mecor

data = pd.DataFrame(
    {
        "y": [0.1, 0.4, -0.2, 0.3, 0.5, 0.0, 0.7, -0.1],
        "x": [1.0, 0.1, 0.5, 0.3, 2.0, 1.5, 0.4, 0.2],
        "z": [0.2, 0.1, -0.3, 0.4, 0.0, 0.3, -0.2, 0.6],
        "s": [0.1, 0.2, 0.3, 0.1, 0.2, 0.2, 0.4, 0.3],
    }
)
spec = bf("y ~ me(x, s) + me(z, s)") + set_mecor(True)
fitted = fit(
    spec,
    data=data,
    priors=PriorSpec(distribution="Normal(0, 2)", class_="meanme"),
    method="svi",
    num_steps=50,
    seed=2,
)
assert {"bsp_mexs", "meanme_mex", "sdme_mez", "corme__mex__mez"} <= set(
    fitted.variables(num_samples=4)
)
```

brms gives `meanme` and `sdme` flat priors, and so does
`default_priors="brms"` (flat on the real line for `meanme`, on `(0, inf)`
for `sdme`), which also scales the default priors of an `mi()` response on
its recorded values, as brms does. A flat prior is improper, so the default
quivers policy uses the formula defaults instead (`fixed_prior` for
`meanme`, `random_scale_prior` for `sdme`, `correlation_prior` for
`corme`), which keep prior draws and `sample_prior` available.

## Sparse imputation models

`index(id)` names the rows of an `mi()` response, and `mi(x, idx)` joins a
predictor to it by those identities, so the imputation model can have
fewer rows than the model that uses it. Together with `subset()`, this
expresses brms's sparse imputation models: `x | mi() + index(id) +
subset(sub) ~ z` models `x` on the rows where `sub` is true, and
`y ~ mi(x, idx)` reads the completed value whose `id` equals each row's
`idx`.

## Prediction, refits, and cross-validation

On the training rows, predictions of a model that uses `mi(x)` replay the
imputed values of each posterior draw. On fresh rows the recorded values
of `x` in `new_data` are used directly. brms returns missing predictions
for fresh rows whose `mi()` or `me()` values are missing, so QVR refuses
such rows with that reason. Pointwise log-likelihoods of an `mi()`
response are zero at its missing rows, which have no recorded likelihood
(brms reports `NA`).
`refit_callback` thus defines held-out units on the rows whose
`mi()` values used by the scored responses are recorded. The true values
of an `mi(sdx)` response are replayed from their posterior draws, which is
brms's behavior when they are saved with `save_pars(latent = TRUE)`.
Without saved values brms redraws them from the measurement model. `update`,
`save_fit`, and `load_fit` work as for any joint fit.

## Multiple imputation before fitting

[`fit_multiple`](../api/formulas/index.md#quivers.formulas.fit_multiple)
fits one formula to each of several imputed datasets, as brms
`brm_multiple` does, and pools the draws with
[`combine_models`](../api/formulas/index.md#quivers.formulas.combine_models).
The pooled posterior is the equally weighted mixture of the per-dataset
posteriors. Chains fitted to different datasets need not overlap, so
convergence is checked within each dataset with `MultipleFit.rhats`.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import fit_multiple

x = np.linspace(-1.0, 1.0, 10)
first = pd.DataFrame({"x": x, "y": 0.3 + 0.8 * x + 0.1 * np.sin(7 * x)})
second = first.assign(x=first["x"] + 0.05)
result = fit_multiple(
    "y ~ x",
    data=[first, second],
    method="nuts",
    num_warmup=20,
    num_samples=10,
    num_chains=2,
    seed=3,
)
assert result.pooled.draws()["beta_x"].shape[:2] == (4, 10)
assert len(result.rhats()) == 2
```

brms treats the combined fit as an ordinary fit, and so does
[`MultipleFit`](../api/formulas/index.md#quivers.formulas.MultipleFit): its
summaries, prediction methods, criteria, `conditional_effects`, `pp_check`,
and `hypothesis` are those of the pooled fit `result.pooled`, evaluated over
the pooled draws. Quantities that depend on the data use the first dataset,
as in brms. `combine=False` returns the separate fits instead, as brms's
`combine = FALSE` does.

<!-- python: run -->
```python
assert result.nchains() == 4
assert result.fixef().rows == ("Intercept", "x")
assert result.posterior_epred(new_data=first.iloc[:3]).shape == (4, 10, 3)
separate = fit_multiple(
    "y ~ x", data=[first, second], combine=False,
    method="svi", num_steps=5, seed=3,
)
assert len(separate) == 2
```

`MultipleFit.update` refits every dataset through the recipe of the first
fit, as brms's `update` of a `brm_multiple` fit does. The datasets must be
given again, the remaining arguments are kept unless overridden, and the
number of datasets may change. `data2` takes one mapping per dataset.

<!-- python: run -->
```python
third = first.assign(x=first["x"] - 0.05)
updated = result.update([first, second, third], num_samples=8)
assert updated.pooled.draws()["beta_x"].shape[:2] == (6, 8)
```
