# Multivariate formula models

A multivariate model gives several responses their own
predictors and families in one probabilistic program. The responses can
share group effects, and Gaussian or Student responses can share a
residual correlation. QVR follows the brms interface
[@burkner2017brms], with Python functions in place of R formula
operators. Every joint model compiles to one QVR program and runs through
the ordinary [`fit`](../api/formulas/fit.md) entry point, so SVI, HMC,
NUTS, prediction, refits, and artifacts use the machinery of a single
response.

## Building a joint formula

[`bf`](../api/formulas/index.md#quivers.formulas.bf) builds one
response. Its positional arguments after the main formula are
distributional formulas such as `"sigma ~ z"`, and its keywords configure
that response (family, fixed parameters, trials, thresholds). Two
responses combine with `+`, as `bf(...) + bf(...)` does in brms, and
[`set_rescor`](../api/formulas/index.md#quivers.formulas.set_rescor)
adds the residual-correlation setting. A left side `mvbind(y1, y2)`
repeats one right side for each named response.

| brms | QVR |
| --- | --- |
| `bf(mvbind(y1, y2) ~ x + (1 \| p \| g))` | `bf("mvbind(y1, y2) ~ x + (1 \| p \| g)")` |
| `bf(y1 ~ x, sigma ~ z) + bf(k ~ x, family = bernoulli())` | `bf("y1 ~ x", "sigma ~ z") + bf("k ~ x", family="bernoulli")` |
| `mvbf(f1, f2, rescor = FALSE)` | `mvbf(f1, f2, rescor=False)` |
| `+ set_rescor(TRUE)` | `+ set_rescor(True)` |
| `set_prior("normal(0, 1)", class = "b", resp = "y1")` | `PriorSpec(distribution="Normal(0, 1)", class_="b", response="y1")` |
| `posterior_epred(fit, resp = "y1")` | `fitted.posterior_epred(resp="y1")` |

A response is named by its left-side variable with every character
other than a letter or digit removed, so `y_1 ~ x` names the response
`y1`. Parameter names follow brms: `b_<resp>_<coef>` for population
coefficients, `b_<dpar>_<resp>_<coef>` for distributional coefficients,
`<dpar>_<resp>` for unregressed auxiliary parameters,
`sd_<group>__<resp>_<coef>`, `cor_<group>__<term>__<term>`, and
`rescor__<a>__<b>`. A raw `"mvbind(...) ~ ..."` string passed to `fit` is
refused with a pointer to `bf`, so a joint model is always an explicit
[`MultivariateFormula`](../api/formulas/index.md#quivers.formulas.MultivariateFormula).
Response-level keywords such as `family` or `weights` given to `fit`
for a joint formula are refused, because they belong to one response.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import PriorSpec, bf, fit, get_prior, set_rescor

rows = np.arange(24)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["tarsus"] = 0.5 * data["x"] + 0.1 * np.sin(3 * rows)
data["back"] = -0.3 * data["x"] + 0.1 * np.cos(5 * rows)
data["k"] = (data["z"] > 0).astype(int)

spec = (
    bf("tarsus ~ x + (1 | p | g)", "sigma ~ z")
    + bf("back ~ x + (1 | p | g)")
    + bf("k ~ x + (1 | p | g)", family="bernoulli")
    + set_rescor(False)
)
priors = PriorSpec(distribution="Normal(0, 1)", class_="b", response="back")
assert get_prior(spec, data=data, priors=priors)["response_back__beta_x"] == (
    "Normal(loc=0, scale=1)"
)
fitted = fit(spec, data=data, priors=priors, method="svi", num_steps=50, seed=1)
assert fitted.response_names == ("tarsus", "back", "k")
assert {"b_tarsus_x", "b_sigma_tarsus_z", "sd_g__k_Intercept"} <= set(
    fitted.variables(num_samples=4)
)
assert "cor_g__tarsus_Intercept__back_Intercept" in fitted.variables(num_samples=4)
```

## Shared group effects

A grouping term `(1 | p | g)` that repeats one identifier `p` across
responses places all of its coefficients in one Gaussian block with one
[LKJ](https://en.wikipedia.org/wiki/Lewandowski-Kurowicka-Joe_distribution)
correlation matrix [@lewandowski2009lkj], exactly as in brms. In the model
above the block has three coordinates, one per response intercept.
Coefficients without a shared identifier keep their own blocks. The
centered and noncentered coordinates of `BayesianFit` apply unchanged.

## Residual correlation

With `set_rescor(True)`, the responses of each row form one vector event.
Each response keeps its own location predictor and scale `sigma_<resp>`,
and one residual correlation matrix joins them through the Cholesky factor
of a multivariate Normal or a multivariate Student law. brms allows this
only when every response uses the same Gaussian or Student family, and so
does QVR. A heterogeneous family set, a censored or truncated response,
and responses with different `subset()` rows are refused with a located
error. Student responses share one degree of freedom `nu`, as in brms.
When `set_rescor` is absent, the brms default applies: residual
correlation is estimated exactly when every response uses one common
Gaussian or Student family and no response is an `mi()` response.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import bf, fit, set_rescor

rows = np.arange(24)
data = pd.DataFrame({"x": np.sin(rows), "z": np.cos(rows)})
data["g"] = np.array(["a", "b", "c"])[rows % 3]
data["tarsus"] = 0.5 * data["x"] + 0.1 * np.sin(3 * rows)
data["back"] = -0.3 * data["x"] + 0.1 * np.cos(5 * rows)
data["k"] = (data["z"] > 0).astype(int)

correlated = fit(
    bf("mvbind(tarsus, back) ~ x") + set_rescor(True),
    data=data,
    method="svi",
    num_steps=50,
    seed=2,
)
draws = correlated.posterior_predict(num_samples=8, seed=3)
assert draws["tarsus"].shape == draws["back"].shape == (1, 8, 24)
scores = correlated.log_likelihood(num_samples=8)
assert set(scores) == {"tarsus__back"}
assert scores["tarsus__back"].shape == (1, 8, 24)
assert correlated.variables(num_samples=8)["rescor__tarsus__back"].shape == (1, 8)
```

## Prediction and post-processing

[`MultivariateFit`](../api/formulas/index.md#quivers.formulas.MultivariateFit)
methods take a `resp` selection: one label, several labels, or `None` for
every response. `posterior_linpred`, `posterior_epred`, and
`posterior_predict` return one tensor per selected response with the
shapes of the `BayesianFit` methods. Fresh rows in `new_data` are bound
by each response's own fitted transforms, contrasts, trials, and response
roles. The joint program is then compiled once per request.
`re_formula`, `allow_new_levels`, and `sample_new_levels` keep their
meaning for a `BayesianFit`. A new level of a shared grouping factor receives
one draw for the whole block, so its effects on different responses keep
their fitted correlation. Under residual correlation every row is drawn as
one vector event before responses are selected.

`log_likelihood` follows the units of brms `log_lik`. Under residual
correlation a row's vector event is one unit, and a strict subset of
responses is scored by the exact marginal of that event. Without residual
correlation, one selected response gives its own pointwise scores and
several selected responses are summed per row. The result has one entry,
keyed by the selected labels joined with `"__"`. `analysis`,
`refit_callback`, `PosteriorAnalysis.reloo`, and `cross_validate` reuse
these units. `fixed_effects`, `random_effects`, and `coefficients` take
`resp=` and return the structure of the corresponding `BayesianFit` methods.
`variables` returns draws under brms names, and `hypothesis` evaluates a
contrast over those names.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import bf, fit, set_rescor

rows = np.arange(24)
data = pd.DataFrame({"x": np.sin(rows), "z": np.cos(rows)})
data["g"] = np.array(["a", "b", "c"])[rows % 3]
data["tarsus"] = 0.5 * data["x"] + 0.1 * np.sin(3 * rows)
data["back"] = -0.3 * data["x"] + 0.1 * np.cos(5 * rows)
data["k"] = (data["z"] > 0).astype(int)

fitted = fit(
    bf("tarsus ~ x + (1 | p | g)", "sigma ~ z")
    + bf("back ~ x + (1 | p | g)")
    + bf("k ~ x + (1 | p | g)", family="bernoulli")
    + set_rescor(False),
    data=data,
    method="svi",
    num_steps=50,
    seed=1,
)
fresh = pd.DataFrame({"x": [0.2, -0.4], "z": [0.1, 0.3], "g": ["a", "new"]})
means = fitted.posterior_epred(
    resp=["tarsus", "k"],
    new_data=fresh,
    allow_new_levels=True,
    sample_new_levels="gaussian",
    num_samples=8,
    seed=4,
)
assert set(means) == {"tarsus", "k"}
assert means["k"].shape == (1, 8, 2)
summary = fitted.hypothesis("b_tarsus_x - b_back_x", num_samples=8)
assert summary.estimate.shape == ()
refit = fitted.update(num_steps=20)
assert refit.response_names == fitted.response_names
```

`update`, `refit_callback`, and the artifact functions
[`save_fit` and `load_fit`](../api/modeling.md) accept a joint fit. A
restored fit rebuilds every response design with the same preparation and
checks the regenerated QVR source and bindings against the saved ones. It
runs no inference.

## Differences from brms

QVR keeps brms's model, names, and prediction units, with these
differences.

- brms centers population design columns before sampling and reports the
  intercept after transforming it back. QVR samples uncentered coefficients
  by default, so `b_<resp>_Intercept` is the sampled intercept and its prior
  applies to the uncentered intercept. `bf(..., center=True)` and
  `fit(..., default_priors="brms")` sample the centered intercept as brms
  does and report `b_<resp>_Intercept` on the original scale.
- Default priors are the QVR formula defaults (for instance a
  `Normal(0, 5)` intercept), not the Student defaults that brms scales by the data.
  The residual correlation matrix receives `LKJCholesky(1)` as in brms.
- Joint prediction binds every response's predictors, because the joint
  program evaluates all responses in each replay.
- Neural predictors are refused, as are differing likelihood weight
  columns under residual correlation.
- A `mi()` response fits and predicts in a joint model. [Missing data and
  measurement error](formula-missing-data.md) states the model and its
  limits.
