# Priors, Formula Options, and Fitting Controls

This page gives the formula counterparts of brms's prior surface
(`set_prior`, `default_prior`, `validate_prior`, `prior_summary`), its
formula options (`bf(center, cmc, sparse, decomp, unused)`), the R formula
operators brms reads through R's `model.matrix`, and the sampler arguments of
`brm` [@burkner2017brms]. The [semantics page](../semantics/programs.md#brms-intercept-centering-and-qr-coordinates)
states the densities, and the
[roadmap](../developer/model-interface-roadmap.md) lists what is supported.

## Default prior policies

Every site no prior rule selects receives a default prior. Two policies are
available through the `fit`, `get_prior`, and `default_prior` keyword
`default_priors`:

- `"quivers"` (the default) keeps the proper defaults described under
  [coefficient priors](analysis-data-and-formulas.md#coefficient-priors-are-autoscaled):
  autoscaled Normal coefficients, half-normal group scales, and the family's
  auxiliary priors.
- `"brms"` reproduces `brms::default_prior`. Population coefficients
  (class `b`) receive the improper flat prior. The centered intercept
  receives a Student-t prior with three degrees of freedom, location the
  rounded median of the linked response (less the mean offset), and scale
  `max(2.5, round(mad(link(y)), 1))`. Standard deviations (`sd`, `sds`,
  `sdgp`, `sderr`, `sdcar`) and an estimated `sigma` receive the half
  Student-t with three degrees of freedom and the same scale. Correlation
  matrices receive `LKJCholesky(1)`, and auxiliary parameters receive the
  defaults of brms's `def_dpar_prior`, including the family-specific ones of
  brms 2.23 (`inv_gamma(0.4, 0.3)` for a negative binomial shape).

Explicit rules win under either policy. The brms policy also centers the
location and distributional intercepts, so the intercept prior means what it
means in brms. The policy applies to categorical, simplex, ordinal, and
mixture models too: each free category's centered intercept receives
`StudentT(df=3.0, loc=0.0, scale=2.5)`, brms's `student_t(3, 0, 2.5)`, and a rule with `dpar="muB"` governs the
parameters of one category. Ordinal thresholds, mixture weights, baseline
hazards, and structured latent terms keep their quivers defaults, because
their coordinates differ from brms's.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import default_prior

table = default_prior("y ~ x + z + (1 | g)", data=data, default_priors="brms")
print(table.to_pandas()[["prior", "class", "coef", "group", "lb"]])
assert table.row("beta_x").prior == "flat"
assert table.row("intercept_centered").prior == "StudentT(df=3.0, loc=37.9, scale=5.6)"
assert table.row("sigma").prior == "HalfStudentT(df=3.0, scale=5.6)"
```

### Improper flat priors

[`ImproperFlat`](../api/formulas/index.md#quivers.formulas.ImproperFlat) is
brms's empty prior: a constant density, that is, the
[Lebesgue measure](https://en.wikipedia.org/wiki/Lebesgue_measure) on the
site's domain. It has no normalizing constant, so the posterior can be proper
while the prior is not. Fitting with NUTS, HMC, or SVI accepts it. Prior
draws, `sample_prior="yes"` or `"only"`, prior initialization, and SMC
evidence refuse it with an error naming the flat sites, as brms refuses prior
sampling with improper priors. A lower bound of zero gives the flat prior on
the positive half line, and two finite bounds give the uniform distribution
on the interval.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import ImproperFlat, PriorSpec, fit

fitted = fit(
    "y ~ x + z",
    data=data,
    priors=PriorSpec(distribution=ImproperFlat(), class_="b"),
    method="svi",
    num_steps=200,
    seed=1,
)
try:
    fitted.prior_draws(num_samples=10)
except ValueError as error:
    assert "improper flat prior" in str(error)
```

### Constants, tags, and prior tables

[`ConstantPrior`](../api/formulas/index.md#quivers.formulas.ConstantPrior) is
brms's `constant(const, broadcast)`: the site becomes a deterministic binding,
and `draws()` and coefficient summaries report the constant. On a correlation
class (`cor`, `rescor`) the constant is the lower
[Cholesky factor](https://en.wikipedia.org/wiki/Cholesky_decomposition) of
the correlation matrix, the parameter brms fixes there: its `K * K` entries
in row order with `broadcast=False`. For two coefficients,
`ConstantPrior(value=(1.0, 0.0, 0.6, 0.8), broadcast=False)` fixes their
correlation at 0.6. A single broadcast value is not a valid factor and is
refused. A rule's `tag` is carried into the prior table, as brms's `tag` is.

A tuple with `broadcast=False` on a class of several coefficients fixes each
of them, as brms's `constant(v, broadcast = FALSE)` fixes a parameter vector
to the vector `v`. `ConstantPrior(value=(8.0, -3.0), broadcast=False)` on
class `b` fixes the first population coefficient at 8 and the second at -3,
and on class `sd` with a `group` it fixes that factor's standard deviations.
The rule needs one value for each coefficient it selects within one
predictor and grouping factor, and a different count is refused with the
selected sites named. A rule that selects several grouping factors or
predictors gives each of them the same vector, as brms does.

The values follow the order of the coefficients in the prior table, which
is the order of the formula's terms. brms follows the column order of R's
model matrix, which places every interaction after the main effects. The
two orders thus differ for a formula such as `y ~ x * z + w`, whose
coefficients are `x`, `z`, `x:z`, `w` here and `x`, `z`, `w`, `x:z` in brms.
brms's separate vectors of smooth and special-term coefficients (`bs`,
`bsp`) are class `b` rows of the same table. In brms a vector constant is a
Stan variable passed through `stanvars`, and here it is the tuple itself.

A rule selects a class by its QVR name or by its brms name. The brms names
`rescor` (residual correlation of a joint model), `lncor` (correlation of a
logistic-normal response), `delta` (spacing of equidistant ordinal
thresholds), and the Cholesky spellings `L`, `Lrescor`, and `Llncor` select
the sites that the prior table lists as `cor`, `rescor`, and
`cutpoint_spacing`.
[`validate_prior`](../api/formulas/index.md#quivers.formulas.validate_prior)
checks rules against a model and returns the complete
[`PriorTable`](../api/formulas/index.md#quivers.formulas.PriorTable), with
brms's columns `prior, class, coef, group, resp, dpar, nlpar, lb, ub, tag,
source`. A fitted model reports the same table with
[`prior_summary`](../api/formulas/index.md#quivers.formulas.prior_summary).
`default_prior` and `validate_prior` take the data options `knots` and
`drop_unused_levels` with the meaning they have in `fit`, so the table lists
the coefficient of a declared factor level that no row uses when
`drop_unused_levels=False`, as brms's `get_prior` does.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import (
    ConstantPrior,
    PriorSpec,
    fit,
    prior_summary,
    validate_prior,
)

rules = (
    PriorSpec(distribution=ConstantPrior(value=8.0), class_="b", coef="x")
    + PriorSpec(distribution="Normal(0, 5)", class_="b", tag="slopes")
)
checked = validate_prior(rules, "y ~ x + z", data=data, default_priors="brms")
assert checked.row("beta_x").prior == "constant(8.0)"
assert checked.row("beta_z").tag == "slopes"

fitted = fit(
    "y ~ x + z", data=data, priors=rules, default_priors="brms",
    method="svi", num_steps=200, seed=2,
)
assert float(fitted.draws(num_samples=4)["beta_x"].min()) == 8.0
assert prior_summary(fitted).rows == checked.rows

both = PriorSpec(
    distribution=ConstantPrior(value=(8.0, -3.0), broadcast=False), class_="b"
)
table = validate_prior(both, "y ~ x + z", data=data)
assert table.row("beta_x").prior == "constant(8.0)"
assert table.row("beta_z").prior == "constant(-3.0)"
```

### Shrinkage across classes

The regularized horseshoe [@piironen2017sparsity; @piironen2016hyperprior]
and the R2D2 prior [@zhang2022r2d2] may cover several classes of one
predictor. A rule marked `main=True` gives the arguments, and every
shrinkage rule of the same response and parameter joins its hierarchy, as
brms's `main = TRUE` does. A shrunken standard deviation is the shrinkage
scale itself, as in brms, so its rule takes no bounds.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import PriorSpec, RegularizedHorseshoe, validate_prior

shrinkage = PriorSpec(
    distribution=RegularizedHorseshoe(df=3.0, par_ratio=0.2, main=True)
) + PriorSpec(distribution=RegularizedHorseshoe(), class_="sd")
joined = validate_prior(shrinkage, "y ~ x + z + (1 | g)", data=data)
assert joined.row("sigma_g_Intercept").prior.startswith("= prior_main_")
```

## Intercept centering and QR coordinates

With `center=True` (the default under the brms policy), the location
intercept is sampled on the column-centered design and reported on the
original scale as `intercept = intercept_centered - mean(X) b`; priors of
class `Intercept` act on `intercept_centered`. With `center=False`, the
intercept is an ordinary class `b` coefficient. The `Intercept` variable gives
the same uncentered intercept explicitly, as in `y ~ 0 + Intercept + x`.

With `decomp="QR"`, the coefficients are sampled in scaled QR coordinates of
the (centered) design, `bQ = XR b`, with class `b` priors on `bQ`, as brms
does [@stanRegression]. Both maps are linear, so the reported coefficients,
predictions on new rows, refits, and saved fits use the original scale.
Both options apply to categorical, simplex, ordinal, and mixture models as
well. A categorical predictor gives each free category its own centered
intercept on the same transformed design, an ordinal predictor centers its
location intercept and keeps its thresholds in place, and category-specific
`cs()` columns stay outside the transformed design, as in brms.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import fit

data["w"] = data["x"] + 0.05 * np.cos(7 * rows)
qr = fit(
    "y ~ x + w + z", data=data, decomp="QR", center=True,
    num_warmup=150, num_samples=100, num_chains=1, seed=3,
)
draws = qr.draws()
assert {"beta_x_qr", "intercept_centered"} <= set(qr.posterior.samples)
assert {"beta_x", "intercept"} <= set(draws)
new_rows = pd.DataFrame({"x": [0.0, 0.5], "w": [0.1, 0.4], "z": [1.0, -1.0]})
print(qr.posterior_epred(new_data=new_rows).mean(dim=(0, 1)))
```

## Formula options and R operators

`cmc=False` evaluates a formula without intercept with treatment coding
rather than cell means. `sparse=True` is the same model with centering
disabled, as brms disables it for sparse designs. `unused=["v"]` keeps the
column `v` with the model rows, so a row missing `v` is excluded with the
other incomplete rows. Joint formulas take the same options through
`bf(..., center=, decomp=, cmc=, sparse=, unused=)`.

The formula layer reads R's operators as R does: `I(x^2)` is a power,
`(a + b + c)^2` expands to the main effects and all pairwise interactions,
`x %in% f` is the nested term `x:f`, `scale(x)` divides by the deviation with
`n - 1`, and `ns(x, df = k)` is the natural cubic spline basis of
`splines::ns` (prediction outside the boundary knots continues linearly).
`bs(x, df = k)` is the [B-spline](https://en.wikipedia.org/wiki/B-spline)
basis of `splines::bs` without its first column, with `degree`, `knots` and
`intercept` as in R (prediction outside the boundary knots continues the
boundary polynomials).
`update(formula=". ~ . + z")` follows R's `update.formula`. A `.` on the left
stands for the fitted response side and a `.` on the right for the fitted
terms. `+ term` adds a term that is not yet present and `- term` removes one,
including a group-level term such as `- (1 | g)`, which is kept, added, or
removed whole. `- 1` or `+ 0` removes the intercept, `+ 1` restores it, and a
one-sided `"~ . + z"` keeps the response. A term is matched by its text, so
`- x:z` after `x * z` is left to the formula algebra.

Factor levels follow brms's `drop_unused_levels`. By default a level that no
row uses has no column, as brms drops it. With `drop_unused_levels=False`, the
declared categories of a pandas categorical column keep their population
columns, in the declared order (columns of zeros, whose coefficients follow
their prior), and a grouping factor keeps its unused levels, whose group
effects follow their prior, so new rows at those levels are predicted without
`allow_new_levels`. A factor with unused levels inside a group-level term is
refused. Under the default, an ordered pandas factor keeps its unused
categories, where brms drops them.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

from quivers.formulas import fit

rows = np.arange(12)
levels = pd.DataFrame(
    {
        "x": np.sin(rows),
        "f": pd.Categorical(np.array(["a", "b"])[rows % 2], categories=["a", "b", "c"]),
        "g": pd.Categorical(np.array(["u", "v"])[rows // 6], categories=["u", "v", "w"]),
    }
)
levels["y"] = levels["x"] + 0.5 * (levels["f"] == "b") + 0.1 * np.cos(rows)
kept = fit("y ~ x + f + (1 | g)", data=levels, drop_unused_levels=False,
           method="svi", num_steps=50)
assert {"b_fc", "r_g[w,Intercept]"} <= set(kept.variables(num_samples=2))
```

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import fit

base = fit("y ~ x + scale(z) + ns(x, df=3)", data=data, method="svi",
           num_steps=100, seed=4)
reduced = base.update(formula=". ~ . - scale(z)", num_steps=100)
assert [c.name for c in reduced.formula.fixed_columns][:2] == ["Intercept", "x"]
```

## Sampler and fitting controls

| brms `brm` argument | quivers |
| --- | --- |
| `chains`, `iter`, `warmup` | `num_chains`, `num_samples` (retained draws per chain), `num_warmup` |
| `thin` | `thin`: each chain runs `num_samples * thin` sampling transitions and keeps every `thin`-th |
| `control = list(adapt_delta, max_treedepth, stepsize, metric)` | `control`: the same names, or a [`SamplerControl`](../api/formulas/index.md#quivers.formulas.SamplerControl) |
| `init = "random"`, `init_r` | `init="random"`, `init_r`: unconstrained starts uniform on `(-init_r, init_r)` |
| `init = 0` | `init=0` |
| `init = list(...)`, `init = function(chain_id)` | `init=[{...}, ...]` (one mapping per chain) or a function of the one-based chain index |
| `algorithm = "sampling"` | `algorithm="sampling"` (NUTS) |
| `algorithm = "meanfield"`, `"fullrank"`, `"laplace"` | SVI with `AutoNormalGuide`, `AutoMultivariateNormalGuide`, `AutoLaplaceApproximation` |
| `algorithm = "fixed_param"` | every draw keeps the chain's initial values |
| `algorithm = "pathfinder"` | multi-path Pathfinder with `num_chains` paths; CmdStan's arguments in `pathfinder` |
| `seed` | `seed` |
| `sample_prior` | `sample_prior` (refused with improper priors) |
| `file`, `file_refit` | `file`, `file_refit` through the artifact store |
| `save_pars` | every latent site is retained |
| `empty = TRUE` | `formula_to_qvr`, `get_prior`, `default_prior` |
| `cores` | `cores`: chains run in up to `cores` worker processes, each rebuilding the fit from its retained arguments; the draws equal the serial run's |
| `control_params(fit)` | `fit.control_params()`: `adapt_delta`, `max_treedepth`, `stepsize` (before adaptation), and `metric` of an MCMC fit |
| `inits(fit)` | `fit.inits()`: one mapping per chain from latent site to its initial value on the constrained scale |

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame(
    {
        "x": np.sin(rows),
        "z": np.cos(2 * rows),
        "g": np.array(["a", "b", "c"])[rows % 3],
    }
)
data["y"] = 40 + 8 * data["x"] - 3 * data["z"] + np.sin(5 * rows)

from quivers.formulas import fit

controlled = fit(
    "y ~ x + z", data=data,
    num_warmup=100, num_samples=20, num_chains=2, thin=2,
    control={"adapt_delta": 0.95, "max_treedepth": 8},
    init="random", init_r=1.0, seed=5,
)
assert controlled.posterior.samples["beta_x"].shape == (2, 20)
fixed = fit(
    "y ~ x + z", data=data, algorithm="fixed_param", num_samples=3, num_chains=1,
    init=[{"intercept": 40.0, "beta_x": 8.0, "beta_z": -3.0, "sigma": 1.0}],
)
assert float(fixed.draws()["beta_x"].max()) == 8.0
```

### Pathfinder

`algorithm="pathfinder"` runs multi-path Pathfinder [@zhang2022pathfinder]:
each path follows L-BFGS on the unconstrained log joint from a uniform
initial point on `(-init_r, init_r)`, keeps the normal approximation along
the path with the largest estimated evidence lower bound, and draws from it.
The paths' pooled draws are then resampled with Pareto-smoothed importance
weights. As in brms, `num_chains` is the number of paths. The remaining
cmdstanr arguments, which brms passes through its `...`, go in `pathfinder`:
`draws` (resampled draws, 1000), `single_path_draws` (draws per path, 1000),
`history_size` (5), `max_lbfgs_iters` (1000), `num_elbo_draws` (25),
`psis_resample`, `calculate_lp`, `save_single_paths`, `init_alpha`, and the
`tol_*` tolerances, with CmdStan's defaults. The draws are approximate and
independent: prediction, `log_likelihood`, `update`, `kfold`, and artifacts
use them as any other draws, and `fit.posterior.pareto_k` reports how far the
mixture of approximations is from the posterior. The
[formal statement](../semantics/programs.md#pathfinder-approximations) and
the [API page](../api/inference/pathfinder.md) give the details.

<!-- python: run -->
```python
import numpy as np
import pandas as pd

rows = np.arange(30)
data = pd.DataFrame({"x": np.sin(rows), "z": np.cos(2 * rows)})
data["y"] = 4 + 0.8 * data["x"] - 0.3 * data["z"] + 0.2 * np.sin(5 * rows)

from quivers.formulas import fit

approximate = fit(
    "y ~ x + z", data=data, algorithm="pathfinder", num_chains=2, seed=3,
    pathfinder={"draws": 200, "single_path_draws": 100, "num_elbo_draws": 10},
)
assert approximate.draws()["beta_x"].shape == (1, 200)
assert approximate.posterior_epred(new_data=data.iloc[:4]).shape == (1, 200, 4)
assert approximate.summary().algorithm == "pathfinder"
```

### Parallel chains

`cores=k` runs the chains in up to `k` worker processes. A compiled program
and a kernel's adaptation state stay in the process that made them, so each
worker rebuilds the fit from the arguments and data the fit retains for
refits, runs its share of the chains from the same per-chain random streams,
and returns the chains' records. The result is the serial run's, draw for
draw.

Each worker is a fresh Python interpreter that does not import the calling
script. A script may thus call `fit(..., cores=k)` at its top level, and it
needs no `if __name__ == "__main__":` block. Two conditions follow from
this. Every argument of the fit must be picklable (a function as `init`, for
instance, and not a lambda). A function or class given as an argument must
also live in an importable module, not in the script that calls `fit`,
because the workers never run that script. `fit` raises an error that states
this before any worker starts. When a worker fails, the other workers are
stopped and the worker's error is raised in the calling process.

<!-- python: run -->
```python
import numpy as np
import pandas as pd
import torch

rows = np.arange(16)
data = pd.DataFrame({"x": np.sin(1.7 * rows)})
data["y"] = 0.4 + 0.9 * data["x"] + 0.2 * np.cos(4.1 * rows)

from quivers.formulas import fit

budget = {"num_warmup": 15, "num_samples": 8, "num_chains": 2, "seed": 11}
serial = fit("y ~ x", data=data, **budget)
parallel = fit("y ~ x", data=data, cores=2, **budget)
assert torch.equal(serial.posterior.samples["beta_x"], parallel.posterior.samples["beta_x"])
```

Several brms arguments configure Stan itself and have no counterpart here:
`backend` (models run natively and transpile to other languages from QVR),
`threads` and `opencl` (within-chain parallelism of Stan), `stanvars`
(Stan code injection, where QVR source and family catalogs extend models),
`stan_model_args` (Stan compiler flags), `rename` (Stan parameter names),
and `normalize` (QVR always scores normalized densities). brms's `knots`
argument places the knots of `s()` terms, as the
[structured predictor guide](formula-special-predictors.md#bases) describes.
