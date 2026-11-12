# 8. Diagnostics and model comparison

Three checks follow a fit: whether the MCMC chains converged, whether data replicated from the fitted model resemble the observed data (a posterior predictive check), and how the model's predictive accuracy compares with another model's (leave-one-out cross-validation). [`quivers.diagnostics`](../../api/diagnostics/index.md) runs all three through [ArviZ](https://python.arviz.org/). This chapter applies them to the regression of [chapter 1](01-first-model.md), fitted with NUTS.

## The DataTree bridge

ArviZ reads an [`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html) with named groups such as `posterior`, `sample_stats`, `posterior_predictive`, `log_likelihood`, `observed_data`, and `constant_data`. [`to_datatree`](../../api/diagnostics/arviz_io.md) converts an `MCMCResult` into such a tree.

```python
import torch
from quivers.dsl import loads
from quivers.inference import NUTSKernel, MCMC, Predictive
from quivers.diagnostics import to_datatree

REGRESSION_SRC = """
object Item : FinSet 100

program regression : Item -> Item
    sample sigma  <- HalfNormal(scale=1.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let mu = beta_0 + beta_1 * x_design
    observe y : Item <- Normal(loc=mu, scale=sigma)
    return y

export regression
"""

program = loads(REGRESSION_SRC)
model = program.morphism

torch.manual_seed(0)
x_data = torch.randn(100)
y_data = 1.5 + 2.7 * x_data + 0.3 * torch.randn(100)

kernel = NUTSKernel(target_accept=0.9)
# Small budget for documentation; bump warmup/samples for real fits.
mcmc = MCMC(kernel, num_warmup=50, num_samples=100, num_chains=2)
result = mcmc.run(model, torch.zeros(1, 1),
                  {"x_design": x_data, "y": y_data})

# Posterior-predictive draws conditional on the fit. Reshape
# ``(num_samples, plate)`` into ``(num_chains, num_samples_per_chain,
# plate)`` for the ArviZ convention.
n_chains, n_per_chain = result.num_chains, result.num_samples
predictive = Predictive(
    model, posterior=result, num_samples=n_chains * n_per_chain,
)
ppc = predictive(torch.zeros(1, 1), {"x_design": x_data})
pp_y = ppc["y"].reshape(n_chains, n_per_chain, 100)

tree = to_datatree(
    result,
    observed_data={"y": y_data},
    posterior_predictive={"y": pp_y},
    constant_data={"x_design": x_data},
    coords={"Item": list(range(100))},
    dims={"y": ["Item"]},
)
```

`coords` and `dims` are optional. They let ArviZ label the axis of `y` by `Item` in tables and plots.

## Convergence diagnostics

```python
import arviz as az

summary = az.summary(tree, var_names=["beta_0", "beta_1", "sigma"])
print(summary)
```

```text
         mean     sd eti89_lb eti89_ub ess_bulk ess_tail r_hat mcse_mean mcse_sd
beta_0  1.544   0.03      1.5      1.6       66       56  1.02    0.0037  0.0028
beta_1   2.67  0.029      2.6      2.7      107       75  1.02     0.003  0.0018
sigma   0.279  0.019     0.25     0.31       36       42  1.01    0.0032  0.0017
```

The diagnostics and the usual thresholds [@vehtari2021rhat]:

| Column | Meaning | Threshold | If it fails |
|---|---|---|---|
| `r_hat` | Rank-normalized split R-hat, the larger of its bulk and folded forms: between-chain against within-chain variance | below 1.01 for every quantity | More warmup, more draws, or a better parameterization. |
| `ess_bulk`, `ess_tail` | Effective sample size: the number of independent draws with the same precision, for the center and the tails of the distribution | at least 100 per chain | Longer runs or a better parameterization. |
| `mcse_mean`, `mcse_sd` | Monte Carlo standard error of the posterior mean and standard deviation | small relative to `sd` | More draws. |
| divergences (`result.total_divergences`) | HMC trajectories whose integration error blew up [@betancourt2015hierarchical] | zero | Raise `target_accept` toward 0.99, or reparameterize ([chapter 3](03-hierarchical.md#non-centered-parameterization)). |

With 50 warmup steps and 100 draws for each of two chains, the run above misses the R-hat and effective sample size thresholds. Raise the budget for real fits.

The same estimators apply to any tensor laid out as `(num_chains, num_draws, *event_shape)`. `result.r_hat` uses [`rhat`](../../api/diagnostics/convergence.md#quivers.diagnostics.convergence.rhat), and `result.ess` and `result.ess_tail` use [`ess`](../../api/diagnostics/convergence.md#quivers.diagnostics.convergence.ess) with `method="bulk"` and `method="tail"`. [`mcse`](../../api/diagnostics/convergence.md#quivers.diagnostics.convergence.mcse) gives the Monte Carlo standard error. They also diagnose a quantity derived from the draws, such as the ratio of the slope to the noise scale:

```python
from quivers.diagnostics import ess, mcse, rhat

ratio = result.samples["beta_1"] / result.samples["sigma"]
print(rhat(ratio), ess(ratio, method="bulk"), ess(ratio, method="tail"))
print(mcse(ratio, method="mean"), mcse(ratio, method="quantile", prob=0.95))
```

Trace and rank plots show the same diagnostics:

<!-- python: skip -->
```python
az.plot_trace(tree, var_names=["beta_0", "beta_1", "sigma"])
az.plot_rank(tree,  var_names=["beta_0", "beta_1", "sigma"])
```

Well-mixed chains give overlapping, stationary traces and rank histograms with no systematic separation between chains.

## Posterior predictive checks

A [posterior predictive check](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html) compares a statistic $T$ of the observed data $y$ with its distribution over replicated data $y^{\text{rep}}$ drawn from the fitted model [@gelman1996predictive; @stanPredictiveChecks]. Plot the replicated data against the observed:

<!-- python: skip -->
```python
az.plot_ppc(tree, var_names=["y"], num_pp_samples=200)
```

[`posterior_predictive_check`](../../api/diagnostics/predictive_checks.md) computes the posterior predictive p-value $\Pr(T(y^{\text{rep}}) \ge T(y))$:

```python
from quivers.diagnostics import posterior_predictive_check

result_dict = posterior_predictive_check(
    tree, observed_name="y", statistic="mean",
)
print(result_dict)            # observed, posterior mean, p-value
```

```text
{'statistic': 'mean', 'observed': 1.6468594074249268, 'predictive_mean': 1.641527646780014, 'ppp': 0.445}
```

A p-value near 0.5 puts the observed statistic in the middle of the replicated distribution. Under this convention a value near 0 means the replicates rarely reach the observed statistic, so the model underpredicts it, and a value near 1 means it overpredicts it. The built-in statistics are `mean`, `median`, `sd`, `var`, `min`, `max`, `q05`, and `q95`, and a callable is also accepted. A regression whose chains agree can still predict too few extreme responses, which `min` and `max` would show.

## Comparing models

[Leave-one-out cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) estimates the expected log pointwise predictive density (elpd) of new data. PSIS-LOO approximates it from one posterior fit by Pareto smoothed importance sampling [@vehtari2017loo; @vehtari2024psis], which needs the log likelihood of each observation under each draw. Compute it for the linear fit, and fit a quadratic model to compare:

```python
from quivers.diagnostics import compare, loo

def normal_log_lik(samples, mu):
    return torch.distributions.Normal(mu, samples["sigma"][..., None]).log_prob(y_data)

s = result.samples                                    # each (chain, draw)
ll_linear = normal_log_lik(s, s["beta_0"][..., None] + s["beta_1"][..., None] * x_data)
tree_linear = to_datatree(
    result, observed_data={"y": y_data}, log_likelihood={"y": ll_linear},
)

QUADRATIC_SRC = REGRESSION_SRC.replace(
    "    let mu = beta_0 + beta_1 * x_design\n",
    "    sample beta_2 <- Normal(loc=0.0, scale=2.0)\n"
    "    let mu = beta_0 + beta_1 * x_design + beta_2 * x_design * x_design\n",
)
quadratic = loads(QUADRATIC_SRC).morphism
result_quad = mcmc.run(quadratic, torch.zeros(1, 1), {"x_design": x_data, "y": y_data})
q = result_quad.samples
mu_q = (q["beta_0"][..., None] + q["beta_1"][..., None] * x_data
        + q["beta_2"][..., None] * x_data**2)
tree_quad = to_datatree(
    result_quad, observed_data={"y": y_data},
    log_likelihood={"y": normal_log_lik(q, mu_q)},
)

linear_loo = loo(tree_linear, pointwise=True, var_name="y")
print(f"elpd_loo = {linear_loo.elpd:.2f} (se {linear_loo.se:.2f}), p_loo = {linear_loo.p:.2f}")

table = compare({"linear": tree_linear, "quadratic": tree_quad}, var_name="y", round_to=2)
print(table.to_string())
```

```text
elpd_loo = -15.87 (se 7.04), p_loo = 3.23
           rank  elpd_diff   dse  p_worse        diag_diff    diag_elpd     p   elpd    se  weight
quadratic     0       0.00  0.00      NaN                   1 k̂ > 0.57  4.16 -15.68  7.25    0.63
linear        1      -0.19  1.22     0.56  |elpd_diff| < 4  1 k̂ > 0.57  3.23 -15.87  7.04    0.37
```

[`compare`](../../api/diagnostics/comparison.md) delegates to [`arviz.compare`](https://python.arviz.org/en/stable/api/generated/arviz.compare.html), which runs PSIS-LOO on each fit and ranks the fits by `elpd`. `elpd_diff` is the difference from the fit ranked first, `dse` its standard error, and `p_worse` the approximate probability that the fit is worse than the top one. `diag_diff` flags a difference too small (below 4) for its standard error to be trusted, and `diag_elpd` counts the observations whose Pareto shape exceeds the reliability threshold. Here the difference is far smaller than its standard error, so the data do not distinguish the two models, as expected for data simulated from the linear one. `p` is $p_{\text{loo}}$, an estimate of the effective number of parameters. The `weight` column holds stacking weights by default [@yao2018stacking]: the weights of the combination of the models' predictive distributions that maximizes leave-one-out performance. `method=` selects another weighting, and `round_to="none"` keeps full precision when the weights will be used numerically.

`linear_loo.pareto_k` holds the estimated Pareto shape $k$ for each observation. With $S$ posterior draws, PSIS is reliable for observations with $k < \min(1 - 1/\log_{10} S, 0.7)$ [@vehtari2024psis], about 0.57 for the 200 draws here. ArviZ warns when an observation exceeds it, and the remedies are moment matching, $K$-fold cross-validation, or exact leave-one-out for the flagged observations. Neither a posterior predictive p-value nor a stacking weight is a probability that a model is true.

## Try this

- Run the eight-schools fit of [chapter 3](03-hierarchical.md) through this pipeline, and locate its divergences in parameter space with the variable `sample_stats/diverging` that the adapter writes from [`MCMCResult.sample_stats`](../../guides/inference-mcmc.md#checkpoints-budgets-and-sampler-statistics).
- Simulate data with a few outliers, fit a `StudentT` likelihood beside the Normal one, and compare the two.
- Select one chain of the posterior (`tree.posterior.sel(chain=0)`) and compare its summary with the other chain's.

## Next

[Chapter 9](09-indexed-data.md) declares finite index types and indexed families and runs total functions over them.
