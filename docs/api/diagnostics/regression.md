# Regression variance ratios

The Bayesian coefficient of determination (R²) compares variation in conditional
response means with unexplained variation at each posterior draw. The pure
`bayesian_r2` function accepts moments from source models, formula models, or
another retained prediction procedure. It preserves chain, draw, and event
coordinates while reducing the observation-row axis.

Supply either `observed` responses or conditional `residual_variance`. With
observations, the unexplained component is the sample variance of observed
minus predicted responses, matching the
[released brms implementation](https://github.com/cran/brms/blob/b124949c14067aa658b176d26477ccf715881bf9/R/bayes_R2.R).
With conditional variances, their row average supplies the unexplained
component, as in the model-based method described in the
[brms development reference](https://paulbuerkner.com/brms/reference/bayes_R2.brmsfit.html).
The two definitions can differ at the same parameter draw.

<!-- python: run -->
```python
import torch
from quivers.diagnostics import bayesian_r2

means = torch.tensor([[[0., 1., 2.], [1., 2., 3.]]])
observed = torch.tensor([0., 2., 2.])
residual_ratio = bayesian_r2(means, observed=observed)
model_ratio = bayesian_r2(means, residual_variance=1.)
assert residual_ratio.shape == model_ratio.shape == (1, 2)
assert ((residual_ratio >= 0) & (residual_ratio <= 1)).all()
```

`expected` has shape `(chain, draw, row, *event)`; observations have shape
`(row, *event)`. Conditional variances broadcast to the expected-response
shape. Each event coordinate receives its own ratio; no independence assumption
is needed to report these marginal quantities. For unordered categories, a
numeric category code does not define a response metric. Use explicitly chosen
indicator coordinates or another justified measurement scale.

Prediction variance uses divisor `n - 1`. The residual definition uses the same
divisor, while conditional variances are averaged over rows. Finite Float64
arithmetic is applied after checking original integer capacity. Common scaling
avoids overflow and underflow in the variance ratio. Constant means have ratio
zero when unexplained variance is positive; zero total variation produces NaN.
No convergence statistic or correction for a variational approximation is
implied. Results use retained, detached values and do not execute a model.

::: quivers.diagnostics.regression
