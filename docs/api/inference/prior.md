# Prior sampling

`prior_draws` generates independent joint draws from a program's retained
conditional priors. It uses the same named draw laws and data dependencies as
fitting, including normalized measure-valued draw boundaries. Its dependency
projection omits downstream responses without evaluating their likelihoods.

```python
import torch
from quivers.dsl import loads
from quivers.inference import prior_draws

program = loads("""object R : Real 1
program model : R -> R
    sample scale <- LogNormal(loc=0.0, scale=0.3)
    sample b <- Normal(loc=0.0, scale=scale)
    observe y <- Normal(loc=b, scale=1.0)
    return b
export model
""").morphism
draws = prior_draws(program, torch.zeros(1, 1), num_samples=8, seed=12)
assert draws.samples["b"].shape == (1, 8)
```

The singleton leading axis of `PriorDraws` preserves the common output layout;
it is not a Markov chain. Prior views reuse predictive, likelihood, and summary
methods and do not report MCMC convergence statistics for independent samples.

An authored `score` is a global potential. Independent forward sampling cannot
normalize an arbitrary potential, so the ancestral method refuses it. Pass a
configured `MCMC` driver to `prior_draws(..., method=driver)` to perform
factor-conditioned prior-only inference; the returned `MCMCResult` then has
genuine chain provenance and the driver's own budgets.

::: quivers.inference.prior
