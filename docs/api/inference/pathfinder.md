# Pathfinder

Pathfinder [@zhang2022pathfinder] is a variational method that follows an
[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) optimization path
of the unconstrained log joint, forms a
[normal](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
approximation at each iterate from the quasi-Newton inverse-Hessian estimate,
and keeps the one with the largest estimated
[evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound)
(ELBO). Multi-path Pathfinder runs several independent paths and resamples
their pooled draws with
[Pareto-smoothed importance weights](https://mc-stan.org/loo/reference/psis.html)
[@vehtari2024psis]. The [formal statement](../../semantics/programs.md#pathfinder-approximations)
gives the approximation, the ELBO estimate, and the resampling step.

[`Pathfinder`][quivers.inference.pathfinder.Pathfinder] takes CmdStan's
argument names. It scores the same Jacobian-corrected
[`PotentialFn`][quivers.inference.mcmc.kernel.PotentialFn] as the
Hamiltonian kernels, over the latents one recorded run reaches. Its result
[`PathfinderDraws`][quivers.inference.pathfinder.PathfinderDraws] keeps the
layout of independent draws, one pseudo-chain of `num_samples` draws, with
each draw's target and approximation log densities (Stan's `lp__` and
`lp_approx__`), the path it came from, each path's ELBO and stopping
reason, and the Pareto shape of the pooled importance ratios.
[`PathfinderMethod`][quivers.modeling.methods.PathfinderMethod] applies it to a
bound model, and `fit(..., algorithm="pathfinder")` to a formula.

<!-- python: run -->
```python
import torch
from quivers.inference import Pathfinder
from quivers.modeling import Model, PathfinderMethod

source = """object Row : FinSet _
object Out : Real 1
program regression(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample b <- Normal(loc=0.0, scale=1.0)
    let mu = b * x
    observe y : Row <- Normal(loc=mu, scale=1.0)
    return b
export regression
"""
x = torch.linspace(-1.0, 1.0, 12)
y = 0.6 * x + 0.1 * torch.sin(7.0 * x)
fitted = Model(source).fit(
    {"x": x, "y": y},
    method=PathfinderMethod(
        pathfinder=Pathfinder(num_paths=2, num_draws=200, num_psis_draws=400, seed=1)
    ),
)
draws = fitted.posterior
assert draws.samples["b"].shape == (1, 400)
assert draws.num_paths == 2 and draws.resampled
```

Cost per path: the L-BFGS evaluations, then for each of the `L` iterates
O(N J^2 + J^3) to factor the approximation by a thin QR and a Cholesky
decomposition, and `K` log-density evaluations at O(N J + J^2) per draw for
the ELBO, with `N` the unconstrained dimension, `J` the history size, and `K`
the ELBO draws. No `N`-by-`N` matrix is formed. Paths run one after another.

The line search enforces the strong Wolfe conditions with cubic
interpolation [@nocedal2006numerical]. Near-zero Wolfe constants make it an
exact line search, under which the approximation on a Gaussian target is the
target itself once the history size reaches the dimension.

::: quivers.inference.pathfinder
