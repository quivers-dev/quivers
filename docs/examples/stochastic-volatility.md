# Stochastic Volatility

## Overview

The stochastic-volatility model of Kim, Shephard, and Chib [@kim1998volatility] explains a series of returns whose spread changes over time. Each return is mean-zero Normal with standard deviation $\exp(h_t / 2)$, and the log-volatility $h_t$ follows a stationary first-order autoregression around a mean $\mu$. The volatility path is never observed, so the model's latent is the whole coupled trajectory $h_{-1}, h_0, \ldots, h_{T-1}$ together with the three parameters of its dynamics. This source states the model with a [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) whose cell draws each $h_t$ given the one before it, started from a stationary draw of its first state, the site `h0`, with `scan(sv_step, init=h0)`. The page checks the program's joint density against a hand-written one on both execution paths. It does not demonstrate recovery of the parameters from simulated returns; the section on recovery below says what was tried and why it does not pass.

## QVR source

```qvr
# Stochastic Volatility
#
# The canonical log-volatility model of Kim, Shephard, and Chib. Returns
# are mean-zero Normal with scale exp(h_t / 2), and the log-volatility
# h_t is a stationary AR(1) process around mu. The trajectory starts at
# h_0, drawn from the chain's stationary distribution, and every later
# h_t is drawn given the one before it, so the program infers one
# coupled latent trajectory.
#
# Generative structure:
#
#   mu       ~ Normal(0, 10)                          log-vol mean
#   phi      ~ Uniform(-1, 1)                         AR(1) coefficient
#   sigma_h  ~ HalfCauchy(2.5)                        log-vol noise
#   h_0      ~ Normal(mu, sigma_h / sqrt(1 - phi^2))  stationary start
#   h_t      ~ Normal(mu + phi * (h_{t-1} - mu), sigma_h)
#   r_t      ~ Normal(0, exp(h_t / 2))                observed return
#
# scan(sv_step, init=h0) runs the step at every position of the input
# sequence (one entry per return; the entries themselves are unused),
# starting from h_0. The step draws h at every position, addressed h@0,
# h@1, ..., and observes the return r there.
#
# Reference: [Kim, Shephard, and Chib 1998](https://doi.org/10.1111/1467-937X.00050).

object Val : Real 1

program sv_step(u_t, h_prev) : Val * Val -> Val
    let h_mean = mu + phi * (h_prev - mu)
    sample h <- Normal(loc=h_mean, scale=sigma_h)
    let scale = exp(0.5 * h)
    observe r <- Normal(loc=0.0, scale=scale)
    return h

define run_all = scan(sv_step, init=h0)

program stochastic_volatility : Val -> Val
    sample mu <- Normal(loc=0.0, scale=10.0)
    sample phi <- Uniform(low=-1.0, high=1.0)
    sample sigma_h <- HalfCauchy(scale=2.5)
    let stationary_scale = sigma_h / sqrt(1.0 - phi * phi)
    sample h0 <- Normal(loc=mu, scale=stationary_scale)
    sample h_last <- run_all
    return h_last

export stochastic_volatility
```

## Understanding the model

The program scores

$$
\begin{aligned}
h_{-1} &\sim \mathcal{N}\bigl(\mu, \sigma_h / \sqrt{1 - \phi^2}\bigr), \\
h_t &\sim \mathcal{N}\bigl(\mu + \phi (h_{t-1} - \mu), \sigma_h\bigr), & t &= 0, \ldots, T - 1, \\
r_t &\sim \mathcal{N}\bigl(0, \exp(h_t / 2)\bigr),
\end{aligned}
$$

with $h_{-1}$ the start, the site `h0`, and $h_t$ the site `h@t`, under priors $\mu \sim \mathcal{N}(0, 10)$, $\phi \sim \mathrm{Uniform}(-1, 1)$, and $\sigma_h \sim \mathrm{HalfCauchy}(2.5)$. A volatile stretch has large returns even though their expected value is zero: raising $h_t$ widens the return distribution at $t$ without moving its center, and $\phi$ near one makes such stretches persist. The start is drawn from the autoregression's stationary distribution, whose standard deviation is $\sigma_h / \sqrt{1 - \phi^2}$, so the series has no transient at its beginning [@kim1998volatility].

Each input row is one run: a series of $T$ positions whose entries the cell ignores (`u_t` is unused; it sets the length), with the returns observed at the sites `r@0`, `r@1`, and so on and the log-volatilities drawn at `h@0`, `h@1`, and so on.

## Walkthrough

- `program sv_step(u_t, h_prev)` is one position. It forms the autoregressive mean from `mu`, `phi`, and the previous state, draws `h` around it at scale `sigma_h`, and observes the return `r` at scale `exp(0.5 * h)`. The cell reads `mu`, `phi`, and `sigma_h` from the scanning program, so every position shares them.
- `define run_all = scan(sv_step, init=h0)` starts the chain at the program's draw `h0` rather than at zero.
- `program stochastic_volatility` draws the three parameters, then `h0` from the stationary distribution, then the path. `sqrt(1.0 - phi * phi)` is real because the `Uniform(-1, 1)` prior keeps $|\phi| < 1$.
- No transpile target lowers a scan, so every target refuses this program with `scan:no-lowering`. The program runs on the PyTorch runtime and the reference machine.

## Try it

The checks below come from a verification script whose tolerances were fixed before it was first run.

### The log joint against a hand-written recursion

At a traced draw of every latent on a twelve-position series, the compiled runtime's log joint must equal a float64 loop over positions to a relative 1e-5. An oracle that starts the trajectory from zero instead of from `h0` must disagree by more than a relative 1e-3, which shows the comparison can see where the chain starts. The reference machine, given fixed values of every latent, must equal the same oracle to a relative 1e-9.

```python
import math

import torch
import torch.distributions as D

from quivers.dsl import load
from quivers.inference import trace

SOURCE = "docs/examples/source/stochastic_volatility.qvr"


def oracle(values, r, start_from_zero=False):
    """Kim, Shephard, and Chib's joint density, by a loop over positions."""
    mu, phi, sigma, h0 = (values[name] for name in ("mu", "phi", "sigma_h", "h0"))
    total = (
        D.Normal(0.0, 10.0).log_prob(mu)
        + D.Uniform(-1.0, 1.0).log_prob(phi)
        + D.HalfCauchy(2.5).log_prob(sigma)
        + D.Normal(mu, sigma / torch.sqrt(1.0 - phi**2)).log_prob(h0)
    )
    previous = torch.zeros_like(h0) if start_from_zero else h0
    for t in range(r.shape[0]):
        h = values[f"h@{t}"]
        total = total + D.Normal(mu + phi * (previous - mu), sigma).log_prob(h)
        total = total + D.Normal(0.0, torch.exp(0.5 * h)).log_prob(r[t])
        previous = h
    return total


program = load(SOURCE)
steps = 12
generator = torch.Generator().manual_seed(0)
r = torch.randn(1, steps, 1, generator=generator)
torch.manual_seed(1)
recorded = trace(
    program.morphism,
    torch.zeros(1, steps, 1),
    observations={f"r@{t}": r[:, t] for t in range(steps)},
)
names = ["mu", "phi", "sigma_h", "h0", *(f"h@{t}" for t in range(steps))]
values = {name: recorded.sites[name].value.reshape(()).double() for name in names}
expected = float(oracle(values, r.reshape(steps).double()))
wrong = float(oracle(values, r.reshape(steps).double(), start_from_zero=True))
got = float(recorded.log_joint)
print(f"compiled runtime {got:.6f}, oracle {expected:.6f}, oracle started from zero {wrong:.6f}")
assert math.isclose(got, expected, rel_tol=1e-5)
assert not math.isclose(got, wrong, rel_tol=1e-3)

fixed = {
    "mu": torch.tensor(-0.5, dtype=torch.float64),
    "phi": torch.tensor(0.85, dtype=torch.float64),
    "sigma_h": torch.tensor(0.3, dtype=torch.float64),
    "h0": torch.tensor(-0.2, dtype=torch.float64),
}
for t in range(steps):
    fixed[f"h@{t}"] = 0.5 * torch.randn((), generator=generator, dtype=torch.float64)
returns = torch.randn(steps, generator=generator, dtype=torch.float64)
sites = {name: float(value) for name, value in fixed.items()}
for t in range(steps):
    sites[f"h@{t}"] = (float(fixed[f"h@{t}"]),)
run = program.run(
    "stochastic_volatility",
    data={"val": tuple((0.0,) for _ in range(steps)), "r": tuple((float(v),) for v in returns)},
    sites=sites,
    seed=0,
)
expected = float(oracle(fixed, returns))
print(f"reference machine {run.log_joint:.9f}, oracle {expected:.9f}")
assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
```

### Recovery: what was tried, and what is not demonstrated

The recovery check simulated $T = 100$ returns from known parameters ($\mu = -1$, $\phi = 0.9$, $\sigma_h = 0.4$) and fitted the program by SVI with a full-rank Gaussian guide over the unconstrained latents ([`AutoMultivariateNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoMultivariateNormalGuide), 2000 Adam steps at learning rate 0.01, 400 posterior draws). Its tolerances were fixed in advance: $|\mathrm{E}[\mu] - \mu| < 1.0$, $|\mathrm{E}[\phi] - \phi| < 0.2$, $\mathrm{E}[\sigma_h] / \sigma_h \in [0.4, 2.5]$, and a correlation above 0.6 between the posterior-mean volatility path and the true one. The fit recovers $\mu$ (posterior mean $-1.14$) but not the dynamics: it estimates $\phi$ at $0.23$ and $\sigma_h$ at $1.47$, $3.7$ times the truth, so two of the four checks fail. The path correlation was printed as $0.6$ to one decimal place; the script stopped at the first failed assertion, on $\phi$, before comparing it with its threshold. A mean-field guide, tried first, did worse, estimating $\phi$ at 0.12, as expected of a guide that cannot represent the posterior coupling between $\phi$ and the path. The tolerances were not widened, and this page makes no recovery claim.

What the failure does and does not show is limited. A Gaussian approximation in the unconstrained space may simply be a poor approximation to this posterior, which is strongly curved in $(\phi, \sigma_h)$; a short series also identifies $\phi$ weakly. The check that would separate the two is NUTS on the same data, or the same fit on a much longer series, and neither runs within this page's budget: one gradient of the log joint at $T = 100$ takes about a second on the compiled runtime, which evaluates the scan position by position, so a NUTS run of a few hundred iterations would take hours.

## Categorical perspective

The cell `sv_step` is a Kleisli morphism $\mathrm{Val} \otimes \mathrm{Val} \to \mathrm{Val}$ of the [Giry monad](https://doi.org/10.1007/BFb0092872) [@giry1982categorical], weighted at each position by the likelihood of the return it observes, and `scan(sv_step, init=h0)` is its composite along the series in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category), precomposed with the stationary draw of the start. The joint the checks above compare is the density of that composite with respect to Lebesgue measure on the parameters and the path.

## Sources and further reading

- *Stochastic volatility: Likelihood inference and comparison with ARCH models* [@kim1998volatility].
- *A categorical approach to probability theory* [@giry1982categorical].
