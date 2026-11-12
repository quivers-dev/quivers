# Inference benchmark results

Five posterior inference algorithms of [`quivers.inference`](../api/index.md#inference-package) are measured on how well they recover known posterior moments on a deterministic suite of synthetic problems. `tests/benchmarks/runner.py` regenerates the grid from the seeded data factories and analytical references in `tests/benchmarks/`. Throughout, $\mathcal{N}(m, s)$ is the Normal distribution with mean $m$ and standard deviation $s$.

## How to read a result

A cell compares one estimated posterior summary with one reference summary.
For instance, `0.0398 / 0.05` means the absolute error in the chosen mean is
`0.0398`, within the allowed error `0.05`. It says nothing directly about
posterior variance, tails, correlations, or a different latent site. A variational
[guide](../tutorials/qvr/06-inference-zoo.md#step-2-which-variational-guide), the family of
approximate posteriors fitted by stochastic variational inference (SVI), can recover a mean while underestimating uncertainty.

The *capture* rows reverse the usual meaning of PASS: they pass when a known failure remains
detectable, so their metric is read before their status. Throughput has different units across
families, since one SVI update and one Markov chain Monte Carlo (MCMC) draw are different amounts of statistical work. The
tables compare the five algorithms at the listed budgets, and changing the number of updates,
draws, or particles changes both cost and accuracy.

## What the suite tests

Every benchmark is an `(algorithm, problem)` cell. A *problem* fixes:

1. A generative model written in QVR and loaded from `tests/benchmarks/models/*.qvr`.
2. A deterministic data generator (fixed `torch.manual_seed`) that produces the observations the model is conditioned on.
3. A *reference posterior* moment for one latent site, computed analytically (conjugate problems), by quadrature on a dense grid (constrained-support problems), or by a long cached NUTS run (Eight Schools).
4. A scalar metric (almost always $|\mathbb{E}_q[\cdot] - \mathbb{E}_{\text{ref}}[\cdot]|$) and a tolerance.

A *cell* runs the algorithm on the problem, draws posterior samples for the target site, and compares the recovered moment against the reference.

Throughput is reported as SVI iterations per second for the variational guides and as posterior draws per second (summed across chains) for the MCMC kernels.

## Cell statuses

- **PASS**: recovered moment is within tolerance of the reference.
- **FAIL**: algorithm runs cleanly but the moment is outside tolerance.
- **ERROR**: algorithm raised during execution (NaN gradient, support-boundary explosion, divergent trajectory, etc.).
- *capture* problems invert the convention: PASS means the metric *exceeds* the tolerance, confirming a documented failure mode.

Determinism: every cell calls `torch.manual_seed(0)` before constructing the problem, so the same `(algorithm, problem)` pair reproduces across runs given fixed PyTorch and NumPy versions.

## Algorithms

All algorithms are evaluated on every problem with hyperparameters held fixed across problems. The grid thus measures performance under that common configuration, and tuning a method for one problem can change its result. The variational guides are fitted by SVI [@blei2017variational], and the samplers are Hamiltonian Monte Carlo (HMC) [@neal2011mcmc] and the No-U-Turn Sampler (NUTS) [@hoffman2014nuts].

| Algorithm | Family | Key hyperparameters |
|---|---|---|
| `AutoNormal` | Mean-field SVI, factorized diagonal Normal in unconstrained space | Adam, lr=0.05, 800 steps (1500 for positive-support sites), 1500 posterior draws |
| `AutoMVN` | Full-covariance SVI, single MVN in unconstrained space | Adam, lr=0.05, 800 steps (1500 for positive-support sites), `init_scale=0.3`, 1500 draws |
| `AutoLaplace` | MAP plus a Gaussian centered at the mode with Hessian covariance [@tierney1986laplace] | Adam, lr=0.05, 500 steps, 1500 draws |
| `HMC` | Hamiltonian Monte Carlo with fixed integrator length | `step_size=0.1` (adapted), `num_steps=10`, diagonal mass matrix (adapted), 200 warmup, 400 samples, 2 chains |
| `NUTS` | No-U-Turn HMC | `target_accept=0.8`, `max_tree_depth=8`, diagonal mass matrix, 200 warmup, 400 samples, 2 chains |

Variational guides operate in unconstrained space via the bijector attached to each latent's support, so positive-support and bounded-support sites are exercised through `exp` / `softplus` / `sigmoid` transforms rather than through constrained Gaussian families.

## Tier 1: conjugate posteriors

These five problems have closed-form posteriors and set a floor: every algorithm should match the analytical moment to within a tight tolerance.

### Beta-Bernoulli

**Model.** Conjugate Beta prior on a Bernoulli rate:

$$
\theta \sim \mathrm{Beta}(2, 2), \qquad
y_i \mid \theta \sim \mathrm{Bernoulli}(\theta), \quad i = 1, \dots, 50.
$$

**Data.** $N = 50$ Bernoulli draws at $\theta^\star = 0.7$.

**Reference.** Conjugacy gives $\theta \mid y \sim \mathrm{Beta}\bigl(\alpha_0 + \sum_i y_i,\ \beta_0 + N - \sum_i y_i\bigr)$ with closed-form mean $\alpha / (\alpha + \beta)$.

**Metric.** `|E[theta]_q - E[theta]_true|`, tolerance `0.05`.

### Normal-Normal

**Model.** Conjugate Normal prior on a Normal mean with known variance:

$$
\mu \sim \mathcal{N}(0, 1), \qquad
y_i \mid \mu \sim \mathcal{N}(\mu, 1), \quad i = 1, \dots, 30.
$$

**Data.** $N = 30$ Normal draws at $\mu^\star = 1.5$, $\sigma = 1$.

**Reference.** Posterior precision $\tau_N = \tau_0 + N / \sigma^2$ gives a Normal posterior with mean $(\tau_0 \mu_0 + N \bar{y} / \sigma^2) / \tau_N$.

**Metric.** `|E[mu]_q - E[mu]_true|`, tolerance `0.15`.

### Normal-Inverse-Gamma

**Model.** Joint conjugate prior on unknown mean *and* variance:

$$
\sigma^2 \sim \mathrm{InverseGamma}(3, 2), \qquad
\mu \mid \sigma^2 \sim \mathcal{N}(0, \sigma), \qquad
y_i \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma), \quad i = 1, \dots, 60.
$$

**Data.** $N = 60$ Normal draws at $\mu^\star = 0.3$, $\sigma^{2\star} = 1.5$.

**Reference.** NIG posterior updates [@murphy2007gaussian] (Section 5) give marginal mean $\mu_N = (\kappa_0 \mu_0 + N \bar{y}) / (\kappa_0 + N)$.

The problem tests guides on two latents with different supports: the unconstrained $\mu$ and the positive $\sigma^2$, whose bijector is $\exp$ or softplus.

**Metric.** `|E[mu]_q - E[mu]_true|`, tolerance `0.2`.

### Gamma-Exponential

**Model.** Conjugate Gamma prior on an Exponential rate:

$$
r \sim \mathrm{Gamma}(2, 1), \qquad
y_i \mid r \sim \mathrm{Exponential}(r), \quad i = 1, \dots, 80.
$$

**Data.** $N = 80$ Exponential draws at $r^\star = 2$.

**Reference.** $r \mid y \sim \mathrm{Gamma}\bigl(a_0 + N,\ b_0 + \sum_i y_i\bigr)$, with mean $a / b$.

**Metric.** `|E[rate]_q - E[rate]_true|`, tolerance `0.3`.

### Bayesian linear regression

**Model.** Linear regression with two parameters, a design drawn iid from $\mathcal{N}(0, 1)$, and known observation noise:

$$
a, b \sim \mathcal{N}(0, 1), \qquad
x_i \sim \mathcal{N}(0, 1), \qquad
y_i \mid a, b \sim \mathcal{N}(a + b x_i, \sigma), \quad i = 1, \dots, 60,
$$

with $\sigma = 0.3$, $a^\star = 0.7$, $b^\star = -0.5$.

**Reference.** Closed-form Gaussian posterior with precision $I + X^\top X / \sigma^2$ and mean $\Sigma X^\top y / \sigma^2$.

**Metric.** `|E[a]_q - E[a]_true|`, tolerance `0.1`.

### Results

Posterior accuracy (metric / tolerance):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---|---|---|---|---|
| Beta-Bernoulli | **PASS** `0.0398 / 0.05` | **PASS** `0.0402 / 0.05` | **PASS** `0.00926 / 0.05` | **PASS** `0.000594 / 0.05` | **PASS** `0.00157 / 0.05` |
| Normal-Normal | **PASS** `0.123 / 0.15` | **PASS** `0.124 / 0.15` | **PASS** `4.77e-07 / 0.15` | **PASS** `0.000607 / 0.15` | **PASS** `0.0225 / 0.15` |
| Normal-Inverse-Gamma | **PASS** `0.0345 / 0.2` | **PASS** `0.0298 / 0.2` | **PASS** `5.96e-08 / 0.2` | **PASS** `0.00715 / 0.2` | **PASS** `0.0068 / 0.2` |
| Gamma-Exponential | **PASS** `0.0513 / 0.3` | **PASS** `0.057 / 0.3` | **PASS** `0.0249 / 0.3` | **PASS** `0.00806 / 0.3` | **PASS** `0.0163 / 0.3` |
| Bayesian linear regression | **PASS** `0.0113 / 0.1` | **PASS** `0.00401 / 0.1` | **PASS** `1.79e-07 / 0.1` | **PASS** `0.000134 / 0.1` | **PASS** `0.00127 / 0.1` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| Beta-Bernoulli | 1149.7 | 868.0 | 1863.5 | 114.2 | 167.9 |
| Normal-Normal | 2013.3 | 1395.7 | 3451.5 | 236.8 | 256.7 |
| Normal-Inverse-Gamma | 883.0 | 571.8 | 1574.8 | 102.6 | 51.0 |
| Gamma-Exponential | 1803.4 | 1153.3 | 3276.4 | 233.1 | 180.0 |
| Bayesian linear regression | 1222.2 | 710.6 | 2305.4 | 163.5 | 40.8 |

## Tier 2: hierarchical posteriors

The Eight Schools problem [@rubin1981parallel] in both parameterizations tests how each algorithm handles the funnel geometry that arises when the group scale $\tau$ shrinks toward zero.

### Eight Schools (centered)

**Model.**

$$
\mu \sim \mathcal{N}(0, 10), \qquad
\tau \sim \mathrm{HalfCauchy}(5), \qquad
\theta_j \mid \mu, \tau \sim \mathcal{N}(\mu, \tau), \qquad
y_j \mid \theta_j \sim \mathcal{N}(\theta_j, 12),
$$

for $j = 1, \dots, 8$ on Rubin's effect sizes $y = (28, 8, -3, 7, -1, 1, 18, 12)$.

**Reference.** Cached NUTS moments (4 chains, 5000 post-warmup draws): $\mathbb{E}[\mu] \approx 5.4$, posterior standard deviation $\approx 4$.

The tolerance is three reference standard deviations, a loose target that reflects how hard the funnel geometry is for variational inference.

**Metric.** `|E[mu]_q - mu_ref|`, tolerance `12`.

### Eight Schools (non-centered)

**Model.** Same priors as the centered model, with the group-level draws reparameterized:

$$
\eta_j \sim \mathcal{N}(0, 1), \qquad
\theta_j = \mu + \tau \eta_j,
$$

decoupling $\tau$ from $\theta_j$ and eliminating the funnel in the prior.

**Reference.** Same cached NUTS moments as the centered model.

The tolerance is two reference standard deviations, tighter than for the centered model, since the non-centered parameterization removes the funnel.

**Metric.** `|E[mu]_q - mu_ref|`, tolerance `8`.

### Results

Posterior accuracy (metric / tolerance):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---|---|---|---|---|
| Eight Schools (centered) | **PASS** `5.4 / 12` | **PASS** `5.51 / 12` | **PASS** `5.4 / 12` | **PASS** `4.38 / 12` | **PASS** `5.75 / 12` |
| Eight Schools (non-centered) | **PASS** `0.891 / 8` | **PASS** `1.06 / 8` | **PASS** `2.01 / 8` | **PASS** `1.74 / 8` | **PASS** `1.45 / 8` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| Eight Schools (centered) | 776.5 | 559.8 | 1498.0 | 113.0 | 28.0 |
| Eight Schools (non-centered) | 729.2 | 572.4 | 1448.6 | 110.3 | 34.1 |

## Tier 3: hard posterior geometry

These problems expose known failure modes of mean-field variational inference and of HMC under poor preconditioning.

### Correlated regression

**Model.** Linear regression as in Tier 1, but with a near-constant design:

$$
a, b \sim \mathcal{N}(0, 1), \qquad
x_i = \rho + (1 - \rho) z_i, \quad z_i \sim \mathcal{N}(0, 1), \qquad
y_i \mid a, b \sim \mathcal{N}(a + b x_i, 0.5),
$$

with $\rho = 0.95$ and $N = 50$.

**Reference.** Closed-form Gaussian posterior in which $a$ and $b$ have a correlation near $-0.99$.

The mean-field guide ignores this correlation. The first-moment metric below still passes, since the underfit shows in the second moments.

**Metric.** `|E[a]_q - E[a]_true|`, tolerance `0.2`.

### Neal's funnel (under-estimation capture) *(capture)*

**Model.** Neal's funnel:

$$
v \sim \mathcal{N}(0, 3), \qquad
x_i \mid v \sim \mathcal{N}(0, e^{v / 2}), \quad i = 1, \dots, 9.
$$

**Data.** Condition on $x_i = 0$ (inference target is $p(v \mid x = 0)$).

**Reference.** The log-likelihood is linear in $v$: $\log p(x_i = 0 \mid v) = -\tfrac{1}{2}\log(2\pi) - v / 2$, so the conditional posterior is Gaussian with mean $-9 N / 2 = -40.5$ and variance $9$ at $N = 9$. The *joint* posterior over $(v, x)$ remains funnel-shaped; only the conditional given $x = 0$ is tractable.

**Capture semantics.** All five algorithms underestimate the magnitude of $v$ [@neal2003slice]. The posterior mean lies forty units below the prior mean, where the scale $e^{v/2}$ of each $x_i$ is about $e^{-20}$. The compiled `Normal` family floors its scale at $10^{-7}$, which $e^{v/2}$ reaches at $v \approx -32.2$, so below that point the compiled log density stops increasing in $v$ and the target the samplers see has its mode near $-32.2$. PASS means the metric *exceeds* the tolerance, confirming the underfit.

**Metric.** `|E[v]_q - E[v]_true|`, tolerance `20.25`.

### Ill-conditioned product Gaussian

**Model.** Five-dimensional product Gaussian with prior scales spanning four orders of magnitude and a fixed observation noise:

$$
x_d \sim \mathcal{N}(0, \sigma_d^{\text{prior}}), \qquad
y_d \mid x_d \sim \mathcal{N}(x_d, 0.1), \qquad d = 1, \dots, 5,
$$

with $\sigma^{\text{prior}} = (100, 10, 1, 0.1, 0.01)$.

**Reference.** A Gaussian in each dimension: $x_d \mid y_d \sim \mathcal{N}\bigl(y_d / (1 + (0.1 / \sigma_d)^2),\ (1 / \sigma_d^2 + 1 / 0.01)^{-1/2}\bigr)$, writing $\sigma_d$ for $\sigma_d^{\text{prior}}$.

The metric tracks the middle scale $x_3$, where the diagonal mass matrix is roughly correct but the gradient signal is small beside that of the dimensions with larger scales.

**Metric.** `|E[x_3]_q - E[x_3]_true|`, tolerance `0.3`.

### Results

Posterior accuracy (metric / tolerance):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---|---|---|---|---|
| Correlated regression | **PASS** `0.0365 / 0.2` | **PASS** `0.0308 / 0.2` | **PASS** `7.41e-05 / 0.2` | **PASS** `0.0512 / 0.2` | **PASS** `0.0569 / 0.2` |
| Neal's funnel (under-estimation capture) *(capture)* | **PASS** `40.6 / 20.2` | **PASS** `40.6 / 20.2` | **PASS** `40.5 / 20.2` | **PASS** `42 / 20.2` | **PASS** `40.9 / 20.2` |
| Ill-conditioned product Gaussian | **PASS** `0.00643 / 0.3` | **PASS** `0.00333 / 0.3` | **PASS** `7.15e-07 / 0.3` | **PASS** `0.086 / 0.3` | **PASS** `0.0424 / 0.3` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| Correlated regression | 1150.7 | 681.6 | 2314.7 | 167.3 | 67.1 |
| Neal's funnel (under-estimation capture) *(capture)* | 1795.3 | 1279.6 | 2807.7 | 281.0 | 1027.6 |
| Ill-conditioned product Gaussian | 482.0 | 461.1 | 876.7 | 68.3 | 54.5 |

## Tier 4: constrained support

These latents lie on a half-line or in a bounded interval. References come from quadrature on a dense grid, and variational guides must pass through a nonlinear bijector to reach the constrained scale.

### HalfNormal scale

**Model.**

$$
\sigma \sim \mathrm{HalfNormal}(2), \qquad
y_i \mid \sigma \sim \mathcal{N}(0, \sigma), \quad i = 1, \dots, 80.
$$

**Reference.** No conjugate form. Integrate

$$
p(\sigma \mid y) \propto \exp(-\sigma^2 / 8) \cdot \sigma^{-N} \cdot \exp\bigl(-\tfrac{1}{2 \sigma^2} \sum_i y_i^2\bigr)
$$

on a 4096-point grid in $[0.05, 6]$ for the reference moments.

**Metric.** `|E[sigma]_q - E[sigma]_true|`, tolerance `0.15`.

### TruncatedNormal recovery

**Model.**

$$
\mu \sim \mathrm{Uniform}(0, 1), \qquad
y_i \mid \mu \sim \mathrm{TruncatedNormal}(\mu, 0.2, 0, 1), \quad i = 1, \dots, 60.
$$

**Reference.** Evaluate the truncated-Normal log-likelihood on a 4096-point $\mu$-grid in $(0, 1)$ with stable log-CDF differences for the truncation constant; normalize for the posterior moments.

**Metric.** `|E[mu]_q - E[mu]_true|`, tolerance `0.05`.

### Results

Posterior accuracy (metric / tolerance):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---|---|---|---|---|
| HalfNormal scale | **PASS** `0.0265 / 0.15` | **PASS** `0.0458 / 0.15` | **PASS** `0.0247 / 0.15` | **PASS** `0.00324 / 0.15` | **PASS** `0.0124 / 0.15` |
| TruncatedNormal recovery | **PASS** `0.0318 / 0.05` | **PASS** `0.0309 / 0.05` | **PASS** `0.000249 / 0.05` | **PASS** `0.00095 / 0.05` | **PASS** `0.00257 / 0.05` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| HalfNormal scale | 1561.4 | 1045.9 | 2725.6 | 190.0 | 90.7 |
| TruncatedNormal recovery | 1355.7 | 1038.6 | 2152.0 | 145.2 | 105.1 |

## Reproducing the grid

```bash
python -m tests.benchmarks.runner
```

The runner accepts `--algorithms` and `--problems` flags for partial runs and writes the regenerated tables to `docs/developer/inference-benchmarks.md` by default. `tests/benchmarks/runner.py` defines the cells and `tests/benchmarks/references.py` the reference posteriors.
