# Inference benchmark results

Five posterior inference algorithms of [`quivers.inference`](../api/index.md#inference-package) are measured on how well they recover known posterior moments on a deterministic suite of synthetic problems. `tests/benchmarks/runner.py` regenerates the grid from the seeded data factories and analytical references in `tests/benchmarks/`. Throughout, $\mathcal{N}(m, s)$ is the Normal distribution with mean $m$ and standard deviation $s$.

## How to read a result

A cell compares one estimated posterior summary with one reference summary.
For instance, `0.0398 / 0.05` means the absolute error in the chosen mean is
`0.0398`, within the allowed error `0.05`. It says nothing directly about
posterior variance, tails, correlations, or a different latent site. A variational
[guide](../tutorials/qvr/06-inference-zoo.md#step-2-which-variational-guide), the family of
approximate posteriors fitted by stochastic variational inference (SVI), can recover a mean while underestimating uncertainty.

Throughput has different units across
families, since one SVI update and one Markov chain Monte Carlo (MCMC) draw are different amounts of statistical work. The
tables compare the five algorithms at the listed budgets, and changing the number of updates,
draws, or particles changes both cost and accuracy.

## What the suite tests

Every benchmark is an `(algorithm, problem)` cell. A *problem* fixes:

1. A generative model written in QVR and loaded from `tests/benchmarks/models/*.qvr`.
2. A deterministic data generator (fixed `torch.manual_seed`) that produces the observations the model is conditioned on.
3. A *reference posterior* moment for one latent site, computed analytically (conjugate problems), by quadrature on a dense grid (problems with constrained support), or by a long cached run of the No-U-Turn Sampler (NUTS) (Eight Schools).
4. A scalar metric (almost always $|\mathbb{E}_q[\cdot] - \mathbb{E}_{\text{ref}}[\cdot]|$) and a tolerance.

A *cell* runs the algorithm on the problem, draws posterior samples for the target site, and compares the recovered moment against the reference.

Throughput is reported as SVI iterations per second for the variational guides and as posterior draws per second (summed across chains) for the MCMC kernels.

## Cell statuses

- **PASS**: recovered moment is within tolerance of the reference.
- **FAIL**: algorithm runs cleanly but the moment is outside tolerance.
- **ERROR**: algorithm raised during execution (NaN gradient, a value that blows up at the boundary of the support, divergent trajectory, etc.).

Determinism: every cell calls `torch.manual_seed(0)` before constructing the problem, so the same `(algorithm, problem)` pair reproduces across runs given fixed PyTorch and NumPy versions.

## Algorithms

All algorithms are evaluated on every problem with hyperparameters held fixed across problems. The grid thus measures performance under that common configuration, and tuning a method for one problem can change its result. The variational guides are fitted by SVI [@blei2017variational], and the samplers are Hamiltonian Monte Carlo (HMC) [@neal2011mcmc] and the No-U-Turn Sampler (NUTS) [@hoffman2014nuts].

| Algorithm | Family | Key hyperparameters |
|---|---|---|
| `AutoNormal` | Mean-field SVI, factorized diagonal Normal in unconstrained space | Adam, lr=0.05, 800 steps (1500 for sites with positive support), 1500 posterior draws |
| `AutoMVN` | Full-covariance SVI, single MVN in unconstrained space | Adam, lr=0.05, 800 steps (1500 for sites with positive support), `init_scale=0.3`, 1500 draws |
| `AutoLaplace` | MAP plus a Gaussian centered at the mode with Hessian covariance [@tierney1986laplace] | Adam, lr=0.05, 500 steps, 1500 draws |
| `HMC` | Hamiltonian Monte Carlo with fixed integrator length | `step_size=0.1` (adapted), `num_steps=10`, diagonal mass matrix (adapted), 200 warmup, 400 samples, 2 chains |
| `NUTS` | No-U-Turn HMC | `target_accept=0.8`, `max_tree_depth=8`, diagonal mass matrix, 200 warmup, 400 samples, 2 chains |

Variational guides operate in unconstrained space via the bijector attached to each latent's support, so sites with positive and with bounded support are exercised through `exp` / `softplus` / `sigmoid` transforms rather than through constrained Gaussian families.

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
| Beta-Bernoulli | **PASS** `0.0398 / 0.05` | **PASS** `0.0402 / 0.05` | **PASS** `0.00926 / 0.05` | **PASS** `0.00269 / 0.05` | **PASS** `0.000477 / 0.05` |
| Normal-Normal | **PASS** `0.123 / 0.15` | **PASS** `0.124 / 0.15` | **PASS** `4.77e-07 / 0.15` | **PASS** `0.000519 / 0.15` | **PASS** `0.00767 / 0.15` |
| Normal-Inverse-Gamma | **PASS** `0.0345 / 0.2` | **PASS** `0.0298 / 0.2` | **PASS** `5.96e-08 / 0.2` | **PASS** `0.00594 / 0.2` | **PASS** `0.0213 / 0.2` |
| Gamma-Exponential | **PASS** `0.0513 / 0.3` | **PASS** `0.057 / 0.3` | **PASS** `0.0249 / 0.3` | **PASS** `0.0016 / 0.3` | **PASS** `0.0212 / 0.3` |
| Bayesian linear regression | **PASS** `0.0113 / 0.1` | **PASS** `0.00401 / 0.1` | **PASS** `1.79e-07 / 0.1` | **PASS** `0.000189 / 0.1` | **PASS** `0.003 / 0.1` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| Beta-Bernoulli | 453.4 | 411.3 | 598.2 | 89.6 | 183.5 |
| Normal-Normal | 587.2 | 501.0 | 592.3 | 138.6 | 237.8 |
| Normal-Inverse-Gamma | 333.1 | 269.8 | 419.2 | 66.6 | 44.4 |
| Gamma-Exponential | 570.1 | 441.7 | 758.5 | 172.3 | 122.7 |
| Bayesian linear regression | 383.5 | 303.6 | 498.8 | 96.1 | 35.9 |

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
| Eight Schools (centered) | **PASS** `4.51 / 12` | **PASS** `3.81 / 12` | **PASS** `4.5 / 12` | **PASS** `1.4 / 12` | **PASS** `1.58 / 12` |
| Eight Schools (non-centered) | **PASS** `2.01 / 8` | **PASS** `1.88 / 8` | **PASS** `2.01 / 8` | **PASS** `1.72 / 8` | **PASS** `1.34 / 8` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| Eight Schools (centered) | 297.0 | 256.2 | 434.3 | 81.3 | 18.3 |
| Eight Schools (non-centered) | 299.4 | 256.2 | 383.5 | 67.8 | 37.4 |

## Tier 3: hard posterior geometry

Correlated and ill-conditioned Gaussian targets test covariance approximation and preconditioning. The conditional from Neal's funnel tests recovery of a Gaussian mean far from initialization.

### Correlated regression

**Model.** Linear regression as in Tier 1, but with a near-constant design:

$$
a, b \sim \mathcal{N}(0, 1), \qquad
x_i = \rho + (1 - \rho) z_i, \quad z_i \sim \mathcal{N}(0, 1), \qquad
y_i \mid a, b \sim \mathcal{N}(a + b x_i, 0.5),
$$

with $\rho = 0.95$ and $N = 50$.

**Reference.** Closed-form Gaussian posterior in which $a$ and $b$ have a correlation near $-0.99$.

The mean-field guide ignores this correlation. The metric on the first moment below still passes, since the underfit shows in the second moments.

**Metric.** `|E[a]_q - E[a]_true|`, tolerance `0.2`.

### Neal's funnel conditional

**Model.** Neal's funnel:

$$
v \sim \mathcal{N}(0, 3), \qquad
x_i \mid v \sim \mathcal{N}(0, e^{v / 2}), \quad i = 1, \dots, 9.
$$

**Data.** Condition on $x_i = 0$ (inference target is $p(v \mid x = 0)$).

**Reference.** The log-likelihood is linear in $v$: $\log p(x_i = 0 \mid v) = -\tfrac{1}{2}\log(2\pi) - v / 2$. Completing the square with the prior gives $v \mid x = 0 \sim \mathcal{N}(-9N/2, 3)$, hence mean $-40.5$ and variance $9$ at $N = 9$.

The joint distribution of $(v, x)$ has Neal's funnel geometry [@neal2003slice]. Conditioning every $x_i$ at zero leaves one Gaussian latent, which a Normal guide can represent exactly. The benchmark measures recovery of this conditional mean, forty units from the initialization. Its tolerance is $1$, one third of the posterior standard deviation, fixed independently of the algorithm outputs. A failed cell measures error at the listed computation budget.

**Metric.** `|E[v]_q - E[v]_true|`, tolerance `1`.

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
| Correlated regression | **PASS** `0.0365 / 0.2` | **PASS** `0.0308 / 0.2` | **PASS** `7.41e-05 / 0.2` | **PASS** `0.0273 / 0.2` | **PASS** `0.104 / 0.2` |
| Neal's funnel conditional | FAIL `10.8 / 1` | FAIL `10.8 / 1` | FAIL `19.3 / 1` | FAIL `2.58 / 1` | FAIL `2.2 / 1` |
| Ill-conditioned product Gaussian | **PASS** `0.00643 / 0.3` | **PASS** `0.00333 / 0.3` | **PASS** `7.15e-07 / 0.3` | **PASS** `0.061 / 0.3` | **PASS** `0.0707 / 0.3` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| Correlated regression | 363.5 | 289.6 | 512.2 | 99.6 | 23.7 |
| Neal's funnel conditional | 478.8 | 448.3 | 596.9 | 226.8 | 404.1 |
| Ill-conditioned product Gaussian | 151.6 | 154.2 | 207.1 | 40.7 | 39.6 |

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

**Reference.** Evaluate the truncated-Normal log-likelihood on a 4096-point $\mu$-grid in $(0, 1)$ with stable log-CDF differences for the truncation constant, and normalize it for the posterior moments.

**Metric.** `|E[mu]_q - E[mu]_true|`, tolerance `0.05`.

### Results

Posterior accuracy (metric / tolerance):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---|---|---|---|---|
| HalfNormal scale | **PASS** `0.0265 / 0.15` | **PASS** `0.0458 / 0.15` | **PASS** `0.0247 / 0.15` | **PASS** `0.00456 / 0.15` | **PASS** `0.0127 / 0.15` |
| TruncatedNormal recovery | **PASS** `0.0318 / 0.05` | **PASS** `0.0309 / 0.05` | **PASS** `0.000249 / 0.05` | **PASS** `0.000353 / 0.05` | **PASS** `0.00189 / 0.05` |

Throughput (iters/s for SVI, draws/s for MCMC):

| Problem | AutoNormal | AutoMVN | AutoLaplace | HMC | NUTS |
|---|---:|---:|---:|---:|---:|
| HalfNormal scale | 476.5 | 445.3 | 589.5 | 115.1 | 36.5 |
| TruncatedNormal recovery | 381.0 | 401.7 | 466.8 | 58.8 | 42.0 |

## Reproducing the grid

```bash
python -m tests.benchmarks.runner
```

The runner accepts `--algorithms` and `--problems` flags for partial runs and writes the regenerated tables to `docs/developer/inference-benchmarks.md` by default. `tests/benchmarks/runner.py` defines the cells and `tests/benchmarks/references.py` the reference posteriors.
