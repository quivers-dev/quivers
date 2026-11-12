# Stan

A Stan program declares data and parameters and computes a log density by adding terms to the
accumulator `target` [@carpenter2017stan]. The Stan renderer writes each draw and observation of a QVR
program as a term `target += f_lpdf(x | args);` and each score as `target += expr;`, so the emitted
program's `target` is the QVR log joint density, up to the conversions below
([Transpilation correctness](index.md)).

## How Stan evaluates a log density

The terms added to `target` are summed, so the program's log density is a sum of factors, the logarithm
of a product of the model's conditional densities. A statement `x ~ f(args)` may omit constants that do
not depend on the parameters, and a term `target += f_lpdf(x | args)` keeps them
([Stan reference manual](https://mc-stan.org/docs/reference-manual/statements.html)). The renderer
writes the second form. The test probe is
[`cmdstanpy.CmdStanModel.log_prob`](https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.log_prob),
which evaluates `target` at a point given in unconstrained coordinates.

## Change of variables to unconstrained coordinates {#unconstrained-space-change-of-variables}

A parameter declared with a constraint is sampled in unconstrained coordinates: $\theta = \Psi(u)$, where
$u$ ranges over $\mathbb R^n$ and $\theta$ satisfies the declaration. For a positive scalar
$\theta = e^u$, and the density in $u$ includes the
[log Jacobian](https://mc-stan.org/docs/stan-users-guide/reparameterization.html#changes-of-variables)
$u$ [@stanReparameterization]. The [Stan transforms reference](https://mc-stan.org/docs/reference-manual/transforms.html)
gives every transform.

| Declaration | $\Psi$ (unconstrained to constrained) | $\log \lvert \det J_\Psi \rvert$ |
|---|---|---|
| `real<lower=0>` | $\theta = \exp(u)$ | $u$ |
| `real<lower=0, upper=1>` | $\theta = \mathrm{logit}^{-1}(u)$ | $u - 2\log(1 + e^{u})$ |
| `vector<lower=0>[K]` | $\exp$ in each coordinate | $\sum_k u_k$ |
| `simplex[K]` | Stan's simplex transform | see the transforms reference |
| `cov_matrix[K]` | Cholesky factorization with a logarithm on the diagonal | see the transforms reference |
| `cholesky_factor_corr[K]` | Stan's transform for Cholesky factors of correlation matrices | see the transforms reference |
| `corr_matrix[K]` | Stan's transform for correlation matrices | see the transforms reference |

The renderer writes the constrained declaration, and Stan adds the log Jacobian to `target` itself, so the
renderer never writes it. When the probe includes the Jacobian, the comparison is

$$
\ell_{\mathsf{Stan},u}(u ; y) - \log\left|\det D\Psi(u)\right| = \ell_{\mathrm{QVR}}(\Psi(u) ; y) + C,
$$

where $C$ may depend on the model and the data but not on $u$. When the probe leaves the Jacobian out, its
value is compared with the QVR density directly.

## Family parameterizations

Stan's families are parameterized as in the
[Stan functions reference](https://mc-stan.org/docs/functions-reference/). The cases that need a
conversion or a constant:

| QVR family | Stan | $\pi_{F, \mathsf{Stan}}$ | $c_{F, \mathsf{Stan}}$ |
|---|---|---|---|
| `Normal(μ, σ)` | `normal(μ, σ)` | identity | 0 |
| `HalfNormal(σ)` | `normal(0, σ)` on a parameter declared `<lower=0>` | $\sigma \mapsto (0, \sigma)$ | $-\log 2$ |
| `HalfCauchy(γ)` | `cauchy(0, γ)` on a parameter declared `<lower=0>` | $\gamma \mapsto (0, \gamma)$ | $-\log 2$ |
| `MultivariateNormal(μ, Σ)` | `multi_normal(μ, Σ)` | identity | 0 |
| `LKJCorrCholesky(η)` | `lkj_corr_cholesky(η)` | identity | 0 |
| `Dirichlet(α)` | `dirichlet(α)` | identity, a scalar broadcast by `rep_vector` | 0 |
| `Categorical(probs=p)` | `categorical(p)` at the code `y + 1` | $y \mapsto y + 1$ | 0 |
| `Bernoulli(p)` | `bernoulli(p)` | identity | 0 |
| `Bernoulli(logits=l)` | `bernoulli_logit(l)` | identity | 0 |
| `LogitNormal(mu=μ, sigma=σ)` | a `logit_normal_lpdf` added to the program, since Stan has none: `normal_lpdf(logit(y) \| μ, σ) - log(y) - log1m(y)`, on a parameter declared `<lower = 0, upper = 1>` | identity | 0 |

For another family, the registry gives the mapping or the refusal. The half families leave out the factor
$2$ of the truncation, so each such site adds $c = -\log 2$ to the difference between Stan's density and
the normalized QVR density. These constants sum to one constant for the program, which normalization
removes.

QVR codes a categorical value from zero, and Stan's `categorical` counts from one. The data keep QVR's
codes (an outcome is declared `int<lower=0, upper=K-1>`), and the emitted log probability reads the code
one higher, `categorical_lpmf(y[n] + 1 | p)`. `ordered_logistic` and `ordered_probit` read their codes the
same way. Inside a `marginalize` block, the counter `k` runs over `1:K` and is Stan's class, so
`categorical_lpmf(k | p)` takes it directly, and the latent variable read as a number is `k - 1`. Stan has
no discrete parameters, so a drawn class is refused with `sample:discrete-latent:<name>`.

## Emitted constructs {#per-construct-emit}

**`sample x <- F(args)`.** A declaration of `x` in `parameters`, constrained by the family's support, and
`target += f_lpdf(x | args);` in `model`, which adds $\log f_{F, \mathsf{Stan}}(x \mid \mathrm{args})$.

**`observe y <- F(args)`.** `y` declared in `data`, and the same term at `y` in `model`.

**Plates.** Nested loops `for (m_<axis> in 1:N_<axis>) { ... }`, with the variable declared as
`array[N_<axis>] <type>`. The terms added in the loop's iterations sum, so the loop adds the logarithm of
the product of the factors ([plates](index.md#5-plates-marginalization-and-via)).

**`marginalize`.** Stan enumerates the latent variable and reduces with `log_sum_exp`. For a grouped block
with a latent $z$ over $K$ values, groups $G$, and an observation whose index map is $g$ from the rows $R$
to $G$:

```stan
{
  array[|G|] vector[K] lps_z;
  for (gi in 1:|G|)
    for (k in 1:K)
      lps_z[gi, k] = categorical_lpmf(k | F_args);
  for (n in 1:|R|)
    for (k in 1:K)
      lps_z[g[n], k] += <scope_lpdf>(y[n] | args_obs(k));
  for (gi in 1:|G|)
    target += log_sum_exp(lps_z[gi]);
}
```

The loops sum the log factors of each group and value, and `log_sum_exp` computes
$\log \sum_k \exp(\cdot)$ stably. The emitted density is then the density with $z$ summed out, up to the
constants above and floating-point error. When the prior of the block is a plate over the groups, the
renderer writes one row of `lps_z` per observation instead, as the
[architecture example](../transpile-architecture.md#6-an-example-from-source-to-stan-and-numpyro) shows.
The renderer refuses a block whose family has no finite support with
`marginalize:non-finite-support:<family>`. The [ZIP regression](../../examples/zip-regression.md) example
sums out a binary gate.

**`score s = expr`.** `target += expr;` in `model`, which adds the weight $\exp(\mathrm{expr})$ in log
space.

**`let v = expr`.** `real v = expr;` in `transformed parameters`. It adds no term.

**`return v`.** `<type> v_value = v;` in `generated quantities`, since Stan does not allow a parameter to be
declared twice. It adds no term.

## Evidence

- **Level 1.** Structural tests check the Stan blocks and the pipeline `Lower`, `StanRenderer`, and
  `EmitPretty(stan)`.
- **Level 2.** `stanc` accepts the programs of the syntax matrix.
- **Level 3.** On selected fixtures, `cmdstanpy`'s `log_prob` is compared with the QVR density on the shared
  grid and tolerance ([evidence](index.md#6-evidence-tiers)), after the corrections above.
