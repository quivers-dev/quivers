# Stan

This page records the per-target obligations and evidence for
[Transpilation correctness](index.md) for $\mathsf{T} = \mathrm{Stan}$.

Stan model code describes a log density to evaluate at candidate parameter
values. With `z ~ normal(0, 1)`, a supplied value of `z` contributes its
Normal log density to `target`. A QVR `sample` from that family contributes
the same prior factor when its result is clamped to that value. The
comparison below checks those factors in the coordinates used by Stan.

## Semantics

Stan specifies a density through contributions to a log-density accumulator
[@carpenter2017stan]. This can be understood as a product of factors
[@koller2009probabilistic]. A `~` statement adds a distribution’s log-density
contribution to `target`; parameter-independent constants may be omitted, as
described in the [Stan reference manual](https://mc-stan.org/docs/reference-manual/statements.html). The log-density probe is
[`cmdstanpy.CmdStanModel.log_prob`](https://mc-stan.org/cmdstanpy/api.html#cmdstanpy.CmdStanModel.log_prob),
which evaluates the program's `target` at a parameter point on the
unconstrained scale.

The reference is Carpenter, Gelman, Hoffman, Lee, Goodrich,
Betancourt, Brubaker, Guo, Li, and Riddell
[@carpenter2017stan].

## Unconstrained-space change of variables

Stan's density probe requires attention to unconstrained coordinates.
Write $\theta=\Psi_{\mathsf{Stan}}(u)$, where $u$ is unconstrained and
$\theta$ is a model parameter satisfying its declared support. For a
positive scalar, $\theta=e^u$, so the density in $u$ includes the [log
Jacobian $u$](https://mc-stan.org/docs/stan-users-guide/reparameterization.html#changes-of-variables). The [Stan transform reference](https://mc-stan.org/docs/reference-manual/transforms.html)
describes these corrections. Other targets may also transform coordinates
inside inference; the relevant question is which coordinates a probe scores.

| Declaration | $\Psi_{\mathsf{Stan}}$ (unconstrained → constrained) | $\log\bigl\lvert\det J_{\Psi_{\mathsf{Stan}}}\bigr\rvert$ |
|---|---|---|
| `real<lower=0>` | $\theta = \exp(\widetilde\theta)$ | $\widetilde\theta$ |
| `real<lower=0, upper=1>` | $\theta = \mathrm{logit}^{-1}(\widetilde\theta)$ | $\widetilde\theta - 2\log(1 + e^{\widetilde\theta})$ |
| `vector<lower=0>[K]` | per-element $\exp$ | $\sum_k \widetilde\theta_k$ |
| `simplex[K]` | simplex transform used by the installed Stan version | see the Stan transform reference |
| `cov_matrix[K]` | Cholesky factorization + log-Cholesky | see the transform reference |
| `cholesky_factor_corr[K]` | correlation Cholesky transform | see the Stan transform reference |
| `corr_matrix[K]` | correlation-matrix transform | see the Stan transform reference |

The Stan renderer emits the constrained-space declaration; Stan's
runtime adds $\log|\det J_{\Psi_{\mathsf{Stan}}}|$ to the `target`
accumulator automatically (see the [transform reference](https://mc-stan.org/docs/reference-manual/transforms.html)). The renderer never emits the Jacobian term itself.

When a target's unconstrained density includes the Jacobian, the desired
constrained-space comparison is

$$
\ell_{\mathsf{Stan},u}(u;y)
-\log\left|\det D\Psi_{\mathsf{Stan}}(u)\right|
=\ell_{\mathrm{QVR}}(\Psi_{\mathsf{Stan}}(u);y)+C.
$$

Here $C$ may depend on the fixed model and data but must not vary with $u$.
If the probe disables Jacobian adjustment, compare its output directly with
the constrained QVR density, omitting the subtraction above.

## Family parameterizations

Stan's families use the canonical parameterizations of the
[Stan Functions Reference](https://mc-stan.org/docs/functions-reference/).
Most map identically from QVR's parameterization (Wikipedia
canonical forms). The non-trivial cases:

| QVR family | Stan call | $\pi_{F, \mathsf{Stan}}$ | $c_{F, \mathsf{Stan}}$ |
|---|---|---|---|
| `Normal(μ, σ)` | `normal(μ, σ)` | identity | 0 |
| `HalfNormal(σ)` | `normal(0, σ)` with `<lower=0>` declaration | $\sigma \mapsto (0, \sigma)$ | $-\log 2$ (target minus normalized QVR density) |
| `HalfCauchy(γ)` | `cauchy(0, γ)` with `<lower=0>` | $\gamma \mapsto (0, \gamma)$ | $-\log 2$ |
| `MultivariateNormal(μ, Σ)` | `multi_normal(μ, Σ)` | identity | 0 |
| `LKJCorrCholesky(η)` | `lkj_corr_cholesky(η)` | identity | 0 |
| `Dirichlet(α)` | `dirichlet(α)` | identity (scalar broadcast via `rep_vector`) | 0 |
| `Categorical(probs=p)` | `categorical(p)` scoring the code `y + 1` | $y \mapsto y + 1$ | 0 |
| `Bernoulli(p)` | `bernoulli(p)` | identity | 0 |
| `LogitNormal(mu=μ, sigma=σ)` | `logit_normal(μ, σ)`, a grafted `logit_normal_lpdf` (Stan ships none): `normal_lpdf(logit(y) \| μ, σ) - log(y) - log1m(y)`, on a parameter declared `<lower = 0, upper = 1>` | identity | 0 |
| other families | consult the target registry | family-specific mapping or refusal | check the emitted density |

QVR codes a Categorical value from zero and Stan's `categorical`
counts from one. The data keeps QVR's coding (an outcome is declared
`int<lower=0, upper=K-1>`), and the emitted log-pmf reads the code one
higher, `categorical_lpmf(y[n] + 1 | p)`; `ordered_logistic` and
`ordered_probit` read their codes the same way. Inside a marginalize
block the enumeration counter `k` runs over `1:K` and already is
Stan's class, so `categorical_lpmf(k | p)` takes it directly, while
the latent read as a number is the code `k - 1`. Stan has no discrete
parameters, so a drawn class is refused with
`sample:discrete-latent:<name>`.

Omitting the half-distribution normalizer contributes target-minus-QVR
$c_{F,\mathsf{Stan}}=-\log 2$ per call site. This is collected into
the per-program constant $C_{\mathsf{Stan}}(M)$ and absorbed by
Bayes-rule normalization.

## Per-construct emit

**`SampleStep(x, F, args)`.** `top_var_decl_no_assign` of type
$\tau_F$ in `parameters` (with the right constraint per the
family's `.support`); `sampling_statement` $x \sim F_{\mathsf{Stan}}(\mathrm{args})$
in `model`. Soundness: by Stan's `~` desugaring, the contribution
to `target` is exactly $\log f_{F, \mathsf{Stan}}(x \mid
\mathrm{args})$.

**`ObserveStep(y, F, args)`.** $y$ declared in `data`; same `~`
statement in `model` contributing $\log f_{F, \mathsf{Stan}}(y \mid
\mathrm{args})$.

**`Plate` (Indexed Bind / Indexed Observe).** Nested
`for (m_<axis> in 1:N_<axis>) { ... }` loops; the latent or
observation is declared as `array[N_<axis>] <type>`. Per Stan
reference manual’s loop semantics, the for-loop's contribution to `target` is
the additive sum of per-iteration `~` contributions. Soundness
follows from [the plate discussion](index.md#5-plates-marginalization-and-via).

**`MarginalizeStep`.** Stan emits the canonical `log_sum_exp`
enumeration. For a grouped marginalize with latent $z : K$ on
group axis $G$ and scope observe via fibration $g : R \to G$:

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

The inner accumulation implements the required finite sum of log factors; the outer `log_sum_exp` is the documented
`log_sum_exp` primitive that computes $\log \sum_k \exp(\cdot)$
accurately. Under [the finite-enumeration strategy](index.md#5-plates-marginalization-and-via),
the emitted log-density equals the marginalized joint up to the
parameter-independent normalizing constants discussed above, subject to
floating-point error and the supported-family contract.

The Stan renderer raises
`UnsupportedConstruct(["marginalize:non-finite-support:<family>"])`
when `finite_enumerable_at_call_site`
returns False; cf.
[ZIP regression](../../examples/zip-regression.md) for the
continuous-support workaround.

**`ScoreStep(name, expr)`.** `real <name> = <expr>;` in
`transformed_parameters`, plus `target += <name>;` in `model`.
Contribution: $\log w_{\mathrm{name}} = \mathrm{expr}$.

**`LetStep(name, expr)`.** `real <name> = <expr>;` in
`transformed_parameters`. Denotationally inert.

**`Return(v_1, \dots, v_m)`.** `generated quantities { <type> v_i_value = v_i; }`
for each return variable that is already a parameter (Stan rejects
re-declaration). Denotationally inert.

## Acceptance

* **Tier 1 pipeline composition.** Structural checks cover the
  expected Stan blocks and the `Lower >> StanRenderer >>
  EmitPretty(stan)` pipeline.
* **Tier 2 external syntax.** `stanc` accepts the generated
  programs in the external-validation matrix.
* **Tier 3 numeric equivalence.** For selected fixtures,
  `cmdstanpy.log_prob` is compared with the QVR reference on the
  shared finite grid and tolerance described in
  [the test contract](index.md#6-evidence-tiers), after
  applying any documented backend-specific correction.
