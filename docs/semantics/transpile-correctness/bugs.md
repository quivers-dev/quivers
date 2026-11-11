# BUGS

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{BUGS}$.

A BUGS model states relationships among variables. Whether a variable is
observed depends on the supplied data, so the same `~` syntax can introduce
an unknown quantity or score a known one. Parameter conventions need care:
a Normal standard deviation of 2 corresponds to precision $1/4$.
The sections below track such conversions before considering test evidence.

## Semantics

BUGS's denotational semantics is the directed graphical model
semantics of Lunn, Spiegelhalter, Thomas, and Best
[@lunn2009bugs]. A `model { ... }`
block declares a set of stochastic nodes (`~`) and deterministic
nodes (`<-`) over which the joint distribution factors. The
shared repository probe compiles and evaluates BUGS-syntax output
through JAGS and `pyjags`, testing the subset shared by the renderer and JAGS.

## Unconstrained-space change of variables

BUGS works on the constrained parameter space directly; no
automatic reparameterization. $\Psi_{\mathsf{BUGS}} = \mathrm{id}$.

## Family parameterizations

The [shared parameterization discussion](index.md#4-distribution-parameterization) separates changing a constructor’s scale argument from transforming its random variable. The [JAGS mapping table](jags.md#family-parameterizations) identifies the closely related target’s naming differences.

BUGS uses precision-parameterized normal-family distributions:
`dnorm(μ, τ)` where $\tau = 1/\sigma^2$. This is the canonical
**parameterization substitution** worked through in
[the parameterization contract](index.md#4-distribution-parameterization): the
algebraic equivalence

$$
\sqrt{\tau/(2\pi)} \exp\!\left(-\tau(v-\mu)^2/2\right)
\;=\;
(2\pi\sigma^2)^{-1/2} \exp\!\left(-(v-\mu)^2/(2\sigma^2)\right)
$$

under $\tau = 1/\sigma^2$ certifies $c_{\mathrm{Normal},
\mathrm{BUGS}} = 0$. The same calculation applies to `dlnorm` (log-
normal), `dt` (Student-t), and `dmnorm` (multivariate normal with
precision matrix $\Omega = \Sigma^{-1}$). The renderer applies
the substitution via the
`FAMILY_META` argument aliases
table (`{"scale": "tau"}`) plus a per-alias arithmetic transform
that wraps the scale arg in `1/(<scale>*<scale>)`. Cf.
[Architecture §5.1](../transpile-architecture.md#51-backend-idioms).

Other parameterization differences:

| QVR family | BUGS call | $\pi_{F, \mathsf{BUGS}}$ |
|---|---|---|
| `Normal(μ, σ)` | `dnorm(μ, τ)` | $\sigma \mapsto 1/\sigma^2$ |
| `Cauchy(μ, γ)` | `dt(μ, 1/γ², 1)` | StudentT-with-1df with precision |
| `Laplace(μ, b)` | `ddexp(μ, 1/b)` | rate = $1/b$ |
| `MultivariateNormal(μ, Σ)` | `dmnorm(μ, Σ⁻¹)` | precision matrix |
| `Dirichlet(α)` | `ddirch(α)` | identity |
| `Categorical(probs=p)` | `dcat(p)` on a drawn class, read back as `code - 1`; an observed code is refused | $y \mapsto y + 1$ |
| `Bernoulli(p)` | `dbern(p)` | identity |
| `LogNormal(μ, σ)` | `dlnorm(μ, 1/σ²)` | precision |
| `StudentT(ν, μ, σ)` | `dt(μ, 1/σ², ν)` | precision |
| `Exponential(λ)` | `dexp(λ)` | identity |
| `Gamma(α, β)` | `dgamma(α, β)` | identity |
| `Beta(α, β)` | `dbeta(α, β)` | identity |
| `Pareto(α, x_m)` | `dpar(α, x_m)` | identity |
| `Weibull(k, λ)` | `dweib(k, 1/λ^k)` | scale to its $k$-th power |
| `Uniform(a, b)` | `dunif(a, b)` | identity |
| `InverseGamma(α, β)` | inverse-transform of `dgamma(α, β)` (composed) | identity on shape, $1/x$ on rate |

Every entry has $c_{F, \mathsf{BUGS}} = 0$ after substitution.

## Per-construct emit

**Sample / observe.** `<name> ~ d<family>(<args>)` inside the
`model { ... }` block. Latent vs observed is inferred from
whether `<name>` appears on the data side: BUGS programs ship
their data as a separate `list(...)` block, and variables not in
the data block are stochastic latents.

**Plate.** `for (m_<axis> in 1:N_<axis>) { <name>[m_<axis>] ~
d<family>(<args>) }`. Under BUGS’s graphical-model interpretation [@lunn2009bugs], the for-loop's
contribution to the joint is the product of per-iteration
factors. Each row's args may index into other plate-shaped nodes
(LDA's `theta[doc[n], 1:K]` form).

**Class codes.** QVR codes a Categorical value from zero and `dcat` counts
from one. A drawn class is drawn as `z_code ~ dcat(p)` and read back as the
QVR code `z <- z_code - 1`, and a subscript reads a code one higher. An
observed code cannot be shifted: BUGS has no `data` block to bind
`y + 1` from the data, a logical node cannot also carry a distribution, and
the zeros trick that would score `log p[y + 1]` needs a data-bound carrier.
An observed class code is therefore refused with `observe:class-code:<name>`,
outside a marginalize block and inside one.

**Marginalize.** A categorical-mixture scope can be collapsed into a `dcat`
distribution whose probabilities already sum over the hidden class. Other
marginalization blocks are refused. The collapsed probabilities represent
$\sum_z p(z)p(y\mid z)$; a model retaining $z$ instead evaluates the joint
$p(z)p(y\mid z)$ at one value of that coordinate.

**Score / let / return.** BUGS has no native `factor` primitive.
The renderer uses the
zero trick:
with host-supplied `zero_name = 0`, the relation
`zero_name ~ dpois(C - expr)` contributes `expr - C` because
`log P(0 | λ) = -λ`. The positive carrier constant $C$ is
parameter-independent. Deterministic bindings emit with `<-`;
return is a QVR-level selection rather than a BUGS statement.

## Acceptance

* **Tier 1 structural.** Every emit has `model { ... }` with `~`
  and `<-` statements wrapped in `for` loops per plate axis.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** JAGS compiles the emitted common-subset syntax.
* **Tier 3 numeric equivalence.** The JAGS/`pyjags` probe evaluates
  emitted BUGS-syntax densities on the selected fixture grids.

### References


For the BUGS project and its modeling language, see [@lunn2009bugs]. For the related JAGS system, see [@plummer2003jags]. The zero trick used above follows directly from the Poisson probability at zero.
