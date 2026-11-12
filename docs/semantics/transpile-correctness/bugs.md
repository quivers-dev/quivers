# BUGS

A BUGS model declares a directed graphical model: stochastic nodes, written `x ~ d<family>(args)`, and
deterministic nodes, written `x <- expr`, over which the joint distribution factors as a product of each
node's density given its parents [@lunn2009bugs; @lunn2012bugs]. Whether a node is observed depends on the
data given with the model, so the same `~` statement introduces a latent variable or scores an observed one.
BUGS is a first-order language in the sense of van de Meent and coauthors, with a fixed finite graph of
random variables [@vandemeent2018introduction]. The shared test probe compiles and evaluates the emitted
BUGS syntax with JAGS through `pyjags`, so it tests the subset of BUGS that JAGS also accepts
([Transpilation correctness](index.md)).

## Change of variables

BUGS works on the constrained parameters directly, with no change of coordinates, so
$\Psi_{\mathsf{BUGS}} = \mathrm{id}$.

## Family parameterizations

The [parameterization discussion](index.md#4-distribution-parameterization) separates a change of a family's
argument from a change of its variable, and the [JAGS table](jags.md#family-parameterizations) gives the
names that differ in JAGS.

The Normal family of BUGS takes a precision: `dnorm(μ, τ)` with $\tau = 1/\sigma^2$. Since

$$
\sqrt{\tau/(2\pi)}\, \exp\!\left(-\tau(v-\mu)^2/2\right) = (2\pi\sigma^2)^{-1/2} \exp\!\left(-(v-\mu)^2/(2\sigma^2)\right)
$$

when $\tau = 1/\sigma^2$, the substitution changes no density, and $c_{\mathrm{Normal}, \mathsf{BUGS}} = 0$. The
same calculation applies to `dlnorm`, `dt`, and `dmnorm`, whose last takes a precision matrix
$\Omega = \Sigma^{-1}$. The renderer applies the substitution through the alias `{"scale": "tau"}` of
`FAMILY_META` and a conversion that writes the scale argument as `1/(<scale>*<scale>)`
([Architecture §5.1](../transpile-architecture.md#51-kinds-of-target)).

| QVR family | BUGS | $\pi_{F, \mathsf{BUGS}}$ |
|---|---|---|
| `Normal(μ, σ)` | `dnorm(μ, τ)` | $\sigma \mapsto 1/\sigma^2$ |
| `Cauchy(μ, γ)` | `dt(μ, 1/γ², 1)` | Student t with one degree of freedom and a precision |
| `Laplace(μ, b)` | `ddexp(μ, 1/b)` | rate $1/b$ |
| `MultivariateNormal(μ, Σ)` | `dmnorm(μ, Σ⁻¹)` | precision matrix |
| `Dirichlet(α)` | `ddirch(α)` | identity |
| `Categorical(probs=p)` | `dcat(p)` for a drawn class, read back as `code - 1`, and an observed code is refused | $y \mapsto y + 1$ |
| `Bernoulli(p)` | `dbern(p)` | identity |
| `Bernoulli(logits=l)` | refused as `family:Bernoulli:logits` | `dbern` takes no log odds |
| `LogNormal(μ, σ)` | `dlnorm(μ, 1/σ²)` | precision |
| `StudentT(ν, μ, σ)` | `dt(μ, 1/σ², ν)` | precision |
| `Exponential(λ)` | `dexp(λ)` | identity |
| `Gamma(α, β)` | `dgamma(α, β)` | identity |
| `Beta(α, β)` | `dbeta(α, β)` | identity |
| `Pareto(α, x_m)` | `dpar(α, x_m)` | identity |
| `Weibull(k, λ)` | `dweib(k, λ^{-k})` | the scale to the power $-k$ |
| `Uniform(a, b)` | `dunif(a, b)` | identity |

The half families are the base distribution truncated to positive values, `dnorm(0, τ) T(0,)` and
`dt(0, τ, 1) T(0,)`, and the truncation is normalized, so they too have $c = 0$. `InverseGamma` has no BUGS
mapping and is refused. Every entry of the table has $c_{F, \mathsf{BUGS}} = 0$ after the substitution.

## Emitted constructs {#per-construct-emit}

**`sample` and `observe`.** `<name> ~ d<family>(<args>)` in `model { ... }`. Whether `<name>` is latent
depends on the data: the data are given separately, as an R `list(...)`, and a node they do not name is a
latent variable.

**Plates.** `for (m_<axis> in 1:N_<axis>) { <name>[m_<axis>] ~ d<family>(<args>) }`. The loop's nodes are
factors of the joint distribution [@lunn2009bugs], so it contributes the product of their densities. The
arguments of a row may index other plated nodes, as in `theta[doc[n], 1:K]`.

**Class codes.** QVR codes a categorical value from zero, and `dcat` counts from one. A drawn class is
`z_code ~ dcat(p)`, read back as the QVR code `z <- z_code - 1`, and a subscript reads a code one higher. An
observed code cannot be shifted: BUGS has no block that transforms the data before the model, so `y + 1`
cannot be bound from the data, a deterministic node cannot also have a distribution, and the zeros trick that
would score `log p[y + 1]` needs a carrier bound to the data. An observed class code is thus refused with
`observe:class-code:<name>`, inside and outside a `marginalize` block.

**`marginalize`.** A block whose scope is a categorical mixture is written as one `dcat` whose probabilities
already sum over the hidden class, $\sum_z p(z) p(y \mid z)$. Other blocks are refused. A model that kept $z$
would instead evaluate the joint $p(z) p(y \mid z)$ at one value of $z$.

**`score`.** BUGS has no factor primitive, and the renderer uses the zeros trick [@lunn2012bugs]: with a
node `zero_s` given the value $0$, the statements `C_s <- C - (expr)` and `zero_s ~ dpois(C_s)` add
$\log P(0 \mid C - \mathrm{expr}) = \mathrm{expr} - C$, since a Poisson distribution with rate $\lambda$ gives
$0$ the probability $e^{-\lambda}$. The constant $C$ ($10^6$ in the emitted code) does not depend on the
parameters, and the rate $C - \mathrm{expr}$ must stay nonnegative on the parameters of interest.

**`let`, `return`.** A deterministic node `<-` for a `let`. A returned value is recorded as a node
`<name>_value <- <name>`.

## Evidence

- **Level 1.** Every emitted model is a `model { ... }` with `~` and `<-` statements in a `for` loop for each
  plate axis, and calling the stages directly or composed gives the same output.
- **Level 2.** JAGS compiles the emitted syntax of the common subset.
- **Level 3.** The `pyjags` probe evaluates the emitted densities on the selected fixture grids.
