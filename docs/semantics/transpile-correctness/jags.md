# JAGS

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{JAGS}$.

JAGS uses BUGS-style model declarations and supplied data to distinguish
unknown variables from observations. A QVR `marginalize` block requests the
finite sum $\sum_z p(z)p(y\mid z)$. The renderer computes that sum and
contributes its logarithm to the joint density.

## Semantics

JAGS's denotational semantics is the directed graphical model
semantics of Plummer
[@plummer2003jags],
inheriting BUGS's `model { ... }` block syntax ([@lunn2009bugs]
[BUGS](bugs.md)) with extensions. The runtime is JAGS itself or
the [`pyjags`](https://github.com/michaelnowotny/pyjags) Python
binding; the log-density probe is JAGS's Gibbs sampler log-joint
accumulator.

The renderer targets the BUGS-like subset accepted by JAGS; its
parameter substitutions and score zero trick follow the BUGS route.

## Unconstrained-space change of variables

Identity, as for BUGS. $\Psi_{\mathsf{JAGS}} = \mathrm{id}$.

## Family parameterizations

JAGS shares BUGS's precision-parameterized normal family
(`dnorm(μ, τ)` with $\tau = 1/\sigma^2$), the same `dlnorm`,
`dt`, `dmnorm` precision conventions, and the same calculation
showing $c_{F, \mathsf{JAGS}} = 0$ for every family. See the
[BUGS page](bugs.md) for the full parameterization table; the
JAGS family-name differences are:

| QVR family | JAGS call | Note |
|---|---|---|
| `Dirichlet(α)` | `ddirich(α)` | (BUGS uses `ddirch`) |
| `Gamma(α, β)` | `dgamma(α, β)` | matches BUGS |
| `Bernoulli(p)` | `dbern(p)` | matches BUGS |
| `Categorical(probs=p)` | `dcat(p)` | matches BUGS |
| `Normal(μ, σ)` | `dnorm(μ, 1/σ²)` | precision parameterization |

JAGS also exposes
[`dgen.gamma`](https://justinribarski.github.io/JAGS/), `pow`,
and `inprod` primitives for use in extensions.

## Per-construct emit

The [BUGS emit rules](bugs.md#per-construct-emit) work through the Poisson zero trick used to add a log score. The [finite marginalization definition](../programs.md#26-marginalize) specifies the weighted sum that the JAGS factor must contain.

Sample, observe, and plate declarations follow the BUGS-style model syntax,
with JAGS-specific distribution names and argument mappings.

**Marginalize.** The renderer constructs the weighted sum over finite support
values and scores its logarithm using the zero trick. Grouped cases accumulate
the rows that share a latent before reducing it. Supported scope shapes and
families are checked explicitly; this path is broader than BUGS's
categorical-mixture collapse.

**Score / let.** The score path uses the documented zero-trick convention;
deterministic bindings use `<-`. A Poisson rate used by the zero trick must
be nonnegative, so any additive carrier constant has to satisfy the relevant
bound throughout the intended parameter domain.

## Acceptance

* **Tier 1 structural.** Same shape as BUGS.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** The test writes a JAGS command file,
  asks JAGS to load the model, and rejects compiler error output.
* **Tier 3 numeric equivalence.** The `pyjags` probe evaluates
  selected emitted densities under the constant-spread criterion.

### References


For JAGS’s design and relation to BUGS, see [@plummer2003jags]. A tutorial with worked Bayesian models is [@kruschke2014doing].
