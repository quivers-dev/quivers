# Turing.jl

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Turing.jl}$.

Turing turns a Julia model function into a record of random choices and
log-density contributions. A variable supplied as observed data contributes
its likelihood; an unknown variable is a latent choice. Evaluating
`logjoint` fixes the choices and scores the model, while running an inference
algorithm explores choices. The density tests call `logjoint` at the
same fixed values used by the QVR reference.

## Semantics

Turing.jl's denotational semantics is the trace semantics of Ge,
Xu, and Ghahramani
[@ge2018turing] implemented
via the `@model` macro that rewrites `~`-statements into joint
log-density accumulator updates. The log-density probe is
[`Turing.logjoint(model, θ)`](https://turinglang.org/Turing.jl/dev/api/Inference/#Turing.logjoint),
returning the joint log-density at a parameter point.

## Unconstrained-space change of variables

Turing applies per-distribution
[Bijectors.jl](https://github.com/TuringLang/Bijectors.jl)
transformations during inference; `logjoint` returns the
constrained-space density. $\Psi_{\mathsf{Turing}} = \mathrm{id}$
at the renderer level.

## Family parameterizations

Turing consumes
[Distributions.jl](https://juliastats.org/Distributions.jl/) which
uses the canonical parameterizations of
[Wikipedia: List of probability distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions).
Some families use the same parameters, but the renderer also translates
conventions: Exponential and Gamma rates become scales by inversion,
NegativeBinomial probabilities are complemented, and Weibull arguments are
reordered. These [parameter substitutions](index.md#4-distribution-parameterization)
preserve densities on the supported domains. The half-family mappings
also use [truncation normalization](https://juliastats.org/Distributions.jl/stable/truncate/#Truncated-Distributions):
restricting a distribution to positive values requires dividing its density
by the probability of that region.

The Turing renderer composes `HalfNormal` / `HalfCauchy` (which
have no Distributions.jl primitive) as `truncated(Normal(0,
sigma), 0, Inf)` via the documented
[`truncated`](https://juliastats.org/Distributions.jl/stable/truncate/)
wrapper. The [truncated-distribution definition](https://juliastats.org/Distributions.jl/stable/truncate/)
divides by the retained probability mass. Thus the half-Normal's factor of
two is already included, and normalized half-Normal densities agree without
an additional $\log 2$ discrepancy.

## Per-construct emit

**Sample / observe.** `<name> ~ <Family>(<args>)` inside the
`@model function model(...)` body; observations appear as function
parameters that bind to specific values when `logjoint` is called.

**Plate.** [`filldist(D, B)`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Distributions.filldist)
for index-independent batches; [`arraydist([D_i for i in 1:B])`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Distributions.arraydist)
for index-dependent batches. Both denote the documented product
measure (Turing.jl documentation, "Composing distributions").

**Marginalize.** The renderer enumerates the finite support, computes a
numerically stable log-sum-exp of the weighted atom scores, and adds the
result with `Turing.@addlogprob!`. Grouped scopes first accumulate row scores
by group. The hidden discrete coordinate has been summed out, leaving
the sampler to update the remaining latents.

**Score / let / return.** [`Turing.@addlogprob!(expr)`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Turing.@addlogprob!)
for score; native `<name> = <expr>` for let; native `return` for
return.

## Acceptance

* **Tier 1 structural.** Every emit has `@model function model(...)
  ... end` with `~` statements wrapping `filldist` / `arraydist`
  per plate axis.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** `julia --eval "Meta.parse(read(stdin, String); raise=true)"`
  accepts every emit.
* **Tier 3 numeric equivalence.** The Turing probe evaluates the
  selected fixture grids under the shared constant-spread tolerance.
