# Turing.jl

Turing.jl is a probabilistic programming language embedded in Julia [@ge2018turing]. Its `@model` macro
rewrites the statements `x ~ D` of a model function into updates of an accumulated log joint density: a
variable given as data adds its likelihood, and a variable not given is a latent variable. The test probe
evaluates the log joint density at the values the QVR density uses
([Transpilation correctness](index.md)).

## How Turing evaluates a log density

The probe is [`Turing.logjoint(model, θ)`](https://turinglang.org/Turing.jl/dev/api/Inference/#Turing.logjoint),
which runs the model with the latent variables fixed to $\theta$ and returns the accumulated log joint
density.

## Change of variables

Turing changes coordinates with [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) during
inference, and `logjoint` returns the density in constrained coordinates, so
$\Psi_{\mathsf{Turing}} = \mathrm{id}$.

## Family parameterizations

Turing's families are those of [Distributions.jl](https://juliastats.org/Distributions.jl/), parameterized
as its documentation states. Some families take the same parameters as QVR's, and the renderer converts the
others: the rates of `Exponential` and `Gamma` become scales by inversion, the probability of
`NegativeBinomial` is complemented, and the arguments of `Weibull` are reordered. These
[substitutions of parameters](index.md#4-distribution-parameterization) keep the densities on the supported
domains.

Distributions.jl has no `HalfNormal` or `HalfCauchy`, and the renderer writes them as
`truncated(Normal(0, sigma), 0, Inf)` with the
[`truncated`](https://juliastats.org/Distributions.jl/stable/truncate/) wrapper, which divides by the
probability of the retained region. The factor $2$ of the half density is thus included, and the densities
agree without a constant.

QVR codes a categorical value from zero, and Distributions.jl's `Categorical` counts from one. The renderer
draws and scores a class through `qvr_categorical(p)`, the same distribution on the support `0:K-1`
(`DiscreteNonParametric(0:K-1, p)`, added from the runtime helper), so observed codes and drawn classes are
QVR codes, and a subscript reads them one higher. Index inputs also arrive counting from zero and are read one
higher where they subscript.

## Emitted constructs

**`sample` and `observe`.** `<name> ~ <Family>(<args>)` in the body of `@model function model(...)`. An
observation is a parameter of the function, bound to its value when `logjoint` is called.

**Plates.** [`filldist(D, B)`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Distributions.filldist)
for $B$ copies of one distribution, and
[`arraydist([D_i for i in 1:B])`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Distributions.arraydist)
for copies whose parameters depend on the index. Both are products of independent distributions.

**`marginalize`.** The renderer enumerates the values of the latent variable, computes a stable log-sum-exp of
their weighted scores, and adds it with `Turing.@addlogprob!`. A grouped block sums the scores of the rows of
each group first. The latent variable is summed out, and the sampler updates the others.

**`score`, `let`, `return`.** [`Turing.@addlogprob!(expr)`](https://turinglang.org/Turing.jl/dev/api/Distributions/#Turing.@addlogprob!)
for a score, `<name> = <expr>` for a `let`, and Julia's `return`.

## Evidence

- **Level 1.** Every emitted model is an `@model function model(...) ... end` with `~` statements over
  `filldist` or `arraydist` for each plate axis, and calling the stages directly or composed gives the same
  output.
- **Level 2.** `julia --eval "Meta.parse(read(stdin, String); raise=true)"` accepts every emitted model.
- **Level 3.** The Turing probe evaluates the selected fixture grids under the shared tolerance.
