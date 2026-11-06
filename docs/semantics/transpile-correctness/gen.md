# Gen.jl

Per-target obligations of [Transpilation correctness](index.md)
for $\mathsf{T} = \mathrm{Gen.jl}$.

Gen identifies random choices by addresses. A choice map supplies values
at those addresses, and `assess` evaluates the corresponding trace weight.
For a plate, an address contains the row position as well as the site name,
so two rows remain distinct choices. This makes address preservation part
of comparing the same parameter point across representations.

## Semantics

The [trace and sample-site guide](../../guides/inference-foundations.md#trace-and-sample-sites) explains the values and log densities being recorded. [Lexical handler execution](../effects.md#8-qiec-rows-and-lexical-handlers) explains how QVR preserves request identities before the Gen renderer assigns addresses.

Gen.jl's denotational semantics is the
generative-function trace semantics of Cusumano-Towner, Saad,
Lew, and Mansinghka
[@cusumanotowner2019gen]. A `@gen
function` is a generative function whose `@trace` invocations
record per-address choices in a `ChoiceMap`. The log-density
probe is [`Gen.assess(generator, args, choicemap)[1]`](https://www.gen.dev/docs/stable/ref/inference/#Gen.assess),
where Julia's `[1]` selects the returned log weight.

## Unconstrained-space change of variables

Identity at the renderer level. Gen's MCMC kernels handle their
own unconstrained-space reparameterizations.

## Family parameterizations

Gen consumes
[Distributions.jl](https://juliastats.org/Distributions.jl/) via
its `Gen.Distribution` wrappers. The renderer applies family-specific parameter mappings, including
rate-to-scale inversion and the NegativeBinomial probability convention.
The half-family mapping exposes a specific limitation. An emitted Normal
base density is half the normalized HalfNormal density at positive values,
so their log scores differ by $-\log 2$ there. But the Normal also assigns
positive probability to negative values, where HalfNormal has zero mass.
Comparisons on positive fixtures thus cover the score offset; forward
sampling would require the corresponding support restriction.

Wrapped families (`Truncated`, `Mixture`) use Gen's
[Distributions.jl composition primitives](https://juliastats.org/Distributions.jl/stable/truncate/)
via a per-renderer `_WRAPPER_BUILDERS` dispatch table; cf.
[Architecture §5.1](../transpile-architecture.md#51-backend-idioms).

## Per-construct emit

**Sample / observe.** Gen has no native plate; the renderer emits
per-batch-axis `for m in 1:B; <name>[m] = @trace(<dist>, (:name,
m)); end` loops over a pre-allocated `Vector{T}(undef, B)`
storage array. Each `@trace` registers an address in the choice
map.

**Plate.** The for-loop over `1:B` ranges produces $B$
independent `@trace` sites, each with the same conditional
distribution. The product measure is described in
[the plate discussion](index.md#5-plates-marginalization-and-via).

**Marginalize.** The renderer evaluates the finite support's atom densities,
combines them with their prior weights, and traces the reduced density as a
factor choice under the `:qvr_factor` namespace. The marginalized latent is
not emitted as an ordinary random-choice site.

**Score / let / return.** [`@addlogprob!(expr)`](https://www.gen.dev/docs/stable/ref/gfi/#Gen.@addlogprob!)
for score; native `<name> = <expr>` for let; native `return`.

## Acceptance

* **Tier 1 structural.** Every emit has `@gen function model(...)
  ... end` with `@trace` calls and per-batch `for` loops.
* **Tier 1 pipeline composition.** Direct and composed pipeline calls agree.
* **Tier 2 external syntax.** `julia --eval "Meta.parse(read(stdin, String); raise=true)"`
  accepts every emit.
* **Tier 3 numeric equivalence.** `Gen.assess(...)[1]` is compared
  with the QVR reference on the selected fixture grids.
