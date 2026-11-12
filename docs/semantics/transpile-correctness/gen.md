# Gen.jl

Gen.jl writes a model as a generative function, whose `@trace` calls record random choices at addresses in
a choice map, and it lets the user program inference over those choices [@cusumanotowner2019gen]. A choice
map gives values at addresses, and `assess` returns the log weight of the corresponding trace. In a plate the
address contains the row as well as the name of the site, so the rows are separate choices
([Transpilation correctness](index.md)). QVR keeps the identity of each request through elaboration
([Effects and type classes §8](../effects.md#8-qiec-rows-and-handlers)) before the renderer assigns
addresses.

## How Gen evaluates a log density

The probe is [`Gen.assess(generator, args, choicemap)[1]`](https://www.gen.dev/docs/stable/ref/inference/#Gen.assess),
where Julia's `[1]` selects the returned log weight of the trace whose choices the choice map fixes.

## Change of variables

The emitted model is in constrained coordinates, and Gen's inference kernels change coordinates themselves,
so $\Psi_{\mathsf{Gen}} = \mathrm{id}$.

## Family parameterizations

Gen's families wrap those of [Distributions.jl](https://juliastats.org/Distributions.jl/). The renderer
converts parameters for each family, inverting rates to scales and complementing the probability of
`NegativeBinomial`.

`HalfNormal` and `HalfCauchy` are written as the base distribution, `normal(0, scale)` and
`cauchy(0, scale)`, without truncation. At a positive value the emitted density is half the QVR density, so
the log densities differ by $-\log 2$ there. But the emitted distribution also gives mass to negative values,
where the half family has none. A comparison at positive points thus sees a constant offset, while
forward sampling from the emitted model can produce negative values the QVR model cannot.

The wrapper families `Truncated` and `Mixture` are written with Distributions.jl's
[composition functions](https://juliastats.org/Distributions.jl/stable/truncate/) through a table of
wrapper builders in the renderer ([Architecture §5.1](../transpile-architecture.md#51-kinds-of-target)).

QVR codes a categorical value from zero, and Gen's `categorical` counts from one. The renderer traces a class
through `qvr_categorical(p)`, a distribution added to the module whose support is `0:K-1` and whose log density
at `x` is that of `categorical` at `x + 1`, so observed codes and traced classes are QVR codes, and a subscript
reads them one higher. Index inputs also count from zero and are read one higher where they subscript.

## Emitted constructs

**`sample` and `observe`.** Gen has no plate primitive, and the renderer writes a loop
`for m in 1:B; <name>[m] = @trace(<dist>, (:name, m)); end` over a preallocated `Vector{T}(undef, B)`, each
`@trace` an address of the choice map.

**Plates.** The loop over `1:B` makes $B$ independent choices with the same conditional distribution, whose
density is the product ([plates](index.md#5-plates-marginalization-and-via)).

**`marginalize`.** The renderer computes the densities at the values of the latent variable, combines them
with their prior weights, and traces the reduced log weight as a factor choice under the address
`:qvr_factor`, reusing the emission of the Turing renderer. The variable summed out is not a choice.

**`score`.** Gen has no factor primitive either. The renderer adds a distribution with one point,
`QvrFactorDist`, whose log density at its point is the weight, and traces a score as
`@trace(_qvr_qiec_factor(s), :qvr_factor => :s)`, so the weight enters the trace's score as a choice does.

**`let`, `return`.** `<name> = <expr>` for a `let`, and Julia's `return`.

## Evidence

- **Level 1.** Every emitted model is a `@gen function model(...) ... end` with `@trace` calls and a `for` loop
  for each batch axis, and calling the stages directly or composed gives the same output.
- **Level 2.** `julia --eval "Meta.parse(read(stdin, String); raise=true)"` accepts every emitted model.
- **Level 3.** `Gen.assess(...)[1]` is compared with the QVR density on the selected fixture grids.
