# DSL Contractions

A `contraction` declaration combines several morphisms in one step by a
[tensor contraction](https://en.wikipedia.org/wiki/Tensor_contraction) under a named
composition rule. The contraction's [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
wiring is either inferred from the declared types or given explicitly.
[Composition Rules](../semantics/composition-rules.md#4-operadic-contractions) gives its
denotation.

## Declaration

```qvr
object A : FinSet 2
object B : FinSet 3
object C : FinSet 4
object D : FinSet 5

contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : B * C -> D,
) : A -> D [rule=product_fuzzy]

morphism a1 : A -> B [role=latent]
morphism a2 : A -> C [role=latent]
morphism k : B * C -> D [role=latent]
define r = op_apply(a1, a2, k)
export r
```

The declared name can be called wherever a morphism expression is allowed, as
`op_apply(a1, a2, k)`. Each call checks the number of arguments and each argument's
domain and codomain sizes (numbers of elements) against the declaration, then computes
the contraction under the named [rule](transformations.md#the-composition-rule-hierarchy).
When the inputs span several lines, each input, the last included, ends with a comma.

## Inferring the wiring from the types {#type-driven-wiring-inference}

An einsum string names each axis by a letter, lists the inputs' axes before the arrow
and the output's after it, and sums every letter missing from the output. For
[matrix multiplication](stochastic.md#markovalgebra), `"ij,jk->ik"` keeps the row `i`
and the column `k` and sums over the middle index `j`, so
`out[i,k] = sum_j f[i,j] * g[j,k]` under the `real` rule. If the matching entries are
`[2, 3]` and `[5, 7]`, that output entry is `2*5 + 3*7 = 31`. Another
[`Algebra`](core.md#algebras-enrichment-algebras) keeps the wiring and replaces the
product and the sum with its $\otimes$ and $\bigoplus$.

The compiler names the axes after the objects in the declared types, so two axes share
a letter exactly when they have the same object, and it infers the wiring by three
rules:

- an axis that appears in the declared output type is kept;
- an axis that appears in two or more inputs and not in the output is contracted: the
  entries are combined with the rule's $\otimes$ and reduced with its $\bigoplus$;
- an axis that appears in only one input and not in the output is refused.

Ordinary einsum would sum such an unshared axis. The compiler instead raises an error at
the declaration, since nothing in the types says whether the axis was meant to be kept
or reduced, and the message suggests adding the axis to the output type, listing it in
a `share` option, or writing the `wiring` explicitly. Two feature axes of the same
length must thus be different objects when the contraction should range over them
independently.

The three rules cover composition, contraction of several inputs over a shared
latent axis, and parallel composition whose output axes pass through, without an
einsum string.

## Shared axes in the output

An axis in the output is not reduced. With `"ij,ij->ij"` each output entry combines
only the corresponding entries of the two inputs, and with `"ij,ij->i"` the column
axis is reduced. An axis named in the declared output is thus kept entry by entry even
when it appears in several inputs:

```qvr
composition real [level=algebra]
object A : FinSet 2
object B : FinSet 3

contraction broadcast_mul (
    f : A -> B,
    g : A -> B,
) : A -> B [rule=real]

morphism f1 : A -> B [role=latent]
define r = broadcast_mul(f1, f1)
export r
```

Both `A` and `B` occur in the output, so the inferred wiring is `ab, ab -> ab`, the
entrywise product. The option `share=[...]` lists axes that may appear in only one
input: such an axis is then reduced with $\bigoplus$ instead of refused. Listing an
axis that the output already keeps changes nothing, and `share` never adds an axis to
the output, whose axes the declared type fixes.

## `wiring`: explicit einsum

The option `wiring="<einsum>"` gives the einsum string directly, for contractions the
three rules do not decide, such as inputs read in a different axis order or types that
are not plain products. The string follows the
[einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) notation of
NumPy and PyTorch: one letter per distinct axis, inputs separated by commas, then `->`
and the kept letters in the order of the declared output. A letter may not repeat
within one input, so a diagonal or a trace is not expressible as a wiring
([`f.trace(A)`](morphisms.md#transpose-trace-cup-and-cap) computes a trace).

```qvr
composition real [level=algebra]
object A : FinSet 2
object B : FinSet 3
object C : FinSet 4
object D : FinSet 5

contraction op_explicit (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : B * C -> D,
) : A -> D [rule=real, wiring="ab,ac,bcd->ad"]

morphism a1 : A -> B [role=latent]
morphism a2 : A -> C [role=latent]
morphism k : B * C -> D [role=latent]
define r = op_explicit(a1, a2, k)
export r
```

## Choosing the right composition rule

`rule=` names a composition rule, built-in or declared in the module, which fixes the
operations used on contracted axes ([Core Types](core.md#algebras-enrichment-algebras)
gives each one's standard name):

- `product_fuzzy` (default): product and noisy-OR on $[0, 1]$, for fuzzy truth values;
- `real`: sum and product on $\mathbb R$, for ordinary linear algebra;
- `markov`: sum and product of conditional probabilities;
- `log_prob`: addition and log-sum-exp, the log semiring, for summing out discrete
  latent variables in log space;
- `tropical`: addition and minimum, for shortest-path costs;
- `max_plus`: addition and maximum, for the log score of the best path;
- `boolean`, `godel`, `lukasiewicz`, `probability`, `counting`: the remaining built-in
  instances.

A rule declared in the module ([Algebra](dsl-declarations.md#algebra)) can be named
the same way. The compiler checks the rule's level against the module's, and a weaker
level allows fewer operations ([levels](transformations.md#the-composition-rule-hierarchy)).

## Worked example: bilinear scoring

Bilinear scoring combines an item embedding, a weight matrix, and a target vector:

```qvr
composition real [level=algebra]
object Item : FinSet 1024
object Scalar : FinSet 1
object LeftFeature : FinSet 64
object RightFeature : FinSet 64

morphism embedding : Item -> LeftFeature [role=latent]
morphism weights : LeftFeature -> RightFeature [role=latent]
morphism target : RightFeature -> Scalar [role=latent]

# Score[i, 0] = sum_d sum_e E[i, d] * W[d, e] * T[e, 0]
# LeftFeature and RightFeature are reduced; Item and Scalar remain.
contraction bilinear_score (
    e : Item -> LeftFeature,
    w : LeftFeature -> RightFeature,
    t : RightFeature -> Scalar,
) : Item -> Scalar [rule=real]

define final = bilinear_score(embedding, weights, target)
export final
```

The inferred wiring is `id,de,es->is`: both feature axes are reduced and the
one-element `Scalar` axis remains. The result has shape `(1024, 1)`, one score per item
against the shared target vector.

## See also

- [Composition Rules](../semantics/composition-rules.md#4-operadic-contractions): the
  denotation of contractions.
- [Algebra](dsl-declarations.md#algebra): declaring the composition rule a contraction
  uses.
- [Transformations and Composition Rules](transformations.md): the levels of
  composition rules and transformations between algebras.
