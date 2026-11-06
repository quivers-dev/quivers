# DSL Contractions

A `contraction` block combines several typed input tensors under a
named composition rule. The declaration gives explicit syntax
for "combine these typed morphisms by joining shared axes," with
the einsum wiring either inferred from the typed signature or
supplied explicitly. The categorical setup is detailed in
[Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions);
this page covers the QVR syntax.

## Declaration

<!-- compile: false -->
```qvr
contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : B * C -> D,
) : A -> D [rule=product_fuzzy]
```

The declared name is callable from any expression site as
`op_apply(arg1_morph, arg2_morph, kernel_morph)`. Each call checks
argument count and per-argument shape (by numel) against the
declared signature, then runs the contraction under the named [rule](transformations.md#the-composition-rule-hierarchy).

## Type-driven wiring inference

For [ordinary matrix multiplication](stochastic.md#markovalgebra), `"ij,jk->ik"` says: keep the row
`i` and final column `k`, and sum over the middle index `j`. Thus
`out[i,k] = sum_j f[i,j] * g[j,k]` under the `real` rule. If the
matching entries are `[2,3]` and `[5,7]`, that output cell is
`2*5 + 3*7 = 31`. Another [algebra](core.md#algebras-enrichment-algebras) retains the wiring but replaces
the product and sum.

Axis names identify which operands share an index. Two feature axes may both
have length 64 and still need distinct names if the contraction must
range over them independently.


The compiler infers the einsum wiring from the typed signature
using three rules:

- Every axis name that appears in the output sequence (the
  contraction's declared dom and cod) is a *kept* axis (propagates
  from inputs to output).
- Every axis name that appears in two or more input sequences but
  not in the output is a *contracted* axis (joined via the rule's
  `tensor_op` then `join`).
- Every axis name that appears in exactly one input sequence and
  not in the output is *anomalous*: it would need to be summed out
  by the rule but no other input shares it. The compiler raises
  with a source-keyed diagnostic and directs the user at the
  explicit `wiring` clause.

This covers the common cases (standard composition, n-ary
contraction over a shared latent axis, parallel composition with
output axes that simply propagate) without requiring the user to
spell out an einsum string for every contraction.

## Shared axes in the output

An axis that appears in the output is not summed out. In `"ij,ij->ij"`,
each output cell combines only the corresponding cells of the two
inputs. In `"ij,ij->i"`, by contrast, the column index disappears and
its contributions are reduced. The arrow's right side is what decides
which information remains in the result.


An axis named in the declared output is kept element-wise even when
it appears in more than one input:

<!-- compile: false -->
```qvr
contraction broadcast_add (
    f : A -> B,
    g : A -> B
) : A -> B [rule=real]
```

Here both `A` and `B` occur in the flattened output sequence, so the
inferred wiring is `ab, ab -> ab`. The grammar also accepts a
`share=[...]` option, but it does not add an axis to the inferred
output sequence; the declared result type still determines that
sequence. Use an explicit `wiring` clause when the result cannot be
expressed by the type-driven rule.

## `wiring`: explicit einsum escape hatch

The `wiring "<einsum>"` clause is the explicit escape hatch for
unusual contractions where the inference rules do not apply.
Common uses:

- **Diagonal extraction**: take the diagonal of a square morphism
  rather than tracing it out.
- **Axis reordering**: route inputs to the contraction in a
  non-standard order.
- **Non-product signatures**: when the contraction signature
  involves coproducts or other non-product `TypeExpr`s that the
  axis-flattening pass cannot handle.

<!-- compile: false -->
```qvr
contraction extract_diag (
    A : I * I -> 1
) : I -> 1 [rule=real, wiring="ii->i"]
```

The wiring string follows the standard numpy / torch
[`einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
notation: one letter per *distinct* axis name, joined with commas
across inputs, then `->`, then the kept axes' letters in the
output's declared order.

## Choosing the right composition rule

The `rule` keyword references a previously declared composition
rule (built-in or user-defined). The rule fixes the algebraic
operations used to contract shared axes:

- **`product_fuzzy`** (default): noisy-OR / product on
  `[0, 1]`. Right for compositions of fuzzy-truth-valued morphisms.
- **`real`**: ordinary sum / product on $\mathbb{R}$. Right for
  ordinary linear-algebra contractions.
- **`log_prob`**: log-sum-exp / sum on the log-probability
  semiring. Right for marginalizing discrete latents in a Bayesian
  composition.
- **`tropical`**: minimum / addition, for shortest-path costs.
- **`max_plus`**: maximum / addition, for best-path log scores.
  Both are tropical-style rules, with opposite optimization directions.
- **`boolean`** / **`godel`** / **`lukasiewicz`** /
  **`probability`**: the other built-in truth algebras and
  probability semirings.

User-defined rules declared at module scope (see
[Algebra](dsl-declarations.md#algebra)) are equally valid `rule`
references; the compiler verifies the rule has the right
algebraic level (`composition_rule` is permissive; weaker rules
restrict which operations are available).

## Worked example: bilinear scoring

A typical use case is bilinear scoring of an embedding against a
weight matrix and a target embedding:

<!-- compile: false -->
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

The inferred wiring is `id,de,es->is`: both feature axes are reduced,
and the singleton `Scalar` axis remains. The result has shape `(1024, 1)`,
one bilinear score per item against the shared target vector.

## See also

- [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions):
  the categorical semantics of operadic contractions.
- [Algebra](dsl-declarations.md#algebra): declaring the composition
  rule used by the contraction.
- [Transformations and Composition Rules](transformations.md): the
  broader `CompositionRule → Semigroupoid → Algebra` hierarchy and
  first-class change-of-base transformations.
