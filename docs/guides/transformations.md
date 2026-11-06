# Transformations and Composition Rules

This guide covers two related mechanisms: first-class *transformations* for
[change of base](categorical.md#base-change) and the *composition-rule hierarchy* beneath [`>>`](morphisms.md#composition-the-operator). Together
they specify how morphisms cross enrichment algebras.

## First-class transformations

Suppose a row contains scores `[0.2, 0.8]`. A threshold at `0.5` makes
separate yes/no decisions `[0, 1]`. Softmax instead compares the entries
and gives approximately `[0.354, 0.646]`, which sums to one. These solve
different problems: thresholding produces a relation, while softmax creates
a conditional probability row from scores. Its object argument identifies
the output axes over which to normalize.

Moving a conversion across composition requires it to preserve the
combining operations. Thresholding at `0.5` fails for product: two scores
of `0.6` each pass the threshold, while their product `0.36` fails.
Applying the threshold before or after composition can thus give different
relations. Row [normalization](stochastic.md#normalization) also uses neighboring entries: increasing
one score lowers the normalized weights of the others.


A *transformation* turns a morphism in one enrichment algebra into a morphism in another. Two kinds exist:

- **Entrywise maps**, exposed through the `AlgebraHomomorphism` API, act pointwise: $\varphi : \mathcal{V} \to \mathcal{W}$ sends each value to the target representation; every entry of the morphism's tensor is sent through $\varphi$.
- **Morphism transformations** act on the whole tensor, consuming axis information; softmax row-normalization, L1/L2 normalization, and [Bayes inversion under a prior](stochastic.md#conditioning-and-bayesian-updates) all live here.

The library treats both as first-class DSL values, living in a transformation namespace disjoint from the morphism namespace. You can let-bind them, compose them with `>>>`, and pass either kind to `change_base`.

### Singletons

Bare-name transformations carry a fixed `(source, target)` pair. The shipped catalog:

| Name | `source` | `target` |
|---|---|---|
| `expectation` | `Markov` | `ProductFuzzyAlgebra` |
| `material_implication` | `ProductFuzzyAlgebra` | `Godel` |
| `log_prob` | `ProductFuzzyAlgebra` | `LogProb` |
| `max_plus` | `LogProb` | `MaxPlus` |
| `threshold` | `ProductFuzzyAlgebra` | `Boolean` |
| `boolean_embedding` | `Boolean` | `ProductFuzzyAlgebra` |
| `probability_clamp` | `Real` | `Probability` |
| `probability_to_real` | `Probability` | `Real` |
| `counting_from_real` | `Real` | `Counting` |
| `counting_to_real` | `Counting` | `Real` |

### Constructors

Parametric transformations take one argument (an object or a morphism) and produce a transformation value:

| Constructor | Argument | `source` | `target` |
|---|---|---|---|
| `softmax(B)` | object | `ProductFuzzyAlgebra` | `Markov` |
| `l1_normalize(B)` | object | `Real` | `Markov` |
| `l2_normalize(B)` | object | `Real` | `Real` |
| `bayes_invert(prior)` | morphism | `Markov` | `Markov` |

### Let-binding and composition

There are two different pipelines: `f >> g` connects model arrows through
a shared object; `t1 >>> t2` applies one representation conversion after
another. Their boundaries are checked against objects and algebras,
respectively: `cod(f) = dom(g)` and `target(t1) = source(t2)`.


Both forms can be let-bound and composed with `>>>`. The compiler checks their boundary: `t1 >>> t2` requires `target(t1) = source(t2)`.

<!-- compile: false -->
```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=latent]

define s    = softmax(B)
define pipe = s >>> expectation
define g    = f.change_base(pipe)
export g
```

`softmax(B)` produces a transformation $\mathcal{V}_\mathrm{pf} \to \mathcal{V}_\mathrm{M}$; `expectation` is $\mathcal{V}_\mathrm{M} \to \mathcal{V}_\mathrm{pf}$; their composition returns a row-normalized morphism in $\mathcal{V}_\mathrm{pf}$. For the row above, the final values remain approximately `[0.354, 0.646]`; `expectation` retains the normalized entries.

The `change_base` postfix accepts any expression that denotes a transformation: a bare singleton, a let-bound name, a constructor call, or a `>>>` chain.

### Python API

The Python core provides the same operations in `quivers.core.trans`:

<!-- python: skip -->
```python
from quivers.core.trans import compose_trans
from quivers.core.morphism_transformations import softmax
from quivers.core.algebra_morphisms import EXPECTATION

pipe = compose_trans(softmax(B), EXPECTATION)
g    = f.change_base(pipe)
```

The DSL and Python APIs use the same flat-sequence representation; the DSL's `>>>` desugars to `compose_trans`.

## The composition-rule hierarchy

The level says which operations the API makes available. Associativity
would permit regrouping `(f >> g) >> h` as `f >> (g >> h)`; an identity
would permit inserting a do-nothing arrow. For matrix-style composition,
associativity follows when the combining operations satisfy the relevant
semiring laws, including distributivity. Product and noisy-OR fail
distributivity, as the calculation in [Core Types](core.md#productfuzzyalgebra)
shows. The construction checks exercise selected inputs; the algebraic
equations specify the requirement across inputs.


Underneath `>>` sits the *composition rule* of the enclosing module. The library's hierarchy:

```
CompositionRule
    ├── BilinearForm     no associativity promise
    └── Semigroupoid     associative ⊗, no identity
            └── Algebra  associative + identity + meet + negate
```

The four `.qvr` level keywords select the level:

| Declaration | Required level | Available operations |
|---|---|---|
| `composition X [level=algebra]` | `Algebra` | `>>`, `@`, `identity(A)`, `f.dagger`, `f.trace(A)`, `cup(A)`, `cap(A)`, all compact-closed operations |
| `composition X [level=semigroupoid]` | `Semigroupoid` | `>>`, `@`, no identity or compact-closed operations |
| `composition X [level=bilinear_form]` | `BilinearForm` | `>>`, `@`, no associativity guarantee |
| `composition X [level=rule]` | `CompositionRule` | permissive; accepts any rule |

A typed `CompileError` flags every Algebra-only operation used inside a non-Algebra module. The diagnostic names the operation and the offending rule's level.

### User-defined inline rules

A composition rule can be defined inline. The body is a sequence of entries whose RHS is a `let`-arithmetic expression:

<!-- compile: false -->
```qvr
composition my_godel [level=algebra]
    tensor_op(a, b) = a * b
    join(t) = sum(t)
    unit = 1.0
    zero = 0.0

composition my_semi [level=semigroupoid]
    tensor_op(a, b) = (1 - a + a * b)
    join(t) = prod(t)

composition signed_dot [level=bilinear_form]
    tensor_op(a, b) = (a + b) * 0.5
    join(t) = sum(t)
```

Required entries per level:

| Level | Required | Optional |
|---|---|---|
| `algebra` | `tensor_op`, `join`, `unit`, `zero` | `negation`, `meet` |
| `semigroupoid` | `tensor_op`, `join` | |
| `bilinear_form` | `tensor_op`, `join` | |

`CustomAlgebra` runs a small sanity check at construction; `CustomSemigroupoid` runs an associativity smoke check on a fixed sample (skippable via `verify_associative=False` in the Python API). `CustomBilinearForm` runs no check, since the type opts out of the associativity promise.

### Material implication

The shipped `material_implication` rule is a `CustomSemigroupoid` with tensor $a \otimes b = 1 - a + a \cdot b$ (Reichenbach implication) and join $\bigvee = \prod$ (product reduction). Use it with the DSL keyword:

<!-- compile: false -->
```qvr
composition material_impl [level=semigroupoid]
object A : FinSet 3
object B : FinSet 3
morphism f : A -> B [role=latent]
morphism g : B -> B [role=latent]
define composed = f >> g
export composed
```

Compact-closed operations are unavailable; the module otherwise behaves normally.

## Operadic contractions

A [contraction](dsl-contractions.md#type-driven-wiring-inference) extends the same hidden-index calculation to several tables
at once. In `"sp,sq,pqo->so"`, `s` and `o` label the output, while `p`
and `q` range over intermediate choices. Each `(p,q)` supplies a product
of three entries, and the rule aggregates those contributions. *Operadic*
means that these operations have multiple typed inputs and one typed output;
you can use the wiring calculation without learning operad theory first.


Binary composition `f >> g` contracts two morphisms along a single shared axis. Many tensor-network patterns contract three or more inputs at a shared reduction; the `contraction` declaration provides this syntax.

### Wiring spec

The wiring spec is an einsum-style string with comma-separated input letter-tuples and an arrow to the output letters:

```
"sp, sq, pqo -> so"
```

Each input lists its axes; the arrow's right-hand side names the surviving output letters. Letters present in inputs but not in the output are contracted under the rule's `join`. Letters in the output must appear in at least one input.

### Declaration

<!-- compile: false -->
```qvr
contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : B * C -> D,
) : A -> D [rule=product_fuzzy, wiring="sp,sq,pqd->sd"]
```

The declared `op_apply` becomes callable from any expression site: `let combined = op_apply(arg1_morph, arg2_morph, kernel_morph)`. Each call checks the argument count against the declared input arity, and each argument's domain / codomain shape (by numel) against the declared per-input signature.

### Validation

The wiring spec is validated at declaration time:

- Each input must be non-empty.
- The low-level `EinsumWiring` API requires distinct letters within each input.
  The DSL's explicit wiring route additionally supports diagonal extraction;
  see [DSL Contractions](dsl-contractions.md).
- The output spec may not repeat a letter.
- Every output letter must appear in at least one input.

A spec violating any of these constraints raises `CompileError` at the declaration, not at the call site.

### Python API

`quivers.core.wiring` provides the corresponding Python functions:

<!-- python: skip -->
```python
from quivers.core.wiring import einsum_wiring, contract

wiring = einsum_wiring(PRODUCT_FUZZY, "sp, sq, pqo -> so")
out    = contract(wiring, arg1, arg2, kernel)
```

`EinsumWiring` accepts any `CompositionRule` instance, including material implication:

<!-- python: skip -->
```python
from quivers.core.algebras import material_implication

mi = material_implication()
wiring = einsum_wiring(mi, "ij, jk -> ik")
out    = wiring.apply(f_tensor, g_tensor)
```

## Further reading

- The [QVR tutorial chapter 7](../tutorials/qvr/07-categorical.md) introduces the categorical foundations of transformations and composition rules.
- The [Python tutorial chapters 6 and 7](../tutorials/python/06-first-class-trans.md) cover the Python API in detail.
- [Composition Rules](../semantics/composition-rules.md) gives the formal semantics.
- [Algebras and Base Change](../semantics/algebras.md) covers the eleven shipped algebras and the homomorphism registry.
