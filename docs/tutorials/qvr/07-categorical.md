# 7. Categorical foundations

Four categorical constructs appear directly in QVR's type system: *objects*, *morphisms*, *algebras*, and *change-of-base*. They determine why a composition may raise `Source algebra 'ProductFuzzyAlgebra' does not match this morphism's algebra 'Real'` and what a call to `change_base` changes. This chapter gives their categorical interpretation.

This is optional reading. Skip it if the DSL is doing what you need.

## Begin with composable tables

Consider a table from people to activities and a table from activities to
outcomes. To find how a person relates to an outcome, inspect all routes
through an intermediate activity. The common activity set tells us which
entries connect. In `A -> B -> C`, both tables must use that same activity
set `B`; matching the number of rows and columns alone would also permit
accidentally connecting activities to an unrelated set of the same size.

The arithmetic determines the question being asked. [Boolean AND/OR](../../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine) asks
whether any route exists. Multiply/sum combines probabilities over mutually
exclusive intermediate choices. Add/min finds the cheapest route from costs.
The shared pattern is composition through an intermediate object; the algebra
supplies its interpretation. A category additionally requires identity and
associativity laws. Check those laws for the chosen operations before using
identity insertion or reassociation to simplify a computation.

## Why have types and algebras at all?

A typed signature like `program f : A -> B` exists so the compiler can reject programs that don't make sense before they hit a tensor. Two examples:

<!-- compile: false -->
```qvr
# A program declared with effects=[Pure], but the body contains `observe`:
program p : Item -> Item [effects=[Pure]]
    observe y : Item <- Normal(loc=0, scale=1)
    return y
# CompileError: line 2, col 4: program declared effects=[Pure]
#               but uses Score effect on `observe y`
```

<!-- compile: false -->
```qvr
# An algebra mismatch under composition:
composition product_fuzzy [level=algebra]
define pipeline = f >> g    # f is ProductFuzzy, g was change_based to Markov
# CompileError: incompatible algebras under `>>`: the left operand
#               is ProductFuzzyAlgebra, the right is Markov
```

Without the effect and algebra tags, the same programs would either run and silently produce nonsense, or fail somewhere deep in a tensor evaluation with a stack trace pointing at the wrong place. The category-theoretic reading below is *why* those tags compose cleanly; the practical payoff is the error messages.

## The setting in one paragraph

A QVR program denotes a morphism in a [$\mathcal{V}$-enriched category](https://ncatlab.org/nlab/show/enriched+category) in the sense developed by Kelly [@kelly1982basic]. *Enriched* means the hom-sets aren't sets, they're objects of a fixed algebra $\mathcal{V}$ called the *enrichment* or *base*. For QVR's category $\mathcal{V}\text{-}\mathbf{Rel}$ of $\mathcal{V}$-valued relations on finite sets, a morphism `f : A -> B` is concretely a tensor of shape `(|A|, |B|)` whose entries live in $\mathcal{V}$. The choice of $\mathcal{V}$ determines what those entries mean: probabilities, log-probabilities, real numbers, fuzzy truth values, Booleans, etc. Composition `f >> g` is parameterized by $\mathcal{V}$ via two operations: a binary tensor product (`a ⊗ b`) that combines entries pointwise, and a join (`⋁_i x_i`) that aggregates over the shared dimension. Different choices of $(\otimes, \bigvee)$ give different composition semantics.

## Algebras: the enrichment algebra

Quivers' [`Algebra`](../../api/core/algebras.md) interface supplies tensor, reduction, unit, zero, meet, and negation operations. The entries below provide different subsets of the lattice or quantale laws; consult the stated assumptions when choosing an interpretation.

| Name | $\otimes$ | $\bigvee$ | Identity | Reading |
|---|---|---|---|---|
| `Boolean` | AND | OR | true | Classical relations. |
| `ProductFuzzyAlgebra` | $a \cdot b$ | $1 - \prod(1-x_i)$ | 1 | Fuzzy "probabilistic AND / noisy-OR". |
| `Lukasiewicz` | $\max(0, a+b-1)$ | $\min(1, \sum x_i)$ | 1 | Probabilistic-sum t-conorm. |
| `Godel` | $\min(a, b)$ | $\max_i x_i$ | 1 | Min-max fuzzy logic. |
| `Tropical` | $a + b$ | $\min_i x_i$ | 0 | Shortest-path cost algebra. |
| `MaxPlus` | $a + b$ | $\max_i x_i$ | 0 | Viterbi / best-path cost algebra. |
| `LogProb` | $a + b$ | $\mathrm{logsumexp}_i$ | 0 | Numerically stable probability. |
| `Markov` | $a \cdot b$ | $\sum_i x_i$ | 1 | Markov-kernel composition (with row-stochasticity). |
| `Real` | $a \cdot b$ | $\sum_i x_i$ | 1 | Real sum-product. |
| `Probability` | $a \cdot b$ | $\sum_i x_i$ (clamped to [0,1]) | 1 | Bounded sum-product. |
| `Counting` | $a \cdot b$ | $\sum_i x_i$ | 1 | Nonnegative-integer counting. |

`composition product_fuzzy [level=algebra]` at the top of a `.qvr` file sets the enrichment for the module. Every `f >> g` composition uses the corresponding $(\otimes, \bigvee)$.

The hierarchy in `quivers.core.algebras` is:

```
CompositionRule
    ├── BilinearForm     (no associativity promise)
    └── Semigroupoid     (associative ⊗, no identity)
            └── Algebra  (associative ⊗ with identity, plus meet/negate)
```

Operations like `identity(A)`, `cup(A)`, `cap(A)`, `f.dagger`, and `f.trace(A)` live on `Algebra`. If your module declares `composition material_impl [level=semigroupoid]`, the compiler rejects `identity(A)` with a typed error pointing at the rule's level. Note that the shipped Reichenbach operation is not associative; its constructor skips the usual semigroupoid associativity check, so its composites must not be reassociated.

## Composition carries the enrichment

The two composition operators both defer to the operands' own algebra, fixed by the module-level `composition NAME [level=...]` declaration:

| Operator | Meaning |
|---|---|
| `>>` | Enriched relation composition in the operands' shared algebra |
| `<<` | The same composite written right to left: `g << f` is `f >> g` |

So `f >> g` composes in whatever enrichment both operands carry, and errors on a mismatch. There is no operator that silently pins a different enrichment; to cross enrichments, you [`change_base`](../../api/core/morphisms.md) one operand into the other's algebra before composing.

For the probability reading, let the two intermediate choices have weights
`0.25` and `0.75`, with onward probabilities `0.8` and `0.6`. The composite
entry is `0.25 * 0.8 + 0.75 * 0.6 = 0.65`. A fuzzy interpretation would use
its own reduction on the route contributions `0.2` and `0.45`; noisy-OR gives
`1 - (1 - 0.2) * (1 - 0.45) = 0.56`. The type checker prevents silently
mixing these two interpretations in one composition.

## Change of base

An `AlgebraHomomorphism` records a source algebra, a target algebra, and an entrywise map. `f.change_base(phi)` applies that map and updates the morphism's algebra tag. Particular instances may support a stronger categorical reading, but the runtime contract is the typed transformation just described.

The DSL provides a catalog of named homomorphisms (`expectation`, `log_prob`, `max_plus`, `material_implication`, `threshold`, `boolean_embedding`, ...) and a small set of *constructors* parameterized by an object or morphism (`softmax(B)`, `l1_normalize(B)`, `l2_normalize(B)`, `bayes_invert(prior)`). Each is a first-class transformation value: you can let-bind it, compose it with `>>>`, and pass it through `change_base`. Python API chapter 6 covers the complete interface; here is the short version:

```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=latent]

define s = softmax(B)
define pipeline = s >>> expectation
define g = f.change_base(pipeline)
```

`softmax(B)` builds a `MorphismTransformation` with `source = ProductFuzzyAlgebra, target = Markov`; `expectation` is a `AlgebraHomomorphism` with `source = Markov, target = ProductFuzzyAlgebra`. The composition `s >>> expectation` round-trips back to ProductFuzzyAlgebra and gives you a morphism whose rows are normalized on the way out.

## Probabilistic programs

An unscored `program` has the usual [Giry
monad](https://ncatlab.org/nlab/show/Giry+monad) interpretation as a Markov
kernel. `observe` and `score` extend this to an unnormalized nonnegative
measure because a density or explicit factor may exceed one. The operational
forms are:

- `sample v <- F(args)` draws from `F(args)` and extends the trace;
- `observe v <- F(args)` preserves the trace and adds the observed value's log
  density;
- `score v = expr` adds an explicit scalar log weight;
- `let v = expr` is a deterministic trace extension;
- finite-support `marginalize v : A <- F(args)` enumerates `v` within its
  indented scope and reduces the resulting weights; a non-enumerable family is
  sampled once rather than integrated continuously; and
- `return v` projects the weighted trace onto `v`.

The effect summaries `Sample`, `Score`, `Marginal`, and `Pure` are checked
against the body. The elaborated QIEC computation carries the more precise row
of lexical `Random` and `Score` instances.

## A worked example: regression as enriched morphism

The chapter-1 regression compiles to a weighted kernel whose trace contains
`beta_0`, `beta_1`, and `sigma`, and whose weight includes the observation
density. SVI optimizes a variational objective for its normalized posterior;
NUTS samples that posterior through the program's log joint. Neither algorithm
changes the checked program that defines the model.

## Further reading

The denotational [semantics](../../semantics/index.md) section gives a rigorous treatment of everything in this chapter and considerably more: the formal grammar, the categorical setting, the compositional semantics of every DSL construct, the adequacy theorem connecting compiler implementation to denotation. It assumes Kelly-level enriched category theory.

The Python API tutorials (chapters 6 and 7) go further on first-class transformations and the composition-rule hierarchy.

The guides under `guides/` cover individual feature areas at the level of "what's the API and how do I use it" (categorical, monadic, enriched, deduction, structural-compression).

## You're done

At this point, you can write models in QVR, fit them with the shipped inference
algorithms, marginalize discrete latents, build hierarchical and
sequence-shaped models, and read the categorical structure underneath when you
need to. Chapters 8–13 continue with diagnostics, indexed data, effects,
generated computation graphs, and release checks. Suggested next stops:

- The [examples gallery](../../examples/index.md) for end-to-end model code (Bayesian regression, mixture models, VAE, transformer, vanilla RNN).
- The [QVR language reference](../../reference/qvr/index.md) for the current syntax and execution behavior.
- The [inference benchmark report](../../developer/inference-benchmarks.md) for the empirical truth-table of which algorithm fits which problem.


## Sources and further reading

- *A categorical approach to probability theory* [@giry1982categorical].
