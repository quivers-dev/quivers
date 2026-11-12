# 7. Categorical foundations

QVR types its declarations by objects and morphisms, composes morphisms under a chosen pair of operations, and converts a morphism from one pair of operations to another with `change_base`. In the terms of [category theory](https://ncatlab.org/nlab/show/category), the morphisms of a module are matrices in a category of matrices, errors such as `incompatible algebras: ProductFuzzyAlgebra() and MarkovAlgebra()` come from composing matrices under different operations, and a probabilistic program denotes a [Markov kernel](https://en.wikipedia.org/wiki/Markov_kernel).

## Morphisms as matrices

A *category* has objects, morphisms between them, an associative composition of morphisms, and an identity morphism on each object [@maclane1998categories]. In QVR an object `A : FinSet n` is a finite set of size $n$, and a morphism `f : A -> B` is a tensor of shape `(|A|, |B|)` whose entries lie in a set $V$ of weights. Composition is the matrix product over two operations on $V$, a product $\otimes$ and a sum $\oplus$:

$$
(f \mathbin{>\!\!>} g)[a, c] = \bigoplus_{b \in B} f[a, b] \otimes g[b, c].
$$

In the Python API the codomain of `f` must equal the domain of `g`, and two [`FinSet`](../../api/core/objects.md) objects are equal only when their names and sizes agree. The QVR compiler compares finite sets by size, so in a `.qvr` file `f : A -> B` and `g : C -> D` compose whenever `B` and `C` have the same size.

The choice of $(\otimes, \oplus)$ fixes what composition computes. With AND and OR it says whether a path exists, with product and sum it adds the probabilities of mutually exclusive routes, and with $+$ and $\min$ it finds the cheapest route. When $(V, \oplus, \otimes)$ is a [semiring](https://en.wikipedia.org/wiki/Semiring), the $V$-valued matrices form a category, often written $\mathbf{Mat}(V)$, because distributivity makes the matrix product associative and the identity matrix is a unit [@mohri2002semiring; @hofmann2014monoidal]. When $V$ is a [quantale](https://ncatlab.org/nlab/show/quantale), a complete lattice with an associative product that distributes over arbitrary joins, the matrices form a [quantaloid](https://ncatlab.org/nlab/show/quantaloid) [@rosenthal1996quantaloids]. Without these laws, rebracketing a composite or inserting an identity may change the result, so check the laws of the chosen operations before simplifying a computation. The [semantics](../../semantics/setting.md#2-v-enriched-relations) states the construction in full.

A $\mathcal V$-[enriched category](https://ncatlab.org/nlab/show/enriched+category) in Kelly's sense has, in place of each hom-set (the set of morphisms between two objects), a hom-object in a monoidal category $\mathcal V$ [@kelly1982basic]. The category of $V$-matrices is not enriched over $V$ in that sense: each of its hom-sets is a set of matrices. For a quantale $V$, each hom-set is itself a complete lattice and composition preserves joins in each argument, which is what makes the category a quantaloid, that is, a category enriched over complete lattices.

## Typed errors

Types and composition rules let the compiler reject two kinds of mistake before any tensor is evaluated. A program declared `Pure` whose body observes:

<!-- compile: false -->
```qvr
object Item : FinSet 3
program p : Item -> Item [effects=[Pure]]
    observe y : Item <- Normal(loc=0.0, scale=1.0)
    return y
export p
# CompileError: line 2, col 0: program 'p' is declared as Pure but body uses effects ['Score']
```

A composite of morphisms with different weights:

<!-- compile: false -->
```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
object C : FinSet 2
morphism f : A -> B [role=latent]
morphism h : B -> C [role=latent]
define g = h.change_base(softmax(C))
define pipeline = f >> g
export pipeline
# CompileError: line 8, col 18: incompatible algebras: ProductFuzzyAlgebra() and MarkovAlgebra()
```

`f` has entries in $[0, 1]$ composed by noisy-OR, while `g` is a row-stochastic matrix composed by sum and product. Without the check, `f >> g` would run and mix the two readings.

## Composition rules and their laws

An [`Algebra`](../../api/core/algebras.md) gives $\otimes$, $\oplus$ (the method `join`), the unit of $\otimes$, a zero, a meet, and a negation. The library checks none of the laws, so each instance below is named by the structure it is:

| Name | $\otimes$ | $\oplus$ | Unit | Structure |
|---|---|---|---|---|
| `Boolean` | AND | OR | true | Boolean semiring. Its matrices are relations. |
| `ProductFuzzy` | $a b$ | $1 - \prod_i (1 - x_i)$ | 1 | [Product t-norm](https://en.wikipedia.org/wiki/T-norm) with the probabilistic sum (noisy-OR) [@klement2000triangular; @pearl1988probabilistic]. Not a semiring. |
| `Lukasiewicz` | $\max(0, a + b - 1)$ | $\min(1, \sum_i x_i)$ | 1 | Łukasiewicz t-norm with the bounded sum [@klement2000triangular]. Not a semiring. |
| `Godel` | $\min(a, b)$ | $\max_i x_i$ | 1 | Minimum (Gödel) t-norm with the maximum, a semiring on $[0, 1]$. |
| `Tropical` | $a + b$ | $\min_i x_i$ | 0 | [Tropical (min-plus) semiring](https://en.wikipedia.org/wiki/Tropical_semiring) on $[0, \infty]$: cheapest paths [@mohri2002semiring]. |
| `MaxPlus` | $a + b$ | $\max_i x_i$ | 0 | Max-plus semiring: the best path, as in the Viterbi algorithm [@goodman1999semiring]. |
| `LogProb` | $a + b$ | $\mathrm{logsumexp}_i\, x_i$ | 0 | [Log semiring](https://en.wikipedia.org/wiki/Log_semiring): sum and product of probabilities in log space [@mohri2002semiring]. |
| `Markov` | $a b$ | $\sum_i x_i$ | 1 | Sum-product semiring, used for row-stochastic matrices. |
| `Real` | $a b$ | $\sum_i x_i$ | 1 | The ring $(\mathbb R, +, \times)$. |
| `Probability` | $a b$ | $\min(1, \sum_i x_i)$ | 1 | Product with a capped sum on $[0, 1]$. Not a semiring. |
| `Counting` | $a b$ | $\sum_i x_i$ | 1 | Counting semiring on $\mathbb N$: the number of paths [@goodman1999semiring]. |

`composition product_fuzzy [level=algebra]` at the top of a `.qvr` file sets the operations for every `>>` in the module. The `level` states which laws the rule claims, unchecked:

```
CompositionRule
    ├── Magmoid           (level bilinear_form: no associativity or identity)
    ├── Semigroupoid      (level semigroupoid: associative, no identity)
    ├── Algebra           (level algebra: identity, unit, zero, meet, and negation)
    └── Semiring          (an Algebra and a Semigroupoid: associative with identity)
            └── IdempotentSemiring  (idempotent sum)
```

Under a rule declared `semigroupoid` the morphisms form a [semicategory](https://ncatlab.org/nlab/show/semicategory), a category without identities [@tilson1987categories]. Under `bilinear_form` composition has no laws at all (the structure is a [magmoid](https://ncatlab.org/nlab/show/magmoid)). The eight semirings in the table are of class `Semiring`, so either level accepts them, and the other three are of class `Algebra` only. `identity(A)` and the operations `cup(A)`, `cap(A)`, `f.dagger`, and `f.trace(A)`, which bend, transpose, and loop the wires of a [compact closed category](https://ncatlab.org/nlab/show/compact+closed+category), need an `Algebra`. Under `composition material_impl [level=bilinear_form]`, `identity(A)` fails with

```text
CompileError: line 3, col 11: identity: requires an algebra-level composition rule (one with an identity, a unit, and a zero); the module declares the bilinear_form 'MaterialImplication', which has none. Declare an algebra, as in `composition product_fuzzy [level=algebra]`, or drop the identity operation.
```

The rule `material_impl` uses the Reichenbach implication $a \to b = 1 - a + ab$ [@baczynski2008fuzzy] as $\otimes$ and the product as $\oplus$. That $\otimes$ is not associative, so the rule is a `Magmoid`, and composites under it must not be rebracketed. [Chapter 7 of the Python track](../python/07-composition-rules.md) covers the hierarchy in detail.

## Composition operators

| Operator | Meaning |
|---|---|
| `>>` | Composition in the operands' shared composition rule. |
| `<<` | The same composite written right to left: `g << f` is `f >> g`. |

`f >> g` fails when the operands' rules differ. No operator changes the rule silently. To compose across rules, apply [`change_base`](../../api/core/morphisms.md) to one operand first.

## Change of base

`f.change_base(phi)` applies a map `phi` to the tensor of `f` and retags the result with the target rule. `phi` is one of two kinds:

- An *entrywise map* (class `CarrierMap`) applies a function $\varphi : V \to W$ to each entry. When $\varphi$ is monotone and lax for $\otimes$, $\oplus$, and the unit, it induces [change of base](https://ncatlab.org/nlab/show/change+of+enriching+category) in Kelly's sense [@kelly1982basic]. The named maps include `expectation` (from `Markov` to `ProductFuzzy`, reading each probability as a degree of membership), `log_prob`, `max_plus`, `threshold`, and `boolean_embedding`.
- A [`MorphismTransformation`](../../api/core/transformations.md) rewrites the whole tensor, for instance normalizing each row. The constructors `softmax(B)`, `l1_normalize(B)`, and `l2_normalize(B)` normalize along the axis of `B`, and `bayes_invert(prior)` computes a Bayesian inversion [@cho2019disintegration]. A row normalization is not a change of base.

Both kinds are values: a `define` can bind them, `>>>` composes them, and `change_base` applies them:

```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=latent]

define s = softmax(B)
define pipeline = s >>> expectation
define g = f.change_base(pipeline)
```

`softmax(B)` maps `ProductFuzzy` to `Markov`, normalizing each row, and `expectation` maps `Markov` back to `ProductFuzzy`. `g` thus has the rule of `f`, with each row normalized to sum to one. [Chapter 6 of the Python track](../python/06-first-class-trans.md) gives the full interface.

## Probabilistic programs as kernels

A *Markov kernel* from $X$ to $Y$ assigns to each $x$ a probability distribution on $Y$. Markov kernels are the morphisms of the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://ncatlab.org/nlab/show/Giry+monad), which sends a space to the space of probability distributions on it [@giry1982categorical]. A `program` without `observe` or `score` denotes such a kernel. `observe` and `score` multiply the result by a density or by $\exp(w)$, which may exceed one, so a program that uses them denotes a kernel into unnormalized measures (an [s-finite kernel](../../semantics/setting.md#4-finite-and-s-finite-kernels)) [@staton2017commutative]. Each step acts on the running set of bindings and its log weight:

- `sample v <- F(args)` draws `v` from `F(args)`.
- `observe v <- F(args)` keeps the observed value of `v` and adds its log density to the log weight.
- `score w = expr` adds `expr` to the log weight.
- `let v = expr` binds `v` to a deterministic value.
- `marginalize v : A <- F(args)` runs its indented body once for each element of the finite set `A` and combines the resulting log weights ([chapter 4](04-marginalize.md)).
- `return v` keeps `v` and discards the other bindings.

For the regression of [chapter 1](01-first-model.md), the kernel draws `sigma`, `beta_0`, and `beta_1`, and its weight is the Normal density of the observed `y`. Stochastic variational inference (SVI) fits a guide to the normalized posterior, and the No-U-Turn Sampler (NUTS) samples it through the log joint density. Neither changes the program that defines the model. The compiler checks the effect summary (`Sample`, `Score`, `Marginal`, `Pure`) against the body, and [chapter 10](10-effects-and-handlers.md) introduces the finer effect rows of the core language.

## Next

[Chapter 8](08-diagnostics-and-comparison.md) checks convergence, compares the data with posterior predictive draws, and compares models by leave-one-out cross-validation. The [semantics](../../semantics/index.md) gives the denotation of every construct, and the [examples gallery](../../examples/index.md) gives complete models.
