# 6. First-class transformations

A *change-of-base* transformation turns a $\mathcal{V}$-enriched morphism into a $\mathcal{W}$-enriched one. Two variants coexist in the library:

- [`AlgebraHomomorphism`](../../api/core/algebras.md) values [@rosenthal1996quantaloids] record typed pointwise maps $\varphi : \mathcal{V} \to \mathcal{W}$: every entry of the morphism's tensor is sent through $\varphi$. A lax monoidal or lattice interpretation requires checking the appropriate laws for that particular map.
- [`MorphismTransformation`](../../api/core/morphisms.md) values act on the whole tensor. Softmax and L1/L2 normalization use row totals or norms; Bayes inversion combines a prior with likelihood entries before normalizing over the input axis.

Both inherit a common interface: a `.source` algebra, a `.target` algebra, and
an `apply` method that accepts a tensor plus a morphism for shape resolution.
The Python API treats transformations as first-class values: you can bind them
to local names, compose them with
[`compose_trans`](../../api/core/algebras.md), written as `>>>` in QVR, and pass
either kind to
[`Morphism.change_base`](../../api/core/morphisms.md).

## Read a transformation as a change in interpretation

A row `[2, 1]` can be a pair of real scores, but it cannot be a probability
row because its entries sum to three. [L1 normalization](02-stochastic-relations.md#creating-stochastic-morphisms) gives `[2/3, 1/3]` for
this nonnegative row. Softmax instead exponentiates its inputs before
normalizing, so it answers a different scoring convention. Both need to see
the whole row; an entrywise map such as taking a logarithm does not.

The type records which arithmetic later composition should use. Normalizing
`[2, 1]` loses its total of three: `[4, 2]` produces the same normalized row.
Changing the result's algebra tag back thus leaves `[2/3, 1/3]` in place.
For a conversion to preserve composition, it must also satisfy the relevant
preservation equations described below.

### Why typed transformations?

In PyTorch, `softmax(logits, dim=-1)` normalizes each row. The returned tensor
contains the values but carries no record of the operations used to interpret
them. A probability calculation sums contributions from mutually exclusive
paths; a fuzzy calculation may combine them with noisy-OR. Recording the
algebra with the tensor lets Quivers check which operations a subsequent
composition expects.

A [`MorphismTransformation`](../../api/core/morphisms.md) is a typed function on morphisms. `softmax(B)` records `ProductFuzzyAlgebra` as its source and `Markov` as its target, so the compiler can check subsequent compositions. Applying it to a morphism `f` returns `f.change_base(softmax(B))` with the algebra tag `Markov`. A later `>>` accepts another Markov morphism and rejects a fuzzy one because the two operands carry different algebras.

The algebra tag records the result's composition semantics and is consumed by later type checks.

## The catalog

```python
from quivers.core.algebra_morphisms import (
    EXPECTATION, MATERIAL_IMPLICATION,
    PROBABILITY_CLAMP, PROBABILITY_TO_REAL,
    COUNTING_FROM_REAL, COUNTING_TO_REAL,
    IdentityHom, Threshold, Embedding,
)
from quivers.core.morphism_transformations import (
    softmax, l1_normalize, l2_normalize, bayes_invert,
    Softmax, L1Normalize, L2Normalize, BayesInvert,
)
```

The shipped *singletons* (no arguments needed):

| Name | `.source` | `.target` | What it does |
|---|---|---|---|
| `EXPECTATION` | `Markov` | `ProductFuzzyAlgebra` | Reinterpret a row-stochastic kernel as fuzzy membership. |
| `MATERIAL_IMPLICATION` | `ProductFuzzyAlgebra` | `Godel` | Reichenbach implication lift. |
| `PROBABILITY_CLAMP` | `Real` | `Probability` | Clamp real entries to `[0, 1]`. |
| `PROBABILITY_TO_REAL` | `Probability` | `Real` | Forget the `[0, 1]` constraint. |
| `COUNTING_FROM_REAL` | `Real` | `Counting` | Floor real entries and clamp them to non-negative integers. |
| `COUNTING_TO_REAL` | `Counting` | `Real` | Embed counts as reals. |

The shipped *constructors* (one argument):

| Constructor | Argument | `.source` | `.target` |
|---|---|---|---|
| `softmax(axis)` | `SetObject` | `ProductFuzzyAlgebra` | `Markov` |
| `l1_normalize(axis)` | `SetObject` | `Real` | `Markov` |
| `l2_normalize(axis)` | `SetObject` | `Real` | `Real` |
| `bayes_invert(prior)` | `Morphism` or `Tensor` | `Markov` | `Markov` |

## Applying one transformation

```python
import torch
from quivers import FinSet, observed
from quivers.core.morphism_transformations import softmax

A = FinSet(name="A", cardinality=3)
B = FinSet(name="B", cardinality=4)
f = observed(A, B, torch.rand(3, 4))      # ProductFuzzyAlgebra-enriched

phi = softmax(B)
g = f.change_base(phi)                     # Markov-enriched
print(g.tensor.sum(dim=-1))                # tensor([1., 1., 1.])
print(g.algebra.name)                     # 'MarkovAlgebra'
```

The result is an `ObservedMorphism` over `Markov`. The transformation is applied to the tensor; the domain and codomain object are preserved for shape-aware transformations like softmax (Bayes inversion is the exception: it swaps them).

## Composing transformations

In Python the call site is
[`compose_trans`](../../api/core/algebras.md); `.qvr` source uses the `>>>`
operator. Both combine two or more
[`MorphismTransformation`](../../api/core/morphisms.md) or
[`AlgebraHomomorphism`](../../api/core/algebras.md) values into a sequence. The
compose-time check verifies that `t1.target` matches `t2.source` and raises a
typed error when they differ.

```python
from quivers.core.algebra_morphisms import EXPECTATION
from quivers.core.morphism_transformations import softmax
from quivers.core.trans import compose_trans

phi   = softmax(B)                # ProductFuzzyAlgebra -> Markov
psi   = EXPECTATION               # Markov       -> ProductFuzzyAlgebra
pipe  = compose_trans(phi, psi)   # ProductFuzzyAlgebra -> ProductFuzzyAlgebra

g = f.change_base(pipe)
print(g.algebra.name)            # back to 'ProductFuzzyAlgebra'
```

If the boundary types do not match (for instance,
`compose_trans(softmax(B), PROBABILITY_TO_REAL)` would try to map `Markov` to
`Probability`), `compose_trans` raises a `TypeError` naming the mismatch.

`compose_trans` returns a [`TransSeq`](../../api/core/algebras.md) value: a flattened sequence of single steps. Calling `change_base` on a `TransSeq` iterates the steps, applying each in turn. Nested compositions flatten so the result is always a flat tuple of steps.

## Three-step pipelines

```python
pipe = compose_trans(
    softmax(B),
    EXPECTATION,
    softmax(B),
)
g = f.change_base(pipe)
```

`compose_trans` takes any number of arguments. Each adjacent pair is type-checked at compose time.

## Bayes inversion

`bayes_invert(prior)` is the one constructor whose argument is a *morphism* rather than an *object*. It builds a `BayesInvert` transformation parameterized by the prior; the transformed kernel is the Bayes-inverted posterior under that prior. Domain and codomain swap.

```python
import torch
from quivers import FinSet, observed, MARKOV
from quivers.core.morphism_transformations import bayes_invert

Unit = FinSet(name="Unit", cardinality=1)
A    = FinSet(name="A", cardinality=3)

prior_tensor = torch.tensor([[0.5, 0.3, 0.2]])
prior        = observed(Unit, A, prior_tensor, algebra=MARKOV)

f = observed(A, A, torch.rand(3, 3), algebra=MARKOV)
g = f.change_base(bayes_invert(prior))
print(g.tensor.sum(dim=-1))      # rows of g sum to 1 (Markov)
```

For instance, suppose a condition has prior probability `0.1`, a test is
positive with probability `0.8` when it holds, and with probability `0.2` when
it does not. A positive test has joint weights `0.08` and `0.18` for the two
possibilities. Dividing by their sum gives posterior probability
`0.08 / 0.26`, about `0.308`, for the condition. Bayes inversion needs that
prior and swaps the direction of the kernel; transposing a probability table
alone does neither.

## From QVR

Inside `.qvr` files, the corresponding forms are the `change_base(t)` postfix
and the `>>>` operator:

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

A `let` binding with a transformation-valued right-hand side enters the
compiler's transformation namespace, which is distinct from the morphism
namespace. The [QVR categorical tutorial](../qvr/07-categorical.md) describes
the QVR syntax; the Python API provides the same operations directly.

## Composition checks

Transformations can be bound to local names, passed to functions, and composed into pipelines. `compose_trans` checks that each step's target algebra matches the next step's source before applying the pipeline to a tensor.

## Try this

- Build a pipeline that round-trips a `ProductFuzzyAlgebra` morphism through `Markov` and back: `compose_trans(softmax(B), EXPECTATION)`. Verify the source and target match.
- Apply the same pipeline to two different morphisms `f` and `h`; confirm the result has the same algebra for both.
- Construct an invalid pipeline (`compose_trans(softmax(B), PROBABILITY_TO_REAL)`) and observe the `TypeError`.

## Next

[Tutorial 7](07-composition-rules.md) introduces the
`CompositionRule → BilinearForm | Semigroupoid → Algebra` hierarchy and the
`EinsumWiring` API for operadic n-ary contractions.
