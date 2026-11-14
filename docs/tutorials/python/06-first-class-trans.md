# 6. First-class transformations

A morphism carries its algebra, the operations under which it composes. Converting a morphism from one algebra $\mathcal V$ to another $\mathcal W$ changes both its entries and that tag. The library has two kinds of conversion, both [`quivers.core`](../../api/core/transformations.md) classes:

- An *entrywise map* (class `CarrierMap`) applies a function $\varphi : \mathcal V \to \mathcal W$ to each entry. When $\varphi$ is monotone and lax for $\otimes$, $\oplus$, and the unit, it induces [change of base](https://ncatlab.org/nlab/show/change+of+enriching+category) in the sense of enriched category theory [@kelly1982basic]. The library does not check these laws.
- A *morphism transformation* (class `MorphismTransformation`) rewrites the whole tensor. Softmax and L1 and L2 normalization divide each row by a total or a norm, and Bayes inversion combines a prior with the kernel before normalizing over the input.

Both have a `.source` algebra, a `.target` algebra, and an `apply` method. Both are values: a variable can hold one, [`compose_trans`](../../api/core/transformations.md) (written `>>>` in QVR) composes them, and [`Morphism.change_base`](../../api/core/morphisms.md) applies either kind.

## Read a transformation as a change in interpretation

The row `[2, 1]` can hold two real scores but not a probability distribution, since it sums to three. [L1 normalization](02-stochastic-relations.md#creating-stochastic-morphisms) gives `[2/3, 1/3]`, and softmax, which exponentiates before normalizing, gives a different row. Both read the whole row, whereas an entrywise map such as the logarithm does not. Normalization also discards the total: `[4, 2]` normalizes to the same row, so retagging the result to the old algebra leaves `[2/3, 1/3]`, not `[2, 1]`.

In PyTorch, `softmax(logits, dim=-1)` returns the normalized values with no record of how they are to be combined. A probability calculation sums the contributions of mutually exclusive paths, whereas a fuzzy calculation combines them by noisy-OR. Since a morphism records its algebra, a later `>>` can check that both operands use the same operations. `softmax(B)` has source `ProductFuzzy` and target `Markov`, so `f.change_base(softmax(B))` is a `Markov` morphism, which composes with another `Markov` morphism and not with a `ProductFuzzy` one.

## The catalog

```python
from quivers.core.algebra_morphisms import (
    MARKOV_TO_PRODUCT_FUZZY, PRODUCT_FUZZY_TO_GODEL,
    PROBABILITY_CLAMP, PROBABILITY_TO_REAL,
    COUNTING_FROM_REAL, COUNTING_TO_REAL,
    IdentityHom, Threshold, Embedding,
)
from quivers.core.morphism_transformations import (
    softmax, l1_normalize, l2_normalize, bayes_invert,
    Softmax, L1Normalize, L2Normalize, BayesInvert,
)
```

Entrywise maps that take no arguments:

| Name | `.source` | `.target` | What `apply` does to each entry |
|---|---|---|---|
| `MARKOV_TO_PRODUCT_FUZZY` | `Markov` | `ProductFuzzy` | Clamps to $[0, 1]$, so a probability is read as a degree of membership. |
| `PRODUCT_FUZZY_TO_GODEL` | `ProductFuzzy` | `Godel` | Clamps to $[0, 1]$. Later composition uses minimum and maximum. |
| `PROBABILITY_CLAMP` | `Real` | `Probability` | Clamps to $[0, 1]$. |
| `PROBABILITY_TO_REAL` | `Probability` | `Real` | Leaves it unchanged. |
| `COUNTING_FROM_REAL` | `Real` | `Counting` | Clamps below at zero and rounds down. |
| `COUNTING_TO_REAL` | `Counting` | `Real` | Converts it to a float. |

Morphism transformations, each built from one argument:

| Constructor | Argument | `.source` | `.target` |
|---|---|---|---|
| `softmax(axis)` | `SetObject` | `ProductFuzzy` | `Markov` |
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
f = observed(A, B, torch.rand(3, 4))      # ProductFuzzy morphism

phi = softmax(B)
g = f.change_base(phi)                     # Markov morphism
print(g.tensor.sum(dim=-1))                # tensor([1., 1., 1.])
print(g.algebra.name)                      # 'Markov'
```

The result is an observed morphism over `Markov`. A transformation along an axis keeps the domain and codomain, and Bayes inversion swaps them.

## Composing transformations

[`compose_trans`](../../api/core/transformations.md) combines two or more transformations of either kind into a sequence, and checks that the target of each step is the source of the next:

```python
from quivers.core.algebra_morphisms import MARKOV_TO_PRODUCT_FUZZY
from quivers.core.morphism_transformations import softmax
from quivers.core.trans import compose_trans

phi   = softmax(B)                # ProductFuzzyAlgebra -> Markov
psi   = MARKOV_TO_PRODUCT_FUZZY   # Markov       -> ProductFuzzyAlgebra
pipe  = compose_trans(phi, psi)   # ProductFuzzyAlgebra -> ProductFuzzyAlgebra

g = f.change_base(pipe)
print(g.algebra.name)            # 'ProductFuzzy'
```

`compose_trans(softmax(B), PROBABILITY_TO_REAL)` raises

```text
TypeError: compose_trans: target of step 0 ('Markov') does not match source of step 1 ('Probability')
```

`compose_trans` returns a [`TransSeq`](../../api/core/transformations.md), a flat tuple of single steps: nested compositions are flattened, and `change_base` applies the steps in order. Any number of steps may be composed:

```python
pipe = compose_trans(
    softmax(B),
    MARKOV_TO_PRODUCT_FUZZY,
    softmax(B),
)
g = f.change_base(pipe)
```

## Bayes inversion

Given a prior $p(a)$ on $A$ and a kernel $f(b \mid a)$ from $A$ to $B$, the Bayesian inversion of $f$ is the kernel from $B$ to $A$ given by Bayes' rule, $f^\dagger(a \mid b) \propto p(a) f(b \mid a)$ [@cho2019disintegration]. `bayes_invert(prior)` is the transformation that computes it, and it swaps the domain and codomain:

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

The transpose of a kernel is not its inversion: inversion needs the prior, and its rows are normalized over the input.

## From QVR

In a `.qvr` file, `change_base` is a method of a morphism and `>>>` composes transformations:

```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=parameter]

define s    = softmax(B)
define pipe = s >>> expectation
define g    = f.change_base(pipe)
export g
```

A `define` whose right side is a transformation binds a name for a transformation, kept apart from the names of morphisms. [Chapter 7 of the QVR track](../qvr/07-categorical.md#change-of-base) describes the QVR forms.

## Try this

- Compose `softmax(B)` with `MARKOV_TO_PRODUCT_FUZZY`, apply the result to two different morphisms, and check that both results are `ProductFuzzy` morphisms whose rows sum to one.
- Build `compose_trans(softmax(B), PROBABILITY_TO_REAL)` and read the `TypeError`.

## Next

[Chapter 7](07-composition-rules.md) covers composition rules with fewer laws than an `Algebra` and contractions of several tensors.
