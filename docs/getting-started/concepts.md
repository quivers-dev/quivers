# Concepts

## 1. An object names the possible values

An *object* is the type of a value. The finite objects are instances of
[`FinSet`](../api/core/objects.md), a named finite set whose elements index one
axis of a tensor. Continuous objects, such as the real line or the probability
simplex, are [continuous spaces](../api/continuous/spaces.md). A *morphism*
`f : A -> B` relates each element of its domain `A` to each element of its
codomain `B`. Between finite sets it is stored as a tensor of shape
`(|A|, |B|)` that holds one weight for each pair.

The examples below use `Weather = {dry, rainy}`, `Route = {walk, bus}`, and
`Arrival = {on_time, late}`, with element 0 standing for the first label of
each set. Two morphisms give conditional probabilities:

| `f` | walk | bus |
|---|---:|---:|
| dry | 0.8 | 0.2 |
| rainy | 0.3 | 0.7 |

| `g` | on_time | late |
|---|---:|---:|
| walk | 0.9 | 0.1 |
| bus | 0.6 | 0.4 |

Objects combine. `Weather * Route` is the Cartesian product, whose elements are
pairs, and `Weather + Route` is the disjoint union, whose elements are tagged
values of either set. Both have four elements, and both are written the same
way in Python and in QVR. In Python, the one-element set `quivers.Unit` is the
domain of a morphism that takes no varying input. [Core types](../guides/core.md)
gives the tensor layout of each.

## 2. Composition eliminates an intermediate value

`f >> g` composes `f : A -> B` with `g : B -> C` into a morphism `A -> C` by
summing out the shared object `B`. Mathematics writes the same composite as
$g \circ f$. When the weights are probabilities and each row of `f` and `g`
sums to one, `f` and `g` are [Markov kernels](https://en.wikipedia.org/wiki/Markov_kernel)
between finite sets, and `f >> g` is their composite: the matrix product, which
gives the probability of each arrival given the weather with the route summed
out.

```python
import torch
from quivers import FinSet, observed
from quivers.core.algebras import MARKOV

Weather = FinSet(name="Weather", cardinality=2)
Route = FinSet(name="Route", cardinality=2)
Arrival = FinSet(name="Arrival", cardinality=2)

f = observed(Weather, Route,
             torch.tensor([[0.8, 0.2], [0.3, 0.7]]), algebra=MARKOV)
g = observed(Route, Arrival,
             torch.tensor([[0.9, 0.1], [0.6, 0.4]]), algebra=MARKOV)
result = (f >> g).tensor
assert torch.allclose(result, torch.tensor([[0.84, 0.16], [0.69, 0.31]]))
```

[`observed`](../api/core/morphisms.md) wraps a fixed tensor as a morphism, and
`MARKOV` selects ordinary multiplication and addition. `observed` does not check
that the rows sum to one. The [stochastic interface](../guides/stochastic.md)
provides morphisms whose rows are normalized by construction.

## 3. An algebra chooses how weights combine

Composition computes

$$
(f \mathbin{>>} g)(a, c) = \bigoplus_{b \in B} f(a, b) \otimes g(b, c),
$$

where $a$ is an element of `A`, $c$ an element of `C`, $\otimes$ the operation
that combines the weights along the path through $b$, and $\bigoplus$ the
operation that combines the paths. An [`Algebra`](../api/core/algebras.md)
fixes the two operations. The built-in ones include these:

| `Algebra` | $\otimes$ | $\bigoplus$ | Structure and typical use |
|---|---|---|---|
| `BOOLEAN` | AND | OR | Boolean semiring, for reachability |
| `MARKOV` | multiplication | sum | sum-product semiring, for probabilities |
| `TROPICAL` | addition | minimum | [tropical semiring](https://en.wikipedia.org/wiki/Tropical_semiring), for shortest paths |
| `LOG_PROB` | addition | log-sum-exp | [log semiring](https://en.wikipedia.org/wiki/Log_semiring), for probabilities in log space |
| `PRODUCT_FUZZY` | multiplication | noisy-OR | product [t-norm](https://en.wikipedia.org/wiki/T-norm) with the probabilistic sum |

The first four are [semirings](https://en.wikipedia.org/wiki/Semiring), as in
semiring parsing [@goodman1999semiring; @mohri2002semiring]. `PRODUCT_FUZZY`
pairs the product t-norm with its dual t-conorm, the probabilistic sum
[@klement2000triangular]. Over many arguments the probabilistic sum is the
*noisy-OR*, $1 - \prod_i (1 - x_i)$, the probability that at least one of
several independent causes occurs [@pearl1988probabilistic]. The
[algebra catalog](../semantics/algebras.md#1-the-eleven-algebras) lists every
built-in `Algebra`.

`PRODUCT_FUZZY` is the default. Composing `f` and `g` without
`algebra=MARKOV` gives $1 - (1 - 0.8 \times 0.1)(1 - 0.2 \times 0.4) = 0.1536$
for late arrival on a dry day, where `MARKOV` gives $0.16$. The noisy-OR
treats walking and taking the bus as independent causes of lateness, while the
sum treats them as exclusive alternatives.

Composition is associative, so that `(a >> b) >> c` and `a >> (b >> c)` agree,
when the two operations form a semiring, in which $\otimes$ distributes over
$\bigoplus$.
Product does not distribute over the probabilistic sum, so under the default
the two groupings can differ:

```python
S = FinSet(name="S", cardinality=2)
a = observed(S, S, torch.tensor([[0.9, 0.5], [0.2, 0.7]]))
b = observed(S, S, torch.tensor([[0.6, 0.8], [0.4, 0.3]]))
c = observed(S, S, torch.tensor([[0.5, 0.1], [0.9, 0.6]]))
left = ((a >> b) >> c).tensor
right = (a >> (b >> c)).tensor
assert not torch.allclose(left, right)
```

[Algebras and base change](../semantics/algebras.md) states which laws each
`Algebra` satisfies.

## 4. A probabilistic program describes how data could arise

A QVR `program` is a [generative model](https://en.wikipedia.org/wiki/Generative_model):
its steps say how the observed values could have been produced
[@goodman2014design]. Each step is one of the following
([step reference](../reference/qvr/probabilistic-programs.md#step-reference)):

- `sample` draws a latent variable from a distribution
- `let` computes a value from earlier ones without drawing
- `observe` scores an observed value under a distribution, which contributes
  the likelihood
- `score` adds an arbitrary log weight
- `marginalize` sums a discrete latent variable out of the likelihood by
  enumerating its values

Running a program forward draws each variable after the variables it depends
on, which is [ancestral sampling](https://en.wikipedia.org/wiki/Bayesian_network)
[@koller2009probabilistic]. Fitting it approximates the posterior over the
latent variables given the observed values, by variational inference or by
Markov chain Monte Carlo. The [quickstart](quickstart.md) fits a regression,
and [Your first model](../tutorials/qvr/01-first-model.md) works through the
same model in detail.

## 5. A distribution is more than one sample

A morphism between continuous spaces is a conditional distribution family, such
as [`ConditionalNormal`](../api/continuous/families.md), which assigns a
distribution over its codomain to each input. It offers two different
operations. `rsample` draws values, with gradients through the draw by the
[reparameterization trick](https://en.wikipedia.org/wiki/Reparameterization_trick)
where the family allows it [@kingma2014autoencoding]. `log_prob` evaluates the
log probability mass or log [density](https://en.wikipedia.org/wiki/Probability_density_function)
of a given value.

The two operations behave differently under composition. `rsample` on `f >> g`
draws an intermediate value from `f` and then a value from `g`. `log_prob` sums
out a discrete intermediate exactly. For a continuous random intermediate it
does not compute the marginal density, which would require an integral. It
scores the composite along a reference path instead, which agrees with the
marginal density when the intermediate is deterministic.
[Continuous spaces](../guides/continuous-spaces.md#composition-sampledcomposition)
gives the interface, and [Correspondence and limitations](../semantics/adequacy.md)
states what each operation computes.

## 6. Monads and handlers describe how computations continue

A [monad](https://en.wikipedia.org/wiki/Monad_(functional_programming)) is an
interface for sequencing computations that produce values in some context, here
a distribution over values [@moggi1991notions; @wadler1992essence]. Its *return*
wraps a value as a computation, and its *bind* runs a computation and passes
each possible result to the next step. The steps of a program are sequenced by
bind. [Kleisli composition](https://ncatlab.org/nlab/show/Kleisli+category) is
the composition of functions that return computations. For the
[Giry monad](https://ncatlab.org/nlab/show/Giry+monad), which sends a space to
the probability distributions on it, those functions are the Markov kernels
[@giry1982categorical]. The library also implements these interfaces in Python
([monads](../guides/monadic.md), [categorical structures](../guides/categorical.md)).

An *effect* is an operation a computation requests, such as drawing a sample,
and a *handler* answers the request
[@plotkin2009handlers; @bauer2015programming]. One handler may draw a fresh
value for a sampling request, another may return an observed value, and a third
may replay a value recorded earlier. The handler receives the
[continuation](https://en.wikipedia.org/wiki/Continuation), the rest of the
computation waiting for the value. QVR declares effects and handlers directly
([Effects and handlers](../tutorials/qvr/10-effects-and-handlers.md)), and the
[effects guide](../guides/effects.md) gives the Python interface.

## 7. Reading the notation and finding the next explanation

| Notation or term | Meaning | Defined in |
|---|---|---|
| $f : A \to B$ | A morphism with domain $A$ and codomain $B$ | [Morphisms](../guides/morphisms.md) |
| $\mathcal V$ | The weights of an `Algebra`, with its operations $\otimes$ and $\bigoplus$ | [Setting](../semantics/setting.md) |
| $\mathcal C(A, B)$, [hom-set](https://ncatlab.org/nlab/show/hom-set) | The morphisms from $A$ to $B$ in a category $\mathcal C$ | [Categorical structures](../guides/categorical.md) |
| [Functor](https://ncatlab.org/nlab/show/functor) | A map between categories that preserves identities and composition | [Functors](../api/categorical/functors.md) |
| [Natural transformation](https://ncatlab.org/nlab/show/natural+transformation) | A family of morphisms between two functors that commutes with every morphism | [Categorical structures](../guides/categorical.md) |
| $\llbracket e \rrbracket_\rho$, [denotation](https://en.wikipedia.org/wiki/Denotational_semantics) | The meaning of expression $e$ when its free names take the values that the environment $\rho$ gives | [Setting](../semantics/setting.md) |
| $\Gamma \vdash e : \tau$, [typing judgment](https://en.wikipedia.org/wiki/Type_system) | Under the typing context $\Gamma$, expression $e$ has type $\tau$ | [Typing](../semantics/typing.md) |
| Effect row | The effects a computation may still perform | [Effects and handlers](../reference/qvr/effects-and-handlers.md) |
| Guide | The family of distributions that variational inference fits to the posterior | [SVI](../guides/inference-svi.md) |
| [Plate](https://en.wikipedia.org/wiki/Plate_notation) | Conditionally independent draws, one for each element of an object | [Programs](../guides/programs.md) |
