# From tables and functions to probabilistic programs

Quivers lets us describe computations in terms of their inputs, outputs, and
how they combine. The categorical vocabulary names patterns in those
computations. We can understand the patterns by calculating with small tables
before learning the general definitions.

This page assumes familiarity with functions, arrays, and elementary
probability. First we compose weighted tables; then we distinguish simulation
from inference; finally we connect those operations to categories, monads,
and effects. The linked guides retain the full interfaces and mathematical
assumptions.

## 1. An object names the possible values

Suppose a model relates weather, travel routes, and arrival times. We use
three [finite sets](../guides/core.md#objects-finite-sets): `Weather = {dry, rainy}`, `Route = {walk, bus}`, and
`Arrival = {on_time, late}`. These sets are **objects**. A `FinSet` records a
name and cardinality; in tensor computations its elements are represented by
indices. Here index 0 represents dry weather and index 1 represents rain.
The string labels are our interpretation of those indices.

An arrow `f : Weather -> Route`, called a **morphism**, specifies a relation
between two objects. The object before the arrow is its **domain**; the
object after it is its **codomain**. For a finite weighted relation, a tensor
stores one weight per input–output pair. Here its shape is `(2, 2)`.

| `f` | walk | bus |
|---|---:|---:|
| dry | 0.8 | 0.2 |
| rainy | 0.3 | 0.7 |

If these rows are probabilities, the dry row means “given dry weather, choose
walking with probability 0.8 and the bus with probability 0.2.” An ordinary
function would choose exactly one route for each weather value. A stochastic
morphism instead specifies a distribution of routes. Other weighted
morphisms can store costs, truth values, or compatibility scores.

Objects also describe structured inputs. `Weather * Route` contains pairs
and has four elements. `Weather + Route` contains tagged alternatives and
also has four elements, but a value is either a weather value or a route
value. A product input thus supplies both pieces of information; a sum
input supplies one piece and a tag identifying its kind. `Unit`
contains one value: an arrow from `Unit` needs no varying input. See
[Core types](../guides/core.md) for their representations.

## 2. Composition eliminates an intermediate value

Now let `g : Route -> Arrival` give the following conditional probabilities:

| `g` | on_time | late |
|---|---:|---:|
| walk | 0.9 | 0.1 |
| bus | 0.6 | 0.4 |

We want the probability of being late given dry weather, without retaining
which route was taken. There are two mutually exclusive possibilities:

1. Walk and arrive late: $0.8 \times 0.1 = 0.08$.
2. Take the bus and arrive late: $0.2 \times 0.4 = 0.08$.

Adding gives $0.16$. Applying the same calculation to every pair gives

$$
f g =
\begin{pmatrix}0.8 & 0.2\\0.3 & 0.7\end{pmatrix}
\begin{pmatrix}0.9 & 0.1\\0.6 & 0.4\end{pmatrix}
=\begin{pmatrix}0.84 & 0.16\\0.69 & 0.31\end{pmatrix}.
$$

Quivers writes sequential composition as `f >> g`. Mathematics often writes
the same order as $g\circ f$, read “$g$ after $f$.” More explicitly,

$$
(g\circ f)(w,a)=\sum_{r\in\mathrm{Route}}f(w,r)g(r,a).
$$

Here $w$ selects the input row, $a$ selects the output column, and $r$ ranges
over the intermediate routes. The shared `Route` axis disappears. This is
both matrix multiplication and the law of total probability, under the
model's assumption that arrival depends on weather through the route.

The following executable example makes the choice of arithmetic explicit:

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

`MARKOV` selects multiplication and summation. In this example we supplied
rows that already sum to one. To represent conditional probabilities, check
that each entry is nonnegative and each row sums to one. The [stochastic interface](../guides/stochastic.md) adds
the probability-specific behavior.

## 3. An algebra chooses how weights combine

The [built-in algebra catalog](../semantics/algebras.md#1-the-eleven-algebras) records the operations behind each interpretation. The [tropical example](../guides/core.md#tropicalalgebra) implements the shortest-path arithmetic used below.

The same pattern answers other questions if we change what the weights mean.
For reachability, combine consecutive edges with AND and alternative paths
with OR. For shortest paths, add the costs along a path and choose the
minimum across alternatives. For instance, routes with costs $5+4$ and
$2+9$ give a shortest cost of $9$.

The shared expression is

$$
(g\circ f)(x,z)=\bigoplus_y f(x,y)\otimes g(y,z).
$$

Read $\otimes$ as “combine these two stages” and $\bigoplus$ as “combine the
alternatives.” They stand for operations chosen by the **algebra**, not
necessarily multiplication and addition.

| Interpretation | Combine stages, $\otimes$ | Combine alternatives, $\bigoplus$ |
|---|---|---|
| Boolean reachability | AND | OR |
| Markov probabilities | multiplication | sum |
| Tropical path costs | addition | minimum |
| Log-probability weights | addition | log-sum-exp |
| Product-fuzzy scores | multiplication | noisy-OR |

Quivers defaults to product-fuzzy scores. Noisy-OR combines our two $0.08$
weights as $1-(1-0.08)^2=0.1536$. Noisy-OR describes the probability that
at least one of two independent events occurs. The walking and bus routes
in our example are mutually exclusive, so their probabilities add to $0.16$.
Choosing the algebra determines which of these calculations Quivers performs.

The algebra also determines which equations composition satisfies. A **category**
has associative composition and identity arrows [@maclane1998categories]. Associativity permits
regrouping three stages without changing their meaning; an identity leaves
its input unchanged. Sum-product matrix composition has these properties.
For general weighted tables, distributivity of the combination operations
is part of what makes reassociation work [@kelly1982basic]. The default noisy-OR/product pair
fails that distributivity law: regrouping its matrix contractions can
change their entries. [Algebras and base change](../semantics/algebras.md)
gives the equations and a counterexample.

## 4. A probabilistic program describes how data could arise

A **[generative model](https://dippl.org/chapters/02-webppl.html#and-inference)**
describes a process that produces values [@goodman2014design]. A
**latent variable** is a value we do not observe. A **prior** describes it
before using the observations; a **likelihood** scores observations given a
particular latent value. The **posterior** describes the latent variable
after taking those observations into account [@bishop2006pattern; @gelman2013bayesian].

Consider a separate, smaller model. Suppose the probability of rain is
$0.3$, and the probability of being late is $0.6$ on rainy days and $0.1$ on
dry days. To simulate a day, first draw the weather, then draw whether we
are late using the probability for that weather. This is **ancestral
sampling**: draw each variable after its parents.

If we observe that we were late, the question changes. We want the weather
distribution conditional on that observation:

$$
P(\mathrm{rain}\mid\mathrm{late})
=\frac{P(\mathrm{rain})P(\mathrm{late}\mid\mathrm{rain})}
{P(\mathrm{rain})P(\mathrm{late}\mid\mathrm{rain})
+P(\mathrm{dry})P(\mathrm{late}\mid\mathrm{dry})}
=\frac{0.3\times0.6}{0.3\times0.6+0.7\times0.1}=0.72.
$$

The numerator is the joint probability of rain and lateness. The denominator
sums over the possible weather values so the posterior probabilities add to
one. **Inference** performs this update, exactly when the required sum or
integral is tractable and approximately otherwise. Among all simulated days, about 30% will be rainy. Among the simulated days
on which we arrive late, about 72% will be rainy. Keeping only the late days
is a simple rejection sampler for this posterior.

In QVR, `sample` introduces a random draw, `let` computes from available
values, and `observe` binds supplied data and contributes its likelihood.
A `let` expression adds no new randomness, but its result can still be
random when its inputs are random. An observation contributes evidence; a
separate inference procedure uses that evidence to approximate or compute
the posterior. The [first-model tutorial](../tutorials/qvr/01-first-model.md)
works through these operations in a regression.

## 5. A distribution is more than one sample

A **Markov kernel** assigns an output probability distribution to each
input. Our first table is a finite kernel. For a real-valued response, a
conditional Normal family plays the same role without enumerating every
possible output. For a Normal kernel, one call might return 1.7; repeated calls produce
outcomes whose frequencies approximate the probabilities assigned by the kernel.

Continuous distributions assign probabilities to events such as intervals.
A **density** describes probability per unit of the reference measure. A
uniform distribution on an interval of length $0.5$ has density $2$ there,
though the interval's total probability is $1$. Thus a continuous
`log_prob` is a log density and may be positive. Probabilities of events
require integration; evaluating a density at a point is a different
operation.

Composing kernels averages over intermediate outcomes. For continuous
outcomes, the finite sum becomes an integral. Running two samplers in
sequence produces one draw from the composite kernel. To estimate the
probability of an output interval, repeat the sequence and count the fraction
of outputs in that interval. Density evaluation requires an analytic integral
or a numerical estimator. See
[Continuous spaces](../guides/continuous-spaces.md) and
[Correspondence and limitations](../semantics/adequacy.md).

Giry gives the construction using probability measures, and Fritz develops a
categorical account of stochastic maps [@giry1982categorical; @fritz2020synthetic].

## 6. Monads and handlers describe how computations continue

An ordinary function returns a value that the next function can consume.
A probabilistic computation instead describes a distribution of possible
values. To continue it, we must specify what happens for each possible
outcome and combine the resulting distributions. That operation is
**[bind](../guides/dsl-programs-and-lets.md#kleisli-bind-syntax)**. **Pure**, also called **return** in the mathematical interface,
embeds a value as a distribution concentrated at that value.

A **monad** packages these operations and their laws. Its laws say, among
other things, that returning a value and then continuing is the same as
continuing with that value directly, and that regrouping successive binds
preserves the computation. **Kleisli composition** is composition using
bind. The finite probability calculation above is one instance; the
[monad guide](../guides/monadic.md) also explains other instances.

Moggi develops the distinction between values and computations; Wadler
connects it to functional programming [@moggi1991notions; @wadler1992essence].

Scoring multiplies a measure by a likelihood. In the rain example, the
resulting weights are $0.18$ for rain and $0.07$ for dry weather, with total
mass $0.25$. Dividing by that mass gives the posterior. More generally, this
normalization requires finite, positive total mass. The Giry monad describes
the probability measures before scoring and after successful normalization;
the intermediate computation uses an unnormalized measure.

An **effect** describes an operation a computation requests, such as drawing
a sample. A **[handler](../guides/effects.md#algebraic-effects-handlers)** supplies an interpretation
of that request [@plotkin2009handlers; @bauer2015programming]. For
instance, a handler may draw a fresh value, supply an observation, or reuse
a recorded value. A **[continuation](https://dippl.org/chapters/03-enumeration.html#continuations)**
is the rest of the computation waiting
for that value. A trace records the sampled values and scores from one execution. An
inference procedure can use many such traces to estimate posterior expectations. The [effect guide](../guides/effects.md) develops
the interfaces and the restrictions on combining handlers.

## 7. Reading the notation and finding the next explanation

| Notation or term | How to read it here | Further detail |
|---|---|---|
| $f:A\to B$ | A computation or relation with input object $A$ and output object $B$ | [Morphisms](../guides/morphisms.md) |
| $\mathcal V$ | The mathematical structure used for weights | [Setting](../semantics/setting.md) |
| $\mathcal C(A,B)$, or a hom-object | The arrows from $A$ to $B$, with the structure appropriate to the category | [Categorical structures](../guides/categorical.md) |
| Functor | A map of categories preserving identities and composition | [Functors](../api/categorical/functors.md) |
| Natural transformation | A compatible family of arrows between two functors | [Categorical structures](../guides/categorical.md) |
| $\llbracket e\rrbracket_\rho$ | The mathematical meaning of expression $e$ using the values named by environment $\rho$ | [Setting](../semantics/setting.md) |
| $\Gamma\vdash e:\tau$ | Under the type assumptions in $\Gamma$, expression $e$ has type $\tau$ | [Typing](../semantics/typing.md) |
| Effect row | The effects still permitted or required at a particular computation boundary | [Effects and handlers](../reference/qvr/effects-and-handlers.md) |
| Variational guide | An adjustable distribution used to approximate a posterior | [SVI](../guides/inference-svi.md) |
| Plate | An indexed collection of conditionally independent draws | [Programs](../guides/programs.md) |

Continue with the [quickstart](quickstart.md) to build weighted tables, or
the [first-model tutorial](../tutorials/qvr/01-first-model.md) to fit a
probabilistic model. For the precise mathematical development, the
[semantics overview](../semantics/index.md) supplies a reading order.
