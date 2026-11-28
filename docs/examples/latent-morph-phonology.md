# Latent morphs and phonology

One underlying form can occur in several observed words. In this model,
each stem and suffix has one latent underlying form, with a uniform
prior on the two segments. A three-tape relation concatenates the stem
and suffix. A phonological transducer then generates the surface form,
conditioning on the whole underlying word. This follows the
generative structure of [Cotterell, Peng, and Eisner (2015)](https://aclanthology.org/Q15-1031/).

Let $x,y$ be the underlying segments of a word and let $f(z\mid x)$
copy its input with probability $1-\eta$ and flip it with probability
$\eta$. The transducer copies the first segment noisily. For the
second segment it first chooses the stem with probability
$\lambda=\sigma(\theta)$ or the suffix with probability $1-\lambda$,
then applies the same noise:

$$
p_\theta(z_1,z_2\mid x,y)=f(z_1\mid x)
\big[(1-\lambda)f(z_2\mid y)+\lambda f(z_2\mid x)\big].
$$

Three observations connect the four latent forms in a tree. Sparse
relation messages give the exact evidence and suffix posterior. We
train $\theta$ on that evidence and compare both quantities and the
training gradient to enumeration over all sixteen lexicons.

## Try it

<!-- python: run -->
```python
from itertools import product
from math import log

import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import (
    Arc,
    Boundary,
    FiniteStringBeliefPlan,
    RationalRelation,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
)

bits = range(2)
tape = StringSet(alphabet=FinSet(name="Segments", cardinality=2))
lexicon = ("stem-a", "stem-b", "suffix-a", "suffix-b")
one_segment = ((0,), (1,))
two_segments = tuple(product(bits, repeat=2))
noise = 0.1


class SoundWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self):
        assimilation = self.theta.sigmoid()
        same_if_different = (1 - noise) - (1 - 2 * noise) * assimilation
        return torch.stack(
            (
                self.theta.new_tensor(log(1 - noise)),
                self.theta.new_tensor(log(noise)),
                same_if_different.log(),
                (-same_if_different).log1p(),
                self.theta.new_zeros(()),
            )
        )


weights = SoundWeights()


def relation(arity, input_tapes, states, arcs, names, values):
    unit = len(names) - 1
    topology = RelationTopology(
        (tape,) * arity,
        input_tapes,
        states,
        arcs,
        (Boundary(0, unit),),
        (Boundary(states - 1, unit),),
        names,
    )
    return RationalRelation(topology, values, algebra=LOG_PROB)


prior = relation(
    1,
    0,
    2,
    tuple(Arc(0, 1, (x,), 0) for x in bits),
    ("symbol", "unit"),
    weights.theta.new_tensor([log(0.5), 0.0]),
)
concat = relation(
    3,
    2,
    3,
    tuple(Arc(0, 1, (x, None, x), 0) for x in bits)
    + tuple(Arc(1, 2, (None, y, y), 0) for y in bits),
    ("unit",),
    weights.theta.new_zeros(1),
)
phonology = relation(
    2,
    1,
    4,
    tuple(
        Arc(0, 1 + x, (x, output), int(x != output))
        for x, output in product(bits, repeat=2)
    )
    + tuple(
        Arc(1 + x, 3, (y, output), 2 * int(x != y) + int(y != output))
        for x, y, output in product(bits, repeat=3)
    ),
    ("copy", "flip", "retain_after_other", "assimilate_after_other", "unit"),
    weights,
)

training = (
    ("stem-a", "suffix-a", (0, 0)),
    ("stem-b", "suffix-a", (1, 0)),
    ("stem-b", "suffix-b", (1, 1)),
)
loopy_data = (*training, ("stem-a", "suffix-b", (0, 1)))


def graph_with(data):
    variables = [StringVariable(name, tape) for name in lexicon]
    supports = dict.fromkeys(lexicon, one_segment)
    factors = [StringFactor((name,), prior) for name in lexicon]
    for i, (stem, suffix, surface) in enumerate(data):
        underlying_name, surface_name = f"underlying-{i}", f"surface-{i}"
        variables.extend(
            (
                StringVariable(underlying_name, tape),
                StringVariable(surface_name, tape, surface),
            )
        )
        factors.extend(
            (
                StringFactor((stem, suffix, underlying_name), concat),
                StringFactor((underlying_name, surface_name), phonology),
            )
        )
        supports[underlying_name] = two_segments
        supports[surface_name] = (surface,)
    return StringFactorGraph(tuple(variables), tuple(factors)), supports


def enumerate_lexicons(data):
    theta = weights.theta
    assimilation = theta.sigmoid()
    masses = []
    assignments = tuple(product(bits, repeat=len(lexicon)))
    for symbols in assignments:
        morph = dict(zip(lexicon, symbols, strict=True))
        mass = theta.new_tensor(0.5 ** len(lexicon))
        for stem, suffix, surface in data:
            x, y = morph[stem], morph[suffix]
            first = 1 - noise if x == surface[0] else noise
            from_suffix = 1 - noise if y == surface[1] else noise
            from_stem = 1 - noise if x == surface[1] else noise
            second = (1 - assimilation) * from_suffix + assimilation * from_stem
            mass = mass * first * second
        masses.append(mass)
    total = torch.stack(masses).sum()
    marginals = {}
    for index, name in enumerate(lexicon):
        marginals[name] = torch.stack(
            tuple(
                torch.stack(
                    [
                        mass
                        for assignment, mass in zip(assignments, masses, strict=True)
                        if assignment[index] == symbol
                    ]
                ).sum()
                / total
                for symbol in bits
            )
        )
    return total.log(), marginals


forest, _ = graph_with(training)
suffix = forest.tree_marginal("suffix-a", max_states=4096, max_work=32768)


def check_forest():
    actual = suffix.partition()
    expected, exact = enumerate_lexicons(training)
    torch.testing.assert_close(actual, expected)
    probabilities = torch.stack(
        tuple(suffix.log_prob(word).exp() for word in one_segment)
    )
    torch.testing.assert_close(probabilities, exact["suffix-a"])
    (actual_gradient,) = torch.autograd.grad(actual, weights.theta)
    (expected_gradient,) = torch.autograd.grad(expected, weights.theta)
    torch.testing.assert_close(actual_gradient, expected_gradient)
    return actual.detach()


before = check_forest()
optimizer = torch.optim.SGD(weights.parameters(), lr=0.8)
for _ in range(12):
    optimizer.zero_grad()
    (-suffix.partition()).backward()
    optimizer.step()
assert check_forest() > before

graph, supports = graph_with(loopy_data)
plan = FiniteStringBeliefPlan(
    graph,
    supports,
    max_entries=192,
    max_states=256,
    max_work=768,
)
with torch.no_grad():
    short = plan.run(max_sweeps=1, tolerance=1e-9, damping=0.2)
    fitted = plan.run(max_sweeps=80, tolerance=1e-9, damping=0.2)
    _, truth = enumerate_lexicons(loopy_data)
    marginal_error = max(
        (fitted.log_marginals[name].exp() - truth[name]).abs().max().item()
        for name in lexicon
    )
assert short.status == "limit" and short.message_residual > 1e-9
assert fitted.status == "converged" and fitted.message_residual <= 1e-9
assert 0 < marginal_error < 0.01
```

The fourth observation closes a cycle through the two stems and two
suffixes. The code then keeps the same factors and declares the
finite support of each variable for loopy belief propagation. The
one-sweep run reports its limit. The longer run reports convergence
when the largest total variation across an undamped message update
falls within tolerance. Enumeration separately measures the error in
the returned marginals for this instance.
