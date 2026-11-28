# Name phylogeny

The observed data are name types $w_1,\ldots,w_n$ and their token counts
$n_1,\ldots,n_n$. Each name has one parent in a directed tree rooted at a
distinguished source $\diamond$. An edge from the source generates a new
name. An edge from another name derives the child by string transduction.

We use the edge-factored approximation in §5.2 of
[Andrews, Eisner, and Dredze (2012)](https://aclanthology.org/D12-1032/).
In this approximation, $n_x/(n_y+1)$ is the expected number of tokens of $x$
before the first token of $y$ in a uniformly random token ordering. The
product of edge weight expectations replaces the expectation of the joint
product over token orders. With source probability $p_0$ and conditional
mutation probability $p_\theta$, the edge weights are

$$
\delta(y\mid\diamond)=\alpha p_0(w_y),
\qquad
\delta(y\mid x)=\mu\frac{n_x}{n_y+1}p_\theta(w_y\mid w_x).
$$

The root language draws a geometric length and then independent uniform
symbols. The transducer deletes each input symbol with probability $d$.
A retained symbol stays unchanged with probability $\sigma(\theta)$ and
flips to the other symbol of the binary alphabet otherwise. These three
possibilities sum to one at each input position, so the transducer is a
normalized conditional distribution for every fixed input. Its fixed-word
queries sum all deletion alignments.

<!-- python: run -->
```python
from itertools import combinations, permutations, product
from math import log

import torch
from torch import nn
from torch.nn import functional as F

from quivers import FinSet, StringSet
from quivers.core import directed_tree
from quivers.core.algebras import LOG_PROB
from quivers.core.relations import Arc, Boundary, RationalRelation, RelationTopology

names = ("ana", "ann", "an")
counts = (7, 4, 2)
symbols = {"a": 0, "n": 1}
words = tuple(tuple(symbols[c] for c in name) for name in names)
deletion, continuation = 0.23, 0.6
alpha, mu = 1.0, 0.2
nodes = range(1, len(words) + 1)


class MutationWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))

    def forward(self):
        keep = log(1 - deletion)
        return torch.stack((
            self.theta.new_tensor(log(deletion)),
            keep + F.logsigmoid(self.theta),
            keep + F.logsigmoid(-self.theta),
            self.theta.new_zeros(()),
        ))


weights = MutationWeights()
tape = StringSet(alphabet=FinSet(name="NameAlphabet", cardinality=2))
arcs = tuple(
    Arc(0, 0, (c, output), slot)
    for c in range(2)
    for output, slot in ((None, 0), (c, 1), (1 - c, 2))
)
boundary = (Boundary(0, 3),)
topology = RelationTopology(
    (tape, tape), 1, 1, arcs, boundary, boundary,
    ("delete", "copy", "substitute", "unit"),
)
channel = RationalRelation(topology, weights, algebra=LOG_PROB)
plans = {
    (x, y): channel.query((words[x - 1], words[y - 1]))
    for x, y in product(nodes, repeat=2)
    if x != y
}


def tree():
    theta = weights.theta
    absent = theta.new_tensor(-torch.inf)
    root = tuple(
        theta.new_tensor(log(alpha * (1 - continuation)) + len(w) * log(continuation / 2))
        for w in words
    )
    rows = [torch.stack((absent, *root))]
    for x in nodes:
        row = [absent]
        for y in nodes:
            if x == y:
                row.append(absent)
            else:
                log_count_factor = log(mu * counts[x - 1] / (counts[y - 1] + 1))
                row.append(plans[x, y].weight() + log_count_factor)
        rows.append(torch.stack(row))
    return directed_tree(torch.stack(rows), root=0)


def enumerate_trees():
    theta = weights.theta
    fidelity = theta.sigmoid()
    edge = {}
    for y in nodes:
        target = words[y - 1]
        edge[0, y] = theta.new_tensor(
            alpha * (1 - continuation) * (continuation / 2) ** len(target)
        )
    for x, y in permutations(nodes, 2):
        source, target = words[x - 1], words[y - 1]
        if len(source) < len(target):
            edge[x, y] = theta.new_zeros(())
            continue
        alignments = []
        for kept in combinations(range(len(source)), len(target)):
            matches = sum(source[i] == c for i, c in zip(kept, target, strict=True))
            alignments.append(
                fidelity**matches * (1 - fidelity)**(len(target) - matches)
            )
        alignment_mass = sum(alignments, theta.new_zeros(()))
        edge[x, y] = (
            mu * counts[x - 1] / (counts[y - 1] + 1)
            * deletion**(len(source) - len(target))
            * (1 - deletion)**len(target) * alignment_mass
        )

    total = theta.new_zeros(())
    marginals = theta.new_zeros((len(words) + 1, len(words) + 1))
    possible_parents = [tuple(x for x in range(len(words) + 1) if x != y) for y in nodes]
    for choices in product(*possible_parents):
        parent = dict(zip(nodes, choices, strict=True))
        for start in nodes:
            seen = set()
            cursor = start
            while cursor and cursor not in seen:
                seen.add(cursor)
                cursor = parent[cursor]
            if cursor:
                break
        else:
            edges = [(parent[y], y) for y in nodes]
            mass = torch.stack(tuple(edge[e] for e in edges)).prod()
            total = total + mass
            for x, y in edges:
                marginals[x, y] += mass.detach()

    return total.log(), marginals / total.detach()


def check_enumeration():
    distribution = tree()
    actual = distribution.partition()
    expected, edge_probabilities = enumerate_trees()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(distribution.edge_marginals(), edge_probabilities)
    (actual_sensitivity,) = torch.autograd.grad(actual, weights.theta)
    (expected_sensitivity,) = torch.autograd.grad(expected, weights.theta)
    torch.testing.assert_close(actual_sensitivity, expected_sensitivity)
    return actual.detach()


before = check_enumeration()
optimizer = torch.optim.SGD(weights.parameters(), lr=0.5)
for _ in range(12):
    optimizer.zero_grad()
    (-tree().partition()).backward()
    optimizer.step()
after = check_enumeration()
assert after > before
```

The root has index zero, and a matrix entry has its parent on the first
axis and its child on the second. The exact tree partition sums the product
of edge weights over root-connected trees. Its derivative with respect to
one edge log weight is that edge's posterior probability. The example
checks the partition, every edge marginal, and the derivative of the
shared mutation parameter by independent enumeration before and after
training. The corpus factor in the approximate likelihood is constant in
$\theta$ for fixed $\alpha$ and $\mu$, so optimizing the log partition
fits the transducer under this approximation.

The fixed-word query plans are cached by ordered pair of names. Each
likelihood evaluation recomputes each pair's score once and reuses it in
the edge matrix. That lets derivatives reach the current transducer
parameter on each update.
