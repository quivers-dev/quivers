# A noisy deletion channel

A source produces a string over a one-symbol alphabet. It repeats the
symbol with probability $p$ and stops with probability $1-p$. The
channel deletes each source symbol with probability $p$ and keeps it
with probability $1-p$. Both relations read the same parameter.

We observe a word of length $m$ and infer the source length $n$. For
$n\geq m$, the weight of the paired words is

$$
P(N=n,Y=m) = (1-p)p^n \binom{n}{m}(1-p)^m p^{n-m}.
$$

The binomial factor counts the distinct choices of which source
positions the channel keeps. A single-tape join retains the source
word through composition, and conditioning fixes the channel's
output. The resulting unary relation sums the compatible alignments
at every source word.

<!-- python: run -->
```python
import torch
from torch import nn

from quivers import FinSet, StringSet
from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    RationalRelation,
    RelationTopology,
    condition,
    join,
)


class ChannelWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self):
        return torch.stack((self.p, 1 - self.p, self.p.new_ones(())))


tape = StringSet(alphabet=FinSet(name="UnaryAlphabet", cardinality=1))
weights = ChannelWeights()
slots = ("repeat_or_delete", "stop_or_keep", "unit")
unit = (Boundary(0, 2),)

source = RationalRelation(
    RelationTopology(
        (tape,), 0, 1, (Arc(0, 0, (0,), 0),), unit, (Boundary(0, 1),), slots
    ),
    weights,
    algebra=MARKOV,
)
channel = RationalRelation(
    RelationTopology(
        (tape, tape), 1, 1,
        (Arc(0, 0, (0, None), 0), Arc(0, 0, (0, 0), 1)),
        unit, unit, slots,
    ),
    weights,
    algebra=MARKOV,
)

joint = join(
    source,
    channel,
    left_tape=0,
    right_tape=0,
    keep_shared=True,
    max_states=64,
    max_work=128,
)
observed = condition(joint, 1, (0, 0), max_states=64, max_work=128)
evidence = observed.total_weight(max_states=64, max_work=128, max_closure_size=2)
posterior_at_three = observed.weight(((0, 0, 0),)) / evidence
(sensitivity,) = torch.autograd.grad(posterior_at_three, weights.p)

assert torch.isclose(evidence.detach(), weights.p.new_tensor(0.032))
assert torch.isfinite(sensitivity)
```

Erasing the remaining source labels for the total-weight calculation exposes
loops for deleted source symbols. The star calculation sums arbitrarily
many deletions before, between, and after the observed symbols. The
evidence is

$$
P(Y=m) = \frac{(1-p)\bigl(p(1-p)\bigr)^m}{(1-p^2)^{m+1}},
$$

and $N-m$ under the posterior has a negative binomial distribution
with $m+1$ successes and success probability $1-p^2$. The code uses a
three-state product topology and a query sized by the observed length.
The source length is unbounded, and the source and channel
contributions both reach the original parameter derivative.
