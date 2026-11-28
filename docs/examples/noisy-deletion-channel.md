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

## Try it

<!-- python: run -->
```python
import torch

from quivers.dsl import Compiler, parse
from quivers.qiec import (
    FixedRelationRuntimeProvider,
    FixedTransformedRelationRequest,
    ParameterId,
    RelationWeightComplement,
    RuntimeConfiguration,
    run_named,
)

source = """
object UnaryAlphabet : FinSet 1

relation geometric [algebra=markov, states=1]
    output UnaryAlphabet
    initial 0 [weight=unit_weight]
    arc 0 -> 0 [labels=(0), weight=repeat_or_delete]
    final 0 [weight=stop_or_keep]

relation deletion [algebra=markov, states=1]
    input UnaryAlphabet
    output UnaryAlphabet
    initial 0 [weight=unit_weight]
    arc 0 -> 0 [labels=(0, eps), weight=repeat_or_delete]
    arc 0 -> 0 [labels=(0, 0), weight=stop_or_keep]
    final 0 [weight=unit_weight]

relation joint = join(geometric, deletion, 0, 0, keep_shared=true)
relation observed = project(joint, (1))
relation_score evidence : Real [operation=pair_weight, relation=observed]
relation_score mass : Real [operation=pair_weight, relation=joint]

define observation_log_prob() : Real !{} =
    let z <- evidence()
    return log(z)

define posterior_length() : Real !{} =
    let z <- evidence()
    let w <- mass()
    return w / z
"""

compiler = Compiler(parse(source), module_name="noisy-deletion")
module = compiler.qiec_module
assert module is not None
p = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
owner = ParameterId.derive(module.module, "relation_weights", "repeat_or_delete")
weights = {
    owner: p,
    ParameterId.derive(module.module, "relation_weights", "stop_or_keep"):
        RelationWeightComplement(owner),
    ParameterId.derive(module.module, "relation_weights", "unit_weight"): p.new_ones(()),
}
providers = []
for score, relation, words in (
    ("evidence", "observed", ((0, 0),)),
    ("mass", "joint", ((0, 0, 0), (0, 0))),
):
    request = FixedTransformedRelationRequest(
        relation=compiler.transformed_relation_specs[relation],
        words=words, max_states=64, max_work=128, max_closure_size=4,
    )
    providers.append(FixedRelationRuntimeProvider.for_score(score, request, weights))
runtime = RuntimeConfiguration(providers=tuple(providers))

log_evidence = run_named(module, "observation_log_prob", runtime=runtime).value
posterior_at_three = run_named(module, "posterior_length", runtime=runtime).value
(sensitivity,) = torch.autograd.grad(posterior_at_three, p)

assert torch.isclose(log_evidence.exp(), p.new_tensor(0.032))
assert torch.isfinite(sensitivity)
```

Projecting onto the observed tape exposes
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
contributions both reach the original parameter derivative. The checked
runtime recomputes the complement from that same leaf after an optimizer
update.
