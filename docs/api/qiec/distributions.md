# Distributions

`RuntimeDistribution` values sample and score
[distribution families](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families)
on the [reference machine](../../developer/qiec.md#reference-evaluator), through the
`DistributionBackend` of the run. The `ReferenceBackend` computes in plain Python numbers, position
by position, through a `HostLaw`; the [tensor backend](../continuous/tensor_backend.md) computes
with torch and keeps every gradient. A transpile target may support different families or
[parameterizations](../../semantics/transpile-correctness/index.md#4-distribution-parameterization),
which its [support table](../../transpile-support.md) lists.

::: quivers.qiec.distributions

## Finite structured sampleables

A [FiniteSampleable][quivers.qiec.structured_sampleables.FiniteSampleable] is a
normalized law on an explicit list of closed discrete QIEC values. A provider
can supply it to a checked `Sampleable[A]` parameter, where `A` is the declared
type of its points. A regular recursive family can be the carrier. The list
specifies the complete candidate support of this law.

For a point $x$ and input log weights $w_i$, the probability is

$$
P(x) = \frac{\sum_{i\colon x_i = x} \exp(w_i - m)}
                  {\sum_i \exp(w_i - m)},
\qquad m = \max_i w_i.
$$

The runtime uses exact structural equality on this carrier. Equal
constructors compare by nominal identity and ordered fields. Equal points
share one probability. An input log weight of $-\infty$ contributes zero.
An empty support or zero total mass cannot be normalized. Centering prevents
a common offset from swamping the log normalizer. A finite difference outside
the run's numerical range raises a distribution error.

Drawing uses the ordinary categorical sampler and follows its numerical and
randomness conventions. Scoring, conditioning, and replay use the normalized
log probability and keep its weight gradient. Exact enumeration visits each
positive-mass value once, using the existing `Random` enumeration handler.
An unrestricted resumption must have an explicit provider assertion that any
captured law may be duplicated, as it does for other opaque runtime values.
The provider may share this law between shots when its support and weights
stay fixed during that invocation. A later run reads the live weights again.

For $n$ weight occurrences and $k$ distinct candidates, indexing the support
validates the reachable carrier declarations and hashes the input term data.
A density query also checks and hashes its argument. A score or enumeration reads $n$
weights in total and uses $O(n + k)$ temporary storage. Enumeration adds the
cost of running the continuation for each positive-mass value. The law has
scalar log probabilities and one term-valued draw. It has no plate axes or
reparameterized sampling rule.

### An exact finite posterior

The existing enumeration handler can also retain one scalar log joint per
point. With `Enumeration(reduction="table", grouped=True)`, the handled
computation returns its observation log likelihood together with its value.
For one sampled structured site, the table's order is the order of
`prior.weighted_support()`, and its declared extent must equal the number
of positive-mass prior points. A provider can use the returned table as the
log weights of a law on those same points.

<!-- python: skip -->
```python
support = tuple(point for point, _ in prior.weighted_support())
posterior = prior.with_(values=support, log_weights=table)
```

The exponentiated entries sum to the evidence. The resulting law
normalizes it to the exact posterior for the declared finite support.
Scoring that posterior preserves the gradient with respect to both the
prior weights and the observation law.

::: quivers.qiec.structured_sampleables
