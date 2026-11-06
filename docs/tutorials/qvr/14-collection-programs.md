# Exact Bayes with finite collection expressions

This tutorial computes a posterior over three candidate means for four
Gaussian observations.
The grid is small enough to enumerate, so the model can expose the collection
operations without introducing an approximation or a sampling algorithm.

We will compute the posterior in three stages. First, a pointwise computation turns a
distribution's log density into a real score. Second, `traverse` applies that
computation across observations and candidate parameter values. Third, `map`
and `logsumexp_over` [normalize](../python/06-first-class-trans.md#read-a-transformation-as-a-change-in-interpretation) the resulting log joint.

Each candidate mean is one complete explanation of the data. To compare
candidates, multiply its [prior probability](../../getting-started/concepts.md#4-a-probabilistic-program-describes-how-data-could-arise) by the likelihoods of the four
observations, then divide by the sum of these weights across candidates.
For illustration, unnormalized weights `[1, 6, 3]` would give probabilities
`[0.1, 0.6, 0.3]`. The code performs this same normalization in log space.

The candidate set is the entire parameter space of this particular model.
Thus enumerating it gives exact finite Bayesian inference; using those three
points to approximate a continuous parameter would be a different claim.
Here collection operations make the repeated scoring explicit.

## 1. Score one observation

`log_prob` returns a nominal `LogWeight`. `weight_value` exposes its real value
for arithmetic:

<!-- compile: qiec-cumulative -->
```qvr
define point_log_likelihood(theta : Real, observation : Real, sigma : Real) : Real !{} =
    return weight_value(log_prob(Normal(theta, sigma), observation))
```

This helper is pure, though it uses a probability distribution. It evaluates
the Gaussian density at an observed value; it does not sample or add a factor
to an enclosing program.

## 2. Traverse the data

The dataset likelihood calls the helper once for each observation and sums the
four contributions:

<!-- compile: qiec-cumulative -->
```qvr
define dataset_log_likelihood(theta : Real, observations : Tensor[Real]([4]), sigma : Real) : Real !{} =
    let contributions <- traverse(
        observations,
        (observation -> point_log_likelihood(theta, observation, sigma)),
    )
    return sum(contributions)
```

The `<-` matters. `point_log_likelihood` is a named computation, so applying it
across a collection uses `traverse` rather than pure `map`. If the helper later
acquires an effect, that effect will appear in the row of
`dataset_log_likelihood` and its callers.

## 3. Normalize the finite posterior

The outer computation traverses the candidate grid, adds a uniform log prior,
and normalizes in log space:

<!-- compile: qiec-cumulative -->
```qvr
define grid_posterior(observations : Tensor[Real]([4]), sigma : Real) : Tensor[Real]([3]) !{} =
    let candidates = [-1.0, 0.0, 1.0]
    let log_likelihoods <- traverse(
        candidates,
        (theta -> dataset_log_likelihood(theta, observations, sigma)),
    )
    let log_prior = log(1.0 / 3.0)
    let log_joint = map(log_likelihoods, likelihood -> likelihood + log_prior)
    let log_evidence = logsumexp_over(log_joint, score -> score)
    return map(log_joint, score -> exp(score - log_evidence))
```

Here `traverse` sequences calls, while `map` transforms values already in hand.
`logsumexp_over` computes the log evidence without exponentiating the raw log
joint. The final `map` returns three probabilities that sum to one.

## 4. Export a program

The complete source is
[`finite-grid-bayes.qvr`](source/finite-grid-bayes.qvr). Its program boundary
takes the observations and known measurement noise as host data:

<!-- compile: qiec-cumulative -->
```qvr
object Analysis : FinSet 1
object Posterior : Real 3

program finite_grid_bayes : Analysis -> Posterior
    let posterior <- grid_posterior(observations, observation_sigma)
    return posterior

export finite_grid_bayes
```

Check the source and run the underlying computation directly:

```bash
qvr check docs/tutorials/qvr/source/finite-grid-bayes.qvr
qvr run docs/tutorials/qvr/source/finite-grid-bayes.qvr grid_posterior "[-0.2, 0.1, 0.3, 0.4]" 0.5 --json
```

The posterior places most mass on the candidate `0.0`. You can verify the
normalization in Python:

```python
from quivers.dsl import load

model = load("docs/tutorials/qvr/source/finite-grid-bayes.qvr")
run = model.run("grid_posterior", (-0.2, 0.1, 0.3, 0.4), 0.5)
assert abs(sum(run.value) - 1.0) < 1e-12
print(run.value)
```

## 5. Try two changes

First, add `2.0` to `candidates` and update the result extent from `3` to `4`.
The checker should force every affected type to agree. Second, replace the
uniform prior with a three-entry tensor and use
`map([0, 1, 2], i -> log(prior[i]))`. This version separates the prior from
the likelihood while keeping the same normalization.

A potential worry is that a three-point grid is too small for a serious
continuous analysis. That is right: this tutorial isolates exact finite
inference. The same factoring applies to lexical uncertainty, discrete model
comparison, and quadrature rules whose nodes and weights are fixed in the
source.

Continue with the [collection reference](../../reference/qvr/collection-expressions.md)
for shape restrictions and target support. The [mixture tutorial](04-marginalize.md)
shows the complementary case in which `marginalize` integrates a finite latent
inside a probabilistic program.
