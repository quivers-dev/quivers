# 14. Exact Bayes with finite collection expressions

When a parameter takes finitely many values, its posterior can be computed exactly by enumeration: weight each value by its prior and the likelihood of the data, and normalize. This chapter computes the posterior of a mean $\theta \in \{-1, 0, 1\}$, under a uniform prior, from four Normal observations with known scale. It writes the computation with QVR's collection expressions: `traverse`, which applies a computation to every element of a collection, `map`, which applies a pure function, and `logsumexp_over`, which reduces in log space. The set of three points is the whole parameter space of this model, so the result is exact. Using the same points to approximate a continuous parameter would be a different claim.

## 1. Score one observation

`log_prob` returns a value of type `LogWeight`, a [nominal type](https://en.wikipedia.org/wiki/Nominal_type_system): a log weight is distinguished by its declared name from an ordinary `Real`. `weight_value` returns its real value for arithmetic:

<!-- compile: qiec-cumulative -->
```qvr
define point_log_likelihood(theta : Real, observation : Real, sigma : Real) : Real !{} =
    return weight_value(log_prob(Normal(theta, sigma), observation))
```

The computation is pure even though it uses a distribution: it evaluates the Normal log density at a value, and it neither samples nor adds to the weight of an enclosing program.

## 2. Traverse the data

The log likelihood of the data calls `point_log_likelihood` once for each observation and sums the results:

<!-- compile: qiec-cumulative -->
```qvr
define dataset_log_likelihood(theta : Real, observations : Tensor[Real]([4]), sigma : Real) : Real !{} =
    let contributions <- traverse(
        observations,
        (observation -> point_log_likelihood(theta, observation, sigma)),
    )
    return sum(contributions)
```

`point_log_likelihood` is a named computation, which in general may perform effects, so applying it to a collection uses `traverse` and binds the result with `<-`. `map` is for pure functions. This is the distinction between `traverse` and `fmap` for a traversable functor in functional programming [@mcbride2008applicative]. If `point_log_likelihood` later performs an effect, that effect appears in the effect row of `dataset_log_likelihood` and of its callers.

## 3. Normalize the finite posterior

The outer computation traverses the candidate means, adds the uniform log prior, and normalizes in log space:

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

`logsumexp_over` computes the log evidence without exponentiating the log joint densities, and the final `map` returns three probabilities that sum to one.

## 4. Export a program

The complete source is [`finite-grid-bayes.qvr`](source/finite-grid-bayes.qvr). Its program takes the observations and the known noise scale as [data inputs](01-first-model.md#data-inputs-and-the-observations-dict):

<!-- compile: qiec-cumulative -->
```qvr
object Analysis : FinSet 1
object Posterior : Real 3

program finite_grid_bayes : Analysis -> Posterior
    let posterior <- grid_posterior(observations, observation_sigma)
    return posterior

export finite_grid_bayes
```

Check the source and run the computation directly:

```bash
qvr check docs/tutorials/qvr/source/finite-grid-bayes.qvr
qvr run docs/tutorials/qvr/source/finite-grid-bayes.qvr grid_posterior "[-0.2, 0.1, 0.3, 0.4]" 0.5 --json
```

or from Python:

```python
from quivers.dsl import load

model = load("docs/tutorials/qvr/source/finite-grid-bayes.qvr")
run = model.run("grid_posterior", (-0.2, 0.1, 0.3, 0.4), 0.5)
assert abs(sum(run.value) - 1.0) < 1e-12
print(run.value)
```

```text
(3.0319443143760076e-05, 0.9962855523615224, 0.00368412819533383)
```

The posterior puts almost all its mass on $\theta = 0$.

## 5. Try two changes

- Add `2.0` to `candidates`. The checker rejects the computation (`computation body returns Tensor[Real]([4]), expected Tensor[Real]([3])`) until the result type of `grid_posterior` and the size of `Posterior` are changed to 4.
- Replace the uniform prior with a tensor `prior` of three probabilities and compute the log prior as `map([0, 1, 2], i -> log(prior[i]))`, which separates the prior from the likelihood.

A potential worry is that a grid of three points is too small for a serious analysis of a continuous parameter. That is right: the example isolates exact inference over a finite set. The same structure applies to lexical uncertainty, comparison of finitely many models, and quadrature rules whose nodes and weights are fixed in the source.

## Next

The [collection reference](../../reference/qvr/collection-expressions.md) gives the shape restrictions and the target support of each collection operation. [Chapter 4](04-marginalize.md) treats the related case in which `marginalize` sums a finite latent variable out inside a probabilistic program.
