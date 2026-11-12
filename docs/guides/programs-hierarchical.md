# Hierarchical Programs

A [hierarchical model](https://en.wikipedia.org/wiki/Multilevel_model) gives related
groups their own parameters and ties those parameters together through shared
higher-level ones, so that groups inform one another (partial pooling)
[@gelman2013bayesian]. QVR expresses such models with
[parametric programs](dsl-programs-and-lets.md#parametric-programs), which reuse one
template for several grouping factors, and with grouped
[marginalization](dsl-programs-and-lets.md#scoped-marginalize), which sums a discrete
class out once per group. [Monadic Programs](programs.md) describes the runtime of a
compiled program.

## Hierarchical models with parametric templates

In the model below every response belongs to both a subject and a verb (crossed
effects). Each subject has an offset, and the shared scale `sigma` controls how much
subjects tend to differ, so data for one subject inform that scale and through it the
estimates for the others. The verbs work the same way. The index arrays `subject_idx`
and `verb_idx` pick out each response's offsets, and adding them to an intercept gives
the response's log-odds.

```qvr
object Subject : FinSet 200
object Verb : FinSet 100
object Resp : FinSet 5000

program random_intercepts (G : FinSet, prior_scale : Real) : G -> Real 1
    sample sigma <- HalfNormal(scale=prior_scale)
    sample v : G <- Normal(loc=0.0, scale=sigma)
    return v

program crossed : Resp -> Resp
    sample intercept <- Normal(loc=0.0, scale=1.0)

    sample by_subject <- random_intercepts(Subject, 1.0)
    sample by_verb    <- random_intercepts(Verb,    1.0)

    let eta = intercept + by_subject[subject_idx] + by_verb[verb_idx]
    observe response : Resp <- Bernoulli(logits=eta)
    return intercept

export crossed
```

The template `random_intercepts` takes a grouping object `G` and a prior scale. Each
call inlines a copy of its body with its names prefixed by the bound name
(`by_subject$sigma`, `by_subject$v`, and so on), so the two grouping factors share the
template's structure but not its latent variables. The responses and the index arrays
are given at run time in the [`observations` dict](programs.md#the-observations-dict),
the index arrays as [data inputs](inference-foundations.md#host-data-per-row-covariates-and-index-arrays).
A monotone ordinal effect is the `cumsum` of `HalfNormal` increments, whose partial sums
increase because each increment is positive, and a discrete latent class is summed out
with a `marginalize` block.

## Grouped marginalization {#grouped-marginalization-fibred-discrete-latents}

Suppose two measurements belong to one item whose class is unknown. Given class zero
their likelihoods are `0.8` and `0.7`, and given class one they are `0.2` and `0.3`.
With equal class probabilities the joint likelihood is
`0.5 * (0.8 * 0.7) + 0.5 * (0.2 * 0.3) = 0.31`. Multiplying two separately mixed
likelihoods would give `0.25`, which is the different model with a separate class
choice for each measurement [@stanFiniteMixtures].

An *index map* is a function $\iota : N \to G$ from observation rows to groups, given
as an integer array. Its *fiber* over a group $g$ is the set
$\iota^{-1}(g) = \{n : \iota(n) = g\}$ of rows assigned to $g$
([fiber](https://en.wikipedia.org/wiki/Fiber_(mathematics))). A `marginalize` block
with `[over=G]` draws one class per element of `G`, and every `observe` in its body
names with `[via=idx]` the index map from its rows to `G`. The block multiplies the
likelihoods of the rows in each fiber under each class and then sums over the class:

<!-- compile: cumulative -->
```qvr
object Item : FinSet 3
object RespA : FinSet 6
object RespB : FinSet 4
object K : FinSet 2
object Out : Real 1

program grouped : Out -> Out
    sample probs <- Dirichlet(concentration=1.0) [over=K]
    sample mu_a : K <- Normal(loc=0.0, scale=1.0)
    sample theta_b : K <- Beta(concentration1=1.0, concentration0=1.0)
    marginalize c : K <- Categorical(probs=probs) [over=Item]
        observe r_a : RespA <- Normal(loc=mu_a[c], scale=1.0) [via=idx_a]
        observe r_b : RespB <- Bernoulli(probs=theta_b[c]) [via=idx_b]
    return probs
```

With $\pi$ the class probabilities, $\iota_m$ the index map of the $m$-th observe, and
$\ell_m(n, k)$ the log likelihood of its row $n$ under class $k$, the block adds

$$
\sum_{g \in G} \log \sum_{k=1}^{K} \exp\Bigl[\log \pi(g, k) + \sum_{m} \sum_{n \in \iota_m^{-1}(g)} \ell_m(n, k)\Bigr]
$$

to the log joint density, computed with log-sum-exp for numerical stability. This is
Stan's `target += log_mix(probs, ...)` accumulation, once per item [@stanFiniteMixtures].

### Special cases

- Without `[over=G]`, the block has one class for all rows, a single mixture over the
  whole data set ([scoped marginalize](dsl-programs-and-lets.md#scoped-marginalize)).
- One observe whose index map sends each row to its own group gives a separate mixture
  per row.
- One observe whose index map groups rows gives one mixture per group.
- Several observes that share the index set give a joint mixture across different kinds
  of response.

### Constraints and runtime

The `over` object must be declared, and every `observe` in the body needs a `via`. The
class probabilities have shape `(K,)`, shared by all groups, or `(|G|, K)`, one row per
group. The function
`quivers.continuous.plate.marginalize_grouped(ll, idx, log_prior, num_groups)` computes
the block. It takes either one `(N, K)` tensor of log likelihoods with its index array,
or a list of `(N_m, K)` tensors with one index array each.

### Product groupings

A group can be identified by a pair, such as a school and a subject. The header then
names the product, `[over=[School, Subject]]`, and each `observe` gives one index array
per factor, `[via=[school_idx, subject_idx]]`. The compiler turns each pair into one
group index in row-major order, the last factor varying fastest, and the block collects
the rows that share a school and a subject before summing over the class:

<!-- compile: cumulative -->
```qvr
object School : FinSet 2
object Subject : FinSet 3
object Row : FinSet 8

program by_pair : Out -> Out
    sample probs2 <- Dirichlet(concentration=1.0) [over=K]
    sample mu : K <- Normal(loc=0.0, scale=1.0)
    marginalize c : K <- Categorical(probs=probs2) [over=[School, Subject]]
        observe y : Row <- Normal(loc=mu[c], scale=1.0) [via=[school_idx, subject_idx]]
    return probs2
```

The number of index arrays must equal the number of factors.
[Composition Rules](../semantics/composition-rules.md) gives the denotation.

## End-to-end fit

The program's input is one row, since the responses and their groups come from the
observations ([Input rows are independent runs](inference-foundations.md#input-rows-are-independent-runs)):

<!-- python: skip -->
```python
import torch
from quivers.dsl import load
from quivers.inference import ELBO, AutoNormalGuide, SVI

model = load("crossed.qvr").morphism  # the crossed program above
observations = {
    "response": response,        # (5000,) zeros and ones
    "subject_idx": subject_idx,  # (5000,) integers in 0..199
    "verb_idx": verb_idx,        # (5000,) integers in 0..99
}

guide = AutoNormalGuide(model, observed_names={"response"})
elbo = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

for _ in range(2000):
    svi.step(torch.zeros(1, 1), observations)
```

## See also

- [Monadic Programs](programs.md): running and scoring a compiled program.
- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md#parametric-programs): the
  syntax of templates and of the `over` clause.
- [Analysis: Data and Formulas](analysis-data-and-formulas.md): formulas in the style
  of brms [@burkner2017brms] that compile random intercepts into a program like the one
  above.
