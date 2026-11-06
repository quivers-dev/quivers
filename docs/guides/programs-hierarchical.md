# Hierarchical Programs

This page describes the program-composition features used to express
hierarchical Bayesian models:
parametric program templates that share structure across grouping
factors, and the grouped [marginalization](dsl-programs-and-lets.md#scoped-marginalize) construct for jointly
identifying a discrete class indicator by multiple heterogeneous
response axes. The basic `MonadicProgram` API is described in the
[programs guide](programs.md).

## Hierarchical models with parametric templates

A hierarchical model gives related groups their own parameters while
linking them through shared higher-level parameters. In this example,
each subject has an offset, and their common scale `sigma` controls how
much subjects tend to differ. Data for one subject can inform that scale
and thus the inference for others. This is partial pooling.

Crossed effects mean that every response belongs to both a subject and a
verb. The two index arrays select the relevant offsets for each row;
adding them to an intercept forms the response's linear predictor.


A [parametric
program](dsl-programs-and-lets.md#parametric-programs) declares a
reusable kernel template polymorphic over typed parameters
(`FinSet`, `Space`, `Object`, `Real`, `Nat`, or `Mor[A, B]`). Each
call site `v <- template(...)` inlines a fresh α-renamed copy of
the template's body, so call sites contribute distinct latents.

<!-- compile: false -->
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
    let probability = sigmoid(eta)
    observe response : Resp <- Bernoulli(probs=probability)
    return intercept

export crossed
```

Each `random_intercepts` call inlines an independent `sigma` and
per-level plate; the observed response is the runtime tensor
supplied alongside the row-to-group indices via
`observations={"response": response_tensor, "subject_idx": subject_idx,
"verb_idx": verb_idx}`. `sigmoid` converts the unconstrained additive
predictor to a probability in `(0,1)`. Monotone
ordinal effects are expressed as `cumsum` of `HalfNormal`
increments (positive support implies monotone partial sums);
discrete latent classes are integrated out with a scoped
`marginalize ...` step followed by an indented body.

The plate-draw, vectorized-observe, parametric-program, and
`marginalize` constructs compose into the standard hierarchical
Bayesian forms. The pattern above shows crossed random intercepts
on two grouping factors, both reusing a single parametric template.
Each call to `random_intercepts` inlines a fresh `sigma` and a
fresh per-level plate `v` under α-renamed names
(`by_subject$sigma`, `by_subject$v`, ...), so the two grouping
factors share *structure* but not *latents*.

## Grouped marginalization: fibred discrete latents

Stan’s [vectorizing mixtures example](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#vectorizing-mixtures) contrasts a separate mixture assignment per observation with one assignment shared by a collection. [@stanFiniteMixtures] That distinction determines where the sum over classes belongs in the grouped model below.

Suppose two measurements belong to the same item, whose class is unknown.
Conditional on class zero their likelihoods are `0.8` and `0.7`;
conditional on class one they are `0.2` and `0.3`. With equal class
probabilities the joint likelihood is
`0.5 * (0.8 * 0.7) + 0.5 * (0.2 * 0.3) = 0.31`.
Multiplying the two independently mixed likelihoods would instead give
`0.25`, a different model with a separate class choice per measurement.

The grouping map collects observations sharing a class draw. A *fiber* is the set of response
rows mapped to one item. First combine their likelihoods conditional on
one class, then sum over that shared class. The logarithms in the formula
below turn products into sums and evaluate the final sum stably.


A scoped `marginalize` block accepts an optional `over G` clause
that declares a *grouping plate* `G`. Inside the body, every
`observe` step carries its own `via <idx>` clause naming the
fibration `idx : Resp_m -> G` from that observe's response plate to
the shared grouping plate. The body's per-axis per-row per-class
log-likelihoods are scatter-summed into a single $(|G|, K)$
accumulator before the log-sum-exp over the class axis:

<!-- compile: false -->
```qvr
marginalize class : K <- Categorical(probs=probs) [over=G]
    observe r_a : Resp_a <- F_a(...) [via=idx_a]
    observe r_b : Resp_b <- F_b(...) [via=idx_b]
    ...
```

The block contributes

$$
\sum_{g \in G}\ \log\sum_{k=1}^{K}\exp\!\left[\log \pi(g, k) + \sum_{m}\sum_{n:\ \mathrm{idx}_m(n)=g}\ell_m(n, k)\right]
$$

to the program-level log-density. The index maps collect each group's observations into a fiber. This is
a finite aggregation along those maps, followed by marginalization of the
shared class. A categorical formulation must specify the weight algebra
and the relevant [Kan-extension](enriched.md#kan-extensions) hypotheses;
the explicit log-density formula above is the operational contract here.

### Degenerate forms

The fibred form is the canonical hierarchical-Bayes likelihood
pattern, equivalent to Stan's per-item
`target += log_mix(probs, sum_m ll_item_m[i])` accumulation. It
degenerates as expected:

- *No grouping plate*: the global mixture form (ungrouped
  `marginalize`).
- *Single observe with identity fibration* (one group per row): the
  per-row mixture.
- *Single observe with coarser fibration*: the per-block
  hierarchical mixture.
- *Multiple observes sharing a per-item class indicator*: the
  joint mixture across heterogeneous response axes.

### Constraints and runtime

The `over` object must be declared. Each observe's `via` name must
be a previously bound plate variable. The categorical family's
first argument is the prior tensor; its shape may be `(K,)` (shared
across groups) or `(|G|, K)` (per-group prior). The runtime
primitive is exposed at
`quivers.continuous.plate.marginalize_grouped(ll, idx, log_prior,
num_groups)` and accepts either a single `(N, K)` tensor
(single-observe case) or a parallel list of `(N_m, K)` tensors with
their per-axis fibrations (multi-observe case).

### Product fibrations

Sometimes a group is identified by a pair, such as school and subject.
A product grouping index retains both labels. The `via` arrays provide
one coordinate per factor, and the implementation converts that tuple
to one flat group index. The marginalization then collects the rows
sharing that school–subject pair before summing over the class.


Product fibrations are supported on each observe via
`via product(idx_a, idx_b)`, paired with an `over G * H` product
grouping plate on the marginalize header. The product-fibration
arity must match the grouping plate's arity. See the
[composition-rule semantics](../semantics/composition-rules.md) for
the denotation and
`tests/test_grouped_marginalize_combinations.py` for examples.

## End-to-end fit

<!-- python: skip -->
```python
from quivers.dsl import load
from quivers.inference import ELBO, AutoNormalGuide, SVI

program = load("docs/examples/source/crossed.qvr")
model = program.morphism  # underlying MonadicProgram
observations = {"response": response_tensor}

guide = AutoNormalGuide(model, observed_names={"response"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

for _ in range(2000):
    svi.step(domain_input, observations)
```

## See also

- [Monadic Programs](programs.md): the basic program semantics and
  the `rsample` / `log_joint` contract this page builds on.
- [DSL Programs and Let-Expressions](dsl-programs-and-lets.md#parametric-programs):
  the syntax for parametric templates and the axis-role
  clause used in the observe / plate steps.
- [Analysis Pipelines: Data and Formulas](analysis-data-and-formulas.md):
  the brms-style formula entry point that compiles a
  random-intercepts formula into this hierarchical program
  automatically.
