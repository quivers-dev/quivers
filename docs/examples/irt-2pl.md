# Two-Parameter Logistic IRT

## Overview

The 2PL [item response theory](https://en.wikipedia.org/wiki/Item_response_theory) model [@birnbaum1968traits] for binary item responses `y_{ij}` of respondent i to item j. Each respondent carries a unidimensional ability `theta_i`, each item carries a difficulty `b_j` and a discrimination `a_j` (positive by construction via a [LogNormal](https://en.wikipedia.org/wiki/Log-normal_distribution) prior), and the probability of a correct response is `sigmoid(a_j * (theta_i - b_j))`.

## QVR source

```qvr
# Two-Parameter Logistic Item Response Theory
#
# The two-parameter logistic IRT model for binary responses
# y_{ij} of respondent i to item j. Each item carries its own
# discrimination and difficulty; respondents carry a
# unidimensional ability.
#
# Generative structure:
#
#   ability_i     ~ Normal(0, 1)                  standardised ability
#   difficulty_j  ~ Normal(0, 1)                  per-item difficulty
#   discrim_j     ~ LogNormal(0, 1)               positive discrimination
#   y_{ij}        ~ Bernoulli(sigmoid(discrim_j * (ability_i - difficulty_j)))
#
# Resp enumerates the full 100 x 20 person-by-item crossing, one
# row per response; the per-row person and item indices are host
# data, read through the plate-gather idiom
# (ability[person_idx], difficulty[item_idx]).

object Person : FinSet 100
object Item : FinSet 20
object Resp : FinSet 2000
object Val : Real 1

program irt_2pl : Resp -> Val
    sample ability : Person <- Normal(loc=0.0, scale=1.0)
    sample difficulty : Item <- Normal(loc=0.0, scale=1.0)
    sample discrim : Item <- LogNormal(loc=0.0, scale=1.0)

    let theta = ability[person_idx]
    let b = difficulty[item_idx]
    let a = discrim[item_idx]
    let eta = a * (theta - b)
    let p = sigmoid(eta)

    observe y : Resp <- Bernoulli(probs=p)
    return p

export irt_2pl
```

## Understanding the model

When a person's ability equals an item's difficulty, the success probability
is [`sigmoid(0) = 0.5`](../tutorials/qvr/02-glms.md#logistic-regression). Raising ability above difficulty increases it. A larger
positive discrimination makes the transition from low to high success sharper,
so the item distinguishes more strongly between nearby abilities.

Each response row identifies both a person and an item. For row indices
`person_idx[n] = 2` and `item_idx[n] = 5`, the likelihood uses ability 2,
difficulty 5, and discrimination 5. These [three parameters](https://mc-stan.org/docs/stan-users-guide/regression.html#multilevel-2pl-model)
determine the response probability together [@stanRegression]. Priors help
anchor the latent ability scale; ability and difficulty must be interpreted
on that same scale. [Posterior predictive](../tutorials/python/08-analysis-pipelines.md#posterior-predictive-checks)
response probabilities express their combined effect on the observed outcome.

## Walkthrough

`Person` and `Item` index the respondent and question parameters. `Resp`
indexes the observed respondent-item pairs; here it enumerates the full
crossing of 100 persons and 20 items, though a design with missing pairs would
simply list fewer rows. The Bernoulli family supplies the binary value type of each
response, while `return p` returns real-valued probabilities, reflected in
`Val : Real 1`. Indexing a response and describing its value are different
roles, which the program's annotations should keep distinct.

`ability`, `difficulty`, and `discrim` are plate-bound on `Person` and `Item`; `discrim` carries a LogNormal prior so it's positive by construction. The runtime supplies `person_idx` and `item_idx` at fit time, and the gather idiom `ability[person_idx]` / `difficulty[item_idx]` / `discrim[item_idx]` realizes the standard cross-classified design. The Bernoulli link `sigmoid(a * (theta - b))` is the canonical [logistic](https://en.wikipedia.org/wiki/Logistic_function) form of the 2PL response function.

## Try it

The blocks below simulate the full 100 x 20 response matrix, check the compiled
log joint against a hand-written density, and then fit with SVI and NUTS. The
program reads its data by name, so its input `x_in` is a single dummy row: that
leading axis is a particle axis, and a longer one would score the whole data
set once per row.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/irt_2pl.qvr")
model = prog.morphism

n_person, n_item = 100, 20
n_resp = n_person * n_item
ability_true = torch.randn(n_person)
difficulty_true = torch.randn(n_item)
discrim_true = (0.5 * torch.randn(n_item)).exp()

# Every person answers every item: row n is (person n // 20, item n % 20).
person_idx = torch.arange(n_person).repeat_interleave(n_item)
item_idx = torch.arange(n_item).repeat(n_person)
eta_true = discrim_true[item_idx] * (
    ability_true[person_idx] - difficulty_true[item_idx]
)
p_true = torch.sigmoid(eta_true)
y = torch.bernoulli(p_true)

observations = {"person_idx": person_idx, "item_idx": item_idx, "y": y}
x_in = torch.zeros(1, 1)
```

### Checking the log joint

```python
D = torch.distributions
truth = {"ability": ability_true, "difficulty": difficulty_true, "discrim": discrim_true}
oracle = (
    D.Normal(0.0, 1.0).log_prob(ability_true).sum()
    + D.Normal(0.0, 1.0).log_prob(difficulty_true).sum()
    + D.LogNormal(0.0, 1.0).log_prob(discrim_true).sum()
    + D.Bernoulli(p_true).log_prob(y).sum()
)
compiled = model.log_joint(x_in, {**truth, **observations}).sum()
print(f"compiled log joint: {float(compiled):.4f}")
print(f"torch log joint:    {float(oracle):.4f}")
assert torch.allclose(compiled, oracle, rtol=1e-5)
```

### SVI fit

The 2PL's location, scale, and sign are fixed only softly, by the Normal(0, 1)
prior on ability, so the checks compare quantities that do not depend on that
indeterminacy. Each ability is estimated from 20 binary responses, which bounds
how well it can be recovered: the check asks for a correlation of at least 0.85
between posterior-mean and true abilities. Each difficulty rests on 100
responses, and the check asks for a correlation of at least 0.9. Finally, the
fitted response probabilities should track the true ones: their mean absolute
error should be at most half of what a constant probability of 0.5 leaves on
these data.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "person_idx", "item_idx"})
optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
for _ in range(1500):
    svi.step(x_in, observations)

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(500)]
post = {k: torch.stack([d[k] for d in draws]).mean(0) for k in truth}
p_svi = torch.stack(
    [
        torch.sigmoid(
            d["discrim"][item_idx] * (d["ability"][person_idx] - d["difficulty"][item_idx])
        )
        for d in draws
    ]
).mean(0)


def corr(a, b):
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


baseline_mae = float((p_true - 0.5).abs().mean())
svi_mae = float((p_svi - p_true).abs().mean())
print(f"ability correlation:    {corr(post['ability'], ability_true):.3f}")
print(f"difficulty correlation: {corr(post['difficulty'], difficulty_true):.3f}")
print(f"probability MAE: {svi_mae:.3f} (constant 0.5: {baseline_mae:.3f})")

assert corr(post["ability"], ability_true) > 0.85
assert corr(post["difficulty"], difficulty_true) > 0.9
assert svi_mae < 0.5 * baseline_mae
```

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md) samples the 140-dimensional posterior
over the full response matrix and is held to the same three checks.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=100, num_samples=100, num_chains=1)
result = mc.run(model, x_in, observations)

s = {k: result.samples[k].reshape(-1, v.numel()) for k, v in truth.items()}
p_nuts = torch.sigmoid(
    s["discrim"][:, item_idx] * (s["ability"][:, person_idx] - s["difficulty"][:, item_idx])
).mean(0)
nuts_mae = float((p_nuts - p_true).abs().mean())
print(f"ability correlation:    {corr(s['ability'].mean(0), ability_true):.3f}")
print(f"difficulty correlation: {corr(s['difficulty'].mean(0), difficulty_true):.3f}")
print(f"probability MAE: {nuts_mae:.3f}")
print(f"divergences: {int(result.divergence_counts.sum())}")

assert corr(s["ability"].mean(0), ability_true) > 0.85
assert corr(s["difficulty"].mean(0), difficulty_true) > 0.9
assert nuts_mae < 0.5 * baseline_mae
```

## Categorical perspective

The 2PL is a Kleisli morphism over the Person + Item plate structure in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The plate-gather operations are pullbacks of indexed kernels along the response-row index maps `person_idx : Resp -> Person` and `item_idx : Resp -> Item`. [@giry1982categorical]

## Sources and further reading

- *A categorical approach to probability theory* [@giry1982categorical].
