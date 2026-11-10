# Bilinear Tensor Contraction

## Overview

Bilinear scoring composes two embeddings through a third-order interaction tensor, the core of the neural tensor layer of Socher and colleagues [@socher2013tensor]. Here the model scores predicate-argument pairs: each item $i$ carries a predicate embedding $P_{i,:}$ and an argument embedding $A_{i,:}$, and a shared interaction tensor $W$ maps each embedding pair to a score for every point $g$ on a judgment scale:

$$
s_{i,g} \;=\; \sum_{b}\sum_{c} P_{i,b}\, A_{i,c}\, W_{b,c,g}.
$$

In quivers this three-way join is a single `contraction` declaration: an [operad](https://ncatlab.org/nlab/show/operad)-style n-ary morphism that combines typed input arrows under a named composition rule. The einsum wiring is inferred from the typed signature, so the source spells out only which arrow plugs into which wire.

## QVR source

```qvr
# Bilinear Tensor Contraction
#
# A bilinear plausibility scorer over predicate-argument pairs.
# Every item (a predicate-argument pair) carries two low-rank
# embeddings: one describing its predicate and one describing its
# argument. A third-order interaction tensor combines the two
# embeddings into a score for each point on a judgment scale.
#
# Structural form:
#
#   pred_embed  : Item -> PredDim               predicate embedding
#   arg_embed   : Item -> ArgDim                argument embedding
#   interaction : PredDim * ArgDim -> Judgment  bilinear form
#
# The contraction joins the three arrows in one operadic step:
# the (i, s) entry of the resulting tensor is
#
#   sum_b sum_c pred_embed[i, b] * arg_embed[i, c] * interaction[b, c, s],
#
# the classic bilinear (neural-tensor-layer) score. The wiring is
# inferred from the typed signature: PredDim and ArgDim appear in
# two inputs each but not in the output, so both are contracted;
# Item and Judgment appear in the output, so both propagate.

composition real [level=algebra]

object Item : FinSet 4
object PredDim, ArgDim : FinSet 2
object Judgment : FinSet 3

morphism pred_embed : Item -> PredDim [role=latent]
morphism arg_embed : Item -> ArgDim [role=latent]
morphism interaction : (PredDim * ArgDim) -> Judgment [role=latent]

contraction bilinear_score (
    p : Item -> PredDim,
    a : Item -> ArgDim,
    w : (PredDim * ArgDim) -> Judgment,
) : Item -> Judgment [rule=real]

define plausibility = bilinear_score(pred_embed, arg_embed, interaction)

export plausibility
```

## Understanding the model

For each item and judgment, the score considers every pair of predicate
and argument features. Multiplying their two coordinates by the matching
interaction coefficient gives one contribution; summing all feature pairs
gives the final score. A two-dimensional predicate and argument thus
contribute four terms to each judgment score.

The `Item` index survives because the calculation produces an answer for
each item; `Judgment` survives because each item has several scores.
`PredDim` and `ArgDim` disappear because they are summed out. This is the
meaning of the [wiring specification](../tutorials/python/07-composition-rules.md#operadic-n-ary-contractions), independently of whether the resulting
score later becomes a Gaussian mean, a classification logit, or another
quantity in a larger model.

## Walkthrough

The three latent declarations introduce the two embedding maps and the interaction tensor as first-class arrows: `pred_embed` and `arg_embed` have tensors of shape `(4, 2)`, and `interaction`, whose domain is the product `PredDim * ArgDim`, has a tensor of shape `(2, 2, 3)`.

The `contraction` declaration types each input wire and names the composition rule:

<!-- compile: false -->
```qvr
contraction bilinear_score (
    p : Item -> PredDim,
    a : Item -> ArgDim,
    w : (PredDim * ArgDim) -> Judgment,
) : Item -> Judgment [rule=real]
```

The required `rule=` option references a registered composition rule; `real` is the ordinary sum-product [semiring](https://en.wikipedia.org/wiki/Semiring) on $\mathbb{R}$, so contracted axes are joined by multiply-then-sum, exactly a [`torch.einsum`](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html). The compiler infers the wiring from the typed signature: `PredDim` and `ArgDim` each appear in two inputs but not in the output, so both are contracted; `Item` and `Judgment` appear in the output, so both propagate. The inferred spec is `ab, ac, bcd -> ad`. For contractions the inference cannot express, the option block also admits a `share=[...]` clause (keep a listed axis out of contraction) and a `wiring="..."` clause (verbatim einsum escape hatch); see the [contractions guide](../guides/dsl-contractions.md).

The `define` binding invokes the contraction on the three declared arrows. Each call site checks the argument count and per-argument shape against the declared wires, then runs the wiring and returns a fresh arrow with the declared `Item -> Judgment` typing; that arrow is what the module exports. Note that a multi-line input list takes a trailing comma before the closing parenthesis.

## Try it

The source defines a deterministic score, not a probability model: its exported arrow is the contraction of the three declared arrows, whose `[role=latent]` tensors are learned parameters with entries in $(0, 1)$. The blocks below build the probabilistic model around the declared contraction in Python. Every entry of the two embeddings and of the interaction tensor gets an independent $\mathcal{N}(0, 1)$ prior, and each of several replicate judgments of every (item, scale point) cell is scored under $y_{r, i, g} \sim \mathcal{N}(s_{i, g}, \sigma^2)$ with $s$ computed by the compiled `bilinear_score` wiring. The model is built in Python because a QVR program cannot yet call a `contraction` or hold a third-order latent tensor it can index by three binders; the compiled contraction is available through the [`Compiler`](../api/dsl/compiler.md#quivers.dsl.compiler.Compiler) environment's `contractions` mapping.

### Generating synthetic data

Draw ground-truth embeddings and an interaction tensor, push them through the bilinear form, and add Normal noise to 50 replicate judgments of every cell. The three `sample` sites of the [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) built here are named after the arrows the source declares, and every shape is read off the compiled module rather than restated. The block first confirms that the compiled wiring is the einsum `ab, ac, bcd -> ad` the typed signature implies.

```python
import torch
import torch.distributions as D

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous import Draw, Let, MonadicProgram, Observe
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file

torch.manual_seed(0)

compiler = Compiler(parse_file("docs/examples/source/tensor_contraction.qvr"))
compiler.compile()
wiring = compiler.contractions["bilinear_score"].wiring

pred_shape = tuple(compiler.morphisms["pred_embed"].tensor.shape)
arg_shape = tuple(compiler.morphisms["arg_embed"].tensor.shape)
inter_shape = tuple(compiler.morphisms["interaction"].tensor.shape)
n_item, n_judgment = pred_shape[0], inter_shape[-1]
n_rep = 50
sigma = 0.5


def bilinear_score(pred, arg, inter):
    """Run the declared `bilinear_score` contraction on three tensors."""
    return wiring.apply(pred, arg, inter)


probe = [torch.randn(pred_shape), torch.randn(arg_shape), torch.randn(inter_shape)]
assert torch.allclose(
    bilinear_score(*probe), torch.einsum("ab,ac,bcd->ad", *probe), atol=1e-6,
)


class EntrywisePrior(ContinuousMorphism):
    """Independent Normal(0, 1) prior over one arrow's entries."""

    def __init__(self, shape):
        dim = 1
        for axis in shape:
            dim *= int(axis)
        super().__init__(Unit, Euclidean(name="Entries", dim=dim))
        self._shape = tuple(int(axis) for axis in shape)

    def rsample(self, x, sample_shape=torch.Size()):
        return D.Normal(torch.zeros(self._shape), 1.0).rsample()

    def log_prob(self, x, y):
        return D.Normal(0.0, 1.0).log_prob(y.reshape(self._shape)).sum()


class JudgmentLikelihood(ContinuousMorphism):
    """Normal replicate judgments around the contracted bilinear score."""

    def __init__(self, sigma):
        super().__init__(
            Euclidean(name="Score", dim=n_judgment),
            Euclidean(name="Judgment", dim=n_judgment),
        )
        self._sigma = sigma

    def rsample(self, x, sample_shape=torch.Size()):
        return D.Normal(x, self._sigma).rsample((n_rep,))

    def log_prob(self, x, y):
        y = y.reshape(n_rep, *x.shape)
        return D.Normal(x, self._sigma).log_prob(y).sum()


model = MonadicProgram(
    domain=Euclidean(name="Ix", dim=1),
    codomain=Euclidean(name="Judgment", dim=n_judgment),
    steps=[
        Draw(names=("pred_embed",), morphism=EntrywisePrior(pred_shape)),
        Draw(names=("arg_embed",), morphism=EntrywisePrior(arg_shape)),
        Draw(names=("interaction",), morphism=EntrywisePrior(inter_shape)),
        Let(
            name="mu",
            value=lambda env: bilinear_score(
                env["pred_embed"].reshape(pred_shape),
                env["arg_embed"].reshape(arg_shape),
                env["interaction"].reshape(inter_shape),
            ),
        ),
        Observe(
            names=("judgment",), morphism=JudgmentLikelihood(sigma), args=("mu",)
        ),
    ],
    return_vars=("judgment",),
)

true_pred_embed = torch.randn(pred_shape)
true_arg_embed = torch.randn(arg_shape)
true_interaction = torch.randn(inter_shape)
true_score = bilinear_score(true_pred_embed, true_arg_embed, true_interaction)
judgment = D.Normal(true_score, sigma).sample((n_rep,))

observations = {"judgment": judgment}
x_in = torch.zeros(1, 1)

print("true scores:", true_score.round(decimals=2).tolist())
```

The `mu` step is a deterministic `let`: it reads the three sampled tensors out of the trace environment, reshapes each back to its declared arrow shape, and returns the $4 \times 3$ contraction. Reshaping is what makes the step total, since a clamped site arrives flattened, and the third-order `interaction` is the one arrow whose shape a flat vector cannot be read off.

### Checking the joint density

At the true tensors, the program's [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) should equal the three prior terms plus the Normal likelihood of every replicate judgment, with the score written as an einsum rather than through the compiled wiring.

```python
truth = {
    "pred_embed": true_pred_embed,
    "arg_embed": true_arg_embed,
    "interaction": true_interaction,
    "judgment": judgment,
}
log_joint = model.log_joint(x_in, truth).reshape(-1)[0]

score = torch.einsum("ab,ac,bcd->ad", true_pred_embed, true_arg_embed, true_interaction)
by_hand = (
    D.Normal(0.0, 1.0).log_prob(true_pred_embed).sum()
    + D.Normal(0.0, 1.0).log_prob(true_arg_embed).sum()
    + D.Normal(0.0, 1.0).log_prob(true_interaction).sum()
    + D.Normal(score, sigma).log_prob(judgment).sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

### SVI fit

Fit an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) by [`SVI`](../api/inference/svi.md#svi) and push 200 guide draws through the contraction. The embeddings and the interaction tensor are not identified: any invertible change of basis in `PredDim` or `ArgDim`, undone in the interaction tensor, leaves every score unchanged. The score tensor is identified, so the check compares fitted scores with true ones, next to the per-cell mean of the replicate judgments as a baseline.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"judgment"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
losses = [svi.step(x_in, observations) for _ in range(1000)]

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(200)]
fit_score = torch.stack([
    bilinear_score(
        d["pred_embed"].reshape(pred_shape),
        d["arg_embed"].reshape(arg_shape),
        d["interaction"].reshape(inter_shape),
    )
    for d in draws
]).mean(0)

fit_rmse = float((fit_score - true_score).pow(2).mean().sqrt())
cell_mean_rmse = float((judgment.mean(0) - true_score).pow(2).mean().sqrt())
print(f"ELBO loss: {losses[0]:.1f} -> {sum(losses[-100:]) / 100:.1f}")
print(f"score RMSE: fitted {fit_rmse:.3f}, replicate means {cell_mean_rmse:.3f}")

assert fit_rmse < 0.15
```

Each cell has 50 judgments with noise scale 0.5, so its replicate mean has a standard error of about 0.07, and the tolerance allows about twice that. With four items and three scale points, the contraction places no real constraint on the twelve scores (the embeddings and interaction tensor have more free entries than there are cells), so the fit should not be expected to beat the replicate means; the check is that the declared contraction, fitted through its factors, reproduces the scores the data identify.

### NUTS posterior

Because the priors are declared inside the program, [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) targets it directly. The chain below is short; the non-identified factors mix poorly, so a real analysis would monitor the scores rather than the factor entries, across several chains.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

assert torch.isfinite(result.log_densities).all()
print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

Binary composition `>>` is the 2-ary case of a wider operadic structure: a `contraction` denotes an n-ary morphism in the [multicategory](https://ncatlab.org/nlab/show/multicategory) of tensor spaces over the active composition rule, with the wiring specification fixing which axes are joined and which survive to the output (see [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions) for the denotation). The bilinear score is the case $n = 3$: two argument tensors over a shared `Item` axis and a kernel over the two embedding axes, folded under $\otimes$ and joined under $\bigoplus$ of the sum-product semiring. Here the three inputs share two different feature axes, which the explicit wiring contracts in one operation.

## See also

- [DSL Contractions](../guides/dsl-contractions.md) for declaration syntax, wiring inference, and the `share=` / `wiring=` clauses.
- [Composition Rules § 4](../semantics/composition-rules.md#4-operadic-contractions) for the categorical semantics of operadic contractions.
- [Probabilistic Matrix Factorization](pmf.md) for the two-way analogue, an inner-product score written as a QVR program over observed (user, item) pairs.

## Sources and further reading

- *Reasoning With Neural Tensor Networks for Knowledge Base Completion* [@socher2013tensor].
