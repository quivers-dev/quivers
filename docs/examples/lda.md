# Document-level topic mixture

## Overview

The source implements a document-level topic mixture with Dirichlet priors
on document topic weights and topic vocabulary distributions. It shares its
building blocks with [latent Dirichlet allocation](https://www.jmlr.org/papers/v3/blei03a.html),
but the grouped block selects one topic for all words in a document. Its
likelihood is [@blei2003lda]

$$
p(w_d \mid \theta_d, \phi)
= \sum_{k=1}^{K} \theta_d[k] \prod_{n:\,d[n]=d} \phi_k[w_n].
$$

Canonical LDA instead assigns a topic independently to each token conditional
on the document mixture, giving a product of per-token mixture sums. The
historical filename `lda.qvr` does not change the grouping written in source.

## QVR source

```qvr
# Document-level topic mixture
#
# Each document draws topic weights, and each topic draws a vocabulary
# distribution. The grouped marginalize block sums over one topic shared
# by every observed token in that document:
#
#   p(w_d | theta_d, phi) = sum_k theta_d[k] * product_n phi[k, w_d[n]].
#
# This differs from canonical LDA, whose independent per-token assignments
# give product_n sum_k theta_d[k] * phi[k, w_d[n]].
# Token indexes observed positions; Vocab indexes possible token values.
# Mix is the real vector space containing the returned topic weights.
#
# Background: Blei, Ng & Jordan (2003), Latent Dirichlet Allocation.

composition log_prob [level=algebra]

object Doc : FinSet 20
object Topic : FinSet 3
object Vocab : FinSet 50
object Token : FinSet 200
object Mix : Real 3

program lda(alpha : Real, beta : Real) : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=beta) [over=Vocab, iid_over=Topic]

    marginalize z : Topic <- Categorical(probs=theta) [over=Doc, reduction=logsumexp]
        observe w : Token <- Categorical(probs=phi[z]) [via=word_idx]

    return theta

export lda
```

## Understanding the model

A document-level topic mixture asks which single topic best explains all
words in a document. For a candidate topic, multiply its probabilities for
those words; then sum these document likelihoods using the document's topic
weights. One rare word can thus change how the entire document is assigned.

Canonical LDA lets each word choose its own topic. Its likelihood is a
product of per-word [mixture sums](mixture-model.md#understanding-the-model).
The grouped block in this source uses [one assignment per document](../tutorials/qvr/04-marginalize.md#hierarchical-mixtures-with-grouping),
so it [sums after multiplying the word likelihoods](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#vectorizing-mixtures)
[@stanFiniteMixtures]. This order makes all words in the document contribute
to the evidence for a single topic.

## Walkthrough

`object Doc : FinSet 20`, `object Topic : FinSet 3`, `object Vocab : FinSet 50`, and `object Token : FinSet 200` declare the four discrete sets: $D = 20$ documents, $K = 3$ topics, $V = 50$ vocabulary items, and $N = 200$ corpus-wide word positions. `Token` is the plate the corpus is observed over and `Vocab` is the set each token takes its value in, so the two stay distinct even when a corpus happens to hold as many positions as types. `object Mix : Real 3` is the value space of the returned per-document topic mixture, a point of the topic simplex embedded in $\mathbb{R}^K$. `composition log_prob [level=algebra]` selects the log-probability semiring, so the log-densities scored by `observe` accumulate additively.

The two `sample` steps draw the document-topic and topic-vocabulary simplex matrices under symmetric Dirichlet priors:

- `sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]` draws a $D \times K$ matrix in which each row is an independent symmetric-$\alpha$ Dirichlet over the topic simplex. The `over=Topic` clause names the family's event axis (Dirichlet event-rank 1), and `iid_over=Doc` asserts each row is independent.
- `sample phi : Topic <- Dirichlet(beta) [over=Vocab, iid_over=Topic]` is the symmetric construction: a $K \times V$ matrix whose every row is an independent Dirichlet over the vocabulary simplex.

The scoped marginalize block

<!-- compile: false -->
```qvr
marginalize z : Topic <- Categorical(probs=theta) [over=Doc, reduction=logsumexp]
    observe w : Token <- Categorical(probs=phi[z]) [via=word_idx]
```

introduces one shared topic latent $z_d$ per document under a Categorical prior
parameterized by the document-shaped `theta`. The body's
`Categorical(probs=phi[z])` looks up that topic's vocabulary row and scores every
observed token in the document. The `[over=Doc]` grouping plate accumulates
each observation into its document; `[via=word_idx]` names the runtime
fibration from each word position into its document. The agenda evaluates the
`[reduction=logsumexp]` reduction over the $K$ topics, recovering the
document-level mixture likelihood
$\sum_k \theta_d[k] \prod_{n:\,d[n]=d} \phi_k[w_n]$.

In the checked computation, this block is a helper that enumerates `Topic`
under the prelude handler and returns one aggregate per `Doc`. The `word_idx`
fibration scatter-sums word likelihoods into those document cells before
`logsumexp` reduces over topics. Thus the reference-machine trace and every
supporting target preserve a shared topic assignment per document, not an
independent marginal for each token. See [Probabilistic
programs](../reference/qvr/probabilistic-programs.md#exact-marginalization).

Finally `return theta` projects the program's joint kernel onto the document-topic mixture; the per-document topic mixture is the natural quantity to inspect after fitting.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

```python
import torch
import torch.nn.functional as F
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/lda.qvr")
fit  = prog.lda(alpha=1.0, beta=0.5)
model = fit.morphism

D, T, V, Npd = 20, 3, 50, 10

# Sharp topic-word and topic-mixture matrices give a recoverable corpus.
phi_true   = F.softmax(torch.randn(T, V) * 2.0, dim=-1)
theta_true = F.softmax(torch.randn(D, T) * 1.5, dim=-1)

word_idx, tokens = [], []
for d in range(D):
    for _ in range(Npd):
        z  = torch.distributions.Categorical(theta_true[d]).sample()
        wt = torch.distributions.Categorical(phi_true[z]).sample()
        word_idx.append(d)
        tokens.append(int(wt))
word_idx = torch.tensor(word_idx, dtype=torch.long)
w        = torch.tensor(tokens,   dtype=torch.long)

N = w.shape[0]
observations = {"w": w, "word_idx": word_idx}
x_in = torch.zeros(1, 1)
```

The per-document topic weights `theta` and per-topic vocabulary distributions `phi` remain continuous simplex sites. The shared document topic `z` is integrated out by the grouped block.

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"w", "word_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(100)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Categorical perspective

Marginalization pushes the joint measure on topic assignments and words onto
the word observations by forgetting the assignments. Here it forgets one
topic per document: the observation-to-document map first groups row scores,
then the finite sum eliminates the shared topic. This is a weighted-kernel
calculation; the grouping map determines which observations share a latent.

## See also

- [Bayesian Gaussian Mixture Model](mixture-model.md) for a per-row mixture whose component assignment is integrated out in closed form by the likelihood, carrying no discrete latent and no `marginalize` block.

## Sources and further reading

- *Latent Dirichlet Allocation* [@blei2003lda].
