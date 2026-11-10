# Latent Dirichlet Allocation

## Overview

[Latent Dirichlet allocation](https://www.jmlr.org/papers/v3/blei03a.html) (LDA) [@blei2003lda] describes a corpus as a collection of documents, each of which mixes a small number of topics. Every document $d$ draws topic weights $\theta_d$ from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) distribution, every topic $k$ draws a distribution $\phi_k$ over the vocabulary, and every token $n$ first draws its own topic $z_n$ from its document's weights and then draws its word from that topic:

$$
\theta_d \sim \mathrm{Dirichlet}(\alpha), \quad
\phi_k \sim \mathrm{Dirichlet}(\beta), \quad
z_n \sim \mathrm{Categorical}(\theta_{d[n]}), \quad
w_n \sim \mathrm{Categorical}(\phi_{z_n}).
$$

The per-token topics are discrete, so the program sums them out exactly. Since each token has its own topic, the likelihood is a product of per-token [mixture](mixture-model.md#understanding-the-model) sums:

$$
p(w \mid \theta, \phi) = \prod_{n=1}^{N} \sum_{k=1}^{K} \theta_{d[n], k}\, \phi_{k, w_n}.
$$

## QVR source

```qvr
# Latent Dirichlet Allocation
#
# Each document draws topic weights, each topic draws a vocabulary
# distribution, and every token draws its own topic from its
# document's weights before drawing a word from that topic:
#
#   theta_d ~ Dirichlet(1.0)                 per-document topic weights
#   phi_k   ~ Dirichlet(0.5)                 per-topic word distribution
#   z_n     ~ Categorical(theta_{d[n]})      per-token topic
#   w_n     ~ Categorical(phi_{z_n})         observed word
#
# The marginalize block sums each token's topic out on its own, so the
# likelihood is a product of per-token mixtures:
#
#   p(w | theta, phi) = product_n sum_k theta[d[n], k] * phi[k, w[n]].
#
# Token indexes observed word positions; Vocab indexes possible word
# types. The host datum doc_idx maps each token to its document, and
# the let gathers that document's topic weights for every token. The
# marginalize block groups over Token itself: token_idx is the
# identity map on token positions, so every group holds one token and
# the topic sum is taken separately for each.
# Mix is the real vector space containing the returned topic weights.
#
# Reference: Blei, Ng & Jordan (2003), Latent Dirichlet Allocation,
# https://www.jmlr.org/papers/v3/blei03a.html

composition log_prob [level=algebra]

object Doc : FinSet 100
object Topic : FinSet 3
object Vocab : FinSet 30
object Token : FinSet 10000
object Mix : Real 3

program lda : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=1.0) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=0.5) [over=Vocab, iid_over=Topic]
    let theta_tok = theta[doc_idx]

    marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token, reduction=logsumexp]
        observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]

    return theta

export lda
```

## Understanding the model

A topic is a distribution over words, and a document is a distribution over topics. Two documents about the same subject tend to put weight on the same topics, so their word frequencies look alike; a document that straddles two subjects can split its weight between two topics. Because every token picks its own topic, a single document can draw some words from one topic and other words from another, which is what lets $\theta_d$ express a genuine mixture rather than a single label per document.

Summing out $z_n$ for one token multiplies each topic's weight in the token's document by that topic's probability of the observed word and adds the products. The order of operations matters: LDA sums over topics for each token and then multiplies across tokens. Multiplying the word probabilities within a document first and then summing over topics would instead give a model with one topic per document (a mixture of unigrams). In the [grouped form of `marginalize`](../tutorials/qvr/04-marginalize.md#hierarchical-mixtures-with-grouping), the grouping plate decides which of the two a program states: grouping tokens by document shares one topic per document, while grouping each token by itself, as this source does, gives every token its own topic.

Topic labels are exchangeable: permuting the $K$ topics, together with the columns of $\theta$, leaves the likelihood unchanged. Fitted topics are thus compared with the truth only after matching them up, and the document-word probabilities $\theta \phi$, which do not depend on the labeling, give a check that needs no matching at all.

## Walkthrough

`object Doc : FinSet 100`, `object Topic : FinSet 3`, `object Vocab : FinSet 30`, and `object Token : FinSet 10000` declare the four discrete sets: $D = 100$ documents, $K = 3$ topics, $V = 30$ word types, and $N = 10000$ corpus-wide word positions. `Token` is the plate the corpus is observed over and `Vocab` is the set each token takes its value in, so the two stay distinct even when a corpus happens to hold as many positions as types. `object Mix : Real 3` is the value space of the returned per-document topic mixture, a point of the topic simplex embedded in $\mathbb{R}^K$. `composition log_prob [level=algebra]` selects the log-probability semiring, so the log-densities scored by `observe` accumulate additively.

The two `sample` steps draw the document-topic and topic-vocabulary simplex matrices under symmetric Dirichlet priors:

- `sample theta : Doc <- Dirichlet(concentration=1.0) [over=Topic, iid_over=Doc]` draws a $D \times K$ matrix in which each row is an independent Dirichlet draw over the topic simplex. The `over=Topic` clause names the family's event axis (Dirichlet event-rank 1), and `iid_over=Doc` asserts each row is independent. A concentration of one is uniform on the simplex, so documents range from single-topic to evenly mixed.
- `sample phi : Topic <- Dirichlet(concentration=0.5) [over=Vocab, iid_over=Topic]` is the symmetric construction: a $K \times V$ matrix whose every row is an independent Dirichlet over the vocabulary simplex. A concentration below one favors topics that put most of their mass on a few words.

`let theta_tok = theta[doc_idx]` gathers, for every token, the topic weights of the document it belongs to. `doc_idx` is host data, an integer vector of length $N$ mapping each token position to its document; it is the only place the document structure of the corpus enters the program.

The marginalize block

<!-- compile: false -->
```qvr
marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token, reduction=logsumexp]
    observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]
```

introduces a topic latent $z$ with support `Topic`, one per group of the grouping plate named by `over=Token`, under a Categorical prior whose probabilities are the group's row of `theta_tok`. The body's `Categorical(probs=phi[z])` looks up the candidate topic's vocabulary row and scores the observed word, and `[via=token_idx]` maps each observed word to its group. The host datum `token_idx` is the identity on token positions, so every group holds exactly one word: the agenda enumerates the $K$ topics for each token, adds the prior log-weight to the word's log-probability, and applies `logsumexp` per token, giving $\log \sum_k \theta_{d[n], k}\, \phi_{k, w_n}$. These per-token terms are then summed over the `Token` plate. Replacing `over=Token` and `token_idx` by `over=Doc` and `doc_idx` would share one topic across each document instead. See [Probabilistic programs](../reference/qvr/probabilistic-programs.md#exact-marginalization) for the checked semantics of the block.

Finally `return theta` projects the program's joint kernel onto the document-topic mixture, the quantity one usually inspects after fitting.

## Try it

The blocks below simulate a corpus from the model itself, check the compiled joint against a hand-written LDA log density, fit the model by SVI and check that it recovers the topics, and run a short NUTS chain on the same joint.

### Generating synthetic data

The corpus has 100 documents of 100 tokens each, generated by the same draws the program describes: Dirichlet document weights and topic-word distributions at the source's concentrations, then a topic and a word for every token.

```python
import itertools

import torch
from quivers.dsl import load

D = torch.distributions

torch.manual_seed(0)
prog = load("docs/examples/source/lda.qvr")
model = prog.morphism

n_doc, n_topic, n_vocab, per_doc = 100, 3, 30, 100
n_token = n_doc * per_doc

true_theta = D.Dirichlet(torch.full((n_topic,), 1.0)).sample((n_doc,))
true_phi = D.Dirichlet(torch.full((n_vocab,), 0.5)).sample((n_topic,))

doc_idx = torch.arange(n_doc).repeat_interleave(per_doc)
topics = D.Categorical(true_theta[doc_idx]).sample()
w = D.Categorical(true_phi[topics]).sample()

token_idx = torch.arange(n_token)

observations = {"w": w, "doc_idx": doc_idx, "token_idx": token_idx}
x_in = torch.zeros(1, 1)
```

The program input `x_in` is a single row; the extents of the `Token` plate and of the gathered document weights come from `w`, `doc_idx`, and `token_idx`. The per-token topics `topics` are part of the simulation only. The model never sees them; it integrates them out.

### Checking the joint density

At the true $(\theta, \phi)$, [`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) should equal the two Dirichlet prior terms plus the LDA log-likelihood $\sum_n \log \sum_k \theta_{d[n], k}\, \phi_{k, w_n}$, written out by hand.

```python
truth = {"theta": true_theta, "phi": true_phi, **observations}
log_joint = model.log_joint(x_in, truth)[0]

token_lik = (true_theta[doc_idx] * true_phi[:, w].T).sum(dim=-1)
by_hand = (
    D.Dirichlet(torch.full((n_topic,), 1.0)).log_prob(true_theta).sum()
    + D.Dirichlet(torch.full((n_vocab,), 0.5)).log_prob(true_phi).sum()
    + token_lik.log().sum()
)
assert torch.allclose(log_joint, by_hand, rtol=1e-5)
print(f"log joint at the truth: {float(log_joint):.2f}")
```

The same check fails for a model with one topic per document, whose log-likelihood $\sum_d \log \sum_k \theta_{d,k} \prod_{n \in d} \phi_{k, w_n}$ differs from this one whenever a document mixes topics.

### SVI fit

Fit an [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) by [`SVI`](../api/inference/svi.md#svi), summarize $\theta$ and $\phi$ by the mean of 100 guide draws, and match fitted topics to true topics by the permutation with the smallest total absolute difference in $\phi$ (with $K = 3$ there are six to try).

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"w", "doc_idx", "token_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))
losses = [svi.step(x_in, observations) for _ in range(800)]

with torch.no_grad():
    draws = [guide.rsample(x_in) for _ in range(100)]
fit_theta = torch.stack([d["theta"] for d in draws]).mean(0).reshape(n_doc, n_topic)
fit_phi = torch.stack([d["phi"] for d in draws]).mean(0).reshape(n_topic, n_vocab)

perm = list(min(
    itertools.permutations(range(n_topic)),
    key=lambda p: float((fit_phi[list(p)] - true_phi).abs().sum()),
))
fit_phi, fit_theta = fit_phi[perm], fit_theta[:, perm]

phi_error = float((fit_phi - true_phi).abs().max())
theta_error = float((fit_theta - true_theta).abs().mean())

# Document-word probabilities: label-free, compared by total variation.
true_dw = true_theta @ true_phi
counts = torch.zeros(n_doc, n_vocab).index_put_(
    (doc_idx, w), torch.ones(n_token), accumulate=True,
)
model_tv = float(0.5 * (fit_theta @ fit_phi - true_dw).abs().sum(-1).mean())
count_tv = float(0.5 * (counts / per_doc - true_dw).abs().sum(-1).mean())

print(f"ELBO loss: {losses[0]:.1f} -> {sum(losses[-100:]) / 100:.1f}")
print(f"max |phi error|:  {phi_error:.3f}")
print(f"mean |theta error|: {theta_error:.3f}")
print(f"doc-word TV, model {model_tv:.3f} vs raw counts {count_tv:.3f}")

assert phi_error < 0.05
assert theta_error < 0.08
assert model_tv < 0.5 * count_tv
```

Each topic sees on the order of 3,300 tokens, so a topic-word probability has a sampling error of at most a few hundredths, and the $\phi$ tolerance allows for that. A document has only 100 tokens, so its weights are less certain; the $\theta$ check is on the mean error. The last assertion is a predictive one: the fitted model's document-word distributions should be much closer to the truth than each document's raw word frequencies, because the model pools what all documents say about each topic.

### NUTS posterior

[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) runs on the same joint, with every per-token topic still summed out and both simplex matrices latent. The chain below is short; a real analysis would run several chains and check their diagnostics.

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

Marginalization pushes the joint measure on topic assignments and words forward onto the words by forgetting the assignments. In LDA the assignments are indexed by `Token`, and the topic prior for each token is pulled back from `Doc` along the map `doc_idx : Token -> Doc`; grouping along the identity `token_idx : Token -> Token` then lets the finite sum eliminate each token's topic independently. Grouping along `doc_idx` instead would push token scores forward to `Doc` before the sum, which shares one topic across a whole document. The two programs differ only in which side of the fibration the finite sum is taken on, and that choice is the difference between LDA and a mixture of unigrams.

## See also

- [Bayesian Gaussian Mixture Model](mixture-model.md) for a per-row mixture whose component assignment is integrated out in closed form by the likelihood, with one global set of weights rather than per-document weights.
- [Finite marginalization tutorial](../tutorials/qvr/04-marginalize.md) for the grouped and ungrouped `marginalize` forms.

## Sources and further reading

- *Latent Dirichlet Allocation* [@blei2003lda].
