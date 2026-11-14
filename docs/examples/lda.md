# Latent Dirichlet Allocation

## Overview

Latent Dirichlet allocation (LDA) [@blei2003lda] models a corpus of documents as mixtures of topics. Each document draws topic weights $\theta_d$ from a Dirichlet, each topic draws a distribution $\phi_k$ over the vocabulary, and each token draws its own topic from its document's weights and then its word from that topic. The program sums each token's topic out, so its likelihood is a product over tokens of per-token mixtures,

$$
p(w \mid \theta, \phi) = \prod_{n} \sum_{k=1}^{K} \theta_{d(n)}[k]\, \phi_k[w_n],
$$

where $d(n)$ identifies the document containing token $n$. The joint differs from a model that draws one topic for a whole document.

## QVR source

```qvr
# Latent Dirichlet Allocation
#
# Each document draws topic weights theta_d, each topic draws a
# vocabulary distribution phi_k, and each token draws its own topic
# from its document's weights and then its word from that topic. The
# per-token topic is summed out, one marginalization per token:
#
#   p(w | theta, phi) = prod_n sum_k theta[d(n), k] * phi[k, w_n].
#
# doc_idx maps each token to its document, so theta_tok gathers one
# row of topic weights per token. Token indexes the corpus positions;
# Vocab indexes the values a token takes.
#
# Reference: [Blei, Ng, and Jordan 2003](https://www.jmlr.org/papers/v3/blei03a.html).

object Doc : FinSet 20
object Topic : FinSet 3
object Vocab : FinSet 50
object Token : FinSet 1000
object Mix : Real 3

program lda(alpha : Real, beta : Real) : Token -> Mix
    sample theta : Doc <- Dirichlet(concentration=alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(concentration=beta) [over=Vocab, iid_over=Topic]
    let theta_tok = theta[doc_idx]

    marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token, reduction=logsumexp]
        observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]

    return theta

export lda
```

## Understanding the model

Each word picks its own topic. A document about two subjects can thus draw some words from one topic and some from the other, and its topic weights record the proportions. The per-token sum is what distinguishes LDA from a document-level mixture, in which one topic is drawn for the whole document and the sum over topics is taken after multiplying the word probabilities, $\sum_k \theta_d[k] \prod_{n: d(n) = d} \phi_k[w_n]$. The second check below shows the program computes the former.

## Walkthrough

- `object Doc : FinSet 20`, `object Topic : FinSet 3`, `object Vocab : FinSet 50`, and `object Token : FinSet 1000` are the documents, topics, vocabulary, and corpus positions. `object Mix : Real 3` is the value space of the returned topic weights.
- `program lda(alpha : Real, beta : Real)` is a template: `prog.lda(alpha=1.0, beta=1.0)` instantiates it with symmetric Dirichlet concentrations.
- `sample theta : Doc <- Dirichlet(...) [over=Topic, iid_over=Doc]` draws one row of topic weights per document, and `sample phi : Topic <- Dirichlet(...) [over=Vocab, iid_over=Topic]` one vocabulary distribution per topic.
- `let theta_tok = theta[doc_idx]` gathers, for each token, its document's row of topic weights. `doc_idx` is input data holding one document index per token.
- `marginalize z : Topic <- Categorical(probs=theta_tok) [over=Token, reduction=logsumexp]` gives every token its own topic and sums it out by log-sum-exp: a grouped `marginalize` scores one class per group, and here each group is a single token. `observe w : Token <- Categorical(probs=phi[z]) [via=token_idx]` scores each token's word under its topic, with `token_idx` mapping each observation to its own token row.
- The [support matrix](../transpile-support.md) records backend coverage for the gathered weights and grouped marginalization.

## Try it

### Generating synthetic data

Sharp topics and topic weights generate 50 tokens per document. A second draw of 50 tokens per document from the same documents is held out.

```python
import torch
import torch.nn.functional as F

from quivers.dsl import load

DOCS, TOPICS, VOCAB, PER_DOC = 20, 3, 50, 50

prog = load("docs/examples/source/lda.qvr")
model = prog.lda(alpha=1.0, beta=1.0).morphism

generator = torch.Generator().manual_seed(2)
true_phi = F.softmax(torch.randn(TOPICS, VOCAB, generator=generator) * 2.0, dim=-1)
true_theta = F.softmax(torch.randn(DOCS, TOPICS, generator=generator) * 1.5, dim=-1)


def corpus(theta, phi, generator):
    doc = torch.arange(DOCS).repeat_interleave(PER_DOC)
    z = torch.multinomial(theta[doc], 1, generator=generator).squeeze(-1)
    return doc, torch.multinomial(phi[z], 1, generator=generator).squeeze(-1)


doc_idx, w = corpus(true_theta, true_phi, generator)
_, w_held_out = corpus(true_theta, true_phi, generator)
token_idx = torch.arange(DOCS * PER_DOC)
observations = {"w": w, "doc_idx": doc_idx, "token_idx": token_idx}
x_in = torch.zeros(1, 1)
```

### The log joint against a per-token mixture

At a traced draw of `theta` and `phi`, the compiled runtime's log joint must equal the Dirichlet priors plus $\sum_n \log \sum_k \theta_{d(n)}[k] \phi_k[w_n]$ in float64 to a relative 1e-5, and the document-level mixture must differ by more than a relative 1e-3. The [reference machine](../developer/qiec.md#reference-evaluator), given fixed `theta` and `phi`, must equal the same reference calculation to a relative 1e-9.

```python
import math

import torch.distributions as D

from quivers.inference import trace


def priors(theta, phi):
    one = torch.tensor(1.0, dtype=torch.float64)
    return D.Dirichlet(one.expand(TOPICS)).log_prob(theta).sum() + D.Dirichlet(one.expand(VOCAB)).log_prob(phi).sum()


def per_token(theta, phi, doc, words):
    return torch.log((theta[doc] * phi[:, words].T).sum(-1))


def document_level(theta, phi, doc, words):
    total = 0.0
    for d in range(DOCS):
        total += float(torch.logsumexp(torch.log(theta[d]) + torch.log(phi[:, words[doc == d]]).sum(-1), 0))
    return total


torch.manual_seed(1)
recorded = trace(model, x_in, observations=observations)
theta = recorded.sites["theta"].value.reshape(DOCS, TOPICS).double()
phi = recorded.sites["phi"].value.reshape(TOPICS, VOCAB).double()
expected = float(priors(theta, phi) + per_token(theta, phi, doc_idx, w).sum())
wrong = float(priors(theta, phi)) + document_level(theta, phi, doc_idx, w)
got = float(recorded.log_joint)
print(f"compiled {got:.6f}, oracle {expected:.6f}, document-level mixture {wrong:.6f}")
assert math.isclose(got, expected, rel_tol=1e-5)
assert not math.isclose(got, wrong, rel_tol=1e-3)

theta = D.Dirichlet(torch.ones(TOPICS, dtype=torch.float64)).sample((DOCS,))
phi = D.Dirichlet(torch.ones(VOCAB, dtype=torch.float64)).sample((TOPICS,))
run = prog.run(
    "lda",
    data={"alpha": 1.0, "beta": 1.0, "w": tuple(w.tolist()), "doc_idx": tuple(doc_idx.tolist()),
          "token_idx": tuple(token_idx.tolist())},
    sites={"theta": tuple(map(tuple, theta.tolist())), "phi": tuple(map(tuple, phi.tolist()))},
    seed=0,
)
expected = float(priors(theta, phi) + per_token(theta, phi, doc_idx, w).sum())
print(f"reference machine {run.log_joint:.9f}, oracle {expected:.9f}")
assert math.isclose(run.log_joint, expected, rel_tol=1e-9)
```

### A fit that did not pass its check

A maximum-likelihood fit on the training tokens used flat `Dirichlet(1)` priors and Adam on unconstrained logits of `theta` and `phi`, 1500 steps at learning rate 0.05. The document-completion criterion required the held-out per-token negative log likelihood under the fit to be more than 0.1 nats below a smoothed unigram model fitted to the same tokens, and less than 0.15 nats above the generating model's. The fit scored 3.25 nats against the unigram's 3.20 and the generating model's 3.00, so it failed. With 50 tokens per document and 50 free word probabilities per topic, the maximum-likelihood topics give held-out words they never saw almost no mass. A smoothing prior or a posterior average would be the next design to try, with its own thresholds. This fit does not demonstrate successful prediction of held-out words.

## Categorical perspective

Marginalization pushes the joint measure on topic assignments and words onto the words by forgetting the assignments. Here it forgets one topic per token: the gathered weights `theta_tok` give each token its own Categorical [kernel](https://en.wikipedia.org/wiki/Markov_kernel) from the one-point space to `Topic`, the topic-word kernel `phi` maps a topic to a word, and their [Kleisli](https://ncatlab.org/nlab/show/Kleisli+category) composite is the per-token mixture. The document-level model forgets one topic per document instead, which composes the kernels after taking the product over the document's words. The two orders give different measures, and the power check above separates them.
