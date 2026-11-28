# Referent choice and pronoun judgments

A sentence introduces one of two discourse referents, $a$ or $b$.
The pronoun either resolves to that referent or picks a fixed alternative
$c$. The QVR program stores the introduced referent in a named state
instance. Two choice instances weight the readings with separate logits.
The handler for the store runs inside the two choices, so each lexical
reading supplies its own value to the pronoun.

## Likelihood

Let $p=\operatorname{sigmoid}(\theta)$ be the prior probability of
introducing $a$ and $\lambda=\operatorname{sigmoid}(\phi)$ the prior
probability of resolving to the stored referent. The observation model
uses $m(a,a)=0.9$, $m(a,c)=0.2$, $m(b,b)=0.6$, and $m(b,c)=0.3$ as
conditional probabilities of a positive judgment. For $x\in\{a,b\}$,
write $A_x=\lambda\,m(x,x)+(1-\lambda)\,m(x,c)$. Then the probability
of a positive judgment is $Z=pA_a+(1-p)A_b$.

For a positive judgment, the lexical gradient is

$$
\frac{\partial\log Z}{\partial\theta}
= \frac{p A_a}{Z}-p
= \Pr(x=a\mid \mathrm{true})-p.
$$

For the pronoun logit, the corresponding derivative is the posterior
probability of the stored reading minus $\lambda$. For a negative
judgment the same identities hold with $m$ replaced by $1-m$. In either
case the identities follow by differentiating the Bernoulli prior masses
before summing the four products.

## QVR source

In the [complete QVR source](source/referent_judgments.qvr), `sentence`
selects and writes the referent, then selects the pronoun reading and reads
back the stored value. The weighted handler resumes both alternatives and
sums their answers. The exported `judgment` program observes one
Bernoulli response at the resulting probability.

## Try it

Run this from the repository root. The finite sum over readings gives an
independent likelihood and computes both posterior counts directly. The
reference and discharged programs also agree on the observation score
address.

<!-- python: run -->
```python
import torch

from quivers.continuous.tensor_backend import TensorBackend
from quivers.dsl import load
from quivers.qiec.builtins import SCORE_ADD
from quivers.qiec.discharge import discharge_module
from quivers.qiec.program_runtime import run_program

module = load("docs/examples/source/referent_judgments.qvr").qiec
assert module is not None
compiled = discharge_module(module)
assert compiled.obstacle("truth_probability") is None
response_prob = torch.tensor([[0.9, 0.2], [0.6, 0.3]], dtype=torch.float64)

for answer in (False, True):
    logits = torch.tensor([0.4, -0.6], dtype=torch.float64, requires_grad=True)
    p, resolution = logits.detach().sigmoid().unbind()
    referents = torch.stack((p, 1 - p))
    pronouns = torch.stack((resolution, 1 - resolution))
    joint = referents[:, None] * pronouns[None, :] * (
        response_prob if answer else 1 - response_prob
    )
    evidence = joint.sum()
    expected_gradient = torch.stack(
        (joint[0, :].sum() / evidence - p, joint[:, 0].sum() / evidence - resolution)
    )
    score_addresses = []
    for evaluator_module in (module, compiled.module):
        run = run_program(
            evaluator_module,
            "judgment",
            data={"lexical_logit": logits[0], "pronoun_logit": logits[1], "truth": answer},
            sites={},
            distributions=TensorBackend(),
        )
        (gradient,) = torch.autograd.grad(run.log_joint, logits)
        torch.testing.assert_close(run.log_joint.detach(), evidence.log())
        torch.testing.assert_close(gradient, expected_gradient)
        addresses = tuple(
            event.detail["address"]
            for event in run.result.trace
            if event.event == "operation.requested"
            and event.detail["operation"] == str(SCORE_ADD)
        )
        assert len(addresses) == 1
        score_addresses.append(addresses)
    assert score_addresses[0] == score_addresses[1]
```
