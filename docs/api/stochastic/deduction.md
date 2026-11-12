# Deduction Systems

A [weighted deduction system](../../guides/deduction.md#the-framework) derives conclusions from
weighted premises by inference rules, and combines the weights of derivations in a semiring
[@shieber1995principles; @goodman1999semiring]. `quivers.stochastic.deduction` re-exports the model
type, its abstract primitives, and the operations below: fitting adjusts rule weights, the Bayesian
wrapper places priors on them, and sampling draws yields from the normalized distribution over
sentences of one length.

| Submodule | Job |
|---|---|
| [`primitives`](deduction/primitives.md) | Abstract building blocks (`Axiom`, `Deduction`, `Goal`, `Schedule`, `DeductiveSystem`), used mainly to write new deduction classes. |
| [`fit`](deduction/fit.md) | Gradient fitting of point estimates (MAP or MLE) of the learnable log weights. |
| [`bayes`](deduction/bayes.md) | A `MonadicProgram` over the learnable log weights whose posterior NUTS or SVI can target. |
| [`sample`](deduction/sample.md) | Exact forward sampling of yields of a given length from the chart's distribution. |

::: quivers.stochastic.deduction
