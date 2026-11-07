# Deduction Systems

[Weighted deduction](../../guides/deduction.md#the-framework) builds conclusions from rules and scored premises [@shieber1995principles; @goodman1999semiring]. Fitting adjusts
rule weights, Bayesian lifting introduces priors over them, and sampling draws from a
normalized distribution over candidate results. The modules below implement these
operations for shared deductive structures.

The public API for weighted chart deductions consists of the model
type, its abstract primitives, and operations for fitting, Bayesian
lifting, and sampling. `quivers.stochastic.deduction` re-exports the
public symbols.

| Submodule | Job |
|---|---|
| [`primitives`](deduction/primitives.md) | Abstract building blocks (`Axiom`, `Deduction`, `Goal`, `Schedule`, `DeductiveSystem`). Most users do not touch these directly. |
| [`fit`](deduction/fit.md) | Point-estimate gradient fitting (MAP / MLE) of the deduction's learnable log-weights. |
| [`bayes`](deduction/bayes.md) | Lift the parameters into a Bayesian `MonadicProgram` whose posterior NUTS / SVI can target. |
| [`sample`](deduction/sample.md) | Exact length-conditional forward sampling of yields from the chart's distribution. |

::: quivers.stochastic.deduction
