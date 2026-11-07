# Deduction Primitives

Separate a [deductive computation](../../../semantics/grammar.md#6-chart-denotation) into its initial facts, rule applications, target
conclusions, and processing order. This makes it possible to change a schedule or
weighting scheme without rewriting the task itself [@shieber1995principles]. The protocol defines the interface;
the concrete agenda system supplies an execution strategy.

Abstract building blocks for chart-based weighted deduction:
`Axiom`, `Deduction`, `Goal`, `Schedule`, and the
`DeductiveSystem` protocol. The concrete agenda-based
`DeductionSystem` (in `quivers.stochastic.agenda`) is the
implementation used by the DSL compiler. The symbols in this module
also support custom-deduction subclasses and the inside-algorithm
framework in [`quivers.stochastic.inside`](../inside.md).

::: quivers.stochastic.deduction.primitives
