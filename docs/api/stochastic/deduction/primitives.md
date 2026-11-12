# Deduction Primitives

These abstract classes split a [deductive computation](../../../semantics/grammar.md#6-chart-denotation)
into its initial facts (`Axiom`), rule applications (`Deduction`), target conclusions (`Goal`), and
processing order (`Schedule`), so a schedule or weighting can change without rewriting the task
[@shieber1995principles]. `DeductiveSystem` is the protocol that ties them together. The
agenda-based `DeductionSystem` in [`quivers.stochastic.agenda`](../agenda.md) is the implementation
the DSL compiler uses, and these symbols also serve custom deduction classes and the
[inside algorithm](../inside.md).

::: quivers.stochastic.deduction.primitives
