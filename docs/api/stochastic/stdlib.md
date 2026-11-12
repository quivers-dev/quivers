# Deduction Library

`STDLIB_DEDUCTIONS` holds ready-made `DeductionSystem` instances for standard
[weighted deduction](../../guides/deduction.md#pre-registered-stdlib) tasks: categorial grammar
recognition, type checking, [Datalog](https://en.wikipedia.org/wiki/Datalog) closure, shortest
paths, hidden Markov models, and edit distance. A DSL `parse(NAME)` expression looks `NAME` up
there before any deduction the module declares. Each system fixes its semiring, so reachability, a
path cost, and a log weight are different outputs.

::: quivers.stochastic.stdlib
