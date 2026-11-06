# Deduction Library

These ready-made systems instantiate weighted deduction for familiar tasks such as
recognition or shortest paths. They share the pattern of initial facts, rules, and
accumulated answers. Choose the system and semiring together: a reachability answer, a
path cost, and a log weight are different outputs.

See [the available deduction systems](../../guides/deduction.md#pre-registered-stdlib) and
[comparing inside scores for a parser](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores).

Pre-registered `DeductionSystem` instances for the standard
specializations of agenda-driven weighted deduction: categorial
grammar recognition, type checking, Datalog closure, shortest paths,
hidden Markov models, and edit distance. A DSL ``parse(NAME)``
expression resolves `NAME` against `STDLIB_DEDUCTIONS` before any
user-declared deduction.

::: quivers.stochastic.stdlib
