# Agenda-based Deduction

An agenda is the work list of chart updates still to process. The system adds axioms,
applies rules to available facts, and accumulates weights at conclusions [@shieber1995principles; @goodman1999semiring]. Different
schedules can change efficiency; [equality of final results](../../guides/deduction.md#strategy-independence) requires the relevant
algebraic and convergence assumptions, especially when rules form cycles.

The concrete chart implementation used for DSL-compiled weighted
deductions. `DeductionSystem` combines an axiom injector, a rule
system, a semiring, and an agenda schedule in an `nn.Module`.
`DeductionSystem.__call__` returns the differentiable `ChartView`
presheaf.

For fitting, sampling, and the NUTS wrapper, see
[`api/stochastic/deduction`](deduction.md).

## Item algebra

Items, patterns, and bindings are plain tuples and dictionaries; these
aliases name their roles in the signatures below.

A* parsing uses a heuristic to prioritize unfinished analyses
[@klein2003astar]. Its correctness depends on the heuristic and search
conditions. Program transformations can instead reduce the work required
to evaluate a deduction system [@eisner2007transformations].

Viewing parsing as deduction separates inference rules from the algorithm
that schedules them [@pereira1983parsing; @klein2001parsing]. This supports
analyses of weighted derivations [@knuth1977generalization; @nederhof2003weighted]
and of the time needed to evaluate them [@mcallester2002complexity]. Compiling
the deduction rules into differentiable computations also connects parsing
to parameter estimation [@eisner2005compiling].

::: quivers.stochastic
    options:
      members:
        - Item
        - Pattern
        - Bindings
      show_root_heading: false
      show_root_toc_entry: false

::: quivers.stochastic.agenda
    options:
      filters: ["!^_", "!^(Item|Pattern|Bindings)$"]
