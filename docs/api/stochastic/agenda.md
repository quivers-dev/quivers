# Agenda-based Deduction

An agenda is the work list of chart updates still to process. The system adds axioms,
applies rules to available facts, and accumulates weights at conclusions. Different
schedules can change efficiency; equality of final results requires the relevant
algebraic and convergence assumptions, especially when rules form cycles.

See [when scheduling preserves the answer](../../guides/deduction.md#strategy-independence) and
[comparing inside scores for a parser](../../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores).

For weighted parsing as deduction, see [@shieber1995principles; @goodman1999semiring].

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

For weighted deduction, see [@shieber1995principles; @goodman1999semiring]. A* parsing requires additional heuristic and search conditions [@klein2003astar]; program transformations provide another route to efficient evaluation [@eisner2007transformations].

Sources for the constructions below: [@pereira1983parsing; @klein2001parsing; @knuth1977generalization; @nederhof2003weighted; @eisner2005compiling; @mcallester2002complexity].

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
