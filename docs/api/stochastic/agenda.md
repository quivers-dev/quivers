# Agenda-based Deduction

`DeductionSystem` is the agenda-based [weighted deduction](../../guides/deduction.md#the-framework)
system that the DSL compiles a `deduction` to [@shieber1995principles; @goodman1999semiring]. It
combines axioms, a rule system, a semiring, and an agenda schedule in a validated model. The
[agenda](https://en.wikipedia.org/wiki/Chart_parser) is the work list of items still to process:
the system adds the axioms, applies rules to the items proved so far, and accumulates weights at
their conclusions [@eisner2005compiling]. Calling the system returns a `ChartView`, a
differentiable view of the chart, the weight of each proved item. Different schedules change the
cost, and [equality of final results](../../guides/deduction.md#strategy-independence) requires the
semiring laws and, when rules form cycles, convergence of the fixed-point iteration. Fitting,
sampling, and the No-U-Turn Sampler (NUTS) wrapper are in the [deduction package](deduction.md).

Construct records with keyword arguments and use `with_` to update their fields. Updating a
deduction retains its live parameter modules and attached loss registries. The `modules`
mapping holds learned parameters under their parameter-name prefixes; `item_signature` and
`item_encoder` describe structured items. An agenda result retains its chart and differentiable
weights, and `with_` attaches an encoder or loss without mutating the result record.

## Items, patterns, and bindings

Items, patterns, and bindings are plain tuples and dictionaries, and these
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
