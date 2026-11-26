# Score units

A score step may declare that its contribution is a sum of one log weight per
[unit](../../semantics/programs.md#271-units-and-row-scores). A
[trace](trace_types.md) of the program then holds a `SiteUnits` at the score's site: the log
weight of each unit, the unit that each row of a response belongs to, and the log likelihood of
each row where the rows factor. `UnfactoredObservationError` says why a response has no log
likelihood per row and which program has one.

::: quivers.effects.units
