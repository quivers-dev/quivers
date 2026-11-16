# Conditional effects

[`prediction_grid`](#quivers.diagnostics.conditional.prediction_grid) specifies
effect coordinates and explicit conditioning or empirical-reference rows.
`BayesianFit.conditional_effects` evaluates that grid through one existing
posterior prediction request. `PredictionGrid.result` also accepts retained
predictions from another checked prediction procedure, with explicit
chain/draw/row axes and provenance.

`ConditionalEffects.draws` retains joint point dependence. Conditional grids
have `(chain, draw, point, condition, *event)` axes; empirical grids reduce the
reference rows within each draw and have `(chain, draw, point, *event)` axes.
`summary` and `analysis` delegate to the common posterior diagnostics. Neither
creates convergence statistics for variational or independent prior draws.
The [guide](../../guides/conditional-effects.md) explains conditioning,
empirical weights, grouping policies, and allocation limits.

::: quivers.diagnostics.conditional
