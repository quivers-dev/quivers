# Unit likelihoods

`UnitLogLikelihood` holds the log likelihood of each independent
[unit](../../semantics/programs.md#271-units-and-row-scores) of a set of responses at each posterior
draw, with the unit that each response row belongs to.
[`ModelFit.unit_log_likelihood`](../modeling.md#quivers.modeling.model.ModelFit.unit_log_likelihood)
returns one for each set of responses whose rows share units.

::: quivers.inference.units
