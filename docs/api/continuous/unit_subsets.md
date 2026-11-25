# Subsampled units

A score that is a sum over [units](../../semantics/programs.md#271-units-and-row-scores) is estimated
without bias by a subset of them
([subsampled units](../../semantics/programs.md#274-subsampled-units)). `unit_coefficients` reads the
coefficient of each unit from the subsampled plates of a run, and refuses a subset that splits a unit.

::: quivers.continuous.unit_subsets
