# Batched evaluation

[`PotentialFn.log_density_batch`](../inference/mcmc.md#quivers.inference.mcmc.PotentialFn)
scores a program at many positions in
[one pass](../../guides/inference-mcmc.md#densities-at-many-positions). Whether each step can be
evaluated this way is declared, by the
[step](../continuous/program_steps.md#quivers.continuous.program_steps.Score) for a Python
callable and by the
[morphism](../continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism) for a
distribution. [`batching_obstacles`](#quivers.effects.batching.batching_obstacles) lists the steps
of a program that declare nothing, or whose terms would differ between positions, and
[`UnbatchableProgramError`](#quivers.effects.batching.UnbatchableProgramError) carries them.

::: quivers.effects.batching
