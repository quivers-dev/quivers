# Continuous Programs

A [program records draws, observations, and deterministic computations](../../guides/programs.md#building-programs-in-python) with named
intermediate values. [Later steps can use earlier values](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram) [@giry1982categorical; @moggi1991notions]. Execution records latent draws
and observation scores in a trace; the return statement selects which values the caller
receives. Inference uses these scores to fit or sample a posterior.

Probabilistic programs in continuous domains.

::: quivers.continuous.programs
