# Continuous Programs

A program records draws, observations, and deterministic computations with named
intermediate values. Later steps can use earlier values. Execution records latent draws
and observation scores in a trace; the return statement selects which values the caller
receives. Inference uses these scores to fit or sample a posterior.

See [constructing a program in Python](../../guides/programs.md#building-programs-in-python) and
[a complete program construction](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram).

For probabilistic composition through bind, see [@giry1982categorical]; for monadic sequencing more generally, see [@moggi1991notions].

Probabilistic programs in continuous domains.

::: quivers.continuous.programs
