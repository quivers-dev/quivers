# Continuous Programs

A `MonadicProgram` is a probabilistic program over continuous and discrete spaces: a sequence of
draws, observations, and deterministic computations with named intermediate values, in which
[later steps read earlier values](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram)
[@moggi1991notions; @giry1982categorical] ([building programs in Python](../../guides/programs.md#building-programs-in-python)).
Running a program records its latent draws and observation scores in an
[execution trace](../../guides/inference-foundations.md#trace-and-sample-sites), and the return
statement selects the values the caller receives. Inference uses these scores to fit or sample a
posterior.

::: quivers.continuous.programs
