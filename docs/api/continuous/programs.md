# Continuous Programs

A `MonadicProgram` is a probabilistic program over continuous and discrete spaces: a sequence of
draws, observations, and deterministic computations with named intermediate values, in which
[later steps read earlier values](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram)
[@moggi1991notions; @giry1982categorical] ([building programs in Python](../../guides/programs.md#building-programs-in-python)).
Running a program records its latent draws and observation scores in an
[execution trace](../../guides/inference-foundations.md#trace-and-sample-sites), and the return
statement selects the values the caller receives. Inference uses these scores to fit or sample a
posterior.

The `Observe.weights` field selects a finite nonnegative power of the observed
likelihood, with a scalar or one value per observation batch position. It
changes the score while retaining the raw response distribution for sampling
and pointwise likelihood extraction. The
[observation-weight reference](../../reference/qvr/probabilistic-programs.md#observation-weights)
explains validation, zero weights, and the reduction order.

::: quivers.continuous.programs
