# Deterministic Morphisms

A [deterministic kernel](https://en.wikipedia.org/wiki/Dirac_measure) sends each input to a Dirac
measure at a function's output, so a transformation such as softmax can take part in a
[probabilistic program](../../tutorials/python/03-probabilistic-programs.md#building-a-monadicprogram)
([let steps](../../guides/programs.md#let-steps)) without a further random draw. Its output is
still random when its input is. The module gives deterministic kernels for cumulative sums of
monotone splines, the softmax projection onto the simplex, and covariance reconstruction from a
correlation Cholesky factor and scales.

[`LearnedFunction`][quivers.continuous.deterministic.LearnedFunction] is the
learned case: a [parameter source](param_source.md) applied to the input's
coordinates, followed by an output nonlinearity. It is what a
`morphism f : X -> Y [role=function]` declaration compiles to, and a program's
`let z = f(a, b)` calls its
[`apply_values`][quivers.continuous.deterministic.LearnedFunction.apply_values]
with one value per domain factor. Its weights are ordinary module parameters,
fit as point estimates, unless it is constructed with `latent=<name>`: its
weights are then read from the sites
[`network_weight_sites`][quivers.continuous.deterministic.network_weight_sites]
names, passed to each evaluation, as a `[weights=latent]` declaration draws
them.

[`SequenceFunction`][quivers.continuous.deterministic.SequenceFunction] is the
learned case over whole sequences, what a `[role=function]` declaration with
`param_source=self_attention` or `param_source=positions(N)` compiles to. Its
input keeps a position axis before the coordinate axis, shape
`(..., T, d)`, and its network reads every position at once: a
[`SelfAttentionSource`][quivers.continuous.param_source.SelfAttentionSource]
mixes the positions, and a
[`PositionSource`][quivers.continuous.param_source.PositionSource] adds a
learned vector to each. A program's `let h = attn(u)` passes it one sequence.

::: quivers.continuous.deterministic
