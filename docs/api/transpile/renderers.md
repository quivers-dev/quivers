# Renderers

Each [renderer](../../semantics/transpile-architecture.md#5-renderert-ir-panprotoschema) emits source
for one backend from the IR, translating draws, observations, and density terms into the target's
conventions. A translation is [checked](../../semantics/transpile-correctness/index.md#2-correctness-criterion)
by fixing the data and latent values and comparing the log densities of the source and target
models.

Stan places integer and Boolean deterministic bindings, together with dependent
bindings, inside its `model` block because transformed parameters must be continuous.
Returning such a value replays its pure dependencies in `generated quantities`;
unrelated continuous bindings remain transformed parameters. Top-level data,
parameter, and output shapes require data-only sizes; the renderer reports a shape
depending on a model-local binding before emitting source.

::: quivers.transpile.renderers
