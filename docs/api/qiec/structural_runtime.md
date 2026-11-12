# Structural runtime

This module runs checked [encoder, decoder, and structural loss computations](../../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders)
through registered Python implementations
([differentiating through a run](../../tutorials/qvr/12-structural-autoencoders.md#4-differentiate-through-the-entry-boundary)).
A run needs a provider for each requested operation, and a transpile target must support those
operations to run the translated computation.

::: quivers.qiec.structural_runtime
