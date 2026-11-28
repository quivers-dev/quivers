# Structural runtime

This module runs checked [encoder, decoder, and structural loss computations](../../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders)
through registered Python implementations
([differentiating through a run](../../tutorials/qvr/12-structural-autoencoders.md#4-differentiate-through-the-entry-boundary)).
A run needs a provider for each requested operation, and a transpile target must support those
operations to run the translated computation.

[`FiniteDecoderSupport`][quivers.qiec.structural_runtime.FiniteDecoderSupport]
checks a supplied set of closed structural terms against one object sort of
the decoder's shared rule presentation. Its `distribution(code)` produces
a finite law on those terms in the presentation's nominal QIEC family.

::: quivers.qiec.structural_runtime
