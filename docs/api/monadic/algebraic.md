# Algebraic Effects

An algebraic-effect program records requests separately from their interpretation. A
handler supplies a result for each request and decides how computation continues. This
permits the same request structure to receive different interpretations. These Python
interfaces describe explicit operation trees; source-level lexical effect instances are
documented separately in the QVR reference.

See [operation trees and their handlers](../../guides/effects.md#algebraic-effects-handlers) and
[the mathematical interpretation of the interfaces](../../semantics/effects.md#2-typeclass-denotations).

For the semantics of monadic computation, see [@moggi1991notions; @wadler1992essence].

Plotkin-style effect signatures and handler-based interpretation via
`Operation`, `EffectSignature`, `Handler`, and `FreeMonad`.

::: quivers.monadic.algebraic
