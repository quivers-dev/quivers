# Effects

An effect interface declares the operations a computation can request and the types of their
arguments and results [@plotkin2009handlers; @bauer2015programming]. An *effect instance* is one
named occurrence of an interface, so two state instances share an interface without sharing their
state ([instances](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler)). An
[effect row](../../reference/qvr/effects-and-handlers.md#effect-rows) lists the instances a
computation may request [@leijen2014koka]. The module gives typed effect interfaces, rows, and
handler signatures.

Discharge allocates the addresses of inlined calls and local instances
at their original entry points. A residual request observes the live
addresses in their original order. Resumptions restore the same
allocations for each shot, which has its own resumption path. Effects in
an intervening clause use the scope outside the captured continuation.

A retained call also inherits its enclosing discharged shots and
clause activations. Discharge keeps the reference handler when the call's
residual row is open or could invoke an outer multi-shot clause inside
that path.

An effectful return clause uses the address of its handler installation.
Different installations of the same declaration have different return
addresses. Multiple shots through one installation share its return
address, with their own shot paths. If an outer handler captures the
return, its operation-clause address includes the captured activation
alongside any captured local instance addresses.

::: quivers.qiec.effects

## Finite laws

The finite law checker compares two named QIEC computations with the same
positional input types and result type. It runs both sides for every assignment
of Booleans, unit values, and their finite products. Each side has an empty
residual effect row, and reachable handler clauses use the same exact finite
fragment.

`finite_continuation_type` gives a total continuation one result cell for every
tuple of arguments. Include handler state among those arguments when the
continuation can observe it. Both sides receive the same table. For a continuation
from a pair of Booleans to a Boolean, the four cells are indexed by
`(false, false)`, `(false, true)`, `(true, false)`, and `(true, true)`, in that
order.

The checker returns the first counterexample with its full assignment and both
results. A positive decision follows only after every assignment completes.
An unsupported term or an exhausted limit raises an error.

::: quivers.qiec.laws
