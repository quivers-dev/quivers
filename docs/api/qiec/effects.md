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

::: quivers.qiec.effects
