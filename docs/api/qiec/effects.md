# Effects

An effect interface declares the operations a computation can request and the types of their
arguments and results [@plotkin2009handlers; @bauer2015programming]. An *effect instance* is one
named occurrence of an interface, so two state instances share an interface without sharing their
state ([instances](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler)). An
[effect row](../../reference/qvr/effects-and-handlers.md#effect-rows) lists the instances a
computation may request [@leijen2014koka]. The module gives typed effect interfaces, rows, and
handler signatures.

Discharge allocates an inlined call's address at its original entry.
Residual requests observe that address while the call is live. A
resumption can restore the same call address more than once, with each
shot keeping its own resumption path. Effects in the intervening clause
use the scope outside the captured call.

::: quivers.qiec.effects
