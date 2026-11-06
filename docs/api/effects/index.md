# Overview

Handlers let one program run with different interpretations of its probabilistic
requests. A sample can be drawn, fixed to an observation, or replayed from a trace; a
score can be scaled or masked. The ordering and scope of handlers matter because each
can change what the next one receives.

See [algebraic effects and handlers](../../guides/effects.md#algebraic-effects-handlers) and [a QVR handler example](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler).

For effects as a way to organize inference, see [@scibior2018functional; @nguyen2023effect].

Effect handlers for probabilistic programs, executed as lexical kernel handlers.

::: quivers.effects
