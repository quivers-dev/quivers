# Overview

Handlers let one program run with [different interpretations](../../guides/effects.md#algebraic-effects-handlers) of its probabilistic
requests [@scibior2018functional; @nguyen2023effect]. A sample can be drawn, fixed to an observation, or replayed from a trace; a
score can be scaled or masked. The [ordering and scope of handlers](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler) matter because each
can change what the next one receives.

Effect handlers for probabilistic programs, executed as lexical kernel handlers.

::: quivers.effects
