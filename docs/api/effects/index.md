# `quivers.effects`

Effect handlers run one probabilistic program under different
[interpretations](../../guides/effects.md#algebraic-effects-handlers) of its requests
[@plotkin2009handlers; @scibior2018functional; @nguyen2023effect]: a draw can be sampled, fixed to
an observation, or replayed from a trace, and a score can be scaled or masked. Each handler is a
lexically scoped handler of the `random`, `score`, or `param` instance of the program's
[Quivers Indexed Effect Core](../qiec/index.md) (QIEC) computation, so the order and scope of
[handlers](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler) matter: each
can change what the next one receives.

::: quivers.effects
