# Handlers

Conditioning fixes a site’s value and includes its likelihood in the score. Intervention
replaces the generating mechanism, changing which density contributions belong in the
model. Replay supplies values from an earlier trace, and scaling multiplies score
contributions. Choose the handler according to the operation the model requires.

See [conditioning a program on observations](../../guides/inference-foundations.md#conditioning-on-observations) and
[inspecting a concrete trace](../../tutorials/python/05-variational-inference.md#tracing-the-model).

For algebraic effects and handler interpretation, see [@plotkin2009handlers; @bauer2015programming].

Conditioning, intervention, reweighting, blocking, replay, and lifting.

::: quivers.effects.clamp

::: quivers.effects.do

::: quivers.effects.scale

::: quivers.effects.mask

::: quivers.effects.block

::: quivers.effects.replay

::: quivers.effects.lift

