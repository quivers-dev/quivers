# Handlers

[Conditioning fixes a site’s value](../../guides/inference-foundations.md#conditioning-on-observations) and includes its likelihood in the score. Intervention
replaces the generating mechanism, changing which density contributions belong in the
model. Replay supplies values from an [earlier trace](../../tutorials/python/05-variational-inference.md#tracing-the-model), and scaling multiplies score
contributions. [`subsample`](#quivers.effects.subsample.subsample) scores a
plate's sites at a subset of its rows, rescaled, for [minibatch
SVI](../../guides/inference-svi.md#minibatches-subsampled-plates). Choose the handler
according to the operation the model requires.

Conditioning, intervention, reweighting, subsampling, blocking, replay, and lifting.

::: quivers.effects.clamp

::: quivers.effects.do

::: quivers.effects.scale

::: quivers.effects.mask

::: quivers.effects.subsample

::: quivers.effects.block

::: quivers.effects.replay

::: quivers.effects.lift

