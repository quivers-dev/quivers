# Handlers

These handlers cover conditioning, intervention, reweighting, subsampling, blocking, replay, and
lifting. [Conditioning](../../guides/inference-foundations.md#conditioning-on-observations) fixes a
site's value and adds its likelihood to the score. Intervention (`do`) replaces the mechanism that
generates a site, which changes which density terms belong to the model. Replay takes values from
an [earlier trace](../../tutorials/python/05-variational-inference.md#tracing-the-model), scaling
multiplies score terms, and [`subsample`](#quivers.effects.subsample.subsample) scores a plate's
sites at a subset of its rows, rescaled, for
[minibatch SVI](../../guides/inference-svi.md#minibatches-subsampled-plates).

::: quivers.effects.clamp

::: quivers.effects.do

::: quivers.effects.scale

::: quivers.effects.mask

::: quivers.effects.subsample

::: quivers.effects.block

::: quivers.effects.replay

::: quivers.effects.lift

