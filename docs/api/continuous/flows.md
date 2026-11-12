# Flows

A [normalizing flow](https://en.wikipedia.org/wiki/Flow-based_generative_model) pushes a simple
distribution through a sequence of invertible maps, and scores a value by the base density times
the inverse maps' Jacobian determinants [@rezende2015flows]
([normalizing flows](../../guides/continuous-spaces.md#normalizing-flows)). `ConditionalFlow` is a
conditional flow built from `AffineCouplingLayer`s [@dinh2017realnvp].

::: quivers.continuous.flows
