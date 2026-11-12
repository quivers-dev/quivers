# Parameter Sources

A *parameter source* is the rule, a table or a network, that turns a
[conditional family](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families)'s
input into the family's parameters, such as a location and a positive scale for a Normal response
([core families](../../guides/continuous-families.md#hand-written-core-families)). For a morphism
declared `~ Family` over a continuous domain, a `ParamSource` computes the parameters of the
corresponding Markov kernel, separately from the draw made with them. Any nonlinear dependence on
the input thus lives in the parameter source.

`LinearSource` contains one `nn.Linear`. `MLPSource` is a multilayer
perceptron with configurable widths and activation, and
`AttentionSource` applies self-attention across a sequence and pools it to one parameter vector. `LookupSource` and
`EmbeddingSource` handle discrete domains. `IdentitySource`,
`FunctionSource`, and `ComposeSource` implement pass-through, a fixed
callable, and source composition, respectively.

Families call the `make_param_source` factory, and
`param_source_from_option` parses the DSL's `[param_source=<kind>]`
morphism option. The default for a continuous domain is `LinearSource`,
so a kernel is linear unless another source is requested. A `SetObject`
domain uses `LookupSource` regardless of the requested kind. The
[Bayesian Neural Network](../../examples/bnn.md) example selects the
MLP source explicitly and relies on it for its nonlinearity.

The hidden widths come either from the option's arguments or from
`hidden_dim`, one width per hidden layer:

<!-- compile: false -->
```qvr
morphism f : X -> Y [param_source=mlp] ~ Normal                      # (64, 64)
morphism f : X -> Y [param_source=mlp(64, 32)] ~ Normal              # (64, 32)
morphism f : X -> Y [param_source=mlp, hidden_dim=[64, 32]] ~ Normal # (64, 32)
morphism f : X -> Y [param_source=mlp, hidden_dim=64] ~ Normal       # (64,)
```

A width given to a source with no hidden layers to apply it to is an
error rather than a silent no-op, and so is `param_source` on a family
whose parameters do not come from a source at all (`Horseshoe`,
`GaussianProcess`, `Independent`, `Transformed`).

A [learned function](deterministic.md) builds its network through
[`function_source`][quivers.continuous.param_source.function_source], which
admits the `linear`, `mlp`, and `identity` kinds and threads the declared
hidden-layer activation into an `MLPSource`. It also admits two networks
over whole sequences, which a
[`SequenceFunction`][quivers.continuous.deterministic.SequenceFunction] holds:
[`SelfAttentionSource`][quivers.continuous.param_source.SelfAttentionSource],
multi-head scaled dot-product self-attention over the position axis,
causally masked or not [@vaswani2017attention], and
[`PositionSource`][quivers.continuous.param_source.PositionSource], a learned
table of position vectors added to a sequence's rows. Unlike
`AttentionSource`, which pools the sequence it attends over into one
parameter vector, both map a sequence of rows to a sequence of rows.

::: quivers.continuous.param_source
