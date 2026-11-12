# Parameter Sources

A [conditional family](../../tutorials/python/03-probabilistic-programs.md#conditional-distribution-families) needs a rule that turns its input into distribution parameters. For
a Normal response this means obtaining a location and a positive scale for each input. A
[parameter source](../../guides/continuous-families.md#hand-written-core-families) supplies that rule, which may be a table or a learned network; it is
separate from the random draw made using those parameters.

`quivers.continuous.param_source` maps a conditional family's input to
its distribution parameters. For a morphism declared `~ Family` over a
continuous domain, a `ParamSource` computes the parameters of the
corresponding Kleisli arrow. Nonlinear dependence on the input is thus
implemented by the parameter source.

`LinearSource` contains one `nn.Linear`. `MLPSource` is a multilayer
perceptron with configurable widths and activation, while
`AttentionSource` is a self-attention head. `LookupSource` and
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
