# Loss registry

A [loss attachment](../../guides/structural-compression-decoders-and-losses.md#loss-declarations)
attaches a weighted scalar training objective to a declared component, such as the
[reconstruction loss](../../tutorials/qvr/12-structural-autoencoders.md#3-invoke-the-checked-loss),
which penalizes a low probability of the original tree. Each compiled module has a `LossRegistry`
that maps attachment sites to weighted loss functions. The sites are program, deduction, encoder,
decoder, and rule names, and the chart and global sites.

The training driver calls `LossRegistry.evaluate(env)` to sum the
registered losses. `evaluate_on(kind, target, env, rule_deduction)`
returns the weighted partial sum for one attachment site.

The QVR compiler compiles loss bodies as let-expression closures with
signature `(env) -> Tensor`. For a global loss, `env` contains the
compiled module's program, deduction, encoder, and decoder bindings as
top-level names. For a rule-attached loss, it also contains `"rule"`,
`"deduction"`, `"antecedents"`, `"conclusion"`, and `"weight"`
keys populated by the agenda's rule-firing callback.

::: quivers.structural.losses
    options:
      filters: ["!^_", "!^(LossBody|LossWeight|TrainEnv|AttachmentKind)$"]

## Type aliases

::: quivers.structural
    options:
      members:
        - LossBody
        - LossWeight
        - TrainEnv
        - AttachmentKind
      show_root_heading: false
      show_root_toc_entry: false
