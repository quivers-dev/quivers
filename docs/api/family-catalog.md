# Custom distribution catalogs

`FamilyCatalog` extends the builtin catalog for one compilation. A custom
definition pairs the existing checked `DistributionFamily` contract with a
native distribution factory or `FamilySpec`. `with_family` returns a new
catalog; definitions and implementations already held by another model retain
their meanings. Constructor schemas follow the declared parameter names.

Pass `families=catalog` to `Model`, `Compiler`, or `lower_qvr_to_qiec`.
Factories receive named natural-scale tensors and return a native distribution;
the existing inference, response-law, raw likelihood, and prediction paths use
that law. Construction validates parameter carriers and constraints, dependent
event dimensions, batch geometry, discreteness, and any declared
reparameterization. These checks cannot prove that a user density integrates
to one or that its sampler implements that density.

Factories are never sampled or called to discover their support during source
checking. A factory with non-real support therefore declares an explicit native
`support` constraint; `FamilySpec` already retains one. The runtime binds the
semantic record to a shallow copy of the returned distribution, retaining its
parameter tensors and autograd graph. This gives the existing measure and
atom machinery with the law's declared reference-measure contract without
mutating a factory-owned object or registering its class globally.

Direct catalog construction preserves integer tensors. The existing packed
source execution path accepts signed integer parameters within `[-2**53, 2**53]`
and refuses larger magnitudes before mixing them with real-valued features.
Boolean parameters retain Boolean tensors; finite trial counts retain their
separate nonnegative bound. Declaring an integer parameter does not turn it
into a trial count.

Catalog copies and refit snapshots retain catalog identity. Pickling executable
factories is refused. QIEC serialization retains only custom semantic records;
execution requires explicitly provided native implementations. Native factories
have no implicit pure-reference or external-target implementation, and those
execution paths report a capability error instead of emitting an undefined
family. Recorded-event censoring and restriction additionally require the law's
existing interval/atom capabilities.

::: quivers.family_catalog
