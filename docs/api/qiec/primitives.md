# Primitives

The [pure primitives](../../reference/qvr/computations.md#built-in-values-and-distributions) are the
core's basic operations on values, held in a closed registry of names, types, and implementations.
A new operation needs a signature the [checker accepts](../../semantics/typing.md#6-computations-and-rows)
and an implementation on each runtime that runs it.
`PrimitiveSignature.total` additionally certifies that every well-typed argument
satisfies the operation's domain. This permits eager selection between proved
total branches; a partial operation in either branch requires lazy evaluation.

## Checked probability operations

Four scalar `Real -> Real` operations carry the `probability` capability.
`log_probability(p)` accepts finite `0 <= p <= 1`, giving `-inf` at
zero. `log1mexp(x)` accepts `x <= 0`, including `-inf`, and gives
`-inf` at zero. `log_ndtr(x)` gives a stable Normal log CDF and accepts
both infinities. `positive_parameter(x)` is the identity on finite
strictly positive reals. All reject Booleans and NaNs; the other endpoint
restrictions are checked at execution. Finite derivatives at singular
logarithmic endpoints are not promised.

Two checked observation operations retain their input roles.
`observation_status : Int -> Int` accepts the codes `-1`, `0`, `1`, and `2`.
`interval_upper : (Real, Real) -> Real` returns the second endpoint when it is
finite and strictly greater than the first, which may be infinite. Both reject
Boolean carriers. Lifted endpoints broadcast to their common array shape; a
scalar upper endpoint thus contributes once per queried row, with its gradient
summed over those rows.

The native and reference implementations preserve these domains. Dynamic
external runtimes supply checked helpers; ordinary Stan calls use checked
functions, while BUGS/JAGS refuse them. Retained canonical computations
require a target with the probability capability. A partial operation in
an inactive conditional branch must stay inactive; unsupported symbolic
control flow raises an explicit error. Registration does not establish
that every target can differentiate or fit such a retained computation.

::: quivers.qiec.primitives
