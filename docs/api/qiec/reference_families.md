# Reference families

These samplers and densities implement the structured and compositional families of the reference
backend, beyond its scalar core, for the
[declared families](../../reference/qvr/probabilistic-programs.md#fixed-and-conditional-families).
A family that several backends implement may differ in its
[event shapes and supports](../../semantics/transpile-correctness/index.md#4-distribution-parameterization).

The count-mixture reference kernels include `ZeroInflatedNegativeBinomial`,
`ZeroInflatedBinomial`, `ZeroInflatedBetaBinomial`, and
`HurdleNegativeBinomial`. They preserve count support and distinguish a
structural-zero mixture from positive truncation; the
[formal count semantics](../../semantics/programs.md#zero-inflated-and-hurdle-count-kernels)
states their masses and endpoint conventions. All 11 external renderers
currently refuse these four count families with a `family:<Family>`
diagnostic, independently of native and reference support.

`GeneralizedPareto(loc, scale, concentration)` also has a reference
density and sampler. Its real tail index controls the finite or infinite
upper endpoint, with the exponential limit at zero; see the
[native tail-family guide](../../guides/continuous-families.md#generalized-pareto-tails).
Reference support does not establish an external target mapping.

`SkewNormal`, `ShiftedLogNormal`, `Frechet`, and
`GeneralizedExtremeValue` also have checked-reference densities and
samplers, independently checked against their native laws. Their
[formal parameter maps](../../semantics/programs.md#scalar-response-kernels-and-regression-parameters)
distinguish native location/scale parameters from regression means.
All 11 external renderers currently refuse these four families explicitly.

`DiscreteWeibull` and `ZeroInflatedAsymmetricLaplace` also have native
and checked-reference implementations, with nonnegative count and mixed
zero-atom/real supports, respectively. Their
[formal definitions](../../semantics/programs.md#discrete-weibull-counts)
distinguish a survival parameter from a response mean and an atom's mass
from a continuous density. Both are explicitly refused by all 11
external renderers. The formula-only `negbinomial2` parameterization
reuses the existing `NegativeBinomial` reference kernel.

::: quivers.qiec.reference_families
