# Measure Algebra

These combinators build measures from measures: `Mixture`, `Restrict`, `Pushforward`, `PointMass`,
`Independent`, and `Normalize`, which back the operator families of the DSL. Restricting a
probability measure to an event of probability 0.2 leaves total mass 0.2, and dividing by 0.2 gives
the conditional distribution [@bernardo1994bayesian]. Each combinator states whether it keeps that
mass or [normalizes it](../../semantics/programs.md#5-algebraic-laws-and-their-scope), and
[`normalize_at_boundary`](#quivers.continuous.measure.normalize_at_boundary) renormalizes a
subprobability measure where it is sampled or scored.

## Raw masses and interval endpoints

`log_interval_mass(low, high, lower_closed=..., upper_closed=...)` integrates
the retained measure over the requested interval. Both endpoints are included
by default; either flag may be false, including for discrete or mixed laws.
An interval with equal endpoints contains only its atom when both flags are
true. `None` leaves that side unbounded.

The four log-tail methods use this same operation:

| Method | Event |
| --- | --- |
| `log_cdf(x)` | $(-\infty,x]$ |
| `log_cdf_left(x)` | $(-\infty,x)$ |
| `log_survival_function(x)` | $(x,\infty)$ |
| `log_survival_function_left(x)` | $[x,\infty)$ |

These methods return **raw mass**. Thus `cdf(x)` can exceed one for an
unnormalized measure; `Normalize(base).cdf(x)` gives its probability CDF.
`Restrict` retains intersection mass, and `Normalize` divides by the retained
total mass once. A mixture's sampling probability for component $j$ is
proportional to its raw weight times its total mass, so a restricted component
is selected according to its remaining mass.

```python
import torch
from quivers.continuous.measure import Mixture, Normalize, PointMass, Restrict

half = Restrict(torch.distributions.Uniform(0.0, 1.0), 0.0, 0.5)
raw = Mixture(torch.tensor([2.0, 3.0]), [half, PointMass(torch.tensor(0.0))])
probability = Normalize(raw)
assert abs(raw.log_normalizer().exp().item() - 4.0) < 1e-6
assert abs(probability.cdf(torch.tensor(0.0)).item() - 0.75) < 1e-6
open_lower = probability.log_interval_mass(0.0, 0.25, lower_closed=False).exp()
assert abs(open_lower.item() - 0.125) < 1e-6
```

## Atoms and density references

A scalar mixture combines component densities against one common reference.
At any retained atom, `log_prob` includes atomic masses alone; an atomless
component's continuous density does not add probability to that point.
Away from the atom carrier, continuous component densities contribute as usual.
Integer-valued scalar laws use counting measure on the integers, including
integer locations where a particular component assigns zero mass.

`atom_reference(value)` identifies the retained atomic carrier, while
`log_atom_mass(value)` reads its raw mass. Restriction can remove an atom's
mass without changing the base carrier, and normalization rescales the mass.
Coupled mixed vector laws need a joint stratum reference; composition that
lacks this information fails explicitly.

Censoring and truncation use these interval masses for different purposes.
An interval-censored observation contributes the event's raw mass. A truncated
response instead uses the normalized restriction as its whole response law.
The `Censored` distribution is a third operation: it clamps sampled values to
its endpoints and retains the resulting endpoint atoms.

Restriction masks excluded values before invoking a base density. This retains
zero mass outside either the interval or a parameter-dependent base support,
without evaluating an invalid logarithm in an inactive branch. Independent event
coordinates reduce the mask to their event's batch axes. Conditional
`families.Truncated` reuses `Normalize(Restrict(...))` for its coordinate laws;
both interfaces share the rejection loop while retaining their sampling policy
and attempt budget. Normal coordinates retain their specialized inverse sampler.

## Batch selection and transformed event shapes

`Measure.select_batch(active, output_shape=...)` selects active batch cells
while retaining each event intact. Mixture components, restrictions, and
normalizers delegate through this operation; a vector response's component
axis is not another batch axis.

A pushforward obtains its output event shape from the bijector's
`forward_shape`; inversion uses `inverse_shape`. Its `jacobian_event_dim`
states which trailing event coordinates the Jacobian already reduces.
Composition reduces any remaining event coordinates exactly once. Thus a
shape-changing stick-breaking map retains its simplex event axis, and a
composed vector map does not duplicate or omit the change-of-variables term.
Parameterized bijectors implement `select_batch` alongside the measure so
masked evaluation uses the corresponding parameters.

::: quivers.continuous.measure
