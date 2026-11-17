# `quivers.formulas.formula`

[Parsing a formula](../../guides/analysis-data-and-formulas.md#formulas-quiversformulas) separates the
ordinary predictor columns (`FixedColumn`) from the group terms (`RandomTerm`), and the `Formula`
record carries the design information from which compilation builds parameter sites and indexed
lookups. `formula_from_data` builds it over `formulae.design_matrices`, so its
[design matrices](../../tutorials/python/08-analysis-pipelines.md#a-first-fit-gaussian-regression-in-one-line)
depend on the data and the category encoding as well as on the formula.

`Formula.design_terms` retains coupled Gaussian bases as `GaussianDesignTerm`
values. Their innovation coordinates and positive penalty scales are distinct
from ordinary fixed columns; `FormulaData.design_terms` preserves the same
immutable records across source-lens round trips. The
[structured predictor guide](../../guides/formula-special-predictors.md)
describes the mgcv `s()` and `t2()` constructors and retained new-data
transforms. `GaussianDesignTerm.scale_group` names a smoothing scale shared
by smooths with one mgcv `id`.
`GaussianDesignTerm.gp` holds a
[`GaussianProcessBlock`](#quivers.formulas.GaussianProcessBlock) for an
approximate `gp()` term: its coordinate scales are the square roots of the
kernel's spectral density at the retained frequencies, as functions of the
sampled `sdgp` and `lscale`. For an exact `gp()` term (no `k`) the block
retains the distinct scaled training inputs, and its coordinates are the
latent function there, `cholesky(K + 1e-12 I)` times standard-Normal
innovations, written with the QVR builtin `cholesky`. Predictions at new
inputs draw from
[`GaussianProcessConditional`](#quivers.formulas.GaussianProcessConditional),
brms's conditional Gaussian law given the training values.

::: quivers.formulas.GaussianProcessBlock

::: quivers.formulas.GaussianProcessConditional

`Formula`, `FormulaData`, and `FixedColumn` are immutable values that
compare by every field. Two formulas are equal only when their formula
strings, response names, fixed columns (labels and data), random
terms, offsets, response modifiers, response values, group levels, and group indices all agree, and
hashing agrees with equality. The arrays with one entry per row are stored as
read-only copies made at construction: writing into
`formula.response_values` or a column's `data` raises `ValueError`, and
changing the source data frame afterwards does not change the formula.
Copy a field (`formula.response_values.copy()`) to get a writable array.

`Formula.response_layout` retains response carrier and event shape, named
vector components, and scalar factor levels with their declared order.
`cbind(a, b, c)` contributes one vector event per row, whereas a factor
response contributes one integer code per row. `FormulaData` retains this
layout through the lens complement; neither serialization nor prediction
flattens the event into observation rows. `FormulaData.group_layouts` also
retains the matching immutable Gaussian coefficient layout. An interleaved
forward pass on another formula cannot change that complement's reconstruction.

`Formula.offsets` retains coefficient-free `offset(expr)` and `exposure(expr)`
columns as read-only `FixedColumn` values. Exposure contributes the logarithm of
positive finite values. The [offset guide](../../guides/analysis-data-and-formulas.md#offsets-and-observation-exposure)
shows how these terms affect fitting and posterior predictors.

`formula_from_data(..., weights="column")` retains a finite nonnegative
numeric column as `Formula.modifiers.weights`. `Formula.modifiers.decision`
retains a normalized binary decision column for a joint response model.
Both belong to `ResponseModifiers`; `FormulaData.modifiers` preserves them
in the lens complement. The
[likelihood-weight guide](../../guides/analysis-data-and-formulas.md#likelihood-weights)
explains the objective, row alignment, and prediction contract.

`ResponseModifiers` additionally retains `CensoringSpec`, `TruncationSpec`,
`StandardErrorSpec`, and `RowSelection`, together with trials, weight-scaling
policy, and the original column-or-constant role sources. Subsetting precedes
all design and role validation, and retained positions align auxiliary and
neural inputs. The [response-modifier guide](../../guides/analysis-data-and-formulas.md#response-modifiers-and-recorded-events)
states event endpoints, scale combination, supported families, and fresh-data
requirements. These records compare by value and retain read-only column data.
Exact observations must lie within their inclusive truncation bounds before
inference begins. Censoring endpoints may lie outside the sampling interval;
their recorded events are intersected with that interval when scored.

`cs(expr)` marks a transition-specific fixed or group column through
`FixedColumn.category_specific`. The formula lens retains those flags and
the authored fixed-column order in `FormulaData.category_specific_columns`
and `fixed_column_order`; serialization through the value IR does not
silently turn a category-specific coefficient into an ordinary scalar.
`cs(1)` uses an internally prepared ones column, with an explicit refusal
if the caller already uses its reserved name. The
[ordinal guide](../../guides/analysis-data-and-formulas.md#ordinal-families-thresholds-and-category-specific-effects)
states supported constructions and prediction shapes.

`Formula.monotonic_terms` retains each brms `mo()` special-effects column as a
`MonotonicTerm`: one `MonotonicFactor` per `mo()` factor, holding the training
matrix of cumulative category indicators and the brms `simo` label, the
covariate of the term's other components, and the brms coefficient label.
`FormulaData.monotonic_terms` preserves them across source-lens round trips.
The `mo()` formulae transform retains the training coding, so fresh rows are
coded and validated the same way. The
[structured predictor guide](../../guides/formula-special-predictors.md#monotonic-effects)
states the model, priors, and prediction rules.

::: quivers.formulas.formula

::: quivers.formulas._monotonic.MonotonicTerm

::: quivers.formulas._monotonic.MonotonicFactor

`Formula.grouping` retains brms's grouping-term syntax as a `GroupingPlan`.
Each `GroupingSpec` builds an interaction factor `g1:g2` or a
multi-membership factor `mm(g1, g2)` from its source columns, for training
and fresh rows alike. A `Membership` holds the member codes and weights of a
multi-membership factor, and a `MemberCovariate` the member values of one of
its coefficients. The [grouping guide](../../guides/formula-grouping.md)
states the syntax and the model.

::: quivers.formulas._grouping.GroupingPlan

::: quivers.formulas._grouping.GroupingSpec

::: quivers.formulas._grouping.Membership

::: quivers.formulas._grouping.MemberCovariate

::: quivers.formulas._grouping.GroupOptions

::: quivers.formulas._grouping.GroupLevelOptions
