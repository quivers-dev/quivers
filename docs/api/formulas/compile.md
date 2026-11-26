# `quivers.formulas.compile`

The `FormulaToQVRModule` lens turns a `Formula` into a `quivers.dsl.ast_nodes.Module`, expanding
formula terms into explicit QVR declarations and program steps
([inspect the generated QVR](../../guides/analysis-data-and-formulas.md#inspect-or-dump-the-generated-qvr)).
A group intercept becomes a set of group effects and a lookup of each observation's group. The
generated module shows exactly which priors, axes, and likelihood the formula produced.
Its optional `distributional` mapping takes parsed `Formula` values for auxiliary
parameters, giving each predictor its own coefficient namespace while sharing group
coding. The lens complement still reconstructs the primary location formula;
auxiliary formulas are retained in the configured lens. Offset and exposure
columns enter native predictor bindings without coefficient draws; their arrays
are retained by `FormulaData.source` and returned by `fixed_column_observations`.

A retained `Formula.modifiers.weights` column becomes an explicitly typed row
input and an observation-only `[weights=formula_weights]` option. The lens binds
its data without coefficient or prior sites and preserves the column through
`backward`. A decision modifier additionally supplies a typed binary row input;
the generated program visibly combines it with the raw response before scoring
the joint law. Modifier declarations describe inputs and do not add effects.

Censoring and truncation become typed row inputs and explicit observation
event metadata. Censoring selects the recorded event; truncation normalizes
the sampling law before that event is scored. Standard-error roles modify
the family scale, and `sigma=False` removes the residual-scale site.
The same role record supplies retained columns for fresh prediction; the
compiler does not create a second response-data schema.

`reparameterize` selects coordinates for all Gaussian group blocks,
including scalar, category-specific, and component-valued members of one
shared correlation. A correlated centered block emits one
`MultivariateNormal` draw; independent coefficients use their scalar or
indexed Normal laws. Noncentered blocks retain their standard Normal
innovation sites.
The lens's immutable `group_layouts` records the coefficient slices used
by posterior replay. `FormulaData.group_layouts` retains the matching tuple
for formula reconstruction, so reusing the lens for a second forward pass
does not alter an earlier module's complement.

Grouped cumulative thresholds use the same hierarchical log-slack law
under both choices. Flexible and sum-to-zero policies have one shape
coordinate per gap; equidistant profiles have one shared gap per group.
The one varying group location is reused or added once, including for
binary responses. The existing named prior catalog accepts
`cutpoint_log_spacing_mean` and `cutpoint_log_spacing_scale` overrides;
scale priors are conditioned on nonnegative values. See the [identified construction](../../semantics/programs.md#ordinal-probability-constructions).

The optional `predictor_graph` retains named nonlinear additive designs,
deterministic intermediates, and family-output expressions. The compiler
emits their dependency order through the existing QVR value AST and applies
the family link at each output once. Shared group blocks collect members
across these leaves through the same layout as ordinary distributional
regression; nonlinear compilation does not introduce a second group emitter.

::: quivers.formulas.compile

::: quivers.formulas.GroupMemberLayout

::: quivers.formulas.GroupBlockLayout

`group_covariances` retains labeled known level covariance independently of
coefficient correlations. Gaussian blocks use the existing MatrixNormal law
on explicit support coordinates; positive-semidefinite rank declarations keep
only genuine stochastic dimensions. Prediction shares the same layout and
conditions new levels through the retained covariance. See the
[supplied-covariance workflow](../../guides/analysis-data-and-formulas.md#supplied-covariance-between-grouping-levels).

::: quivers.formulas.GroupCovariance

::: quivers.formulas.GroupCovarianceBasis

## Typed prior plans

`user_priors` accepts an exact-name mapping, `PriorSpec`, `PriorSet`, or a
sequence of specs. The compiler resolves selectors against one immutable
`PriorTarget` catalogue built from the same design and group layouts as the
model. `prior_targets` and `FormulaData.prior_targets` expose that catalogue;
`prior_set` retains the authored specification. Ordinary priors use the existing
compositional distribution parser, including normalized restrictions.
Sampled horseshoe and R2D2 hierarchies add hyperprior sites while preserving
coefficient binders and conditional group geometry. See the
[prior workflow](../../guides/analysis-data-and-formulas.md#prior-selectors-and-shared-shrinkage).

::: quivers.formulas.PriorSpec

::: quivers.formulas.PriorSet

::: quivers.formulas.PriorTarget

::: quivers.formulas.RegularizedHorseshoe

::: quivers.formulas.R2D2

::: quivers.formulas.normalize_priors


::: quivers.formulas.PriorInput

::: quivers.formulas.PriorDistribution
