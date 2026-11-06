# Implementation Correspondence and Limits

For each supported QVR phrase, the compiler should construct the runtime object described by its semantic clause. This page connects those clauses to the operations that implement them and the tests that check them.

## 1. What the compiler returns

`Compiler` resolves a parsed module into an environment of objects, morphisms, programs, deductions, structural components, and exports. Evaluation returns an artifact-specific value: a tensor for a discrete morphism, a sample for a sampler, or a dictionary of returned values for a program.

Thus correspondence must be stated per artifact:

| Phrase | Runtime object | Evidence-bearing operation |
|---|---|---|
| `object` | `SetObject` or `ContinuousSpace` | structural equality, shape, membership where implemented |
| discrete `morphism` | `Morphism` | `.tensor` |
| family morphism | `ContinuousMorphism` | `.rsample`, `.log_prob` |
| `program` | `MonadicProgram` or compiled program wrapper | sampling and `.log_joint` |
| `deduction` | `DeductionSystem` | agenda/chart evaluation |
| structural declaration | encoder, decoder, signature, or loss object | component-specific method |

## 2. Direct correspondences

Several clauses are close enough to their implementations to check pointwise.

First, object resolution is a structural walk over `ObjectExpr`. Nested products and coproducts are flattened without reordering components. Continuous constructors dispatch through the compiler's constructor table.

Second, a discrete latent morphism stores a raw parameter and exposes `sigmoid(raw)` as its tensor. An observed morphism exposes its supplied tensor. Composition calls the active composition rule's `tensor_op` and `join`; tensor product applies `tensor_op` pointwise.

Third, the expression combinators have distinct operational contracts:

| Combinator | Current implementation |
|---|---|
| `.marginalize(X, ...)` | reduces named **codomain** axes with the active algebra's `join` |
| `fan(f, g, ...)` | feeds one input to each component and concatenates outputs |
| `repeat(f, n)` | sequentially composes the same instance, sharing parameters |
| `stack(f, n)` | sequentially composes deep copies, giving each layer independent parameters |
| `scan(cell)` | iterates a continuous cell `A * H -> H` over an implicit time axis and returns the final state |

With `stack(f, 3)`, the output of the first copy enters the second, then the third. With `scan(cell)`, the hidden state passes between successive time steps.

## 3. Probabilistic composition

The primer first performs this elimination as a [finite matrix product](../getting-started/concepts.md#2-composition-eliminates-an-intermediate-value). The [continuous composition equations](morphisms.md#3-continuous-morphisms) replace that sum with an integral, and the [Python guide](../guides/continuous-spaces.md#composition-sampledcomposition) exercises the estimator discussed here.

Suppose $f$ chooses an intermediate state and $g$ draws an output conditional
on that state. Generating one output is straightforward: draw the state,
then draw the output. Computing the density at a fixed output is a different
task, because every possible intermediate state contributes. With a finite
two-state kernel this is $p(y)=p(0)p(y\mid0)+p(1)p(y\mid1)$; a continuous
intermediate state requires an integral. The sampler visits one intermediate state per draw; density evaluation
accumulates contributions across the intermediate space.

Sampling from `f >> g` is ancestral: sample the intermediate value with `f.rsample`, then pass it to `g.rsample`. This produces samples from the composite kernel when the component samplers implement their declared kernels.

Density evaluation is different. `SampledComposition.log_prob` draws a fixed number of intermediate samples, evaluates the second density, and combines the values with `logsumexp - log(n)`. The density-scale average is a Monte Carlo estimator of the integral; taking its logarithm introduces finite-sample bias. The default sample count is 100. Increasing the sample count reduces Monte Carlo error; analytic compositions provide a reference for checking the estimator.

`ScanMorphism.log_prob(x, y)` constructs the canonical reference trajectory
ending at `y` and returns `log_joint` for that fixed trajectory. The marginal
density of the final state would require summing or integrating these joint
scores across all intermediate trajectories. A cell without a
conditional density contributes zero instead.

## 4. Programs and data

Program statements preserve their leading-keyword distinction:

- `sample` introduces a random value;
- `observe` adds a family log density at supplied data;
- `let` binds deterministic tensor arithmetic;
- `score` adds an explicitly computed scalar to the log joint;
- finite-support `marginalize` enumerates a scoped latent and removes it from
  the outer scope; a non-enumerable family currently follows the ordinary
  single-sample path rather than computing a continuous integral;
- `return` selects the program result.

`from_data("key")` retrieves data bound to the compiler before expression compilation. Bind the key before compiling the expression that uses it.

## 5. What the tests establish

The tests cover four parts of the implementation:

- compiler and DSL tests exercise parsing, resolution, typing errors, expression composition, contractions, programs, and structural declarations;
- `tests/test_program_theory.py` checks extraction and validation of program-shape schemas;
- `tests/test_model_roundtrips.py` checks JSON round trips for `dx.Model` values;
- `tests/transpile/test_structural.py`, `test_lens_laws.py`, `test_external_syntax.py`, and `test_numeric_equivalence.py` test the separate transpilation pipeline described in [Transpilation correctness](transpile-correctness/index.md).

## 6. Conditions and open gaps

Five conditions determine how these correspondences can be used.

1. Categorical equations that use distributivity, arbitrary joins, or compact closure require those laws of the active algebra. The counterexamples in [Algebras](algebras.md#21-a-note-on-the-product-fuzzy-and-ukasiewicz-pairs) show why these hypotheses are needed.
2. Floating-point operations may differ from exact real arithmetic, especially near support boundaries and saturated reductions.
3. Monte Carlo density calculations are approximate at finite sample counts.
4. External family comparisons require matching support, parameter maps, and normalizing constants; the [target pages](transpile-correctness/index.md) record these maps.
5. Schema extraction records program shape. Two models with the same extracted schema can carry different learned weights and assign different probabilities.

Useful extensions are property tests for each algebraic law, comparisons of sampled compositions with analytic Gaussian cases, and compiler tests for each declarative typing rule.
