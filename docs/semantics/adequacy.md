# Implementation correspondence and limits

For each phrase, the compiler builds a runtime object meant to realize the phrase's denotation. The
tables below pair each denotation with the operation that computes it, and say where the runtime
computes it exactly, where it approximates it, and where it computes something else. In programming
language semantics, *adequacy* is the different statement that an operational semantics agrees with a
denotational one [@plotkin1977lcf], and it is not established here.

## 1. What the compiler returns

`Compiler` resolves a parsed module into an environment of objects, morphisms, programs, deductions,
structural components, and exports. What a phrase computes depends on its runtime object:

| Phrase | Runtime object | Operation that realizes the denotation |
|---|---|---|
| `object` | `SetObject` or `ContinuousSpace` | structural equality, shape, and membership where implemented |
| finite `morphism` | `Morphism` | `.tensor`, the matrix |
| family `morphism` | `ContinuousMorphism` | `.rsample` to draw, `.log_prob` to score |
| `program` | `MonadicProgram` | drawing, and `.log_joint`, the log joint density |
| `deduction` | `DeductionSystem` | evaluation of the chart by the agenda |
| `signature`, `encoder`, `decoder`, `loss` | the component's class | the component's methods |

## 2. Direct correspondences {#2-direct-correspondences}

For these clauses the runtime computes the denotation, up to floating-point error.

- An object expression is resolved by one recursive walk over `ObjectExpr`, which flattens nested
  products and coproducts without reordering them ([Types and spaces](types-and-spaces.md)). Space
  constructors are looked up in the compiler's table of constructors.
- A learnable table (`role=latent`) stores unconstrained parameters and returns their sigmoid as its
  matrix, and an observed morphism returns the matrix it was given. Composition applies the rule's
  `tensor_op` and `join` as in [Morphisms §1.1](morphisms.md#11-composition-tensor-identity), and the
  tensor product applies `tensor_op` entrywise.
- The expression combinators:

| Combinator | Computes |
|---|---|
| `.marginalize(X, ...)` | reduces the named codomain axes with the rule's `join` |
| `fan(f, g, ...)` | gives one input to each component and concatenates their outputs |
| `repeat(f, n)` | composes one instance with itself, its parameters shared |
| `stack(f, n)` | composes copies in sequence, each with its own parameters |
| `scan(cell)` | runs a cell `A * H -> H` along the sequence axis and returns the last state, or every state with `outputs=all`, starting from the last position with `reverse=true` |

## 3. Composition of kernels {#3-probabilistic-composition}

The composite of two kernels integrates over the intermediate value
([Morphisms §3](morphisms.md#3-continuous-morphisms)). Drawing from it and evaluating its density are
different computations: a draw visits one intermediate value, while the density at a fixed output sums
the contributions of every intermediate value.

Drawing from `f >> g` is [ancestral sampling](https://en.wikipedia.org/wiki/Bayesian_network): draw the
intermediate value with `f.rsample`, then the output with `g.rsample`. When each factor draws from its
declared kernel, the result is a draw from the composite.

The density, `SampledComposition.log_prob`, depends on the intermediate:

- For a finite intermediate set, it sums over the set, and the result is the density of the composite.
- For a continuous intermediate space, it does not integrate. It sets the noise of each intermediate
  draw to zero, follows the resulting path $y_k = T_k(y_{k-1}, 0)$, for a Normal kernel the path of
  means, and returns $\sum_{k<n} \log p_k(y_k \mid y_{k-1}) + \log p_n(z \mid y_{n-1})$. This is the joint
  density of one path, not the density of the composite at $z$, and it raises an error when a factor
  cannot be reparameterized or has no density.

The composite density can thus be used as a likelihood only when every intermediate is finite or
deterministic. A program that needs the density of a composite through a continuous intermediate should
draw the intermediate as a site of its own, so that inference integrates over it.

`ScanMorphism.log_prob(x, y)` scores a given trajectory `y`, one state for each position, as the sum of
the densities of its transitions, as a scan inside a program scores the site it reaches at each position.
It refuses a final state given alone, whose density would be the marginal over every earlier state. A
composition whose first factor is a scan scores the trajectory its path selects. For a cell with no
conditional density, such as a deterministic network, the method returns $0$ for every trajectory,
including one that the cell cannot produce.

## 4. Programs and data

The steps of a program keep their keywords' meanings
([Programs as kernels](programs.md)):

- `sample` draws a value;
- `observe` adds the log density of a family at a given value;
- `let` binds deterministic arithmetic on tensors;
- `score` adds a computed scalar to the log joint density;
- `marginalize` over a family with finite support sums the latent variable out and removes it from the
  outer scope, and over any other family it draws one value and runs its scope once, without integrating;
- `return` selects the result.

`from_data("key")` reads data bound to the compiler before the expression is compiled, so the key must be
bound first.

## 5. Tests

- Tests of the compiler and the language exercise parsing, resolution, type errors, composition of
  expressions, contractions, programs, and structural declarations.
- The tests of the [program-shape protocol](program-theory.md) check that extraction is deterministic and
  that every extracted schema validates.
- Round-trip tests check that the data values of the library survive serialization to JSON.
- The transpilation tests check the separate pipeline of
  [Transpilation correctness](transpile-correctness/index.md).

## 6. Conditions

1. An equation that uses distributivity, arbitrary joins, or compact closure needs those laws of the
   module's composition rule, and three built-in instances, the default among them, fail distributivity
   ([Semirings, t-norms, and change of base §2](algebras.md#2-algebraic-scope)).
2. Floating-point arithmetic may differ from exact arithmetic, especially near the boundary of a support
   and in reductions that saturate.
3. The density of a composite through a continuous intermediate is the density of one path (§3), not the
   integral.
4. Comparing a family with another system's needs matching supports, parameterizations, and normalizing
   constants, which the [target pages](transpile-correctness/index.md#4-distribution-parameterization)
   record.
5. A [program-shape schema](program-theory.md) records shape: two models with one schema can have different
   learned weights and assign different probabilities.
