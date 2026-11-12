# Finite collection expressions

QVR applies a function across a tensor of fixed shape in two ways. A pure [`map`](#pure-map) builds
a value, and [`traverse`](#effect-aware-traverse) runs a [computation](computations.md#declaration-form)
once per element, each run finishing before the next begins. Both keep the collection's shape in
the result type.

## Operation reference

| Form | Position | Result | Main restriction |
| --- | --- | --- | --- |
| `map(xs, x -> body)` | pure expression | tensor with the leading extent of `xs` | `body` is pure |
| `traverse(xs, x -> computation(...))` | right side of `<-` | tensor of computation results | leading extent is known statically |
| `fold(xs, init, acc -> x -> body)` | pure expression | type of `init` | leading extent is known statically |
| `length(xs)` | pure expression | `Int` | leading extent is a literal |
| `logsumexp_over(xs, x -> score)` | pure expression | `Real` | `score` returns one `Real` per item |

All five operations run along the leading axis. If `xs` has type `Tensor[A]([n, d])` and the lambda
returns `Tensor[B]([k])`, then `map` and `traverse` return `Tensor[B]([n, k])`.

## Pure map

`map` takes an inline lambda or a lambda bound by a local pure binding:

<!-- compile: qiec -->
```qvr
define standardized(xs : Tensor[Real]([3]), center : Real, scale : Real) : Tensor[Real]([3]) !{} =
    let z_score = x -> (x - center) / scale
    return map(xs, z_score)
```

The lambda may capture pure values in scope, such as `center` and `scale`, and its parameter
shadows an outer name with the same spelling. Elaboration to the [Quivers Indexed Effect Core](index.md) (QIEC) lowers the form to a typed array comprehension,
so the reference machine runs it and a target that supports comprehensions renders it.

## Effect-aware traverse

`traverse` passes each item to a named computation and sequences the requests those calls make. It
is computation syntax and uses `<-`, even when the callee performs no effect:

<!-- compile: qiec -->
```qvr
define center(x : Real, mean : Real) : Real !{} =
    return x - mean

define center_all(xs : Tensor[Real]([3]), mean : Real) : Tensor[Real]([3]) !{} =
    let centered <- traverse(xs, x -> center(x, mean))
    return centered
```

The lambda body is one call of a named computation, and the callee's
[effect row](effects-and-handlers.md#effect-rows) joins the row of the enclosing computation. QIEC
expands a fixed collection into calls and bindings in source order, which gives each call its own
stable structural path, so interpreters and transpilers need no separate `traverse` primitive.

The lambda's callee receives no static arguments. When the callee has a static
[telescope](types-and-indexed-families.md#kinds-and-binders), wrap the specialized call in a helper
with none.

## Fold and derived reductions

`fold` is a [left fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)). Its lambda is
curried, taking the accumulator first and the current item second:

<!-- compile: qiec -->
```qvr
define weighted_total(values : Tensor[Real]([3]), weights : Tensor[Real]([3])) : Real !{} =
    let products = map([0, 1, 2], i -> values[i] * weights[i])
    return fold(products, 0.0, total -> value -> total + value)
```

QIEC elaborates a fold at its fixed extent, one application of the body per item. When the body
reads the accumulator more than once, or reads it inside a lambda or `factor` whose body runs once
per item, each intermediate accumulator is bound once and every read refers to that binding. The
elaborated term thus grows linearly in the collection's length, and each step runs once, even for a
body that reads the whole state at every step, such as the message update of the
[forward algorithm](https://en.wikipedia.org/wiki/Forward_algorithm) [@rabiner1989hmm], in which each
new component reads every old one:

<!-- compile: qiec -->
```qvr
define forward(start : Tensor[Real]([2]), transition : Tensor[Real]([2, 2]), emissions : Tensor[Real]([5, 2])) : Real !{} =
    let alpha = fold(emissions, start, message -> emission -> map([0, 1], j -> logsumexp_over([0, 1], i -> message[i] + transition[i, j]) + emission[j]))
    return logsumexp_over(alpha, score -> score)
```

The same sharing applies to a `map` or `fold` source that is not a variable, and to a lambda bound
by a local binding whose body reads its argument more than once: the source or argument is
evaluated once and bound.

`logsumexp_over(xs, f)` computes $\log \sum_x \exp f(x)$ over the items of `xs`, the
[log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) of `map(xs, f)` computed stably. It
normalizes a finite posterior given its log joint, as in
[normalization](../../guides/stochastic.md#normalization):

<!-- compile: qiec -->
```qvr
define normalize(log_joint : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    let log_evidence = logsumexp_over(log_joint, score -> score)
    return map(log_joint, score -> exp(score - log_evidence))
```

`length(xs)` returns the leading extent as an `Int`. An extent bound by a static binder is an index
term, not a runtime integer, so it is specialized before `length` or `fold` can use it.

## Dynamic filtering

The checked surface refuses `filter(xs, predicate)` with the stable diagnostic
`qiec-collection:filter:dynamic-shape`. The length of a filtered tensor depends on runtime values,
and every QIEC tensor has a fixed shape, so typing the result would need a length quantified
existentially or a ragged tensor type. A Boolean mask with a reduction keeps the original shape and
expresses the same computation. The eager PyTorch expression evaluator accepts `filter` outside
the checked surface.

## Diagnostics and target support

Collection errors are ordinary checked diagnostics. A wrong arity, a source that is not a tensor, a
function argument that is not a lambda, a leading extent known only at run time, or a result of the
wrong type fails during `qvr check`. A program that cannot elaborate is reported as
`qiec-program-gap` and is never omitted from the checked module without a diagnostic.

The reference machine runs every checked collection form. What each transpile target accepts is
listed under [Transpilation](execution-and-tooling.md#transpilation), and
`qvr check --target TARGET` reports the features a target lacks before transpiling.

The [finite-grid Bayes tutorial](../../tutorials/qvr/14-collection-programs.md) computes a complete
posterior with these forms.
