# Finite collection expressions

QVR has two operations for applying a function across a tensor. A pure
[`map`](#pure-map) builds a value; [`traverse`](#effect-aware-traverse) runs a [computation](computations.md#declaration-form) for each element in
order. Both retain the collection’s shape in its type. For `traverse`,
each computation completes before the next begins.

## Operation reference

| Form | Position | Result | Main restriction |
| --- | --- | --- | --- |
| `map(xs, x -> body)` | pure expression | tensor with the leading extent of `xs` | `body` is pure |
| `traverse(xs, x -> computation(...))` | right side of `<-` | tensor of computation results | leading extent is statically known |
| `fold(xs, init, acc -> x -> body)` | pure expression | type of `init` | leading extent is statically known |
| `length(xs)` | pure expression | `Int` | leading extent is a literal |
| `logsumexp_over(xs, x -> score)` | pure expression | `Real` | `score` returns one `Real` per item |

All five operations use the leading axis. If `xs` has type
`Tensor[A]([n, d])` and the lambda returns `Tensor[B]([k])`, `map` and
`traverse` return `Tensor[B]([n, k])`.

## Pure map

`map` accepts an inline lambda or a lambda named by a local pure binding:

<!-- compile: qiec -->
```qvr
define standardized(xs : Tensor[Real]([3]), center : Real, scale : Real) : Tensor[Real]([3]) !{} =
    let z_score = x -> (x - center) / scale
    return map(xs, z_score)
```

The lambda may capture surrounding pure values such as `center` and `scale`.
Its parameter shadows an outer name with the same spelling. QIEC lowers the
form to a typed comprehension, so the result remains available to the
reference machine and to targets with comprehension support.

## Effect-aware traverse

Mapping a pure function over `[1,2,3]` can directly build a new tensor.
Calling a computation for each item must also sequence any requests
those calls make. `traverse` makes that order explicit. Even when
`center` is pure, it is declared as a computation, so its calls use
`traverse` and `<-`; a pure arithmetic lambda can instead use `map`.


Use `traverse` when each item must be passed to a named computation. Its
surface is a computation call and thus uses `<-`:

<!-- compile: qiec -->
```qvr
define center(x : Real, mean : Real) : Real !{} =
    return x - mean

define center_all(xs : Tensor[Real]([3]), mean : Real) : Tensor[Real]([3]) !{} =
    let centered <- traverse(xs, x -> center(x, mean))
    return centered
```

The lambda body must be one named computation call. The callee's [effect row](effects-and-handlers.md#effect-rows)
joins the row of the enclosing computation. QIEC expands a fixed collection
into calls and binds in source order, which gives each call its own stable
structural path. This expansion also means that existing interpreters and
transpilers need no separate `traverse` runtime primitive.

The current form does not pass static arguments to the lambda's callee. Wrap a
specialized call in a monomorphic helper when a computation has a static
[telescope](types-and-indexed-families.md#kinds-and-binders).

## Fold and derived reductions

A left fold carries one running value through the collection. With
`[2,3,4]`, initial total `0`, and addition, the intermediate totals
are `2`, `5`, and `9`. The curried lambda `total -> value -> ...`
means "first receive the current total, then the next item."
`logsumexp_over` performs a different reduction: if log weights
correspond to `0.2` and `0.3`, it returns `log(0.5)`. Subtracting
that value and exponentiating normalizes them to `0.4` and `0.6`.


`fold` is a left fold. The lambda is curried with the accumulator first and
the current item second:

<!-- compile: qiec -->
```qvr
define weighted_total(values : Tensor[Real]([3]), weights : Tensor[Real]([3])) : Real !{} =
    let products = map([0, 1, 2], i -> values[i] * weights[i])
    return fold(products, 0.0, total -> value -> total + value)
```

`logsumexp_over(xs, f)` is the stable log-space reduction of `map(xs, f)`.
It is useful for finite Bayesian [normalization](../../guides/stochastic.md#normalization):

<!-- compile: qiec -->
```qvr
define normalize(log_joint : Tensor[Real]([3])) : Tensor[Real]([3]) !{} =
    let log_evidence = logsumexp_over(log_joint, score -> score)
    return map(log_joint, score -> exp(score - log_evidence))
```

`length(xs)` returns the leading extent as an `Int`. A statically polymorphic
extent is an index term rather than a runtime integer, so it must be
specialized before `length` or `fold` can use it.

## Dynamic filtering

Filtering can change the length according to runtime values: three
inputs might yield zero, one, two, or three outputs. A checked
fixed-shape tensor cannot assign one known extent to all those cases.
A mask keeps the original shape and marks which positions contribute,
so reductions can ignore positions without creating a variable-length
result.


The eager PyTorch expression evaluator retains `filter(xs, predicate)` for
host-only workflows. Checked QIEC tensors have fixed shape, so an unmasked
filter would need an existential length or a ragged-tensor type. The checked
surface thus rejects `filter` with the stable
`qiec-collection:filter:dynamic-shape` diagnostic. Use a Boolean mask and a
reduction when the output can stay at its original shape.

## Diagnostics and target support

Collection errors are ordinary checked diagnostics: wrong arity, a non-tensor
source, a non-lambda function argument, a dynamic leading extent, or a result
of the wrong type fails during `qvr check`. A program that cannot elaborate is
reported as `qiec-program-gap`; it is not silently omitted from the checked
module.

The reference machine executes every checked collection form. Dynamic targets lower the
resulting comprehensions, gathers, calls, and binds through the common QIEC
runtime. Stan, BUGS, and JAGS may reject those core capabilities. Run
`qvr check --target TARGET` to test the selected backend before transpiling.

See the [finite-grid Bayes tutorial](../../tutorials/qvr/14-collection-programs.md)
for a complete posterior calculation, and [Computations](computations.md) for
the distinction between `=` and `<-`.
