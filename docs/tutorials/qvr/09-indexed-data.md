# Indexed data and total functions

This tutorial builds a length-indexed vector and uses its index to rule out
invalid states. The statistical reason to care is familiar: [shape, missingness, and support assumptions](02-glms.md#common-mistakes) often begin as comments or runtime checks. An indexed
family can make one of those assumptions part of the computation's type.

We will build a summary function whose input type records the vector length
and whose recursive branches preserve that information.

An ordinary list says what its elements are; a length-indexed vector also
states how many it contains. `Nil` constructs length zero. Adding a head to
that vector with `Cons` constructs length one; adding another constructs
length two. The length in the type is evidence supplied by the constructor,
not a second, independently editable integer.

When a function inspects a nonempty constructor, it learns that a tail exists
and what its length is. The *case motive* states the result type expected
from that inspection. This example always returns an integer or a real sum,
so the result type does not itself need to vary with the learned length.

## 1. Declare the index

`Nat` is a closed index sort. `Z` represents zero and `S(n)` its successor.

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define length[A : Type, n : Nat](xs : Vec[A](n)) : Int !{} =
    case xs motive (m : Nat) => Int
        Nil =>
            return 0
        Cons[k](head, tail) =>
            let rest <- length[A, k](tail)
            return 1 + rest
```

The family has one uniform parameter `A` and one refinable index `n`. `Nil`
constructs only `Vec[A](Z)`. `Cons[m]` accepts a tail of length `m` and
constructs a vector of length `S(m)`.

The case motive says that every branch returns `Int`, regardless of the
refined length. In the `Cons` branch, `tail` has type `Vec[A](k)`, so the
recursive call must be specialized at `k`.

## 2. Fold without a shape check

Add a real-valued fold:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define sum[n : Nat](xs : Vec[Real](n), acc : Real) : Real !{} =
    case xs motive (m : Nat) => Real
        Nil =>
            return acc
        Cons[k](head, tail) =>
            let rest <- sum[k](tail, acc + head)
            return rest
```

No branch asks whether the tail is empty, and no runtime integer is decremented.
The constructor proof selects the valid recursive argument.

## 3. Build a closed vector

Constructor values state their family application explicitly:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define triple(a : Real, b : Real, c : Real) : Vec[Real](S(S(S(Z)))) !{} =
    let last = construct Cons[Real, Z](c, construct Nil[Real]() as Vec[Real](Z)) as Vec[Real](S(Z))
    let pair = construct Cons[Real, S(Z)](b, last) as Vec[Real](S(S(Z)))
    return construct Cons[Real, S(S(Z))](a, pair) as Vec[Real](S(S(S(Z))))
```

The static arguments to `Cons` are the uniform element type followed by its
local predecessor index. The `as` annotation closes the resulting family
application.

## 4. Run the summary

The complete source is
[`indexed-vectors.qvr`](source/indexed-vectors.qvr). Check it, list its entries,
and invoke the pure `summarize` computation:

```bash
qvr check docs/tutorials/qvr/source/indexed-vectors.qvr
qvr run docs/tutorials/qvr/source/indexed-vectors.qvr --list
qvr run docs/tutorials/qvr/source/indexed-vectors.qvr summarize 1.5 2.0 3.5 --json
```

The same entry boundary is available in Python:

```python
from quivers.dsl import load

indexed = load("docs/tutorials/qvr/source/indexed-vectors.qvr")
run = indexed.run("summarize", 1.5, 2.0, 3.5)
assert run.value == (7.0, 3)
print(run.value)
```

`run.entry.kind` is `"computation"`, the runtime is `core`, and the trace
contains the recursive calls as checked evaluator events.

## 5. Transfer the pattern to statistical data

Length is only one useful index. Missingness can be made explicit with
`Measurement(Observed)` and `Measurement(Missing)`; censoring can distinguish
`Exact`, `Left`, `Right`, and `Interval`; a covariance factor can distinguish
`Raw`, `Cholesky`, and `PositiveDefinite` states. The design test is whether a
constructor establishes an invariant that later case analysis can use.

A potential worry is that the index merely moves validation into more syntax.
For data arriving from an untyped host, a boundary check is still necessary.
But after that check constructs the indexed value, every downstream
computation receives the refined type, so the invariant is checked once rather
than rediscovered at each use.

Next, [Effects and handlers](10-effects-and-handlers.md) adds an explicit
interpretation to an operation while keeping the calling computation abstract.
The [type reference](../../reference/qvr/types-and-indexed-families.md) gives
the complete family and case rules.
