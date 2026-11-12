# 9. Indexed data and total functions

An [inductive family](https://ncatlab.org/nlab/show/inductive+family) is a datatype indexed by values, whose constructors fix the index of the values they build [@dybjer1994inductive]. Generalized algebraic datatypes ([GADTs](https://en.wikipedia.org/wiki/Generalized_algebraic_data_type)) are the same idea in functional programming [@xi2003guarded]. Declare vectors indexed by their length and write functions over them that need no length check at run time. A shape assumption that would otherwise be a comment or a runtime assertion becomes part of a type.

## 1. Declare the index

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

`index Nat = Z | S(Nat)` declares a finite enumerated index type whose values are the natural numbers in unary: `Z` is zero and `S(n)` is the successor of `n`. `Vec[A](n)` has a *parameter* `A`, the element type, which is the same in every constructor, and an *index* `n`, the length, which each constructor fixes: `Nil` builds only `Vec[A](Z)`, and `Cons[m]` takes a tail of length `m` and builds a vector of length `S(m)`. The length is thus established by the constructors and cannot disagree with the elements.

`length` is a computation (`define`) whose effect row `!{}` is empty: it performs no effects ([chapter 10](10-effects-and-handlers.md) introduces effect rows). Its `case` analysis states a *motive*, the result type of the analysis as a function of the inspected value's index [@mcbride2002elimination]. Here the motive `(m : Nat) => Int` returns `Int` whatever the length. In the `Cons` branch the checker learns that the input has length `S(k)` for some `k`, so `tail` has type `Vec[A](k)` and the recursive call is made at `k`.

## 2. Fold without a shape check

A sum over a real vector:

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

No branch tests whether the tail is empty, and no runtime counter is decremented. Matching `Cons[k]` gives the index `k` of the tail, which fixes the type of the recursive call.

## 3. Build a vector

A constructor application names the family it builds:

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

The bracketed arguments of `Cons` are the parameter (the element type) followed by the constructor's own index `m`, the length of the tail. The `as` annotation states the type of the result.

## 4. Run the summary

The complete source is [`indexed-vectors.qvr`](source/indexed-vectors.qvr), whose computation `summarize` builds a vector of three elements and returns its sum and its length. Check the file, list its [entry points](../../reference/qvr/execution-and-tooling.md#list-entry-points) (the computations and programs a module exposes to `qvr run` and `Program.run`), and run `summarize`:

```bash
qvr check docs/tutorials/qvr/source/indexed-vectors.qvr
qvr run docs/tutorials/qvr/source/indexed-vectors.qvr --list
qvr run docs/tutorials/qvr/source/indexed-vectors.qvr summarize 1.5 2.0 3.5 --json
```

The same entry point from Python:

```python
from quivers.dsl import load

indexed = load("docs/tutorials/qvr/source/indexed-vectors.qvr")
run = indexed.run("summarize", 1.5, 2.0, 3.5)
assert run.value == (7.0, 3)
print(run.value)
```

`run.entry.kind` is `"computation"`. The computation runs on the reference machine, the library's interpreter for the Quivers Indexed Effect Core ([QIEC](../../reference/qvr/index.md)), the typed core language that QVR declarations elaborate to. `run.result.runtime` is `"core"`, and `run.result.trace` records each step of the run, the recursive calls included, as an event.

## 5. Transfer the pattern to statistical data

Length is one useful index among several. An index can record whether a measurement is observed or missing (the [type reference](../../reference/qvr/types-and-indexed-families.md) uses a family `Measurement` indexed by `Observed` and `Missing`), whether a value is exact or censored on the left, on the right, or to an interval, or whether a covariance parameter is raw, a Cholesky factor, or positive definite. An index earns its place when a constructor establishes an invariant that later case analysis uses.

A potential worry is that the index only moves validation into more syntax. Data from an untyped source still needs a check where it enters. But once that check has built the indexed value, every later computation receives the refined type, so the invariant is checked once and not at each use.

## Next

[Chapter 10](10-effects-and-handlers.md) declares an effect and gives its operations an interpretation with a handler. The [type reference](../../reference/qvr/types-and-indexed-families.md) gives the complete rules for families and case analysis.
