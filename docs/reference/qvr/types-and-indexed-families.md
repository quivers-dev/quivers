# Types and indexed families

An *indexed family* is a data type whose type carries an index, such as a vector's length, that
each constructor fixes. It is the
[inductive family](https://ncatlab.org/nlab/show/inductive+family) of dependent type theory
[@dybjer1994inductive], and in a language without dependent types the
[generalized algebraic data type](https://en.wikipedia.org/wiki/Generalized_algebraic_data_type)
(GADT) [@xi2003guarded]. A value of `Measurement(Observed)` and one of `Measurement(Missing)`
belong to one family, but a well-typed computation cannot use one where the other is expected. The
constructor determines the index, and [case analysis](#indexed-case-analysis) refines it inside
each branch.

## Kinds and binders

A *telescope* is an ordered list of binders in which later types may mention earlier binders
[@debruijn1991telescopic]. QVR writes static binders in square brackets and runtime binders in
round brackets:

| Form | Meaning | Used by |
| --- | --- | --- |
| `[A : Type]` | static type binder | families, constructors, effects, [handlers](effects-and-handlers.md#handler-declarations), computations |
| `[n : Nat]` | static index binder | constructors, operations, computations |
| `[E : Effect]` | static binder for an effect interface | declarations that abstract over an effect |
| `(n : Nat)` | family index, refined by case analysis | family declarations and case motives |
| `(x : T)` | runtime value parameter | computations and handler clauses |

Static arguments are written in brackets, as in `length[Real, S(Z)](xs)`, and runtime values in
parentheses. An integer literal may fill a `Nat` static binder, so `window[32](xs)` is valid when
`window` binds `[n : Nat]`.

The built-in value types are `Unit`, `Bool`, `Int`, `Real`, `String`, `LogWeight`,
`Tensor[A]([d1, ...])`, `Sampleable[A]`, and `Site[A]`. A
[product type](../../guides/core.md#productset-and-coproductset) is written `A * B`. A tuple value
is written `(a, b)`, and `pair[0]` and `pair[1]` project its components.

## Index types

An `index` declaration introduces a type of static indices together with all of its constructors:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)
index Availability = Observed | Missing
```

A constructor may be nullary or recursive. No constructor can be added after the declaration, so
the checker knows every constructor a `case` must cover. Index values are static terms, not runtime
integers. A `Nat` literal stands for an index only where a declaration binds a `Nat`.

## Family declarations

`Vec[A](n)` is a list of elements of type `A` whose type also records its length `n`. `Nil` has
length `Z`, and `Cons` has length one more than its tail, so a computation that requires a nonempty
vector cannot receive `Nil`. In the terms of inductive families, `A` is a *parameter*, fixed across
all constructors, and `n` is an *index*, which each constructor sets [@dybjer1994inductive]. QVR
writes parameters in brackets and indices in parentheses:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
```

A case branch cannot change `A`. It can learn `n`: matching `Nil` proves that the index is `Z`, and
matching `Cons[m]` proves that it is `S(m)`. A constructor's fields precede `->`, and a nullary
constructor has none. The [declaration form of a computation](computations.md#declaration-form)
shows how static arguments are received, and the
[finite collection operations](collection-expressions.md#operation-reference) use the extents they
fix to compute result shapes.

Family names and constructor names live in separate namespaces. A constructor's static arguments
are the family's parameters followed by the constructor's own telescope. A real vector of
three elements is thus built as follows:

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

Both `construct` and `as` are required. The annotation fixes the family application the value is
expected to have, so a constructor never takes its type from unrelated context.

## Indexed case analysis

A `case` expression states a *motive*, the result type written as a function of the scrutinee's
indices [@mcbride2002elimination]. The motive below is `Int` at every length, and the same form
allows a result type that depends on the index:

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

Matching `Cons[k]` shows that the vector has a head and a tail and that its length is one more than
the tail's. The branch binder `k` is a *rigid* index, a fresh variable that stands for an unknown
index inside one branch [@peytonjones2006simple]. It may fix the type of `tail` and the static
argument of the recursive call, but it cannot appear in the type of the branch's result.

Coverage is conservative. A constructor may be omitted only when its result index cannot equal the
scrutinee's, which the checker decides from the fact that distinct constructors never build equal
indices. This rule permits a total `head` with no `Nil` branch:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[k](first, rest) =>
            return first
```

`Nil` has index `Z`, which is distinct from `S(n)`, so no value of `Vec[A](S(n))` was built by
`Nil`, and the single branch covers every value of the declared input type.

## Pure expressions

Computation values and `program` expressions draw on one registry of checked primitives, the
portable set that lowering to the [Quivers Indexed Effect Core](index.md) (QIEC) and program elaboration share. The eager PyTorch compiler accepts
further names of its own, and those names do not become QIEC primitives.

Operators associate to the left within each precedence level:

| Precedence, low to high | Forms |
| --- | --- |
| Boolean disjunction | `||` |
| Boolean conjunction | `&&` |
| comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| additive | `+`, `-` |
| multiplicative | `*`, `/`, `%` |
| unary | `-`, `not` |

The registry holds typed scalar primitives, reductions, rowwise operations, and the collection forms
of fixed shape `map`, `fold`, `length`, and `logsumexp_over`.

| Form | Result |
| --- | --- |
| `sum`, `mean`, `prod`, `max`, `min`, `logsumexp` | collapse the last axis, so shape `(n, k)` becomes `(n,)` and a vector becomes a number |
| `softmax`, `log_softmax`, `cumsum`, `sort`, `normalize` | act along the last axis and keep the shape |
| `concat(a, b, ...)` | join tensors along the last axis, whose extents add, with the leading axes equal |
| `slice(v, start, length)` | keep `length` entries of the last axis from `start`, with both bounds integer literals |
| `select(c, a, b)` | `a` where the `Bool` `c` holds and `b` elsewhere, entry by entry, with both evaluated |
| `real(c)`, `int(c)` | read a `Bool` as one or zero |
| `matmul(a, b)` | sum the last axis of `a` against the first axis of a vector or matrix `b` |
| `layer_norm(v)`, `rms_norm(v)` | normalize each row along the last axis, with an optional real `eps` |

`traverse` is computation syntax, not a pure value. The collection forms and their restrictions
are specified in [Finite collection expressions](collection-expressions.md), and every primitive is
listed in the
[let-expression guide](../../guides/dsl-programs-and-lets.md#checked-primitive-reference).
Integer division and remainder truncate toward zero on every target. An operation with an `Int`
and a `Real` operand converts the `Int` explicitly where the surrounding form allows it. A
comparison with a negative literal is written `x < (-1)`, because `<-` is the token of effectful
binding.

### Reserved words

A let expression reads five words as something other than a name:

| Word | Meaning in an expression |
| --- | --- |
| `unit` | the unit value, of type `Unit` |
| `true`, `false` | the Boolean literals |
| `not` | Boolean negation |
| `factor` | the start of a factor comprehension |

None of them may name a binder: a `sample`, `observe`, or `marginalize` variable (alone or in a
tuple pattern), a `let` or `score` name, a factor binder, a lambda parameter, or a program
parameter. Every later reference to such a binder would read the literal or the keyword, so the
source is refused where the binder is written, as in
`reserved word at line 5, col 11: 'unit' names the unit value in expressions, so it cannot name a sample site`.
Names that contain or capitalize a reserved word (`units`, `Unit`, `is_true`) are ordinary names,
and so is `in`, which is a keyword only between a factor's binders and its body.

List literals build tensors. `[1, 2, 3]` has integer elements, `[1.0, 2.0, 3.0]` has real
elements, and nested lists add dimensions. A distribution family application builds a `Sampleable`
value, `site("x")` builds a typed site, and `log_prob(distribution, value)` returns a `LogWeight`.

## Common failures

- **Missing branch.** Add the constructor, unless its result index provably differs from the
  scrutinee's.
- **Escaped index.** Return a type that does not mention a static binder local to a branch, or
  package the index inside a family whose outer type is fixed.
- **Wrong constructor telescope.** Give the family's parameters first, then the constructor's own
  binders.
- **Value and static arguments confused.** Put types and indices in `[...]` and runtime values in
  `(...)`.
- **Shadowed primitive type.** A telescope binder named `Real` is a local type variable, not the
  primitive type. Rename it unless that abstraction is intended.

[Computations](computations.md) covers recursion and binding, and
[Effects and handlers](effects-and-handlers.md) covers indexed computations that perform requests.
