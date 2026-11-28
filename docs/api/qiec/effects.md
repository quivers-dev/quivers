# Effects

An effect interface declares the operations a computation can request and the types of their
arguments and results [@plotkin2009handlers; @bauer2015programming]. An *effect instance* is one
named occurrence of an interface, so two state instances share an interface without sharing their
state ([instances](../../tutorials/qvr/10-effects-and-handlers.md#2-write-an-authored-handler)). An
[effect row](../../reference/qvr/effects-and-handlers.md#effect-rows) lists the instances a
computation may request [@leijen2014koka]. The module gives typed effect interfaces, rows, and
handler signatures.

Discharge allocates the addresses of inlined calls and local instances
at their original entry points. A residual request observes the live
addresses in their original order. Resumptions restore the same
allocations for each shot, which has its own resumption path. Effects in
an intervening clause use the scope outside the captured continuation.

A retained call also inherits its enclosing discharged shots and
clause activations. Discharge keeps the reference handler when the call's
residual row is open or could invoke an outer multi-shot clause inside
that path.

An effectful return clause uses the address of its handler installation.
Different installations of the same declaration have different return
addresses. Multiple shots through one installation share its return
address, with their own shot paths. If an outer handler captures the
return, its operation-clause address includes the captured activation
alongside any captured local instance addresses.

## Exact scalar derivatives

`exact_jvp` runs a checked scalar computation and returns its value and
Jacobian-vector product as rational numbers. It follows the reference
machine through authored clauses, parameter updates, resumptions, and
recursive calls. Its arithmetic includes division by nonzero values and
powers with fixed integral exponents.

Every comparison reached by the run must either have unequal operands
or be independent of the input. This gives a neighborhood where the
same branch is taken. At an input-dependent equality, the checker raises
`JvpBoundaryError`, even when the supplied tangent is zero. Unsupported
primitives, undefined arithmetic, and exhausted limits raise errors.

The handler below doubles the value supplied to its continuation, which
then squares it. The resulting function is $f(x)=4x^2$. At $x=3$, a
direction of $1/2$ gives $Df(3)[1/2]=12$.

```python
from fractions import Fraction

from quivers.dsl import Compiler, parse
from quivers.qiec import exact_jvp

source = """
effect Staging
    shift : Real -> Real

instance stage : Staging

handler double for Staging : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shift(v : Real) resumes 1 =>
        resume(2.0 * v)

define squared(x : Real) : Real !{} =
    handle stage with double in
        let y <- perform stage.shift(x)
        return y * y
"""
module = Compiler(parse(source), module_name="exact-jvp-example").qiec_module
assert module is not None
result = exact_jvp(module, "squared", (3,), (Fraction(1, 2),))
assert (result.value, result.tangent) == (36, 12)
```

`exact_mixed_derivative` seeds two perturbations independently. Their
product gives the mixed directional derivative. For the same $f$,
$D^2 f(3)[1/2,2]=8$.

```python
from quivers.qiec import exact_mixed_derivative

result = exact_mixed_derivative(module, "squared", (3,), (Fraction(1, 2),), (2,))
assert (result.value, result.first, result.second, result.mixed) == (36, 12, 48, 8)
```

`exact_gradient` records the scalar operations executed by the reference
machine and then accumulates their cotangents in reverse order. Every use
of a value contributes, including uses of a captured value in different
resumptions. The result lists partial derivatives in parameter order.

```python
from quivers.qiec import exact_gradient

result = exact_gradient(module, "squared", (3,))
assert (result.value, result.gradient) == (36, (24,))
```

::: quivers.qiec.differentiation

::: quivers.qiec.effects

## Finite laws

The finite law checker compares two named QIEC computations with the same
positional input types and result type. It runs both sides for every assignment
of Booleans, unit values, and their finite products. Each side has an empty
residual effect row, and reachable handler clauses use the same exact finite
fragment.

`finite_continuation_type` gives a total continuation one result cell for every
tuple of arguments. Include handler state among those arguments when the
continuation can observe it. Both sides receive the same table. For a continuation
from a pair of Booleans to a Boolean, the four cells are indexed by
`(false, false)`, `(false, true)`, `(true, false)`, and `(true, true)`, in that
order.

The checker returns the first counterexample with its full assignment and both
results. A positive decision follows only after every assignment completes.
An unsupported term or an exhausted limit raises an error.

::: quivers.qiec.laws

## Polynomial laws

The polynomial checker compares scalar real computations after discharging
their authored handlers. It interprets the admitted arithmetic over rational
coefficients. A real literal contributes the exact rational value it represents,
and a quotient of two constants can supply an exact coefficient such as
`1.0 / 3.0`.

The checker supports addition, subtraction, multiplication, division by a
closed nonzero constant, and nonnegative integral constant powers. It checks
divisors in the source before using the discharged result, including divisors
in bindings whose results are unused. A runtime branch condition leaves the
polynomial fragment.

Each normal form lists monomial exponents in parameter order and their
coefficients. Equal forms certify equality on the scalar polynomial domain.
When the forms differ, the checker gives integer arguments and the two exact
rational results.

::: quivers.qiec.polynomial_laws

## Rational laws

The rational checker runs the original, checked computation with exact
symbolic scalar arithmetic. A divisor contributes a nonzero condition
whenever it is evaluated, including in an unused binding. Both sides of an
equation must have the same nonzero domain polynomial after multiplying
their distinct monic guards. When the checker cannot establish this domain
agreement, it reports an unsupported law.

On a shared domain the checker compares exact numerators over a common
denominator. If they differ, it returns integer arguments where both
computations are defined and gives their exact rational values. The
checker supports authored handlers whose branch choices can be resolved
during the symbolic run, including branches selected by resumed Booleans.

::: quivers.qiec.rational_laws
