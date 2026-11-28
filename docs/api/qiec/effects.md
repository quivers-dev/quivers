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

### Implicit roots

`exact_implicit_gradient` takes a scalar residual $F(z,\theta)$ and an
exact root $z$ at the supplied parameter values. It requires
$F(z,\theta)=0$, a regular scalar execution, and a nonzero state derivative
$F_z(z,\theta)$. These conditions give a local solution branch through
$z$, with gradient $-F_\theta/F_z$. The result records the implicit
derivative conditions and the exact state derivative.

```python
from fractions import Fraction

from quivers.dsl import Compiler, parse
from quivers.qiec import exact_implicit_gradient

source = """
define residual(z : Real, x : Real) : Real !{} =
    return z * z - x
"""
module = Compiler(parse(source), module_name="exact-root-example").qiec_module
assert module is not None
result = exact_implicit_gradient(module, "residual", 3, (9,))
assert (result.value, result.gradient) == (3, (Fraction(1, 6),))
assert result.semantics.kind == "implicit"
```

::: quivers.qiec.differentiation

### Checkpointed scalar iterations

`exact_checkpointed_gradient` differentiates a finite sequence of
deterministic scalar transitions. Each transition reads the current
state and the same parameters. Replay recovers a segment's midpoint
state, reverses the later segment, and uses the saved initial state to
reverse the earlier segment.

For $n$ transitions, bisection saves at most $1+\lceil\log_2 n\rceil$
states when $n>0$. It builds and uses one local reverse graph at a time.
Each transition is reversed once and replayed at most once per level.
The result counts both kinds of evaluations and the peak number of saved
states. The reported derivative is for the specified finite iteration.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import exact_checkpointed_gradient

source = """
define advance(state : Real, rate : Real, offset : Real) : Real !{} =
    return state * rate + offset
"""
module = Compiler(parse(source), module_name="scalar-checkpoint-example").qiec_module
assert module is not None
result = exact_checkpointed_gradient(module, "advance", 3, (1, 2), 8)
assert (result.value, result.gradient) == (19, (1, 80, 8))
assert (result.replay_steps, result.reverse_steps, result.peak_saved_states) == (12, 8, 4)
```

::: quivers.qiec.checkpointing

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

### Nominal finite effect equations

`EffectEquation` names one effect interface, a concrete instance, the
operations in the equation, and two checked computations over that
instance. `check_finite_effect_law` installs the selected authored handler
around each computation. Each assignment includes the initial values of
the handler's parameters. An equation can use a finite product as a
continuation table, so the checker sees every possible response in the
declared carrier.

The decision binds the checked module, equation and handler identities,
and the finite checking domain. Passing an earlier decision checks that
binding and runs the finite check again.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import EffectEquation, check_finite_effect_law

source = """
effect ReadOnly
    read_value : Unit -> Bool

instance reader : ReadOnly

handler fixed for ReadOnly : Bool -> Bool with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    read_value resumes 1 =>
        resume(current)

define repeated() : Bool !{reader} =
    let first <- perform reader.read_value()
    let second <- perform reader.read_value()
    if first then
        return second
    else
        if second then
            return false
        else
            return true

define agreed() : Bool !{} =
    return true
"""
module = Compiler(parse(source), module_name="read-effect-law").qiec_module
assert module is not None
instance = next(item.entry for item in module.instances if item.name == "reader")
effect = next(item for item in module.effects if item.ref.id == instance.effect.id)
roots = {item.name: item.id for item in module.computations}
equation = EffectEquation(
    effect.ref.id, "same-read", instance.instance, (effect.operations[0].id,),
    roots["repeated"], roots["agreed"],
)
decision = check_finite_effect_law(module, equation, module.handlers[0].id)
assert decision.certified
assert decision.result.checked == 2
```

::: quivers.qiec.effect_laws

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

## Rational derivative rules

An authored scalar derivative rule is a checked computation. A forward
rule takes the primal arguments followed by their tangents. Each reverse
component takes the primal arguments and one output cotangent. Reverse
components follow the order of the primal parameters.

The checker differentiates the primal's symbolic rational form, then
compares each supplied rule with the resulting derivative action for all
seeds. This establishes linearity in the seeds and, when both rules are
supplied, transposition. A false identity has an exact separating input.

Every evaluated divisor also supplies a nonzero guard. The checker must
establish that all candidate rules are defined everywhere the primal is
defined, including at zero seeds. A rule may divide by the square of a
divisor already checked by the primal. When the checker cannot establish
domain coverage, it raises `RationalLawUnsupported`.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import check_rational_derivative_rule

source = """
define ratio(x : Real, y : Real) : Real !{} =
    return x / y

define push(x : Real, y : Real, dx : Real, dy : Real) : Real !{} =
    return (dx * y - x * dy) / (y * y)

define pull_x(x : Real, y : Real, seed : Real) : Real !{} =
    return seed / y

define pull_y(x : Real, y : Real, seed : Real) : Real !{} =
    return (0.0 - seed * x) / (y * y)
"""
module = Compiler(parse(source), module_name="checked-ratio-rule").qiec_module
assert module is not None
decision = check_rational_derivative_rule(
    module, "ratio", jvp="push", vjp=("pull_x", "pull_y")
)
assert decision.certified
assert decision.domain_polynomial == (((0, 1), 1),)
```

::: quivers.qiec.derivative_rules
