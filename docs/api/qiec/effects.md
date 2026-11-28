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

## Superposition, repeated draws, and completion

In `superposed`, the measure at `x` is
$p\,\mathcal N(\mu_a,1)+(1-p)\,\mathcal N(\mu_b,1)$. Replay at
$x$ scores the logarithm of that sum and records one `Random` address.
In `two_shots`, the handler evaluates two resumption branches. Each reaches
the same source origin with a distinct resumption ordinal. Their replay
joint is the product of the two Normal densities. The handler returns the
weighted arithmetic mean of the two drawn values.

The last two programs give a product-space completion with different sites.
In `disjoint`, the selector $z$ determines the sole reached Bernoulli site.
Write its conditional mass as $f_z(x)$ and the selector mass as $q_z$.
`completed` also draws the inactive site from the normalized law
$f_{1-z}(y)$. For each selector and active result, summing its extended
mass $q_z f_z(x) f_{1-z}(y)$ over $y$ gives $q_z f_z(x)$. The returned
value and its marginal score under the selector are preserved. The
completed program has three realized addresses where the disjoint program
has two. An auxiliary factor with total mass other than one changes the
marginal by that factor.

<!-- compile: qiec -->
```qvr
object UnitInput : FinSet 1
object Output : Real 1

instance random : Random

effect Choice
    split : Real -> Bool

instance choice : Choice

handler both for Choice : Real -> Real [coverage=total, forwards=none, implementation=authored]
    split(probability : Real) resumes omega =>
        let first <- resume(true)
        let second <- resume(false)
        return probability * first + (1.0 - probability) * second

define choose_mean(first : Bool, left : Real, right : Real) : Real !{} =
    if first then
        return left
    else
        return right

define draw_twice(p : Real, left : Real, right : Real) : Real !{random} =
    handle choice with both in
        let first <- perform choice.split(p)
        let mean <- choose_mean(first, left, right)
        perform random.sample[Real](site("x"), Normal(mean, 1.0))

define lift_truth(flag : Bool) : Real !{} =
    if flag then
        return 1.0
    else
        return 0.0

define selected_draw(first : Bool, p_left : Real, p_right : Real) : Real !{random} =
    if first then
        let value <- perform random.sample[Bool](site("left"), Bernoulli(p_left))
        lift_truth(value)
    else
        let value <- perform random.sample[Bool](site("right"), Bernoulli(p_right))
        lift_truth(value)

define selected_value(first : Bool, left : Bool, right : Bool) : Real !{} =
    if first then
        lift_truth(left)
    else
        lift_truth(right)

define draw_mixture(p : Real, mu_a : Real, mu_b : Real) : Real !{random} =
    let weights = [p, 1.0 - p]
    let locations = [mu_a, mu_b]
    perform random.sample[Real](site("x"), MixtureNormal(weights, locations, [1.0, 1.0]))

program superposed(p : Real, mu_a : Real, mu_b : Real) : UnitInput -> Output
    let x <- draw_mixture(p, mu_a, mu_b)
    return x

program two_shots(p : Real, mu_a : Real, mu_b : Real) : UnitInput -> Output
    let x <- draw_twice(p, mu_a, mu_b)
    return x

program disjoint(p : Real, p_left : Real, p_right : Real) : UnitInput -> Output
    sample first <- Bernoulli(probs=p)
    let x <- selected_draw(first, p_left, p_right)
    return x

program completed(p : Real, p_left : Real, p_right : Real) : UnitInput -> Output
    sample first <- Bernoulli(probs=p)
    sample left <- Bernoulli(probs=p_left)
    sample right <- Bernoulli(probs=p_right)
    let x <- selected_value(first, left, right)
    return x

export superposed
export two_shots
export disjoint
export completed
```

::: quivers.qiec.effects

## Handler carrier shapes

`classify_handler_carrier` inspects the actual checked clauses of a
handler. A `tail-state` clause resumes exactly once in tail position.
A `tail-or-abort` clause can also return an answer directly. The
classifier reports `continuation` when a clause uses general control,
including a resumption followed by other work.

The classifier checks both arms of a conditional unless its condition
is a Boolean literal. Closed calls in a prefix are checked through their
reachable bodies.

The report retains the nominal operation, the declared resumption
grade, and the handler's return action. A compiler uses separate
conditions for each transformation, including exchanging handlers.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import classify_handler_carrier

source = """
effect Offset
    advance : Real -> Real

instance offset : Offset

handler carry for Offset : Real -> Real with (current : Real) [coverage=total, forwards=none, implementation=authored]
    return answer =>
        return answer
    advance(v : Real) resumes 1 =>
        resume(v + current) with (v)
"""
module = Compiler(parse(source), module_name="carrier-example").qiec_module
assert module is not None
shape = classify_handler_carrier(module, module.handlers[0].id)
assert shape.kind == "tail-state"
assert shape.return_action == "identity"
assert shape.clauses[0].completions == ("resume",)
```

::: quivers.qiec.handler_carriers


## Recursive state specialization

`specialize_recursive_state` rewrites a named installation of a single
nominal state handler into pure QIEC computations. The handler must be
total and authored. Each operation must resume exactly once in tail
position through ordinary closed prefixes. The specialization currently
requires monomorphic declarations with no residual effects.

The specialized body passes the current state to each effectful callee.
Each callee returns its result and the new state. A clause evaluates the
reply and all updated state coordinates in the old state, in declaration
order. Recursive calls reuse the same companion definitions, so the size
of the rewritten module does not depend on how many calls a run takes.
Only the original installation applies the handler's return action.

Module-level discharge uses the specialization when it reaches a
recursive handled call within this fragment. The original and specialized
computations produce the same values under the checked contracts, and
surviving effects in the caller keep their addresses. The specialization
removes the handled requests and can take a different number of runtime
steps. A target still needs support for the resulting products and
recursive calls. A direct call to the specialization reports unsupported
control or an exceeded construction limit.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import run_named, specialize_recursive_state

source = """
effect Counter
    read_value : Unit -> Real
    write_value : Real -> Unit

instance counter : Counter

handler threaded for Counter : Real -> (Real * Real) with (state : Real) [coverage=total, forwards=none, implementation=authored]
    return answer =>
        return (answer, state)
    read_value resumes 1 =>
        resume(state)
    write_value(value : Real) resumes 1 =>
        resume() with (value)

define walk(n : Real) : Real !{counter} =
    if n > 0.0 then
        let prior <- perform counter.read_value()
        perform counter.write_value(prior + 1.0)
        let answer <- walk(n - 1.0)
        return answer + prior
    else
        perform counter.read_value()

define execute(n : Real, initial : Real) : (Real * Real) !{} =
    handle counter with threaded(initial) in
        walk(n)
"""
module = Compiler(parse(source), module_name="recursive-counter").qiec_module
assert module is not None
entry = next(item for item in module.computations if item.name == "execute")
specialized = specialize_recursive_state(module, entry.id)
result = run_named(specialized, "execute", (2, 3))
assert tuple(map(float, result.value)) == (12.0, 5.0)
```

::: quivers.qiec.state_specialization

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

### Nominal scalar effect equations

`check_scalar_effect_law` checks an `EffectEquation` whose inputs,
handler state and observed answers are scalar reals. Choose the
polynomial or rational interpretation when running the check. The
rational method requires agreement of the two strict domains. A prior
decision is bound to its selected method as well as the equation and
handler.

The example checks additivity for an operation whose handler divides
its argument by the supplied scale. Both sides are defined when that
scale is nonzero.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import EffectEquation, check_scalar_effect_law

source = """
effect Scale
    transform : Real -> Real

instance scaling : Scale

handler divide for Scale : Real -> Real with (s : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    transform(v : Real) resumes 1 =>
        resume(v / s)

define combined(a : Real, b : Real) : Real !{scaling} =
    perform scaling.transform(a + b)

define separate(a : Real, b : Real) : Real !{scaling} =
    let first <- perform scaling.transform(a)
    let second <- perform scaling.transform(b)
    return first + second
"""
module = Compiler(parse(source), module_name="scale-effect-law").qiec_module
assert module is not None
instance = next(item.entry for item in module.instances if item.name == "scaling")
effect = next(item for item in module.effects if item.ref.id == instance.effect.id)
roots = {item.name: item.id for item in module.computations}
equation = EffectEquation(
    effect.ref.id, "additivity", instance.instance, (effect.operations[0].id,),
    roots["combined"], roots["separate"],
)
decision = check_scalar_effect_law(module, equation, module.handlers[0].id, method="rational")
assert decision.certified
assert decision.result.domain_polynomial == (((0, 0, 1), 1),)
```

::: quivers.qiec.effect_laws

### Effect law obligations

An `EffectLawBundle` pairs a checked module with a set of
`EffectLawObligation` records. Each obligation names the equation,
handler and checking method and can carry named hypotheses. The bundle
checks these identifiers against the module and assigns one statement to
each effect's law name. Use the indicated exact checker on the module
and equation to obtain a decision.

The bundle round-trips with `dumps` and `loads`. The obligation in this
example requests a rational check of the scale law above.

```python
from quivers.qiec import EffectLawBundle, EffectLawObligation, dumps, loads

bundle = EffectLawBundle(
    module=module,
    obligations=(
        EffectLawObligation(equation=equation, handler=module.handlers[0].id, method="rational"),
    ),
)
decoded = loads(dumps(bundle))
assert isinstance(decoded, EffectLawBundle)
obligation = decoded.obligations[0]
assert obligation.method == "rational"
rechecked = check_scalar_effect_law(
    decoded.module, obligation.equation, obligation.handler, method=obligation.method
)
assert rechecked == decision
```

::: quivers.qiec.effect_law_schema

### Checked effect plans

`Compiler.effect_plan` reports the carrier shapes of handlers in a selected
entry's checked call graph, including calls made from authored clauses.
It also reports ordinary discharge and its obstacles. An
`EffectLawBundle` can be inspected directly with `checked_effect_plan`.
`EffectLawEvidence.for_module` attaches a `Tested` or `Assumed` claim
to that exact module and obligation. Pass these records as `law_evidence`
to `Compiler.effect_plan`. Each comparison reports its exact check
status and the held declaration. A refuted equation retains its
counterexample even if the attached record declares an assumption.
The digest changes when the source or its checked operations change.

The report gives every requested law its own result. A checked law carries
the evidence for its actual named handler. A refuted law carries the
checker’s separating assignment. An unsupported comparison, exhausted
limit, or unproved named condition leaves that obligation unresolved.
Ordinary discharge applies its own checked rules independently of these
reports.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import EffectEquation, EffectLawObligation

source = """
effect Reader
    read_value : Unit -> Bool

instance reader : Reader

handler fixed for Reader : Bool -> Bool with (value : Bool) [coverage=total, forwards=none, implementation=authored]
    read_value resumes 1 =>
        resume(value)

define read_once() : Bool !{reader} =
    perform reader.read_value()

define read_twice() : Bool !{reader} =
    let ignored <- perform reader.read_value()
    perform reader.read_value()

define example(value : Bool) : Bool !{} =
    handle reader with fixed(value) in
        read_once()
"""
compiler = Compiler(parse(source), module_name="planned-reader")
module = compiler.qiec_module
assert module is not None
effect = next(item for item in module.effects if item.ref.name == "Reader")
instance = next(item.entry for item in module.instances if item.name == "reader")
handler = next(item for item in module.handlers if item.name == "fixed")
roots = {item.name: item.id for item in module.computations}
equation = EffectEquation(
    effect.ref.id, "repeated-read", instance.instance,
    tuple(item.id for item in effect.operations), roots["read_once"], roots["read_twice"],
)
obligation = EffectLawObligation(equation=equation, handler=handler.id, method="finite")
plan = compiler.effect_plan("example", obligations=(obligation,))
assert plan.carriers[0].kind == "tail-state"
assert plan.laws[0].status == "checked"
```

::: quivers.qiec.effect_plan

## Finite elimination

`eliminate_finite` contracts explicit finite tables. Each factor gives its
axes in table order, with the last axis varying fastest. The order of the
factors is the written product order.

The checker evaluates the actual named QIEC operations on every pair of
carrier values. It checks the additive and multiplicative identities,
associativity, additive commutativity, both distributive laws and the
annihilating zero. Carrier values are Booleans, unit values and their
finite products. A failing law returns unequal answers and their exact
arguments in a `FiniteAlgebraLawError`.

With a noncommutative product, eliminating a variable joins the interval
from its first to last participating factor, including intervening
factors. Each resulting factor occupies the original interval's position.
With a commutative product, the planner may join a smaller bucket by
leaving those intervening factors in place. The report records each
step's factor positions and the largest table cardinality. This contract
starts from explicit factors. Deriving those factors from an effectful
program requires a separate justification.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import FiniteAlgebra, FiniteFactor, eliminate_finite

source = """
define add(a : Bool, b : Bool) : Bool !{} =
    if a then
        return true
    else
        return b

define multiply(a : Bool, b : Bool) : Bool !{} =
    if a then
        return b
    else
        return false
"""
module = Compiler(parse(source), module_name="boolean-elimination").qiec_module
assert module is not None
ids = {item.name: item.id for item in module.computations}
algebra = FiniteAlgebra(ids["add"], ids["multiply"], zero=False, one=True)
factors = (
    FiniteFactor(("x",), (True, False)),
    FiniteFactor(("y",), (False, True)),
    FiniteFactor(("x",), (True, True)),
)
answer = eliminate_finite(module, algebra, factors, ("y",), {"x": 2, "y": 2})
assert answer.factor.cells == (False, True)
assert answer.steps[0].positions == (0, 2)
```

::: quivers.qiec.finite_elimination

## Finite Bernoulli estimators

`plan_bernoulli_gradient` checks a named scalar computation that performs
direct, unplated Boolean Bernoulli draws on one nominal `Random` instance.
Each reached static site needs an explicit `enumerate`, `score`, or
`measure` policy. Repeated calls to the same site share its policy and keep their
distinct request addresses. At an adaptive site, the report gives the
conditional probability and its derivative for each preceding history.

The plan enumerates the finite source traces at the supplied point. Its
exact result is the expected return and its directional derivative.
Calling `sample` runs the chosen estimator over that tree. Enumeration
evaluates both branches. The score estimator draws one branch and adds the
return times the directional derivative of that branch's log probability.
The weak derivative, selected by `measure`, draws one primal branch and
evaluates both branches. It adds $p'(V_1 - V_0)$ to the selected
branch's tangent. Both estimators are unbiased in the checked local domain.
Here $p$ is the probability of true and $V_1$ and $V_0$ are the branch values.
The result records visited requests, score draws, weak selectors, and
the ordered launch of each weak branch.

Conditional probabilities must lie strictly between zero and one. The
exact scalar checker also requires comparisons to be stable at the
supplied point. Construction checks a complete-path bound and a step
bound for each replay. With $n$ complete paths it needs $2n - 1$
reference replays and reports their total steps. The sampled estimator
does not truncate rare large score weights.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import Bind, Perform, SiteEstimator, plan_bernoulli_gradient

source = """
instance random : Random

define reward(p : Real) : Real !{random} =
    let chosen <- perform random.sample[Bool](site("chosen"), Bernoulli(p))
    if chosen then
        return 1.0
    else
        return 0.0
"""
module = Compiler(parse(source), module_name="bernoulli-gradient").qiec_module
assert module is not None
owner = next(item for item in module.computations if item.name == "reward")
assert isinstance(owner.body, Bind) and isinstance(owner.body.first, Perform)
site = owner.body.first.request.origin.static_site
policy = SiteEstimator(site=site, estimator="score")
plan = plan_bernoulli_gradient(module, owner.id, (Fraction(1, 3),), (2,), (policy,))
assert (plan.exact.value, plan.exact.tangent) == (Fraction(1, 3), 2)
assert len(plan.sample(seed=42).draws) == 1

weak_policy = SiteEstimator(site=site, estimator="measure")
weak = plan_bernoulli_gradient(module, owner.id, (Fraction(1, 3),), (2,), (weak_policy,))
estimate = weak.sample(seed=42)
assert estimate.tangent == 2
assert len(estimate.selectors) == 1
assert estimate.virtual_branches == ((False,), (True,))
```

::: quivers.qiec.bernoulli_estimators

## Finite handler interchange

A `FiniteHandlerInterchange` names two checked computations whose only
structural difference is the order of two distinct handler installations.
They supply the same initial state to each installation and invoke the
same body. `check_finite_handler_interchange` tests the returned value
on every assignment to the finite inputs.

In the example below, `cells_lr` and `cells_rl` agree on all eight
Boolean input triples. Each cell updates its own component of the
product state, and the body still sequences every read and write.
With a multi-shot choice, `state_first(false)` returns `true` because
the first resumption updates the outer cell before the second reads it.
`choice_first(false)` returns `false` because each resumption of the
inner cell starts from the captured initial state. The certificate
concerns the returned value for this body. Trace observations and
other computations require their own evidence. Pass both statements to
`Compiler.effect_plan(..., interchanges=...)` to report the comparisons
as checked and refuted beside the ordinary discharge plan.

<!-- compile: qiec -->
```qvr
effect Cell
    get : Unit -> Bool
    put : Bool -> Unit

instance left_cell : Cell
instance right_cell : Cell

effect Choice
    pick : Unit -> Bool

instance choice : Choice

handler state for Cell : Bool -> Bool with (current : Bool) [coverage=total, forwards=none, implementation=authored]
    return v =>
        return v
    get resumes 1 =>
        resume(current)
    put(v : Bool) resumes 1 =>
        resume(unit) with (v)

handler some for Choice : Bool -> Bool [coverage=total, forwards=none, implementation=authored]
    return v =>
        return v
    pick resumes omega =>
        let a <- resume(true)
        let b <- resume(false)
        if a then
            return true
        else
            return b

define two_cells(new : Bool) : Bool !{left_cell, right_cell} =
    let old <- perform right_cell.get()
    perform left_cell.put(new)
    let current <- perform left_cell.get()
    perform right_cell.put(current)
    let check <- perform right_cell.get()
    if old then
        return check
    else
        if check then
            return false
        else
            return true

define cells_lr(a : Bool, b : Bool, new : Bool) : Bool !{} =
    handle left_cell with state(a) in
        handle right_cell with state(b) in
            two_cells(new)

define cells_rl(a : Bool, b : Bool, new : Bool) : Bool !{} =
    handle right_cell with state(b) in
        handle left_cell with state(a) in
            two_cells(new)

define cell_choice() : Bool !{left_cell, choice} =
    let positive <- perform choice.pick()
    if positive then
        perform left_cell.put(true)
        return false
    else
        perform left_cell.get()

define state_first(initial : Bool) : Bool !{} =
    handle left_cell with state(initial) in
        handle choice with some in
            cell_choice()

define choice_first(initial : Bool) : Bool !{} =
    handle choice with some in
        handle left_cell with state(initial) in
            cell_choice()
```

::: quivers.qiec.handler_interchange

## Scalar Normal pathwise derivatives

`plan_normal_pathwise` checks a named scalar computation with direct,
unplated `Normal` draws on one nominal `Random` instance. Each reached
static site needs an explicit `reparam` policy. The plan assigns a fresh
standard Normal coordinate to each addressed occurrence, including
repeated calls to the same site.

A location can depend polynomially on earlier draws. The checker
requires a strictly positive, noise-independent scale, noise-independent
comparisons and divisors, and fixed integral exponents. It also checks
that parameter-dependent decisions stay on the same branch near the
supplied point. In this fragment the return and its directional
derivative are polynomials in the base noise with locally bounded
coefficients. Their growth has an integrable Gaussian bound, which
permits differentiation under the expectation.

`plan.exact` gives the expected return and its directional derivative
from exact Normal moments. `plan.at_noise` evaluates the return and
its directional derivative at the supplied base-noise coordinates.
Coordinates follow the order in `plan.visits` and use integers or
fractions. Construction checks limits on source steps, noise
coordinates, and polynomial and moment work.

```python
from quivers.dsl import Compiler, parse
from quivers.qiec import Bind, Perform, SiteEstimator, plan_normal_pathwise

source = """
instance random : Random

define squared(m : Real, s : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(m, s))
    return x * x
"""
module = Compiler(parse(source), module_name="normal-pathwise").qiec_module
assert module is not None
owner = next(item for item in module.computations if item.name == "squared")
assert isinstance(owner.body, Bind) and isinstance(owner.body.first, Perform)
site = owner.body.first.request.origin.static_site
policy = SiteEstimator(site=site, estimator="reparam")
plan = plan_normal_pathwise(module, owner.id, (2, 3), (1, 2), (policy,))
assert (plan.exact.value, plan.exact.tangent) == (13, 16)
action = plan.at_noise((1,))
assert (action.value, action.tangent) == (25, 30)
```

::: quivers.qiec.normal_pathwise

## Mixed Bernoulli and Normal estimators

`plan_mixed_site_gradient` checks both kinds of request in the same
scalar source computation. A Bernoulli site's probability may depend on
earlier Boolean draws and on the inputs. It must stay strictly between
zero and one and be independent of Normal noise. Each Normal site
follows the scalar pathwise rules.

The finite estimator combines the conditional returns and derivatives
linearly, including negative returns. A Bernoulli action accounts for
its own changing probability exactly once. A Normal action
differentiates its location and scale at fixed base noise before taking
the conditional moments. The plan uses finite enumeration to check all
local hypotheses and compute the exact expectation and derivative.
Calling `sample` estimates these quantities by the selected finite
estimator with the Normal moments integrated at the leaves.

`at_noise` instead fixes all Normal coordinates before applying the
finite estimators. Coordinates follow `normal_visits`. Each occurrence
is keyed by its canonical address and the finite choices made before
it. Branches share the coordinates from their common history and use
separate coordinates after they diverge.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import Bind, Perform, SiteEstimator, plan_mixed_site_gradient

source = """
instance random : Random

define signed_reward(p : Real, s : Real) : Real !{random} =
    let x <- perform random.sample[Real](site("x"), Normal(0.0, s))
    let positive <- perform random.sample[Bool](site("positive"), Bernoulli(p))
    if positive then
        return x * x
    else
        return -x * x
"""
module = Compiler(parse(source), module_name="mixed-estimator").qiec_module
assert module is not None
owner = next(item for item in module.computations if item.name == "signed_reward")
first = owner.body
assert isinstance(first, Bind) and isinstance(first.first, Perform)
second = first.then
assert isinstance(second, Bind) and isinstance(second.first, Perform)
normal = SiteEstimator(site=first.first.request.origin.static_site, estimator="reparam")
finite = SiteEstimator(site=second.first.request.origin.static_site, estimator="score")
plan = plan_mixed_site_gradient(
    module, owner.id, (Fraction(3, 4), 2), (1, 1), (normal, finite),
)
assert (plan.exact.value, plan.exact.tangent) == (2, 10)
assert len(plan.normal_visits) == 1
assert plan.at_noise((2,)).exact.tangent == 40
assert len(plan.sample(seed=42).draws) == 1
```

::: quivers.qiec.mixed_site_gradients

## Exact finite marginalization

`marginalize_bernoulli_unit` takes a scalar source computation whose
return is an observation factor. It enumerates the latent Bernoulli
histories and differentiates each prior probability and observation
factor. Each site needs an explicit `enumerate` policy. The factor
on every complete history must be strictly positive. This ensures the
complete-data log scores exist on the checked local support.

For history $h$, write $w_h$ for the joint weight and $Z = \sum_h w_h$
for the marginal observation weight. The result records $Z$ and its
directional derivative, the posterior weights $w_h / Z$ and their
derivatives, and the complete-data log scores $w_h' / w_h$. The
posterior mean of these scores equals $Z' / Z$ by the finite Fisher
identity. The reference computation checks this equality in rational
arithmetic.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import Bind, Perform, SiteEstimator, marginalize_bernoulli_unit

source = """
instance random : Random

define observed_unit(p : Real) : Real !{random} =
    let group <- perform random.sample[Bool](site("group"), Bernoulli(p))
    if group then
        return 3.0
    else
        return 1.0
"""
module = Compiler(parse(source), module_name="finite-marginalization").qiec_module
assert module is not None
entry = next(item for item in module.computations if item.name == "observed_unit")
assert isinstance(entry.body, Bind) and isinstance(entry.body.first, Perform)
policy = SiteEstimator(site=entry.body.first.request.origin.static_site, estimator="enumerate")
fit = marginalize_bernoulli_unit(module, entry.id, (Fraction(1, 4),), (1,), (policy,))
assert (fit.normalizer.value, fit.normalizer.tangent) == (Fraction(3, 2), 2)
assert fit.marginal_score == Fraction(4, 3)
assert [outcome.posterior.value for outcome in fit.outcomes] == [Fraction(1, 2)] * 2
```

::: quivers.qiec.finite_marginalization

## Finite importance sampling

`plan_bernoulli_importance` checks a target and a proposal at two
separately supplied points for the same scalar source. The target
supplies the prior probability times the positive observation factor.
The proposal uses only its prior probabilities. These probabilities
are held fixed in the target direction. Both evaluations use explicit
`enumerate` policies to check every complete history.

The coverage check compares the ordered addresses and Boolean outcomes
in each history. Distinct addresses can give distinct histories even
when the Boolean outcomes are identical. A missing history raises
`ImportanceCoverageError` with that original history and its positive
target weight.

If $w_h$ is the joint target weight and $q_h$ is the proposal
probability, a proposal draw returns $w_h / q_h$ and $w_h' / q_h$.
Their means are the raw evidence and its directional derivative.
Use the separate finite marginalization result for normalized
posterior weights and the marginal log score.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import Bind, Perform, SiteEstimator, plan_bernoulli_importance

source = """
instance random : Random

define observed_unit(p : Real) : Real !{random} =
    let group <- perform random.sample[Bool](site("group"), Bernoulli(p))
    if group then
        return 3.0
    else
        return 1.0
"""
module = Compiler(parse(source), module_name="finite-importance").qiec_module
assert module is not None
entry = next(item for item in module.computations if item.name == "observed_unit")
assert isinstance(entry.body, Bind) and isinstance(entry.body.first, Perform)
policy = SiteEstimator(site=entry.body.first.request.origin.static_site, estimator="enumerate")
plan = plan_bernoulli_importance(
    module, entry.id, (Fraction(1, 4),), (1,), (policy,),
    proposal_arguments=(Fraction(1, 2),), proposal_policies=(policy,),
)
assert (plan.exact.value, plan.exact.tangent) == (Fraction(3, 2), 2)
assert [outcome.estimate.tangent for outcome in plan.outcomes] == [-2, 6]
assert plan.sample(seed=12).estimate.value == Fraction(3, 2)
```

::: quivers.qiec.finite_importance

## Finite observation and intervention

`compare_bernoulli_do_and_condition` compares the effect of observing
one addressed Boolean draw and fixing that draw to the same value.
Both queries use a checked finite source model in which each draw
is a separate probabilistic mechanism. The treatment and response
addresses must occur on every complete history. The source returns
a positive observation factor on each history.

Conditioning on the treatment retains its probability in the source
weights. The intervention replaces that kernel by a point mass at the
chosen value. The returned interventional weights omit the original
probability of the treated draw and its derivative. The other kernels
use their source probabilities at each history. Both queries include
the source's observation factor and normalize with their own evidence.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    AddressedBernoulliEvent,
    Bind,
    Perform,
    SiteEstimator,
    compare_bernoulli_do_and_condition,
)

source = """
instance random : Random

define propensity(c : Bool, p : Real) : Real !{} =
    if c then
        return 0.75
    else
        return p

define model(p : Real) : Real !{random} =
    let c <- perform random.sample[Bool](site("confounder"), Bernoulli(0.5))
    let q <- propensity(c, p)
    let t <- perform random.sample[Bool](site("treatment"), Bernoulli(q))
    return 1.0
"""
module = Compiler(parse(source), module_name="finite-intervention").qiec_module
assert module is not None
entry = next(item for item in module.computations if item.name == "model")
policies = []
term = entry.body
while isinstance(term, Bind):
    if isinstance(term.first, Perform):
        site = term.first.request.origin.static_site
        policies.append(SiteEstimator(site=site, estimator="enumerate"))
    term = term.then
confounder, treatment = policies
comparison = compare_bernoulli_do_and_condition(
    module, entry.id, (Fraction(1, 4),), (1,), tuple(policies),
    treatment=AddressedBernoulliEvent((str(treatment.site), (), ()), True),
    outcome=AddressedBernoulliEvent((str(confounder.site), (), ()), True),
)
assert (comparison.conditioned.value, comparison.conditioned.tangent) == (
    Fraction(3, 4), Fraction(-3, 4),
)
assert (comparison.intervened.value, comparison.intervened.tangent) == (
    Fraction(1, 2), 0,
)
```

::: quivers.qiec.finite_intervention

## Finite multinomial resampling

`plan_multinomial_resampling` checks one stage with an ordered input
table of particle values and positive weights. Each value and weight
has a directional derivative. Each offspring slot independently
chooses an original position according to its normalized weight.
Equal particle values remain distinct positions.

The native scalar objective receives the selected values in slot
order, followed by its direct arguments. The derivative estimate
adds the objective times the sum of the selected log probability
derivatives to the objective's derivative at fixed ancestry. Repeated
positions contribute a score for each selecting slot. The plan
enumerates all ordered assignments up to its bound and reports
the exact expectation and derivative for this finite-step objective.

The expectation depends on the declared objective. For instance,
an unnormalized estimate of 1 or 3 with equal probability has mean 2.
Its expected logarithm is $\log\sqrt{3}$, which is below $\log 2$.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import (
    Bind,
    Perform,
    ResamplingParticle,
    SiteEstimator,
    marginalize_bernoulli_unit,
    plan_multinomial_resampling,
)

source = """
instance random : Random

define observed(p : Real) : Real !{random} =
    let chosen <- perform random.sample[Bool](site("chosen"), Bernoulli(p))
    if chosen then
        return 3.0
    else
        return 1.0

define objective(first : Real, second : Real) : Real !{} =
    return first + 2.0 * second
"""
module = Compiler(parse(source), module_name="multinomial-resampling").qiec_module
assert module is not None
entry = next(item for item in module.computations if item.name == "observed")
assert isinstance(entry.body, Bind) and isinstance(entry.body.first, Perform)
policy = SiteEstimator(site=entry.body.first.request.origin.static_site, estimator="enumerate")
fit = marginalize_bernoulli_unit(module, entry.id, (Fraction(1, 4),), (1,), (policy,))
particles = tuple(ResamplingParticle(x.observation, x.joint) for x in fit.outcomes)
plan = plan_multinomial_resampling(module, "objective", particles)
assert len(plan.outcomes) == 4
assert (plan.exact.value, plan.exact.tangent) == (6, 8)
assert len(plan.sample(seed=13).positions) == 2
```

::: quivers.qiec.finite_resampling

## Finite unit subsampling

`plan_unit_subsampling` checks a fixed design for an additive
finite population total. The design lists complete subsets of
whole unit positions and their exact probabilities. Every unit
needs positive inclusion probability. The estimator divides
each selected unit's value and directional derivative by that
unit's inclusion probability, then sums the selected contributions.
Its expectation gives the full total and its derivative.

Use `UnitSubset.from_plate_rows` for the existing plate index format
when each row is a whole unit. Uniform samples of a fixed size
recover the existing size-over-sample-size coefficient. A design
with unequal inclusion probabilities uses a separate coefficient
for each unit.

The log likelihood of positive observation factors $L_i$ has an
additive directional score $\sum_i L_i' / L_i$. Applying the
inclusion weights to those scores estimates that separate sum.
The log of a sampled raw total instead depends on the distribution
of that whole estimate.

```python
from fractions import Fraction
import torch
from quivers.dsl import Compiler, parse
from quivers.qiec import UnitSubset, exact_jvp, plan_unit_subsampling

source = """
define unit(x : Real, theta : Real) : Real !{} =
    return x * theta
"""
module = Compiler(parse(source), module_name="unit-subsampling").qiec_module
assert module is not None
units = tuple(exact_jvp(module, "unit", (x, 2), (0, 1)) for x in (1, 3))
design = tuple(
    UnitSubset.from_plate_rows("Units", torch.tensor([i]), Fraction(1, 2))
    for i in range(2)
)
plan = plan_unit_subsampling(units, design)
assert plan.inclusion_probabilities == (Fraction(1, 2), Fraction(1, 2))
assert (plan.exact.value, plan.exact.tangent) == (8, 4)
assert [action.estimate.value for action in plan.outcomes] == [4, 12]
```

::: quivers.qiec.finite_subsampling

## Finite-step Metropolis derivatives

`finite_metropolis_jvp` takes an initial law, positive target weights,
and a proposal kernel on the same finite state space. Its inputs are
values and tangents of differentiable curves. The initial law and each
proposal row are normalized, with zero total tangent. The Metropolis
transition from $i$ to $j\ne i$ has mass
$P_{ij}=\min(q_{ij},\,w_jq_{ji}/w_i)$. The diagonal collects the remaining
mass, including rejected proposals. [Hastings (1970, §2.2)](https://doi.org/10.1093/biomet/57.1.97)

At a strict minimum the active expression gives the tangent. At an
equality, the two expressions must have the same directional slope for
this comparison to be admitted. If the initial row is $u_0$, the
recurrence propagates

$$
u_{k+1}=u_kP,
\qquad
\dot u_{k+1}=\dot u_k P+u_k\dot P.
$$

The result is the law after the requested number of steps, with
separate contributions from the initial law and the transition law. For $n$ states
and $k$ positive steps, the recurrence uses $O((1+k)n^2)$ rational
arithmetic operations. Rational coefficient bit lengths contribute
separately to runtime.

<!-- python: run -->
```python
from fractions import Fraction

from quivers.dsl import loads
from quivers.qiec import ExactJvp, exact_jvp, finite_metropolis_jvp

module = loads("""
define target_weight(t : Real) : Real !{} =
    return t
""").qiec
assert module is not None

def constant(value):
    return ExactJvp(Fraction(value), Fraction(0))

second_weight = exact_jvp(module, "target_weight", (Fraction(1, 4),), (1,))
proposal = (
    (constant(Fraction(3, 4)), constant(Fraction(1, 4))),
    (constant(Fraction(1, 2)), constant(Fraction(1, 2))),
)
law = finite_metropolis_jvp(
    (constant(1), constant(0)), (constant(1), second_weight), proposal, steps=2
)
assert law[1] == ExactJvp(Fraction(11, 64), Fraction(5, 8))
```

::: quivers.qiec.finite_metropolis

## Bias of finite gradient clipping

`audit_clipped_bernoulli_gradient` checks a finite Bernoulli source
with explicit estimator policies, then caps each realized final
directional estimate at the supplied magnitude. It enumerates the
estimator's own score choices and weak selectors, including choices
made in both branches of a weak estimator. Each outcome retains its
source addresses, selector choices, and exact probability.

The signed bias is the mean clipped tangent minus the exact source
tangent. The report belongs to the checked source, parameter point,
direction, policies, and clipping limit. A different point or direction
requires its own calculation.

```python
from fractions import Fraction
from quivers.dsl import Compiler, parse
from quivers.qiec import Bind, Perform, SiteEstimator, audit_clipped_bernoulli_gradient

source = """
instance random : Random

define reward(p : Real) : Real !{random} =
    let chosen <- perform random.sample[Bool](site("chosen"), Bernoulli(p))
    if chosen then
        return 1.0
    else
        return 0.0
"""
module = Compiler(parse(source), module_name="clipped-gradient").qiec_module
assert module is not None
entry = next(item for item in module.computations if item.name == "reward")
assert isinstance(entry.body, Bind) and isinstance(entry.body.first, Perform)
policy = SiteEstimator(site=entry.body.first.request.origin.static_site, estimator="score")
audit = audit_clipped_bernoulli_gradient(
    module, entry.id, (Fraction(1, 4),), (1,), (policy,), clip_limit=2,
)
assert audit.source.exact.tangent == 1
assert audit.mean.tangent == Fraction(1, 2)
assert audit.signed_bias == Fraction(-1, 2)
```

::: quivers.qiec.finite_clipping

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

A `RationalDerivativeDeclaration` identifies its primal, forward action,
and reverse components by computation ID. It can be serialized
independently of the executable module. Passing it to `exact_jvp` or
`exact_gradient` checks the identities against the supplied module before
executing the named action. The checker returns the primal guard and
any separating input. Each application keeps the primal evaluation's
ordinary domain and resource limits.

<!-- python: run -->
```python
from fractions import Fraction

from quivers.dsl import loads
from quivers.qiec import RationalDerivativeDeclaration, dumps, exact_jvp
from quivers.qiec import loads as read_qiec

module = loads("""
define mean_term(x : Real) : Real !{} =
    return 1.0 / x

define mean_direction(x : Real, step : Real) : Real !{} =
    return -step / (x * x)
""").qiec
assert module is not None
ids = {computation.name: computation.id for computation in module.computations}
rule = RationalDerivativeDeclaration(primal=ids["mean_term"], jvp=ids["mean_direction"])
decoded = read_qiec(dumps(rule))
assert isinstance(decoded, RationalDerivativeDeclaration)
result = exact_jvp(module, "mean_term", (2,), (3,), rule=decoded)
assert (result.value, result.tangent) == (Fraction(1, 2), Fraction(-3, 4))
```

::: quivers.qiec.derivative_rule_schema.RationalDerivativeDeclaration

::: quivers.qiec.derivative_rules
