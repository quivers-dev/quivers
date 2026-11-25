# 10. Effects and handlers

An [algebraic effect](https://en.wikipedia.org/wiki/Effect_system) is an operation that a computation requests without saying how it is carried out, and a *handler* is the code that answers the request and resumes the computation [@plotkin2003algebraic; @plotkin2009handlers; @bauer2015programming]. Pyro builds its inference on the same separation: one model, run under different handlers, draws a fresh value, conditions on an observation, or replays a recorded value ([Pyro's effect handlers](https://pyro.ai/examples/effect_handlers.html)) [@pyroEffects]. Declare an operation `shrink`, which asks for an adjusted value, and give it an interpretation with a handler.

When a computation runs `perform robust.shrink(8.0)`, it pauses. The rest of the computation, waiting for the answer, is the [continuation](https://en.wikipedia.org/wiki/Continuation). A handler that answers `resume(4.0)` passes `4.0` to the continuation as the result of `perform`. QVR uses three notions:

- An *effect* (interface) lists operations and their types.
- An *effect instance* is one named occurrence of an effect, so two instances of one interface can be handled differently.
- An *effect row* lists the instances a computation may still perform. Handling an instance completely removes it from the row.

## 1. Declare the request

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

define robust_request(x : Real) : Real !{robust | rho lacks robust} =
    let adjusted <- perform robust.shrink(x)
    return adjusted
```

`Robust` is the effect and `robust` an instance of it. The row `!{robust | rho lacks robust}` says that `robust_request` performs `robust` and passes through whatever other instances the caller's row `rho` holds. A row with such a variable tail is *open*, and abstracting over the tail is [row polymorphism](https://en.wikipedia.org/wiki/Row_polymorphism) [@leijen2014koka]. The constraint `lacks robust` keeps `rho` from naming `robust` a second time.

## 2. Write an authored handler

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

handler half_weight for Robust : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shrink(x : Real) resumes 1 =>
        resume(0.5 * x)

define robustify(x : Real) : Real !{} =
    handle robust with half_weight in
        let adjusted <- perform robust.shrink(x)
        return adjusted
```

The handler's options say that it has a clause for every operation of `Robust` (`coverage=total`), that it passes no request on to an outer handler (`forwards=none`), and that its clauses are written in QVR (`implementation=authored`). The `return` clause passes the final value through unchanged. The clause for `shrink` resumes the continuation with half its argument, and `resumes 1` declares that it resumes exactly once. Since the handler covers `robust` completely, the row of `robustify` is empty (`!{}`), and the computation continues under the same handler after each `resume`, as a *deep handler* does [@kammar2013handlers]. The [handler options](../../reference/qvr/effects-and-handlers.md#handler-options) list the alternatives.

## 3. Check the row

The complete source is [`authored-handler.qvr`](source/authored-handler.qvr). Ask the REPL for the rows of both computations:

```text
$ qvr repl docs/tutorials/qvr/source/authored-handler.qvr
qvr> :effects robust_request
qvr> :effects robustify
```

The row of `robust_request` keeps `robust`, and that of `robustify` is empty. `qvr run` can thus run `robustify` on the reference machine, the library's interpreter for the core language, the [Quivers Indexed Effect Core](../../reference/qvr/index.md) (QIEC) ([chapter 9](09-indexed-data.md#4-run-the-summary)), with no handler from outside:

```bash
qvr run docs/tutorials/qvr/source/authored-handler.qvr robustify 8.0 --trace
```

From Python, with the trace of the run:

```python
from quivers.dsl import load

effects = load("docs/tutorials/qvr/source/authored-handler.qvr")
run = effects.run("robustify", 8.0)
assert run.value == 4.0

events = [event.event for event in run.result.trace]
assert "operation.requested" in events
assert "operation.handled" in events
assert "resumption.invoked" in events
print(run.value, events)
```

```text
4.0 ['run.started', 'runtime.attached', 'argument.validated', 'handler.entered', 'operation.requested', 'operation.handled', 'clause.entered', 'resumption.invoked', 'handler.returned', 'clause.answered', 'run.completed']
```

The trace records the request, the dispatch to the handler, the entry into the clause, the resumption, and the final answer. Each event also carries the source location and the identifiers of the declarations involved.

## 4. Choose a resumption grade

A *resumption grade* states how many times a clause may resume its continuation. The grades are the usage annotations of [substructural type systems](https://en.wikipedia.org/wiki/Substructural_type_system) and quantitative type theory [@atkey2018syntax]:

| Grade | Resumptions | Typical use |
|---|---|---|
| `0` | never | aborting, or replacing the rest of the computation |
| `aff` (affine) | at most once | a clause that may discard the continuation |
| `1` (linear) | exactly once | answering a request, as `half_weight` does, or a state update |
| `omega` (unrestricted) | any number | search that resumes once for each alternative |

Grades `0`, `aff`, and `1` make the continuation one-shot. `omega` makes it multi-shot. Declare the most restrictive grade that the policy allows. The reference machine counts resumptions at run time, so a clause declared `1` that resumes twice fails. A multi-shot resumption also needs a context that can be copied, and an accumulator such as the chart of a generated parser ([chapter 11](11-parsing-and-search.md)) is copied for each resumption.

## 5. Decide between authored and foreign handlers

An *authored* handler has its clause bodies written in QVR, so it can be inspected, serialized, and emitted to a transpile target that supports its operations. A *foreign* handler declares only its type, and a *provider*, an object registered with the runtime, gives its behavior when the module runs. This is the role of a [foreign function interface](https://en.wikipedia.org/wiki/Foreign_function_interface). Use a foreign handler for a sampler, a PyTorch module, a database, or another resource that lives only in the running Python process. Write `[implementation=foreign]` and omit the clause bodies. Omitting them without that option is an error.

A process that loads the serialized module must install a provider for each foreign handler. The runtime configuration selects the provider by name, and the Python callables themselves are never serialized ([foreign handlers and providers](../../reference/qvr/effects-and-handlers.md#foreign-handlers-and-providers)).

## 6. Carry state in the handler

A handler's *parameters* carry values from one request to the next [@leijen2017type]. They are typed binders written after the answer type. `handle` gives their first values, every clause reads them, and `resume(v) with (e)` gives the values they have while the continuation runs. The module below is a small model of discourse reference in the style of effect-driven semantics. A sentence may introduce a referent (`introduce`), a definite description reads the latest one (`latest`), and a description whose referent is not positive has no value (`undefined`):

```python
from quivers.dsl import loads

DISCOURSE = """
effect Store
    latest : Unit -> Real
    introduce : Real -> Unit

effect Failure
    undefined : Unit -> Real

instance referents : Store
instance presupposition : Failure

handler remembering for Store : Real -> Real with (r : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    latest resumes 1 =>
        resume(r)
    introduce(v : Real) resumes 1 =>
        resume() with (v)

handler floored for Failure : Real -> Real with (floor : Real) [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    undefined resumes 0 =>
        return floor

define definite() : Real !{presupposition, referents} =
    let x <- perform referents.latest()
    let checked <- positive(x)
    return checked

define positive(x : Real) : Real !{presupposition} =
    if x > 0.0 then
        return x
    else
        let y <- perform presupposition.undefined()
        return y

define discourse(a : Real) : Real !{presupposition, referents} =
    perform referents.introduce(2.0 * a)
    let d <- definite()
    return d + a

define judged(a : Real) : Real !{} =
    handle presupposition with floored(0.0 - 1.0) in
        handle referents with remembering(0.0) in
            let m <- discourse(a)
            return m

object Trial : FinSet 4

program judgments : Trial -> Trial
    sample a <- Normal(loc=0.0, scale=1.0)
    let m <- judged(a)
    observe y : Trial <- Normal(loc=m, scale=1.0)
    return y

export judgments
"""
model = loads(DISCOURSE)
assert model.run("judged", 0.5).value == 1.5
assert model.run("judged", -0.5).value == -1.0
```

`remembering` starts with the referent `0.0`, and its clause for `introduce` resumes with the new one in its place. `floored` discards the rest of the computation when a description has no value and answers its parameter `floor` in place of it. So `judged(0.5)` introduces `1.0`, reads it back, and returns `1.5`, while `judged(-0.5)` returns the floor.

## 7. Fit a program whose meaning is stated with effects

The program `judgments` calls `judged`, which installs handlers, and the compiled runtime and Stan run none. Both read `judged` with its handlers [*discharged*](../../reference/qvr/effects-and-handlers.md#discharged-handlers): each request is replaced by the answer of its clause, and the request under the `if` by a choice between the rest of the computation and the floor. The program thus has gradients, and it transpiles to Stan with no handler in it:

```python
import torch
from quivers.dsl.parser import parse
from quivers.inference import MCMC, NUTSKernel
from quivers.transpile import transpile

y = torch.tensor([1.4, 0.9, 1.8, 1.1])
fit = MCMC(NUTSKernel(max_tree_depth=4), num_warmup=50, num_samples=50, seed=0).run(
    model.morphism, torch.zeros(1, 1), {"y": y}
)
assert fit.samples["a"].isfinite().all()

stan = transpile(parse(DISCOURSE), target="stan").decode()
assert "?" in stan and "handle" not in stan
```

The order of the handlers is part of the analysis, since each request is answered by the innermost handler of its instance and a clause runs outside its own handler. [Which calls are fit](../../reference/qvr/probabilistic-programs.md#which-calls-are-fit) lists the calls that are discharged and the ones that are refused.

## Next

[Chapter 11](11-parsing-and-search.md) uses multi-shot handlers to search over parses. The [effect reference](../../reference/qvr/effects-and-handlers.md) lists the built-in effects, rows, options, and failure modes.
