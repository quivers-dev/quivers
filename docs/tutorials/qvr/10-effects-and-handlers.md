# 10. Effects and handlers

An [algebraic effect](https://en.wikipedia.org/wiki/Effect_system) is an operation that a computation requests without saying how it is carried out, and a *handler* is the code that answers the request and resumes the computation [@plotkin2003algebraic; @plotkin2009handlers; @bauer2015programming]. Pyro builds its inference on the same separation: one model, run under different handlers, draws a fresh value, conditions on an observation, or replays a recorded value ([Pyro's effect handlers](https://pyro.ai/examples/effect_handlers.html)) [@pyroEffects]. This chapter declares an operation `shrink`, which asks for an adjusted value, and gives it an interpretation with a handler.

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

The row of `robust_request` keeps `robust`, and that of `robustify` is empty. `qvr run` can thus run `robustify` on the reference machine, the library's interpreter for the core language QIEC ([chapter 9](09-indexed-data.md#4-run-the-summary)), with no handler from outside:

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

An *authored* handler has its clause bodies written in QVR, so it can be inspected, serialized, and emitted to a transpile target that supports its operations. A *foreign* handler declares only its type, and a *provider*, an object registered with the runtime, gives its behaviour when the module runs. This is the role of a [foreign function interface](https://en.wikipedia.org/wiki/Foreign_function_interface). Use a foreign handler for a sampler, a PyTorch module, a database, or another resource that lives only in the running Python process. Write `[implementation=foreign]` and omit the clause bodies. Omitting them without that option is an error.

A process that loads the serialized module must install a provider for each foreign handler. The runtime configuration selects the provider by name, and the Python callables themselves are never serialized ([foreign handlers and providers](../../reference/qvr/effects-and-handlers.md#foreign-handlers-and-providers)).

## Next

[Chapter 11](11-parsing-and-search.md) uses multi-shot handlers to search over parses. The [effect reference](../../reference/qvr/effects-and-handlers.md) lists the built-in effects, rows, options, and failure modes.
