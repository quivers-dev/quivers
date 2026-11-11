# Generated computation graphs

Several high-level QVR declarations generate checked QIEC definitions. Their
surface remains the convenient way to write the model, while the generated
graph supplies one execution, inspection, and diagnostic boundary. Elaboration may add families,
instances, [handlers](effects-and-handlers.md#handler-declarations), and helper computations, but every generated node is
typed and remains visible to tooling.

## What each declaration generates

Generated computations are explicit expansions of convenient source
forms. A parser declaration, for instance, implies choices among
rules and accumulation of weights; its generated graph spells out
those operations. You normally call the [public entry](execution-and-tooling.md#list-entry-points) and inspect
the helpers when debugging types, effects, or backend support.
The expansion retains source information so a generated failure can
still point back to the declaration you wrote.


| Source declaration | Principal QIEC graph | Runtime requirement |
| --- | --- | --- |
| `program` | one computation over canonical `random` and `score`; helpers for [marginalization](probabilistic-programs.md#exact-marginalization) and scans | core reference handlers; optional model attachments |
| `deduction` | closed item family, recursive derivation, `Search`, `Weight[K]`, `Param`, collecting handlers, run entry | core search, semiring, and parameter providers |
| `define p = parser(...)` | finite item family and a goal-directed [deduction](../../guides/deduction.md#the-framework) instantiated from schemas and bundles | same search runtime as a deduction |
| `signature` | opaque term family indexed by declared sorts | structural term values at the host boundary |
| `encoder` | typed `Compute[Term, Code]` request and named computation | attached PyTorch [encoder](../../guides/structural-compression-signatures-and-encoders.md#encoder-blocks) |
| `decoder` | typed `Compute[Code, Term]` and negative-log-likelihood computations | attached PyTorch [decoder](../../guides/structural-compression-decoders-and-losses.md#decoder-blocks) |
| `loss` | named computation calling its encoder/decoder dependencies | attachments reachable from the loss |

`Program.entry_points()` lists the public computations a compiled module can
invoke. Generated implementation helpers may also be inspectable in the
checked module; callers should select the documented entry rather than depend
on a helper's generated name.

## Deduction declarations

A weighted deduction system becomes a search computation whose weights are
interpreted in the declared semiring. For a span grammar, elaboration creates:

1. an item family with constructors for atoms, applied symbols, and positions;
2. a recursive derive computation that chooses axioms or applicable rules,
   recurring on smaller spans (or smaller yields, for items with several spans),
   with a closure computation for unary rules and, when `depth` is declared, a
   depth computation that bounds each conclusion's category;
3. structural equality and display computations for items;
4. a `Search` instance for alternatives;
5. a `Weight[K]` instance for rule and lexical contributions;
6. a `Param` instance for learned weights; and
7. a run computation that handles search and accumulation.

Shared pattern variables are checked by structural equality after recursive
premises return. Failed matches perform an empty choice. A learned lexical or
rule weight is fetched from `Param` under the same stable name used by the
agenda engine, so the classic chart and reference computation can share a
parameter store. Under `normalize=lhs`, each left-hand side of several
productions with a learnable member gains a computation `<D>__lhs<i>` that reads
its productions' free weights from `Param` and answers their log-sum-exp,
computed about the largest; every rule and lexicon entry adds its own free
weight less that normalizer, its entry of the group's `log_softmax`; see
[production probabilities](../../guides/deduction.md#production-probabilities-per-left-hand-side).
When a program binds the grammar's production probabilities, the deduction is
elaborated a second time: `<D>__weighted_axiom`, `<D>__weighted_derive`, and
the rest of the computations that read weights take a table of the productions'
log-probabilities as their last argument and read each weight from it, and
`<D>__weighted_run` is the entry the program calls. A program that parses a
corpus gains a helper `<program>__<let>_corpus` that calls the entry on each
sentence and multiplies the answers in the semiring.

For `LogProb`, the result is a `LogWeight`; for `Boolean`, it is a `Bool`; for
`Counting`, it is an `Int`. `run_deduction` returns the goal's inside value and
may accept tokens, explicit axioms, axiom weights, and parameter values,
depending on the deduction's shape.

See the [weighted deduction guide](../../guides/deduction.md) and the
[parsing tutorial](../../tutorials/qvr/11-parsing-and-search.md).

## Schema-backed parsers

`parser(rules=[...], terminal=Token, start=S, depth=n)` first instantiates
pattern-polymorphic schemas over the finite category inventory induced by the
declared free residuated object. It then lowers that concrete rule system to
the same search-and-weight graph as a deduction.

The classic `ChartParser` and the QIEC entry share lexical and rule weights and
compute the same bounded inside score. This dual view is useful during model
development: the classic object exposes batched PyTorch fitting, while the
checked computation exposes effect handling, entry-point execution, stable
serialization, and target diagnostics.

A target that lacks the search runtime reports
`qiec:capability:search:<entry>`. The parser is not dropped when another
program in the module calls it; the reachable call graph determines whether a
target can emit the model.

## Structural signatures and autoencoders

A checked module can describe the type of a neural computation without
containing the neural module itself. Its attachment descriptor is like
a typed reference to an implementation; the runtime provider supplies
the actual PyTorch object. Saving the checked description thus
differs from saving learned parameters or serializing a runnable
Python module. Both pieces must be supplied at the execution boundary.


A `signature` defines typed host values, including data leaves, recursive
constructors, binders, and de Bruijn variables. It lowers to an opaque family
because the stable kernel tracks the term's sort but does not serialize the
process-local Python object.

An encoder or decoder then becomes a `Compute` instance and a named
computation. For the shipped term autoencoder:

```text
reconstruct(term)
  -> Enc.apply(term)
  -> Dec__nll.apply((term, code))
  -> LogWeight
```

`Program.run("reconstruct", term)` attaches the compiled `torch.nn.Module`
objects, validates the host values against the checked input and output types,
and preserves the autograd graph. The encoder and decoder are registered under
stable source-derived module names, so `parameters()`, `named_parameters()`,
and `state_dict()` include them.

The stable serialized module contains attachment descriptors, not PyTorch
modules or callables. A process that deserializes it must attach compatible
providers before invoking the affected entry. A transpile target without such
a provider refuses it under `qiec:capability:neural-attachment:<entry>`.

See the [structural autoencoder tutorial](../../tutorials/qvr/12-structural-autoencoders.md)
and the [Term Autoencoder](../../examples/term-autoencoder.md) example.

## Programs that call generated entries

The surfaces compose through ordinary calls. A program can call a deduction:

<!-- compile: qiec -->
```qvr
object Term : FinSet 16
object Weight : Real 1

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "dog" : N = dog #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]

program fit : Term -> Weight
    let chart = parse(AB, sentence)
    score evidence = chart.goal_weight()
    return evidence

export fit
```

The parser result is a source convenience. Elaboration replaces the
`goal_weight()` access with a call to the deduction's checked run computation
and a score request. The program's target-capability set thus includes search,
weight accumulation, parameter lookup when weights are learned, and the
ordinary probabilistic effects of its other steps.

## Inspect the generated graph

Use these surfaces at increasing levels of detail:

```bash
qvr run model.qvr --list
qvr repl model.qvr
# then: :info entry
#       :effects entry
#       :dump entry --json
#       :graph program --mermaid
```

In Python, `program.entry_points()` gives the public invocation surface and
`program.qiec` gives the checked module. Invoke the entries returned by
`entry_points()`; use the internal helpers in `program.qiec` to inspect
how those entries were compiled.

## Capability failures are part of the contract

A generated computation can introduce a requirement not apparent from one
surface line. For instance, `parser(...)` introduces search, and a loss over a
decoder introduces neural attachments. Quivers attributes the diagnostic to
the reachable computation and preserves its source range. Use
`qvr check --target TARGET` or set the LSP target while editing, so core QIEC
requirements appear while editing. Then run the transpiler to check
restrictions imposed by the target renderer.
