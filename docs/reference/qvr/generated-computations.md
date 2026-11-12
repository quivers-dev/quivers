# Generated computation graphs

Several QVR declarations elaborate to QIEC definitions that the declaration does not spell out.
Elaboration may add families, instances, [handlers](effects-and-handlers.md#handler-declarations),
and helper computations, and every generated definition is typed, visible to the tooling, and
carries the source range of the declaration it came from, so a diagnostic in generated code points
to the source line.

## What each declaration generates

| Source declaration | Principal QIEC graph | Runtime requirement |
| --- | --- | --- |
| `program` | one computation over the built-in `random` and `score` instances, with helpers for [marginalization](probabilistic-programs.md#exact-marginalization) and scans | the core handlers, and any model attachments |
| `deduction` | a family of items, a recursive derivation, `Search`, `Weight[K]`, `Param`, handlers that collect the weights, and a run entry | the core search, semiring, and parameter providers |
| `define p = parser(...)` | a finite family of items and a [deduction](../../guides/deduction.md#the-framework) instantiated from rule schemata and bundles, searched from the goal | the search runtime of a deduction |
| `signature` | an opaque family of terms indexed by the declared sorts | structural term values held in the Python process |
| `encoder` | a typed `Compute[Term, Code]` request and a named computation | an attached PyTorch [encoder](../../guides/structural-compression-signatures-and-encoders.md#encoder-blocks) |
| `decoder` | a typed `Compute[Code, Term]` request and computations for the negative log-likelihood | an attached PyTorch [decoder](../../guides/structural-compression-decoders-and-losses.md#decoder-blocks) |
| `loss` | a named computation that calls its encoder and decoder | the attachments the loss reaches |

`Program.entry_points()` lists the public [entry points](execution-and-tooling.md#list-entry-points)
of a compiled module. The generated helpers are visible in the checked module, but their names are
not part of the interface, and a caller selects a documented entry.

## Deduction declarations

A *weighted deduction system* is a set of inference rules over items, each derivation carrying a
weight combined in a semiring [@shieber1995principles; @goodman1999semiring; @nederhof2003weighted].
A `deduction` elaborates to a search computation whose weights are interpreted in the declared
semiring. For a span grammar, elaboration creates:

1. a family of items with constructors for atoms, applied symbols, and positions;
2. a recursive derivation that chooses an axiom or an applicable rule and recurs on smaller spans
   (on smaller yields, for items with several spans), with a closure computation for unary rules
   and, when `depth` is declared, a computation that bounds the depth of each conclusion's category;
3. computations for structural equality and display of items;
4. a `Search` instance for the alternatives;
5. a `Weight[K]` instance for the rule and lexical weights;
6. a `Param` instance for learned weights; and
7. a run computation that handles search and accumulation.

Pattern variables shared between premises are checked by structural equality after the recursive
premises return, and a failed match performs an empty choice. A learned lexical or rule weight is
read from `Param` under the stable name that the compiled runtime's agenda-based chart parser in
`quivers.stochastic` uses [@eisner2005compiling], so the two share one parameter store.

Under `normalize=lhs`, the rules that share a left-hand side are normalized as a group
([production probabilities](../../guides/deduction.md#production-probabilities-per-left-hand-side)).
Each left-hand side with several productions, at least one of them learnable, gains a computation
`<D>__lhs<i>`. It reads the free weights of those productions from `Param` and returns their
log-sum-exp, computed relative to the largest. Each rule and lexicon entry then adds its own free
weight minus that normalizer, which is its entry of the group's `log_softmax`.

When a program binds the grammar's production probabilities, the deduction is elaborated a second
time. `<D>__weighted_axiom`, `<D>__weighted_derive`, and the other computations that read weights
take a table of the productions' log probabilities as their last argument and read each weight from
it, and `<D>__weighted_run` is the entry the program calls. A program that parses a corpus gains a
helper `<program>__<let>_corpus` that calls the entry on each sentence and multiplies the answers in
the semiring.

The result type follows the semiring: a `LogWeight` for `LogProb`, a `Bool` for `Boolean`, and an
`Int` for `Counting`. `run_deduction` returns the goal's inside weight, the semiring sum over all
derivations of the goal of the product of their rule weights
([inside algorithm](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm))
[@goodman1999semiring]. Depending on the deduction's shape, it accepts tokens, explicit axioms, axiom
weights, and parameter values.

The [weighted deduction guide](../../guides/deduction.md) and the
[parsing tutorial](../../tutorials/qvr/11-parsing-and-search.md) build deductions step by step.

## Parsers from rule schemata {#schema-backed-parsers}

`parser(rules=[...], terminal=Token, start=S, depth=n)` builds a parser from *rule schemata*, rule
templates with pattern variables [@shieber1995principles]. It instantiates the schemata over the
finite set of syntactic categories built from the atoms of the declared `FreeResiduated` object with
its connectives (by default the slashes `/` and `\`), nested up to depth `n`, as in the
[Lambek calculus](https://ncatlab.org/nlab/show/Lambek+calculus)
[@lambek1958mathematics; @moortgat1997categorial]. It then lowers the resulting rules to the same
graph of search and weights as a deduction.

The compiled runtime's `ChartParser` and the QIEC entry share their lexical and rule weights and
compute the same inside weight, bounded by `depth`. The `ChartParser` fits by batched PyTorch
optimization, and the checked computation gives effect handling, entry point execution, stable
serialization, and target diagnostics.

A target without the search runtime reports `qiec:capability:search:<entry>`. A parser that another
program in the module calls is kept, and the reachable call graph determines whether a target can
emit the model.

## Structural signatures and autoencoders

A checked module can give the type of a neural computation without containing the network. It holds
an *attachment descriptor*, a typed reference to an implementation, and a runtime provider gives the
PyTorch object when the module runs. Saving the checked module thus saves neither the learned
parameters nor a runnable Python module, and both are given at invocation.

A `signature` declares typed terms: data leaves, recursive constructors, binders, and variables
written as [de Bruijn indices](https://en.wikipedia.org/wiki/De_Bruijn_index). It lowers to an opaque
family, because the core tracks a term's sort but does not serialize the Python object that holds
it.

An encoder or a decoder becomes a `Compute` instance and a named computation. For the term
autoencoder:

```text
reconstruct(term)
  -> Enc.apply(term)
  -> Dec__nll.apply((term, code))
  -> LogWeight
```

`Program.run("reconstruct", term)` attaches the compiled `torch.nn.Module` objects, validates the
Python values against the checked input and output types, and keeps the autograd graph. The encoder
and decoder are registered under stable module names derived from the source, so `parameters()`,
`named_parameters()`, and `state_dict()` include them.

A process that deserializes the module attaches compatible providers before it invokes an entry that
needs them. A transpile target without such a provider refuses the entry with
`qiec:capability:neural-attachment:<entry>`.

The [structural autoencoder tutorial](../../tutorials/qvr/12-structural-autoencoders.md) and the
[Term Autoencoder](../../examples/term-autoencoder.md) example build one end to end.

## Programs that call generated entries

Generated entries compose through ordinary calls. A program can call a deduction:

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

Elaboration replaces the `goal_weight()` access with a call to the deduction's run computation and a
request to `score`. The features a target must support for `fit` thus include search, weight
accumulation, parameter lookup when weights are learned, and the probabilistic effects of the
program's other steps.

## Inspect the generated graph

From the least to the most detailed:

```bash
qvr run model.qvr --list
qvr repl model.qvr
# then: :info entry
#       :effects entry
#       :dump entry --json
#       :graph program --mermaid
```

In Python, `program.entry_points()` gives the public entries and `program.qiec` gives the checked
module. The entries are invoked, and the helpers in `program.qiec` show how they were compiled.

## Target requirements

A generated computation can require a feature that no single source line shows: `parser(...)`
requires search, and a loss over a decoder requires neural attachments. The diagnostic names the
reachable computation and keeps its source range. `qvr check --target TARGET`, or a target set in the
language server, reports these requirements while the source is edited, and `qvr transpile` then
reports any restriction particular to the target's renderer.
