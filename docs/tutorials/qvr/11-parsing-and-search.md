# Parsing as an effectful computation

This tutorial runs a schema-backed chart parser in two ways: as a batched
PyTorch `ChartParser` and as a checked search computation. Both views use the
same instantiated grammar and learned weights.

The source is the gallery's
[`schema_chart_parser.qvr`](../../examples/source/schema_chart_parser.qvr).
It declares a free residuated category universe, forward and backward
application schemas, a permutation schema, one bundle, and:

<!-- compile: false -->
```qvr
define lp_parser = parser(rules=[lp_rules], terminal=Token, start=S, depth=1)
```

[A chart](../../examples/custom-rules.md#understanding-the-model) stores
possible analyses for [intervals of a sentence](https://www.nltk.org/book/ch08.html#well-formed-substring-tables)
[@nltkSyntax2019]. In `the dog sleeps`, one cell can say that `the dog` forms
a noun phrase, and another that `sleeps` expects that phrase on its left.
A rule combines them into a sentence analysis. Several derivations may reach
the same full-sentence goal, so its score must aggregate their contributions.

An *inside weight* aggregates those derivations. Rule weights multiply along
a derivation, and alternative derivations add; in log space these become
addition and [`logsumexp`](../../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine). A chart can compute the shared subproblems once,
while reference search enumerates alternatives explicitly. Agreement between
the two checks the computation; a raw inside weight is not automatically a
normalized probability over all possible sentences.

## 1. Compile the schema parser

```python
import torch
import math
from quivers.dsl import load
from quivers.qiec.program_runtime import run_deduction

torch.manual_seed(0)
parsing = load("docs/examples/source/schema_chart_parser.qvr")
classic = parsing.morphism

assert classic.n_rules == 18
assert classic.n_unary_rules == 9
print([entry.name for entry in parsing.entry_points()])
```

The classic object is a `torch.nn.Module`. It is the right view for batched
gradient fitting. The entry list exposes the generated equality, display,
goal, failure, axiom, derivation, and run computations. Those definitions are
the right view for effect inspection and reference execution.

## 2. Compare inside scores

The token vocabulary is declared in source order, so `the dog sleeps` is
`[0, 1, 3]` in the classic view:

```python
sentence = ["dog", "sleeps"]
token_ids = torch.tensor([1, 3])

# Give both views the same explicit uniform lexical table.
with torch.no_grad():
    classic.axiom.lexicon_logits.zero_()
classic_weight = float(classic(token_ids).detach())
n_categories = classic.rule_system.n_categories
n_terminals = classic.axiom.lexicon_logits.shape[0]
uniform = -math.log(n_categories)
parameters = {
    f"lp_parser.lex.{index}": uniform
    for index in range(n_categories * n_terminals)
}
checked = run_deduction(
    parsing.qiec,
    "lp_parser",
    tokens=sentence,
    parameters=parameters,
    fuel=1_000_000,
)

assert abs(classic_weight - checked.weight) < 1e-6
print("inside log weight:", checked.weight)
```

`run_deduction` handles the generated `Search` instance by resuming once per
alternative and handles `Weight[LogWeight]` with a forkable accumulator. The
result is a log-sum-exp over complete derivations of the goal, matching the
inside chart.

## 3. See the generated effects

Open the module in the REPL:

```text
$ qvr repl docs/examples/source/schema_chart_parser.qvr
qvr> :effects lp_parser__derive
qvr> :info lp_parser__run
qvr> :dump lp_parser__run --json
```

The derivation performs search choices, weight additions, and parameter reads.
The run entry installs collecting handlers, so its residual row is closed. The
generated item family remains visible in the dump, including constructors for
the finite category inventory and span positions.

## 4. Call a deduction from a probabilistic program

A direct `deduction` can be called from a `program`:

<!-- compile: qiec -->
```qvr
object Term : FinSet 12
object Weight : Real 1

deduction Tiny : Term -> Term [semiring=LogProb, start=S, depth=5]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "dog" : N = dog #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]

program score_sentence : Term -> Weight
    let chart = parse(Tiny, sentence)
    score evidence = chart.goal_weight()
    return evidence

export score_sentence
```

The source expression `chart.goal_weight()` elaborates to the deduction's
checked run call. The score step then adds its result to the probabilistic
program's joint. This composition permits a Bayesian model to place priors on
other parameters while exactly summing a bounded discrete derivation space.

## 5. Check deployment before fitting

Search is not available on every transpile target. Ask the checker before a
long fit or packaging step:

```bash
qvr check --target pyro docs/examples/source/schema_chart_parser.qvr
qvr check --target stan docs/examples/source/schema_chart_parser.qvr
```

An unsupported target reports `qiec:capability:search:<entry>` and preserves
the source range that introduced the parser. It does not emit an empty parser
or drop the score.

The classic view executes the fitted batched chart module. The checked view
exposes the search computation for reference evaluation and composition, and
reports unsupported target operations. The call site selects which view executes.

Continue with [Structural autoencoders](12-structural-autoencoders.md), or see
the [generated-computation reference](../../reference/qvr/generated-computations.md)
for the complete elaboration map.
