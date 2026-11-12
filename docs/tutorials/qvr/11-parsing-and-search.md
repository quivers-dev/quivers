# 11. Parsing as an effectful computation

A weighted chart parser can be written as a weighted deduction system: inference rules over items such as "category $X$ spans words $i$ to $j$", with derivations weighted in a semiring [@shieber1995principles; @goodman1999semiring]. QVR compiles such a parser two ways from one source: as a batched PyTorch module for gradient fitting (the compiled runtime) and as a QIEC computation that searches over derivations and runs on the reference machine ([chapter 9](09-indexed-data.md#4-run-the-summary)). Check that the two agree, inspect the effects of the search, and call a parser from a probabilistic program.

The grammar is a [categorial grammar](https://en.wikipedia.org/wiki/Categorial_grammar) [@lambek1958mathematics]. Its categories are built from the atoms `N`, `NP`, and `S` by two slashes, written in Steedman's notation with the result on the left [@steedman2000syntactic]: `X / Y` combines with a `Y` on its right to give an `X`, and `X \ Y` combines with a `Y` on its left to give an `X`. The gallery program [`schema_chart_parser.qvr`](../../examples/source/schema_chart_parser.qvr) declares:

- The categories of depth at most one over the atoms, `object Cat : FreeResiduated(Atoms)`, that is, the atoms and every `A / B` and `A \ B` with atomic `A` and `B`.
- Two *rule schemata*, rule templates with variables `X` and `Y` ranging over `Cat`: forward application `X / Y, Y |- X` and backward application `Y, X \ Y |- X` (the rules of AB grammar).
- A unary schema `X / Y |- X \ Y`, which permits the permutation of the commutative Lambek calculus LP [@moortgat1997categorial].
- A *bundle* `lp_rules`, the library's name for a list of schemata, and the parser:

<!-- compile: false -->
```qvr
define lp_parser = parser(rules=[lp_rules], terminal=Token, start=S, depth=1)
```

A [chart](https://en.wikipedia.org/wiki/Chart_parser) stores the analyses of each span of the input [@nltkSyntax2019]. Several derivations may reach the goal item, an `S` spanning the whole input, and its *inside weight* sums their weights, where the weight of one derivation is the product of its rule weights [@goodman1999semiring]. In the log semiring the product becomes a sum and the sum a `logsumexp`. A chart computes shared subderivations once, whereas the search computation enumerates the derivations, so agreement between the two checks both. An inside weight is not in general a probability normalized over sentences.

## 1. Compile the schema parser

```python
import torch
import math
from quivers.dsl import load
from quivers.qiec.program_runtime import run_deduction

torch.manual_seed(0)
parsing = load("docs/examples/source/schema_chart_parser.qvr")
compiled = parsing.morphism

assert compiled.n_rules == 18
assert compiled.n_unary_rules == 9
print([entry.name for entry in parsing.entry_points()])
```

```text
['lp_parser__eq', 'lp_parser__show', 'lp_parser__goal', 'lp_parser__fail', 'lp_parser__axiom', 'lp_parser__derive', 'lp_parser__run']
```

`compiled` is a `torch.nn.Module`, the form to use for batched gradient fitting. The entry points are the computations the compiler generates for the parser: equality and display of items, the goal test, failure, the lexical axioms, one derivation step, and the full run. These are the form to use for inspecting effects and for running on the reference machine.

## 2. Compare inside scores

`Token` lists the vocabulary `the`, `dog`, `cat`, `sleeps` in that order, so the input `dog sleeps` is `[1, 3]` for the compiled module. Give both forms the same uniform lexical weights and compare:

```python
sentence = ["dog", "sleeps"]
token_ids = torch.tensor([1, 3])

# Give both views the same explicit uniform lexical table.
with torch.no_grad():
    compiled.axiom.lexicon_logits.zero_()
compiled_weight = float(compiled(token_ids).detach())
n_categories = compiled.rule_system.n_categories
n_terminals = compiled.axiom.lexicon_logits.shape[0]
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

assert abs(compiled_weight - checked.weight) < 1e-6
print("inside log weight:", checked.weight)
```

```text
inside log weight: -3.3809946743446355
```

`run_deduction` runs the search computation on the reference machine. It handles the generated `Search` effect by resuming once for each alternative, a multi-shot handler ([chapter 10](10-effects-and-handlers.md#4-choose-a-resumption-grade)), and handles the weight effect `Weight[LogWeight]` with an accumulator copied for each alternative. `fuel` bounds the number of evaluation steps. The result is the `logsumexp` over the complete derivations of the goal, equal to the inside weight of the chart.

## 3. See the generated effects

Open the module in the REPL:

```text
$ qvr repl docs/examples/source/schema_chart_parser.qvr
qvr> :effects lp_parser__derive
qvr> :info lp_parser__run
qvr> :dump lp_parser__run --json
```

The derivation step performs search choices, weight additions, and reads of parameters. The run entry point installs handlers that collect every derivation, so its residual row, the effects left after its own handlers, is empty. The dump shows the generated family of items, with constructors for the categories and the span positions.

## 4. Call a deduction from a probabilistic program

A `deduction` block declares a weighted deduction system directly, and a `program` can call it:

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

`atoms` declares the constructor symbols of the items, and each `rule` is an inference rule whose premises are left of `|-` and whose conclusion is right of it. `#[learnable]` gives the rule or lexical entry a weight that is fitted. `chart.goal_weight()` elaborates to a call of the deduction's run computation, and `score` adds the result to the program's log joint density. A Bayesian model can thus put priors on other parameters while summing exactly over a bounded set of derivations.

## 5. Check transpile targets before fitting

Not every transpile target supports search. Ask the checker before a long fit or before packaging:

```bash
qvr check --target pyro docs/examples/source/schema_chart_parser.qvr
qvr check --target stan docs/examples/source/schema_chart_parser.qvr
```

A target without search reports the diagnostic `qiec:capability:search:<entry>` with the source range of the parser. It does not emit an empty parser or drop the score.

## Next

[Chapter 12](12-structural-autoencoders.md) attaches neural encoders and decoders to typed terms. The [generated-computation reference](../../reference/qvr/generated-computations.md) lists everything the compiler generates for a parser.
