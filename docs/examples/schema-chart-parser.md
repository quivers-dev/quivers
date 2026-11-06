# Schema-Bundled Chart Parser

## Overview

A learnable [CKY](https://en.wikipedia.org/wiki/CYK_algorithm) chart parser assembled from pattern-polymorphic `schema` declarations rather than a `deduction` block. Two binary schemas give forward and backward application (the AB grammar core of the [Lambek calculus](https://doi.org/10.2307/2310058)); one unary schema adds the directional permutation of the commutative Lambek calculus LP. A `bundle` names the rule set, and `parser(...)` splices the bundle into a [`ChartParser`](../api/stochastic/parsers.md#quivers.stochastic.parsers.ChartParser) whose lexical category assignments and per-rule weights are learnable log-weights. [@lambek1958mathematics]

## QVR source

```qvr
# Schema-Bundled Chart Parser
#
# A learnable CKY chart parser assembled from pattern-polymorphic
# rule schemas over a free residuated category universe, rather
# than from a deduction block. Two binary schemas give forward
# and backward application (the AB grammar core); one unary
# schema adds the directional permutation of the commutative
# Lambek calculus LP. A bundle names the rule set, and
# parser(...) splices the bundle into a differentiable chart
# parser whose lexical category assignments and per-rule weights
# are learnable log-weights.
#
# Rule schemas (X, Y range over Cat):
#
#   fwd_app : X/Y, Y  |- X      forward application
#   bwd_app : Y, X\Y  |- X      backward application
#   perm    : X/Y     |- X\Y    directional permutation (LP)
#
# The chart's category inventory is the depth-1 residuated
# closure of the atoms {N, NP, S}: the atoms plus every A/B and
# A\B with atomic A and B. Token indexes the four-word
# vocabulary {the, dog, cat, sleeps} in declaration order; the
# parser's forward pass scores a token sequence by the inside
# log-weight of the S-rooted span covering the whole input.

object Atoms : {N, NP, S}
object Cat : FreeResiduated(Atoms)
object Token : {the, dog, cat, sleeps}

schema fwd_app (X, Y : Cat) : (X / Y) * Y -> X
schema bwd_app (X, Y : Cat) : Y * (X \ Y) -> X
schema perm (X, Y : Cat) : X / Y -> X \ Y

bundle lp_rules : [fwd_app, bwd_app, perm]

define lp_parser = parser(rules=[lp_rules], terminal=Token, start=S, depth=1)

export lp_parser
```

## Understanding the model

A schema is a reusable [rule template](custom-rules.md#understanding-the-model): “a phrase needing `Y` on the right,
followed by a `Y`, yields `X`.” Instantiating it with `X = NP` and `Y = N`
gives the familiar combination of `the : NP/N` with `dog : N`. A bundle
is a named list of these templates, while the finite category inventory
specifies which substitutions the compiler actually tries. The NLTK book’s [context-free grammar examples](https://www.nltk.org/book/ch08.html#context-free-grammar) introduce the phrase categories and production notation behind this template [@nltkSyntax2019].

The parser returns one aggregate score for all derivations reaching the
sentence goal. This tells you how the model weights those analyses; without
additional normalization it is not a probability over sentences. Comparing
the chart and [reference-search score](../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) is useful because they calculate that
same aggregate by different execution strategies.

## Walkthrough

`object Atoms : {N, NP, S}` declares the category atoms as an enum set, and `object Cat : FreeResiduated(Atoms)` binds their [free residuated closure](../semantics/schemas.md#4-the-free-residuated-universe) as the universe the schema parameters range over. `object Token : {the, dog, cat, sleeps}` declares the terminal vocabulary; token indices follow declaration order, so `the` is token 0 and `sleeps` is token 3.

Each `schema` declaration is a family of chart rules parameterized by typed meta-variables ([Schemas](../semantics/schemas.md#6-schema-declarations)). The arity is read off the domain shape: a product domain compiles to a [`PatternBinarySchema`](../api/stochastic/biclosed.md#quivers.stochastic.schema.PatternBinarySchema) that fires on adjacent chart cells, and any other domain compiles to a [`PatternUnarySchema`](../api/stochastic/biclosed.md#quivers.stochastic.schema.PatternUnarySchema) that fires on a single cell. QVR's slash convention is biclosed: in both `X / Y` and `X \ Y` the left operand is the result and the right operand is the argument, with the slash direction marking where the argument is sought. Thus `fwd_app` consumes a rightward functor and its right neighbor, `bwd_app` consumes a leftward functor and its left neighbor, and `perm` flips a functor's seek direction while fixing its result and argument, which is exactly the structural permutation that separates LP from the directional calculus.

`bundle lp_rules : [fwd_app, bwd_app, perm]` binds the rule set as a first-class name ([Bundles](../semantics/schemas.md#7-bundles)). The `parser(...)` expression resolves each entry in `rules=` against declared rules, schema primitives, and bundles; a bundle reference is spliced, that is, replaced by its members exactly as if they had been listed verbatim.

`parser(rules=[lp_rules], terminal=Token, start=S, depth=1)` builds the chart parser via [`ChartParser.from_schema`](../api/stochastic/parsers.md#quivers.stochastic.parsers.ChartParser.from_schema). The category atoms are inferred from the uniquely declared `FreeResiduated` object in scope, and `depth=1` bounds the [`CategorySystem`](../api/stochastic/categories.md#quivers.stochastic.categories.CategorySystem) to the atoms plus every depth-1 slash category, 21 categories in all. Instantiating the three schemas over this inventory yields a [`RuleSystem`](../api/stochastic/rules.md#quivers.stochastic._rule_system.RuleSystem) with 18 binary and 9 unary firings, each carrying its own learnable log-weight; the lexical axiom contributes a learnable `(4, 21)` token-to-category log-weight table.

The QIEC lowering instantiates that same finite rule system as a deduction. Its recursive computation carries the expected ground category, chooses only rules that can produce it, chooses split points for binary rules, and adds lexical and rule weights through the core effects. Thus `run_deduction(module, "lp_parser", tokens=...)` computes the same inside weight as the classic chart. A transpile target without the search runtime rejects this computation under `qiec:capability:search:lp_parser__run`.

## Try it

```python
import math
import torch
from quivers.dsl import load
from quivers.qiec.program_runtime import run_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/schema_chart_parser.qvr")
parser = prog.morphism

print(parser.n_rules, "binary rules;", parser.n_unary_rules, "unary rules")

# token indices: the=0, dog=1, cat=2, sleeps=3
corpus = torch.tensor([[0, 1, 3], [0, 2, 3]])  # the dog sleeps / the cat sleeps
print("initial scores:", parser(corpus))

# Give the checked search computation and classic chart the same explicit
# uniform lexical table, then compare their inside scores.
with torch.no_grad():
    parser.axiom.lexicon_logits.zero_()
n_categories = parser.rule_system.n_categories
n_terminals = parser.axiom.lexicon_logits.shape[0]
uniform = -math.log(n_categories)
parameters = {
    f"lp_parser.lex.{index}": uniform
    for index in range(n_categories * n_terminals)
}
checked = run_deduction(
    prog.qiec,
    "lp_parser",
    tokens=["dog", "sleeps"],
    parameters=parameters,
    fuel=1_000_000,
)
classic = float(parser(torch.tensor([1, 3])).detach())
assert abs(checked.weight - classic) < 1e-6

# LP's permutation is a structural postulate rather than a fitted
# preference, so the corpus adjusts the lexical table and the two
# application rules and leaves the unary weights at their zero
# initialisation.
structural = {id(p) for p in parser.deductions[1].parameters()}
fitted = [p for p in parser.parameters() if id(p) not in structural]

opt = torch.optim.Adam(fitted, lr=5e-2)
for _ in range(40):
    opt.zero_grad()
    loss = -parser(corpus).sum()
    loss.backward()
    opt.step()

print("fitted scores:", parser(corpus))
```

The parser is callable on integer token tensors, `(seq_len,)` for one sentence or `(batch, seq_len)` for a batch, and returns the inside log-weight of the `S`-rooted span covering the whole input. The score is differentiable in the lexical table and the per-rule weights, so fitting the grammar to a corpus is ordinary gradient ascent on the summed scores.

`run_deduction` exercises the QIEC view of the same parser. It handles the
generated search and weight effects on the reference machine and returns the
goal's inside weight. Pass an explicit parameter store when comparing after an
optimizer step; the call above deliberately compares the shared initial
weights before fitting mutates the classic module.

## The chart_fold primitive

`parser(rules=...)` is syntactic sugar for the `chart_fold(...)` primitive,
which builds the same inside computation from morphism-valued arguments rather
than named schemas. Given a lexical kernel `emit : NT -> Tok` and a branching
kernel `grow : NT -> NT * NT`, `chart_fold` constructs an
[`InsideAlgorithm`](../api/stochastic/inside.md#quivers.stochastic.inside.InsideAlgorithm)
directly, so the rule tensors are user-declared morphisms rather than schema
instantiations:

```qvr
object NT : FinSet 3
object Tok : FinSet 5

morphism grow : NT -> NT * NT
morphism emit : NT -> Tok

define inside = chart_fold(lex=emit, binary=grow, start=0)

export inside
```

This form is the PCFG inside algorithm with `grow[A, B, C]` playing the role of the production probability P(A -> B C) and `emit[A, t]` the emission probability P(A -> t); `start=0` selects the root nonterminal by index.

## Categorical perspective

A schema denotes a uniform family of morphisms indexed by substitutions of its parameters into the residuated universe ([Schemas](../semantics/schemas.md#6-schema-declarations)); instantiating the family over a finite [`CategorySystem`](../api/stochastic/categories.md#quivers.stochastic.categories.CategorySystem) is precisely the evaluation of a functor from category inventories to rule systems, and the union of schemas corresponds to the coproduct of the induced rule systems. Under QVR's biclosed convention `fwd_app` and `bwd_app` are the counits of the right and left residuation adjunctions, and `perm` is the structural isomorphism that a symmetric monoidal (rather than merely monoidal) base makes available. The bundle is denotationally inert: `parser(rules=[lp_rules])` and `parser(rules=[fwd_app, bwd_app, perm])` compile to the same rule system.
