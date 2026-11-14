# Schema-Bundled Chart Parser

## Overview

A learnable [CKY](https://en.wikipedia.org/wiki/CYK_algorithm) chart parser assembled from pattern-polymorphic `schema` declarations rather than a `deduction` block. Two binary schemas give forward and backward application (the AB grammar core of the [Lambek calculus](https://doi.org/10.2307/2310058)). One unary schema adds the directional permutation of the commutative Lambek calculus LP. A `bundle` names the rule set, and `parser(...)` splices the bundle into a [`ChartParser`](../api/stochastic/parsers.md#quivers.stochastic.parsers.ChartParser) whose lexical category assignments and per-rule weights are learnable log-weights. [@lambek1958mathematics]

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

A schema is a reusable [rule template](custom-rules.md#understanding-the-model):
“a phrase needing `Y` on the right, followed by a `Y`, yields `X`.”
The [phrase categories](https://www.nltk.org/book/ch08.html#context-free-grammar)
`NP` and `N` denote noun phrases and nouns [@nltkSyntax2019]. Instantiating
`X = NP` and `Y = N` gives the combination of `the : NP/N` with `dog : N`.
A bundle names a list of templates. The finite category inventory determines
which substitutions the compiler tries.

The parser returns one aggregate score for all derivations reaching the
sentence goal. This tells you how the model weights those analyses. Without
additional normalization it is not a probability over sentences. Comparing
the chart and [reference-search score](../tutorials/qvr/11-parsing-and-search.md#2-compare-inside-scores) is useful because they calculate that
same aggregate by different execution strategies.

The score becomes a distribution once it is normalized over a finite set of
strings. Fixing a length $n$ and writing $Z(s)$ for the exponentiated score
of a token string $s$, the *length-conditional distribution*

$$
P(s \mid n) = \frac{Z(s)}{\sum_{|s'| = n} Z(s')}
$$

sums to one over the $|\mathrm{Token}|^n$ strings of that length. With four
tokens and $n = 3$ there are 64 strings, few enough to score in one batch,
so the denominator is computed exactly. The fit below maximizes this
likelihood rather than the raw score, which is unbounded above in the
per-rule log-weights.

## Walkthrough

`object Atoms : {N, NP, S}` declares the category atoms as an enum set, and `object Cat : FreeResiduated(Atoms)` binds their [free residuated closure](../semantics/schemas.md#4-the-free-residuated-universe) as the universe the schema parameters range over. `object Token : {the, dog, cat, sleeps}` declares the terminal vocabulary. Token indices follow declaration order, so `the` is token 0 and `sleeps` is token 3.

Each `schema` declaration is a family of chart rules parameterized by typed meta-variables ([Schemas](../semantics/schemas.md#6-schema-declarations)). The arity is read off the domain shape: a product domain compiles to a [`PatternBinarySchema`](../api/stochastic/biclosed.md#quivers.stochastic.schema.PatternBinarySchema) that fires on adjacent chart cells, and any other domain compiles to a [`PatternUnarySchema`](../api/stochastic/biclosed.md#quivers.stochastic.schema.PatternUnarySchema) that fires on a single cell. The slash syntax uses Steedman notation [@steedman2000syntactic]: in both `X / Y` and `X \ Y` the left operand is the result and the right operand is the argument, with the slash direction marking where the argument is sought. Thus `fwd_app` consumes a rightward functor and its right neighbor, `bwd_app` consumes a leftward functor and its left neighbor, and `perm` flips a functor's seek direction while fixing its result and argument, the directional conversion induced by exchange in LP.

`bundle lp_rules : [fwd_app, bwd_app, perm]` binds the rule set as a first-class name ([Bundles](../semantics/schemas.md#7-bundles)). The `parser(...)` expression resolves each entry in `rules=` against declared rules, schema primitives, and bundles. A bundle reference is spliced, that is, replaced by its members exactly as if they had been listed verbatim.

`parser(rules=[lp_rules], terminal=Token, start=S, depth=1)` builds the chart parser via [`ChartParser.from_schema`](../api/stochastic/parsers.md#quivers.stochastic.parsers.ChartParser.from_schema). The category atoms are inferred from the uniquely declared `FreeResiduated` object in scope, and `depth=1` bounds the [`CategorySystem`](../api/stochastic/categories.md#quivers.stochastic.categories.CategorySystem) to the atoms plus every depth-1 slash category, 21 categories in all. Instantiating the three schemas over this inventory yields a [`RuleSystem`](../api/stochastic/rules.md#quivers.stochastic._rule_system.RuleSystem) with 18 binary and 9 unary firings, each carrying its own learnable log-weight. The lexical axiom contributes a learnable `(4, 21)` token-to-category table whose rows are log-softmax normalized, so each token's entry is a log-probability over the 21 categories.

The QIEC lowering instantiates that same finite rule system as a deduction. Its recursive computation carries the expected ground category, chooses only rules that can produce it, chooses split points for binary rules, and adds lexical and rule weights through the core effects. Thus `run_deduction(module, "lp_parser", tokens=...)` computes the same inside weight as the classic chart. Backend coverage is recorded in the [support matrix](../transpile-support.md).

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
assert parser.rule_system.n_categories == 21
assert (parser.n_rules, parser.n_unary_rules) == (18, 9)

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
```

The parser is callable on integer token tensors, `(seq_len,)` for one sentence or `(batch, seq_len)` for a batch, and returns the inside log-weight of the `S`-rooted span covering the whole input. The score is differentiable in the lexical table and the per-rule weights.

`run_deduction` exercises the QIEC view of the same parser. It handles the
generated search and weight effects on the [reference machine](../developer/qiec.md#reference-evaluator) and returns the
goal's inside weight. Pass an explicit parameter store when comparing after an
optimizer step. The call above deliberately compares the shared initial
weights before any fitting mutates the classic module.

### Recovering a length-conditional distribution

A second parser with randomly drawn lexical and application weights serves as
the ground truth. Its length-conditional distribution over the 64 three-token
strings generates a corpus, and a freshly initialized parser is fit by
maximizing the corpus log-likelihood under the same normalization. LP's
permutation is a structural postulate rather than a fitted preference, so
both parsers keep the nine unary weights at zero and the fit adjusts only the
lexical table and the application rules. The fit is judged by the
[Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
from the true distribution, against the same divergence for the initial
parser.

```python
import itertools

torch.manual_seed(1)
truth = load("docs/examples/source/schema_chart_parser.qvr").morphism
model = load("docs/examples/source/schema_chart_parser.qvr").morphism
strings = torch.tensor(list(itertools.product(range(4), repeat=3)))


def log_conditional(chart_parser):
    """log P(s | n = 3) for every three-token string."""
    scores = chart_parser(strings)
    return scores - torch.logsumexp(scores, dim=0)


def kl(log_p, log_q):
    support = torch.isfinite(log_p)
    return float((log_p[support].exp() * (log_p[support] - log_q[support])).sum())


with torch.no_grad():
    truth.axiom.lexicon_logits.normal_(0.0, 1.5)
    truth.deductions[0].weights.normal_(0.0, 1.0)
    true_log_p = log_conditional(truth)

draws = torch.multinomial(true_log_p.exp(), 2000, replacement=True)
counts = torch.bincount(draws, minlength=len(strings)).float()
observed = counts > 0

structural = {id(p) for p in model.deductions[1].parameters()}
fitted = [p for p in model.parameters() if id(p) not in structural]
unary_before = model.deductions[1].weights.detach().clone()
with torch.no_grad():
    kl_initial = kl(true_log_p, log_conditional(model))

opt = torch.optim.Adam(fitted, lr=5e-2)
for _ in range(200):
    opt.zero_grad()
    log_p = log_conditional(model)
    loss = -(counts[observed] * log_p[observed]).sum() / counts.sum()
    loss.backward()
    opt.step()

with torch.no_grad():
    kl_fitted = kl(true_log_p, log_conditional(model))
print(f"KL(true || initial) = {kl_initial:.4f}")
print(f"KL(true || fitted)  = {kl_fitted:.4f}")
assert kl_fitted < 0.05 and kl_fitted < 0.25 * kl_initial
assert torch.equal(model.deductions[1].weights.detach(), unary_before)
```

The tolerance follows from the corpus size: 2,000 draws over 64 outcomes put
the sampling error of an unconstrained maximum-likelihood estimate near
$63 / (2 \cdot 2000) \approx 0.016$ nats, so a fitted divergence below 0.05
indicates that the fit has recovered the distribution up to sampling noise.
The last assertion checks that the permutation weights were indeed left
untouched.

## The chart_fold primitive

`parser(rules=...)` is syntactic sugar for the `chart_fold(...)` primitive,
which builds the same inside computation from morphism-valued arguments rather
than named schemas. Given a lexical [kernel](https://en.wikipedia.org/wiki/Markov_kernel) `emit : NT -> Tok` and a branching
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

This form runs the inside recursion of a context-free grammar in [Chomsky normal form](https://en.wikipedia.org/wiki/Chomsky_normal_form), with `grow[A, B, C]` weighting the production A -> B C and `emit[A, t]` the production A -> t. `start=0` selects the root nonterminal by index. Both declarations default to [stochastic morphisms](../guides/stochastic.md#stochastic-morphisms), so each kernel is row-normalized on its own: for every nonterminal A, the binary weights sum to one over pairs (B, C), and the lexical weights sum to one over tokens t. A probabilistic context-free grammar instead requires the binary and lexical productions of A to share one distribution, summing to one jointly. With two separately normalized kernels each nonterminal carries a total mass of two, so the inside score is that of a weighted grammar rather than a sentence probability, and the length-conditional normalization above applies to it as well.

## Categorical perspective

A schema describes a family of rules obtained by substituting categories for its parameters. A finite [`CategorySystem`](../api/stochastic/categories.md#quivers.stochastic.categories.CategorySystem) restricts the substitutions used to instantiate that family. A bundle collects the resulting rules: `parser(rules=[lp_rules])` and `parser(rules=[fwd_app, bwd_app, perm])` compile to the same rule system.

In a [biclosed monoidal category](https://ncatlab.org/nlab/show/closed+monoidal+category), the two slash types are internal function types for arguments on opposite sides. Application evaluates such a function against its argument. Formally, `fwd_app` and `bwd_app` are the counits of the two tensor-residuation adjunctions. A [symmetric](https://ncatlab.org/nlab/show/symmetric+monoidal+category) tensor also permits exchange, which induces the change of direction represented by `perm`. These interpretations concern the rule shapes. Independently learned numerical rule weights are not required to satisfy categorical coherence equations.
