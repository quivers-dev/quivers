# Weighted Deduction Fragment

The QVR weighted-deduction fragment is declared by the
`deduction NAME : Domain -> Codomain { … }` block. A single block
realizes grammar-style parsers, type-theoretic proof systems,
Datalog-shaped fixed-point evaluations, and graph algorithms as
parameter settings on the same agenda engine. See
[Weighted Deduction Systems](../guides/deduction.md) for the user
guide; this page defines the denotation.

A weighted deduction system accumulates consequences of rules. Its *chart*
is a table from established items to their weights, and its *agenda* is a
work list of consequences still to process. Sharing a chart avoids
recomputing the same subproblem every time it appears in a derivation.

For a graph example, two known edges $a\to b$ and $b\to c$ support a path
$a\to c$. If their costs are 2 and 3, that derivation costs 5. A direct
edge of cost 4 gives another derivation of the same conclusion. A
minimum-cost algebra keeps 4; a counting algebra would instead count the
two derivations. Parsing uses the same pattern, with spans and grammatical
categories in place of graph endpoints.

The following definitions separate (i) what counts as an item, (ii) which
rules connect items, and (iii) how derivation weights combine. The long
grammar inventory in §10 is a syntax reference; the preceding sections
explain what that syntax computes.

## 1. Item algebra

A `deduction` block fixes a single item algebra `I` via the
`atoms NAME, NAME, ...` clause. The atoms are the closed set of nullary and
parametric constructor symbols; an item `i ∈ I` is a constructor
application `c(a_1, …, a_n)` with `c` an atom and `a_1, …, a_n`
arbitrary nested items. Concretely, `I` is the free algebra over
the atoms quotiented by the underlying carrier object `Term`'s
identification of structurally equal applications.

Pattern variables range over `I`; the unifier supplies first-order
substitutions. An identifier in a rule body is a variable exactly when it
is neither a declared atom nor a numeral, whatever its spelling (single
uppercase letters are the convention). A declared atom's name always
denotes that atom, in the agenda chart and in the checked QIEC search
alike, so a rule that means a variable must not spell it like an atom.
In a category slot the atom is a legitimate ground pattern; in a
chart-position slot (see [§2.1](#21-learnable-rule-weights-learnable))
it is rejected at compile time with a diagnostic at the rule that names
the identifier, since a position slot holds a position variable or an
integer.

## 2. Rules

A rule lists premises on the left and a conclusion on the right of `|-`.
It can fire when chart items match all its premise patterns under one
consistent substitution for their variables. A *hyperedge* is the graph
analogue of this many-input connection: several premises jointly support
one conclusion. It is not a set of independent one-premise implications.

A rule declaration

```
rule r : π_1, …, π_m |- π
```

introduces a hyperedge in the rule-system multicategory: a
universally quantified sequent

$$
\frac{\pi_1[\bar X / \bar\pi] \quad \cdots \quad \pi_m[\bar X / \bar\pi]}{\pi[\bar X / \bar\pi]} \quad (r)
$$

over substitutions `[\bar X / \bar\pi]` for the rule's free pattern
variables. The hyperedge carries a log-weight `β_r ∈ K` for the
declared semiring `K`.

The collection of all rules forms a rule system
`Σ : I^{op} × I → K-Vect` assigning to each pair `(i, i')` the
`K`-module spanned by `Σ`-derivations `i ⊢ i'`. Rules are
hyperedges in the multicategory of items; the agenda fires each
rule by substitution along a pattern morphism into the chart.

### 2.1 Learnable rule weights (`#[learnable]`)

A rule may carry a trailing pragma `#[learnable]`:

```
rule r : π_1, …, π_m |- π #[learnable]
```

Rather than the default constant log-weight $\beta_r$, the rule's
contribution is a learnable parameter keyed by the rule's
*non-position* bindings. The variables of a rule fall into two
sorts. A *position variable* fills a chart-position slot: one of
the first two slots of `span` in a deduction with a lexicon (whose
axioms are `span(i, i + 1, c)` items), a slot an integer literal
fills, or any slot a variable shared with a position slot fills in
the same rule, closed under that last clause. Every other variable
is a *category variable*: it ranges over categories, logical forms,
or other structured items. Writing $P(r)$ for the position
variables and $C(r) = \mathrm{Vars}(r) \setminus P(r)$ for the
category variables, each firing produces a binding map
$\sigma : \mathrm{Vars}(r) \to I$ and the conclusion's weight
becomes

$$
\beta_r(\sigma) \;=\; \otimes_{\ell=1}^{m} \alpha[\sigma(\pi_\ell)] \;\otimes\; \theta_{r, \sigma|_{C(r)}},
$$

where $\theta_{r, \kappa} \in K$ is one learnable scalar per
distinct restriction $\kappa = \sigma|_{C(r)}$ of the binding to
the category variables, keyed by a canonical string form of
$\kappa$. Chart positions never enter the key: a rule has the
same weight at every position of every input, so a weight learned
on one sentence applies to the same construction anywhere in any
other, and the parameter set does not grow with input length. The
parameters live on a per-deduction `nn.Module` side-table that
participates in the model's `.parameters()`; the weight of a
category binding is allocated the first time a firing produces it.

Two specializations of the surface deserve explicit names:

* When the rule has no category variable (every variable is a
  position, as in `np : span(I, K, Det), span(K, J, N) |- span(I, J, NP)`),
  $\theta_{r, \kappa}$ reduces to a single global rule weight
  $\theta_r$, the PCFG-style per-production weight. It is
  allocated when the deduction is compiled.
* When the rule has category variables (e.g. `fwd : span(I, K,
  Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)`), each distinct
  ground value of $(X, Y)$ pulls a distinct parameter, recovering
  the *partial-application weight* of
  [§2.2](#22-parent-rule-specialization-parent).

A lexicon entry marked `#[learnable]` has one weight per entry
(and, for a wildcard or restricted category, one per candidate
category), shared across every position at which its word occurs.

The checked QIEC elaboration reads the same weights from its
`Param` store under the names `<D>.rule.<r>:<κ>` and `<D>.lex.<i>`,
with $\kappa$ the category bindings rendered in variable-name
order and joined by `|` (empty for a rule without category
variables), so both evaluators agree on which firings share a
weight.

### 2.2 Parent rule specialization (`parent=`)

A rule may declare a *parent* via the `parent=` option in its
pragma:

```
rule r' : π'_1, …, π'_m |- π' #[learnable, parent=r]
```

The conclusion weight then composes additively in the log semiring
with the parent's weight under the *same* category bindings:

$$
\beta_{r'}(\sigma) \;=\; \otimes_\ell \alpha[\sigma(\pi'_\ell)]
\;\otimes\; \theta_{r', \kappa}
\;\otimes\; \theta_{r, \kappa},
\qquad \kappa = \sigma|_{C(r')}.
$$

The parent chain is acyclic and terminates at a non-parented
rule. This recovers a *hierarchical* / *back-off* parametrization
in which the specialization contributes a correction term over
the parent's weight without retraining the parent
from scratch.

### 2.3 Bounded rule weights (`bounded`)

A learnable rule whose pragma additionally lists `bounded` is
reparameterized to keep its surface log-weight strictly negative:

$$
\theta_{r, \kappa} \;=\; -\mathrm{softplus}(\mathit{raw}_{r,\kappa})
\;-\; \log n_b,
$$

where $n_b$ counts the deduction's bounded rules. The per-firing
factor $\exp(\theta_{r,\kappa})$ is then strictly below $1 / n_b$,
so the total mass a chart item can push through the deduction's
bounded rules is strictly below 1. This is the correct
parameterization when the rules close cycles in the deduction
graph and the semiring is non-idempotent (`LogProb`, `Counting`,
`Inside`): for cycles built from unary bounded rules, each of
which contributes at most one out-edge per item, the joint cap
bounds the row sums of the chart's update operator below 1, the
Kleene-star series converges, and the chart's fixed point is
finite for every parameter value (see
[§7](#7-strategy-independence) on convergence vs. divergence
detection). Note that a per-rule cap of 1 alone would not
suffice: interlocking cycles (e.g. an introduction / elimination
pair over nested constructors) can diverge even when every
individual cycle's log-weight is negative.

The separation between a deduction structure and the semiring used to score
it follows the semiring-parsing perspective [@goodman1999semiring].

## 3. Semiring

The primer’s [reachability and shortest-path calculations](../getting-started/concepts.md#3-an-algebra-chooses-how-weights-combine) illustrate how changing two operations changes the answer computed by a shared recurrence. [Learnable rule weights](#21-learnable-rule-weights-learnable) explain where the scalar factors in a derivation come from.

The semiring field selects the scoring algebra `K`:

| Field value | $\oplus$ (plus) | $\otimes$ (times) | Use |
|-------------|-----------------|--------------------|-----|
| `LogProb` | logsumexp | + | marginal log-probability |
| `Viterbi` | max | + | best-derivation decoding |
| `Boolean` | or | and | recognition |
| `Counting` | + | × | derivation counts |

The chart is enriched over `K`; the agenda enumerates derivations
under `K`'s monoidal operations.

## 3a. Binders and alpha-renaming

A deduction may declare a `binders` block listing constructor
names whose first argument is a *bound variable*:

```
deduction D : … {
    atoms NP, S, Bwd, App, Var, bark
    binders Lam
    …
}
```

For every $c \in \mathsf{binders}(D)$, an item of the form
$c(x, t)$ denotes the same equivalence class as $c(y,
t[x := y])$ for any fresh $y$. The lexicon-LF compiler enforces
this *structurally* by alpha-renaming every binder application
at construction time:

$$
c(x, t) \;\longmapsto\; c(\mathsf{fresh}, t[x := \mathsf{fresh}]),
$$

with $\mathsf{fresh}$ drawn from a per-deduction fresh-symbol
pool. Two lexicon entries that both write `Lam(x, body)` in
source thus produce chart items with *distinct* canonical
bound names, so the chart's structural identity on item tuples
coincides with alpha-equivalence on the source terms.

The compiler additionally collects every bound variable name
appearing in any lexicon LF and extends the constructor set
$\mathsf{atoms}(D)$ with those names for the duration of LF
compilation, so the user need not duplicate them in `atoms`.
References to a bound name inside a binder body resolve through
the alpha-renamed canonical symbol. An occurrence outside the
binder's scope is rejected unless the same spelling is also a
declared nullary constant.

The public chart representation distinguishes the two kinds of
leaf explicitly:

| Surface term | Runtime term |
|--------------|--------------|
| nullary constant `dog_p` or `dog_p()` | `("atom", "dog_p")` |
| bound variable `x` | `("var", "#v1")` after alpha-renaming |
| application `App(f, x)` | `("App", f_term, x_term)` |

Thus a rule pattern that mentions `dog_p` has exactly the same
runtime shape as `dog_p` in an inline or TSV-backed lexicon LF.
The names `atom` and `var` are reserved term-algebra tags and
cannot be declared in `atoms`.

The compiler exposes capture-avoiding substitution as a
[`let`-expression builtin](#93-let-expression-builtins) `subst(t,
v, w)`: because every binder has been alpha-renamed to a unique
canonical symbol at compile time, naive structural substitution
is already capture-avoiding.

## 3b. Item shape and pattern arity

Span axioms emitted by an indented `lexicon` body carry the
lexical category in their fourth slot and (when any rule
references an LF binding, or the deduction declares a
`binders` block) the logical form in their fifth slot:

$$
\mathsf{item} \;=\; (\mathsf{span}, i, j, \mathit{cat}) \quad\text{or}\quad
(\mathsf{span}, i, j, \mathit{cat}, \mathit{lf}).
$$

The compiler detects whether the deduction *uses LFs* by
scanning rule patterns for any 4-arity `span(I, J, C, F)`
expression, and turns the LF slot on or off uniformly per
deduction. Patterns of the shorter form `span(I, J, C)` are
auto-extended at compile time with a fresh LF wildcard so both
surface forms match the same chart shape; this preserves
backwards compatibility for LF-free deductions while enabling
the LF-binding surface where needed.

## 4. Axiom injector

A deduction needs an *axiom injector*
`ax : Input → List(I × K)` producing the initial chart from an
input. The block admits three surface forms:

- `lexicon` block with `"word" : Cat = lf #[learnable]` entries: label-indexed
  lookup table inline.
- `lexicon from "path.tsv" with learnable`: same shape loaded
  from a TSV; its logical forms use the same constant tagging,
  binder normalization, and scope checks as inline entries.
- `axioms = some_kernel_morphism`: a declared Kleisli morphism
  `Input → List(I × K)`.

## 5. Goal, depth, and convergence

A depth limit and a convergence criterion solve different problems. Bounding
category-expression depth prevents ever-larger category terms from being
introduced. A rule can still send an existing item back to itself at the
same depth. Repeated traversal then contributes further weight, so the
scheduling and convergence conditions below determine whether that sum
reaches a finite value.

`start s` declares the start atom; the goal predicate is
"there exists a derivation whose conclusion is `c(s, …)` for
some configured wrapper constructor (e.g. a `span(0, n, s)` over
the full input range)."

`depth d` bounds the maximum *constructor-tree depth* of the
conclusion category of any rule firing. The compiler installs a
uniform side-condition on every rule that rejects firings whose
conclusion category exceeds depth $d$ in the constructor tree
(counting every wrapping $(c, a_1, \dots, a_n)$ tuple whose head
is a category constructor as one unit of depth). This bounds the
chart's category-constructor tower for rule systems with growing
unary intro / elim cycles (modal, continuation, residuation),
without privileging any particular category constructor. A `span`
item measures as its category; positions, atoms, bound variables, and
integer literals measure zero. The bound never limits how many rules
a derivation applies: a sentence whose derivations are deeper than
$d$ is parsed in full. The checked QIEC search applies the same
measure to the same conclusions, so the two evaluators cut off exactly
the same items.

`tolerance ε` controls convergence detection in the chart's
weight aggregation. For non-idempotent semirings with cyclic
rule graphs, each re-derivation of an item along a new path
increases its weight via `semiring.plus`, even when the item is
already in the chart; with contractive cycle weights (e.g. the
joint sub-stochastic cap of [§2.3 `bounded`](#23-bounded-rule-weights-bounded)),
the sequence of updates converges to a Kleene-star limit, and
the chart terminates re-firings on an item once successive
updates fall below $\varepsilon$. With a divergent cycle the
agenda hits `max_iterations` instead (the correct rejection of
an ill-posed model). Setting $\varepsilon = 0$ recovers strict
structural equality and the default acyclic-friendly semantics. The
checked QIEC search reaches the same limit by enumeration: it bounds
how often a chain of unary rules may revisit an item and raises the
bound until two successive goal weights differ by at most
$\varepsilon$ (are equal, for $\varepsilon = 0$); a divergent cycle
runs until the run's step budget is spent.

`max_iterations N` caps the number of agenda pops; the agenda
raises on exceeding it. Default is $10^5$.

## 6. Chart denotation

A [chart](https://www.nltk.org/book/ch08.html#well-formed-substring-tables) stores
partial analyses for reuse in larger analyses [@nltkSyntax2019]. In a weighted
chart, each entry aggregates the derivations of one item [@goodman1999semiring].
Within a derivation,
combine the premise contributions and rule weight using $\otimes$; across
derivations, combine alternatives using $\oplus$. This is why switching a
semiring can reuse the chart algorithm while changing its answer from
“possible?” to “how many?” or “what total probability?”. For cyclic systems,
“all derivations” can be an infinite collection, and the convergence
conditions are part of the semantic claim. In a finite computation, [chart entries and their gradients](../guides/deduction.md#charts-as-first-class-differentiable-values)
can be inspected to identify which rules contribute to a result.

Fix the deduction `D = (I, Σ, K, ax, goal, s, d)` and an input
`w`. The *chart* is the function

$$
\alpha : I \to K, \qquad
\alpha[i] = \bigoplus_{\text{deriv } d : i_1, …, i_k \vdash i}
            \bigotimes_{\ell=1}^{k} \alpha[i_{\ell}]
$$

: the `K`-join over all `Σ`-derivations of `i`, of the product
of the children's weights. The base case is the axiom injector:
`α[i] = w` for every `(i, w) ∈ ax(input)`, all other unproven
items at `⊥`.

The denotation of the deduction is the goal weight:

$$
\llbracket D \rrbracket(w) = \bigoplus_{i ∈ \mathrm{goal}} α[i].
$$

When `K = LogProb` this is the inside log-probability of the
input under the rule system; when `K = Viterbi` it is the
best-derivation score; when `K = Boolean` it is the membership
predicate `w ∈ L(D)`.

## 7. Strategy independence

For a complete lattice of chart values and a monotone rule operator, the
Knaster–Tarski theorem gives a least fixed point. In an acyclic finite chart,
the same values can be computed directly in dependency order. Cyclic weighted
charts need additional assumptions that make their potentially infinite sums
and fixed-point evaluation well defined.

The agenda implements a schedule for evaluating rule instances. A strategy is
a tuple `(Agenda, π, stop)`, where `Agenda` supplies a queue discipline
(FIFO / LIFO / priority), `π` a priority function, and `stop` a termination
predicate. Strategy independence requires a correct schedule for the chosen
algebra: necessary deductions must be reached, repeated deductions must be
accounted for correctly, and termination must preserve the requested answer.
Under these conditions, the semiring interpretation assigns the same chart
values independently of scheduling [@goodman1999semiring]. Pruning removes
derivations from the sum, and early stopping can leave their contributions
unprocessed. The runtime picks a default
strategy from rule arities and the semiring's algebraic properties
(CKY-sweep for context-free + idempotent, A\* for weighted +
idempotent + admissible heuristic, semi-naive for Datalog-shaped,
Viterbi for `(max, ⊗)`, depth-first for proof-search-shaped).

## 8. Charts as first-class differentiable values

The chart's underlying weight storage is a `torch.Tensor` (dense)
or `dict[Item, torch.Tensor]` (sparse) with `requires_grad`
flowing from rule-weight parameters through the agenda's
`semiring.times` / `semiring.plus` operations.

- **Finite-iteration** deductions (no fixpoint cycles): autodiff
  through the bounded number of semiring operations is automatic.
- **Fixed-point** deductions (cyclic): the gradient `∂C^* / ∂θ`
  is solved via the implicit-function theorem at convergence
  (`torch.linalg.solve` over `∂F/∂C^*`).
- **Inside-outside-friendly** semirings (commutative + idempotent
  + with inverses, e.g. `LogProb`, `Viterbi`): the runtime emits
  an analytic outside computation, related to weighted dynamic programming [@eisner2005compiling], as a
  second agenda pass that propagates outside weights from the
  goal back to all items.

## 9. Chart access from program bodies

A `program` body may invoke a compiled deduction $D$ by name
through the `parse` builtin:

```
let chart = parse(D, sentence)
```

The right-hand side denotes a *chart-view* object whose API
exposes the chart's K-presheaf as differentiable measurable maps
on the surrounding trace context:

| Method | Denotation |
|---|---|
| `chart.weight(item)` | $\alpha[i]$ at the ground item $i$ (raises if absent). |
| `chart.try_weight(item, default)` | $\alpha[i]$ or `default` or the semiring's zero. |
| `chart.enumerate(pattern)` | the indexed family $\{(i, \alpha[i]) \mid \pi \sim i\}$ for items matching pattern $\pi$. |
| `chart.goal_weight()` | $\bigoplus_{i \in \mathrm{goal}} \alpha[i]$, the deduction's inside-marginal. |
| `chart.embedding(item)` | the vector embedding of an item under the attached encoder, if any. |
| `chart.attached_loss` | the cumulative loss declared on attached `loss` blocks. |
| `chart.semiring` | the chart's underlying semiring $K$. |

Each of these returns a `torch.Tensor` carrying gradients
through the deduction's learnable parameters (lexicon log-weights
and the rule-weight side-table from [§2.1](#21-learnable-rule-weights-learnable)).

The natural lift into the program's log-joint is via a
[`score` step](programs.md#210-score-factor):

```
let chart = parse(D, sentence)
score log_Z = chart.goal_weight()
```

The body's denotation extends the program's log-joint by the
inside log-marginal of the input under the deduction's current
parameters: SVI and NUTS over the program's parameter set then
descend to the deduction's parameters automatically.

### 9.1 Composing deductions

The let-expression builtin `compose(D_1, D_2)` builds a deduction
system whose axiom injector chains $D_1$'s goal items into
$D_2$'s axiom set:

$$
\mathsf{ax}_{D_1 \mathbin{\mathrm{c}} D_2}(\mathit{inp})
\;=\;
\mathsf{goal}\_\mathsf{items}\bigl(D_1(\mathit{inp})\bigr)
\;\cup\;
\mathsf{ax}_{D_2}(\mathit{inp}).
$$

The result's semiring, agenda, and goal are inherited from
$D_2$; the parameters of *both* factors appear in the composed
system's `.parameters()` via cross-attachment of the
`_axiom_module` / `_rule_module` side-tables. A composed
deduction is thus directly usable inside `parse(..., …)`
calls without any further wrapper.

The composition is associative up to chart equality of the
intermediate goal-item set, and identity-on-the-right under the
trivial deduction with no rules.

### 9.2 Forward sampling

The [`quivers.deduction.sample_corpus`](../api/stochastic/deduction/sample.md#quivers.stochastic.deduction.sample.sample_corpus)
helper realizes exact forward sampling from a deduction's
distribution over yields: it enumerates the deduction's surface
vocabulary to a chosen length, evaluates $\alpha[\mathsf{goal}]$
for every candidate yield, and draws categorically from the
softmax of the resulting log-weights.

### 9.3 Let-expression builtins

The let-expression compiler ships three deduction-related
builtins, all callable from inside a `program` or lexicon LF
body:

| Builtin | Signature | Denotation |
|---|---|---|
| `parse(D, x)` | `(DeductionSystem, Input) -> ChartView` | invoke $D$ on input $x$. |
| `compose(D_1, D_2)` | `(DeductionSystem, DeductionSystem) -> DeductionSystem` | as in [§9.1](#91-composing-deductions). |
| `subst(t, v, w)` | `(Term, Term, Term) -> Term` | structural substitution; capture-avoiding under the [`binders` invariant](#3a-binders-and-alpha-renaming). |

`subst(t, v, w)` walks $t$ replacing every subterm structurally
equal to $v$ with $w$. Because every binder application has
been alpha-renamed to a fresh canonical symbol at lexicon-LF
compile time, naive structural substitution is already
capture-avoiding: no two distinct bound variables share a
canonical name in any reachable term.

## 10. Program-fragment integration

The Bayesian-modeling step kinds, effect signatures, the
`over`-modifier, the operadic-contraction call, and the
transformation-composition operator introduce additional
productions in the QVR grammar. The shapes below mirror the
tree-sitter source at `grammars/qvr/grammar.js`; semantics is
given in [Programs](programs.md), [Expressions](expressions.md),
and [Composition Rules](composition-rules.md).

```ebnf
# The unified option block is the same production every declaration
# kind (morphism, kernel, contraction, deduction, lexicon-from-file,
# pragma) reads. Values are bare identifiers (flags), identifiers
# (named options), numeric / string literals, comma-separated lists,
# or single function-call expressions.
option_block        := '[' option_entry (',' option_entry)* ']'
option_entry        := IDENT [ '=' option_value ]
option_value        := IDENT
                     | INT | FLOAT | STRING
                     | '[' option_value (',' option_value)* ']'
                     | IDENT '(' option_value (',' option_value)* ')'

# Outer / inner pragmas. The outer form decorates the next
# declaration; the inner form decorates the enclosing module.
pragma_outer        := '#['  pragma_entry (',' pragma_entry)* ']'
pragma_inner        := '#!['  pragma_entry (',' pragma_entry)* ']'
pragma_entry        := IDENT [ '=' option_value ]

deduction_decl      := 'deduction' IDENT ':' type_expr '->' type_expr
                       [ option_block ]
                       deduction_body_entry*

deduction_body_entry
                    := deduction_atoms
                     | deduction_binders
                     | deduction_rule
                     | deduction_lexicon
                     | deduction_lexicon_from_file

deduction_atoms     := 'atoms' IDENT (',' IDENT)*
deduction_binders   := 'binders' IDENT (',' IDENT)*

deduction_rule      := 'rule' IDENT ':' premise_list ('|-' | '⊢')
                       conclusion [ rule_pragma ]
rule_pragma         := '#[' rule_pragma_entry (',' rule_pragma_entry)* ']'
rule_pragma_entry   := 'learnable'
                     | 'bounded'
                     | 'parent' '=' IDENT

# Deduction-level options recognized by the agenda compiler;
# extend the standard ``[k=v, ...]`` option list.
deduction_option    := 'semiring'   '=' IDENT
                     | 'start'      '=' IDENT
                     | 'depth'      '=' INT
                     | 'tolerance'  '=' NUMBER
                     | 'max_iterations' '=' INT
                     | 'axioms'     '=' IDENT
                     | 'signature'  '=' IDENT
                     | 'encoder'    '=' IDENT

typed_program_param := IDENT ':' param_kind
param_kind          := object_kind | scalar_kind | morphism_kind
object_kind         := 'FinSet' | 'Space' | 'Object'
scalar_kind         := 'Real'   | 'Nat'
morphism_kind       := 'Mor' '[' type_expr ',' type_expr ']'

effect_set          := effect (',' effect)*
effect              := 'Sample' | 'Score' | 'Marginal' | 'Pure'

bind_step           := var_pattern [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]

observe_step        := 'observe' IDENT [ ':' type_expr ]
                       '<-' IDENT [ '(' draw_arg_list ')' ]
                       [ option_block ]
# Option-block entries recognized on observe steps include
# ``via = idx`` (a fibration into a marginalize grouping plate)
# and any family-specific keyword overrides.

score_step          := 'score' IDENT '=' let_arith

call_step           := 'let' IDENT '<-' qiec_call
# ``let x <- helper[Real](a, 2.0)`` binds the result of a named
# computation declared with ``define``; the callee's effect row joins
# the program's. A call step runs only through the checked QIEC
# module, since the runtime compiler has no computation to invoke.
# A call step that names a program with open input extents writes no
# static arguments: the extents are read off the arguments' types.

marginalize_step    := 'marginalize' IDENT [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]
                       [ option_block ]
                       NEWLINE INDENT program_step+ DEDENT
# Option-block entries on marginalize steps include
# ``over = G`` (or ``over = product(G_1, …)`` / ``over = [G_1, …]``
# for a product grouping plate) and ``reduction = R`` selecting
# the aggregation (R ∈ {logsumexp, sum, mean}, default logsumexp).
# Each observe step inside the body carries its own
# ``via = idx`` option naming the fibration into the shared plate;
# the runtime scatter-sums each per-(N_m, K) log-likelihood into
# the same (|G|, K) accumulator before the reduction.

let_index           := IDENT '[' let_arith (',' let_arith)* ']'

program_decl        := 'program' IDENT [ '(' param_list ')' ]
                       ':' type_expr '->' type_expr
                       [ option_block ]
                       NEWLINE INDENT program_step* return_step DEDENT
# Option-block entries on program declarations include
# ``effects = [E_1, ..., E_n]`` (the declared effect signature; each
# E_i is one of {Sample, Score, Marginal, Pure}) and
# ``over = M`` (posterior consumer over model M's latents).

# Top-level composition-rule selection. The single `composition`
# keyword takes an optional `[level=LEVEL]` option naming the declared
# algebraic level; the optional indented body declares a fresh
# rule inline.
composition_decl    := 'composition' IDENT
                       [ option_block ]
                       [ NEWLINE INDENT composition_rule_entry+ DEDENT ]

composition_level   := 'algebra' | 'semigroupoid'
                     | 'bilinear_form' | 'rule'

composition_rule_entry
                    := IDENT '(' IDENT (',' IDENT)* ')' '=' let_expr
                     | IDENT '=' let_expr

# Operadic n-ary contraction declaration.
contraction_decl    := 'contraction' IDENT
                       '(' contraction_input (',' contraction_input)* ')'
                       ':' type_expr '->' type_expr
                       'rule' IDENT
                       'wiring' STRING

contraction_input   := IDENT ':' type_expr '->' type_expr

# Transformation-valued expressions used inside change_base.
# Distinct from morphism composition `>>` (see Expressions § 2.7).
trans_compose       := expr '>>>' expr

# Method-call postfix on a morphism expression. The change_base
# argument is any expression that resolves to a transformation
# (singleton, constructor call, let-bound trans, or `>>>` chain).
change_base_postfix := expr '.' 'change_base' '(' expr ')'

# Operadic call site: `IDENT(arg_1, ..., arg_n)` resolves to a
# registered contraction or a parametric program template.
morphism_call       := IDENT '(' IDENT (',' IDENT)* ')'
```

A `program_decl` is *parametric* iff its parameter list contains any `typed_program_param`; the walker dispatches parametric programs to the call-site inliner rather than to the runtime program compiler. A program whose option block carries `[effects = [...]]` has its body checked against the declared capability set: the actual effects of the body must form a subset of the listed set, and `[effects = [Pure]]` rejects any `sample_step` / `observe_step` / `marginalize_step`. A program whose option block carries `[over = M]` is a posterior block consuming the latents of model `M`; the consumed latents appear as data parameters in the program's parameter list.

A `composition_decl` selects the module's underlying composition rule. With no body and no `[level=...]` option, the keyword resolves the named rule from the built-in catalog and registers it. With a `[level=LEVEL]` option but no body, the resolved built-in rule is verified to match the declared algebraic level (`algebra`, `semigroupoid`, `bilinear_form`, or `rule`, the last covering any `CompositionRule`). With a body, the entries declare the rule's operations inline; the `[level=LEVEL]` option fixes the algebraic level, and the compiler verifies that the required entries (`tensor_op`, `join`, plus `unit`, `zero` for `algebra`) are present. See [Composition Rules](composition-rules.md) for the denotation.

A `contraction_decl` declares an n-ary operadic morphism whose action contracts its input morphisms under the named composition rule using the wiring spec. Call sites `IDENT(arg_1, …, arg_n)` route through `morphism_call`; the compiler resolves `IDENT` against the contraction registry, the parametric-program template table, and the morphism scope in that order. See [Expressions § 2.13](expressions.md#213-operadic-contraction-call) for the call-site denotation.

## 11. QIEC fragment

The Quivers Indexed Effect Core (QIEC) adds a disjoint declaration and
computation fragment to the module grammar. Square-bracket telescopes bind
static type, index, or effect-interface variables; parenthesized telescopes bind
family indices or computation values. The following compressed EBNF records
the boundary, while `grammars/qvr/grammar.js` remains authoritative:

```ebnf
index_decl        := 'index' IDENT '=' index_constructor ('|' index_constructor)*
family_decl       := 'family' IDENT [static_telescope] [index_telescope]
                     ':' KIND constructor_decl+
constructor_decl  := 'constructor' IDENT [static_telescope] ':'
                     [type ('*' type)* '->'] type

effect_decl       := 'effect' IDENT [static_telescope] operation_decl+
operation_decl    := IDENT [static_telescope] ':' [type ('*' type)* '->'] type
instance_decl     := 'instance' IDENT ':' effect_ref
handler_decl      := 'handler' IDENT [static_telescope] 'for' effect_ref ':'
                     type '->' type [handler_options] handler_clause+
handler_clause    := return_clause | operation_clause
return_clause     := 'return' local '=>' computation
operation_clause  := IDENT [static_telescope] ['(' local (',' local)* ')']
                     'resumes' grade ['=>' computation]
grade             := '0' | 'aff' | '1' | 'omega'

computation_decl  := 'define' IDENT [static_telescope] ['(' value_params ')']
                     ':' type effect_row '=' computation
effect_row        := '!{' [IDENT (',' IDENT)*]
                     ['|' IDENT ['lacks' IDENT (',' IDENT)*]] '}'
computation       := 'return' value
                   | 'let' local '=' value computation
                   | 'let' local '<-' inline computation
                   | inline computation
                   | 'handle' IDENT 'with' handler_application 'in' computation
                   | 'with' 'instance' IDENT ':' effect_ref 'in' computation
                   | 'case' value 'motive' [index_telescope] '=>' type branch+
                   | 'if' expression 'then' computation 'else' computation
inline            := perform | call | resume
value             := 'construct' IDENT [static_arguments] '(' [value (',' value)*] ')'
                     'as' type
                   | expression
expression        := expression binop expression | ('-' | 'not') atom | atom
atom              := IDENT | NUMBER | STRING | 'true' | 'false' | 'unit'
                   | '(' expression ')'
                   | '(' expression ',' expression (',' expression)* ')'
                   | IDENT '[' INT ']'
                   | IDENT '(' [expression (',' expression)*] ')'
binop             := '||' | '&&' | '==' | '!=' | '<' | '<=' | '>' | '>=' | '+'
                   | '-' | '*' | '/' | '%'
perform           := 'perform' IDENT '.' IDENT [static_arguments]
                     '(' [value (',' value)*] ')'
call              := IDENT [static_arguments] '(' [value (',' value)*] ')'
resume            := 'resume' '(' [value] ')'
static_arguments  := '[' static_argument (',' static_argument)* ']'
static_argument   := type | INT
```

A static argument is type syntax, read as a type or an effect according to
the binder it fills, or an integer literal, which fills a `Nat` index
binder: `pad[3](xs)` applies a computation declared
`define pad[n : Nat](xs : Tensor[Real]([n])) ...` at extent three.

### Disambiguation

Three forms share the shape `NAME(...)`, and each is separated by a keyword
or a qualifier rather than by precedence:

- a **call** is a bare `NAME(...)` in computation position;
- an **effect request** is qualified by its instance, `inst.op(...)`, and
  introduced by `perform`; and
- a **constructor value** is introduced by `construct` and closed by
  `as TYPE`, and occurs in value position.

There is one call syntax. A computation is invoked by ordinary application,
and a handler clause resumes its continuation with `resume`, which is a
keyword rather than a bound name, so no second spelling can drift from the
first.

### Bindings

`let x = VALUE` binds a pure expression and `let x <- COMPUTATION` binds a
computation's result. They are separate productions with separate typing
rules, and neither is sugar for the other.

### Pure expressions

QIEC values and `program` let and score steps share one expression grammar.
Operators bind from loosest to tightest as `||`, `&&`, comparison, `+ -`,
`* / %`, then unary `-` and `not`; equal precedence associates left. A
comparison against a negative literal needs parentheses, `x < (-1)`, because
`<-` is the binding arrow.

In a QIEC value every operator resolves to one primitive of the closed
registry by the type of its operands, and both operands must have the same
type: `+` is integer, real, or string concatenation; `%` is integer only;
`==` and `!=` compare integers, reals, Booleans, or strings; `&&`, `||`, and
`not` take Booleans. Integer division and remainder truncate toward zero on
every target. The builtin applications `real`, `int`, `exp`, `log`, `sqrt`,
`pow`, `abs`, `min`, and `max` resolve the same way. A tuple `(a, b)` builds a
finite product and `t[i]` with an integer literal selects its component.
A list literal `[a, b, c]` is a `Tensor` whose leading dimension is its
entry count and whose entries share one type, so `[[1.0, 2.0], [3.0, 4.0]]`
is a `Tensor[Real]([2, 2])`; a numeric literal list with any fractional
entry is real. Applying a distribution family, `Normal(mu, 1.0)` or
`Dirichlet([1.0, 2.0])`, fills the family's parameters in registry order and
builds a `Sampleable`; `site("x")` is a `Site` at the type its position
expects; `log_prob(d, x)` is the `LogWeight` of `d` at `x`.
`if COND then ... else ...` branches on a Boolean at computation level, so the
untaken branch is never entered and a recursive computation can stop.

### Layout

Every construct that takes a nested computation opens an indented block after
a trailing `=>` or `in`: handler clauses, case branches, `handle ... in`, and
`with instance ... in`. An inline computation, that is a call, a request, or
a resumption, is one line and is followed by the rest of the computation at
the same indentation.

### Authored and foreign handlers

An operation clause with a `=>` body is authored. A clause without one is a
signature, and its declaration must say `[implementation=foreign]`, so a
missing body never silently means "look up a callback at runtime". The
complete grammar, worked examples, and implementation limits appear in the
[QIEC developer note](../developer/qiec.md).

## References

Parsing as deduction is developed in [@pereira1983parsing; @shieber1995principles], and weighted alternatives in [@goodman1999semiring]. The hypergraph view [@klein2001parsing] explains how deductions share intermediate results. For priority-based evaluation, see [@knuth1977generalization; @nederhof2003weighted]. For compiled weighted dynamic programs and complexity analysis, see [@eisner2005compiling; @mcallester2002complexity]. These references provide the algorithmic background; the restrictions and evaluation schedule of QVR are specified above.
