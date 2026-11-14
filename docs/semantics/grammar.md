# Weighted deduction

A `deduction` declaration denotes a weighted deduction system: a set of items, inference rules
over them, a semiring of weights, the axioms an input gives, and the goal items
[@shieber1995principles; @goodman1999semiring; @nederhof2003weighted]. Its denotation is a
function from inputs to weights: the chart assigns each item the $\oplus$-sum, over its
derivations, of the $\otimes$-product of the weights each derivation uses, and the value is the
sum of the chart over the goal items. The [Weighted deduction systems guide](../guides/deduction.md)
shows the declarations in use. The same form expresses parsers, proof search, Datalog programs,
and graph algorithms, which differ in their items, rules, and semiring.

A *chart* is the table of items proved so far and their weights, and an *agenda* is the queue of
items still to be processed [@eisner2005compiling]. A chart lets a derivation reuse an item
another derivation has already proved. In a graph, edges $a \to b$ and $b \to c$ of costs $2$ and
$3$ derive a path $a \to c$ of cost $5$, and a direct edge of cost $4$ derives it again: under
the tropical semiring the chart keeps $4$, and under the counting semiring it counts two
derivations.

## 1. Items {#1-item-algebra}

The `atoms` clause of a `deduction` declares the constructor symbols. An item is a term over
them: a symbol applied to items, `c(a_1, ..., a_n)`, with nullary symbols as constants. The set
$I$ of items is the [term algebra](https://en.wikipedia.org/wiki/Term_algebra) over the declared
symbols, and two items are equal when they are the same term. The keyword `atoms` names these
symbols, which logic programming would call function symbols.

A rule's patterns may contain variables, which range over items, and rules fire under the
substitutions that [unification](https://en.wikipedia.org/wiki/Unification_(computer_science))
finds for them. An identifier in a rule is a variable exactly when it is neither a declared symbol
nor a numeral, whatever its spelling, and single uppercase letters are the convention. A declared
symbol always denotes itself, in the agenda's chart and in the search computation of the
[Quivers Indexed Effect Core](../reference/qvr/index.md) (QIEC) alike, so a
variable must not be spelled like a symbol. A symbol is a valid ground pattern in a category slot.
In a chart position slot (§2.1) it is a compile error naming the rule and the identifier, since a
position slot holds a position variable or an integer.

## 2. Rules

A rule declaration

```
rule r : π_1, ..., π_m |- π
```

states the inference rule

$$
\frac{\pi_1\sigma \quad \cdots \quad \pi_m\sigma}{\pi\sigma} \quad (r)
$$

for every substitution $\sigma$ of items for its variables: when every premise instance
$\pi_\ell\sigma$ is in the chart, the conclusion $\pi\sigma$ is derived. A rule with several
premises is a hyperedge of a [hypergraph](https://en.wikipedia.org/wiki/Hypergraph), an edge from
several premises to one conclusion [@klein2001parsing]. It needs all its premises at once, so it
is not a set of separate rules with one premise. Each instance has a weight $\beta_r(\sigma) \in K$ in
the deduction's semiring $K$, which is $\mathbf 1_K$ unless the rule is learnable or weighted.

### 2.1 Learnable rule weights (`#[learnable]`) {#21-learnable-rule-weights-learnable}

A rule marked `#[learnable]`,

```
rule r : π_1, ..., π_m |- π #[learnable]
```

has a learnable weight that depends on some of its variables. The variables of a rule are of two
kinds. A *position variable* fills a chart position: one of the first two slots of `span` in a
deduction with a lexicon (whose axioms are the items `span(i, i + 1, c)`), a slot that an integer
literal fills, or a slot that a variable shared with a position slot fills in the same rule,
closed under this last condition. Every other variable is a *category variable*, ranging over
categories, logical forms, or other items. Write $P(r)$ for the position variables of $r$ and
$C(r)$ for its category variables. A firing with substitution $\sigma$ gives its conclusion the
weight

$$
\bigotimes_{\ell=1}^{m} \alpha\bigl(\pi_\ell\sigma\bigr) \otimes \theta_{r, \sigma|_{C(r)}},
$$

where $\alpha$ is the chart, and $\theta_{r, \kappa} \in K$ is one learnable scalar for each
restriction $\kappa$ of the substitution to the category variables. Positions never enter $\kappa$,
so a rule has the same weight at every position of every input, and the number of parameters does
not grow with the length of the input. The weights are parameters of a PyTorch module owned by the
deduction, and the weight of a category binding is created the first time a firing produces it.

- A rule with no category variable, such as `np : span(I, K, Det), span(K, J, N) |- span(I, J, NP)`,
  has one weight $\theta_r$, the weight of a production of a weighted context-free grammar. It is
  created when the deduction is compiled.
- A rule with category variables, such as
  `fwd : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X)`, has a separate weight for each
  ground value of $(X, Y)$.

A lexicon entry marked `#[learnable]` has one weight per entry, and one per candidate category for a
wildcard or restricted category, shared by every position of its word. The checked QIEC
elaboration reads the same weights from its parameter store under the names `<D>.rule.<r>:<κ>` and
`<D>.lex.<i>`, with $\kappa$ the category bindings in the order of their variable names joined by
`|` (empty for a rule with no category variables), so the agenda and the QIEC search share a weight
between the same firings.

### 2.2 Parent rules (`parent=`) {#22-parent-rule-specialization-parent}

A rule may name a parent rule:

```
rule r' : π'_1, ..., π'_m |- π' #[learnable, parent=r]
```

Its weight is then the product, in $K$, of its own learnable weight and the parent's weight under
the same category bindings:

$$
\bigotimes_\ell \alpha\bigl(\pi'_\ell\sigma\bigr) \otimes \theta_{r', \kappa} \otimes \theta_{r, \kappa},
\qquad \kappa = \sigma|_{C(r')}.
$$

Parent links form chains that end at a rule with no parent, and cycles are rejected. In the log
semiring the weights add, so the child's weight is a correction to its parent's, a form of
additive hierarchical sharing of parameters.

### 2.3 Bounded rule weights (`bounded`) {#23-bounded-rule-weights-bounded}

A learnable rule with `bounded` in its pragma has the weight

$$
\theta_{r, \kappa} = -\mathrm{softplus}(\mathit{raw}_{r, \kappa}) - \log n_b,
$$

where $\mathit{raw}_{r, \kappa}$ is the unconstrained parameter and $n_b$ is the number of bounded
rules in the deduction. Each firing of a bounded rule thus multiplies by less than $1 / n_b$. When
the bounded rules form cycles of unary rules, each giving at most one conclusion per item, the
total weight one item passes on through all bounded rules is below $1$. Under a semiring that is not
idempotent, such as `LogProb` or `Counting`, the weight of an item on a cycle is the sum of a series
over every number of trips around the cycle, the Kleene star $a^* = \bigoplus_{n \ge 0} a^n$ of the
cycle weight [@mohri2002semiring], and this bound makes the series converge for every value of the
parameters (§5). A bound of $1$ on each rule separately would not suffice: interlocking cycles, such
as an introduction and an elimination rule over nested constructors, can diverge when each single
cycle has weight below one.

### 2.4 Production probabilities (`normalize=lhs`) {#24-production-probabilities-normalizelhs}

With the deduction option `normalize=lhs`, the rules and the lexicon entries are the productions $p$
of a grammar, each with a left-hand side $\mathrm{lhs}(p)$: the category of a `span` conclusion or of
a lexical axiom, and otherwise the conclusion with its position slots erased. Each production has a
log weight $\phi_p$, its learnable parameter or $0$ when it is not learnable, and is used with the
weight

$$
\theta_p = \phi_p - \log \sum_{q \,:\, \mathrm{lhs}(q) = \mathrm{lhs}(p)} e^{\phi_q},
$$

so that $\sum_{q : \mathrm{lhs}(q) = A} e^{\theta_q} = 1$ for every left-hand side $A$, and the
deduction is a [probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar)
(PCFG) [@booth1973applying]. The groups must be fixed by the declaration: no rule has a category
variable, `parent`, or `bounded`, the lexicon categories are ground, items carry no logical forms,
the axioms come from a lexicon, and the semiring is `LogProb` or `Viterbi`. Any other deduction with
this option is a compile error. The checked elaboration computes the same $\theta_p$ from the
parameter reads of each group.

## 3. Semirings

The `semiring` option chooses $K$:

| `semiring` | $\oplus$ | $\otimes$ | Chart value at the goal |
|-------------|-----------------|--------------------|-----|
| `LogProb` (default) | log-sum-exp | $+$ | log of the total weight of all derivations |
| `Viterbi` | $\max$ | $+$ | log weight of the best derivation |
| `Boolean` | or | and | whether a derivation exists |
| `Counting` | $+$ | $\times$ | number of derivations |

`LogProb` is the log semiring, `Viterbi` the max-plus semiring, and `Counting` the counting semiring
([Semirings, t-norms, and change of base](algebras.md)). They are the instances of the
class `ChartSemiring` that a deduction accepts, a different class from the `Algebra` instances that compose morphisms. The
same rules give the answers in the last column under the different semirings, as in semiring
parsing [@goodman1999semiring]. A value of `semiring` outside these four is read as `LogProb`
without an error.

## 4. Binders, logical forms, and axioms

### 4.1 Binders {#3a-binders-and-alpha-renaming}

A deduction may list constructors whose first argument is a bound variable:

```
deduction D : A -> B
    atoms NP, S, Bwd, App, Var, bark
    binders Lam
    ...
```

For $c \in \mathsf{binders}(D)$, the items $c(x, t)$ and $c(y, t[x := y])$ for a fresh $y$ are
[alpha-equivalent](https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence): they differ
only in the name of the bound variable. The compiler renames every bound variable of a lexicon
logical form to a fresh symbol when it builds the form,

$$
c(x, t) \longmapsto c(\mathsf{fresh}, t[x := \mathsf{fresh}]),
$$

with $\mathsf{fresh}$ drawn from a pool owned by the deduction. Two lexicon entries that both write
`Lam(x, body)` thus build items with different bound names, and equality of items in the chart
agrees with alpha-equivalence of the source terms. The bound names of the lexicon are added to the
symbols of the deduction while its logical forms are compiled, so they need not be listed in
`atoms`. A use of a bound name outside its binder is rejected, unless the same name is also a
declared constant.

The runtime represents the two kinds of leaf differently:

| Source term | Runtime term |
|--------------|--------------|
| constant `dog_p` or `dog_p()` | `("atom", "dog_p")` |
| bound variable `x` | `("var", "#v1")` after renaming |
| application `App(f, x)` | `("App", f_term, x_term)` |

A rule pattern that mentions `dog_p` has the same runtime shape as `dog_p` in a lexicon, inline or
read from a file. The tags `atom` and `var` are reserved and cannot be declared in `atoms`. Since
every bound variable has a name of its own, substituting by structure,
[`subst(t, v, w)`](#93-let-expression-builtins), does not capture variables.

### 4.2 The shape of span items {#3b-item-shape-and-pattern-arity}

A lexicon produces axioms with the lexical category in the fourth slot, and with the logical form
(LF) in a fifth slot when some rule binds an LF or the deduction declares `binders`:

$$
(\mathsf{span}, i, j, \mathit{cat}) \quad\text{or}\quad (\mathsf{span}, i, j, \mathit{cat}, \mathit{lf}).
$$

The compiler decides per deduction whether items carry LFs, by looking for a pattern
`span(I, J, C, F)` with four arguments in its rules. In a deduction with LFs, a pattern
`span(I, J, C)` is extended with a fresh LF variable, so both forms match the same items.

### 4.3 Axioms {#4-axiom-injector}

The axioms map an input to the weighted items that start the chart. They are given in one of three
ways:

- a `lexicon` block of entries `"word" : Cat = lf #[learnable]`, looked up by word;
- `lexicon from "path.tsv"`, the same table read from a file, whose logical forms get the same
  tagging of constants, renaming of binders, and scope checks as inline entries;
- the option `axioms = m`, naming a declared morphism from the input to weighted items.

## 5. Goal, depth, and convergence

The deduction options `start`, `depth`, `tolerance`, and `max_iterations` (in the option block of
the declaration, as `[start=S, depth=2]`) fix the goal and bound the evaluation.

`start = s` declares the start symbol. The goal items are the conclusions $c(s, \dots)$ of the
configured form, such as `span(0, n, s)` over the whole input of length $n$.

`depth = k` bounds the depth of the category in a rule's conclusion, counting each constructor
application as one level. A `span` item has the depth of its category, and positions, constants,
bound variables, and integer literals have depth zero. The compiler adds to every rule the side
condition that rejects a firing whose conclusion is deeper than $k$. This bounds the categories that
cycles of unary introduction and elimination rules (modal, continuation, residuation) can build. It
does not bound the number of rules a derivation uses: a sentence whose derivations use many rules is
still parsed in full. The checked QIEC search applies the same measure to the same conclusions.

`tolerance = ε` controls the evaluation of cyclic rule systems. Under a semiring that is not
idempotent, every new derivation of an item adds to its weight, so on a cycle the weight is the
limit of a series (§2.3). The agenda stops re-deriving an item once an update changes its weight by
less than $\varepsilon$. When the series converges, as it does under the bound of `bounded`, this
reaches its limit to within the tolerance. When it diverges, the agenda reaches `max_iterations`
and raises an error. With $\varepsilon = 0$, the default, any change in a stored weight counts as an
update, so on a cycle the agenda stops only when the weights stop changing in floating point or when
`max_iterations` is reached. The checked QIEC search computes the same limit by enumeration: it bounds how often a
chain of unary rules may revisit an item and raises the bound until two successive goal weights
differ by at most $\varepsilon$.

`max_iterations = N` bounds the number of items taken from the agenda, $10^5$ by default, and the
agenda raises an error when it is exceeded.

## 6. Chart denotation {#6-chart-denotation}

Write $D = (I, R, K, \mathrm{ax}, \mathrm{goal})$ for the deduction's items, rules, semiring, axioms,
and goal, and fix an input $x$. The *chart* is the function $\alpha : I \to K$ with

$$
\alpha(i) = \bigoplus_{\delta \in \mathrm{Der}(i)} \mathrm{wt}(\delta),
$$

where $\mathrm{Der}(i)$ is the set of derivations of $i$ from the axioms $\mathrm{ax}(x)$, and the
weight $\mathrm{wt}(\delta)$ of a derivation is the $\otimes$-product of the weights of the axioms and
rule instances it uses. Equivalently, $\alpha$ satisfies

$$
\alpha(i) = a(i) \oplus \bigoplus_{(r, \sigma) \,:\, \pi\sigma = i} \beta_r(\sigma) \otimes \bigotimes_{\ell=1}^{m} \alpha(\pi_\ell\sigma),
$$

where $a(i)$ is the weight of $i$ among the axioms ($\mathbf 0_K$ if it is not an axiom), and is
$\mathbf 0_K$ on items with no derivation. When $\mathrm{Der}(i)$ is infinite, as on a cycle, the sum
is the limit of §5, and the equation holds when that limit exists. The denotation of the deduction is

$$
\llbracket D \rrbracket(x) = \bigoplus_{i \in \mathrm{goal}} \alpha(i).
$$

Under `LogProb` it is the log of the total weight of the derivations of the input, which for a PCFG
is the log [inside probability](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm)
[@baker1979trainable]. Under `Viterbi` it is the log weight of the best derivation, and under
`Boolean` whether the input is in the language.

## 7. Independence of the evaluation strategy {#7-strategy-independence}

The agenda processes rule instances in some order. For an acyclic system, every evaluation that
processes each item after all items it depends on, and combines every derivation once, computes the
chart of §6, whatever the order [@goodman1999semiring]. For a cyclic system under an idempotent
semiring such as `Boolean` or `Viterbi`, the chart is the least fixed point of the equation of §6,
which exists by the [Knaster–Tarski theorem](https://en.wikipedia.org/wiki/Knaster%E2%80%93Tarski_theorem)
when the weights form a complete lattice and the rules are monotone. For a cyclic system under a
semiring that is not idempotent, the chart is the limit of §5, which depends on convergence. Pruning
an item removes its derivations from the sum, and stopping early leaves some unprocessed, so neither
preserves the chart.

The compiler chooses the agenda from the semiring and the rules: a depth-first agenda for `Boolean`
when some rule has one premise, semi-naive evaluation for `Boolean` otherwise, and a bottom-up CKY
agenda for every other semiring. The runtime class `DeductionSystem` accepts other agendas,
including a priority agenda for best-first and A* search [@knuth1977generalization; @klein2003astar].

## 8. Gradients

The chart stores its weights as PyTorch tensors, dense for span charts and in a dictionary keyed by
item otherwise. Gradients flow from the learnable weights of §2 through every $\oplus$ and $\otimes$ the
agenda performs. For a cyclic system, the gradient is that of the computed approximation, obtained
by differentiating through every iteration up to convergence. The gradient of the log inside weight
with respect to the weight of a rule instance is its expected count, which gives the outside weights
of the [inside–outside algorithm](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm)
by backpropagation [@eisner2016inside].

## 9. Charts in program bodies {#9-chart-access-from-program-bodies}

A `program` body evaluates a compiled deduction $D$ with the `parse` builtin:

```
let chart = parse(D, sentence)
```

The value is a chart, whose methods return tensors that carry gradients through the deduction's
learnable weights (§2.1):

| Method | Value |
|---|---|
| `chart.weight(item)` | $\alpha(i)$ at the ground item $i$, an error if $i$ is not proved |
| `chart.try_weight(item, default)` | $\alpha(i)$, or `default` if $i$ is not proved, or $\mathbf 0_K$ if `default` is omitted |
| `chart.enumerate(pattern)` | the pairs $(i, \alpha(i))$ for the proved items $i$ that match the pattern |
| `chart.goal_weight()` | $\bigoplus_{i \in \mathrm{goal}} \alpha(i)$ |
| `chart.embedding(item)` | the vector of an item under the attached encoder, if there is one |
| `chart.attached_loss` | the sum of the losses declared in attached `loss` blocks |
| `chart.semiring` | the semiring $K$ |

A [`score` step](programs.md#210-score-factor) adds the goal weight to the program's log joint
density:

```
let chart = parse(D, sentence)
score log_Z = chart.goal_weight()
```

Under `LogProb` the score is the log inside weight of the sentence under the deduction's current
weights, and stochastic variational inference (SVI) and the No-U-Turn Sampler (NUTS) over the program's parameters reach the deduction's weights through it.

A corpus of independent sentences of one length is one input of one run, a table with one sentence
for each element of a plate $P$:

```
let chart = parse(D, corpus, P)
score log_L = chart.goal_weight()
```

and `chart.goal_weight()` is then $\bigotimes_{n \in P} \bigoplus_{i \in \mathrm{goal}_n} \alpha_n(i)$,
which under `LogProb` is the log likelihood of the corpus, $\sum_n \log Z(s_n)$, with $Z(s)$ the inside
weight of the sentence $s$.

For a grammar normalized by left-hand side (§2.4), the call may bind the production probabilities of
each left-hand side $A$ with more than one production to a program value $p_A$, a vector over the
productions of $A$ in the order `production_groups` lists:

```
let chart = parse(D, corpus, P, ("A", p_A), ("B", p_B))
```

The chart then uses $\theta_q = \log p_{A,q} - \log \sum_{q'} p_{A,q'}$ for each production $q$ of $A$,
which is $\log p_{A,q}$ when $p_A$ lies on the simplex, in place of the learned weights, and a
production alone in its group has $\theta_q = 0$. With $p_A \sim \mathrm{Dirichlet}(\alpha_A)$ drawn in
the program, the program's log joint density is
$\sum_A \log \mathrm{Dir}(p_A ; \alpha_A) + \sum_n \log Z(s_n ; p)$, that of a Bayesian PCFG
[@johnson2007bayesian]. The compiler checks the binding when it lowers the program: each bound label
is a left-hand side of several productions, each such left-hand side is bound once, each value is a
vector of its group's size, and a value drawn at a sample site comes from a family whose support is
the simplex.

### 9.1 Composing deductions {#91-composing-deductions}

`compose(D_1, D_2)` builds a deduction whose axioms are the goal items of $D_1$ together with the
axioms of $D_2$:

$$
\mathrm{ax}_{\mathsf{compose}(D_1, D_2)}(x) = \mathrm{goalitems}\bigl(D_1(x)\bigr) \cup \mathrm{ax}_{D_2}(x).
$$

Its rules, semiring, agenda, and goal are those of $D_2$, and its parameters are those of both, so
the result may be passed to `parse` and fitted as one system. No associativity or identity law is
claimed for `compose`.

### 9.2 Sampling sentences

[`sample_corpus`](../api/stochastic/deduction/sample.md) draws sentences of one given length. It
enumerates every string of that length over the deduction's vocabulary, computes the goal weight of
each, and samples from the softmax of these log weights. Under `LogProb` this samples exactly from the
distribution of sentences conditional on their length, and the cost grows exponentially with the
length.

### 9.3 Builtins {#93-let-expression-builtins}

| Builtin | Type | Value |
|---|---|---|
| `parse(D, x)` | `(DeductionSystem, Input) -> Chart` | the chart of $D$ on $x$ |
| `compose(D_1, D_2)` | `(DeductionSystem, DeductionSystem) -> DeductionSystem` | §9.1 |
| `subst(t, v, w)` | `(Term, Term, Term) -> Term` | $t$ with each subterm equal to $v$ replaced by $w$ |

`subst` replaces subterms by structure. Because every binder of a lexicon logical form has been renamed
to a fresh symbol (§4.1), no two different bound variables share a name, and the replacement does not
capture a variable.

## 10. Grammar of the program and composition forms {#10-program-fragment-integration}

The forms below are written in EBNF after the tree-sitter grammar `grammars/qvr/grammar.js`, which is
authoritative. Their semantics is on [Programs as kernels](programs.md),
[Expressions](expressions.md), and [Composition rules](composition-rules.md).

```ebnf
# The option block is one production read by every declaration and step
# that takes options. A value is a name, a number, a string, a list, or
# one call.
option_block        := '[' option_entry (',' option_entry)* ']'
option_entry        := IDENT [ '=' option_value ]
option_value        := IDENT
                     | INT | FLOAT | STRING
                     | '[' option_value (',' option_value)* ']'
                     | IDENT '(' option_value (',' option_value)* ')'

# The outer pragma decorates the next declaration, the inner pragma the module.
pragma_outer        := '#['  pragma_entry (',' pragma_entry)* ']'
pragma_inner        := '#!['  pragma_entry (',' pragma_entry)* ']'
pragma_entry        := IDENT [ '=' option_value ]

deduction_decl      := 'deduction' IDENT ':' type_expr '->' type_expr
                       [ option_block ] NEWLINE
                       INDENT deduction_body_entry+ DEDENT

deduction_body_entry
                    := deduction_atoms
                     | deduction_binders
                     | deduction_rule
                     | deduction_lexicon
                     | deduction_lexicon_from_file

deduction_atoms     := 'atoms' IDENT (',' IDENT)*
deduction_binders   := 'binders' IDENT (',' IDENT)*

deduction_rule      := 'rule' IDENT ':' premise_list ('|-' | '⊢')
                       conclusion [ '#[' pragma_entry (',' pragma_entry)* ']' ]
# Rule pragma entries: learnable, bounded, parent = IDENT, weight = expr.

# Deduction options: semiring, start, depth, tolerance, max_iterations,
# axioms, signature, encoder, normalize.

typed_program_param := IDENT ':' param_kind
param_kind          := object_kind | scalar_kind | morphism_kind
object_kind         := 'FinSet' | 'Space' | 'Object'
scalar_kind         := 'Real'   | 'Nat'
morphism_kind       := 'Mor' '[' type_expr ',' type_expr ']'

effect_set          := effect (',' effect)*
effect              := 'Sample' | 'Score' | 'Marginal' | 'Pure'

sample_step         := 'sample' var_pattern [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ] [ option_block ]
# Draw options: over, iid_over.

observe_step        := 'observe' IDENT [ ':' type_expr ]
                       '<-' IDENT [ '(' draw_arg_list ')' ]
                       [ option_block ]
# Observation options: via = idx (an index map into the group of an
# enclosing grouped marginalize), over, iid_over.

score_step          := 'score' IDENT '=' let_arith

call_step           := 'let' IDENT '<-' qiec_call
# Binds the result of a computation declared with `define`; the callee's
# effect row joins the program's. Only the checked QIEC module runs a call
# step. A call of a program with open input extents writes no static
# arguments, since the extents are read from the arguments' types.

marginalize_step    := 'marginalize' IDENT [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]
                       [ option_block ]
                       NEWLINE INDENT program_step+ DEDENT
# Options: over = G, over = product(G_1, ...), or over = [G_1, ...] for the
# group set, and reduction = logsumexp | sum | mean (default logsumexp).

let_index           := IDENT '[' let_arith (',' let_arith)* ']'

program_decl        := 'program' IDENT [ '(' param_list ')' ]
                       ':' type_expr '->' type_expr
                       [ option_block ]
                       NEWLINE INDENT program_step* return_step DEDENT
# Options: effects = [E_1, ..., E_n], and over = M for a program over the
# posterior draws of the model M.

composition_decl    := 'composition' IDENT
                       [ option_block ]
                       [ NEWLINE INDENT composition_rule_entry+ DEDENT ]
# Option: level = algebra | semigroupoid | magmoid | rule.

composition_rule_entry
                    := IDENT '(' IDENT (',' IDENT)* ')' '=' let_arith
                     | IDENT '=' let_arith

contraction_decl    := 'contraction' IDENT
                       '(' contraction_input (',' contraction_input)* ')'
                       ':' type_expr '->' type_expr
                       [ option_block ]
# Options: rule = IDENT (required), wiring = STRING.

contraction_input   := IDENT ':' type_expr '->' type_expr

trans_compose       := expr '>>>' expr
change_base_postfix := expr '.' 'change_base' '(' expr ')'
morphism_call       := IDENT '(' expr (',' expr)* ')'
```

A program is *parametric* when its parameter list has a `typed_program_param`. The compiler expands a
parametric program at each call site instead of compiling it on its own
([Programs as kernels §3.2](programs.md#32-calls-of-programs)). The option `effects` is checked as in
[Programs as kernels §2.8](programs.md#28-effect-signatures), and `over = M` declares a program over the
posterior draws of `M`, whose latent variables become parameters of the program. The `composition`
declaration is described on [Composition rules §3](composition-rules.md#3-declaring-a-composition-rule).
A call `IDENT(arg_1, ..., arg_n)` is resolved against the contractions, the parametric programs, and the
morphisms of the module, in that order ([Expressions §2.13](expressions.md#213-operadic-contraction-call)).

## 11. QIEC fragment {#11-qiec-fragment}

The Quivers Indexed Effect Core (QIEC) adds its own declarations and computations to the module
grammar ([Typing and checking](typing.md)). A *telescope* is an ordered list of binders in which later
types may mention earlier variables [@debruijn1991telescopic]. Square brackets bind static types,
indices, or effect interfaces, and parentheses bind family indices or computation values. The EBNF below
is abridged, and `grammars/qvr/grammar.js` is authoritative:

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

A static argument is type syntax, read as a type or an effect according to the binder it fills, or an
integer literal, which fills a `Nat` index: `pad[3](xs)` applies a computation declared
`define pad[n : Nat](xs : Tensor[Real]([n])) ...` at extent three. The `grade` of an operation clause is
its [resumption grade](typing.md#7-handlers-and-scoped-instances), the number of times the clause may
resume its continuation: never (`0`), at most once (`aff`), exactly once (`1`), or any number of times
(`omega`) [@atkey2018syntax].

### Calls, requests, and constructors

Three forms have the shape `NAME(...)`, and a keyword or a qualifier separates them: a *call* is a bare
`NAME(...)` in computation position, an *effect request* is `perform inst.op(...)`, qualified by its
effect instance, and a *constructor value* is `construct NAME(...) as TYPE` in value position. A handler
clause resumes its continuation with `resume`, a keyword and not a bound name.

### Bindings

`let x = VALUE` binds a pure expression and `let x <- COMPUTATION` binds the result of a computation.
They are separate productions with separate typing rules.

### Pure expressions

QIEC values and the `let` and `score` steps of programs share one expression grammar. From loosest to
tightest, the operators are `||`, `&&`, comparison, `+ -`, `* / %`, and the unary `-` and `not`, and
operators of equal precedence associate to the left. A comparison with a negative literal needs
parentheses, `x < (-1)`, because `<-` is the binding arrow.

In a QIEC value each operator resolves by the type of its operands to one primitive of the closed
registry, and both operands must have the same type: `+` is integer or real addition or string
concatenation, `%` is integer only, `==` and `!=` compare integers, reals, Booleans, or strings, and
`&&`, `||`, and `not` take Booleans. Integer division and remainder truncate toward zero on every
target. The builtins `real`, `int`, `exp`, `log`, `sqrt`, `pow`, `abs`, `min`, and `max` resolve the same
way. A tuple `(a, b)` builds a finite product, and `t[i]` with an integer literal selects a component. A
list literal `[a, b, c]` is a `Tensor` whose leading dimension is its number of entries and whose entries
share one type, so `[[1.0, 2.0], [3.0, 4.0]]` is a `Tensor[Real]([2, 2])`, and a numeric list with any
fractional entry is real. Applying a distribution family, `Normal(mu, 1.0)` or `Dirichlet([1.0, 2.0])`,
fills the family's parameters in registry order and builds a `Sampleable`. `site("x")` is a `Site` at the
type its position expects, and `log_prob(d, x)` is the `LogWeight` of `d` at `x`. `if COND then ... else
...` branches on a Boolean in a computation, so the untaken branch is never entered, and a recursive
computation can stop.

### Layout

A construct that takes a nested computation opens an indented block after a trailing `=>` or `in`:
handler clauses, case branches, `handle ... in`, and `with instance ... in`. An inline computation (a
call, a request, or a resumption) takes one line and is followed by the rest of the computation at the
same indentation.

### Authored and foreign handlers

An operation clause with a `=>` body is *authored*. A clause without one is a signature, and its handler
must say `[implementation=foreign]`, so that a missing body never means "look up a callback at run time".
A foreign handler's clause behavior is given at run time by a provider, an arrangement close to a
[foreign function interface](https://en.wikipedia.org/wiki/Foreign_function_interface). The full grammar,
worked examples, and limits are in the [QIEC developer note](../developer/qiec.md).
