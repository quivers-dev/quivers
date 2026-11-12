# Expressions

An expression is the body of a `let` binding or an `export` declaration. Its combinators take
morphisms, and for the parser combinators grammar data, and build new morphisms. The semantics is
a compositional denotational semantics: $\llbracket e \rrbracket_\rho$ is defined by one clause for
each form of $e$, from the denotations of its immediate subexpressions. We write $g \circ f$ in
function order, $f$ first, and `f >> g` writes the same composite from left to right.

| Combinator | Builds |
|---|---|
| `f >> g`, `g << f` | `f` followed by `g` |
| `f @ g` | `f` and `g` on the two components of a paired input |
| `fan(f, g)` | `f` and `g` on one shared input |
| `f.marginalize(X)` | `f` with the codomain factor `X` reduced away |
| `repeat(f, n)` | `f` applied $n$ times with shared parameters |
| `stack(f, n)` | $n$ copies of `f` with separate parameters, in sequence |
| `scan(cell)` | a recurrent cell run along a sequence |
| `f.change_base(t)` | `f` converted by a transformation `t` |

## 1. Syntax

$$
\begin{array}{rcl}
e & ::= & x
        \;\big|\; \mathsf{identity}(\tau)
        \;\big|\; e_1 \mathbin{>\!\!>} e_2
        \;\big|\; e_2 \mathbin{<\!\!<} e_1
        \;\big|\; e_1 \mathbin{@} e_2
        \;\big|\; e.\mathsf{marginalize}(\bar X)
        \;\big|\; e.\mathsf{change\_base}(t) \\
&  \big| & e.\mathsf{dagger}
        \;\big|\; e.\mathsf{trace}(A)
        \;\big|\; \mathsf{cup}(A)
        \;\big|\; \mathsf{cap}(A)
        \;\big|\; e.\mathsf{curry\_right}
        \;\big|\; e.\mathsf{curry\_left}
        \;\big|\; e.\mathsf{freeze}
        \;\big|\; \mathsf{from\_data}(\text{"K"}) \\
&  \big| & \mathsf{fan}(e_1, \dots, e_n)
        \;\big|\; \mathsf{repeat}(e, n)
        \;\big|\; \mathsf{stack}(e, n)
        \;\big|\; \mathsf{scan}(e, \dots) \\
&  \big| & \mathsf{parser}(\dots)
        \;\big|\; \mathsf{ccg}(\dots)
        \;\big|\; \mathsf{lambek}(\dots)
        \;\big|\; \mathsf{chart\_fold}(\dots)
        \;\big|\; \mathit{op}(e_1, \dots, e_n) \\[1ex]
t & ::= & \mathit{name}
        \;\big|\; \mathit{name}(\overline{\mathit{arg}})
        \;\big|\; t_1 \mathbin{>\!\!>\!\!>} t_2
        \;\big|\; y
\end{array}
$$

The second grammar gives the transformations used by $\mathsf{change\_base}$: a built-in
transformation with no argument, a constructor applied to arguments, a sequential composite, and a
`let`-bound transformation $y$ ([Composition rules §5](composition-rules.md#5-first-class-transformations)).
In $\mathit{op}(e_1, \dots, e_n)$, $\mathit{op}$ is a `contraction` declared in the module.

Each expression denotes a morphism in one of the three fragments of [Morphisms](morphisms.md).
We write $\llbracket e \rrbracket_{\rho}$ for its denotation in the environment $\rho$
([Setting §6](setting.md#5-environments)) and drop $\rho$ when only morphism names occur.

## 2. Identifiers, identity, and the monoidal combinators

### 2.1 Identifiers

$\llbracket x \rrbracket_{\rho} = \rho_{\mathrm{mor}}(x)$. A name outside
$\mathrm{dom}(\rho_{\mathrm{mor}})$ is a `CompileError`.

### 2.2 Identity

$\llbracket \mathsf{identity}(\tau) \rrbracket_{\rho} = 1_{\llbracket \tau \rrbracket_{\rho}}$, the
identity morphism on the denotation of $\tau$: the identity matrix in $\mathbf{Mat}(\mathcal V)$
for an object expression, and the kernel $s \mapsto \delta_s$ in $\mathbf{BorelStoch}$ for a space
expression.

### 2.3 Composition

$$
\llbracket e_1 \mathbin{>\!\!>} e_2 \rrbracket = \llbracket e_2 \rrbracket \circ \llbracket e_1 \rrbracket = \llbracket e_1 \rrbracket ; \llbracket e_2 \rrbracket,
\qquad
\llbracket e_2 \mathbin{<\!\!<} e_1 \rrbracket = \llbracket e_1 \mathbin{>\!\!>} e_2 \rrbracket.
$$

The codomain of $e_1$ must equal the domain of $e_2$. For two finite weighted morphisms the
composite is the matrix product over the module's composition rule $\mathcal V$:

$$
\llbracket f \mathbin{>\!\!>} g \rrbracket(x, z) = \bigoplus_{y} \llbracket f \rrbracket(x, y) \otimes \llbracket g \rrbracket(y, z).
$$

Both operands must have the same rule, and the operators never convert one. Composing across two
rules takes an explicit [`change_base`](composition-rules.md#5-first-class-transformations). For
kernels the composite is the integral of
[Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels). When one operand is a finite
matrix and the other a kernel, the runtime first wraps the matrix as a kernel, as described in
[Morphisms §4](morphisms.md#4-tensor-product-across-fragments).

### 2.4 Tensor product

$\llbracket e_1 \mathbin{@} e_2 \rrbracket = \llbracket e_1 \rrbracket \boxtimes \llbracket e_2 \rrbracket$,
the tensor product of [Morphisms §4](morphisms.md#4-tensor-product-across-fragments).

### 2.5 Marginalization

For $\llbracket e \rrbracket : A \to X_1 \times \cdots \times X_k \times Y$,

$$
\llbracket e.\mathsf{marginalize}(X_1, \dots, X_k) \rrbracket(a, y) = \bigoplus_{x_1, \dots, x_k} \llbracket e \rrbracket\bigl(a, (x_1, \dots, x_k, y)\bigr),
$$

a morphism $A \to Y$ that reduces the named codomain factors with $\bigoplus$. Under the probability
semiring this is the marginal distribution. Under the Boolean semiring it asks whether some value
of the reduced factors relates $a$ to $y$. This method reduces an existing matrix. A
[`marginalize` step](programs.md#26-marginalize) in a program instead sums a latent variable out of
a likelihood.

### 2.6 Change of base

For $\llbracket e \rrbracket : A \to B$ over $\mathcal V$ and $t : \mathrm{Trans}[\mathcal V, \mathcal W]$,

$$
\llbracket e.\mathsf{change\_base}(t) \rrbracket = \llbracket t \rrbracket\bigl(\llbracket e \rrbracket\bigr),
$$

a morphism over $\mathcal W$. The action of $t$ is entrywise for an `AlgebraHomomorphism` and acts
on whole rows or matrices for a `MorphismTransformation`. Bayesian inversion also swaps the domain
and the codomain.

### 2.7 Sequential composition of transformations

$\llbracket t_1 \mathbin{>\!\!>\!\!>} t_2 \rrbracket(f) = \llbracket t_2 \rrbracket(\llbracket t_1 \rrbracket(f))$,
defined when the target of $t_1$ is the source of $t_2$. The checker rejects a mismatch with the
line and column of the expression. Since this is composition of functions, the two groupings of
$t_1 \mathbin{>\!\!>\!\!>} t_2 \mathbin{>\!\!>\!\!>} t_3$ denote the same transformation
([Composition rules §5](composition-rules.md#5-first-class-transformations)).

### 2.8 Two spellings of composition

`>>` and `<<` are the only composition operators. Both compose with the module's composition rule,
which the `composition` declaration sets
([Composition rules §3](composition-rules.md#3-declaring-a-composition-rule)), and `g << f` is
`f >> g` written from right to left.

### 2.9 Compact closed structure {#29-compact-closed-structure}

A [*compact closed category*](https://ncatlab.org/nlab/show/compact+closed+category) is a symmetric
monoidal category in which every object $A$ has a dual $A^*$, with a unit
$\eta_A : I \to A^* \boxtimes A$ (the *cup*) and a counit $\varepsilon_A : A \boxtimes A^* \to I$
(the *cap*) that satisfy the two snake equations below [@kelly1980coherence]. In
[string diagrams](https://ncatlab.org/nlab/show/string+diagram), where morphisms are drawn as boxes
and objects as wires, the cup bends a wire and the snake equations straighten it again
[@selinger2011survey].

When the module's rule $\mathcal V$ is a commutative semiring, $\mathbf{Mat}(\mathcal V)$ is compact
closed with each object its own dual. The compiler accepts the four expressions below only when the
rule is an `Algebra` ([Composition rules §2](composition-rules.md#2-the-denotation-of-at-each-level)),
which the default `ProductFuzzyAlgebra` is although it is not a semiring.

**Cup and cap.** With $\delta(a, a') = \mathbf 1$ if $a = a'$ and $\mathbf 0$ otherwise,

$$
\llbracket \mathsf{cup}(A) \rrbracket\bigl((), (a, a')\bigr) = \delta(a, a'),
\qquad
\llbracket \mathsf{cap}(A) \rrbracket\bigl((a, a'), ()\bigr) = \delta(a, a'),
$$

morphisms $\mathbf 1 \to A \times A$ and $A \times A \to \mathbf 1$. The snake equations

$$
(\mathrm{id}_A \boxtimes \mathsf{cup}(A)) ; (\mathsf{cap}(A) \boxtimes \mathrm{id}_A) = \mathrm{id}_A,
\qquad
(\mathsf{cup}(A) \boxtimes \mathrm{id}_A) ; (\mathrm{id}_A \boxtimes \mathsf{cap}(A)) = \mathrm{id}_A
$$

reduce on entries to $\bigoplus_{a, a'} \delta(x, a) \otimes \delta(a, a') \otimes \delta(a', y) = \delta(x, y)$.
Every term but one is a product with a factor $\mathbf 0$, so the equations hold whenever
$\mathbf 1$ is a unit of $\otimes$, $\mathbf 0$ absorbs, and $\mathbf 0$ is the unit of $\bigoplus$,
which is true of every built-in instance. That cup, cap, and the tensor product make
$\mathbf{Mat}(\mathcal V)$ a compact closed category needs the semiring laws as well.

**Dagger.** For $f : A \to B$, $\llbracket f.\mathsf{dagger} \rrbracket(b, a) = \llbracket f \rrbracket(a, b)$,
the transpose. Transposition reverses composition,
$(f ; g)^\dagger = g^\dagger ; f^\dagger$, when $\otimes$ is commutative, so it is a
[dagger](https://ncatlab.org/nlab/show/dagger+category) on $\mathbf{Mat}(\mathcal V)$ for a
commutative semiring, and together with cup and cap it makes $\mathbf{Mat}(\mathcal V)$
[dagger compact](https://ncatlab.org/nlab/show/dagger-compact+category) [@selinger2007dagger].

**Trace.** A [traced monoidal category](https://ncatlab.org/nlab/show/traced+monoidal+category) has
an operation that feeds an output wire back into an input wire [@joyal1996traced]. For
$f : A \times X \to A \times Y$, the partial trace along $A$ is

$$
\llbracket f.\mathsf{trace}(A) \rrbracket(x, y) = \bigoplus_{a \in \llbracket A \rrbracket} \llbracket f \rrbracket\bigl((a, x), (a, y)\bigr).
$$

For a commutative semiring it equals the composite
$(\mathsf{cup}(A) \boxtimes \mathrm{id}_X) ; (\mathrm{id}_A \boxtimes f) ; (\mathsf{cap}(A) \boxtimes \mathrm{id}_Y)$,
up to the associativity isomorphisms, and so satisfies the trace axioms of every compact closed
category [@joyal1996traced]. The categorical trace is unrelated to an execution trace, the record
of the random choices of a run.

### 2.10 Currying {#210-residuation-witnesses}

For $f : X \times Y \to Z$, the postfixes `f.curry_right` and `f.curry_left` are meant to denote
the two currying bijections of a biclosed monoidal category, $\mathcal C(X \otimes Y, Z) \cong
\mathcal C(X, Z / Y)$ and $\mathcal C(X \otimes Y, Z) \cong \mathcal C(Y, X \backslash Z)$
([Schemas §3](schemas.md#3-residuated-type-formers)). In $\mathbf{Mat}(\mathcal V)$, where each
object is its own dual, the corresponding bijection sends $f$ to the matrix
$(x, (y, z)) \mapsto f((x, y), z)$ from $X$ to $Y \times Z$.

The runtime class `CurriedMorphism` computes neither. `f.curry_right` has the domain $X$ (the
domain of $f$ without its last factor), the codomain $Z$ unchanged, and the matrix of $f$
unchanged, of shape $X \times Y \times Z$, so its declared type does not match its entries.
`f.curry_left` drops the first factor instead. When $Z$ is not a set of categorial types, the
object also records a slash type built from $Z$ and the dropped factor, beside the unchanged
codomain.

### 2.11 Freeze

$\llbracket f.\mathsf{freeze} \rrbracket = \llbracket f \rrbracket$. The runtime copies the current
entries into an `ObservedMorphism`, so that training does not change them.

### 2.12 Data

$\llbracket \mathsf{from\_data}(\text{"K"}) \rrbracket = \rho_{\mathrm{data}}(\text{"K"})$, the matrix
given at fit time under the key `"K"`. Its shape must match the declared type, and training does not
change it.

### 2.13 Contraction calls {#213-operadic-contraction-call}

For a `contraction` $\mathit{op}$ with $n$ inputs of types $A_i \to B_i$ and wiring $w$,

$$
\llbracket \mathit{op}(e_1, \dots, e_n) \rrbracket = \Phi_w\bigl(\llbracket e_1 \rrbracket, \dots, \llbracket e_n \rrbracket\bigr),
$$

where $\Phi_w$ is the einsum of [Composition rules §4](composition-rules.md#4-operadic-contractions).
The call checks that each argument has as many entries as its declared type, and a mismatch is a
`CompileError`.

## 3. Copying, iteration, and recurrence

### 3.1 Fan {#31-fan}

For $e_i : X \to Y_i$,

$$
\llbracket \mathsf{fan}(e_1, \dots, e_n) \rrbracket\bigl(x, (y_1, \dots, y_n)\bigr) = \prod_{i=1}^{n} \llbracket e_i \rrbracket(x, y_i),
$$

the composite of the copy map $X \to X^n$ with $e_1 \boxtimes \cdots \boxtimes e_n$. For kernels, the
outputs are independent given the shared input $x$, and they may be dependent once $x$ is averaged
over. A kernel $X \to Y_1 \times Y_2$ is not determined by its two marginals, since other
[couplings](https://en.wikipedia.org/wiki/Coupling_(probability)) have the same marginals, so `fan`
is not a categorical product: $\mathbf{FinStoch}$ and $\mathbf{BorelStoch}$ are Markov categories,
not Cartesian ones [@fritz2020synthetic]. For finite weighted morphisms, $\otimes$ replaces the
product of numbers.

### 3.2 Stack

For $e : X \to X$ and $n \ge 1$,
$\llbracket \mathsf{stack}(e, n) \rrbracket = \llbracket e_1 \rrbracket ; \cdots ; \llbracket e_n \rrbracket$,
where $e_1, \dots, e_n$ are copies of $e$ with separate parameters. The output of copy $i$ is the
input of copy $i + 1$.

### 3.3 Repeat

For $e : X \to X$, $\llbracket \mathsf{repeat}(e, n) \rrbracket = \llbracket e \rrbracket ; \cdots ; \llbracket e \rrbracket$
($n$ times), with one set of parameters shared by every step. `repeat(e)` has $n = 1$.

### 3.4 Scan

A cell $e : A \times H \to H$ maps an input and a state to a new state. For an input sequence
$x = (x_1, \dots, x_T) \in A^T$, the scan runs

$$
h_0 = \mathit{init},
\qquad
h_t = e(x_t, h_{t-1}) \quad (t = 1, \dots, T),
$$

and $\mathsf{scan}(e) : A^T \to H$ returns $h_T$. With `outputs=all` it returns the trajectory
$(h_1, \dots, h_T) \in H^T$, a sequence along the same axis, so that a second scan can run over it.
With `reverse=true` the recursion runs $h_t = e(x_t, h_{t+1})$ from $h_{T+1} = \mathit{init}$, returns
$h_1$, and still indexes the trajectory by input position. The initial state is zeros, a learned
vector, or a binding of the enclosing program (`init=h0`).

For a stochastic cell, the density of a whole trajectory is
$\sum_t \log p(h_t \mid x_t, h_{t-1})$, the joint of the states. The density of the final state alone
is a different quantity, which integrates over every intermediate trajectory. The runtime class
[`ScanMorphism`](../api/continuous/scan.md) scores trajectories: `log_joint(x, hidden_states)`
takes the trajectory as a tensor of shape (batch, sequence length, state dimension), or as a
dictionary keyed by `state_key` (default `"h"`).

When a program draws a scan, the sites of its cell become sites of the program, visited once per
position. The joint density is $\sum_t \sum_s \log p(s_t \mid \cdots)$ over the sites $s$ of the
cell, and the $t$-th visit of a site $s$ has the address `s@t`. `sample h <- scan(cell)` thus scores
the trajectory at `h@0`, ..., `h@{T-1}`. A reversed scan numbers its visits in the order it makes
them, so `s@n` is input position $T - n$ (positions counted from 1), while observations are read by
input position. The [scans reference](../reference/qvr/probabilistic-programs.md#scans) gives the
scoping and addressing rules.

## 4. Parser combinators

### 4.1 `parser`, `ccg`, and `lambek`

The keywords `parser`, `ccg`, and `lambek` are three spellings of one form. Each takes a list of
rule names, resolved through the morphism, schema, and bundle environments
([Schemas §7](schemas.md#7-bundles)), the atomic syntactic categories, a start category $s$, a
bound $d$ on the nesting depth of categorial types, and the type constructors allowed. From these
it builds a lexical morphism, a binary rule morphism, and optionally a unary rule morphism over the
finite set of categorial types of depth at most $d$ ([Schemas §4](schemas.md#4-the-free-residuated-universe)),
and passes them to the class `InsideAlgorithm`.

The denotation sends a token string $w = w_1 \cdots w_n$ to the log of the inside weight of the start
category over the whole string,

$$
\llbracket \mathsf{parser}(\dots) \rrbracket(w) = \log \beta(s, 0, n),
$$

where $\beta(c, i, j)$ is the sum, over every derivation of $w_{i+1} \cdots w_j$ from the category
$c$, of the product of the weights of the rules it uses. The runtime computes $\beta$ by the CKY
recursion in the log semiring, whatever the module's composition rule. When the lexical and rule
morphisms are stochastic, so that the rules form a probabilistic context-free grammar,
$\beta(s, 0, n)$ is the [inside probability](https://en.wikipedia.org/wiki/Inside%E2%80%93outside_algorithm)
of $w$ [@baker1979trainable; @goodman1999semiring]. The value depends only on the rules and the
string, not on the order in which the chart is filled.

That description holds for grammars without unary rules. With a unary rule matrix $U$, the inside
weights of a cell should be closed under unary chains, $\beta \mapsto \beta (I + U + U^2 + \cdots)$.
The runtime instead repeats the update $\beta \leftarrow \beta \oplus \beta U$ until the cell stops
changing or eight times, which gives $\beta (I + U)^k$ and counts a chain of unary rules more than
once. For one rule $A \to B$ of probability $\tfrac12$ and $\beta(A) = 1$, it returns
$\beta(B) = 4$ in place of $\tfrac12$. With unary rules the value is thus not the inside
probability.

### 4.2 `chart_fold`

The [chart recursion](grammar.md#6-chart-denotation) states which derivations a cell sums. The
primitive `chart_fold` takes the three morphisms directly:

$$
\mathit{args} = \bigl(\mathit{lex} = \ell,\ \mathit{binary} = B,\ \mathit{unary} = U,\ \mathit{start} = s,\ \mathit{depth} = d,\ \mathit{effect\_depth} = k\bigr),
$$

where $\ell : \mathrm{Token} \to \mathrm{Cat}$ gives the lexical axioms, $B : \mathrm{Cat} \times \mathrm{Cat} \to \mathrm{Cat}$
the binary rules, the optional $U : \mathrm{Cat} \to \mathrm{Cat}$ the unary rules, $s$ the start
category, $d$ the bound on the depth of categorial types, and $k$ the number of effects a cell may
stack (default 0). It denotes the same function of $w$ as `parser`. With $k > 0$ each chart entry also
carries a stack of at most $k$ effects, and the rules fire as described in
[Effects §4](effects.md#4-joint-type-and-effect-dispatch).

### 4.3 Deduction calls

Three builtins act on [weighted deduction systems](grammar.md#93-let-expression-builtins):

- `parse(D, x)` evaluates the deduction system $D$ on the input $x$ and returns its chart, the
  weight function $\alpha : I \to K$ from the items $D$ proves to its semiring $K$;
- `compose(D_1, D_2)` builds the system whose axioms are the goal items of $D_1$ and whose rules,
  semiring, agenda, and goal are those of $D_2$ ([Weighted deduction §9.1](grammar.md#91-composing-deductions));
  fitting it fits the parameters of both;
- `subst(t, v, w)` replaces each subterm of the item $t$ equal to $v$ by $w$
  ([Weighted deduction §9.3](grammar.md#93-let-expression-builtins)).

A chart $\alpha$ answers these method calls:

| Call | Value |
|---|---|
| `chart.weight(i)` | $\alpha(i)$ for a proved item $i$ |
| `chart.try_weight(i, d)` | $\alpha(i)$ if $i$ is proved, and otherwise $d$, or $\mathbf 0_K$ when $d$ is omitted |
| `chart.enumerate(p)` | the pairs $(i, \alpha(i))$ for the proved items $i$ that match the pattern $p$ |
| `chart.goal_weight()` | $\bigoplus_{i \in \mathrm{goal}} \alpha(i)$ |
| `chart.attached_loss` | the sum of the losses attached to the chart |
| `chart.semiring` | the semiring $K$ |

## 5. Equations

When the module's rule $\mathcal V$ is a commutative semiring, the denotations satisfy the equations
of a symmetric monoidal category ([Morphisms §1.1](morphisms.md#11-composition-tensor-identity)):

| Law | Statement |
|-----|-----------|
| Associativity of composition | $\llbracket (e_1 \mathbin{>\!\!>} e_2) \mathbin{>\!\!>} e_3 \rrbracket = \llbracket e_1 \mathbin{>\!\!>} (e_2 \mathbin{>\!\!>} e_3) \rrbracket$ |
| Units of composition | $\llbracket \mathsf{identity}(\tau) \mathbin{>\!\!>} e \rrbracket = \llbracket e \rrbracket = \llbracket e \mathbin{>\!\!>} \mathsf{identity}(\tau') \rrbracket$ |
| Associativity of `@` | $\llbracket (e_1 \mathbin{@} e_2) \mathbin{@} e_3 \rrbracket = \llbracket e_1 \mathbin{@} (e_2 \mathbin{@} e_3) \rrbracket$, up to the regrouping of [Types and spaces §5](types-and-spaces.md#5-regrouping-and-order-of-factors) |
| Interchange | $\llbracket (e_1 \mathbin{>\!\!>} e_2) \mathbin{@} (e_3 \mathbin{>\!\!>} e_4) \rrbracket = \llbracket (e_1 \mathbin{@} e_3) \mathbin{>\!\!>} (e_2 \mathbin{@} e_4) \rrbracket$ |
| Symmetry | $\llbracket e_1 \mathbin{@} e_2 \rrbracket = \sigma_{X_1, X_2} ; \llbracket e_2 \mathbin{@} e_1 \rrbracket ; \sigma_{Y_2, Y_1}$ |

The same equations hold for stochastic matrices and Markov kernels, whose composition and tensor
product are those of $\mathbf{FinStoch}$ and $\mathbf{BorelStoch}$. For a rule that is not a
semiring they may fail. The compiler evaluates an expression in the grouping written, so a rewrite
that regroups is justified only under these hypotheses.
[Implementation correspondence](adequacy.md) lists the tests of these operations.
