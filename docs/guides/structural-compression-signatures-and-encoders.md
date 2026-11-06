# Structural Compression: Signatures and Encoders

An encoder turns a structured value, such as a syntax tree, into a
fixed-length vector. A [decoder](structural-compression-decoders-and-losses.md#decoder-blocks) assigns probabilities to structures given
such a vector. The structure determines the recursion; learnable functions
determine how child representations combine. This lets a model reuse one
function at every occurrence of a constructor, including occurrences in
trees of different sizes.

For instance, encode `App(Const("dog"), Const("cat"))` by first encoding
the two constants, then applying the learned `App` function to their
vectors. Formally,
$C(\mathrm{App}(t,u)) = \widehat C_{\mathrm{App}}(C(t),C(u))$.
This equation explains the algebra-homomorphism terminology below:
constructing a term and then encoding agrees with encoding its children
and combining them with the corresponding vector operation. Two terms
can receive the same vector; the decoder models which reconstructions
are likely from that vector.

Milewski’s [F-algebras chapter](https://bartoszmilewski.com/2017/02/28/f-algebras/#:~:text=Category%20of%20F-Algebras), in its “Category of F-Algebras” and “Catamorphisms” sections, constructs an evaluator by assigning an operation to each constructor. [@milewski2017falgebras] The [algebras and coalgebras guide](monadic.md#algebras-and-coalgebras) gives the corresponding equations.


Quivers exposes a uniform algebraic interface for compressing
arbitrary structured objects to fixed-length vectors and decoding
them back under a learned distribution. The interface is built on
two categorical primitives:

- An **encoder** is a [Σ-algebra
  homomorphism](https://ncatlab.org/nlab/show/algebra+over+an+endofunctor)
  $T_\Sigma \to \mathrm{Vec}_D$: for each operation $\mathrm{op} :
  s_1 \times \cdots \times s_n \to s$ in a multi-sorted signature
  $\Sigma$, a parametric function $\hat{C}_{\mathrm{op}} :
  \mathrm{Vec}_D \times \cdots \times \mathrm{Vec}_D \to
  \mathrm{Vec}_D$. The recursion over the term tree is supplied by
  the framework; the analyst supplies only the per-operation
  parametric functions.
- A **decoder** is a [Kleisli
  coalgebra](https://ncatlab.org/nlab/show/coalgebra+over+an+endofunctor)
  $\mathrm{Vec}_D \to \mathrm{Kern}(T_\Sigma)$: a structure choice,
  primitive choice, factor split, and binder selection together
  corecursively generate a `Term` (with per-term `log_prob`
  available for observed terms).

Encoder plus decoder is a stochastic autoencoder on $T_\Sigma$. The
same interface represents transformers (sequence encoder +
autoregressive decoder), tree-LSTMs (tree encoder), graph neural
networks (message-passing graph encoder), variational autoencoders
(encoder + decoder pair on $T_\Sigma$ with a KL term), the
vector inside-outside parser of Le-Zuidema / Drozdov / Kim
(chart-item signature + attached encoder), and typed-LF induction
pipelines (binders with annotation sorts threading the type of each
bound variable through a de-Bruijn context).

This page covers signature and encoder declarations; the
[decoder and loss declarations](structural-compression-decoders-and-losses.md#decoder-blocks)
are documented on a separate page.

## Signature blocks

A **signature** lists the kinds of value, called sorts, and the
constructors that build them. In the example, `Const` builds a term from
a name, while `App` builds a term from two terms. `Term` and `Type`
specify which constructors are allowed; they can have different
constructors even when their embeddings have the same length. The `dim` fields specify representations used by the neural modules.


A `signature` declares the sorts, constructors, binders, and (for
graph signatures) vertex / edge kinds of an inductive (or
graph-shaped) algebra.

<!-- compile: false -->
```qvr
signature LF
    sorts
        Term : object [dim=64]
        Type : object [dim=32]
        Name : data [dim=32, vocab=["dog", "cat", "every", "some"]]
    constructors
        Const : Name -> Term
        App : Term, Term -> Term
    binders
        Lam : binds (x : Term : ty : Type) in (body : Term) -> Term
        All : binds (x : Term : ty : Type) in (body : Term) -> Term
```

Three sort kinds:

- `object`: the principal sorts whose terms are compressed /
  decoded recursively.
- `data`: opaque atoms (string / int / float / bytes / bool)
  consumed by per-data-sort embedders. A data sort may declare a
  closed vocabulary inline via `vocab { ... }` listing string,
  integer, and / or float literals; the decoder samples and scores
  data-sorted children from exactly these tokens.
- `index`: [de-Bruijn
  index](https://en.wikipedia.org/wiki/De_Bruijn_index) slots.

Reserved op names: `BoundVar` and `Data`. These are
framework-built-in op tags. The compiler rejects them as
user-declared constructor / binder names.

### Binders, de-Bruijn context, and type annotations

A binder makes a variable available only inside a particular body.
De Bruijn indices identify bound variables by their distance from the
innermost binder: in nested binders, index zero refers to the most
recently introduced variable, and index one to the next outer variable.
Renaming a bound variable thus does not change this representation.
The context `Gamma` records the available variables together with their
embeddings and annotations, so nested scopes remain distinct.


A binder declaration introduces one or more typed scoped variables.
`binds (x : Term : ty : Type) in (body : Term) -> Term` reads:
"constructs a value of sort `Term`; in the scope of argument
`body`, introduces a variable `x` of sort `Term` annotated by a
`Type`-sorted term `ty`."

The framework threads a de-Bruijn context $\Gamma$ through both the
encoder's recursion and the decoder's corecursion. Each entry of
$\Gamma$ is a triple `(var_sort, embedding, type_term)` that records
the variable's sort, vector representation, and type annotation.
`BoundVar(i)` at any object-sorted
position reads $\Gamma$ at depth `i`. The encoder's `var_init`
function (see below) computes the variable's vector embedding from
its type annotation.

Strict declaration rules:

- Every sort a constructor or binder mentions (domain, codomain,
  annotation sort) must be declared in the signature's
  `sorts { ... }` block. There is no silent auto-registration.
- Every sort with no inline `dim` must have its dim supplied by
  every encoder / decoder over the signature. The compiler raises
  if a sort's dim is unresolved.

## Encoder blocks

The encoder replaces each constructor by a function on child vectors.
A leaf embedding handles a value with no recursive children; a binary
constructor consumes two already-computed vectors. This is a bottom-up
fold over a tree. The framework supplies the traversal, while each
`op` body supplies the local computation. The formal account uses
algebras over a signature functor; the concrete recurrence above is the
behavior a model author needs to implement.


An encoder declares the carrier dim per sort plus one parametric
function per constructor / binder. Bodies are optional; the
compiler scaffolds a 2-layer MLP per omitted op with the correct
per-arg dim sequence.

<!-- compile: false -->
```qvr
encoder C : LF
    dim Term = 64
    dim Type = 32

    op Const(n) |-> name_embed
    op App(f, x) |-> mlp_app([f, x])

    op Lam(ty, body) |-> mlp_lam([ty, body])
    op All(ty, body) |-> mlp_all([ty, body])

    var_init Term from Type as ty |-> mlp_typed_var(ty)
```

For binder constructors, the framework's calling convention is:

1. The annotation arguments (one per annotated bound variable) are
   compressed in the **outer** context $\Gamma$.
2. Each `var_init <var_sort> from <annot_sort> as <name> |-> body`
   declaration is invoked to mint a fresh variable embedding from
   the annotation embedding; the framework pushes
   `(var_sort, embedding, annot_term)` onto $\Gamma$.
3. The scoped arguments are compressed in the **extended** context
   $\Gamma'$.
4. The per-op function receives the flat child-embedding list
   `[annot_1, ..., annot_k, scoped_1, ..., scoped_m]` in
   declaration order.

Multiple `var_init` declarations are allowed per encoder, one per
`(var_sort, annot_sort)` pair the signature's binders introduce.
Unannotated binders (those without an annotation sort) use a
learnable nullary constant keyed by `var_sort` alone.

### Primitives and user-defined callables in rule bodies

Rule bodies are let-expressions over the same fixed primitive pool
the rest of the DSL uses (every `torch.nn.functional` activation,
the simplex maps, pointwise transcendentals, `dim=-1` reductions,
layer-norm, dropout; see the
[let-expression primitive reference](dsl-programs-and-lets.md#checked-primitive-reference))
plus any callable declared at the module level.

**Module-scope callables resolve by name.** Top-level `morphism`,
`program`, `encoder`, `decoder`, and `deduction` declarations are
visible inside encoder bodies and may be called as `f(arg, ...)`.

**Resolution order**: the dispatcher resolves builtins first, then
user globals, then user-declared constructors. This means a
let-expression primitive name shadows a user constructor with the
same name (the compiler will refuse to register such a constructor
when it's first declared).

**Arity checking.** Each call site checks the number of positional
arguments against the callee's declared arity at compile time. The
arity rules:

- A `MonadicProgram` with named `params`: arity equals
  `len(params)`.
- A `MonadicProgram` without `params`: arity is 1 (the packed
  program input).
- A `Morphism`: arity is 1 (the domain tensor).
- Any other callable: positional parameters without defaults; a
  `*args` parameter makes arity unknowable, in which case the
  compiler accepts the call.

**Shape-error reporting.** Errors raised inside a user callable are
reported with the encoder call site named, rather than being buried in
the PyTorch traceback. The dispatcher captures the call site
(filename, line, column from the tree-sitter parse) and re-raises
with that frame attached.

<!-- compile: false -->
```qvr
program embed : Pixel -> Latent
    let h = relu(layer_norm(W1 * x))
    return h

encoder C : LF
    dim Term = 64
    op App(f, x) |-> gelu(embed(f) + embed(x))
```

### Factory form

For encoders that fit one of the standard architectures, the
explicit `{ ... }` body can be replaced by a factory invocation:

<!-- compile: false -->
```qvr
encoder rnn_enc : seq [factory=rnn_encoder, dim=128]
encoder tfm_enc : seq [factory=transformer_encoder, dim=256]
encoder bow_enc : seq [factory=bow_encoder]
encoder tree : lf [factory=tree_lstm_encoder]
encoder graph : mol [factory=gnn_encoder]
```

Factories live in `quivers.structural.shapes` (see the [encoder API
page](../api/structural/encoder.md)); the registry mapping factory
name to import path is
`quivers.dsl.compiler.structural._ENCODER_FACTORY_REGISTRY`.
Bracketed kwargs are forwarded to the factory and validated against
its signature. The two forms (explicit body and factory) are
mutually exclusive on a single declaration but coexist in the same
module.

The factory signatures and what they produce:

| Factory | Signature | What it builds |
|---|---|---|
| `rnn_encoder(sig, dim)` | sequence signature | GRU-cell right-fold over the chain |
| `transformer_encoder(sig, dim)` | sequence signature | Per-`Cons` head + tail-projection MLP over their concatenation |
| `bow_encoder(sig, dim)` | sequence signature | Order-independent sum over chain elements |
| `tree_lstm_encoder(sig, dim)` | tree signature | Child-sum binary-tree LSTM [@tai2015treelstm] |
| `gnn_encoder(sig, iterations, dim, readout)` | graph signature | Per-edge-kind message MLP, per-vertex-kind GRU update, mean / sum / max readout |

All factories return an
[`Encoder`](../api/structural/encoder.md) instance over the
underlying signature; the explicit-body form and the factory form
produce interchangeable runtime values.

### Sequence-shaped sugar

For sequence signatures (`Seq[A] = Nil | Cons(A, Seq)`), two extra
body shapes are available:

<!-- compile: false -->
```qvr
encoder RNN : Seq
    op Nil |-> 0.0
    op Cons(head, tail) recurrent state |-> gru_step(head, state)

encoder Tfm : Seq
    op Nil |-> 0.0
    op Cons(head, tail) attention prefix |-> tfm_step(head, prefix)
```

- `recurrent <state>` binds the named state variable to the
  recursive child's already-computed embedding, exactly the
  standard right-fold F-algebra recursion, with the user's chosen
  name for the running state.
- `attention <prefix>` iteratively walks the chain of recursive
  applications outside-in. At step $i$, the body sees `prefix =
  [head_0_emb, ..., head_{i-1}_emb]` (the running list of
  non-recursive children's embeddings collected outside-in). The
  encoder's final embedding is the deepest (innermost) step's
  output.

### Graph-shaped sugar

Trees have a direct parent-child evaluation order, but graphs can contain
cycles. Message passing instead updates vertex representations for a
fixed number of rounds. Each edge sends information from a source to a
target; each target combines incoming messages with its current state;
a readout aggregates the resulting vertex states. The iteration count
controls how many rounds of local communication are available.


Graph signatures declare `vertex_kinds` and `edge_kinds` (with
typed endpoints), and the encoder body uses message-passing:

<!-- compile: false -->
```qvr
signature Mol
    vertex_kinds
        Atom : data [dim=32]
        Bond : data [dim=32]
    edge_kinds
        bonded : Atom -- Atom
        in_bond : Atom -> Bond

encoder GNN : Mol
    iterations 4

    init Atom(a) |-> atom_embed[a]
    init Bond(b) |-> bond_embed[b]
    message[bonded](src, tgt) |-> mlp_msg([src, tgt])
    message[in_bond](src, tgt) |-> mlp_in([src, tgt])
    update[Atom](self, msgs) |-> gru_update_atom(self, mean(msgs))
    update[Bond](self, msgs) |-> gru_update_bond(self, mean(msgs))
    readout |-> mean_pool
```

Undirected edges (`Atom -- Atom`) emit messages in both directions;
directed edges (`Atom -> Bond`) emit only in the declared
direction.

## Stdlib shapes

`quivers.structural.shapes` provides ready-to-use signatures and
encoders / decoders for the three principal compressible shapes:

- `seq_signature(name, dim)`: `Seq[A] = Nil | Cons(A, Seq)`.
- `rnn_encoder(sig, dim)`: GRU-cell right-fold.
- `transformer_encoder(sig, dim)`: head + tail-projection MLP.
- `bow_encoder(sig, dim)`: order-independent sum.

- `tree_signature(name, dim)`:
  `Tree[L, B] = Leaf(L) | Node(B, Tree, Tree)`.
- `tree_lstm_encoder(sig, dim)`: child-sum binary-tree LSTM.

- `graph_signature(name, vertex_kinds, edge_kinds)`: vertex /
  edge-kinded graph signature.
- `gnn_encoder(sig, iterations, dim, readout)`: per-edge-kind
  message MLP, per-vertex-kind GRU update, mean / sum / max
  readout.

All shapes are realized on top of the generic
[`Encoder`](../api/structural/encoder.md) and
[`Decoder`](../api/structural/decoder.md) runtimes, the same
algebra-homomorphism / Kleisli coalgebra pattern, no special-casing.

## See also

- [Structural Compression: Decoders and Losses](structural-compression-decoders-and-losses.md):
  the dual Kleisli arrow back to terms and the loss declarations.
- [Weighted Deduction Systems](deduction.md): the chart-parser
  representation to which an encoder can attach, exposing
  `chart.embedding(item)` as a differentiable value.
- [Structural autoencoders tutorial](../tutorials/qvr/12-structural-autoencoders.md):
  checked entry invocation, autograd, and checkpoint registration.

## Further reading

- [@tai2015treelstm]
