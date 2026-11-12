# Tree and Graph Autoencoders: Signatures and Encoders

An encoder turns a structured value, such as a syntax tree, into a fixed-length
vector, and a [decoder](structural-compression-decoders-and-losses.md#decoder-blocks)
assigns probabilities to structures given such a vector. Together they form an
[autoencoder](https://en.wikipedia.org/wiki/Autoencoder) over terms, the design of
recursive autoencoders [@pollack1990recursive], tree-LSTM encoders
[@tai2015treelstm], and message-passing graph encoders
[@gilmer2017neural; @battaglia2018relational]. `quivers.structural` and the QVR
declarations `signature`, `encoder`, `decoder`, and `loss` implement them.
[Decoders and losses](structural-compression-decoders-and-losses.md) covers the last
two.

A *signature* $\Sigma$ lists sorts and the constructors that build terms of each
sort, and $T_\Sigma$ is the set of terms it generates. An encoder gives each
constructor $\mathrm{op} : s_1 \times \dots \times s_n \to s$ a learnable function
$\widehat C_{\mathrm{op}}$ on vectors and computes the code of a term bottom-up:

$$
C(\mathrm{op}(t_1, \dots, t_n)) = \widehat C_{\mathrm{op}}(C(t_1), \dots, C(t_n)).
$$

To encode `App(Const("dog"), Const("cat"))`, for instance, it encodes the two
constants and applies the learned `App` function to their vectors. One function is
reused at every occurrence of a constructor, in trees of any size. In algebraic terms
the functions $\widehat C_{\mathrm{op}}$ make the vector space $\mathrm{Vec}_D$ an
[algebra for the signature](https://ncatlab.org/nlab/show/algebra+for+an+endofunctor),
and $C$ is the unique map from the term algebra $T_\Sigma$ that respects every
constructor, the [catamorphism](https://en.wikipedia.org/wiki/Catamorphism) or fold
[@meijer1991functional]. Different terms can receive the same vector, and the decoder
models which terms are likely given a vector.

## Signature blocks

A `signature` declares the sorts, the constructors, the binders, and, for graphs,
the vertex and edge kinds. In the example, `Const` builds a `Term` from a `Name`, and
`App` builds a `Term` from two `Term`s. `Term` and `Type` admit different
constructors even though both are vectors in the end, and the `dim` options give the
vector length of each sort.

<!-- compile: cumulative -->
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

A sort is one of three kinds:

- `object`: a sort whose terms are encoded and decoded recursively;
- `data`: leaves that carry a string, integer, float, byte string, or Boolean, each
  embedded by an embedder for its sort. A data sort may list a closed vocabulary
  with the option `vocab=[...]`, and the decoder then samples and scores leaves of
  that sort from exactly those values;
- `index`: positions for [de Bruijn indices](https://en.wikipedia.org/wiki/De_Bruijn_index)
  [@debruijn1972lambda].

A constructor with no arguments is written with an empty domain, `Leaf : -> Tree`.
The names `BoundVar` and `Data` are reserved for the library's own constructors and
cannot be declared.

### Binders, de Bruijn context, and type annotations

A binder makes a variable available inside one body only. A
[de Bruijn index](https://en.wikipedia.org/wiki/De_Bruijn_index) names a bound
variable by its distance from the innermost binder, so index 0 is the most recently
bound variable, and renaming a bound variable does not change the representation. The
declaration `binds (x : Term : ty : Type) in (body : Term) -> Term` builds a `Term`
and, in the argument `body`, binds a variable `x` of sort `Term` annotated by a term
`ty` of sort `Type`. Signatures of this kind, with constructors that bind variables in
some arguments, are binding signatures [@fiore1999abstract].

The encoder and the decoder both carry a context $\Gamma$ of bound variables. Each
entry is a triple `(var_sort, embedding, type_term)` of the variable's sort, its
vector, and its annotation, and `BoundVar(i)` in an object-sorted position reads the
entry at depth `i`. The encoder's `var_init` rule, below, computes a variable's vector
from its annotation.

Every sort that a constructor or binder mentions must be declared under `sorts`, and
every sort without a `dim` option must receive one from each encoder and decoder over
the signature. The compiler raises an error otherwise.

## Encoder blocks

An `encoder` gives the vector length of each sort and a body for each constructor and
binder. A body is a let-expression over the children's vectors, and an omitted body
defaults to a learned two-layer MLP with the right input and output lengths:

<!-- compile: cumulative -->
```qvr
encoder C : LF
    dim Term = 64
    dim Type = 32
    op App(f, x) |-> gelu(f + x)
```

Here `App` sums its children's vectors and applies GELU, and `Const`, `Lam`, and `All`
take the default MLPs. For a binder the library proceeds in four steps:

1. it encodes the annotation arguments in the outer context $\Gamma$;
2. for each annotated variable, it runs the matching
   `var_init <var_sort> from <annot_sort> as <name> |-> body` rule to compute the
   variable's vector from its annotation's vector, and pushes
   `(var_sort, embedding, annot_term)` onto $\Gamma$;
3. it encodes the scoped arguments in the extended context;
4. it passes the binder's body the list `[annot_1, ..., annot_k, scoped_1, ..., scoped_m]`
   in declaration order.

An encoder may give one `var_init` rule for each `(var_sort, annot_sort)` pair its
signature's binders use. A pair without a rule gets a learned two-layer MLP, and an
unannotated variable gets a learned constant vector for its sort.

### Primitives and user-defined callables in rule bodies

Rule bodies use the same primitives as let-expressions elsewhere in QVR, listed in the
[primitive reference](dsl-programs-and-lets.md#checked-primitive-reference): the
activations of `torch.nn.functional`, the simplex maps, pointwise functions,
reductions over the last axis, `layer_norm`, and `dropout`. They can also call any
`morphism`, `program`, `encoder`, `decoder`, or `deduction` declared at the top level
of the module, as `f(arg, ...)`.

A name in a body resolves first to a primitive, then to a top-level declaration, then
to a constructor. A constructor with a primitive's name, such as `relu`, is accepted,
but a call `relu(x)` in a body reaches the primitive.

The compiler checks the number of arguments at each call:

- a `MonadicProgram` with named `params` takes `len(params)` arguments;
- a `MonadicProgram` without `params` takes one, its packed input;
- a `Morphism` takes one, the domain tensor;
- any other callable takes its positional parameters without defaults, and a
  `*args` parameter disables the check.

An error raised inside a called function is re-raised with the file, line, and column
of the encoder call that reached it.

<!-- compile: false -->
```qvr
program embed : Pixel -> Latent
    let h = relu(layer_norm(W1 * x))
    return h

encoder C : LF
    dim Term = 64
    op App(f, x) |-> gelu(embed(f) + embed(x))
```

The program `embed` here assumes objects `Pixel` and `Latent` and a parameter `W1`
declared elsewhere in the module.

### Factory form

An encoder can instead name a factory from `quivers.structural.shapes` (the
[encoder API](../api/structural/encoder.md)). The bracketed options other than
`factory` are passed to it and checked against its parameters. One declaration uses
either a body or a factory, and a module may mix both kinds of declaration.

<!-- compile: cumulative -->
```qvr
signature Seq0
    sorts
        A : data [dim=16, vocab=["a", "b"]]
        Seq : object [dim=16]
    constructors
        Nil : -> Seq
        Cons : A, Seq -> Seq

encoder rnn_enc : Seq0 [factory=rnn_encoder, dim=16]
encoder tfm_enc : Seq0 [factory=transformer_encoder, dim=16]
encoder bow_enc : Seq0 [factory=bow_encoder]
```

| Factory | Signature | What it computes |
|---|---|---|
| `rnn_encoder(sig, dim)` | sequence | a right fold over the list with a GRU cell [@cho2014encoder] |
| `transformer_encoder(sig, dim)` | sequence | a right fold in which each `Cons` step applies an MLP to the concatenation of a linear projection of the head and a linear projection of the tail's vector |
| `bow_encoder(sig, dim)` | sequence | the sum of the element embeddings, which ignores order |
| `tree_lstm_encoder(sig, dim)` | tree | a child-sum tree-LSTM over binary trees [@tai2015treelstm] |
| `gnn_encoder(sig, iterations, dim, readout)` | graph | message passing with one MLP per edge kind, one GRU update per vertex kind, and a mean, sum, or max readout |

`transformer_encoder` contains no attention layer, so despite its name it is not a
transformer in the sense of Vaswani et al. [@vaswani2017attention]. Every factory
returns an [`Encoder`](../api/structural/encoder.md), the same runtime value an
explicit body produces.

### Sequence-shaped sugar

For sequence signatures, $\mathrm{Seq} = \mathrm{Nil} \mid \mathrm{Cons}(A, \mathrm{Seq})$,
a `Cons` rule can take one of two extra forms:

<!-- compile: false -->
```qvr
encoder RNN : Seq
    op Nil |-> 0.0
    op Cons(head, tail) recurrent state |-> gru_step(head, state)

encoder Tfm : Seq
    op Nil |-> 0.0
    op Cons(head, tail) attention prefix |-> tfm_step(head, prefix)
```

- `recurrent <state>` binds the named variable to the tail's vector, so the body is
  the step function of an ordinary right fold.
- `attention <prefix>` walks the list from the outside in. At step $i$ the body
  receives `prefix = [head_0, ..., head_{i-1}]`, the vectors of the heads already
  passed, and the encoder returns the output of the innermost step. Any attention over
  the prefix is computed by the body, and the form itself computes no attention weights.

`gru_step` and `tfm_step` stand for functions the module declares.

### Graph-shaped sugar

Graphs can contain cycles, so there is no bottom-up order. Message passing instead
updates every vertex's vector for a fixed number of rounds: each edge sends a message
from its source to its target, each vertex combines its incoming messages with its
current vector, and a readout pools the final vectors [@gilmer2017neural]. A graph
signature declares `vertex_kinds` and `edge_kinds`, and the encoder gives the
initialization, message, update, and readout functions:

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

An undirected edge kind (`Atom -- Atom`) sends messages both ways, and a directed one
(`Atom -> Bond`) only in its declared direction. The embedding tables and MLPs named
in the bodies stand for declarations elsewhere in the module.

## Stdlib shapes

`quivers.structural.shapes` provides signatures to go with the factories above:

- `seq_signature(name, dim)`: $\mathrm{Seq} = \mathrm{Nil} \mid \mathrm{Cons}(A, \mathrm{Seq})$;
- `tree_signature(name, dim)`:
  $\mathrm{Tree} = \mathrm{Leaf}(L) \mid \mathrm{Node}(B, \mathrm{Tree}, \mathrm{Tree})$;
- `graph_signature(name, vertex_kinds, edge_kinds)`: a graph signature with the given
  vertex and edge kinds.

They run on the same [`Encoder`](../api/structural/encoder.md) and
[`Decoder`](../api/structural/decoder.md) classes as declared encoders and decoders.

## See also

- [Decoders and losses](structural-compression-decoders-and-losses.md): the decoder,
  a Markov kernel from vectors to terms, and the loss declarations.
- [Weighted Deduction Systems](deduction.md): the chart parser, to which an encoder
  can attach to expose `chart.embedding(item)` as a differentiable value.
- [Tree autoencoders tutorial](../tutorials/qvr/12-structural-autoencoders.md):
  running entries, gradients, and checkpoints.
