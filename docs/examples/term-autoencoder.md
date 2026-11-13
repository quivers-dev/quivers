# Term Autoencoder

## Overview

A stochastic autoencoder on simply typed lambda terms, built from the four structural declarations: a `signature` fixing the multi-sorted term algebra, an [`Encoder`](../api/structural/encoder.md#quivers.structural.encoder.Encoder) folding each term bottom-up into one fixed-width code vector, a [`Decoder`](../api/structural/decoder.md#quivers.structural.decoder.Decoder) corecursively expanding a code back into a term, and a loss head scoring the round trip by reconstruction negative log-likelihood.

This example wires all four declarations together in one module. It shows the keyword-led encoder body (an explicit `op` rule alongside compiler-scaffolded defaults), the per-encoder / per-decoder `dim` override, a binder that threads a [de Bruijn](https://en.wikipedia.org/wiki/De_Bruijn_index) type annotation through the recursion, and a loss attached to a named site via `[on=...]`. The `#!` doc comments attach to the declaration that follows and are carried on its AST node's `docs` field.

## QVR source

```qvr
# Term Autoencoder
#
# A stochastic autoencoder on simply typed lambda terms: a
# multi-sorted signature fixes the term algebra, an encoder folds
# each term bottom-up into one fixed-width code vector, a decoder
# corecursively expands a code back into a term, and a loss head
# scores the round trip by reconstruction negative log-likelihood.

#! Simply typed lambda terms: Term and Type are the recursively
#! decoded object sorts; Name is a data sort with a closed
#! vocabulary that the decoder samples and scores leaves from.
signature STLC
    sorts
        Term : object
        Type : object [dim=24]
        Name : data   [dim=24, vocab=["f", "x", "plus", "base"]]
    constructors
        Const : Name -> Term
        App   : Term, Term -> Term
        Base  : Name -> Type
    binders
        Lam : binds (x : Term : ty : Type) in (body : Term) -> Term

#! Fold a term bottom-up into one code vector. App carries an
#! explicit op rule; every other operator keeps its learned
#! scaffold. Bound variables enter the context as their type's
#! annotation embedding.
encoder Enc : STLC
    dim Term = 24
    op App(fun, arg) |-> relu(fun + arg)
    var_init Term from Type as ty |-> ty

#! Expand a code vector back into a term, scoring constructor,
#! binder, and bound-variable candidates at every node down to a
#! bounded depth.
decoder Dec : STLC [depth=6]
    dim Term = 24
    body |-> recursive

#! Reconstruction negative log-likelihood of the round trip.
loss reconstruct [weight=1.0, on=decoder(Dec)]
    -Dec.log_prob(term, Enc(term))
```

## Understanding the model

Consider an application tree `App(function, argument)`. The encoder first
encodes the function and argument subtrees, then combines those vectors into
one representation. A decoder runs in the generative direction: it chooses
a node kind and then predicts children. It thus defines a distribution
over possible trees rather than an exact inverse of compression.

[Reconstruction negative log likelihood](../tutorials/qvr/12-structural-autoencoders.md#3-invoke-the-checked-loss) rewards assigning high probability
to the original tree. It can train without sampling the correct tree on every
iteration. Binder scope is an additional structural constraint: a [de Bruijn index](../tutorials/qvr/12-structural-autoencoders.md#2-construct-typed-host-data) `0` refers to the nearest enclosing binder, so changing a variable's
printed name does not change which binder it denotes.

## Walkthrough

### The signature

<!-- compile: false -->
```qvr
signature STLC
    sorts
        Term : object
        Type : object [dim=24]
        Name : data   [dim=24, vocab=["f", "x", "plus", "base"]]
    constructors
        Const : Name -> Term
        App   : Term, Term -> Term
        Base  : Name -> Type
    binders
        Lam : binds (x : Term : ty : Type) in (body : Term) -> Term
```

`Term` and `Type` are `object` sorts, decoded recursively; `Name` is a `data` sort whose closed `vocab` supplies the tokens the decoder samples and scores leaves from. `Term` deliberately omits its `dim`: each encoder or decoder over the signature must then supply one via a `dim Term = 24` body entry, and the compiler raises when neither the signature nor the block does. The `Lam` binder introduces a `Term`-sorted variable annotated by a `Type`; the framework threads that annotation through a typed de Bruijn context during both encoding and decoding.

### The encoder

<!-- compile: false -->
```qvr
encoder Enc : STLC
    dim Term = 24
    op App(fun, arg) |-> relu(fun + arg)
    var_init Term from Type as ty |-> ty
```

Every encoder body entry is keyword-led. The `op` keyword introduces a per-operation rewrite rule: `App`'s two child embeddings are summed and passed through `relu`. Operators without a rule (`Const`, `Base`, `Lam`) receive a compiler-scaffolded 2-layer MLP with the correct per-argument dimensions, so a body may override as few or as many operations as the model demands. The `var_init` entry sets the embedding a `Lam`-bound variable enters the context with; here the type annotation's embedding is passed through unchanged.

### The decoder

<!-- compile: false -->
```qvr
decoder Dec : STLC [depth=6]
    dim Term = 24
    body |-> recursive
```

The decoder header names its signature after `:`, exactly like the encoder header. `body |-> recursive` accepts the scaffolded corecursion: at every node the runtime scores constructor, binder, and bound-variable candidates for the current sort, splits the parent vector into per-child vectors, and recurses down to the `depth` bound. Each slot (structure, primitive, factor, binder selection) can instead be overridden with an explicit rule, as described in the [decoders and losses guide](../guides/structural-compression-decoders-and-losses.md).

### The loss

<!-- compile: false -->
```qvr
loss reconstruct [weight=1.0, on=decoder(Dec)]
    -Dec.log_prob(term, Enc(term))
```

A `loss` declaration registers a weighted scalar head in the module's [`LossRegistry`](../api/structural/losses.md#quivers.structural.losses.LossRegistry). The body is compiled as a let-expression closure over an environment: `Enc` and `Dec` resolve to the compiled module's own artifacts, while `term` is supplied by the caller at evaluation time. The `on=decoder(Dec)` option attaches the entry to a named site, so a training driver may evaluate it selectively via [`evaluate_on`](../api/structural/losses.md#quivers.structural.losses.LossRegistry.evaluate_on); `weight=1.0` sets the multiplier applied by [`evaluate`](../api/structural/losses.md#quivers.structural.losses.LossRegistry.evaluate).

The same declarations lower to a checked QIEC graph. `STLC` becomes an opaque term family indexed by `Term`, `Type`, and `Name`; `Enc`, `Dec`, and `Dec__nll` become typed host-computation requests; and `reconstruct` calls `Enc` followed by `Dec__nll`. `program.run("reconstruct", term)` supplies the compiled PyTorch modules for those requests and validates their host results against the checked types. Its result is the same tensor as `losses.evaluate({"term": term})`, including the autograd edges to encoder and decoder parameters. The encoder and decoder are registered submodules, so `program.parameters()` and `program.state_dict()` expose their learned tensors. Targets without a structural provider reject the named computation under `qiec:capability:neural-attachment`.

## Try it

### Compile, encode, decode

Compile the module with [`quivers.dsl.load`](../api/dsl/parser.md), build a term with [`make_term`](../api/structural/signature.md#quivers.structural.signature.make_term) and [`bound_var`](../api/structural/signature.md#quivers.structural.signature.bound_var), and run it through the round trip.

```python
import torch
from quivers.dsl import load
from quivers.structural import bound_var, make_term

torch.manual_seed(0)
prog = load("docs/examples/source/term_autoencoder.qvr")
enc = prog.encoders["Enc"]
dec = prog.decoders["Dec"]

# \x : base. plus x
term = make_term(
    "Lam",
    make_term("Base", "base"),
    make_term("App", make_term("Const", "plus"), bound_var(0)),
)
code = enc(term)
print("code shape:", tuple(code.shape))     # (24,)

recon = dec(code)
print("reconstruction root op:", recon.op)
```

### Score and train the round trip

[`Decoder.log_prob`](../api/structural/decoder.md) scores an observed term against a code vector; the registry's `evaluate` sums every registered loss under a caller-supplied environment, and `prog.run("reconstruct", term)` computes the same loss through the checked `reconstruct -> Enc -> Dec__nll` graph. The block first asserts that the two paths agree on every term. It then trains the encoder and decoder on six distinct terms, two of which bind variables (one with a nested binder whose body refers to the outer variable, de Bruijn index 1), and checks reconstruction two ways: the decoder must assign each training term probability above 0.99 given its own code, and 20 independent decodes of each code must reproduce the term exactly, compared as full trees with [`Term.to_tuple`](../api/structural/signature.md#quivers.structural.signature.Term.to_tuple) so that binders and indices count. Two held-out terms are reported separately.

```python
import torch
from quivers.dsl import load
from quivers.structural import bound_var, make_term

torch.manual_seed(0)
prog = load("docs/examples/source/term_autoencoder.qvr")
enc = prog.encoders["Enc"]
dec = prog.decoders["Dec"]

const = lambda name: make_term("Const", name)
app = lambda fun, arg: make_term("App", fun, arg)
lam = lambda body: make_term("Lam", make_term("Base", "base"), body)

train = {
    "plus": const("plus"),
    "f x": app(const("f"), const("x")),
    "\\v. v": lam(bound_var(0)),
    "\\v. plus v": lam(app(const("plus"), bound_var(0))),
    "\\v. f (plus v)": lam(app(const("f"), app(const("plus"), bound_var(0)))),
    "\\v. \\w. plus v": lam(lam(app(const("plus"), bound_var(1)))),
}
held_out = {
    "\\v. f v": lam(app(const("f"), bound_var(0))),
    "f (plus x)": app(const("f"), app(const("plus"), const("x"))),
}


def loss_of(term):
    return prog.losses.evaluate({"term": term})


# The data-leaf embedding tables allocate parameters on first lookup,
# so every term is scored once before the optimizer collects them.
for term in [*train.values(), *held_out.values()]:
    checked = prog.run("reconstruct", term)
    torch.testing.assert_close(checked.value, loss_of(term))
    assert checked.result.runtime == "core+structural"

params = list(enc.parameters()) + list(dec.parameters())
opt = torch.optim.Adam(params, lr=1e-2)
for _ in range(300):
    opt.zero_grad()
    loss = sum(loss_of(term) for term in train.values())
    loss.backward()
    opt.step()


def recovered(term, n_draws=20):
    """How many independent decodes of the term's code reproduce it."""
    code = enc(term)
    return sum(dec(code).to_tuple() == term.to_tuple() for _ in range(n_draws))


with torch.no_grad():
    for name, term in train.items():
        p_self = float(torch.exp(-loss_of(term)))
        hits = recovered(term)
        print(f"train     {name:18} p(term | code) = {p_self:.3f}, exact decodes {hits}/20")
        assert p_self > 0.99 and hits == 20, name
    for name, term in held_out.items():
        p_self = float(torch.exp(-loss_of(term)))
        print(f"held out  {name:18} p(term | code) = {p_self:.3f}, exact decodes {recovered(term)}/20")
```

The six training terms are reconstructed exactly, binders and de Bruijn indices included. The held-out terms are not: six training terms teach the decoder to reproduce those trees, not to generalize to new combinations of their parts, and a reconstruction objective on so few terms gives no pressure to do so. Generalization would need a larger and more varied training set.

## Categorical perspective

The encoder is a [Σ-algebra homomorphism](https://ncatlab.org/nlab/show/algebra+over+an+endofunctor) $T_\Sigma \to \mathrm{Vec}_D$: the framework supplies the recursion over the term tree, and the analyst supplies (or accepts scaffolds for) one parametric function per operation. The decoder is a [Kleisli coalgebra](https://ncatlab.org/nlab/show/coalgebra+over+an+endofunctor) $\mathrm{Vec}_D \to \mathrm{Kern}(T_\Sigma)$, corecursively generating a distribution over terms with a per-term `log_prob`. Their composite is a stochastic endomorphism on $T_\Sigma$, and the loss head scores how far that endomorphism sits from the identity on the observed term.

## See also

- [Structural Compression: Signatures and Encoders](../guides/structural-compression-signatures-and-encoders.md)
- [Structural Compression: Decoders and Losses](../guides/structural-compression-decoders-and-losses.md)
- [Structural Compression semantics](../semantics/structural.md)
- API reference: [Signatures](../api/structural/signature.md), [Encoders](../api/structural/encoder.md), [Decoders](../api/structural/decoder.md), [Loss Registry](../api/structural/losses.md)
