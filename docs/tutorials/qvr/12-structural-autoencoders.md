# Structural autoencoders

This tutorial executes a structural reconstruction loss through the same
entry boundary as an ordinary QIEC computation. The source is the gallery's
[`term_autoencoder.qvr`](../../examples/source/term_autoencoder.qvr), which
declares a typed lambda-term signature, an encoder, a decoder, and a loss.

The runtime attachment boundary separates two responsibilities: the stable
module records `Compute` requests and their types, while `Program.run` attaches
the compiled PyTorch modules and preserves their autograd graph.

A term is a [tree with typed nodes](../../examples/term-autoencoder.md#understanding-the-model). The encoder replaces each subtree with a
vector, combining child vectors until one vector describes the whole term.
The decoder uses that vector to assign probabilities to choices of nodes and
children. Reconstruction loss asks how much probability it assigns to the
original tree; sampling a decoded tree need not reproduce the original exactly.

The checked computation records which encoder and decoder should be called
and the types they accept. The runtime supplies their trained PyTorch modules.
Preserving the returned tensor's [gradient history](../python/01-first-quiver.md#working-as-a-differentiable-module) is what lets a reconstruction
error update both modules through the same entry point. The [PyTorch autograd walkthrough](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) shows how a loss retains its computation graph and how `backward()` accumulates parameter gradients [@pytorchAutogradTutorial].

## 1. Compile and inspect the entries

```python
import torch
from quivers.dsl import load
from quivers.structural import bound_var, make_term

torch.manual_seed(0)
structural = load("docs/examples/source/term_autoencoder.qvr")
names = [entry.name for entry in structural.entry_points()]
assert names == ["Enc", "Dec", "Dec__nll", "reconstruct"]
print(names)
```

`Enc`, `Dec`, and `Dec__nll` are host-computation entries. `reconstruct` is the
loss graph that calls the first and third entries in sequence.

## 2. Construct typed host data

The signature describes simply typed lambda terms. Build
`λx : base. plus x` with the structural host API:

```python
term = make_term(
    "Lam",
    make_term("Base", "base"),
    make_term("App", make_term("Const", "plus"), bound_var(0)),
)
```

`bound_var(0)` refers to the nearest binder. The encoder checks constructors,
argument arities, binder scope, and data vocabulary while it folds the term.

## 3. Invoke the checked loss

```python
run = structural.run("reconstruct", term)
assert torch.is_tensor(run.value)
assert run.value.shape == ()
assert run.result.runtime == "core+structural"

classic_loss = structural.losses.evaluate({"term": term})
torch.testing.assert_close(run.value, classic_loss)
print(float(run.value.detach()))
```

The checked entry and classic loss registry produce the same scalar. The
runtime label names both providers: `core` evaluates the stable computation,
and `structural` handles the encoder and decoder attachments.

## 4. Differentiate through the entry boundary

```python
for parameter in structural.parameters():
    parameter.grad = None

run.value.backward()
assert any(parameter.grad is not None for parameter in structural.parameters())
print(len(structural.state_dict()), "checkpoint tensors")
```

The attachment handler returns a PyTorch tensor without converting it to a
plain scalar, so the autograd edges to both modules survive the reference
machine. The compiled encoder and decoder are registered under stable names;
an optimizer over `structural.parameters()` and a checkpoint from
`structural.state_dict()` thus include them.

## 5. Understand serialization and targets

Serializing `structural.qiec` writes attachment descriptors, stable IDs, and
checked types. It does not serialize the PyTorch modules. A different process
may deserialize the core graph, but it must compile or otherwise attach
compatible encoder and decoder providers before invoking `reconstruct`.

Current probabilistic-programming targets do not carry these neural
attachments. Checking one produces
`qiec:capability:neural-attachment:<entry>`:

```bash
qvr check --target numpyro docs/examples/source/term_autoencoder.qvr
```

This refusal preserves the loss graph. Silently emitting only the source
declarations would suggest that the target can compute a reconstruction value
when it cannot.

The [Term Autoencoder example](../../examples/term-autoencoder.md) develops the
encoder and decoder rules in more detail. The [generated-computation
reference](../../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders)
specifies the attachment boundary.
