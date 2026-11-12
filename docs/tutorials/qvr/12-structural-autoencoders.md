# 12. Tree autoencoders

A tree autoencoder encodes a tree into a vector by combining the vectors of each node's children from the leaves up, and decodes a vector into a distribution over trees [@pollack1990recursive; @tai2015treelstm]. The encoder is a fold over the tree, a [catamorphism](https://en.wikipedia.org/wiki/Catamorphism) [@meijer1991functional]. The package `quivers.structural` implements them: the trees are terms of a typed signature with binders, and the loss is the negative log probability that the decoder assigns to the original term. Run that loss as a computation of the [Quivers Indexed Effect Core](../../reference/qvr/index.md) (QIEC), the typed core language of QVR, on the reference machine, the library's interpreter for QIEC ([chapter 9](09-indexed-data.md#4-run-the-summary)), and differentiate through it.

The gallery program [`term_autoencoder.qvr`](../../examples/source/term_autoencoder.qvr) declares a signature for [simply typed lambda terms](https://en.wikipedia.org/wiki/Simply_typed_lambda_calculus), an encoder `Enc`, a decoder `Dec`, and a loss `reconstruct`. Its elaboration to QIEC records which encoder and decoder each step calls and the types they accept, as `Compute` requests. The encoder and decoder themselves are PyTorch modules, which `Program.run` binds to those requests as *attachments*: values held only in the running Python process and never serialized with the module. The returned loss keeps its [autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html#computing-gradients) history [@pytorchAutogradTutorial], so `.backward()` reaches the parameters of both modules.

## 1. Compile and inspect the entry points

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

`Enc`, `Dec`, and `Dec__nll` (the decoder's negative log likelihood) are entry points answered by attachments. `reconstruct` is the loss, which calls `Enc` and then `Dec__nll`.

## 2. Construct a typed term {#2-construct-typed-host-data}

Build the term $\lambda x : \mathit{base}.\ \mathit{plus}\ x$ with the Python constructors of `quivers.structural`:

```python
term = make_term(
    "Lam",
    make_term("Base", "base"),
    make_term("App", make_term("Const", "plus"), bound_var(0)),
)
```

`bound_var(0)` is a [de Bruijn index](https://en.wikipedia.org/wiki/De_Bruijn_index): it refers to the nearest enclosing binder. While folding the term, the encoder checks each constructor, the number of its arguments, the scope of each bound variable, and the vocabulary of each data leaf.

## 3. Run the loss {#3-invoke-the-checked-loss}

```python
run = structural.run("reconstruct", term)
assert torch.is_tensor(run.value)
assert run.value.shape == ()
assert run.result.runtime == "core+structural"

classic_loss = structural.losses.evaluate({"term": term})
torch.testing.assert_close(run.value, classic_loss)
print(float(run.value.detach()))
```

```text
8.16720962524414
```

The reference machine and the loss registry (`structural.losses`) of the [compiled runtime](../../guides/dsl-overview.md#compilation-pipeline), the PyTorch implementation of the module, give the same scalar. The runtime label names both providers: `core` runs the QIEC computation, and `structural` answers the encoder and decoder requests.

## 4. Differentiate through the run {#4-differentiate-through-the-entry-boundary}

```python
for parameter in structural.parameters():
    parameter.grad = None

run.value.backward()
assert any(parameter.grad is not None for parameter in structural.parameters())
print(len(structural.state_dict()), "checkpoint tensors")
```

```text
34 checkpoint tensors
```

The handler for the attachments returns PyTorch tensors unchanged, so the autograd graph to both modules survives the run on the reference machine. The encoder and decoder are registered under fixed names, so an optimizer over `structural.parameters()` and a checkpoint from `structural.state_dict()` include them.

## 5. Serialization and transpile targets

Serializing `structural.qiec` writes the descriptors of the attachments, their identifiers, and their checked types, but not the PyTorch modules. Another process may load the serialized module, but it must compile or attach compatible encoder and decoder providers before running `reconstruct`.

No probabilistic programming target supports these attachments. Checking against one reports `qiec:capability:neural-attachment:<entry>`:

```bash
qvr check --target numpyro docs/examples/source/term_autoencoder.qvr
```

The [term autoencoder example](../../examples/term-autoencoder.md) develops the encoder and decoder rules. The [generated-computation reference](../../reference/qvr/generated-computations.md#structural-signatures-and-autoencoders) specifies how attachments are bound.

## Next

[Chapter 13](13-release-workflow.md) checks a module against its targets, inspects its entry points, and verifies the editor tooling before a release.
