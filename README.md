<h1 align="center">Quivers</h1>

<p align="center">
  <em>Typed, composable models with PyTorch.</em>
</p>

<p align="center">
  <a href="https://github.com/quivers-dev/quivers/actions/workflows/ci.yml"><img src="https://github.com/quivers-dev/quivers/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://quivers.dev"><img src="https://github.com/quivers-dev/quivers/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <a href="https://pypi.org/project/quivers/"><img src="https://img.shields.io/pypi/v/quivers" alt="PyPI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14%2B-blue" alt="Python 3.14+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="https://quivers.dev/getting-started/quickstart/"><strong>Quickstart</strong></a>
  ·
  <a href="https://quivers.dev/tutorials/"><strong>Tutorials</strong></a>
  ·
  <a href="https://quivers.dev/examples/"><strong>Examples</strong></a>
  ·
  <a href="https://quivers.dev/reference/qvr/"><strong>Language reference</strong></a>
  ·
  <a href="https://quivers.dev/api/"><strong>Python API</strong></a>
</p>

---

Quivers is a Python library for constructing and fitting models from typed,
composable components. It provides weighted relations, Markov kernels, and
probabilistic programs backed by PyTorch tensors and automatic differentiation.
The Python API exposes these components directly, while QVR, its typed
functional probabilistic programming language, lets you describe models in
source files that can be checked, executed, and translated to other languages.

The library draws on category theory: components have explicit input and output
types, and composition connects compatible components. A choice of algebra
specifies how weights combine, supporting probabilistic, Boolean, fuzzy, and
optimization-based interpretations. Applications include Bayesian regression,
sequence models, weighted parsing, and models of structured data.

## Features

- **Composable models:** build weighted relations and discrete or continuous
  probabilistic computations through the Python API or QVR.
- **Learning and inference:** train differentiable components with PyTorch
  optimizers, or fit probabilistic programs with variational inference and
  Markov chain Monte Carlo (MCMC).
- **Statistical modeling:** write mixed-effects regressions with formulas,
  work with dataframes, and inspect fits with posterior predictive checks,
  convergence diagnostics, and ArviZ integration.
- **Structured models:** define weighted deduction systems, including parsers,
  and encoders and decoders for recursively structured data.
- **Language tools and interoperability:** check QVR programs from the command
  line, use editor and notebook integrations, and transpile supported models
  to other probabilistic programming languages. See the
  [target support matrix](https://quivers.dev/transpile-support/) for supported
  features and restrictions.

## Installation

Quivers requires **Python 3.14 or later**:

```bash
python -m pip install quivers
```

Install optional dependencies for the features you use:

```bash
python -m pip install 'quivers[formulas,diagnostics]'
```

Other extras include `data` for dataframe utilities, `repl` for the interactive
shell and Jupyter kernel, `lsp` for the language server, and `targets` for Python
target runtimes. The [installation guide](https://quivers.dev/getting-started/installation/)
covers these extras, editor setup, and dependencies for individual targets.

## Quick start

### Compose a model in Python

A Markov kernel maps each input to a probability distribution over outputs.
This example composes two learnable kernels through a shared hidden state:

```python
from quivers import FinSet, Program, stochastic

inputs = FinSet(name="Input", cardinality=3)
hidden = FinSet(name="Hidden", cardinality=4)
outputs = FinSet(name="Output", cardinality=2)

encode = stochastic(inputs, hidden)
decode = stochastic(hidden, outputs)
model = Program(encode >> decode)

probabilities = model()
print(probabilities.shape)  # torch.Size([3, 2])
```

Each row is a distribution over the two outputs, with the hidden state summed
out. `Program` is a PyTorch `nn.Module`, so its learnable parameters can be
trained with PyTorch optimizers. The
[Python tutorials](https://quivers.dev/tutorials/python/01-first-quiver/)
introduce weighted relations, composition, and learning.

### Write a probabilistic program in QVR

QVR separates a model's generative assumptions from the inference algorithm.
For instance, save this Bayesian linear regression as `regression.qvr`:

<!-- GitHub Linguist fallback: QVR currently uses the R lexer. -->
```R
object Item : FinSet 100

program regression : Item -> Item
    sample sigma  <- HalfNormal(scale=1.0)
    sample beta_0 <- Normal(loc=0.0, scale=5.0)
    sample beta_1 <- Normal(loc=0.0, scale=2.0)
    let mu = beta_0 + beta_1 * x
    observe y : Item <- Normal(loc=mu, scale=sigma)
    return y

export regression
```

The `sample` steps specify priors, and `observe` specifies the likelihood.
The predictor `x` and observed responses `y` are provided as tensors when
fitting the model. Check the file from the command line:

```bash
qvr check regression.qvr
```

Load the checked model in Python:

```python
from quivers.dsl import load

regression = load("regression.qvr").morphism
```

The [end-to-end quickstart](https://quivers.dev/getting-started/quickstart/)
shows how to supply data, fit this model, and summarize its posterior. For
regression from a dataframe, the
[formula interface](https://quivers.dev/guides/analysis-data-and-formulas/)
provides an alternative to writing QVR by hand.

## Documentation

- [Getting started](https://quivers.dev/getting-started/quickstart/): installation,
  a first fit, and the core concepts.
- [Tutorials](https://quivers.dev/tutorials/): separate tracks for QVR and the
  Python API.
- [Examples](https://quivers.dev/examples/): complete models, including
  hierarchical models, state-space models, neural networks, and grammars.
- [Guides](https://quivers.dev/guides/): inference, data and formulas,
  diagnostics, and transpilation.
- [Language reference](https://quivers.dev/reference/qvr/),
  [Python API](https://quivers.dev/api/), and
  [semantics](https://quivers.dev/semantics/): syntax, public interfaces, and
  mathematical foundations.

## Project status

Quivers is alpha software. The language and Python API may change; see the
[changelog](CHANGELOG.md) and
[releases](https://github.com/quivers-dev/quivers/releases) for details.

## Contributing and support

Bug reports, feature requests, and documentation improvements are welcome through
[GitHub Issues](https://github.com/quivers-dev/quivers/issues). For a bug report,
include a minimal Python or QVR example, the package version, and the full error
message. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and
the contribution workflow.

## Acknowledgments

This project was developed by [Aaron Steven White](https://aaronstevenwhite.io/)
at the University of Rochester with support from the National Science
Foundation (NSF-BCS-2237175 *CAREER: Logical Form Induction*,
NSF-BCS-2040831 *Computational Modeling of the Internal Structure of Events*).

## License

Quivers is released under the [MIT License](LICENSE).
