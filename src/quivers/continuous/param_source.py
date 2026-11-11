"""Pluggable parameter sources for conditional distribution families.

A [`ParamSource`][quivers.continuous.param_source.ParamSource]
produces a `(batch, param_dim)` tensor from a per-row input. Every
`ConditionalX` family in
[`quivers.continuous.families`][quivers.continuous.families] uses
one to convert its input into the flattened parameter vector its
underlying distribution needs.

The primitives:

* [`LinearSource`][quivers.continuous.param_source.LinearSource]:
  the default; one `nn.Linear`, no nonlinearity. Matches the
  single-linear layer the transpile backends emit exactly, so a
  kernel morphism on this source is numerically equivalent to its
  transpiled counterpart.
* [`MLPSource`][quivers.continuous.param_source.MLPSource]:
  a multi-layer perceptron with user-configurable hidden widths and
  activation. Selected by `[param_source=mlp]`; a kernel is linear
  unless it asks for this.
* [`LookupSource`][quivers.continuous.param_source.LookupSource]:
  a learnable per-entry embedding table, the discrete-domain
  standard.
* [`EmbeddingSource`][quivers.continuous.param_source.EmbeddingSource]:
  an embedding table piped through a downstream `ParamSource`
  (embedding + MLP head, the standard categorical-input pattern).
* [`AttentionSource`][quivers.continuous.param_source.AttentionSource]:
  single-head self-attention over the input dimension, a useful
  primitive for set-valued inputs.
* [`IdentitySource`][quivers.continuous.param_source.IdentitySource]:
  pass the input through unchanged (parameters supplied as data).
* [`FunctionSource`][quivers.continuous.param_source.FunctionSource]:
  wraps an arbitrary `Callable`, letting a user drop in any
  `nn.Module` without subclassing.
* [`ComposeSource`][quivers.continuous.param_source.ComposeSource]:
  categorical composition of two param sources, useful for
  building sequential architectures out of primitives.

The DSL surface accepts a `[param_source=...]` option on
`morphism` declarations, with the hidden widths given either as the
option's arguments or through `hidden_dim`:

    morphism trans : State -> State [param_source=linear] ~ Normal
    morphism trans : State -> State [param_source=mlp] ~ Normal
    morphism trans : State -> State [param_source=mlp(64, 64)] ~ Normal
    morphism trans : State -> State [param_source=mlp, hidden_dim=[64, 32]] ~ Normal

One width per hidden layer, so the sequence says how many as well as
how wide; a bare `hidden_dim=64` is the one-layer case. The call form
is parsed by
[`param_source_from_option`][quivers.continuous.param_source.param_source_from_option].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from quivers.core.objects import SetObject
from quivers.continuous.morphisms import AnySpace
from quivers.continuous.spaces import ContinuousSpace


class ParamSource(nn.Module, ABC):
    """A learnable map from per-row input to a flat parameter vector.

    Every conditional distribution family holds a `ParamSource`
    instance and defers its parameter computation to it. Subclasses
    override `forward` and declare `param_dim`.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        ...

    @property
    @abstractmethod
    def param_dim(self) -> int:  # pragma: no cover - abstract
        ...


class LinearSource(ParamSource):
    """Single `nn.Linear(domain_dim, param_dim)`; no nonlinearity.

    This is the parameter source that matches the transpile
    backends' emit: the DSL `morphism f : A -> B [role=kernel] ~
    Family` lowers to a single linear layer `Family(loc = W x, ...)`
    on every backend. Configuring a runtime kernel with
    `LinearSource(...)` makes the runtime numerically equivalent to
    its transpiled counterpart.
    """

    def __init__(self, domain_dim: int, param_dim: int, bias: bool = True) -> None:
        super().__init__()
        self._domain_dim = int(domain_dim)
        self._param_dim = int(param_dim)
        self.linear = nn.Linear(domain_dim, param_dim, bias=bias)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.linear(x)


class MLPSource(ParamSource):
    """Multi-layer perceptron with configurable hidden widths and
    activation. The default `hidden_dims=(64, 64)` yields a
    two-hidden-layer, tanh-activated network; a user who wants a
    wider, deeper, or differently-activated network drops in the
    same constructor.
    """

    def __init__(
        self,
        domain_dim: int,
        param_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._domain_dim = int(domain_dim)
        self._param_dim = int(param_dim)
        layers: list[nn.Module] = []
        widths = [domain_dim, *hidden_dims, param_dim]
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1], bias=bias))
            if i < len(widths) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class LookupSource(ParamSource):
    """Per-entry learnable parameter table indexed by an integer
    input. The categorical / discrete-input parameter source: the
    input is a `LongTensor` of category indices, the output is the
    per-index parameter vector.
    """

    def __init__(self, n_entries: int, param_dim: int) -> None:
        super().__init__()
        self._n_entries = int(n_entries)
        self._param_dim = int(param_dim)
        self.table = nn.Parameter(torch.randn(n_entries, param_dim) * 0.01)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def forward(self, x: Tensor) -> Tensor:
        idx = x.long().reshape(-1)
        return self.table[idx]


class EmbeddingSource(ParamSource):
    """Embedding table followed by a downstream `ParamSource`. The
    canonical embedding + MLP head pattern: categorical inputs pass
    through an `nn.Embedding` and the resulting dense vector feeds a
    user-supplied head (linear, MLP, or attention).
    """

    def __init__(
        self,
        n_entries: int,
        embed_dim: int,
        head: ParamSource,
    ) -> None:
        super().__init__()
        if head._domain_dim != embed_dim:
            raise ValueError(
                "EmbeddingSource: head's domain_dim must equal embed_dim; "
                f"got head.domain_dim={head._domain_dim}, embed_dim={embed_dim}"
            )
        self.embed = nn.Embedding(n_entries, embed_dim)
        self.head = head
        self._embed_dim = int(embed_dim)

    @property
    def param_dim(self) -> int:
        return self.head.param_dim

    def forward(self, x: Tensor) -> Tensor:
        idx = x.long().reshape(-1)
        emb = self.embed(idx)
        return self.head(emb)


class AttentionSource(ParamSource):
    """Single-head self-attention over the input feature dimension,
    followed by an aggregation and a linear head. Useful when the
    input is a set of features whose ordering carries no information
    and the model should be permutation-equivariant.

    For a `(batch, seq_len, domain_dim)` input, computes
    scaled-dot-product attention across the sequence dimension,
    aggregates via mean-pool, and projects to `param_dim` via a
    linear head. When the input is `(batch, domain_dim)`, treats it
    as a length-one sequence.
    """

    def __init__(
        self,
        domain_dim: int,
        param_dim: int,
        num_heads: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self._domain_dim = int(domain_dim)
        self._param_dim = int(param_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=domain_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=bias,
        )
        self.head = nn.Linear(domain_dim, param_dim, bias=bias)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        attended, _ = self.attn(x, x, x, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.head(pooled)


class IdentitySource(ParamSource):
    """Passes the input through unchanged. Useful when parameters
    come directly from the host (e.g. a design matrix or an already-
    computed feature vector) and no further transformation is
    needed. `param_dim` equals the input's last-axis size and must
    be declared at construction.
    """

    def __init__(self, param_dim: int) -> None:
        super().__init__()
        self._param_dim = int(param_dim)

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x


class FunctionSource(ParamSource):
    """Wraps an arbitrary `nn.Module` or callable as a `ParamSource`.
    The user supplies the module and declares its output dim; the
    machinery around conditional families sees a uniform interface.
    """

    def __init__(self, fn: Callable[[Tensor], Tensor], param_dim: int) -> None:
        super().__init__()
        self._param_dim = int(param_dim)
        if isinstance(fn, nn.Module):
            self._module: nn.Module | None = fn
            self._fn: Callable[[Tensor], Tensor] | None = None
        else:
            self._module = None
            self._fn = fn

    @property
    def param_dim(self) -> int:
        return self._param_dim

    def forward(self, x: Tensor) -> Tensor:
        if self._module is not None:
            return self._module(x)
        assert self._fn is not None
        return self._fn(x)


class ComposeSource(ParamSource):
    """Categorical composition of two param sources: `outer(inner(x))`.
    `inner.param_dim` must equal `outer._domain_dim`. Enables
    building sequential architectures from primitives (e.g.
    `Compose(MLPSource(...), LinearSource(...))`).
    """

    def __init__(self, outer: ParamSource, inner: ParamSource) -> None:
        super().__init__()
        if inner.param_dim != outer._domain_dim:
            raise ValueError(
                "ComposeSource: inner.param_dim must equal outer.domain_dim; "
                f"got inner.param_dim={inner.param_dim}, "
                f"outer.domain_dim={outer._domain_dim}"
            )
        self.outer = outer
        self.inner = inner

    @property
    def param_dim(self) -> int:
        return self.outer.param_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.outer(self.inner(x))


#: The kind a family gets when nothing selects one.
_DEFAULT_SOURCE_KIND = "linear"


def _split_source_option(option_value: str | None) -> tuple[str, dict]:
    """Split a ``[param_source=...]`` option into its kind and the
    arguments it carries.

    ``"mlp"`` is a bare kind; ``"mlp(64, 32)"`` carries positional
    hidden widths; ``"attention(heads=4)"`` carries keywords. Absent,
    the kind is the default.
    """
    if option_value is None:
        return _DEFAULT_SOURCE_KIND, {}
    val = option_value.strip()
    if "(" not in val:
        return val, {}
    kind, rest = val.split("(", 1)
    rest = rest.rstrip(")").strip()
    if not rest:
        return kind.strip(), {}
    kwargs: dict[str, object] = {}
    positional: list[int] = []
    for token in rest.split(","):
        token = token.strip()
        if "=" in token:
            k, v = token.split("=", 1)
            kwargs[k.strip()] = int(v.strip())
        else:
            positional.append(int(token))
    if positional:
        kwargs["hidden_dims"] = tuple(positional)
    return kind.strip(), kwargs


def make_param_source(
    domain: AnySpace,
    param_dim: int,
    kind: str = _DEFAULT_SOURCE_KIND,
    **kwargs,
) -> ParamSource:
    """Factory that dispatches the `[param_source=...]` DSL option
    to the concrete class.

    The default is `LinearSource`, so a morphism declared
    ``f : X -> Y ~ Normal`` maps its input to the family's parameters
    the way its arrow reads: linearly. A model that wants a
    nonlinearity between its input and its parameters asks for one,
    and the fact then appears in the source rather than in a default.

    Recognised kinds:
    * ``"lookup"`` — always used when the domain is a `SetObject`,
      regardless of the requested kind.
    * ``"linear"`` — the default; one `nn.Linear`.
    * ``"mlp"`` — `hidden_dims=(64, 64)` unless overridden by
      ``hidden_dims`` or ``hidden_dim`` kwargs.
    * ``"identity"`` — pass through.
    * ``"attention"`` — self-attention head.
    """
    if isinstance(domain, SetObject):
        return LookupSource(domain.size, param_dim)
    if not isinstance(domain, ContinuousSpace):
        raise TypeError(
            f"make_param_source: unsupported domain type {type(domain).__name__}"
        )
    dim = domain.dim
    if kind == "linear":
        return LinearSource(dim, param_dim, **kwargs)
    if kind == "identity":
        return IdentitySource(param_dim)
    if kind == "attention":
        if "heads" in kwargs:
            kwargs["num_heads"] = kwargs.pop("heads")
        return AttentionSource(dim, param_dim, **kwargs)
    if kind == "mlp":
        hidden_dims = kwargs.pop("hidden_dims", None)
        if hidden_dims is None:
            hidden_dim = kwargs.pop("hidden_dim", 64)
            hidden_dims = (int(hidden_dim), int(hidden_dim))
        return MLPSource(dim, param_dim, hidden_dims=hidden_dims, **kwargs)
    raise ValueError(f"make_param_source: unknown kind {kind!r}")


#: Source kinds with hidden layers, so a width is something they read.
_HIDDEN_LAYER_KINDS: frozenset[str] = frozenset({"mlp"})


def _make_source(
    domain: AnySpace,
    param_dim: int,
    hidden_dim: int | Sequence[int] | None = None,
    param_source: ParamSource | None = None,
    param_source_option: str | None = None,
) -> ParamSource:
    """Create the parameter source a conditional family reads from.

    This is the one seam every family builds its source through, so
    each of them takes the same three ways of choosing it:

    * ``param_source``: an instance, which wins outright.
    * ``param_source_option``: the DSL's ``[param_source=...]`` text,
      a kind with optional arguments (``"mlp"``, ``"mlp(64, 32)"``).
    * ``hidden_dim``: the hidden widths, as one width or a sequence of
      them. A sequence gives one hidden layer per entry.

    Without any of them the default is linear for a continuous domain
    and a lookup table for a discrete one.

    A width that the chosen source has no layers to apply it to raises
    rather than being dropped: asking a single matrix how wide its
    hidden layers should be has no answer, and silence would read as
    one.
    """
    if param_source is not None:
        return param_source
    kind, kwargs = _split_source_option(param_source_option)
    if hidden_dim is not None:
        if kind not in _HIDDEN_LAYER_KINDS:
            raise ValueError(
                f"hidden_dim is not read by the {kind!r} parameter source, "
                f"which has no hidden layers; select one that does with "
                f"param_source, or drop the width"
            )
        if "hidden_dims" not in kwargs:
            widths = (hidden_dim,) if isinstance(hidden_dim, int) else tuple(hidden_dim)
            kwargs["hidden_dims"] = tuple(int(w) for w in widths)
    return make_param_source(domain, param_dim, kind=kind, **kwargs)


def param_source_from_option(
    domain: AnySpace,
    param_dim: int,
    option_value: str | None,
) -> ParamSource:
    """Parse a `[param_source=...]` option string into a
    `ParamSource`. Accepts:

    * ``"mlp"``, ``"linear"``, ``"identity"``, ``"attention"``
    * ``"mlp(64, 64)"`` — parenthesised hidden widths
    * ``"mlp(32)"`` — single hidden width
    * ``"attention(heads=4)"`` — keyword-only options

    Unrecognised syntax raises `ValueError` so parse errors surface
    at compile time rather than as silent identity fallthrough.
    """
    kind, kwargs = _split_source_option(option_value)
    return make_param_source(domain, param_dim, kind=kind, **kwargs)


type FunctionActivation = Literal["tanh", "relu", "sigmoid", "softplus"]
"""The nonlinearity between the hidden layers of a learned function's network."""

#: The module each hidden-layer activation name builds.
FUNCTION_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}

#: The source kinds a learned function may use.
FUNCTION_SOURCE_KINDS: frozenset[str] = frozenset({"linear", "mlp", "identity"})


def function_source(
    domain: AnySpace,
    param_dim: int,
    option_value: str | None,
    *,
    activation: FunctionActivation | None = None,
) -> ParamSource:
    """Build the network of a learned function from its options.

    Parameters
    ----------
    domain : AnySpace
        The function's input space.
    param_dim : int
        The width of its output.
    option_value : str | None
        The ``[param_source=...]`` text: ``"linear"`` (the default when
        ``None``), ``"mlp"``, ``"mlp(64, 32)"``, or ``"identity"``.
    activation : FunctionActivation | None
        The nonlinearity between hidden layers, read only by ``mlp``;
        ``tanh`` when ``None``.

    Returns
    -------
    ParamSource
        A `LinearSource`, `MLPSource`, or `IdentitySource`.

    Raises
    ------
    ValueError
        If the kind is not a learned-function source, an activation is
        given to a source without hidden layers, the activation is
        unknown, or an identity source changes the width.
    """
    kind, kwargs = _split_source_option(option_value)
    if kind not in FUNCTION_SOURCE_KINDS:
        raise ValueError(
            f"a learned function's param_source is one of "
            f"{sorted(FUNCTION_SOURCE_KINDS)}, not {kind!r}"
        )
    if activation is not None and kind not in _HIDDEN_LAYER_KINDS:
        raise ValueError(
            f"activation is read by the hidden layers of an mlp source; the "
            f"{kind!r} source has none, so select param_source=mlp or drop "
            f"the activation"
        )
    if activation is not None and activation not in FUNCTION_ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; expected one of "
            f"{sorted(FUNCTION_ACTIVATIONS)}"
        )
    if not isinstance(domain, ContinuousSpace):
        raise ValueError(
            f"a learned function maps real coordinates, but its domain "
            f"{domain!s} is not a continuous space"
        )
    if kind == "identity" and int(domain.dim) != int(param_dim):
        raise ValueError(
            f"an identity source passes its {domain.dim} input coordinates "
            f"through unchanged, so it cannot write {param_dim}; select a "
            f"learned source (linear or mlp)"
        )
    if kind == "mlp":
        kwargs["activation"] = FUNCTION_ACTIVATIONS[activation or "tanh"]
    return make_param_source(domain, param_dim, kind=kind, **kwargs)


__all__ = [
    "AttentionSource",
    "FunctionActivation",
    "function_source",
    "ComposeSource",
    "EmbeddingSource",
    "FunctionSource",
    "IdentitySource",
    "LinearSource",
    "LookupSource",
    "MLPSource",
    "ParamSource",
    "make_param_source",
    "param_source_from_option",
]
