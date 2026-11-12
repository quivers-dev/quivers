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

import math
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


def attention_draw_message(name: str) -> str:
    """The refusal of a program step that draws through an attention source.

    An `AttentionSource` attends over the sequence axis of its input and
    pools it away, so it reads one parameter vector off a whole sequence
    of rows. A program step hands a kernel one row per draw, a sequence
    of length one, over which attention computes no weighting.

    Parameters
    ----------
    name : str
        The name the step draws through.

    Returns
    -------
    str
        The message every path raises.
    """
    return (
        f"the step draws from {name!r}, whose parameters come from a "
        f"`param_source=attention` network; that network pools a sequence of "
        f"rows into one parameter vector, and a program step hands a kernel "
        f"one row per draw, so no draw reads it. Attend over a sequence with "
        f"a `[role=function, param_source=self_attention]` morphism bound by "
        f"`let`, and draw from a family around its value"
    )


class SelfAttentionSource(ParamSource):
    """Multi-head scaled dot-product self-attention over a sequence.

    The source maps a sequence of ``T`` rows to a sequence of ``T`` rows,
    keeping the position axis, which is the one before the coordinate
    axis. Each row is projected to a query, a key, and a value of the
    model width ``m``; each of ``heads`` heads reads a block of
    ``e = m / heads`` coordinates of them and weighs the value at every
    position ``j`` by

    .. math::

        a_{ij} = \\frac{\\exp(q_i \\cdot k_j / \\sqrt{e})}
                      {\\sum_{j'} \\exp(q_i \\cdot k_{j'} / \\sqrt{e})},

    with the sum over ``j' <= i`` only, and ``a_{ij} = 0`` for ``j > i``,
    when the attention is causal. The heads' weighted values, laid end
    to end, pass through an output projection.

    Parameters
    ----------
    domain_dim : int
        The width ``d`` of an input row.
    model_dim : int
        The model width ``m``: of the queries, keys, values, and output.
    heads : int
        The number of heads; it divides ``model_dim``.
    causal : bool
        Whether position ``i`` attends only to positions ``j <= i``.

    Raises
    ------
    ValueError
        If a width or the head count is not positive, or the head count
        does not divide the model width.
    """

    def __init__(
        self, domain_dim: int, model_dim: int, heads: int, causal: bool
    ) -> None:
        super().__init__()
        if domain_dim < 1 or model_dim < 1 or heads < 1:
            raise ValueError(
                f"self-attention takes positive widths and a positive head "
                f"count, got domain {domain_dim}, model {model_dim}, heads {heads}"
            )
        if model_dim % heads != 0:
            raise ValueError(
                f"self-attention splits its model width {model_dim} into heads "
                f"of equal width, which {heads} heads do not divide"
            )
        self._domain_dim = int(domain_dim)
        self._param_dim = int(model_dim)
        self._heads = int(heads)
        self._causal = bool(causal)
        self.query = nn.Linear(domain_dim, model_dim)
        self.key = nn.Linear(domain_dim, model_dim)
        self.value = nn.Linear(domain_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)

    @property
    def param_dim(self) -> int:
        """The model width, the width of an output row.

        Returns
        -------
        int
            The width.
        """
        return self._param_dim

    @property
    def heads(self) -> int:
        """The number of attention heads.

        Returns
        -------
        int
            The count.
        """
        return self._heads

    @property
    def causal(self) -> bool:
        """Whether each position attends only to itself and those before it.

        Returns
        -------
        bool
            The flag.
        """
        return self._causal

    def forward(self, x: Tensor) -> Tensor:
        """Attend over the position axis of a sequence.

        Parameters
        ----------
        x : Tensor
            Shape ``(..., T, domain_dim)``.

        Returns
        -------
        Tensor
            Shape ``(..., T, model_dim)``.

        Raises
        ------
        ValueError
            If ``x`` has no position axis or its rows are not
            ``domain_dim`` wide.
        """
        if x.dim() < 2 or int(x.shape[-1]) != self._domain_dim:
            raise ValueError(
                f"self-attention reads a sequence of {self._domain_dim}-wide "
                f"rows, shape (..., T, {self._domain_dim}), got {tuple(x.shape)}"
            )
        width = self._param_dim // self._heads

        def split(projected: Tensor) -> Tensor:
            return projected.unflatten(-1, (self._heads, width)).transpose(-3, -2)

        query = split(self.query(x))
        key = split(self.key(x))
        value = split(self.value(x))
        scores = query @ key.transpose(-1, -2) / math.sqrt(width)
        if self._causal:
            length = int(x.shape[-2])
            later = torch.ones(length, length, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(later.triu(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        context = (weights @ value).transpose(-3, -2).flatten(-2)
        return self.output(context)


class PositionSource(ParamSource):
    """A learned table of position vectors added to a sequence's rows.

    Row ``t`` of a sequence gains row ``t`` of the table, so a sequence
    of ``T`` positions reads the table's first ``T`` rows. The table's
    entries start as draws from a Normal of standard deviation 0.02.

    Parameters
    ----------
    dim : int
        The width of a row.
    max_length : int
        The number of positions the table holds, the longest sequence
        the source reads.

    Raises
    ------
    ValueError
        If a size is not positive.
    """

    def __init__(self, dim: int, max_length: int) -> None:
        super().__init__()
        if dim < 1 or max_length < 1:
            raise ValueError(
                f"a position table takes a positive width and length, got "
                f"width {dim} and length {max_length}"
            )
        self._param_dim = int(dim)
        self._max_length = int(max_length)
        self.table = nn.Parameter(torch.randn(max_length, dim) * 0.02)

    @property
    def param_dim(self) -> int:
        """The width of a row.

        Returns
        -------
        int
            The width.
        """
        return self._param_dim

    @property
    def max_length(self) -> int:
        """The number of positions the table holds.

        Returns
        -------
        int
            The count.
        """
        return self._max_length

    def forward(self, x: Tensor) -> Tensor:
        """Add each position's vector to the row at that position.

        Parameters
        ----------
        x : Tensor
            Shape ``(..., T, dim)``.

        Returns
        -------
        Tensor
            ``x`` plus the table's first ``T`` rows.

        Raises
        ------
        ValueError
            If ``x`` has no position axis, its rows are not ``dim`` wide,
            or it is longer than the table.
        """
        if x.dim() < 2 or int(x.shape[-1]) != self._param_dim:
            raise ValueError(
                f"a position table reads a sequence of {self._param_dim}-wide "
                f"rows, shape (..., T, {self._param_dim}), got {tuple(x.shape)}"
            )
        length = int(x.shape[-2])
        if length > self._max_length:
            raise ValueError(
                f"a sequence of {length} positions is longer than the "
                f"{self._max_length} positions the table holds"
            )
        return x + self.table[:length].to(x.dtype)


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
FUNCTION_SOURCE_KINDS: frozenset[str] = frozenset(
    {"linear", "mlp", "identity", "self_attention", "positions"}
)

#: The learned-function sources that map a whole sequence of rows to a
#: sequence of rows, rather than one row to one row.
SEQUENCE_SOURCE_KINDS: frozenset[str] = frozenset({"self_attention", "positions"})


def function_source(
    domain: AnySpace,
    param_dim: int,
    option_value: str | None,
    *,
    activation: FunctionActivation | None = None,
    heads: int | None = None,
    causal: bool | None = None,
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
        ``None``), ``"mlp"``, ``"mlp(64, 32)"``, ``"identity"``,
        ``"self_attention"``, or ``"positions(<max_length>)"``.
    activation : FunctionActivation | None
        The nonlinearity between hidden layers, read only by ``mlp``;
        ``tanh`` when ``None``.
    heads : int | None
        The number of attention heads, read only by ``self_attention``;
        one when ``None``.
    causal : bool | None
        Whether self-attention is causally masked, which
        ``self_attention`` requires and no other source reads.

    Returns
    -------
    ParamSource
        A `LinearSource`, `MLPSource`, `IdentitySource`,
        `SelfAttentionSource`, or `PositionSource`.

    Raises
    ------
    ValueError
        If the kind is not a learned-function source, an option is given
        to a source that does not read it, the activation is unknown, an
        identity or position source changes the width, self-attention
        is not told whether it is causal, or a position source does not
        name its length.
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
    for option, given in (("heads", heads), ("causal", causal)):
        if given is not None and kind != "self_attention":
            raise ValueError(
                f"{option} is read by a self_attention source; the {kind!r} "
                f"source does not attend, so drop {option}"
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
    if kind == "self_attention":
        if kwargs:
            raise ValueError(
                "self_attention takes no arguments; give the head count as "
                "the heads option, as in heads=4"
            )
        if causal is None:
            raise ValueError(
                "self_attention is causal or not by declaration: add "
                "causal=true for each position to attend to itself and the "
                "positions before it, or causal=false for every position"
            )
        return SelfAttentionSource(
            int(domain.dim), int(param_dim), 1 if heads is None else int(heads), causal
        )
    if kind == "positions":
        length = kwargs.get("hidden_dims")
        if (
            set(kwargs) != {"hidden_dims"}
            or not isinstance(length, tuple)
            or (len(length) != 1)
        ):
            raise ValueError(
                "positions takes the number of positions its table holds, the "
                "longest sequence it reads, as in positions(64)"
            )
        if int(domain.dim) != int(param_dim):
            raise ValueError(
                f"a position table adds to the {domain.dim} coordinates of each "
                f"row, so it cannot write {param_dim}"
            )
        return PositionSource(int(domain.dim), int(length[0]))
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
    "PositionSource",
    "SelfAttentionSource",
    "make_param_source",
    "param_source_from_option",
]
