"""Hierarchical Bayesian modelling primitives.

Adds the operations needed to express models of the kind found in
hierarchical Bayesian regression / latent-class analysis (the
canonical Stan workhorse): finite-domain-indexed draws (plates),
vectorized observations with gather indexing, LKJ priors on
correlation matrices, ordinal monotone splines via cumulative
sum, and generic distribution truncation. Each primitive is
declared with its categorical denotation in :math:`\\mathbf{Kern}`;
the runtime is a straight realisation of those denotations.

Categorical foundations
-----------------------
* **Plate (indexed draw)**. Given a finite index set :math:`A` and a
  parameterized family :math:`F : \\Theta \\to \\mathcal{G}(B)`, a
  *plate draw* declares the kernel

  .. math::

      v : A \\to B,\\quad v(a) \\sim F(\\theta(a))

  equivalently a single morphism into the function space
  :math:`\\mathbf{1} \\to \\mathcal{G}(B^A)` factoring as the
  independent product :math:`\\prod_{a \\in A} F(\\theta(a))`. Under
  the natural isomorphism
  :math:`\\mathbf{Kern}(\\mathbf{1}, B^A) \\cong \\mathbf{Kern}(A, B)`
  the plate variable IS a :math:`\\mathbf{Kern}`-morphism
  :math:`A \\to B`; in the variational backend it is held as a
  tensor of shape ``(|A|, *B.shape)`` whose prior contribution to
  the ELBO is the per-row log-density.

* **Gather**. Given a finite index map :math:`\\iota : N \\to A` (a
  per-observation grouping assignment) and a plate variable
  :math:`v : A \\to B`, the gathered morphism is the pullback
  :math:`\\iota^* v = v \\circ \\iota : N \\to B`. Categorically a
  textbook reindexing morphism in :math:`\\mathbf{Kern}`; runtime is
  ``v[indices]`` along the leading axis.

* **Vectorised observe**. A batched observation step

  .. math::

      \\mathcal{S}\\llbracket\\,\\mathsf{observe}\\ r[n] \\sim
      F(\\theta[n])\\ \\mathsf{for}\\ n\\in N\\,\\rrbracket :
      \\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)

  has score :math:`\\prod_{n \\in N} p_F(r_{\\text{obs}}(n);\\,
  \\theta(n,\\phi))`. The single-observation form is the
  ``|N| = 1`` special case; mass-correctness follows from the
  Cho-Jacobs Markov-with-conditioning calculus.

* **Marginalise (program-level)**. Given a previously-drawn
  discrete latent :math:`c : \\Phi \\to \\mathcal{G}(C)`, the
  marginalisation step is the pushforward through the projection
  :math:`\\pi_{\\Phi \\setminus C} : \\Phi \\times C \\to \\Phi`:

  .. math::

      \\mathsf{marg}(c) = \\mathcal{G}(\\pi_{\\Phi \\setminus C})
                         \\circ \\mathcal{S}\\llbracket\\mathsf{draw}\\ c\\rrbracket

  numerically realised as :math:`\\log \\sum_c \\exp(\\cdot)` over
  the :math:`C` axis of the per-class log-likelihoods.

* **cumsum**. The deterministic morphism
  :math:`\\mathrm{cumsum} : \\mathrm{Euclidean}(K) \\to \\mathrm{Euclidean}(K)`,
  :math:`(x_1, \\dots, x_K) \\mapsto (x_1, x_1 + x_2, \\dots,
  \\sum_i x_i)`. Used for the standard monotonic-spline
  parameterization of ordinal coefficients.

* **LKJ correlation prior**. The Lewandowski-Kurowicka-Joe
  distribution :math:`\\mathrm{LKJ}(K, \\eta)` on the manifold
  :math:`\\mathrm{Corr}_K` of :math:`K \\times K` correlation
  matrices, parameterized via the Cholesky factor for numerical
  stability. The accompanying ``cholesky_quad_form`` deterministic
  morphism reconstructs a full covariance
  :math:`\\Sigma = \\mathrm{diag}(s)\\, R\\, \\mathrm{diag}(s)`
  from a correlation :math:`R` and a per-component scale
  :math:`s \\in \\mathrm{PositiveReals}(K)`.

* **Truncated**. Given a base family :math:`F` and an interval
  :math:`[a, b]`, the truncated family
  :math:`F_{|[a, b]}` has density
  :math:`p_F(x) / (F_{\\text{cdf}}(b) - F_{\\text{cdf}}(a))` on
  :math:`[a, b]` and zero elsewhere. Realised via inverse-CDF
  sampling when available; rejection otherwise.

References
----------
- Cho, K. and Jacobs, B. (2019). *Disintegration and Bayesian
  inversion via string diagrams*. Mathematical Structures in
  Computer Science 29(7), 938–971. doi:10.1017/S0960129518000488.
- Lewandowski, D., Kurowicka, D. and Joe, H. (2009). *Generating
  random correlation matrices based on vines and extended onion
  method*. Journal of Multivariate Analysis 100(9), 1989–2001.
  doi:10.1016/j.jmva.2009.04.008.
"""

from __future__ import annotations

import math
from typing import cast

import torch


from quivers.core.objects import EnumSet, FinSet
from quivers.continuous.morphisms import ContinuousMorphism, AnySpace
from quivers.continuous.param_source import LookupSource
from quivers.continuous.spaces import (
    CholeskyFactor,
    Correlation,
    Covariance,
    Euclidean,
    LowerTriangular,
    Orthogonal,
    Stiefel,
)


# ---------------------------------------------------------------------------
# Plate / vectorized observe / program-level marginalise Python builders
# ---------------------------------------------------------------------------


_MATRIX_SPACES = (
    CholeskyFactor,
    Correlation,
    Covariance,
    LowerTriangular,
    Orthogonal,
    Stiefel,
)
"""Spaces whose ``dim`` is a matrix size and whose ``shape`` is flat."""

_CLASS_SPACES = (FinSet, EnumSet)
"""Finite sets a discrete family draws one class index of per row.

Their ``shape`` is the one-hot width of an element, not the extent of a
draw: a row of a plated draw over such a set holds a single index.
"""


class PlateDraw(ContinuousMorphism):
    """A finite-domain-indexed draw, as a Kern-morphism ``A → B``.

    Concretely: ``v : A → B ~ F(theta)`` becomes a tensor of shape
    ``(|A|, *B.shape)`` whose ``a``-th row is an independent
    :math:`F(\\theta_a)`-distributed random variable. The variational
    posterior factorizes across rows by default; the prior's ELBO
    contribution is :math:`\\sum_a \\log p_F(v_a; \\theta_a)`.

    Categorically: by the natural isomorphism
    :math:`\\mathbf{Kern}(\\mathbf{1}, B^A) \\cong \\mathbf{Kern}(A, B)`,
    the plate variable IS a Kern-morphism :math:`A \\to B`. The
    PlateDraw is realised as a `ContinuousMorphism` whose
    codomain is the flat product-space of ``index_size`` copies of
    the per-row family's codomain.

    The plate may range over several finite axes at once,
    :math:`A_1 \\times \\dots \\times A_k`: the draw is then an
    independent array of shape ``(|A_1|, ..., |A_k|, *B.shape)``, one
    independent ``F`` variate per element of the product.

    Parameters
    ----------
    index_size : int
        Cardinality :math:`|A|`, the number of independent rows: the
        product of ``index_shape`` when that is given.
    family : ContinuousMorphism
        Per-row distribution family.
    domain : AnySpace
        The program's input space (broadcast conditioning).
    index_shape : tuple[int, ...] | None
        The extents of the plate's axes, outermost first, when the
        plate ranges over more than one axis; ``None`` for the single
        axis ``(index_size,)``.

    Raises
    ------
    ValueError
        If ``index_shape`` does not multiply out to ``index_size``.
    """

    def __init__(
        self,
        index_size: int,
        family: ContinuousMorphism,
        domain: AnySpace | None = None,
        *,
        index_shape: tuple[int, ...] | None = None,
    ) -> None:
        shape = (index_size,) if index_shape is None else tuple(index_shape)
        if not shape or math.prod(shape) != index_size:
            raise ValueError(
                f"PlateDraw: index_shape {shape} does not multiply out to "
                f"index_size {index_size}"
            )
        # A matrix-valued space (a Cholesky factor, a covariance) keeps
        # `dim` as its matrix size and states its flat element count as
        # `shape`; other continuous spaces state only `dim`.
        per_row_shape: tuple[int, ...]
        if isinstance(family.codomain, _CLASS_SPACES):
            per_row_shape = ()
            per_row_dim = 1
        elif isinstance(family.codomain, _MATRIX_SPACES):
            per_row_shape = family.codomain.shape
            per_row_dim = math.prod(per_row_shape)
        elif hasattr(family.codomain, "dim"):
            per_row_dim = int(family.codomain.dim)
            per_row_shape = (per_row_dim,)
        else:
            per_row_shape = tuple(family.codomain.shape)
            per_row_dim = (
                int(torch.tensor(per_row_shape).prod().item()) if per_row_shape else 1
            )
        # `plate_rows` keeps the row structure of the flattened
        # codomain recoverable: a consumer that reads the variable as
        # a *parameter* needs the per-row width, since the plate axis
        # is a batch axis in that position rather than an event axis.
        # The rows a gather along the outermost axis selects are the
        # slabs over the remaining axes, so that axis is the codomain's
        # row count.
        flat_codomain = Euclidean(
            name=f"plate({'x'.join(map(str, shape))}x{family.codomain!s})",
            dim=index_size * per_row_dim,
            plate_rows=shape[0],
        )
        actual_domain = domain if domain is not None else family.domain
        super().__init__(actual_domain, flat_codomain)
        self._index_size = index_size
        self._index_shape = shape
        self._family = family
        self._per_row_shape = per_row_shape
        self._per_row_dim = per_row_dim
        # Width of the conditioning input a single row of the family
        # consumes. Fixed families ignore it (they read only the row
        # count); conditional families feed it through their parameter
        # network, so it must equal the domain's feature width.
        dom = actual_domain
        if hasattr(dom, "dim"):
            self._domain_width = max(1, int(dom.dim))
        elif hasattr(dom, "shape"):
            shape = tuple(dom.shape)
            self._domain_width = int(torch.tensor(shape).prod().item()) if shape else 1
        else:
            self._domain_width = 1

    @property
    def index_size(self) -> int:
        return self._index_size

    @property
    def index_shape(self) -> tuple[int, ...]:
        """The extents of the plate's axes, outermost first.

        Returns
        -------
        tuple[int, ...]
            The extents.
        """
        return self._index_shape

    @property
    def family(self) -> ContinuousMorphism:
        return self._family

    def _row_conditioning(
        self,
        x: torch.Tensor | None,
        n_rows: int,
        device: torch.device | None,
    ) -> torch.Tensor:
        """Conditioning input each of ``n_rows`` per-row families reads.

        A fixed (all-literal) family ignores the conditioning and needs
        only the row count, so a value whose width does not match the
        family's domain falls back to a zero input of the expected
        shape. A conditional family reads its parameters from the
        resolved input ``x`` (the scale of ``by_subj <- Normal(0, sigma)``
        is the latent ``sigma``), so ``x`` is threaded through, broadcast
        from a single shared row to every row when needed rather than
        discarded. A family whose parameters are a table indexed by the
        elements of its finite domain is drawn one row per element by
        a plate over that domain, so each row is conditioned on its own
        index.
        """
        width = self._domain_width
        if isinstance(getattr(self._family, "param_source", None), LookupSource):
            rows = torch.arange(n_rows, device=device) % self._index_size
            return rows.reshape(n_rows, 1)
        if x is None or x.numel() == 0:
            return torch.zeros(n_rows, width, device=device)
        xf = x if x.dim() == 2 else x.reshape(1, -1)
        if xf.shape[-1] != width:
            if xf.numel() == width:
                xf = xf.reshape(1, width)
            elif xf.numel() == n_rows * width:
                xf = xf.reshape(n_rows, width)
            else:
                return torch.zeros(n_rows, width, device=device)
        if xf.shape[0] == n_rows:
            return xf
        if xf.shape[0] == 1:
            return xf.expand(n_rows, width)
        return torch.zeros(n_rows, width, device=device)

    def rsample(
        self,
        x: torch.Tensor | None = None,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        """Reparameterized sample from the plate's prior.

        A plate draw is *batch-invariant*: the latent vector is a
        global model quantity shared across every row of an observed
        plate. The returned tensor has shape ``(|A|, *B.shape)``
        regardless of the program input's leading batch dimension,
        with each row drawn independently from the per-row family
        ``F`` at its declared parameters. ``x`` supplies only the
        device; it is optional so batch-invariant callers such as
        `gather` need not fabricate one.

        This is the standard Pyro / NumPyro semantic: a sample inside
        a ``plate("subj", n_subj)`` context is one ``(n_subj,)``
        vector, *not* a per-particle (batch, n_subj) replication. The
        gather ``arr[idx]`` along the plate axis then composes
        cleanly with per-row observed-plate axes downstream.
        """
        del sample_shape  # plate latents are batch-invariant
        device = x.device if x is not None else None
        # Draw ``|A|`` independent rows by sampling the per-row family
        # with a leading sample axis of length ``|A|`` from a single
        # conditioning row. Threading the row count through
        # ``sample_shape`` rather than the input's batch axis keeps the
        # draw genuinely independent even for families whose fixed
        # parameterization advertises a scalar batch shape (a
        # positive-support family realised as a restricted base
        # measure, say); those ignore the input's leading axis but
        # still honour the explicit sample shape.
        row_input = self._row_conditioning(x, 1, device)
        base = self._family.rsample(
            row_input, sample_shape=torch.Size((self._index_size,))
        )
        expected = self._index_size * self._per_row_dim
        if base.numel() != expected:
            raise ValueError(
                f"plate rsample: per-row family produced {base.numel()} "
                f"elements for {self._index_size} rows of per-row width "
                f"{self._per_row_dim}; expected {expected}"
            )
        return self._to_structured(base)

    def _to_structured(self, value: torch.Tensor) -> torch.Tensor:
        """Reshape a plate value to its structured ``(*index_shape, *B.shape)``.

        The per-row event axis ``B`` is carried explicitly so a
        downstream gather ``v[idx]`` along the plate axis preserves the
        event coordinates. Scalar-per-row plates
        (``per_row_shape == (1,)``) drop the trailing length-1 axis so
        the latent has the natural ``(|A|,)`` shape and ``v[idx]``
        advance-indexing produces ``(N,)`` without a manual squeeze.
        """
        sample = value.reshape(*self._index_shape, *self._per_row_shape)
        if len(self._per_row_shape) == 1 and self._per_row_shape[0] == 1:
            sample = sample.squeeze(-1)
        return sample

    def canonical_latent(self, value: torch.Tensor) -> torch.Tensor:
        """Give a clamped plate value its structured ``(|A|, *B.shape)``.

        A plate latent supplied as a conditioning / observed value may
        arrive flattened to ``(|A| * prod(B.shape),)`` (the shape a
        test point or host-data channel produces when it flattens a
        matrix-valued latent). Restoring the structured shape here lets
        a gather ``v[idx]`` over the plate axis yield ``(N, *B.shape)``
        with the per-row event axis intact.

        A value whose element count is not exactly ``|A| * prod(B)`` is
        a genuinely batched latent (a leading parameter-sample axis);
        it is returned unchanged so the batched scoring path handles
        it.
        """
        if value.numel() != self._index_size * self._per_row_dim:
            return value
        return self._to_structured(value)

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Prior log-density of the plate sample.

        The plate variable ``v : A -> B`` scores as the independent
        product ``prod_a p_F(v_a)`` of the per-row family ``F``; the
        contribution is ``sum_a log p_F(v_a)``.

        Two layouts are accepted. The natural *plate latent* carries the
        plate's axes leading (or holds exactly ``|A| * prod(event)``
        elements in any layout, a flat vector included): its rows are scored
        under ``F`` and reduced to a scalar, wrapped in a length-1
        tensor so the downstream log-joint accumulator sees a uniform
        1-d shape it can broadcast against the response plate. A
        *batched* value carries a genuine leading batch axis distinct
        from ``|A|``; each batch element is scored independently and a
        ``(batch,)`` vector of per-element densities is returned.

        The per-row event size is read from the value's element count
        rather than solely from the family's advertised codomain, so a
        scalar family carrying an explicit event axis (``over=...``)
        is scored over the full event without loss.
        """
        index_size = self._index_size
        per_row_family = self._per_row_dim
        axes = len(self._index_shape)
        structured = tuple(y.shape[:axes]) == self._index_shape
        batched = (
            y.dim() >= 2 and not structured and y.numel() != index_size * per_row_family
        )
        if not batched:
            total = y.numel()
            if total % index_size != 0:
                raise ValueError(
                    f"plate latent has {total} elements, not divisible "
                    f"by the plate index size {index_size}"
                )
            row_numel = total // index_size
            event_shape = (
                self._per_row_shape if row_numel == per_row_family else (row_numel,)
            )
            sample = y.reshape(index_size, *event_shape)
            row_input = self._row_conditioning(x, index_size, y.device)
            per_row_lp = self._family.log_prob(row_input, sample)
            return per_row_lp.reshape(-1).sum().unsqueeze(0)
        batch = y.shape[0]
        per_batch = y.numel() // batch
        if per_batch % index_size != 0:
            raise ValueError(
                f"batched plate latent has {per_batch} elements per "
                f"batch row, not divisible by the plate index size "
                f"{index_size}"
            )
        row_numel = per_batch // index_size
        event_shape = (
            self._per_row_shape if row_numel == per_row_family else (row_numel,)
        )
        flat = y.reshape(batch * index_size, *event_shape)
        row_input = self._row_conditioning(x, batch * index_size, y.device)
        per_row_lp = self._family.log_prob(row_input, flat)
        return per_row_lp.reshape(batch, index_size).sum(dim=-1)

    def log_prob_rows(
        self, x: torch.Tensor, y: torch.Tensor, *rows: torch.Tensor
    ) -> torch.Tensor:
        """Prior log-density of the plate's elements at a subset of its rows.

        The elements kept are the product of the rows kept along each
        axis; each is scored under its row's family, as `log_prob` scores
        it, and the scores are summed.

        Parameters
        ----------
        x : torch.Tensor
            The conditioning input, as `log_prob` reads it.
        y : torch.Tensor
            The plate's value at every element, with the plate's axes
            leading or flat in their order.
        *rows : torch.Tensor
            One index per plate axis, outermost first: the rows kept
            along that axis.

        Returns
        -------
        torch.Tensor
            The summed log-density of the kept elements, shape ``(1,)``.

        Raises
        ------
        ValueError
            If ``rows`` does not give one index per axis, an index leaves
            its axis, or ``y`` does not hold every element of the plate.
        """
        shape = self._index_shape
        if len(rows) != len(shape):
            raise ValueError(
                f"PlateDraw.log_prob_rows: {len(rows)} row indices for a plate "
                f"of {len(shape)} axes"
            )
        positions = torch.arange(self._index_size, device=y.device).reshape(shape)
        for axis, (index, extent) in enumerate(zip(rows, shape, strict=True)):
            if index.numel() == 0 or int(index.min()) < 0 or int(index.max()) >= extent:
                raise ValueError(
                    f"PlateDraw.log_prob_rows: the rows kept along axis {axis} "
                    f"must be a nonempty index into its {extent} rows"
                )
            positions = positions.index_select(axis, index.to(y.device))
        total = y.numel()
        if total % self._index_size != 0:
            raise ValueError(
                f"PlateDraw.log_prob_rows: the plate value has {total} elements, "
                f"not a whole number of rows for the plate's {self._index_size} "
                "elements; a subsampled plate latent is given at every row"
            )
        row_numel = total // self._index_size
        event_shape = (
            self._per_row_shape if row_numel == self._per_row_dim else (row_numel,)
        )
        kept = positions.reshape(-1)
        sample = y.reshape(self._index_size, *event_shape)[kept]
        row_input = self._row_conditioning(x, self._index_size, y.device)[kept]
        return self._family.log_prob(row_input, sample).reshape(-1).sum().unsqueeze(0)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """Pullback ``v[indices]`` along a finite index map.

        ``indices`` is a long-tensor of shape ``(N,)`` with entries
        in ``[0, |A|)``; returns a tensor of shape
        ``(N, *B.shape)``.
        """
        return self.rsample()[indices]

    def __repr__(self) -> str:
        return f"PlateDraw(index_size={self._index_size}, family={self._family!r})"


class VectorisedObserve(ContinuousMorphism):
    """A batched observation step accumulating per-row log-likelihoods.

    Categorically, the batched-likelihood kernel
    :math:`\\Phi \\to \\mathcal{G}_{\\le 1}(\\Phi)` whose score is
    :math:`\\prod_{n \\in N} p_F(r_{\\text{obs}}(n);\\, \\theta(n,\\phi))`.
    Realised as a `ContinuousMorphism` whose domain is the
    parameter-input space (the morphism conditions on θ) and whose
    codomain is the per-observation response space — so the
    existing `MonadicProgram` observe-step machinery treats
    it as an observed site and threads the score through
    ``log_joint`` via the usual ``morph.log_prob(theta, response)``
    call, with ``log_prob`` here summing over the leading index axis.

    The observed response tensor is registered as a buffer so the
    parent program's optimizer tracks it and the runtime never
    has to thread it through ``observations=...``.

    Parameters
    ----------
    family : ContinuousMorphism
        The per-observation distribution family.
    response : torch.Tensor
        Observed values ``r_obs`` of shape ``(N, *codom.shape)``. A
        one-dimensional continuous codomain also accepts ``(N,)``,
        which gains its event axis in `_align_response`; a discrete
        codomain takes ``(N,)`` indices and has no event axis.
    """

    def __init__(
        self,
        family: ContinuousMorphism,
        response: torch.Tensor,
        *,
        array_shape: tuple[int, ...] = (),
    ) -> None:
        super().__init__(family.domain, family.codomain)
        self._family = family
        self._array_shape = tuple(array_shape)
        self.register_buffer("_response", response.detach())

    @property
    def array_shape(self) -> tuple[int, ...]:
        """The extents of the independent array each observation row holds.

        Empty for one variate per row; ``(|B|, |C|)`` for an observation
        drawn over leading ``over`` axes beyond its family's event rank.

        Returns
        -------
        tuple[int, ...]
            The extents.
        """
        return self._array_shape

    def _array_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Lay each row's parameters out for every entry of its array.

        A parameter one value per row is repeated for every entry of the
        row's array; a parameter holding one value per entry, a slice
        of a table over the array's axes, gives each entry its own.

        Parameters
        ----------
        x : torch.Tensor
            The parameters, one row per observation row.

        Returns
        -------
        torch.Tensor
            The parameters, one row per entry of every row's array.
        """
        entries = math.prod(self._array_shape)
        spec = getattr(self._family, "_param_spec", None)
        if spec is not None and x.dim() == 2:
            widths = [width for kind, width in spec if kind == "var"]
            if (
                widths
                and entries > 1
                and all(width is None or width == 1 for width in widths)
                and int(x.shape[-1]) == len(widths) * entries
            ):
                # Every parameter holds one value per entry of the row's
                # array, laid out parameter by parameter: each entry
                # reads its own value of each.
                return (
                    x.reshape(x.shape[0], len(widths), entries)
                    .transpose(1, 2)
                    .reshape(x.shape[0] * entries, len(widths))
                )
        return x.repeat_interleave(entries, dim=0)

    @property
    def response(self) -> torch.Tensor:
        return cast("torch.Tensor", self._response)

    @property
    def support(self):
        """Inherit the underlying per-row family's support so guides
        and validators see the correct constraint for the observation."""
        return getattr(self._family, "support", super().support)

    @property
    def _param_event_ranks(self) -> tuple[int, ...] | None:
        """Forward the per-parameter event ranks from the wrapped
        family so `_resolve_input` can broadcast shared vector-typed
        parameters (e.g. `OrderedLogistic` cutpoints) before stacking.
        """
        return getattr(self._family, "_param_event_ranks", None)

    @property
    def _param_spec(self) -> list[tuple[str, int | float | None]] | None:
        """Forward the per-parameter spec (kinds and dims) from the
        wrapped family so `_resolve_input` can stack multiple vector-
        typed parameters (e.g. `MixtureNormal` weights / locs / scales)
        by their declared dims rather than by tensor rank.
        """
        return getattr(self._family, "_param_spec", None)

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample the per-observation family at the supplied θ.

        ``x`` is the θ-tensor (one row per observation index); the
        result is the per-observation response sample. Used in
        prior-predictive simulation; never called during inference
        when the response is observed.

        A θ shared by every observation (a single row: literal or
        program-wide parameters) is drawn once per row of the plate,
        so the draw has the plate's rows whatever mix of shared and
        per-row parameters the observation reads.
        """
        rows = self.response.shape[0]
        if x.dim() >= 1 and x.shape[0] == 1 and rows > 1:
            x = x.expand(rows, *x.shape[1:])
        if not self._array_shape:
            return self._family.rsample(x, sample_shape)
        draws = self._family.rsample(self._array_rows(x), sample_shape)
        lead = tuple(sample_shape)
        event = tuple(draws.shape[len(lead) + 1 :])
        if event == (1,):
            event = ()
        return draws.reshape(*lead, x.shape[0], *self._array_shape, *event)

    def _align_response(self, target: torch.Tensor) -> torch.Tensor:
        """Give the response its codomain's event axis before scoring.

        A conditional family reads its parameters out of a network, so
        for a codomain of dimension ``d`` they arrive as ``(N, d)``. The
        natural way to write N scalar responses against a ``d = 1``
        codomain is ``(N,)``, and subtracting that from an ``(N, 1)``
        mean broadcasts to ``(N, N)``. Because the score is then summed,
        the result is finite but incorrect, so the axis
        is restored here.

        An inline family is the other layout and is left alone: its
        parameters come from the program's environment as per-row
        scalars already aligned with the plate, so the response's last
        axis indexes the plate rather than an event. ``_param_spec``
        marks that case.
        """
        if getattr(self._family, "_param_spec", None) is not None:
            return target
        codomain = self._family.codomain
        dim = getattr(codomain, "dim", None)
        if dim is None:
            # Discrete codomain: one index per row, no event axis.
            return target
        dim = int(dim)
        if target.dim() == 1:
            if dim != 1:
                raise ValueError(
                    f"observed response for codomain {codomain!r} has shape "
                    f"{tuple(target.shape)}; expected (N, {dim}) for a "
                    f"{dim}-dimensional codomain"
                )
            return target.unsqueeze(-1)
        if target.shape[-1] != dim:
            raise ValueError(
                f"observed response for codomain {codomain!r} has shape "
                f"{tuple(target.shape)}; its last axis must be the "
                f"codomain's event axis of size {dim}"
            )
        return target

    def log_prob(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """Sum of per-observation log-densities.

        ``y`` defaults to the registered response buffer; passing a
        different value (e.g. a clamped observation) is supported
        for fast prior-predictive checks.
        """
        target = y if y is not None else cast("torch.Tensor", self._response)
        if self._array_shape:
            # Each entry of a row's array is one observation of the
            # per-row family at that row's parameters.
            entries = math.prod(self._array_shape)
            flat = target.reshape(target.shape[0] * entries, -1).squeeze(-1)
            rows = (
                x
                if x.shape[0] == target.shape[0]
                else x.expand(target.shape[0], *x.shape[1:])
            )
            return self._family.log_prob(
                self._array_rows(rows), self._align_response(flat)
            ).sum()
        return self._family.log_prob(x, self._align_response(target)).sum()

    def log_prob_rows(
        self, x: torch.Tensor, y: torch.Tensor, *rows: torch.Tensor
    ) -> torch.Tensor:
        """Sum of the per-observation log-densities at a subset of the rows.

        Parameters
        ----------
        x : torch.Tensor
            The parameters, one row per observation row or one row every
            observation shares, as `log_prob` reads them.
        y : torch.Tensor
            The observed rows: every row of the plate, or only the kept
            rows in the order of the index.
        *rows : torch.Tensor
            One index per plate axis of the observation, outermost first:
            the rows kept along the observation's rows, then every entry
            of each axis of its array.

        Returns
        -------
        torch.Tensor
            The summed log-density of the kept rows.

        Raises
        ------
        ValueError
            If ``rows`` does not give one index per axis, keeps a subset of
            an array axis, or leaves its axis, or if ``x`` or ``y`` has
            neither the plate's rows nor the kept rows.
        """
        if len(rows) != 1 + len(self._array_shape):
            raise ValueError(
                f"VectorisedObserve.log_prob_rows: {len(rows)} row indices for "
                f"an observation over {1 + len(self._array_shape)} axes"
            )
        for axis, (index, extent) in enumerate(
            zip(rows[1:], self._array_shape, strict=True), start=1
        ):
            if index.numel() != extent or not bool(
                (index.cpu() == torch.arange(extent)).all()
            ):
                raise ValueError(
                    f"VectorisedObserve.log_prob_rows: axis {axis} indexes each "
                    "row's array, whose entries are scored together; subsample "
                    "the plate of the observation's rows instead"
                )
        kept = rows[0]
        plate_rows = int(self.response.shape[0])
        if kept.numel() == 0 or int(kept.min()) < 0 or int(kept.max()) >= plate_rows:
            raise ValueError(
                "VectorisedObserve.log_prob_rows: the rows kept must be a "
                f"nonempty index into the observation's {plate_rows} rows"
            )
        count = int(kept.numel())
        target = y if y is not None else self.response
        if int(target.shape[0]) == plate_rows:
            target = target.index_select(0, kept.to(target.device))
        elif int(target.shape[0]) != count:
            raise ValueError(
                f"VectorisedObserve.log_prob_rows: the observation has "
                f"{int(target.shape[0])} rows, neither the plate's {plate_rows} "
                f"nor the {count} kept"
            )
        theta = x
        if x.dim() >= 1 and int(x.shape[0]) == plate_rows and plate_rows != 1:
            theta = x.index_select(0, kept.to(x.device))
        elif x.dim() >= 1 and int(x.shape[0]) not in (1, count):
            raise ValueError(
                f"VectorisedObserve.log_prob_rows: the parameters have "
                f"{int(x.shape[0])} rows, neither the plate's {plate_rows}, "
                f"the {count} kept, nor one shared row"
            )
        return self.log_prob(theta, target)

    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """Alias for ``log_prob(theta)``; preserved for the Python
        builder API."""
        return self.log_prob(theta)

    def __repr__(self) -> str:
        return (
            f"VectorisedObserve(family={self._family!r}, "
            f"N={cast('torch.Tensor', self._response).shape[0]})"
        )


def marginalize_categorical(log_probs_per_class: torch.Tensor) -> torch.Tensor:
    """Program-level marginalisation over a discrete latent class.

    Given per-class log-likelihoods of shape ``(N, K)`` (one row per
    observation, one column per latent-class assignment), returns
    the marginalised log-likelihood of shape ``(N,)``:

    .. math::

        \\log p(r_n) = \\log \\sum_{k=1}^{K} p(c_n = k)\\, p(r_n | c_n = k).

    Realises the program-level ``marginalize c`` step as the
    pushforward through :math:`\\pi_{\\Phi \\setminus C}`.
    """
    return torch.logsumexp(log_probs_per_class, dim=-1)


def _flatten_product_indices(
    group_indices: tuple[torch.Tensor, ...],
    group_sizes: tuple[int, ...],
) -> tuple[torch.Tensor, int]:
    """Compose a tuple of co-indexed index map tensors into a
    single flat group index on the product grouping plate.

    The product :math:`G_1 \\times G_2 \\times \\dots \\times G_r`
    is row-major: ``flat = idx_1 * (|G_2| ... |G_r|) + idx_2 * (|G_3| ...) + ... + idx_r``.

    Returns ``(flat_index, total_groups)`` where ``total_groups =
    \\prod_i |G_i|``.
    """
    if not group_indices:
        raise ValueError("_flatten_product_indices: need at least one index map tensor")
    if len(group_indices) != len(group_sizes):
        raise ValueError(
            "_flatten_product_indices: number of indices "
            f"({len(group_indices)}) must equal number of sizes "
            f"({len(group_sizes)})"
        )
    n_rows = int(group_indices[0].shape[0])
    for j, idx in enumerate(group_indices):
        if idx.shape != (n_rows,):
            raise ValueError(
                f"_flatten_product_indices: index map {j} has shape "
                f"{tuple(idx.shape)}; expected ({n_rows},) to match "
                f"the first index map"
            )
    total = 1
    for size in group_sizes:
        if size <= 0:
            raise ValueError(
                "_flatten_product_indices: all group sizes must be "
                f"positive; got {group_sizes}"
            )
        total *= int(size)
    flat = torch.zeros(n_rows, dtype=torch.long, device=group_indices[0].device)
    running = 1
    for j in range(len(group_indices) - 1, -1, -1):
        size = int(group_sizes[j])
        idx_j = group_indices[j].to(torch.long)
        if (idx_j < 0).any() or (idx_j >= size).any():
            raise ValueError(
                f"_flatten_product_indices: index map {j} has entries "
                f"outside [0, {size})"
            )
        flat = flat + idx_j * running
        running *= size
    return flat, total


def marginalize_grouped(
    log_likelihood_per_row_per_class: (
        torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
    ),
    group_index: (
        torch.Tensor
        | tuple[torch.Tensor, ...]
        | list[torch.Tensor | tuple[torch.Tensor, ...]]
    ),
    log_prior_per_group_per_class: torch.Tensor,
    num_groups: int | tuple[int, ...],
    *,
    reduction: str = "logsumexp",
    per_group: bool = False,
) -> torch.Tensor:
    """Per-group marginalisation over a discrete latent class.

    Two surface shapes are accepted, the second a strict
    generalisation of the first.

    **Single response axis.** Given a per-(response, class)
    log-likelihood tensor of shape ``(N, K)`` and an index map
    :math:`r : \\text{Resp} \\to G` (or a tuple of co-indexed
    index maps into a product grouping plate
    :math:`G_1 \\times \\dots \\times G_r`), return the scalar

    .. math::

        \\sum_{g \\in G}\\, \\rho\\!\\left[
            \\log \\pi(g, \\cdot) + \\sum_{n:\\, r(n) = g} \\ell(n, \\cdot)
        \\right].

    **Multiple heterogeneous response axes sharing a per-group
    class indicator.** Given ``M`` per-axis log-likelihood tensors
    :math:`\\ell_m \\in \\mathbb{R}^{N_m \\times K}` and a parallel
    list of index maps :math:`r_m : \\text{Resp}_m \\to G`, return

    .. math::

        \\sum_{g \\in G}\\, \\rho\\!\\left[
            \\log \\pi(g, \\cdot)
            + \\sum_{m=1}^{M} \\sum_{n:\\, r_m(n) = g} \\ell_m(n, \\cdot)
        \\right].

    Each per-axis log-likelihood is scatter-added into the same
    ``(|G|, K)`` accumulator before the reduction. Categorically
    this is the right Kan extension along the coproduct index map
    :math:`\\coprod_m r_m : \\coprod_m \\text{Resp}_m \\to G`. The
    single-axis form is the unary case (``M = 1``).

    The reduction :math:`\\rho` over the class axis is selected by
    ``reduction``: ``logsumexp`` for the canonical mixture
    marginalisation, ``sum`` for joint scoring (no
    marginalisation), ``mean`` for the symmetric average.

    Parameters
    ----------
    log_likelihood_per_row_per_class : torch.Tensor or list/tuple of torch.Tensor
        Single tensor: a per-(response, class) log-likelihood of
        shape ``(N, K)``, used in the single-axis form. List or
        tuple of tensors: per-axis log-likelihoods of shapes
        ``(N_m, K)``, used in the multi-axis form. The class axis
        ``K`` must agree across all entries.
    group_index : torch.Tensor, tuple of torch.Tensor, or list of either
        Single-axis form: a long tensor of shape ``(N,)`` (single
        index map) or a tuple of co-indexed long tensors (product
        index map). Multi-axis form: a list of length ``M``,
        parallel to the log-likelihood list; each entry is itself
        either a single tensor or a product-index-map tuple, exactly
        as in the single-axis form.
    log_prior_per_group_per_class : torch.Tensor
        Per-(group, class) log-prior of shape ``(num_groups, K)``,
        ``(K,)`` (broadcast over the group axis), or, for product
        index maps, ``(num_groups_1, ..., num_groups_r, K)``.
    num_groups : int or tuple[int, ...]
        Cardinality of the (product) group plate.
    reduction : {"logsumexp", "sum", "mean"}
        Per-group reduction over the class axis. ``logsumexp`` is
        the canonical mixture marginalisation; ``sum`` joint-scores
        without marginalising the class; ``mean`` averages
        symmetrically. Default ``logsumexp``.
    per_group : bool
        Whether to return the reduced per-group aggregates
        ``(|G|, *extra)`` (flat over a product plate) instead of
        their sum over the group axis. A block nested inside
        another grouped block contributes one aggregate per
        position of its group to the enclosing accumulator, so
        the enclosing block consumes this form.

    Returns
    -------
    torch.Tensor
        Scalar program-level log-density contribution, or the
        per-group aggregates when ``per_group`` is set.

    Raises
    ------
    ValueError
        If ``reduction`` is not one of the three reductions, if the
        index map and log-likelihood shapes disagree, if a prior has
        a shape that fits neither the group plate nor the class
        axis, or if ``per_group`` is requested for a log-likelihood
        with no row axis (there is no group axis to keep).

    Notes
    -----
    Edge cases:

    * ``K == 1``: the log-sum-exp collapses to the body's
      log-likelihood plus the constant log-prior.
    * Identity index map (``group_index = arange(N)`` and
      ``num_groups == N``): each row is its own group, recovering
      the per-row mixture.
    * Empty fibre: contributes a multiplicative identity to the
      product (``log 1 = 0``), so an unused group adds zero per
      class before the reduction.
    * Multi-axis with a single entry (``M = 1``): exactly
      equivalent to the single-axis form, modulo unboxing the
      singleton list.
    """
    if isinstance(log_likelihood_per_row_per_class, (list, tuple)):
        if not isinstance(group_index, list):
            raise ValueError(
                "marginalize_grouped: multi-axis log-likelihood "
                "requires a parallel list of group_index entries; "
                f"got {type(group_index).__name__}"
            )
        if len(log_likelihood_per_row_per_class) != len(group_index):
            raise ValueError(
                "marginalize_grouped: log-likelihood list has "
                f"{len(log_likelihood_per_row_per_class)} entries "
                f"but group_index list has {len(group_index)}"
            )
        return _marginalize_grouped_multi(
            tuple(log_likelihood_per_row_per_class),
            tuple(group_index),
            log_prior_per_group_per_class,
            num_groups,
            reduction=reduction,
            per_group=per_group,
        )
    if reduction not in ("logsumexp", "sum", "mean"):
        raise ValueError(
            f"reduction must be one of 'logsumexp', 'sum', 'mean'; got {reduction!r}"
        )
    if log_likelihood_per_row_per_class.dim() < 1:
        raise ValueError(
            "log_likelihood_per_row_per_class must have at least one "
            "axis (the class axis); got "
            f"{log_likelihood_per_row_per_class.shape}"
        )
    # Two operating modes:
    #
    # 1. ``ll`` has a leading row axis (shape ``(N, *extra, K)``):
    #    scatter-add along the index map to obtain
    #    ``(G, *extra, K)``, apply the per-cell prior, reduce over
    #    the class axis, and sum over groups. The shape of the
    #    return value is ``(*extra,)`` — a scalar if there are no
    #    extra (outer-class) axes, otherwise a tensor whose axes
    #    are the outer-block class axes still in scope.
    #
    # 2. ``ll`` has no row axis (shape ``(*extra, K)``): this
    #    occurs for *intermediate* levels of a nested marginalize
    #    stack, where the innermost level has already integrated
    #    the per-row contributions. There is no scatter step; the
    #    prior is added across the extra axes and the class axis is
    #    reduced. The return shape is ``(*extra_except_class_axis,)``.
    # The mode is detected from the shape of ``group_index``.
    flat_group_index_shape: tuple[int, ...]
    if isinstance(group_index, tuple):
        flat_group_index_shape = group_index[0].shape
    else:
        flat_group_index_shape = group_index.shape
    if flat_group_index_shape:
        n_idx = flat_group_index_shape[0]
        # If the user supplied an index map of length N > 0, the ll
        # MUST carry a leading axis of length N. A length-N index map
        # paired with an ll whose leading axis differs is a shape
        # mismatch — not the "no row axis" intermediate path.
        if (
            n_idx > 0
            and log_likelihood_per_row_per_class.dim() >= 2
            and log_likelihood_per_row_per_class.shape[0] != n_idx
        ):
            raise ValueError(
                "group_index must have shape (N,) matching the "
                "leading axis of the log-likelihood; got "
                f"{flat_group_index_shape} vs N="
                f"{log_likelihood_per_row_per_class.shape[0]}"
            )
        has_row_axis = (
            n_idx > 0
            and log_likelihood_per_row_per_class.dim() >= 2
            and log_likelihood_per_row_per_class.shape[0] == n_idx
        )
    else:
        has_row_axis = False
    if not has_row_axis:
        # No N-axis: apply prior + reduce over the class axis only.
        # ``log_prior_per_group_per_class`` is broadcastable across
        # the extra axes (typically just (K,)).
        if per_group:
            raise ValueError(
                "marginalize_grouped: per_group requires a log-likelihood "
                "with a leading row axis; got shape "
                f"{tuple(log_likelihood_per_row_per_class.shape)}"
            )
        weighted = log_prior_per_group_per_class + log_likelihood_per_row_per_class
        if reduction == "logsumexp":
            return torch.logsumexp(weighted, dim=-1)
        if reduction == "sum":
            return weighted.sum(dim=-1)
        return weighted.mean(dim=-1)
    n_rows = int(log_likelihood_per_row_per_class.shape[0])
    n_classes = int(log_likelihood_per_row_per_class.shape[-1])
    extra_axes = tuple(log_likelihood_per_row_per_class.shape[1:-1])

    # Resolve single-vs-product index map into a flat group index.
    if isinstance(group_index, tuple):
        if not isinstance(num_groups, tuple):
            raise ValueError(
                "marginalize_grouped: product index map (tuple "
                "group_index) requires num_groups to be a tuple of "
                f"matching length; got {type(num_groups).__name__}"
            )
        flat_index, total_groups = _flatten_product_indices(group_index, num_groups)
        # Flatten an (G_1, ..., G_r, K) prior to (G_1 * ... * G_r, K).
        if log_prior_per_group_per_class.dim() == 1:
            log_prior_flat: torch.Tensor = log_prior_per_group_per_class
        elif log_prior_per_group_per_class.dim() == 2 and (
            log_prior_per_group_per_class.shape[0] == total_groups
            or log_prior_per_group_per_class.shape[0] == 1
        ):
            log_prior_flat = log_prior_per_group_per_class
        else:
            expected_axes = len(num_groups) + 1
            if log_prior_per_group_per_class.dim() != expected_axes:
                raise ValueError(
                    "marginalize_grouped: product-index-map prior "
                    f"must have shape (*{num_groups}, K) or (K,); "
                    f"got {tuple(log_prior_per_group_per_class.shape)}"
                )
            log_prior_flat = log_prior_per_group_per_class.reshape(
                total_groups, n_classes
            )
    else:
        if isinstance(num_groups, tuple):
            raise ValueError(
                "marginalize_grouped: scalar group_index requires "
                f"scalar num_groups; got {num_groups!r}"
            )
        if group_index.shape != (n_rows,):
            raise ValueError(
                "group_index must have shape (N,) matching the "
                "leading axis of the log-likelihood; got "
                f"{tuple(group_index.shape)} vs N={n_rows}"
            )
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive; got {num_groups}")
        if (group_index < 0).any() or (group_index >= num_groups).any():
            raise ValueError(
                f"group_index entries must lie in [0, {num_groups}); "
                "out-of-range index detected"
            )
        flat_index = group_index.to(torch.long)
        total_groups = int(num_groups)
        log_prior_flat = log_prior_per_group_per_class

    # Scatter-add the (N, *extra, K) per-row log-likelihood along
    # the index map to obtain a (|G|, *extra, K) per-group
    # accumulator. ``index_add`` operates on the leading axis;
    # ``extra`` axes ride along as broadcast.
    grouped_shape: tuple[int, ...] = (total_groups,) + extra_axes + (n_classes,)
    grouped = torch.zeros(
        grouped_shape,
        dtype=log_likelihood_per_row_per_class.dtype,
        device=log_likelihood_per_row_per_class.device,
    )
    grouped = grouped.index_add(0, flat_index, log_likelihood_per_row_per_class)
    # Broadcast log_prior_flat shaped ``(K,)`` or ``(G, K)``
    # against the accumulator ``(G, *extra, K)``. We insert
    # singleton axes for ``*extra`` so the broadcast is well-formed.
    prior_view = log_prior_flat
    if prior_view.dim() == 1:
        # (K,) → (1,) * len(extra) + (K,) → broadcasts against
        # (G, *extra, K).
        for _ in extra_axes:
            prior_view = prior_view.unsqueeze(0)
    elif prior_view.dim() == 2 and prior_view.shape[0] == total_groups:
        # (G, K) → (G, *(1,)*len(extra), K).
        for _ in extra_axes:
            prior_view = prior_view.unsqueeze(1)
    weighted = prior_view + grouped
    if reduction == "logsumexp":
        per_group_aggregates = torch.logsumexp(weighted, dim=-1)
    elif reduction == "sum":
        per_group_aggregates = weighted.sum(dim=-1)
    else:  # mean
        per_group_aggregates = weighted.mean(dim=-1)
    if per_group:
        return per_group_aggregates
    # Sum over the group axis; extra axes (outer-block class
    # broadcasts) pass through unchanged.
    return per_group_aggregates.sum(dim=0)


def _marginalize_grouped_multi(
    log_likelihoods: tuple[torch.Tensor, ...],
    group_indices: tuple[torch.Tensor | tuple[torch.Tensor, ...], ...],
    log_prior_per_group_per_class: torch.Tensor,
    num_groups: int | tuple[int, ...],
    *,
    reduction: str,
    per_group: bool,
) -> torch.Tensor:
    """Multi-axis grouped marginalisation.

    Each ``(ll_m, idx_m)`` pair is scatter-added into a shared
    ``(|G|, K)`` accumulator before the per-group reduction. This
    is the right Kan extension along the coproduct index map
    :math:`\\coprod_m r_m`.

    Empty list raises: the surface form ``in { }`` is rejected at
    compile time, so a caller-supplied empty list is a programming
    error, not a degenerate case to silently absorb.

    Parameters
    ----------
    log_likelihoods : tuple[torch.Tensor, ...]
        Per-axis log-likelihoods of shapes ``(N_m, *extra, K)``.
    group_indices : tuple[torch.Tensor | tuple[torch.Tensor, ...], ...]
        The index map of each axis, parallel to ``log_likelihoods``.
    log_prior_per_group_per_class : torch.Tensor
        The per-(group, class) log-prior, shaped as for
        :func:`marginalize_grouped`.
    num_groups : int | tuple[int, ...]
        Cardinality of the (product) group plate.
    reduction : str
        The per-group reduction over the class axis.
    per_group : bool
        Whether to return the per-group aggregates instead of their
        sum over the group axis.

    Returns
    -------
    torch.Tensor
        The summed contribution, or the per-group aggregates.

    Raises
    ------
    ValueError
        If the reduction is unknown, the list is empty, the class or
        extra axes disagree across entries, an index map does not fit
        its log-likelihood or the group plate, the prior has an
        unusable shape, or ``per_group`` is requested with no row
        axis.
    """
    if reduction not in ("logsumexp", "sum", "mean"):
        raise ValueError(
            f"reduction must be one of 'logsumexp', 'sum', 'mean'; got {reduction!r}"
        )
    if not log_likelihoods:
        raise ValueError(
            "marginalize_grouped: multi-axis form requires at least "
            "one (ll, idx) pair; got an empty list"
        )
    # Validate K agrees across axes.
    n_classes = int(log_likelihoods[0].shape[-1])
    for m, ll in enumerate(log_likelihoods):
        if int(ll.shape[-1]) != n_classes:
            raise ValueError(
                "marginalize_grouped: class axis disagrees across "
                f"axes: axis 0 has K={n_classes} but axis {m} has "
                f"K={int(ll.shape[-1])}"
            )

    # Resolve total_groups from the supplied ``num_groups``.
    if isinstance(num_groups, tuple):
        if any(g <= 0 for g in num_groups):
            raise ValueError(
                f"num_groups entries must all be positive; got {num_groups}"
            )
        total_groups = 1
        for g in num_groups:
            total_groups *= int(g)
    else:
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive; got {num_groups}")
        total_groups = int(num_groups)

    # Determine the "extra" middle axes (outer-block class
    # broadcasts) from the first log-likelihood. All axes are
    # required to agree on these.
    first_ll = log_likelihoods[0]
    # No-row-axis case: every entry's ll has shape ``(*extra, K)``
    # with no leading row axis. This is the "intermediate level
    # of a nested marginalize stack" path, where the inner block
    # has already produced its per-(outer-class) tensor. We sum
    # the entries (treating each as an additive contribution to
    # the same outer-class log-likelihood), add the prior, and
    # reduce over the class axis. No scatter-add takes place.
    if all(ll.dim() == first_ll.dim() and ll.dim() < 2 for ll in log_likelihoods):
        if per_group:
            raise ValueError(
                "marginalize_grouped: per_group requires log-likelihoods "
                "with a leading row axis; got shape "
                f"{tuple(first_ll.shape)}"
            )
        acc = log_likelihoods[0]
        for ll in log_likelihoods[1:]:
            acc = acc + ll
        weighted = log_prior_per_group_per_class + acc
        if reduction == "logsumexp":
            return torch.logsumexp(weighted, dim=-1)
        if reduction == "sum":
            return weighted.sum(dim=-1)
        return weighted.mean(dim=-1)
    if first_ll.dim() < 2:
        raise ValueError(
            "marginalize_grouped: multi-axis form requires each "
            "log-likelihood to carry a leading row axis and a "
            f"trailing class axis; got shape {tuple(first_ll.shape)}"
        )
    extra_axes = tuple(first_ll.shape[1:-1])

    # Build the shared accumulator and scatter each (ll, idx) pair
    # into it. The first pair seeds the dtype / device.
    grouped_shape: tuple[int, ...] = (total_groups, *extra_axes, n_classes)
    grouped = torch.zeros(
        grouped_shape,
        dtype=first_ll.dtype,
        device=first_ll.device,
    )
    for m, (ll, idx) in enumerate(zip(log_likelihoods, group_indices)):
        if ll.dim() < 2:
            raise ValueError(
                "marginalize_grouped: each log-likelihood must "
                "carry a leading row axis and a trailing class "
                f"axis; axis {m} has shape {tuple(ll.shape)}"
            )
        n_rows_m = int(ll.shape[0])
        if tuple(ll.shape[1:-1]) != extra_axes:
            raise ValueError(
                "marginalize_grouped: extra (outer-class) axes "
                "must agree across all axes; axis 0 has "
                f"{extra_axes} but axis {m} has "
                f"{tuple(ll.shape[1:-1])}"
            )
        # Resolve this axis's index map into a flat group index.
        if isinstance(idx, tuple):
            if not isinstance(num_groups, tuple):
                raise ValueError(
                    "marginalize_grouped: product index map on "
                    f"axis {m} requires num_groups to be a tuple"
                )
            flat_idx, _ = _flatten_product_indices(idx, num_groups)
        else:
            if isinstance(num_groups, tuple):
                raise ValueError(
                    f"marginalize_grouped: scalar index map on axis "
                    f"{m} cannot pair with product num_groups "
                    f"{num_groups}"
                )
            if idx.shape != (n_rows_m,):
                raise ValueError(
                    "marginalize_grouped: each group_index must "
                    "have shape (N_m,) matching its log-likelihood's "
                    f"leading axis; axis {m} got "
                    f"{tuple(idx.shape)} vs N_m={n_rows_m}"
                )
            if (idx < 0).any() or (idx >= total_groups).any():
                raise ValueError(
                    "marginalize_grouped: group_index entries must "
                    f"lie in [0, {total_groups}); axis {m} has an "
                    "out-of-range index"
                )
            flat_idx = idx.to(torch.long)
        grouped = grouped.index_add(0, flat_idx, ll)

    # Apply prior, reduce over class axis, sum over groups.
    if log_prior_per_group_per_class.dim() == 1:
        prior_flat = log_prior_per_group_per_class
    elif (
        log_prior_per_group_per_class.dim() == 2
        and log_prior_per_group_per_class.shape[0] in (total_groups, 1)
    ):
        prior_flat = log_prior_per_group_per_class
    else:
        if isinstance(num_groups, tuple):
            expected_axes = len(num_groups) + 1
            if log_prior_per_group_per_class.dim() != expected_axes:
                raise ValueError(
                    "marginalize_grouped: multi-axis prior must "
                    f"have shape (*{num_groups}, K) or (K,); got "
                    f"{tuple(log_prior_per_group_per_class.shape)}"
                )
            prior_flat = log_prior_per_group_per_class.reshape(total_groups, n_classes)
        else:
            raise ValueError(
                "marginalize_grouped: prior shape must be (K,) or "
                f"(num_groups, K)=({total_groups}, {n_classes}); "
                f"got {tuple(log_prior_per_group_per_class.shape)}"
            )
    prior_view = prior_flat
    if prior_view.dim() == 1:
        for _ in extra_axes:
            prior_view = prior_view.unsqueeze(0)
    elif prior_view.dim() == 2 and prior_view.shape[0] == total_groups:
        for _ in extra_axes:
            prior_view = prior_view.unsqueeze(1)
    weighted = prior_view + grouped
    if reduction == "logsumexp":
        per_group_aggregates = torch.logsumexp(weighted, dim=-1)
    elif reduction == "sum":
        per_group_aggregates = weighted.sum(dim=-1)
    else:
        per_group_aggregates = weighted.mean(dim=-1)
    if per_group:
        return per_group_aggregates
    return per_group_aggregates.sum(dim=0)


__all__ = [
    "PlateDraw",
    "VectorisedObserve",
    "marginalize_categorical",
    "marginalize_grouped",
]
