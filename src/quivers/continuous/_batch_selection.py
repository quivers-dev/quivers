"""Select native distribution batches from declared constructor parameter ranks."""

from __future__ import annotations

import inspect
from typing import cast

import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Independent,
    TransformedDistribution,
    MixtureSameFamily,
)
from torch.distributions import transforms as T

from quivers.continuous._atoms import _family_metadata


def _constructor_parameters(distribution, record):
    """Choose one faithful native representation of alternative parameters.

    Parameters
    ----------
    distribution : Distribution
        Registered native distribution.
    record : DistributionFamily
        Semantic parameter names and intrinsic ranks.

    Returns
    -------
    dict
        Constructor parameter names mapped to their intrinsic ranks.
    """
    ranks = {parameter.name: parameter.rank for parameter in record.parameters}
    if "probs" in ranks and "logits" in ranks:
        raw = getattr(distribution, "_param", None)
        closed = getattr(distribution, "_closed_probs", None)
        probability = getattr(distribution, "probs", None)
        use_probs = closed is not None or (raw is not None and raw is probability)
        ranks.pop("logits" if use_probs else "probs")
    # A Cholesky factor is a complete differentiable representation regardless
    # of which covariance parameterization the original constructor accepted.
    if "scale_tril" in ranks:
        ranks.pop("covariance_matrix", None)
        ranks.pop("precision_matrix", None)
    if "log_rate" in ranks and "rate" in ranks:
        boundary = getattr(distribution, "_boundary_rate", None)
        preserve_rate = boundary is getattr(distribution, "rate", None)
        ranks.pop("log_rate" if preserve_rate else "rate")
    return ranks


def _parameter_at(value, batch_shape, rank, indices, output_shape):
    """Select declared batch axes while retaining all intrinsic parameter axes.

    Parameters
    ----------
    value : Tensor
        Original parameter tensor.
    batch_shape : torch.Size
        Declared native batch shape.
    rank : int
        Intrinsic parameter rank.
    indices : Tensor
        Selected flattened batch indices.
    output_shape : torch.Size
        Selected batch layout.

    Returns
    -------
    Tensor
        Selected parameter with unchanged intrinsic axes.
    """
    if value.ndim < rank:
        return value
    intrinsic = value.shape[-rank:] if rank else torch.Size()
    full = value.expand(batch_shape + intrinsic)
    return (
        full.reshape((batch_shape.numel(),) + intrinsic)
        .index_select(0, indices.to(value.device))
        .reshape(output_shape + intrinsic)
    )


def select_parameter(value, batch_shape, rank, active, *, output_shape=None):
    """Select a tensor using its declared intrinsic rank, never equal-size guesses.

    Parameters
    ----------
    value : Tensor
        Broadcast tensor parameter.
    batch_shape : torch.Size
        Native batch axes.
    rank : int
        Intrinsic trailing parameter rank.
    active : Tensor
        Boolean mask over native batch axes.
    output_shape : torch.Size or None
        Optional layout of selected batch entries; default is one flat axis.

    Returns
    -------
    Tensor
        Selected tensor with all intrinsic parameter axes retained.
    """
    shape = torch.Size(batch_shape)
    indices = active.expand(shape).reshape(-1).nonzero(as_tuple=False).flatten()
    target = (
        torch.Size((indices.numel(),))
        if output_shape is None
        else torch.Size(output_shape)
    )
    return _parameter_at(value, shape, rank, indices, target)


def _reconstruct(distribution, transform_parameter):
    """Apply one parameter-axis operation through declared native constructors.

    Parameters
    ----------
    distribution : Distribution
        Registered native family or explicit constructor-rank capability.
    transform_parameter : Callable
        Maps each tensor and its intrinsic rank to new batch axes.

    Returns
    -------
    Distribution
        Reconstructed law retaining scalar constructor resource policies.

    Raises
    ------
    NotImplementedError
        If required constructor metadata is unavailable.
    """
    metadata = _family_metadata(distribution)
    declared = getattr(distribution, "_batch_parameter_ranks", None)
    if metadata is None and declared is None:
        raise NotImplementedError(
            f"{type(distribution).__name__} requires declared native batch-selection metadata"
        )
    if metadata is None:
        ranks = dict(cast(dict[str, int], declared))
        declared_names = set(ranks)
    else:
        _, record = metadata
        ranks = _constructor_parameters(distribution, record)
        declared_names = {p.name for p in record.parameters}
    parameters = {}
    for name, rank in ranks.items():
        value = getattr(distribution, name, None)
        if not isinstance(value, Tensor):
            raise NotImplementedError(
                f"{type(distribution).__name__}.{name} is not a tensor constructor parameter"
            )
        parameters[name] = transform_parameter(value, rank)
    signature = inspect.signature(type(distribution)).parameters
    for name, parameter in signature.items():
        if (
            name not in parameters
            and name not in declared_names
            and name != "validate_args"
        ):
            value = getattr(distribution, name, parameter.default)
            if value is not inspect.Parameter.empty and not isinstance(value, Tensor):
                parameters[name] = value
    if "validate_args" in signature:
        parameters["validate_args"] = distribution._validate_args
    return type(distribution)(**parameters)


def expand_registered_batch(distribution, batch_shape):
    """Broadcast native constructor parameters through the same rank contract.

    Parameters
    ----------
    distribution : Distribution
        Registered native law.
    batch_shape : torch.Size
        Requested broadcast batch axes.

    Returns
    -------
    Distribution
        Law preserving its native class and original parameter coordinates.
    """
    shape = torch.Size(batch_shape)

    def expand(value, rank):
        """Expand one tensor while retaining its declared intrinsic axes.

        Parameters
        ----------
        value : Tensor
            Native tensor parameter.
        rank : int
            Intrinsic parameter rank.

        Returns
        -------
        Tensor
            Expanded parameter view.
        """
        intrinsic = value.shape[-rank:] if rank else torch.Size()
        return value.expand(shape + intrinsic)

    return _reconstruct(distribution, expand)


def _select_transform(transform, batch_shape, indices, output_shape):
    """Select declared tensor parameters of native transform compositions.

    Parameters
    ----------
    transform : Transform
        Native bijective or deterministic transform.
    batch_shape : torch.Size
        Native distribution batch axes.
    indices : Tensor
        Selected flat batch indices.
    output_shape : torch.Size
        Selected batch layout.

    Returns
    -------
    Transform
        Transform with selected batch parameters and unchanged event metadata.

    Raises
    ------
    NotImplementedError
        If a custom transform supplies no parameter-axis capability.
    """
    if isinstance(transform, T.ComposeTransform):
        return T.ComposeTransform(
            [
                _select_transform(part, batch_shape, indices, output_shape)
                for part in transform.parts
            ]
        )
    if isinstance(transform, T._InverseTransform):
        return _select_transform(transform.inv, batch_shape, indices, output_shape).inv
    if isinstance(transform, T.IndependentTransform):
        return T.IndependentTransform(
            _select_transform(
                transform.base_transform, batch_shape, indices, output_shape
            ),
            transform.reinterpreted_batch_ndims,
        )
    if isinstance(transform, T.AffineTransform):
        rank = transform.event_dim

        def parameter(value):
            """Select affine tensor coordinates under the declared event rank.

            Parameters
            ----------
            value : Tensor or float
                Affine parameter.

            Returns
            -------
            Tensor or float
                Selected tensor or unchanged shared scalar.
            """
            return (
                _parameter_at(value, batch_shape, rank, indices, output_shape)
                if isinstance(value, Tensor)
                else value
            )

        return T.AffineTransform(
            parameter(transform.loc), parameter(transform.scale), event_dim=rank
        )
    if isinstance(transform, T.PowerTransform):
        return T.PowerTransform(
            _parameter_at(transform.exponent, batch_shape, 0, indices, output_shape)
        )
    stateless = (
        T.SigmoidTransform,
        T.ExpTransform,
        T.AbsTransform,
        T.TanhTransform,
        T.SoftplusTransform,
        T.SoftmaxTransform,
        T.StickBreakingTransform,
        T.LowerCholeskyTransform,
        T.PositiveDefiniteTransform,
        T.CorrCholeskyTransform,
    )
    if type(transform) in stateless:
        return transform
    raise NotImplementedError(
        f"{type(transform).__name__} requires declared transform batch parameters"
    )


def _select(distribution, indices, output_shape):
    """Reconstruct selected batches using declared intrinsic parameter axes.

    Parameters
    ----------
    distribution : Distribution
        Registered distribution or Independent composition.
    indices : Tensor
        Flat selected batch indices.
    output_shape : torch.Size
        Shape of the selected batch, before intrinsic event dimensions.

    Returns
    -------
    Distribution
        Distribution with the selected batch axes.
    Raises
    ------
    NotImplementedError
        If a nested custom law or transform has no selection capability.
    """
    method = getattr(distribution, "select_batch", None)
    if callable(method):
        active = (
            torch.zeros(
                distribution.batch_shape.numel(),
                dtype=torch.bool,
                device=indices.device,
            )
            .scatter(0, indices, True)
            .reshape(distribution.batch_shape)
        )
        if "output_shape" in inspect.signature(method).parameters:
            return cast(Distribution, method(active, output_shape=output_shape))
        if output_shape != torch.Size((indices.numel(),)):
            raise NotImplementedError(
                "custom batch selection requires an output_shape capability for nested event axes"
            )
        return cast(Distribution, method(active))
    if isinstance(distribution, Independent):
        inner = distribution.base_dist.batch_shape[len(distribution.batch_shape) :]
        count = inner.numel()
        expanded = (
            indices[:, None] * count + torch.arange(count, device=indices.device)
        ).reshape(-1)
        base = _select(distribution.base_dist, expanded, output_shape + inner)
        return Independent(
            base,
            distribution.reinterpreted_batch_ndims,
            validate_args=distribution._validate_args,
        )

    if isinstance(distribution, MixtureSameFamily):
        count = distribution.component_distribution.batch_shape[-1]
        mixture = _select(
            distribution.mixture_distribution.expand(distribution.batch_shape),
            indices,
            output_shape,
        )
        expanded = (
            indices[:, None] * count + torch.arange(count, device=indices.device)
        ).reshape(-1)
        components = _select(
            distribution.component_distribution.expand(
                distribution.batch_shape + (count,)
            ),
            expanded,
            output_shape + (count,),
        )
        return MixtureSameFamily(
            mixture, components, validate_args=distribution._validate_args
        )
    if (
        isinstance(distribution, TransformedDistribution)
        and _family_metadata(distribution) is None
        and getattr(distribution, "_batch_parameter_ranks", None) is None
    ):
        if type(distribution) is not TransformedDistribution:
            raise NotImplementedError(
                "custom transformed distributions require native constructor metadata"
            )
        base = _select(
            distribution.base_dist.expand(distribution.batch_shape),
            indices,
            output_shape,
        )
        transforms = [
            _select_transform(
                transform, distribution.batch_shape, indices, output_shape
            )
            for transform in distribution.transforms
        ]
        return TransformedDistribution(
            base, transforms, validate_args=distribution._validate_args
        )

    def select(value, rank):
        """Select a parameter using the enclosing native batch contract.

        Parameters
        ----------
        value : Tensor
            Native tensor parameter.
        rank : int
            Intrinsic parameter rank.

        Returns
        -------
        Tensor
            Selected parameter view.
        """
        return _parameter_at(
            value, distribution.batch_shape, rank, indices, output_shape
        )

    return _reconstruct(distribution, select)


def select_batch(
    distribution: Distribution,
    active: Tensor,
    *,
    output_shape: torch.Size | None = None,
) -> Distribution:
    """Select active batch rows without sampling or inspecting cached shapes.

    Parameters
    ----------
    distribution : Distribution
        Native law with registered constructor ranks, or a custom
        ``select_batch(active)`` capability.
    active : Tensor
        Boolean mask broadcastable to the native batch shape.
    output_shape : torch.Size or None
        Optional selected batch layout for nested independent event axes.

    Returns
    -------
    Distribution
        Law whose single batch axis contains only selected entries.

    Raises
    ------
    TypeError
        If the mask is not Boolean.
    NotImplementedError
        If the law has no declared reconstruction capability.
    """
    if active.dtype != torch.bool:
        raise TypeError("native batch selection requires a Boolean mask")
    active = active.expand(distribution.batch_shape)
    indices = active.reshape(-1).nonzero(as_tuple=False).flatten()
    target = (
        torch.Size((indices.numel(),))
        if output_shape is None
        else torch.Size(output_shape)
    )
    return _select(distribution, indices, target)


def sample_active(
    distribution: Distribution, sample_shape: torch.Size, active: Tensor
) -> Tensor:
    """Draw only selected native batch rows and scatter unused placeholders.

    Parameters
    ----------
    distribution : Distribution
        Native distribution with declared batch reconstruction metadata.
    sample_shape : torch.Size
        Leading independent sampling dimensions.
    active : Tensor
        Boolean mask over the native batch axes.

    Returns
    -------
    Tensor
        Samples with unchanged sample/batch/event axes; inactive entries are zero.
    """
    active = active.expand(distribution.batch_shape)
    selected = select_batch(distribution, active)
    draws = selected.sample(sample_shape)
    indices = active.reshape(-1).nonzero(as_tuple=False).flatten().to(draws.device)
    flat = draws.new_zeros(
        sample_shape + (distribution.batch_shape.numel(),) + distribution.event_shape
    )
    flat = flat.index_copy(len(sample_shape), indices, draws)
    return flat.reshape(
        sample_shape + distribution.batch_shape + distribution.event_shape
    )
