"""Immutable distribution contracts and native bindings for one compilation."""

from collections.abc import Callable, Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Never, SupportsIndex

import torch
from torch.distributions import Distribution, constraints

from quivers.qiec.families import FAMILIES, DistributionFamily
from quivers.qiec.family_shapes import match_family_shapes
from quivers.qiec.kinds import NAT
from quivers.qiec.types import IndexLiteral
from quivers.continuous.family_spec import FamilySpec, build_torch_distribution
from quivers.continuous.param_transforms import _validate_parameter

type DistributionFactory = Callable[[Mapping[str, torch.Tensor]], Distribution]
"""Build one native law from checked, named natural-scale parameters."""


@dataclass(frozen=True, eq=False, init=False)
class FamilyCatalog(Mapping[str, DistributionFamily]):
    """An immutable, compilation-local extension of the builtin family catalog.

    A catalog compares by identity and holds native implementations by identity.
    It never registers callbacks globally or serializes their executable code.
    Use :meth:`with_family` to obtain a new catalog sharing existing definitions.
    """

    _definitions: Mapping[str, DistributionFamily]
    _implementations: Mapping[str, DistributionFactory | FamilySpec]
    _supports: Mapping[str, constraints.Constraint]

    def __init__(self) -> None:
        object.__setattr__(self, "_definitions", MappingProxyType({}))
        object.__setattr__(self, "_implementations", MappingProxyType({}))
        object.__setattr__(self, "_supports", MappingProxyType({}))

    def __getitem__(self, name: str) -> DistributionFamily:
        """Look up a checked contract.

        Parameters
        ----------
        name : str
            Builtin or custom family name.

        Returns
        -------
        DistributionFamily
            Contract retained by this catalog.
        """
        return self._definitions[name] if name in self._definitions else FAMILIES[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate over family names.

        Returns
        -------
        Iterator[str]
            Builtin names followed by custom names.
        """
        return iter((*FAMILIES, *self._definitions))

    def __len__(self) -> int:
        """Count retained contracts.

        Returns
        -------
        int
            Number of builtin and custom families.
        """
        return len(FAMILIES) + len(self._definitions)

    def __eq__(self, other: object) -> bool:
        """Compare compilation contexts by identity.

        Parameters
        ----------
        other : object
            Candidate compilation context.

        Returns
        -------
        bool
            Whether both references denote this catalog.
        """
        return self is other

    def __hash__(self) -> int:
        """Hash the immutable compilation context by identity.

        Returns
        -------
        int
            Identity hash for this catalog.
        """
        return object.__hash__(self)

    @property
    def definitions(self) -> tuple[DistributionFamily, ...]:
        """Serializable custom contracts, excluding builtin records and callbacks.

        Returns
        -------
        tuple[DistributionFamily, ...]
            Checked custom contracts in insertion order.
        """
        return tuple(self._definitions.values())

    def is_custom(self, name: str) -> bool:
        """Whether a name has a compilation-local native implementation.

        Parameters
        ----------
        name : str
            Family name to inspect.

        Returns
        -------
        bool
            Whether the name denotes a custom family.
        """
        return name in self._definitions

    def support(self, name: str) -> constraints.Constraint:
        """Read declared support without constructing or sampling a law.

        Parameters
        ----------
        name : str
            Custom family name.

        Returns
        -------
        torch.distributions.constraints.Constraint
            Declared native support.
        """
        return self._supports[name]

    def parameter_names(self, name: str) -> tuple[str, ...]:
        """Return the canonical named constructor schema of a checked family.

        Parameters
        ----------
        name : str
            Family name in this catalog.

        Returns
        -------
        tuple[str, ...]
            Natural-scale parameter names in their declared order.

        Raises
        ------
        KeyError
            If the family is not in this catalog.
        """
        return self[name].parameter_names

    def with_family(
        self,
        semantic: DistributionFamily,
        implementation: DistributionFactory | FamilySpec,
        *,
        support: constraints.Constraint | None = None,
    ) -> FamilyCatalog:
        """Return an isolated catalog containing one additional native law.

        Parameters
        ----------
        semantic : DistributionFamily
            Checked parameter, sample-carrier, shape, support and measure contract.
        implementation : DistributionFactory or FamilySpec
            Factory receiving named tensors, or an existing native family record.
            The implementation must return the normalized law declared by the
            contract; mathematical correctness remains the author's obligation.
        support : torch.distributions.constraints.Constraint or None
            Native support for a factory. Real-valued support defaults to real;
            other supports must be explicit. FamilySpec already retains support.

        Returns
        -------
        FamilyCatalog
            New catalog; existing compilations retain their original bindings.

        Raises
        ------
        ValueError
            If a name or stable identity collides, or metadata disagrees.
        TypeError
            If the implementation is not a native record or callable.
        """
        if semantic.name in self or any(
            item.id == semantic.id for item in self.values()
        ):
            raise ValueError(
                f"distribution family {semantic.name!r} is already defined"
            )
        if isinstance(implementation, FamilySpec):
            if (
                implementation.name != semantic.name
                or implementation.param_names != semantic.parameter_names
            ):
                raise ValueError(
                    "native family name and parameters must match its semantic contract"
                )
            if support is not None:
                raise ValueError("FamilySpec already declares its native support")
            support = implementation.support
        elif not callable(implementation):
            raise TypeError(
                "a native family implementation must be a FamilySpec or factory"
            )
        if semantic.compositional:
            raise ValueError(
                "custom native families declare tensor parameters, not measure operators"
            )
        if semantic.targets:
            raise ValueError(
                "native custom families cannot claim external target implementations"
            )
        semantic = semantic.with_(targets=MappingProxyType({}))
        if support is None:
            if semantic.support != "real":
                raise ValueError(
                    "custom factories with non-real support require explicit native support"
                )
            support = constraints.real
        if not isinstance(support, constraints.Constraint):
            raise TypeError("custom native support must be a torch Constraint")
        result = FamilyCatalog()
        object.__setattr__(
            result,
            "_definitions",
            MappingProxyType({**self._definitions, semantic.name: semantic}),
        )
        object.__setattr__(
            result,
            "_implementations",
            MappingProxyType({**self._implementations, semantic.name: implementation}),
        )
        object.__setattr__(
            result,
            "_supports",
            MappingProxyType({**self._supports, semantic.name: support}),
        )
        return result

    def with_catalog(self, other: FamilyCatalog) -> FamilyCatalog:
        """Combine immutable extensions without replacing executable identities.

        Parameters
        ----------
        other : FamilyCatalog
            Independently retained definitions to add to this catalog.

        Returns
        -------
        FamilyCatalog
            A catalog sharing the original implementations and support objects.
            Identical repeated definitions are retained once.

        Raises
        ------
        TypeError
            If other is not a FamilyCatalog.
        ValueError
            If a repeated family name has another contract, implementation,
            or support identity, or a new name collides with a stable family id.
        """
        if not isinstance(other, FamilyCatalog):
            raise TypeError("catalog composition requires a FamilyCatalog")
        result = self
        for name, semantic in other._definitions.items():
            implementation = other._implementations[name]
            support = other._supports[name]
            if name in result._definitions:
                if (
                    result._definitions[name] != semantic
                    or result._implementations[name] is not implementation
                    or result._supports[name] is not support
                ):
                    raise ValueError(
                        f"distribution family {name!r} has conflicting catalog definitions"
                    )
                continue
            result = result.with_family(
                semantic,
                implementation,
                **(
                    {}
                    if isinstance(implementation, FamilySpec)
                    else {"support": support}
                ),
            )
        return result

    def renamed(self, names: Mapping[str, str]) -> FamilyCatalog:
        """Return this catalog with custom families under other names.

        Parameters
        ----------
        names : Mapping[str, str]
            New name of each renamed family; other families keep theirs.

        Returns
        -------
        FamilyCatalog
            A catalog sharing the original implementations and supports.

        Raises
        ------
        ValueError
            If two families would share one name.
        """
        result = FamilyCatalog()
        for name, semantic in self._definitions.items():
            implementation = self._implementations[name]
            target = names.get(name, name)
            if isinstance(implementation, FamilySpec):
                result = result.with_family(
                    semantic.with_(name=target), implementation.with_(name=target)
                )
                continue
            result = result.with_family(
                semantic.with_(name=target),
                implementation,
                support=self._supports[name],
            )
        return result

    def build(self, name: str, arguments: Mapping[str, torch.Tensor]) -> Distribution:
        """Construct and validate a custom law without sampling or detaching it.

        Parameters
        ----------
        name : str
            Custom family name.
        arguments : Mapping[str, torch.Tensor]
            Named natural-scale values retaining batch and intrinsic axes.

        Returns
        -------
        torch.distributions.Distribution
            Native law with the declared event and parameter batch geometry.

        Raises
        ------
        KeyError
            If the custom implementation is absent.
        TypeError, ValueError
            If arguments or the returned native law violate the checked contract.
        """
        record = self._definitions[name]
        if set(arguments) != set(record.parameter_names):
            raise TypeError(
                f"custom family {name!r} requires parameters {record.parameter_names}"
            )
        values = dict(arguments)
        shapes = {}
        batches = []
        for parameter in record.parameters:
            value = values[parameter.name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"custom family parameter {parameter.name!r} must be a tensor"
                )
            if parameter.constraint == "boolean":
                if value.dtype != torch.bool:
                    raise TypeError(
                        f"custom family parameter {parameter.name!r} requires Bool"
                    )
            elif value.dtype == torch.bool or value.is_complex():
                raise TypeError(
                    f"custom family parameter {parameter.name!r} requires real numeric values"
                )
            condition = getattr(constraints, parameter.constraint, None)
            if isinstance(condition, constraints.Constraint):
                _validate_parameter(
                    value,
                    condition,
                    description=f"custom family parameter {parameter.name!r} violates {parameter.constraint}",
                )
            if value.ndim < parameter.rank:
                raise ValueError(
                    f"custom family parameter {parameter.name!r} has insufficient intrinsic rank"
                )
            shapes[parameter.name] = (
                tuple(value.shape[-parameter.rank :]) if parameter.rank else ()
            )
            batches.append(
                value.shape[: -parameter.rank] if parameter.rank else value.shape
            )
        implementation = self._implementations[name]
        result = (
            implementation.mixed_builder_override(
                [values[name] for name in record.parameter_names]
            )
            if isinstance(implementation, FamilySpec)
            and implementation.mixed_builder_override is not None
            else build_torch_distribution(implementation, values)
            if isinstance(implementation, FamilySpec)
            else implementation(MappingProxyType(values))
        )
        if not isinstance(result, Distribution):
            raise TypeError(f"custom family {name!r} must return a torch Distribution")
        native_support = result.support
        if (
            native_support is None
            or bool(native_support.is_discrete) != record.discrete
        ):
            raise TypeError(
                f"custom family {name!r} returned a different sample carrier"
            )
        if len(result.event_shape) != record.event_rank:
            raise ValueError(
                f"custom family {name!r} returned event rank {len(result.event_shape)}, expected {record.event_rank}"
            )
        match_family_shapes(
            record.effective_shape_signature,
            {
                key: tuple(IndexLiteral(value=size, sort=NAT) for size in shape)
                for key, shape in shapes.items()
            },
            tuple(IndexLiteral(value=size, sort=NAT) for size in result.event_shape),
        )
        expected_batch = torch.broadcast_shapes(*batches) if batches else torch.Size()
        if result.batch_shape != expected_batch:
            raise ValueError(
                f"custom family {name!r} returned batch shape {result.batch_shape}, expected {expected_batch}"
            )
        if record.reparameterizable and not result.has_rsample:
            raise TypeError(
                f"custom family {name!r} declares reparameterized sampling but its native law does not provide it"
            )
        result = copy(result)
        setattr(result, "_family_contract", record)
        return result

    def __copy__(self) -> FamilyCatalog:
        """Share this immutable catalog while retaining implementation identities.

        Returns
        -------
        FamilyCatalog
            This immutable catalog.
        """
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> FamilyCatalog:
        """Retain catalog identity when snapshotting fitting settings.

        Parameters
        ----------
        memo : dict[int, object]
            Objects already copied by the snapshot operation.

        Returns
        -------
        FamilyCatalog
            This immutable catalog.
        """
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        """Reject serialization of executable implementations.

        Parameters
        ----------
        protocol : SupportsIndex
            Requested pickle protocol.

        Raises
        ------
        TypeError
            Always; implementations must be explicitly reattached.
        """
        raise TypeError(
            "family catalogs contain live native implementations; retain semantic metadata and explicitly reattach implementations"
        )


__all__ = ["DistributionFactory", "FamilyCatalog"]
