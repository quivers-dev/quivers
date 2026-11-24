"""Custom response laws extend the existing formula and native family paths."""

from __future__ import annotations

from collections.abc import Mapping
import re

from quivers.family_catalog import FamilyCatalog
from quivers.formulas._custom_types import CustomFamilySpec
from quivers.formulas.family import AuxParam, Family, Link, links
from quivers.formulas._predictor_graph import NonlinearExpression


def custom_family(
    name: str,
    *,
    catalog: FamilyCatalog,
    location: str,
    link: str | Link = "identity",
    auxiliary: tuple[AuxParam, ...] | Mapping[str, AuxParam] = (),
    data: Mapping[str, str] | None = None,
) -> Family:
    """Bind an arbitrary checked native law to ordinary regression predictors.

    Parameters
    ----------
    name : str
        Custom distribution name already defined in the supplied catalog.
    catalog : FamilyCatalog
        Immutable semantic contracts and native distribution factories.
    location : str
        Native parameter receiving the linked main predictor named ``mu``.
    link : str or Link
        Existing inverse link for that predictor.
    auxiliary : tuple[AuxParam, ...] or Mapping[str, AuxParam]
        Other native parameters, with ordinary priors and distributional links.
        A mapping separates native names from formula names, for instance
        native ``scale`` mapped to an auxiliary named ``sigma``.
    data : Mapping[str, str] or None
        Native parameters supplied by typed numeric data columns. These enter
        the existing predictor graph as deterministic identity-linked roots.

    Returns
    -------
    Family
        Response law using the same compiler, inference and replay operations.

    Raises
    ------
    ValueError, TypeError
        If parameter roles overlap, omit a constructor parameter, or use an
        unknown link, noncustom name, or invalid deterministic input name.
    """
    if not isinstance(catalog, FamilyCatalog):
        raise TypeError("catalog must be an immutable FamilyCatalog")
    if name not in {record.name for record in catalog.definitions}:
        raise ValueError(f"{name!r} is not a custom law in this catalog")
    record = catalog[name]
    if isinstance(link, str):
        if link not in links:
            raise ValueError(f"unknown inverse link {link!r}")
        link = links[link]
    if not isinstance(link, Link):
        raise TypeError("link must be a registered inverse link or Link")
    if link.name not in links or link != links[link.name]:
        raise ValueError("custom families use a registered inverse link")
    extra = dict(data or {})
    if not isinstance(auxiliary, (tuple, Mapping)):
        raise TypeError("auxiliary must be a tuple or mapping of AuxParam values")
    auxiliary_values = (
        tuple(auxiliary.values()) if isinstance(auxiliary, Mapping) else auxiliary
    )
    if any(not isinstance(item, AuxParam) for item in auxiliary_values):
        raise TypeError("auxiliary parameters must be AuxParam values")
    auxiliary_map = (
        dict(auxiliary)
        if isinstance(auxiliary, Mapping)
        else {item.name: item for item in auxiliary_values}
    )
    names = [item.name for item in auxiliary_values]
    if len(auxiliary_map) != len(auxiliary) or len(set(names)) != len(names):
        raise ValueError("custom auxiliary formula names must be distinct")
    roles = (location, *auxiliary_map, *extra)
    if len(set(roles)) != len(roles) or set(roles) != set(record.parameter_names):
        raise ValueError(
            "location, auxiliary and data roles must partition all native parameters"
        )
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
        for value in (*names, *extra, *extra.values())
    ):
        raise ValueError(
            "custom parameter and data column names require QVR identifiers"
        )
    if "mu" in names:
        raise ValueError("mu is reserved for the main formula predictor")
    # Deterministic inputs use private graph roots, avoiding collisions with
    # authored constructor names and ordinary population coefficient sites.
    data_roots = {parameter: f"custom_data_{parameter}" for parameter in extra}
    if set(names) & set(data_roots.values()):
        raise ValueError("custom_data_ names are reserved for declared data parameters")
    deterministic = tuple(
        AuxParam(name=root, prior="Normal(0.0, 1.0)", link=links["identity"])
        for root in data_roots.values()
    )
    binding_names = {
        location: "mu",
        **{native: value.name for native, value in auxiliary_map.items()},
        **data_roots,
    }
    bindings = tuple(
        (parameter, binding_names[parameter]) for parameter in record.parameter_names
    )
    return Family(
        name=name,
        observe_family=name,
        location_link=link,
        # The native law constrains its own parameters and recorded events.
        location_links=tuple(links),
        recorded_events=True,
        aux_params=tuple(
            aux if aux.links is not None else aux.with_(links=tuple(links))
            for aux in (*auxiliary_values, *deterministic)
        ),
        custom=CustomFamilySpec(
            catalog=catalog,
            bindings=bindings,
            data=tuple((data_roots[name], column) for name, column in extra.items()),
        ),
    )


def custom_distributional(
    family: Family,
    distributional: Mapping | None,
    *,
    fixed_parameters: Mapping | None = None,
) -> Mapping | None:
    """Inject declared deterministic inputs into the existing checked graph.

    Parameters
    ----------
    family : Family
        Retained native constructor and input-role bindings.
    distributional : Mapping or None
        Authored parameter regressions and deterministic roots.
    fixed_parameters : Mapping or None
        Explicit fixed auxiliary values, checked for conflicting data roles.

    Returns
    -------
    Mapping or None
        Original roots extended with deterministic data parameters.

    Raises
    ------
    ValueError
        If a declared data parameter is also regressed or fixed.
    """
    if family.custom is None or not family.custom.data:
        return distributional
    roots = dict(distributional or {})
    for name, column in family.custom.data:
        if name in roots or name in (fixed_parameters or {}):
            raise ValueError(
                f"data parameter {name!r} cannot also be regressed or fixed"
            )
        roots[name] = NonlinearExpression(expression=column)
    return roots
