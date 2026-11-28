"""The backend protocol and the support-tier check.

[`unsupported_for`][quivers.transpile.unsupported_for] reads a parsed
module's top-level statements against a backend's support tier, and
[`Backend`][quivers.transpile.Backend] is the protocol a backend module
satisfies. Both name the DSL's AST types, so they live apart from
`quivers.transpile._api`, which the DSL compiler reaches during its own
initialization.
"""

from __future__ import annotations

from typing import Protocol

from quivers.dsl.ast_nodes import Module, Statement
from quivers.transpile._api import (
    CATEGORICAL_METADATA_IGNORABLE,
    QIEC_SURFACE,
    STRUCTURAL_QIEC,
    UnsupportedConstruct,
)


class Backend(Protocol):
    """The protocol every backend module satisfies.

    Backends register themselves via
    `didactic.codegen.emitter` under a
    ``"qvr-<name>"`` key. Quivers' top-level
    [`transpile`][quivers.transpile.transpile] dispatches by looking up
    the registered emitter, then delegates to its
    `emit_instance`.

    Attributes
    ----------
    file_extension
        Canonical filename extension (``"stan"``, ``"py"``, ``"jl"``,
        ``"js"``, ``"scm"``).
    grammar
        The tree-sitter grammar name backing this backend, as accepted by
        `panproto.AstParserRegistry.parse_with_protocol`.
    support
        The probabilistic-subset support tier accepted by this backend.
    """

    file_extension: str
    grammar: str
    support: frozenset[str]

    def emit_instance(self, module: Module) -> bytes:
        """Transpile a parsed QVR module to bytes.

        Parameters
        ----------
        module
            The parsed module.

        Returns
        -------
        bytes
            The target program's source.

        Raises
        ------
        UnsupportedConstruct
            If the module holds constructs the backend cannot represent.
        """
        ...


def unsupported_for(target: str, module: Module, *, allow: frozenset[str]) -> None:
    """Raise [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
    if ``module`` contains statement kinds outside ``allow``.

    Walks the module's top-level statements; the ``kind`` field is the
    didactic `TaggedUnion` discriminator
    (``"program_decl"``, ``"morphism_decl"``, etc.). Any kind not in
    ``allow`` is collected; if the resulting set is non-empty, raises.

    [`CATEGORICAL_METADATA_IGNORABLE`][quivers.transpile.CATEGORICAL_METADATA_IGNORABLE]
    kinds (``composition_decl``, ``category_decl``, ``schema_decl``,
    ``bundle_decl``, ``rule_decl``, ``contraction_decl``,
    ``signature_decl``, ``deduction_decl``) are accepted ALONGSIDE a
    ``program_decl``. The QIEC forms in
    [`QIEC_SURFACE`][quivers.transpile.QIEC_SURFACE]
    are always admitted because the caller has already lowered and checked the
    complete QIEC submodule. Renderer-level capability analysis decides which
    executable QIEC features the selected target can preserve.

    Parameters
    ----------
    target
        The backend name reported in the error, such as ``"qvr-stan"``.
    module
        The parsed module to check.
    allow
        The statement kinds the backend accepts, usually one of the
        support tiers such as [`STAN_LIKE`][quivers.transpile.STAN_LIKE].

    Raises
    ------
    UnsupportedConstruct
        If a top-level statement's kind is outside the effective
        allowance.
    """
    kinds = {cast_kind(s) for s in module.statements}
    has_program = "program_decl" in kinds
    has_schema_parser = any(
        cast_kind(statement) == "define_decl"
        and str(getattr(getattr(statement, "expr", None), "kind", "")) == "expr_parser"
        for statement in module.statements
    )
    has_structural = "signature_decl" in kinds and bool(
        kinds & {"encoder_decl", "decoder_decl", "interpretation_decl"}
    )
    effective_allow = allow | QIEC_SURFACE
    if has_program or has_schema_parser:
        effective_allow |= CATEGORICAL_METADATA_IGNORABLE
    if has_structural:
        effective_allow |= STRUCTURAL_QIEC
    bad: set[str] = set()
    for statement in module.statements:
        kind = cast_kind(statement)
        if kind not in effective_allow:
            bad.add(kind)
    if bad:
        raise UnsupportedConstruct(target, sorted(bad))


def cast_kind(statement: Statement) -> str:
    """Return ``statement.kind`` as a string.

    The didactic `TaggedUnion` discriminator
    is typed `Literal[...]`; the cast is a single boundary line so the
    caller stays free of literal-narrowing noise.
    """
    return str(getattr(statement, "kind"))
