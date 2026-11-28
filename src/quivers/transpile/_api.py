"""The refusal error and support tiers of `quivers.transpile`.

Defines the [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
error raised when a backend cannot represent a QVR construct, and the
support-tier frozensets every backend declares. The module imports
nothing from the DSL, so every transpile module can import it while the
DSL package is still initializing.
"""

from __future__ import annotations

from quivers.transpile._diagnostics import (
    RefusedDeclaration,
    user_facing_message,
)


class UnsupportedConstruct(Exception):
    """Raised when a backend cannot transpile one or more QVR constructs.

    The exception message describes the unsupported constructs and possible
    replacements. The structured identifiers remain available in ``kinds``.

    Parameters
    ----------
    target
        Backend name, such as ``"qvr-stan"``.
    kinds
        Construct identifiers suitable for programmatic matching; they are
        sorted and deduplicated on construction.
    declarations
        Affected top-level declarations, when applicable.
    module_has_program
        Whether the refused module declares a probabilistic program.

    Attributes
    ----------
    target
        Backend name, such as ``"qvr-stan"``.
    kinds
        Sorted, deduplicated construct identifiers.
    declarations
        Affected top-level declarations, when applicable.
    module_has_program
        Whether the refused module declares a probabilistic program.
    """

    def __init__(
        self,
        target: str,
        kinds: list[str],
        *,
        declarations: tuple[RefusedDeclaration, ...] = (),
        module_has_program: bool = False,
    ) -> None:
        self.target = target
        self.kinds = sorted(set(kinds))
        self.declarations = declarations
        self.module_has_program = module_has_program
        super().__init__(
            user_facing_message(
                target,
                tuple(self.kinds),
                declarations,
                module_has_program,
            )
        )


_NO_TARGET_HEADS: frozenset[str] = frozenset(
    {
        "no-stan-target",
        "no-bugs-target",
        "no-jags-target",
        "no-target-name",
        "no-webppl-target",
        "no-pymc-target",
        "no-edward2-target",
        "no-numpyro-target",
        "no-pyro-target",
        "no-gen-target",
        "no-turing-target",
        "no-church-target",
    }
)


STAN_LIKE: frozenset[str] = frozenset(
    {
        "object_decl",
        "morphism_decl",
        "define_decl",
        "program_decl",
        "export_decl",
    }
)
"""Statement kinds every PPL backend accepts.

The probabilistic-program surface: declarations and program bodies of
sample, observe, let, score, return, and marginalize steps. It excludes
categorical-algebra and neural-network declarations.
"""

CATEGORICAL_METADATA_IGNORABLE: frozenset[str] = frozenset(
    {
        "category_decl",
        "schema_decl",
        "composition_decl",
        "bundle_decl",
        "rule_decl",
        "contraction_decl",
        "signature_decl",
        "deduction_decl",
    }
)
"""Categorical declaration kinds the walker ignores beside a ``program_decl``.

A module carrying only these declarations still raises ``UnsupportedConstruct``
naming them, so standalone categorical declarations remain rejected.
"""

QIEC_SURFACE: frozenset[str] = frozenset(
    {
        "index_decl",
        "indexed_family_decl",
        "effect_decl",
        "effect_instance_decl",
        "handler_decl",
        "computation_decl",
        "interpretation_decl",
    }
)
"""QIEC declaration kinds admitted once the shared boundary has checked the module.

These forms are first-class inputs to the structural IR and need no
accompanying ``program_decl``; each renderer applies its capability policy
after lowering, where a diagnostic can name the computation and feature.
"""

STRUCTURAL_QIEC: frozenset[str] = frozenset(
    {"signature_decl", "encoder_decl", "decoder_decl", "loss_decl"}
)
"""Structural declarations elaborated as host-backed QIEC computations."""

PYTHON_DEEP: frozenset[str] = STAN_LIKE | frozenset({"encoder_decl", "decoder_decl"})
"""[`STAN_LIKE`][quivers.transpile.STAN_LIKE] plus encoder and decoder declarations.

The tier of backends with a deep-learning idiom: Pyro and NumPyro
modules.
"""

CHURCH_LIKE: frozenset[str] = STAN_LIKE
"""The probabilistic subset Church and WebPPL accept.

The same kinds as [`STAN_LIKE`][quivers.transpile.STAN_LIKE]; these
targets realise ``marginalize`` as a continuation-style ``Infer`` or
``enumerate-query``.
"""
