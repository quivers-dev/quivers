"""Structural compression: signatures, encoders, decoders, losses.

Runtime substrate for QVR signatures, encoders, decoders, and attached losses:
a uniform algebraic interface for compressing structured objects to
fixed-length vectors and decoding them under a learned distribution.
"""

from __future__ import annotations

from quivers.structural.signature import (
    DataLeaf,
    TermArg,
    SortKind,
    SortVocabEntry,
    Sort,
    Constructor,
    BinderVarSpec,
    BinderArgSpec,
    Binder,
    VertexKind,
    EdgeKind,
    Signature,
    Term,
    bound_var,
    make_term,
    ContextEntry,
    Context,
    EMPTY_CONTEXT,
    Substitution,
    SubstitutionError,
    free_indices,
    substitute,
)
from quivers.structural.encoder import (
    BOUND_VAR_OP,
    PerOpMode,
    PerOpFn,
    make_default_op_fn,
    make_default_var_init,
    Encoder,
)
from quivers.structural.decoder import (
    Decoder,
)
from quivers.structural.losses import (
    LossBody,
    LossWeight,
    TrainEnv,
    AttachmentKind,
    LossEntry,
    LossRegistry,
)

__all__ = [
    # signature
    "DataLeaf",
    "TermArg",
    "SortKind",
    "SortVocabEntry",
    "Sort",
    "Constructor",
    "BinderVarSpec",
    "BinderArgSpec",
    "Binder",
    "VertexKind",
    "EdgeKind",
    "Signature",
    "Term",
    "bound_var",
    "make_term",
    "ContextEntry",
    "Context",
    "EMPTY_CONTEXT",
    "Substitution",
    "SubstitutionError",
    "free_indices",
    "substitute",
    # encoder
    "BOUND_VAR_OP",
    "PerOpMode",
    "PerOpFn",
    "make_default_op_fn",
    "make_default_var_init",
    "Encoder",
    # decoder
    "Decoder",
    # losses
    "LossBody",
    "LossWeight",
    "TrainEnv",
    "AttachmentKind",
    "LossEntry",
    "LossRegistry",
]
