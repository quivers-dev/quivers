"""Elaborate structural encoders, decoders, and losses into QIEC.

A structural signature becomes an opaque term family indexed by its declared
sorts. Encoders and decoders become typed computations over that family and
fixed-width real tensors. Their implementation remains a runtime attachment:
the core graph names the input, output, effect instance, and call edges, while
the PyTorch module and its learned tensors stay at the provider boundary.

The reconstruction form used by the term-autoencoder example is lowered as a
call from the loss to the encoder and the decoder's negative-log-likelihood
computation. Other loss bodies remain one typed host computation. Both forms
therefore expose the dependency in the core without serializing a Python
closure or a tensor parameter into the module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import didactic.api as dx

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.dsl import ast_nodes as surface
from quivers.qiec.builtins import COMPUTE_APPLY, COMPUTE_EFFECT
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.checking import ComputationSignature
from quivers.qiec.declarations import FamilyDecl
from quivers.qiec.effects import EffectRequest, EffectRow, HandlerDef, SiteProvenance
from quivers.qiec.identifiers import ComputationId, FamilyId
from quivers.qiec.kinds import NAT, IndexBinder, UserIndexSort
from quivers.qiec.module import NamedComputation, NamedEffectInstance
from quivers.qiec.structural_runtime import structural_handler_definition
from quivers.qiec.terms import Bind, Call, Handle, Perform, Return, TupleValue, Var
from quivers.qiec.types import (
    REAL,
    IndexConstructor,
    IndexLiteral,
    TypeApplication,
    TypeExpr,
    product_type,
)
from quivers.qiec.effects import instantiate_effect
from quivers.qiec.terms import Local

if TYPE_CHECKING:
    from quivers.dsl.qiec_lowering import _Elaborator


class _StructuralSignature(QiecModel):
    """One signature's indexed term family.

    Parameters
    ----------
    declaration
        The source signature.
    sort
        The closed index sort naming its term sorts.
    family
        The opaque family of host-backed terms.
    root
        The principal object sort.
    """

    declaration: surface.SignatureDecl = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    sort: UserIndexSort = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    family: FamilyDecl = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    root: str

    def term(self, sort: str | None = None) -> TypeApplication:
        """Return the family type at one declared sort.

        Parameters
        ----------
        sort
            The sort name, or the principal sort when omitted.

        Returns
        -------
        TypeApplication
            The indexed term type.
        """

        name = self.root if sort is None else sort
        return TypeApplication(
            constructor=self.family.type_constructor,
            arguments=(IndexConstructor(name=name, arguments=(), sort=self.sort),),
        )


class _StructuralComponent(QiecModel):
    """A host-backed computation and its lexical ``Compute`` instance."""

    declaration: surface.EncoderDecl | surface.DecoderDecl | surface.LossDecl = (
        dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    )
    name: str
    role: str
    argument: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    result: TypeExpr = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    instance: NamedEffectInstance = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    handler: HandlerDef = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    signature: ComputationSignature | None = dx.field(
        default=None, opaque=True, extras=OPAQUE_BY_VALUE
    )


def _option_number(options: tuple[surface.OptionEntry, ...], key: str) -> int | None:
    """Read an integral numeric option.

    Parameters
    ----------
    options
        The source option entries.
    key
        The option name.

    Returns
    -------
    int or None
        The value when present and integral.
    """

    for entry in options:
        if entry.key == key and isinstance(entry.value, surface.OptionNumber):
            value = float(entry.value.value)
            return int(value) if value.is_integer() else None
    return None


def _root_sort(declaration: surface.SignatureDecl) -> str:
    """Return a signature's principal object sort."""

    for sort in declaration.sorts:
        if sort.kind == "object":
            return sort.name
    if declaration.sorts:
        return declaration.sorts[0].name
    if declaration.vertex_kinds:
        return declaration.vertex_kinds[0].name
    return "Value"


def _component_dim(
    declaration: surface.EncoderDecl | surface.DecoderDecl,
    signature: surface.SignatureDecl,
    root: str,
) -> int:
    """Resolve the fixed code width visible at the QIEC boundary."""

    for item in declaration.dims:
        if item.sort == root:
            return item.dim
    option_dim = _option_number(declaration.options, "dim")
    if option_dim is not None:
        return option_dim
    for sort in (*signature.sorts, *signature.vertex_kinds):
        if sort.name == root:
            declared = _option_number(sort.options, "dim")
            if declared is not None:
                return declared
    # The classic compiler reports the missing dimension before it asks for
    # the QIEC projection. This fallback keeps direct structural inspection
    # total while giving the unresolved boundary a concrete checked type.
    return 1


def _reconstruction_loss(
    declaration: surface.LossDecl,
) -> tuple[str, str] | None:
    """Recognize ``-Dec.log_prob(term, Enc(term))``.

    Parameters
    ----------
    declaration
        The loss declaration.

    Returns
    -------
    tuple[str, str] or None
        The encoder and decoder names, respectively.
    """

    body = declaration.body
    if not isinstance(body, surface.LetExprUnaryOp) or body.op != "-":
        return None
    method = body.operand
    if not isinstance(method, surface.LetExprMethodCall):
        return None
    if method.method != "log_prob" or len(method.args) != 2:
        return None
    if not isinstance(method.receiver, surface.LetExprVar):
        return None
    term, encoded = method.args
    if not isinstance(term, surface.LetExprVar) or term.name != "term":
        return None
    if not isinstance(encoded, surface.LetExprCall) or len(encoded.args) != 1:
        return None
    encoded_term = encoded.args[0]
    if not isinstance(encoded_term, surface.LetExprVar) or encoded_term.name != "term":
        return None
    return encoded.func, method.receiver.name


class _StructuralElaboration:
    """Mixin implementing the structural QIEC declaration passes."""

    def _declare_structural_declarations(self: _Elaborator) -> None:
        """Declare indexed term families and host-computation instances."""

        signatures: dict[str, _StructuralSignature] = {}
        for declaration in self._items(surface.SignatureDecl):
            names = tuple(
                item.name for item in (*declaration.sorts, *declaration.vertex_kinds)
            ) or ("Value",)
            sort_name = f"{declaration.name}__Sort"
            sort = UserIndexSort(
                name=sort_name, constructors=names, arities=tuple(0 for _ in names)
            )
            family_name = f"{declaration.name}__Term"
            family = FamilyDecl(
                id=FamilyId.derive(self.source.module_name, "family", family_name),
                name=family_name,
                parameters=(),
                indices=(IndexBinder(name="sort", sort=sort, refinable=True),),
                constructors=(),
                closed=False,
            )
            self.index_sorts[sort_name] = sort
            self.families[family_name] = family
            signatures[declaration.name] = _StructuralSignature(
                declaration=declaration,
                sort=sort,
                family=family,
                root=_root_sort(declaration),
            )
        self._structural_signatures = signatures

        declarations = tuple(
            item
            for item in self.statements
            if isinstance(item, surface.EncoderDecl | surface.DecoderDecl)
        )
        if declarations:
            self.effects.setdefault("Compute", COMPUTE_EFFECT)

        components: dict[str, _StructuralComponent] = {}
        for declaration in declarations:
            structural = signatures.get(declaration.signature)
            if structural is None:
                continue
            width = _component_dim(
                declaration,
                structural.declaration,
                structural.root,
            )
            code = tensor_type(REAL, (IndexLiteral(value=width, sort=NAT),))
            if isinstance(declaration, surface.EncoderDecl):
                self._structural_component(
                    components,
                    declaration,
                    declaration.name,
                    "encoder",
                    structural.term(),
                    code,
                )
            else:
                self._structural_component(
                    components,
                    declaration,
                    declaration.name,
                    "decoder",
                    code,
                    structural.term(),
                )
                self._structural_component(
                    components,
                    declaration,
                    f"{declaration.name}__nll",
                    "decoder-nll",
                    product_type(structural.term(), code),
                    LOG_WEIGHT,
                )

        for declaration in self._items(surface.LossDecl):
            recognized = _reconstruction_loss(declaration)
            if recognized is not None and all(
                name in components for name in (recognized[0], f"{recognized[1]}__nll")
            ):
                continue
            if not signatures:
                continue
            structural = next(iter(signatures.values()))
            self.effects.setdefault("Compute", COMPUTE_EFFECT)
            self._structural_component(
                components,
                declaration,
                declaration.name,
                "loss",
                structural.term(),
                LOG_WEIGHT,
            )
        self._structural_components = components

    def _structural_component(
        self: _Elaborator,
        components: dict[str, _StructuralComponent],
        declaration: surface.EncoderDecl | surface.DecoderDecl | surface.LossDecl,
        name: str,
        role: str,
        argument: TypeExpr,
        result: TypeExpr,
    ) -> None:
        """Declare one component's specialized ``Compute`` instance."""

        effect = COMPUTE_EFFECT.apply((argument, result))
        instance_name = f"{name}__compute"
        entry = instantiate_effect(
            effect,
            module=self.source.module_name,
            lexical_path=("structural", role, name, "instance"),
        )
        instance = NamedEffectInstance(
            name=instance_name,
            entry=entry,
            origin=self._origin(
                declaration,
                ("structural", role, name, "instance"),
                "effect-instance",
            ),
        )
        self.instances[instance_name] = instance
        handler = structural_handler_definition(
            instance.entry.effect,
            result,
            instance_name=instance_name,
            computation_name=name,
        )
        self.handlers[handler.name] = handler
        components[name] = _StructuralComponent(
            declaration=declaration,
            name=name,
            role=role,
            argument=argument,
            result=result,
            instance=instance,
            handler=handler,
        )

    def _declare_structural_signatures(self: _Elaborator) -> None:
        """Register every structural computation signature."""

        for component in getattr(self, "_structural_components", {}).values():
            signature = ComputationSignature(
                id=ComputationId.derive(
                    self.source.module_name,
                    "computation",
                    component.name,
                ),
                name=component.name,
                telescope=(),
                parameters=(component.argument,),
                result=component.result,
                effects=EffectRow(),
            )
            self.registry.register_computation(signature)
            self.computation_signatures[component.name] = signature
            self._structural_components[component.name] = component.with_(
                signature=signature
            )

        for declaration in self._items(surface.LossDecl):
            recognized = _reconstruction_loss(declaration)
            if recognized is None:
                continue
            encoder = self._structural_components.get(recognized[0])
            decoder = self._structural_components.get(f"{recognized[1]}__nll")
            if encoder is None or decoder is None:
                continue
            signature = ComputationSignature(
                id=ComputationId.derive(
                    self.source.module_name,
                    "computation",
                    declaration.name,
                ),
                name=declaration.name,
                telescope=(),
                parameters=(encoder.argument,),
                result=LOG_WEIGHT,
                effects=EffectRow(),
            )
            self.registry.register_computation(signature)
            self.computation_signatures[declaration.name] = signature

    def _declare_structural_computations(
        self: _Elaborator,
    ) -> tuple[NamedComputation, ...]:
        """Build the structural component and loss bodies."""

        computations: list[NamedComputation] = []
        for component in getattr(self, "_structural_components", {}).values():
            assert component.signature is not None
            argument = Local(name="argument", type=component.argument)
            request = EffectRequest(
                instance=component.instance.entry.instance,
                effect=component.instance.entry.effect,
                operation=COMPUTE_APPLY,
                static_arguments=(),
                arguments=(Var(local=argument),),
                result_type=component.result,
                origin=SiteProvenance(
                    origin=self._origin(
                        component.declaration,
                        ("structural", component.role, component.name, "apply"),
                        "effect-request",
                    )
                ),
            )
            computations.append(
                self._checked_computation(
                    component.signature,
                    (argument,),
                    Handle(
                        instance=component.instance.entry.instance,
                        handler=component.handler.id,
                        computation=Perform(request=request),
                        static_arguments=(),
                    ),
                    component.declaration,
                    ("structural", component.role, component.name),
                )
            )

        for declaration in self._items(surface.LossDecl):
            recognized = _reconstruction_loss(declaration)
            if recognized is None:
                continue
            encoder = self._structural_components.get(recognized[0])
            decoder = self._structural_components.get(f"{recognized[1]}__nll")
            signature = self.computation_signatures.get(declaration.name)
            if encoder is None or decoder is None or signature is None:
                continue
            assert encoder.signature is not None and decoder.signature is not None
            term = Local(name="term", type=encoder.argument)
            code = Local(name="code", type=encoder.result)
            loss = Local(name="loss", type=LOG_WEIGHT)
            body = Bind(
                binder=code,
                first=Call(
                    callee=encoder.signature.id,
                    name=encoder.name,
                    static_arguments=(),
                    arguments=(Var(local=term),),
                    result_type=encoder.result,
                    effects=encoder.signature.effects,
                    origin=self._origin(
                        declaration,
                        ("structural", "loss", declaration.name, "encoder"),
                        "call",
                    ),
                ),
                then=Bind(
                    binder=loss,
                    first=Call(
                        callee=decoder.signature.id,
                        name=decoder.name,
                        static_arguments=(),
                        arguments=(
                            TupleValue(
                                items=(Var(local=term), Var(local=code)),
                                result_type=decoder.argument,
                            ),
                        ),
                        result_type=LOG_WEIGHT,
                        effects=decoder.signature.effects,
                        origin=self._origin(
                            declaration,
                            ("structural", "loss", declaration.name, "decoder"),
                            "call",
                        ),
                    ),
                    then=Return(value=Var(local=loss)),
                ),
            )
            computations.append(
                self._checked_computation(
                    signature,
                    (term,),
                    body,
                    declaration,
                    ("structural", "loss", declaration.name),
                )
            )
        return tuple(computations)


__all__ = []
