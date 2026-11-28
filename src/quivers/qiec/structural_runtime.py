"""Run host-backed structural computations on the reference machine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, lru_cache
from graphlib import TopologicalSorter
from itertools import product
from types import MappingProxyType
from typing import cast
import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, OPAQUE_BY_VALUE
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import (
    LOG_WEIGHT,
    sampleable_type,
    sampled_element,
    tensor_shape,
)
from quivers.qiec.deduction_outcomes import validate_outcome_key
from quivers.qiec.effects import EffectRef, HandlerDef
from quivers.qiec.evaluator import (
    ClauseContext,
    Evaluator,
    RuntimeAttachments,
    RuntimeConstructor,
    RuntimeHandler,
)
from quivers.qiec.execution import (
    ExecutionResult,
    RuntimeConfiguration,
    RuntimeValidator,
    run_named,
)
from quivers.qiec.identifiers import ConstructorId, TypeId
from quivers.qiec.module import NamedEffectInstance, QiecModule
from quivers.qiec.numbers import host_index
from quivers.qiec.rules import (
    Derivation,
    RulePresentation,
    closed_derivation_value,
    presentation_of,
    term_derivation,
)
from quivers.qiec.structured_sampleables import FiniteSampleable, _check_finite_point
from quivers.qiec.terms import Return
from quivers.qiec.types import REAL, TypeApplication, TypeExpr
from quivers.structural import (
    EMPTY_CONTEXT,
    Decoder,
    Encoder,
    Signature,
    Term,
    TermArg,
    make_term,
)


type _DecoderLayout = tuple[
    Signature, int, tuple[tuple[str, tuple[tuple[type, TermArg], ...]], ...]
]


def _finite_term_name(signature: str, sort: str) -> str:
    """Name the complete decoder's nominal value family.

    Parameters
    ----------
    signature, sort
        The structural signature and object sort.

    Returns
    -------
    str
        A qualified name distinct from the opaque structural term family.
    """
    return f"{signature}__{sort}Value"


def _decoder_layout(decoder: Decoder) -> _DecoderLayout:
    """Snapshot the declarations that determine the decoder's support.

    Parameters
    ----------
    decoder
        The decoder, whose learnable tensors are deliberately omitted.

    Returns
    -------
    tuple
        The signature, depth and finite vocabularies in stable order.
    """
    vocabularies = tuple(
        (sort, tuple((type(value), value) for value in vocabulary))
        for sort, vocabulary in sorted(decoder.data_vocab.items())
    )
    return decoder.signature, decoder.depth, vocabularies


@lru_cache(maxsize=128)
def _qualified_structural_presentation(signature: Signature) -> RulePresentation:
    """Qualify a shared term presentation's source names by its signature.

    Parameters
    ----------
    signature
        The structural signature whose families the source module imports.

    Returns
    -------
    RulePresentation
        The original rule and family identities with qualified names.
    """
    original = presentation_of(signature)
    prefix = f"{signature.name}__"
    return original.with_(
        judgments=tuple(
            item.with_(name=_finite_term_name(signature.name, item.name))
            for item in original.judgments
        ),
        rules=tuple(item.with_(name=prefix + item.name) for item in original.rules),
    )


def _decoder_presentation(
    signature: Signature, sort: str, *, qualified: bool
) -> RulePresentation:
    """Require a first-order structural presentation at the chosen sort.

    Parameters
    ----------
    signature, sort
        The decoder's structural signature and root object sort.
    qualified
        Whether the source names carry the signature's prefix.

    Returns
    -------
    RulePresentation
        The structural signature's shared checked declaration.

    Raises
    ------
    ValueError
        If the decoder uses a binding signature or names a non-object sort.
    """
    if signature.binders:
        raise ValueError("finite decoder support requires a first-order signature")
    root = signature.sorts.get(sort)
    if root is None or root.kind != "object":
        raise ValueError(f"decoder support sort {sort!r} is not an object sort")
    return (
        _qualified_structural_presentation(signature)
        if qualified
        else presentation_of(signature)
    )


@dataclass(frozen=True)
class _PreparedDecoderSupport:
    """Static candidates, kept independently of a live neural decoder.

    Parameters
    ----------
    sort, element
        The structural sort and its nominal checked type.
    max_candidates
        The bound used to prepare this support.
    presentation
        The shared checked term declarations.
    candidates
        Pairs of typed run-time values and their source structural terms.
    layout
        The support-defining decoder declarations.
    """

    sort: str
    element: TypeApplication
    max_candidates: int
    presentation: RulePresentation
    candidates: tuple[tuple[RuntimeConstructor, Term], ...]
    layout: _DecoderLayout

    def distribution(self, decoder: Decoder, code: torch.Tensor) -> FiniteSampleable:
        """Score this support with the decoder at the current invocation.

        Parameters
        ----------
        decoder, code
            The same structural decoder configuration and a floating code of
            the declared width, with any live parameters.

        Returns
        -------
        FiniteSampleable
            The fixed typed points with fresh differentiable log weights.

        Raises
        ------
        ValueError
            If the code or decoder configuration does not match the support.
        """
        width = decoder.sort_dims.get(self.sort)
        if width is None:
            raise ValueError(
                f"the decoder has no embedding width for sort {self.sort!r}"
            )
        if (
            not isinstance(code, torch.Tensor)
            or code.shape != (width,)
            or not code.is_floating_point()
        ):
            raise ValueError(
                f"decoder support expects one code of shape ({width},) with floating dtype"
            )
        if _decoder_layout(decoder) != self.layout:
            raise ValueError(
                "the finite decoder support's signature, depth or vocabulary changed"
            )
        values, terms = zip(*self.candidates, strict=True)
        masses = torch.stack(
            [decoder.log_prob(term, code, sort=self.sort) for term in terms]
        )
        return FiniteSampleable(
            element=self.element,
            values=values,
            log_weights=masses,
            registry=self.presentation.registry,
        )


class FiniteDecoderSupport(QiecModel):
    """A decoder restricted to an explicit finite set of closed terms.

    The candidates are checked once against the decoder's structural signature
    and represented by constructors of its ordinary QIEC rule presentation.
    Repeated identical candidates name the same term. Only first-order term
    signatures with a closed discrete QIEC carrier are supported.

    Parameters
    ----------
    decoder
        The decoder whose log probability determines the conditional law.
    sort
        The object sort of the candidates.
    candidates
        The supplied finite support. Every term must be in the decoder's
        domain, including its depth and data-vocabulary restrictions.
    max_candidates
        A positive bound on the number of supplied entries, checked before
        validating terms. Exceeding the bound refuses the entire support.
    qualified
        Use the signature-qualified names of the source-level term carrier.
    """

    decoder: Decoder = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sort: str
    candidates: tuple[Term, ...] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_candidates: int
    qualified: bool = False

    @dx.model_validator()
    def _validate_support(self) -> None:
        """Ensure the candidate set fits the bound and the checked carrier.

        Raises
        ------
        ValueError
            If the set is empty or the bound does not admit all entries.
        KernelError
            If the selected carrier is outside the finite sampleable domain.
        RuleError
            If one term does not inhabit the declared structural signature.
        """
        if type(self.max_candidates) is not int or self.max_candidates < 1:
            raise ValueError(
                "a finite decoder support needs a positive candidate bound"
            )
        if not self.candidates or len(self.candidates) > self.max_candidates:
            raise ValueError(
                f"decoder support must have from 1 to {self.max_candidates} candidate entries"
            )
        _ = self._prepared

    @cached_property
    def _prepared(self) -> _PreparedDecoderSupport:
        """The checked topology, code-independent and reusable between fits.

        Returns
        -------
        _PreparedDecoderSupport
            Typed points and their static declaration, holding no live decoder.
        """
        return _PreparedDecoderSupport(
            sort=self.sort,
            element=self.element,
            max_candidates=self.max_candidates,
            presentation=self.presentation,
            candidates=self._checked_candidates,
            layout=_decoder_layout(self.decoder),
        )

    @cached_property
    def presentation(self) -> RulePresentation:
        """The shared typed rule presentation, for the consuming module.

        Returns
        -------
        RulePresentation
            The declaration of the candidate terms' nominal families.

        Raises
        ------
        ValueError
            If the decoder uses a binding signature or the sort is not an
            object sort in its signature.
        """
        return _decoder_presentation(
            self.decoder.signature, self.sort, qualified=self.qualified
        )

    @classmethod
    def complete(
        cls,
        decoder: Decoder,
        *,
        sort: str,
        max_candidates: int,
        qualified: bool = False,
    ) -> FiniteDecoderSupport:
        """Prepare the complete support of a productive depth-bounded decoder.

            Every constructor the decoder can choose at a reachable sort and
            depth must be productive. Data leaves range over their decoder
            vocabulary, which must be finite, nonempty and free of duplicates.
            The support is counted up to the declared limit before making terms.

            Parameters
            ----------
            decoder
                The first-order decoder and its depth bound.
            sort
                The root object sort to expand.
            max_candidates
                The most terms to materialize. It must be positive.
        qualified
            Whether to qualify the names of the shared term declarations.

            Returns
            -------
            FiniteDecoderSupport
                Every term admitted by the given structural decoder at this root.
                When its local distributions are valid, its normalized law
                agrees with the original decoder.

            Raises
            ------
            ValueError
                If a reachable branch is unproductive, the domain is unsupported,
                or the number of terms exceeds the bound.
        """
        _decoder_presentation(decoder.signature, sort, qualified=qualified)
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError(
                "a finite decoder support needs a positive candidate bound"
            )
        return cls(
            decoder=decoder,
            candidates=_complete_decoder_terms(decoder, sort, max_candidates),
            sort=sort,
            max_candidates=max_candidates,
            qualified=qualified,
        )

    @cached_property
    def element(self) -> TypeApplication:
        """The nominal QIEC type of a point of this support.

        Returns
        -------
        TypeApplication
            The closed family for the selected structural sort.
        """
        families, _ = self.presentation.declarations()
        name = (
            _finite_term_name(self.decoder.signature.name, self.sort)
            if self.qualified
            else self.sort
        )
        family = next(item for item in families if item.name == name)
        return TypeApplication(constructor=family.type_constructor)

    @cached_property
    def _checked_candidates(self) -> tuple[tuple[RuntimeConstructor, Term], ...]:
        """Check and retain each distinct term with its typed representation.

        Returns
        -------
        tuple
            Constructor and structural term pairs in first appearance order.

        Raises
        ------
        ValueError
            If a candidate has a different sort.
        """
        validate_outcome_key(self.element, self.presentation.registry)
        evaluator = Evaluator()
        candidates: dict[RuntimeConstructor, Term] = {}
        for term in self.candidates:
            derivation = term_derivation(self.decoder.signature, term).body
            if not isinstance(derivation, Derivation):
                raise ValueError("a finite decoder candidate must be a closed term")
            value = closed_derivation_value(self.presentation, derivation)
            if value.result_type != self.element:
                raise ValueError(
                    f"a decoder candidate does not inhabit sort {self.sort!r}"
                )
            runtime = evaluator.evaluate(Return(value=value))
            assert isinstance(runtime, RuntimeConstructor)
            candidates.setdefault(runtime, term)
        return tuple(candidates.items())

    def distribution(self, code: torch.Tensor) -> FiniteSampleable:
        """Build the conditional decoder law at one unbatched code.

        Parameters
        ----------
        code
            One embedding of the selected object sort.

        Returns
        -------
        FiniteSampleable
            The same supplied terms with freshly evaluated log masses,
            normalized over this set by the finite sampleable.

        Raises
        ------
        ValueError
            If the code has the wrong shape or the decoder's signature,
            depth or vocabularies have changed.
        RuntimeError
            If a supplied term violates the decoder's depth or vocabulary
            restriction, or the decoder lacks a required component.
        DistributionError
            If the finite law has no positive mass or invalid log masses.
        """
        return self._prepared.distribution(self.decoder, code)


type _DecoderState = tuple[str, int]


def _complete_decoder_terms(
    decoder: Decoder, sort: str, limit: int
) -> tuple[Term, ...]:
    """Count all reachable structural alternatives before materializing terms.

    Parameters
    ----------
    decoder
        The first-order decoder with closed vocabularies.
    sort
        The root sort.
    limit
        The most terms the caller is willing to materialize.

    Returns
    -------
    tuple of Term
        The full root support, in constructor and vocabulary order.

    Raises
    ------
    ValueError
        If a reachable branch can get stuck or the complete support is too large.
    """
    signature = decoder.signature
    root = (sort, decoder.depth)
    dependencies: TopologicalSorter[_DecoderState] = TopologicalSorter()
    productions: dict[
        _DecoderState, tuple[tuple[str, tuple[_DecoderState, ...]], ...]
    ] = {}
    leaves: dict[_DecoderState, tuple[TermArg, ...]] = {}
    pending = [root]
    while pending:
        state = pending.pop()
        if state in productions or state in leaves:
            continue
        current, depth = state
        kind = signature.sorts[current].kind
        if kind == "data":
            words = tuple(decoder.data_vocab.get(current, ()))
            if not words or len(set(words)) != len(words):
                raise ValueError(
                    f"reachable decoder data sort {current!r} needs a nonempty vocabulary "
                    "without duplicates"
                )
            leaves[state] = words
            dependencies.add(state)
            continue
        if kind != "object":
            raise ValueError(
                f"reachable decoder sort {current!r} needs a variable in an empty context"
            )
        try:
            choices = decoder._object_candidates(EMPTY_CONTEXT, current, depth)
        except RuntimeError as error:
            raise ValueError(
                f"decoder sort {current!r} at depth {depth}: {error}"
            ) from error
        rules = tuple(
            (
                name,
                tuple(
                    (child, -1 if signature.sorts[child].kind == "data" else depth - 1)
                    for child in signature.constructors[name].domain
                ),
            )
            for name in choices
        )
        children = tuple(child for _, child_states in rules for child in child_states)
        productions[state] = rules
        dependencies.add(state, *children)
        pending.extend(children)

    order = tuple(dependencies.static_order())
    counts: dict[_DecoderState, int] = {}
    overflow = limit + 1
    for state in order:
        if state in leaves:
            counts[state] = min(overflow, len(leaves[state]))
            continue
        total = 0
        for _, child_states in productions[state]:
            ways = 1
            for child in child_states:
                ways = min(overflow, ways * counts[child])
            total = min(overflow, total + ways)
        counts[state] = total
    if counts[root] > limit:
        raise ValueError(
            f"complete decoder support exceeds the {limit} candidate limit"
        )

    terms = dict(leaves)
    for state in order:
        if state in productions:
            terms[state] = tuple(
                make_term(name, *arguments)
                for name, child_states in productions[state]
                for arguments in product(*(terms[child] for child in child_states))
            )
    # Every production of the checked object sort above constructs a Term.
    return cast(tuple[Term, ...], terms[root])


@dataclass(frozen=True)
class _StructuralTermLayout:
    """The inverse of the shared presentation's first-order value encoding.

    Parameters
    ----------
    element, presentation
        The exact root carrier and the checked structural declarations.
    constructors
        The original constructor names and positional sorts, keyed by the
        same nominal identities used by the qualified presentation.
    """

    element: TypeApplication
    presentation: RulePresentation
    constructors: Mapping[ConstructorId, tuple[str, tuple[str, ...]]]

    def term(self, value: object) -> Term:
        """Restore the source order of the two checked field partitions.

        QIEC places the object premises before the scalar payload fields.
        The signature fixes which source positions each partition occupies.
        Both orders within a partition are preserved, so interleaving by
        that signature is the inverse of value encoding. The argument for a
        whole term follows by induction on its finite tree.

        Parameters
        ----------
        value
            A closed nominal QIEC value from this signature's root carrier.

        Returns
        -------
        Term
            The structural term consumed by the existing encoder.

        Raises
        ------
        DistributionError
            If a root or child constructor, static argument, or scalar
            payload violates the checked declaration.
        ValueError
            If an index payload is negative.
        """
        _check_finite_point(value, self.element, self.presentation.registry)
        assert isinstance(value, RuntimeConstructor)
        pending = [(value, False)]
        results: list[Term] = []
        while pending:
            current, entered = pending.pop()
            name, layout = self.constructors[current.constructor]
            premise_count = layout.count("object")
            if not entered:
                pending.append((current, True))
                for child in reversed(current.fields[:premise_count]):
                    assert isinstance(child, RuntimeConstructor)
                    pending.append((child, False))
                continue
            children = iter(results[-premise_count:] if premise_count else ())
            if premise_count:
                del results[-premise_count:]
            payload = iter(current.fields[premise_count:])
            arguments: list[TermArg] = []
            for kind in layout:
                if kind == "object":
                    arguments.append(next(children))
                    continue
                leaf = next(payload)
                if isinstance(leaf, str):
                    arguments.append(leaf)
                else:
                    literal = host_index(leaf, "a structural encoder's integer payload")
                    if kind == "index" and literal < 0:
                        raise ValueError(
                            "a structural encoder's index payload cannot be negative"
                        )
                    arguments.append(literal)
            results.append(Term(op=name, args=tuple(arguments)))
        (result,) = results
        return result


@lru_cache(maxsize=128)
def _structural_term_layout(
    signature: Signature, element: TypeApplication
) -> _StructuralTermLayout:
    """Resolve a structural encoder's nominal term carrier once per signature.

    Parameters
    ----------
    signature, element
        The compiled source signature and the checked encoder argument type.

    Returns
    -------
    _StructuralTermLayout
        The checked inverse, independent of the live encoder weights.

    Raises
    ------
    ValueError
        If this carrier does not belong to the signature's first-order terms.
    KernelError
        If a reachable payload lacks the supported discrete representation.
    """
    if signature.binders:
        raise ValueError("a typed structural encoder requires a first-order signature")
    presentation = _qualified_structural_presentation(signature)
    families, _ = presentation.declarations()
    if not any(
        element == TypeApplication(constructor=family.type_constructor)
        for family in families
    ):
        raise ValueError(
            "structural encoder term carrier differs from its checked signature"
        )
    validate_outcome_key(element, presentation.registry)
    declarations = presentation_of(signature).rules
    constructors = {
        rule.id: (
            rule.name,
            tuple(
                signature.sorts[sort].kind for sort in signature.domain_of(rule.name)
            ),
        )
        for rule in declarations
    }
    return _StructuralTermLayout(element, presentation, MappingProxyType(constructors))


def _tensor_value(type_: TypeExpr, value: object) -> bool:
    """Return whether a torch tensor has the checked tensor shape.

    Parameters
    ----------
    type_
        The closed QIEC type whose tensor shape supplies the expected rank
        and any literal dimensions.
    value
        The host value to validate.

    Returns
    -------
    bool
        Whether ``value`` is a tensor matching every statically known
        dimension of ``type_``.
    """

    shape = tensor_shape(type_)
    if shape is None or not isinstance(value, torch.Tensor):
        return False
    dimensions = shape[1]
    if value.ndim != len(dimensions):
        return False
    for actual, expected in zip(value.shape, dimensions, strict=True):
        literal = getattr(expected, "value", None)
        if isinstance(literal, int) and actual != literal:
            return False
    return True


class StructuralRuntimeProvider(QiecModel):
    """PyTorch components and validators for one structural invocation.

    Parameters
    ----------
    runtimes
        The host handlers implementing the module's structural computations.
    term_types
        Type identities whose host values are structural :class:`Term`
        instances.
    name
        The provider name recorded in execution results and traces.
    """

    runtimes: tuple[RuntimeHandler, ...] = dx.field(
        opaque=True, extras=OPAQUE_BY_IDENTITY
    )
    term_types: frozenset[TypeId] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    name: str = "structural"

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach each host computation handler.

        Parameters
        ----------
        module
            The checked module, unused because the handlers were selected
            while constructing this provider.
        attachments
            The invocation's attachment table.
        """

        del module
        for runtime in self.runtimes:
            attachments.bind_handler(runtime)

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Return the validator for structural terms and torch values.

        Parameters
        ----------
        type_
            The closed QIEC type a host argument or result must inhabit.

        Returns
        -------
        RuntimeValidator | None
            A predicate for scalar weights, structural terms, tensors, and
            structural products, or ``None`` when the provider does not cover
            ``type_``.
        """

        if type_ == REAL or type_ == LOG_WEIGHT:
            return lambda value: (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                or isinstance(value, torch.Tensor)
                and value.shape == ()
            )
        if _is_term_type(type_, self.term_types):
            return lambda value: isinstance(value, Term)
        if tensor_shape(type_) is not None:
            return lambda value: _tensor_value(type_, value)
        if isinstance(type_, TypeApplication) and type_.constructor.name.startswith(
            "Product"
        ):
            components = tuple(
                item for item in type_.arguments if isinstance(item, TypeApplication)
            )
            validators = tuple(self.validator_for(item) for item in components)
            if len(validators) != len(type_.arguments) or any(
                validator is None for validator in validators
            ):
                return None

            def product(value: object) -> bool:
                """Check a structural product componentwise.

                Parameters
                ----------
                value
                    The host value to validate against the product type.

                Returns
                -------
                bool
                    Whether ``value`` is a tuple whose components pass the
                    corresponding validators.
                """

                return (
                    isinstance(value, tuple)
                    and len(value) == len(validators)
                    and all(
                        validator is not None and validator(item)
                        for validator, item in zip(validators, value, strict=True)
                    )
                )

            return product
        return None


def _is_term_type(type_: TypeExpr, identities: frozenset[TypeId]) -> bool:
    """Return whether a type is one of the module's structural families.

    Parameters
    ----------
    type_
        The closed QIEC type to inspect.
    identities
        Type-constructor identities belonging to structural term families.

    Returns
    -------
    bool
        Whether ``type_`` applies one of those family constructors.
    """

    return isinstance(type_, TypeApplication) and type_.constructor.id in identities


def _validator_for(
    type_: TypeExpr,
    term_types: frozenset[TypeId],
    *,
    decoder_law: bool = False,
) -> RuntimeValidator | None:
    """Return the structural validator for one closed runtime type.

    Parameters
    ----------
    type_
        The closed QIEC type to validate at the host boundary.
    term_types
        Type identities whose host values are structural terms.
    decoder_law
        Whether the requested operation returns a checked finite decoder law.

    Returns
    -------
    RuntimeValidator | None
        The matching predicate, or ``None`` when the structural provider does
        not cover ``type_``.
    """

    if decoder_law:
        element = sampled_element(type_)
        if element is not None:
            return lambda value: (
                isinstance(value, FiniteSampleable) and value.element == element
            )
    provider = StructuralRuntimeProvider(runtimes=(), term_types=term_types)
    return provider.validator_for(type_)


def structural_handler_definition(
    effect: EffectRef,
    result: TypeExpr,
    *,
    instance_name: str,
    computation_name: str,
) -> HandlerDef:
    """Build the stable checked handler signature for one neural attachment.

    Parameters
    ----------
    effect
        The applied ``Compute`` interface served by the attachment.
    result
        The closed result type returned by the host component.
    instance_name
        The source-derived effect-instance name.
    computation_name
        The source-derived structural computation name.

    Returns
    -------
    HandlerDef
        A stable foreign handler declaration for the component.
    """

    runtime = compute_handler(
        lambda argument, _context: argument,
        effect=effect,
        result_validator=lambda _value: True,
        answer_type=result,
        key=f"structural-{instance_name}-{computation_name}",
    )
    return runtime.definition.with_(
        name=f"__structural_{instance_name}_{computation_name}",
    )


def _loss_value(
    program: quivers.program.Program, name: str, argument: object
) -> torch.Tensor:
    """Evaluate one named loss entry under its term argument.

    Parameters
    ----------
    program
        The compiled program containing the structural loss registry.
    name
        The loss entry to evaluate.
    argument
        The structural term supplied to the loss body and optional weight.

    Returns
    -------
    torch.Tensor
        The weighted loss value, retaining its autograd graph.
    """

    registry = program.losses
    entry = next(item for item in registry.entries if item.name == name)
    environment = {"term": argument}
    value = entry.body(environment)
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(float(value))
    if entry.weight is None:
        return value
    weight = entry.weight(environment)
    if not isinstance(weight, torch.Tensor):
        weight = torch.tensor(float(weight))
    return value * weight


def _component_function(
    program: quivers.program.Program,
    instance: NamedEffectInstance,
) -> Callable[[object, ClauseContext], object]:
    """Resolve one structural instance to its compiled PyTorch operation.

    Parameters
    ----------
    program
        The compiled container supplying encoder, decoder, and loss objects.
    instance
        The QIEC effect instance whose source provenance names the component.

    Returns
    -------
    Callable[[object, ClauseContext], object]
        The host operation installed as the instance's ``Compute.run``
        clause.

    Raises
    ------
    KeyError
        If the structural registry has no component named by ``instance``.
    ValueError
        If the instance lacks structural provenance or names an unknown role.
    """

    path = instance.origin.structural_path
    if len(path) < 4 or path[0] != "structural":
        raise ValueError(f"instance {instance.name!r} is not structural")
    role = str(path[1])
    name = str(path[2])
    if role in {"encoder", "encoder-value"}:
        encoder_name = (
            name if role == "encoder" else name.removesuffix("__encode_value")
        )
        encoders = cast(Mapping[str, Callable[[object], object]], program.encoders)
        encoder = encoders[encoder_name]
        if role == "encoder":
            return lambda argument, _context: encoder(argument)
        signatures = getattr(program, "signatures", {})
        carrier = instance.entry.effect.arguments[0]
        signature = signatures.get(path[4]) if len(path) == 5 else None
        if not isinstance(signature, Signature) or not isinstance(
            carrier, TypeApplication
        ):
            raise TypeError("a typed structural encoder needs its checked signature")
        if isinstance(encoder, Encoder) and encoder.signature != signature:
            raise TypeError("the live structural encoder has a different signature")
        layout = _structural_term_layout(signature, carrier)
        return lambda argument, _context: encoder(layout.term(argument))
    if role == "decoder":
        decoder = program.decoders[name]
        return lambda argument, _context: decoder(argument)
    if role == "decoder-law":
        decoder = program.decoders[name.removesuffix("__law")]
        support = getattr(decoder, "_qiec_finite_support", None)
        expected = sampled_element(instance.entry.effect.arguments[1])
        if (
            not isinstance(support, _PreparedDecoderSupport)
            or support.element != expected
        ):
            raise TypeError(
                f"decoder {decoder.name!r} has no complete support of the checked type"
            )

        def law(argument: object, context: ClauseContext) -> FiniteSampleable:
            """Evaluate the prepared support at one checked embedding.

            Parameters
            ----------
            argument
                The embedding and the limit recorded in the checked module.
            context
                The invocation holding the producer’s exact ownership assertion.

            Returns
            -------
            FiniteSampleable
                A freshly scored law with the checked nominal carrier.

            Raises
            ------
            TypeError
                If the surviving runtime argument has the wrong shape or a
                limit different from the prepared support.
            """
            if (
                not isinstance(argument, tuple)
                or len(argument) != 2
                or not isinstance(argument[0], torch.Tensor)
                or host_index(argument[1], "decoder support limit")
                != support.max_candidates
            ):
                raise TypeError(
                    f"decoder {decoder.name!r} does not match the checked support limit"
                )
            result = support.distribution(decoder, argument[0])
            # This producer owns the fresh log-mass tensor and the law only
            # reads it. The validated candidate terms are immutable machine
            # values. Native Random owns the draw effect and this result
            # carries no mutable backend override.
            context.attach(
                result,
                sampleable_type(support.element),
                role="finite-decoder-law",
                duplicable=True,
            )
            return result

        return law
    if role == "decoder-nll":
        decoder_name = name.removesuffix("__nll")
        decoder = program.decoders[decoder_name]

        def nll(argument: object, _context: ClauseContext) -> torch.Tensor:
            """Score one observed term and code as negative log likelihood.

            Parameters
            ----------
            argument
                A ``(term, code)`` pair supplied to the decoder likelihood.
            _context
                The enclosing clause context, unused by this pure host call.

            Returns
            -------
            torch.Tensor
                The decoder's negative log probability, retaining its
                autograd graph.

            Raises
            ------
            TypeError
                If ``argument`` is not a pair.
            """

            if not isinstance(argument, tuple) or len(argument) != 2:
                raise TypeError("decoder negative log likelihood takes (term, code)")
            term, code = argument
            return -decoder.log_prob(term, code)

        return nll
    if role == "loss":
        return lambda argument, _context: _loss_value(program, name, argument)
    raise ValueError(f"unknown structural component role {role!r}")


def structural_runtime_provider(
    module: QiecModule,
    program: quivers.program.Program,
) -> StructuralRuntimeProvider:
    """Build the typed neural-attachment provider for a compiled module.

    Parameters
    ----------
    module
        The checked QIEC module whose structural instances need handlers.
    program
        The compiled container supplying the corresponding PyTorch objects.

    Returns
    -------
    StructuralRuntimeProvider
        A provider containing one validated runtime handler per structural
        instance.

    Raises
    ------
    TypeError
        If a structural component returns a type with no host validator.
    """

    term_types = frozenset(
        family.type_constructor.id
        for family in module.families
        if family.name.endswith("__Term")
    )
    runtimes: list[RuntimeHandler] = []
    for instance in module.instances:
        if instance.origin.structural_path[:1] != ("structural",):
            continue
        result_type = instance.entry.effect.arguments[1]
        assert isinstance(result_type, TypeApplication)
        component_name = str(instance.origin.structural_path[2])
        validator = _validator_for(
            result_type,
            term_types,
            decoder_law=instance.origin.structural_path[1] == "decoder-law",
        )
        if validator is None:
            raise TypeError(
                f"structural component {component_name!r} has no validator for "
                f"{result_type!r}"
            )
        runtime = compute_handler(
            _component_function(program, instance),
            effect=instance.entry.effect,
            result_validator=validator,
            answer_type=result_type,
            key=f"structural-{instance.name}-{component_name}",
        )
        runtime = runtime.with_(
            definition=structural_handler_definition(
                instance.entry.effect,
                result_type,
                instance_name=instance.name,
                computation_name=component_name,
            )
        )
        runtimes.append(runtime)
    return StructuralRuntimeProvider(runtimes=tuple(runtimes), term_types=term_types)


def structural_runtime_configuration(
    module: QiecModule,
    program: quivers.program.Program,
) -> RuntimeConfiguration:
    """Return the standard runtime configuration for structural entries.

    Parameters
    ----------
    module
        The checked QIEC module whose structural entries may be invoked.
    program
        The compiled container supplying their PyTorch attachments.

    Returns
    -------
    RuntimeConfiguration
        A configuration selecting the structural provider when the module has
        structural runtimes, or no providers otherwise.
    """

    provider = structural_runtime_provider(module, program)
    return RuntimeConfiguration(
        providers=(provider,) if provider.runtimes else (),
    )


def run_structural(
    module: QiecModule,
    program: quivers.program.Program,
    name: str,
    arguments: Sequence[object],
    *,
    fuel: int | None = None,
) -> ExecutionResult:
    """Run a structural computation through the compiled PyTorch attachments.

    Parameters
    ----------
    module
        The checked QIEC module.
    program
        The classic compiled container supplying encoders, decoders, losses,
        and their learned tensors.
    name
        The structural computation name.
    arguments
        Its host arguments.
    fuel
        Optional reference-machine step budget.

    Returns
    -------
    ExecutionResult
        The typed result and trace. Torch tensor results retain their autograd
        graph back to the attached component parameters.

    Raises
    ------
    KeyError
        If ``name`` is not a structural computation.
    """

    computation = next(
        (
            item
            for item in module.computations
            if item.name == name and item.origin.structural_path[:1] == ("structural",)
        ),
        None,
    )
    if computation is None:
        raise KeyError(
            f"module {module.module!r} has no structural computation {name!r}"
        )
    return run_named(
        module,
        computation.name,
        tuple(arguments),
        runtime=structural_runtime_configuration(module, program),
        fuel=fuel,
    )


__all__ = [
    "FiniteDecoderSupport",
    "StructuralRuntimeProvider",
    "run_structural",
    "structural_handler_definition",
    "structural_runtime_configuration",
    "structural_runtime_provider",
]

# `quivers.program` imports this module to configure the entry points it
# runs, and the annotations above name its `Program`. The import binds the
# package path rather than the class and sits at the foot of the file, so
# whichever module loads first, the other finds every name it needs, and
# the annotations resolve once both have finished loading.
import quivers.program  # noqa: E402
