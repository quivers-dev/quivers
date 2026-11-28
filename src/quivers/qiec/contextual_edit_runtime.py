"""Checked native QIEC scores over the existing packed contextual edit plan."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_VALUE
from quivers.core.algebras import LOG_PROB
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import ContextualEditChannel, ContextualEditPlan, Words
from quivers.core.relations.contextual_edit import _validate_words
from quivers.core.relations.queries import _validate_limits
from quivers.qiec._models import QiecModel
from quivers.qiec.builtins import compute_handler
from quivers.qiec.canonical import LOG_WEIGHT, tensor_type
from quivers.qiec.evaluator import ClauseContext, RuntimeAttachments
from quivers.qiec.execution import CoreRuntimeProvider, RuntimeValidator
from quivers.qiec.identifiers import AttachmentId
from quivers.qiec.kinds import NAT
from quivers.qiec.module import QiecModule
from quivers.qiec.relation_runtime import _fixed_compute_definition
from quivers.qiec.types import IndexLiteral, TypeExpr


class FixedContextualEditRequest(QiecModel):
    """Request one packed batch of alignment-marginal conditional scores.

    The request carries nominal alphabets, paired fixed observations,
    the insertion bound and a stable channel attachment. The caller
    supplies the existing channel with its original live scorer. Each
    result entry is the log probability of the corresponding output
    under the channel conditioned on that entry's complete input.

    Parameters
    ----------
    channel : AttachmentId
        The stable runtime attachment for the original model.
    input_alphabet, output_alphabet : tuple of str and int
        The nominal alphabet names and sizes in their respective roles.
    inputs, outputs : Words
        Nonempty batches containing one input and output per position.
    max_insertions : int
        The channel's maximum number of consecutive insertions.
    max_states : int
        The total permitted observation configurations across the batch.
    max_work : int
        The total permitted padded action cells and compatible edit arcs.
    """

    channel: AttachmentId
    input_alphabet: tuple[str, int] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    output_alphabet: tuple[str, int] = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    inputs: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    outputs: Words = dx.field(opaque=True, extras=OPAQUE_BY_VALUE)
    max_insertions: int
    max_states: int
    max_work: int

    @dx.model_validator()
    def _validate_request(self) -> None:
        """Check finite nominal alphabets, aligned words, and limits.

        Raises
        ------
        ValueError
            A tape, word, batch shape, insertion or resource limit is invalid.
        """
        _validate_limits(self.max_states, self.max_work)
        if type(self.max_insertions) is not int or self.max_insertions < 0:
            raise ValueError("contextual edit requires a nonnegative insertion limit")
        if not self.inputs or len(self.inputs) != len(self.outputs):
            raise ValueError(
                "contextual edit requires a nonempty batch of paired observations"
            )
        for (name, size), words, role in (
            (self.input_alphabet, self.inputs, "input"),
            (self.output_alphabet, self.outputs, "output"),
        ):
            if (
                not isinstance(name, str)
                or not name
                or type(size) is not int
                or size < 0
            ):
                raise ValueError(f"invalid contextual edit {role} alphabet")
            _validate_words(
                words, StringSet(alphabet=FinSet(name=name, cardinality=size)), role
            )

    @property
    def handler_name(self) -> str:
        """Return the checked foreign name for this channel and batch.

        Returns
        -------
        str
            A digest binding the channel, nominal alphabets, ordered
            observations and insertion bound. Limits are invocation settings.
        """
        from quivers.qiec.serialization import dumps

        owner = (
            self.channel,
            self.input_alphabet,
            self.output_alphabet,
            self.inputs,
            self.outputs,
            self.max_insertions,
        )
        digest = sha256(dumps(owner).encode("utf-8")).hexdigest()
        return f"__contextual_edit_likelihood_{digest}"

    @property
    def result_type(self) -> TypeExpr:
        """Return the precise batch type of the conditional log scores.

        Returns
        -------
        TypeExpr
            `Tensor[LogWeight]` with exactly one entry per pair.
        """
        return tensor_type(
            LOG_WEIGHT, (IndexLiteral(value=len(self.inputs), sort=NAT),)
        )

    def plan(
        self, channels: Mapping[AttachmentId, ContextualEditChannel]
    ) -> ContextualEditPlan:
        """Bind one original channel and construct one reusable packed plan.

        Parameters
        ----------
        channels : Mapping of AttachmentId to ContextualEditChannel
            Live channels supplied by the reference runtime caller.

        Returns
        -------
        ContextualEditPlan
            The bounded plan sharing the supplied channel's live scorer.

        Raises
        ------
        TypeError
            The mapped runtime value is outside the contextual channel type.
        ValueError
            The requested channel is missing, a nominal tape or insertion
            bound differs, or a construction budget is exhausted.
        """
        if not isinstance(channels, Mapping):
            raise TypeError(
                "contextual edit requires an explicit channel attachment mapping"
            )
        if self.channel not in channels:
            raise ValueError("missing original contextual edit channel attachment")
        channel = channels[self.channel]
        if not isinstance(channel, ContextualEditChannel):
            raise TypeError("contextual edit requires its original channel type")
        input_ = channel.input_tape.alphabet
        output = channel.output_tape.alphabet
        if (
            (input_.name, input_.cardinality) != self.input_alphabet
            or (output.name, output.cardinality) != self.output_alphabet
            or channel.max_insertions != self.max_insertions
        ):
            raise ValueError(
                "contextual edit channel has different nominal tapes or insertion bound"
            )
        return channel.plan(
            self.inputs,
            self.outputs,
            max_states=self.max_states,
            max_work=self.max_work,
        )


class ContextualEditRuntimeProvider:
    """Bind an authored packed score to the original contextual edit channel.

    Parameters
    ----------
    request : FixedContextualEditRequest
        The typed batch and limits carried by checked QIEC data.
    channels : Mapping of AttachmentId to ContextualEditChannel
        Original process-local channels indexed by attachment identity.
    """

    name = "contextual-edit"

    def __init__(
        self,
        request: FixedContextualEditRequest,
        channels: Mapping[AttachmentId, ContextualEditChannel],
    ) -> None:
        self._handler_name = request.handler_name
        self._result_type = request.result_type
        self._plan = request.plan(channels)

    def _evaluate(self, argument: object, _context: ClauseContext) -> torch.Tensor:
        """Score the packed batch through its current network parameters.

        Parameters
        ----------
        argument : object
            The evaluated unit argument of the checked Compute operation.
        _context : ClauseContext
            The reference machine's clause bindings.

        Returns
        -------
        torch.Tensor
            The ordered conditional log scores with live gradients.

        Raises
        ------
        TypeError
            The machine supplied an argument outside the unit carrier.
        """
        if argument is not None:
            raise TypeError("a fixed contextual edit score takes a unit argument")
        return self._plan.log_prob()

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Attach to the exact foreign handler declared by the checked program.

        Parameters
        ----------
        module : QiecModule
            The checked module with the original channel and batch identity.
        attachments : RuntimeAttachments
            The reference machine's process-local bindings.

        Raises
        ------
        ValueError
            The module lacks the required checked handler name.
        TypeError
            The declaration lacks the exact batch and Compute contract.
        """
        result = self._result_type
        definition = _fixed_compute_definition(module, self._handler_name, result)
        validation = self.validator_for(result)
        assert validation is not None
        runtime = compute_handler(
            self._evaluate,
            effect=definition.effect,
            result_validator=validation,
            answer_type=result,
            key=self._handler_name,
        )
        attachments.bind_handler(runtime.with_(definition=definition))

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Validate the exact ordered batch and its log carrier.

        Parameters
        ----------
        type_ : TypeExpr
            The checked type requested by the runtime.

        Returns
        -------
        RuntimeValidator or None
            The validator for this batch type, including log zero, or no
            validation contribution for other types.
        """
        if type_ != self._result_type:
            return None
        core = CoreRuntimeProvider().validator_for(type_)
        assert core is not None
        return lambda value: (
            isinstance(value, torch.Tensor)
            and core(value)
            and bool(LOG_PROB.contains(value).all())
        )


__all__ = ["ContextualEditRuntimeProvider", "FixedContextualEditRequest"]
