"""Reference operational tests for QIEC handlers and built-in effects."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from quivers.qiec.checking import KernelError, KernelRegistry, infer_computation
from quivers.qiec.builtins import (
    ABORT_ABORT,
    ABORT_EFFECT,
    CHOOSE,
    CHOOSE_CHOOSE,
    CHOICES_A,
    LOG_WEIGHT,
    RANDOM,
    RANDOM_EFFECT,
    RANDOM_SAMPLE,
    SAMPLEABLE_CONSTRUCTOR,
    SCORE,
    SCORE_ADD,
    SCORE_EFFECT,
    SITE_CONSTRUCTOR,
    STATE_EFFECT,
    STATE_GET,
    STATE_PUT,
    WEIGHT_ADD,
    WEIGHT_EFFECT,
    Aborted,
    ExtraValuePolicy,
    ReplayPolicy,
    TraceRecorder,
    abort_handler,
    choose_handler,
    condition_handler,
    draw_handler,
    replay_handler,
    score_handler,
    state_effect,
    state_handler,
    trace_handler,
    weight_handler,
)
from quivers.qiec.effects import (
    EffectRequest,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
)
from quivers.qiec.evaluator import (
    ClauseComputation,
    Evaluator,
    HandlerManifest,
    InvalidHandlerError,
    MissingAttachmentError,
    NonDuplicableContinuationError,
    ResumptionUsageError,
    RuntimeAttachments,
    RuntimeClause,
    RuntimeHandler,
    RuntimeTypeMismatch,
    UnhandledEffectError,
)
from quivers.qiec.identifiers import (
    AttachmentId,
    ConstructorId,
    EffectInstanceId,
    HandlerId,
    SiteProvenance,
    SourceOrigin,
    StaticScopeId,
)
from quivers.qiec.terms import (
    AttachmentRef,
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    ConstructorValue,
    Handle,
    LiteralValue,
    Local,
    Perform,
    Return,
    Var,
)
from quivers.qiec.types import (
    INT,
    STRING,
    UNIT,
    TypeApplication,
)

INT_SITE = TypeApplication(constructor=SITE_CONSTRUCTOR, arguments=(INT,))
INT_SAMPLEABLE = TypeApplication(constructor=SAMPLEABLE_CONSTRUCTOR, arguments=(INT,))


def _origin(index: int, role: str = "test") -> SiteProvenance:
    return SiteProvenance(
        origin=SourceOrigin(
            module="test",
            structural_path=("body", index),
            role=role,
            source_protocol="qvr-source/test",
            file="test.qvr",
            line=index + 1,
            column=1,
        )
    )


def _request(
    instance: EffectInstanceId,
    effect,
    operation,
    arguments,
    result_type,
    *,
    static_arguments=(),
    index: int = 0,
) -> EffectRequest:
    return EffectRequest(
        instance=instance,
        effect=effect,
        operation=operation,
        static_arguments=tuple(static_arguments),
        arguments=tuple(arguments),
        result_type=result_type,
        origin=_origin(index, effect.name.lower()),
    )


def _instance(name: str) -> EffectInstanceId:
    return EffectInstanceId.derive("test", name)


def _attach_handler(
    attachments: RuntimeAttachments,
    handler: RuntimeHandler,
) -> HandlerId:
    return attachments.bind_handler(handler)


def _int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _multiply(left: object, right: object) -> object:
    assert _int(left) and _int(right)
    return left * right  # type: ignore[operator]


def _tracked_resource_handler(
    key: str,
) -> tuple[RuntimeHandler, list[str]]:
    """Build a per-installation resource with exact-once lifecycle assertions."""
    definition = HandlerDef(
        id=HandlerId.derive("test", "tracked-resource", key),
        name=f"tracked-resource-{key}",
        effect=state_effect(INT),
        clauses=(
            HandlerClauseDef(operation=STATE_GET, grade=ResumptionGrade.LINEAR),
            HandlerClauseDef(operation=STATE_PUT, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    finalized: list[str] = []

    def make_context() -> RuntimeHandler:
        done = [False]

        def get(_request_value, resume, _context):
            return resume(0)

        def put(_request_value, resume, _context):
            return resume(None)

        def finish(mode: str) -> None:
            assert not done[0]
            done[0] = True
            finalized.append(mode)

        return RuntimeHandler(
            definition=definition,
            clauses={
                STATE_GET: RuntimeClause(invoke=get, result_validator=_int),
                STATE_PUT: RuntimeClause(
                    invoke=put, result_validator=lambda value: value is None
                ),
            },
            input_validator=_int,
            output_validator=_int,
            mutable_context=True,
            on_exit=lambda: finish("exit"),
            on_drop=lambda: finish("drop"),
        )

    prototype = make_context()
    prototype = prototype.with_(context_factory=make_context)
    return prototype, finalized


def test_return_bind_and_case_have_one_structural_semantics() -> None:
    local = Local(name="x", type=INT)
    bound = Bind(
        binder=local,
        first=Return(value=LiteralValue(value=2, type=INT)),
        then=Return(value=Var(local=local)),
    )
    assert Evaluator().evaluate(bound) == 2

    constructor = ConstructorId.derive("test", "int")
    field = Local(name="field", type=INT)
    scrutinee = ConstructorValue(
        constructor=constructor,
        static_arguments=(),
        fields=(LiteralValue(value=7, type=INT),),
        result_type=INT,
    )
    case = Case(
        scrutinee=scrutinee,
        motive=CaseMotive(indices=(), result_type=INT),
        branches=(
            CaseBranch(
                constructor=constructor,
                static_arguments=(),
                fields=(field,),
                body=Return(value=Var(local=field)),
                scope=StaticScopeId.derive("test", "runtime-case"),
            ),
        ),
    )
    assert Evaluator().evaluate(case) == 7


def test_attachment_boundary_checks_identity_type_and_value() -> None:
    attachments = RuntimeAttachments()
    id = AttachmentId.derive("test", "integer")
    ref = attachments.bind_value(id, 3, INT, validator=_int, duplicable=True)
    assert Evaluator(attachments).evaluate(Return(value=ref)) == 3

    wrong_type = AttachmentRef(attachment=id, type=STRING)
    with pytest.raises(RuntimeTypeMismatch, match="core reference expects"):
        Evaluator(attachments).evaluate(Return(value=wrong_type))
    with pytest.raises(MissingAttachmentError):
        Evaluator().evaluate(Return(value=ref))


def test_deep_resumption_reinstalls_the_matching_handler() -> None:
    instance = _instance("deep-random")
    attachments = RuntimeAttachments()
    calls: list[object] = []
    definition = HandlerDef(
        id=HandlerId.derive("test", "deep"),
        name="deep",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )

    def clause(request, resume, _context):
        calls.append(request.address)
        return resume(len(calls))

    handler = RuntimeHandler(
        definition=definition,
        clauses={RANDOM_SAMPLE: RuntimeClause(invoke=clause, result_validator=_int)},
        input_validator=_int,
        output_validator=_int,
        duplicable_context=True,
    )
    _attach_handler(attachments, handler)
    first = _request(instance, RANDOM, RANDOM_SAMPLE, (), INT, index=1)
    second = _request(instance, RANDOM, RANDOM_SAMPLE, (), INT, index=2)
    ignored = Local(name="ignored", type=INT)
    computation = Handle(
        instance=instance,
        handler=definition.id,
        computation=Bind(
            binder=ignored, first=Perform(request=first), then=Perform(request=second)
        ),
    )

    assert Evaluator(attachments).evaluate(computation) == 2
    assert len(calls) == 2


def test_clause_body_effects_run_only_in_the_outer_context() -> None:
    random_instance = _instance("clause-random")
    score_instance = _instance("clause-score")
    attachments = RuntimeAttachments()
    scorer, accumulator = score_handler(answer_type=INT, key="clause-score")
    _attach_handler(attachments, scorer)

    definition = HandlerDef(
        id=HandlerId.derive("test", "clause-outside"),
        name="clause-outside",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )

    def clause(_request_value, resume, context):
        weight = context.attach(
            2.5,
            LOG_WEIGHT,
            role="outer-score",
            validator=lambda value: isinstance(value, float),
        )
        score = _request(
            score_instance,
            SCORE,
            SCORE_ADD,
            (weight,),
            UNIT,
            index=4,
        )
        assert context.evaluate(Perform(request=score)) is None
        return resume(9)

    random_handler = RuntimeHandler(
        definition=definition,
        clauses={RANDOM_SAMPLE: RuntimeClause(invoke=clause, result_validator=_int)},
        input_validator=_int,
        output_validator=_int,
        duplicable_context=True,
    )
    _attach_handler(attachments, random_handler)
    sample = _request(random_instance, RANDOM, RANDOM_SAMPLE, (), INT, index=3)
    computation = Handle(
        instance=score_instance,
        handler=scorer.definition.id,
        computation=Handle(
            instance=random_instance,
            handler=definition.id,
            computation=Perform(request=sample),
        ),
    )

    assert Evaluator(attachments).evaluate(computation) == 9
    assert accumulator.contributions == [2.5]


@pytest.mark.parametrize(
    ("fail_locally", "expected_finalization"),
    [(False, ["exit"]), (True, ["drop"])],
)
def test_clause_context_evaluate_finalizes_locally_installed_handlers(
    fail_locally: bool,
    expected_finalization: list[str],
) -> None:
    random_instance = _instance(f"clause-resource-random-{fail_locally}")
    resource_instance = _instance(f"clause-resource-state-{fail_locally}")
    unhandled_instance = _instance(f"clause-resource-unhandled-{fail_locally}")
    attachments = RuntimeAttachments()
    resource, finalized = _tracked_resource_handler(f"clause-evaluate-{fail_locally}")
    definition = HandlerDef(
        id=HandlerId.derive("test", "clause-evaluate", fail_locally),
        name="clause-evaluate",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    unhandled = _request(
        unhandled_instance,
        RANDOM,
        RANDOM_SAMPLE,
        (),
        INT,
        index=41,
    )

    def clause(_request_value, resume, context):
        body = (
            Perform(request=unhandled)
            if fail_locally
            else Return(value=LiteralValue(value=1, type=INT))
        )
        local = Handle(
            instance=resource_instance, handler=resource.definition.id, computation=body
        )
        if fail_locally:
            with pytest.raises(UnhandledEffectError):
                context.evaluate(local)
        else:
            assert context.evaluate(local) == 1
        return resume(7)

    handler = RuntimeHandler(
        definition=definition,
        clauses={RANDOM_SAMPLE: RuntimeClause(invoke=clause, result_validator=_int)},
        input_validator=_int,
        output_validator=_int,
    )
    _attach_handler(attachments, resource)
    _attach_handler(attachments, handler)
    request = _request(
        random_instance,
        RANDOM,
        RANDOM_SAMPLE,
        (),
        INT,
        index=42,
    )
    computation = Handle(
        instance=random_instance,
        handler=definition.id,
        computation=Perform(request=request),
    )

    assert Evaluator(attachments).evaluate(computation) == 7
    assert finalized == expected_finalization


@pytest.mark.parametrize(
    ("fail_locally", "expected_finalization"),
    [(False, ["exit"]), (True, ["drop"])],
)
def test_clause_computation_finalizes_locally_installed_handlers(
    fail_locally: bool,
    expected_finalization: list[str],
) -> None:
    random_instance = _instance(f"clause-answer-random-{fail_locally}")
    resource_instance = _instance(f"clause-answer-state-{fail_locally}")
    attachments = RuntimeAttachments()
    resource, finalized = _tracked_resource_handler(
        f"clause-computation-{fail_locally}"
    )
    definition = HandlerDef(
        id=HandlerId.derive("test", "clause-computation", fail_locally),
        name="clause-computation",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.ZERO),
        ),
        input_type=INT,
        output_type=INT,
    )
    body = (
        Perform(
            request=_request(
                _instance(f"clause-answer-unhandled-{fail_locally}"),
                RANDOM,
                RANDOM_SAMPLE,
                (),
                INT,
                index=43,
            )
        )
        if fail_locally
        else Return(value=LiteralValue(value=3, type=INT))
    )

    def clause(_request_value, _resume, _context):
        return ClauseComputation(
            computation=Handle(
                instance=resource_instance,
                handler=resource.definition.id,
                computation=body,
            )
        )

    handler = RuntimeHandler(
        definition=definition,
        clauses={RANDOM_SAMPLE: RuntimeClause(invoke=clause, result_validator=_int)},
        output_validator=_int,
    )
    _attach_handler(attachments, resource)
    _attach_handler(attachments, handler)
    request = _request(
        random_instance,
        RANDOM,
        RANDOM_SAMPLE,
        (),
        INT,
        index=44,
    )
    computation = Handle(
        instance=random_instance,
        handler=definition.id,
        computation=Perform(request=request),
    )
    evaluator = Evaluator(attachments)
    if fail_locally:
        with pytest.raises(UnhandledEffectError):
            evaluator.evaluate(computation)
    else:
        assert evaluator.evaluate(computation) == 3
    assert finalized == expected_finalization


@pytest.mark.parametrize(
    ("fail_after_resume", "expected_finalization"),
    [(False, ["exit"]), (True, ["drop"])],
)
def test_resumption_finalizes_handlers_installed_after_execution_starts(
    fail_after_resume: bool,
    expected_finalization: list[str],
) -> None:
    random_instance = _instance(f"resume-resource-random-{fail_after_resume}")
    resource_instance = _instance(f"resume-resource-state-{fail_after_resume}")
    attachments = RuntimeAttachments()
    resource, finalized = _tracked_resource_handler(
        f"resumed-continuation-{fail_after_resume}"
    )
    definition = HandlerDef(
        id=HandlerId.derive("test", "resume-resource", fail_after_resume),
        name="resume-resource",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )

    def clause(_request_value, resume, _context):
        return resume(7)

    handler = RuntimeHandler(
        definition=definition,
        clauses={RANDOM_SAMPLE: RuntimeClause(invoke=clause, result_validator=_int)},
        input_validator=_int,
        output_validator=_int,
    )
    _attach_handler(attachments, resource)
    _attach_handler(attachments, handler)
    request = _request(
        random_instance,
        RANDOM,
        RANDOM_SAMPLE,
        (),
        INT,
        index=45,
    )
    selected = Local(name="resumed_resource_value", type=INT)
    if fail_after_resume:
        resource_body = Perform(
            request=_request(
                _instance("resume-resource-unhandled"),
                RANDOM,
                RANDOM_SAMPLE,
                (),
                INT,
                index=46,
            )
        )
    else:
        resource_body = Return(value=Var(local=selected))
    continuation = Bind(
        binder=selected,
        first=Perform(request=request),
        then=Handle(
            instance=resource_instance,
            handler=resource.definition.id,
            computation=resource_body,
        ),
    )
    computation = Handle(
        instance=random_instance, handler=definition.id, computation=continuation
    )
    evaluator = Evaluator(attachments)
    if fail_after_resume:
        with pytest.raises(UnhandledEffectError):
            evaluator.evaluate(computation)
    else:
        assert evaluator.evaluate(computation) == 7
    assert finalized == expected_finalization


def test_clause_body_cannot_recapture_itself_but_can_reach_outer_instance_handler() -> (
    None
):
    instance = _instance("same-instance-clause")
    attachments = RuntimeAttachments()
    events: list[str] = []
    outer_definition = HandlerDef(
        id=HandlerId.derive("test", "outer-same-instance"),
        name="outer-same-instance",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    inner_definition = HandlerDef(
        id=HandlerId.derive("test", "inner-same-instance"),
        name="inner-same-instance",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )

    def outer_clause(_request_value, resume, _context):
        events.append("outer")
        return resume(41)

    generated = _request(instance, RANDOM, RANDOM_SAMPLE, (), INT, index=21)

    def inner_clause(_request_value, resume, context):
        events.append("inner")
        assert context.evaluate(Perform(request=generated)) == 41
        return resume(9)

    outer = RuntimeHandler(
        definition=outer_definition,
        clauses={
            RANDOM_SAMPLE: RuntimeClause(invoke=outer_clause, result_validator=_int)
        },
        input_validator=_int,
        output_validator=_int,
    )
    inner = RuntimeHandler(
        definition=inner_definition,
        clauses={
            RANDOM_SAMPLE: RuntimeClause(invoke=inner_clause, result_validator=_int)
        },
        input_validator=_int,
        output_validator=_int,
    )
    _attach_handler(attachments, outer)
    _attach_handler(attachments, inner)
    original = _request(instance, RANDOM, RANDOM_SAMPLE, (), INT, index=22)
    computation = Handle(
        instance=instance,
        handler=outer_definition.id,
        computation=Handle(
            instance=instance,
            handler=inner_definition.id,
            computation=Perform(request=original),
        ),
    )

    assert Evaluator(attachments).evaluate(computation) == 9
    assert events == ["inner", "outer"]


@dataclass
class _Sampleable:
    draw: int

    def rsample(self) -> int:
        return self.draw

    def log_prob(self, value: int) -> float:
        return -abs(value - self.draw) - 0.25


def _random_request(
    attachments: RuntimeAttachments,
    instance: EffectInstanceId,
    *,
    site: str = "z",
    sampleable: _Sampleable | None = None,
    index: int = 0,
) -> EffectRequest:
    distribution = sampleable or _Sampleable(7)
    attachment = AttachmentId.derive("test", "sampleable", site, index)
    sampleable_ref = attachments.bind_value(
        attachment,
        distribution,
        INT_SAMPLEABLE,
        validator=lambda value: isinstance(value, _Sampleable),
        duplicable=True,
    )
    site_ref = attachments.bind_value(
        AttachmentId.derive("test", "site", site, index),
        site,
        INT_SITE,
        validator=lambda value: isinstance(value, str),
        duplicable=True,
    )
    return _request(
        instance,
        RANDOM,
        RANDOM_SAMPLE,
        (site_ref, sampleable_ref),
        INT,
        static_arguments=(INT,),
        index=index,
    )


def test_forwarding_trace_observes_draw_reply_in_lexical_order() -> None:
    instance = _instance("traced-random")
    attachments = RuntimeAttachments()
    draw = draw_handler(_int, answer_type=INT, key="traced-draw")
    recorder = TraceRecorder()
    trace = trace_handler(
        RANDOM,
        (RANDOM_SAMPLE,),
        recorder,
        validators={RANDOM_SAMPLE: _int},
        answer_type=INT,
        key="traced-forward",
    )
    _attach_handler(attachments, draw)
    _attach_handler(attachments, trace)
    request = _random_request(attachments, instance, index=5)
    computation = Handle(
        instance=instance,
        handler=draw.definition.id,
        computation=Handle(
            instance=instance,
            handler=trace.definition.id,
            computation=Perform(request=request),
        ),
    )

    assert Evaluator(attachments).evaluate(computation) == 7
    assert [event.result for event in recorder.events] == [7]
    assert recorder.events[0].address[2] == ()


def test_condition_supplies_value_and_emits_explicit_score() -> None:
    random_instance = _instance("condition-random")
    score_instance = _instance("condition-score")
    attachments = RuntimeAttachments()
    draw = draw_handler(_int, answer_type=INT, key="condition-draw")
    condition = condition_handler(
        {"z": 5},
        score_instance=score_instance,
        result_validator=_int,
        answer_type=INT,
        key="condition",
    )
    score, accumulator = score_handler(answer_type=INT, key="condition-score")
    for handler in (draw, condition, score):
        _attach_handler(attachments, handler)
    request = _random_request(attachments, random_instance, index=6)
    computation = Handle(
        instance=score_instance,
        handler=score.definition.id,
        computation=Handle(
            instance=random_instance,
            handler=draw.definition.id,
            computation=Handle(
                instance=random_instance,
                handler=condition.definition.id,
                computation=Perform(request=request),
            ),
        ),
    )

    evaluator = Evaluator(attachments)
    assert evaluator.evaluate(computation) == 5
    # Generated Score attachments are evaluation-local, so an evaluator and
    # its handler state can be reused without aliasing the first run's value.
    assert evaluator.evaluate(computation) == 5
    assert accumulator.contributions == [-2.25]

    # A fresh dynamic installation must not inherit the previous run's seen
    # set; otherwise a now-unused observation would be silently accepted.
    empty_body = Handle(
        instance=random_instance,
        handler=condition.definition.id,
        computation=Return(value=LiteralValue(value=0, type=INT)),
    )
    with pytest.raises(InvalidHandlerError, match="unused observations"):
        evaluator.evaluate(empty_body)

    registry = KernelRegistry()
    registry.register_effect(RANDOM_EFFECT)
    registry.register_effect(SCORE_EFFECT)
    for handler in (draw, condition, score):
        registry.register_handler(handler.definition)
    checked = infer_computation(computation, registry)
    assert checked.effects.entries == ()
    assert checked.result == INT


def test_condition_rejects_unused_observation_names() -> None:
    random_instance = _instance("extra-random")
    score_instance = _instance("extra-score")
    attachments = RuntimeAttachments()
    draw = draw_handler(_int, answer_type=INT, key="extra-draw")
    condition = condition_handler(
        {"other": 3},
        score_instance=score_instance,
        result_validator=_int,
        extra=ExtraValuePolicy.ERROR,
        answer_type=INT,
        key="extra-condition",
    )
    for handler in (draw, condition):
        _attach_handler(attachments, handler)
    request = _random_request(attachments, random_instance, index=7)
    computation = Handle(
        instance=random_instance,
        handler=draw.definition.id,
        computation=Handle(
            instance=random_instance,
            handler=condition.definition.id,
            computation=Perform(request=request),
        ),
    )
    with pytest.raises(InvalidHandlerError, match="unused observations"):
        Evaluator(attachments).evaluate(computation)


def test_replay_policies_keep_scoring_explicit() -> None:
    instance = _instance("replay-random")
    attachments = RuntimeAttachments()
    draw = draw_handler(_int, answer_type=INT, key="replay-draw")
    replay = replay_handler(
        {"z": 11},
        result_validator=_int,
        policy=ReplayPolicy.INTERVENE,
        answer_type=INT,
        key="replay-intervene",
    )
    _attach_handler(attachments, draw)
    _attach_handler(attachments, replay)
    request = _random_request(attachments, instance, index=8)
    computation = Handle(
        instance=instance,
        handler=draw.definition.id,
        computation=Handle(
            instance=instance,
            handler=replay.definition.id,
            computation=Perform(request=request),
        ),
    )
    evaluator = Evaluator(attachments)
    assert evaluator.evaluate(computation) == 11
    empty_body = Handle(
        instance=instance,
        handler=replay.definition.id,
        computation=Return(value=LiteralValue(value=0, type=INT)),
    )
    with pytest.raises(InvalidHandlerError, match="unused replay values"):
        evaluator.evaluate(empty_body)


def test_condition_and_replay_bookkeeping_is_per_dynamic_installation() -> None:
    outer_random = _instance("nested-random-outer")
    inner_random = _instance("nested-random-inner")
    score_instance = _instance("nested-condition-score")
    attachments = RuntimeAttachments()
    condition = condition_handler(
        {"z": 5},
        score_instance=score_instance,
        result_validator=_int,
        answer_type=INT,
        key="nested-condition",
    )
    replay = replay_handler(
        {"z": 5},
        result_validator=_int,
        policy=ReplayPolicy.INTERVENE,
        answer_type=INT,
        key="nested-replay",
    )
    score, _accumulator = score_handler(
        answer_type=INT,
        key="nested-condition-score",
    )
    for handler in (condition, replay, score):
        _attach_handler(attachments, handler)

    condition_request = _random_request(
        attachments,
        inner_random,
        index=27,
    )
    nested_condition = Handle(
        instance=score_instance,
        handler=score.definition.id,
        computation=Handle(
            instance=outer_random,
            handler=condition.definition.id,
            computation=Handle(
                instance=inner_random,
                handler=condition.definition.id,
                computation=Perform(request=condition_request),
            ),
        ),
    )
    with pytest.raises(InvalidHandlerError, match="unused observations"):
        Evaluator(attachments).evaluate(nested_condition)

    replay_request = _random_request(attachments, inner_random, index=28)
    nested_replay = Handle(
        instance=outer_random,
        handler=replay.definition.id,
        computation=Handle(
            instance=inner_random,
            handler=replay.definition.id,
            computation=Perform(request=replay_request),
        ),
    )
    with pytest.raises(InvalidHandlerError, match="unused replay values"):
        Evaluator(attachments).evaluate(nested_replay)


def test_state_get_put_and_final_state() -> None:
    instance = _instance("state")
    attachments = RuntimeAttachments()
    handler, cell = state_handler(
        1,
        _int,
        state_type=INT,
        expose_final=True,
        answer_type=INT,
        answer_validator=_int,
    )
    _attach_handler(attachments, handler)
    applied_state = state_effect(INT)
    get = _request(
        instance,
        applied_state,
        STATE_GET,
        (),
        INT,
        index=9,
    )
    put = _request(
        instance,
        applied_state,
        STATE_PUT,
        (LiteralValue(value=4, type=INT),),
        UNIT,
        index=10,
    )
    old = Local(name="old", type=INT)
    ignored = Local(name="ignored", type=UNIT)
    body = Bind(
        binder=old,
        first=Perform(request=get),
        then=Bind(
            binder=ignored,
            first=Perform(request=put),
            then=Return(value=Var(local=old)),
        ),
    )

    assert Evaluator(attachments).evaluate(
        Handle(instance=instance, handler=handler.definition.id, computation=body)
    ) == (1, 4)
    assert cell.value == 4


def test_state_context_is_fresh_for_nested_and_reused_installations() -> None:
    outer = _instance("nested-state-outer")
    inner = _instance("nested-state-inner")
    attachments = RuntimeAttachments()
    handler, published = state_handler(
        0,
        _int,
        state_type=INT,
        answer_type=INT,
        answer_validator=_int,
        key="nested-state",
    )
    _attach_handler(attachments, handler)
    effect = state_effect(INT)
    inner_put = _request(
        inner,
        effect,
        STATE_PUT,
        (LiteralValue(value=5, type=INT),),
        UNIT,
        index=29,
    )
    inner_get = _request(inner, effect, STATE_GET, (), INT, index=30)
    outer_get = _request(outer, effect, STATE_GET, (), INT, index=31)
    ignored_put = Local(name="nested_put", type=UNIT)
    inner_value = Local(name="nested_inner_value", type=INT)
    inner_body = Bind(
        binder=ignored_put,
        first=Perform(request=inner_put),
        then=Perform(request=inner_get),
    )
    body = Bind(
        binder=inner_value,
        first=Handle(
            instance=inner, handler=handler.definition.id, computation=inner_body
        ),
        then=Perform(request=outer_get),
    )
    evaluator = Evaluator(attachments)
    assert (
        evaluator.evaluate(
            Handle(instance=outer, handler=handler.definition.id, computation=body)
        )
        == 0
    )
    assert published.value == 0

    outer_put = _request(
        outer,
        effect,
        STATE_PUT,
        (LiteralValue(value=9, type=INT),),
        UNIT,
        index=32,
    )
    changed = Bind(
        binder=ignored_put,
        first=Perform(request=outer_put),
        then=Perform(request=outer_get),
    )
    assert (
        evaluator.evaluate(
            Handle(instance=outer, handler=handler.definition.id, computation=changed)
        )
        == 9
    )
    assert (
        evaluator.evaluate(
            Handle(
                instance=outer,
                handler=handler.definition.id,
                computation=Perform(request=outer_get),
            )
        )
        == 0
    )
    assert published.value == 0


def test_score_and_weight_accumulators_are_per_dynamic_installation() -> None:
    attachments = RuntimeAttachments()
    outer_score = _instance("nested-score-outer")
    inner_score = _instance("nested-score-inner")
    score, published_score = score_handler(
        identity=0,
        weight_validator=_int,
        total_validator=_int,
        answer_type=STRING,
        answer_validator=lambda value: isinstance(value, str),
        key="nested-score",
    )
    outer_weight = _instance("nested-weight-outer")
    inner_weight = _instance("nested-weight-inner")
    weight, published_weight = weight_handler(
        1,
        _multiply,
        _int,
        weight_type=INT,
        expose_total=False,
        answer_type=STRING,
        answer_validator=lambda value: isinstance(value, str),
        key="nested-weight",
    )
    _attach_handler(attachments, score)
    _attach_handler(attachments, weight)

    score_refs = tuple(
        attachments.bind_value(
            AttachmentId.derive("test", "nested-score", value),
            value,
            LOG_WEIGHT,
            validator=_int,
            duplicable=True,
        )
        for value in (2, 3, 5)
    )
    score_requests = tuple(
        _request(
            instance,
            SCORE,
            SCORE_ADD,
            (ref,),
            UNIT,
            index=33 + offset,
        )
        for offset, (instance, ref) in enumerate(
            (
                (outer_score, score_refs[0]),
                (inner_score, score_refs[1]),
                (outer_score, score_refs[2]),
            )
        )
    )
    ignored_unit = Local(name="nested_score_unit", type=UNIT)
    ignored_answer = Local(name="nested_score_answer", type=STRING)
    inner_score_body = Bind(
        binder=ignored_unit,
        first=Perform(request=score_requests[1]),
        then=Return(value=LiteralValue(value="inner", type=STRING)),
    )
    outer_score_body = Bind(
        binder=ignored_unit,
        first=Perform(request=score_requests[0]),
        then=Bind(
            binder=ignored_answer,
            first=Handle(
                instance=inner_score,
                handler=score.definition.id,
                computation=inner_score_body,
            ),
            then=Bind(
                binder=ignored_unit,
                first=Perform(request=score_requests[2]),
                then=Return(value=LiteralValue(value="outer", type=STRING)),
            ),
        ),
    )
    evaluator = Evaluator(attachments)
    assert (
        evaluator.evaluate(
            Handle(
                instance=outer_score,
                handler=score.definition.id,
                computation=outer_score_body,
            )
        )
        == "outer"
    )
    assert published_score.total == 7
    assert published_score.contributions == [2, 5]

    weight_effect = WEIGHT_EFFECT.apply((INT,))
    weight_requests = tuple(
        _request(
            instance,
            weight_effect,
            WEIGHT_ADD,
            (LiteralValue(value=value, type=INT),),
            UNIT,
            index=36 + offset,
        )
        for offset, (instance, value) in enumerate(
            (
                (outer_weight, 2),
                (inner_weight, 3),
                (outer_weight, 5),
            )
        )
    )
    inner_weight_body = Bind(
        binder=ignored_unit,
        first=Perform(request=weight_requests[1]),
        then=Return(value=LiteralValue(value="inner", type=STRING)),
    )
    outer_weight_body = Bind(
        binder=ignored_unit,
        first=Perform(request=weight_requests[0]),
        then=Bind(
            binder=ignored_answer,
            first=Handle(
                instance=inner_weight,
                handler=weight.definition.id,
                computation=inner_weight_body,
            ),
            then=Bind(
                binder=ignored_unit,
                first=Perform(request=weight_requests[2]),
                then=Return(value=LiteralValue(value="outer", type=STRING)),
            ),
        ),
    )
    assert (
        evaluator.evaluate(
            Handle(
                instance=outer_weight,
                handler=weight.definition.id,
                computation=outer_weight_body,
            )
        )
        == "outer"
    )
    assert published_weight.total == 10
    assert published_weight.contributions == [2, 5]

    assert (
        evaluator.evaluate(
            Handle(
                instance=outer_score,
                handler=score.definition.id,
                computation=Return(value=LiteralValue(value="clean", type=STRING)),
            )
        )
        == "clean"
    )
    assert published_score.total == 0
    assert published_score.contributions == []


def test_parameterized_builtin_handler_uses_the_applied_interface() -> None:
    instance = _instance("checked-state")
    handler, _cell = state_handler(
        1,
        _int,
        state_type=INT,
        answer_type=INT,
        expose_final=False,
        key="checked-state",
    )
    applied_state = state_effect(INT)
    request = _request(
        instance,
        applied_state,
        STATE_GET,
        (),
        INT,
        index=20,
    )
    computation = Handle(
        instance=instance,
        handler=handler.definition.id,
        computation=Perform(request=request),
    )
    registry = KernelRegistry()
    registry.register_effect(STATE_EFFECT)
    registry.register_handler(handler.definition)

    inferred = infer_computation(computation, registry)
    assert inferred.effects.entries == ()
    assert inferred.result == INT
    assert handler.definition.effect == applied_state


def test_generic_handler_static_arguments_drive_runtime_dispatch() -> None:
    instance = _instance("runtime-generic-state")
    attachments = RuntimeAttachments()
    handler, _cell = state_handler(
        4,
        _int,
        key="runtime-generic-state",
    )
    _attach_handler(attachments, handler)
    applied_state = state_effect(INT)
    request = _request(
        instance,
        applied_state,
        STATE_GET,
        (),
        INT,
        index=40,
    )
    static_arguments = tuple(INT for _binder in handler.definition.telescope)
    computation = Handle(
        instance=instance,
        handler=handler.definition.id,
        computation=Perform(request=request),
        static_arguments=static_arguments,
    )
    registry = KernelRegistry()
    registry.register_effect(STATE_EFFECT)
    registry.register_handler(handler.definition)

    inferred = infer_computation(computation, registry)
    assert inferred.effects.entries == ()
    assert inferred.result == INT
    assert Evaluator(attachments).evaluate_checked(computation, registry) == 4

    mismatched = _request(
        instance,
        state_effect(STRING),
        STATE_GET,
        (),
        STRING,
        index=47,
    )
    with pytest.raises(InvalidHandlerError, match="disagree about the interface"):
        Evaluator(attachments).evaluate(
            Handle(
                instance=instance,
                handler=handler.definition.id,
                computation=Perform(request=mismatched),
                static_arguments=static_arguments,
            )
        )


def test_checked_evaluation_uses_the_typed_runtime_environment() -> None:
    local = Local(name="input", type=INT)
    computation = Return(value=Var(local=local))
    registry = KernelRegistry()

    assert Evaluator().evaluate_checked(computation, registry, {local: 7}) == 7

    mistyped_local = Local(name="input", type=STRING)
    with pytest.raises(KernelError, match="unbound or mistyped local 'input'"):
        Evaluator().evaluate_checked(
            computation,
            registry,
            {mistyped_local: "seven"},
        )


def test_checked_evaluation_rejects_handler_id_collision_with_divergent_definition() -> (
    None
):
    instance = _instance("manifest-state")
    expected, _cell = state_handler(
        1,
        _int,
        state_type=INT,
        answer_type=INT,
        answer_validator=_int,
        expose_final=False,
        key="manifest-state",
    )
    divergent_definition = expected.definition.with_(
        name="same-id-different-definition",
    )
    divergent = RuntimeHandler(
        definition=divergent_definition,
        clauses=expected.clauses,
        return_clause=expected.return_clause,
        input_validator=expected.input_validator,
        output_validator=expected.output_validator,
    )
    attachments = RuntimeAttachments()
    _attach_handler(attachments, divergent)
    applied_state = state_effect(INT)
    request = _request(
        instance,
        applied_state,
        STATE_GET,
        (),
        INT,
        index=23,
    )
    computation = Handle(
        instance=instance,
        handler=expected.definition.id,
        computation=Perform(request=request),
    )
    registry = KernelRegistry()
    registry.register_effect(STATE_EFFECT)
    registry.register_handler(expected.definition)
    definitions = dict(registry.handlers)
    manifest = HandlerManifest(definitions=definitions)
    definitions.clear()
    assert manifest.definitions[expected.definition.id] is expected.definition
    assert manifest == HandlerManifest.from_registry(registry)

    # The ergonomic evaluator can execute deliberately unchecked core.
    assert Evaluator(attachments).evaluate(computation) == 1
    checked = Evaluator(
        attachments,
        handler_manifest=manifest,
    )
    with pytest.raises(InvalidHandlerError, match="checked definition"):
        checked.evaluate(computation)
    with pytest.raises(InvalidHandlerError, match="checked definition"):
        Evaluator(attachments).evaluate_checked(computation, registry)


def test_abort_is_zero_shot_and_discards_the_bind_continuation() -> None:
    instance = _instance("abort")
    attachments = RuntimeAttachments()
    handler = abort_handler(
        error_type=STRING,
        input_type=INT,
        output_type=INT,
        key="abort-test",
    )
    _attach_handler(attachments, handler)
    request = _request(
        instance,
        ABORT_EFFECT.apply((STRING,)),
        ABORT_ABORT,
        (LiteralValue(value="stop", type=STRING),),
        INT,
        static_arguments=(INT,),
        index=11,
    )
    never = Local(name="never", type=INT)
    body = Bind(
        binder=never,
        first=Perform(request=request),
        then=Return(value=LiteralValue(value=999, type=INT)),
    )
    assert Evaluator(attachments).evaluate(
        Handle(instance=instance, handler=handler.definition.id, computation=body)
    ) == Aborted("stop")


def test_choose_uses_multi_shot_resumption_and_tracks_branch_addresses() -> None:
    choose_instance = _instance("choose")
    random_instance = _instance("choose-random")
    attachments = RuntimeAttachments()
    choose = choose_handler(_int, answer_type=INT, answer_validator=_int)
    recorder = TraceRecorder()

    # A deterministic Random interpreter makes the nested request a convenient
    # witness that each resumed branch receives a distinct resumption path.
    draw = draw_handler(
        _int,
        draw=lambda _site, _sampleable, _request: 9,
        answer_type=INT,
        key="branch-draw",
    )
    trace = trace_handler(
        RANDOM,
        (RANDOM_SAMPLE,),
        recorder,
        validators={RANDOM_SAMPLE: _int},
        answer_type=INT,
        key="branch-trace",
    )
    for handler in (choose, draw, trace):
        _attach_handler(attachments, handler)

    choices_id = AttachmentId.derive("test", "choices")
    choices_ref = attachments.bind_value(
        choices_id,
        (1, 2),
        CHOICES_A,
        validator=lambda value: isinstance(value, tuple),
        duplicable=True,
    )
    choose_request = _request(
        choose_instance,
        CHOOSE,
        CHOOSE_CHOOSE,
        (choices_ref,),
        INT,
        static_arguments=(INT,),
        index=12,
    )
    random_request = _random_request(attachments, random_instance, index=13)
    selected = Local(name="selected", type=INT)
    ignored = Local(name="draw", type=INT)
    body = Bind(
        binder=selected,
        first=Perform(request=choose_request),
        then=Bind(
            binder=ignored,
            first=Perform(request=random_request),
            then=Return(value=Var(local=selected)),
        ),
    )
    computation = Handle(
        instance=random_instance,
        handler=draw.definition.id,
        computation=Handle(
            instance=random_instance,
            handler=trace.definition.id,
            computation=Handle(
                instance=choose_instance, handler=choose.definition.id, computation=body
            ),
        ),
    )

    assert Evaluator(attachments).evaluate(computation) == (1, 2)
    assert [event.address[2] for event in recorder.events] == [(0,), (1,)]


def test_multishot_branches_fork_from_one_pristine_handler_snapshot() -> None:
    choose_instance = _instance("forked-choose")
    state_instance = _instance("forked-state")
    attachments = RuntimeAttachments()
    choose = choose_handler(
        _int,
        answer_type=INT,
        answer_validator=_int,
        key="forked-choose",
    )
    applied_state = state_effect(INT)
    state_definition = HandlerDef(
        id=HandlerId.derive("test", "forkable-state"),
        name="forkable-state",
        effect=applied_state,
        clauses=(
            HandlerClauseDef(operation=STATE_GET, grade=ResumptionGrade.LINEAR),
            HandlerClauseDef(operation=STATE_PUT, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    fork_sources: list[int] = []

    def make_state_handler(initial: int) -> RuntimeHandler:
        cell = {"value": initial}

        def get(_request_value, resume, _context):
            return resume(cell["value"])

        def put(request, resume, _context):
            (new_value,) = request.arguments
            assert _int(new_value)
            cell["value"] = new_value
            return resume(None)

        def fork(_handler):
            fork_sources.append(cell["value"])
            return make_state_handler(cell["value"])

        return RuntimeHandler(
            definition=state_definition,
            clauses={
                STATE_GET: RuntimeClause(invoke=get, result_validator=_int),
                STATE_PUT: RuntimeClause(
                    invoke=put, result_validator=lambda value: value is None
                ),
            },
            input_validator=_int,
            output_validator=_int,
            duplicable_context=True,
            mutable_context=True,
            fork_context=fork,
        )

    state = make_state_handler(0)
    state = state.with_(context_factory=lambda: make_state_handler(0))
    _attach_handler(attachments, choose)
    _attach_handler(attachments, state)
    choices = attachments.bind_value(
        AttachmentId.derive("test", "forked-choices"),
        (10, 20),
        CHOICES_A,
        validator=lambda value: isinstance(value, tuple),
        duplicable=True,
    )
    choose_request = _request(
        choose_instance,
        CHOOSE,
        CHOOSE_CHOOSE,
        (choices,),
        INT,
        static_arguments=(INT,),
        index=24,
    )
    get_request = _request(
        state_instance,
        applied_state,
        STATE_GET,
        (),
        INT,
        index=25,
    )
    put_request = _request(
        state_instance,
        applied_state,
        STATE_PUT,
        (LiteralValue(value=1, type=INT),),
        UNIT,
        index=26,
    )
    selected = Local(name="selected", type=INT)
    before = Local(name="before", type=INT)
    ignored = Local(name="put", type=UNIT)
    body = Bind(
        binder=selected,
        first=Perform(request=choose_request),
        then=Bind(
            binder=before,
            first=Perform(request=get_request),
            then=Bind(
                binder=ignored,
                first=Perform(request=put_request),
                then=Return(value=Var(local=before)),
            ),
        ),
    )
    computation = Handle(
        instance=choose_instance,
        handler=choose.definition.id,
        computation=Handle(
            instance=state_instance, handler=state.definition.id, computation=body
        ),
    )

    assert Evaluator(attachments).evaluate(computation) == (0, 0)
    # One pristine capture plus one fork per shot. Shot one cannot mutate the
    # seed from which shot two is created.
    assert fork_sources == [0, 0, 0]


@pytest.mark.parametrize(
    ("raise_on_return", "expected_finalization"),
    [
        (False, {0: "drop", 1: "drop", 2: "exit", 3: "exit"}),
        (True, {0: "drop", 1: "drop", 2: "drop"}),
    ],
)
def test_multishot_handler_contexts_finalize_exactly_once(
    raise_on_return: bool,
    expected_finalization: dict[int, str],
) -> None:
    choose_instance = _instance(f"resource-choose-{raise_on_return}")
    resource_instance = _instance(f"resource-state-{raise_on_return}")
    attachments = RuntimeAttachments()
    choose = choose_handler(
        _int,
        answer_type=INT,
        answer_validator=_int,
        key=f"resource-choose-{raise_on_return}",
    )
    effect = state_effect(INT)
    definition = HandlerDef(
        id=HandlerId.derive("test", "resource-state", raise_on_return),
        name="resource-state",
        effect=effect,
        clauses=(
            HandlerClauseDef(operation=STATE_GET, grade=ResumptionGrade.LINEAR),
            HandlerClauseDef(operation=STATE_PUT, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    created: list[int] = []
    finalized: dict[int, str] = {}

    def make_context(initial: int, *, tracked: bool = True) -> RuntimeHandler:
        label = len(created) if tracked else -1
        if tracked:
            created.append(label)
        cell = {"value": initial}

        def get(_request_value, resume, _context):
            return resume(cell["value"])

        def put(request, resume, _context):
            (new_value,) = request.arguments
            cell["value"] = new_value
            return resume(None)

        def fork(_handler):
            return make_context(cell["value"])

        def finish(value, _context):
            if raise_on_return:
                raise RuntimeError("resource return failed")
            return value

        def finalize(mode: str) -> None:
            assert label not in finalized
            finalized[label] = mode

        return RuntimeHandler(
            definition=definition,
            clauses={
                STATE_GET: RuntimeClause(invoke=get, result_validator=_int),
                STATE_PUT: RuntimeClause(
                    invoke=put, result_validator=lambda value: value is None
                ),
            },
            return_clause=finish,
            input_validator=_int,
            output_validator=_int,
            duplicable_context=True,
            mutable_context=True,
            fork_context=fork,
            on_exit=lambda: finalize("exit"),
            on_drop=lambda: finalize("drop"),
        )

    resource = make_context(0, tracked=False)
    resource = resource.with_(context_factory=lambda: make_context(0))
    _attach_handler(attachments, choose)
    _attach_handler(attachments, resource)
    choices = attachments.bind_value(
        AttachmentId.derive("test", "resource-choices", raise_on_return),
        (1, 2),
        CHOICES_A,
        validator=lambda value: isinstance(value, tuple),
        duplicable=True,
    )
    request = _request(
        choose_instance,
        CHOOSE,
        CHOOSE_CHOOSE,
        (choices,),
        INT,
        static_arguments=(INT,),
        index=39,
    )
    computation = Handle(
        instance=choose_instance,
        handler=choose.definition.id,
        computation=Handle(
            instance=resource_instance,
            handler=resource.definition.id,
            computation=Perform(request=request),
        ),
    )
    evaluator = Evaluator(attachments)
    if raise_on_return:
        with pytest.raises(RuntimeError, match="resource return failed"):
            evaluator.evaluate(computation)
    else:
        assert evaluator.evaluate(computation) == (1, 2)

    assert created == list(expected_finalization)
    assert finalized == expected_finalization


def test_mutable_duplicable_handler_requires_a_fork_context() -> None:
    definition = HandlerDef(
        id=HandlerId.derive("test", "missing-fork"),
        name="missing-fork",
        effect=RANDOM,
        clauses=(
            HandlerClauseDef(operation=RANDOM_SAMPLE, grade=ResumptionGrade.LINEAR),
        ),
        input_type=INT,
        output_type=INT,
    )
    with pytest.raises(InvalidHandlerError, match="needs fork_context"):
        RuntimeHandler(
            definition=definition,
            clauses={
                RANDOM_SAMPLE: RuntimeClause(
                    invoke=lambda _request, resume, _context: resume(1),
                    result_validator=_int,
                )
            },
            duplicable_context=True,
            mutable_context=True,
        )


def test_unrestricted_resumption_rejects_captured_mutable_state() -> None:
    choose_instance = _instance("nonduplicable-choose")
    state_instance = _instance("captured-state")
    attachments = RuntimeAttachments()
    choose = choose_handler(
        _int,
        answer_type=INT,
        answer_validator=_int,
        key="nonduplicable-choose",
    )
    state, _cell = state_handler(
        0,
        _int,
        state_type=INT,
        answer_type=INT,
        answer_validator=_int,
        key="captured-state",
    )
    _attach_handler(attachments, choose)
    _attach_handler(attachments, state)
    choices_ref = attachments.bind_value(
        AttachmentId.derive("test", "nonduplicable-choices"),
        (1, 2),
        CHOICES_A,
        validator=lambda value: isinstance(value, tuple),
        duplicable=True,
    )
    request = _request(
        choose_instance,
        CHOOSE,
        CHOOSE_CHOOSE,
        (choices_ref,),
        INT,
        static_arguments=(INT,),
        index=14,
    )
    computation = Handle(
        instance=choose_instance,
        handler=choose.definition.id,
        computation=Handle(
            instance=state_instance,
            handler=state.definition.id,
            computation=Perform(request=request),
        ),
    )
    with pytest.raises(NonDuplicableContinuationError, match="State.run"):
        Evaluator(attachments).evaluate(computation)


@pytest.mark.parametrize(
    ("grade", "clause", "message"),
    [
        (
            ResumptionGrade.ZERO,
            lambda _request, resume, _context: resume(1),
            "grade-0",
        ),
        (
            ResumptionGrade.AFFINE,
            lambda _request, resume, _context: (resume(1), resume(2))[1],
            "more than once",
        ),
        (
            ResumptionGrade.LINEAR,
            lambda _request, _resume, _context: 1,
            "exactly once",
        ),
    ],
)
def test_resumption_grades_are_enforced_in_every_execution(
    grade,
    clause,
    message,
) -> None:
    instance = _instance(f"grade-{grade.value}")
    definition = HandlerDef(
        id=HandlerId.derive("test", "grade", grade.value),
        name=f"grade-{grade.value}",
        effect=RANDOM,
        clauses=(HandlerClauseDef(operation=RANDOM_SAMPLE, grade=grade),),
        input_type=INT,
        output_type=INT,
    )
    handler = RuntimeHandler(
        definition=definition,
        clauses={RANDOM_SAMPLE: RuntimeClause(invoke=clause, result_validator=_int)},
        duplicable_context=True,
    )
    attachments = RuntimeAttachments()
    _attach_handler(attachments, handler)
    request = _request(instance, RANDOM, RANDOM_SAMPLE, (), INT, index=15)
    with pytest.raises(ResumptionUsageError, match=message):
        Evaluator(attachments).evaluate(
            Handle(
                instance=instance,
                handler=definition.id,
                computation=Perform(request=request),
            )
        )


def test_random_handler_validates_the_resumed_result() -> None:
    instance = _instance("bad-random")
    attachments = RuntimeAttachments()
    handler = draw_handler(
        _int,
        draw=lambda _site, _sampleable, _request: "not-an-int",
        answer_type=INT,
        key="bad-random",
    )
    _attach_handler(attachments, handler)
    request = _random_request(attachments, instance, index=16)
    with pytest.raises(RuntimeTypeMismatch, match="result of operation"):
        Evaluator(attachments).evaluate(
            Handle(
                instance=instance,
                handler=handler.definition.id,
                computation=Perform(request=request),
            )
        )


def test_weight_accumulates_with_caller_supplied_semiring_operation() -> None:
    instance = _instance("weight")
    attachments = RuntimeAttachments()
    handler, accumulator = weight_handler(
        1,
        _multiply,
        _int,
        weight_type=INT,
        answer_type=STRING,
        answer_validator=lambda value: isinstance(value, str),
    )
    _attach_handler(attachments, handler)
    weight_effect = WEIGHT_EFFECT.apply((INT,))
    first = _request(
        instance,
        weight_effect,
        WEIGHT_ADD,
        (LiteralValue(value=2, type=INT),),
        UNIT,
        index=17,
    )
    second = _request(
        instance,
        weight_effect,
        WEIGHT_ADD,
        (LiteralValue(value=3, type=INT),),
        UNIT,
        index=18,
    )
    left = Local(name="left", type=UNIT)
    right = Local(name="right", type=UNIT)
    body = Bind(
        binder=left,
        first=Perform(request=first),
        then=Bind(
            binder=right,
            first=Perform(request=second),
            then=Return(value=LiteralValue(value="done", type=STRING)),
        ),
    )
    assert Evaluator(attachments).evaluate(
        Handle(instance=instance, handler=handler.definition.id, computation=body)
    ) == ("done", 6)
    assert accumulator.contributions == [2, 3]


def test_builtin_answer_carriers_validate_structure_and_components() -> None:
    state, _state_cell = state_handler(
        0,
        _int,
        state_type=INT,
        expose_final=True,
        answer_type=INT,
        answer_validator=_int,
        key="validated-state-output",
    )
    score, _score_accumulator = score_handler(
        identity=0,
        weight_validator=_int,
        total_validator=_int,
        expose_total=True,
        answer_type=INT,
        answer_validator=_int,
        key="validated-score-output",
    )
    weight, _weight_accumulator = weight_handler(
        1,
        _multiply,
        _int,
        weight_type=INT,
        expose_total=True,
        answer_type=INT,
        answer_validator=_int,
        total_validator=_int,
        key="validated-weight-output",
    )
    choose = choose_handler(
        _int,
        answer_type=INT,
        answer_validator=_int,
        key="validated-choice-output",
    )

    for handler in (state, score, weight):
        assert handler.output_validator((1, 2)) is not False
        assert handler.output_validator((1, "wrong")) is False
        assert handler.output_validator((1, 2, 3)) is False
    assert choose.output_validator((1, 2, 3)) is not False
    assert choose.output_validator((1, "wrong")) is False
    assert choose.output_validator([1, 2, 3]) is False

    with pytest.raises(ValueError, match="output validator"):
        choose_handler(_int, combine=lambda results: results)

    custom = choose_handler(
        _int,
        combine=lambda results: {"branches": results},
        combine_validator=lambda value: (
            isinstance(value, dict) and set(value) == {"branches"}
        ),
        answer_type=INT,
        answer_validator=_int,
        key="validated-custom-choice-output",
    )
    assert custom.output_validator({"branches": [(1,), (2,)]}) is not False
    assert custom.output_validator((1, 2)) is False


def test_unhandled_effect_is_explicit() -> None:
    instance = _instance("unhandled")
    request = _request(
        instance,
        RANDOM,
        RANDOM_SAMPLE,
        (),
        INT,
        index=19,
    )
    with pytest.raises(UnhandledEffectError):
        Evaluator().evaluate(Perform(request=request))
