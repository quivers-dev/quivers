"""The schema follows and independently rechecks packed interpretation uses."""

from math import log

import pytest

from quivers.dsl import Compiler, loads as load_source
from quivers.dsl.parser import parse
from quivers.modeling import Model
from quivers.qiec import (
    BOOL,
    PackedBooleanInterpretationBinding,
    RuleError,
    dumps,
    loads,
    validate_module,
)
from quivers.qiec.terms import (
    Bind,
    Call,
    Handle,
    LiteralValue,
    Perform,
    TensorValue,
    Var,
)
from tests.dsl.test_packed_boolean_outcomes import EXPLICIT, SOURCE


INDEPENDENT = """\
program unrelated : Token -> Result
    sample value <- Normal(loc=0.0, scale=1.0)
    return value

export unrelated
"""


def test_schema_preserves_either_interpretation_use_and_omits_unrelated_entries():
    source = SOURCE + EXPLICIT + INDEPENDENT
    model = Model(source, entry="fit")
    (ordinary,) = model.schema.closed_interpretations
    (packed,) = model.schema.packed_interpretations
    assert packed.interpretation == ordinary.interpretation.id
    assert ordinary.carrier == BOOL

    explicit = Model(source, entry="explicit").schema
    assert explicit.closed_interpretations == (ordinary,)
    assert explicit.packed_interpretations == ()
    inactive = Model(source, entry="unrelated").schema
    assert inactive.closed_interpretations == ()
    assert inactive.packed_interpretations == ()

    standalone = loads(dumps(packed))
    assert isinstance(standalone, PackedBooleanInterpretationBinding)
    assert standalone == packed
    executable = loads(dumps(model.checked_module))
    standalone.validate(executable, loads(dumps(ordinary)))


def _rewrite_terminal(computation, change):
    if isinstance(computation, Bind):
        return computation.with_(then=_rewrite_terminal(computation.then, change))
    return change(computation)


def _replace_body(module, identity, change):
    return module.with_(
        computations=tuple(
            item.with_(body=_rewrite_terminal(item.body, change))
            if item.id == identity
            else item
            for item in module.computations
        )
    )


def test_companion_rejects_same_type_operation_substitution_and_table_reordering():
    model = Model(SOURCE)
    (ordinary,) = model.schema.closed_interpretations
    (packed,) = model.schema.packed_interpretations
    module = model.checked_module
    ids = {computation.name: computation.id for computation in module.computations}
    swapped = ordinary.with_(
        interpretation=ordinary.interpretation.with_(
            operations=tuple(
                (rule, ids["reject"] if op == ids["accept"] else op)
                for rule, op in ordinary.interpretation.operations
            )
        )
    )
    with pytest.raises(RuleError, match="differs from its checked fold"):
        packed.validate(module, swapped)

    def reverse(terminal):
        assert isinstance(terminal, Call)
        tokens, table, mode = terminal.arguments
        assert isinstance(table, TensorValue)
        return terminal.with_(
            arguments=(tokens, table.with_(items=table.items[::-1]), mode)
        )

    wrong_order = _replace_body(module, packed.computation, reverse)
    validate_module(wrong_order)
    with pytest.raises(RuleError, match="does not compute its checked truth table"):
        packed.validate(wrong_order, ordinary)


def test_companion_rejects_an_adapter_that_discards_checked_truths():
    model = Model(SOURCE)
    (ordinary,) = model.schema.closed_interpretations
    (packed,) = model.schema.packed_interpretations

    def suppress(terminal):
        assert isinstance(terminal, Handle)
        assert isinstance(terminal.computation, Perform)
        request = terminal.computation.request
        words, terms, weights, table, normalize = request.arguments
        assert isinstance(table, Var)
        forged = TensorValue(
            items=tuple(LiteralValue(value=False, type=BOOL) for _ in range(11)),
            result_type=table.local.type,
        )
        return terminal.with_(
            computation=terminal.computation.with_(
                request=request.with_(
                    arguments=(words, terms, weights, forged, normalize)
                )
            )
        )

    erased = _replace_body(model.checked_module, packed.query, suppress)
    validate_module(erased)
    with pytest.raises(RuleError, match="does not forward its checked truth table"):
        packed.validate(erased, ordinary)


def test_authored_handler_clause_reaches_the_same_packed_companion():
    source = (
        SOURCE.replace(" #[learnable]", "")
        + """\

effect Retrieval
    fetch : Unit -> Real

instance retrieve : Retrieval

handler reading for Retrieval : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return value =>
        return value
    fetch() resumes 1 =>
        let value <- packed_reply()
        resume(value)

define packed_reply() : Real !{} =
    let rows <- Parity__parity__packed_boolean_outcomes[2](["x", "x"], true)
    truth_mass(rows)

define query_reading() : Real !{} =
    handle retrieve with reading in
        perform retrieve.fetch()

program through_handler : Token -> Result
    let mass <- query_reading()
    score observed = log(mass)
    return observed

export through_handler
"""
    )
    model = Model(source, entry="through_handler")
    (ordinary,) = model.schema.closed_interpretations
    (packed,) = model.schema.packed_interpretations
    packed.validate(model.checked_module, ordinary)
    bound = model.bind({})
    assert bound.schema.closed_interpretations == (ordinary,)
    assert bound.schema.packed_interpretations == (packed,)
    assert load_source(source).run("through_handler").log_joint == pytest.approx(
        log(0.5)
    )


def test_companion_rejects_nominal_substitution_with_identical_operations():
    begin = SOURCE.index("interpret parity")
    end = SOURCE.index("define lexical")
    alias = SOURCE[begin:end].replace("interpret parity", "interpret alias")
    compiler = Compiler(parse(SOURCE[:end] + alias + SOURCE[end:]))
    module = compiler.qiec_module
    (original, duplicate) = compiler._closed_interpretations
    (packed,) = compiler._packed_interpretations
    assert original.interpretation.operations == duplicate.interpretation.operations
    forged = packed.with_(interpretation=duplicate.interpretation.id)
    with pytest.raises(RuleError, match="stale nominal use"):
        forged.validate(module, duplicate)
