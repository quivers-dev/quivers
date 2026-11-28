"""Compare the checked continuation selector with an independent minimax game."""

from fractions import Fraction

import pytest

from quivers.dsl import loads as load_source
from quivers.qiec import ExactJvp, JvpBoundaryError, JvpLimit, JvpUnsupported
from quivers.qiec import finite_selection_jvp
from quivers.qiec.module import QiecModule
from quivers.qiec.serialization import dumps, loads


_SOURCE = """
effect Selection
    choose : Bool -> Bool
    loss : Real -> Unit
    unrelated : Real -> Real

instance select_effect : Selection

define minimizing_reply() : Bool !{select_effect} =
    perform select_effect.choose(false)

define game(x : Real) : Real !{select_effect} =
    let start <- perform select_effect.loss(10.0 * x)
    let left <- perform select_effect.choose(true)
    let right <- minimizing_reply()
    if left then
        if right then
            let observed <- perform select_effect.loss(9.0 * x)
            return 17.0 * x
        else
            let observed <- perform select_effect.loss(2.0 * x)
            return 11.0 * x
    else
        if right then
            let observed <- perform select_effect.loss(3.0 * x)
            return 7.0 * x
        else
            let observed <- perform select_effect.loss(5.0 * x)
            return 19.0 * x
"""


def _module(source=_SOURCE):
    source_module = load_source(source).qiec
    assert isinstance(source_module, QiecModule)
    module = loads(dumps(source_module))
    assert isinstance(module, QiecModule)
    ids = {op.name: op.id for op in module.effects[0].operations}
    entry = next(item.id for item in module.computations if item.name == "game")
    return module, entry, ids


@pytest.mark.parametrize("x", (Fraction(-2, 3), Fraction(1, 5), Fraction(7, 3)))
def test_later_selection_is_applied_before_earlier_choice(x):
    module, entry, ids = _module()
    direction = Fraction(4, 5)
    loss_factors = ((5, 3), (2, 9))
    value_factors = ((19, 7), (11, 17))
    opponent = [min(range(2), key=lambda j: x * factors[j]) for factors in loss_factors]
    first = max(range(2), key=lambda i: x * loss_factors[i][opponent[i]])
    second = opponent[first]
    result = finite_selection_jvp(
        module, entry, (x,), (direction,), choose=ids["choose"], loss=ids["loss"]
    )
    value = value_factors[first][second]
    loss = 10 + loss_factors[first][second]
    assert result.value == ExactJvp(x * value, direction * value)
    assert result.loss == ExactJvp(x * loss, direction * loss)
    assert result.ties == 0 and result.paths == 4 and result.source_steps > 0
    assert [record.choice for record in result.decisions] == [bool(first), bool(second)]
    assert result.decisions[0].losses == tuple(
        x * (10 + loss_factors[i][j]) for i, j in enumerate(opponent)
    )
    a, b = (record.address for record in result.decisions)
    assert a[0] != b[0] and not a[1] and any(frame[0] == "call" for frame in b[1])

    delta = Fraction(1, 1000)
    other = finite_selection_jvp(
        module,
        entry,
        (x + direction * delta,),
        (0,),
        choose=ids["choose"],
        loss=ids["loss"],
    )
    assert (other.loss.value - result.loss.value) / delta == result.loss.tangent
    assert (other.value.value - result.value.value) / delta == result.value.tangent


def test_loss_mutation_switches_demonstrated_source_path():
    module, entry, ids = _module(_SOURCE.replace("loss(3.0 * x)", "loss(1.0 * x)"))
    changed = finite_selection_jvp(
        module, entry, (2,), (1,), choose=ids["choose"], loss=ids["loss"]
    )
    assert tuple(decision.choice for decision in changed.decisions) == (True, False)
    assert changed.loss == ExactJvp(Fraction(24), Fraction(12))
    assert changed.value == ExactJvp(Fraction(22), Fraction(11))


def test_ties_are_refused_or_explicitly_return_a_branch_tangent():
    module, entry, ids = _module()
    with pytest.raises(JvpBoundaryError, match="equal continuation losses"):
        finite_selection_jvp(
            module, entry, (0,), (1,), choose=ids["choose"], loss=ids["loss"]
        )
    result = finite_selection_jvp(
        module,
        entry,
        (0,),
        (1,),
        choose=ids["choose"],
        loss=ids["loss"],
        choose_first_at_ties=True,
    )
    assert result.ties == 3
    assert tuple(decision.choice for decision in result.decisions) == (False, False)
    assert result.value == ExactJvp(Fraction(), Fraction(19))
    assert result.loss == ExactJvp(Fraction(), Fraction(15))


def test_checks_operations_context_and_budget_before_reporting():
    module, entry, ids = _module()
    with pytest.raises(JvpUnsupported, match="nominal interface and signature"):
        finite_selection_jvp(
            module, entry, (1,), (1,), choose=ids["loss"], loss=ids["choose"]
        )
    with pytest.raises(JvpLimit, match="exceeds 3 complete paths"):
        finite_selection_jvp(
            module,
            entry,
            (1,),
            (1,),
            choose=ids["choose"],
            loss=ids["loss"],
            max_paths=3,
        )
    with pytest.raises(JvpLimit):
        finite_selection_jvp(
            module, entry, (1,), (1,), choose=ids["choose"], loss=ids["loss"], fuel=1
        )
    foreign, entry, ids = _module(
        _SOURCE.replace("return 17.0 * x", "perform select_effect.unrelated(x)")
    )
    with pytest.raises(JvpUnsupported, match="outside the finite selection interface"):
        finite_selection_jvp(
            foreign, entry, (1,), (1,), choose=ids["choose"], loss=ids["loss"]
        )


@pytest.mark.parametrize("depth", (0, 1, 3, 5))
def test_complete_alternatives_have_the_exact_binary_tree_bound(depth):
    declarations = _SOURCE.split("define minimizing_reply", 1)[0]
    fixture = (
        declarations
        + """
define add_if(choice : Bool, amount : Real) : Unit !{select_effect} =
    if choice then
        perform select_effect.loss(amount)
    else
        perform select_effect.loss(0.0)

define game(x : Real) : Real !{select_effect} =
    let initial <- perform select_effect.loss(x)
"""
    )
    for i in range(1, depth + 1):
        fixture += f"    let choice_{i} <- perform select_effect.choose(true)\n"
        fixture += f"    let wrote_{i} <- add_if(choice_{i}, {i}.0 * x)\n"
    module, entry, ids = _module(fixture + "    return x\n")
    options = dict(choose=ids["choose"], loss=ids["loss"])
    paths = 2**depth
    result = finite_selection_jvp(module, entry, (3,), (1,), max_paths=paths, **options)
    total = 1 + depth * (depth + 1) // 2
    assert result.paths == paths
    assert result.loss == ExactJvp(Fraction(3 * total), Fraction(total))
    assert len(result.decisions) == depth
    assert all(decision.choice for decision in result.decisions)
    if depth:
        with pytest.raises(JvpLimit, match="complete paths"):
            finite_selection_jvp(
                module, entry, (3,), (1,), max_paths=paths - 1, **options
            )
