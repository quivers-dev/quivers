"""Effect-lift dispatch emits only the implemented fixed schema shapes."""

import pytest

from quivers.arrows.instances import Function, Stochastic, VRel
from quivers.dsl.ast_nodes import (
    ObjectEffectApply,
    ObjectProduct,
    ObjectSlash,
    SchemaDecl,
    TypeName,
)
from quivers.monadic.instances import Alternative_, Identity
from quivers.monadic.typeclasses import Functor
from quivers.stochastic.effect_lifts import class_directed_lifts


class _FunctorOnly:
    """A nominal registration used only to test schema dispatch."""

    name = "Bare"


Functor.register(_FunctorOnly)


@pytest.mark.parametrize("effect", [_FunctorOnly(), Function(), Stochastic(), VRel()])
def test_functor_and_arrow_membership_does_not_invent_schema_lifts(effect) -> None:
    """An unsupported class yields no rule, rather than a misleading pure lift."""
    schema = SchemaDecl(
        name="arbitrary",
        parameters=(),
        domain=TypeName(name="A"),
        codomain=TypeName(name="B"),
    )
    assert class_directed_lifts(schema, effect) == ()


@pytest.mark.parametrize(
    "effect,prefixes",
    [
        (Identity(), ("pure", "apply", "bind")),
        (Alternative_(), ("pure", "apply", "bind", "alt")),
    ],
)
def test_lifts_have_fixed_shapes_and_preserve_the_base_name(effect, prefixes) -> None:
    """Changing the base rule's types does not pretend to derive a new lift."""
    schema = SchemaDecl(
        name="r", parameters=(), domain=TypeName(name="A"), codomain=TypeName(name="B")
    )
    changed = schema.with_(domain=TypeName(name="C"), codomain=TypeName(name="D"))
    lifts = class_directed_lifts(schema, effect)
    assert lifts == class_directed_lifts(changed, effect)
    assert tuple(item.name for item in lifts) == tuple(
        f"{prefix}_{effect.name}_r" for prefix in prefixes
    )
    assert lifts[0].domain == TypeName(name="X")
    assert lifts[0].codomain == ObjectEffectApply(
        effect=effect.name, args=(TypeName(name="X"),)
    )
    apply = lifts[1].domain
    assert isinstance(apply, ObjectProduct)
    left, right = apply.components
    assert isinstance(left, ObjectEffectApply)
    assert left.args == (
        ObjectSlash(
            result=TypeName(name="X"), argument=TypeName(name="Y"), direction="/"
        ),
    )
    assert right == ObjectEffectApply(effect=effect.name, args=(TypeName(name="Y"),))
