"""Declaration-option migration preserves source outside named header fields."""

from __future__ import annotations

import pytest

from quivers.cli import migrations
from quivers.cli.migrations import MigrationError, compose_migration
from quivers.cli.migrations import v0_19_0_to_v0_62_0 as preceding
from quivers.cli.migrations import v0_62_0_to_head as keywords
from quivers.dsl import CompileError, ParseError, loads, parse


SOURCE = """# λ: role=latent, over=model, level=bilinear_form stay in this comment.
object latent : FinSet 2
object bilinear_form : FinSet 2
morphism over : latent -> bilinear_form [role = latent, scale=0.1]

composition material_impl [level = bilinear_form]
program posterior : latent -> bilinear_form [effects=Pure, over = model]
    sample x <- Normal(loc=0.0, scale=1.0) [over=latent]
    return x
""".encode()


def test_option_renames_preserve_every_unrelated_byte() -> None:
    expected = (
        SOURCE.replace(b"role = latent", b"role = parameter")
        .replace(b"level = bilinear_form", b"level = magmoid")
        .replace(b"over = model", b"posterior_of = model")
    )
    result = compose_migration("v0.62.0", "HEAD")(SOURCE)
    assert result == expected
    assert result != SOURCE.replace(b"latent", b"parameter")
    assert keywords.migrate(result) == result
    parse(result.decode())


def test_same_words_in_other_options_and_strings_are_unchanged() -> None:
    source = b"""morphism f : A -> B [role=kernel, weights=latent, over=axis]
morphism g : A -> B [role="latent"]
composition C [level="bilinear_form"]
program p : A -> B
    sample z <- Normal(loc=0.0, scale=1.0) [over=axis]
    return z
"""
    assert keywords.migrate(source) == source


@pytest.mark.parametrize("source", [b"object A :\n", b"\x00\x00"])
def test_malformed_source_is_rejected(source: bytes) -> None:
    with pytest.raises(MigrationError):
        keywords.migrate(source)


def test_preceding_hop_is_pinned_and_preserves_released_option_names() -> None:
    assert preceding._TARGET_REV == "v0.62.0"
    source = b"morphism f : A -> B [role=latent]\n"
    assert compose_migration("v0.19.0", "v0.62.0")(source) == source
    assert compose_migration("v0.19.0", "HEAD")(source) == source.replace(
        b"role=latent", b"role=parameter"
    )


def test_semantic_hop_cannot_be_declared_an_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migrations, "_SEMANTIC_PAIRS", frozenset())
    with pytest.raises(MigrationError, match="identity migrators"):
        migrations.vcs_coverage_report()


def test_parameter_role_compiles_and_rejects_removed_role() -> None:
    source = "object A : FinSet 2\nmorphism f : A -> A [role=parameter]\nexport f\n"
    program = loads(source)
    assert list(program.parameters())
    with pytest.raises(CompileError, match="latent"):
        loads(source.replace("role=parameter", "role=latent"))


def test_removed_composition_level_is_rejected() -> None:
    parse("composition material_impl [level=magmoid]\n")
    with pytest.raises(ParseError, match="bilinear_form"):
        parse("composition material_impl [level=bilinear_form]\n")


def test_posterior_option_compiles_and_rejects_removed_option() -> None:
    source = """object Item : FinSet 1
object R : Real 1
program model : Item -> R
    sample z <- Normal(loc=0.0, scale=1.0)
    return z
program post(z : Real) : Item -> R [effects=Pure, posterior_of=model]
    return z
export model
"""
    loads(source)
    with pytest.raises(CompileError, match="unknown option 'over'"):
        loads(source.replace("posterior_of=model", "over=model"))
