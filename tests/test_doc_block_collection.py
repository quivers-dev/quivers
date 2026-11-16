"""Explicit example markers extend the tutorial execution contract to all docs."""

from tests import test_doc_blocks as examples


def test_python_collection_respects_page_defaults_and_explicit_markers(
    tmp_path, monkeypatch
):
    root = tmp_path / "repo"
    docs = root / "docs"
    tutorial = docs / "tutorials" / "chapter.md"
    api = docs / "api" / "binding.md"
    guide = docs / "guides" / "binding.md"
    readme = root / "README.md"
    for path in (tutorial, api, guide, readme):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "```python\nimplicit = 1\n```\n\n"
            "<!-- python: skip -->\n```python\nskipped = 2\n```\n\n"
            "<!-- python: run -->\n```python\nexplicit = 3\n```\n\n"
            "<!-- python: run -->\n```python\nassert explicit == 3\n```\n"
        )
    monkeypatch.setattr(examples, "_REPO_ROOT", root)
    monkeypatch.setattr(examples, "_DOC_ROOTS", (docs, readme))
    monkeypatch.setattr(examples, "_TUTORIAL_ROOT", tutorial.parent)
    collected = examples._collect_py_blocks()
    for path in (api, guide, readme, tutorial):
        blocks = [
            (index, body.strip())
            for name, index, _, body in collected
            if name == str(path.relative_to(root))
        ]
        assert blocks == ([(0, "implicit = 1")] if path == tutorial else []) + [
            (2, "explicit = 3"),
            (3, "assert explicit == 3"),
        ]
