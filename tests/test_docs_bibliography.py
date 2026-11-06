"""Check citation links without changing executable documentation examples."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


@pytest.fixture
def bibliography(tmp_path):
    pytest.importorskip("pybtex")
    pytest.importorskip("mkdocs")
    path = Path(__file__).parents[1] / "docs/hooks/bibliography.py"
    spec = spec_from_file_location("docs_bibliography", path)
    hook = module_from_spec(spec)
    spec.loader.exec_module(hook)
    (tmp_path / "references.bib").write_text(
        """@book{first,
          author = {Example, Ada and Other, Ben},
          title = {A Book}, year = {2020}, publisher = {A Publisher},
          url = {https://example.org/book}
        }
        @article{second,
          author = {Else, Cora}, title = {Another Work},
          journal = {A Journal}, year = {2021}, volume = {1}, pages = {1--4}
        }"""
    )
    hook.on_config({"docs_dir": str(tmp_path)})
    return hook


def render(hook, source):
    return hook.on_page_markdown(
        source, SimpleNamespace(file=SimpleNamespace(src_uri="example.md"))
    )


def test_grouped_citations_link_to_one_entry_per_source(bibliography):
    html = render(bibliography, "Read [@first; @second], then [@first].")
    assert html.count('href="#ref-first"') == 2
    assert html.count('id="ref-first"') == 1
    assert 'href="#ref-second"' in html
    assert "Example and Other, 2020" in html
    assert "https://example.org/book" in html
    assert "[@" not in html


def test_literal_citation_syntax_in_examples_is_untouched(bibliography):
    source = "Use `[@missing]` literally.\n\n```python\nx = '[@missing]'\n```\n"
    source += "\n    ~~~text\n    [@missing]\n    ~~~\n"
    assert render(bibliography, source) == source


def test_unknown_citation_fails_with_page_and_key(bibliography):
    from mkdocs.exceptions import PluginError

    with pytest.raises(PluginError, match="Unknown citation 'missing' in example.md"):
        render(bibliography, "An unsupported reference [@missing].")


@pytest.mark.parametrize(
    "source",
    [
        "Literal `code\n[@missing]\ncode` here.\n",
        "Literal ``code `[@missing]` code`` here.\n",
        "\n    [@missing]\n",
        "> ```text\n> [@missing]\n> ```\n",
    ],
)
def test_other_markdown_code_forms_are_preserved(bibliography, source):
    assert render(bibliography, source) == source


def test_complete_bibliography_does_not_depend_on_page_order(bibliography):
    html = render(bibliography, "<!-- bibliography: all -->\n")
    assert html.count('id="ref-first"') == 1
    assert html.count('id="ref-second"') == 1
    assert "<!-- bibliography: all -->" not in html


def test_citations_do_not_leak_between_pages(bibliography):
    render(bibliography, "First page [@first].")
    second = render(bibliography, "Second page [@second].")
    assert 'id="ref-first"' not in second
    assert 'id="ref-second"' in second


def test_same_author_year_is_disambiguated(bibliography, tmp_path):
    source = tmp_path / "references.bib"
    with source.open("a") as stream:
        stream.write("""@book{third,
          author = {Example, Ada and Other, Ben},
          title = {Z Book}, year = {2020}, publisher = {A Publisher}
        }""")
    bibliography.on_config({"docs_dir": str(tmp_path)})
    html = render(bibliography, "Read [@first; @third].")
    assert "Example and Other, 2020a" in html
    assert "Example and Other, 2020b" in html
    assert html.count("2020a") == 2
    assert html.count("2020b") == 2
    assert "2020a" not in source.read_text()


def test_bibtex_url_escapes_do_not_break_links(bibliography, tmp_path):
    source = tmp_path / "references.bib"
    source.write_text(
        source.read_text().replace(
            "https://example.org/book",
            r"https://example.org/\textasciitilde reader/a\_book",
        )
    )
    bibliography.on_config({"docs_dir": str(tmp_path)})
    html = render(bibliography, "Read [@first].")
    assert 'href="https://example.org/~reader/a_book"' in html


def test_undated_tutorials_are_formatted_and_disambiguated(bibliography, tmp_path):
    source = tmp_path / "references.bib"
    source.write_text("""@misc{alpha,
      author = {{Project Developers}}, title = {A Tutorial},
      url = {https://example.org/alpha}
    }
    @misc{beta,
      author = {{Project Developers}}, title = {B Tutorial},
      url = {https://example.org/beta}
    }
    @misc{single,
      author = {{Other Developers}}, title = {Single Tutorial}
    }""")
    bibliography.on_config({"docs_dir": str(tmp_path)})
    html = render(bibliography, "Read [@alpha; @beta; @single].")
    visible_text = re.sub(r"<[^>]+>", "", html)
    assert "Project Developers, n.d.-a" in visible_text
    assert "Project Developers, n.d.-b" in visible_text
    assert "Other Developers, n.d." in visible_text
    assert html.count("n.d.-a") == 2
    assert html.count("n.d.-b") == 2
    assert 'href="https://example.org/alpha"' in html
    assert "year" not in source.read_text()
