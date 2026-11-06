"""Render author-year citations and page bibliographies from one BibTeX file.

Use ``[@key]`` or ``[@key; @another]`` in prose. Fenced and inline code are
left alone. Unknown keys fail the build rather than producing broken citations.
The references page uses ``<!-- bibliography: all -->`` for the full list.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

from mkdocs.exceptions import PluginError
from pybtex.backends.html import Backend
from pybtex.database import parse_file
from pybtex.plugin import find_plugin
from pybtex.richtext import Text


_CITATION = re.compile(r"\[@([^\]\n]+)\]")
_KEYS = re.compile(r"[A-Za-z0-9_:-]+(?:\s*;\s*@[A-Za-z0-9_:-]+)*")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)", re.DOTALL)
_ALL = "<!-- bibliography: all -->"
_data = None
_formatted: dict[str, str] = {}
_backend = Backend()


def on_config(config):
    """Parse and format all entries once, including uncited entries."""
    global _data, _formatted
    source = Path(config["docs_dir"]) / "references.bib"
    try:
        _data = parse_file(str(source))
        for entry in _data.entries.values():
            # Maintainer tutorials often have a version but no publication date.
            # Keep that absence explicit in both labels and formatted entries.
            entry.fields.setdefault("year", "n.d.")
            for field in ("url", "doi"):
                if field in entry.fields:
                    entry.fields[field] = re.sub(
                        r"\\+([_%&#])", r"\1", entry.fields[field]
                    )
                    entry.fields[field] = re.sub(
                        r"\\+textasciitilde(?:\{\})?\s*", "~", entry.fields[field]
                    )
        # Disambiguate distinct works with the same author-year label in both
        # the inline citation and the formatted entry, without altering BibTeX.
        groups: dict[str, list[str]] = {}
        for key in _data.entries:
            groups.setdefault(_label(key), []).append(key)
        for keys in groups.values():
            if len(keys) > 1:
                keys.sort(key=lambda key: _data.entries[key].fields["title"])
                for index, key in enumerate(keys):
                    entry = _data.entries[key]
                    separator = "-" if entry.fields["year"] == "n.d." else ""
                    entry.fields["year"] += separator + chr(ord("a") + index)
        style = find_plugin("pybtex.style.formatting", "plain")()
        _formatted = {
            entry.key: entry.text.render(_backend)
            for entry in style.format_bibliography(_data)
        }
    except Exception as error:
        raise PluginError(f"Cannot format {source}: {error}") from error
    return config


def _label(key: str) -> str:
    entry = _data.entries[key]
    people = entry.persons.get("author") or entry.persons.get("editor") or []
    names = [" ".join(person.prelast_names + person.last_names) for person in people]
    if not names:
        author = entry.fields.get("organization", entry.fields["title"])
    elif len(names) == 1:
        author = names[0]
    elif len(names) == 2:
        author = " and ".join(names)
    else:
        author = names[0] + " et al."
    year = entry.fields.get("year", "n.d.")
    if entry.fields.get("pubstate") == "forthcoming":
        year = "forthcoming"
    return Text.from_latex(f"{author}, {year}").render(_backend)


def _references(keys: list[str]) -> str:
    return (
        '\n\n<div class="qv-references">\n'
        + "\n".join(
            f'<p id="ref-{escape(key, quote=True)}">{_formatted[key]}</p>'
            for key in keys
        )
        + "\n</div>\n"
    )


def on_page_markdown(markdown, page, **kwargs):
    """Resolve citations outside code and append only the page's references."""
    used: list[str] = []

    def replace(match):
        raw = match.group(1)
        if not _KEYS.fullmatch(raw):
            raise PluginError(
                f"Invalid citation {match.group(0)!r} in {page.file.src_uri}"
            )
        keys = [key.strip().lstrip("@") for key in raw.split(";")]
        links = []
        for key in keys:
            if key not in _formatted:
                raise PluginError(f"Unknown citation {key!r} in {page.file.src_uri}")
            if key not in used:
                used.append(key)
            links.append(
                f'<a class="qv-citation" href="#ref-{escape(key, quote=True)}">{_label(key)}</a>'
            )
        return "(" + "; ".join(links) + ")"

    lines = []
    prose = []

    def flush_prose():
        source = "".join(prose)
        start = 0
        for code in _INLINE_CODE.finditer(source):
            lines.extend(
                (_CITATION.sub(replace, source[start : code.start()]), code[0])
            )
            start = code.end()
        lines.append(_CITATION.sub(replace, source[start:]))
        prose.clear()

    fence = None
    full_bibliography = False
    for line in markdown.splitlines(keepends=True):
        # Quoted fences also contain literal code. Keep their quote markers.
        logical = re.sub(r"^(?: {0,3}> ?)+", "", line)
        marker = re.match(r"^\s*(`{3,}|~{3,})", logical)
        if fence is not None:
            lines.append(line)
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= len(fence)
                and not logical[marker.end() :].strip()
            ):
                fence = None
            continue
        if marker:
            flush_prose()
            fence = marker[1]
            lines.append(line)
            continue
        if re.match(r"^(?: {4}|\t)", logical):
            flush_prose()
            lines.append(line)
            continue
        if line.strip() == _ALL:
            flush_prose()
            full_bibliography = True
            lines.append(_references(list(_formatted)))
            continue
        prose.append(line)
    flush_prose()
    result = "".join(lines)
    if used and not full_bibliography:
        result += "\n\n## Cited sources\n" + _references(used)
    return result
