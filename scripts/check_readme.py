# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Check the built wheel's long description as PyPI will show it.

Usage: python scripts/check_readme.py DIST_DIR [REPO_DIR]

Needs readme_renderer (PyPI's renderer). The README is read from the wheel's metadata,
not the checkout, and rendered as PyPI renders it. PyPI resolves no relative paths and
gives headings no anchors, so every image and link must be an absolute https URL. Those
into this repository must name a file that exists on its main branch (checked against
REPO_DIR, the checkout, default "."), or a heading of the README by its GitHub anchor.
Prints a report and exits non-zero on the first failure.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from email.parser import Parser
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from readme_renderer.markdown import render

REPO = ("personalrobotics", "mj_viser")
BRANCH = "main"


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and (tag, name) in (("img", "src"), ("a", "href"), ("source", "srcset")):
                self.refs.append((tag, value))


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[`*_]", "", heading).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _description(dist: Path) -> tuple[str, str]:
    wheel = next(dist.glob("ssmjviser-*.whl"))
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(z.read(name).decode())
    return str(metadata.get("Description-Content-Type", "")), str(metadata.get_payload())


def _check(url: str, repo: Path, anchors: set[str]) -> str:
    """Why ``url`` would not work on PyPI, or "" if it does."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        return "not an absolute https URL"
    segments = [s for s in parts.path.split("/") if s]
    owner_repo = tuple(segments[:2])
    if parts.hostname == "raw.githubusercontent.com" and owner_repo == REPO:
        if segments[2:3] != [BRANCH] or not (repo / "/".join(segments[3:])).is_file():
            return f"no file {'/'.join(segments[3:])} on {BRANCH}"
    elif parts.hostname == "github.com" and owner_repo == REPO and len(segments) > 2:
        if segments[2] in ("blob", "tree"):
            if segments[3:4] != [BRANCH] or not (repo / "/".join(segments[4:])).exists():
                return f"no path {'/'.join(segments[4:])} on {BRANCH}"
        elif segments[2] not in ("issues", "releases", "pulls", "actions"):
            return "an unrecognized repository URL"
    if parts.fragment and parts.hostname == "github.com" and owner_repo == REPO and len(segments) == 2:
        if parts.fragment not in anchors:
            return f"no heading with anchor #{parts.fragment}"
    return ""


def main(dist_dir: str, repo_dir: str = ".") -> None:
    content_type, text = _description(Path(dist_dir))
    if not content_type.startswith("text/markdown"):
        raise SystemExit(f"long description content type is {content_type!r}, not Markdown")
    rendered = render(text, variant="GFM")
    if rendered is None:
        raise SystemExit("the long description does not render")
    anchors = {_slug(m) for m in re.findall(r"^#+ (.+)$", text, re.M)}
    refs = _Refs()
    refs.feed(rendered)
    raw = _Refs()  # PyPI drops <source>; GitHub shows it
    raw.feed(text)
    seen = sorted(set(refs.refs) | {r for r in raw.refs if r[0] == "source"})
    failures = [(tag, url, why) for tag, url in seen if (why := _check(url, Path(repo_dir), anchors))]
    report = {
        "content_type": content_type,
        "images": sorted({u for t, u in seen if t in ("img", "source")}),
        "links": sorted({u for t, u in seen if t == "a"}),
        "failures": [{"tag": t, "url": u, "why": w} for t, u, w in failures],
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} image or link would not work on PyPI")


if __name__ == "__main__":
    main(*sys.argv[1:])
