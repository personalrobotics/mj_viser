# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Decide whether, and where, a release run may publish; refuse a mismatched release.

Usage: python3 scripts/release/guard.py REF EVENT [PUBLISH_TESTPYPI]

REF is the ref the run is for (``refs/tags/v1.0.0rc1``, or a branch for a manual run),
EVENT the GitHub event (``push`` or ``workflow_dispatch``), PUBLISH_TESTPYPI ``true`` when
a manual run asked for TestPyPI. Prints ``version=``, ``tag=``, and ``index=`` lines (for
$GITHUB_OUTPUT); ``index`` is ``testpypi``, ``pypi``, or ``none`` (build and verify only).

A tag must be annotated, be ``v`` plus the package's version exactly (``vX.Y.Z`` or
``vX.Y.ZrcN``), and name a version the index does not have. A release candidate goes to
TestPyPI only; a release (1.0.0 or later) goes to PyPI only, from a commit on main,
with its CHANGELOG section. A manual run may publish only a pre-release (``rcN`` or
``.devN``), and only to TestPyPI. Standard library only.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "ssmjviser"
TAG = re.compile(r"^v(\d+\.\d+\.\d+)(rc\d+)?$")
INDEXES = {"testpypi": "https://test.pypi.org", "pypi": "https://pypi.org"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _published(index: str, version: str) -> bool:
    url = f"{INDEXES[index]}/pypi/{NAME}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise SystemExit(f"cannot tell whether {index} has {NAME} {version}: HTTP {e.code}") from e


def decide(ref: str, event: str, publish_testpypi: bool) -> dict[str, str]:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    if not ref.startswith("refs/tags/"):
        if event != "workflow_dispatch":
            raise SystemExit(f"a release runs for a tag or a manual run, not {event} on {ref}")
        index = "testpypi" if publish_testpypi else "none"
        if index == "testpypi" and not re.search(r"(rc|\.dev)\d+$", version):
            raise SystemExit(f"a manual run publishes only a pre-release to TestPyPI, not {version}")
        if index == "testpypi" and _published(index, version):
            raise SystemExit(f"TestPyPI already has {NAME} {version}; bump the version")
        return {"version": version, "tag": "", "index": index}

    tag = ref.removeprefix("refs/tags/")
    match = TAG.match(tag)
    if match is None:
        raise SystemExit(f"tag {tag} is not vX.Y.Z or vX.Y.ZrcN")
    if tag != f"v{version}":
        raise SystemExit(f"tag {tag} does not match the package's version {version}")
    if _git("cat-file", "-t", f"refs/tags/{tag}") != "tag":
        raise SystemExit(f"tag {tag} is not annotated; tag with git tag -a")
    index = "testpypi" if match.group(2) else "pypi"
    if index == "pypi":
        if int(match.group(1).split(".")[0]) < 1:
            raise SystemExit(f"tag {tag}: mj_viser is published to PyPI from 1.0.0 on")
        commit = _git("rev-list", "-n", "1", f"refs/tags/{tag}")
        if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT).returncode:
            raise SystemExit(f"tag {tag} is not on main")
        section = rf"^## {re.escape(version)} — \d{{4}}-\d{{2}}-\d{{2}}$"
        if not re.search(section, (ROOT / "CHANGELOG.md").read_text(), re.M):
            raise SystemExit(f"CHANGELOG.md has no '## {version} — YYYY-MM-DD' section")
    if _published(index, version):
        raise SystemExit(f"{index} already has {NAME} {version}; versions cannot be replaced")
    return {"version": version, "tag": tag, "index": index}


def main(ref: str, event: str, publish_testpypi: str = "false") -> None:
    for key, value in decide(ref, event, publish_testpypi == "true").items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main(*sys.argv[1:])
