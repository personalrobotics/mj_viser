# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""After publication: the installed ssmjviser runs the README's quick start.

Usage: python -I scripts/release/installed.py RELEASE_REPORT_JSON PUBLISHED_JSON

Run with the interpreter of a fresh environment that installed ssmjviser from the index.
Checks that the installed package is the released version, runs the quick start check
(scripts/e2e/readme.py), and adds it, with PUBLISHED_JSON (scripts/release/published.py:
the index serves exactly the verified files), to the release report.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.readme import check_quick_start  # noqa: E402


def main(report_path: str, published_path: str) -> None:
    report = json.loads(Path(report_path).read_text())
    published = json.loads(Path(published_path).read_text())
    if published["version"] != report["version"] or published["files"] != report["artifacts"]:
        raise SystemExit(f"the index's files {published} are not the report's {report['artifacts']}")
    dist = distribution("ssmjviser")
    if dist.version != report["version"]:
        raise SystemExit(f"installed ssmjviser {dist.version}, not {report['version']}")
    report["installed_from_index"] = {
        "index": os.environ.get("INDEX", report["index"]),
        "version": dist.version,
        "location": str(dist.locate_file("")),
        "python": sys.version.split()[0],
        "quick_start": check_quick_start(),
        "index_files": published["files"],
    }
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["installed_from_index"], indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
