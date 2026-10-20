# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""After publication: the installed mj-viser runs the README's quick start.

Usage: python -I scripts/release/installed.py RELEASE_REPORT_JSON

Run with the interpreter of a fresh environment that installed mj-viser from the index.
Checks that the installed package is the released version, runs the quick start check
(scripts/e2e/readme.py), and adds the result to the release report.
"""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.readme import check_quick_start  # noqa: E402


def main(report_path: str) -> None:
    report = json.loads(Path(report_path).read_text())
    dist = distribution("mj-viser")
    if dist.version != report["version"]:
        raise SystemExit(f"installed mj-viser {dist.version}, not {report['version']}")
    report["installed_from_index"] = {
        "index": os.environ.get("INDEX", report["index"]),
        "version": dist.version,
        "location": str(dist.locate_file("")),
        "python": sys.version.split()[0],
        "quick_start": check_quick_start(),
    }
    Path(report_path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["installed_from_index"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
