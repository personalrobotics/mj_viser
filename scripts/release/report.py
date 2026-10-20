# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The release report: what a release run verified, and exactly what it will upload.

Usage: python scripts/release/report.py ARTIFACTS_DIR OUT_JSON

ARTIFACTS_DIR holds the run's artifacts, one directory each (as actions/download-artifact
writes them). Reads VERSION, TAG, and INDEX (the guard's decision) and GitHub's run
variables from the environment. Refuses unless every clean-wheel check installed the wheel
that will be uploaded, the sdist check installed the sdist that will be uploaded, and the
typed-consumer and ssrobot checks reported. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX_URLS = {"testpypi": "https://test.pypi.org/project/mj-viser/", "pypi": "https://pypi.org/project/mj-viser/"}


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    data: dict[str, object] = json.loads(path.read_text())
    return data


def main(artifacts_dir: str, out: str) -> None:
    root = Path(artifacts_dir)
    version, tag, index = os.environ["VERSION"], os.environ.get("TAG", ""), os.environ["INDEX"]
    sums = dict(line.split()[::-1] for line in (root / "dist" / "SHA256SUMS").read_text().splitlines())
    wheel, sdist = f"mj_viser-{version}-py3-none-any.whl", f"mj_viser-{version}.tar.gz"
    for name in (wheel, sdist):
        if hashlib.sha256((root / "dist" / name).read_bytes()).hexdigest() != sums.get(name):
            raise SystemExit(f"{name} does not match SHA256SUMS")

    wheel_checks = []
    for report in sorted(root.glob("e2e-py*/*/smoke_report.json")):
        r = _load(report)
        installed = r["wheel"]
        assert isinstance(installed, dict)
        if installed != {"file": wheel, "sha256": sums[wheel]}:
            raise SystemExit(f"{report} checked {installed}, not the wheel to upload")
        deps = r["installed"]
        assert isinstance(deps, dict)
        wheel_checks.append(
            {
                "environment": r["environment"],
                "python": r["python"],
                "installed": {k: deps[k] for k in ("mujoco", "viser", "numpy", "trimesh", "pillow") if k in deps},
                "readme_quick_start": r["readme_quick_start"],
            }
        )
    if not wheel_checks:
        raise SystemExit("no clean-wheel check reported")

    sdist_report = _load(root / "sdist-e2e" / "locked" / "smoke_report.json")
    if sdist_report["wheel"] != {"file": sdist, "sha256": sums[sdist]}:
        raise SystemExit(f"the sdist check installed {sdist_report['wheel']}, not the sdist to upload")

    ssrobot = {}
    for lane, path in (
        ("live", "ssrobot-live/ssrobot_live.json"),
        ("replay", "ssrobot-replay/ssrobot_replay.json"),
        ("preview", "ssrobot-preview/ssrobot_preview.json"),
        ("contacts", "ssrobot-preview/contacts/ssrobot_contacts.json"),
    ):
        lane_report = _load(root / path)
        ssrobot[lane] = lane_report.get("versions", "reported")

    api = (ROOT / "docs" / "api.md").read_text()
    server, repo, run = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    report = {
        "tag": tag,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "version": version,
        "run": f"{server}/{repo}/actions/runs/{run}" if run else "",
        "public_api": {
            "document": "docs/api.md",
            "sha256": hashlib.sha256(api.encode()).hexdigest(),
            "names": re.findall(r"^- `(mj_viser\.[\w.]+)`$", api, re.M),
        },
        "artifacts": {name: sums[name] for name in (wheel, sdist)},
        "wheel_checks": wheel_checks,
        "sdist_check": {"python": sdist_report["python"], "environment": sdist_report["environment"]},
        "typed_consumer": _load(root / "typed-consumer" / "typed-consumer.json"),
        "ssrobot": ssrobot,
        "index": index,
        "index_url": INDEX_URLS[index] + f"{version}/" if index in INDEX_URLS else "",
    }
    Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("tag", "version", "index", "artifacts")}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
