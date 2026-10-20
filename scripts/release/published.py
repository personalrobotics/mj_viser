# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""After an upload: the index serves exactly the files the run verified.

Usage: python scripts/release/published.py INDEX VERSION DIST_DIR OUT_JSON

INDEX is ``testpypi`` or ``pypi``; DIST_DIR holds the verified files and SHA256SUMS. Waits,
a bounded time, for the index's JSON API to list the version with both files (a new
version can take minutes to appear), then requires each file's digest there, and the
sha256 of the wheel's bytes downloaded from the index, to equal SHA256SUMS. Standard
library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = {"testpypi": "https://test.pypi.org/pypi", "pypi": "https://pypi.org/pypi"}
ATTEMPTS, PAUSE_S = 20, 30  # up to ten minutes
PROJECT, DIST_NAME = "ssmjviser", "ssmjviser"


def _listing(index: str, version: str, names: set[str]) -> dict[str, dict[str, str]]:
    url = f"{API[index]}/{PROJECT}/{version}/json"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                files = {f["filename"]: f for f in json.load(response)["urls"]}
            if names <= files.keys():
                return {n: {"sha256": files[n]["digests"]["sha256"], "url": files[n]["url"]} for n in names}
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
        print(f"{index} does not list {PROJECT} {version} with both files yet ({attempt}/{ATTEMPTS})", flush=True)
        if attempt < ATTEMPTS:
            time.sleep(PAUSE_S)
    raise SystemExit(f"{index} did not list {PROJECT} {version} with {sorted(names)} in time")


def main(index: str, version: str, dist_dir: str, out: str) -> None:
    dist = Path(dist_dir)
    sums = dict(line.split()[::-1] for line in (dist / "SHA256SUMS").read_text().splitlines())
    wheel, sdist = f"{DIST_NAME}-{version}-py3-none-any.whl", f"{DIST_NAME}-{version}.tar.gz"
    listed = _listing(index, version, {wheel, sdist})
    for name in (wheel, sdist):
        if listed[name]["sha256"] != sums[name]:
            raise SystemExit(f"{index} has {name} with sha256 {listed[name]['sha256']}, not {sums[name]}")
    with urllib.request.urlopen(listed[wheel]["url"], timeout=60) as response:
        served = hashlib.sha256(response.read()).hexdigest()
    if served != sums[wheel]:
        raise SystemExit(f"{index} served {wheel} with sha256 {served}, not {sums[wheel]}")
    result = {"index": index, "version": version, "files": {n: listed[n]["sha256"] for n in (wheel, sdist)}}
    Path(out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
