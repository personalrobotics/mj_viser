#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa
#
# Build the mj_viser wheel, install it with only its declared dependencies into
# a fresh virtualenv, and run the end-to-end check scripts/smoke_wheel.py against it.
#
# Usage: scripts/smoke_wheel.sh [OUT_DIR]   (default: dist/smoke)
# Artifacts: OUT_DIR/smoke_report.json (results, input-state hashes, wheel sha256)
#            OUT_DIR/two_states.viser (Viser recording of the two rendered states)
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mkdir -p "${1:-$repo/dist/smoke}" && cd "${1:-$repo/dist/smoke}" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv build --wheel --out-dir "$work/dist" "$repo" >/dev/null
wheel="$(ls "$work"/dist/mj_viser-*.whl)"

uv venv --quiet "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$wheel"

# -I: isolated mode, so nothing from the repo checkout shadows the installed wheel.
(cd "$work" && "$work/venv/bin/python" -I "$repo/scripts/smoke_wheel.py" "$out")

sha="$(shasum -a 256 "$wheel" | cut -d' ' -f1)"
"$work/venv/bin/python" -I - "$out/smoke_report.json" "$(basename "$wheel")" "$sha" <<'EOF'
import json, sys
path, name, sha = sys.argv[1:]
report = json.load(open(path))
report["wheel"] = {"file": name, "sha256": sha}
json.dump(report, open(path, "w"), indent=2, sort_keys=True)
EOF
cat "$out/smoke_report.json"
