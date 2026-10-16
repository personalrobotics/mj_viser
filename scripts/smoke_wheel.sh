#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa
#
# Build the mj_viser wheel once, install it into one fresh virtualenv per
# environment, and run the end-to-end check scripts/smoke_wheel.py in each.
#
# Usage: scripts/smoke_wheel.sh [OUT_DIR] [ENV...]   (default: dist/smoke latest)
# ENV:
#   latest  newest releases allowed by pyproject.toml (what `pip install` gets today)
#   locked  exact versions from uv.lock
#   lowest  every direct dependency at its declared minimum, on Python 3.10
# Artifacts, per ENV:
#   OUT_DIR/ENV/smoke_report.json  results, input-state hashes, installed versions, wheel sha256
#   OUT_DIR/ENV/two_states.viser   Viser recording of the two rendered states
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mkdir -p "${1:-$repo/dist/smoke}" && cd "${1:-$repo/dist/smoke}" && pwd)"
shift || true
envs=("${@:-latest}")
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv build --wheel --out-dir "$work/dist" "$repo" >/dev/null
wheel="$(ls "$work"/dist/mj_viser-*.whl)"
sha="$(shasum -a 256 "$wheel" | cut -d' ' -f1)"

for env in "${envs[@]}"; do
  venv="$work/venv-$env"
  case "$env" in
    latest)
      uv venv --quiet "$venv"
      uv pip install --quiet --python "$venv/bin/python" "$wheel"
      ;;
    locked)
      uv venv --quiet "$venv"
      uv export --quiet --frozen --no-dev --no-emit-project --no-hashes --project "$repo" -o "$work/locked.txt"
      uv pip install --quiet --python "$venv/bin/python" -r "$work/locked.txt"
      uv pip install --quiet --python "$venv/bin/python" --no-deps "$wheel"
      ;;
    lowest)
      uv venv --quiet --python 3.10 "$venv"
      uv pip compile --quiet --resolution lowest-direct --python-version 3.10 --no-header \
        "$repo/pyproject.toml" -o "$work/lowest.txt"
      uv pip install --quiet --python "$venv/bin/python" -r "$work/lowest.txt"
      uv pip install --quiet --python "$venv/bin/python" --no-deps "$wheel"
      ;;
    *)
      echo "unknown environment: $env" >&2
      exit 2
      ;;
  esac

  mkdir -p "$out/$env"
  # -I: isolated mode, so nothing from the repo checkout shadows the installed wheel.
  (cd "$work" && "$venv/bin/python" -I "$repo/scripts/smoke_wheel.py" "$out/$env" "$env")

  "$venv/bin/python" -I - "$out/$env/smoke_report.json" "$(basename "$wheel")" "$sha" <<'EOF'
import json, sys
path, name, sha = sys.argv[1:]
report = json.load(open(path))
report["wheel"] = {"file": name, "sha256": sha}
json.dump(report, open(path, "w"), indent=2, sort_keys=True)
EOF
  echo "$env: OK ($out/$env)"
done
