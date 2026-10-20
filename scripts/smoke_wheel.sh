#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa
#
# Build the mj_viser wheel once (or take MJ_VISER_WHEEL=path/to/wheel, as CI does
# to test the exact artifact it uploads), install it into one fresh virtualenv per
# environment, and run the end-to-end check (scripts/e2e/run.py) in each.
#
# Usage: [MJ_VISER_WHEEL=...] scripts/smoke_wheel.sh [OUT_DIR] [ENV...]   (default: dist/smoke latest)
# ENV:
#   latest  newest releases allowed by pyproject.toml (what `pip install` gets today)
#   locked  exact versions from uv.lock
#   lowest  every direct dependency at its declared minimum, on Python 3.11
# Artifacts, per ENV:
#   OUT_DIR/ENV/smoke_report.json  results, input-state hashes, installed versions, wheel sha256
#   OUT_DIR/ENV/two_states.viser   Viser recording of the two rendered states
# With MJ_VISER_ROBOT_XML=a/scene.xml:b/robot.xml (colon-separated), also per ENV and robot:
#   OUT_DIR/ENV/robot_<model>.json, OUT_DIR/ENV/robot_<model>.viser   that robot at two states
set -euo pipefail
if [ -n "${MJ_VISER_ROBOT_XML:-}" ]; then
  absolute=()
  IFS=: read -ra robots <<< "$MJ_VISER_ROBOT_XML"
  for robot in "${robots[@]}"; do
    absolute+=("$(cd "$(dirname "$robot")" && pwd)/$(basename "$robot")")
  done
  MJ_VISER_ROBOT_XML="$(IFS=:; echo "${absolute[*]}")"
  export MJ_VISER_ROBOT_XML
fi

repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mkdir -p "${1:-$repo/dist/smoke}" && cd "${1:-$repo/dist/smoke}" && pwd)"
shift || true
envs=("${@:-latest}")
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -n "${MJ_VISER_WHEEL:-}" ]; then
  wheel="$(cd "$(dirname "$MJ_VISER_WHEEL")" && pwd)/$(basename "$MJ_VISER_WHEEL")"
else
  uv build --wheel --out-dir "$work/dist" "$repo" >/dev/null
  wheel="$(ls "$work"/dist/ssmjviser-*.whl)"
fi
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
      uv venv --quiet --python 3.11 "$venv"
      uv pip compile --quiet --resolution lowest-direct --python-version 3.11 --no-header \
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
  (cd "$work" && "$venv/bin/python" -I "$repo/scripts/e2e/run.py" "$out/$env" "$env")

  "$venv/bin/python" -I - "$out/$env/smoke_report.json" "$(basename "$wheel")" "$sha" <<'EOF'
import json, sys
path, name, sha = sys.argv[1:]
report = json.load(open(path))
report["wheel"] = {"file": name, "sha256": sha}
json.dump(report, open(path, "w"), indent=2, sort_keys=True)
EOF
  echo "$env: OK ($out/$env)"
done
