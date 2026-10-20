#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa
#
# The release gate, locally: what the release workflow checks before it publishes, on one
# build, before you push a tag. Needs uv, and port 8080 free (the README's quick start).
#
# Usage: scripts/release-precheck.sh [TAG]     e.g. scripts/release-precheck.sh v1.0.0rc1
#
# Lint and types; build the sdist and wheel once; the README as PyPI shows it; a strict
# downstream type check; the clean-wheel end-to-end check of the wheel and of the sdist
# (locked versions). With TAG (created first, annotated), also the workflow's guard: the
# tag matches the version, and the index it would publish to does not have that version.
# The ssrobot lanes run in CI only.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "[precheck] lint and types"
uv run ruff check .
uv run ruff format --check .
uv run mypy src

if [ -n "${1:-}" ]; then
  echo "[precheck] guard for $1"
  uv run --no-project --python 3.12 python scripts/release/guard.py "refs/tags/$1" push
fi

echo "[precheck] build once"
uv build --out-dir "$work/dist" >/dev/null
wheel="$(ls "$work"/dist/mj_viser-*.whl)"
sdist="$(ls "$work"/dist/mj_viser-*.tar.gz)"
(cd "$work/dist" && shasum -a 256 mj_viser-*)

echo "[precheck] README as PyPI shows it"
uv run --no-project --with "readme-renderer[md]==44.0" python scripts/check_readme.py "$work/dist" . >/dev/null

echo "[precheck] strict downstream type check"
scripts/typing/check.sh "$wheel"

echo "[precheck] clean-wheel end-to-end check: wheel, then sdist"
MJ_VISER_WHEEL="$wheel" UV_PYTHON="${UV_PYTHON:-3.12}" scripts/smoke_wheel.sh "$work/e2e-wheel" locked
MJ_VISER_WHEEL="$sdist" UV_PYTHON="${UV_PYTHON:-3.12}" scripts/smoke_wheel.sh "$work/e2e-sdist" locked

echo "[precheck] all green: $(basename "$wheel"), $(basename "$sdist")"
