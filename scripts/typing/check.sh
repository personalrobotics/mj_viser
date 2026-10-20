#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa
#
# A downstream project's strict type check against an installed mj_viser wheel.
#
# Usage: scripts/typing/check.sh WHEEL [REPORT_JSON]
#
# Installs WHEEL and mypy into a fresh virtualenv, outside the checkout, then runs
# mypy --strict on consumer.py (must pass: mj_viser's own types, through py.typed) and on
# misuse.py (must report exactly its marked errors: those types are not Any).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
wheel="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
report="${2:-}"
mypy_version="1.19.1"  # uv.lock's
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv venv --quiet --python "${UV_PYTHON:-3.12}" "$work/venv"
uv pip install --quiet --python "$work/venv/bin/python" "$wheel" "mypy==$mypy_version"
cp "$here/consumer.py" "$here/misuse.py" "$here/mypy.ini" "$work/"
cd "$work"

venv/bin/mypy --config-file mypy.ini --no-incremental consumer.py

expected="$(grep -n '# expect:' misuse.py | sed -E 's/^([0-9]+):.*# expect: (.*)$/misuse.py:\1: \2/' | sort)"
set +e
venv/bin/mypy --config-file mypy.ini --no-incremental --show-error-codes misuse.py > misuse.out
set -e
actual="$(sed -nE 's/^(misuse\.py:[0-9]+): error: .*\[([a-z-]+)\]$/\1: \2/p' misuse.out | sort)"
if [ "$actual" != "$expected" ]; then
  echo "misuse.py: expected errors:"; echo "$expected"
  echo "got:"; cat misuse.out
  exit 1
fi
echo "typed consumer: OK (mypy $mypy_version --strict; $(echo "$expected" | wc -l | tr -d ' ') misuses rejected)"
if [ -n "$report" ]; then
  printf '{"mypy": "%s", "consumer": "passed", "misuses_rejected": %s, "wheel": "%s"}\n' \
    "$mypy_version" "$(echo "$expected" | wc -l | tr -d ' ')" "$(basename "$wheel")" > "$report"
fi
