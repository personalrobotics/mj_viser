# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The installed package: every module imports, with no robot control, stepping, or private Viser."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import pkgutil
import sys
import tokenize

import mj_viser


def check_package() -> list[str]:
    """Import every mj_viser module and check its source; returns the module names."""
    modules = sorted(m.name for m in pkgutil.walk_packages(mj_viser.__path__, "mj_viser."))
    for name in modules:
        importlib.import_module(name)
    forbidden = [m for m in sys.modules if m.split(".")[0] == "mj_manipulator"]
    if forbidden:
        raise SystemExit(f"mj_viser imported robot-control modules: {forbidden}")

    # The viewer code (not its docstrings) must contain no stepping, reset, or worker thread of its own.
    # Nor may it reach into Viser's private modules or transport (viser._x, ._websock_*).
    names = set()
    private_viser = []
    for m in modules:
        src = io.StringIO(inspect.getsource(sys.modules[m]))
        toks = [t.string for t in tokenize.generate_tokens(src.readline) if t.type in (tokenize.NAME, tokenize.OP)]
        names |= set(toks)
        private_viser += [
            f"{m}: viser.{c}"
            for a, b, c in zip(toks, toks[1:], toks[2:])
            if a == "viser" and b == "." and c.startswith("_")
        ]
        private_viser += [f"{m}: {t}" for t in toks if t.startswith("_websock")]
    banned = sorted(names & {"mj_step", "mj_resetData", "Thread"})
    if banned:
        raise SystemExit(f"viewer lifecycle contains {banned}")
    if private_viser:
        raise SystemExit(f"private Viser access: {private_viser}")
    if any(hasattr(mj_viser.MujocoViewer, a) for a in ("launch", "launch_passive", "sync", "is_running")):
        raise SystemExit("legacy launch/sync API still present")

    return modules
