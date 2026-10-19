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

INTEGRATIONS = "mj_viser.integrations."
# Core modules that still reach ssrobot, until #51 moves them behind the integration.
_PENDING = {"mj_viser.replay"}


def check_boundary(modules: list[str]) -> dict[str, object]:
    """The standalone lane: ssrobot is absent, core never imports it (not even lazily),
    and its integration says what it needs."""
    if importlib.util.find_spec("ssrobot") is not None:
        raise SystemExit("ssrobot is installed: the standalone lane must prove mj_viser works without it")
    core = [m for m in modules if not m.startswith(INTEGRATIONS) and m != INTEGRATIONS.rstrip(".")]
    reaching = []
    for m in core:
        src = io.StringIO(inspect.getsource(sys.modules[m]))
        toks = [
            t for t in tokenize.generate_tokens(src.readline) if t.type in (tokenize.NAME, tokenize.STRING, tokenize.OP)
        ]
        for a, b, c in zip(toks, toks[1:], toks[2:]):
            imports = a.string in ("import", "from") and b.string == "ssrobot"
            dynamic = a.string in ("import_module", "__import__") and c.string.strip("'\"").split(".")[0] == "ssrobot"
            if imports or dynamic:
                reaching.append(m)
    reaching = sorted(set(reaching) - _PENDING)
    if reaching:
        raise SystemExit(f"core modules reach ssrobot: {reaching}")
    if any(name.split(".")[0] == "ssrobot" for name in sys.modules):
        raise SystemExit("importing mj_viser imported ssrobot")
    try:
        importlib.import_module("mj_viser.integrations.ssrobot")
    except ModuleNotFoundError as e:
        if "needs ssrobot" not in str(e):
            raise SystemExit(f"the ssrobot integration fails unclearly without ssrobot: {e}")
        integration = str(e)
    else:
        raise SystemExit("the ssrobot integration imported without ssrobot")
    return {
        "ssrobot_installed": False,
        "core_modules": core,
        "pending": sorted(_PENDING),
        "integration": integration,
        "broken_installations": _broken_installations(),
    }


def _broken_installations() -> dict[str, str]:
    """A broken ssrobot is reported as it is; only a missing ssrobot (or MuJoCo extra) is
    reported as missing. Each case imports the integration over a stand-in ssrobot."""
    import subprocess
    import tempfile
    from pathlib import Path

    cases = {
        # ssrobot is there, but its own dependency is not: report that dependency.
        "missing_dependency_of_ssrobot": ("import mj_viser_test_absent_dependency\n", None),
        # ssrobot is there without its MuJoCo extra's module: report ssrobot as needed.
        "missing_mujoco_extra": ("", "ssrobot.mujoco"),
    }
    out = {}
    for case, (init, expected_name) in cases.items():
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "ssrobot").mkdir()
            (Path(root) / "ssrobot" / "__init__.py").write_text(init)
            probe = (
                f"import sys; sys.path.insert(0, {root!r})\n"
                "try:\n    import mj_viser.integrations.ssrobot\n"
                "except ModuleNotFoundError as e:\n    print(repr(e.name)); print(e)\n"
            )
            lines = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.splitlines()
        name = eval(lines[0]) if lines else None  # noqa: S307 - the repr printed just above
        message = lines[1] if len(lines) > 1 else ""
        if case == "missing_dependency_of_ssrobot":
            if name != "mj_viser_test_absent_dependency" or "needs ssrobot" in message:
                raise SystemExit(f"a broken ssrobot was reported as {name!r}: {message!r}")
        elif name != expected_name or "needs ssrobot" not in message:
            raise SystemExit(f"ssrobot without its MuJoCo extra was reported as {name!r}: {message!r}")
        out[case] = message
    return out


def check_package() -> list[str]:
    """Import every mj_viser core module and check its source; returns the module names."""
    modules = sorted(
        m.name for m in pkgutil.walk_packages(mj_viser.__path__, "mj_viser.") if not m.name.startswith(INTEGRATIONS)
    )
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
