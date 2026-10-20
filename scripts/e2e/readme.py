# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The README's quick start, run exactly as written against the installed package.

The block after the README's quick-start marker is written to a file and run in a fresh
interpreter. While it runs, a browser connects to the URL it prints and must receive the
pendulum's scene; then the program must exit cleanly on its own. The quick start serves on
a free port, so the check needs none of its own, and its server is gone when it returns.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import msgspec
from websockets.sync.client import connect

from e2e.channels import CAMERA, _frames

README = Path(__file__).resolve().parents[2] / "README.md"
POSE = {"FrameMessage", "SetPositionMessage", "SetOrientationMessage", "SetSceneNodeVisibilityMessage"}
MARKER = "<!-- The block below is run as written by the clean-wheel check (scripts/e2e/readme.py). -->"


def quick_start() -> str:
    text = README.read_text()
    if MARKER not in text:
        raise SystemExit("the README has no quick-start marker")
    block = re.search(r"```python\n(.*?)```", text[text.index(MARKER) :], re.S)
    if block is None:
        raise SystemExit("no python block follows the README's quick-start marker")
    return block.group(1)


def _listening(url: str) -> bool:
    parts = urlsplit(url)
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((parts.hostname or "127.0.0.1", parts.port or 80)) == 0


def check_quick_start() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "quick_start.py"
        script.write_text(quick_start())
        errors = Path(tmp) / "stderr.txt"
        with errors.open("w") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-I", "-u", str(script)], cwd=tmp, stdout=subprocess.PIPE, stderr=stderr, text=True
            )
        try:
            assert process.stdout is not None
            line = ""
            while not line.startswith("Open "):
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    raise SystemExit(f"the quick start exited before serving: {errors.read_text()}")
            url = line.split()[-1]
            ws = connect(url.replace("http://", "ws://"), subprotocols=[f"viser-v{version('viser')}"], max_size=None)
            try:
                ws.send(msgspec.msgpack.encode(CAMERA))
                frames = _frames(ws, 2.0)
            finally:
                ws.close()
            # The pendulum's floor, rod, and bob, under the model's root; and the bob swinging.
            meshes = [m for m in frames if str(m.get("name", "")).startswith("/mujoco/") and m["type"] not in POSE]
            moved = [m for m in frames if str(m.get("name", "")).startswith("/mujoco/") and m["type"] in POSE]
            if len(meshes) < 3 or len(moved) < 10:
                raise SystemExit(f"the quick start's browser saw {len(meshes)} geometry and {len(moved)} pose messages")
            start = time.monotonic()
            process.stdout.read()  # drain to exit
            code = process.wait(timeout=120)
            if code != 0:
                raise SystemExit(f"the quick start exited {code}: {errors.read_text()}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        if _listening(url):  # the viewer released its port when the program ended
            raise SystemExit(f"something still serves {url} after the quick start exited")
    return {
        "url": url,
        "geometry_messages": len(meshes),
        "pose_messages_in_2s": len(moved),
        "exit": code,
        "ran_after_browser_s": round(time.monotonic() - start, 1),
        "port_released": True,
    }
