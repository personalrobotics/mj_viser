# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Smoke test for an installed mj_viser wheel. Run by scripts/smoke_wheel.sh.

Imports every mj_viser module, opens the viewer on a small model, renders a
frame, closes it, and writes a JSON report to the path given as argv[1].
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pkgutil
import socket
import sys
from importlib.metadata import version

import mujoco

import mj_viser

MJCF = """
<mujoco>
  <worldbody>
    <body name="link" pos="0 0 0.5">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" rgba="0.8 0.3 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main(report_path: str) -> None:
    modules = sorted(m.name for m in pkgutil.walk_packages(mj_viser.__path__, "mj_viser."))
    for name in modules:
        importlib.import_module(name)

    forbidden = [m for m in sys.modules if m.split(".")[0] in {"mj_manipulator"}]
    if forbidden:
        raise SystemExit(f"mj_viser imported robot-control modules: {forbidden}")

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    viewer = mj_viser.MujocoViewer(model, data, host="127.0.0.1", port=free_port(), show_gui=True)
    viewer.launch_passive(open_browser=False)
    viewer.sync()
    viewer.close()

    report = {
        "python": sys.version.split()[0],
        "mj_viser": version("mj-viser"),
        "mujoco": version("mujoco"),
        "viser": version("viser"),
        "mj_manipulator_installed": importlib.util.find_spec("mj_manipulator") is not None,
        "modules_imported": modules,
        "public_api": sorted(mj_viser.__all__),
        "viewer_opened_and_closed": True,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main(sys.argv[1])
