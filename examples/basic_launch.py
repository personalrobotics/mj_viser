# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Basic example: load any MJCF model, step it yourself, and render each step.

Usage:
    uv run python examples/basic_launch.py path/to/model.xml

    # With MuJoCo Menagerie (clone it first):
    # git clone https://github.com/google-deepmind/mujoco_menagerie.git
    uv run python examples/basic_launch.py mujoco_menagerie/universal_robots_ur5e/scene.xml
    uv run python examples/basic_launch.py mujoco_menagerie/franka_emika_panda/scene.xml
"""

from __future__ import annotations

import sys
import time

import mujoco

from mj_viser import MujocoViewer


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.xml>")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(sys.argv[1])
    data = mujoco.MjData(model)

    with MujocoViewer(model) as viewer:
        print(f"Viewer at {viewer.url}. Ctrl+C to stop.")
        try:
            while True:
                # Your control logic here, e.g. data.ctrl[:] = ...
                mujoco.mj_step(model, data)
                viewer.update(data)
                time.sleep(model.opt.timestep)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
