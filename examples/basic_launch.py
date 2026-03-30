"""Basic example: load any MJCF model and launch the interactive viewer.

Usage:
    uv run python examples/basic_launch.py path/to/model.xml

    # With MuJoCo Menagerie (clone it first):
    # git clone https://github.com/google-deepmind/mujoco_menagerie.git
    uv run python examples/basic_launch.py mujoco_menagerie/universal_robots_ur5e/scene.xml
    uv run python examples/basic_launch.py mujoco_menagerie/franka_emika_panda/scene.xml
"""

from __future__ import annotations

import sys

import mujoco

from mj_viser import MujocoViewer


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.xml>")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(sys.argv[1])
    data = mujoco.MjData(model)

    viewer = MujocoViewer(model, data)
    viewer.launch()


if __name__ == "__main__":
    main()
