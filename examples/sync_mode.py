"""Example: user-controlled simulation loop with sync().

Demonstrates how to run your own control logic and push state to the viewer.

Usage:
    uv run python examples/sync_mode.py path/to/model.xml
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

    viewer = MujocoViewer(model, data)
    viewer.launch_passive()

    print("Running user-controlled loop. Ctrl+C to stop.")
    dt = model.opt.timestep
    try:
        while viewer.is_running():
            # Your control logic here — e.g., set data.ctrl[:] = ...
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
