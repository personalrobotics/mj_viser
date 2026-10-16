# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Quick demo: every primitive geom type, one per geom group, falling under gravity.

Run with: uv run python examples/gui_demo.py
Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import time

import mujoco

from mj_viser import MujocoViewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="box" pos="0 0 0.5">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1" group="0"/>
    </body>
    <body name="sphere" pos="0.5 0 0.5">
      <joint type="free"/>
      <geom type="sphere" size="0.08" rgba="0.2 0.8 0.2 1" group="1"/>
    </body>
    <body name="capsule" pos="-0.5 0 0.5">
      <joint type="free"/>
      <geom type="capsule" size="0.05 0.15" rgba="0.2 0.2 0.8 1" group="2"/>
    </body>
    <body name="cylinder" pos="0 0.5 0.5">
      <joint type="free"/>
      <geom type="cylinder" size="0.06 0.1" rgba="0.8 0.8 0.2 1" group="3"/>
    </body>
    <body name="ellipsoid" pos="0 -0.5 0.5">
      <joint type="free"/>
      <geom type="ellipsoid" size="0.1 0.06 0.04" rgba="0.8 0.2 0.8 1" group="4"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    with MujocoViewer(model, port=8080) as viewer:
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
