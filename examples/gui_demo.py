"""Quick demo: loads a simple MuJoCo scene and shows the GUI controls + rendered geoms.

Run with: uv run python examples/gui_demo.py
Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import time

import mujoco
import viser

from mj_viser.gui import GuiManager
from mj_viser.scene import SceneManager

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
    mujoco.mj_forward(model, data)

    server = viser.ViserServer(port=8080)
    scene_mgr = SceneManager(server, model, data)
    scene_mgr.build_scene()
    gui_mgr = GuiManager(server, model)

    print("Open http://localhost:8080 — try Play/Pause/Step/Reset and visibility toggles")

    dt = model.opt.timestep
    prev_groups: set[int] = set()

    while True:
        # Handle reset
        if gui_mgr.should_reset:
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

        # Handle play / step
        if gui_mgr.is_playing or gui_mgr.should_step:
            mujoco.mj_step(model, data)

        # Update scene
        scene_mgr.update_transforms()

        # Update visibility if changed
        groups = gui_mgr.visible_groups()
        if groups != prev_groups:
            scene_mgr.update_visibility(groups)
            prev_groups = groups

        time.sleep(dt / gui_mgr.speed)


if __name__ == "__main__":
    main()
