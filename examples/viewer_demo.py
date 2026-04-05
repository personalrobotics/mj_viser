# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Demo: MujocoViewer with built-in sim loop and a custom joint readout panel.

Run with: uv run python examples/viewer_demo.py
Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import numpy as np
import viser

from mj_viser import MujocoViewer, PanelBase

try:
    import mujoco
except ImportError:
    raise SystemExit("mujoco is required: pip install mujoco")

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="box" pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
    <body name="sphere" pos="0.4 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.08" rgba="0.2 0.8 0.2 1"/>
    </body>
    <body name="capsule" pos="-0.4 0 1">
      <joint type="free"/>
      <geom type="capsule" size="0.05 0.15" rgba="0.2 0.2 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


class SimInfoPanel(PanelBase):
    """Custom panel showing simulation time and body positions."""

    def name(self) -> str:
        return "Sim Info"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder(self.name(), order=10):
            self._time_text = gui.add_text("Time", initial_value="0.000", disabled=True)
            self._pos_text = gui.add_text("Box pos", initial_value="", disabled=True)

    def on_sync(self, viewer: MujocoViewer) -> None:
        self._time_text.value = f"{viewer.data.time:.3f}"
        pos = viewer.data.xpos[1]  # body 1 = box
        self._pos_text.value = np.array2string(pos, precision=3, suppress_small=True)


def main() -> None:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    viewer = MujocoViewer(model, data, port=8080)
    viewer.add_panel(SimInfoPanel())
    viewer.launch()


if __name__ == "__main__":
    main()
