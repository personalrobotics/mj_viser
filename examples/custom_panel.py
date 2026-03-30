"""Example: custom panel showing joint positions and simulation time.

Usage:
    uv run python examples/custom_panel.py path/to/model.xml
"""

from __future__ import annotations

import sys

import numpy as np
import viser

import mujoco

from mj_viser import MujocoViewer, PanelBase


class JointPanel(PanelBase):
    """Displays joint positions (qpos) in a text readout."""

    def name(self) -> str:
        return "Joint Positions"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder(self.name(), order=10):
            self._time_text = gui.add_text("Time", initial_value="0.000", disabled=True)
            self._qpos_text = gui.add_text("qpos", initial_value="", disabled=True)
            self._nq_text = gui.add_text(
                "nq", initial_value=str(viewer.model.nq), disabled=True
            )

    def on_sync(self, viewer: MujocoViewer) -> None:
        self._time_text.value = f"{viewer.data.time:.3f}"
        self._qpos_text.value = np.array2string(
            viewer.data.qpos, precision=3, suppress_small=True, max_line_width=60
        )


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.xml>")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(sys.argv[1])
    data = mujoco.MjData(model)

    viewer = MujocoViewer(model, data)
    viewer.add_panel(JointPanel())
    viewer.launch()


if __name__ == "__main__":
    main()
