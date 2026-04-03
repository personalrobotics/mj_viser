# mj_viser

Web-based MuJoCo viewer built on [Viser](https://github.com/nerfstudio-project/viser). Interactive simulation in the browser with extensible GUI panels for robotics applications.

<p align="center">
  <img src="assets/g1_viewer.png" alt="Unitree G1 humanoid in mj_viser viewer" width="800"/>
</p>

## Features

- **All primitive geom types** — box, sphere, cylinder, capsule, ellipsoid, and mesh
- **Textures** — PBR materials, per-face-vertex UV mapping, STL double-sided rendering
- **Two API modes** — built-in simulation loop or user-controlled sync
- **Extensible panels** — add custom GUI panels for sensors, cameras, controls, etc.
- **Sensor plots** — built-in `SensorPanel` with real-time scrolling time series (uPlot)
- **HUD overlay** — fixed-positioned status text over the 3D viewport
- **Click-to-select** — click any geom to show its body name; label follows the object
- **Visibility groups** — toggle MuJoCo geom groups on/off
- **Granular GUI control** — show/hide simulation controls and visibility panel independently
- **Teleop panel** — SE(3) gizmo + ghost hand for interactive arm control with collision feedback
- **Multi-client** — multiple browser tabs viewing the same simulation
- **Beautiful rendering** — three-point lighting, proper materials, transparency support

## Installation

```bash
git clone https://github.com/personalrobotics/mj_viser.git
cd mj_viser
pip install -e .
```

## Quick Start

### Built-in simulation loop

```python
import mujoco
from mj_viser import MujocoViewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

viewer = MujocoViewer(model, data)
viewer.launch()  # Opens browser at http://localhost:8080
```

### User-controlled loop

```python
import mujoco
from mj_viser import MujocoViewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

viewer = MujocoViewer(model, data)
viewer.launch_passive()

while viewer.is_running():
    data.ctrl[:] = my_controller(data)
    mujoco.mj_step(model, data)
    viewer.sync()
```

### Sensor plots

```python
from mj_viser import MujocoViewer, SensorPanel, SensorChannel

viewer = MujocoViewer(model, data)
viewer.add_panel(SensorPanel(
    title="Wrist F/T",
    channels=[
        SensorChannel(0, "Fx", "#e74c3c"),
        SensorChannel(1, "Fy", "#2ecc71"),
        SensorChannel(2, "Fz", "#3498db"),
    ],
    window_seconds=5.0,
))
viewer.launch()
```

### HUD overlay

```python
viewer.set_hud("status", "L: [13N] can_0 | R: — | physics", "bottom-left")
```

### Click-to-select

```python
# Labels appear automatically on click. Register a callback for custom behavior:
viewer.on_select(lambda geom_id, body_name: print(f"Selected: {body_name}"))
```

### Custom panels

```python
from mj_viser import MujocoViewer, PanelBase
import viser

class MyPanel(PanelBase):
    def name(self) -> str:
        return "My Panel"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        with gui.add_folder(self.name()):
            self._text = gui.add_text("Status", initial_value="", disabled=True)

    def on_sync(self, viewer: MujocoViewer) -> None:
        self._text.value = f"Time: {viewer.data.time:.2f}"

viewer = MujocoViewer(model, data)
viewer.add_panel(MyPanel())
viewer.launch()
```

### Teleop panel

```python
from mj_viser import MujocoViewer, TeleopPanel
from mj_manipulator.teleop import TeleopController

controller = TeleopController(arm, ctx)
panel = TeleopPanel(
    arm=arm, controller=controller,
    model=model, data=data,
    gripper_body_prefix="ur5e/gripper/",
    arm_label="Right Arm",
)
viewer.add_panel(panel)
viewer.launch()
# Teleop tab: activate gizmo, drag to control arm, toggle gripper, record demos
```

### Granular GUI control

```python
# Disable built-in panels for apps that manage their own GUI
viewer = MujocoViewer(model, data, show_sim_controls=False, show_visibility=False)
```

## Examples

```bash
# Clone MuJoCo Menagerie for robot models
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Interactive viewer
uv run python examples/basic_launch.py mujoco_menagerie/universal_robots_ur5e/scene.xml

# User-controlled loop
uv run python examples/sync_mode.py mujoco_menagerie/franka_emika_panda/scene.xml

# Custom panel demo
uv run python examples/custom_panel.py mujoco_menagerie/unitree_g1/scene.xml
```

## Development

```bash
git clone https://github.com/personalrobotics/mj_viser.git
cd mj_viser
uv sync --all-extras
uv run pytest
uv run ruff check src/ tests/
```

## License

MIT
