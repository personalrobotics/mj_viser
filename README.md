# mj_viser

Web-based MuJoCo viewer built on [Viser](https://github.com/nerfstudio-project/viser). Renders caller-supplied MuJoCo state in the browser, with extensible GUI panels for robotics applications.

mj_viser renders MuJoCo state; it does not step simulations or command robots. Your code owns the model state, the clock, and when to render. Teleoperation, task controls, arm ownership, safety policy, and recording belong to the application (for example Geodude or ADA), which decides what to execute.

<p align="center">
  <img src="assets/g1_viewer.png" alt="Unitree G1 humanoid in mj_viser viewer" width="800"/>
</p>

## Features

- **All primitive geom types** — box, sphere, cylinder, capsule, ellipsoid, and mesh
- **Textures** — PBR materials, per-face-vertex UV mapping, STL double-sided rendering
- **Caller-driven lifecycle** — `open()`, `update(data)`, `close()`; the viewer copies state and never mutates yours
- **Extensible panels** — add custom GUI panels for sensors, cameras, controls, etc.
- **Sensor plots** — built-in `SensorPanel` with real-time scrolling time series (uPlot)
- **HUD overlay** — fixed-positioned status text over the 3D viewport
- **Click-to-select** — click any geom to show its body name; label follows the object
- **Visibility groups** — toggle MuJoCo geom groups on/off
- **Loopback by default** — binds `127.0.0.1`; serving other machines is an explicit `host=` choice
- **Multi-client** — multiple browser tabs viewing the same simulation
- **Beautiful rendering** — three-point lighting, proper materials, transparency support

## Installation

```bash
git clone https://github.com/personalrobotics/mj_viser.git
cd mj_viser
pip install -e .
```

## Quick Start

### Render your own loop

```python
import mujoco
from mj_viser import MujocoViewer

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

viewer = MujocoViewer(model)       # host="127.0.0.1", port=8080 by default; port=0 picks a free port
url = viewer.open()                # raises OSError if the address cannot be bound
print(url)

while running:
    data.ctrl[:] = my_controller(data)
    mujoco.mj_step(model, data)    # you step; the viewer never does
    viewer.update(data)            # copies data and renders the copy

viewer.close()                     # releases the port; open() again to reuse the viewer
```

`with MujocoViewer(model) as viewer:` opens on entry and closes on exit. `update()` recomputes kinematics on its copy, so setting `qpos` alone (for example when replaying a trace) is enough. Panels read that copy as `viewer.data`.

### Sensor plots

```python
from mj_viser import MujocoViewer, SensorPanel, SensorChannel

viewer = MujocoViewer(model)
viewer.add_panel(SensorPanel(
    title="Wrist F/T",
    channels=[
        SensorChannel(0, "Fx", "#e74c3c"),
        SensorChannel(1, "Fy", "#2ecc71"),
        SensorChannel(2, "Fz", "#3498db"),
    ],
    window_seconds=5.0,
))
viewer.open()
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

viewer = MujocoViewer(model)
viewer.add_panel(MyPanel())
viewer.open()
```

### Hide the visibility panel

```python
# For apps that manage their own GUI
viewer = MujocoViewer(model, show_visibility=False)
```

## Compatibility

| | Supported |
| --- | --- |
| Python | 3.10–3.12 |
| MuJoCo | `>=3.5` (first release with `mj_copyData` in the Python bindings) |
| Viser | `>=1.1.1,<1.2` |
| NumPy | `>=1.24` |
| trimesh | `>=3.21.7` |

CI runs the same end-to-end check from a clean wheel install against the locked versions (`uv.lock`) and the newest allowed releases on every supported Python, and against every declared minimum on Python 3.10.

## Examples

```bash
# Clone MuJoCo Menagerie for robot models
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Step a model and render each step
uv run python examples/basic_launch.py mujoco_menagerie/universal_robots_ur5e/scene.xml

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
scripts/smoke_wheel.sh dist/smoke lowest locked latest   # end-to-end check from a clean wheel install, per environment
```

## License

MIT
