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
- **HUD overlay** — fixed-positioned status text over the 3D viewport (shown as text; markup is escaped)
- **Body-frame scene** — one Viser frame per MuJoCo body, with geoms and sites beneath it at stable, unique paths
- **Click-to-select** — click any geom to show its body name, to that browser only; the label follows the object
- **Sites** — MuJoCo sites drawn as their shapes, beneath their bodies
- **Live view** — `LiveView` follows a running simulation through a read-only source, without being able to affect it
- **Channels, recordings, previews, and contacts** — `ChannelBoard` panels, `StateRecorder` recordings, `MotionPreview` ghosts, and `ContactOverlay` points, normals, and force arrows, for any MuJoCo model
- **ssrobot integration** — optional `mj_viser.integrations.ssrobot`: live view, plan preview, and run-bundle replay for ssrobot runs
- **Visibility groups** — show or hide MuJoCo geom and site groups, from the panel or `set_visible_groups()`
- **Loopback by default** — binds `127.0.0.1`; serving other machines is an explicit `host=` choice
- **Multi-client** — multiple browser tabs viewing the same simulation
- **Scene report** — `viewer.scene_report()` lists every body, geom, and site path and every element not rendered, with the reason
- **Geometry and materials** — every MuJoCo primitive, planes, and meshes, with MuJoCo's color, transparency, and 2D-texture rules (see below)

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

### Follow a running simulation

`LiveView` follows a simulation through a read-only render source (`mj_viser.RenderSource`): an object with its own `model` and `data`, a `world` identity, and `capture()`, which copies the simulation's state into `data` and returns a frame (`time_ns`, `revision`, `world`), or raises `mj_viser.SourceStale` once it can no longer describe the simulation. Any runtime that can expose a MuJoCo model and state can provide one.

```python
from mj_viser import LiveView

view = LiveView(source, max_rate_hz=30)   # source: your RenderSource
print(view.open())                        # raises SourceStale if the first frame cannot be shown
while running:
    step_my_simulation()                  # your runtime owns time and stepping
    view.refresh()                        # captures and renders, at most 30 times a second
view.close()                              # the simulation keeps running
```

- Nothing the viewer does reaches the simulation: the source holds copies.
- `refresh()` never raises because the view was closed or a browser went away, so it cannot stop your loop.
- The HUD shows the simulation's time and scene revision. `expected_world=` refuses a source for any other scene.
- Attach and detach freely with new sources. After a source goes stale, its view renders nothing more.

### Named channels and status

Give a `LiveView` a `mj_viser.ChannelBoard` and feed it: `declare` channels (a vector of values with element names, an RGB image, or unsupported), give each a `reading(name, stamp_ns, value)`, set `status` lines, note `event`s, and `advance` its time.

```python
from mj_viser import ChannelBoard

board = ChannelBoard()
board.declare("arm_q", quantity="joint_position", labels=("shoulder", "elbow"))
board.declare("wrist_rgb", kind="image")
view = LiveView(source, channels=board)
board.reading("arm_q", stamp_ns, (0.3, 0.1))
board.set_status("task", "reaching")
```

Each browser gets its own panels:
- each channel's latest reading by element name, with its time, marked `current`, `stale`, `awaiting`, `malformed`, `unavailable` (not provided), or `unsupported`;
- each image channel's latest image, sent at most once per new reading;
- the status lines and recent events;
- a plot of the channel that browser picks.

The panels only show; they hold no control of the run, and one browser's choices never reach another.

### Record states

`mj_viser.StateRecorder` writes any model's states, text, and RGB images into a native Viser recording at the times you give, for Viser's own player:

```python
from mj_viser import StateRecorder

with StateRecorder(model) as recorder:
    for time_ns, data, lines in my_states:
        recorder.advance(time_ns)
        recorder.show(data)
        recorder.text(lines)
    open("run.viser", "wb").write(recorder.finish())
```

Nothing is interpolated: each state shows from its time until the next. Play a recording with Viser's player: `viser-build-client --out-dir client/`, serve the directory (`python -m http.server`), and open `http://localhost:8000/client/?playbackPath=http://localhost:8000/run.viser`.

### Preview joint motions

`mj_viser.MotionPreview` shows named `JointMotion`s as a translucent ghost of the bodies they move. Each browser picks one and scrubs its waypoints, or its timed motion at exact times, in its own scene. A motion may carry a `Highlight`, such as an invalid state, shown with its caption. Nothing is executed. `mj_viser.preview.record_motion` writes a motion as a recording.

### Contacts

`mj_viser.ContactOverlay` draws the contacts of the state the viewer shows, as the simulation computed them (nothing is recomputed): a point, a normal, and a force arrow per contact, from MuJoCo's own contact list and `mj_contactForce`, named by body. Each browser chooses whether contacts show and how long a newton's arrow is; the caption gives the frame's time, the scale, and each contact's force in newtons, and says so when there are none or more than are drawn.

```python
from mj_viser import ContactOverlay

viewer.add_panel(ContactOverlay())   # or live.viewer.add_panel(...)
```

`mj_viser.contact_set(model, data)` returns the same contacts as data, and `StateRecorder(model, contacts=True)` records them with each state, logging every contact and arrow drawn.

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

Each browser has its own selection; one viewer's click does not change another's.

### Scene report

```python
report = viewer.scene_report()  # plain JSON-ready data; works before open()
report["bodies"][1]["path"]     # e.g. "/mujoco/b0_world/b1_trunk"
report["unsupported"]           # e.g. [{"kind": "light", "name": "spotlight", "reason": "MuJoCo lights are not rendered; ..."}, ...]
report["coverage"]["site"]      # e.g. {"total": 9, "rendered": 8, "filtered": 1, "unsupported": 0}
```

`coverage` accounts for every geom, site, light, camera, tendon, skin, and flex in the model: each is rendered, filtered (with the reason in its entry), or listed in `unsupported`. The report depends only on the model, so it is the same wherever the model is rendered.

Paths are built from MuJoCo ids plus sanitized names (`b3_trunk`, `g7`, `s2_imu`), so they are stable for a model and unique even for unnamed elements or names that differ only in punctuation.

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

### Groups

As in MuJoCo, geom and site groups 0–2 are shown and 3–5 hidden by default. The Visibility panel has a checkbox for each group in use; applications can set the same state directly, for example to show sensor sites kept in group 4:

```python
viewer.set_visible_groups(sites=range(6))           # show every site group
viewer.set_visible_groups(geoms={0, 1, 2, 3})       # also show geom group 3
viewer.visible_groups()                             # {"geoms": {0, 1, 2, 3}, "sites": {0, ..., 5}}

viewer = MujocoViewer(model, show_visibility=False)  # no panel; groups still apply
```

Group ids are MuJoCo's 0–5; anything else (a float, a string, a bool, or an id out of range) raises `TypeError` or `ValueError` and leaves the current setting unchanged.

## ssrobot integration

`mj_viser.integrations.ssrobot` connects mj_viser to [ssrobot](https://github.com/personalrobotics/ssrobot) runs. It is the only part of mj_viser that imports ssrobot, which you install yourself (with its MuJoCo extra); importing it without ssrobot says what is needed.

### Follow a run

```python
from mj_viser import LiveView
from mj_viser.integrations.ssrobot import TraceChannels, render_source

channels = TraceChannels(description)   # an ssrobot trace sink that feeds a ChannelBoard
with RobotContext(description, runtime, sinks=[trace, channels]) as ctx:
    view = LiveView(render_source(ctx), channels=channels.board)
    view.open()
    while running:
        ctx.step()
        view.refresh()
```

`render_source(ctx)` adapts ssrobot's render handle; traces are byte-identical with and without a view attached. A `ContactOverlay` on the view draws the run's contacts from the forwarded state the handle captures, and agrees with ssrobot's privileged `contact` channels (touching, summed normal force), which `TraceChannels` shows by name. After the runtime reopens, an old view goes stale. `TraceChannels` shows the description's channels by joint and component name, each execution's command and state history, the scene revision and what is held, and runtime events.

### Preview a plan

```python
from ssrobot.mujoco import materialize_world_from
from mj_viser.integrations.ssrobot import PlanPreview

world = materialize_world_from(package, world=runtime.world(), scene=scene)
preview = PlanPreview(world, description, live=view)
view.viewer.add_panel(preview)
snapshot = ctx.snapshot()
preview.add_result("lift", arm.plan_to_end_effector_offset((0, 0, 1), 0.05, snapshot=snapshot, planner=planner), snapshot)
```

- Each browser picks a result and scrubs its path (the planner's own waypoints) or its trajectory (evaluated exactly from the planner's timing law). Held objects ride their end effectors.
- A result is refused if it was planned in another snapshot, world, or (with `live=`) scene revision, and a shown preview goes stale when the scene changes.
- `add_result(..., validation=)` takes an invalid `MotionValidation` of the plan's own motion and shows its first invalid state, with its location and contacts.
- `record_preview` writes a preview as a recording.

### Replay a run bundle

A run saved as an ssrobot run bundle (`ssrobot.write_bundle`) converts into a native Viser recording. The bundle is verified and its world rebuilt by ssrobot without a runtime, before anything renders:

```bash
python -m mj_viser.integrations.ssrobot.replay run-bundle/ replay/
# replay/recording.viser   the recording
# replay/conversion.json   what is shown when, what is not, and every interval without a frame
```

- Frames are the run's `SceneSnapshot`s, at the cadence the run took them with `ctx.snapshot()`. Nothing is interpolated.
- Text in the scene shows the run's `TraceChannels` board at each record's time, with camera images beside it. `conversion.json` lists every channel as displayed, malformed, unavailable, or unsupported, and every image reading as shown or malformed.
- A bundle with a missing or changed file is refused, and nothing is written. Moving the bundle does not change the recording.
- `record_trace(records, world)` in the same module converts a trace without a bundle.
- Contact geometry is not in the trace: ssrobot records each contact channel's touching and summed normal force, which the replay shows, but not contact points, normals, or per-contact forces.

## How MuJoCo geometry is converted

Each rule follows MuJoCo's own visualizer, and the end-to-end check compares the result with it.

- **Types**: box, sphere, cylinder, capsule, ellipsoid, plane, and mesh render. Height fields and SDFs do not and are listed in `scene_report()["unsupported"]`.
- **Color**: a material's rgba applies only while the geom's rgba is MuJoCo's default `0.5 0.5 0.5 1`; an explicit geom rgba wins.
- **Filtering**: a geom or site whose resolved alpha is 0 is not drawn (`filtered` in the report). Groups 0–2 are visible and 3–5 hidden by default, as in MuJoCo; see [Groups](#groups).
- **Sites**: sphere, capsule, ellipsoid, cylinder, and box sites render with MuJoCo's size, local pose, and color rules (the same material rule as geoms).
- **Textures**: 2D textures render on meshes with texture coordinates and on planes. Meshes and planes honor the material's `texrepeat`, and planes also `texuniform`, as MuJoCo's renderer does. Textures on other geom types, and cube or skybox textures, are not drawn; the geom keeps its base color and the report lists it.
- **Infinite planes** (size 0) render as a square of half-size 50 m, noted in `scene_report()["diagnostics"]`.

## Compatibility

| | Supported |
| --- | --- |
| Python | 3.11–3.14 (3.15 awaits wheels for Viser's `zstandard` dependency) |
| MuJoCo | `>=3.12` (the texture layout mj_viser reproduces); locked to 3.14.0, the version ssrobot pins |
| Viser | `>=1.1.1,<1.2` |
| NumPy | `>=1.24` |
| trimesh | `>=3.21.7` |
| Pillow | `>=9.2` |

CI runs the same end-to-end check from a clean wheel install, in parallel jobs, on the exact wheel it builds and uploads together with each run's report and recording: the locked versions (`uv.lock`) on every supported Python, every declared minimum on Python 3.11, and the newest allowed releases (on Python 3.14 for pull requests; on every Python for `main` and a nightly run). It renders public robots from MuJoCo Menagerie (`trossen_wxai`, `robotiq_2f85`, `unitree_go1`) at a pinned commit.

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
uv sync --locked --all-extras
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

The end-to-end check builds the wheel, installs it into a clean virtualenv, and exercises the installed package:

```bash
UV_PYTHON=3.12 scripts/smoke_wheel.sh dist/smoke locked
```

It writes `dist/smoke/locked/smoke_report.json` (results, input-state hashes, installed versions, wheel sha256) and `dist/smoke/locked/two_states.viser`, a recording of two rendered states; open it in the Viser client to replay. Add `lowest` and `latest` to run the declared-minimum and newest-release environments too.

## License

MIT
