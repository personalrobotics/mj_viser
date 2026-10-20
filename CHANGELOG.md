# Changelog

## 1.0.0 — 2026-10-20

The first release on PyPI, as `ssmjviser`. From this release, the public API in
`docs/api.md` follows Semantic Versioning.

### Added

- A documented public API with a Semantic Versioning and deprecation contract
  (`docs/api.md`), checked against what the package exports; `py.typed`, with a strict
  downstream type check of the installed wheel in CI; `CITATION.cff`; and a release
  workflow that publishes tags to TestPyPI and PyPI with Trusted Publishing, from the files
  CI verified, with `scripts/release-precheck.sh` as its local mirror (#49).

### Changed

- `mj_viser.integrations.ssrobot` no longer re-exports `open_bundle` and
  `materialize_world`, which are ssrobot's own functions (#49). Migration: import them
  from ssrobot, `from ssrobot import open_bundle` and
  `from ssrobot.mujoco import materialize_world`.
- The README's quick start serves on a free port and prints its URL.
- The distribution is named `ssmjviser` (PyPI does not allow `mj-viser`, which is too close
  to the existing `mjviser`); the import name stays `mj_viser` (#49).
- The README leads with what mj_viser does, shown in clips captured from real runs, and
  installs from the release wheel; it is ready to be the PyPI project description: the
  build points its relative images and links at the repository, and each is checked from
  the built wheel (#48). CONTRIBUTING and
  the issue and pull request templates treat mj_viser as a standalone package.
- The README installs with `pip install ssmjviser`.

### Known limitations

- Viser sends a node's new pose only when it is not `np.allclose` (rtol 1e-5) to the
  last one sent, so a body can lag a supplied state by about 1e-5 per local transform,
  live and in recordings (#44).
- ssrobot traces carry contact channels per declared pair, not contact geometry, so
  ssrobot replays show contact channels but no contact points or arrows (#56).

## 0.4.0 — 2026-10-19

### Changed

- The live render source is an mj_viser protocol (#51). `RenderFrame` has `time_ns`
  (it had `time.time_ns`), and a source signals that it can no longer describe its
  simulation by raising `mj_viser.SourceStale(reason)`. `LiveView` no longer
  interprets other libraries' error codes, so any other error propagates. `open()`
  raises `SourceStale`, and closes the viewer, when the first frame cannot be rendered.
- ssrobot adapters live in the optional `mj_viser.integrations.ssrobot`, the only part of
  mj_viser that imports ssrobot (#51). Migration:
  - `LiveView(render_handle(ctx))` becomes `LiveView(render_source(ctx))`.
  - `ChannelModel(description)` becomes `TraceChannels(description)`, still the trace
    sink; pass its `.board` to `LiveView(channels=)`.
  - `python -m mj_viser.replay` becomes `python -m mj_viser.integrations.ssrobot.replay`,
    and `record_trace` and `convert_bundle` move to that module.
  - `mj_viser.PlanPreview` becomes `mj_viser.integrations.ssrobot.PlanPreview`, and
    `preview.add(name, result, snapshot)` becomes `preview.add_result(...)`;
    `record_preview` moves there too.
- `ChannelBoard` is the general channel and status model that `LiveView` panels and
  recordings show; callers feed it named readings, status lines, and events. The panels'
  "Executions" folder is now "Status".

### Added

- `ContactOverlay` draws the contacts of the state a viewer shows: a point, a normal, and
  a force arrow per contact, from MuJoCo's own contact list and `mj_contactForce`, named
  by body, with per-browser visibility and arrow scale (#11). `contact_set` returns them
  as data, and `StateRecorder(contacts=True)` records and logs them. The ssrobot
  integration labels contact channels (touching, normal force in N), and a replay's
  report says contact geometry is not in the trace.
- `MotionPreview` previews named `JointMotion`s of any MuJoCo model as per-browser ghosts,
  with waypoints, timed motions at exact times, and an optional `Highlight`;
  `record_motion` records one (#51). ssrobot's `PlanPreview` is built on it.
- `StateRecorder` records caller-supplied states, text, and RGB images of any MuJoCo model
  as a native Viser recording, at the times the caller gives (#51).
- `PlanPreview` shows ssrobot planning results as a ghost of the moving robot in the
  scene they were planned in (#3). Each browser picks a result and scrubs its exact path
  waypoints or its trajectory, evaluated from the planner's timing law, in its own scene.
  Results from another snapshot, world, or live scene revision are refused, shown ones
  go stale when the scene changes, and a supplied invalid validation's first invalid
  state is shown as reported. `record_preview` writes a preview as a recording.
  `LiveView.frame` is the latest frame rendered.

### Known limitations

- Viser sends a node's new pose only when it is not `np.allclose` (rtol 1e-5) to the
  last one sent, so a body can lag a supplied state by about 1e-5 per local transform,
  live and in recordings (#44).
- ssrobot traces carry contact channels per declared pair, not contact geometry, so
  ssrobot replays show contact channels but no contact points or arrows (#56).

## 0.3.0 — 2026-10-18

### Added

- `LiveView` follows a running simulation through a read-only render source, such as
  ssrobot's `render_handle(ctx)`, without importing ssrobot (#28). `refresh()` captures
  and renders at most `max_rate_hz` times a second and never raises because the view
  closed or a browser disconnected; `expected_world` refuses a source for another scene;
  a source that reports `stale_render` or `not_open`, or a frame from another scene,
  leaves the view `stale`. The HUD shows simulation time and scene revision.
- `ChannelModel` reads trace records, live as an ssrobot sink or from a recorded trace,
  and keeps each declared channel's latest reading by name and its status, each
  execution's state history, the scene revision, and recent events (#6). `LiveView(...,
  channels=)` shows it in per-browser panels with a per-browser plot; replays show it
  in their scene label and list every channel's coverage in `conversion.json`.
- RGB image channels, such as ADA's wrist camera, are shown live in per-browser image
  panels and in replays as scene images at their recorded times (#24). An image is sent
  at most once per new reading; a reading that is not an H×W×3 `uint8` image is marked
  `malformed`, and a replay's report accounts for every image reading.
- `python -m mj_viser.replay BUNDLE OUT_DIR` converts an ssrobot run bundle into a
  native Viser recording and a conversion report (#4). Frames are the run's recorded
  snapshots, with no interpolation; a scene label shows commands, execution states,
  scene revisions, observations, violations, and runtime events at their recorded
  times. The bundle is verified and its world rebuilt by ssrobot before anything renders.
  `mj_viser.replay.record_trace` converts any trace and world of the same shape.

### Known limitations

- Viser sends a node's new pose only when it is not `np.allclose` (rtol 1e-5) to the
  last one sent, so a body can lag a supplied state by about 1e-5 per local transform,
  live and in recordings (#44).

## 0.2.0 — 2026-10-17

### Changed

- The viewer lifecycle is caller-driven and read-only (#30). `MujocoViewer(model)`
  no longer takes `MjData`; call `url = viewer.open()`, then `viewer.update(data)`
  whenever you want a frame, then `viewer.close()`. The viewer copies `data`,
  recomputes kinematics on the copy, and never steps, resets, retains, or
  mutates the caller's state. `viewer.data` is that copy.
- Defaults to `host="127.0.0.1"`. Pass `host="0.0.0.0"` to serve other machines.
- `port=0` binds a free port; `viewer.url` reports the bound address.
- `open()` raises `OSError` when the address cannot be bound, instead of hanging
  or silently moving to another port. `update()` raises `ValueError` unless
  `data.model is viewer.model` (data made from a reloaded or copied model is
  rejected) and `RuntimeError` when the viewer is not open.
- `viewer.url` brackets IPv6 literals (`http://[::1]:8080`).
- `close()` is idempotent and releases the port; a closed viewer can be reopened.
  `server`, `url`, and `set_hud` are valid only while open.
- Requires `mujoco>=3.12` (whose texture layout mj_viser reproduces), `viser>=1.1.1,<1.2`,
  `numpy>=1.24`, `trimesh>=3.21.7`, and `pillow>=9.2` (trimesh and Pillow are imported
  directly and are now declared), on Python 3.11–3.14 (`requires-python = ">=3.11,<3.15"`,
  matching ssrobot's minimum; #29). The lock pins mujoco 3.14.0, the version ssrobot uses.
- `label` is passed to Viser's public `label` and appears as the GUI panel heading.
  It no longer sets the browser tab title, which needed a private Viser message (#27).
- The scene mirrors MuJoCo's body tree (#31): one Viser frame per body at a nested path
  such as `/mujoco/b0_world/b1_trunk/g3`, with geoms and sites as children at their static
  local poses. Each update poses only the non-welded body frames, relative to their parent.
  Node paths change from `/mujoco/geoms/geom_<id>`.
- Selection is per browser connection (#31): each client gets its own label, and one
  client's click no longer replaces another's.
- `viewer.scene_report()` (#31) returns every body, geom, and site path, geom status
  (`rendered`, `filtered`, `unsupported`), and every unrendered element (planes and other
  unsupported geom types, lights, cameras, tendons, skins, flexes, skyboxes)
  with a reason.
- Geometry and material conversion follows MuJoCo's visualizer (#32): planes render,
  infinite ones as a 50 m square (reported in `diagnostics`); an explicit geom rgba
  overrides its material's; mesh textures use MuJoCo's per-corner texture coordinates and
  `texrepeat`; textured planes honor `texrepeat` and `texuniform`; material alpha is kept on
  textured meshes. Textures that are not drawn are reported in `unsupported`.
- Sites render as their MuJoCo shapes beneath their bodies, with geom color rules (#23).
- Geom and site group visibility is viewer state (#23): `set_visible_groups(geoms=…,
  sites=…)` and `visible_groups()`, MuJoCo's defaults (groups 0–2 shown), and per-group
  checkboxes for both kinds in the Visibility panel. Groups now apply even with
  `show_visibility=False`, which previously showed every group. Group ids must be
  integers 0–5; other values raise and change nothing.
- `scene_report()["coverage"]` counts every geom, site, light, camera, tendon, skin, and
  flex as rendered, filtered, or unsupported (#23).
- `set_hud()` content and `SensorPanel` titles, channel labels, and colors are treated as
  text: markup is escaped instead of rendered (#29).

### Fixed

- Mesh textures were mapped with per-vertex texture coordinates, scrambling textures on
  meshes whose texture coordinates are indexed per face corner (#32).
- A material's rgba replaced an explicit geom rgba (#32).

- No geoms rendered with recent MuJoCo releases (seen with 3.16), whose enums no
  longer compare equal to the numpy ints in `model.geom_type`.

### Removed

- `launch()`, `launch_passive()`, `sync()`, `is_running()`, the built-in
  simulation thread, the Play/Pause/Step/Reset/Speed panel, and the `show_gui`
  and `show_sim_controls` arguments (#30).
- Geoms are no longer hidden below z = -0.5, and the default ground grid and three
  directional lights are gone; scenes use Viser's default lighting (#31).
- `mesh_utils.make_capsule_mesh`, `make_ellipsoid_mesh`, and
  `extract_mujoco_mesh_textured`: capsules and ellipsoids use trimesh primitives, and
  `extract_mujoco_mesh_corners` replaces the textured extractor (#32).
- `mj_viser.transforms` (`xmat_to_wxyz`, `mj_pos_to_viser`, `configure_scene`):
  use `viser.transforms.SO3.from_matrix(xmat.reshape(3, 3)).wxyz` (#27).
- The unused `plotly` dependency (#27).
- `TeleopPanel` and `GhostHand` (`mj_viser.teleop_panel`). mj_viser no longer
  knows about `mj_manipulator` arms, teleop controllers, ownership, safety, or
  recording. Rendering is read-only; applications own commands. (#26)

### Migration from launch/sync

| Before | After |
| --- | --- |
| `MujocoViewer(model, data)` | `MujocoViewer(model)` |
| `viewer.launch()` | your loop: `mj_step` then `viewer.update(data)`, between `open()` and `close()` |
| `viewer.launch_passive()` | `viewer.open()` |
| `viewer.sync()` | `viewer.update(data)` |
| `while viewer.is_running():` | your own loop condition; `viewer.is_open` reports lifecycle state |
| `show_sim_controls=False` | (nothing to hide) |
| `show_gui=False` | `show_visibility=False` |
| writing through `viewer.data` | write your own `MjData` and pass it to `update` |
| `viewer._server`, `viewer._panels.append(p)` | `viewer.server` (while open), `viewer.add_panel(p)` |

### Migration for Geodude and ADA

Each removed control moves into the application, which already owns the arm,
controller, and runtime. Build the widget with Viser primitives on
`viewer.server` and route its events to your own controller.

| Removed from `TeleopPanel` | Application-owned replacement |
| --- | --- |
| SE(3) gizmo driving `TeleopController.set_target_pose` | App panel (`geodude.panels.teleop_panel`, ADA console) adds `viewer.server.scene.add_transform_controls(...)` and forwards `on_update` poses to its own controller |
| Ghost gripper (`GhostHand`) | App builds the gripper mesh (`mj_viser.mesh_utils.extract_mujoco_mesh` is still public) and adds it with `add_mesh_trimesh` as a child of its gizmo |
| Activate / Deactivate, ownership acquire / preempt / release | App's ownership registry and event loop (`mj_manipulator.ownership`, `PhysicsEventLoop`) |
| Abort request and clear | App's abort policy |
| Snap to EE, `CartesianController` reset | App controller API; no viewer access to `_cart_ctrl` |
| Toggle Gripper | App gripper command |
| Safety mode dropdown | App `SafetyMode` setting |
| Record / Stop Recording | App or ssrobot trace recorder |
| Tracking / collision / unreachable status | App sets text with `viewer.set_hud(...)` or its own GUI markdown |
