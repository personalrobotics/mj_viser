# Changelog

## Unreleased (0.2.0)

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
- Requires `mujoco>=3.5` (`mj_copyData` in the Python bindings), `viser>=1.1.1,<1.2`,
  `numpy>=1.24`, and `trimesh>=3.21.7` (now declared; it was imported but undeclared).
- `label` is passed to Viser's public `label` and appears as the GUI panel heading.
  It no longer sets the browser tab title, which needed a private Viser message (#27).

### Fixed

- No geoms rendered with recent MuJoCo releases (seen with 3.16), whose enums no
  longer compare equal to the numpy ints in `model.geom_type`.

### Removed

- `launch()`, `launch_passive()`, `sync()`, `is_running()`, the built-in
  simulation thread, the Play/Pause/Step/Reset/Speed panel, and the `show_gui`
  and `show_sim_controls` arguments (#30).
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
