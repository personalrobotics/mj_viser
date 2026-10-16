# Changelog

## Unreleased

### Removed

- `TeleopPanel` and `GhostHand` (`mj_viser.teleop_panel`). mj_viser no longer
  knows about `mj_manipulator` arms, teleop controllers, ownership, safety, or
  recording. Rendering is read-only; applications own commands. (#26)

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
