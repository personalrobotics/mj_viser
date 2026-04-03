"""Teleop panel: SE(3) gizmo + ghost hand for interactive arm control.

Provides a Viser TransformControls gizmo that drives a TeleopController.
A transparent ghost gripper shows the target pose with color feedback:
green = tracking, orange = collision warning, red = unreachable.

Generic — works with any arm + gripper. Robot-specific code supplies
the gripper body prefix and arm reference.

Usage::

    from mj_viser import TeleopPanel
    from mj_manipulator.teleop import TeleopController

    controller = TeleopController(arm, ctx)
    panel = TeleopPanel(
        arm=arm, controller=controller,
        model=model, gripper_body_prefix="right_ur5e/gripper/",
    )
    panel.setup(gui, viewer)
    # panel.on_sync(viewer) called each frame by the viewer
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import trimesh
import viser

from mj_viser.mesh_utils import extract_mujoco_mesh
from mj_viser.panels import PanelBase
from mj_viser.transforms import xmat_to_wxyz

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.teleop import SafetyMode, TeleopController

    from mj_viser.viewer import MujocoViewer

logger = logging.getLogger(__name__)

# Ghost hand colors (RGB 0-255)
_COLOR_TRACKING = (100, 200, 100)       # green
_COLOR_COLLISION = (230, 160, 50)       # orange
_COLOR_UNREACHABLE = (200, 80, 80)      # red
_GHOST_OPACITY = 0.35


class GhostHand:
    """Transparent gripper mesh rendered in the Viser scene.

    Extracts collision mesh geometry from a MuJoCo model for all geoms
    under a body prefix, merges into a single trimesh, and renders as
    a single semi-transparent scene node.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gripper_body_prefix: str,
        ee_site_id: int,
        opacity: float = _GHOST_OPACITY,
        name: str = "/teleop/ghost",
    ):
        self._server = server
        self._name = name
        self._handle = None

        # Build merged mesh from all collision geoms under the prefix
        mesh = self._extract_gripper_mesh(model, data, gripper_body_prefix, ee_site_id)
        if mesh is not None and len(mesh.vertices) > 0:
            self._handle = server.scene.add_mesh_trimesh(
                name, mesh=mesh, visible=False,
            )

    def set_pose(self, pose_4x4: np.ndarray) -> None:
        """Move the ghost to a new EE pose."""
        if self._handle is None:
            return
        self._handle.position = pose_4x4[:3, 3]
        self._handle.wxyz = xmat_to_wxyz(pose_4x4[:3, :3].flatten())

    def set_color(self, rgb: tuple[int, int, int]) -> None:
        """Change ghost color (for reachability feedback)."""
        # Viser doesn't support runtime color change on trimesh easily.
        # We'd need to remove and re-add. For now, skip — the gizmo itself
        # provides visual feedback. This is a placeholder for future
        # Viser API improvements.
        pass

    def set_visible(self, visible: bool) -> None:
        """Show or hide the ghost."""
        if self._handle is not None:
            self._handle.visible = visible

    def remove(self) -> None:
        """Remove ghost from scene."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @staticmethod
    def _extract_gripper_mesh(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        body_prefix: str,
        ee_site_id: int,
    ) -> trimesh.Trimesh | None:
        """Extract and merge all collision geoms under a body prefix.

        Transforms each mesh from its geom frame into the EE site frame
        so the merged mesh is centered at the grasp_site origin.
        """
        # Get EE site world transform
        site_pos = data.site_xpos[ee_site_id].copy()
        site_mat = data.site_xmat[ee_site_id].reshape(3, 3)
        T_world_site = np.eye(4)
        T_world_site[:3, :3] = site_mat
        T_world_site[:3, 3] = site_pos
        T_site_world = np.linalg.inv(T_world_site)

        meshes = []
        for geom_id in range(model.ngeom):
            body_id = model.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None or not body_name.startswith(body_prefix):
                continue
            # Only collision geoms (contype or conaffinity > 0)
            if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
                continue

            geom_type = model.geom_type[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = model.geom_dataid[geom_id]
                verts, faces = extract_mujoco_mesh(model, mesh_id)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                size = model.geom_size[geom_id]
                box = trimesh.creation.box(extents=size * 2)
                verts = np.array(box.vertices, dtype=np.float32)
                faces = np.array(box.faces, dtype=np.int32)
            else:
                continue

            # Transform from geom local → world → EE site frame
            geom_pos = data.geom_xpos[geom_id]
            geom_mat = data.geom_xmat[geom_id].reshape(3, 3)
            T_world_geom = np.eye(4)
            T_world_geom[:3, :3] = geom_mat
            T_world_geom[:3, 3] = geom_pos
            T_site_geom = T_site_world @ T_world_geom

            # Apply transform to vertices
            verts_h = np.c_[verts, np.ones(len(verts))]
            verts_site = (T_site_geom @ verts_h.T).T[:, :3]

            meshes.append(trimesh.Trimesh(
                vertices=verts_site.astype(np.float32),
                faces=faces,
            ))

        if not meshes:
            logger.warning("No gripper meshes found under prefix '%s'", body_prefix)
            return None

        merged = trimesh.util.concatenate(meshes)
        # Set a neutral color
        merged.visual = trimesh.visual.ColorVisuals(
            mesh=merged,
            face_colors=np.tile([100, 200, 100, int(255 * _GHOST_OPACITY)], (len(merged.faces), 1)),
        )
        return merged


class TeleopPanel(PanelBase):
    """Viser panel for SE(3) gizmo teleop control.

    Creates a 6DOF TransformControls gizmo and a ghost gripper mesh.
    Drives a TeleopController on gizmo updates.

    Args:
        arm: Arm instance (for reading EE pose).
        controller: TeleopController to drive.
        model: MuJoCo model (for extracting gripper mesh).
        data: MuJoCo data (for initial gripper poses).
        gripper_body_prefix: Body name prefix for gripper geoms.
        arm_label: Display label (e.g., "Right Arm").
    """

    def __init__(
        self,
        arm: Arm,
        controller: TeleopController,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        gripper_body_prefix: str,
        arm_label: str = "Arm",
        abort_fn: object | None = None,
        clear_abort_fn: object | None = None,
    ):
        self._arm = arm
        self._controller = controller
        self._model = model
        self._data = data
        self._gripper_prefix = gripper_body_prefix
        self._arm_label = arm_label
        self._abort_fn = abort_fn  # callable → bool, checked in teleop loop
        self._clear_abort_fn = clear_abort_fn  # callable → None, clears abort
        self._gizmo = None
        self._ghost = None
        self._is_teleop_active = False

    def name(self) -> str:
        return "Teleop"

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        self._viewer = viewer
        server = viewer._server
        scene = server.scene

        # Gizmo (hidden initially) — unique name per arm
        arm_name = self._arm.config.name
        gizmo_name = f"/teleop/{arm_name}/gizmo"
        ee_pose = self._arm.get_ee_pose()
        self._gizmo = scene.add_transform_controls(
            gizmo_name,
            scale=0.3,
            depth_test=False,
            wxyz=xmat_to_wxyz(ee_pose[:3, :3].flatten()),
            position=tuple(ee_pose[:3, 3]),
            visible=False,
        )

        # Ghost hand as child of gizmo — moves automatically when gizmo moves
        self._ghost = GhostHand(
            server, self._model, self._data,
            self._gripper_prefix, self._arm.ee_site_id,
            name=f"{gizmo_name}/ghost",
        )

        # Wire gizmo callbacks
        @self._gizmo.on_update
        def _on_gizmo_update(event) -> None:
            if not self._is_teleop_active:
                return
            pose = np.eye(4)
            wxyz = np.array(event.target.wxyz)
            w, x, y, z = wxyz
            pose[:3, :3] = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
            ])
            pose[:3, 3] = event.target.position
            # Only buffer the target — stepping happens in _teleop_loop thread
            self._controller.set_target_pose(pose)

        # GUI controls
        self._activate_btn = gui.add_button(
            f"Activate Teleop ({self._arm_label})", color="green",
        )
        self._snap_btn = gui.add_button("Snap to EE")
        self._gripper_btn = gui.add_button("Toggle Gripper")

        # Safety mode dropdown
        from mj_manipulator.teleop import SafetyMode
        self._safety_dropdown = gui.add_dropdown(
            "Safety Mode",
            options=["allow", "reject"],
            initial_value=self._controller.safety_mode.value,
        )

        @self._safety_dropdown.on_update
        def _on_safety_change(event) -> None:
            self._controller.safety_mode = SafetyMode(self._safety_dropdown.value)

        # Status label (markdown for colored text)
        self._status_md = gui.add_markdown("⚪ **Idle**")

        # Record button
        self._record_btn = gui.add_button("Record")

        @self._activate_btn.on_click
        def _on_activate(event) -> None:
            if self._is_teleop_active:
                self._deactivate_teleop()
            else:
                self._activate_teleop()

        @self._snap_btn.on_click
        def _on_snap(event) -> None:
            if self._gizmo is not None:
                ee_pose = self._arm.get_ee_pose()
                self._gizmo.wxyz = xmat_to_wxyz(ee_pose[:3, :3].flatten())
                self._gizmo.position = tuple(ee_pose[:3, 3])

        @self._gripper_btn.on_click
        def _on_gripper(event) -> None:
            self._controller.toggle_gripper()

        @self._record_btn.on_click
        def _on_record(event) -> None:
            if self._controller.is_recording:
                frames = self._controller.stop_recording()
                self._record_btn.name = "Record"
                logger.info("Recorded %d frames", len(frames))
            else:
                self._controller.start_recording()
                self._record_btn.name = "Stop Recording"

    def on_sync(self, viewer: MujocoViewer) -> None:
        """Called each frame by viewer.sync(). No-op — teleop stepping
        happens in _teleop_loop to avoid recursion (step → sync → on_sync)."""
        pass

    def _activate_teleop(self) -> None:
        # Clear any stale abort from Stop button
        if self._clear_abort_fn is not None:
            self._clear_abort_fn()

        ee_pose = self._controller.activate()
        self._is_teleop_active = True

        if self._gizmo is not None:
            self._gizmo.wxyz = xmat_to_wxyz(ee_pose[:3, :3].flatten())
            self._gizmo.position = tuple(ee_pose[:3, 3])
            self._gizmo.visible = True

        if self._ghost is not None:
            self._ghost.set_visible(True)

        self._activate_btn.name = f"Deactivate Teleop ({self._arm_label})"
        self._activate_btn.color = "red"

        # Start background loop that steps the controller + syncs viewer
        self._teleop_thread = threading.Thread(
            target=self._teleop_loop, daemon=True,
        )
        self._teleop_thread.start()

    def _deactivate_teleop(self) -> None:
        self._is_teleop_active = False  # signals thread to stop
        self._controller.deactivate()

        if self._gizmo is not None:
            self._gizmo.visible = False
        if self._ghost is not None:
            self._ghost.set_visible(False)

        self._activate_btn.name = f"Activate Teleop ({self._arm_label})"
        self._activate_btn.color = "green"
        self._status_md.content = "⚪ **Idle**"

    def _teleop_loop(self) -> None:
        """Background loop: step controller + sync viewer at ~30 Hz."""
        dt = 1.0 / 30.0
        while self._is_teleop_active and self._controller.is_active:
            # Check abort (Stop button / reset)
            if self._abort_fn is not None and self._abort_fn():
                self._controller.deactivate()
                break
            t0 = time.monotonic()
            try:
                from mj_manipulator.teleop import TeleopState
                state = self._controller.step()
                # Update status label with color
                if state == TeleopState.TRACKING:
                    self._status_md.content = "🟢 **Tracking**"
                elif state == TeleopState.TRACKING_COLLISION:
                    self._status_md.content = "🔴 **Collision**"
                elif state == TeleopState.UNREACHABLE:
                    self._status_md.content = "🟠 **Unreachable**"
                else:
                    self._status_md.content = "⚪ **Idle**"
                if self._viewer is not None:
                    self._viewer.sync()
            except Exception as e:
                logger.warning("Teleop step error: %s", e)
                break
            elapsed = time.monotonic() - t0
            sleep = dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # Loop exited (abort, error, or deactivate) — reset panel UI
        self._is_teleop_active = False
        if self._gizmo is not None:
            self._gizmo.visible = False
        if self._ghost is not None:
            self._ghost.set_visible(False)
        self._activate_btn.name = f"Activate Teleop ({self._arm_label})"
        self._activate_btn.color = "green"
        self._status_md.content = "⚪ **Idle**"
