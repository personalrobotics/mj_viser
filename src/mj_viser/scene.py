"""Scene manager: builds and updates Viser scene nodes from MuJoCo model/data."""

from __future__ import annotations

import math

import mujoco
import viser

from mj_viser.geom_builders import GEOM_BUILDERS
from mj_viser.transforms import configure_scene, mj_pos_to_viser, xmat_to_wxyz


class SceneManager:
    """Manages the mapping between MuJoCo geoms and Viser scene nodes.

    Responsible for one-time scene construction, per-frame transform updates,
    lighting, and visibility toggling.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self._server = server
        self._model = model
        self._data = data
        self._geom_handles: dict[int, viser.SceneNodeHandle] = {}
        self._hidden_groups: set[int] = set()  # groups toggled off by user
        self._selected_geom: int | None = None
        self._label_handle: viser.SceneNodeHandle | None = None
        self._on_select_callbacks: list = []  # [(callable(geom_id, body_name))]

    def build_scene(self) -> None:
        """Create Viser scene nodes for all supported MuJoCo geoms."""
        configure_scene(self._server)
        self._setup_lighting()
        self._setup_ground_grid()

        scene = self._server.scene
        for geom_id in range(self._model.ngeom):
            # Skip invisible geoms (collision-only or hidden)
            mat_id = self._model.geom_matid[geom_id]
            if mat_id >= 0:
                alpha = self._model.mat_rgba[mat_id][3]
            else:
                alpha = self._model.geom_rgba[geom_id][3]
            if alpha == 0:
                continue

            geom_type = self._model.geom_type[geom_id]
            builder = GEOM_BUILDERS.get(geom_type)
            if builder is None:
                continue
            handle = builder(scene, geom_id, self._model)
            self._geom_handles[geom_id] = handle

            # Click handler for selection
            gid = geom_id  # capture for closure
            @handle.on_click
            def _(event: viser.ScenePointerEvent, _gid=gid) -> None:
                self._handle_click(_gid)

        # Set initial transforms
        self.update_transforms()

    def update_transforms(self) -> None:
        """Update all geom positions, orientations, and visibility."""
        with self._server.atomic():
            for geom_id, handle in self._geom_handles.items():
                pos = self._data.geom_xpos[geom_id]

                # Always update transforms (even for hidden geoms, so they're
                # correct if toggled back on)
                handle.position = mj_pos_to_viser(pos)
                handle.wxyz = xmat_to_wxyz(self._data.geom_xmat[geom_id])

                # Visibility: hidden by group toggle or underground
                group = int(self._model.geom_group[geom_id])
                handle.visible = group not in self._hidden_groups and float(pos[2]) >= -0.5

            # Move selection label with the selected geom
            if self._selected_geom is not None and self._label_handle is not None:
                pos = self._data.geom_xpos[self._selected_geom]
                if float(pos[2]) < -0.5:
                    self._clear_label()
                    self._selected_geom = None
                else:
                    self._label_handle.position = tuple(
                        p + o for p, o in zip(mj_pos_to_viser(pos), (0, 0, 0.05))
                    )

    def update_visibility(self, visible_groups: set[int]) -> None:
        """Toggle geom visibility based on MuJoCo geom groups."""
        all_groups = {int(self._model.geom_group[gid]) for gid in self._geom_handles}
        self._hidden_groups = all_groups - visible_groups
        # Apply immediately (don't wait for update_transforms)
        with self._server.atomic():
            for geom_id, handle in self._geom_handles.items():
                group = int(self._model.geom_group[geom_id])
                if group in self._hidden_groups:
                    handle.visible = False
                else:
                    # Only show if not underground (hidden by registry)
                    pos = self._data.geom_xpos[geom_id]
                    handle.visible = bool(pos[2] >= -0.5)

    def _handle_click(self, geom_id: int) -> None:
        """Handle a click on a geom — toggle selection label."""
        if self._selected_geom == geom_id:
            # Click same geom again — deselect
            self._clear_label()
            self._selected_geom = None
            return

        self._selected_geom = geom_id

        # Resolve body name
        body_id = self._model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(
            self._model, mujoco.mjtObj.mjOBJ_BODY, body_id,
        ) or f"body_{body_id}"
        geom_name = mujoco.mj_id2name(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id,
        ) or f"geom_{geom_id}"

        # Show label at geom position
        pos = self._data.geom_xpos[geom_id]
        self._show_label(body_name, pos)

        # Notify callbacks
        for cb in self._on_select_callbacks:
            cb(geom_id, body_name)

    def _show_label(self, text: str, position: object) -> None:
        """Show a 3D label at the given position."""
        self._clear_label()
        self._label_handle = self._server.scene.add_label(
            "/mujoco/selection_label",
            text=text,
            wxyz=(1, 0, 0, 0),
            position=tuple(p + o for p, o in zip(mj_pos_to_viser(position), (0, 0, 0.05))),
        )

    def _clear_label(self) -> None:
        """Remove the selection label."""
        if self._label_handle is not None:
            self._label_handle.remove()
            self._label_handle = None

    def on_select(self, callback) -> None:
        """Register a callback for geom selection: callback(geom_id, body_name)."""
        self._on_select_callbacks.append(callback)

    def _setup_lighting(self) -> None:
        """Create a clean three-point lighting setup."""
        scene = self._server.scene

        # Key light: warm, from upper-right-front
        scene.add_light_directional(
            "/lights/key",
            color=(255, 250, 240),
            intensity=1.0,
            wxyz=_euler_to_wxyz(-math.pi / 4, math.pi / 4, 0),
        )

        # Fill light: cool, from upper-left
        scene.add_light_directional(
            "/lights/fill",
            color=(210, 220, 240),
            intensity=0.4,
            wxyz=_euler_to_wxyz(-math.pi / 6, -math.pi / 3, 0),
        )

        # Rim light: from behind, subtle
        scene.add_light_directional(
            "/lights/rim",
            color=(240, 240, 255),
            intensity=0.3,
            wxyz=_euler_to_wxyz(-math.pi / 5, math.pi, 0),
        )

    def _setup_ground_grid(self) -> None:
        """Add a ground-plane grid at z=0."""
        self._server.scene.add_grid(
            "/ground",
            width=20.0,
            height=20.0,
            plane="xy",
            cell_size=0.5,
            cell_color=(180, 180, 180),
            section_color=(120, 120, 120),
            section_size=1.0,
        )


def _euler_to_wxyz(pitch: float, yaw: float, roll: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (XYZ intrinsic) to (w, x, y, z) quaternion.

    Used only for configuring light directions.
    """
    cx, sx = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cz, sz = math.cos(roll / 2), math.sin(roll / 2)

    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return (w, x, y, z)
