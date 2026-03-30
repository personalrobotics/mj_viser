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

    def build_scene(self) -> None:
        """Create Viser scene nodes for all supported MuJoCo geoms."""
        configure_scene(self._server)
        self._setup_lighting()
        self._setup_ground_grid()

        scene = self._server.scene
        for geom_id in range(self._model.ngeom):
            geom_type = self._model.geom_type[geom_id]
            builder = GEOM_BUILDERS.get(geom_type)
            if builder is None:
                continue
            handle = builder(scene, geom_id, self._model)
            self._geom_handles[geom_id] = handle

        # Set initial transforms
        self.update_transforms()

    def update_transforms(self) -> None:
        """Update all geom positions and orientations from current MjData state."""
        with self._server.atomic():
            for geom_id, handle in self._geom_handles.items():
                handle.position = mj_pos_to_viser(self._data.geom_xpos[geom_id])
                handle.wxyz = xmat_to_wxyz(self._data.geom_xmat[geom_id])

    def update_visibility(self, visible_groups: set[int]) -> None:
        """Toggle geom visibility based on MuJoCo geom groups."""
        for geom_id, handle in self._geom_handles.items():
            group = int(self._model.geom_group[geom_id])
            handle.visible = group in visible_groups

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
