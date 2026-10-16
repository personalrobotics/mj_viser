# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Built-in GUI panel for geom-group visibility toggles."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import viser


class GuiManager:
    """Creates and manages the built-in visibility panel."""

    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel) -> None:
        self._model = model

        # Discover which geom groups have rendered geoms (skip unsupported types like plane/hfield).
        from mj_viser.geom_builders import GEOM_BUILDERS

        used_groups: set[int] = set()
        for geom_id in range(model.ngeom):
            if model.geom_type[geom_id] in GEOM_BUILDERS:
                used_groups.add(int(model.geom_group[geom_id]))

        gui = server.gui

        # --- Visibility toggles (only for groups that have geoms) ---
        self._visibility_callbacks: list[Callable[[], None]] = []
        self._group_toggles: dict[int, viser.GuiCheckboxHandle] = {}
        with gui.add_folder("Visibility", order=1):
            for g in sorted(used_groups):
                # MuJoCo default: groups 0-2 visible, groups 3-5 hidden
                toggle = gui.add_checkbox(f"Group {g}", initial_value=(g <= 2))
                self._group_toggles[g] = toggle

                @toggle.on_update
                def _(_: viser.GuiEvent[viser.GuiCheckboxHandle]) -> None:
                    for cb in self._visibility_callbacks:
                        cb()

    def on_visibility_change(self, callback: Callable[[], None]) -> None:
        """Register a callback fired when any group toggle changes."""
        self._visibility_callbacks.append(callback)

    def visible_groups(self) -> set[int]:
        """Return the set of currently visible geom group indices."""
        return {g for g, toggle in self._group_toggles.items() if toggle.value}
