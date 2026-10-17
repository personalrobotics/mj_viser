# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Built-in GUI panel for geom- and site-group visibility toggles."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import viser

from mj_viser.geom_builders import GEOM_BUILDERS, PRIMITIVE_TYPES


class GuiManager:
    """The Visibility panel: one checkbox per MuJoCo geom group and site group in use.

    The viewer owns which groups are visible; this panel shows that state and reports
    the user's changes through ``on_change(kind, group, visible)``.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        model: mujoco.MjModel,
        visible: dict[str, set[int]],
        on_change: Callable[[str, int, bool], None],
    ) -> None:
        # Only groups that hold something this viewer draws get a checkbox.
        used = {
            "geoms": {int(model.geom_group[g]) for g in range(model.ngeom) if model.geom_type[g] in GEOM_BUILDERS},
            "sites": {int(model.site_group[s]) for s in range(model.nsite) if model.site_type[s] in PRIMITIVE_TYPES},
        }
        self._toggles: dict[tuple[str, int], viser.GuiCheckboxHandle] = {}
        with server.gui.add_folder("Visibility", order=1):
            for kind, label in (("geoms", "Geom group"), ("sites", "Site group")):
                for group in sorted(used[kind]):
                    toggle = server.gui.add_checkbox(f"{label} {group}", initial_value=group in visible[kind])
                    self._toggles[(kind, group)] = toggle

                    @toggle.on_update
                    def _(event: viser.GuiEvent[viser.GuiCheckboxHandle], kind: str = kind, group: int = group) -> None:
                        on_change(kind, group, bool(event.target.value))

    def show(self, visible: dict[str, set[int]]) -> None:
        """Reflect the viewer's visibility state in the checkboxes."""
        for (kind, group), toggle in self._toggles.items():
            # Viser defines this setter apart from its getter, which mypy reads as read-only.
            toggle.value = group in visible[kind]  # type: ignore[misc]
