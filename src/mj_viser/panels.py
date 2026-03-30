"""Extension point for custom GUI panels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import viser

if TYPE_CHECKING:
    from mj_viser.viewer import MujocoViewer


class PanelBase(ABC):
    """Base class for user-defined GUI panels.

    Subclass this to add custom GUI elements (sensor readouts, camera feeds,
    control interfaces, etc.) to the viewer sidebar.

    Example::

        class JointPanel(PanelBase):
            def name(self) -> str:
                return "Joints"

            def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
                with gui.add_folder(self.name()):
                    self._text = gui.add_text("qpos", initial_value="", disabled=True)

            def on_sync(self, viewer: MujocoViewer) -> None:
                import numpy as np
                self._text.value = np.array2string(viewer.data.qpos, precision=3)
    """

    @abstractmethod
    def name(self) -> str:
        """Panel folder label in the GUI sidebar."""
        ...

    @abstractmethod
    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        """Called once when the panel is registered. Add GUI elements here.

        Args:
            gui: The viser GUI API (``server.gui``).
            viewer: The MujocoViewer instance for accessing model/data.
        """
        ...

    def on_sync(self, viewer: MujocoViewer) -> None:
        """Called each frame after transforms are updated.

        Override to update panel state (e.g., read sensor values).
        Default implementation does nothing.
        """
