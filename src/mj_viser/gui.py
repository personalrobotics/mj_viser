"""Built-in GUI panel for simulation controls and visibility toggles."""

from __future__ import annotations

import threading

import mujoco
import viser
from viser import Icon


class GuiManager:
    """Creates and manages the built-in simulation control panel.

    Exposes state properties that the viewer's simulation loop reads each tick.
    """

    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel) -> None:
        self._model = model
        self._lock = threading.Lock()
        self._playing = False
        self._step_requested = False
        self._reset_requested = False

        gui = server.gui

        # --- Simulation controls ---
        with gui.add_folder("Simulation", order=0):
            self._play_btn = gui.add_button("Play", icon=Icon.PLAYER_PLAY, color="green")
            self._pause_btn = gui.add_button("Pause", icon=Icon.PLAYER_PAUSE, color="yellow")
            self._step_btn = gui.add_button("Step", icon=Icon.PLAYER_SKIP_FORWARD)
            self._reset_btn = gui.add_button("Reset", icon=Icon.REFRESH, color="red")
            self._speed_slider = gui.add_slider(
                "Speed",
                min=0.1,
                max=5.0,
                step=0.1,
                initial_value=1.0,
            )

        # --- Visibility toggles ---
        with gui.add_folder("Visibility", order=1):
            self._group_toggles: dict[int, viser.GuiCheckboxHandle] = {}
            for g in range(6):
                self._group_toggles[g] = gui.add_checkbox(
                    f"Group {g}",
                    initial_value=True,
                )

        # --- Wire up callbacks ---
        @self._play_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            with self._lock:
                self._playing = True

        @self._pause_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            with self._lock:
                self._playing = False

        @self._step_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            with self._lock:
                self._step_requested = True

        @self._reset_btn.on_click
        def _(_: viser.GuiEvent) -> None:
            with self._lock:
                self._reset_requested = True

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def speed(self) -> float:
        return float(self._speed_slider.value)

    @property
    def should_step(self) -> bool:
        """Return True if a single step was requested, consuming the request."""
        with self._lock:
            if self._step_requested:
                self._step_requested = False
                return True
            return False

    @property
    def should_reset(self) -> bool:
        """Return True if a reset was requested, consuming the request."""
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False

    def visible_groups(self) -> set[int]:
        """Return the set of currently visible geom group indices."""
        return {g for g, toggle in self._group_toggles.items() if toggle.value}
