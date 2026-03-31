"""MujocoViewer: the main public class tying together scene, GUI, and simulation."""

from __future__ import annotations

import threading
import time
import webbrowser

import mujoco
import viser

from mj_viser.gui import GuiManager
from mj_viser.panels import PanelBase
from mj_viser.scene import SceneManager


class MujocoViewer:
    """Web-based MuJoCo viewer using Viser.

    Two modes of operation:

    **Built-in simulation loop** (blocking)::

        viewer = MujocoViewer(model, data)
        viewer.launch()  # runs sim + opens browser

    **User-controlled loop** (non-blocking)::

        viewer = MujocoViewer(model, data)
        viewer.launch_passive()
        while viewer.is_running():
            # your control logic here
            mujoco.mj_step(model, data)
            viewer.sync()
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        show_gui: bool = True,
        show_sim_controls: bool | None = None,
        show_visibility: bool | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._host = host
        self._port = port
        self._show_gui = show_gui
        # None = inherit from show_gui
        self._show_sim_controls = show_sim_controls if show_sim_controls is not None else show_gui
        self._show_visibility = show_visibility if show_visibility is not None else show_gui
        self._running = False
        self._sim_thread: threading.Thread | None = None
        self._panels: list[PanelBase] = []

        self._server = viser.ViserServer(host=host, port=port)
        self._scene_mgr = SceneManager(self._server, model, data)
        self._gui_mgr: GuiManager | None = None

    @property
    def model(self) -> mujoco.MjModel:
        """The MuJoCo model."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """The MuJoCo data (current state)."""
        return self._data

    @property
    def server(self) -> viser.ViserServer:
        """The underlying Viser server, for advanced usage."""
        return self._server

    def add_panel(self, panel: PanelBase) -> None:
        """Register a custom GUI panel.

        Must be called before ``launch()`` or ``launch_passive()``.
        """
        self._panels.append(panel)

    def launch(self, open_browser: bool = True) -> None:
        """Start the built-in simulation loop (blocking).

        This builds the scene, starts a simulation thread, and blocks until
        the viewer is closed.

        Args:
            open_browser: If True, open the viewer URL in the default browser.
        """
        self._build(open_browser=open_browser)
        self._running = True
        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def launch_passive(self, open_browser: bool = True) -> None:
        """Start the viewer in the background (non-blocking).

        After calling this, use :meth:`sync` in your own loop to push state
        updates to the viewer.

        Args:
            open_browser: If True, open the viewer URL in the default browser.
        """
        self._build(open_browser=open_browser)
        self._running = True

    def sync(self) -> None:
        """Push current ``MjData`` state to the viewer.

        Call this in your own loop after stepping the simulation.
        """
        self._scene_mgr.update_transforms()

        if self._gui_mgr is not None:
            self._scene_mgr.update_visibility(self._gui_mgr.visible_groups())

        for panel in self._panels:
            panel.on_sync(self)

    def close(self) -> None:
        """Shut down the viewer and stop any simulation thread."""
        self._running = False
        if self._sim_thread is not None:
            self._sim_thread.join(timeout=2.0)
            self._sim_thread = None
        self._server.stop()

    def is_running(self) -> bool:
        """True if the viewer is still active."""
        return self._running

    def __enter__(self) -> MujocoViewer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self, open_browser: bool) -> None:
        """One-time scene + GUI setup."""
        # Compute forward kinematics so geom positions are valid before rendering.
        mujoco.mj_forward(self._model, self._data)
        self._scene_mgr.build_scene()

        if self._show_gui or self._show_visibility:
            self._gui_mgr = GuiManager(
                self._server, self._model,
                show_sim_controls=self._show_sim_controls,
                show_visibility=self._show_visibility,
            )
            # Apply initial group visibility (groups 3+ hidden by default)
            self._scene_mgr.update_visibility(self._gui_mgr.visible_groups())
            # Wire visibility toggles to update immediately (no sync needed)
            self._gui_mgr._visibility_callback.append(
                lambda: self._scene_mgr.update_visibility(self._gui_mgr.visible_groups())
            )

        for panel in self._panels:
            panel.setup(self._server.gui, self)

        if open_browser:
            webbrowser.open(f"http://localhost:{self._port}")

    def _sim_loop(self) -> None:
        """Simulation thread for launch() mode."""
        dt = self._model.opt.timestep
        prev_groups: set[int] = set()

        while self._running:
            gui = self._gui_mgr

            # Reset
            if gui is not None and gui.should_reset:
                mujoco.mj_resetData(self._model, self._data)
                mujoco.mj_forward(self._model, self._data)

            # Step
            if gui is None or gui.is_playing or gui.should_step:
                mujoco.mj_step(self._model, self._data)

            # Update scene
            self._scene_mgr.update_transforms()

            # Visibility
            if gui is not None:
                groups = gui.visible_groups()
                if groups != prev_groups:
                    self._scene_mgr.update_visibility(groups)
                    prev_groups = groups

            # Notify panels
            for panel in self._panels:
                panel.on_sync(self)

            # Pace to real-time
            speed = gui.speed if gui is not None else 1.0
            time.sleep(dt / speed)
