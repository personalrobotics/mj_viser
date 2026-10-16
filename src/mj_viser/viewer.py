# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MujocoViewer: the main public class tying together scene, GUI, and server lifecycle."""

from __future__ import annotations

import socket
import sys
import threading
import webbrowser
from collections.abc import Callable

import mujoco
import viser

from mj_viser.gui import GuiManager
from mj_viser.panels import PanelBase
from mj_viser.scene import SceneManager


class MujocoViewer:
    """Web-based MuJoCo viewer using Viser.

    The viewer only renders. The caller owns the model state, the simulation
    clock, and the update schedule; the viewer never steps, resets, or mutates
    the caller's ``MjData``::

        viewer = MujocoViewer(model)
        url = viewer.open()  # starts the server; raises if it cannot
        while running:
            mujoco.mj_step(model, data)  # your code
            viewer.update(data)  # copies data, then renders the copy
        viewer.close()

    ``with MujocoViewer(model) as viewer:`` opens on entry and closes on exit.
    A closed viewer can be opened again.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        label: str | None = None,
        show_visibility: bool = True,
    ) -> None:
        """
        Args:
            model: The MuJoCo model to render.
            host: Interface to bind. Defaults to loopback only; pass
                ``"0.0.0.0"`` to serve other machines.
            port: Port to bind, or 0 for an OS-assigned port. A requested
                port that is unavailable raises instead of moving to another.
            label: Browser tab title.
            show_visibility: Show the geom-group visibility panel.
        """
        self._model = model
        self._host = host
        self._port = port
        self._label = label
        self._show_visibility = show_visibility
        self._panels: list[PanelBase] = []
        self._select_callbacks: list[Callable[[int, str], None]] = []

        # Viewer-owned copy of the last rendered state. Never the caller's MjData.
        self._data = mujoco.MjData(model)
        # Guards self._data and scene handles against concurrent Viser GUI callbacks.
        self._lock = threading.RLock()

        self._server: viser.ViserServer | None = None
        self._scene_mgr: SceneManager | None = None
        self._gui_mgr: GuiManager | None = None
        self._huds: dict[str, viser.GuiHtmlHandle] = {}

    @property
    def model(self) -> mujoco.MjModel:
        """The MuJoCo model."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """The viewer's copy of the last state passed to :meth:`update`.

        Read it from panels; do not modify it. Changes are overwritten by the
        next update and never reach the caller's simulation.
        """
        return self._data

    @property
    def server(self) -> viser.ViserServer:
        """The underlying Viser server, for advanced usage. Only valid while open."""
        if self._server is None:
            raise RuntimeError("MujocoViewer is not open; call open() first")
        return self._server

    @property
    def is_open(self) -> bool:
        """True between a successful :meth:`open` and :meth:`close`."""
        return self._server is not None

    @property
    def url(self) -> str:
        """The URL the server is actually bound to. Only valid while open."""
        server = self.server
        host = server.get_host()
        if ":" in host:  # IPv6 literal
            host = f"[{host}]"
        return f"http://{host}:{server.get_port()}"

    def set_hud(
        self,
        name: str,
        content: str,
        position: str = "bottom-left",
    ) -> None:
        """Set or update an HTML overlay on the 3D viewport.

        Creates a fixed-positioned HTML element that overlays the canvas,
        outside the sidebar. Multiple HUDs can coexist with different names.
        Only valid while open.

        Args:
            name: Unique identifier for this HUD element.
            content: HTML content to display.
            position: One of "top-left", "top-right", "bottom-left",
                "bottom-right", "top", "bottom".
        """
        css_pos = {
            "top-left": "top:8px;left:8px;",
            "top-right": "top:8px;right:320px;",
            "bottom-left": "bottom:8px;left:8px;",
            "bottom-right": "bottom:8px;right:320px;",
            "top": "top:8px;left:50%;transform:translateX(-50%);",
            "bottom": "bottom:8px;left:50%;transform:translateX(-50%);",
        }.get(position, "bottom-left:8px;left:8px;")

        html = (
            f'<div style="position:fixed;{css_pos}z-index:1000;'
            f"background:rgba(0,0,0,0.7);color:white;padding:4px 10px;"
            f"border-radius:4px;font-family:monospace;font-size:12px;"
            f'pointer-events:none;white-space:nowrap;">'
            f"{content}</div>"
        )

        if name in self._huds:
            self._huds[name].content = html
        else:
            self._huds[name] = self.server.gui.add_html(html)

    def remove_hud(self, name: str) -> None:
        """Remove an HUD overlay."""
        if name in self._huds:
            self._huds[name].remove()
            del self._huds[name]

    def on_select(self, callback: Callable[[int, str], None]) -> None:
        """Register a callback for geom selection: callback(geom_id, body_name)."""
        self._select_callbacks.append(callback)
        if self._scene_mgr is not None:
            self._scene_mgr.on_select(callback)

    def add_panel(self, panel: PanelBase) -> None:
        """Register a custom GUI panel. Panels added while open are set up immediately."""
        self._panels.append(panel)
        if self._server is not None:
            panel.setup(self._server.gui, self)

    def open(self, open_browser: bool = False) -> str:
        """Start the server and build the scene from the model's default state.

        Returns:
            The URL the server is bound to.

        Raises:
            RuntimeError: If the viewer is already open.
            OSError: If ``host``/``port`` cannot be bound.
        """
        if self._server is not None:
            raise RuntimeError("MujocoViewer is already open")

        # Viser blocks forever when it cannot bind (its worker exits without
        # signalling ready), and silently moves to the next port when the
        # requested one is taken. Probe first so both surface here as OSError.
        _check_bindable(self._host, self._port)
        server = viser.ViserServer(host=self._host, port=self._port, verbose=False)
        try:
            if self._port != 0 and server.get_port() != self._port:
                raise OSError(f"port {self._port} on {self._host} became unavailable during startup")
            self._server = server
            self._build()
        except BaseException:
            self._teardown()
            server.stop()
            raise

        url = self.url
        if open_browser:
            webbrowser.open(url)
        return url

    def update(self, data: mujoco.MjData) -> None:
        """Render ``data``.

        Copies ``data`` into the viewer's own ``MjData`` and recomputes
        kinematics on the copy, so the rendered pose always follows ``qpos``
        and mocap poses even if the caller did not run forward kinematics.
        ``data`` is not modified or retained.

        Raises:
            RuntimeError: If the viewer is not open.
            ValueError: If ``data`` was not created from this viewer's ``MjModel`` object.
        """
        if self._scene_mgr is None:
            raise RuntimeError("MujocoViewer is not open; call open() first")
        # Size checks are not enough: same-sized models can differ structurally
        # (a slide read as a hinge), and differing layouts crash mj_copyData.
        if data.model is not self._model:
            raise ValueError("data was not created from this viewer's MjModel")

        with self._lock:
            mujoco.mj_copyData(self._data, self._model, data)
            mujoco.mj_kinematics(self._model, self._data)
            self._scene_mgr.update_transforms()
            for panel in self._panels:
                panel.on_sync(self)

    def close(self) -> None:
        """Stop the server and release its port. Safe to call more than once."""
        server = self._server
        self._teardown()
        if server is not None:
            server.stop()

    def __enter__(self) -> MujocoViewer:
        if self._server is None:
            self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Scene + GUI setup for a freshly started server."""
        assert self._server is not None
        if self._label is not None:
            from viser._messages import RunJavascriptMessage

            self._server._websock_interface.queue_message(
                RunJavascriptMessage(source=f"document.title = {self._label!r};")
            )

        with self._lock:
            # Render the model's default pose until the first update().
            self._data = mujoco.MjData(self._model)
            mujoco.mj_kinematics(self._model, self._data)
            self._scene_mgr = SceneManager(self._server, self._model, self._data, self._lock)
            for cb in self._select_callbacks:
                self._scene_mgr.on_select(cb)
            self._scene_mgr.build_scene()

            if self._show_visibility:
                self._gui_mgr = GuiManager(self._server, self._model)
                scene_mgr, gui_mgr = self._scene_mgr, self._gui_mgr
                scene_mgr.update_visibility(gui_mgr.visible_groups())
                gui_mgr.on_visibility_change(lambda: scene_mgr.update_visibility(gui_mgr.visible_groups()))

        for panel in self._panels:
            panel.setup(self._server.gui, self)

    def _teardown(self) -> None:
        with self._lock:
            self._server = None
            self._scene_mgr = None
            self._gui_mgr = None
            self._huds.clear()


def _check_bindable(host: str, port: int) -> None:
    """Raise OSError if a TCP server could not bind ``host:port`` right now.

    Mirrors asyncio's server socket setup (every resolved address, SO_REUSEADDR
    on POSIX, IPV6_V6ONLY) so a port just released by close() is not reported
    as busy.
    """
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    ):
        with socket.socket(family, socktype, proto) as sock:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind(sockaddr)
            except OSError as e:
                raise OSError(e.errno, f"cannot serve on {host}:{port}: {e.strerror}") from e
