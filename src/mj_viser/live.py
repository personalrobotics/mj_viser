# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Live view of a running simulation through a read-only render source.

A render source (:class:`RenderSource`) owns its own copy of the model and state,
refreshed only when asked, so nothing a viewer does can reach the simulation. Pass one
to :class:`LiveView` and call :meth:`LiveView.refresh` from your loop. Any runtime that
can expose a MuJoCo model and state can provide one;
``mj_viser.integrations.ssrobot.render_source(ctx)`` adapts ssrobot's.
"""

from __future__ import annotations

import math
import time
import webbrowser
from typing import Protocol

import mujoco

from mj_viser.channels import ChannelBoard, ChannelPanels
from mj_viser.viewer import MujocoViewer


class SourceStale(Exception):
    """Raised by :meth:`RenderSource.capture` when the source can no longer describe
    its simulation (for example, the simulation reopened); make a new source."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RenderFrame(Protocol):
    """What a source's data shows after a capture."""

    time_ns: int
    """Simulation time, in nanoseconds."""
    revision: int
    """Scene revision."""
    world: str
    """Identity of the model and scene the data describes."""


class RenderSource(Protocol):
    """A read-only, caller-refreshed view of a simulation's model and state."""

    model: mujoco.MjModel
    """The source's own copy of the model."""
    data: mujoco.MjData
    """The source's own ``MjData`` of ``model``, as of the last capture."""
    world: str
    """Identity of the model and scene ``model`` was copied from."""

    def capture(self) -> RenderFrame:
        """Copy the simulation's current state into ``data`` and describe it.

        Raises:
            SourceStale: If the source can no longer describe its simulation.
        """
        ...


class LiveView:
    """A :class:`MujocoViewer` that follows a running simulation through a render source.

    The simulation owns time and stepping; this view only reads. Call :meth:`refresh`
    whenever you like, typically after each step: it captures the source and renders the
    frame, at most ``max_rate_hz`` times a second, and never raises because the viewer
    was closed or a browser went away. Close it at any time; the simulation keeps
    running. Attach a new view, with a new source, whenever you like.
    """

    def __init__(
        self,
        source: RenderSource,
        *,
        expected_world: str | None = None,
        channels: ChannelBoard | None = None,
        max_rate_hz: float | None = 30.0,
        host: str = "127.0.0.1",
        port: int = 8080,
        label: str | None = None,
        show_visibility: bool = True,
    ) -> None:
        """
        Args:
            source: The render source, for example
                ``mj_viser.integrations.ssrobot.render_source(ctx)``.
            expected_world: If given, the scene identity the source must describe;
                anything else is refused before rendering.
            channels: A :class:`~mj_viser.ChannelBoard` the caller feeds. Each browser
                then gets panels of its channels, status, and events, updated with each
                frame.
            max_rate_hz: Most frames rendered per second. Faster calls to
                :meth:`refresh` are coalesced. None renders every call.
            host, port, label, show_visibility: As for :class:`MujocoViewer`.

        Raises:
            ValueError: If the source describes a different scene than
                ``expected_world``, or ``max_rate_hz`` is not positive.
        """
        if expected_world is not None and source.world != expected_world:
            raise ValueError(f"source describes scene {source.world!r}, expected {expected_world!r}")
        if max_rate_hz is not None and not max_rate_hz > 0:
            raise ValueError(f"max_rate_hz must be positive or None, got {max_rate_hz!r}")
        self._source = source
        self._world = source.world
        self._min_interval = 0.0 if max_rate_hz is None else 1.0 / max_rate_hz
        self._last_render = -math.inf
        self._stale: str | None = None
        self._frame: RenderFrame | None = None
        self._panels = None if channels is None else ChannelPanels(channels)
        self.viewer = MujocoViewer(source.model, host=host, port=port, label=label, show_visibility=show_visibility)
        """The underlying viewer, for panels, HUDs, and visibility groups."""

    @property
    def world(self) -> str:
        """The scene identity this view renders."""
        return self._world

    @property
    def stale(self) -> str | None:
        """Why the source can no longer be rendered (the :class:`SourceStale` reason, or
        ``"world_changed"``), or None. A stale view renders nothing more;
        attach a new one with a new source."""
        return self._stale

    @property
    def frame(self) -> RenderFrame | None:
        """The latest frame rendered, or None before the first."""
        return self._frame

    @property
    def is_open(self) -> bool:
        return self.viewer.is_open

    def open(self, open_browser: bool = False) -> str:
        """Start the viewer and show the simulation's current state. Returns the URL.

        The first frame must render: if it cannot, the viewer is closed again, releasing
        its port, and nothing is left showing the model's default state.

        Raises:
            SourceStale: If the source is stale at the first capture, or its first frame
                describes another world (reason ``"world_changed"``).
        """
        url = self.viewer.open()
        try:
            if self._panels is not None:
                self._panels.attach(self.viewer.server)
            if self.refresh(force=True) is None:
                raise SourceStale(self._stale or "no frame")
        except BaseException:
            self.close()
            raise
        if open_browser:
            webbrowser.open(url)
        return url

    def close(self) -> None:
        """Stop the viewer. The simulation is unaffected. Safe to call more than once."""
        if self._panels is not None:
            self._panels.detach()
        self.viewer.close()

    def refresh(self, force: bool = False) -> RenderFrame | None:
        """Capture the source and render it; returns the frame shown, or None if nothing was.

        Nothing is rendered when the view is closed or stale, or when the last frame was
        rendered less than ``1 / max_rate_hz`` seconds ago (unless ``force``).
        """
        if not self.viewer.is_open or self._stale is not None:
            return None
        now = time.monotonic()
        if not force and now - self._last_render < self._min_interval:
            return None
        try:
            frame = self._source.capture()
        except SourceStale as e:
            self._mark_stale(e.reason)
            return None
        if frame.world != self._world:
            self._mark_stale("world_changed")
            return None
        self._frame = frame
        self.viewer.update(self._source.data)
        self.viewer.set_hud("runtime", f"t = {frame.time_ns / 1e9:.3f} s · scene revision {frame.revision}")
        if self._panels is not None:
            self._panels.render()
        self._last_render = now
        return frame

    def _mark_stale(self, reason: str) -> None:
        self._stale = reason
        self.viewer.set_hud("runtime", f"stopped following the simulation: {reason}")
