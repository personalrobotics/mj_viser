# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Named, timestamped channels and status, for live panels and recordings.

A :class:`ChannelBoard` holds what a viewer shows about a run, by name: each declared
channel's latest reading (a vector of values or an RGB image) with its time and status,
ordered status lines, and recent events. The caller feeds it; it interprets nothing
itself. :class:`~mj_viser.LiveView` shows a board in per-browser panels, and
:class:`~mj_viser.StateRecorder` can record its text and images.
``mj_viser.integrations.ssrobot`` feeds one from ssrobot traces.
"""

from __future__ import annotations

import html
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from viser import uplot

ChannelKind = Literal["values", "image", "unsupported"]


@dataclass
class _Channel:
    name: str
    kind: ChannelKind
    quantity: str
    labels: tuple[str, ...]
    note: str = ""
    available: bool = True
    values: tuple[float, ...] | None = None
    stamp_ns: int | None = None
    readings: int = 0
    history: deque[tuple[int, tuple[float, ...]]] = field(default_factory=lambda: deque(maxlen=500))
    image: np.ndarray | None = None
    malformed: str | None = None


@dataclass(frozen=True)
class ChannelState:
    """One channel as a viewer shows it."""

    name: str
    quantity: str
    """What it measures, as declared (e.g. ``joint_position``); free text."""
    kind: ChannelKind
    status: str
    """``current``; ``stale`` (no reading for ``stale_after_ns``); ``awaiting`` (no
    reading yet); ``malformed`` (the latest reading is not what the kind needs);
    ``unavailable`` (declared, but not provided); or ``unsupported`` (a kind this
    viewer does not show)."""
    text: str
    """The latest reading, with element names, or why there is none."""
    stamp_ns: int | None
    """When the latest reading was taken."""


class ChannelBoard:
    """Named channels with their latest readings, status lines, and recent events.

    Thread-safe: a simulation's thread may feed it while a viewer's threads read it.
    """

    def __init__(self, *, stale_after_ns: int = 1_000_000_000, events: int = 5) -> None:
        """
        Args:
            stale_after_ns: A channel whose latest reading is older than this, in the
                board's time, is ``stale``.
            events: How many recent events to keep.
        """
        self._lock = threading.RLock()
        self._stale_after = stale_after_ns
        self._channels: dict[str, _Channel] = {}
        self._status: dict[str, str] = {}
        self._events: deque[str] = deque(maxlen=events)
        self._now: int | None = None

    # -- feeding -----------------------------------------------------------------

    def declare(
        self,
        name: str,
        *,
        kind: ChannelKind = "values",
        quantity: str = "",
        labels: Sequence[str] = (),
        available: bool = True,
        note: str = "",
    ) -> None:
        """Declare a channel: ``values`` (a vector, named element-wise by ``labels``),
        ``image`` (H×W×3 ``uint8``), or ``unsupported`` (listed, with ``note``, but not
        shown). Declaring again updates the declaration and keeps the readings."""
        with self._lock:
            c = self._channels.get(name)
            if c is None:
                self._channels[name] = _Channel(name, kind, quantity, tuple(labels), note, available)
            else:
                c.kind, c.quantity, c.labels, c.available, c.note = kind, quantity, tuple(labels), available, note

    def set_available(self, name: str, available: bool) -> None:
        """Mark a declared channel as provided, or not (``unavailable``)."""
        with self._lock:
            self._channels[name].available = available

    def reading(self, name: str, stamp_ns: int, value: Sequence[float] | np.ndarray) -> None:
        """A channel's reading taken at ``stamp_ns``. An undeclared channel is declared
        from it. An image channel's reading that is not an H×W×3 ``uint8`` array is
        recorded as malformed."""
        with self._lock:
            c = self._channels.get(name)
            if c is None:
                kind: ChannelKind = "image" if isinstance(value, np.ndarray) and value.ndim == 3 else "values"
                c = self._channels[name] = _Channel(name, kind, "", ())
            c.stamp_ns = int(stamp_ns)
            c.readings += 1
            if c.kind == "image":
                c.image, c.malformed = _image(value)
            elif c.kind == "values":
                c.values = tuple(float(v) for v in value)
                c.malformed = None
                c.history.append((c.stamp_ns, c.values))

    def malformed(self, name: str, stamp_ns: int, reason: str) -> None:
        """A channel's reading at ``stamp_ns`` that cannot be shown, and why."""
        with self._lock:
            c = self._channels[name]
            c.stamp_ns, c.malformed, c.image = int(stamp_ns), reason, None
            c.readings += 1

    def advance(self, now_ns: int) -> None:
        """The board's time: readings older than ``stale_after_ns`` before it are stale."""
        with self._lock:
            self._now = int(now_ns)

    def set_status(self, key: str, text: str | None) -> None:
        """Set a status line, kept in first-set order, or remove it (``None``)."""
        with self._lock:
            if text is None:
                self._status.pop(key, None)
            else:
                self._status[key] = text

    def event(self, text: str) -> None:
        """Note an event; the most recent few are kept."""
        with self._lock:
            self._events.append(text)

    # -- reading -----------------------------------------------------------------

    @property
    def now_ns(self) -> int | None:
        return self._now

    def status(self) -> list[str]:
        """Status lines, in first-set order."""
        with self._lock:
            return list(self._status.values())

    def events(self) -> list[str]:
        """Recent events, oldest first."""
        with self._lock:
            return list(self._events)

    def channels(self) -> list[ChannelState]:
        """Every channel, sorted by name."""
        with self._lock:
            return [self._state(c) for _, c in sorted(self._channels.items())]

    def history(self, name: str) -> tuple[tuple[str, ...], list[tuple[int, tuple[float, ...]]]]:
        """A values channel's element labels and its recent (stamp, values) readings."""
        with self._lock:
            c = self._channels[name]
            return c.labels, list(c.history)

    def image(self, name: str) -> tuple[np.ndarray | None, int | None]:
        """An image channel's latest well-formed image and its stamp; no image if there
        is no reading yet or the latest is malformed."""
        with self._lock:
            c = self._channels[name]
            return c.image, c.stamp_ns

    def coverage(self) -> dict[str, dict[str, Any]]:
        """Every channel: ``displayed``, ``malformed``, ``unavailable``, or
        ``unsupported``, with its quantity and how many readings it had."""
        out = {}
        with self._lock:
            for name, c in sorted(self._channels.items()):
                status = self._state(c).status
                shown = "displayed" if status in ("current", "stale", "awaiting") else status
                out[name] = {"status": shown, "quantity": c.quantity, "readings": c.readings}
        return out

    def lines(self) -> list[str]:
        """Everything as text: time, status lines, channels with readings, and the
        latest event."""
        with self._lock:
            lines = [f"t = {(self._now or 0) / 1e9:.3f} s", *self._status.values()]
            shown = ("current", "stale", "malformed")
            lines += [f"{c.name} = {c.text}" for c in self.channels() if c.status in shown]
            return lines + list(self._events)[-1:]

    def _state(self, c: _Channel) -> ChannelState:
        def state(status: str, text: str, stamp: int | None) -> ChannelState:
            return ChannelState(c.name, c.quantity, c.kind, status, text, stamp)

        if c.kind == "unsupported":
            return state("unsupported", c.note or "not shown", c.stamp_ns)
        if not c.available:
            return state("unavailable", "not provided", None)
        if c.stamp_ns is None:
            return state("awaiting", "no reading yet", None)
        if c.malformed is not None:
            return state("malformed", c.malformed, c.stamp_ns)
        stale = self._now is not None and self._now - c.stamp_ns > self._stale_after
        if c.kind == "image":
            assert c.image is not None
            shown = "uint8 " + "×".join(str(n) for n in c.image.shape)
        else:
            shown = _values(c.values or (), c.labels)
        return state(
            "stale" if stale else "current",
            f"{shown} (t = {c.stamp_ns / 1e9:.3f} s{', stale' if stale else ''})",
            c.stamp_ns,
        )


def _image(value: Any) -> tuple[np.ndarray | None, str | None]:
    """An image reading as an H×W×3 ``uint8`` array, or why it is not one."""
    if not isinstance(value, np.ndarray):
        return None, "malformed: not an image array"
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape) <= 0:
        return None, f"malformed: {value.dtype} {'×'.join(map(str, value.shape))}, not uint8 H×W×3"
    return value, None


def _values(values: Sequence[float], labels: tuple[str, ...]) -> str:
    if labels and len(labels) == len(values):
        return ", ".join(f"{label} {float(v):.3f}" for label, v in zip(labels, values, strict=True))
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"


_NO_PLOT = "(none)"
# A series without a stroke draws nothing.
_COLORS = ("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f")


class ChannelPanels:
    """GUI panels of a :class:`ChannelBoard` for each connected browser.

    Each client gets its own panels, so one client's choices (which channel to plot,
    which folders are open) never reach another. The panels only show; they hold no
    control of the run.
    """

    def __init__(self, model: ChannelBoard) -> None:
        self._model = model
        self._lock = threading.Lock()
        self._clients: dict[int, _ClientPanels] = {}

    def attach(self, server: Any) -> None:
        """Build panels for every client of ``server``, now and as they connect."""
        server.on_client_connect(self._connect)
        server.on_client_disconnect(lambda client: self._drop(client.client_id))
        for client in server.get_clients().values():
            self._connect(client)

    def detach(self) -> None:
        with self._lock:
            self._clients.clear()

    def render(self) -> None:
        """Show the model's current state in every client's panels."""
        with self._lock:
            clients = list(self._clients.values())
        for panels in clients:
            panels.render(self._model)

    def client(self, client_id: int) -> _ClientPanels | None:
        """A client's panels, if it is connected."""
        with self._lock:
            return self._clients.get(client_id)

    def _connect(self, client: Any) -> None:
        panels = _ClientPanels(client.gui)
        with self._lock:
            self._clients[client.client_id] = panels
        panels.render(self._model)

    def _drop(self, client_id: int) -> None:
        with self._lock:
            self._clients.pop(client_id, None)


def _table(rows: Sequence[Sequence[str]]) -> str:
    cells = "".join(
        "<tr>" + "".join(f'<td style="padding:0 6px 0 0;vertical-align:top">{html.escape(c)}</td>' for c in r) + "</tr>"
        for r in rows
    )
    return f'<table style="font-size:11px;border-collapse:collapse">{cells}</table>'


@dataclass
class _ImagePanel:
    caption: Any
    handle: Any = None
    stamp: int | None = None


class _ClientPanels:
    def __init__(self, gui: Any) -> None:
        self._gui = gui
        # A client's panels are rendered from the simulation's thread (each frame) and
        # from Viser's (when the client connects); one at a time.
        self._lock = threading.Lock()
        with gui.add_folder("Run", order=10):
            self.run = gui.add_html("")
        with gui.add_folder("Status", order=11):
            self.status = gui.add_html("")
        with gui.add_folder("Channels", order=12):
            self.channels = gui.add_html("")
        with gui.add_folder("Plot", order=13) as self._plot_folder:
            self.choice = gui.add_dropdown("Channel", (_NO_PLOT,), initial_value=_NO_PLOT)
        self._images_folder = gui.add_folder("Images", order=14)
        self._plot: Any = None
        self._plotted: str | None = None
        self._images: dict[str, _ImagePanel] = {}
        self.images_sent: dict[str, int] = {}
        """Images sent to this client, by channel: one per new reading at most."""

    def render(self, model: ChannelBoard) -> None:
        with self._lock:
            self._render(model)

    def _render(self, model: ChannelBoard) -> None:
        now = model.now_ns
        self.run.content = _table([(f"t = {(now or 0) / 1e9:.3f} s",)] + [(e,) for e in model.events()])
        self.status.content = _table([(line,) for line in model.status()] or [("nothing yet",)])
        states = model.channels()
        self.channels.content = _table([(c.name, c.status, c.text) for c in states])
        numeric = tuple(c.name for c in states if c.kind == "values" and c.status in ("current", "stale"))
        numeric = tuple(n for n in numeric if model.history(n)[1])
        if self.choice.options != (_NO_PLOT, *numeric):
            self.choice.options = (_NO_PLOT, *numeric)
        self._render_plot(model, self.choice.value)
        for c in states:
            if c.kind == "image":
                self._render_image(model, c)

    def _render_image(self, model: ChannelBoard, c: ChannelState) -> None:
        panel = self._images.get(c.name)
        if panel is None:
            with self._images_folder:
                panel = self._images[c.name] = _ImagePanel(self._gui.add_html(""))
        panel.caption.content = _table([(c.name, c.status, c.text)])
        image, stamp = model.image(c.name)
        if image is None or c.status not in ("current", "stale"):
            if panel.handle is not None:
                panel.handle.visible = False
            return
        if panel.handle is None:
            with self._images_folder:
                panel.handle = self._gui.add_image(image, label=c.name, format="jpeg")
        elif stamp != panel.stamp:
            panel.handle.image = image
        if not panel.handle.visible:
            panel.handle.visible = True
        if stamp == panel.stamp:
            return
        panel.stamp = stamp
        self.images_sent[c.name] = self.images_sent.get(c.name, 0) + 1

    def _render_plot(self, model: ChannelBoard, name: str) -> None:
        if name != self._plotted:
            if self._plot is not None:
                self._plot.remove()
                self._plot = None
            self._plotted = name
        if name == _NO_PLOT:
            return
        labels, history = model.history(name)
        width = len(history[-1][1])
        labels = labels if len(labels) == width else tuple(str(i) for i in range(width))
        times = np.array([t / 1e9 for t, _ in history])
        values = np.array([v for _, v in history]).reshape(len(history), width)
        data = (times, *(values[:, i] for i in range(width)))
        if self._plot is None:
            with self._plot_folder:
                self._plot = self._gui.add_uplot(
                    data=data,
                    series=(
                        uplot.Series(label="t (s)"),
                        *(
                            uplot.Series(label=label, stroke=_COLORS[i % len(_COLORS)], width=2)
                            for i, label in enumerate(labels)
                        ),
                    ),
                    aspect=1.5,
                )
        else:
            self._plot.data = data
