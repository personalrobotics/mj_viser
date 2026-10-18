# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""What a run's trace says, by name: channels, executions, scene, and runtime events.

:class:`ChannelModel` reads trace records, as ssrobot writes them, and keeps what a
viewer shows: each declared channel's latest reading and status, each execution's
command and state history, the scene revision and what is held, and recent runtime
events. It is the one presentation path for live views and replays: pass it to a
``RobotContext`` as a sink (it is callable) and to :class:`~mj_viser.LiveView` for GUI
panels, or feed it a recorded trace. Names come from the trace and the robot
description, never from MuJoCo indices. Reading records changes nothing else.
"""

from __future__ import annotations

import html
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from viser import uplot

# Quantities this viewer does not show; depth waits for a consumer.
UNSUPPORTED = frozenset({"depth_image"})
IMAGE = "rgb_image"
_LATEST = 3  # executions shown; earlier ones are counted, so panels and labels keep their size


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))


@dataclass
class _Channel:
    name: str
    quantity: str
    source: str
    labels: tuple[str, ...]
    confirmed: bool = False
    value: Any = None
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
    """What it measures, e.g. ``joint_position``; ``unknown`` without a description."""
    status: str
    """``current``; ``stale`` (no reading for ``stale_after_ns``); ``awaiting`` (no
    reading yet); ``malformed`` (the latest reading is not what the quantity needs);
    ``unavailable`` (declared, but the runtime does not provide it); or ``unsupported``
    (a quantity this viewer does not show)."""
    text: str
    """The latest reading, with element names, or why there is none."""
    stamp_ns: int | None
    """When the latest reading was taken."""


@dataclass
class _Execution:
    command: str
    states: list[str]


class ChannelModel:
    """Named observations, executions, scene, and events from trace records.

    Thread-safe: a runtime's thread may feed it while a viewer's threads read it.
    """

    def __init__(self, description: Any = None, *, stale_after_ns: int = 1_000_000_000, events: int = 5) -> None:
        """
        Args:
            description: The robot description (e.g. ``ctx.description``), for channel
                quantities and joint names. Without it, channels are named by the trace
                alone.
            stale_after_ns: A channel whose latest reading is older than this, in trace
                time, is ``stale``.
            events: How many recent runtime events to keep.
        """
        self._lock = threading.RLock()
        self._stale_after = stale_after_ns
        self._channels: dict[str, _Channel] = {}
        if description is not None:
            joints = {g.name: tuple(g.joints) for g in description.groups}
            for c in description.channels:
                quantity, source = _enum(c.quantity), str(c.source)
                labels = joints.get(source, ()) if quantity.startswith("joint_") else _LABELS.get(quantity, ())
                self._channels[c.name] = _Channel(c.name, quantity, source, labels)
        self._declared = description is not None
        self._executions: dict[str, _Execution] = {}
        self._revision = 0
        self._held: tuple[str, ...] = ()
        self._events: deque[str] = deque(maxlen=events)
        self._now: int | None = None
        self._opened = False

    def __call__(self, record: Any) -> None:
        """Accept one trace record: the ssrobot sink interface."""
        self.accept(record)

    def accept(self, record: Any) -> None:
        """Accept one trace record, in trace order."""
        kind, p = _enum(record.kind), record.payload
        with self._lock:
            self._now = record.time_ns
            if kind == "opened":
                self._opened = True
                for name in p.channels:
                    channel = self._channels.get(name)
                    if channel is None:
                        channel = self._channels[name] = _Channel(name, "unknown", "", ())
                    channel.confirmed = True
                self._events.append(f"opened {p.runtime} ({_enum(p.clock_mode)} clock)")
            elif kind == "observed":
                for r in p.readings:
                    channel = self._channels.get(r.channel)
                    if channel is None:  # not declared: shown, as the trace has it
                        channel = self._channels[r.channel] = _Channel(r.channel, "unknown", "", (), True)
                    channel.value = r.value
                    channel.stamp_ns = r.stamp.time_ns
                    channel.readings += 1
                    if channel.quantity == IMAGE:
                        channel.image, channel.malformed = _image(r.value)
                    elif getattr(r.value, "shape", None) is None:
                        channel.history.append((r.stamp.time_ns, tuple(float(v) for v in r.value)))
            elif kind == "submitted":
                self._executions[p.execution] = _Execution(_command(p.command), ["submitted"])
            elif kind == "status":
                why = f" ({p.diagnostic.code})" if p.diagnostic is not None else ""
                self._executions.setdefault(p.execution, _Execution("?", [])).states.append(_enum(p.state) + why)
            elif kind == "applied" and p.modifications:
                changes = ", ".join(f"{_enum(m.kind)} {m.target}" for m in p.modifications)
                self._events.append(f"{p.execution}: applied, {changes}")
            elif kind == "scene":
                self._revision = p.revision
                self._held = tuple(f"{a.object} held by {a.end_effector}" for a in p.attachments if a.held)
            elif kind == "violation":
                self._events.append(f"{p.object} slipped {p.position_error * 1000:.1f} mm, {p.rotation_error:.3f} rad")
            elif kind == "health":
                why = f" ({p.diagnostic.code})" if p.diagnostic is not None else ""
                self._events.append(f"runtime {_enum(p.state)}{why}")
            elif kind == "compiled":
                self._events.append(f"compiled a trajectory for {p.command.group}")
            elif kind == "closed":
                self._events.append("closed")

    def feed(self, records: Iterable[Any]) -> None:
        for r in records:
            self.accept(r)

    @property
    def now_ns(self) -> int | None:
        """Time of the latest record."""
        return self._now

    def scene(self) -> str:
        with self._lock:
            return f"scene revision {self._revision}" + (f": {', '.join(self._held)}" if self._held else "")

    def executions(self, latest: int | None = None) -> list[tuple[str, str]]:
        """Executions in submission order, the ``latest`` few if given: each one's id and
        its command with its states. An earlier ones' count leads, as ``("…", "N earlier")``."""
        with self._lock:
            items = list(self._executions.items())
            shown = items if latest is None else items[-latest:]
            out = [(e, f"{x.command} · {' → '.join(x.states)}") for e, x in shown]
            earlier = len(items) - len(shown)
            return ([("…", f"{earlier} earlier")] if earlier else []) + out

    def events(self) -> list[str]:
        """Recent runtime events, oldest first."""
        with self._lock:
            return list(self._events)

    def channels(self) -> list[ChannelState]:
        """Every channel, sorted by name."""
        with self._lock:
            return [self._state(c) for _, c in sorted(self._channels.items())]

    def history(self, name: str) -> tuple[tuple[str, ...], list[tuple[int, tuple[float, ...]]]]:
        """A numeric channel's element labels and its recent (stamp, values) readings."""
        with self._lock:
            c = self._channels[name]
            return c.labels, list(c.history)

    def image(self, name: str) -> tuple[np.ndarray | None, int | None]:
        """An image channel's latest well-formed image (H×W×3 ``uint8``) and its stamp;
        no image if there is no reading yet or the latest is malformed."""
        with self._lock:
            c = self._channels[name]
            return c.image, c.stamp_ns

    def coverage(self) -> dict[str, dict[str, Any]]:
        """Every channel: ``displayed``, ``unavailable``, or ``unsupported``, with its
        quantity and how many readings it had."""
        out = {}
        with self._lock:
            for name, c in sorted(self._channels.items()):
                status = self._state(c).status
                shown = "displayed" if status in ("current", "stale", "awaiting") else status
                out[name] = {"status": shown, "quantity": c.quantity, "readings": c.readings}
        return out

    def lines(self) -> list[str]:
        """The whole state as text: time, scene, executions, channels with readings,
        and recent events."""
        with self._lock:
            lines = [f"t = {(self._now or 0) / 1e9:.3f} s", self.scene()]
            lines += [f"{e}: {text}" for e, text in self.executions(latest=_LATEST)]
            shown = ("current", "stale", "malformed")
            lines += [f"{c.name} = {c.text}" for c in self.channels() if c.status in shown]
            return lines + self.events()[-1:]

    def _state(self, c: _Channel) -> ChannelState:
        if c.quantity in UNSUPPORTED:
            return ChannelState(c.name, c.quantity, "unsupported", f"{c.quantity} is not shown yet", c.stamp_ns)
        if self._opened and not c.confirmed:
            return ChannelState(c.name, c.quantity, "unavailable", "the runtime does not provide it", None)
        if c.stamp_ns is None:
            return ChannelState(c.name, c.quantity, "awaiting", "no reading yet", None)
        if c.malformed is not None:
            return ChannelState(c.name, c.quantity, "malformed", c.malformed, c.stamp_ns)
        stale = self._now is not None and self._now - c.stamp_ns > self._stale_after
        text = _values(c.value, c.labels) + f" (t = {c.stamp_ns / 1e9:.3f} s{', stale' if stale else ''})"
        return ChannelState(c.name, c.quantity, "stale" if stale else "current", text, c.stamp_ns)


_LABELS = {
    "pose": ("x", "y", "z", "qw", "qx", "qy", "qz"),
    "wrench": ("fx", "fy", "fz", "tx", "ty", "tz"),
    "gripper_opening": ("opening",),
}


def _image(value: Any) -> tuple[np.ndarray | None, str | None]:
    """An RGB reading as an H×W×3 ``uint8`` array, or why it is not one."""
    shape, dtype, data = getattr(value, "shape", None), getattr(value, "dtype", None), getattr(value, "data", None)
    if shape is None or data is None:
        return None, "malformed: not an image array"
    shape = tuple(int(n) for n in shape)
    if _enum(dtype) != "uint8" or len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
        return None, f"malformed: {_enum(dtype)} {'×'.join(map(str, shape))}, not uint8 H×W×3"
    if len(data) != shape[0] * shape[1] * 3:
        return None, f"malformed: {len(data)} bytes for {'×'.join(map(str, shape))}"
    return np.frombuffer(data, np.uint8).reshape(shape), None


def _values(value: Any, labels: tuple[str, ...]) -> str:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{getattr(value, 'dtype', 'array')} {'×'.join(str(n) for n in shape)}"
    if len(labels) == len(value):
        return ", ".join(f"{label} {float(v):.3f}" for label, v in zip(labels, value, strict=True))
    return "[" + ", ".join(f"{float(v):.3f}" for v in value) + "]"


def _command(command: Any) -> str:
    inner = getattr(command, "trajectory", None) or command
    target = getattr(inner, "group", None) or getattr(inner, "gripper", None)
    return type(command).__name__ + (f" on {target}" if target else "")


_NO_PLOT = "(none)"
# A series without a stroke draws nothing.
_COLORS = ("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f")


class ChannelPanels:
    """GUI panels of a :class:`ChannelModel` for each connected browser.

    Each client gets its own panels, so one client's choices (which channel to plot,
    which folders are open) never reach another. The panels only show; they hold no
    control of the run.
    """

    def __init__(self, model: ChannelModel) -> None:
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
        with gui.add_folder("Executions", order=11):
            self.executions = gui.add_html("")
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

    def render(self, model: ChannelModel) -> None:
        with self._lock:
            self._render(model)

    def _render(self, model: ChannelModel) -> None:
        now = model.now_ns
        run = [(f"t = {(now or 0) / 1e9:.3f} s",), (model.scene(),)] + [(e,) for e in model.events()]
        self.run.content = _table(run)
        self.executions.content = _table(model.executions(latest=_LATEST) or [("no executions yet",)])
        states = model.channels()
        self.channels.content = _table([(c.name, c.status, c.text) for c in states])
        numeric = tuple(c.name for c in states if c.stamp_ns is not None and c.status in ("current", "stale"))
        numeric = tuple(n for n in numeric if model.history(n)[1])
        if self.choice.options != (_NO_PLOT, *numeric):
            self.choice.options = (_NO_PLOT, *numeric)
        self._render_plot(model, self.choice.value)
        for c in states:
            if c.quantity == IMAGE:
                self._render_image(model, c)

    def _render_image(self, model: ChannelModel, c: ChannelState) -> None:
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

    def _render_plot(self, model: ChannelModel, name: str) -> None:
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
