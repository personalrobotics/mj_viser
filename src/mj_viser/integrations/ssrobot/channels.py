# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""ssrobot trace records as a :class:`mj_viser.ChannelBoard`.

:class:`TraceChannels` is an ssrobot trace sink: pass it to ``RobotContext(...,
sinks=[...])`` and its ``board`` to :class:`mj_viser.LiveView`, or feed it a recorded
trace. It declares the robot description's channels by name (joint names for joint
channels, components for poses, wrenches, and openings), and turns each record into
board readings, status lines (the scene revision, and each execution's command and
states), and events.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from mj_viser.channels import ChannelBoard, ChannelKind

LATEST = 3
"""Executions shown; earlier ones are counted, so panels and labels keep their size."""

_LABELS = {
    "pose": ("x", "y", "z", "qw", "qx", "qy", "qz"),
    "wrench": ("fx", "fy", "fz", "tx", "ty", "tz"),
    "gripper_opening": ("opening",),
}
_UNSHOWN = {"depth_image": "depth images are not shown", "unknown": "not described: quantity unknown"}
_UNKNOWN = _UNSHOWN["unknown"]


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value))


class TraceChannels:
    """An ssrobot trace sink that keeps a :class:`mj_viser.ChannelBoard` up to date."""

    def __init__(self, description: Any = None, *, stale_after_ns: int = 1_000_000_000) -> None:
        """
        Args:
            description: The robot description (``ctx.description``), for channel
                quantities and names. Without it, or for a channel it does not name,
                a channel is listed by the trace's name, as unsupported (quantity
                unknown) until a numeric reading shows it is a vector of values.
            stale_after_ns: As for :class:`mj_viser.ChannelBoard`.
        """
        self.board = ChannelBoard(stale_after_ns=stale_after_ns)
        self._quantity: dict[str, str] = {}
        if description is not None:
            joints = {g.name: tuple(g.joints) for g in description.groups}
            for c in description.channels:
                quantity = _enum(c.quantity)
                labels = joints.get(str(c.source), ()) if quantity.startswith("joint_") else _LABELS.get(quantity, ())
                kind: ChannelKind = (
                    "unsupported" if quantity in _UNSHOWN else "image" if quantity == "rgb_image" else "values"
                )
                self.board.declare(c.name, kind=kind, quantity=quantity, labels=labels, note=_UNSHOWN.get(quantity, ""))
                self._quantity[c.name] = quantity
        self._executions: dict[str, tuple[str, list[str]]] = {}
        self.board.set_status("scene", "scene revision 0")

    def __call__(self, record: Any) -> None:
        """Accept one trace record: the ssrobot sink interface."""
        self.accept(record)

    def feed(self, records: Iterable[Any]) -> None:
        for r in records:
            self.accept(r)

    def accept(self, record: Any) -> None:
        """Accept one trace record, in trace order."""
        kind, p, board = _enum(record.kind), record.payload, self.board
        board.advance(record.time_ns)
        if kind == "opened":
            confirmed = set(p.channels)
            for name in self._quantity:
                if self._quantity[name] not in _UNSHOWN:
                    board.set_available(name, name in confirmed)
            for name in sorted(confirmed - set(self._quantity)):
                self._unknown(name)
            board.event(f"opened {p.runtime} ({_enum(p.clock_mode)} clock)")
        elif kind == "observed":
            for r in p.readings:
                stamp = r.stamp.time_ns
                # Not described: listed as of unknown quantity, and shown once a numeric
                # reading says it is a vector; nothing in the trace is dropped.
                self._unknown(r.channel)
                if self._quantity[r.channel] == "unknown" and getattr(r.value, "shape", None) is None:
                    self._quantity[r.channel] = "unknown numeric"
                    board.declare(r.channel, quantity="unknown")
                if self._quantity.get(r.channel) == "rgb_image":
                    image, reason = _image(r.value)
                    if image is None:
                        board.malformed(r.channel, stamp, reason)
                    else:
                        board.reading(r.channel, stamp, image)
                elif self._quantity[r.channel] in _UNSHOWN or getattr(r.value, "shape", None) is not None:
                    board.reading(r.channel, stamp, ())  # counted, not shown
                else:
                    board.reading(r.channel, stamp, r.value)
        elif kind == "submitted":
            self._executions[p.execution] = (_command(p.command), ["submitted"])
            self._show_executions()
        elif kind == "status":
            why = f" ({p.diagnostic.code})" if p.diagnostic is not None else ""
            self._executions.setdefault(p.execution, ("?", []))[1].append(_enum(p.state) + why)
            self._show_executions()
        elif kind == "applied" and p.modifications:
            changes = ", ".join(f"{_enum(m.kind)} {m.target}" for m in p.modifications)
            board.event(f"{p.execution}: applied, {changes}")
        elif kind == "scene":
            held = ", ".join(f"{a.object} held by {a.end_effector}" for a in p.attachments if a.held)
            board.set_status("scene", f"scene revision {p.revision}" + (f": {held}" if held else ""))
        elif kind == "violation":
            board.event(f"{p.object} slipped {p.position_error * 1000:.1f} mm, {p.rotation_error:.3f} rad")
        elif kind == "health":
            why = f" ({p.diagnostic.code})" if p.diagnostic is not None else ""
            board.event(f"runtime {_enum(p.state)}{why}")
        elif kind == "compiled":
            board.event(f"compiled a trajectory for {p.command.group}")
        elif kind == "closed":
            board.event("closed")

    def _unknown(self, name: str) -> None:
        """List a channel the description does not name, until its readings say more."""
        if name not in self._quantity:
            self._quantity[name] = "unknown"
            self.board.declare(name, kind="unsupported", quantity="unknown", note=_UNKNOWN)

    def _show_executions(self) -> None:
        """The latest executions as status lines, after a count of earlier ones."""
        for key in [k for k in list(self._executions) + ["earlier"]]:
            self.board.set_status(f"execution {key}", None)
        items = list(self._executions.items())
        if len(items) > LATEST:
            self.board.set_status("execution earlier", f"…: {len(items) - LATEST} earlier")
        for execution, (command, states) in items[-LATEST:]:
            self.board.set_status(f"execution {execution}", f"{execution}: {command} · {' → '.join(states)}")


def _image(value: Any) -> tuple[np.ndarray | None, str]:
    """An ssrobot array reading as an H×W×3 ``uint8`` image, or why it is not one."""
    shape, dtype, data = getattr(value, "shape", None), getattr(value, "dtype", None), getattr(value, "data", None)
    if shape is None or data is None:
        return None, "malformed: not an image array"
    shape = tuple(int(n) for n in shape)
    if _enum(dtype) != "uint8" or len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
        return None, f"malformed: {_enum(dtype)} {'×'.join(map(str, shape))}, not uint8 H×W×3"
    if len(data) != shape[0] * shape[1] * 3:
        return None, f"malformed: {len(data)} bytes for {'×'.join(map(str, shape))}"
    return np.frombuffer(data, np.uint8).reshape(shape), ""


def _command(command: Any) -> str:
    inner = getattr(command, "trajectory", None) or command
    target = getattr(inner, "group", None) or getattr(inner, "gripper", None)
    return type(command).__name__ + (f" on {target}" if target else "")
