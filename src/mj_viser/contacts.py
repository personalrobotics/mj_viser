# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo contacts, as the state shown supplies them: points, normals, and forces.

:func:`contact_set` reads an ``MjData``'s own contact list, as the simulation computed
it; nothing is recomputed. Each contact names its two bodies, and carries its point,
normal, constraint force (MuJoCo's ``mj_contactForce``, turned into world coordinates),
and signed distance. :class:`ContactOverlay` draws the set the viewer shows, per
browser: points, normals, and force arrows, with each browser's own visibility and
scale. Force arrows are scaled for display; the caption states the scale and the units.
"""

from __future__ import annotations

import html
import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from mj_viser.panels import PanelBase

if TYPE_CHECKING:
    import viser

    from mj_viser.viewer import MujocoViewer

LIMIT = 200
"""The most contacts drawn; more are reported as truncated."""
_NORMAL_M = 0.05  # display length of a normal, in meters
_COLORS = {"point": (255, 200, 0), "normal": (60, 160, 255), "force": (230, 60, 60)}


@dataclass(frozen=True)
class Contact:
    """One contact between two bodies, in world coordinates."""

    first: str
    """The first body's name (``body <id>`` if unnamed; ``world`` for world geoms)."""
    second: str
    geoms: tuple[int, int]
    position: tuple[float, float, float]
    """The contact point, in meters."""
    normal: tuple[float, float, float]
    """Unit normal, pointing from the first geom to the second."""
    force: tuple[float, float, float] | None
    """The constraint force on the second body, in newtons, or None when the contact
    has no active constraint (excluded or inactive)."""
    distance: float
    """Signed distance, in meters; negative is penetration."""


@dataclass(frozen=True)
class ContactSet:
    """The contacts of one state."""

    time_s: float
    """The state's simulation time."""
    contacts: tuple[Contact, ...]
    """The contacts drawn, at most the limit."""
    total: int
    """How many contacts the state has."""

    @property
    def truncated(self) -> bool:
        return self.total > len(self.contacts)


def check_limit(limit: int) -> int:
    """``limit`` if it is a positive integer, the most contacts to draw.

    Raises:
        ValueError: Otherwise.
    """
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise ValueError(f"the contact limit must be a positive integer, got {limit!r}")
    return int(limit)


def check_scale(scale_m_per_n: float) -> float:
    """``scale_m_per_n`` if it is finite and positive: meters of arrow per newton.

    Raises:
        ValueError: Otherwise.
    """
    if isinstance(scale_m_per_n, bool) or not isinstance(scale_m_per_n, (int, float, np.number)):
        raise ValueError(f"the force arrow scale must be a number, got {scale_m_per_n!r}")
    if not (math.isfinite(scale_m_per_n) and scale_m_per_n > 0):
        raise ValueError(f"the force arrow scale must be finite and positive, got {scale_m_per_n!r}")
    return float(scale_m_per_n)


def contact_set(model: mujoco.MjModel, data: mujoco.MjData, *, limit: int = LIMIT) -> ContactSet:
    """The contacts ``data`` holds, as its simulation computed them.

    ``data`` must be of ``model``; its contact list and constraint forces are read as
    they are (a caller that steps or forwards the simulation supplies them).

    Raises:
        ValueError: If ``limit`` is not a positive integer.
    """
    limit = check_limit(limit)
    total = int(data.ncon)
    out = []
    wrench = np.zeros(6)
    for i in range(min(total, limit)):
        c = data.contact[i]
        frame = np.asarray(c.frame, float).reshape(3, 3)  # rows: normal, then two tangents
        force: tuple[float, float, float] | None = None
        if c.efc_address >= 0:
            mujoco.mj_contactForce(model, data, i, wrench)
            f = frame.T @ wrench[:3]  # contact frame to world
            force = (float(f[0]), float(f[1]), float(f[2]))
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        out.append(
            Contact(
                first=_body_name(model, b1),
                second=_body_name(model, b2),
                geoms=(int(c.geom1), int(c.geom2)),
                position=tuple(float(v) for v in c.pos),  # type: ignore[arg-type]
                normal=tuple(float(v) for v in frame[0]),  # type: ignore[arg-type]
                force=force,
                distance=float(c.dist),
            )
        )
    return ContactSet(time_s=float(data.time), contacts=tuple(out), total=total)


def _body_name(model: mujoco.MjModel, body: int) -> str:
    if body == 0:
        return "world"
    return model.body(body).name or f"body {body}"


def contact_vectors(contacts: ContactSet, scale_m_per_n: float) -> dict[str, np.ndarray]:
    """What is drawn for a set: ``points`` (N×3), and ``normals`` and ``forces`` as
    arrows (N×2×3, start and end), forces scaled by ``scale_m_per_n``."""
    points = np.array([c.position for c in contacts.contacts], float).reshape(-1, 3)
    normals = np.array(
        [[c.position, np.add(c.position, np.multiply(c.normal, _NORMAL_M))] for c in contacts.contacts], float
    ).reshape(-1, 2, 3)
    forces = np.array(
        [
            [c.position, np.add(c.position, np.multiply(c.force, scale_m_per_n))]
            for c in contacts.contacts
            if c.force is not None
        ],
        float,
    ).reshape(-1, 2, 3)
    return {"points": points, "normals": normals, "forces": forces}


def describe(contacts: ContactSet | None, scale_m_per_n: float) -> list[str]:
    """Caption lines: the frame's time, the units and scale, and each contact."""
    if contacts is None:
        return ["no state yet"]
    lines = [f"t = {contacts.time_s:.3f} s · force arrows: 1 N = {scale_m_per_n * 100:g} cm"]
    if contacts.total == 0:
        return [*lines, "no contacts"]
    if contacts.truncated:
        lines.append(f"truncated: {len(contacts.contacts)} of {contacts.total} contacts shown")
    for c in contacts.contacts:
        force = "no active force" if c.force is None else f"{math.hypot(*c.force):.3f} N"
        lines.append(f"{c.first} / {c.second}: {force}, distance {c.distance * 1000:.2f} mm")
    return lines


class ContactOverlay(PanelBase):
    """Contacts of the state the viewer shows, per browser.

    Add it to a viewer (``viewer.add_panel(ContactOverlay())``); each ``update`` (or a
    ``LiveView`` frame) refreshes it. Each browser chooses, for itself, whether contacts
    show and how long a newton's force arrow is. Nothing here changes the simulation.
    """

    def __init__(self, *, limit: int = LIMIT, scale_m_per_n: float = 0.01) -> None:
        self._limit = check_limit(limit)
        self._scale = check_scale(scale_m_per_n)
        self._lock = threading.Lock()
        self._clients: dict[int, _ClientContacts] = {}
        self._contacts: ContactSet | None = None

    def name(self) -> str:
        return "Contacts"

    @property
    def contacts(self) -> ContactSet | None:
        """The contacts of the latest state shown, or None before the first."""
        return self._contacts

    def setup(self, gui: viser.GuiApi, viewer: MujocoViewer) -> None:
        server = viewer.server
        server.on_client_connect(self._connect)
        server.on_client_disconnect(lambda client: self._drop(client.client_id))
        for client in server.get_clients().values():
            self._connect(client)

    def on_sync(self, viewer: MujocoViewer) -> None:
        self._contacts = contact_set(viewer.model, viewer.data, limit=self._limit)
        with self._lock:
            clients = list(self._clients.values())
        for client in clients:
            client.render(self._contacts)

    def client(self, client_id: int) -> _ClientContacts | None:
        with self._lock:
            return self._clients.get(client_id)

    def _connect(self, client: Any) -> None:
        panel = _ClientContacts(client, self._scale)
        with self._lock:
            self._clients[client.client_id] = panel
        panel.render(self._contacts)

    def _drop(self, client_id: int) -> None:
        with self._lock:
            self._clients.pop(client_id, None)


class _ClientContacts:
    """One browser's contact display: its own visibility, scale, and nodes."""

    def __init__(self, client: Any, scale: float) -> None:
        self._client = client
        self._lock = threading.RLock()
        self._rendering = False
        self._contacts: ContactSet | None = None
        self._nodes: dict[str, Any] = {}
        gui = client.gui
        with gui.add_folder("Contacts", order=30):
            self.show = gui.add_checkbox("Show contacts", initial_value=True)
            self.scale_cm = gui.add_number("Force arrow (cm per N)", initial_value=scale * 100, min=0.01, step=0.1)
            self.caption = gui.add_html("")
        for handle in (self.show, self.scale_cm):
            handle.on_update(lambda _event: self._changed())

    @property
    def scale_m_per_n(self) -> float:
        return float(self.scale_cm.value) / 100

    def _changed(self) -> None:
        if not self._rendering:
            self.render(self._contacts)

    def render(self, contacts: ContactSet | None) -> None:
        with self._lock:
            self._rendering = True
            try:
                self._contacts = contacts
                self._render(contacts)
            finally:
                self._rendering = False

    def _render(self, contacts: ContactSet | None) -> None:
        scale = self.scale_m_per_n
        lines = "<br>".join(html.escape(line) for line in describe(contacts, scale))
        self.caption.content = f'<div style="font-size:11px">{lines}</div>'  # the channel panels' size
        for node in self._nodes.values():
            node.remove()
        self._nodes = {}
        if contacts is None or not contacts.contacts or not self.show.value:
            return
        scene = self._client.scene
        drawn = contact_vectors(contacts, scale)
        self._nodes["points"] = scene.add_point_cloud(
            "/contacts/points", drawn["points"], _COLORS["point"], point_size=0.01, point_shape="circle"
        )
        self._nodes["normals"] = scene.add_arrows(
            "/contacts/normals",
            drawn["normals"],
            _COLORS["normal"],
            shaft_radius=0.002,
            head_radius=0.005,
            head_length=0.01,
        )
        if len(drawn["forces"]):
            self._nodes["forces"] = scene.add_arrows(
                "/contacts/forces",
                drawn["forces"],
                _COLORS["force"],
                shaft_radius=0.003,
                head_radius=0.008,
                head_length=0.015,
            )
