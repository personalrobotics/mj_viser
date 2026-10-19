# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Contacts as the state supplies them, drawn and recorded, without ssrobot.

A 1 kg box dropped onto a floor: in the air it has no contacts; at rest it has four,
whose forces carry its weight. The checks: contact_set reports each contact's bodies,
point, normal, distance, and force exactly as MuJoCo's own contact list and
mj_contactForce give them, with the forces summing to the box's weight; a limit
truncates explicitly; a StateRecorder with contacts draws, at each state's time, a
point, a normal, and a force arrow per contact (none for the state in the air), each
arrow ending at its point plus the force at the stated scale; and with a
ContactOverlay, one browser hiding contacts or rescaling its arrows leaves another's
unchanged, while a browser connected before any state reads "no state yet". Writes
contacts.viser and contacts_report.json.
"""

from __future__ import annotations

import json
import os
import time
from importlib.metadata import version
from typing import Any

import msgspec
import mujoco
import numpy as np
from websockets.sync.client import connect

from e2e.channels import CAMERA, _frames
from e2e.fixtures import expect_raises
from e2e.oracles import replay_arrays
from mj_viser import ContactOverlay, MujocoViewer, StateRecorder, contact_set
from mj_viser.recording import CONTACTS

BOX = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.5"><freejoint/><geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
SCALE = 0.01  # m per N


def _states(model: mujoco.MjModel) -> dict[str, mujoco.MjData]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    air = mujoco.MjData(model)
    mujoco.mj_copyData(air, model, data)
    for _ in range(1000):  # 2 s: dropped, landed, at rest
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    return {"air": air, "rest": data}


def _independent(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Each contact's point, world force, and distance, straight from MuJoCo."""
    out, wrench = [], np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        mujoco.mj_contactForce(model, data, i, wrench)
        out.append((np.array(c.pos), np.asarray(c.frame).reshape(3, 3).T @ wrench[:3], float(c.dist)))
    return out


def _choose(ws: Any, frames: list[dict[str, Any]], label: str, value: Any) -> None:
    widget = [m for m in frames if m["type"].startswith("Gui") and m.get("props", {}).get("label") == label][-1]
    ws.send(msgspec.msgpack.encode({"type": "GuiUpdateMessage", "uuid": widget["uuid"], "updates": {"value": value}}))


def check_contacts(out_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    model = mujoco.MjModel.from_xml_string(BOX)
    states = _states(model)

    air, rest = contact_set(model, states["air"]), contact_set(model, states["rest"])
    if air.total != 0 or air.contacts:
        raise SystemExit(f"the box in the air has contacts: {air}")
    truth = _independent(model, states["rest"])
    if rest.total != 4 or len(truth) != 4:
        raise SystemExit(f"the resting box has {rest.total} contacts")
    for c, (point, force, distance) in zip(rest.contacts, truth, strict=True):
        if (c.first, c.second) != ("world", "box") or c.force is None:
            raise SystemExit(f"contact {c}")
        if not (np.allclose(c.position, point) and np.allclose(c.force, force) and c.distance == distance):
            raise SystemExit(f"contact {c} differs from MuJoCo's {point}, {force}, {distance}")
    weight = float(sum(c.force[2] for c in rest.contacts if c.force is not None))
    if abs(weight - 9.81) > 0.01:
        raise SystemExit(f"the contacts carry {weight} N, not the box's 9.81 N")
    out["resting_contacts"] = {"count": rest.total, "vertical_force_n": round(weight, 6)}
    limited = contact_set(model, states["rest"], limit=2)
    if not limited.truncated or len(limited.contacts) != 2 or limited.total != 4:
        raise SystemExit(f"a limit of 2 gave {limited}")
    out["truncated"] = f"{len(limited.contacts)} of {limited.total}"

    # Bad limits and scales are refused where they come in; the recorder refuses before
    # it opens anything.
    def opened(*args: Any, **kwargs: Any) -> None:
        raise SystemExit("StateRecorder opened a viewer for a scale it refuses")

    refused = {}
    for limit in (0, -1, 1.5, True):
        refused[f"contact_set limit {limit!r}"] = expect_raises(
            ValueError, lambda limit=limit: contact_set(model, states["rest"], limit=limit)
        )
        refused[f"ContactOverlay limit {limit!r}"] = expect_raises(
            ValueError, lambda limit=limit: ContactOverlay(limit=limit)
        )
    original = MujocoViewer.open
    MujocoViewer.open = opened  # type: ignore[method-assign]
    try:
        for scale in (-0.01, 0.0, float("nan"), float("inf")):
            refused[f"StateRecorder scale {scale!r}"] = expect_raises(
                ValueError, lambda scale=scale: StateRecorder(model, contacts=True, contact_scale_m_per_n=scale)
            )
            refused[f"ContactOverlay scale {scale!r}"] = expect_raises(
                ValueError, lambda scale=scale: ContactOverlay(scale_m_per_n=scale)
            )
    finally:
        MujocoViewer.open = original  # type: ignore[method-assign]
    out["refused"] = refused

    # Recorded: no contact nodes in the air; at rest a point, normal, and force each.
    with StateRecorder(model, contacts=True, contact_scale_m_per_n=SCALE) as recorder:
        recorder.advance(0)
        recorder.show(states["air"])
        recorder.advance(1_000_000_000)
        recorder.show(states["rest"])
        recording = recorder.finish()
        log = recorder.contact_log
    messages = replay_arrays(recording)
    forces = [
        (round(t, 6), m) for t, m in messages if m.get("name") == f"{CONTACTS}/forces" and m["type"] == "ArrowMessage"
    ]
    if [t for t, _ in forces] != [1.0]:
        raise SystemExit(f"force arrows recorded at {[t for t, _ in forces]}, expected only at 1.0 s")
    arrows = np.asarray(forces[0][1]["props"]["points"], float).reshape(-1, 2, 3)
    expected = np.array([[p, p + f * SCALE] for p, f, _ in truth])
    if not np.allclose(arrows, expected, atol=1e-6):  # float32 on the wire
        raise SystemExit(f"recorded force arrows {arrows}, expected {expected}")
    if [entry["drawn"] for entry in log] != [
        {"points": 0, "normals": 0, "forces": 0},
        {"points": 4, "normals": 4, "forces": 4},
    ]:
        raise SystemExit(f"contact log {[entry['drawn'] for entry in log]}")
    out["recorded"] = {"air": log[0]["drawn"], "rest": log[1]["drawn"]}
    with open(os.path.join(out_dir, "contacts.viser"), "wb") as f:
        f.write(recording)
    with open(os.path.join(out_dir, "contacts_report.json"), "w") as f:
        json.dump(log, f, indent=2, sort_keys=True)
        f.write("\n")

    # Live, per browser: hiding and rescaling are each browser's own.
    overlay = ContactOverlay(scale_m_per_n=SCALE)
    with MujocoViewer(model, port=0) as viewer:
        viewer.add_panel(overlay)
        url = viewer.url.replace("http://", "ws://")
        clients = [connect(url, subprotocols=[f"viser-v{version('viser')}"], max_size=None) for _ in range(2)]
        try:
            for ws in clients:
                ws.send(msgspec.msgpack.encode(CAMERA))
            seen = [_frames(ws, 1.0) for ws in clients]
            captions = [m["props"]["content"] for m in seen[0] if m["type"] == "GuiHtmlMessage"]
            if not any("no state yet" in c for c in captions):
                raise SystemExit("a browser connected before any state does not read 'no state yet'")
            viewer.update(states["rest"])
            seen = [s + _frames(ws, 1.0) for s, ws in zip(seen, clients, strict=True)]
            _choose(clients[0], seen[0], "Show contacts", False)
            _choose(clients[1], seen[1], "Force arrow (cm per N)", 2.0)
            time.sleep(0.3)
            after = [_frames(ws, 1.0) for ws in clients]
            panels = [overlay.client(i) for i in sorted(viewer.server.get_clients())]
            shown = [p.show.value for p in panels if p is not None]
            scales = [p.scale_m_per_n for p in panels if p is not None]
            if sorted(shown) != [False, True] or sorted(scales) != [0.01, 0.02]:
                raise SystemExit(f"per-browser choices leaked: show {shown}, scales {scales}")
            removed = [
                m for m in after[0] if m["type"] == "RemoveSceneNodeMessage" and "/contacts/" in m.get("name", "")
            ]
            readded = [m for m in after[0] if m["type"] == "ArrowMessage" and m.get("name") == "/contacts/forces"]
            if not removed or readded:
                raise SystemExit("the browser that hid contacts still draws them")
            out["per_browser"] = {"hidden_by_one": True, "scales_m_per_n": sorted(scales)}
        finally:
            for ws in clients:
                ws.close()
    return out
