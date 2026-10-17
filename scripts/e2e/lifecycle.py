# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Lifecycle failures and network binding: update and startup errors, IPv6, repeated open/close."""

from __future__ import annotations

import socket
import urllib.parse

import mujoco

import mj_viser
from e2e.fixtures import MJCF, expect_raises, free_port
from e2e.oracles import frozen, live_local_poses


def check_lifecycle(model: mujoco.MjModel, sources: dict[str, mujoco.MjData], report: dict[str, object]) -> None:
    """Failures return to the caller, close() releases the port, and URLs are well formed."""
    closed = mj_viser.MujocoViewer(model, port=0)
    closed.open()
    closed.close()
    # --- Update failures return to the caller ---------------------------------------------
    # Adversarial models: every array the old size check compared is the same length.
    # "swapped" is the same layout with hinge and slide exchanged (0.5 m would render as 0.5 rad);
    # "camera" adds a camera, changing MjData's internal layout so mj_copyData would raise FatalError.
    adversarial = {
        "swapped": MJCF.replace('type="hinge"', 'type="TMP"')
        .replace('type="slide"', 'type="hinge"')
        .replace('type="TMP"', 'type="slide"'),
        "camera": MJCF.replace("<worldbody>", '<worldbody><camera name="cam" pos="0 0 2"/>'),
    }
    report["update_while_closed"] = expect_raises(RuntimeError, lambda: closed.update(sources["A"]))
    with mj_viser.MujocoViewer(model, port=0) as v:
        v.update(sources["A"])
        good = frozen(live_local_poses(v))
        for name, xml in adversarial.items():
            m = mujoco.MjModel.from_xml_string(xml)
            if (m.nq, m.nv, m.nbody, m.ngeom, m.nsensordata) != (model.nq, model.nv, model.nbody, model.ngeom, 0):
                raise SystemExit(f"adversarial model {name} is not size-identical")
            d = mujoco.MjData(m)
            d.qpos[:] = [0.5, 0.5]
            report[f"update_{name}_model"] = expect_raises(ValueError, lambda: v.update(d))
            if frozen(live_local_poses(v)) != good:
                raise SystemExit(f"rejected {name} update changed the rendered scene")
        report["open_twice"] = expect_raises(RuntimeError, v.open)

    # --- IPv6 literal hosts produce a bracketed URL ---------------------------------------
    try:
        with socket.socket(socket.AF_INET6) as s6:
            s6.bind(("::1", 0))
    except OSError:
        report["ipv6_url"] = "skipped: ::1 not bindable on this host"
    else:
        with mj_viser.MujocoViewer(model, host="::1", port=0) as v:
            parsed = urllib.parse.urlsplit(v.url)
            if parsed.hostname != "::1" or not parsed.port:
                raise SystemExit(f"invalid IPv6 URL {v.url}")
            report["ipv6_url"] = v.url.replace(str(parsed.port), "<port>")

    # --- Repeated open/close on one fixed port releases it each time ----------------------
    fixed = free_port()
    v = mj_viser.MujocoViewer(model, port=fixed)
    for _ in range(3):
        if v.open() != f"http://127.0.0.1:{fixed}":
            raise SystemExit("reopen did not bind the requested port")
        v.update(sources["B"])
        v.close()
    v.close()  # idempotent
    with socket.socket() as s:
        s.bind(("127.0.0.1", fixed))  # port is free again
    report["open_close_cycles_on_fixed_port"] = 3

    # --- Startup failures return to the caller instead of hanging -------------------------
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        blocked = mj_viser.MujocoViewer(model, port=taken)
        report["startup_port_in_use"] = expect_raises(OSError, blocked.open).replace(str(taken), "<port>")
        if blocked.is_open:
            raise SystemExit("viewer reports open after failed startup")
    # 192.0.2.1 is TEST-NET-1 (RFC 5737): never assigned to a local interface.
    report["startup_unbindable_host"] = expect_raises(OSError, mj_viser.MujocoViewer(model, host="192.0.2.1").open)
