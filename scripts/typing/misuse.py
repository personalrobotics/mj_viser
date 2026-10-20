# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Misuses of mj_viser's public API that a strict type checker must reject.

scripts/typing/check.sh requires exactly these errors, so an installed mj_viser whose
types the checker could not see (no py.typed, or everything Any) fails the check.
"""

from __future__ import annotations

import mujoco

from mj_viser import LiveView, MujocoViewer, contact_set


def misuse(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    url: int = MujocoViewer(model).open()  # expect: assignment
    LiveView(object())  # expect: arg-type
    contact_set(model, data, limit="3")  # expect: arg-type
    del url
