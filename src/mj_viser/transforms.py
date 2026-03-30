"""Coordinate and rotation conversion between MuJoCo and Viser."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

import viser


def configure_scene(server: viser.ViserServer) -> None:
    """Configure viser scene to match MuJoCo's Z-up coordinate convention."""
    server.scene.set_up_direction("+z")


def xmat_to_wxyz(xmat: npt.NDArray[np.floating]) -> tuple[float, float, float, float]:
    """Convert MuJoCo's flat 9-element row-major rotation matrix to (w, x, y, z) quaternion.

    Uses Shepperd's method for numerical stability.

    Args:
        xmat: Flat array of 9 floats (row-major 3x3 rotation matrix),
              as stored in ``MjData.geom_xmat[geom_id]``.

    Returns:
        Quaternion as (w, x, y, z).
    """
    m = xmat.reshape(3, 3)

    # Shepperd's method: pick the largest diagonal element to avoid division by ~0.
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (w, x, y, z)


def mj_pos_to_viser(pos: npt.NDArray[np.floating]) -> tuple[float, float, float]:
    """Convert MuJoCo position to viser position.

    With ``set_up_direction("+z")``, this is the identity transform.
    """
    return (float(pos[0]), float(pos[1]), float(pos[2]))
