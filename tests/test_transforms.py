"""Tests for coordinate and rotation conversion utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mj_viser.transforms import mj_pos_to_viser, xmat_to_wxyz


def _wxyz_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert (w,x,y,z) quaternion back to 3x3 rotation matrix for round-trip testing."""
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_equiv(q1: tuple[float, ...], q2: tuple[float, ...], atol: float = 1e-7) -> bool:
    """Check if two quaternions represent the same rotation (q == -q)."""
    a = np.array(q1)
    b = np.array(q2)
    return bool(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol))


class TestXmatToWxyz:
    def test_identity(self) -> None:
        identity = np.eye(3).flatten()
        w, x, y, z = xmat_to_wxyz(identity)
        assert _quat_equiv((w, x, y, z), (1.0, 0.0, 0.0, 0.0))

    def test_90_deg_around_z(self) -> None:
        # Rotation of 90 degrees around Z axis
        mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float).flatten()
        w, x, y, z = xmat_to_wxyz(mat)
        expected_w = math.cos(math.pi / 4)  # cos(45°)
        expected_z = math.sin(math.pi / 4)  # sin(45°)
        assert _quat_equiv((w, x, y, z), (expected_w, 0.0, 0.0, expected_z))

    def test_90_deg_around_x(self) -> None:
        mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float).flatten()
        w, x, y, z = xmat_to_wxyz(mat)
        expected_w = math.cos(math.pi / 4)
        expected_x = math.sin(math.pi / 4)
        assert _quat_equiv((w, x, y, z), (expected_w, expected_x, 0.0, 0.0))

    def test_90_deg_around_y(self) -> None:
        mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float).flatten()
        w, x, y, z = xmat_to_wxyz(mat)
        expected_w = math.cos(math.pi / 4)
        expected_y = math.sin(math.pi / 4)
        assert _quat_equiv((w, x, y, z), (expected_w, 0.0, expected_y, 0.0))

    def test_180_deg_around_z(self) -> None:
        mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float).flatten()
        w, x, y, z = xmat_to_wxyz(mat)
        assert _quat_equiv((w, x, y, z), (0.0, 0.0, 0.0, 1.0))

    def test_round_trip_random(self) -> None:
        """Generate random quaternion, convert to matrix, convert back."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            if q[0] < 0:
                q = -q  # Normalize to positive w
            mat = _wxyz_to_mat(*q)
            result = xmat_to_wxyz(mat.flatten())
            assert _quat_equiv(tuple(q), result, atol=1e-6), (
                f"Round-trip failed: input={q}, output={result}"
            )

    def test_unit_quaternion_output(self) -> None:
        """Output quaternion should have unit norm."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            mat = _wxyz_to_mat(*q)
            result = xmat_to_wxyz(mat.flatten())
            norm = math.sqrt(sum(c * c for c in result))
            assert abs(norm - 1.0) < 1e-6


class TestMjPosToViser:
    def test_passthrough(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        assert mj_pos_to_viser(pos) == (1.0, 2.0, 3.0)

    def test_zeros(self) -> None:
        pos = np.zeros(3)
        assert mj_pos_to_viser(pos) == (0.0, 0.0, 0.0)

    def test_returns_python_floats(self) -> None:
        pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = mj_pos_to_viser(pos)
        assert all(isinstance(v, float) for v in result)
