"""Tests for procedural mesh generation and MuJoCo mesh extraction."""

from __future__ import annotations

import numpy as np

from mj_viser.mesh_utils import make_capsule_mesh, make_ellipsoid_mesh


class TestMakeCapsuleMesh:
    def test_returns_correct_types(self) -> None:
        verts, faces = make_capsule_mesh(0.5, 1.0)
        assert verts.dtype == np.float32
        assert faces.dtype == np.int32
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3

    def test_z_extent_matches_dimensions(self) -> None:
        radius, half_height = 0.3, 0.8
        verts, _ = make_capsule_mesh(radius, half_height)
        z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
        expected = half_height + radius
        assert abs(z_max - expected) < 1e-5
        assert abs(z_min + expected) < 1e-5

    def test_xy_extent_matches_radius(self) -> None:
        radius = 0.5
        verts, _ = make_capsule_mesh(radius, 1.0)
        xy_dist = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
        assert xy_dist.max() <= radius + 1e-5

    def test_face_indices_valid(self) -> None:
        verts, faces = make_capsule_mesh(0.5, 1.0)
        assert faces.min() >= 0
        assert faces.max() < len(verts)

    def test_no_degenerate_faces(self) -> None:
        _, faces = make_capsule_mesh(0.5, 1.0)
        for f in faces:
            assert len(set(f)) == 3, f"Degenerate face: {f}"

    def test_watertight(self) -> None:
        """Every edge should appear in exactly 2 triangles for a watertight mesh."""
        _, faces = make_capsule_mesh(0.5, 1.0, rings=4, segments=8)
        edge_count: dict[tuple[int, int], int] = {}
        for f in faces:
            for i in range(3):
                e = (min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
                edge_count[e] = edge_count.get(e, 0) + 1
        non_manifold = {e: c for e, c in edge_count.items() if c != 2}
        assert len(non_manifold) == 0, f"Non-manifold edges: {non_manifold}"


class TestMakeEllipsoidMesh:
    def test_returns_correct_types(self) -> None:
        verts, faces = make_ellipsoid_mesh(np.array([1.0, 2.0, 3.0]))
        assert verts.dtype == np.float32
        assert faces.dtype == np.int32

    def test_extent_matches_semi_axes(self) -> None:
        semi = np.array([1.0, 2.0, 3.0])
        verts, _ = make_ellipsoid_mesh(semi, subdivisions=3)
        for axis in range(3):
            assert abs(verts[:, axis].max() - semi[axis]) < 0.1
            assert abs(verts[:, axis].min() + semi[axis]) < 0.1

    def test_sphere_when_equal_axes(self) -> None:
        r = 1.5
        verts, _ = make_ellipsoid_mesh(np.array([r, r, r]), subdivisions=3)
        dists = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(dists, r, atol=0.05)

    def test_face_indices_valid(self) -> None:
        verts, faces = make_ellipsoid_mesh(np.array([1.0, 1.0, 1.0]))
        assert faces.min() >= 0
        assert faces.max() < len(verts)


class TestExtractMujocoMesh:
    """Tests that require MuJoCo — run only if a model with meshes is available."""

    def test_extract_from_simple_model(self) -> None:
        """Create a simple model with a box mesh to verify extraction works."""
        import mujoco

        xml = """
        <mujoco>
          <asset>
            <mesh name="box_mesh"
                  vertex="0 0 0  1 0 0  1 1 0  0 1 0  0 0 1  1 0 1  1 1 1  0 1 1"/>
          </asset>
          <worldbody>
            <geom type="mesh" mesh="box_mesh"/>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        from mj_viser.mesh_utils import extract_mujoco_mesh

        verts, faces = extract_mujoco_mesh(model, mesh_id=0)
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert verts.dtype == np.float32
        assert faces.dtype == np.int32
        assert len(verts) == 8  # 8 box vertices
