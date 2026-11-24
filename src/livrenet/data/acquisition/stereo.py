from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SensorProfile, StereoConfig
from .geometry import Camera, MeshData
from .rasterizer import RasterizedSurface, rasterize, surface_points


@dataclass
class StereoVisibility:
    left: RasterizedSurface
    right: RasterizedSurface | None
    mutual_mask: np.ndarray
    right_uv: np.ndarray
    right_camera: Camera | None


def make_right_camera(
    left: Camera, profile: SensorProfile, mesh_unit_to_mm: float
) -> Camera:
    baseline_world = float(profile.camera["baseline_mm"]) / float(mesh_unit_to_mm)
    return left.shifted(left.right * baseline_world)


def compute_stereo_visibility(
    mesh: MeshData,
    left_camera: Camera,
    profile: SensorProfile,
    mesh_unit_to_mm: float,
    config: StereoConfig,
) -> StereoVisibility:
    left = rasterize(mesh, left_camera)
    if not config.enabled:
        return StereoVisibility(
            left=left,
            right=None,
            mutual_mask=left.valid.copy(),
            right_uv=np.full((*left.depth.shape, 2), -1, dtype=np.int32),
            right_camera=None,
        )
    right_camera = make_right_camera(left_camera, profile, mesh_unit_to_mm)
    right = rasterize(mesh, right_camera)
    mutual = np.zeros_like(left.valid)
    right_uv_map = np.full((*left.depth.shape, 2), -1, dtype=np.int32)
    if not left.valid.any() or not right.valid.any():
        return StereoVisibility(left, right, mutual, right_uv_map, right_camera)

    world, _, _ = surface_points(mesh, left, left.valid)
    uv, z = right_camera.project(world)
    x = np.rint(uv[:, 0] - 0.5).astype(int)
    y = np.rint(uv[:, 1] - 0.5).astype(int)
    in_bounds = (
        (x >= 0)
        & (x < right_camera.width)
        & (y >= 0)
        & (y < right_camera.height)
        & (z > right_camera.near)
    )
    valid_ids = np.flatnonzero(left.valid)
    accepted = np.zeros(len(valid_ids), dtype=bool)
    if in_bounds.any():
        ids = np.flatnonzero(in_bounds)
        right_depth = right.depth[y[ids], x[ids]].astype(float)
        tol_world = np.maximum(
            float(config.occlusion_tolerance_mm) / float(mesh_unit_to_mm),
            float(config.occlusion_relative_tolerance) * np.maximum(z[ids], 1e-6),
        )
        accepted[ids] = (right_depth > 0) & (np.abs(right_depth - z[ids]) <= tol_world)
    flat = mutual.reshape(-1)
    flat[valid_ids] = accepted if config.require_mutual_visibility else in_bounds
    right_flat = right_uv_map.reshape(-1, 2)
    right_flat[valid_ids[in_bounds]] = np.column_stack([x[in_bounds], y[in_bounds]])
    return StereoVisibility(left, right, mutual, right_uv_map, right_camera)
