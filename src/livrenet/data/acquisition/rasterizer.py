from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Camera, MeshData


@dataclass
class RasterizedSurface:
    depth: np.ndarray
    face_id: np.ndarray
    barycentric: np.ndarray
    incidence: np.ndarray
    surface_weight: np.ndarray
    projected_area_px: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return self.face_id >= 0


def rasterize(mesh: MeshData, camera: Camera) -> RasterizedSurface:
    """Perspective-correct z-buffer with approximate per-pixel surface area.

    ``surface_weight`` allocates each triangle's 3-D area over its complete
    projected image area. Summing it on any mask therefore estimates how much
    physical surface survives FOV clipping, self-occlusion and later dropout.
    """
    h, w = camera.height, camera.width
    depth = np.full((h, w), np.inf, dtype=np.float64)
    face_map = np.full((h, w), -1, dtype=np.int64)
    bary_map = np.zeros((h, w, 3), dtype=np.float32)
    incidence = np.zeros((h, w), dtype=np.float32)
    surface_weight = np.zeros((h, w), dtype=np.float64)
    camera_points = camera.world_to_camera(mesh.points)
    normals = mesh.normals
    centers = mesh.points[mesh.faces].mean(axis=1)
    view_dirs = camera.position[None, :] - centers
    view_dirs /= np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-12)
    face_incidence = np.clip(np.sum(normals * view_dirs, axis=1), 0.0, 1.0)
    projected_area_px = np.zeros(len(mesh.faces), dtype=float)
    areas = mesh.areas

    for fi, vertex_ids in enumerate(mesh.faces):
        tri = camera_points[vertex_ids]
        if np.any(tri[:, 2] <= camera.near) or face_incidence[fi] <= 0.0:
            continue
        uv = np.column_stack(
            [
                camera.fx * tri[:, 0] / tri[:, 2] + camera.cx,
                camera.fy * tri[:, 1] / tri[:, 2] + camera.cy,
            ]
        )
        v1 = uv[1] - uv[0]
        v2 = uv[2] - uv[0]
        full_area = 0.5 * abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
        if full_area < 1e-10:
            continue
        projected_area_px[fi] = full_area
        xmin = max(0, int(np.floor(uv[:, 0].min())))
        xmax = min(w - 1, int(np.ceil(uv[:, 0].max())))
        ymin = max(0, int(np.floor(uv[:, 1].min())))
        ymax = min(h - 1, int(np.ceil(uv[:, 1].max())))
        if xmin > xmax or ymin > ymax:
            continue
        x0, y0 = uv[0]
        x1, y1 = uv[1]
        x2, y2 = uv[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denom)) < 1e-12:
            continue
        ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        px = xs + 0.5
        py = ys + 0.5
        b0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        b1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-7) & (b1 >= -1e-7) & (b2 >= -1e-7)
        if not inside.any():
            continue
        inv_z = b0 / tri[0, 2] + b1 / tri[1, 2] + b2 / tri[2, 2]
        z = 1.0 / np.maximum(inv_z, 1e-12)
        sub_depth = depth[ymin : ymax + 1, xmin : xmax + 1]
        update = inside & (z < sub_depth)
        if not update.any():
            continue
        sub_depth[update] = z[update]
        face_map[ymin : ymax + 1, xmin : xmax + 1][update] = fi
        corrected = np.stack([b0 / tri[0, 2], b1 / tri[1, 2], b2 / tri[2, 2]], axis=-1)
        corrected /= np.maximum(corrected.sum(axis=-1, keepdims=True), 1e-12)
        bary_map[ymin : ymax + 1, xmin : xmax + 1][update] = corrected[update]
        incidence[ymin : ymax + 1, xmin : xmax + 1][update] = face_incidence[fi]
        surface_weight[ymin : ymax + 1, xmin : xmax + 1][update] = areas[fi] / full_area

    depth[~np.isfinite(depth)] = 0.0
    return RasterizedSurface(
        depth=depth.astype(np.float32),
        face_id=face_map,
        barycentric=bary_map,
        incidence=incidence,
        surface_weight=surface_weight.astype(np.float32),
        projected_area_px=projected_area_px.astype(np.float32),
    )


def backproject(
    depth: np.ndarray, valid: np.ndarray, camera: Camera
) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(valid)
    z = depth[y, x].astype(float)
    camera_points = np.column_stack(
        [
            (x + 0.5 - camera.cx) * z / camera.fx,
            (y + 0.5 - camera.cy) * z / camera.fy,
            z,
        ]
    )
    return camera.camera_to_world(camera_points), np.column_stack([x, y]).astype(
        np.int32
    )


def surface_points(
    mesh: MeshData, raster: RasterizedSurface, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask)
    face_ids = raster.face_id[y, x]
    bary = raster.barycentric[y, x].astype(float)
    points = np.sum(mesh.points[mesh.faces[face_ids]] * bary[..., None], axis=1)
    return points, face_ids.astype(np.int64), bary.astype(np.float32)
