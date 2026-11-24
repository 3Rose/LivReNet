"""Sensor blueprint and canonical point attributes."""

from dataclasses import dataclass

import numpy as np

from .geometry import Camera, MeshData
from .rasterizer import backproject
from .sensor import SensorParameters, ToolMaskSpec


@dataclass(frozen=True)
class ObservationBlueprint:
    camera: Camera
    sensor_seed: int
    tool: ToolMaskSpec
    sensor_parameters: SensorParameters
    blueprint_id: str
    template_geometric_coverage: float
    training_randomized: bool
    camera_sampling_attempts: int
    camera_sampling_fallback: bool
    camera_sampling_policy: str


def _anchor_visible(template: MeshData, camera: Camera, face_map: np.ndarray) -> bool:
    if camera.anchor_face_id < 0:
        return True
    tri = template.faces[camera.anchor_face_id]
    bary = np.asarray(camera.anchor_barycentric, dtype=float)
    anchor = np.sum(template.points[tri] * bary[:, None], axis=0)
    uv, z = camera.project(anchor[None, :])
    x = int(round(float(uv[0, 0]) - 0.5))
    y = int(round(float(uv[0, 1]) - 0.5))
    if z[0] <= camera.near or x < 0 or y < 0 or x >= camera.width or y >= camera.height:
        return False
    return int(face_map[y, x]) == int(camera.anchor_face_id)


def _canonical_input_mask(
    valid: np.ndarray, budget: int, strategy: str, rng: np.random.Generator
) -> np.ndarray:
    ids = np.flatnonzero(valid)
    if ids.size <= budget:
        return valid.copy()
    if strategy == "uniform_random":
        chosen = rng.choice(ids, size=budget, replace=False)
    elif strategy == "image_density_balanced":
        h, w = valid.shape
        y, x = np.unravel_index(ids, valid.shape)
        bins_y, bins_x = 16, 20
        by = np.minimum(bins_y - 1, y * bins_y // h)
        bx = np.minimum(bins_x - 1, x * bins_x // w)
        key = by * bins_x + bx
        counts = np.bincount(key, minlength=bins_y * bins_x)
        weights = 1.0 / np.maximum(counts[key], 1)
        weights /= weights.sum()
        chosen = rng.choice(ids, size=budget, replace=False, p=weights)
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    out = np.zeros(valid.size, dtype=bool)
    out[chosen] = True
    return out.reshape(valid.shape)


def _gather_point_attributes(
    mesh: MeshData,
    template: MeshData,
    raster,
    valid: np.ndarray,
    noisy_depth: np.ndarray,
    camera: Camera,
    gaussian_sigma_world: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    noisy_points, pixel_uv = backproject(noisy_depth, valid, camera)
    y = pixel_uv[:, 1] if len(pixel_uv) else np.zeros(0, dtype=int)
    x = pixel_uv[:, 0] if len(pixel_uv) else np.zeros(0, dtype=int)
    face_ids = raster.face_id[y, x].astype(np.int64)
    bary = raster.barycentric[y, x].astype(np.float32)
    if gaussian_sigma_world > 0 and len(noisy_points):
        noisy_points = noisy_points + rng.normal(
            0.0, gaussian_sigma_world, size=noisy_points.shape
        )
    if len(face_ids):
        surface_gt = np.sum(mesh.points[mesh.faces[face_ids]] * bary[..., None], axis=1)
        template_gt = np.sum(
            template.points[template.faces[face_ids]] * bary[..., None], axis=1
        )
        normal = mesh.normals[face_ids]
        nearest_vertex = mesh.faces[face_ids, np.argmax(bary, axis=1)]
    else:
        surface_gt = np.zeros((0, 3), dtype=float)
        template_gt = np.zeros((0, 3), dtype=float)
        normal = np.zeros((0, 3), dtype=float)
        nearest_vertex = np.zeros(0, dtype=np.int64)
    return {
        "points": noisy_points.astype(np.float32),
        "pixel_uv": pixel_uv.astype(np.int32),
        "face_ids": face_ids,
        "barycentric": bary,
        "nearest_vertex_ids": nearest_vertex.astype(np.int64),
        "surface_points_gt": surface_gt.astype(np.float32),
        "template_correspondence_points": template_gt.astype(np.float32),
        "correspondence_displacement_gt": (surface_gt - template_gt).astype(np.float32),
        "noise_vector": (noisy_points - surface_gt).astype(np.float32),
        "surface_normals": normal.astype(np.float32),
    }
