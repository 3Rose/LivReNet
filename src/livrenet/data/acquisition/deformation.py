from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import DeformationRegionConfig
from .geometry import MeshData


@dataclass
class DeformationRegions:
    magnitude: np.ndarray
    vertex_area: np.ndarray
    peak_core: np.ndarray
    high_deformation: np.ndarray
    support: np.ndarray
    energy_region: np.ndarray
    max_displacement: float
    total_energy: float
    high_threshold: float
    support_threshold: float
    peak_area_fraction: float

    def as_npz(self) -> dict[str, np.ndarray | float]:
        return {
            "displacement_magnitude": self.magnitude.astype(np.float32),
            "vertex_area": self.vertex_area.astype(np.float32),
            "peak_core_mask": self.peak_core.astype(np.uint8),
            "high_deformation_mask": self.high_deformation.astype(np.uint8),
            "deformation_support_mask": self.support.astype(np.uint8),
            "energy_region_mask": self.energy_region.astype(np.uint8),
            "max_displacement": np.float32(self.max_displacement),
            "total_deformation_energy": np.float32(self.total_energy),
            "high_deformation_threshold": np.float32(self.high_threshold),
            "support_threshold": np.float32(self.support_threshold),
            "peak_region_area_fraction": np.float32(self.peak_area_fraction),
        }


def compute_deformation_regions(
    mesh: MeshData, config: DeformationRegionConfig, mesh_unit_to_mm: float
) -> DeformationRegions:
    if mesh.displacement is None:
        magnitude = np.zeros(len(mesh.points), dtype=float)
    else:
        magnitude = np.linalg.norm(mesh.displacement, axis=1)
    vertex_area = mesh.vertex_areas
    max_disp = float(magnitude.max()) if len(magnitude) else 0.0
    absolute_world = float(config.support_absolute_mm) / float(mesh_unit_to_mm)
    high_threshold = float(config.high_relative_threshold) * max_disp
    support_threshold = max(
        float(config.support_relative_threshold) * max_disp, absolute_world
    )
    if max_disp <= 1e-12:
        peak = np.zeros(len(mesh.points), dtype=bool)
        high = np.zeros(len(mesh.points), dtype=bool)
        support = np.zeros(len(mesh.points), dtype=bool)
        energy_region = np.zeros(len(mesh.points), dtype=bool)
        total_energy = 0.0
    else:
        peak = magnitude >= float(config.peak_relative_threshold) * max_disp
        high = magnitude >= high_threshold
        support = magnitude >= support_threshold
        energy = vertex_area * magnitude**2
        total_energy = float(energy.sum())
        order = np.argsort(-energy)
        cumulative = np.cumsum(energy[order])
        count = int(
            np.searchsorted(
                cumulative, float(config.energy_fraction) * total_energy, side="left"
            )
            + 1
        )
        energy_region = np.zeros(len(mesh.points), dtype=bool)
        energy_region[order[:count]] = True
    peak_area_fraction = (
        float(vertex_area[peak].sum() / max(float(vertex_area.sum()), 1e-12))
        if max_disp > 1e-12
        else 0.0
    )
    return DeformationRegions(
        magnitude,
        vertex_area,
        peak,
        high,
        support,
        energy_region,
        max_disp,
        total_energy,
        high_threshold,
        support_threshold,
        peak_area_fraction,
    )


def splat_vertex_support(
    n_vertices: int,
    faces: np.ndarray,
    face_ids: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    support = np.zeros(n_vertices, dtype=float)
    if len(face_ids) == 0:
        return support
    vertices = faces[np.asarray(face_ids, dtype=np.int64)]
    for local in range(3):
        np.add.at(support, vertices[:, local], barycentric[:, local])
    return support


def hop_distance_to_observed(
    n_vertices: int,
    edges: np.ndarray,
    observed: np.ndarray,
    max_hops: int | None = None,
    adjacency: list[list[int]] | None = None,
) -> np.ndarray:
    if adjacency is None:
        adjacency = [[] for _ in range(n_vertices)]
        for a, b in np.asarray(edges, dtype=np.int64):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
    distance = np.full(n_vertices, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for idx in np.flatnonzero(observed):
        distance[idx] = 0
        queue.append(int(idx))
    while queue:
        node = queue.popleft()
        if max_hops is not None and distance[node] >= max_hops:
            continue
        for other in adjacency[node]:
            if distance[other] < 0:
                distance[other] = distance[node] + 1
                queue.append(other)
    return distance


def area_fraction(
    mask: np.ndarray, region: np.ndarray, vertex_area: np.ndarray
) -> float:
    denom = float(vertex_area[region].sum())
    if denom <= 0:
        return 0.0
    return float(vertex_area[mask & region].sum() / denom)
