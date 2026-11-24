from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def bbox_diag(points: np.ndarray) -> float:
    p = np.asarray(points, float)
    return float(np.linalg.norm(p.max(0) - p.min(0)))


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def read_surface(path: str):
    import pyvista as pv

    m = pv.read(path)
    if not isinstance(m, pv.PolyData):
        m = m.extract_surface()
    m = m.triangulate().clean()
    m = m.compute_normals(
        point_normals=True,
        cell_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        inplace=False,
    )
    return m


def triangle_array(mesh) -> np.ndarray:
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:4].astype(np.int64)


def area_uniform_vertex(mesh, rng: np.random.Generator, candidates: np.ndarray) -> int:
    faces = triangle_array(mesh)
    pts = np.asarray(mesh.points)
    keep = np.any(np.isin(faces, candidates), axis=1)
    faces = faces[keep]
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if not np.any(areas > 0):
        return int(rng.choice(candidates))
    f = faces[int(rng.choice(len(faces), p=areas / areas.sum()))]
    valid = f[np.isin(f, candidates)]
    return int(rng.choice(valid if len(valid) else candidates))


def edge_graph(mesh) -> coo_matrix:
    from DatasetFactory.ScientificPipelineShared.graph import surface_graph

    return surface_graph(mesh.points, triangle_array(mesh))


def geodesic_distances(graph, center: int, limit: float) -> np.ndarray:
    return np.asarray(
        dijkstra(graph, directed=False, indices=int(center), limit=float(limit))
    )


def map_surface_to_volume(
    surface_points: np.ndarray, volume_points: np.ndarray, tol: float
) -> np.ndarray:
    dist, idx = cKDTree(volume_points).query(surface_points, k=1)
    if float(np.max(dist)) > float(tol):
        raise ValueError(
            f"Surface-volume mapping max distance {float(np.max(dist)):.6g} > {tol:.6g}"
        )
    return idx.astype(np.int32)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_sample(
    path: Path,
    template,
    deformed: np.ndarray,
    disp: np.ndarray,
    point_data: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> None:
    """Write one self-contained, pickle-free compressed NPZ sample."""

    reference = np.asarray(template.points, dtype=np.float32)
    displacement = np.asarray(disp, dtype=np.float32)
    payload = {
        "schema_version": np.asarray(1, dtype=np.int16),
        "reference_points": reference,
        "points": np.asarray(deformed, dtype=np.float32),
        "triangles": triangle_array(template).astype(np.int32),
        "displacement": displacement,
        "displacement_magnitude": np.linalg.norm(displacement, axis=1).astype(
            np.float32
        ),
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }
    for key, value in point_data.items():
        payload[key] = np.asarray(value)
    for key, value in (extra_arrays or {}).items():
        payload[key] = np.asarray(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def read_sample_npz(path: str | Path) -> dict[str, Any]:
    """Load a generated sample without enabling pickle deserialization."""

    with np.load(Path(path), allow_pickle=False) as archive:
        output = {
            key: np.asarray(archive[key]).copy()
            for key in archive.files
            if key != "metadata_json"
        }
        output["metadata"] = json.loads(str(archive["metadata_json"]))
    return output
