from __future__ import annotations

import base64
import json
import math
import zlib
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from .config import CameraConfig


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / max(float(np.linalg.norm(x)), eps)


def face_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )


def orient_faces_consistently(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Repair triangle winding without assuming convexity.

    Adjacent faces are first oriented consistently through shared edges. For a
    closed mesh, the whole component is then flipped when its signed volume is
    negative. Open components keep the locally consistent orientation.
    """
    faces = np.asarray(faces, dtype=np.int64).copy()
    edge_map: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for fi, tri in enumerate(faces):
        for local in range(3):
            a = int(tri[local])
            b = int(tri[(local + 1) % 3])
            edge_map.setdefault((min(a, b), max(a, b)), []).append((fi, a, b))
    neighbors: list[list[tuple[int, bool]]] = [[] for _ in range(len(faces))]
    for uses in edge_map.values():
        if len(uses) != 2:
            continue
        (fa, aa, ab), (fb, ba, bb) = uses
        same_direction = aa == ba and ab == bb
        neighbors[fa].append((fb, same_direction))
        neighbors[fb].append((fa, same_direction))
    state = np.zeros(len(faces), dtype=np.int8)
    components: list[list[int]] = []
    for start in range(len(faces)):
        if state[start] != 0:
            continue
        state[start] = 1
        component = []
        queue = deque([start])
        while queue:
            current = queue.popleft()
            component.append(current)
            for other, same_direction in neighbors[current]:
                desired = -state[current] if same_direction else state[current]
                if state[other] == 0:
                    state[other] = desired
                    queue.append(other)
        components.append(component)
    faces[state < 0] = faces[state < 0][:, [0, 2, 1]]
    for component in components:
        tri = points[faces[np.asarray(component)]]
        signed_volume = float(
            np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])))
            / 6.0
        )
        if signed_volume < 0:
            ids = np.asarray(component, dtype=np.int64)
            faces[ids] = faces[ids][:, [0, 2, 1]]
    return faces


def face_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces]
    raw = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)


def vertex_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = face_areas(points, faces)
    result = np.zeros(len(points), dtype=float)
    np.add.at(result, faces.reshape(-1), np.repeat(areas / 3.0, 3))
    return result


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


@dataclass
class MeshData:
    points: np.ndarray
    faces: np.ndarray
    displacement: np.ndarray | None = None
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must contain triangles with shape (F, 3)")
        if self.displacement is not None:
            self.displacement = np.asarray(self.displacement, dtype=float)
            if self.displacement.shape != self.points.shape:
                raise ValueError("displacement must have the same shape as points")

    @cached_property
    def bbox_diag(self) -> float:
        return float(np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0)))

    @cached_property
    def areas(self) -> np.ndarray:
        return face_areas(self.points, self.faces)

    @cached_property
    def vertex_areas(self) -> np.ndarray:
        return vertex_areas(self.points, self.faces)

    @cached_property
    def normals(self) -> np.ndarray:
        return face_normals(self.points, self.faces)

    @cached_property
    def edges(self) -> np.ndarray:
        return mesh_edges(self.faces)

    @cached_property
    def vertex_adjacency(self) -> list[list[int]]:
        adjacency: list[list[int]] = [[] for _ in range(len(self.points))]
        for a, b in self.edges:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        return adjacency

    def with_faces(self, faces: np.ndarray) -> "MeshData":
        return MeshData(
            self.points, faces, self.displacement, self.point_data, self.metadata
        )

    def load_center_vertex(self) -> int | None:
        explicit = self.metadata.get("load_center_vertex")
        if explicit is not None:
            return int(explicit)
        for name in (
            "load_center_vertex",
            "is_load_center",
            "force_application_vertex",
        ):
            arr = self.point_data.get(name)
            if arr is not None and np.asarray(arr).size == len(self.points):
                ids = np.flatnonzero(np.asarray(arr).reshape(-1) > 0)
                if ids.size:
                    return int(ids[0])
        return None


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    anchor_face_id: int = -1
    anchor_barycentric: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        forward = normalize(np.asarray(self.look_at) - np.asarray(self.position))
        right = normalize(np.cross(forward, np.asarray(self.up)))
        if np.linalg.norm(right) < 1e-8:
            right = normalize(np.cross(forward, np.array([1.0, 0.0, 0.0])))
        up = normalize(np.cross(right, forward))
        rotation = np.stack([right, -up, forward], axis=0)
        return rotation, np.asarray(self.position, dtype=float)

    @property
    def right(self) -> np.ndarray:
        return self.frame()[0][0]

    def shifted(self, offset_world: np.ndarray) -> "Camera":
        offset_world = np.asarray(offset_world, dtype=float)
        return Camera(
            position=np.asarray(self.position) + offset_world,
            look_at=np.asarray(self.look_at) + offset_world,
            up=np.asarray(self.up),
            width=self.width,
            height=self.height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            near=self.near,
            anchor_face_id=self.anchor_face_id,
            anchor_barycentric=self.anchor_barycentric,
        )

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        rotation, origin = self.frame()
        return (rotation @ (np.asarray(points) - origin).T).T

    def camera_to_world(self, points: np.ndarray) -> np.ndarray:
        rotation, origin = self.frame()
        return (rotation.T @ np.asarray(points).T).T + origin

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.world_to_camera(points)
        z = p[:, 2]
        uv = np.column_stack(
            [
                self.fx * p[:, 0] / np.maximum(z, 1e-12) + self.cx,
                self.fy * p[:, 1] / np.maximum(z, 1e-12) + self.cy,
            ]
        )
        return uv, z

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": np.asarray(self.position).tolist(),
            "look_at": np.asarray(self.look_at).tolist(),
            "up": np.asarray(self.up).tolist(),
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "near": self.near,
            "anchor_face_id": self.anchor_face_id,
            "anchor_barycentric": list(self.anchor_barycentric),
        }


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    return normalize(
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - math.cos(angle))
    )


def sample_camera(
    template: MeshData,
    config: CameraConfig,
    rng: np.random.Generator,
    surface_weights: np.ndarray | None = None,
) -> Camera:
    """Sample a camera from the reference anatomy only."""
    areas = template.areas
    weights = areas.copy()
    if surface_weights is not None:
        sw = np.asarray(surface_weights, dtype=float).reshape(-1)
        if sw.size not in (len(template.points), len(template.faces)):
            raise ValueError("surface_weights must be per-vertex or per-face")
        if sw.size == len(template.points):
            sw = sw[template.faces].mean(axis=1)
        weights *= np.clip(sw, 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("camera surface weights contain no positive finite mass")
    face_id = int(rng.choice(len(template.faces), p=weights / weights.sum()))
    tri_ids = template.faces[face_id]
    r1, r2 = rng.random(2)
    s = math.sqrt(r1)
    bary = np.array([1.0 - s, s * (1.0 - r2), s * r2])
    anchor = (template.points[tri_ids] * bary[:, None]).sum(axis=0)
    normal = template.normals[face_id]
    tangent_a = normalize(np.cross(normal, np.array([0.0, 0.0, 1.0])))
    if np.linalg.norm(tangent_a) < 1e-8:
        tangent_a = normalize(np.cross(normal, np.array([0.0, 1.0, 0.0])))
    tangent_b = normalize(np.cross(normal, tangent_a))
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    tangent = math.cos(theta) * tangent_a + math.sin(theta) * tangent_b
    offset = float(rng.uniform(*config.tangent_offset_range_diag)) * template.bbox_diag
    look_at = anchor + tangent * offset
    look_at += (
        rng.normal(size=2)
        @ np.stack([tangent_a, tangent_b])
        * config.look_at_jitter_diag
        * template.bbox_diag
    )
    axis = normalize(rng.normal(size=3))
    tilt = math.radians(float(rng.normal(0.0, config.tilt_std_deg)))
    outward = _rotate(normal, axis, tilt)
    distance = float(rng.uniform(*config.distance_range_diag)) * template.bbox_diag
    position = look_at + outward * distance
    forward = normalize(look_at - position)
    roll = math.radians(float(rng.uniform(*config.roll_range_deg)))
    up = _rotate(tangent_b, forward, roll)
    fov = math.radians(float(rng.uniform(*config.horizontal_fov_deg_range)))
    fx = 0.5 * config.width / math.tan(0.5 * fov)
    cx = float(rng.uniform(*config.principal_x_fraction_range)) * config.width
    cy = float(rng.uniform(*config.principal_y_fraction_range)) * config.height
    return Camera(
        position,
        look_at,
        up,
        config.width,
        config.height,
        fx,
        fx,
        cx,
        cy,
        config.near_diag * template.bbox_diag,
        face_id,
        tuple(float(x) for x in bary),
    )


def load_mesh(
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    orient: bool = True,
) -> MeshData:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        # Native format emitted by the multi-anatomy deformation pipeline.
        # It stores the undeformed reference explicitly, preserving exact
        # correspondence without requiring meshio or a sidecar VTP.
        with np.load(path, allow_pickle=False) as raw:
            required = {"points", "triangles"}
            missing = required.difference(raw.files)
            if missing:
                raise ValueError(
                    f"{path} is missing required mesh arrays: {sorted(missing)}"
                )
            embedded_metadata: dict[str, Any] = {}
            if "metadata_json" in raw.files:
                try:
                    embedded_metadata = json.loads(str(raw["metadata_json"].item()))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path} contains invalid metadata_json") from exc
            points = np.asarray(raw["points"], dtype=float)
            faces = np.asarray(raw["triangles"], dtype=np.int64)
            displacement = (
                np.asarray(raw["displacement"], dtype=float)
                if "displacement" in raw.files
                else None
            )
            if displacement is None and "reference_points" in raw.files:
                displacement = points - np.asarray(raw["reference_points"], dtype=float)
            point_data = {
                name: np.asarray(raw[name])
                for name in raw.files
                if name
                not in {
                    "schema_version",
                    "reference_points",
                    "points",
                    "triangles",
                    "displacement",
                    "displacement_magnitude",
                    "metadata_json",
                }
            }
            if "reference_points" in raw.files:
                point_data["reference_points"] = np.asarray(
                    raw["reference_points"], dtype=float
                )
        mesh = MeshData(
            points,
            faces,
            displacement,
            point_data,
            {**embedded_metadata, **dict(metadata or {})},
        )
    elif path.suffix.lower() == ".vtp":
        mesh = _load_vtp(path, metadata)
    else:
        try:
            import meshio
        except ImportError as exc:
            raise ImportError("Mesh loading requires meshio") from exc
        raw = meshio.read(str(path))
        triangles = [
            np.asarray(block.data, dtype=np.int64)
            for block in raw.cells
            if block.type == "triangle"
        ]
        if not triangles:
            raise ValueError(f"{path} contains no triangle cells")
        point_data = {name: np.asarray(value) for name, value in raw.point_data.items()}
        displacement = None
        for name in ("displacement", "Displacement", "disp", "u"):
            value = point_data.get(name)
            if value is not None and value.ndim == 2 and value.shape[1] >= 3:
                displacement = np.asarray(value[:, :3], dtype=float)
                break
        mesh = MeshData(
            np.asarray(raw.points[:, :3], dtype=float),
            np.concatenate(triangles),
            displacement,
            point_data,
            dict(metadata or {}),
        )
    if orient:
        mesh.faces = orient_faces_consistently(mesh.points, mesh.faces)
    return mesh


_VTK_DTYPES = {
    "Float32": np.dtype("<f4"),
    "Float64": np.dtype("<f8"),
    "Int8": np.dtype("i1"),
    "UInt8": np.dtype("u1"),
    "Int16": np.dtype("<i2"),
    "UInt16": np.dtype("<u2"),
    "Int32": np.dtype("<i4"),
    "UInt32": np.dtype("<u4"),
    "Int64": np.dtype("<i8"),
    "UInt64": np.dtype("<u8"),
}


def _decode_vtk_data_array(
    element: ET.Element, compressed: bool, header_dtype: np.dtype
) -> np.ndarray:
    vtk_type = element.attrib.get("type", "Float32")
    if vtk_type not in _VTK_DTYPES:
        raise ValueError(f"Unsupported VTK data type {vtk_type}")
    dtype = _VTK_DTYPES[vtk_type]
    fmt = element.attrib.get("format", "ascii")
    raw_text = element.text or ""
    if fmt == "ascii":
        values = np.fromstring(raw_text, sep=" ", dtype=dtype)
    elif fmt == "binary":
        text = "".join(raw_text.split())
        if not text:
            return np.array([], dtype=dtype)
        if compressed:
            # VTK stores a base64-encoded header followed immediately by a
            # separately base64-encoded zlib payload. Header layout is:
            # n_blocks, block_size, last_block_size, compressed_sizes[].
            prefix_chars = 4 * ((header_dtype.itemsize + 2) // 3)
            prefix = base64.b64decode(text[:prefix_chars])
            n_blocks = int(
                np.frombuffer(prefix[: header_dtype.itemsize], dtype=header_dtype)[0]
            )
            header_bytes = (3 + n_blocks) * header_dtype.itemsize
            header_chars = 4 * ((header_bytes + 2) // 3)
            header_raw = base64.b64decode(text[:header_chars])
            header = np.frombuffer(header_raw[:header_bytes], dtype=header_dtype)
            compressed_sizes = header[3 : 3 + n_blocks].astype(int)
            payload = base64.b64decode(text[header_chars:])
            chunks = []
            cursor = 0
            for size in compressed_sizes:
                block = payload[cursor : cursor + size]
                chunks.append(zlib.decompress(block))
                cursor += int(size)
            raw = b"".join(chunks)
        else:
            header_chars = 4 * ((header_dtype.itemsize + 2) // 3)
            header = base64.b64decode(text[:header_chars])
            byte_count = int(
                np.frombuffer(header[: header_dtype.itemsize], dtype=header_dtype)[0]
            )
            raw = base64.b64decode(text[header_chars:])[:byte_count]
        values = np.frombuffer(raw, dtype=dtype).copy()
    else:
        raise ValueError(f"Unsupported VTK DataArray format {fmt}")
    components = int(element.attrib.get("NumberOfComponents", "1"))
    return values.reshape((-1, components)) if components > 1 else values


def _load_vtp(path: Path, metadata: dict[str, Any] | None = None) -> MeshData:
    root = ET.parse(path).getroot()
    if root.attrib.get("type") != "PolyData":
        raise ValueError(f"{path} is not a VTK PolyData file")
    compressed = "ZLib" in root.attrib.get("compressor", "")
    header_dtype = _VTK_DTYPES[root.attrib.get("header_type", "UInt32")]
    piece = root.find("./PolyData/Piece")
    if piece is None:
        raise ValueError(f"{path} contains no PolyData/Piece")

    points_element = piece.find("./Points/DataArray")
    if points_element is None:
        raise ValueError(f"{path} contains no point coordinates")
    points = np.asarray(
        _decode_vtk_data_array(points_element, compressed, header_dtype), dtype=float
    )

    polygons = piece.findall("./Polys/DataArray")
    arrays_by_name = {
        element.attrib.get("Name", ""): _decode_vtk_data_array(
            element, compressed, header_dtype
        )
        for element in polygons
    }
    connectivity = np.asarray(
        arrays_by_name.get("connectivity", []), dtype=np.int64
    ).reshape(-1)
    offsets = np.asarray(arrays_by_name.get("offsets", []), dtype=np.int64).reshape(-1)
    starts = (
        np.concatenate([[0], offsets[:-1]])
        if offsets.size
        else np.array([], dtype=np.int64)
    )
    sizes = offsets - starts
    if not offsets.size or not np.all(sizes == 3):
        raise ValueError(f"{path} must contain triangle polygons")
    faces = np.stack(
        [connectivity[start:stop] for start, stop in zip(starts, offsets)]
    ).astype(np.int64)

    point_data = {}
    for element in piece.findall("./PointData/DataArray"):
        name = element.attrib.get("Name")
        if name:
            point_data[name] = _decode_vtk_data_array(element, compressed, header_dtype)
    field_data = {}
    for element in root.findall("./PolyData/FieldData/DataArray"):
        name = element.attrib.get("Name")
        if name:
            value = _decode_vtk_data_array(element, compressed, header_dtype)
            field_data[name] = value.item() if value.size == 1 else value
    merged_metadata = {**field_data, **dict(metadata or {})}

    displacement = None
    for name in ("displacement", "Displacement", "disp", "u"):
        value = point_data.get(name)
        if value is not None and value.ndim == 2 and value.shape[1] >= 3:
            displacement = np.asarray(value[:, :3], dtype=float)
            break
    return MeshData(
        points=points,
        faces=faces,
        displacement=displacement,
        point_data=point_data,
        metadata=merged_metadata,
    )
