"""Native implementation of the reference renderer; fallback remains exact Python."""

from __future__ import annotations
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np

from .rasterizer import RasterizedSurface, surface_points
from .stereo import StereoVisibility, make_right_camera


@lru_cache(maxsize=1)
def native_library():
    source = Path(__file__).with_name("raster.c")
    key = hashlib.sha256(
        source.read_bytes() + b"cc-O3-c11-ffp-contract-off-v1"
    ).hexdigest()[:20]
    folder = Path(tempfile.gettempdir()) / "thesis_native_raster" / key
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "raster.so"
    if not target.exists():
        temporary = folder / f"raster_{os.getpid()}.so"
        result = subprocess.run(
            [
                "cc",
                "-O3",
                "-std=c11",
                "-ffp-contract=off",
                "-shared",
                "-fPIC",
                str(source),
                "-lm",
                "-o",
                str(temporary),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                "Native raster compilation failed: " + result.stderr[-2000:]
            )
        os.replace(temporary, target)
    library = ctypes.CDLL(str(target))
    library.raster_kernel.argtypes = (
        [ctypes.c_void_p] * 4
        + [ctypes.c_int64, ctypes.c_int, ctypes.c_int]
        + [ctypes.c_double] * 5
        + [ctypes.c_void_p] * 6
    )
    library.raster_kernel.restype = None
    return library


def rasterize(mesh, camera, backend="native"):
    if backend == "python":
        from .rasterizer import rasterize as reference

        return reference(mesh, camera)
    if backend != "native":
        raise ValueError("Renderer must be native or python")
    h, w = camera.height, camera.width
    cp = np.ascontiguousarray(camera.world_to_camera(mesh.points), np.float64)
    faces = np.ascontiguousarray(mesh.faces, np.int64)
    if (
        h <= 0
        or w <= 0
        or not np.isfinite(cp).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(cp)
    ):
        raise ValueError("Invalid renderer geometry/dimensions")
    centers = mesh.points[faces].mean(1)
    directions = camera.position[None, :] - centers
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    fi = np.ascontiguousarray(
        np.clip(np.sum(mesh.normals * directions, axis=1), 0.0, 1.0), np.float64
    )
    areas = np.ascontiguousarray(mesh.areas, np.float64)
    depth = np.empty((h, w), np.float64)
    face_map = np.empty((h, w), np.int64)
    bary = np.zeros((h, w, 3), np.float32)
    incidence = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float64)
    projected = np.zeros(len(faces), np.float64)
    native_library().raster_kernel(
        *(a.ctypes.data for a in (cp, faces, fi, areas)),
        len(faces),
        w,
        h,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.near,
        *(a.ctypes.data for a in (depth, face_map, bary, incidence, weight, projected)),
    )
    return RasterizedSurface(
        depth.astype(np.float32),
        face_map,
        bary,
        incidence,
        weight.astype(np.float32),
        projected.astype(np.float32),
    )


def stereo_visibility(mesh, camera, profile, unit_to_mm, config, backend="native"):
    left = rasterize(mesh, camera, backend)
    if not config.enabled:
        return StereoVisibility(
            left,
            None,
            left.valid.copy(),
            np.full((*left.depth.shape, 2), -1, np.int32),
            None,
        )
    right_camera = make_right_camera(camera, profile, unit_to_mm)
    right = rasterize(mesh, right_camera, backend)
    mutual = np.zeros_like(left.valid)
    right_uv = np.full((*left.depth.shape, 2), -1, np.int32)
    if not left.valid.any() or not right.valid.any():
        return StereoVisibility(left, right, mutual, right_uv, right_camera)
    world, _, _ = surface_points(mesh, left, left.valid)
    uv, z = right_camera.project(world)
    x, y = np.rint(uv[:, 0] - 0.5).astype(int), np.rint(uv[:, 1] - 0.5).astype(int)
    bounds = (
        (x >= 0)
        & (x < camera.width)
        & (y >= 0)
        & (y < camera.height)
        & (z > right_camera.near)
    )
    ids = np.flatnonzero(left.valid)
    accepted = np.zeros(len(ids), bool)
    valid = np.flatnonzero(bounds)
    rd = right.depth[y[valid], x[valid]].astype(float)
    tolerance = np.maximum(
        config.occlusion_tolerance_mm / unit_to_mm,
        config.occlusion_relative_tolerance * np.maximum(z[valid], 1e-6),
    )
    accepted[valid] = (rd > 0) & (np.abs(rd - z[valid]) <= tolerance)
    mutual.reshape(-1)[ids] = accepted if config.require_mutual_visibility else bounds
    right_uv.reshape(-1, 2)[ids[bounds]] = np.column_stack((x[bounds], y[bounds]))
    return StereoVisibility(left, right, mutual, right_uv, right_camera)
