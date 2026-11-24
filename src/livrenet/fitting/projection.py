"""Exhaustive point-to-triangle projection; no nearest-vertex approximation.

CUDA uses an installed Triton JIT, without nvcc or new packages. CPU uses the
reference Open3D scene. Search is float32 on both backends; anchors interpolate
the original tensor vertices (including float64 for the solver).
"""

import torch


def closest(vertices, faces, points):
    """Return face IDs, barycentrics, anchors, unit normals for [B,N,3]."""
    if vertices.ndim == 2:
        vertices = vertices[None]
    if vertices.device.type == "cuda":
        from .projection_triton import search

        ids, bary = search(vertices, faces, points)
        bary = bary.to(vertices.dtype)
    elif vertices.device.type == "cpu":
        import numpy as np
        import open3d as o3d

        all_ids, all_bary = [], []
        for v in vertices:
            scene = o3d.t.geometry.RaycastingScene(nthreads=1)
            scene.add_triangles(
                o3d.core.Tensor(v.numpy().astype("f4")),
                o3d.core.Tensor(faces.numpy().astype("uint32")),
            )
            hit = scene.compute_closest_points(
                o3d.core.Tensor(points.numpy().astype("f4")), nthreads=1
            )
            uv = hit["primitive_uvs"].numpy().astype(float)
            all_bary.append(np.c_[1 - uv.sum(1), uv])
            all_ids.append(hit["primitive_ids"].numpy().astype("i8"))
        ids = torch.as_tensor(np.stack(all_ids), dtype=torch.long)
        bary = torch.as_tensor(np.stack(all_bary), dtype=vertices.dtype)
    else:
        raise ValueError("Only CPU and CUDA are supported")
    batch = torch.arange(len(vertices), device=vertices.device)[:, None, None]
    triangle = vertices[batch, faces[ids]]
    anchor = (triangle * bary[..., None]).sum(-2)
    cross = torch.linalg.cross(
        triangle[..., 1, :] - triangle[..., 0, :],
        triangle[..., 2, :] - triangle[..., 0, :],
    )
    normal = cross / torch.linalg.vector_norm(cross, dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    return ids, bary, anchor, normal
