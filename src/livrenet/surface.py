"""Surface geometry and observation-derived features."""

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


class Scene:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.scene.add_triangles(
            o3d.core.Tensor(self.vertices.astype("float32")),
            o3d.core.Tensor(self.faces.astype("uint32")),
        )

    def closest(self, points):
        q = self.scene.compute_closest_points(
            o3d.core.Tensor(np.asarray(points, "float32")), nthreads=2
        )
        ids = q["primitive_ids"].numpy().astype(np.int64)
        uv = q["primitive_uvs"].numpy().astype(float)
        bary = np.maximum(np.column_stack((1 - uv.sum(1), uv)), 0.0)
        bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
        anchor = np.einsum("pvd,pv->pd", self.vertices[self.faces[ids]], bary)
        return ids, bary, anchor, q["primitive_normals"].numpy().astype(float)

    def distance(self, points):
        return (
            self.scene.compute_distance(
                o3d.core.Tensor(np.asarray(points, "float32")), nthreads=2
            )
            .numpy()
            .astype(float)
        )


def edges(faces):
    f = np.asarray(faces, np.int64)
    e = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    return np.unique(np.sort(e, axis=1), axis=0)


def graph(vertices, faces):
    e = edges(faces)
    length = np.maximum(
        np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1), 1e-8
    )
    return sparse.coo_matrix(
        (np.r_[length, length], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()


def normals_areas(vertices, faces):
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    vnormal = np.zeros_like(vertices)
    varea = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(vnormal, faces[:, k], n)
        np.add.at(varea, faces[:, k], area / 3)
    return vnormal / np.maximum(
        np.linalg.norm(vnormal, axis=1, keepdims=True), 1e-10
    ), varea


def observation_features(template, faces, field, points, noise_sigma_mm=0.0):
    """Current PC-surface residual, barycentric splat, and geodesic support.

    No material correspondence or GT mask. Coherent sensor bias can mimic signal.
    """
    current = template + field
    ids, bary, anchor, normal = Scene(current, faces).closest(points)
    residual = points - anchor
    vertices = faces[ids]
    weight = np.zeros(len(template))
    sums = np.zeros_like(template)
    second = np.zeros(len(template))
    for k in range(3):
        np.add.at(weight, vertices[:, k], bary[:, k])
        np.add.at(sums, vertices[:, k], residual * bary[:, k, None])
        np.add.at(second, vertices[:, k], (residual**2).sum(1) * bary[:, k])
    avg = sums / np.maximum(weight[:, None], 1e-8)
    rms = np.sqrt(np.maximum(second, 0.0) / np.maximum(weight, 1e-8))
    seed = np.flatnonzero(weight > 0.05)
    geo = graph(current, faces)
    distance = (
        dijkstra(geo, directed=False, indices=seed, min_only=True)
        if len(seed)
        else np.full(len(template), np.inf)
    )
    support = np.exp(-np.minimum(distance, 200.0) / 15.0)
    vn, _ = normals_areas(current, faces)
    position = (template - template.mean(0)) / 100.0
    features = np.column_stack(
        (
            position,
            vn,
            field / 10.0,
            avg / 5.0,
            np.log1p(weight),
            rms / 5.0,
            support,
            np.full(len(template), noise_sigma_mm / 5.0),
        )
    )
    return features.astype("float32"), support.astype("float32"), edges(faces)


def _weighted_mesh_graph(points: np.ndarray, faces: np.ndarray) -> sparse.csr_matrix:
    """Build the template surface graph used by the locked support ring."""

    faces = np.asarray(faces, dtype=np.int64)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    weights = np.linalg.norm(
        np.asarray(points, dtype=np.float64)[edges[:, 0]]
        - np.asarray(points, dtype=np.float64)[edges[:, 1]],
        axis=1,
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((weights, weights))
    return sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(points), len(points))
    ).tocsr()
