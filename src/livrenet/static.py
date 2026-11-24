"""Anatomy-specific encoder and refiner graphs."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from .surface import _weighted_mesh_graph, edges


def build_static(a):
    scale = float(a["normalization_scale"])
    xyz = (a["template_points"] - a["normalization_center"]) / scale
    ctrl = xyz[a["control_indices"]]
    count = len(ctrl)
    nn = cKDTree(ctrl).query(ctrl, k=8 + 1)[1][:, 1:]
    edge = np.stack((np.repeat(np.arange(count), nn.shape[1]), nn.reshape(-1)))
    edge = np.unique(np.concatenate((edge, edge[::-1]), axis=1), axis=1)
    rel = ctrl[edge[0]] - ctrl[edge[1]]
    metric = a["metric"].astype("f8")
    metric = 0.5 * (metric + metric.T)
    eigen, q = np.linalg.eigh(metric)
    if eigen.min() < -1e-6 * max(1.0, eigen.max()):
        raise ValueError("Indefinite FEM metric")
    metric = (q * np.maximum(eigen, 0)) @ q.T
    metric /= max(float(np.diag(metric).mean()), 1e-6)
    return dict(
        control_xyz=ctrl.astype("f4"),
        edge=edge.astype("i8"),
        edge_geometry=np.c_[rel, np.linalg.norm(rel, axis=1)].astype("f4")[None],
        degree=np.bincount(edge[1], minlength=count)
        .clip(1)
        .astype("f4")[None, :, None],
        control_basis=a["basis"][:, a["control_indices"]]
        .transpose(1, 2, 0)
        .reshape(-1, 32)
        .astype("f4"),
        energy=metric.astype("f8"),
        basis=a["basis"].astype("f4"),
    )


def _fps(distance, count):
    chosen = [0]
    nearest = distance[0].copy()
    while len(chosen) < count:
        j = int(nearest.argmax())
        chosen.append(j)
        nearest = np.minimum(nearest, distance[j])
    return np.asarray(chosen, dtype="i8")


def _coarse_edges(distance):
    distance = distance.copy()
    np.fill_diagonal(distance, np.inf)
    nn = np.argsort(distance, axis=1)[:, :8]
    pairs = np.column_stack([np.repeat(np.arange(len(distance)), 8), nn.reshape(-1)])
    return np.unique(np.r_[pairs, pairs[:, ::-1]], axis=0).T.astype("i8")


def refiner_graph(a):
    t = a["template_points"]
    f = a["triangles"]
    chosen = []
    nearest = np.full(len(t), np.inf)
    j = 0
    for _ in range(512):
        chosen.append(j)
        nearest = np.minimum(nearest, ((t - t[j]) ** 2).sum(-1))
        nearest[chosen] = -1
        j = int(nearest.argmax())
    control = np.asarray(chosen, dtype="i8")
    distances = dijkstra(_weighted_mesh_graph(t, f), directed=False, indices=control)
    if not np.isfinite(distances).all():
        raise ValueError("Disconnected anatomy")
    dd = distances[:, control]
    coarse = _fps(dd, 128)
    owner0 = distances.argmin(0).astype("i8")
    owner1 = dd[coarse].argmin(0).astype("i8")
    if len(np.unique(owner0)) != 512 or len(np.unique(owner1)) != 128:
        raise ValueError("Empty anatomical graph cluster")
    fine = edges(f)
    fine = np.concatenate([fine, fine[:, ::-1]], axis=0).T
    value = dict(
        e0=fine,
        e1=_coarse_edges(dd),
        e2=_coarse_edges(dd[np.ix_(coarse, coarse)]),
        owner0=owner0,
        owner1=owner1,
        n1=np.asarray(512),
        n2=np.asarray(128),
    )
    return value
