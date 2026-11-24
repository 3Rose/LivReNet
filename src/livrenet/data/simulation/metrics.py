from __future__ import annotations

import numpy as np


def _unique_edges(faces):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def _component_count(mask, edges):
    adjacency = [[] for _ in range(len(mask))]
    for a, b in edges:
        if mask[a] and mask[b]:
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
    seen = set()
    components = 0
    for index in np.where(mask)[0]:
        index = int(index)
        if index in seen:
            continue
        components += 1
        stack = [index]
        seen.add(index)
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return components


def prepare_surface_metric_context(points, faces):
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    edges = _unique_edges(faces)
    original_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    triangles = points[faces]
    triangle_areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    vertex_areas = np.zeros(len(points), dtype=float)
    for corner in range(3):
        np.add.at(vertex_areas, faces[:, corner], triangle_areas / 3.0)
    return {
        "edges": edges,
        "original_lengths": original_lengths,
        "valid_edges": original_lengths > 1.0e-12,
        "vertex_areas": vertex_areas,
    }


def _area_fraction(mask, vertex_areas):
    mask = np.asarray(mask, dtype=bool)
    areas = np.asarray(vertex_areas, dtype=float)
    total = float(np.sum(areas))
    if total <= 0.0:
        return float(np.mean(mask)) if len(mask) else 0.0
    return float(np.sum(areas[mask]) / total)


def surface_deformation_metrics(
    points, faces, displacement, load_weight, bc_mask, *, context=None
):
    points = np.asarray(points, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if points.shape != displacement.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and displacement must both have shape [N, 3]")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("Non-finite surface displacement")
    deformed = points + displacement
    magnitude = np.linalg.norm(displacement, axis=1)
    maximum = float(magnitude.max())
    peak = int(np.argmax(magnitude))
    mean = float(magnitude.mean())

    prepared = context or prepare_surface_metric_context(points, faces)
    edges = prepared["edges"]
    original_lengths = prepared["original_lengths"]
    vertex_areas = prepared["vertex_areas"]
    deformed_lengths = np.linalg.norm(
        deformed[edges[:, 0]] - deformed[edges[:, 1]], axis=1
    )
    valid = prepared["valid_edges"]
    stretch = deformed_lengths[valid] / original_lengths[valid]
    jumps = np.abs(magnitude[edges[:, 0]] - magnitude[edges[:, 1]])

    active = (
        magnitude >= 0.20 * maximum
        if maximum > 0
        else np.zeros_like(magnitude, dtype=bool)
    )
    high = (
        magnitude >= 0.50 * maximum
        if maximum > 0
        else np.zeros_like(magnitude, dtype=bool)
    )
    support = np.asarray(load_weight) > 0

    return {
        "max_disp_mm": maximum,
        "mean_disp_mm": mean,
        "p95_disp_mm": float(np.quantile(magnitude, 0.95)),
        "peak_vertex": peak,
        "peak_to_mean": maximum / max(mean, 1e-12) if maximum > 0 else 0.0,
        "active_fraction_rel20": float(active.mean()),
        "halfmax_fraction": float(high.mean()),
        "support_fraction": float(support.mean()),
        "active_area_fraction_rel20": _area_fraction(active, vertex_areas),
        "halfmax_area_fraction": _area_fraction(high, vertex_areas),
        "support_area_fraction": _area_fraction(support, vertex_areas),
        "moving_area_fraction_ge_1mm": _area_fraction(magnitude >= 1.0, vertex_areas),
        "moving_area_fraction_ge_2mm": _area_fraction(magnitude >= 2.0, vertex_areas),
        "moving_area_fraction_ge_5mm": _area_fraction(magnitude >= 5.0, vertex_areas),
        "edge_jump_p95_mm": float(np.quantile(jumps, 0.95)),
        "edge_jump_p95_to_peak": (
            float(np.quantile(jumps, 0.95) / maximum) if maximum > 0 else 0.0
        ),
        "edge_stretch_min": float(stretch.min()) if len(stretch) else 1.0,
        "edge_stretch_p01": float(np.quantile(stretch, 0.01)) if len(stretch) else 1.0,
        "edge_stretch_p05": float(np.quantile(stretch, 0.05)) if len(stretch) else 1.0,
        "edge_stretch_p95": float(np.quantile(stretch, 0.95)) if len(stretch) else 1.0,
        "edge_stretch_p99": float(np.quantile(stretch, 0.99)) if len(stretch) else 1.0,
        "edge_stretch_max": float(stretch.max()) if len(stretch) else 1.0,
        "high_disp_components": (
            int(_component_count(high, edges)) if maximum > 0 else 0
        ),
        "peak_inside_load_patch": bool(support[peak]) if maximum > 0 else True,
        "bc_mean_disp_mm": (
            float(magnitude[np.asarray(bc_mask, bool)].mean())
            if np.any(bc_mask)
            else 0.0
        ),
        "bc_max_disp_mm": (
            float(magnitude[np.asarray(bc_mask, bool)].max())
            if np.any(bc_mask)
            else 0.0
        ),
    }


def _tetra_matrices(points, tetrahedra):
    vertices = np.asarray(points, dtype=float)[np.asarray(tetrahedra, dtype=np.int64)]
    return np.stack(
        [
            vertices[:, 1] - vertices[:, 0],
            vertices[:, 2] - vertices[:, 0],
            vertices[:, 3] - vertices[:, 0],
        ],
        axis=2,
    )


def reference_tetrahedron_metrics(points, tetrahedra):
    matrices = _tetra_matrices(points, tetrahedra)
    determinant = np.linalg.det(matrices)
    absolute = np.abs(determinant)
    nondegenerate = absolute > 1.0e-14
    condition = np.full(len(matrices), np.inf, dtype=float)
    if np.any(nondegenerate):
        condition[nondegenerate] = np.linalg.cond(matrices[nondegenerate])
    return {
        "reference_tetra_count": int(len(matrices)),
        "reference_degenerate_fraction": float(np.mean(~nondegenerate)),
        "reference_negative_orientation_fraction": float(np.mean(determinant < 0)),
        "reference_volume_min_mm3": float(absolute.min() / 6.0),
        "reference_volume_p01_mm3": float(np.quantile(absolute / 6.0, 0.01)),
        "reference_volume_median_mm3": float(np.median(absolute / 6.0)),
        "reference_condition_p95": float(np.quantile(condition, 0.95)),
        "reference_condition_max": float(condition.max(initial=0.0)),
    }


def prepare_volume_metric_context(points, tetrahedra):
    reference = _tetra_matrices(points, tetrahedra)
    determinant_reference = np.linalg.det(reference)
    nondegenerate = np.abs(determinant_reference) > 1.0e-14
    if not np.any(nondegenerate):
        raise ValueError("All tetrahedra are degenerate")
    valid_reference = reference[nondegenerate]
    valid_determinant = determinant_reference[nondegenerate]
    return {
        "nondegenerate": nondegenerate,
        "inverse_reference": np.linalg.inv(valid_reference),
        "weights": np.abs(valid_determinant),
    }


def volume_deformation_metrics(points, tetrahedra, displacement, *, context=None):
    points = np.asarray(points, dtype=float)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=float)
    if points.shape != displacement.shape:
        raise ValueError("Volume points and displacement shapes differ")
    if not np.all(np.isfinite(displacement)):
        raise ValueError("Non-finite volume displacement")

    current = _tetra_matrices(points + displacement, tetrahedra)
    prepared = context or prepare_volume_metric_context(points, tetrahedra)
    current = current[prepared["nondegenerate"]]

    deformation_gradient = current @ prepared["inverse_reference"]
    jacobian = np.linalg.det(deformation_gradient)
    cauchy_green = np.swapaxes(deformation_gradient, 1, 2) @ deformation_gradient
    squared_stretches = np.clip(np.linalg.eigvalsh(cauchy_green), 0.0, None)
    principal_stretches = np.sqrt(squared_stretches)
    green_lagrange = 0.5 * (squared_stretches - 1.0)
    maximum_abs_strain = np.max(np.abs(green_lagrange), axis=1)

    weights = prepared["weights"]
    global_volume_ratio = float(np.sum(weights * jacobian) / np.sum(weights))
    return {
        "jacobian_min": float(jacobian.min()),
        "jacobian_p01": float(np.quantile(jacobian, 0.01)),
        "jacobian_p05": float(np.quantile(jacobian, 0.05)),
        "jacobian_p95": float(np.quantile(jacobian, 0.95)),
        "jacobian_p99": float(np.quantile(jacobian, 0.99)),
        "jacobian_max": float(jacobian.max()),
        "inverted_fraction": float(np.mean(jacobian <= 0.0)),
        "global_volume_ratio": global_volume_ratio,
        "principal_stretch_min": float(principal_stretches.min()),
        "principal_stretch_p01": float(np.quantile(principal_stretches, 0.01)),
        "principal_stretch_p99": float(np.quantile(principal_stretches, 0.99)),
        "principal_stretch_max": float(principal_stretches.max()),
        "green_lagrange_abs_p95": float(np.quantile(maximum_abs_strain, 0.95)),
        "green_lagrange_abs_p99": float(np.quantile(maximum_abs_strain, 0.99)),
        "green_lagrange_abs_max": float(maximum_abs_strain.max()),
    }


deformation_metrics = surface_deformation_metrics
