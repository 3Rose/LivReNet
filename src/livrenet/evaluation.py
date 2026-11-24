"""Material displacement metrics and paired observation aggregation."""

from collections import defaultdict

import numpy as np

from scipy.sparse.csgraph import dijkstra

from .surface import Scene, _weighted_mesh_graph


def measure(anatomy, points, predicted, target, support, identity):
    error = np.linalg.norm(predicted - target, axis=1)
    magnitude = np.linalg.norm(target, axis=1)
    motion = np.linalg.norm(predicted, axis=1)
    maximum = magnitude.max()

    def region(mask):
        return float(error[mask].mean()) if mask.any() else None

    reference = anatomy.vertices[anatomy.faces]
    deformed = (anatomy.vertices + predicted)[anatomy.faces]
    ref_normal = np.cross(
        reference[:, 1] - reference[:, 0], reference[:, 2] - reference[:, 0]
    )
    normal = np.cross(deformed[:, 1] - deformed[:, 0], deformed[:, 2] - deformed[:, 0])
    ratios = np.linalg.norm(normal, axis=1) / np.linalg.norm(ref_normal, axis=1).clip(
        1e-12
    )
    truth = anatomy.vertices + target
    prediction = anatomy.vertices + predicted
    truth_mesh = truth[anatomy.faces]
    areas = (
        np.linalg.norm(
            np.cross(
                truth_mesh[:, 1] - truth_mesh[:, 0], truth_mesh[:, 2] - truth_mesh[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    weights = np.zeros(len(truth))
    np.add.at(weights, anatomy.faces.ravel(), np.repeat(areas / 3, 3))
    surface = Scene(prediction, anatomy.faces)
    truth_surface = Scene(truth, anatomy.faces)
    surface_error = 0.5 * (truth_surface.distance(prediction) + surface.distance(truth))
    result = dict(
        global_epe_mm=float(error.mean()),
        p95_epe_mm=float(np.quantile(error, 0.95)),
        peak_epe_mm=region(magnitude >= 0.8 * maximum) if maximum >= 1 else None,
        quiet_epe_mm=region(magnitude <= 0.5),
        active_epe_mm=region(magnitude >= max(1, 0.2 * maximum))
        if maximum >= 1
        else None,
        surface_mm=float(np.dot(surface_error, weights) / weights.sum()),
        observation_to_mesh_mm=float(surface.distance(points).mean()),
        min_area_ratio=float(ratios.min()),
        flagged_faces=int(
            np.count_nonzero((ratios < 0.1) | ((ref_normal * normal).sum(1) <= 0))
        ),
        identity_tail_mm=float(np.sort(motion)[-max(1, len(motion) // 100) :].mean())
        if identity
        else None,
        identity_max_mm=float(motion.max()) if identity else None,
    )
    if support is not None:
        distance = dijkstra(
            _weighted_mesh_graph(anatomy.vertices, anatomy.faces),
            directed=False,
            indices=np.flatnonzero(support),
            min_only=True,
        )
        result.update(
            observed_epe_mm=region(support),
            hidden_epe_mm=region(~support),
            nearby_hidden_epe_mm=region(~support & (distance <= 20)),
            distant_hidden_epe_mm=region(~support & (distance > 20)),
        )
    return result


def macro(records, metric):
    """Average cameras within dose and parent, parents within anatomy, then anatomies."""
    groups = defaultdict(list)
    for row in records:
        value = row["metrics"].get(metric)
        if value is not None:
            groups[(row["anatomy_id"], row["parent_id"], row["noise_mm"])].append(value)
    parents = defaultdict(list)
    for (anatomy, parent, _), observations in groups.items():
        parents[(anatomy, parent)].append(np.mean(observations))
    anatomies = defaultdict(list)
    for (anatomy, _), doses in parents.items():
        anatomies[anatomy].append(np.mean(doses))
    return (
        float(np.mean([np.mean(values) for values in anatomies.values()]))
        if anatomies
        else None
    )


def summary(records):
    if not records:
        raise ValueError("No evaluation records to summarize.")
    metrics = sorted({key for row in records for key in row["metrics"]})
    stages = sorted({row["stage"] for row in records})
    groups = {
        "nominal": lambda row: row["noise_mm"] <= 0.5,
        "stress": lambda row: row["noise_mm"] > 0.5,
    }
    for dose in sorted({row["noise_mm"] for row in records}):
        groups[f"rms_{dose:g}_mm"] = lambda row, dose=dose: row["noise_mm"] == dose
    output = {}
    for stage in stages:
        output[stage] = {}
        for group, check in groups.items():
            for identity in (False, True):
                subset = [
                    row
                    for row in records
                    if row["stage"] == stage
                    and check(row)
                    and row["is_identity"] == identity
                ]
                if subset:
                    key = f"{group}_{'identity' if identity else 'deforming'}"
                    output[stage][key] = {
                        metric: macro(subset, metric) for metric in metrics
                    }
    return output
