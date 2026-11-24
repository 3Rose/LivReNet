from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import ProjectConfig, load_config, validate_config
from .fem import LinearElasticSolver, SolveResult
from .geometry import (
    append_jsonl,
    bbox_diag,
    edge_graph,
    map_surface_to_volume,
    read_surface,
    triangle_array,
    write_sample,
)
from .metrics import (
    prepare_surface_metric_context,
    prepare_volume_metric_context,
    reference_tetrahedron_metrics,
    surface_deformation_metrics,
    volume_deformation_metrics,
)
from .sampling import (
    SampleIntent,
    anatomy_amplitude_scale,
    build_anchors,
    prepare_anatomy_support,
    sample_deformation_load,
    sample_intent,
    surface_characteristics,
)


_WORKER_CACHE = {}
_WORKER_SPECS = {}
_WORKER_CFG = None


def _configure_native_worker_threads(cfg) -> None:
    """Set native thread limits before spawning serial FEM workers."""

    count = str(int(cfg.solver.native_threads_per_worker))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = count


def _load_surface_transfer(path, source_count, target_count):
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        indices = np.asarray(archive["aligned_vertex_indices"], dtype=np.int64)
        weights = np.asarray(archive["weights"], dtype=np.float64)
    if indices.shape != weights.shape or indices.shape[0] != target_count:
        raise ValueError(f"Invalid surface transfer shape in {path}")
    if indices.size and (indices.min() < 0 or indices.max() >= source_count):
        raise ValueError(f"Surface transfer indices out of bounds in {path}")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-5):
        raise ValueError(f"Surface transfer weights do not sum to one in {path}")
    return indices, weights


def _transfer_result_to_output(result, transfer):
    """Transfer smooth FEM fields from the aligned mesh to atlas topology."""

    if transfer is None:
        return result
    indices, weights = transfer
    displacement = np.asarray(result["displacement"])
    result["displacement"] = np.sum(displacement[indices] * weights[..., None], axis=1)
    transferred = {}
    nearest_source = indices[np.arange(len(indices)), np.argmax(weights, axis=1)]
    for name, values in result["point_data"].items():
        values = np.asarray(values)
        # Masks and signed/sentinel-valued geodesics should preserve a real
        # source vertex value. Smooth scalar weights use interpolation.
        if values.dtype.kind in "biu" or name == "geodesic_distance_to_load":
            transferred[name] = values[nearest_source]
        elif values.ndim == 1:
            transferred[name] = np.sum(values[indices] * weights, axis=1)
        else:
            transferred[name] = np.sum(values[indices] * weights[..., None], axis=1)
    result["point_data"] = transferred
    metadata = result["meta"]
    if "max_disp_mm" in metadata:
        metadata["simulation_max_disp_mm"] = float(metadata["max_disp_mm"])
    output_magnitude = np.linalg.norm(result["displacement"], axis=1)
    metadata["max_disp_mm"] = float(output_magnitude.max(initial=0.0))
    metadata["output_representation"] = "atlas_transferred_from_aligned_fem"
    metadata["load_center_vertex"] = (
        int(np.argmax(transferred["load_weight"]))
        if np.any(transferred.get("load_weight", 0.0) > 0.0)
        else -1
    )
    return result


class ValidityError(RuntimeError):
    def __init__(self, reasons, metrics):
        self.reasons = tuple(reasons)
        self.metrics = dict(metrics)
        super().__init__("Validity rejection: " + ", ".join(self.reasons))


class SampleGenerationExhausted(RuntimeError):
    """All screened proposals for one immutable quota cell were rejected."""


def _stable_seed(base_seed, *parts):
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _sample_split(cfg, template_split, rng):
    if template_split is not None:
        split = str(template_split).lower()
        return split, bool(
            split == "test" and rng.random() < cfg.splits.ood_fraction_within_test
        )
    value = rng.random()
    if value < cfg.splits.train_fraction:
        return "train", False
    if value < cfg.splits.train_fraction + cfg.splits.val_fraction:
        return "val", False
    return "test", bool(rng.random() < cfg.splits.ood_fraction_within_test)


def _largest_remainder_counts(total, weighted_labels):
    """Allocate an integer total while preserving requested proportions."""

    labels = [label for label, _ in weighted_labels]
    weights = np.asarray([weight for _, weight in weighted_labels], dtype=float)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("Quota weights must be non-negative and non-zero")
    raw = int(total) * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    for position in np.argsort(-(raw - counts), kind="stable")[
        : int(total) - int(counts.sum())
    ]:
        counts[position] += 1
    return {label: int(count) for label, count in zip(labels, counts)}


def _accepted_bin_quota_targets(cfg, total):
    bins = list(cfg.displacement.bins)
    bin_weights = np.asarray([float(item[3]) for item in bins], dtype=float)
    if np.any(bin_weights < 0) or float(bin_weights.sum()) <= 0:
        raise ValueError("displacement bin weights must be non-negative and non-zero")
    return _largest_remainder_counts(
        total,
        [
            ("identity", float(cfg.displacement.identity_fraction)),
            *[
                (
                    str(item[0]),
                    (1.0 - cfg.displacement.identity_fraction) * float(weight),
                )
                for item, weight in zip(bins, bin_weights / bin_weights.sum())
            ],
        ],
    )


def _accepted_family_quota_targets(cfg, total):
    """Exact integer family margins among non-identity accepted samples."""

    identity = _accepted_bin_quota_targets(cfg, total)["identity"]
    return _largest_remainder_counts(
        int(total) - int(identity),
        [
            (str(family), float(weight))
            for family, weight in cfg.deformation.family_weights.items()
        ],
    )


def _residual_transport(row_remaining, col_remaining, allowed, priority):
    """Integral bipartite max-flow used to round a fractional quota table."""

    families = list(row_remaining)
    bins = list(col_remaining)
    source, sink = ("source",), ("sink",)
    capacity = {}
    adjacency = {}

    def add_edge(left, right, value):
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
        capacity[(left, right)] = int(value)
        capacity[(right, left)] = 0

    for family in families:
        add_edge(source, ("family", family), row_remaining[family])
    for family in families:
        ordered = sorted(
            (name for name in bins if allowed[(family, name)]),
            key=lambda name: (-priority[(family, name)], name),
        )
        for name in ordered:
            add_edge(
                ("family", family),
                ("bin", name),
                sum(row_remaining.values()),
            )
    for name in bins:
        add_edge(("bin", name), sink, col_remaining[name])

    flow = {edge: 0 for edge in capacity}
    required = int(sum(row_remaining.values()))
    delivered = 0
    while delivered < required:
        parent = {source: None}
        queue = [source]
        for node in queue:
            if sink in parent:
                break
            for neighbour in adjacency.get(node, ()):  # deterministic order
                if neighbour not in parent and (
                    capacity[(node, neighbour)] - flow[(node, neighbour)] > 0
                ):
                    parent[neighbour] = node
                    queue.append(neighbour)
        if sink not in parent:
            raise ValueError("Family/bin quota margins are not jointly feasible")
        increment = required - delivered
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            increment = min(
                increment,
                capacity[(previous, node)] - flow[(previous, node)],
            )
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            flow[(previous, node)] += increment
            flow[(node, previous)] -= increment
            node = previous
        delivered += increment

    return {
        (family, name): int(flow[(("family", family), ("bin", name))])
        for family in families
        for name in bins
        if allowed[(family, name)]
    }


def _accepted_joint_quota_targets(cfg, total):
    """Joint accepted family x max-displacement-bin quota table.

    Both requested margins are exact after integer rounding. Structurally
    impossible cells (for example ``local_narrow/large``) receive zero.
    """

    bin_targets_all = _accepted_bin_quota_targets(cfg, total)
    identity_target = int(bin_targets_all["identity"])
    bin_targets = {
        str(name): int(count)
        for name, count in bin_targets_all.items()
        if name != "identity"
    }
    family_targets = _accepted_family_quota_targets(cfg, total)
    families = list(family_targets)
    bins = list(bin_targets)
    non_identity = sum(family_targets.values())
    if non_identity != sum(bin_targets.values()):
        raise RuntimeError("Family and displacement quota totals differ")

    allowed = {}
    for family in families:
        family_max = float(cfg.deformation.family_max_displacement_mm[family])
        for name, lower, _upper, _weight in cfg.displacement.bins:
            allowed[(family, str(name))] = family_max > float(lower)

    matrix = np.outer(
        np.asarray([family_targets[name] for name in families], dtype=float),
        np.asarray([bin_targets[name] for name in bins], dtype=float),
    ) / max(non_identity, 1)
    mask = np.asarray(
        [[allowed[(family, name)] for name in bins] for family in families],
        dtype=bool,
    )
    matrix[~mask] = 0.0
    row_goal = np.asarray([family_targets[name] for name in families], dtype=float)
    col_goal = np.asarray([bin_targets[name] for name in bins], dtype=float)
    for _ in range(1000):
        row_sum = matrix.sum(axis=1)
        if np.any((row_goal > 0) & (row_sum <= 0)):
            raise ValueError("A requested family has no feasible displacement bin")
        matrix *= np.divide(
            row_goal, row_sum, out=np.ones_like(row_goal), where=row_sum > 0
        )[:, None]
        col_sum = matrix.sum(axis=0)
        if np.any((col_goal > 0) & (col_sum <= 0)):
            raise ValueError("A displacement bin has no feasible family")
        matrix *= np.divide(
            col_goal, col_sum, out=np.ones_like(col_goal), where=col_sum > 0
        )[None, :]
        if (
            max(
                np.max(np.abs(matrix.sum(axis=1) - row_goal), initial=0.0),
                np.max(np.abs(matrix.sum(axis=0) - col_goal), initial=0.0),
            )
            < 1.0e-10
        ):
            break
    else:
        raise ValueError("Family/bin quota margins did not converge")

    rounded = np.floor(matrix + 1.0e-12).astype(int)
    row_remaining = {
        family: int(family_targets[family] - rounded[row].sum())
        for row, family in enumerate(families)
    }
    col_remaining = {
        name: int(bin_targets[name] - rounded[:, col].sum())
        for col, name in enumerate(bins)
    }
    priority = {
        (family, name): float(matrix[row, col] - rounded[row, col])
        for row, family in enumerate(families)
        for col, name in enumerate(bins)
    }
    residual = _residual_transport(row_remaining, col_remaining, allowed, priority)
    joint = {
        (family, name): int(rounded[row, col] + residual.get((family, name), 0))
        for row, family in enumerate(families)
        for col, name in enumerate(bins)
        if allowed[(family, name)]
    }
    if {
        family: sum(joint.get((family, name), 0) for name in bins)
        for family in families
    } != family_targets:
        raise RuntimeError("Rounded joint quota does not preserve family margins")
    if {
        name: sum(joint.get((family, name), 0) for family in families) for name in bins
    } != bin_targets:
        raise RuntimeError("Rounded joint quota does not preserve bin margins")
    return {
        "identity": identity_target,
        "bins": bin_targets_all,
        "families": family_targets,
        "joint": joint,
    }


def _balanced_joint_counts(row_targets, col_targets):
    """Round an unconstrained proportional table with both margins exact."""

    row_targets = {str(key): int(value) for key, value in row_targets.items()}
    col_targets = {str(key): int(value) for key, value in col_targets.items()}
    total = int(sum(row_targets.values()))
    if total != int(sum(col_targets.values())):
        raise ValueError("Joint quota margins have different totals")
    if any(value < 0 for value in (*row_targets.values(), *col_targets.values())):
        raise ValueError("Joint quota margins cannot be negative")
    if total == 0:
        return {(row, col): 0 for row in row_targets for col in col_targets}

    rows = list(row_targets)
    cols = list(col_targets)
    matrix = (
        np.outer(
            np.asarray([row_targets[row] for row in rows], dtype=float),
            np.asarray([col_targets[col] for col in cols], dtype=float),
        )
        / total
    )
    rounded = np.floor(matrix + 1.0e-12).astype(int)
    row_remaining = {
        row: int(row_targets[row] - rounded[index].sum())
        for index, row in enumerate(rows)
    }
    col_remaining = {
        col: int(col_targets[col] - rounded[:, index].sum())
        for index, col in enumerate(cols)
    }
    allowed = {(row, col): True for row in rows for col in cols}
    priority = {
        (row, col): float(matrix[row_index, col_index] - rounded[row_index, col_index])
        for row_index, row in enumerate(rows)
        for col_index, col in enumerate(cols)
    }
    residual = _residual_transport(row_remaining, col_remaining, allowed, priority)
    joint = {
        (row, col): int(rounded[row_index, col_index] + residual.get((row, col), 0))
        for row_index, row in enumerate(rows)
        for col_index, col in enumerate(cols)
    }
    if {row: sum(joint[(row, col)] for col in cols) for row in rows} != row_targets:
        raise RuntimeError("Rounded joint quota does not preserve row margins")
    if {col: sum(joint[(row, col)] for row in rows) for col in cols} != col_targets:
        raise RuntimeError("Rounded joint quota does not preserve column margins")
    return joint


def _difficulty_aware_anatomy_bin_counts(cfg, row_targets, col_targets):
    """Preserve exact margins while keeping severe bins off fragile meshes."""

    scores = cfg.quotas.anatomy_difficulty_scores
    strength = float(cfg.quotas.difficulty_bias_strength)
    if not scores or strength <= 0:
        return _balanced_joint_counts(row_targets, col_targets)
    rows = list(row_targets)
    cols = list(col_targets)
    total = int(sum(row_targets.values()))
    if total == 0:
        return {(row, col): 0 for row in rows for col in cols}

    raw_scores = np.asarray([float(scores[row]) for row in rows], dtype=float)
    reference = float(np.median(raw_scores))
    difficulty = np.maximum(np.log(raw_scores / max(reference, 1.0e-12)), 0.0)
    configured_order = [str(item[0]) for item in cfg.displacement.bins]
    rank = {
        name: index / max(len(configured_order) - 1, 1)
        for index, name in enumerate(configured_order)
    }
    severity = np.asarray([rank[name] for name in cols], dtype=float)
    thresholds = cfg.quotas.max_anatomy_difficulty_by_bin
    allowed = {
        (row, col): (
            col not in thresholds or float(scores[row]) <= float(thresholds[col])
        )
        for row in rows
        for col in cols
    }
    mask = np.asarray(
        [[allowed[(row, col)] for col in cols] for row in rows], dtype=bool
    )
    matrix = (
        np.outer(
            np.asarray([row_targets[row] for row in rows], dtype=float),
            np.asarray([col_targets[col] for col in cols], dtype=float),
        )
        / total
    )
    matrix *= np.exp(-strength * difficulty[:, None] * severity[None, :])
    matrix[~mask] = 0.0
    row_goal = np.asarray([row_targets[row] for row in rows], dtype=float)
    col_goal = np.asarray([col_targets[col] for col in cols], dtype=float)
    for _ in range(1000):
        row_sum = matrix.sum(axis=1)
        if np.any((row_goal > 0) & (row_sum <= 0)):
            raise ValueError(
                "An anatomy has no feasible displacement bin under the "
                "configured difficulty thresholds"
            )
        matrix *= np.divide(
            row_goal, row_sum, out=np.ones_like(row_goal), where=row_sum > 0
        )[:, None]
        col_sum = matrix.sum(axis=0)
        if np.any((col_goal > 0) & (col_sum <= 0)):
            raise ValueError(
                "A displacement bin has no feasible anatomy under the "
                "configured difficulty thresholds"
            )
        matrix *= np.divide(
            col_goal, col_sum, out=np.ones_like(col_goal), where=col_sum > 0
        )[None, :]
        if (
            max(
                np.max(np.abs(matrix.sum(axis=1) - row_goal), initial=0.0),
                np.max(np.abs(matrix.sum(axis=0) - col_goal), initial=0.0),
            )
            < 1.0e-10
        ):
            break
    else:
        raise ValueError("Difficulty-aware quota margins did not converge")

    rounded = np.floor(matrix + 1.0e-12).astype(int)
    row_remaining = {
        row: int(row_targets[row] - rounded[index].sum())
        for index, row in enumerate(rows)
    }
    col_remaining = {
        col: int(col_targets[col] - rounded[:, index].sum())
        for index, col in enumerate(cols)
    }
    priority = {
        (row, col): float(matrix[row_index, col_index] - rounded[row_index, col_index])
        for row_index, row in enumerate(rows)
        for col_index, col in enumerate(cols)
    }
    residual = _residual_transport(row_remaining, col_remaining, allowed, priority)
    return {
        (row, col): int(rounded[row_index, col_index] + residual.get((row, col), 0))
        for row_index, row in enumerate(rows)
        for col_index, col in enumerate(cols)
    }


def _accepted_anatomy_joint_quota_targets(cfg, template_ids, total):
    """Exact anatomy x family x displacement-bin accepted-output plan."""

    template_ids = [str(template_id) for template_id in template_ids]
    if not template_ids or len(template_ids) != len(set(template_ids)):
        raise ValueError("Anatomy quota planning requires unique template ids")
    if not cfg.quotas.anatomy_difficulty_scores:
        discovered = {
            str(template.template_id): float(template.difficulty_score)
            for template in cfg.templates
            if template.enabled and template.difficulty_score is not None
        }
        if set(discovered) == set(template_ids):
            cfg.quotas.anatomy_difficulty_scores = discovered
    weights = cfg.quotas.anatomy_weights
    anatomy_targets = _largest_remainder_counts(
        int(total),
        [
            (
                template_id,
                float(weights[template_id]) if weights else 1.0,
            )
            for template_id in template_ids
        ],
    )
    global_targets = _accepted_joint_quota_targets(cfg, int(total))
    category_targets = {
        "identity": int(global_targets["identity"]),
        **{family: int(count) for family, count in global_targets["families"].items()},
    }
    anatomy_family = _balanced_joint_counts(anatomy_targets, category_targets)

    anatomy_family_bin = {
        (template_id, "identity", "identity"): int(
            anatomy_family[(template_id, "identity")]
        )
        for template_id in template_ids
    }
    for family in global_targets["families"]:
        family_bins = {
            bin_name: int(count)
            for (candidate_family, bin_name), count in global_targets["joint"].items()
            if candidate_family == family
        }
        allocated = _difficulty_aware_anatomy_bin_counts(
            cfg,
            {
                template_id: anatomy_family[(template_id, family)]
                for template_id in template_ids
            },
            family_bins,
        )
        anatomy_family_bin.update(
            {
                (template_id, family, bin_name): int(count)
                for (template_id, bin_name), count in allocated.items()
            }
        )

    if {
        template_id: sum(
            count
            for (candidate, _family, _bin), count in anatomy_family_bin.items()
            if candidate == template_id
        )
        for template_id in template_ids
    } != anatomy_targets:
        raise RuntimeError("Anatomy/family/bin plan changed anatomy margins")
    return {
        **global_targets,
        "anatomies": anatomy_targets,
        "joint_anatomy_family": anatomy_family,
        "joint_anatomy_family_bin": anatomy_family_bin,
    }


def accepted_quota_plan(cfg, total):
    """Return the deterministic accepted-output quota plan as JSON-safe data."""

    targets = _accepted_joint_quota_targets(cfg, int(total))
    output = {
        "identity": int(targets["identity"]),
        "bins": {name: int(value) for name, value in targets["bins"].items()},
        "families": {name: int(value) for name, value in targets["families"].items()},
        "joint": {
            family: {
                bin_name: int(targets["joint"].get((family, bin_name), 0))
                for bin_name in targets["bins"]
                if bin_name != "identity"
            }
            for family in targets["families"]
        },
    }
    template_ids = [
        template.template_id for template in cfg.templates if template.enabled
    ]
    if template_ids:
        anatomy_targets = _accepted_anatomy_joint_quota_targets(
            cfg, template_ids, int(total)
        )
        output["anatomies"] = anatomy_targets["anatomies"]
        output["anatomy_family"] = {
            anatomy: {
                family: int(anatomy_targets["joint_anatomy_family"][(anatomy, family)])
                for family in ("identity", *targets["families"])
            }
            for anatomy in template_ids
        }
        output["anatomy_family_bin"] = {
            anatomy: {
                family: {
                    bin_name: int(count)
                    for (
                        candidate_anatomy,
                        candidate_family,
                        bin_name,
                    ), count in anatomy_targets["joint_anatomy_family_bin"].items()
                    if candidate_anatomy == anatomy and candidate_family == family
                }
                for family in ("identity", *targets["families"])
            }
            for anatomy in template_ids
        }
    return output


def _plan_accepted_joint_assignments(cfg, final_count, existing):
    targets = _accepted_joint_quota_targets(cfg, final_count)
    flat_targets = {("identity", "identity"): targets["identity"]}
    flat_targets.update(targets["joint"])
    observed = {key: 0 for key in flat_targets}
    for index, row in existing.items():
        if index >= final_count:
            continue
        bin_name = str(row.get("quota_target_disp_bin", row.get("target_disp_bin", "")))
        family = str(
            row.get(
                "quota_target_family",
                row.get("deformation_family", "identity"),
            )
        )
        key = ("identity", "identity") if bin_name == "identity" else (family, bin_name)
        if key in observed:
            observed[key] += 1

    missing = [index for index in range(final_count) if index not in existing]
    labels = [
        key
        for key, count in flat_targets.items()
        for _ in range(max(0, int(count) - observed[key]))
    ]
    while len(labels) < len(missing):
        key = min(
            flat_targets,
            key=lambda item: (observed[item] - flat_targets[item], item),
        )
        labels.append(key)
        observed[key] += 1
    rng = np.random.default_rng(_stable_seed(cfg.random_seed, "accepted-joint-plan"))
    rng.shuffle(labels)
    assignments = {}
    for index, (family, bin_name) in zip(missing, labels[: len(missing)]):
        assignments[index] = (
            bin_name,
            None if family == "identity" else family,
        )
    return assignments, targets


def _plan_accepted_anatomy_joint_assignments(cfg, template_ids, final_count, existing):
    """Assign every missing index an immutable anatomy/family/bin quota cell."""

    targets = _accepted_anatomy_joint_quota_targets(cfg, template_ids, final_count)
    flat_targets = targets["joint_anatomy_family_bin"]
    observed = {key: 0 for key in flat_targets}
    for index, row in existing.items():
        if index >= final_count:
            continue
        anatomy = str(row.get("quota_target_template_id", row.get("template_id", "")))
        bin_name = str(row.get("quota_target_disp_bin", row.get("target_disp_bin", "")))
        family = str(
            row.get(
                "quota_target_family",
                row.get("deformation_family", "identity"),
            )
        )
        key = (
            (anatomy, "identity", "identity")
            if bin_name == "identity"
            else (anatomy, family, bin_name)
        )
        if key not in observed:
            raise RuntimeError(
                "Existing checkpoint contains a sample outside the current "
                f"anatomy/family/bin quota plan: {key}"
            )
        observed[key] += 1

    nominal_overfilled = {
        key: {"target": int(flat_targets[key]), "existing": int(value)}
        for key, value in observed.items()
        if value > int(flat_targets[key])
    }

    # Fine cells may be above a newly computed nominal allocation after a
    # configuration/template calibration change. Preserve every accepted
    # sample and reallocate only the remaining capacity. Exact coarser margins
    # (anatomy, anatomy x family, and global family x bin) remain invariant.
    if nominal_overfilled:
        observed_anatomy = {
            anatomy: sum(
                value
                for (candidate, _family, _bin_name), value in observed.items()
                if candidate == anatomy
            )
            for anatomy in targets["anatomies"]
        }
        observed_anatomy_family = {
            key: sum(
                value
                for (candidate, family, _bin_name), value in observed.items()
                if (candidate, family) == key
            )
            for key in targets["joint_anatomy_family"]
        }
        global_cell_targets = {
            ("identity", "identity"): int(targets["identity"]),
            **{key: int(value) for key, value in targets["joint"].items()},
        }
        observed_global_cells = {
            key: sum(
                value
                for (_anatomy, family, bin_name), value in observed.items()
                if (family, bin_name) == key
            )
            for key in global_cell_targets
        }
        impossible = {
            "anatomies": {
                key: (observed_anatomy[key], int(target))
                for key, target in targets["anatomies"].items()
                if observed_anatomy[key] > int(target)
            },
            "anatomy_families": {
                key: (observed_anatomy_family[key], int(target))
                for key, target in targets["joint_anatomy_family"].items()
                if observed_anatomy_family[key] > int(target)
            },
            "family_bins": {
                key: (observed_global_cells[key], int(target))
                for key, target in global_cell_targets.items()
                if observed_global_cells[key] > int(target)
            },
        }
        impossible = {key: value for key, value in impossible.items() if value}
        if impossible:
            raise RuntimeError(
                "Existing checkpoint is already overfilled at a required quota "
                f"margin and cannot reach the requested distribution: {impossible}"
            )

        effective_targets = dict(flat_targets)
        for anatomy in template_ids:
            key = (anatomy, "identity", "identity")
            effective_targets[key] = int(
                targets["joint_anatomy_family"][(anatomy, "identity")]
            )
        for family in targets["families"]:
            bins = [
                bin_name
                for candidate_family, bin_name in targets["joint"]
                if candidate_family == family
            ]
            row_remaining = {
                anatomy: int(
                    targets["joint_anatomy_family"][(anatomy, family)]
                    - observed_anatomy_family[(anatomy, family)]
                )
                for anatomy in template_ids
            }
            col_remaining = {
                bin_name: int(
                    targets["joint"][(family, bin_name)]
                    - observed_global_cells[(family, bin_name)]
                )
                for bin_name in bins
            }
            residual = _difficulty_aware_anatomy_bin_counts(
                cfg, row_remaining, col_remaining
            )
            for anatomy in template_ids:
                for bin_name in bins:
                    key = (anatomy, family, bin_name)
                    effective_targets[key] = int(
                        observed[key] + residual[(anatomy, bin_name)]
                    )
        flat_targets = effective_targets
        targets = {
            **targets,
            "joint_anatomy_family_bin": effective_targets,
        }
        print(
            "Resume quota adaptation: preserved existing samples and "
            f"rebalanced {len(nominal_overfilled)} overfilled fine cells "
            "across the remaining anatomy/family/bin assignments."
        )

    missing = [index for index in range(final_count) if index not in existing]
    labels = [
        key
        for key, count in flat_targets.items()
        for _ in range(int(count) - observed[key])
    ]
    if len(labels) != len(missing):
        raise RuntimeError("Anatomy/family/bin quota deficit does not match holes")
    rng = np.random.default_rng(
        _stable_seed(cfg.random_seed, "accepted-anatomy-joint-plan")
    )
    rng.shuffle(labels)
    assignments = {}
    for index, (anatomy, family, bin_name) in zip(missing, labels):
        assignments[index] = (
            anatomy,
            bin_name,
            None if family == "identity" else family,
        )
    return assignments, targets


def _plan_accepted_bin_assignments(cfg, final_count, existing):
    """Assign each missing index one final-output amplitude bin.

    The accepted samples, rather than the attempted proposals, are balanced.
    Existing checkpoint rows count toward the target so a resume preserves the
    same contract whenever possible.
    """

    targets = _accepted_bin_quota_targets(cfg, final_count)
    observed = {label: 0 for label in targets}
    for index, row in existing.items():
        if index >= final_count:
            continue
        label = str(row.get("quota_target_disp_bin", row.get("target_disp_bin", "")))
        if label in observed:
            observed[label] += 1
    missing = [index for index in range(final_count) if index not in existing]
    labels = [
        label
        for label, count in targets.items()
        for _ in range(max(0, count - observed[label]))
    ]
    # A resume created by an older policy can already overfill one or more
    # bins. Fill the remaining slots with the currently most underfilled bin;
    # the final audit below enforces the configured tolerance transparently.
    while len(labels) < len(missing):
        label = min(
            targets,
            key=lambda name: (observed[name] - targets[name], name),
        )
        labels.append(label)
        observed[label] += 1
    rng = np.random.default_rng(_stable_seed(cfg.random_seed, "accepted-bin-plan"))
    rng.shuffle(labels)
    return dict(zip(missing, labels[: len(missing)])), targets


def _audit_accepted_bin_quotas(cfg, dataframe, targets):
    labels = list(targets)
    counts = {
        label: int(
            np.sum(
                dataframe.get(
                    "quota_target_disp_bin", dataframe["target_disp_bin"]
                ).astype(str)
                == label
            )
        )
        for label in labels
    }
    total = max(len(dataframe), 1)
    deviations = {
        label: 100.0 * (counts[label] - targets[label]) / total for label in labels
    }
    tolerance = float(cfg.displacement.accepted_quota_tolerance_percentage_points)
    violations = {
        label: value
        for label, value in deviations.items()
        if abs(value) > tolerance + 1.0e-12
    }
    if violations:
        raise RuntimeError(
            "Accepted displacement-bin quotas exceed the configured "
            f"±{tolerance:g} percentage-point tolerance: {violations}"
        )
    return {
        "targets": targets,
        "accepted_counts": counts,
        "deviation_percentage_points": deviations,
        "tolerance_percentage_points": tolerance,
    }


def _audit_accepted_family_quotas(cfg, dataframe, targets):
    deformed = dataframe[dataframe["deformation_family"] != "identity"]
    counts = {
        family: int(np.sum(deformed["deformation_family"].astype(str) == family))
        for family in targets
    }
    total = max(len(deformed), 1)
    deviations = {
        family: 100.0 * (counts[family] - targets[family]) / total for family in targets
    }
    tolerance = float(cfg.displacement.accepted_quota_tolerance_percentage_points)
    violations = {
        family: value
        for family, value in deviations.items()
        if abs(value) > tolerance + 1.0e-12
    }
    if violations:
        raise RuntimeError(
            "Accepted deformation-family quotas exceed the configured "
            f"±{tolerance:g} percentage-point tolerance: {violations}"
        )
    return {
        "targets": targets,
        "accepted_counts": counts,
        "deviation_percentage_points": deviations,
        "tolerance_percentage_points": tolerance,
    }


def _audit_accepted_joint_quotas(dataframe, targets):
    counts = {
        (family, bin_name): int(
            np.sum(
                (dataframe["deformation_family"].astype(str) == family)
                & (dataframe["target_disp_bin"].astype(str) == bin_name)
            )
        )
        for family, bin_name in targets
    }
    mismatches = {
        f"{family}/{bin_name}": {
            "target": int(targets[(family, bin_name)]),
            "accepted": int(counts[(family, bin_name)]),
        }
        for family, bin_name in targets
        if counts[(family, bin_name)] != int(targets[(family, bin_name)])
    }
    if mismatches:
        raise RuntimeError(f"Accepted joint family/bin quotas differ: {mismatches}")
    return {
        family: {
            bin_name: {
                "target": int(targets[(family, bin_name)]),
                "accepted": int(counts[(family, bin_name)]),
            }
            for candidate_family, bin_name in targets
            if candidate_family == family
        }
        for family in dict.fromkeys(family for family, _ in targets)
    }


def _audit_accepted_anatomy_quotas(dataframe, targets):
    counts = {
        anatomy: int(np.sum(dataframe["template_id"].astype(str) == anatomy))
        for anatomy in targets
    }
    mismatches = {
        anatomy: {"target": int(targets[anatomy]), "accepted": counts[anatomy]}
        for anatomy in targets
        if counts[anatomy] != int(targets[anatomy])
    }
    if mismatches:
        raise RuntimeError(f"Accepted anatomy quotas differ: {mismatches}")
    return {
        anatomy: {"target": int(targets[anatomy]), "accepted": counts[anatomy]}
        for anatomy in targets
    }


def _audit_accepted_anatomy_family_quotas(dataframe, targets):
    counts = {
        (anatomy, family): int(
            np.sum(
                (dataframe["template_id"].astype(str) == anatomy)
                & (dataframe["deformation_family"].astype(str) == family)
            )
        )
        for anatomy, family in targets
    }
    mismatches = {
        f"{anatomy}/{family}": {
            "target": int(targets[(anatomy, family)]),
            "accepted": counts[(anatomy, family)],
        }
        for anatomy, family in targets
        if counts[(anatomy, family)] != int(targets[(anatomy, family)])
    }
    if mismatches:
        raise RuntimeError(f"Accepted anatomy/family quotas differ: {mismatches}")
    return {
        anatomy: {
            family: {
                "target": int(targets[(anatomy, family)]),
                "accepted": counts[(anatomy, family)],
            }
            for candidate_anatomy, family in targets
            if candidate_anatomy == anatomy
        }
        for anatomy in dict.fromkeys(anatomy for anatomy, _ in targets)
    }


def _audit_accepted_anatomy_family_bin_quotas(dataframe, targets):
    counts = {
        (anatomy, family, bin_name): int(
            np.sum(
                (dataframe["template_id"].astype(str) == anatomy)
                & (dataframe["deformation_family"].astype(str) == family)
                & (dataframe["target_disp_bin"].astype(str) == bin_name)
            )
        )
        for anatomy, family, bin_name in targets
    }
    mismatches = {
        f"{anatomy}/{family}/{bin_name}": {
            "target": int(targets[(anatomy, family, bin_name)]),
            "accepted": counts[(anatomy, family, bin_name)],
        }
        for anatomy, family, bin_name in targets
        if counts[(anatomy, family, bin_name)]
        != int(targets[(anatomy, family, bin_name)])
    }
    if mismatches:
        raise RuntimeError(f"Accepted anatomy/family/bin quotas differ: {mismatches}")
    return {
        anatomy: {
            family: {
                bin_name: {
                    "target": int(targets[(anatomy, family, bin_name)]),
                    "accepted": counts[(anatomy, family, bin_name)],
                }
                for candidate_anatomy, candidate_family, bin_name in targets
                if candidate_anatomy == anatomy and candidate_family == family
            }
            for candidate_anatomy, family, _bin_name in targets
            if candidate_anatomy == anatomy
        }
        for anatomy in dict.fromkeys(anatomy for anatomy, _family, _bin_name in targets)
    }


def _map_surface_weights_to_volume(mapping, surface_weights):
    surface_weights = np.asarray(surface_weights, dtype=float)
    selected = surface_weights > 0
    if not np.any(selected):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=float)
    volume_nodes, inverse = np.unique(
        np.asarray(mapping)[selected], return_inverse=True
    )
    output = np.zeros(len(volume_nodes), dtype=float)
    np.maximum.at(output, inverse, surface_weights[selected])
    return volume_nodes.astype(np.int32), output


def _surface_reference_condition_map(volume_points, volume_cells, mapping):
    """Worst reference-tetrahedron condition incident to each surface node."""

    points = np.asarray(volume_points, dtype=float)
    cells = np.asarray(volume_cells, dtype=np.int64)
    matrices = np.stack(
        [
            points[cells[:, 1]] - points[cells[:, 0]],
            points[cells[:, 2]] - points[cells[:, 0]],
            points[cells[:, 3]] - points[cells[:, 0]],
        ],
        axis=2,
    )
    condition = np.asarray(np.linalg.cond(matrices), dtype=float)
    finite = condition[np.isfinite(condition)]
    replacement = float(finite.max()) if len(finite) else 1.0e12
    condition = np.nan_to_num(
        condition, nan=replacement, posinf=replacement, neginf=replacement
    )
    vertex_condition = np.zeros(len(points), dtype=float)
    for corner in range(4):
        np.maximum.at(vertex_condition, cells[:, corner], condition)
    return vertex_condition[np.asarray(mapping, dtype=np.int64)]


def _validate_metrics(cfg, intent, surface_metrics, volume_metrics):
    validity = cfg.validity
    combined = {**surface_metrics, **volume_metrics}
    numeric = [
        value
        for value in combined.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    reasons = []
    if not np.all(np.isfinite(np.asarray(numeric, dtype=float))):
        reasons.append("non_finite_metric")
    if volume_metrics["jacobian_min"] < validity.min_jacobian:
        reasons.append("jacobian_min")
    if volume_metrics["jacobian_max"] > validity.max_jacobian:
        reasons.append("jacobian_max")
    if volume_metrics["inverted_fraction"] > validity.max_inverted_fraction:
        reasons.append("inverted_tetrahedra")
    if not (
        validity.min_global_volume_ratio
        <= volume_metrics["global_volume_ratio"]
        <= validity.max_global_volume_ratio
    ):
        reasons.append("global_volume_ratio")
    if surface_metrics["edge_stretch_min"] < validity.min_edge_stretch_ratio:
        reasons.append("edge_stretch_min")
    if surface_metrics["edge_stretch_max"] > validity.max_edge_stretch_ratio:
        reasons.append("edge_stretch_max")
    if volume_metrics["green_lagrange_abs_p95"] > validity.max_green_lagrange_abs_p95:
        reasons.append("strain_p95")
    if volume_metrics["green_lagrange_abs_p99"] > validity.max_green_lagrange_abs_p99:
        reasons.append("strain_p99")
    if volume_metrics["green_lagrange_abs_max"] > validity.max_green_lagrange_abs_max:
        reasons.append("strain_max")
    if surface_metrics["edge_jump_p95_to_peak"] > validity.max_edge_jump_to_peak_ratio:
        reasons.append("surface_jump")
    if surface_metrics["bc_mean_disp_mm"] > validity.max_bc_mean_displacement_mm:
        reasons.append("hard_bc_motion")
    if (
        validity.require_peak_inside_support
        and intent.family != "lobe_motion"
        and not surface_metrics["peak_inside_load_patch"]
    ):
        reasons.append("peak_outside_support")
    if (
        validity.require_single_high_displacement_component
        and intent.family != "identity"
        and surface_metrics["high_disp_components"] != 1
    ):
        reasons.append("multiple_high_displacement_components")
    spread_bounds = validity.family_spread_bounds.get(intent.family)
    if spread_bounds:
        active_min, active_max, halfmax_min, peak_to_mean_max = map(
            float, spread_bounds
        )
        active = float(surface_metrics["active_area_fraction_rel20"])
        halfmax = float(surface_metrics["halfmax_area_fraction"])
        concentration = float(surface_metrics["peak_to_mean"])
        if active < active_min:
            reasons.append("spread_too_local")
        if active > active_max:
            reasons.append("spread_too_global")
        if halfmax < halfmax_min:
            reasons.append("halfmax_area_too_small")
        if concentration > peak_to_mean_max:
            reasons.append("peak_too_concentrated")
    bounds = getattr(cfg, "principal_stretch_screen", None)
    if bounds and (
        volume_metrics["principal_stretch_min"] < bounds[0]
        or volume_metrics["principal_stretch_max"] > bounds[1]
    ):
        reasons.append("principal_stretch_screen")
    if reasons:
        raise ValidityError(reasons, combined)


_AMPLITUDE_SENSITIVE_REJECTIONS = {
    "principal_stretch_screen",
    "jacobian_min",
    "jacobian_max",
    "inverted_tetrahedra",
    "global_volume_ratio",
    "edge_stretch_min",
    "edge_stretch_max",
    "strain_p95",
    "strain_p99",
    "strain_max",
}


def _evaluate_solve_result(
    cfg,
    intent,
    solve_result,
    mapping,
    surface,
    faces,
    load,
    anchors,
    solver,
    surface_metric_context,
    volume_metric_context,
):
    surface_displacement = solve_result.displacement[mapping]
    surface_metrics = surface_deformation_metrics(
        surface.points,
        faces,
        surface_displacement,
        load.load_weight,
        anchors.hard_mask,
        context=surface_metric_context,
    )
    surface_metrics.update(
        {
            "load_support_area_fraction": load.support_area_fraction,
            "target_load_area_fraction": load.target_area_fraction,
            "hard_anchor_area_fraction": anchors.hard_area_fraction,
            "soft_anchor_area_fraction": anchors.soft_area_fraction,
            "soft_anchor_effective_area_fraction": (
                anchors.soft_effective_area_fraction
            ),
            "target_hard_anchor_area_fraction": (anchors.target_hard_area_fraction),
            "target_soft_anchor_area_fraction": (anchors.target_soft_area_fraction),
        }
    )
    volume_metrics = volume_deformation_metrics(
        solver.volume_points,
        solver.volume_cells,
        solve_result.displacement,
        context=volume_metric_context,
    )
    _validate_metrics(cfg, intent, surface_metrics, volume_metrics)
    if cfg.validity.strict_surface_integrity:
        from DatasetFactory.ScientificPipelineShared.integrity import (
            quality,
            surface_failures,
        )

        audit = quality(surface.points + surface_displacement, faces)
        failures = surface_failures(audit)
        if failures:
            raise ValidityError(
                ["surface_integrity:" + x for x in failures], surface_metrics
            )
    return surface_displacement, surface_metrics, volume_metrics


def _capacity_adjusted_result(
    cfg,
    intent,
    solve_result,
    initial_error,
    evaluator,
):
    """Reuse one linear FEM response to find the largest valid target in-bin."""

    calibration = cfg.anatomy_calibration
    if (
        not calibration.enabled
        or not calibration.capacity_adaptation_enabled
        or not initial_error.reasons
        or not set(initial_error.reasons).issubset(_AMPLITUDE_SENSITIVE_REJECTIONS)
    ):
        raise initial_error
    requested = float(intent.target_max_mm)
    lower = float(intent.target_bin_lower_mm)
    upper = float(intent.target_bin_upper_mm)
    minimum = lower + float(calibration.capacity_min_bin_fraction) * (upper - lower)
    if requested <= minimum + 1.0e-9 or requested <= 0:
        raise initial_error

    def scaled(candidate_target):
        factor = float(candidate_target) / requested
        candidate_solve = SolveResult(
            displacement=solve_result.displacement * factor,
            traction_magnitude=solve_result.traction_magnitude * factor,
            linear_scale=solve_result.linear_scale * factor,
            attempts=solve_result.attempts,
        )
        candidate_intent = replace(intent, target_max_mm=float(candidate_target))
        metrics = evaluator(candidate_intent, candidate_solve)
        return candidate_intent, candidate_solve, metrics

    try:
        best = scaled(minimum)
    except ValidityError:
        raise initial_error
    low, high = minimum, requested
    for _ in range(calibration.capacity_bisection_steps):
        middle = 0.5 * (low + high)
        try:
            candidate = scaled(middle)
        except ValidityError:
            high = middle
        else:
            low = middle
            best = candidate
    return best


def _init_worker(cfg, template_specs):
    global _WORKER_CACHE, _WORKER_SPECS, _WORKER_CFG
    _WORKER_CACHE = {}
    _WORKER_SPECS = {spec[0]: spec for spec in template_specs}
    _WORKER_CFG = cfg


def _get_worker_template(template_id):
    global _WORKER_CACHE
    if template_id in _WORKER_CACHE:
        return _WORKER_CACHE[template_id]

    _WORKER_CACHE.clear()
    spec = _WORKER_SPECS[template_id]
    (
        template_id,
        surface_path,
        volume_path,
        mesh_name,
        anatomical_paths,
        template_split,
        young,
        poisson,
    ) = spec
    surface = read_surface(surface_path)
    diagonal = bbox_diag(surface.points)
    graph = edge_graph(surface)
    solver = LinearElasticSolver(
        volume_path,
        mesh_name,
        young,
        poisson,
        _WORKER_CFG.solver.petsc_options,
        require_serial=_WORKER_CFG.solver.require_serial,
    )
    mapping = map_surface_to_volume(
        surface.points,
        solver.volume_points,
        _WORKER_CFG.solver.mapping_tolerance_ratio * diagonal,
    )
    faces = triangle_array(surface)
    anatomy_support = prepare_anatomy_support(
        surface.points, anatomical_paths, faces=faces
    )
    surface_center_quality = _surface_reference_condition_map(
        solver.volume_points, solver.volume_cells, mapping
    )
    anatomy_support = replace(
        anatomy_support, surface_center_quality=surface_center_quality
    )
    surface_metric_context = prepare_surface_metric_context(surface.points, faces)
    volume_metric_context = prepare_volume_metric_context(
        solver.volume_points, solver.volume_cells
    )
    diagnostics = {
        **solver.mapping_diagnostics(),
        **reference_tetrahedron_metrics(solver.volume_points, solver.volume_cells),
        "surface_vertices": int(surface.n_points),
        "surface_faces": int(surface.n_cells),
        "surface_open_edges": int(surface.n_open_edges),
        "surface_to_volume_unique_fraction": float(
            len(np.unique(mapping)) / max(len(mapping), 1)
        ),
        "bbox_diagonal_mm": float(diagonal),
        "surface_area_mm2": float(anatomy_support.surface_area),
        "surface_enclosed_volume_mm3": float(anatomy_support.enclosed_volume),
        "characteristic_length_mm": float(anatomy_support.characteristic_length),
        "amplitude_scale": anatomy_amplitude_scale(
            _WORKER_CFG, anatomy_support.characteristic_length
        ),
        "surface_incident_condition_p50": float(
            np.quantile(surface_center_quality, 0.50)
        ),
        "surface_incident_condition_p80": float(
            np.quantile(surface_center_quality, 0.80)
        ),
        "surface_incident_condition_p95": float(
            np.quantile(surface_center_quality, 0.95)
        ),
        "quota_anatomy_difficulty_score": float(
            _WORKER_CFG.quotas.anatomy_difficulty_scores.get(template_id, 1.0)
        ),
    }
    if diagnostics["surface_open_edges"] != 0:
        raise ValueError(
            f"{template_id}: liver surface is not closed "
            f"({diagnostics['surface_open_edges']} open edges)"
        )
    if diagnostics["reference_degenerate_fraction"] > 0:
        raise ValueError(
            f"{template_id}: reference tetrahedral mesh contains degenerate cells"
        )
    loaded = (
        surface,
        diagonal,
        graph,
        solver,
        mapping,
        anatomical_paths,
        template_split,
        young,
        poisson,
        diagnostics,
        anatomy_support,
        faces,
        surface_metric_context,
        volume_metric_context,
    )
    _WORKER_CACHE[template_id] = loaded
    return loaded


def _intent_metadata(intent: SampleIntent):
    return {
        "split": intent.split,
        "is_ood": intent.is_ood,
        "deformation_family": intent.family,
        "mechanism": intent.mechanism,
        "load_sign": intent.sign,
        "direction_mode": intent.direction_mode,
        "target_disp_bin": intent.target_bin,
        "target_max_disp_mm": intent.target_max_mm,
        "normalized_target_max_disp_mm": (
            intent.target_max_mm / max(intent.amplitude_scale, 1.0e-12)
        ),
        "target_bin_lower_mm": intent.target_bin_lower_mm,
        "target_bin_upper_mm": intent.target_bin_upper_mm,
        "anatomy_amplitude_scale": intent.amplitude_scale,
        "target_load_area_fraction": intent.target_load_area_fraction,
        "profile": intent.profile,
        "planned_load_radius_ratio": intent.radius_ratio,
        "anchor_regime": intent.anchor_regime,
    }


def _base_metadata(cfg, index, template_id, intent, young, poisson):
    return {
        "sample_index": index,
        "sample_id": f"{cfg.sample_prefix}_{index:0{cfg.sample_id_width}d}",
        "template_id": template_id,
        **_intent_metadata(intent),
        "young_modulus": young,
        "poisson_ratio": poisson,
    }


def _identity_result(
    cfg,
    index,
    template_id,
    intent,
    surface,
    solver,
    anchors,
    young,
    poisson,
    diagnostics,
    faces,
    surface_metric_context,
    volume_metric_context,
):
    surface_displacement = np.zeros((surface.n_points, 3), dtype=float)
    volume_displacement = np.zeros_like(solver.volume_points)
    load_weight = np.zeros(surface.n_points, dtype=np.float32)
    geodesic = np.full(surface.n_points, -1.0, dtype=np.float32)
    surface_metrics = surface_deformation_metrics(
        surface.points,
        faces,
        surface_displacement,
        load_weight,
        anchors.hard_mask,
        context=surface_metric_context,
    )
    volume_metrics = volume_deformation_metrics(
        solver.volume_points,
        solver.volume_cells,
        volume_displacement,
        context=volume_metric_context,
    )
    metadata = {
        **_base_metadata(cfg, index, template_id, intent, young, poisson),
        "load_radius_mm": 0.0,
        "load_center_vertex": -1,
        "force_direction": [0.0, 0.0, 0.0],
        "hard_anchor_radius_mm": anchors.hard_radius,
        "soft_anchor_radius_mm": anchors.soft_radius,
        "soft_anchor_relative_stiffness": anchors.soft_relative_stiffness,
        "anchor_released_fraction": anchors.released_fraction,
        "hard_anchor_area_fraction": anchors.hard_area_fraction,
        "soft_anchor_area_fraction": anchors.soft_area_fraction,
        "soft_anchor_effective_area_fraction": anchors.soft_effective_area_fraction,
        "traction_magnitude": 0.0,
        "linear_scale": 0.0,
        "generation_attempts": 0,
        **surface_metrics,
        **volume_metrics,
    }
    return {
        "meta": metadata,
        "displacement": surface_displacement,
        "volume_displacement": volume_displacement.astype(np.float32),
        "point_data": {
            "hard_bc_mask": anchors.hard_mask.astype(np.uint8),
            "soft_anchor_weight": anchors.soft_weight,
            "load_weight": load_weight,
            "geodesic_distance_to_load": geodesic,
        },
        "failures": [],
        "template_diagnostics": diagnostics,
    }


def _failure_record(index, template_id, attempt, intent, exc):
    row = {
        "sample_index": index,
        "attempt": attempt,
        "template_id": template_id,
        **_intent_metadata(intent),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if isinstance(exc, ValidityError):
        row["rejection_reasons"] = list(exc.reasons)
        row["metrics"] = exc.metrics
    return row


def _generate_one(
    task,
    _intent_resample_round=0,
    _locked_intent=None,
    _prior_failures=(),
    _sample_replacement_round=0,
    _forced_family=None,
):
    cfg, index, template_id, seed, *task_options = task
    initial_template_id = template_id
    requested_target_bin = task_options[0] if task_options else None
    requested_family = task_options[1] if len(task_options) > 1 else None
    # Anatomy, family and displacement bin are an immutable accepted-output
    # quota cell. Retries may change anchors, load placement and random draws,
    # but never move the sample to an easier patient.
    replacement_seed = (
        seed
        if _sample_replacement_round == 0
        else _stable_seed(seed, "sample_replacement", _sample_replacement_round)
    )
    proposal_seed = (
        replacement_seed
        if _intent_resample_round == 0
        else _stable_seed(replacement_seed, "intent_resample", _intent_resample_round)
    )
    (
        surface,
        diagonal,
        graph,
        solver,
        mapping,
        anatomical_paths,
        template_split,
        young,
        poisson,
        diagnostics,
        anatomy_support,
        faces,
        surface_metric_context,
        volume_metric_context,
    ) = _get_worker_template(template_id)

    plan_rng = np.random.default_rng(_stable_seed(proposal_seed, "intent"))
    if _locked_intent is None:
        split, is_ood = _sample_split(cfg, template_split, plan_rng)
        force_identity = (
            requested_target_bin == "identity"
            if requested_target_bin is not None
            else None
        )
        target_bin = (
            None if requested_target_bin in (None, "identity") else requested_target_bin
        )
        intent = sample_intent(
            cfg,
            plan_rng,
            diagonal=diagonal,
            split=split,
            is_ood=is_ood,
            force_identity=force_identity,
            family=_forced_family or requested_family,
            target_bin=target_bin,
            characteristic_length=anatomy_support.characteristic_length,
        )
    else:
        # A failed realization is re-sampled only within its original family
        # and amplitude bin.  This prevents acceptance bias from turning a
        # difficult large deformation into an easier small one.
        intent = sample_intent(
            cfg,
            plan_rng,
            diagonal=diagonal,
            split=_locked_intent.split,
            is_ood=_locked_intent.is_ood,
            force_identity=False,
            family=_locked_intent.family,
            target_bin=_locked_intent.target_bin,
            characteristic_length=anatomy_support.characteristic_length,
        )
        is_ood = intent.is_ood

    if intent.is_identity:
        rng = np.random.default_rng(_stable_seed(proposal_seed, "identity"))
        anchors = build_anchors(
            surface,
            anatomical_paths,
            cfg,
            rng,
            diagonal,
            is_ood,
            regime_name=intent.anchor_regime,
            anatomy_support=anatomy_support,
        )
        result = _identity_result(
            cfg,
            index,
            template_id,
            intent,
            surface,
            solver,
            anchors,
            young,
            poisson,
            diagnostics,
            faces,
            surface_metric_context,
            volume_metric_context,
        )
        result["meta"]["intent_resample_round"] = _intent_resample_round
        result["meta"]["initial_template_id"] = initial_template_id
        result["meta"]["anatomy_fallback_round"] = 0
        result["meta"]["sample_replacement_round"] = _sample_replacement_round
        result["meta"]["quota_target_template_id"] = initial_template_id
        result["meta"]["quota_target_disp_bin"] = (
            requested_target_bin or intent.target_bin
        )
        result["meta"]["quota_target_family"] = "identity"
        result["failures"] = [*_prior_failures, *result.get("failures", [])]
        return result

    failures = []
    max_attempts = cfg.validity.max_generation_attempts
    anchor_realizations = min(max_attempts, cfg.validity.max_anchor_realizations)
    loads_per_anchor = int(np.ceil(max_attempts / anchor_realizations))
    attempt = 0

    # The attachment state belongs to one anatomical realization.  We retry
    # load placement under that same state before moving to another plausible
    # attachment realization.  This preserves the complete strict validation
    # budget while avoiding a direct LU factorization for every failed load.
    for anchor_index in range(anchor_realizations):
        if attempt >= max_attempts:
            break
        prepared = None
        try:
            anchor_rng = np.random.default_rng(
                _stable_seed(proposal_seed, "anchors", anchor_index)
            )
            anchors = build_anchors(
                surface,
                anatomical_paths,
                cfg,
                anchor_rng,
                diagonal,
                is_ood,
                regime_name=intent.anchor_regime,
                anatomy_support=anatomy_support,
            )
            hard_nodes = np.unique(mapping[anchors.hard_mask])
            soft_nodes, soft_weights = _map_surface_weights_to_volume(
                mapping, anchors.soft_weight
            )
            prepared = solver.prepare_system(
                hard_nodes,
                soft_nodes,
                soft_weights,
                anchors.soft_relative_stiffness,
            )
        except Exception as exc:
            failures.append(_failure_record(index, template_id, attempt, intent, exc))
            attempt += 1
            continue

        try:
            for _ in range(min(loads_per_anchor, max_attempts - attempt)):
                current_attempt = attempt
                attempt += 1
                rng = np.random.default_rng(
                    _stable_seed(proposal_seed, "load", current_attempt)
                )
                try:
                    load = sample_deformation_load(
                        surface, graph, anchors, intent, cfg, rng, diagonal
                    )
                    load_nodes, load_weights = _map_surface_weights_to_volume(
                        mapping, load.load_weight
                    )
                    solve_result = solver.solve_prepared(
                        prepared=prepared,
                        load_nodes=load_nodes,
                        load_weights=load_weights,
                        direction=load.direction,
                        hard_nodes=hard_nodes,
                        soft_nodes=soft_nodes,
                        soft_weights=soft_weights,
                        soft_relative_stiffness=anchors.soft_relative_stiffness,
                        target_max=intent.target_max_mm,
                        initial_traction=cfg.displacement.initial_traction,
                        target_nodes=mapping,
                    )
                    requested_target = float(intent.target_max_mm)

                    def evaluator(candidate_intent, candidate_solve):
                        return _evaluate_solve_result(
                            cfg,
                            candidate_intent,
                            candidate_solve,
                            mapping,
                            surface,
                            faces,
                            load,
                            anchors,
                            solver,
                            surface_metric_context,
                            volume_metric_context,
                        )

                    capacity_adjusted = False
                    try:
                        (
                            surface_displacement,
                            surface_metrics,
                            volume_metrics,
                        ) = evaluator(intent, solve_result)
                    except ValidityError as initial_error:
                        intent, solve_result, evaluated = _capacity_adjusted_result(
                            cfg,
                            intent,
                            solve_result,
                            initial_error,
                            evaluator,
                        )
                        (
                            surface_displacement,
                            surface_metrics,
                            volume_metrics,
                        ) = evaluated
                        capacity_adjusted = True

                    metadata = {
                        **_base_metadata(
                            cfg, index, template_id, intent, young, poisson
                        ),
                        "load_radius_mm": load.radius,
                        "load_radius_ratio": load.radius / diagonal,
                        "load_support_area_fraction": load.support_area_fraction,
                        "target_load_area_fraction": load.target_area_fraction,
                        "load_center_vertex": load.center_vertex,
                        "load_center_reference_condition": (
                            load.center_reference_quality
                        ),
                        "load_center_condition_threshold": (
                            load.center_quality_threshold
                        ),
                        "force_direction": load.direction.tolist(),
                        "hard_anchor_radius_mm": anchors.hard_radius,
                        "soft_anchor_radius_mm": anchors.soft_radius,
                        "soft_anchor_relative_stiffness": anchors.soft_relative_stiffness,
                        "anchor_released_fraction": anchors.released_fraction,
                        "hard_anchor_area_fraction": anchors.hard_area_fraction,
                        "soft_anchor_area_fraction": anchors.soft_area_fraction,
                        "soft_anchor_effective_area_fraction": (
                            anchors.soft_effective_area_fraction
                        ),
                        "target_hard_anchor_area_fraction": (
                            anchors.target_hard_area_fraction
                        ),
                        "target_soft_anchor_area_fraction": (
                            anchors.target_soft_area_fraction
                        ),
                        "anatomy_characteristic_length_mm": (
                            anatomy_support.characteristic_length
                        ),
                        "requested_target_max_disp_mm": requested_target,
                        "capacity_adjusted": capacity_adjusted,
                        "capacity_scale": (
                            float(intent.target_max_mm) / requested_target
                        ),
                        "traction_magnitude": solve_result.traction_magnitude,
                        "linear_scale": solve_result.linear_scale,
                        "generation_attempts": current_attempt + 1,
                        "intent_resample_round": _intent_resample_round,
                        "initial_template_id": initial_template_id,
                        "anatomy_fallback_round": 0,
                        "sample_replacement_round": _sample_replacement_round,
                        "quota_target_template_id": initial_template_id,
                        "quota_target_disp_bin": (
                            requested_target_bin or intent.target_bin
                        ),
                        "quota_target_family": (requested_family or intent.family),
                        **surface_metrics,
                        **volume_metrics,
                    }
                    return {
                        "meta": metadata,
                        "displacement": surface_displacement,
                        "volume_displacement": np.asarray(
                            solve_result.displacement, dtype=np.float32
                        ),
                        "point_data": {
                            "hard_bc_mask": anchors.hard_mask.astype(np.uint8),
                            "soft_anchor_weight": anchors.soft_weight,
                            "load_weight": load.load_weight.astype(np.float32),
                            "geodesic_distance_to_load": load.geodesic_distance.astype(
                                np.float32
                            ),
                        },
                        "failures": [*_prior_failures, *failures],
                        "template_diagnostics": diagnostics,
                    }
                except Exception as exc:
                    failures.append(
                        _failure_record(
                            index, template_id, current_attempt, intent, exc
                        )
                    )
        finally:
            solver.release_system(prepared)

    if _intent_resample_round < cfg.validity.max_intent_resamples:
        # Draw another boundary/load realization of the same requested family
        # and amplitude bin.  Validity thresholds are never relaxed.
        return _generate_one(
            task,
            _intent_resample_round + 1,
            _locked_intent=intent,
            _prior_failures=(*_prior_failures, *failures),
            _sample_replacement_round=_sample_replacement_round,
            _forced_family=_forced_family,
        )

    all_failures = (*_prior_failures, *failures)
    if _sample_replacement_round < cfg.validity.max_sample_replacements:
        # Start a fresh bounded realization while keeping the entire accepted
        # anatomy/family/bin quota cell locked. Validity is never relaxed.
        next_family = requested_family or intent.family
        return _generate_one(
            task,
            _intent_resample_round=0,
            _locked_intent=None,
            _prior_failures=all_failures,
            _sample_replacement_round=_sample_replacement_round + 1,
            _forced_family=next_family,
        )

    summary = failures[-1]["error"] if failures else "unknown error"
    raise SampleGenerationExhausted(
        f"Could not generate sample {index} after {len(all_failures)} screened proposals "
        f"for quota cell {initial_template_id}/{intent.family}/{intent.target_bin}: "
        f"{summary}"
    )


def _read_existing_rows(
    output, prefix, id_width, *, require_volume_displacement: bool = False
):
    jsonl_path = output / "json" / "mesh_manifest.jsonl"
    legacy_jsonl_path = output / "logs" / "generation_index.jsonl"
    csv_path = output / "logs" / "generation_index.csv"
    candidates = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    elif legacy_jsonl_path.exists():
        for line in legacy_jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    elif csv_path.exists():
        try:
            candidates = pd.read_csv(csv_path).to_dict("records")
        except pd.errors.EmptyDataError:
            candidates = []

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.(?:npz|vtp)$")
    by_index = {}
    for row in candidates:
        try:
            index = int(row["sample_index"])
        except (KeyError, TypeError, ValueError):
            continue
        template_id = str(row.get("template_id", row.get("anatomy_id", "")))
        default_path = (
            output / template_id / "meshes" / f"{prefix}_{index:0{id_width}d}.npz"
        )
        legacy_path = output / "meshes" / f"{prefix}_{index:0{id_width}d}.vtp"
        row_path = Path(str(row.get("path", row.get("mesh_path", default_path))))
        if not row_path.exists():
            row_path = default_path if default_path.exists() else legacy_path
        if not row_path.exists():
            continue
        if require_volume_displacement and row_path.suffix == ".npz":
            try:
                with np.load(row_path, allow_pickle=False) as archive:
                    if "volume_displacement" not in archive.files:
                        continue
            except (OSError, ValueError):
                continue
        if pattern.match(row_path.name):
            row["path"] = str(row_path.resolve())
            row["mesh_path"] = str(row_path.resolve())
            row["template_id"] = template_id
            by_index[index] = row

    # A hard interruption can happen after the atomic NPZ rename and before
    # its manifest line is appended. Recover those valid orphan checkpoints so
    # resume never throws away already completed FEM work.
    for row_path in sorted(output.glob("*/meshes/*.npz")):
        match = pattern.match(row_path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if index in by_index:
            continue
        try:
            with np.load(row_path, allow_pickle=False) as archive:
                if (
                    require_volume_displacement
                    and "volume_displacement" not in archive.files
                ):
                    continue
                metadata = json.loads(str(archive["metadata_json"].item()))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        metadata_index = int(metadata.get("sample_index", index))
        if metadata_index != index:
            continue
        template_id = str(metadata.get("template_id", row_path.parent.parent.name))
        metadata.update(
            {
                "sample_index": index,
                "template_id": template_id,
                "path": str(row_path.resolve()),
                "mesh_path": str(row_path.resolve()),
            }
        )
        by_index[index] = metadata
    return by_index


def _read_executor_recovery_rounds(output):
    """Return the next deterministic recovery round for failed sample ids."""

    path = output / "json" / "generation_failures.jsonl"
    rounds = {}
    if not path.exists():
        return rounds
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            index = int(row["sample_index"])
            recovery_round = int(row.get("executor_recovery_round", 0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        rounds[index] = max(rounds.get(index, 0), recovery_round + 1)
    return rounds


def _task_for_recovery_round(task, recovery_round):
    if recovery_round <= 0:
        return task
    cfg, index, template_id, seed, *options = task
    return (
        cfg,
        index,
        template_id,
        _stable_seed(seed, "executor-recovery", recovery_round),
        *options,
    )


def _validate_resume_topology(existing, parent_cache):
    """Refuse to mix samples made from different versions of one anatomy."""

    for index, row in existing.items():
        template_id = str(row.get("template_id", ""))
        if template_id not in parent_cache:
            continue
        path = Path(str(row["path"]))
        expected_points = np.asarray(parent_cache[template_id].points, dtype=np.float32)
        expected_triangles = triangle_array(parent_cache[template_id]).astype(np.int32)
        try:
            with np.load(path, allow_pickle=False) as archive:
                reference_points = np.asarray(archive["reference_points"])
                triangles = np.asarray(archive["triangles"])
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Checkpoint NPZ unreadable for sample {index}: {path}"
            ) from exc
        if (
            reference_points.shape != expected_points.shape
            or triangles.shape != expected_triangles.shape
            or not np.allclose(reference_points, expected_points, rtol=0.0, atol=1.0e-5)
            or not np.array_equal(triangles, expected_triangles)
        ):
            raise RuntimeError(
                "Resume topology mismatch: existing sample "
                f"{path} was not generated from the current {template_id} "
                "template. Use a clean output directory instead of mixing "
                "incompatible meshes."
            )


def _abort_executor(pool):
    """Cancel queued work and stop the bounded set of running workers."""

    processes = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2.0)


def _read_previous_template_diagnostics(output):
    manifest_path = output / "json" / "dataset.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    diagnostics = manifest.get("template_diagnostics", {})
    return diagnostics if isinstance(diagnostics, dict) else {}


def _prepare_output(cfg, resume):
    output = Path(cfg.out_dir)
    if not resume:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        existing = {}
    else:
        existing = _read_existing_rows(
            output,
            cfg.sample_prefix,
            cfg.sample_id_width,
            require_volume_displacement=cfg.solver.store_volume_displacement,
        )
    (output / "json").mkdir(parents=True, exist_ok=True)
    if resume and existing:
        # Deduplicate interrupted appends and make recovered orphan NPZ files
        # immediately visible to every later resume.
        _atomic_write_jsonl(
            output / "json" / "mesh_manifest.jsonl",
            [existing[index] for index in sorted(existing)],
        )
    return output, existing


def _atomic_write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _manifest_row(row, template_sources):
    output = dict(row)
    template_id = str(output["template_id"])
    mesh_id = str(output["sample_id"])
    mesh_path = str(Path(output["path"]).resolve())
    output.update(
        {
            "anatomy_id": template_id,
            "mesh_id": mesh_id,
            "mesh_path": mesh_path,
            "template_path": str(Path(template_sources[template_id]).resolve()),
            "mesh_format": "npz",
        }
    )
    return output


def _write_universal_manifest(output, dataframe, template_sources):
    rows = []
    for row in dataframe.to_dict("records"):
        rows.append(_manifest_row(row, template_sources))
    _atomic_write_jsonl(output / "json" / "mesh_manifest.jsonl", rows)


def _atomic_write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_anatomy_metadata(output, enabled, dataframe, diagnostics_by_template, cfg):
    for template in enabled:
        anatomy_root = output / template.template_id
        (anatomy_root / "meshes").mkdir(parents=True, exist_ok=True)
        json_dir = anatomy_root / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        group = dataframe[dataframe["template_id"] == template.template_id]
        payload = {
            "schema_version": 1,
            "anatomy_id": template.template_id,
            "split": template.split,
            "sample_count": int(len(group)),
            "source": {
                "simulation_surface_path": str(Path(template.surface_path).resolve()),
                "output_surface_path": str(
                    Path(
                        template.output_surface_path or template.surface_path
                    ).resolve()
                ),
                "surface_transfer_path": (
                    str(Path(template.surface_transfer_path).resolve())
                    if template.surface_transfer_path is not None
                    else None
                ),
                "volume_xdmf_path": str(Path(template.volume_xdmf_path).resolve()),
                "anatomical_paths": {
                    key: str(Path(value).resolve())
                    for key, value in template.anatomical_paths.items()
                },
            },
            "diagnostics": diagnostics_by_template.get(template.template_id, {}),
        }
        _atomic_write_json(json_dir / "anatomy.json", payload)
        if not cfg.solver.store_volume_displacement:
            continue
        # Materialize once while the FEM runtime is available. v4 can then
        # consume this immutable topology contract from a lightweight kernel.
        surface = read_surface(template.surface_path)
        simulation_points = np.asarray(surface.points, dtype=np.float32)
        simulation_triangles = triangle_array(surface).astype(np.int32)
        solver = LinearElasticSolver(
            template.volume_xdmf_path,
            template.xdmf_mesh_name,
            cfg.material.young_modulus,
            cfg.material.poisson_ratio,
            cfg.solver.petsc_options,
            require_serial=cfg.solver.require_serial,
        )
        try:
            diagonal = max(
                float(
                    np.linalg.norm(simulation_points.max(0) - simulation_points.min(0))
                ),
                1e-6,
            )
            mapping = map_surface_to_volume(
                simulation_points,
                solver.volume_points,
                cfg.solver.mapping_tolerance_ratio * diagonal,
            )
            output_surface = read_surface(
                template.output_surface_path or template.surface_path
            )
            arrays = {
                "volume_nodes": np.asarray(solver.volume_points, np.float32),
                "tetrahedra": np.asarray(solver.volume_cells, np.int32),
                "simulation_surface_to_volume_map": mapping.astype(np.int32),
                "simulation_surface_points": simulation_points,
                "simulation_surface_triangles": simulation_triangles,
                "output_surface_points": np.asarray(output_surface.points, np.float32),
                "output_surface_triangles": triangle_array(output_surface).astype(
                    np.int32
                ),
            }
            if template.surface_transfer_path:
                with np.load(
                    template.surface_transfer_path, allow_pickle=False
                ) as transfer:
                    arrays["output_surface_to_simulation_indices"] = np.asarray(
                        transfer["aligned_vertex_indices"], np.int32
                    )
                    arrays["output_surface_to_simulation_weights"] = np.asarray(
                        transfer["weights"], np.float32
                    )
            contract_npz = anatomy_root / "physics_contract.npz"
            temporary = contract_npz.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, contract_npz)
            contract = {
                "schema_version": 4,
                "anatomy_id": template.template_id,
                "arrays_path": contract_npz.name,
                "topology_chain": "simulation_surface -> FEM_volume -> optional_surface_transfer -> output_atlas_surface",
                "counts": {key: int(len(value)) for key, value in arrays.items()},
                "source_paths": payload["source"],
            }
            _atomic_write_json(anatomy_root / "physics_contract.json", contract)
        finally:
            del solver


def run_project(
    cfg_or_path: ProjectConfig | str,
    *,
    n_samples=None,
    resume=False,
):
    cfg = (
        load_config(cfg_or_path)
        if isinstance(cfg_or_path, (str, Path))
        else cfg_or_path
    )
    if not cfg.templates:
        raise ValueError("The recipe must specify at least one input template.")
    validate_config(cfg, check_paths=True)
    output, existing = _prepare_output(cfg, resume)

    enabled = [template for template in cfg.templates if template.enabled]
    final_count = cfg.n_samples if n_samples is None else int(n_samples)
    if final_count < 1:
        raise ValueError("n_samples must be positive")

    parent_cache = {}
    output_transfers = {}
    template_sources = {}
    specs = []
    characteristic_lengths = []
    discovered_difficulty_scores = {}
    for template in enabled:
        surface = read_surface(template.surface_path)
        output_surface = (
            read_surface(template.output_surface_path)
            if template.output_surface_path is not None
            else surface
        )
        output_transfers[template.template_id] = _load_surface_transfer(
            template.surface_transfer_path,
            surface.n_points,
            output_surface.n_points,
        )
        _, _, _, characteristic_length = surface_characteristics(
            surface.points, triangle_array(surface)
        )
        characteristic_lengths.append(characteristic_length)
        if template.difficulty_score is not None:
            discovered_difficulty_scores[template.template_id] = float(
                template.difficulty_score
            )
        else:
            report_path = (
                Path(template.surface_path).parent
                / cfg.discovery.preparation_report_filename
            )
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                discovered_difficulty_scores[template.template_id] = float(
                    report["volume_quality"]["edge_ratio_stats"]["q95"]
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                discovered_difficulty_scores[template.template_id] = 1.0
        anatomy_root = output / template.template_id
        (anatomy_root / "meshes").mkdir(parents=True, exist_ok=True)
        (anatomy_root / "json").mkdir(parents=True, exist_ok=True)
        template_sources[template.template_id] = (
            template.output_surface_path or template.surface_path
        )
        material = cfg.material
        material_rng = np.random.default_rng(
            _stable_seed(cfg.random_seed, "material", template.template_id)
        )
        if material.randomize_per_template:
            young = material.young_modulus * np.exp(
                material_rng.normal(0.0, material.young_modulus_log_std)
            )
            poisson = material_rng.uniform(
                material.poisson_ratio_min, material.poisson_ratio_max
            )
        else:
            young = material.young_modulus
            poisson = material.poisson_ratio
        parent_cache[template.template_id] = output_surface
        specs.append(
            (
                template.template_id,
                template.surface_path,
                template.volume_xdmf_path,
                template.xdmf_mesh_name,
                template.anatomical_paths,
                template.split,
                young,
                poisson,
            )
        )

    calibration = cfg.anatomy_calibration
    if calibration.enabled and calibration.characteristic_length_reference_mm <= 0:
        calibration.characteristic_length_reference_mm = float(
            np.median(characteristic_lengths)
        )
    if not cfg.quotas.anatomy_difficulty_scores:
        cfg.quotas.anatomy_difficulty_scores = discovered_difficulty_scores

    if resume and existing:
        _validate_resume_topology(existing, parent_cache)

    missing_indices = [index for index in range(final_count) if index not in existing]
    joint_assignments, quota_targets = _plan_accepted_anatomy_joint_assignments(
        cfg,
        [template.template_id for template in enabled],
        final_count,
        existing,
    )
    tasks = [
        (
            cfg,
            index,
            joint_assignments[index][0],
            _stable_seed(cfg.random_seed, "sample", index),
            joint_assignments[index][1],
            joint_assignments[index][2],
        )
        for index in missing_indices
    ]
    template_order = {
        template.template_id: position for position, template in enumerate(enabled)
    }
    tasks.sort(key=lambda task: (template_order[task[2]], task[1]))

    accepted_checkpoint_count = final_count - len(missing_indices)
    if resume:
        print(
            "Resume dataset: "
            f"{accepted_checkpoint_count}/{final_count} samples already accepted; "
            f"{len(missing_indices)} to generate."
        )

    # Preserve diagnostics on a completed or partially resumed run. Each newly
    # loaded worker will replace its template entry with current diagnostics.
    diagnostics_by_template = _read_previous_template_diagnostics(output)
    if tasks:
        workers = min(max(1, int(cfg.n_workers)), len(tasks))
        _configure_native_worker_threads(cfg)
        context = mp.get_context("spawn")
        recovery_starts = _read_executor_recovery_rounds(output)
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(cfg, specs),
            max_tasks_per_child=int(cfg.max_tasks_per_worker),
        )
        task_iterator = iter(tasks)
        futures = {}

        def submit(task, recovery_round, retry_count):
            future = pool.submit(
                _generate_one,
                _task_for_recovery_round(task, recovery_round),
            )
            futures[future] = (task, recovery_round, retry_count)

        try:
            for _ in range(workers):
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                submit(task, recovery_starts.get(task[1], 0), 0)

            with tqdm(
                total=final_count,
                initial=accepted_checkpoint_count,
                desc="Generating physically screened meshes",
                unit="sample",
            ) as progress:
                while futures:
                    done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    fatal_error = None
                    for future in done:
                        task, recovery_round, retry_count = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            failure = {
                                "sample_index": int(task[1]),
                                "quota_target_template_id": str(task[2]),
                                "quota_target_disp_bin": str(task[4]),
                                "quota_target_family": str(task[5]),
                                "executor_recovery_round": int(recovery_round),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            append_jsonl(
                                output / "json" / "generation_failures.jsonl",
                                failure,
                            )
                            can_retry = not isinstance(
                                exc, BrokenProcessPool
                            ) and retry_count < int(cfg.executor_retries_per_sample)
                            if can_retry:
                                submit(task, recovery_round + 1, retry_count + 1)
                                progress.set_postfix_str(f"retry sample {task[1]}")
                            elif fatal_error is None:
                                fatal_error = RuntimeError(
                                    "Generation worker failed for immutable quota "
                                    f"cell {task[2]}/{task[5]}/{task[4]} at "
                                    f"sample {task[1]}. The checkpoint is intact; "
                                    "resume will use a fresh deterministic recovery "
                                    "seed. See json/generation_failures.jsonl."
                                )
                                fatal_error.__cause__ = exc
                            continue

                        metadata = result["meta"]
                        metadata["executor_recovery_round"] = int(recovery_round)
                        template_id = metadata["template_id"]
                        result = _transfer_result_to_output(
                            result, output_transfers[template_id]
                        )
                        metadata = result["meta"]
                        diagnostics_by_template[template_id] = result[
                            "template_diagnostics"
                        ]
                        surface = parent_cache[template_id]
                        metadata["rejection_history"] = result["failures"]
                        path = (
                            output
                            / template_id
                            / "meshes"
                            / f"{metadata['sample_id']}.npz"
                        )
                        write_sample(
                            path,
                            surface,
                            np.asarray(surface.points) + result["displacement"],
                            result["displacement"],
                            result["point_data"],
                            metadata,
                            extra_arrays=(
                                {"volume_displacement": result["volume_displacement"]}
                                if cfg.solver.store_volume_displacement
                                else None
                            ),
                        )
                        row = {**metadata, "path": str(path.resolve())}
                        manifest_row = _manifest_row(row, template_sources)
                        # The atomic NPZ plus this fsynced JSONL append are the
                        # durable per-sample checkpoint used by resume.
                        append_jsonl(
                            output / "json" / "mesh_manifest.jsonl",
                            manifest_row,
                        )
                        existing[int(metadata["sample_index"])] = manifest_row
                        progress.update(1)

                    if fatal_error is not None:
                        raise fatal_error

                    while len(futures) < workers:
                        try:
                            task = next(task_iterator)
                        except StopIteration:
                            break
                        submit(task, recovery_starts.get(task[1], 0), 0)
        except BaseException as exc:
            cause = getattr(exc, "__cause__", None)
            if isinstance(cause, SampleGenerationExhausted):
                # A scientifically rejected quota cell is not a PETSc crash.
                # Let currently running native solvers exit normally; forcibly
                # terminating them emits misleading PETSc/MPI signal-15 errors.
                pool.shutdown(wait=True, cancel_futures=True)
            else:
                _abort_executor(pool)
            raise
        else:
            pool.shutdown(wait=True)

    rows = [existing[index] for index in sorted(existing) if index < final_count]
    if len(rows) != final_count:
        missing = sorted(set(range(final_count)) - set(existing))
        raise RuntimeError(f"Generation incomplete; missing indices: {missing[:20]}")
    dataframe = pd.DataFrame(rows).sort_values("sample_index")
    accepted_bin_quota = _audit_accepted_bin_quotas(
        cfg, dataframe, quota_targets["bins"]
    )
    accepted_family_quota = _audit_accepted_family_quotas(
        cfg, dataframe, quota_targets["families"]
    )
    accepted_joint_quota = _audit_accepted_joint_quotas(
        dataframe, quota_targets["joint"]
    )
    if "quota_target_template_id" in dataframe:
        anatomy_drift = dataframe[
            dataframe["template_id"].astype(str)
            != dataframe["quota_target_template_id"].astype(str)
        ]
        if len(anatomy_drift):
            raise RuntimeError(
                "Accepted samples moved outside their planned anatomy quota: "
                + ", ".join(anatomy_drift["sample_id"].astype(str).head(20))
            )
    accepted_anatomy_quota = _audit_accepted_anatomy_quotas(
        dataframe, quota_targets["anatomies"]
    )
    accepted_anatomy_family_quota = _audit_accepted_anatomy_family_quotas(
        dataframe, quota_targets["joint_anatomy_family"]
    )
    accepted_anatomy_family_bin_quota = _audit_accepted_anatomy_family_bin_quotas(
        dataframe, quota_targets["joint_anatomy_family_bin"]
    )
    _write_universal_manifest(output, dataframe, template_sources)

    split_by_template = {
        template_id: sorted(
            set(dataframe.loc[dataframe["template_id"] == template_id, "split"])
        )
        for template_id in dataframe["template_id"].unique()
    }
    leakage = {
        template_id: splits
        for template_id, splits in split_by_template.items()
        if len(splits) > 1 and len(enabled) > 1
    }
    if leakage:
        raise RuntimeError(f"Patient-level split leakage detected: {leakage}")

    manifest = {
        "schema_version": 8,
        "generator": "medical_plausible_linear_fem_v3",
        "layout": {
            "universal_json_dir": "json",
            "sample_pattern": "<anatomy_id>/meshes/<sample_id>.npz",
            "anatomy_json_pattern": "<anatomy_id>/json/anatomy.json",
        },
        "n_samples": len(dataframe),
        "templates": [template.template_id for template in enabled],
        "deformation_families": list(cfg.deformation.family_weights),
        "solver_backend": cfg.solver.backend,
        "patient_split_leakage": leakage,
        "accepted_displacement_bin_quota": accepted_bin_quota,
        "accepted_deformation_family_quota": accepted_family_quota,
        "accepted_joint_family_bin_quota": accepted_joint_quota,
        "accepted_anatomy_quota": accepted_anatomy_quota,
        "accepted_joint_anatomy_family_quota": accepted_anatomy_family_quota,
        "accepted_joint_anatomy_family_bin_quota": (accepted_anatomy_family_bin_quota),
        "template_diagnostics": diagnostics_by_template,
        "mesh_manifest": str((output / "json" / "mesh_manifest.jsonl").resolve()),
        "config": cfg.to_dict(),
    }
    _write_anatomy_metadata(output, enabled, dataframe, diagnostics_by_template, cfg)
    _atomic_write_json(output / "json" / "dataset.json", manifest)
    return manifest
