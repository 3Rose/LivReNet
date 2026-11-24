from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import DeformationRegionConfig
from .deformation import (
    DeformationRegions,
    area_fraction,
    hop_distance_to_observed,
    splat_vertex_support,
)
from .geometry import MeshData
from .rasterizer import RasterizedSurface


def _spatial_support(valid: np.ndarray, reference: np.ndarray, bins: int = 8) -> float:
    h, w = valid.shape
    occupied = 0
    possible = 0
    for iy in range(bins):
        for ix in range(bins):
            ys = slice(iy * h // bins, (iy + 1) * h // bins)
            xs = slice(ix * w // bins, (ix + 1) * w // bins)
            if reference[ys, xs].any():
                possible += 1
                occupied += int(valid[ys, xs].any())
    return occupied / max(possible, 1)


def _surface_coverage(
    raster: RasterizedSurface, mask: np.ndarray, total_area: float
) -> float:
    return float(
        np.clip(np.sum(raster.surface_weight[mask]) / max(total_area, 1e-12), 0.0, 1.0)
    )


def _pixel_interpolated_vertex_values(
    mesh: MeshData, raster: RasterizedSurface, mask: np.ndarray, values: np.ndarray
) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return np.zeros(0, dtype=float)
    face_ids = raster.face_id[y, x]
    bary = raster.barycentric[y, x].astype(float)
    return np.sum(values[mesh.faces[face_ids]] * bary, axis=1)


def _region_coverage(
    mesh: MeshData,
    raster: RasterizedSurface,
    mask: np.ndarray,
    vertex_region: np.ndarray,
    vertex_area: np.ndarray,
) -> float:
    denom = float(vertex_area[vertex_region].sum())
    if denom <= 0:
        return 0.0
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return 0.0
    face_ids = raster.face_id[y, x]
    bary = raster.barycentric[y, x].astype(float)
    membership = np.sum(
        vertex_region[mesh.faces[face_ids]].astype(float) * bary, axis=1
    )
    weights = raster.surface_weight[y, x].astype(float)
    return float(np.clip(np.sum(weights * membership) / denom, 0.0, 1.0))


def _energy_evidence(
    mesh: MeshData,
    raster: RasterizedSurface,
    mask: np.ndarray,
    regions: DeformationRegions,
) -> float:
    if regions.total_energy <= 0 or mesh.displacement is None:
        return 0.0
    y, x = np.nonzero(mask)
    if len(y) == 0:
        return 0.0
    face_ids = raster.face_id[y, x]
    bary = raster.barycentric[y, x].astype(float)
    disp = np.sum(mesh.displacement[mesh.faces[face_ids]] * bary[..., None], axis=1)
    weights = raster.surface_weight[y, x].astype(float)
    return float(
        np.clip(
            np.sum(weights * np.sum(disp * disp, axis=1)) / regions.total_energy,
            0.0,
            1.0,
        )
    )


def input_vertex_metrics(
    mesh: MeshData,
    regions: DeformationRegions,
    face_ids: np.ndarray,
    barycentric: np.ndarray,
    vertex_support: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Area-based observability metrics for the canonical sparse point cloud.

    A 2,500-point cloud is obtained from a dense depth image. Summing the
    original per-pixel areas only measures the sampling rate (roughly 2,500 / N
    pixels), not the surface extent represented by the PC.  We instead splat
    input correspondences to template vertices and measure the supported
    physical area and deformation energy there.
    """
    support = (
        splat_vertex_support(len(mesh.points), mesh.faces, face_ids, barycentric)
        if vertex_support is None
        else np.asarray(vertex_support, dtype=float)
    )
    observed = support > 0
    total_area = float(regions.vertex_area.sum())
    surface_coverage = float(
        regions.vertex_area[observed].sum() / max(total_area, 1e-12)
    )
    if regions.total_energy > 0 and mesh.displacement is not None:
        energy = regions.vertex_area * np.sum(
            mesh.displacement * mesh.displacement, axis=1
        )
        evidence = float(
            np.clip(energy[observed].sum() / regions.total_energy, 0.0, 1.0)
        )
    else:
        evidence = 0.0
    return {
        "input_surface_coverage": float(np.clip(surface_coverage, 0.0, 1.0)),
        "deformation_evidence_input": evidence,
        "peak_core_coverage_input": area_fraction(
            observed, regions.peak_core, regions.vertex_area
        ),
        "input_observed_surface_vertex_fraction": float(
            np.clip(surface_coverage, 0.0, 1.0)
        ),
        "input_observed_support_fraction": area_fraction(
            observed, regions.support, regions.vertex_area
        ),
    }, support


def classify_training_stratum(metrics: dict[str, Any]) -> str:
    if metrics.get("sample_type") == "identity":
        return "identity"
    quality = float(metrics["measurement_quality_sensor"])
    coverage = float(metrics["sensor_surface_coverage"])
    evidence = float(metrics["deformation_evidence_sensor"])
    peak = float(metrics["peak_core_coverage_sensor"])
    peak_area = float(metrics.get("peak_region_area_fraction", 0.0))
    adjacent = float(metrics["unobserved_support_adjacent_fraction"])
    snr2 = float(metrics["observed_signal_fraction_snr_gt2"])
    if quality < 0.25 or coverage < 0.04:
        return "stress"
    if evidence < 0.04 and float(metrics.get("max_displacement_mm", 0.0)) > 1.0:
        return "hidden_deformation"
    if peak >= 0.20 and snr2 >= 0.25:
        return "informative_peak"
    if (
        peak_area >= float(metrics.get("continuation_peak_area_min", 0.15))
        and evidence >= 0.08
        and adjacent >= 0.05
        and float(metrics["observed_support_fraction"]) < 0.85
    ):
        return "informative_continuation"
    if evidence >= 0.05 and snr2 >= 0.10:
        return "moderate"
    return "weak_signal"


def bin_label(value: float, edges: tuple[float, ...], prefix: str) -> str:
    idx = int(
        np.clip(np.digitize([value], np.asarray(edges)[1:-1])[0], 0, len(edges) - 2)
    )
    return f"{prefix}_{edges[idx]:.2f}_{edges[idx + 1]:.2f}"


def compute_observation_metrics(
    mesh: MeshData,
    raster: RasterizedSurface,
    geometric_valid: np.ndarray,
    sensor_valid: np.ndarray,
    input_valid: np.ndarray,
    residual_px: np.ndarray,
    residual_p95_px: float,
    regions: DeformationRegions,
    raw_face_ids: np.ndarray,
    raw_barycentric: np.ndarray,
    raw_noisy_points: np.ndarray,
    raw_surface_points: np.ndarray,
    input_face_ids: np.ndarray,
    input_barycentric: np.ndarray,
    input_vertex_support: np.ndarray | None,
    mesh_unit_to_mm: float,
    region_config: DeformationRegionConfig,
    sample_type: str,
) -> dict[str, float | int | bool | str]:
    total_area = float(mesh.areas.sum())
    rendered = raster.valid
    geometric_coverage = _surface_coverage(raster, geometric_valid, total_area)
    sensor_coverage = _surface_coverage(raster, sensor_valid, total_area)
    input_metrics, _ = input_vertex_metrics(
        mesh,
        regions,
        input_face_ids,
        input_barycentric,
        vertex_support=input_vertex_support,
    )

    n_rendered = int(rendered.sum())
    n_geometric = int(geometric_valid.sum())
    n_sensor = int(sensor_valid.sum())
    n_input = int(input_valid.sum())
    sensor_valid_fraction = n_sensor / max(n_geometric, 1)
    input_retention_fraction = n_input / max(n_sensor, 1)
    incidence_score = (
        float(np.mean(raster.incidence[sensor_valid])) if n_sensor else 0.0
    )
    median_error = (
        float(np.median(np.abs(residual_px[sensor_valid])))
        if n_sensor
        else float("inf")
    )
    p95_error = (
        float(np.quantile(np.abs(residual_px[sensor_valid]), 0.95))
        if n_sensor
        else float("inf")
    )
    residual_quality = (
        math.exp(-median_error / max(float(residual_p95_px), 1e-6)) if n_sensor else 0.0
    )
    spatial_score = _spatial_support(sensor_valid, geometric_valid)
    density_score = min(1.0, n_sensor / max(min(2500, int(0.35 * n_geometric)), 1))
    components = np.asarray(
        [
            sensor_valid_fraction,
            incidence_score,
            residual_quality,
            density_score,
            spatial_score,
        ]
    )
    weights_q = np.asarray([0.30, 0.15, 0.25, 0.10, 0.20])
    quality = float(np.exp(np.sum(weights_q * np.log(np.clip(components, 1e-6, 1.0)))))

    metrics: dict[str, Any] = {
        "sample_type": sample_type,
        "geometric_surface_coverage": geometric_coverage,
        "sensor_surface_coverage": sensor_coverage,
        "input_surface_coverage": input_metrics["input_surface_coverage"],
        "deformation_evidence_geometric": _energy_evidence(
            mesh, raster, geometric_valid, regions
        ),
        "deformation_evidence_sensor": _energy_evidence(
            mesh, raster, sensor_valid, regions
        ),
        "deformation_evidence_input": input_metrics["deformation_evidence_input"],
        "peak_core_coverage_geometric": _region_coverage(
            mesh, raster, geometric_valid, regions.peak_core, regions.vertex_area
        ),
        "peak_core_coverage_sensor": _region_coverage(
            mesh, raster, sensor_valid, regions.peak_core, regions.vertex_area
        ),
        "peak_core_coverage_input": input_metrics["peak_core_coverage_input"],
        "input_observed_surface_vertex_fraction": input_metrics[
            "input_observed_surface_vertex_fraction"
        ],
        "input_observed_support_fraction": input_metrics[
            "input_observed_support_fraction"
        ],
        "input_metric_definition": "vertex_support_area_v1",
        "deformation_support_coverage_sensor": _region_coverage(
            mesh, raster, sensor_valid, regions.support, regions.vertex_area
        ),
        "energy_region_coverage_sensor": _region_coverage(
            mesh, raster, sensor_valid, regions.energy_region, regions.vertex_area
        ),
        "sensor_valid_fraction": float(sensor_valid_fraction),
        "input_retention_fraction": float(input_retention_fraction),
        "measurement_quality_sensor": float(np.clip(quality, 0.0, 1.0)),
        "mean_incidence_sensor": incidence_score,
        "median_abs_disparity_error_px": median_error
        if np.isfinite(median_error)
        else 0.0,
        "p95_abs_disparity_error_px": p95_error if np.isfinite(p95_error) else 0.0,
        "residual_quality": residual_quality,
        "spatial_support_score": spatial_score,
        "n_rendered_left_pixels": n_rendered,
        "n_stereo_geometric_pixels": n_geometric,
        "n_sensor_points_raw": n_sensor,
        "n_input_points": n_input,
        "max_displacement_mm": float(regions.max_displacement * mesh_unit_to_mm),
        "peak_region_area_fraction": float(regions.peak_area_fraction),
        "continuation_peak_area_min": float(region_config.continuation_peak_area_min),
        "continuation_eligible": bool(
            regions.peak_area_fraction >= region_config.continuation_peak_area_min
        ),
        "deformation_energy_defined": bool(regions.total_energy > 0),
    }

    if len(raw_face_ids) and mesh.displacement is not None:
        gt_disp = np.sum(
            mesh.displacement[mesh.faces[raw_face_ids]] * raw_barycentric[..., None],
            axis=1,
        )
        signal_mm = np.linalg.norm(gt_disp, axis=1) * mesh_unit_to_mm
        noise_mm = (
            np.linalg.norm(raw_noisy_points - raw_surface_points, axis=1)
            * mesh_unit_to_mm
        )
        snr = signal_mm / np.maximum(noise_mm, 0.5)
        metrics.update(
            {
                "observed_signal_mm_median": float(np.median(signal_mm)),
                "observed_signal_mm_p95": float(np.quantile(signal_mm, 0.95)),
                "observed_noise_mm_median": float(np.median(noise_mm)),
                "observed_signal_to_noise_median": float(np.median(snr)),
                "observed_signal_fraction_snr_gt1": float(np.mean(snr > 1.0)),
                "observed_signal_fraction_snr_gt2": float(np.mean(snr > 2.0)),
            }
        )
    else:
        metrics.update(
            {
                "observed_signal_mm_median": 0.0,
                "observed_signal_mm_p95": 0.0,
                "observed_noise_mm_median": 0.0,
                "observed_signal_to_noise_median": 0.0,
                "observed_signal_fraction_snr_gt1": 0.0,
                "observed_signal_fraction_snr_gt2": 0.0,
            }
        )

    vertex_support = splat_vertex_support(
        len(mesh.points), mesh.faces, raw_face_ids, raw_barycentric
    )
    observed_vertices = vertex_support > 0
    metrics["observed_surface_vertex_fraction"] = float(
        regions.vertex_area[observed_vertices].sum()
        / max(regions.vertex_area.sum(), 1e-12)
    )
    metrics["observed_support_fraction"] = area_fraction(
        observed_vertices, regions.support, regions.vertex_area
    )
    if regions.support.any() and observed_vertices.any():
        distance = hop_distance_to_observed(
            len(mesh.points),
            mesh.edges,
            observed_vertices,
            max_hops=int(region_config.continuation_hops),
            adjacency=mesh.vertex_adjacency,
        )
        unobserved_support = regions.support & ~observed_vertices
        adjacent = (
            unobserved_support
            & (distance >= 0)
            & (distance <= int(region_config.continuation_hops))
        )
        remote = unobserved_support & (
            (distance < 0) | (distance > int(region_config.continuation_hops))
        )
        metrics["unobserved_support_adjacent_fraction"] = area_fraction(
            adjacent, regions.support, regions.vertex_area
        )
        metrics["hidden_remote_support_fraction"] = area_fraction(
            remote, regions.support, regions.vertex_area
        )
    else:
        metrics["unobserved_support_adjacent_fraction"] = 0.0
        metrics["hidden_remote_support_fraction"] = 0.0

    metrics["effective_informativeness_rank_only"] = float(
        np.clip(
            metrics["deformation_evidence_sensor"]
            * metrics["measurement_quality_sensor"],
            0.0,
            1.0,
        )
    )
    metrics["training_stratum"] = classify_training_stratum(metrics)
    return metrics
