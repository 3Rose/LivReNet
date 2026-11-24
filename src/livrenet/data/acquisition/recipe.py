"""Configured camera, stereo and deformation-aware acquisition."""

from typing import Any

import numpy as np

from .config import (
    CameraConfig,
    DeformationRegionConfig,
    GenerationConfig,
    LandmarkConfig,
    SensorConfig,
    SensorRandomizationConfig,
    StereoConfig,
    ToolOcclusionConfig,
)


def _build_engine_config(config: dict[str, Any]) -> GenerationConfig:
    pc = config.get("pcgen", {})
    camera = pc.get("camera", {})
    stereo = pc.get("stereo", {})
    sensor = pc.get("sensor", {})
    tool = pc.get("tool", {})
    regions = pc.get("deformation_regions", {})
    coverage = config.get("coverage", {})
    generation = config.get("generation", {})
    camera_cfg = CameraConfig(
        width=int(camera.get("width", 320)),
        height=int(camera.get("height", 256)),
        horizontal_fov_deg_range=tuple(
            camera.get("horizontal_fov_deg_range", [39.7, 42.2])
        ),
        principal_x_fraction_range=tuple(
            camera.get("principal_x_fraction_range", [0.445, 0.521])
        ),
        principal_y_fraction_range=tuple(
            camera.get("principal_y_fraction_range", [0.417, 0.474])
        ),
        distance_range_diag=tuple(camera.get("distance_range_diag", [0.25, 0.80])),
        tangent_offset_range_diag=tuple(
            camera.get("tangent_offset_range_diag", [0.0, 0.12])
        ),
        look_at_jitter_diag=float(camera.get("look_at_jitter_diag", 0.025)),
        tilt_std_deg=float(camera.get("tilt_std_deg", 7.0)),
        roll_range_deg=tuple(camera.get("roll_range_deg", [-15.0, 15.0])),
        near_diag=float(camera.get("near_diag", 0.01)),
        max_sampling_attempts=int(camera.get("max_sampling_attempts", 64)),
        template_visible_surface_range=tuple(
            camera.get("template_visible_surface_range", [0.12, 0.30])
        ),
        target_surface_coverage_center=float(
            camera.get("target_surface_coverage_center", 0.225)
        ),
        target_surface_coverage_sigma=float(
            camera.get("target_surface_coverage_sigma", 0.045)
        ),
        require_anchor_visible=bool(camera.get("require_anchor_visible", True)),
    )
    sensor_cfg = SensorConfig(
        profile_path=str(
            pc.get("profile_path") or config.get("paths", {}).get("profile") or ""
        ),
        mesh_unit_to_mm=float(sensor.get("mesh_unit_to_mm", 1.0)),
        stochastic_validity_strength=float(
            sensor.get("stochastic_validity_strength", 0.08)
        ),
        local_bias_probability=float(sensor.get("local_bias_probability", 0.03)),
        local_bias_scale=float(sensor.get("local_bias_scale", 0.35)),
        random_hole_scale=float(sensor.get("random_hole_scale", 0.2)),
        flying_point_scale=float(sensor.get("flying_point_scale", 0.3)),
        residual_scale=float(sensor.get("residual_scale", 0.3)),
        gaussian_xyz_sigma_mm=float(sensor.get("gaussian_xyz_sigma_mm", 0.35)),
        correlated_depth_sigma_mm=float(sensor.get("correlated_depth_sigma_mm", 0.1)),
        canonical_point_budget=int(
            generation.get(
                "canonical_point_budget", sensor.get("canonical_point_budget", 2500)
            )
        ),
        minimum_raw_points=int(
            generation.get("minimum_raw_points", sensor.get("minimum_raw_points", 128))
        ),
        sampling_strategy=str(
            sensor.get("sampling_strategy", "image_density_balanced")
        ),
        store_raw_points=False,
    )
    region_cfg = DeformationRegionConfig(
        peak_relative_threshold=float(
            regions.get(
                "peak_relative_threshold", coverage.get("peak_relative_threshold", 0.8)
            )
        ),
        high_relative_threshold=float(
            regions.get(
                "high_relative_threshold", coverage.get("high_relative_threshold", 0.3)
            )
        ),
        support_relative_threshold=float(
            regions.get("support_relative_threshold", 0.1)
        ),
        support_absolute_mm=float(regions.get("support_absolute_mm", 0.5)),
        energy_fraction=float(regions.get("energy_fraction", 0.8)),
        continuation_hops=int(regions.get("continuation_hops", 5)),
        continuation_peak_area_min=float(
            coverage.get("continuation_peak_region_area_min", 0.15)
        ),
        continuation_fallback_to_core=bool(
            coverage.get("continuation_fallback_to_core", True)
        ),
    )
    probabilities = tool.get(
        "probabilities",
        {
            "none": 0.82,
            "partial_contact": 0.08,
            "total_contact": 0.02,
            "distractor": 0.08,
        },
    )
    return GenerationConfig(
        camera=camera_cfg,
        stereo=StereoConfig(
            enabled=bool(stereo.get("enabled", True)),
            require_mutual_visibility=bool(
                stereo.get("require_mutual_visibility", True)
            ),
            occlusion_tolerance_mm=float(stereo.get("occlusion_tolerance_mm", 1.5)),
            occlusion_relative_tolerance=float(
                stereo.get("occlusion_relative_tolerance", 0.01)
            ),
        ),
        sensor=sensor_cfg,
        sensor_randomization=SensorRandomizationConfig(enabled=False),
        tool=ToolOcclusionConfig(
            probabilities={str(k): float(v) for k, v in probabilities.items()},
            width_fraction_range=tuple(tool.get("width_fraction_range", [0.02, 0.045])),
            partial_offset_fraction=tuple(
                tool.get("partial_offset_fraction", [0.5, 1.5])
            ),
            edge_flying_probability=float(tool.get("edge_flying_probability", 0.08)),
        ),
        deformation_regions=region_cfg,
        landmarks=LandmarkConfig(global_landmark_count=128, hidden_landmark_count=32),
        seed=int(config.get("seed", 20260828)),
        identity_fractions={
            str(k): float(v)
            for k, v in generation.get(
                "identity_fractions", {"train": 0.03, "val": 0.05, "test": 0.10}
            ).items()
        },
    )


def _guided_surface_weights(regions, role: str, settings: dict[str, Any]) -> np.ndarray:
    """Return a positive camera-anchor distribution for one semantic role.

    This is a data-generation policy only: it uses the known synthetic GT to
    make informative views easier to find. The selected masks and weights are
    not copied into the deployable point-cloud input.
    """
    peak = np.asarray(regions.peak_core, dtype=float)
    high = np.asarray(regions.high_deformation, dtype=float)
    support = np.asarray(regions.support, dtype=float)
    energy = np.asarray(regions.energy_region, dtype=float)
    background = np.ones_like(peak, dtype=float) * float(
        settings.get("background_weight", 0.03)
    )
    if role == "core":
        weights = background
        weights = weights + float(settings.get("core_peak_weight", 1.0)) * peak
        weights = weights + float(settings.get("core_high_weight", 0.35)) * high
    elif role == "continuation":
        ring = support * (1.0 - peak)
        if float(ring.sum()) <= 0:
            ring = high * (1.0 - peak)
        if float(ring.sum()) <= 0:
            ring = energy * (1.0 - peak)
        weights = background
        weights = weights + float(settings.get("continuation_ring_weight", 1.0)) * ring
        weights = weights + float(settings.get("continuation_high_weight", 0.35)) * high
        weights = weights + float(settings.get("continuation_peak_weight", 0.10)) * peak
    elif role == "safety":
        weights = background + float(settings.get("safety_background_weight", 1.0)) * (
            1.0 - support
        )
        weights = (
            weights + float(settings.get("safety_support_penalty", 0.05)) * support
        )
    else:
        weights = np.ones_like(peak, dtype=float)
    return np.maximum(weights, 1e-6)
