from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraConfig:
    """Camera prior defined only from the reference anatomy and access prior."""

    width: int = 360
    height: int = 288
    horizontal_fov_deg_range: tuple[float, float] = (39.7, 42.2)
    principal_x_fraction_range: tuple[float, float] = (0.445, 0.521)
    principal_y_fraction_range: tuple[float, float] = (0.417, 0.474)
    distance_range_diag: tuple[float, float] = (0.25, 0.80)
    tangent_offset_range_diag: tuple[float, float] = (0.0, 0.12)
    look_at_jitter_diag: float = 0.025
    tilt_std_deg: float = 7.0
    roll_range_deg: tuple[float, float] = (-15.0, 15.0)
    near_diag: float = 0.01
    max_sampling_attempts: int = 64
    # The primary benchmark concentrates around 20--25% physical surface
    # coverage; the lower bound leaves only a small tail of genuinely sparse
    # views while retaining a non-trivial complete-to-partial task.
    template_visible_surface_range: tuple[float, float] = (0.12, 0.30)
    target_surface_coverage_center: float = 0.225
    target_surface_coverage_sigma: float = 0.045
    require_anchor_visible: bool = True


@dataclass(frozen=True)
class StereoConfig:
    enabled: bool = True
    require_mutual_visibility: bool = True
    occlusion_tolerance_mm: float = 1.5
    occlusion_relative_tolerance: float = 0.01


@dataclass(frozen=True)
class ToolOcclusionConfig:
    probabilities: dict[str, float] = field(
        default_factory=lambda: {
            # Instruments are useful augmentation, but frequent contact
            # occlusion should not define the primary registration task.
            "none": 0.75,
            "partial_contact": 0.12,
            "total_contact": 0.03,
            "distractor": 0.10,
        }
    )
    width_fraction_range: tuple[float, float] = (0.02, 0.045)
    partial_offset_fraction: tuple[float, float] = (0.5, 1.5)
    edge_flying_probability: float = 0.08


@dataclass(frozen=True)
class SensorConfig:
    profile_path: str | None = None
    mesh_unit_to_mm: float = 1.0
    # Primary benchmark: calibrated depth residuals with modest coordinate
    # noise.  The default intentionally does not inherit the sparse/harsh
    # SGBM validity profile; holes, flying points and local biases are stress
    # factors, not a prerequisite for the core registration question.
    stochastic_validity_strength: float = 0.0
    local_bias_probability: float = 0.0
    local_bias_scale: float = 0.0
    random_hole_scale: float = 0.0
    flying_point_scale: float = 0.0
    residual_scale: float = 0.30
    gaussian_xyz_sigma_mm: float = 0.35
    correlated_depth_sigma_mm: float = 0.0
    canonical_point_budget: int = 2500
    minimum_raw_points: int = 128
    sampling_strategy: str = "image_density_balanced"
    store_raw_points: bool = False


@dataclass(frozen=True)
class SensorRandomizationConfig:
    """Bounded sensor randomization used to populate noise strata."""

    enabled: bool = True
    apply_to_evaluation: bool = True
    residual_scale_range: tuple[float, float] = (0.20, 0.40)
    validity_strength_range: tuple[float, float] = (0.0, 0.0)
    hole_scale_range: tuple[float, float] = (0.0, 0.0)
    flying_scale_range: tuple[float, float] = (0.0, 0.0)
    local_bias_probability_range: tuple[float, float] = (0.0, 0.0)
    local_bias_scale_range: tuple[float, float] = (0.0, 0.0)
    gaussian_xyz_sigma_mm_range: tuple[float, float] = (0.20, 0.50)
    correlated_depth_sigma_mm_range: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class DeformationRegionConfig:
    peak_relative_threshold: float = 0.80
    high_relative_threshold: float = 0.30
    support_relative_threshold: float = 0.10
    support_absolute_mm: float = 0.50
    energy_fraction: float = 0.80
    continuation_hops: int = 5
    continuation_peak_area_min: float = 0.15
    continuation_fallback_to_core: bool = True
    continuation_max_attempts: int = 3


@dataclass(frozen=True)
class TrainingSelectionConfig:
    """GT-aware composition policy. Applied only to the training split."""

    candidate_views_per_mesh: int = 16
    selected_views_per_mesh: int = 8
    selection_policy: str = "continuous_balanced_v2"
    identity_fraction: float = 0.20
    keep_candidate_bank: bool = False
    quotas: dict[str, int] = field(
        default_factory=lambda: {
            "informative_peak": 3,
            "informative_continuation": 3,
            "moderate": 3,
            "hidden_deformation": 2,
            "stress": 2,
            "weak_signal": 1,
        }
    )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation composition; balanced is the default controlled benchmark."""

    raw_views_per_mesh: int = 8
    candidate_views_per_mesh: int = 16
    selected_views_per_mesh: int = 8
    balanced_selection: bool = True
    include_identity: bool = True
    identity_fraction: float = 0.20
    benchmark_point_budgets: tuple[int, ...] = (1000, 2500, 4000)
    visibility_bins: tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 1.0)
    signal_bins: tuple[float, ...] = (0.0, 0.05, 0.20, 0.50, 1.0)


@dataclass(frozen=True)
class LandmarkConfig:
    global_landmark_count: int = 128
    hidden_landmark_count: int = 32


@dataclass(frozen=True)
class OutputConfig:
    # ``output_dir`` is the pointclouds/ directory. Point clouds live directly
    # in it, while all manifests and provenance live under pointclouds/doc/.
    observations_subdir: str = ""
    meshes_subdir: str = "legacy_mesh_targets"
    metadata_subdir: str = "doc"
    # The original template/deformed meshes are already authoritative. Writing
    # NPZ copies is optional and only needed by legacy standalone loaders.
    write_mesh_targets: bool = False
    write_pcd_ascii: bool = False
    overwrite: bool = False
    fail_if_nonempty: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    stereo: StereoConfig = field(default_factory=StereoConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    sensor_randomization: SensorRandomizationConfig = field(
        default_factory=SensorRandomizationConfig
    )
    tool: ToolOcclusionConfig = field(default_factory=ToolOcclusionConfig)
    deformation_regions: DeformationRegionConfig = field(
        default_factory=DeformationRegionConfig
    )
    training: TrainingSelectionConfig = field(default_factory=TrainingSelectionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    generation_size: str = "custom"
    seed: int = 42
    # Identity controls are specified per split and interpreted at deformation
    # group level.  Keeping this explicit prevents a global train-only
    # percentage from silently producing no validation/test controls.
    identity_fractions: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "GenerationConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera=CameraConfig(**raw.get("camera", {})),
            stereo=StereoConfig(**raw.get("stereo", {})),
            sensor=SensorConfig(**raw.get("sensor", {})),
            sensor_randomization=SensorRandomizationConfig(
                **raw.get("sensor_randomization", {})
            ),
            tool=ToolOcclusionConfig(**raw.get("tool", {})),
            deformation_regions=DeformationRegionConfig(
                **raw.get("deformation_regions", {})
            ),
            training=TrainingSelectionConfig(**raw.get("training", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            landmarks=LandmarkConfig(**raw.get("landmarks", {})),
            output=OutputConfig(**raw.get("output", {})),
            generation_size=str(raw.get("generation_size", "custom")),
            seed=int(raw.get("seed", 42)),
            identity_fractions={
                str(split): float(value)
                for split, value in raw.get(
                    "identity_fractions",
                    {},
                ).items()
            },
        )

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


GENERATION_SIZE_PRESETS: dict[str, tuple[int, int]] = {
    # (selected observations, candidate bank). A 2x bank is enough for
    # continuous maximin selection while avoiding the old 64-candidate cost.
    "lite": (4, 8),
    "default": (8, 16),
    "large": (16, 32),
}


def apply_generation_size(config: GenerationConfig, size: str) -> GenerationConfig:
    if size not in GENERATION_SIZE_PRESETS:
        raise ValueError(
            f"Unknown generation size {size!r}; choose {sorted(GENERATION_SIZE_PRESETS)}"
        )
    selected, candidates = GENERATION_SIZE_PRESETS[size]
    return replace(
        config,
        generation_size=size,
        training=replace(
            config.training,
            selected_views_per_mesh=selected,
            candidate_views_per_mesh=candidates,
            selection_policy="continuous_balanced_v2",
        ),
        evaluation=replace(
            config.evaluation,
            selected_views_per_mesh=selected,
            candidate_views_per_mesh=candidates,
            balanced_selection=True,
            raw_views_per_mesh=selected,
        ),
    )


@dataclass(frozen=True)
class SensorProfile:
    schema_version: str
    source: dict[str, Any]
    image: dict[str, Any]
    camera: dict[str, Any]
    reference_geometry: dict[str, Any]
    stereo_estimator: dict[str, Any]
    validity_model: dict[str, Any]
    residual_model: dict[str, Any]
    spatial_artifacts: dict[str, Any]

    @classmethod
    def from_json(cls, path: str | Path) -> "SensorProfile":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
