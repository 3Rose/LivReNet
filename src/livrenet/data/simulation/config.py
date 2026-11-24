from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALID_SPLITS = {"train", "val", "test"}
VALID_FAMILIES = {"local_narrow", "local_broad", "lobe_motion"}


@dataclass
class DiscoveryConfig:
    """Automatic patient discovery from an IRCAD prepared-data root."""

    enabled: bool = False
    prepared_root: str = ""
    dataset_manifest_filename: str = "dataset_manifest.json"
    anatomy_variant: str = "atlas"
    surface_filename: str = "liver_surface_from_volume.vtp"
    volume_xdmf_filename: str = "liver_volume.xdmf"
    volume_data_filename: str = "liver_volume.h5"
    portal_filename: str = "portalvein_surface_clean.vtp"
    caval_filenames: tuple[str, ...] = (
        "venacava_surface_clean.vtp",
        "venoussystem_surface_clean.vtp",
    )
    preparation_report_filename: str = "preparation_report.json"
    skip_failed_quality: bool = True
    validate_surface_topology: bool = True
    max_volume_points: int = 0
    split_seed: int = 20260729


@dataclass
class TemplateConfig:
    template_id: str
    surface_path: str
    volume_xdmf_path: str
    anatomical_paths: Mapping[str, str]
    output_surface_path: str | None = None
    surface_transfer_path: str | None = None
    difficulty_score: float | None = None
    xdmf_mesh_name: str = "Grid"
    enabled: bool = True
    split: str | None = None


@dataclass
class MaterialConfig:
    """Homogeneous small-strain material for the fast production backend.

    With millimetre geometry, ``3e-3 N/mm²`` is ``3 kPa``. Young's modulus
    randomization is per anatomy because displacement-controlled rescaling makes
    per-sample E changes almost geometrically redundant.
    """

    model: str = "linear_elastic"
    young_modulus: float = 3.0e-3
    poisson_ratio: float = 0.45
    randomize_per_template: bool = False
    young_modulus_log_std: float = 0.15
    poisson_ratio_min: float = 0.42
    poisson_ratio_max: float = 0.47


@dataclass
class AnchorRegimeConfig:
    name: str
    weight: float
    hard_radius_scale: float
    soft_radius_scale: float
    stiffness_scale: float
    min_hard_vertices: int
    release_fraction: float = 0.0


@dataclass
class AnchorConfig:
    hard_radius_ratio: float = 0.020
    soft_radius_ratio: float = 0.075
    soft_anchor_relative_stiffness_min: float = 0.02
    soft_anchor_relative_stiffness_max: float = 0.08
    radius_jitter: float = 0.12
    load_avoid_radius_ratio: float = 0.055
    regimes: tuple[AnchorRegimeConfig, ...] = field(
        default_factory=lambda: (
            AnchorRegimeConfig("anchored", 0.45, 1.00, 1.00, 1.00, 5, 0.00),
            AnchorRegimeConfig("partially_mobilized", 0.35, 0.75, 0.90, 0.55, 4, 0.30),
            AnchorRegimeConfig("softly_supported", 0.20, 0.40, 1.25, 0.30, 3, 0.55),
        )
    )


@dataclass
class DeformationConfig:
    """Biomechanically inspired, deliberately interpretable load families."""

    family_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "local_narrow": 0.20,
            "local_broad": 0.40,
            "lobe_motion": 0.40,
        }
    )
    radius_ratio_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "local_narrow": (0.100, 0.160),
            "local_broad": (0.200, 0.380),
            "lobe_motion": (0.380, 0.650),
        }
    )
    family_max_displacement_mm: Mapping[str, float] = field(
        default_factory=lambda: {
            # Very focal 20+ mm motions repeatedly violate the strain envelope
            # on the calibrated IRCAD mesh. Keep the narrow family in the
            # small/moderate range; larger motions use broader support.
            "local_narrow": 15.0,
            "local_broad": 28.0,
            "lobe_motion": 35.0,
        }
    )
    # Also cap targets relative to anatomy size for cross-patient robustness.
    family_max_displacement_ratio: Mapping[str, float] = field(
        default_factory=lambda: {
            "local_narrow": 0.10,
            "local_broad": 0.13,
            "lobe_motion": 0.16,
        }
    )
    push_probability: Mapping[str, float] = field(
        default_factory=lambda: {
            "local_narrow": 0.55,
            "local_broad": 0.50,
            "lobe_motion": 0.35,
        }
    )
    profile_choices: tuple[str, ...] = (
        "gaussian",
        "cosine",
        "wendland",
        "flat_smooth",
    )
    # ``flat_smooth`` is opt-in by family; it is inappropriate as an implicit
    # default for narrow point-contact-like loads.
    profile_weights: tuple[float, ...] = (0.35, 0.25, 0.40, 0.0)
    # Family-specific profile priors. Broad/lobar motions deliberately avoid
    # the sharply concentrated Wendland kernel used by local contact.
    profile_weights_by_family: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            "local_narrow": {
                "flat_smooth": 0.60,
                "gaussian": 0.25,
                "cosine": 0.15,
            },
            "local_broad": {"flat_smooth": 1.0},
            "lobe_motion": {"flat_smooth": 1.0},
        }
    )
    # Raise the minimum sampled radius as target displacement approaches the
    # upper end of a family's feasible range. This prevents tall, narrow
    # broad/lobar deformations while keeping max-displacement quotas intact.
    radius_amplitude_coupling: Mapping[str, float] = field(
        default_factory=lambda: {
            "local_narrow": 0.55,
            "local_broad": 0.85,
            "lobe_motion": 0.90,
        }
    )
    oblique_angle_min_deg: float = 20.0
    oblique_angle_max_deg: float = 60.0
    tangential_normal_component_max: float = 0.15
    # Prefer instrument-accessible peripheral parenchyma without restricting
    # every lobe manipulation to the most compliant extreme tips.
    lobe_peripheral_quantile: float = 0.45
    min_patch_vertices: int = 10


@dataclass
class DisplacementConfig:
    # This generator is scoped to instrument-induced deformation samples.
    # Identity pairs, if needed downstream, belong to the observation pipeline.
    identity_fraction: float = 0.0
    bins: tuple[tuple[str, float, float, float], ...] = (
        ("small", 3.0, 8.0, 0.15),
        ("moderate", 8.0, 18.0, 0.45),
        ("large", 18.0, 30.0, 0.35),
        ("severe", 30.0, 35.0, 0.05),
    )
    initial_traction: float = 1.0e-4
    # The bin weights describe the accepted non-identity dataset, not merely
    # the proposal stream.  The final audit permits this many percentage
    # points of deviation per bin (and for the identity fraction).
    accepted_quota_tolerance_percentage_points: float = 5.0


@dataclass
class ValidityConfig:
    """Physics-informed acceptance envelope.

    These limits reject geometric failure and implausibly concentrated strain.
    They are not a substitute for clinical validation.
    """

    min_jacobian: float = 0.20
    max_jacobian: float = 2.50
    max_inverted_fraction: float = 0.0
    min_global_volume_ratio: float = 0.90
    max_global_volume_ratio: float = 1.10
    min_edge_stretch_ratio: float = 0.50
    max_edge_stretch_ratio: float = 1.80
    max_green_lagrange_abs_p95: float = 0.40
    max_green_lagrange_abs_p99: float = 0.75
    max_green_lagrange_abs_max: float = 2.50
    max_edge_jump_to_peak_ratio: float = 0.20
    max_bc_mean_displacement_mm: float = 1.0e-6
    require_peak_inside_support: bool = True
    require_single_high_displacement_component: bool = True
    # Opt in for new datasets; historical configurations keep their old envelope.
    strict_surface_integrity: bool = False
    max_generation_attempts: int = 40
    # Several load proposals share one attachment realization.  A later
    # realization is still available when the first is incompatible with the
    # requested deformation, without paying one LU factorization per proposal.
    max_anchor_realizations: int = 4
    # If no physically valid realization exists for a sampled parameter set,
    # discard that draw and draw a new one instead of aborting the whole run.
    max_intent_resamples: int = 3
    # After exhausting one family, keep the requested displacement bin but
    # draw an analogous realization from another compatible family. This is a
    # bounded outer replacement loop, not a relaxation of validity checks.
    max_sample_replacements: int = 3
    # Optional family-specific spatial-shape envelope. Fractions are weighted
    # by surface area rather than vertex count:
    # [min active area above 20% of peak, max active area,
    #  min area above half peak, max peak/mean displacement].
    # Empty by default so generic/example configurations remain compatible.
    family_spread_bounds: Mapping[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


@dataclass
class SplitConfig:
    train_fraction: float = 0.75
    val_fraction: float = 0.10
    test_fraction: float = 0.15
    require_patient_level_for_multiple_templates: bool = True
    allow_single_template_sample_split: bool = True
    ood_fraction_within_test: float = 0.25
    ood_radius_multiplier: tuple[float, float] = (0.75, 1.25)
    ood_oblique_angle_deg: tuple[float, float] = (60.0, 80.0)
    ood_anchor_stiffness_multiplier: tuple[float, float] = (0.40, 1.80)


@dataclass
class QuotaConfig:
    """Accepted-output anatomy quotas.

    An empty mapping assigns the same weight to every enabled anatomy.  When
    provided, the mapping must name every enabled anatomy and contain strictly
    positive weights.  These weights control accepted samples, not proposals.
    """

    anatomy_weights: Mapping[str, float] = field(default_factory=dict)
    # Optional/preflight-populated mesh difficulty (larger is harder). It only
    # shapes the anatomy x family x bin table while preserving every margin.
    anatomy_difficulty_scores: Mapping[str, float] = field(default_factory=dict)
    difficulty_bias_strength: float = 2.0
    # Do not assign high-amplitude bins to reference FEM meshes whose q95
    # tetrahedral edge ratio already indicates a fragile discretization.
    max_anatomy_difficulty_by_bin: Mapping[str, float] = field(
        default_factory=lambda: {"severe": 3.0}
    )


@dataclass
class AnatomyCalibrationConfig:
    """Normalize sampling by anatomy rather than by bounding box alone.

    Surface fractions make load and attachment supports comparable across
    differently tessellated patients.  The characteristic length is the cube
    root of enclosed liver volume; a zero reference requests automatic use of
    the median enabled anatomy at run start.
    """

    enabled: bool = True
    characteristic_length_reference_mm: float = 0.0
    amplitude_scale_exponent: float = 1.0
    amplitude_scale_bounds: tuple[float, float] = (0.85, 1.15)
    hard_area_fraction_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "anchored": (0.07, 0.11),
            "partially_mobilized": (0.04, 0.075),
            "softly_supported": (0.015, 0.035),
        }
    )
    soft_area_fraction_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "anchored": (0.55, 0.72),
            "partially_mobilized": (0.45, 0.62),
            "softly_supported": (0.55, 0.72),
        }
    )
    load_area_fraction_ranges: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "local_narrow": (0.015, 0.05),
            "local_broad": (0.06, 0.16),
            "lobe_motion": (0.16, 0.32),
        }
    )
    # Avoid placing the deformation peak on the worst-conditioned incident
    # tetrahedra of each mesh. The remaining quantile is anatomy-relative.
    load_center_quality_quantile: float = 1.0
    capacity_adaptation_enabled: bool = True
    capacity_bisection_steps: int = 8
    # Do not collapse accepted samples onto the lower edge of their quota bin.
    capacity_min_bin_fraction: float = 0.02


@dataclass
class SolverConfig:
    backend: str = "linear"
    petsc_options: Mapping[str, str] = field(
        default_factory=lambda: {"ksp_type": "preonly", "pc_type": "lu"}
    )
    mapping_tolerance_ratio: float = 1.0e-5
    require_serial: bool = True
    # MeshGen parallelizes over serial FEM processes. Keep each process's
    # native BLAS/OpenMP runtime bounded to avoid nested oversubscription.
    native_threads_per_worker: int = 1
    # Required by v4 physics diagnostics. Kept opt-in so legacy MeshGen runs
    # do not silently multiply their storage footprint.
    store_volume_displacement: bool = False


@dataclass
class ProjectConfig:
    templates: tuple[TemplateConfig, ...]
    out_dir: str
    n_samples: int = 1000
    n_workers: int = 1
    # Recycle native PETSc/DOLFINx state during long notebook runs.
    max_tasks_per_worker: int = 32
    # Retry a worker-level failure with a fresh deterministic seed without
    # changing the immutable anatomy/family/bin quota assignment.
    # Fresh deterministic recovery seeds for a quota cell that exhausts its
    # screened proposal budget. Difficult high-amplitude cells can be valid
    # but sparse; retrying them is preferable to relaxing physical gates.
    executor_retries_per_sample: int = 4
    random_seed: int = 42
    overwrite_existing: bool = False
    sample_prefix: str = "mesh"
    sample_id_width: int = 6
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    deformation: DeformationConfig = field(default_factory=DeformationConfig)
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    quotas: QuotaConfig = field(default_factory=QuotaConfig)
    anatomy_calibration: AnatomyCalibrationConfig = field(
        default_factory=AnatomyCalibrationConfig
    )
    solver: SolverConfig = field(default_factory=SolverConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple(value):
    return tuple(value) if isinstance(value, list) else value


def _tupleize(data: dict[str, Any]) -> ProjectConfig:
    data = dict(data)
    data["templates"] = tuple(TemplateConfig(**item) for item in data["templates"])

    if isinstance(data.get("discovery"), dict):
        discovery_data = dict(data["discovery"])
        if "caval_filenames" in discovery_data:
            discovery_data["caval_filenames"] = _tuple(
                discovery_data["caval_filenames"]
            )
        data["discovery"] = DiscoveryConfig(**discovery_data)

    if isinstance(data.get("material"), dict):
        data["material"] = MaterialConfig(**data["material"])

    if isinstance(data.get("anchors"), dict):
        anchor_data = dict(data["anchors"])
        if "regimes" in anchor_data:
            anchor_data["regimes"] = tuple(
                AnchorRegimeConfig(**item) for item in anchor_data["regimes"]
            )
        data["anchors"] = AnchorConfig(**anchor_data)

    if isinstance(data.get("deformation"), dict):
        deformation_data = dict(data["deformation"])
        for key in ("profile_choices", "profile_weights"):
            if key in deformation_data:
                deformation_data[key] = _tuple(deformation_data[key])
        if "radius_ratio_ranges" in deformation_data:
            deformation_data["radius_ratio_ranges"] = {
                name: tuple(bounds)
                for name, bounds in deformation_data["radius_ratio_ranges"].items()
            }
        data["deformation"] = DeformationConfig(**deformation_data)

    if isinstance(data.get("displacement"), dict):
        displacement_data = dict(data["displacement"])
        if "bins" in displacement_data:
            displacement_data["bins"] = tuple(
                tuple(item) for item in displacement_data["bins"]
            )
        data["displacement"] = DisplacementConfig(**displacement_data)

    if isinstance(data.get("validity"), dict):
        validity_data = dict(data["validity"])
        if "family_spread_bounds" in validity_data:
            validity_data["family_spread_bounds"] = {
                name: tuple(bounds)
                for name, bounds in validity_data["family_spread_bounds"].items()
            }
        data["validity"] = ValidityConfig(**validity_data)

    if isinstance(data.get("splits"), dict):
        split_data = dict(data["splits"])
        for key in (
            "ood_radius_multiplier",
            "ood_oblique_angle_deg",
            "ood_anchor_stiffness_multiplier",
        ):
            if key in split_data:
                split_data[key] = tuple(split_data[key])
        data["splits"] = SplitConfig(**split_data)

    if isinstance(data.get("quotas"), dict):
        data["quotas"] = QuotaConfig(**data["quotas"])

    if isinstance(data.get("anatomy_calibration"), dict):
        calibration_data = dict(data["anatomy_calibration"])
        if "amplitude_scale_bounds" in calibration_data:
            calibration_data["amplitude_scale_bounds"] = tuple(
                calibration_data["amplitude_scale_bounds"]
            )
        for key in (
            "hard_area_fraction_ranges",
            "soft_area_fraction_ranges",
            "load_area_fraction_ranges",
        ):
            if key in calibration_data:
                calibration_data[key] = {
                    name: tuple(bounds)
                    for name, bounds in calibration_data[key].items()
                }
        data["anatomy_calibration"] = AnatomyCalibrationConfig(**calibration_data)

    if isinstance(data.get("solver"), dict):
        data["solver"] = SolverConfig(**data["solver"])

    return ProjectConfig(**data)


def _positive_weights(values: Mapping[str, float], label: str) -> None:
    if not values or any(float(value) < 0 for value in values.values()):
        raise ValueError(f"{label} must contain non-negative weights")
    if sum(float(value) for value in values.values()) <= 0:
        raise ValueError(f"{label} weights must sum to a positive value")


def validate_config(cfg: ProjectConfig, *, check_paths: bool = False) -> None:
    if cfg.n_workers < 1:
        raise ValueError("n_workers must be positive")
    if cfg.max_tasks_per_worker < 1:
        raise ValueError("max_tasks_per_worker must be positive")
    if cfg.executor_retries_per_sample < 0:
        raise ValueError("executor_retries_per_sample cannot be negative")
    if cfg.solver.native_threads_per_worker < 1:
        raise ValueError("solver.native_threads_per_worker must be positive")
    enabled = [template for template in cfg.templates if template.enabled]
    discovery_pending = bool(cfg.discovery.enabled and cfg.discovery.prepared_root)
    if not enabled and not discovery_pending:
        raise ValueError(
            "At least one template must be enabled, or discovery must provide "
            "a prepared_root"
        )
    ids = [template.template_id for template in enabled]
    if len(ids) != len(set(ids)):
        raise ValueError("Enabled template_id values must be unique")

    for template in enabled:
        if not template.anatomical_paths:
            raise ValueError(
                f"{template.template_id}: anatomical_paths cannot be empty"
            )
        if (
            template.difficulty_score is not None
            and float(template.difficulty_score) <= 0
        ):
            raise ValueError(
                f"{template.template_id}: difficulty_score must be positive"
            )
        if template.split is not None and template.split.lower() not in VALID_SPLITS:
            raise ValueError(
                f"{template.template_id}: invalid split {template.split!r}"
            )
        if check_paths:
            paths = [
                template.surface_path,
                template.volume_xdmf_path,
                *template.anatomical_paths.values(),
            ]
            if template.output_surface_path is not None:
                paths.append(template.output_surface_path)
            if template.surface_transfer_path is not None:
                paths.append(template.surface_transfer_path)
            missing = [path for path in paths if not Path(path).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{template.template_id}: missing input paths: {missing}"
                )

    explicit_splits = [template.split is not None for template in enabled]
    if len(enabled) > 1 and cfg.splits.require_patient_level_for_multiple_templates:
        if not all(explicit_splits):
            raise ValueError(
                "Multiple anatomies require an explicit patient-level split on every "
                "enabled template"
            )
    if len(enabled) == 1 and not explicit_splits[0]:
        if not cfg.splits.allow_single_template_sample_split:
            raise ValueError("Single-template sample-level splitting is disabled")

    anatomy_weights = cfg.quotas.anatomy_weights
    if anatomy_weights and enabled:
        if set(anatomy_weights) != set(ids):
            raise ValueError(
                "anatomy_weights must define exactly every enabled template_id"
            )
        if any(float(weight) <= 0.0 for weight in anatomy_weights.values()):
            raise ValueError("anatomy_weights must be strictly positive")
    difficulty_scores = cfg.quotas.anatomy_difficulty_scores
    if difficulty_scores and enabled:
        if set(difficulty_scores) != set(ids):
            raise ValueError(
                "anatomy_difficulty_scores must define exactly every enabled "
                "template_id"
            )
        if any(float(value) <= 0.0 for value in difficulty_scores.values()):
            raise ValueError("anatomy_difficulty_scores must be strictly positive")
    if cfg.quotas.difficulty_bias_strength < 0:
        raise ValueError("difficulty_bias_strength cannot be negative")
    bin_names = {str(item[0]) for item in cfg.displacement.bins}
    thresholds = cfg.quotas.max_anatomy_difficulty_by_bin
    if any(name not in bin_names for name in thresholds):
        raise ValueError("max_anatomy_difficulty_by_bin contains an unknown bin")
    if any(float(value) <= 0.0 for value in thresholds.values()):
        raise ValueError("max_anatomy_difficulty_by_bin values must be positive")

    split_total = (
        cfg.splits.train_fraction + cfg.splits.val_fraction + cfg.splits.test_fraction
    )
    if abs(split_total - 1.0) > 1.0e-9:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1")
    if not 0.0 <= cfg.splits.ood_fraction_within_test <= 1.0:
        raise ValueError("ood_fraction_within_test must lie in [0, 1]")

    if cfg.material.young_modulus <= 0:
        raise ValueError("young_modulus must be positive")
    if not 0.0 < cfg.material.poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie strictly between 0 and 0.5")
    if not 0.0 <= cfg.displacement.identity_fraction < 1.0:
        raise ValueError("identity_fraction must lie in [0, 1)")
    if not 0.0 <= cfg.displacement.accepted_quota_tolerance_percentage_points <= 100.0:
        raise ValueError(
            "accepted_quota_tolerance_percentage_points must lie in [0, 100]"
        )

    _positive_weights(cfg.deformation.family_weights, "family_weights")
    if set(cfg.deformation.family_weights) != VALID_FAMILIES:
        raise ValueError(f"family_weights must define exactly {sorted(VALID_FAMILIES)}")
    for mapping_name in (
        "radius_ratio_ranges",
        "family_max_displacement_mm",
        "family_max_displacement_ratio",
        "push_probability",
    ):
        mapping = getattr(cfg.deformation, mapping_name)
        if set(mapping) != VALID_FAMILIES:
            raise ValueError(
                f"{mapping_name} must define exactly {sorted(VALID_FAMILIES)}"
            )
    if len(cfg.deformation.profile_choices) != len(cfg.deformation.profile_weights):
        raise ValueError("profile_choices and profile_weights lengths differ")
    if sum(cfg.deformation.profile_weights) <= 0:
        raise ValueError("profile_weights must sum to a positive value")
    for family, bounds in cfg.deformation.radius_ratio_ranges.items():
        if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
            raise ValueError(f"{family}: invalid radius_ratio_ranges bounds")
    for family, coupling in cfg.deformation.radius_amplitude_coupling.items():
        if family not in VALID_FAMILIES or not 0.0 <= float(coupling) <= 1.0:
            raise ValueError(f"{family}: invalid radius_amplitude_coupling")
    for family, weights in cfg.deformation.profile_weights_by_family.items():
        if family not in VALID_FAMILIES:
            raise ValueError(f"Unknown profile family: {family}")
        _positive_weights(weights, f"{family} profile")
        if any(name not in cfg.deformation.profile_choices for name in weights):
            raise ValueError(f"{family}: unknown profile in family-specific weights")
    for family, probability in cfg.deformation.push_probability.items():
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{family}: push_probability must lie in [0, 1]")

    regime_names = [regime.name for regime in cfg.anchors.regimes]
    if len(regime_names) != len(set(regime_names)) or not regime_names:
        raise ValueError("Anchor regime names must be non-empty and unique")
    if sum(regime.weight for regime in cfg.anchors.regimes) <= 0:
        raise ValueError("Anchor regime weights must sum to a positive value")
    for regime in cfg.anchors.regimes:
        if not 0.0 <= regime.release_fraction < 1.0:
            raise ValueError(f"{regime.name}: release_fraction must lie in [0, 1)")

    calibration = cfg.anatomy_calibration
    if calibration.characteristic_length_reference_mm < 0:
        raise ValueError("characteristic_length_reference_mm cannot be negative")
    if calibration.amplitude_scale_exponent < 0:
        raise ValueError("amplitude_scale_exponent cannot be negative")
    scale_low, scale_high = map(float, calibration.amplitude_scale_bounds)
    if not 0 < scale_low <= scale_high:
        raise ValueError("Invalid amplitude_scale_bounds")
    regime_name_set = set(regime_names)
    for mapping_name in (
        "hard_area_fraction_ranges",
        "soft_area_fraction_ranges",
    ):
        mapping = getattr(calibration, mapping_name)
        if set(mapping) != regime_name_set:
            raise ValueError(f"{mapping_name} must define exactly every anchor regime")
        for name, bounds in mapping.items():
            if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] < 1:
                raise ValueError(f"{name}: invalid {mapping_name} bounds")
    if set(calibration.load_area_fraction_ranges) != VALID_FAMILIES:
        raise ValueError("load_area_fraction_ranges must define exactly every family")
    for family, bounds in calibration.load_area_fraction_ranges.items():
        if len(bounds) != 2 or not 0 < bounds[0] < bounds[1] < 1:
            raise ValueError(f"{family}: invalid load area fraction bounds")
    if not 0 < calibration.load_center_quality_quantile <= 1:
        raise ValueError("load_center_quality_quantile must lie in (0, 1]")
    if calibration.capacity_bisection_steps < 0:
        raise ValueError("capacity_bisection_steps cannot be negative")
    if not 0 <= calibration.capacity_min_bin_fraction < 1:
        raise ValueError("capacity_min_bin_fraction must lie in [0, 1)")

    if not cfg.displacement.bins:
        raise ValueError("At least one displacement bin is required")
    bin_names = []
    bin_weights = []
    for name, lower, upper, weight in cfg.displacement.bins:
        if not name or not 0 <= lower < upper or weight < 0:
            raise ValueError(
                f"Invalid displacement bin: {(name, lower, upper, weight)}"
            )
        bin_names.append(name)
        bin_weights.append(weight)
    if len(bin_names) != len(set(bin_names)) or sum(bin_weights) <= 0:
        raise ValueError(
            "Displacement bins need unique names and positive total weight"
        )

    validity = cfg.validity
    if not 0 < validity.min_jacobian < 1 < validity.max_jacobian:
        raise ValueError("Jacobian limits must bracket 1")
    if not 0 < validity.min_global_volume_ratio < 1 < validity.max_global_volume_ratio:
        raise ValueError("Global volume limits must bracket 1")
    if validity.max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be positive")
    if validity.max_anchor_realizations < 1:
        raise ValueError("max_anchor_realizations must be positive")
    if validity.max_intent_resamples < 0:
        raise ValueError("max_intent_resamples cannot be negative")
    if validity.max_sample_replacements < 0:
        raise ValueError("max_sample_replacements cannot be negative")
    for family, bounds in validity.family_spread_bounds.items():
        if family not in VALID_FAMILIES or len(bounds) != 4:
            raise ValueError(f"{family}: invalid family_spread_bounds")
        active_min, active_max, halfmax_min, peak_to_mean_max = map(float, bounds)
        if not (0.0 <= active_min < active_max <= 1.0):
            raise ValueError(f"{family}: invalid active-fraction bounds")
        if not (0.0 <= halfmax_min <= active_max):
            raise ValueError(f"{family}: invalid halfmax minimum")
        if peak_to_mean_max <= 1.0:
            raise ValueError(f"{family}: invalid peak-to-mean maximum")
    if cfg.solver.backend != "linear":
        raise NotImplementedError("This release implements only the linear backend")


def load_config(path: str | Path) -> ProjectConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = config_from_dict(json.load(handle))
    return cfg


def config_from_dict(data: Mapping[str, Any]) -> ProjectConfig:
    cfg = _tupleize(dict(data))
    validate_config(cfg, check_paths=False)
    return cfg


def save_config(path: str | Path, cfg: ProjectConfig) -> None:
    validate_config(cfg, check_paths=False)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
