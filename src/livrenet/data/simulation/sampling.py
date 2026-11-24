from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import AnchorRegimeConfig, ProjectConfig
from .geometry import area_uniform_vertex, geodesic_distances, normalize


@dataclass(frozen=True)
class SampleIntent:
    """Statistical identity fixed before geometric retries.

    Keeping this immutable prevents validity rejection from silently replacing a
    difficult lobe/severe sample with an easier family or an identity sample.
    """

    split: str
    is_ood: bool
    is_identity: bool
    family: str
    mechanism: str
    sign: str
    direction_mode: str
    profile: str
    radius_ratio: float
    target_bin: str
    target_max_mm: float
    anchor_regime: str
    target_bin_lower_mm: float = 0.0
    target_bin_upper_mm: float = 0.0
    target_load_area_fraction: float = 0.0
    amplitude_scale: float = 1.0


@dataclass
class AnchorState:
    hard_mask: np.ndarray
    soft_weight: np.ndarray
    distance_to_anatomy: np.ndarray
    hard_radius: float
    soft_radius: float
    soft_relative_stiffness: float
    regime: str
    anatomy_centroid: np.ndarray
    released_fraction: float
    hard_area_fraction: float = 0.0
    soft_area_fraction: float = 0.0
    soft_effective_area_fraction: float = 0.0
    target_hard_area_fraction: float = 0.0
    target_soft_area_fraction: float = 0.0
    vertex_areas: np.ndarray | None = None
    surface_area: float = 0.0
    surface_center_quality: np.ndarray | None = None


@dataclass
class LoadState:
    family: str
    mechanism: str
    sign: str
    center_vertex: int
    center_point: np.ndarray
    normal: np.ndarray
    direction: np.ndarray
    direction_mode: str
    profile: str
    radius: float
    geodesic_distance: np.ndarray
    load_weight: np.ndarray
    support_area_fraction: float = 0.0
    target_area_fraction: float = 0.0
    center_reference_quality: float = 0.0
    center_quality_threshold: float = 0.0


@dataclass(frozen=True)
class AnatomySupport:
    """Anatomical distances that are invariant across generated samples."""

    distance_to_surface: np.ndarray
    centroid: np.ndarray
    vertex_areas: np.ndarray
    surface_area: float
    enclosed_volume: float
    characteristic_length: float
    surface_center_quality: np.ndarray | None = None


def surface_characteristics(surface_points, faces):
    """Return barycentric vertex areas, total area, volume and V^(1/3)."""

    points = np.asarray(surface_points, dtype=float)
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim == 1:
        triangles = triangles.reshape(-1, 4)[:, 1:]
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Expected triangular surface faces")
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    triangle_areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    vertex_areas = np.zeros(len(points), dtype=float)
    for corner in range(3):
        np.add.at(vertex_areas, triangles[:, corner], triangle_areas / 3.0)
    surface_area = float(triangle_areas.sum())
    signed_volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
    enclosed_volume = abs(signed_volume)
    characteristic_length = (
        enclosed_volume ** (1.0 / 3.0)
        if enclosed_volume > 1.0e-12
        else np.sqrt(max(surface_area, 1.0e-12))
    )
    return vertex_areas, surface_area, enclosed_volume, float(characteristic_length)


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        raise ValueError("Weighted quantile has no finite positive-weight values")
    values = values[finite]
    weights = weights[finite]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    threshold = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")), len(order) - 1
    )
    return float(values[order[index]])


def prepare_anatomy_support(
    surface_points, anatomical_paths, *, faces=None
) -> AnatomySupport:
    """Read vascular structures and build their surface-distance field once."""

    import pyvista as pv

    points = []
    for path in anatomical_paths.values():
        mesh = pv.read(path)
        mesh = mesh if isinstance(mesh, pv.PolyData) else mesh.extract_surface()
        if mesh.n_points:
            points.append(np.asarray(mesh.points, dtype=float))
    if not points:
        raise ValueError("At least one non-empty anatomical structure is required")
    anatomy = np.concatenate(points, axis=0)
    distance, _ = cKDTree(anatomy).query(np.asarray(surface_points), k=1)
    if faces is None:
        vertex_areas = np.ones(len(surface_points), dtype=float)
        surface_area = float(len(surface_points))
        enclosed_volume = 0.0
        characteristic_length = float(
            np.linalg.norm(np.ptp(np.asarray(surface_points), axis=0))
        )
    else:
        (
            vertex_areas,
            surface_area,
            enclosed_volume,
            characteristic_length,
        ) = surface_characteristics(surface_points, faces)
    return AnatomySupport(
        distance_to_surface=np.asarray(distance, dtype=float),
        centroid=np.mean(anatomy, axis=0),
        vertex_areas=vertex_areas,
        surface_area=surface_area,
        enclosed_volume=enclosed_volume,
        characteristic_length=characteristic_length,
    )


def _weighted_choice(mapping, rng):
    names = list(mapping)
    weights = np.asarray([mapping[name] for name in names], dtype=float)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid sampling weights")
    return str(rng.choice(names, p=weights / weights.sum()))


def choose_anchor_regime(cfg: ProjectConfig, rng) -> AnchorRegimeConfig:
    regimes = cfg.anchors.regimes
    weights = np.asarray([regime.weight for regime in regimes], dtype=float)
    return regimes[int(rng.choice(len(regimes), p=weights / weights.sum()))]


def _regime_by_name(cfg: ProjectConfig, name: str) -> AnchorRegimeConfig:
    for regime in cfg.anchors.regimes:
        if regime.name == name:
            return regime
    raise KeyError(f"Unknown anchor regime: {name}")


def _apply_partial_release(surface_points, candidate_mask, fraction, centroid, rng):
    """Release a contiguous side of the attachment region.

    A random plane through the vascular centroid approximates partial surgical
    mobilization without pretending to model individual ligaments.
    """

    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    indices = np.where(candidate_mask)[0]
    if fraction <= 0 or len(indices) < 2:
        return candidate_mask, 0.0
    axis = normalize(rng.normal(size=3))
    score = (np.asarray(surface_points)[indices] - centroid) @ axis
    threshold = np.quantile(score, 1.0 - float(fraction))
    released = indices[score >= threshold]
    output = candidate_mask.copy()
    output[released] = False
    return output, float(len(released) / max(len(indices), 1))


def build_anchors(
    surface,
    anatomical_paths,
    cfg: ProjectConfig,
    rng,
    diag: float,
    ood: bool,
    *,
    regime_name: str,
    anatomy_support: AnatomySupport | None = None,
) -> AnchorState:
    support = anatomy_support or prepare_anatomy_support(
        surface.points, anatomical_paths, faces=surface.faces
    )
    anatomy_centroid = np.asarray(support.centroid, dtype=float)
    distance = np.asarray(support.distance_to_surface, dtype=float)
    regime = _regime_by_name(cfg, regime_name)

    vertex_areas = np.asarray(support.vertex_areas, dtype=float)
    total_area = float(vertex_areas.sum())
    calibration = cfg.anatomy_calibration
    target_hard_area = 0.0
    target_soft_area = 0.0
    if calibration.enabled and total_area > 0:
        target_hard_area = float(
            rng.uniform(*calibration.hard_area_fraction_ranges[regime.name])
        )
        target_soft_area = float(
            rng.uniform(*calibration.soft_area_fraction_ranges[regime.name])
        )
        # The release is applied after selecting a contiguous vascular patch;
        # compensate its expected removed fraction so the final area remains
        # close to the requested regime-level fraction.
        initial_hard_fraction = min(
            target_hard_area / max(1.0 - regime.release_fraction, 1.0e-6),
            0.98,
        )
        initial_soft_fraction = min(
            (target_soft_area + target_hard_area)
            / max(1.0 - 0.5 * regime.release_fraction, 1.0e-6),
            0.98,
        )
        hard_radius = _weighted_quantile(distance, vertex_areas, initial_hard_fraction)
        soft_radius = _weighted_quantile(distance, vertex_areas, initial_soft_fraction)
    else:
        jitter = cfg.anchors.radius_jitter
        hard_radius = (
            cfg.anchors.hard_radius_ratio
            * regime.hard_radius_scale
            * diag
            * rng.uniform(1.0 - jitter, 1.0 + jitter)
        )
        soft_radius = (
            cfg.anchors.soft_radius_ratio
            * regime.soft_radius_scale
            * diag
            * rng.uniform(1.0 - jitter, 1.0 + jitter)
        )

    hard_initial = distance <= hard_radius
    hard, released_fraction = _apply_partial_release(
        surface.points,
        hard_initial,
        regime.release_fraction,
        anatomy_centroid,
        rng,
    )
    if hard.sum() < regime.min_hard_vertices:
        eligible = np.where(hard_initial)[0]
        if len(eligible) < regime.min_hard_vertices:
            eligible = np.argsort(distance)
        selected = eligible[np.argsort(distance[eligible])[: regime.min_hard_vertices]]
        hard[selected] = True

    if calibration.enabled and total_area > 0:
        best_soft = None
        search_fraction = initial_soft_fraction
        for _ in range(6):
            candidate_radius = _weighted_quantile(
                distance, vertex_areas, search_fraction
            )
            candidate_soft = np.clip(
                1.0 - distance / max(candidate_radius, 1e-12), 0.0, 1.0
            )
            if regime.release_fraction > 0:
                kept, _ = _apply_partial_release(
                    surface.points,
                    candidate_soft > 0,
                    0.5 * regime.release_fraction,
                    anatomy_centroid,
                    rng,
                )
                candidate_soft[~kept] = 0.0
            candidate_soft[hard] = 0.0
            achieved = float(
                vertex_areas[candidate_soft > 0].sum() / max(total_area, 1.0e-12)
            )
            score = abs(achieved - target_soft_area)
            if best_soft is None or score < best_soft[0]:
                best_soft = (
                    score,
                    candidate_soft,
                    candidate_radius,
                )
            if score <= 0.005:
                break
            search_fraction = float(
                np.clip(
                    search_fraction * target_soft_area / max(achieved, 1.0e-6),
                    target_soft_area,
                    0.995,
                )
            )
        _, soft, soft_radius = best_soft
    else:
        soft = np.clip(1.0 - distance / max(soft_radius, 1e-12), 0.0, 1.0)
        if regime.release_fraction > 0:
            soft_candidate = soft > 0
            kept, _ = _apply_partial_release(
                surface.points,
                soft_candidate,
                0.5 * regime.release_fraction,
                anatomy_centroid,
                rng,
            )
            soft[~kept] = 0.0
        soft[hard] = 0.0

    relative_stiffness = (
        rng.uniform(
            cfg.anchors.soft_anchor_relative_stiffness_min,
            cfg.anchors.soft_anchor_relative_stiffness_max,
        )
        * regime.stiffness_scale
    )
    if ood:
        relative_stiffness *= rng.uniform(*cfg.splits.ood_anchor_stiffness_multiplier)

    hard_area_fraction = float(vertex_areas[hard].sum() / max(total_area, 1.0e-12))
    soft_area_fraction = float(vertex_areas[soft > 0].sum() / max(total_area, 1.0e-12))
    soft_effective_area_fraction = float(
        np.dot(vertex_areas, soft) / max(total_area, 1.0e-12)
    )
    return AnchorState(
        hard_mask=hard,
        soft_weight=soft.astype(np.float32),
        distance_to_anatomy=distance,
        hard_radius=float(hard_radius),
        soft_radius=float(soft_radius),
        soft_relative_stiffness=float(relative_stiffness),
        regime=regime.name,
        anatomy_centroid=anatomy_centroid,
        released_fraction=released_fraction,
        hard_area_fraction=hard_area_fraction,
        soft_area_fraction=soft_area_fraction,
        soft_effective_area_fraction=soft_effective_area_fraction,
        target_hard_area_fraction=target_hard_area,
        target_soft_area_fraction=target_soft_area,
        vertex_areas=vertex_areas,
        surface_area=total_area,
        surface_center_quality=support.surface_center_quality,
    )


def _tangent(normal, rng):
    normal = normalize(normal)
    for _ in range(20):
        vector = rng.normal(size=3)
        tangent = vector - np.dot(vector, normal) * normal
        if np.linalg.norm(tangent) > 1e-10:
            return normalize(tangent)
    fallback = [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0]
    return normalize(np.cross(normal, fallback))


def _sample_mechanism(family, push, rng):
    sign = "push" if push else "pull"
    if family == "local_narrow":
        if push:
            mechanism = "indentation_push"
            mode = str(
                rng.choice(["normal", "oblique", "tangential"], p=[0.65, 0.30, 0.05])
            )
        else:
            mechanism = str(rng.choice(["local_lift", "local_drag"], p=[0.65, 0.35]))
            mode = (
                "tangential"
                if mechanism.endswith("drag")
                else str(rng.choice(["normal", "oblique"], p=[0.60, 0.40]))
            )
    elif family == "local_broad":
        if push:
            mechanism = "broad_compression"
            mode = str(
                rng.choice(["normal", "oblique", "tangential"], p=[0.50, 0.40, 0.10])
            )
        else:
            mechanism = str(
                rng.choice(["broad_retraction", "broad_drag"], p=[0.65, 0.35])
            )
            mode = (
                "tangential"
                if mechanism.endswith("drag")
                else str(rng.choice(["normal", "oblique"], p=[0.45, 0.55]))
            )
    elif family == "lobe_motion":
        if push:
            mechanism = "lobe_push"
            mode = str(rng.choice(["axial", "oblique"], p=[0.55, 0.45]))
        else:
            mechanism = str(rng.choice(["lobe_lift", "lobe_drag"], p=[0.55, 0.45]))
            mode = (
                "tangential"
                if mechanism.endswith("drag")
                else str(rng.choice(["axial", "oblique"], p=[0.55, 0.45]))
            )
    else:
        raise ValueError(f"Unknown deformation family: {family}")
    return mechanism, sign, mode


def _sample_target(
    cfg: ProjectConfig,
    family: str,
    diagonal: float,
    rng,
    *,
    target_bin: str | None = None,
    amplitude_scale: float = 1.0,
    return_bounds: bool = False,
):
    family_max = min(
        float(cfg.deformation.family_max_displacement_mm[family]) * amplitude_scale,
        float(cfg.deformation.family_max_displacement_ratio[family]) * diagonal,
    )
    available = []
    for name, lower, upper, weight in cfg.displacement.bins:
        scaled_lower = float(lower) * amplitude_scale
        scaled_upper = float(upper) * amplitude_scale
        clipped_upper = min(scaled_upper, family_max)
        if clipped_upper > scaled_lower:
            available.append((name, scaled_lower, clipped_upper, float(weight)))
    if not available:
        raise ValueError(f"No displacement bin available for family {family}")
    if target_bin is not None:
        matches = [item for item in available if item[0] == target_bin]
        if not matches:
            raise ValueError(
                f"Displacement bin {target_bin!r} is unavailable for family "
                f"{family!r} at this anatomy size"
            )
        chosen = matches[0]
        sampled = float(rng.uniform(chosen[1], chosen[2]))
        return (
            (chosen[0], sampled, chosen[1], chosen[2])
            if return_bounds
            else (chosen[0], sampled)
        )
    weights = np.asarray([item[3] for item in available], dtype=float)
    chosen = available[int(rng.choice(len(available), p=weights / weights.sum()))]
    sampled = float(rng.uniform(chosen[1], chosen[2]))
    return (
        (chosen[0], sampled, chosen[1], chosen[2])
        if return_bounds
        else (chosen[0], sampled)
    )


def anatomy_amplitude_scale(cfg: ProjectConfig, characteristic_length: float) -> float:
    calibration = cfg.anatomy_calibration
    reference = float(calibration.characteristic_length_reference_mm)
    if not calibration.enabled or reference <= 0 or characteristic_length <= 0:
        return 1.0
    raw = (float(characteristic_length) / reference) ** float(
        calibration.amplitude_scale_exponent
    )
    return float(np.clip(raw, *calibration.amplitude_scale_bounds))


def sample_intent(
    cfg: ProjectConfig,
    rng,
    *,
    diagonal: float,
    split: str,
    is_ood: bool,
    force_identity: bool | None = None,
    family: str | None = None,
    target_bin: str | None = None,
    characteristic_length: float = 0.0,
) -> SampleIntent:
    amplitude_scale = anatomy_amplitude_scale(cfg, characteristic_length)
    regime = choose_anchor_regime(cfg, rng)
    is_identity = (
        bool(rng.random() < cfg.displacement.identity_fraction)
        if force_identity is None
        else bool(force_identity)
    )
    if is_identity and (family is not None or target_bin not in (None, "identity")):
        raise ValueError(
            "An identity intent cannot request a deformation family or bin"
        )
    if is_identity:
        return SampleIntent(
            split=split,
            is_ood=is_ood,
            is_identity=True,
            family="identity",
            mechanism="identity",
            sign="none",
            direction_mode="none",
            profile="none",
            radius_ratio=0.0,
            target_bin="identity",
            target_max_mm=0.0,
            anchor_regime=regime.name,
            amplitude_scale=amplitude_scale,
        )

    if family is None:
        possible = []
        for name, weight in cfg.deformation.family_weights.items():
            try:
                _sample_target(
                    cfg,
                    name,
                    diagonal,
                    rng,
                    target_bin=target_bin,
                    amplitude_scale=amplitude_scale,
                )
            except ValueError:
                continue
            possible.append((name, float(weight)))
        if not possible:
            raise ValueError(
                f"No deformation family can realize bin {target_bin!r} for this anatomy"
            )
        names = [item[0] for item in possible]
        weights = np.asarray([item[1] for item in possible], dtype=float)
        family = str(rng.choice(names, p=weights / weights.sum()))
    push = bool(rng.random() < cfg.deformation.push_probability[family])
    mechanism, sign, mode = _sample_mechanism(family, push, rng)
    target_bin, target_max, target_lower, target_upper = _sample_target(
        cfg,
        family,
        diagonal,
        rng,
        target_bin=target_bin,
        amplitude_scale=amplitude_scale,
        return_bounds=True,
    )
    family_profiles = cfg.deformation.profile_weights_by_family.get(family)
    if family_profiles:
        profile = _weighted_choice(family_profiles, rng)
    else:
        profile = str(
            rng.choice(
                cfg.deformation.profile_choices,
                p=np.asarray(cfg.deformation.profile_weights, dtype=float)
                / np.sum(cfg.deformation.profile_weights),
            )
        )

    radius_low, radius_high = map(float, cfg.deformation.radius_ratio_ranges[family])
    family_max = min(
        float(cfg.deformation.family_max_displacement_mm[family]) * amplitude_scale,
        float(cfg.deformation.family_max_displacement_ratio[family]) * diagonal,
    )
    family_min = min(float(item[1]) for item in cfg.displacement.bins) * amplitude_scale
    amplitude_phase = float(
        np.clip(
            (target_max - family_min) / max(family_max - family_min, 1.0e-12),
            0.0,
            1.0,
        )
    )
    coupling = float(cfg.deformation.radius_amplitude_coupling.get(family, 0.0))
    # Large displacements require a sufficiently broad application region.
    # This changes the load geometry, not the requested max-displacement bin.
    adaptive_low = radius_low + coupling * amplitude_phase * (radius_high - radius_low)
    radius_ratio = float(rng.uniform(adaptive_low, radius_high))
    area_low, area_high = map(
        float, cfg.anatomy_calibration.load_area_fraction_ranges[family]
    )
    adaptive_area_low = area_low + coupling * amplitude_phase * (area_high - area_low)
    target_load_area = float(rng.uniform(adaptive_area_low, area_high))
    if is_ood:
        ood_multiplier = float(rng.uniform(*cfg.splits.ood_radius_multiplier))
        radius_ratio *= ood_multiplier
        target_load_area = float(
            np.clip(target_load_area * ood_multiplier**2, 1.0e-4, 0.95)
        )
    return SampleIntent(
        split=split,
        is_ood=is_ood,
        is_identity=False,
        family=family,
        mechanism=mechanism,
        sign=sign,
        direction_mode=mode,
        profile=profile,
        radius_ratio=radius_ratio,
        target_bin=target_bin,
        target_max_mm=target_max,
        anchor_regime=regime.name,
        target_bin_lower_mm=target_lower,
        target_bin_upper_mm=target_upper,
        target_load_area_fraction=target_load_area,
        amplitude_scale=amplitude_scale,
    )


def _direction(intent, normal, center, anchors, cfg, rng):
    push = intent.sign == "push"
    if intent.family == "lobe_motion":
        outward = normalize(np.asarray(center) - anchors.anatomy_centroid)
        axis = -outward if push else outward
        tangent = _tangent(outward, rng)
    else:
        axis = -normal if push else normal
        tangent = _tangent(normal, rng)

    mode = intent.direction_mode
    if mode in ("normal", "axial"):
        return axis
    if mode == "oblique":
        bounds = (
            cfg.splits.ood_oblique_angle_deg
            if intent.is_ood
            else (
                cfg.deformation.oblique_angle_min_deg,
                cfg.deformation.oblique_angle_max_deg,
            )
        )
        angle = np.deg2rad(rng.uniform(*bounds))
        return normalize(np.cos(angle) * axis + np.sin(angle) * tangent)
    if mode == "tangential":
        alpha = rng.uniform(
            -cfg.deformation.tangential_normal_component_max,
            cfg.deformation.tangential_normal_component_max,
        )
        return normalize(tangent + alpha * axis)
    raise ValueError(mode)


def _profile(distance, radius, kind):
    q = np.asarray(distance, dtype=float) / max(float(radius), 1e-12)
    inside = q <= 1.0
    weight = np.zeros_like(q, dtype=float)
    if kind == "gaussian":
        weight[inside] = np.exp(-0.5 * (q[inside] / 0.42) ** 2)
    elif kind == "cosine":
        weight[inside] = 0.5 * (1.0 + np.cos(np.pi * q[inside]))
    elif kind == "wendland":
        weight[inside] = (1.0 - q[inside]) ** 4 * (4.0 * q[inside] + 1.0)
    elif kind == "flat_smooth":
        # Broad distributed traction: flat near the centre, continuously
        # tapered to zero at the patch boundary. Both value and slope vanish
        # at q=1, avoiding the ring discontinuity of a clipped flat load.
        local_q = q[inside]
        weight[inside] = (1.0 - local_q**4) ** 2
    else:
        raise ValueError(kind)
    maximum = weight.max()
    return weight / maximum if maximum > 0 else weight


def sample_deformation_load(
    surface,
    graph,
    anchors: AnchorState,
    intent: SampleIntent,
    cfg: ProjectConfig,
    rng,
    diag: float,
) -> LoadState:
    avoid = cfg.anchors.load_avoid_radius_ratio * diag
    valid = (~anchors.hard_mask) & (anchors.distance_to_anatomy > avoid)
    candidates = np.where(valid)[0]
    if not len(candidates):
        raise RuntimeError("No valid load centers")

    quality_threshold = float("inf")
    if anchors.surface_center_quality is not None:
        quality = np.asarray(anchors.surface_center_quality, dtype=float)
        finite_quality = np.isfinite(quality[candidates])
        if np.any(finite_quality):
            quality_threshold = float(
                np.quantile(
                    quality[candidates][finite_quality],
                    cfg.anatomy_calibration.load_center_quality_quantile,
                )
            )
            screened = candidates[
                np.isfinite(quality[candidates])
                & (quality[candidates] <= quality_threshold)
            ]
            if len(screened) >= cfg.deformation.min_patch_vertices:
                candidates = screened

    if intent.family == "lobe_motion":
        threshold = np.quantile(
            anchors.distance_to_anatomy[candidates],
            cfg.deformation.lobe_peripheral_quantile,
        )
        peripheral = candidates[anchors.distance_to_anatomy[candidates] >= threshold]
        if len(peripheral) >= cfg.deformation.min_patch_vertices:
            candidates = peripheral

    fallback_radius = intent.radius_ratio * diag
    family_radius_high = (
        float(cfg.deformation.radius_ratio_ranges[intent.family][1]) * diag
    )
    # With area calibration the old per-family radius is no longer the family
    # definition. Use the largest configured geodesic search envelope so the
    # requested area is actually reachable on elongated/irregular anatomies.
    radius_high = (
        max(float(bounds[1]) for bounds in cfg.deformation.radius_ratio_ranges.values())
        * diag
        if cfg.anatomy_calibration.enabled
        else family_radius_high
    )
    # A valid centre can lie on a locally coarse part of the tetrahedral
    # boundary. Finding another centre is cheap geometry sampling, not a failed
    # FEM realization, so do it here rather than consuming a generation retry.
    center_vertex = None
    geodesic = None
    weight = None
    radius = None
    achieved_area = 0.0
    vertex_areas = (
        np.asarray(anchors.vertex_areas, dtype=float)
        if anchors.vertex_areas is not None
        else np.ones(surface.n_points, dtype=float)
    )
    total_area = float(vertex_areas.sum())
    best = None
    for _ in range(8):
        proposed = area_uniform_vertex(surface, rng, candidates)
        if cfg.anatomy_calibration.enabled and intent.target_load_area_fraction > 0:
            proposed_geodesic = geodesic_distances(graph, proposed, radius_high)
            reachable_area = float(vertex_areas[np.isfinite(proposed_geodesic)].sum())
            local_quantile = min(
                intent.target_load_area_fraction
                * total_area
                / max(reachable_area, 1.0e-12),
                1.0,
            )
            proposed_radius = min(
                _weighted_quantile(
                    proposed_geodesic,
                    vertex_areas,
                    local_quantile,
                ),
                radius_high,
            )
        else:
            proposed_radius = fallback_radius
            proposed_geodesic = geodesic_distances(graph, proposed, proposed_radius)
        proposed_weight = _profile(proposed_geodesic, proposed_radius, intent.profile)
        patch = proposed_weight > 0
        if int(np.sum(patch)) < cfg.deformation.min_patch_vertices:
            continue
        proposed_area = float(vertex_areas[patch].sum() / max(total_area, 1.0e-12))
        score = abs(proposed_area - intent.target_load_area_fraction)
        if best is None or score < best[0]:
            best = (
                score,
                proposed,
                proposed_geodesic,
                proposed_weight,
                proposed_radius,
                proposed_area,
            )
        if score <= max(0.002, 0.05 * intent.target_load_area_fraction):
            break
    if best is not None:
        _, center_vertex, geodesic, weight, radius, achieved_area = best
    if center_vertex is None:
        raise RuntimeError("Load patch too small")

    center = np.asarray(surface.points)[center_vertex].copy()
    normal = normalize(np.asarray(surface.point_data["Normals"])[center_vertex])
    direction = _direction(intent, normal, center, anchors, cfg, rng)

    return LoadState(
        family=intent.family,
        mechanism=intent.mechanism,
        sign=intent.sign,
        center_vertex=int(center_vertex),
        center_point=center,
        normal=normal,
        direction=direction,
        direction_mode=intent.direction_mode,
        profile=intent.profile,
        radius=float(radius),
        geodesic_distance=geodesic,
        load_weight=weight.astype(np.float32),
        support_area_fraction=achieved_area,
        target_area_fraction=intent.target_load_area_fraction,
        center_reference_quality=(
            float(anchors.surface_center_quality[center_vertex])
            if anchors.surface_center_quality is not None
            else 0.0
        ),
        center_quality_threshold=quality_threshold,
    )


# Compatibility helper for external notebooks.
def sample_target_displacement(cfg: ProjectConfig, family: str, rng, diagonal=None):
    if diagonal is None:
        diagonal = float("inf")
    return _sample_target(cfg, family, float(diagonal), rng)
