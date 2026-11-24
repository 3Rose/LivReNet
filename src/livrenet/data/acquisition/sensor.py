from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    SensorConfig,
    SensorProfile,
    SensorRandomizationConfig,
    ToolOcclusionConfig,
)
from .geometry import Camera
from .rasterizer import RasterizedSurface


@dataclass(frozen=True)
class ToolMaskSpec:
    scenario: str
    start_uv: tuple[float, float]
    end_uv: tuple[float, float]
    radius_fraction: float
    contact_available: bool
    edge_flying_probability: float


@dataclass(frozen=True)
class SensorParameters:
    stochastic_validity_strength: float
    local_bias_probability: float
    local_bias_scale: float
    random_hole_scale: float
    flying_point_scale: float
    residual_scale: float
    gaussian_xyz_sigma_mm: float
    correlated_depth_sigma_mm: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


@dataclass
class SensorRealization:
    depth: np.ndarray
    valid: np.ndarray
    residual_px: np.ndarray
    tool_mask: np.ndarray
    random_hole_mask: np.ndarray
    flying_mask: np.ndarray
    validity_probability: np.ndarray
    correlated_depth_bias_mm: np.ndarray


def bundled_profile_path() -> Path:
    return Path(__file__).with_name("sensor_profile.json")


def load_sensor_profile(path: str | None) -> SensorProfile:
    resolved = Path(path) if path else bundled_profile_path()
    if not resolved.exists():
        raise FileNotFoundError(f"Sensor profile not found at {resolved}")
    return SensorProfile.from_json(resolved)


def realize_sensor_parameters(
    base: SensorConfig,
    randomization: SensorRandomizationConfig,
    rng: np.random.Generator,
    training: bool,
) -> SensorParameters:
    if not training or not randomization.enabled:
        return SensorParameters(
            base.stochastic_validity_strength,
            base.local_bias_probability,
            base.local_bias_scale,
            base.random_hole_scale,
            base.flying_point_scale,
            base.residual_scale,
            base.gaussian_xyz_sigma_mm,
            base.correlated_depth_sigma_mm,
        )

    def uniform(bounds):
        return float(rng.uniform(*bounds))

    return SensorParameters(
        uniform(randomization.validity_strength_range),
        uniform(randomization.local_bias_probability_range),
        uniform(randomization.local_bias_scale_range),
        uniform(randomization.hole_scale_range),
        uniform(randomization.flying_scale_range),
        uniform(randomization.residual_scale_range),
        uniform(randomization.gaussian_xyz_sigma_mm_range),
        uniform(randomization.correlated_depth_sigma_mm_range),
    )


def choose_tool_spec(
    camera: Camera,
    load_center_world: np.ndarray | None,
    config: ToolOcclusionConfig,
    rng: np.random.Generator,
) -> ToolMaskSpec:
    names = list(config.probabilities)
    probabilities = np.asarray([config.probabilities[x] for x in names], dtype=float)
    if probabilities.sum() <= 0:
        raise ValueError("Tool probabilities must contain positive mass")
    probabilities /= probabilities.sum()
    scenario = str(rng.choice(names, p=probabilities))
    contact_available = load_center_world is not None
    if contact_available:
        uv, z = camera.project(np.asarray(load_center_world, dtype=float)[None, :])
        contact = np.array([uv[0, 0] / camera.width, uv[0, 1] / camera.height])
        if (
            z[0] <= camera.near
            or not np.isfinite(contact).all()
            or np.any(contact < 0.0)
            or np.any(contact > 1.0)
        ):
            contact_available = False
    if not contact_available:
        contact = rng.uniform([0.2, 0.2], [0.8, 0.8])
        if scenario in {"partial_contact", "total_contact"}:
            scenario = "distractor"
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    direction = np.array([math.cos(theta), math.sin(theta)])
    radius = float(rng.uniform(*config.width_fraction_range))
    if scenario == "total_contact":
        start, end = contact - 0.45 * direction, contact + 0.45 * direction
    elif scenario == "partial_contact":
        offset = float(rng.uniform(*config.partial_offset_fraction)) * radius
        normal = np.array([-direction[1], direction[0]])
        shifted = contact + offset * normal
        start, end = shifted - 0.45 * direction, shifted + 0.45 * direction
    elif scenario == "distractor":
        center = rng.uniform([0.15, 0.15], [0.85, 0.85])
        if contact_available and np.linalg.norm(center - contact) < 4.0 * radius:
            center = np.clip(contact + np.array([0.35, -0.35]), 0.1, 0.9)
        start, end = center - 0.45 * direction, center + 0.45 * direction
    else:
        start = end = contact
        scenario = "none"
    return ToolMaskSpec(
        scenario=scenario,
        start_uv=(float(start[0]), float(start[1])),
        end_uv=(float(end[0]), float(end[1])),
        radius_fraction=radius,
        contact_available=contact_available,
        edge_flying_probability=float(config.edge_flying_probability),
    )


def tool_mask(shape: tuple[int, int], spec: ToolMaskSpec) -> np.ndarray:
    h, w = shape
    if spec.scenario == "none":
        return np.zeros(shape, dtype=bool)
    yy, xx = np.mgrid[:h, :w]
    p = np.stack([(xx + 0.5) / w, (yy + 0.5) / h], axis=-1)
    a, b = np.asarray(spec.start_uv), np.asarray(spec.end_uv)
    ab = b - a
    t = np.clip(
        np.sum((p - a) * ab, axis=-1) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0
    )
    closest = a + t[..., None] * ab
    dx = (p[..., 0] - closest[..., 0]) * (w / min(h, w))
    dy = (p[..., 1] - closest[..., 1]) * (h / min(h, w))
    return dx * dx + dy * dy <= spec.radius_fraction**2


def _lookup(values: np.ndarray, edges: list[float], rates: list[float]) -> np.ndarray:
    ids = np.clip(np.digitize(values, np.asarray(edges)[1:-1]), 0, len(rates) - 1)
    return np.asarray(rates, dtype=float)[ids]


def _edge_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid, depth, 0.0)
    gx = np.abs(np.diff(safe, axis=1, prepend=safe[:, :1]))
    gy = np.abs(np.diff(safe, axis=0, prepend=safe[:1, :]))
    scale = np.median(depth[valid]) if valid.any() else 1.0
    return (gx + gy) > max(0.01 * scale, 1e-8)


def _boundary(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    out[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    out[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    out[1:, :] |= mask[1:, :] != mask[:-1, :]
    out[:-1, :] |= mask[1:, :] != mask[:-1, :]
    return out


def _random_holes(
    shape: tuple[int, int],
    surface_valid: np.ndarray,
    profile: SensorProfile,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = shape
    quantiles = np.asarray(
        profile.spatial_artifacts["invalid_component_area_fraction_quantiles"],
        dtype=float,
    )
    quantiles = quantiles[np.isfinite(quantiles) & (quantiles > 0)]
    if quantiles.size == 0 or scale <= 0:
        return np.zeros(shape, dtype=bool)
    count = int(np.clip(rng.poisson(2.0 * scale), 0, 10))
    yy, xx = np.mgrid[:h, :w]
    result = np.zeros(shape, dtype=bool)
    valid_ids = np.argwhere(surface_valid)
    if not valid_ids.size:
        return result
    for _ in range(count):
        cy, cx = valid_ids[int(rng.integers(len(valid_ids)))]
        fraction = float(rng.choice(quantiles)) * scale
        area_px = max(4.0, fraction * float(surface_valid.sum()))
        aspect = float(np.exp(rng.normal(0.0, 0.45)))
        ry = math.sqrt(area_px / (math.pi * aspect))
        rx = ry * aspect
        result |= ((xx - cx) / max(rx, 1.0)) ** 2 + (
            (yy - cy) / max(ry, 1.0)
        ) ** 2 <= 1.0
    return result & surface_valid


def _smooth_random_field(
    shape: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    """Low-frequency image-space field without external dependencies."""
    h, w = shape
    gh, gw = 6, 8
    coarse = rng.normal(size=(gh, gw))
    x_old = np.linspace(0.0, 1.0, gw)
    x_new = np.linspace(0.0, 1.0, w)
    tmp = np.vstack([np.interp(x_new, x_old, row) for row in coarse])
    y_old = np.linspace(0.0, 1.0, gh)
    y_new = np.linspace(0.0, 1.0, h)
    field = np.column_stack([np.interp(y_new, y_old, tmp[:, x]) for x in range(w)])
    field -= field.mean()
    field /= max(float(field.std()), 1e-8)
    return field


def apply_sensor_model(
    raster: RasterizedSurface,
    camera: Camera,
    profile: SensorProfile,
    base_config: SensorConfig,
    parameters: SensorParameters,
    mask_spec: ToolMaskSpec,
    seed: int,
    initial_valid: np.ndarray | None = None,
) -> SensorRealization:
    rng = np.random.default_rng(seed)
    surface = raster.valid & (raster.depth > 0)
    if initial_valid is not None:
        surface &= np.asarray(initial_valid, dtype=bool)
    z_mm = raster.depth.astype(float) * base_config.mesh_unit_to_mm
    valid_cfg = profile.validity_model
    overall = max(float(valid_cfg["overall_trusted_rate"]), 1e-4)
    low, high = np.asarray(valid_cfg["depth_normalization_mm"], dtype=float)
    depth_n = np.clip((z_mm - low) / max(high - low, 1e-6), 0.0, 1.0)
    yy, xx = np.mgrid[: camera.height, : camera.width]
    radial = np.sqrt(
        ((xx - camera.cx) / max(camera.width / 2.0, 1.0)) ** 2
        + ((yy - camera.cy) / max(camera.height / 2.0, 1.0)) ** 2
    )
    depth_rate = _lookup(
        depth_n, valid_cfg["depth_edges"], valid_cfg["depth_success_rate"]
    )
    incidence_rate = _lookup(
        raster.incidence,
        valid_cfg["incidence_edges"],
        valid_cfg["incidence_success_rate"],
    )
    radial_rate = _lookup(
        radial, valid_cfg["radial_edges"], valid_cfg["radial_success_rate"]
    )
    scene_edge = _edge_mask(raster.depth, surface)
    edge_rate = np.where(
        scene_edge,
        valid_cfg["depth_edge_success_rate"],
        valid_cfg["smooth_region_success_rate"],
    )
    ratios = np.stack(
        [
            depth_rate / overall,
            incidence_rate / overall,
            radial_rate / overall,
            edge_rate / overall,
        ]
    )
    probability = overall * np.exp(np.mean(np.log(np.clip(ratios, 0.05, 20.0)), axis=0))
    probability = np.clip(probability, 0.01, 0.995)
    probability = 1.0 - parameters.stochastic_validity_strength * (1.0 - probability)
    probability = np.clip(probability, 0.0, 1.0)

    tmask = tool_mask(surface.shape, mask_spec)
    holes = _random_holes(
        surface.shape, surface & ~tmask, profile, parameters.random_hole_scale, rng
    )
    valid = surface & ~tmask & ~holes & (rng.random(surface.shape) < probability)

    baseline_mm = float(profile.camera["baseline_mm"])
    source_width = float(profile.image["width"])
    pixel_scale = camera.width / source_width
    disparity = camera.fx * baseline_mm / np.maximum(z_mm, 1e-6)
    q_prob = np.asarray(profile.residual_model["quantile_probabilities"], dtype=float)
    q_residual = np.asarray(
        profile.residual_model["signed_disparity_residual_px_quantiles"], dtype=float
    )
    q_residual *= pixel_scale * parameters.residual_scale
    residual_uniform = rng.random(surface.shape)
    residual = np.zeros(surface.shape, dtype=float)
    conditional = profile.residual_model.get(
        "signed_disparity_residual_px_quantiles_by_depth"
    )
    if conditional:
        residual_edges = np.asarray(profile.residual_model["depth_edges"], dtype=float)
        residual_ids = np.clip(
            np.digitize(depth_n, residual_edges[1:-1]), 0, len(conditional) - 1
        )
        for depth_id, row in enumerate(conditional):
            selected = residual_ids == depth_id
            residual[selected] = np.interp(
                residual_uniform[selected],
                q_prob,
                np.asarray(row, dtype=float) * pixel_scale * parameters.residual_scale,
            )
    else:
        residual = np.interp(residual_uniform, q_prob, q_residual)

    if (
        parameters.local_bias_probability > 0
        and rng.random() < parameters.local_bias_probability
    ):
        centers = rng.uniform(
            [0.15, 0.15], [0.85, 0.85], size=(int(rng.integers(1, 4)), 2)
        )
        amps = rng.normal(
            0.0,
            np.quantile(np.abs(q_residual), 0.9) * parameters.local_bias_scale,
            len(centers),
        )
        field = np.zeros(surface.shape, dtype=float)
        for center, amp in zip(centers, amps):
            radius = float(rng.uniform(0.08, 0.25))
            d2 = (
                (xx / camera.width - center[0]) ** 2
                + (yy / camera.height - center[1]) ** 2
            ) / radius**2
            field += amp * np.exp(-0.5 * d2)
        residual += field

    noisy_disparity = disparity + residual
    valid &= noisy_disparity > 1e-4
    noisy_z_mm = camera.fx * baseline_mm / np.maximum(noisy_disparity, 1e-4)
    correlated_bias = np.zeros(surface.shape, dtype=float)
    if parameters.correlated_depth_sigma_mm > 0:
        correlated_bias = (
            _smooth_random_field(surface.shape, rng)
            * parameters.correlated_depth_sigma_mm
        )
        noisy_z_mm += correlated_bias
    noisy_depth = noisy_z_mm / base_config.mesh_unit_to_mm

    tool_edge = _boundary(tmask)
    candidate_flying = valid & (scene_edge | tool_edge)
    base_flying = float(
        profile.spatial_artifacts["gross_outlier_rate_on_returned_median"]
    )
    probability_flying = np.where(
        tool_edge, mask_spec.edge_flying_probability, base_flying
    )
    probability_flying = np.clip(
        probability_flying * parameters.flying_point_scale, 0.0, 0.35
    )
    flying = candidate_flying & (rng.random(surface.shape) < probability_flying)
    if flying.any():
        direction = rng.choice(np.array([-1.0, 1.0]), size=int(flying.sum()))
        magnitude = rng.uniform(0.02, 0.12, size=int(flying.sum()))
        noisy_depth[flying] *= 1.0 + direction * magnitude

    noisy_depth[~valid] = 0.0
    residual[~valid] = 0.0
    correlated_bias[~valid] = 0.0
    return SensorRealization(
        depth=noisy_depth.astype(np.float32),
        valid=valid,
        residual_px=residual.astype(np.float32),
        tool_mask=tmask,
        random_hole_mask=holes,
        flying_mask=flying,
        validity_probability=probability.astype(np.float32),
        correlated_depth_bias_mm=correlated_bias.astype(np.float32),
    )
