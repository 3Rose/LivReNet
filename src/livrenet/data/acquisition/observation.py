"""Same acquisition/noise random draws in both modes; diagnostic work is optional."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .generator import (
    ObservationBlueprint,
    _anchor_visible,
    _canonical_input_mask,
    _gather_point_attributes,
)
from .geometry import sample_camera
from .sensor import realize_sensor_parameters, choose_tool_spec, apply_sensor_model
from .rasterizer import backproject
from .deformation import compute_deformation_regions, splat_vertex_support
from .metrics import compute_observation_metrics
from .render import stereo_visibility


@dataclass
class Packet:
    arrays: dict
    metrics: dict
    masks: dict


def make_blueprint(
    template,
    engine,
    profile,
    rng,
    bid,
    *,
    weights=None,
    policy="natural_reference_only",
    backend="native",
):
    parameters = realize_sensor_parameters(
        engine.sensor, engine.sensor_randomization, rng, False
    )
    best, best_stereo, best_score, best_cov, best_attempt = (
        None,
        None,
        float("inf"),
        0.0,
        0,
    )
    accepted = False
    lo, hi = engine.camera.template_visible_surface_range
    for attempt in range(1, engine.camera.max_sampling_attempts + 1):
        camera = sample_camera(template, engine.camera, rng, surface_weights=weights)
        stereo = stereo_visibility(
            template,
            camera,
            profile,
            engine.sensor.mesh_unit_to_mm,
            engine.stereo,
            backend,
        )
        coverage = float(
            stereo.left.surface_weight[stereo.mutual_mask].sum()
            / max(template.areas.sum(), 1e-12)
        )
        anchor = not engine.camera.require_anchor_visible or _anchor_visible(
            template, camera, stereo.left.face_id
        )
        score = abs(coverage - engine.camera.target_surface_coverage_center) / max(
            engine.camera.target_surface_coverage_sigma, 1e-6
        ) + (0.0 if anchor else 10.0)
        if score < best_score:
            best, best_stereo, best_score, best_cov, best_attempt = (
                camera,
                stereo,
                score,
                coverage,
                attempt,
            )
        if lo <= coverage <= hi and anchor:
            best, best_stereo, best_cov, best_attempt = (
                camera,
                stereo,
                coverage,
                attempt,
            )
            accepted = True
            break
    if best is None:
        raise ValueError("No camera proposal")
    # Natural acquisition never reads load/peak/field metadata.
    load_id = template.load_center_vertex()
    tool = choose_tool_spec(
        best, None if load_id is None else template.points[load_id], engine.tool, rng
    )
    bp = ObservationBlueprint(
        best,
        int(rng.integers(0, 2**32 - 1)),
        tool,
        parameters,
        bid,
        best_cov,
        False,
        best_attempt,
        not accepted,
        policy,
    )
    return bp, best_stereo


def observe(
    mesh, template, bp, engine, profile, stereo, *, diagnostic=False, regions=None
):
    realization = apply_sensor_model(
        stereo.left,
        bp.camera,
        profile,
        engine.sensor,
        bp.sensor_parameters,
        bp.tool,
        bp.sensor_seed,
        initial_valid=stereo.mutual_mask,
    )
    rng = np.random.default_rng(bp.sensor_seed ^ 0xA5A5A5A5)
    selected = _canonical_input_mask(
        realization.valid,
        engine.sensor.canonical_point_budget,
        engine.sensor.sampling_strategy,
        rng,
    )
    raw_ids = np.flatnonzero(realization.valid)
    input_ids = np.flatnonzero(selected)
    lookup = np.full(realization.valid.size, -1, np.int64)
    lookup[raw_ids] = np.arange(len(raw_ids))
    raw_index = lookup[input_ids]
    sigma = bp.sensor_parameters.gaussian_xyz_sigma_mm / engine.sensor.mesh_unit_to_mm
    if diagnostic:
        raw = _gather_point_attributes(
            mesh,
            template,
            stereo.left,
            realization.valid,
            realization.depth,
            bp.camera,
            sigma,
            rng,
        )
        inp = {k: v[raw_index] for k, v in raw.items()}
    else:
        points, uv = backproject(realization.depth, selected, bp.camera)
        if sigma > 0 and len(raw_ids):
            # Draw the same Gaussian array as the reference before subsetting.
            # Changing mode never changes the model's points or noise seeds.
            points += rng.normal(0.0, sigma, size=(len(raw_ids), 3))[raw_index]
        y, x = uv[:, 1], uv[:, 0]
        face = stereo.left.face_id[y, x].astype(np.int64)
        bary = stereo.left.barycentric[y, x].astype(np.float32)
        inp = {
            "points": points.astype(np.float32),
            "pixel_uv": uv,
            "face_ids": face,
            "barycentric": bary,
        }
    a = {k + "_input": v for k, v in inp.items()}
    uv = inp["pixel_uv"]
    y, x = uv[:, 1], uv[:, 0]
    a["is_sensor_inlier_gt_input"] = (~realization.flying_mask[y, x]).astype(bool)
    if not diagnostic:
        return Packet(a, {}, {})
    regions = regions or compute_deformation_regions(
        mesh, engine.deformation_regions, engine.sensor.mesh_unit_to_mm
    )
    support = splat_vertex_support(
        len(mesh.points), mesh.faces, inp["face_ids"], inp["barycentric"]
    )
    p95 = (
        float(profile.residual_model["absolute_disparity_residual_px_quantiles"][95])
        * bp.camera.width
        / float(profile.image["width"])
    )
    metrics = compute_observation_metrics(
        mesh=mesh,
        raster=stereo.left,
        geometric_valid=stereo.mutual_mask,
        sensor_valid=realization.valid,
        input_valid=selected,
        residual_px=realization.residual_px,
        residual_p95_px=p95,
        regions=regions,
        raw_face_ids=raw["face_ids"],
        raw_barycentric=raw["barycentric"],
        raw_noisy_points=raw["points"],
        raw_surface_points=raw["surface_points_gt"],
        input_face_ids=inp["face_ids"],
        input_barycentric=inp["barycentric"],
        input_vertex_support=support,
        mesh_unit_to_mm=engine.sensor.mesh_unit_to_mm,
        region_config=engine.deformation_regions,
        sample_type="identity" if regions.max_displacement <= 1e-12 else "deformed",
    )
    a.update(
        confidence_input=realization.validity_probability[y, x].astype(np.float32),
        incidence_input=stereo.left.incidence[y, x],
        surface_weight_input=stereo.left.surface_weight[y, x],
        residual_px_input=realization.residual_px[y, x].astype(np.float32),
        right_uv_input=stereo.right_uv[y, x],
        correlated_depth_bias_mm_input=realization.correlated_depth_bias_mm[
            y, x
        ].astype(np.float32),
        is_flying_point_gt_input=realization.flying_mask[y, x],
        is_depth_corrupted_gt_input=(np.abs(realization.residual_px[y, x]) > 1e-12)
        | (np.abs(realization.correlated_depth_bias_mm[y, x]) > 1e-12),
        vertex_observation_weight_input=support.astype(np.float32),
    )
    noise = inp["points"].astype(float) - inp["surface_points_gt"].astype(float)
    norm = np.linalg.norm(noise, axis=1) * engine.sensor.mesh_unit_to_mm
    valid = a["is_sensor_inlier_gt_input"]
    metrics.update(
        realized_noise_rms_mm=float(np.sqrt(np.mean(norm**2))) if len(norm) else None,
        realized_noise_p95_mm=float(np.percentile(norm, 95)) if len(norm) else None,
        inlier_noise_rms_mm=float(np.sqrt(np.mean(norm[valid] ** 2)))
        if valid.any()
        else None,
        flying_point_fraction=float(np.mean(~valid)) if len(valid) else None,
    )
    masks = {
        k: np.packbits(value.reshape(-1), bitorder="little")
        for k, value in {
            "geometric": stereo.mutual_mask,
            "sensor_valid": realization.valid,
            "input_valid": selected,
            "tool": realization.tool_mask,
            "hole": realization.random_hole_mask,
            "flying": realization.flying_mask,
        }.items()
    }
    return Packet(a, metrics, masks)
