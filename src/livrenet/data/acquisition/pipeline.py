"""Build the thesis stereo acquisition and controlled noise from a known FEM field."""

from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np

from .deformation import compute_deformation_regions, splat_vertex_support
from .geometry import MeshData
from .noise import rms, unit_field
from .observation import make_blueprint, observe
from .recipe import _build_engine_config, _guided_surface_weights
from .render import stereo_visibility
from .sensor import SensorParameters, load_sensor_profile
from .stereo import make_right_camera


def stable_id(kind, *parts):
    key = "|".join((kind, *map(str, parts))).encode()
    return f"{kind}_{hashlib.sha256(key).hexdigest()[:20]}"


def stable_seed(seed, *parts):
    key = "|".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def contains_query(vertices, faces):
    import pyvista as pv
    from vtk import vtkSelectEnclosedPoints

    mesh = pv.PolyData(vertices, np.c_[np.full(len(faces), 3), faces].ravel())
    query = vtkSelectEnclosedPoints()
    query.SetTolerance(1e-8)
    query.Initialize(mesh)
    return mesh, query


class StereoAcquisition:
    def __init__(self, config=None, sensor_profile=None):
        path = config or Path(__file__).with_name("thesis_acquisition.json")
        self.config = json.loads(path.read_text())
        c = self.config
        thresholds = c["closeout"]
        self.noise = c["recipe"]["noise"]
        self.seed = c["seed"]
        self.bounds = (thresholds["coverage_min"], thresholds["coverage_max"])
        engine = _build_engine_config(c["legacy_PC"])
        self.engine = replace(
            engine,
            camera=replace(
                engine.camera,
                template_visible_surface_range=self.bounds,
                target_surface_coverage_center=0.30,
                target_surface_coverage_sigma=0.05,
                distance_range_diag=(0.12, 1.5),
                max_sampling_attempts=1,
            ),
            sensor=replace(
                engine.sensor,
                canonical_point_budget=c["recipe"]["acquisition"]["point_budget"],
                minimum_raw_points=c["recipe"]["acquisition"]["minimum_points"],
                **self.noise["profiles"]["clean"],
            ),
            tool=replace(
                engine.tool,
                probabilities=c["recipe"]["acquisition"]["tool_probabilities"],
            ),
            deformation_regions=replace(
                engine.deformation_regions, support_absolute_mm=0.005
            ),
        )
        self.profile = load_sensor_profile(
            sensor_profile or Path(__file__).with_name("sensor_profile.json")
        )

    def acquire(self, reference, faces, field, key, ordinal=0, identity=False):
        reference = reference.astype(float)
        field = field.astype(float)
        template = MeshData(reference, faces, np.zeros_like(reference))
        mesh = MeshData(reference + field, faces, field)
        qref = contains_query(reference, faces)
        qdef = qref if identity else contains_query(reference + field, faces)
        engine, profile, c = self.engine, self.profile, self.config
        regions = compute_deformation_regions(mesh, engine.deformation_regions, 1.0)
        weights = (
            None
            if identity
            else _guided_surface_weights(
                regions, "core", c["legacy_PC"]["guided_view_search"]["core_weights"]
            )
        )
        camera_id = stable_id("closeout_camera_v1", self.seed, key, ordinal)
        thresholds = c["closeout"]
        peak = np.linalg.norm(field, axis=1)
        peak_mask = (
            peak >= c.get("suite", {}).get("peak_relative_threshold", 0.8) * peak.max()
        )
        for attempt in range(
            thresholds["camera_attempts"] + thresholds["neutral_attempts"]
        ):
            seed = stable_seed(self.seed, camera_id, attempt)
            bp, reference_stereo = make_blueprint(
                template,
                engine,
                profile,
                np.random.default_rng(seed),
                camera_id,
                weights=weights if attempt < thresholds["camera_attempts"] else None,
                policy="closeout_broad_informative",
            )
            if bp.camera_sampling_fallback:
                continue
            right = make_right_camera(bp.camera, profile, 1.0)
            if any(
                query.IsInsideSurface(*camera.position)
                for _, query in (qref, qdef)
                for camera in (bp.camera, right)
            ):
                continue
            stereo = (
                reference_stereo
                if identity
                else stereo_visibility(mesh, bp.camera, profile, 1.0, engine.stereo)
            )
            coverage = float(
                stereo.left.surface_weight[stereo.mutual_mask].sum() / mesh.areas.sum()
            )
            if not self.bounds[0] <= coverage <= self.bounds[1]:
                continue
            bp = replace(
                bp,
                sensor_parameters=SensorParameters(**self.noise["profiles"]["clean"]),
                sensor_seed=stable_seed(self.seed, camera_id, "clean") % 2**32,
            )
            clean = observe(
                mesh,
                template,
                bp,
                engine,
                profile,
                stereo,
                diagnostic=True,
                regions=regions,
            )
            arrays = clean.arrays
            if len(arrays["points_input"]) != 2500:
                continue
            face_ids, bary = arrays["face_ids_input"], arrays["barycentric_input"]
            support = splat_vertex_support(len(reference), faces, face_ids, bary) > 0
            if not identity:
                areas = mesh.vertex_areas
                peak_area = areas[peak_mask].sum()
                if (
                    peak_area <= 0
                    or areas[peak_mask & support].sum() / peak_area
                    < thresholds["peak_visibility_min"]
                ):
                    continue
            strict_support = np.zeros(len(reference), dtype=bool)
            strict_support[
                faces[face_ids[arrays["is_sensor_inlier_gt_input"]]].ravel()
            ] = True
            points = arrays["points_input"]
            noise_seed = (
                stable_seed(self.noise["seed"], camera_id, "v5_noise_unit_field")
                % 2**32
            )
            noise = unit_field(
                points.astype(float),
                arrays["pixel_uv_input"],
                bp.camera,
                self.noise,
                noise_seed,
                self.noise["evaluation_mode"],
            )
            half = (points.astype(float) + 0.5 * noise).astype(np.float32)
            noise_step = half.astype(float) - points.astype(float)
            actual = rms(noise_step)
            if abs(actual - 0.5) > 1e-4:
                raise ValueError(
                    f"Unexpected noise magnitude for {camera_id}: {actual}"
                )
            return (
                camera_id,
                strict_support,
                {
                    0.0: points,
                    0.5: half,
                    1.0: (points + noise_step / actual).astype(np.float32),
                    1.5: (points + 1.5 * noise_step / actual).astype(np.float32),
                },
            )
        raise RuntimeError(f"No admissible stereo acquisition for {camera_id}")
