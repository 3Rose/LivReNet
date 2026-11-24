"""Inference from a registered point cloud and an anatomy-specific prior."""

from functools import cached_property, lru_cache
from pathlib import Path
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch

from .encoder import CompactEvidencePCA
from .fitting.staged import fit
from .refiner import FreeG
from .static import build_static, refiner_graph
from .surface import observation_features


def configure_torch(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose a CPU or CUDA device.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for a CPU run.")
    return device


def tensors(arrays, device, batch=False):
    return {
        name: torch.as_tensor(value, device=device)[None]
        if batch
        else torch.as_tensor(value, device=device)
        for name, value in arrays.items()
    }


def point_cloud(points):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (2500, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 2,500 finite Cartesian points in millimetres.")
    return points


class Anatomy:
    """Reusable geometry for a single template and its reduced deformation basis."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as arrays:
            self.assets = {
                key: arrays[key]
                for key in (
                    "template_points",
                    "triangles",
                    "normalization_center",
                    "normalization_scale",
                    "control_indices",
                    "basis",
                    "metric",
                )
            }
        a = self.assets
        self.vertices = a["template_points"]
        self.faces = a["triangles"].astype(np.int64)
        self.center = a["normalization_center"]
        self.scale = float(a["normalization_scale"])
        self.basis = a["basis"].astype(np.float32)
        self.metric = a["metric"].astype(np.float32)
        controls = a["control_indices"]
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
            or len(self.vertices) < 512
            or self.faces.ndim != 2
            or self.faces.shape[1] != 3
            or not len(self.faces)
            or self.faces.min() < 0
            or self.faces.max() >= len(self.vertices)
            or self.basis.shape != (32, len(self.vertices), 3)
            or self.metric.shape != (32, 32)
            or controls.ndim != 1
            or len(controls) < 9
            or controls.min() < 0
            or controls.max() >= len(self.vertices)
            or self.center.shape != (3,)
            or not np.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError(f"Incompatible anatomy arrays in {self.path}")
        self.normalized = ((self.vertices - self.center) / self.scale).astype(
            np.float32
        )
        self.control_tree = cKDTree(
            (self.vertices[controls] - self.center) / self.scale
        )
        self.vertex_tree = cKDTree(self.normalized)
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self.scene.add_triangles(
            o3d.core.Tensor(self.vertices.astype(np.float32)),
            o3d.core.Tensor(self.faces.astype(np.uint32)),
        )

    @cached_property
    def encoder_arrays(self):
        return build_static(self.assets)

    @cached_property
    def refiner_arrays(self):
        return refiner_graph(self.assets) | {"ref": self.vertices - self.center}

    @cached_property
    def incident_faces(self):
        incident = [[] for _ in self.vertices]
        for index, face in enumerate(self.faces):
            for vertex in face:
                incident[vertex].append(index)
        lookup = np.full((len(incident), max(map(len, incident))), -1, dtype=np.int64)
        for index, faces in enumerate(incident):
            lookup[index, : len(faces)] = faces
        return lookup

    @lru_cache(maxsize=24)
    def encoder_graph(self, device):
        return tensors(self.encoder_arrays, device)

    @lru_cache(maxsize=24)
    def refiner_graph(self, device):
        return tensors(self.refiner_arrays, device)

    def encoder_inputs(self, points):
        points = point_cloud(points)
        hit = self.scene.compute_closest_points(o3d.core.Tensor(points), nthreads=1)
        residual = points - hit["points"].numpy()
        coordinates = (points - self.center) / self.scale
        return {
            "points": coordinates.astype(np.float32),
            "knn": cKDTree(points).query(points, k=16)[1].astype(np.int64),
            "candidates": self.control_tree.query(coordinates, k=8)[1].astype(np.int64),
            "surface_residual_mm": residual.astype(np.float32),
            "distance_mm": np.linalg.norm(residual, axis=1).astype(np.float32)[:, None],
        }

    def fitting_inputs(self, points):
        raw = ((points - self.center) / self.scale).astype(np.float32)
        nearby = self.vertex_tree.query(raw, k=64)[1]
        rows = self.incident_faces[nearby].reshape(len(raw), -1)
        candidates = np.zeros((len(raw), 192), dtype=np.int64)
        valid = np.zeros_like(candidates, dtype=bool)
        for index, row in enumerate(rows):
            distinct, first = np.unique(row, return_index=True)
            faces = row[np.sort(first[distinct >= 0])][:192]
            if len(faces):
                candidates[index, :] = faces[0]
                candidates[index, : len(faces)] = faces
                valid[index, : len(faces)] = True
        all_indices = np.flatnonzero(valid[:, :96].any(axis=1))
        if not len(all_indices):
            raise ValueError("No fitting constraints could be constructed.")
        fit_indices = all_indices[
            np.linspace(
                0, len(all_indices) - 1, min(512, len(all_indices)), dtype=np.int64
            )
        ]
        return dict(
            template_norm=self.normalized,
            faces=self.faces,
            raw=raw,
            scale=np.asarray(self.scale, np.float32),
            basis=self.basis,
            metric=self.metric,
            candidate_faces=candidates,
            candidate_valid=valid,
            all_indices=all_indices,
            fit_indices=fit_indices,
        )

    def refiner_inputs(self, points, fitted):
        features, _, _ = observation_features(self.vertices, self.faces, fitted, points)
        features[:, 15] = (np.expm1(features[:, 12]) < 1e-6).astype(np.float32)
        nearest = (
            cKDTree(points).query(self.vertices + fitted, k=1)[0].astype(np.float32)
        )
        raw = np.column_stack(
            (
                features,
                (self.vertices + fitted - self.center) / self.scale,
                nearest / 10,
            )
        ).astype(np.float16)
        if not np.isfinite(raw).all():
            raise FloatingPointError("Refiner inputs contain nonfinite values.")
        return dict(raw=raw.astype(np.float32), u0=fitted, no_signal=np.asarray(False))


class Predictor:
    def __init__(self, encoder_checkpoint, config, device, refiner_checkpoint=None):
        self.device = configure_torch(device)
        self.config = config
        self.encoder = None
        if encoder_checkpoint is not None:
            checkpoint = torch.load(
                encoder_checkpoint, map_location="cpu", weights_only=True
            )
            self.encoder = CompactEvidencePCA().to(self.device).eval()
            self.encoder.load_state_dict(checkpoint["model"], strict=True)
        self.refiner = None
        if refiner_checkpoint is not None:
            checkpoint = torch.load(
                refiner_checkpoint, map_location="cpu", weights_only=True
            )
            self.refiner = FreeG(checkpoint["normalization"]).to(self.device).eval()
            self.refiner.load_state_dict(checkpoint["model"], strict=True)

    @torch.inference_mode()
    def upstream(self, anatomy, points, caps=None):
        points = point_cloud(points)
        device = self.device
        if self.encoder is None:
            initial = torch.zeros(
                anatomy.vertices.shape, device=device, dtype=torch.float64
            )
            center = torch.zeros(32, device=device, dtype=torch.float64)
            no_signal = np.asarray(
                anatomy.encoder_inputs(points)["distance_mm"].max() <= 1e-4
            )
        else:
            inputs = tensors(anatomy.encoder_inputs(points), device, batch=True)
            output = self.encoder(inputs, anatomy.encoder_graph(device))
            initial = output["displacement_mm"][0]
            center = output["coefficients"][0]
            no_signal = output["numeric_no_signal"][0]
        observation = tensors(anatomy.fitting_inputs(points), device)
        observation["numeric_no_signal"] = torch.as_tensor(no_signal, device=device)
        packet = dict(
            template=torch.as_tensor(
                anatomy.vertices - anatomy.center, device=device, dtype=torch.float64
            ),
            faces=observation["faces"],
            basis=observation["basis"].double(),
            metric=observation["metric"].double(),
            points=observation["raw"].double() * anatomy.scale,
            initial=initial.double(),
            fixed=torch.zeros_like(initial, dtype=torch.float64),
            center=center.double(),
            mass=torch.tensor(300.0, device=device, dtype=torch.float64),
            coefficient_ridge_center=torch.zeros(
                32, device=device, dtype=torch.float64
            ),
        )
        caps = tuple(self.config["fitting_caps"]) if caps is None else caps
        fitted, diagnostics = fit(packet, observation, self.config["fitter"], caps)
        if bool(diagnostics["numerical_fallback"]):
            raise FloatingPointError(
                "Reduced fitting failed to return a finite solution."
            )
        return initial.cpu().numpy(), fitted.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def predict(self, anatomy, points, caps=None):
        solve, fitted = self.upstream(anatomy, points, caps)
        fields = dict(solve=solve, fitted=fitted)
        if self.refiner is not None:
            inputs = tensors(
                anatomy.refiner_inputs(points, fitted), self.device, batch=True
            )
            graph = anatomy.refiner_graph(self.device)
            fields["final"] = self.refiner(inputs, graph)[0][0].cpu().numpy()
        if any(not np.isfinite(field).all() for field in fields.values()):
            raise FloatingPointError("Prediction contains nonfinite displacements.")
        return fields
