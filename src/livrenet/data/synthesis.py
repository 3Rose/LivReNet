"""Generate a controlled complete-to-partial FEM registration dataset."""

from itertools import permutations
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.sparse.linalg import splu

from .archive import save_array
from .fem import assemble_linear_tetra_stiffness
from .pca import zero_origin_svd
from ..io import load_npz, save_npz, write_jsonl


def farthest_points(vertices, count):
    selected = np.empty(count, dtype=np.int64)
    nearest = np.full(len(vertices), np.inf)
    index = 0
    for position in range(count):
        selected[position] = index
        nearest = np.minimum(nearest, np.sum((vertices - vertices[index]) ** 2, axis=1))
        nearest[selected[: position + 1]] = -1
        index = int(nearest.argmax())
    return selected


def tetra_boundary(nodes, tetrahedra):
    local = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    faces = tetrahedra[:, local].reshape(-1, 3)
    _, first, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    indices = first[counts == 1]
    boundary = faces[indices].copy()
    triangle = nodes[boundary]
    toward_opposite = nodes[tetrahedra[indices // 4, indices % 4]] - triangle[:, 0]
    normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
    inward = np.einsum("ij,ij->i", normal, toward_opposite) > 0
    boundary[inward] = boundary[inward][:, [0, 2, 1]]
    surface, inverse = np.unique(boundary, return_inverse=True)
    return surface, inverse.reshape(-1, 3)


def fixture_mesh():
    """Return an artificial warped volume with enough vertices for all graph levels."""
    n = 11
    grid = np.indices((n, n, n)).reshape(3, -1).T
    u, v, w = (grid / (n - 1) * 2 - 1).T
    vertices = np.column_stack(
        (42 * u + 3 * (v * w), 30 * v + 4 * u**2, 23 * w + 3 * (u * v))
    )
    cells = np.indices((n - 1, n - 1, n - 1)).reshape(3, -1).T
    steps = np.eye(3, dtype=int)
    tetrahedra = []
    for order in permutations(range(3)):
        corners = np.vstack((np.zeros(3, dtype=int), steps[list(order)].cumsum(axis=0)))
        tetrahedra.append(
            np.ravel_multi_index(
                (cells[:, None, :] + corners).transpose(2, 0, 1), (n,) * 3
            )
        )
    return vertices, np.concatenate(tetrahedra)


class ElasticVolume:
    def __init__(self, nodes, tetrahedra, young, poisson):
        self.nodes, self.tetrahedra = nodes, tetrahedra
        if (
            nodes.ndim != 2
            or nodes.shape[1] != 3
            or not np.isfinite(nodes).all()
            or tetrahedra.ndim != 2
            or tetrahedra.shape[1] != 4
            or not len(tetrahedra)
            or tetrahedra.min() < 0
            or tetrahedra.max() >= len(nodes)
            or len(np.unique(tetrahedra)) != len(nodes)
        ):
            raise ValueError(
                "Expected a finite tetrahedral volume with no unused nodes."
            )
        self.surface, self.faces = tetra_boundary(nodes, tetrahedra)
        if len(self.surface) < 512:
            raise ValueError(
                "The volume must have at least 512 distinct boundary vertices."
            )
        self.vertices = nodes[self.surface]
        self.extent = np.linalg.norm(np.ptp(nodes, axis=0))
        self.stiffness = assemble_linear_tetra_stiffness(
            nodes, tetrahedra, young=young, poisson=poisson
        )
        reference = nodes[tetrahedra]
        self.reference_det = np.linalg.det(reference[:, 1:] - reference[:, :1])
        if np.any(np.abs(self.reference_det) < 1e-12):
            raise ValueError("Degenerate reference tetrahedra.")

    def field(self, rng, amplitude):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        axis = (self.vertices - self.vertices.mean(axis=0)) @ direction
        anchors = self.surface[axis <= np.quantile(axis, 0.18)]
        candidates = np.flatnonzero(axis >= np.quantile(axis, 0.60))
        anchor_dofs = (3 * anchors[:, None] + np.arange(3)).ravel()
        free = np.setdiff1d(np.arange(self.stiffness.shape[0]), anchor_dofs)
        factor = splu(self.stiffness[free][:, free].tocsc())
        load_center = self.vertices[rng.choice(candidates)]
        radii = np.linalg.norm(self.vertices - load_center, axis=1)
        weights = np.exp(-0.5 * (radii / (rng.uniform(0.08, 0.2) * self.extent)) ** 2)
        triangle = self.vertices[self.faces]
        area = (
            np.linalg.norm(
                np.cross(
                    triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
                ),
                axis=1,
            )
            / 2
        )
        lumped_area = np.bincount(
            self.faces.ravel(), np.repeat(area / 3, 3), minlength=len(self.vertices)
        )
        load = np.zeros_like(self.nodes)
        load[self.surface] = (weights * lumped_area)[:, None] * rng.normal(size=(1, 3))
        field = np.zeros(self.stiffness.shape[0])
        field[free] = factor.solve(load.ravel()[free])
        field = field.reshape(-1, 3)
        peak = np.linalg.norm(field[self.surface], axis=1).max()
        if not np.isfinite(peak) or peak <= 0:
            raise FloatingPointError(
                "The applied traction produced an invalid displacement."
            )
        field *= amplitude / peak
        deformed = (self.nodes + field)[self.tetrahedra]
        relative_det = (
            np.linalg.det(deformed[:, 1:] - deformed[:, :1]) / self.reference_det
        )
        return field.astype(np.float32) if relative_det.min() > 0.2 else None


def visible_patch(vertices, faces, field, rng, n_points=2500):
    """Sample mutually visible surface points using two perspective viewpoints."""
    deformed = (vertices + field).astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(
        o3d.core.Tensor(deformed), o3d.core.Tensor(faces.astype(np.uint32))
    )
    triangle = deformed[faces]
    areas = np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]),
        axis=1,
    )
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(np.ptp(vertices, axis=0)) * 1.15
    sampled_faces = rng.choice(len(faces), 8 * n_points, p=areas / areas.sum())
    weights = rng.dirichlet(np.ones(3), size=len(sampled_faces))
    targets = np.einsum("nv,nvd->nd", weights, triangle[sampled_faces]).astype(
        np.float32
    )
    motion = np.linalg.norm(field, axis=1)
    peak_faces = (
        (motion[faces] >= 0.8 * motion.max()).any(axis=1) if motion.max() >= 1 else None
    )
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        position = center + radius * axis
        tangent = np.cross(axis, [0, 0, 1])
        tangent /= np.linalg.norm(tangent)
        other = position + 5 * tangent

        def visible(view):
            origins = np.broadcast_to(view, targets.shape)
            rays = np.column_stack((origins, targets - origins)).astype(np.float32)
            hits = scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()
            return np.abs(hits - 1) <= 3e-5

        accepted = visible(position) & visible(other)
        if not 0.20 <= accepted.mean() <= 0.55 or accepted.sum() < n_points:
            continue
        if (
            peak_faces is not None
            and accepted[peak_faces[sampled_faces]].sum()
            < 0.25 * peak_faces[sampled_faces].sum()
        ):
            continue
        selected = rng.choice(np.flatnonzero(accepted), n_points, replace=False)
        support = np.zeros(len(vertices), dtype=bool)
        support[faces[sampled_faces[selected]].ravel()] = True
        return targets[selected], support
    raise RuntimeError("Could not sample a patch satisfying the visibility criteria.")


def paired_noise(points, rng):
    noise = rng.normal(size=points.shape)
    noise *= np.minimum(1, 2.5 / np.linalg.norm(noise, axis=1, keepdims=True))
    return noise / np.sqrt(np.mean(np.sum(noise**2, axis=1)))


def build_prior(volume, fit_fields):
    surface_fields = np.stack(fit_fields)[:, volume.surface]
    basis = zero_origin_svd(surface_fields, (32,))[32]["components_flat"]
    coefficients = (
        surface_fields.astype(np.float64).reshape(len(fit_fields), -1)
        @ basis.astype(np.float64).T
    )
    volume_matrix = np.stack(fit_fields).astype(np.float64).reshape(len(fit_fields), -1)
    lift = np.linalg.lstsq(coefficients, volume_matrix, rcond=1e-8)[0].T
    energy = lift.T @ (volume.stiffness @ lift)
    return basis.reshape(32, len(volume.surface), 3), 0.5 * (energy + energy.T)


def generate(output, inputs, *, seed, fit_count, heldout_count, fixture=False):
    """Write separate FIT/train, validation and test manifests."""
    output = Path(output)
    if fit_count < 32 or heldout_count < 1:
        raise ValueError("At least 32 FIT fields and one held-out field are required.")
    if fixture:
        nodes, tetrahedra = fixture_mesh()
        inputs = {
            f"fixture_{split}": (nodes, tetrahedra, split)
            for split in ("train", "validation", "test")
        }
    rows = {"train": [], "validation": [], "test": []}
    for number, (anatomy, (nodes, tetrahedra, split)) in enumerate(
        sorted(inputs.items())
    ):
        rng = np.random.default_rng(np.random.SeedSequence([seed, number]))
        volume = ElasticVolume(nodes, tetrahedra, young=2000, poisson=0.45)
        vertices, faces = volume.vertices.astype(np.float32), volume.faces
        fields = []
        attempts = 0
        count = fit_count + (heldout_count if split != "train" else 0)
        while len(fields) < count and attempts < 20 * count:
            attempts += 1
            field = volume.field(
                rng, rng.uniform(0.9, 4.0) if fixture else rng.uniform(1.0, 12.0)
            )
            if field is not None:
                fields.append(field)
        if len(fields) != count:
            raise RuntimeError(f"FEM deformation budget exhausted for {anatomy}.")

        basis, metric = build_prior(volume, fields[:fit_count])
        fit_ids = np.array([f"fem_{i:04d}" for i in range(fit_count)])
        anatomy_path = Path("anatomies") / f"{anatomy}.npz"
        save_npz(
            output / anatomy_path,
            template_points=vertices,
            triangles=faces,
            basis=basis,
            metric=metric,
            control_indices=farthest_points(vertices, 512),
            fit_ids=fit_ids,
            normalization_center=(vertices.max(0) + vertices.min(0)) / 2,
            normalization_scale=np.asarray(np.linalg.norm(np.ptp(vertices, axis=0))),
        )
        selected = range(fit_count) if split == "train" else range(fit_count, count)
        if fixture and split == "train":
            selected = range(2)
        for index in [*selected, -1]:
            identity = index < 0
            parent = "identity" if identity else f"fem_{index:04d}"
            target = (
                np.zeros_like(vertices) if identity else fields[index][volume.surface]
            )
            points, support = visible_patch(vertices, faces, target, rng)
            noise = paired_noise(points, rng)
            base = Path(split) / anatomy
            target_path = base / f"{parent}_target.npy"
            support_path = base / f"{parent}_support.npy"
            if not identity:
                save_array(output / target_path, target)
            save_array(output / support_path, support)
            for dose in (0.0, 0.5, 1.0, 1.5) if split == "test" else (0.0, 0.5):
                sample = f"{anatomy}_{parent}_{dose:g}"
                points_path = base / f"{sample}_points.npy"
                save_array(
                    output / points_path, (points + dose * noise).astype(np.float32)
                )
                rows[split].append(
                    dict(
                        sample_id=sample,
                        anatomy_id=anatomy,
                        anatomy=anatomy_path.as_posix(),
                        parent_id=parent,
                        camera_id=f"{anatomy}_{parent}",
                        partition=split,
                        is_identity=identity,
                        noise_mm=dose,
                        points=points_path.as_posix(),
                        support=support_path.as_posix(),
                        target=None if identity else target_path.as_posix(),
                    )
                )
    for split, samples in rows.items():
        if samples:
            write_jsonl(output / f"{split}.jsonl", samples)
    return {split: len(samples) for split, samples in rows.items()}


def read_volumes(config, directory):
    volumes = {}
    for name, row in config.items():
        if Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Anatomy IDs must be single path components.")
        split = row["partition"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unexpected partition: {split}")
        mesh = load_npz(Path(directory) / row["volume"])
        volumes[name] = (
            mesh["nodes"].astype(np.float64),
            mesh["tetrahedra"].astype(np.int64),
            split,
        )
    if not volumes:
        raise ValueError("The volume configuration is empty.")
    return volumes
