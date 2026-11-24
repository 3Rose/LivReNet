"""Convert locally held thesis assets and observations to portable manifests."""

from functools import lru_cache
from pathlib import Path
import os

import numpy as np

from ..io import atomic, load_npz, read_jsonl, save_npz, write_jsonl


def pack_anatomy(assets, anatomy, output):
    template = load_npz(assets / "patient_templates" / f"{anatomy}.npz")
    basis = load_npz(assets / "deformation_basis" / f"{anatomy}.npz")
    energy = load_npz(assets / "pca_energy" / f"{anatomy}_pca_energy_rank32.npz")
    count = len(template["template_points"])
    save_npz(
        output,
        **template,
        basis=basis["rank32_components_flat"].reshape(32, count, 3),
        metric=energy["H_fem"],
        fit_ids=basis["pca_fit_deformation_ids"],
    )
    return template, set(basis["pca_fit_deformation_ids"].astype(str))


@lru_cache(maxsize=8)
def read_shard(path):
    return load_npz(path)


def save_array(path, array):
    with atomic(path) as temporary, temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)


def pack_archive(
    manifest,
    assets,
    output,
    name,
    partition,
    limit=None,
    source_prefix=None,
    local_root=None,
):
    """Export observations from an explicit archive manifest without scanning for files."""
    manifest, assets, output = map(Path, (manifest, assets, output))
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("The output name must be a single path component.")
    if bool(source_prefix) != bool(local_root):
        raise ValueError("--source-prefix and --local-root must be provided together.")
    if limit is not None and limit < 1:
        raise ValueError("The sample limit must be positive.")

    def resolve(value):
        path = Path(value)
        if path.is_absolute() and source_prefix is not None:
            return Path(local_root) / path.relative_to(source_prefix)
        return path if path.is_absolute() else manifest.parent / path

    original = read_jsonl(manifest)
    rows = []
    templates = {}
    seen_targets = set()
    for row in original[:limit]:
        sample, anatomy = row["sample_id"], row["anatomy_id"]
        if any(
            Path(item).name != item or item in {"", ".", ".."}
            for item in (sample, anatomy)
        ):
            raise ValueError("Invalid filename component in archive manifest.")
        anatomy_path = output / "anatomies" / f"{anatomy}.npz"
        if anatomy not in templates:
            templates[anatomy] = pack_anatomy(assets, anatomy, anatomy_path)
        template, fit_ids = templates[anatomy]
        parent = row["parent_mesh_id"]
        identity = bool(row["is_identity"])
        if not identity and ((parent in fit_ids) != (partition == "train")):
            raise ValueError(f"Unexpected PCA FIT membership for {parent}")
        shard = read_shard(resolve(row["shard_path"]))
        start, stop = map(
            int, shard["offsets"][row["shard_index"] : row["shard_index"] + 2]
        )
        points = (
            load_npz(resolve(row["study_points_path"]))["points"]
            if "study_points_path" in row
            else shard["points"][start:stop]
        )
        support = np.zeros(len(template["template_points"]), dtype=bool)
        face_ids = shard["face"][start:stop][shard["valid"][start:stop].astype(bool)]
        support[template["triangles"][face_ids].ravel()] = True
        points_path = output / name / "points" / f"{sample}.npy"
        support_path = output / name / "support" / f"{sample}.npy"
        if not identity:
            if Path(parent).name != parent or parent in {"", ".", ".."}:
                raise ValueError("Invalid deformation identifier.")
            scale = float(row.get("deformation_scale", 1))
            target_key = (anatomy, parent, scale)
            target_path = (
                output / name / "targets" / anatomy / f"{parent}_{scale:g}.npy"
            )
            if target_key not in seen_targets:
                mesh = load_npz(resolve(row["mesh_path"]))
                if not np.array_equal(
                    mesh["triangles"], template["triangles"]
                ) or not np.allclose(
                    mesh["reference_points"],
                    template["template_points"],
                    atol=1e-5,
                    rtol=0,
                ):
                    raise ValueError(f"Frozen target topology differs for {parent}")
                save_array(
                    target_path, (mesh["displacement"] * scale).astype(np.float32)
                )
                seen_targets.add(target_key)

        save_array(points_path, points.astype(np.float32))
        save_array(support_path, support)
        rows.append(
            dict(
                sample_id=sample,
                anatomy_id=anatomy,
                anatomy=os.path.relpath(anatomy_path, output),
                parent_id=row["camera_id"] if identity else parent,
                camera_id=row["camera_id"],
                partition=partition,
                is_identity=identity,
                noise_mm=float(
                    row.get(
                        "study_noise_rms_mm",
                        0 if row["noise_profile"] == "clean" else 0.5,
                    )
                ),
                points=os.path.relpath(points_path, output),
                support=os.path.relpath(support_path, output),
                target=None if identity else os.path.relpath(target_path, output),
            )
        )
    if not rows:
        raise ValueError("The archive manifest is empty.")
    path = output / f"{name}.jsonl"
    write_jsonl(path, rows)
    return path
