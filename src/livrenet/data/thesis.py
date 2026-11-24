"""Explicit entry points for the thesis deformation and acquisition recipes."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .archive import save_array
from .fem import assemble_linear_tetra_stiffness
from .pca import zero_origin_svd
from ..io import load_npz, read_jsonl, save_npz, write_json, write_jsonl


def locate(path, manifest, source_prefix=None, local_root=None):
    path = Path(path)
    if path.is_absolute() and source_prefix is not None:
        return local_root / path.relative_to(source_prefix)
    return path if path.is_absolute() else manifest.parent / path


def rebuild_prior(
    manifest,
    nodes_path,
    anatomy,
    output,
    *,
    young,
    poisson,
    controls=256,
    source_prefix=None,
    local_root=None,
):
    """Rebuild the exact rank-32 deformation coordinates using only FIT fields."""
    from .synthesis import farthest_points

    if bool(source_prefix) != bool(local_root):
        raise ValueError("Both archive roots are required to remap paths.")
    rows = [
        row
        for row in read_jsonl(manifest)
        if row["anatomy_id"] == anatomy and row["partition"] == "fit"
    ]
    ids = [row["mesh_id"] for row in rows]
    if len(ids) < 32 or len(set(ids)) != len(ids):
        raise ValueError("At least 32 distinct FIT fields are required.")
    topology = load_npz(nodes_path)
    nodes = topology["volume_nodes"]
    tetrahedra = topology["tetrahedra"]
    stiffness = assemble_linear_tetra_stiffness(
        nodes, tetrahedra, young=young, poisson=poisson
    )
    surfaces, volume = [], []
    template, faces = None, None
    for row in rows:
        mesh = load_npz(locate(row["mesh_path"], manifest, source_prefix, local_root))
        if template is None:
            template, faces = mesh["reference_points"], mesh["triangles"]
        if not np.array_equal(template, mesh["reference_points"]) or not np.array_equal(
            faces, mesh["triangles"]
        ):
            raise ValueError("FIT fields have inconsistent surface topology.")
        if mesh["volume_displacement"].shape != nodes.shape:
            raise ValueError("Volume field does not use the supplied volume indexing.")
        surfaces.append(mesh["displacement"])
        volume.append(mesh["volume_displacement"])
    basis = zero_origin_svd(surfaces, (32,))[32]["components_flat"]
    matrix = np.stack(surfaces).astype(np.float64).reshape(len(ids), -1)
    coefficients = matrix @ basis.astype(np.float64).T
    lift = np.linalg.lstsq(
        coefficients,
        np.stack(volume).astype(np.float64).reshape(len(ids), -1),
        rcond=1e-8,
    )[0].T
    metric = lift.T @ (stiffness @ lift)
    metric = 0.5 * (metric + metric.T)
    eigen, vectors = np.linalg.eigh(metric)
    if eigen.min() < -1e-8 * max(1, np.abs(eigen).max()):
        raise ValueError("The projected FEM energy is indefinite.")
    metric = (vectors * eigen.clip(0)) @ vectors.T
    save_npz(
        output,
        template_points=template,
        triangles=faces,
        basis=basis.reshape(32, len(template), 3),
        metric=0.5 * (metric + metric.T),
        control_indices=farthest_points(template, min(controls, len(template))),
        normalization_center=(template.max(0) + template.min(0)) / 2,
        normalization_scale=np.asarray(np.linalg.norm(np.ptp(template, axis=0))),
        fit_ids=np.array(ids),
    )


def simulate(recipe, prepared, output, workers, samples, job, partition):
    """Run a frozen single-anatomy FEM recipe on explicitly prepared templates."""
    from .simulation import runner
    from .simulation.config import config_from_dict

    config = json.loads(recipe.read_text())
    if "jobs" in config:
        if job not in config["jobs"]:
            raise ValueError("Choose a job listed by --list-jobs.")
        job_config = config["jobs"][job].copy()
        profile = config["profiles"][job_config.pop("profile")]
        config = config["defaults"] | profile | job_config
    elif job is not None:
        raise ValueError("--job is only valid for a recipe bank.")
    config["out_dir"] = str(output.resolve())
    if workers is not None:
        config["n_workers"] = workers
    for template in config["templates"]:
        for key in ("surface_path", "volume_xdmf_path"):
            template[key] = str(
                (
                    prepared / template["template_id"] / Path(template[key]).name
                ).resolve()
            )
        template["anatomical_paths"] = {
            key: str((prepared / template["template_id"] / Path(path).name).resolve())
            for key, path in template.get("anatomical_paths", {}).items()
        }
    config["overwrite_existing"] = False
    if samples is not None:
        config["n_samples"] = samples
    screen = config.get("principal_stretch_screen")
    cfg = config_from_dict(
        {
            key: value
            for key, value in config.items()
            if key != "principal_stretch_screen"
        }
    )
    cfg.principal_stretch_screen = screen
    receipt = output / "simulation_recipe.json"
    if receipt.exists() and json.loads(receipt.read_text()) != config:
        raise ValueError("The existing simulation directory uses a different recipe.")
    write_json(receipt, config)
    runner.run_project(cfg, resume=True)
    raw_manifest = output / "json" / "mesh_manifest.jsonl"
    records = [row | {"partition": partition} for row in read_jsonl(raw_manifest)]
    manifest = output / "fields.jsonl"
    write_jsonl(manifest, records)
    return manifest


def acquire(
    fields,
    assets,
    output,
    name,
    partition,
    cameras,
    identities,
    config=None,
    sensor_profile=None,
    source_prefix=None,
    local_root=None,
):
    """Generate portable observation pairs with the thesis camera and noise model."""
    from .acquisition.pipeline import StereoAcquisition, stable_id

    if bool(source_prefix) != bool(local_root):
        raise ValueError("Both archive roots are required to remap paths.")
    if (
        Path(name).name != name
        or name in {"", ".", ".."}
        or cameras < 1
        or identities < 0
    ):
        raise ValueError("Invalid output name or acquisition count.")
    acquisition = StereoAcquisition(config, sensor_profile)
    source = read_jsonl(fields)
    rows, template_cache = [], {}
    expected = {"train": "fit", "validation": "calibration", "test": "evaluation"}[
        partition
    ]
    for entry in source:
        if entry.get("partition", expected) != expected:
            continue
        entry = deepcopy(entry)
        anatomy, parent = entry["anatomy_id"], entry["mesh_id"]
        if any(
            Path(item).name != item or item in {"", ".", ".."}
            for item in (anatomy, parent)
        ):
            raise ValueError("Invalid anatomy or parent identifier.")
        template_path = Path("anatomies") / f"{anatomy}.npz"
        if anatomy not in template_cache:
            template_cache[anatomy] = load_npz(assets / f"{anatomy}.npz")
            save_npz(output / template_path, **template_cache[anatomy])
            include_identity = True
        else:
            include_identity = False
        template = template_cache[anatomy]
        mesh = load_npz(locate(entry["mesh_path"], fields, source_prefix, local_root))
        reference, faces = template["template_points"], template["triangles"]
        if not np.array_equal(
            reference, mesh["reference_points"]
        ) or not np.array_equal(faces, mesh["triangles"]):
            raise ValueError(
                "Acquisition fields must share the frozen template indexing."
            )
        if (parent in set(template["fit_ids"].astype(str))) != (partition == "train"):
            raise ValueError("Acquisition parent has incorrect prior FIT membership.")
        folder = Path(name) / anatomy
        target_path = folder / f"{parent}_target.npy"
        save_array(output / target_path, mesh["displacement"])
        roles = [(False, parent, mesh["displacement"], cameras)]
        if include_identity and identities:
            roles.append((True, "identity", np.zeros_like(reference), identities))
        for identity, suffix, displacement, count in roles:
            key = f"{anatomy}__{expected}__{suffix}"
            for ordinal in range(count):
                camera, support, clouds = acquisition.acquire(
                    reference, faces, displacement, key, ordinal, identity
                )
                support_path = folder / f"{camera}_support.npy"
                save_array(output / support_path, support)
                for dose in (0.0, 0.5, 1.0, 1.5) if partition == "test" else (0.0, 0.5):
                    tag = {
                        0.0: "clean",
                        0.5: "low" if partition == "train" else "rms_0p5",
                        1.0: "rms_0p5",
                        1.5: "rms_0p5",
                    }[dose]
                    sample = stable_id("closeout_sample", camera, tag)
                    if dose > 0.5:
                        sample += f"__RMS_{dose:g}"
                    points_path = folder / f"{sample}_points.npy"
                    save_array(output / points_path, clouds[dose])
                    rows.append(
                        dict(
                            sample_id=sample,
                            anatomy_id=anatomy,
                            anatomy=template_path.as_posix(),
                            parent_id=camera if identity else parent,
                            camera_id=camera,
                            partition=partition,
                            is_identity=identity,
                            noise_mm=dose,
                            points=points_path.as_posix(),
                            support=support_path.as_posix(),
                            target=None if identity else target_path.as_posix(),
                        )
                    )
    if not rows:
        raise ValueError("No deformation records for the selected partition.")
    path = output / f"{name}.jsonl"
    write_jsonl(path, rows)
    return path
