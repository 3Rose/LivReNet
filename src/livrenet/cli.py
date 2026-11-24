"""Command-line tools for preparing data, training and benchmarking LivReNet."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data.archive import pack_archive
from .io import save_npz, write_json, write_jsonl


def settings(args):
    path = args.config or Path(__file__).with_name("thesis.json")
    return json.loads(path.read_text())


def predict(args):
    from .runtime import Anatomy, Predictor

    anatomy = Anatomy(args.anatomy)
    model = Predictor(args.encoder, settings(args), args.device, args.refiner)
    fields = model.predict(anatomy, np.load(args.points, allow_pickle=False))
    save_npz(
        args.output,
        **fields,
        triangles=anatomy.faces,
        **{
            f"{name}_vertices": anatomy.vertices + field
            for name, field in fields.items()
        },
    )
    print(args.output)


def evaluate(args):
    from .data.manifest import Dataset
    from .evaluation import measure, summary
    from .runtime import Predictor

    dataset = Dataset(args.manifest)
    configs = settings(args)
    network = Predictor(args.encoder, configs, args.device, args.refiner)
    baseline = Predictor(None, configs, args.device) if args.baselines else None
    records = []
    for index, row in enumerate(dataset.rows, 1):
        anatomy = dataset.geometry(row)
        points = dataset.points(row)
        fields = network.predict(anatomy, points)
        if baseline is not None:
            fields["template"] = np.zeros_like(anatomy.vertices)
            fields["geometric_fit"] = baseline.upstream(anatomy, points)[1]
            fields["geometric_fit_extended"] = baseline.upstream(
                anatomy, points, (5, 1, 2, 24)
            )[1]
        if args.predictions:
            save_npz(args.output / "predictions" / f"{row['sample_id']}.npz", **fields)
        target, support = dataset.labels(row)
        fields_info = {
            key: row[key]
            for key in (
                "sample_id",
                "anatomy_id",
                "parent_id",
                "noise_mm",
                "is_identity",
            )
        }
        for stage, field in fields.items():
            records.append(
                fields_info
                | {
                    "stage": stage,
                    "metrics": measure(
                        anatomy, points, field, target, support, row["is_identity"]
                    ),
                }
            )
        if index == len(dataset.rows) or index % 25 == 0:
            print(f"Evaluated {index}/{len(dataset.rows)} observations.", flush=True)
    write_jsonl(args.output / "per_observation.jsonl", records)
    write_json(args.output / "summary.json", summary(records))


def train(args):
    from .data.manifest import Dataset
    from .training import train_stage

    config = settings(args)
    training, validation = (
        Dataset(args.train_manifest),
        Dataset(args.validation_manifest),
    )
    if args.primary_validation is not None:
        config["primary_validation_anatomies"] = args.primary_validation
    if args.epochs is not None:
        config["encoder_epochs"] = config["refiner_epochs"] = args.epochs
    encoder = args.encoder
    if args.stage in {"encoder", "both"}:
        encoder = train_stage(
            "encoder", training, validation, args.output, config, args.device
        )
        print(encoder)
    if args.stage in {"refiner", "both"}:
        print(
            train_stage(
                "refiner",
                training,
                validation,
                args.output,
                config,
                args.device,
                encoder,
            )
        )


def synthesize(args):
    from .data.synthesis import generate, read_volumes

    if args.fixture:
        volumes = {}
    else:
        volumes = read_volumes(
            json.loads(args.volumes.read_text()), args.volumes.parent
        )
    counts = generate(
        args.output,
        volumes,
        seed=args.seed,
        fit_count=args.fit_count,
        heldout_count=args.heldout_count,
        fixture=args.fixture,
    )
    print(json.dumps(counts))


def convert_volume(args):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(args.surface))
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    if not mesh.is_watertight() or args.max_volume <= 0:
        raise ValueError(
            "Provide a watertight surface and a positive maximum tetrahedron volume."
        )
    try:
        import tetgen
    except ImportError as error:
        raise RuntimeError(
            "The volume command requires the tetgen Python package."
        ) from error
    tetra = tetgen.TetGen(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    tetra.tetrahedralize(switches=f"pq1.5a{args.max_volume:g}Q")
    save_npz(args.output, nodes=tetra.node, tetrahedra=tetra.elem)


def import_archive(args):
    path = pack_archive(
        args.manifest,
        args.assets,
        args.output,
        args.name,
        args.partition,
        args.limit,
        args.source_prefix,
        args.local_root,
    )
    print(path)


def simulate_thesis(args):
    from .data.thesis import simulate

    if args.list_jobs:
        for job in sorted(json.loads(args.recipe.read_text()).get("jobs", {})):
            print(job)
        return
    if args.prepared is None or args.output is None or args.partition is None:
        raise ValueError(
            "--prepared, --output and --partition are required for FEM simulation."
        )
    print(
        simulate(
            args.recipe,
            args.prepared,
            args.output,
            args.workers,
            args.samples,
            args.job,
            args.partition,
        )
    )


def prior_thesis(args):
    from .data.thesis import rebuild_prior

    rebuild_prior(
        args.fields,
        args.volume,
        args.anatomy_id,
        args.output,
        young=args.young,
        poisson=args.poisson,
        source_prefix=args.source_prefix,
        local_root=args.local_root,
    )
    print(args.output)


def acquire_thesis(args):
    from .data.thesis import acquire

    print(
        acquire(
            args.fields,
            args.assets,
            args.output,
            args.name,
            args.partition,
            args.cameras,
            args.identities,
            args.acquisition_config,
            args.sensor_profile,
            args.source_prefix,
            args.local_root,
        )
    )


def assemble_manifest(args):
    from .data.catalog import assemble

    print(assemble(args.manifests, args.lineage, args.output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="command")
    p = sub.add_parser(
        "predict", help="Reconstruct a displacement field from one aligned point cloud."
    )
    p.set_defaults(function=predict)
    p.add_argument("--anatomy", type=Path, required=True)
    p.add_argument("--points", type=Path, required=True)

    e = sub.add_parser(
        "evaluate", help="Evaluate frozen weights on an observation manifest."
    )
    e.set_defaults(function=evaluate)
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--baselines", action="store_true")
    e.add_argument("--predictions", action="store_true")
    for command in (p, e):
        command.add_argument("--encoder", type=Path, required=True)
        command.add_argument("--refiner", type=Path)

    t = sub.add_parser("train", help="Train an encoder and residual refiner.")
    t.set_defaults(function=train)
    t.add_argument("--train-manifest", type=Path, required=True)
    t.add_argument("--validation-manifest", type=Path, required=True)
    t.add_argument("--stage", choices=("encoder", "refiner", "both"), default="both")
    t.add_argument(
        "--encoder", type=Path, help="Frozen encoder for refiner-only training."
    )
    t.add_argument("--primary-validation", nargs="+", metavar="ANATOMY")
    t.add_argument("--epochs", type=int, help="Override the budget for a short run.")
    for command in (p, e, t):
        command.add_argument("--device", default="cpu")
        command.add_argument(
            "--config",
            type=Path,
            help="Override the bundled training and fitting configuration.",
        )
        command.add_argument("--output", type=Path, required=True)

    d = sub.add_parser(
        "generate",
        help="Create FEM fields, anatomy priors and paired partial observations.",
    )
    d.set_defaults(function=synthesize)
    mode = d.add_mutually_exclusive_group(required=True)
    mode.add_argument("--volumes", type=Path)
    mode.add_argument("--fixture", action="store_true")
    d.add_argument("--seed", type=int, default=13)
    d.add_argument("--fit-count", type=int, default=40)
    d.add_argument("--heldout-count", type=int, default=1)
    d.add_argument("--output", type=Path, required=True)

    v = sub.add_parser(
        "volume", help="Tetrahedralize a local watertight triangle mesh."
    )
    v.set_defaults(function=convert_volume)
    v.add_argument("--surface", type=Path, required=True)
    v.add_argument(
        "--max-volume",
        type=float,
        required=True,
        help="Maximum tetrahedron volume in cubic millimetres.",
    )
    v.add_argument("--output", type=Path, required=True)

    a = sub.add_parser(
        "import-thesis", help="Convert a local thesis archive into portable manifests."
    )
    a.set_defaults(function=import_archive)
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--assets", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--name", required=True)
    a.add_argument(
        "--partition", choices=("train", "validation", "test"), required=True
    )
    a.add_argument(
        "--source-prefix",
        type=Path,
        help="Original root of absolute paths in the archive.",
    )
    a.add_argument(
        "--local-root", type=Path, help="Current location of the same archive tree."
    )
    a.add_argument("--limit", type=int)

    f = sub.add_parser(
        "simulate-thesis", help="Run one configured DOLFINx thesis simulation job."
    )
    f.set_defaults(function=simulate_thesis)
    f.add_argument("--recipe", type=Path, required=True)
    f.add_argument("--list-jobs", action="store_true")
    f.add_argument("--job")
    f.add_argument("--prepared", type=Path)
    f.add_argument("--output", type=Path)
    f.add_argument("--partition", choices=("fit", "calibration", "evaluation"))
    f.add_argument("--workers", type=int)
    f.add_argument(
        "--samples", type=int, help="Reduced quota for testing a simulation recipe."
    )

    prior = sub.add_parser(
        "prior-thesis", help="Rebuild a rank-32 prior from local FIT fields."
    )
    prior.set_defaults(function=prior_thesis)
    prior.add_argument("--fields", type=Path, required=True)
    prior.add_argument("--volume", type=Path, required=True)
    prior.add_argument("--anatomy-id", required=True)
    prior.add_argument("--young", type=float, default=0.003)
    prior.add_argument("--poisson", type=float, default=0.45)
    prior.add_argument("--output", type=Path, required=True)

    camera = sub.add_parser(
        "acquire-thesis", help="Render paired stereo observations from FEM fields."
    )
    camera.set_defaults(function=acquire_thesis)
    camera.add_argument("--fields", type=Path, required=True)
    camera.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Directory of consolidated anatomy archives.",
    )
    camera.add_argument("--output", type=Path, required=True)
    camera.add_argument("--name", required=True)
    camera.add_argument(
        "--partition", choices=("train", "validation", "test"), required=True
    )
    camera.add_argument("--cameras", type=int, default=1)
    camera.add_argument(
        "--identities", type=int, default=0, help="Stationary views per anatomy."
    )
    camera.add_argument("--acquisition-config", type=Path)
    camera.add_argument("--sensor-profile", type=Path)
    for command in (prior, camera):
        command.add_argument("--source-prefix", type=Path)
        command.add_argument("--local-root", type=Path)

    assembly = sub.add_parser(
        "assemble-manifest",
        help="Select and order generated records from frozen lineage.",
    )
    assembly.set_defaults(function=assemble_manifest)
    assembly.add_argument("--manifests", type=Path, nargs="+", required=True)
    assembly.add_argument("--lineage", type=Path, required=True)
    assembly.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
