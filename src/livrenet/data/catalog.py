"""Assemble generated manifests in the order recorded by a frozen lineage table."""

import csv
import os

from ..io import read_jsonl, write_jsonl

PATH_FIELDS = ("anatomy", "points", "target", "support", "mesh_path", "template_path")


def assemble(manifests, lineage, output):
    with lineage.open(newline="") as stream:
        reader = csv.DictReader(stream)
        identifier = "mesh_id" if "mesh_id" in reader.fieldnames else "sample_id"
        order = [(row["anatomy_id"], row[identifier]) for row in reader]
    if not order or len(order) != len(set(order)):
        raise ValueError("The lineage table must contain distinct records.")
    wanted = set(order)
    available = {}
    for manifest in manifests:
        for row in read_jsonl(manifest):
            key = (row["anatomy_id"], row[identifier])
            if key not in wanted:
                continue
            for name in PATH_FIELDS:
                if row.get(name):
                    row[name] = os.path.relpath(
                        manifest.parent / row[name], output.parent
                    )
            previous = available.setdefault(key, row)
            if previous != row:
                raise ValueError(f"Conflicting paths or metadata for {key}")
    missing = wanted - available.keys()
    if missing:
        examples = ", ".join("/".join(key) for key in sorted(missing)[:3])
        raise ValueError(
            f"{len(missing)} lineage records are missing, including {examples}"
        )
    write_jsonl(output, [available[key] for key in order])
    return output
