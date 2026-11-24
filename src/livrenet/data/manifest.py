"""Shared records for training and evaluation."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..io import read_jsonl
from ..runtime import Anatomy, point_cloud


class Dataset:
    def __init__(self, manifest):
        manifest = Path(manifest)
        self.root = manifest.parent
        self.rows = read_jsonl(manifest)
        if not self.rows:
            raise ValueError(f"Empty observation manifest: {manifest}")
        ids = [row["sample_id"] for row in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate observation identifiers.")
        if any(Path(name).name != name or name in {"", ".", ".."} for name in ids):
            raise ValueError("Sample identifiers must be single path components.")
        anatomy_files = {}
        for row in self.rows:
            name, path = row["anatomy_id"], row["anatomy"]
            if anatomy_files.setdefault(name, path) != path:
                raise ValueError(
                    f"Anatomy {name} is bound to more than one asset file."
                )

    @lru_cache(maxsize=24)
    def anatomy(self, path):
        return Anatomy(self.root / path)

    def geometry(self, row):
        return self.anatomy(row["anatomy"])

    def points(self, row):
        return point_cloud(np.load(self.root / row["points"], allow_pickle=False))

    def target(self, row):
        vertices = self.geometry(row).vertices
        if row["is_identity"]:
            return np.zeros_like(vertices, dtype=np.float32)
        displacement = np.load(self.root / row["target"], allow_pickle=False)
        if displacement.shape != vertices.shape or not np.isfinite(displacement).all():
            raise ValueError(f"Invalid displacement target for {row['sample_id']}")
        return displacement.astype(np.float32, copy=False)

    def labels(self, row):
        displacement = self.target(row)
        support_path = row.get("support")
        support = (
            np.load(self.root / support_path, allow_pickle=False)
            if support_path
            else None
        )
        if support is not None and (
            support.shape != (len(displacement),) or support.dtype != bool
        ):
            raise ValueError(f"Invalid visible support for {row['sample_id']}")
        return displacement, support
