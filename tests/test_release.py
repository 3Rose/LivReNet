import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from livrenet.data.catalog import assemble
from livrenet.data.synthesis import fixture_mesh, generate
from livrenet.data.manifest import Dataset
from livrenet.evaluation import macro
from livrenet.io import read_jsonl, sha256, write_jsonl
from livrenet.runtime import Predictor

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_checkpoint_manifest(self):
        folder = ROOT / "checkpoints"
        meta = json.loads((folder / "manifest.json").read_text())
        for name, epoch in (("encoder", 58), ("refiner", 31)):
            checkpoint = folder / f"{name}.pt"
            self.assertEqual(sha256(checkpoint), meta[f"{name}_sha256"])
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(payload["selected_epoch"], epoch)
            self.assertFalse(payload["test_used_for_selection"])
        self.assertEqual(
            sha256(folder / "normalization.json"), meta["normalization_sha256"]
        )

    def test_fem_fixture_and_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nodes, cells = fixture_mesh()
            counts = generate(
                root,
                {"small": (nodes, cells, "test")},
                seed=13,
                fit_count=36,
                heldout_count=1,
            )
            self.assertEqual(counts["test"], 8)
            data = Dataset(root / "test.jsonl")
            row = data.rows[0]
            anatomy = data.geometry(row)
            lookup = np.load(anatomy.path)
            self.assertNotIn(row["parent_id"], set(lookup["fit_ids"]))
            config = json.loads((ROOT / "src/livrenet/thesis.json").read_text())
            model = Predictor(
                ROOT / "checkpoints/encoder.pt",
                config,
                "cpu",
                ROOT / "checkpoints/refiner.pt",
            )
            fields = model.predict(anatomy, data.points(row))
            self.assertEqual(set(fields), {"solve", "fitted", "final"})
            for output in fields.values():
                self.assertEqual(output.shape, anatomy.vertices.shape)
                self.assertTrue(np.isfinite(output).all())

    def test_macro_keeps_anatomies_balanced(self):
        observations = [
            dict(anatomy_id="one", parent_id="a", noise_mm=0.0, metrics={"error": 2.0}),
            dict(anatomy_id="one", parent_id="a", noise_mm=0.5, metrics={"error": 4.0}),
            dict(anatomy_id="one", parent_id="b", noise_mm=0.0, metrics={"error": 5.0}),
            dict(anatomy_id="two", parent_id="c", noise_mm=0.0, metrics={"error": 8.0}),
        ]
        self.assertEqual(macro(observations, "error"), 6.0)

    def test_assemble_preserves_lineage_order_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "raw" / "records.jsonl"
            write_jsonl(
                source,
                [
                    {"sample_id": "a", "anatomy_id": "one", "points": "a.npy"},
                    {"sample_id": "b", "anatomy_id": "one", "points": "b.npy"},
                ],
            )
            lineage = folder / "lineage.csv"
            lineage.write_text("anatomy_id,sample_id\none,b\none,a\n")
            path = assemble([source], lineage, folder / "ordered.jsonl")
            self.assertEqual(
                [row["points"] for row in read_jsonl(path)], ["raw/b.npy", "raw/a.npy"]
            )


if __name__ == "__main__":
    unittest.main()
