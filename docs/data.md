# Data and anatomy preparation

Two data paths are provided. `livrenet generate` creates a small independent synthetic study from tetrahedral inputs. The thesis pipeline uses `simulate-thesis`, `prior-thesis` and `acquire-thesis` with the frozen experiment recipes. `import-thesis` can also convert an existing local archive for evaluation or training.

## Input sources

- **3D-IRCADb-01.** Download scans and geometry from [IRCAD](https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/). The source declares [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Keep source scans, prepared meshes, anatomy priors and synthetic observations derived from those anatomies in your local data directory.
- **SERV-CT.** The acquisition profile contains calibrated numerical summaries derived from the dataset by Edwards et al., [reference publication](https://doi.org/10.1016/j.media.2021.102302) and [dataset](https://doi.org/10.5522/04/26352199.v1). The redistributed profile uses [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) in line with the archive's terms. Image pairs, masks and raw depth maps are obtained from the source.
- **P2I-LReg.** The thesis uses observations available through [Self-P2IR](https://github.com/junzastar/Self-P2IR). The repository's synthetic evaluation uses vertex displacement ground truth. The real-data analysis in the thesis uses agreement with observed surfaces.

## Generate an independent synthetic study

Use tetrahedral meshes in a common millimetre coordinate system. Each input `.npz` contains `nodes` of shape `(N, 3)` and integer `tetrahedra` of shape `(T, 4)`. A watertight triangle surface can be converted using TetGen.

```bash
python -m pip install -e '.[mesh]'
livrenet volume --surface liver.stl --max-volume 10 --output data/liver.npz
```

A volume configuration maps each anatomy to its mesh and split. Paths are relative to the configuration file.

```json
{
  "development_liver": {"volume": "liver.npz", "partition": "train"},
  "validation_liver": {"volume": "validation.npz", "partition": "validation"}
}
```

```bash
livrenet generate --volumes data/volumes.json --output data/study \
  --fit-count 144 --heldout-count 54 --seed 13
```

The generator solves P1 linear elasticity with fixed patches and applied traction, rejects excessive local volume change, builds a zero-origin rank-32 prior from FIT fields, and records visible partial observations with paired noise doses. This smaller simulator uses two-view visibility and bounded independent noise. Use the thesis renderer below for the calibrated camera, occlusion and coherent-noise recipe.

## Run the thesis data pipeline

Prepare native closed liver surfaces and corresponding tetrahedral volumes in millimetres. The FEM entry point uses explicit filenames in the recipe bank, resolved below the prepared template directory. Existing prepared meshes are required for numerical replay of the frozen study. Retetrahedralizing a source surface changes its vertices, FEM fields and prior.

```text
prepared/<anatomy-id>/
  liver_surface_from_volume.vtp
  liver_volume.xdmf
  liver_volume.h5
  portalvein_surface_clean.vtp
  venacava_surface_clean.vtp
```

The FEM stage runs in a DOLFINx/PETSc environment. The neural environment can run the prior, acquisition, training and evaluation commands. The stereo renderer also requires a C compiler and PyVista/VTK.

```bash
conda env create -f configs/fem-environment.yml
conda activate livrenet-fem
python -m pip install -e . --no-deps
livrenet simulate-thesis --recipe configs/fem_expanded_test.json --list-jobs
livrenet simulate-thesis --recipe configs/fem_expanded_test.json \
  --job 3Dircadb1.15__weak --partition evaluation \
  --prepared data/prepared --workers 1 --output data/fem/expanded-1.15-weak
```

The three banks cover the donor FIT, extension FIT plus held-out fields, and expanded evaluation fields. A job writes its generated surface and volume fields, the physics contract and `fields.jsonl`. The short `--samples` option changes the simulation budget for a smoke test. Use the stored quotas to regenerate the study.

The frozen prior uses 72 donor and 72 extension FIT fields per anatomy. Combine their records in the corresponding study order. Validation and test deformation records use separate partitions. The prior command filters for `fit` and uses the paired surface and volume displacements to project the FEM stiffness into 32 dimensions.

```bash
livrenet prior-thesis --fields data/local-fit-manifest.jsonl \
  --volume data/fem/fit-1.15/3Dircadb1.15/physics_contract.npz \
  --anatomy-id 3Dircadb1.15 --output data/assets/3Dircadb1.15.npz

livrenet acquire-thesis --fields data/fem/expanded-1.15-weak/fields.jsonl \
  --assets data/assets --partition test --name expanded-1.15-weak \
  --output data/thesis
```

Use `--identities 30` on one combined expanded field catalog per anatomy to add its stationary cameras. Use `--acquisition-config configs/training_acquisition.json` for the extension training and validation observations. The detailed donor camera reuse and test assembly are recorded in the [reproduction protocol](reproduction.md).

## Reuse a local thesis archive

The importer reads explicit original JSONL records plus the three prior asset directories. It handles both fixed-shard observations and the additional noise doses stored as individual point arrays.

```bash
livrenet import-thesis --manifest data/archive/expanded_test.jsonl \
  --assets data/archive/assets --partition test --name expanded \
  --output data/thesis
```

Use `--source-prefix` and `--local-root` together if the original manifest contains absolute paths and the archived tree has moved. The importer maps that prefix directly. The destination contains a self-contained relative-path manifest and local copies of its inputs and labels.

## File contract

| File | Contents |
| --- | --- |
| Anatomy NPZ | `template_points` in mm, `triangles`, `control_indices`, normalization center and scale, rank-32 `basis`, and reduced FEM `metric` |
| Measured points NPY | A finite `(2500, 3)` Cartesian array in the anatomy's frame |
| Target NPY | Displacement in mm on the exact reference vertex indexing |
| Support NPY | Boolean per-vertex ground-truth visibility for metric reporting |
| Observation JSONL | Relative paths, anatomy, parent field, camera, identity flag, split and RMS dose |

The network receives the anatomy and measured points. The training objective and evaluation metrics read targets separately.

The [lineage tables](../configs/lineage/) contain the archived case identifiers and ordering. To combine generated files for the FIT prior or select the exact observations for a study split, pass the corresponding table to `assemble-manifest`.

```bash
livrenet assemble-manifest --manifests data/fem/*/fields.jsonl \
  --lineage configs/lineage/fit.csv --output data/local-fit-manifest.jsonl
```

For the donor training acquisition use `configs/donor_acquisition.json` and generate both views per FIT field and the 16 donor identity views. The training lineage table selects the retained view and preserves the original training order. The extension acquisition uses `configs/training_acquisition.json`; the default acquisition configuration is the expanded test.
