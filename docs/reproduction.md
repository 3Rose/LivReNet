# Reproduction protocol

## Frozen model

The checkpoints are the two seed-13 models selected after training from scratch for the thesis synthetic comparison. The encoder was selected at epoch 58 of 80. The refiner was selected at epoch 31 of 40. The [checkpoint manifest](../checkpoints/manifest.json) binds both weight files and the training-only normalization via SHA-256.

The fitted variant returns the numerical PCA/FEM output initialized by the encoder. The full variant applies the learned residual on the native mesh. Fitting uses the balanced 2/1/1/3 iteration schedule. The extended geometric control uses 5/1/2/24. CUDA reproduces the thesis execution path. CPU uses Open3D for mesh projection and can introduce small numerical differences.

## Split and preparation

The [cohort file](../configs/cohorts.json) defines 13 training, three validation and three test anatomies. The checkpoint objective is averaged over validation anatomies 1.7 and 1.19. Case 1.11 is monitored separately. All blocks in the model are shared across anatomies. The fixed PCA/FEM prior is specific to each anatomy, including those used for validation and test.

The training archive has 4,160 measured point clouds and validation has 360, with 0 and 0.5 mm 3-D RMS noise. The donor FIT block uses the first camera in lexicographic order per deformation and retains the 16 donor stationary cameras per training anatomy. The extension uses one view per field. The training manifest in the retained archive is the direct source for the exact donor-camera selection and ordering.

The expanded test contains 810 independent deformation fields and 90 stationary camera acquisitions across 1.13, 1.15 and 1.16. Each has four matched noise doses of 0, 0.5, 1 and 1.5 mm RMS. The generator accepts informative views with at least 20% visible surface and at least 25% peak-region support on deforming cases. The higher doses are out of the training noise range. The panel was evaluated retrospectively with frozen weights and priors.

For exact matching with archived observations, use the retained native templates, FIT fields and manifests from the thesis study. A new generation run can be compared against the archive by checking topology, prior assets and observation IDs before comparing errors. The repository includes the FEM and acquisition recipes, along with an archive importer for replay from these retained artifacts.

## Run the primary comparison

Once the expanded manifest has been imported or materialized, run the frozen model and geometric controls.

```bash
livrenet evaluate --manifest data/thesis/expanded.jsonl \
  --encoder checkpoints/encoder.pt --refiner checkpoints/refiner.pt \
  --baselines --predictions --device cuda --output outputs/expanded
```

The evaluation writes per-observation metrics and aggregate summaries. It averages cameras within noise dose and parent, doses within parent, parents within anatomy, then anatomies. Deforming and stationary cases are reported separately. Peak EPE uses vertices where target displacement reaches at least 80% of the case maximum, for cases with maximum displacement of at least 1 mm. The original nine-system summaries include separately trained ablation variants and synchronized timing. The evaluation command covers the principal model's internal stages and geometric controls.

For retraining, import the retained training and validation manifests with the corresponding partitions, then run:

```bash
livrenet train --train-manifest data/thesis/train.jsonl \
  --validation-manifest data/thesis/validation.jsonl \
  --device cuda --output runs/thesis
```

The command writes the selected checkpoints, training history and resumable epoch state. The refiner's fitted inputs and normalization are bound to the selected encoder and data manifests.

## Release verification

The extracted CUDA pipeline reproduced all three stored displacement fields on 24 observations covering the three test anatomies, four noise doses, and deforming and stationary cases. A complete 30-field weak-deformation job reproduced the stored surface and volumetric fields for anatomy 1.15. Rebuilding that anatomy's prior reproduced the frozen basis, projected energy and control indices. A replayed stereo acquisition reproduced its archived clean point cloud.

Run the portable regression tests with `python -m unittest discover -s tests`. The synthetic fixture can also be used to exercise both training stages and the evaluation CLI on CPU.
