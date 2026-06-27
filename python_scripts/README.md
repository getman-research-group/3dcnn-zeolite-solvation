# Python Scripts

This directory contains the Python scripts used to prepare molecular-simulation
data, build voxel datasets, train the 3D-CNN models, and generate the analysis
figures reported in the study.

The scripts are organized around one main workflow:

1. Read one LAMMPS topology file and one sampled MD snapshot.
2. Convert one snapshot into an adsorbate-centered voxel grid.
3. Apply the 24 rotational augmentations used for model training.
4. Store one pickle file for each zeolite-environment-adsorbate system.
5. Load the serialized voxel data and train the attention-based 3D-CNN with
   grouped cross-validation.
6. Plot model performance, training behavior, attention patterns, feature
   attribution results, and data-distribution summaries.

## Workflow

```text
database/*.csv + md_simulations/
          |
          v
read_md_snapshot.py
          |
          v
extract_lammpsdata_info.py + extract_hbonds.py
          |
          v
generate_voxel_grids.py
          |
          v
augment_voxel_grids.py
          |
          v
store_grids_pickle.py
          |
          v
load_grids_pickle.py
          |
          v
train_3d_cnn.py + model_3d_cnn.py
          |
          v
plot_*.py / interpretation analyses
```

## Shared Configuration

- `core/path.py` resolves repository paths for the supported local and cluster
  environments. The repository root can also be supplied through the
  `ZEOLITE_SOLVATION_PATH` environment variable.
- `core/global_vars.py` stores the shared configuration used across the
  workflow, including zeolite types, environment-to-adsorbate mappings,
  feature names, and label-table definitions.

## Step 1: Read MD Topology and Snapshots

- `read_md_snapshot.py` loads one molecular system from the repository's MD
  simulation layout:
  `md_simulations/<zeolite>/<solvent>-<pore_type>/<adsorbate>/`.
  It combines the shared `data_nvt_samp_new.lammpsdata` topology with one
  sampled snapshot trajectory `intE<index>/intE<index>.traj`, constructs an
  `MDAnalysis.Universe`, and assigns consistent residue names for water,
  methanol, zeolite, and adsorbate atoms.
- `view_md_snapshot.py` creates ASE-based visualizations of either the full MD
  system or a local adsorbate-centered environment for inspection.

## Step 2: Extract Atom-Level Descriptors

- `extract_lammpsdata_info.py` reads the topology and annotated snapshot to
  derive atom-level descriptors needed for voxelization. These include
  Lennard-Jones epsilon and sigma values, inferred adsorbate connectivity,
  atom valence, hydrophobic indicators, and donor/acceptor-capacity flags.
- `extract_hbonds.py` detects adsorbate-solvent hydrogen bonds in one snapshot
  using `MDAnalysis.analysis.hydrogenbonds.HydrogenBondAnalysis`. It produces
  per-atom Boolean descriptors such as `is_hbonded`, `is_hbonded_donor`, and
  `is_hbonded_acceptor` that are later written into voxel channels.

## Step 3: Generate One Voxel Grid from One Snapshot

- `generate_voxel_grids.py` is the central preprocessing step for one sampled
  MD configuration. For a selected zeolite, environment, adsorbate, and
  snapshot index, it:

  - loads the MD snapshot and corresponding DFT interaction-energy label,
  - centers a cubic grid on the adsorbate center of mass,
  - applies the minimum-image convention to select nearby atoms,
  - accumulates atom-level descriptors into a 4D voxel tensor,
  - stores adsorbate features in channels `0-13` and solvent features in
    channels `14-27`,
  - normalizes continuous channels and applies the PCMax-style van der Waals
    saturation to discrete channels.

  Under the manuscript defaults, one snapshot is represented as a
  `20 x 20 x 20 x 28` grid.

## Step 4: Apply Rotational Augmentation

- `augment_voxel_grids.py` applies the 24 proper rotational symmetries of a
  cube to one voxel grid using exact 90-degree rotations implemented with
  `numpy.rot90`. These transformations preserve voxel values, channel order,
  and the DFT target label while changing only the spatial orientation.
- The module also provides helper functions to visualize the rotated grids and
  to check augmentation quality, including sum preservation, label consistency,
  and duplicate detection.

## Step 5: Build and Store System-Level Pickle Files

- `store_grids_pickle.py` connects snapshot-level voxel generation to
  augmentation and dataset serialization.
- For one zeolite-environment-adsorbate system, it processes snapshots
  `intE01` through `intE10`, generates one original voxel grid per snapshot,
  augments each grid into 24 orientations, and stores the results as one pickle
  file in `dataset_cnn/`.
- Under the manuscript defaults, each system-level pickle contains:

  - `10` original snapshot grids kept for inspection,
  - `240` augmented grids used for machine learning,
  - top-level `metadata` describing the grid geometry, feature ordering,
    channel mapping, and system identity,
  - a `snapshots` dictionary keyed by snapshot index.

- `load_grids_pickle.py` reads these serialized pickle files back into memory,
  checks their expected dimensions and metadata, and assembles them for model
  training.

## Step 6: Train the Attention-Based 3D-CNN

- `model_3d_cnn.py` defines the dual-branch 3D-CNN used in this project. The
  architecture processes adsorbate and solvent channel groups separately,
  fuses them, and applies CBAM-based attention to learn spatially resolved
  interaction patterns.
- `train_3d_cnn.py` loads the serialized voxel data, converts voxel arrays from
  `(N, 20, 20, 20, 28)` to PyTorch's `(N, 28, 20, 20, 20)` format, scales the
  target energies, and trains the model with grouped cross-validation.
- The training code supports several split modes, including `random_split`,
  `solvent_split`, and `pore_type_split`. In the grouped split used for model
  evaluation, all data points from the same adsorbate are kept in the same fold
  to avoid leakage across environments.
- Model checkpoints, training records, fold-level predictions, and summary
  results are written to `output_model_cnn/`.
- `train_3d_cnn.sh` is the original SLURM launcher for GPU training on the
  research cluster. Its account name, environment activation, and resource
  requests should be adapted before use on another cluster.

## Step 7: Analyze the Dataset and Model Results

- `plot_data_distribution.py` loads the DFT label tables from `database/` and
  generates the distribution and MD-sampling figures used to summarize the
  dataset.
- `plot_3d_cnn_results.py` loads saved training results, extracts fold-level
  predictions and performance metrics, and generates model-evaluation plots.
- `plot_training_info.py` visualizes learning curves, validation behavior, and
  other training diagnostics from saved model outputs.
- `plot_attention_results.py` extracts and visualizes spatial attention maps
  from the trained dual-branch model.
- `plot_importance_captum.py` performs Captum-based feature-attribution
  analysis on trained models.
- `plot_importance_maps.py` generates 3D voxel importance maps from trained
  models for interpretation of learned interactions.

All analysis figures are written to subdirectories of `output_figures/`.

## Environment Test

- `test_pytorch.py` checks the local Python and PyTorch installation, reports
  CPU/CUDA/MPS availability, and runs a small forward/backward smoke test.
  Example:

  ```bash
  python python_scripts/test_pytorch.py --skip-benchmark
  ```

- `test_pytorch.sh` is a companion launcher for running the same environment
  check locally or through SLURM.

## Practical Notes

- The individual scripts contain example parameters or control flags in their
  `__main__` blocks. Review those settings before launching a full dataset or
  training run.
- `store_grids_pickle.py` now supports a focused test run for one selected
  system as well as a full dataset-generation mode.
- The repository's dataset organization is system-based: one pickle corresponds
  to one zeolite-environment-adsorbate combination rather than to an adsorbate
  pooled across multiple environments.
