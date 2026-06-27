# Python Scripts

This directory contains the data-processing, voxel-generation, model-training,
evaluation, and visualization scripts used in the study.

## Workflow

```text
MD topology and snapshots
        |
        v
Molecular features and voxel grids
        |
        v
Serialized voxel datasets
        |
        v
3D-CNN training and evaluation
        |
        v
Figures and interpretation analyses
```

## Configuration

- `core/path.py` resolves repository paths on supported local and cluster
  environments. The repository location can also be supplied through the
  `ZEOLITE_SOLVATION_PATH` environment variable.
- `core/global_vars.py` defines the systems, adsorbates, energy-table names,
  and molecular features shared by the workflow.

## MD Processing and Molecular Features

- `read_md_snapshot.py` combines a LAMMPS topology file with one sampled MD
  configuration and returns an annotated MDAnalysis universe.
- `view_md_snapshot.py` prepares ASE visualizations of either the full system or
  a local environment centered on the adsorbate.
- `extract_lammpsdata_info.py` extracts force-field and atom-level descriptors
  from the LAMMPS data and snapshot files.
- `extract_hbonds.py` identifies adsorbate-solvent hydrogen bonds and related
  donor/acceptor information.

## Voxel Dataset Preparation

- `generate_voxel_grids.py` converts one MD snapshot into a voxel-grid
  representation and associates it with the corresponding DFT label from
  `database/`.
- `augment_voxel_grids.py` applies the rotational data augmentation used for
  the voxelized configurations.
- `store_grids_pickle.py` generates and serializes complete voxel datasets to
  `dataset_cnn/`.
- `load_grids_pickle.py` loads the serialized grids and verifies their metadata,
  dimensions, labels, and completeness before model training.

## Model Training

- `model_3d_cnn.py` defines the attention-enhanced 3D-CNN architecture.
- `train_3d_cnn.py` loads voxel datasets, performs cross-validation, trains and
  evaluates the model, and stores checkpoints and training records in
  `output_model_cnn/`.
- `train_3d_cnn.sh` is the Slurm launcher used for GPU training on the original
  computing cluster. Its account, environment, and resource settings should be
  adapted for another cluster.

## Analysis and Visualization

- `plot_data_distribution.py` summarizes the DFT energy labels and MD sampling
  distributions. It generates the manuscript data-distribution figure and
  additional diagnostic plots.
- `plot_3d_cnn_results.py` extracts cross-validation predictions and metrics and
  generates model-performance figures.
- `plot_training_info.py` visualizes losses, learning behavior, and training
  diagnostics.
- `plot_attention_results.py` analyzes and visualizes learned spatial-attention
  maps.
- `plot_importance_captum.py` performs Captum-based feature-attribution
  analysis.
- `plot_importance_maps.py` generates three-dimensional feature-importance maps
  from trained models.

Generated figures are written to subdirectories of `output_figures/`.

## Environment Test

`test_pytorch.py` reports the Python and PyTorch versions, detects available CPU,
CUDA, and Apple MPS devices, and performs a small forward/backward test. Run:

```bash
python python_scripts/test_pytorch.py --skip-benchmark
```

`test_pytorch.sh` provides a portable local or Slurm launcher for the same test.
The individual research scripts also contain example parameters or control
settings in their main blocks; review these settings before processing a new
system or launching a complete dataset run.
