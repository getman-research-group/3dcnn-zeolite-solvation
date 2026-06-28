# Python Scripts

This directory contains the scripts used to process the molecular-simulation
data, build voxel datasets, train the 3D-CNN models, and generate the figures
and interpretation analyses reported in the study.

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
  and molecular features used consistently throughout the workflow.

## MD Processing and Molecular Features

- `read_md_snapshot.py` combines a LAMMPS topology file with a sampled MD
  configuration and returns an annotated MDAnalysis universe.
- `view_md_snapshot.py` prepares ASE visualizations of either the full system or
  a local environment centered on the adsorbate.
- `extract_lammpsdata_info.py` extracts force-field parameters and atom-level
  features from the LAMMPS data and snapshot files.
- `extract_hbonds.py` identifies adsorbate-solvent hydrogen bonds and assigns
  the related donor/acceptor information.

## Voxel Dataset Preparation

- `generate_voxel_grids.py` converts one MD snapshot into a voxel-grid
  representation and associates it with the corresponding DFT label from
  `database/`.
- `augment_voxel_grids.py` applies the rotational data augmentation used for
  the voxelized configurations.
- `store_grids_pickle.py` generates complete voxel datasets and serializes them
  to `dataset_cnn/`.
- `load_grids_pickle.py` loads the serialized grids and checks their metadata,
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
  distributions. It generates the manuscript data-distribution figure together
  with additional diagnostic plots.
- `plot_3d_cnn_results.py` extracts cross-validation predictions and metrics and
  generates the model parity plot (Figure 6) and test-time-augmentation
  comparison (Figure S5.4).
- `plot_training_info.py` visualizes losses, learning behavior, and training
  diagnostics.
- `plot_attention_results.py` extracts the spatial-attention weights learned by
  the adsorbate, solvent, and interaction modules and generates the attention
  heatmaps reported as Figure 7 in the manuscript.
- `plot_importance_spatial.py` uses Captum Integrated Gradients and Layer
  Conductance to generate the input- and processor-level spatial attribution
  maps reported as Figures S5.5, S5.6, and S5.7 in the Supporting Information.

Generated figures are written to subdirectories of `output_figures/`.

## Environment Test

`test_pytorch.py` reports the Python and PyTorch versions, detects available CPU,
CUDA, and Apple MPS devices, and performs a small forward/backward test. Run:

```bash
python python_scripts/test_pytorch.py --skip-benchmark
```

`test_pytorch.sh` provides a portable local or Slurm launcher for the same test.
Several research scripts also include example parameters or control settings in
their main blocks; review these settings before processing a new system or
launching a complete dataset run.
