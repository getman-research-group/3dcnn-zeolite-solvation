# 3D-CNN Zeolite Solvation

This repository accompanies the manuscript:

**Predicting Solvation Thermodynamics of Adsorbates in Zeolite Pores Using
Convolutional Neural Networks with Attention Mechanisms**

It contains the interaction-energy database, molecular-simulation examples,
voxelization and model-training code, trained five-fold 3D-CNN models, and the
analysis scripts used to generate the manuscript and Supporting Information
figures.

![Workflow overview](overall_workflow.png)

**Workflow overview.** Molecular dynamics (MD) configurations are annotated
with molecular and force-field descriptors, converted into 28-channel voxel
grids, and used by an attention-enhanced 3D-CNN to predict DFT-derived
adsorbate–solvent interaction energies in Ti-FAU zeolite pores.

## Repository structure

```text
3dcnn-zeolite-solvation/
├── database/                # DFT and MD interaction-energy tables
├── dataset_cnn/             # Generated voxel datasets (large pickles ignored by Git)
├── md_simulations/          # Two representative FAU MD/DFT systems
├── output_model_cnn/        # Five trained fold checkpoints and result summary
├── output_figures/          # Manuscript, SI, and supporting visualizations
├── python_scripts/          # Preprocessing, training, evaluation, and interpretation
│   └── core/                # Shared paths, systems, features, and configuration
├── environment.yml          # Reproducible CPU-compatible Conda environment
├── overall_workflow.png
├── LICENSE
└── README.md
```

Additional directory-level documentation is available for the
[Python scripts](python_scripts/README.md),
[energy database](database/README.md),
[representative simulations](md_simulations/FAU/README.md), and
[voxel dataset configuration](dataset_cnn/size_16.0-box_0.8-shape_20_20_20_28/README.md).

## Data provided

### Energy labels

`database/` contains eight CSV files covering 800 snapshots from 80 FAU
zeolite–environment–adsorbate systems. The data include hydrophilic and
hydrophobic pores, pure water, and three methanol–water compositions. The
snapshot-level DFT interaction energy in the `intE` column is the target used
for model training.

CSV filenames follow:

```text
<zeolite>-<solvent_composition>-<pore_type>.csv
```

### Representative MD and DFT systems

`md_simulations/FAU/` contains two representative systems:

- 2-propanol in hydrophilic Ti-FAU with 240 methanol and 960 water molecules.
- Propylene glycol in hydrophilic Ti-FAU with pure water.

Each example documents the LAMMPS topology and sampling trajectory, ten sampled
configurations, CP2K inputs and outputs, optimized structures, and the
bare-zeolite reference calculation.

### Trained models

`output_model_cnn/` contains:

- `model-fold_0.pth` through `model-fold_4.pth`: PyTorch checkpoints for the
  five cross-validation folds.
- `model.pkl`: the corresponding result summary, including fold predictions,
  performance metrics, training histories, and data-split information used by
  the analysis scripts.

### Voxel datasets and GitHub file-size limits

Each system-level voxel pickle is larger than 400 MB and therefore exceeds
GitHub's 100 MB per-file limit. These pickle files are intentionally excluded
from Git and must be generated locally from the supplied simulation files. A
pickle contains ten MD snapshots, each expanded through the 24 proper cube
rotations, for 240 augmented `20 × 20 × 20 × 28` voxel grids.

The 28 channels are separated into:

- channels 0–13: adsorbate atomic features;
- channels 14–27: solvent atomic features.

## Installation

Create the tested CPU-compatible environment from the repository root:

```bash
conda env create -f environment.yml
conda activate zeolite-3dcnn
```

Verify Python, PyTorch, and the available compute devices:

```bash
python python_scripts/test_pytorch.py --skip-benchmark
```

The supplied environment supports preprocessing, plotting, model loading, and
CPU evaluation. GPU training requires a CUDA-enabled PyTorch build compatible
with the CUDA installation on the target system.

Scripts resolve the repository root automatically. A custom location may be
specified when needed:

```bash
export ZEOLITE_SOLVATION_PATH=/path/to/3dcnn-zeolite-solvation
```

## Reproducing the preprocessing workflow

Run commands from the repository root.

Generate the representative mixed-solvent voxel pickle:

```bash
python python_scripts/store_grids_pickle.py \
  --test \
  --zeolite FAU \
  --environment methanol_240_water_960-hydrophilic \
  --adsorbate 02_01_02_propanol
```

The generated file is written to:

```text
dataset_cnn/size_16.0-box_0.8-shape_20_20_20_28/
```

Add `--force-regenerate` to rebuild an existing file. To generate every system
configured in `python_scripts/core/global_vars.py`, use:

```bash
python python_scripts/store_grids_pickle.py --all
```

Full-dataset generation requires the corresponding simulation directories for
all configured systems. The GitHub repository contains two representative
systems rather than the complete collection of raw trajectories.

## Model training

After the required voxel pickles are present, train the five-fold 3D-CNN with:

```bash
python python_scripts/train_3d_cnn.py
```

For a Slurm GPU cluster, adapt the account, environment, and resource settings
in `python_scripts/train_3d_cnn.sh` and submit that launcher instead. Training
produces one `.pth` checkpoint per fold and one `.pkl` result summary in
`output_model_cnn/`.

## Reproducing figures and interpretation analyses

The principal figure scripts are:

| Script | Output |
| --- | --- |
| `plot_data_distribution.py` | DFT energy-distribution analysis, including Figure 2 |
| `plot_3d_cnn_results.py` | Figure 6 and Figures S5.1 and S5.4 |
| `plot_attention_results.py` | Learned spatial-attention heatmaps for Figure 7 |
| `plot_importance_spatial.py` | Integrated Gradients and Layer Conductance spatial attribution maps for Figures S5.5–S5.7 |
| `plot_training_info.py` | Fold-averaged loss and gradient-norm training figure |

For example:

```bash
python python_scripts/plot_data_distribution.py
python python_scripts/plot_3d_cnn_results.py
python python_scripts/plot_training_info.py
```

The attention and spatial-attribution scripts operate on a user-selected
zeolite, environment, adsorbate, snapshot, and voxel rotation. Their parameter
blocks should be checked before execution, and the corresponding voxel pickle
must exist locally. Generated figures are written to subdirectories of
`output_figures/`.

## Reproducibility scope

The repository supports:

- inspection of the complete interaction-energy label tables;
- end-to-end voxel generation for the representative systems;
- inspection and evaluation of the supplied trained fold models;
- regeneration of model-performance, attention, attribution, and training
  visualizations when their required local voxel inputs are present.

The complete raw MD/DFT collection and complete voxel dataset are substantially
larger than a standard GitHub repository. They are not distributed here in
full; the included examples document the expected directory structure and
preprocessing procedure.

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@article{shi_zeolite_3dcnn,
  title   = {Predicting Solvation Thermodynamics of Adsorbates in Zeolite Pores Using Convolutional Neural Networks with Attention Mechanisms},
  author  = {Shi, Jiexin and Chen, Xiuting and Getman, Rachel B.},
  journal = {Journal of Chemical Information and Modeling},
  note    = {Manuscript submitted}
}
```

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).

## Contact

Rachel B. Getman  
William G. Lowrie Department of Chemical and Biomolecular Engineering  
The Ohio State University  
getman.11@osu.edu
