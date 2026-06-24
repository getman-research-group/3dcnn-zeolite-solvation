# 3D-CNN Zeolite Solvation

This repository accompanies the manuscript:

**Predicting Solvation Thermodynamics of Adsorbates in Zeolite Pores Using Convolutional Neural Networks with Attention Mechanisms**

Jiexin Shi, Xiuting Chen, and Rachel B. Getman

The repository contains the data, scripts, trained models, example molecular simulation files, and generated figures used to support the manuscript. The workflow uses molecular dynamics (MD) snapshots of adsorbates in Ti-FAU zeolite pores, converts local adsorbate-solvent configurations into voxel-grid inputs, and trains attention-enhanced three-dimensional convolutional neural networks to predict DFT-derived adsorbate-solvent interaction energies.

![Workflow overview](figure_1.png)

## Repository Contents

```text
3dcnn-zeolite-solvation/
├── python_scripts/          # Data processing, voxelization, training, evaluation, and plotting scripts
├── database/                # Energy tables and source data used by the training scripts
├── dataset_cnn/             # Voxel-grid datasets used as model inputs
├── output_model_cnn/        # Trained model checkpoints and training records
├── output_figures/          # Figures and visualization outputs generated from the analysis
├── md_simulations/          # Representative MD and DFT input/output files
├── figure_1.png             # Workflow overview figure
├── LICENSE
└── README.md
```

The `md_simulations/` directory includes representative LAMMPS and CP2K input files needed to document the molecular simulation setup used in the manuscript. The `database/` and `dataset_cnn/` directories provide the energy labels and voxelized molecular configurations used by the machine-learning workflow.

## Installation

Create a Python environment with the scientific Python and deep-learning packages used by the scripts. A typical setup is:

```bash
conda create -n zeolite-3dcnn python=3.10
conda activate zeolite-3dcnn
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch mdanalysis captum
```

GPU-enabled PyTorch is recommended for model training. CPU execution is sufficient for inspecting datasets and running smaller analysis scripts.

## Basic Usage

Run commands from the repository root.

Generate voxel-grid datasets from simulation files:

```bash
python python_scripts/generate_voxel_grids_type_2.py
```

Train the 3D-CNN model:

```bash
python python_scripts/train_3d_cnn_2_8.py
```

Generate parity plots and related analysis figures:

```bash
python python_scripts/plot_3d_cnn_results.py
```

Some scripts contain path and run-configuration variables near the top of the file. Update those paths if the repository is moved or if a subset of the data is being analyzed.

## Reproducing the Manuscript Workflow

The main workflow is:

1. Prepare energy-label tables in `database/`.
2. Convert representative MD snapshots in `md_simulations/` into voxel-grid datasets in `dataset_cnn/`.
3. Train and evaluate the 3D-CNN models using scripts in `python_scripts/`.
4. Save trained checkpoints to `output_model_cnn/`.
5. Generate manuscript and supporting-information figures in `output_figures/`.

The uploaded data include a clean energy table, representative voxelized systems, trained model checkpoints, and example LAMMPS and CP2K input files so that the key data-processing, model-training, and analysis steps can be reproduced.

## Citation

If you use this repository, please cite the accompanying manuscript:

```bibtex
@article{shi_zeolite_3dcnn,
  title = {Predicting Solvation Thermodynamics of Adsorbates in Zeolite Pores Using Convolutional Neural Networks with Attention Mechanisms},
  author = {Shi, Jiexin and Chen, Xiuting and Getman, Rachel B.},
  journal = {Journal of Chemical Information and Modeling},
  note = {Manuscript submitted}
}
```

## License

This project is distributed under the MIT License. See `LICENSE` for details.

## Contact

Rachel B. Getman  
William G. Lowrie Department of Chemical and Biomolecular Engineering  
The Ohio State University  
getman.11@osu.edu
