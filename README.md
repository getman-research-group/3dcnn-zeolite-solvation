
# 3D-CNN Zeolite Solvation

**Predicting Solvation Thermodynamics of Adsorbates in Zeolite Pores Using 3D Convolutional Neural Networks**

---

## 📘 Overview

This repository contains the data processing scripts, voxelization utilities, and deep learning models developed for predicting **adsorbate–solvent interaction energies** in zeolite pores using **3D Convolutional Neural Networks (3D-CNNs)** with attention mechanisms.

The workflow integrates **classical molecular dynamics (MD)** sampling, **density functional theory (DFT)** single-point energy calculations, and **deep learning–based regression** to efficiently reproduce solvation thermodynamic properties that would otherwise require extensive first-principles computations.

---

## 🧪 Background

Understanding solvation thermodynamics in confined environments such as **zeolite nanopores** is crucial for modeling liquid-phase catalysis.  
Our approach builds upon the *Multiscale Sampling (MSS)* framework originally developed in **Chen et al., J. Phys. Chem. C 2024, 128, 19367–19379**, which combined MD and DFT calculations to evaluate solvation free energies of C₁–C₃ oxygenates in hydrophilic and hydrophobic Ti-FAU zeolite pores.

This project extends that framework by introducing a **data-driven 3D deep learning model** that predicts adsorbate–solvent interaction energies directly from MD snapshots, eliminating the need for DFT evaluations in large-scale screening studies.

---

## 🧰 Repository Structure

---

## 🧩 Methodology

### 1. Dataset Generation
- **Systems:** 80 unique *adsorbate–environment* combinations (C₁–C₃ oxygenates) in hydrophilic/hydrophobic Ti-FAU pores.
- **Snapshots:** 10 MD snapshots per system (800 total), each labeled with DFT-calculated adsorbate–solvent interaction energy.
- **Augmentation:** 24 symmetry rotations per snapshot → 19,200 total samples.

### 2. Voxelization
Each MD snapshot is converted into a **20×20×20 voxel grid** (0.8 Å resolution) centered on the adsorbate’s center of mass.
- 28 feature channels (14 for adsorbate, 14 for solvent).
- Features include atom types, charges, masses, Lennard-Jones parameters, and hydrogen bonding indicators.
- Zeolite framework atoms are excluded (fixed during MD).
- Implemented with **MDAnalysis** and **NumPy**.

### 3. Neural Network
A **dual-branch 3D CNN** with **channel & spatial attention modules (CBAM)** learns correlations between adsorbate and solvent voxel features.

Key architecture features:
- Dual processing branches for adsorbate and solvent channels.
- Group-aware interaction attention layer for adsorbate–solvent coupling.
- Residual CNN backbone for hierarchical spatial feature extraction.
- Final regression head outputs ΔE<sub>int</sub> predictions.

---

## 📊 Model Performance

| Split Type | Metric | MAE (eV) | RMSE (eV) | R² |
|-------------|---------|-----------|-----------|-----|
| Adsorbate-based | Snapshot ΔE<sub>int</sub> | 0.083 | 0.110 | 0.82 |
| Adsorbate-based | Adsorbate ⟨ΔE<sub>int</sub>⟩ | 0.037 | 0.047 | 0.95 |
| Solvent-based | Adsorbate ⟨ΔE<sub>int</sub>⟩ | 0.036 | 0.048 | 0.94 |
| Pore-type-based | Adsorbate ⟨ΔE<sub>int</sub>⟩ | 0.059 | 0.071 | 0.89 |

Performance meets **DFT-level accuracy (<0.1 eV)** with **>90% variance explained** for ensemble-averaged predictions.

---

## 🔍 Visualization

Attention heatmaps reveal how the model identifies physically relevant spatial regions:

<p align="center">
  <img src="results/attention_heatmaps/example_attention.png" width="600">
</p>

The attention focuses on **adsorbate–solvent interfacial regions**, validating the interpretability of learned spatial features.

---

## ⚙️ Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt

Core packages:
	•	Python ≥ 3.9
	•	PyTorch ≥ 2.0
	•	MDAnalysis ≥ 2.6
	•	NumPy, SciPy, scikit-learn
	•	Matplotlib, Seaborn
	•	tqdm, h5py

🚀 Training
Example command to train the 3D-CNN:
python scripts/train_3dcnn_model.py --data_dir ./data/voxels/ --epochs 200 --batch_size 32 --lr 2e-4



🧾 License

This project is released under the MIT License.
See the LICENSE￼ file for details.

© 2025 Getman Research Group – The Ohio State University