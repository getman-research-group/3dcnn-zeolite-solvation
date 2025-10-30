
# 3D-CNN for Zeolite Solvation Prediction

**Predicting Adsorbate–Solvent Interaction Energies in Zeolite Nanopores Using 3D Convolutional Neural Networks**

<p align="center">
  <img src="figure_1.png" width="800" alt="Graphical Abstract">
</p>

---

## Overview

This repository contains code and models for predicting **adsorbate–solvent interaction energies (ΔE<sub>int</sub>)** in zeolite nanopores using **3D Convolutional Neural Networks** with attention mechanisms.

Our approach combines:
- **Molecular Dynamics (MD)** sampling for structural configurations
- **Density Functional Theory (DFT)** for reference energy calculations
- **3D-CNN with attention** for fast, accurate energy prediction

By learning from 3D voxelized representations of MD snapshots, our model achieves **DFT-level accuracy** while reducing computational cost by orders of magnitude—enabling high-throughput screening of solvation thermodynamics in confined environments.

### Key Features
✅ **DFT-level accuracy**: MAE < 0.05 eV for ensemble predictions  
✅ **Strong generalization**: R² > 0.90 across unseen adsorbates and solvent compositions  
✅ **Physically interpretable**: Attention maps reveal chemically meaningful interaction sites  
✅ **Efficient surrogate**: Replaces expensive DFT calculations for screening studies

---

## Background

Understanding solvation thermodynamics in confined nanoporous environments is critical for modeling liquid-phase catalysis in zeolites. Traditional approaches require extensive DFT calculations, making large-scale screening computationally prohibitive.

This work builds on the **Multiscale Sampling (MSS) framework** ([Chen et al., *J. Phys. Chem. C* 2024, 128, 19367](https://doi.org/10.1021/acs.jpcc.4c05344)), which combined MD and DFT to evaluate solvation free energies of C₁–C₃ oxygenates in Ti-FAU zeolite pores. We extend this by introducing a **data-driven 3D-CNN surrogate model** that predicts interaction energies directly from MD snapshots.

---

## Methodology

### Dataset
- **80 adsorbate–environment combinations**  
  C₁–C₃ oxygenates in hydrophilic/hydrophobic Ti-FAU pores with water and water–methanol solvents
- **800 MD snapshots** (10 per system)  
  Each labeled with DFT-calculated ΔE<sub>int</sub>
- **19,200 training samples**  
  24 rotational augmentations per snapshot

### 3D Voxelization
MD snapshots → **20×20×20 voxel grids** (0.8 Å resolution)
- **28 feature channels**: 14 for adsorbate, 14 for solvent  
  (element types, charges, masses, LJ parameters, H-bond indicators)
- Centered on adsorbate center of mass
- Zeolite framework excluded (implicit environment)

### Neural Network
**Dual-branch 3D-CNN with CBAM attention**
- Separate adsorbate/solvent processing pathways
- Channel & spatial attention for interpretability
- Residual blocks with progressive downsampling (20³ → 10³ → 5³)
- Trained with MSE loss, Adam optimizer (lr=2×10⁻⁴), 200 epochs

---

## Model Performance

| Evaluation Scenario | MAE (eV) | RMSE (eV) | R² |
|---------------------|----------|-----------|-----|
| **Unseen adsorbates** (snapshot-level) | 0.083 | 0.110 | 0.82 |
| **Unseen adsorbates** (ensemble avg.) | 0.037 | 0.047 | 0.95 |
| **New solvent mixtures** (ensemble avg.) | 0.036 | 0.048 | 0.94 |
| **Different pore types** (ensemble avg.) | 0.059 | 0.071 | 0.89 |

*Ensemble predictions average over 10 snapshots per adsorbate*

### Attention Visualization
<p align="center">
  <img src="results/attention_heatmaps/example_attention.png" width="600" alt="Attention maps highlight adsorbate–solvent interfacial regions">
</p>

Attention heatmaps demonstrate that the model learns physically meaningful features, focusing on hydrogen bonding sites and polar interaction zones.

---

## Repository Structure

```
3dcnn-zeolite-solvation/
├── python_scripts/         # Data processing and model training scripts
├── database/               # Raw MD trajectories and DFT energies
├── output_model_cnn/       # Trained models and checkpoints
├── output_figures/         # Visualization outputs
├── md_simulations/         # MD simulation input files
└── figure_1.png           # Graphical abstract
```

---

## Installation

```bash
git clone https://github.com/getman-research-group/3dcnn-zeolite-solvation.git
cd 3dcnn-zeolite-solvation
pip install -r requirements.txt
```

**Core dependencies:** PyTorch, MDAnalysis, NumPy, SciPy, scikit-learn, Matplotlib

---

## Citation

If you use this work, please cite:

```bibtex
@article{zeolite3dcnn2025,
  title={Predicting Adsorbate–Solvent Interaction Energies in Zeolite Nanopores 
         Using 3D Convolutional Neural Networks},
  author={[Authors]},
  journal={[Journal]},
  year={2025},
  note={In preparation}
}
```

**Related work:**
```bibtex
@article{chen2024multiscale,
  title={Multiscale Sampling of Solvation Free Energies in Confined Environments},
  author={Chen, Jiaqi and Getman, Rachel B.},
  journal={J. Phys. Chem. C},
  volume={128},
  pages={19367--19379},
  year={2024},
  doi={10.1021/acs.jpcc.4c05344}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Getman Research Group**  
The Ohio State University

For questions, please open an issue or contact: [getman.11@osu.edu](mailto:getman.11@osu.edu)

---

<p align="center">
  <i>© 2025 Getman Research Group – The Ohio State University</i>
</p>