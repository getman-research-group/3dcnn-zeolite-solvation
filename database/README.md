# Energy Label Database

This directory contains the interaction-energy tables used to generate and
evaluate the machine-learning datasets in this repository. Each row represents
one molecular dynamics (MD) snapshot and its corresponding energy label.

The database contains 800 snapshots from 80 FAU zeolite systems. Each system is
represented by ten sampled configurations. The data cover hydrophilic and
hydrophobic pores in pure water and three methanol-water compositions.

## Files

The CSV filenames follow the convention:

```text
<zeolite>-<solvent_composition>-<pore_type>.csv
```

The eight files comprise:

- Pure water: hydrophilic and hydrophobic FAU systems, with 22 adsorbates in
  each pore environment.
- 120 methanol and 1080 water molecules: hydrophilic and hydrophobic FAU
  systems, with 6 adsorbates in each pore environment.
- 240 methanol and 960 water molecules: hydrophilic and hydrophobic FAU
  systems, with 6 adsorbates in each pore environment.
- 600 methanol and 600 water molecules: hydrophilic and hydrophobic FAU
  systems, with 6 adsorbates in each pore environment.

## Columns

| Column | Description |
| --- | --- |
| `zeolite` | Zeolite framework used in the simulation. |
| `environment` | Solvent composition and pore type. |
| `adsorbate` | Identifier of the adsorbate molecule. |
| `snapshot` | Sampled MD configuration index from 1 to 10. |
| `intE` | Snapshot-level DFT interaction energy in eV; this is the target used for model training. |
| `intE_avg` | Mean DFT interaction energy over the ten snapshots of the corresponding system, in eV. |
| `intE_MD` | Interaction energy obtained from the molecular simulation for the corresponding system, in eV. |

The processing scripts select columns by name. In particular,
`python_scripts/generate_voxel_grids.py` associates each voxelized snapshot with
its `intE` target, and `python_scripts/load_grids_pickle.py` loads these labels
for model training and evaluation.

