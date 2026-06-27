# Representative MD and DFT System

This directory contains the molecular simulation and electronic-structure files
for one representative system used in this study: 2-propanol in hydrophilic
Ti-FAU containing 240 methanol and 960 water molecules.

The files document the connection between the molecular dynamics (MD) sampling
trajectory, ten sampled configurations, and the corresponding density functional
theory (DFT) calculations used to determine adsorbate-solvent interaction
energies.

## Directory Structure

```text
02_01_02_propanol/
|-- data_nvt_samp_new.lammpsdata
|-- dump_nvt_samp.lammpsdata.lammpstrj
|-- FAU_Ti_O1_3SiOH_3C_1OH_converged.xyz
|-- converged_one_unit.xyz
|-- 00_intE_zeo/
|-- intE01/
|   |-- intE01.traj
|   |-- 01_intE01_aq/
|   |-- 02_intE01_aq_zeo/
|   `-- 03_intE01_g_sp/
|-- ...
`-- intE10/
```

## MD Files

- `data_nvt_samp_new.lammpsdata` contains the LAMMPS topology, atom types,
  molecule IDs, charges, force-field parameters, and bonding information. It is
  used as the topology file by the snapshot-processing and voxel-generation
  scripts in this repository.
- `dump_nvt_samp.lammpsdata.lammpstrj` is the LAMMPS sampling trajectory. It
  contains an initial frame followed by the ten configurations retained for DFT
  calculations.
- `intE01/intE01.traj` through `intE10/intE10.traj` are the individual sampled
  configurations extracted from the full trajectory. Their coordinates are
  combined with the shared topology in `data_nvt_samp_new.lammpsdata` when the
  MD snapshots are read.

## DFT Reference Structures

- `FAU_Ti_O1_3SiOH_3C_1OH_converged.xyz` is the DFT-optimized
  zeolite-adsorbate structure prior to extraction of the representative unit
  cell.
- `converged_one_unit.xyz` is the extracted zeolite-adsorbate unit-cell
  structure used to construct the DFT systems.
- `00_intE_zeo/` contains the CP2K calculation of the bare zeolite reference
  energy.

## Snapshot DFT Calculations

The directories `intE01/` through `intE10/` correspond to the ten sampled MD
configurations. Each snapshot directory contains three calculation components:

- `01_intE##_aq/`: geometry optimization of the zeolite, adsorbate, and nearby
  solvent molecules.
- `02_intE##_aq_zeo/`: single-point calculation of the zeolite and the same
  nearby solvent environment after removal of the adsorbate.
- `03_intE##_g_sp/`: single-point calculation of the zeolite-adsorbate system
  without the nearby solvent molecules.

Here, `##` denotes the snapshot number from `01` to `10`. Together with the bare
zeolite reference calculation, these components provide the electronic energies
required to evaluate the adsorbate-solvent interaction energy for each snapshot.

## Calculation Files

Within each CP2K calculation directory:

- `*.inp` is the CP2K input file.
- `*.xyz` contains the atomic coordinates used in the calculation.
- `BASIS_file`, `POTENTIALS_file`, and `dftd3.dat` provide the basis sets,
  pseudopotentials, and dispersion parameters required by the CP2K input.
- `sub_*.sh` is the Slurm submission script used on the original computing
  cluster.
- `slurm-*.out` and `rfm_cp2k_test_mpi_job.err` are job-scheduler log files.
- `00-OUTPUT/` contains archived calculation inputs and outputs, including the
  CP2K output (`*.out`) and, where applicable, the converged geometry.

The Slurm scripts retain settings specific to the original computing
environment and may require adaptation before use on another cluster.
