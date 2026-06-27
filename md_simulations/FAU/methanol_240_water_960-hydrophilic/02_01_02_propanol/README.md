# Representative MD and DFT System

This directory contains the simulation files for one representative system used
in the study: 2-propanol in hydrophilic Ti-FAU with a solvent composition of
240 methanol and 960 water molecules.

The files document the workflow from an MD sampling trajectory to ten selected
snapshots and their associated DFT interaction-energy calculations. Some of the
historical workflow scripts retain absolute paths from the original computing
cluster and are provided for methodological transparency rather than direct
execution.

## MD Files

- `data_nvt_samp.lammpsdata`: Original LAMMPS data file containing atom types,
  molecule IDs, charges, force-field topology, coordinates, and velocities.
- `data_nvt_samp_new.lammpsdata`: Topology file exported with annotated atom and
  residue names. This is the topology read by the current snapshot-processing
  and voxel-generation scripts.
- `dump_nvt_samp.lammpsdata.lammpstrj`: MD sampling trajectory containing the
  initial frame and ten sampled configurations.
- `fau_ad_co.lammpsdata`: Intermediate annotated LAMMPS structure used when
  preparing `data_nvt_samp_new.lammpsdata`.

## DFT Reference Structures

- `FAU_Ti_O1_3SiOH_3c_1oh_converged.xyz`: DFT-optimized zeolite-adsorbate
  structure before extraction of the representative unit cell.
- `converged_one_unit.xyz`: Extracted unit-cell structure used to construct the
  DFT interaction-energy systems.
- `00_intE_zeo/`: CP2K calculation for the bare zeolite reference energy.

## Sampled Snapshots

The directories `intE01/` through `intE10/` contain the ten sampled MD
configurations. Each directory includes an `intE##.traj` snapshot and three DFT
calculation components:

- `01_intE##_aq/`: Geometry optimization of the zeolite, adsorbate, and nearby
  solvent molecules.
- `02_intE##_aq_zeo/`: Single-point calculation of the zeolite and the same
  nearby solvent molecules after removal of the adsorbate.
- `03_intE##_g_sp/`: Single-point calculation of the zeolite-adsorbate system
  without the nearby solvent molecules.

The CP2K input (`.inp`) and coordinate (`.xyz`) files specify the calculations.
The `00-OUTPUT/` directories contain archived CP2K and job-scheduler outputs.

## Historical Workflow Scripts

- `01_bash_split_def_cocolvent.sh`: Splits the MD trajectory, prepares the ten
  snapshot directories, constructs solvated DFT systems, and submits the first
  two groups of CP2K calculations.
- `02_bash_g_sp_cosolvent.sh`: Prepares and submits the zeolite-adsorbate
  single-point calculations.
- `03_bash_traj_intE_calc_2.sh` and `03_bash_traj_intE_calc_def.sh`: Collect the
  calculated energies using two historical zeolite-reference locations.
- `traj_2_intE_cosol_3.0.py`: Selects water and methanol molecules within 7
  angstrom of the adsorbate and writes the solvated CP2K coordinate files.
- `zeolite_ad_g.py`: Extracts the representative FAU unit cell from the
  optimized structure.
- `calc_intE_2.0_9.26.py`: Calculates snapshot interaction energies and their
  average from the collected CP2K energies.

`traj_intE.txt` records the collected energies and final interaction-energy
summary. `process.txt` preserves historical source-directory information from
the original cluster workflow.
