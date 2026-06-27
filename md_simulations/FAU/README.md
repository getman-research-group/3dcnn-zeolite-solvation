# Representative FAU Simulation Systems

This directory contains two representative Ti-FAU systems from the molecular
simulation and DFT workflow used in this study. They illustrate the organization
of the LAMMPS and CP2K files for both mixed-solvent and pure-water environments.
The complete dataset is described in the manuscript and the energy tables in
`database/`.

## Mixed-Solvent Example

[`methanol_240_water_960-hydrophilic/02_01_02_propanol/`](methanol_240_water_960-hydrophilic/02_01_02_propanol/)

This example contains 2-propanol in hydrophilic Ti-FAU surrounded by a mixed
solvent of 240 methanol and 960 water molecules.

## Pure-Water Example

[`water_pure-hydrophilic/11_01_propylene_glycol/`](water_pure-hydrophilic/11_01_propylene_glycol/)

This example contains propylene glycol in hydrophilic Ti-FAU surrounded by pure
water.

## Included Files

Each representative system includes the LAMMPS topology and sampling trajectory,
ten selected MD snapshots, the corresponding CP2K inputs and outputs, converged
DFT structures, and the bare-zeolite reference calculation. These files document
how sampled molecular configurations were connected to the DFT interaction-energy
labels used by the machine-learning workflow.
