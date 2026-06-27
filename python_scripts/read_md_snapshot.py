# -*- coding: utf-8 -*-
"""
read_md_snapshot.py

This script loads one molecular dynamics snapshot from the LAMMPS simulation
files and stores it as an MDAnalysis Universe for downstream voxel generation.

For each selected system, two files are read:
1. data_nvt_samp_new.lammpsdata: topology, atom types, molecule IDs, charges,
   bonds, and simulation-cell information shared by all snapshots.
2. intE<index>/intE<index>.traj: atomic coordinates for one sampled snapshot,
   for example intE01/intE01.traj.

The expected simulation directory is:
md_simulations/<zeolite>/<solvent>-<pore_type>/<adsorbate>/

The snapshotMDAnalysis class also assigns residue names for water, methanol,
zeolite, and adsorbate atoms so that downstream scripts can select molecular
components consistently. This script reads existing simulation files and does
not modify them.

MDAnalysis reference: https://docs.mdanalysis.org/
"""

## Importing Modules
import os
import time
import numpy as np
import copy
import pickle
from datetime import datetime

## MDAnalysis Modules
import MDAnalysis as mda

## Custom Functions
# from core.global_vars import ADSORBATE_TO_NAME_DICT
from core.path import get_paths


class snapshotMDAnalysis:
    
    '''
    Load one LAMMPS topology file and one snapshot trajectory file, then create
    an annotated MDAnalysis Universe for the selected molecular system.
    '''

    ### Initializing
    def __init__(   self,
                    zeolite_type = 'FAU',           # e.g. "FAU", "BEA" or "MFI"
                    solvent_type = 'water_pure',    # e.g. "water_pure"
                    pore_type = 'hydrophilic',      # e.g. "hydrophilic", "hydrophobic"
                    adsorbate = '01_methanol',      # e.g. "01_methanol", "02_01_02_propanol"
                    snapshot_index = 1,
                    verbose = False,
                    ):
        
        ## If verbose is True, print the information
        self.verbose = verbose
        
        ## Simulation Details
        self.zeolite_type = zeolite_type
        self.solvent_type = solvent_type
        self.pore_type = pore_type
        self.adsorbate = adsorbate
        self.snapshot_index = snapshot_index
        
        ## Parse solvent composition
        self._parse_solvent_composition()
        
        ## Get the root directory containing all MD simulation files
        self.md_simulations = get_paths('simulation_path')
        
        ## Construct the system directory, e.g., .../FAU/water_pure-hydrophilic/01_methanol
        folder_name = f"{self.solvent_type}-{self.pore_type}"
        self.sim_dir = os.path.join(self.md_simulations, self.zeolite_type, folder_name, self.adsorbate)

        ## Define the shared LAMMPS topology file
        self.path_lammpsdata = os.path.join(self.sim_dir,'data_nvt_samp_new.lammpsdata')
        
        ## Read atom names and molecule IDs from the LAMMPS topology file
        self.read_lammpsdata()
        
        # Load the selected snapshot coordinates as an MDAnalysis Universe
        self.load_snapshot(self.snapshot_index)
        
        ## Add atom and residue names used by downstream selections
        self.add_attributes()
        
        ## Print General Trajectory Information
        if self.verbose:
            self.print_config_general_info()
        

    def _parse_solvent_composition(self):
        """
        Parse solvent type and determine molecule ID ranges for different components.
        """
        if self.verbose:
            print(f"\n--- Loading MDAnalysis Universe.")
            print(f"    Zeolite: {self.zeolite_type}")
            print(f"    Solvent: {self.solvent_type}")
            print(f"    Pore: {self.pore_type}")
            print(f"    Adsorbate: {self.adsorbate}")
            print(f"    Snapshot index: {self.snapshot_index}")
        
        if self.solvent_type == 'water_pure':
            # Pure water system: 1-1200 mol_ids are all water
            self.water_mol_range = (1, 1200)
            self.methanol_mol_range = None
            self.has_methanol = False
            if self.verbose:
                print(f"    Pure water system: Water mol_ids 1-1200")
                
        elif self.solvent_type == 'methanol_120_water_1080':
            # Mixed system: 1080 water + 120 methanol
            self.water_mol_range = (1, 1080)
            self.methanol_mol_range = (1081, 1200)
            self.has_methanol = True
            if self.verbose:
                print(f"    Mixed solvent system: Water mol_ids 1-1080, Methanol mol_ids 1081-1200")
                
        elif self.solvent_type == 'methanol_240_water_960':
            # Mixed system: 960 water + 240 methanol
            self.water_mol_range = (1, 960)
            self.methanol_mol_range = (961, 1200)
            self.has_methanol = True
            if self.verbose:
                print(f"    Mixed solvent system: Water mol_ids 1-960, Methanol mol_ids 961-1200")
                
        elif self.solvent_type == 'methanol_600_water_600':
            # Mixed system: 600 water + 600 methanol
            self.water_mol_range = (1, 600)
            self.methanol_mol_range = (601, 1200)
            self.has_methanol = True
            if self.verbose:
                print(f"    Mixed solvent system: Water mol_ids 1-600, Methanol mol_ids 601-1200")
                
        else:
            raise ValueError(f"Unsupported solvent_type: {self.solvent_type}. "
                           "Supported types: 'water_pure', 'methanol_120_water_1080', "
                           "'methanol_240_water_960', 'methanol_600_water_600'")
        
        # Fixed ranges for other components
        self.zeolite_mol_id = 1201
        self.adsorbate_mol_id = 1202
        

    def read_lammpsdata(self):
        """
        Read the Atoms section of the LAMMPS data file and store mappings from
        atom IDs to atom names and molecule IDs.
        """
        with open(self.path_lammpsdata, 'r', encoding = "utf-8") as lammpsdata:
            lines = lammpsdata.readlines()
            
            ## Find The First Line Of Atoms Information
            for i, line in enumerate(lines):
                if line.startswith("Atoms"):
                    atoms_start = i + 2
                    break
            
            ## Find The Last Line Of Atoms Information
            for i, line in enumerate(lines[atoms_start:]):
                if line.strip() == "" or line.strip().startswith("Bonds"):
                    atoms_end = i + atoms_start - 1
                    break
            
            lines_atoms = lines[atoms_start: atoms_end + 1]
            lines_atoms = [line.split() for line in lines_atoms]
            self.lines_atoms = lines_atoms
            self.atom_types = []
            self.atom_num_to_name = {}
            self.atom_num_to_molid = {}
            
            for info in lines_atoms:
                if len(info) >= 7:  # Make sure we have enough columns
                    atom_id = int(info[0])
                    mol_id = int(info[1])  # molecule ID is typically the second column
                    # Skip partial_charge at index 3, coordinates start at index 4
                    atom_name = info[-1]  # atom name is in the comment at the end
                    self.atom_num_to_name[atom_id] = atom_name
                    self.atom_num_to_molid[atom_id] = mol_id
                    if atom_name not in self.atom_types:
                        self.atom_types.append(atom_name)
        
    
    def load_snapshot(self, snapshot_index: int):
        '''
        Load both the LAMMPS data (topology) and a single-frame LAMMPS dump,
        then return an MDAnalysis Universe containing both topology and coordinates.
        '''
        # Construct snapshot subfolder and trajectory filename, e.g., intE01/intE01.traj
        subfolder = f"intE{snapshot_index:02d}"
        traj_name = f"intE{snapshot_index:02d}.traj"
        self.path_lammpstrj = os.path.join(self.sim_dir, subfolder, traj_name)
        
        if self.verbose:
            print(f"\n--- Loading lammps data: {self.path_lammpsdata}")
            print(f"--- Loading lammps dump: {self.path_lammpstrj}")
        
        # Create MDAnalysis Universe
        self.universe = mda.Universe(
                                     self.path_lammpsdata,
                                     self.path_lammpstrj,
                                     topology_format='DATA',
                                     format='LAMMPSDUMP',
                                     dt = 1.0,)
        
        return self.universe
    
    
    def add_attributes(self):
        """Assign atom names and component-specific residue names to the Universe."""
        
        ## Add attributes to topology information
        self.universe.add_TopologyAttr('resnames')
        self.universe.add_TopologyAttr('names')
        if self.verbose:
            print("\n--- Adding Topology Attributes ---")
            print("    Adding Residue Names")
            print("    Adding Atom Names")
        
        ## Create residue names based on molecule IDs from LAMMPS data
        temp_resnames = []
        for resid in self.universe.residues.resids:
            # Get first atom of this residue to determine molecule type
            first_atom = self.universe.residues[resid-1].atoms[0]
            mol_id = self.atom_num_to_molid.get(first_atom.id, 0)
            
            if self.water_mol_range[0] <= mol_id <= self.water_mol_range[1]:
                temp_resnames.append('HOH')
            elif self.has_methanol and self.methanol_mol_range[0] <= mol_id <= self.methanol_mol_range[1]:
                temp_resnames.append('MEO')  # Methanol residue name
            elif mol_id == self.zeolite_mol_id:
                temp_resnames.append(self.zeolite_type)  # Use actual zeolite type (FAU, BEA, MFI, etc.)
            elif mol_id == self.adsorbate_mol_id:
                temp_resnames.append('ADS')
            else:
                temp_resnames.append('UNK')
        
        self.universe.residues.resnames = np.array(temp_resnames)
        
        ## Add atoms name information to the object
        self.universe.atoms.names = np.array(list(map(self.atom_num_to_name.get, self.universe.atoms.ids)))
    
    
    ### Function to Print General Config Information
    def print_config_general_info(self):
        '''
        The function takes the trajectory and prints the information of the configuration
        INPUTS:
            self: class object
        OUTPUTS:
            Printed output
        '''
        self.num_water_atoms = len([num for num in (self.universe.atoms.resids) if self.water_mol_range[0] <= num <= self.water_mol_range[1]])
        self.num_zeolite_atoms = len([num for num in (self.universe.atoms.resids) if num == self.zeolite_mol_id])
        self.num_ads_atoms = len([num for num in (self.universe.atoms.resids) if num == self.adsorbate_mol_id])
        
        # Count methanol atoms if present
        if self.has_methanol:
            self.num_methanol_atoms = len([num for num in (self.universe.atoms.resids) if self.methanol_mol_range[0] <= num <= self.methanol_mol_range[1]])
        else:
            self.num_methanol_atoms = 0
        
        # Calculate molecular counts
        self.num_water_molecules = len([resid for resid in self.universe.residues.resids if self.water_mol_range[0] <= resid <= self.water_mol_range[1]])
        self.num_zeolite_molecules = len([resid for resid in self.universe.residues.resids if resid == self.zeolite_mol_id])
        self.num_ads_molecules = len([resid for resid in self.universe.residues.resids if resid == self.adsorbate_mol_id])
        
        # Count methanol molecules if present
        if self.has_methanol:
            self.num_methanol_molecules = len([resid for resid in self.universe.residues.resids if self.methanol_mol_range[0] <= resid <= self.methanol_mol_range[1]])
        else:
            self.num_methanol_molecules = 0
        
        # Calculate system volume and density
        box_volume = np.prod(self.universe.dimensions[:3])  # Volume in Å³
        
        # Count different atom types
        atom_type_counts = {}
        for atom_type in self.atom_types:
            atom_type_counts[atom_type] = len([name for name in self.universe.atoms.names if name == atom_type])
        
        # Count residues by type
        residue_counts = {}
        for resname in set(self.universe.residues.resnames):
            residue_counts[resname] = len([r for r in self.universe.residues.resnames if r == resname])
        
        # Count bonds by residue type and atom type
        bond_counts_residue = {}
        bond_counts_atom = {}
        total_bonds = len(self.universe.bonds)
        if total_bonds > 0:
            for bond in self.universe.bonds:
                atom1 = bond.atoms[0]
                atom2 = bond.atoms[1]
                atom1_resname = atom1.resname
                atom2_resname = atom2.resname
                atom1_name = atom1.name
                atom2_name = atom2.name
                
                # Classify bond type by residue
                if atom1_resname == atom2_resname:
                    bond_type_res = f"{atom1_resname}-{atom1_resname}"
                else:
                    bond_type_res = f"{min(atom1_resname, atom2_resname)}-{max(atom1_resname, atom2_resname)}"
                bond_counts_residue[bond_type_res] = bond_counts_residue.get(bond_type_res, 0) + 1
                
                # Classify bond type by atom names
                bond_type_atom = f"{min(atom1_name, atom2_name)}-{max(atom1_name, atom2_name)}"
                bond_counts_atom[bond_type_atom] = bond_counts_atom.get(bond_type_atom, 0) + 1

        ## Print system parameters
        print ("\n--- System Parameters ---")
        print ("    Zeolite Type: %s" % self.zeolite_type)
        print ("    Solvent Type: %s" % self.solvent_type)
        print ("    Pore Type: %s" % self.pore_type)
        print ("    Adsorbate: %s" % self.adsorbate)
        print ("    Snapshot Index: %d" % self.snapshot_index)
                
        print ("\n--- General Information about MDAnalysis Configuration ---")
        print ("    Total frame is:         ", len(self.universe.trajectory))
        
        ## Print total number of atoms and residues
        print ("    Total atom number is:   ", len(self.universe.atoms.ix))
        print ("    Total residue number is:", len(self.universe.residues.ix))
        print ("    Total bond number is:   ", len(self.universe.bonds))
        
        ## Print Unit Cell info
        print ("\n--- Unit Cell Information ---")
        print ("    Unit Cell Dimension:    %.3f × %.3f × %.3f Å" % tuple(self.universe.dimensions[0:3]))
        print ("    Unit Cell Angles:       %.1f° × %.1f° × %.1f°" % tuple(self.universe.dimensions[3:6]))
        print ("    Unit Cell Volume:       %.2f Å³" % box_volume)
        print ("")
        
        ## Print molecular information
        print ("--- Molecular Information ---")
        print ("    H2O molecules:          %d (atoms: %d)" % (self.num_water_molecules, self.num_water_atoms))
        if self.has_methanol:
            print ("    Methanol molecules:     %d (atoms: %d)" % (self.num_methanol_molecules, self.num_methanol_atoms))
        print ("    %s zeolite molecules:  %d (atoms: %d)" % (self.zeolite_type, self.num_zeolite_molecules, self.num_zeolite_atoms))
        print ("    Adsorbate molecules:    %d (atoms: %d)" % (self.num_ads_molecules, self.num_ads_atoms))
        
        ## Print residue counts
        print ("\n--- Residue Distribution ---")
        for resname, count in sorted(residue_counts.items()):
            print ("    %s residues: %d" % (resname, count))
        
        ## Print bond information
        print ("\n--- Bond Information ---")
        print ("    Total bonds: %d" % total_bonds)
        if bond_counts_residue:
            print ("    Bonds by residue type:")
            for bond_type, count in sorted(bond_counts_residue.items()):
                # Calculate average bonds per residue
                if '-' in bond_type and bond_type.split('-')[0] == bond_type.split('-')[1]:
                    # Same residue type (e.g., HOH-HOH, MEO-MEO)
                    res_type = bond_type.split('-')[0]
                    if res_type in residue_counts:
                        avg_bonds = count / residue_counts[res_type]
                        print ("    - %s bonds: %d (%d bonds per %s)" % (bond_type, count, avg_bonds, res_type))
                    else:
                        print ("    - %s bonds: %d" % (bond_type, count))
                else:
                    # Different residue types (e.g., HOH-MEO)
                    print ("    - %s bonds: %d" % (bond_type, count))
        
        if not bond_counts_residue and not bond_counts_atom:
            print ("    No bond information available")
        
        ## Print atom type distribution
        print ("\n--- Atom Type Distribution ---")
        for atom_type, count in sorted(atom_type_counts.items()):
            print ("    %s atoms: %d" % (atom_type, count))

        

if __name__ == "__main__":
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'           # e.g. "FAU", "BEA" or "MFI"
    # "water_pure" "methanol_120_water_1080" "methanol_240_water_960" "methanol_600_water_600"
    solvent_type = 'methanol_240_water_960'
    pore_type = 'hydrophilic'      # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '02_01_02_propanol'      # e.g. "01_methanol", "02_01_02_propanol"
    snapshot_index = 1
                    
    snapshot_mda = snapshotMDAnalysis(
                                      zeolite_type = zeolite_type,
                                      solvent_type = solvent_type,
                                      pore_type = pore_type,
                                      adsorbate = adsorbate,
                                      snapshot_index = snapshot_index,
                                      verbose = True,
                                      )

