# -*- coding: utf-8 -*-
"""

"""

## Importing Modules
import numpy as np
import sys
import os
import mdtraj as md     # https://mdtraj.org/
import pickle           # Used to store variables
import matplotlib.pyplot as plt
from datetime import datetime

## Custom Functions
from core.path import get_paths

# -*- coding: utf-8 -*-
"""
import_tools.py
This contains codes on importing functions.

Functions:
    read_file_as_line: reads a file as a line

Classes:
    import_traj: class that can import trajectory information (uses md.load from mdtraj module)
        
"""

import os
import time

# Mdtraj to Read Trajectories
import mdtraj as md

# Function To Get The Path On The Server
from core.global_vars import ADSORBATE_TO_NAME_DICT

class snapshot_MDTraj:
    """
    Load a single MD snapshot from LAMMPS data and dump files using MDTraj.
    """

    def __init__(self,
                 zeolite_type: str = 'FAU',
                 solvent_type: str = 'water_pure',
                 pore_type: str = 'hydrophilic',
                 adsorbate: str = '01_methanol',
                 snapshot_index: int = 1,
                 verbose: bool = False):
        
        # Store basic settings
        self.verbose = verbose
        self.zeolite_type = zeolite_type
        self.solvent_type = solvent_type
        self.pore_type = pore_type
        self.adsorbate = adsorbate
        self.snapshot_index = snapshot_index

        # Build simulation directory path
        root = get_paths('simulation_path')
        folder = f"{self.solvent_type}-{self.pore_type}"
        
        self.sim_dir = os.path.join(root, self.zeolite_type, folder, self.adsorbate)
        
        # Define file paths
        self.topology_file = os.path.join(self.sim_dir, 'data_nvt_samp.lammpsdata')
        if self.verbose:
            print(f"Topology file: {self.topology_file}")

        # Load the specified snapshot
        self.traj = self.load_snapshot(self.snapshot_index)

    def load_snapshot(self, index: int):
        """
        Load a single-frame dump and combine with topology.
        Returns an MDTraj Trajectory object for that snapshot.
        """
        # Construct dump path
        sub = f"intE{index:02d}"
        dump_file = f"intE{index:02d}.traj"
        self.dump_file = os.path.join(self.sim_dir, sub, dump_file)
        if self.verbose:
            print(f"Snapshot file: {self.dump_file}")

        # Load dump as trajectory with topology from data file
        traj = md.load_lammpstrj(self.dump_file,
                                 top=self.topology_file,
                                 frame=0)
        if self.verbose:
            num_atoms = traj.n_atoms
            print(f"Loaded trajectory with {num_atoms} atoms and 1 frame")

        # Optionally, extract arrays or assign attribute arrays
        # e.g., atom types: traj.topology.atom_slice(...) or traj.xyz

        return traj

    
    
    
    ### Function to Print General Trajectory Information
    def print_traj_general_info(self):
        '''
        The function takes the trajectory and prints the residue names, corresponding number, and time length of the trajectory
        INPUTS:
            self: class object
        OUTPUTS:
            Printed output
        '''

        def findUniqueResNames(traj):
            ''' This function finds all the residues in the trajectory and outputs its unique residue name
            INPUTS:
                traj: trajectory from md.traj
            OUTPUTS:
                List of unique residues
            '''
            return list(set([ residue.name for residue in traj.topology.residues ]))

        def findTotalResidues(traj, resname):
            ''' This function takes the residue name and finds the residue indexes and the total number of residues
            INPUTS:
                traj: trajectory from md.traj
                resname: Name of the residue
            OUTPUTS:
                num_residues, index_residues
            '''
            # Finding residue index
            index_residues = [ residue.index for residue in traj.topology.residues if residue.name == resname ]
            
            # Finding total number of residues
            num_residues = len(index_residues)
            
            return num_residues, index_residues
        
        if self.verbose:
            print("\n--- General Information about MDtraj Trajectory ---")
            print("    %s\n"%(self.traj))
            print("    Slicing Trajectory into 5 frames we want")
            print("    %s\n"%(self.traj_5_snapshots))
            print("    Unit Cell Angles:  %i, %i, %i" % (self.traj.unitcell_angles[0][0], self.traj.unitcell_angles[0][1], self.traj.unitcell_angles[0][2]))
            print("    Unit Cell Lengths: %.3f nm, %.3f nm, %.3f nm" % (self.traj.unitcell_lengths[0][0], self.traj.unitcell_lengths[0][1], self.traj.unitcell_lengths[0][2]))
            print("")
        
        
        
        ## Storing Total Residues
        self.residue_num = {}
        
        # Finding unique residues
        unique_res_names = findUniqueResNames(self.traj)

        for currentResidueName in unique_res_names:
            # Finding total number of residues, and their indexes
            num_residues, index_residues = findTotalResidues(self.traj, resname = currentResidueName)
            
            ## Storing
            self.residue_num[currentResidueName] = num_residues
            
            # Printing an output
            if self.verbose:
                print ("    Total number of residue %s is: %s" % (currentResidueName, num_residues))
        
        ## Number of Atoms that Belongs to Pt Slab Surface
        num_pt = [i for i in self.traj.topology._residues if str(i).startswith('PTS')]
        num_pt_atom = sum(residue.n_atoms for residue in num_pt)
        
        ## Number of Atoms that Belongs to Adsorbate
        num_ad = [i for i in self.traj.topology._residues if str(i).startswith('A')]
        num_ad_atom = sum(residue.n_atoms for residue in num_ad)
        
        ## Number of Atoms that Belongs to Water
        num_water = [i for i in self.traj.topology._residues if str(i).startswith('HOH')]
        num_water_atom = sum(residue.n_atoms for residue in num_water)
        
        if self.verbose:
            print('\n    Total Atom number of Pt Slab is:   %d' % num_pt_atom)
            print('    Total Atom number of H20 is:       %d' % num_ad_atom)
            print('    Total Atom number of Adsorbate is: %d' % num_water_atom)
            
            # Finding total time length of simulation
            # print("\nTime length of trajectory: %s ps"%(self.traj.time[-1] - self.traj.time[0]))
            # The traj time given by MDTraj is wrong, is one-tenth of the correct length
            print("\n    Time length of trajectory: %s ns"%(((self.traj.time[-1] - self.traj.time[0]) / 100)))
        
        return



if __name__ == "__main__":
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'           # e.g. "FAU", "BEA" or "MFI"
    solvent_type = 'water_pure'    # e.g. "water_pure"
    pore_type = 'hydrophilic'      # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '01_methanol'      # e.g. "01_methanol", "02_01_02_propanol"
    snapshot_index = 1
    
    
    ### Loading Trajectory
    snapshot_mdtraj = snapshot_MDTraj(
                                      zeolite_type = zeolite_type,
                                      solvent_type = solvent_type,
                                      pore_type = pore_type,
                                      adsorbate = adsorbate,
                                      snapshot_index = snapshot_index,
                                      verbose = True,
                                      )