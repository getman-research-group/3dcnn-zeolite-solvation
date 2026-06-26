# -*- coding: utf-8 -*-
"""
generate_voxel_grids_type_2.py
    The purpose of this code is to generate a grid interpolation from a MD trajectory.
    So we can represent our system in a way that can be represented for a machine learning approach.

Functions:
    normalize_3d_rdf: normalizes 3D RDFs

"""

## Importing Modules

import sys
import os
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import maxabs_scale, StandardScaler
from typing import Optional


## Custom Functions
from core.path import get_paths
from core.global_vars import FEATURE_LIST, LABEL_CSV_FILES
from read_md_snapshot import snapshotMDAnalysis
from extract_lammpsdata_info import extract_LJ_parameter_info, extract_total_valence_info
from extract_lammpsdata_info import extract_is_hydrophobic_info, extract_is_donor_acceptor
from extract_hbonds import HydrogenBondDetector


# Class Function To Generate Grid Interpolation Array Per Frame
class GenerateVoxelGrids:    ### Initializing
    def __init__(
                self,
                zeolite_type = 'FAU',           # e.g. "FAU"
                solvent_type = 'water_pure',    # e.g. "water_pure"
                pore_type = 'hydrophilic',      # e.g. "hydrophilic", "hydrophobic"
                adsorbate = '01_methanol',      # e.g. "01_methanol", "02_01_02_propanol"
                snapshot_index = 1,             # Single snapshot index
                box_grids_size = 20.0,          # Box length in all three dimensions (Angstrom)
                box_increment = 1.0,
                feature_list = ['mol_type'],
                include_zeolite = False,        # Whether to include zeolite atoms
                include_solvent = True,         # Whether to include solvent (water) atoms
                include_adsorbate = True,       # Whether to include adsorbate atoms
                verbose = False,
                ):
        
        ## Storing Initial Information
        self.zeolite_type = zeolite_type
        self.solvent_type = solvent_type
        self.pore_type = pore_type
        self.adsorbate = adsorbate
        self.snapshot_index = snapshot_index
        self.box_grids_size = box_grids_size    # Box length in all three dimensions (Angstrom)
        self.box_increment = box_increment      # Voxel box size in Angstrom
        self.feature_list = feature_list
        self.include_zeolite = include_zeolite
        self.include_solvent = include_solvent
        self.include_adsorbate = include_adsorbate
        self.verbose = verbose

        # Load labels from CSV files (similar to build_graph.py)
        self.load_labels()

        # Use target interaction energy from labels
        self.target_interaction_energy = self.get_target_interaction_energy()

        # Define actual atomic features (excluding mol_type which was just a molecular identity marker)
        self.atomic_features = ['atom_type_C',
                               'atom_type_H',
                               'atom_type_O',
                               'is_hydrophobic',
                               'is_donor',
                               'is_acceptor',
                               'is_hbonded',
                               'is_hbonded_donor',
                               'is_hbonded_acceptor',
                               'atom_mass',
                               'partial_charge',
                               'valence',
                               'LJ_epsilon',
                               'LJ_sigma']
        
        # Define feature categories for consistent naming throughout the class
        self.feature_categories = {
            # Discrete features that use binary/one-hot encoding and should be saturated
            'discrete': ['atom_type_C',
                         'atom_type_H',
                         'atom_type_O',
                         'is_hydrophobic',
                         'is_donor',
                         'is_acceptor',
                         'is_hbonded',
                         'is_hbonded_donor',
                         'is_hbonded_acceptor',
                         ],
            
            # Continuous features that should be normalized but not saturated
            'continuous': ['atom_mass',
                           'partial_charge',
                           'valence',
                           'LJ_epsilon',
                           'LJ_sigma',
                           ]
        }
        
        # Print initialization information
        if self.verbose:
            print(f"\n=== Initializing Voxel Grid Generator ===")
            print(f"    System: {self.zeolite_type}-{self.solvent_type}-{self.pore_type}-{self.adsorbate}")
            print(f"    Snapshot: {snapshot_index}")
            print(f"    Target interaction energy: {self.target_interaction_energy} (eV)")
            print(f"    Box grids size: {self.box_grids_size} Angstrom")
            print(f"    Box increment: {self.box_increment} Angstrom")
            print(f"    This generator instance handles ONE specific snapshot only.")
            print(f"    Using separate channel groups: adsorbate channels + solvent channels")
            print(f"    Atomic features: {self.atomic_features}")
            print(f"    Total channels: {2 * len(self.atomic_features)} (adsorbate: {len(self.atomic_features)}, solvent: {len(self.atomic_features)})")

        # Load single MD snapshot using MDAnalysis
        print(f"\n--- Loading MD Snapshot {self.snapshot_index} for {self.zeolite_type}-{self.solvent_type}-{self.pore_type}-{self.adsorbate}")
        
        self.snapshot_mda = snapshotMDAnalysis(
                                                zeolite_type = self.zeolite_type,
                                                solvent_type = self.solvent_type,
                                                pore_type = self.pore_type,
                                                adsorbate = self.adsorbate,
                                                snapshot_index = self.snapshot_index,
                                                verbose = False,
                                            )
        
        print(f"    Loaded snapshot {self.snapshot_index}")
        
        
        # Extract unique element types if any atom_type features are requested
        atom_type_features = [f for f in self.atomic_features if f.startswith('atom_type_')]
        if atom_type_features:
            self.extract_element_types()
        
        
        # Extract hydrophobic information if hydrophobic feature is requested
        if 'is_hydrophobic' in self.atomic_features:
            if verbose:
                print(f"\n--- Extracting Hydrophobic Information ---")
            self.atom_id_to_is_hydrophobic = extract_is_hydrophobic_info(
                self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                verbose=verbose
            )
        
        
        # Extract hydrogen bond potential (donor/acceptor capacity) if features are requested
        donor_acceptor_features = ['is_donor', 'is_acceptor']
        if any(feature in self.atomic_features for feature in donor_acceptor_features):
            if verbose:
                print(f"\n--- Extracting Hydrogen Bond Potential Information ---")
            
            # Use extract_is_donor_acceptor to get H-bond potential properties
            donor_acceptor_info = extract_is_donor_acceptor(
                zeolite_type=self.zeolite_type,
                solvent_type=self.solvent_type,
                pore_type=self.pore_type,
                adsorbate=self.adsorbate,
                snapshot_index=self.snapshot_index,
                r_cut=5.0,  # Same cutoff as used in HydrogenBondDetector
                verbose=verbose
            )
            
            # Store the atom ID to donor/acceptor status mappings
            self.atom_id_to_is_donor = donor_acceptor_info['atom_id_to_is_donor']
            self.atom_id_to_is_acceptor = donor_acceptor_info['atom_id_to_is_acceptor']
        
                
        # Extract hydrogen bond information if HB features are requested
        hb_features = ['is_hbonded', 'is_hbonded_donor', 'is_hbonded_acceptor']
        if any(feature in self.atomic_features for feature in hb_features):
            if verbose:
                print(f"\n--- Extracting Hydrogen Bond Information ---")
            
            # Use HydrogenBondDetector to get H-bond properties
            hbond_detector = HydrogenBondDetector(
                zeolite_type=self.zeolite_type,
                solvent_type=self.solvent_type,
                pore_type=self.pore_type,
                adsorbate=self.adsorbate,
                snapshot_index=self.snapshot_index,
                verbose=verbose
            )
            
            # Get atom ID to H-bond properties mapping directly from the detector
            self.atom_id_to_hbond_props = hbond_detector.atom_hbond_properties

        
        # Extract valence information if valence feature is requested
        if 'valence' in self.atomic_features:
            if verbose:
                print(f"\n--- Extracting Valence Information ---")
            self.atom_id_to_valence = extract_total_valence_info(
                self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                verbose=verbose
            )
        
        # Extract LJ parameters if LJ features are requested
        if 'LJ_epsilon' in self.atomic_features or 'LJ_sigma' in self.atomic_features:
            if verbose:
                print(f"\n--- Extracting LJ Parameters ---")
            
            if 'LJ_epsilon' in self.atomic_features:
                self.atom_type_to_epsilon = extract_LJ_parameter_info(
                    self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                    parameter='epsilon', verbose=verbose
                )
            
            if 'LJ_sigma' in self.atomic_features:
                self.atom_type_to_sigma = extract_LJ_parameter_info(
                    self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                    parameter='sigma', verbose=verbose
                )
        
        if verbose:
            discrete_in_list = [f for f in self.feature_categories['discrete'] if f in self.atomic_features]
            continuous_in_list = [f for f in self.feature_categories['continuous'] if f in self.atomic_features]
            print(f"\n--- Feature Categories ---")
            print(f"    Discrete features (binary/one-hot, will be saturated): {discrete_in_list}")
            print(f"    Continuous features (will be normalized): {continuous_in_list}")

        # Calculate box grid parameters
        self.find_box_range()
        
        # Find adsorbate center of mass
        self.COM_adsorbate = self.find_adsorbate_COM()
        
        # Create voxel grid based on occupancy (centered on adsorbate COM)
        self.voxel_occupancy_grid = self.create_voxel_occupancy_grid()
        
        # Create voxel features grid
        self.voxel_grid = self.create_voxel_features_grid()
        
        # Normalize the voxel features grid (only continuous features)
        self.voxel_grid = self.normalize_voxel_features_grid()

        # Saturate the voxel features grid
        self.voxel_grid = self.saturate_voxel_features_grid()
    
    
    
    
    def find_box_range(self):
        '''
        The purpose of this function is to take the input data and find the bounds of the box.
        INPUTS:
            self: class object
        OUTPUTS:
            self.max_bin_num: maximum bin number (integer)
            self.grid_half_box_length: half box length in Angstrom
            self.plot_axis_range: plotting range if you plotted in Cartesian coordinates, tuple (min, max)
            self.r_range: Range that we are interested in as a tuple (-half_box, +half_box)
            self.bin_volume: [float] bin volume
        '''
        ## Find the maximum bin number
        self.max_bin_num = int(np.floor(self.box_grids_size / self.box_increment))
        
        ## Find the half box length
        self.grid_half_box_length = self.box_grids_size / 2.0
        
        ## Find the plot axis range
        self.plot_axis_range = (0, self.max_bin_num)  # Plot length for axis in number of bins
        
        ## Find the range of the box
        self.r_range = (-self.grid_half_box_length, self.grid_half_box_length)
        
        ## Define the bin array
        self.bin_array = (self.max_bin_num, self.max_bin_num, self.max_bin_num)
        
        ## Calculate the size of the voxel grid
        self.box_grids_volume = self.box_grids_size**3
        
        ## Calculate the bin volume
        self.bin_volume = self.box_increment**3
        
        if self.verbose:
            print(f"\n--- Grid parameters:")
            print(f"    Bins per dimension:  {self.max_bin_num}")
            print(f"    Grid range:          {self.r_range} Å")
            print(f"    Voxel grids volume:  {self.box_grids_volume} Å^3")
            print(f"    Bin volume:          {self.bin_volume} Å^3")
        
        return

    
    def find_adsorbate_COM(self):
        '''
        Calculate the center of mass of the adsorbate molecule using MDAnalysis.
        Following the pattern used throughout the codebase for adsorbate COM calculations.
        
        OUTPUTS:
            COM_adsorbate: numpy array of adsorbate center of mass coordinates [x, y, z] in Angstrom
        '''
        print(f'\n--- Calculating Adsorbate Center of Mass for adsorbate "{self.adsorbate}"')
        
        # Get the universe from the snapshot
        universe = self.snapshot_mda.universe
        
        # Select adsorbate atoms using the standard pattern from the codebase
        adsorbate_atoms = universe.select_atoms('resname ADS')
        
        if len(adsorbate_atoms) == 0:
            raise ValueError("No adsorbate atoms found! Check that adsorbate residue is named 'ADS'")
        
        if self.verbose:
            print(f'    Adsorbate atoms found: {len(adsorbate_atoms)}')
        
        # Print all adsorbate atom coordinates for verification
        if self.verbose:
            print(f'    Adsorbate atom coordinates:')
            for i, atom in enumerate(adsorbate_atoms):
                pos = atom.position
                print(f'        Atom {i+1} ({atom.name}): [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] Å')
        
        # Calculate center of mass using MDAnalysis built-in method
        # This automatically handles mass-weighted calculation
        COM_adsorbate = adsorbate_atoms.center_of_mass()
        
        print(f'\n    Adsorbate COM coordinates: [{COM_adsorbate[0]:.3f}, {COM_adsorbate[1]:.3f}, {COM_adsorbate[2]:.3f}] Å')
        
        return COM_adsorbate

    
    def create_voxel_occupancy_grid(self):
        '''
        Create voxel grid centered on adsorbate COM with specified box_grids_size.

        INPUTS:
            None (uses self.include_zeolite, self.include_solvent, self.include_adsorbate)

        OUTPUTS:
            voxel_occupancy_grid: Voxel grid showing occupancy counts, shape (max_bin_num, max_bin_num, max_bin_num)
        '''
        print(f'\n--- Creating Voxel Occupancy Representation for snapshot {self.snapshot_index}')
        print(f'    Centering grid on adsorbate COM with box size: {self.box_grids_size} Å')
        
        # Get the universe from the snapshot
        universe = self.snapshot_mda.universe
        
        # Get the cell dimensions from the universe for periodic boundary condition handling
        box_dimensions = universe.dimensions[:3]  # x, y, z dimensions
        print(f'    System box dimensions: {box_dimensions}')
        
        # Select atoms based on user preferences
        selected_atoms = []
        print('')
        if self.include_zeolite:
            zeolite_atoms = universe.select_atoms('resname FAU')  # More generic zeolite selection
            if len(zeolite_atoms) == 0:
                # Fallback to the zeolite type name
                zeolite_atoms = universe.select_atoms(f'resname {self.zeolite_type}')
            selected_atoms.append(zeolite_atoms)
            
            print(f'    Including zeolite atoms: {len(zeolite_atoms)}')
        
        if self.include_solvent:
            # Select both water and methanol atoms for mixed solvent systems
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            
            # Combine water and methanol atoms
            if len(water_atoms) > 0 and len(methanol_atoms) > 0:
                solvent_atoms = water_atoms + methanol_atoms
                selected_atoms.append(solvent_atoms)
                print(f'    Including solvent atoms: {len(solvent_atoms)} ({len(water_atoms)} water + {len(methanol_atoms)} methanol)')
            elif len(water_atoms) > 0:
                selected_atoms.append(water_atoms)
                print(f'    Including solvent (water) atoms: {len(water_atoms)}')
            elif len(methanol_atoms) > 0:
                selected_atoms.append(methanol_atoms)
                print(f'    Including solvent (methanol) atoms: {len(methanol_atoms)}')
            
        if self.include_adsorbate:
            adsorbate_atoms = universe.select_atoms('resname ADS')
            selected_atoms.append(adsorbate_atoms)
            print(f'    Including adsorbate atoms: {len(adsorbate_atoms)}')
        
        # Combine all selected atoms
        if selected_atoms:
            all_selected = selected_atoms[0]
            for atoms in selected_atoms[1:]:
                all_selected = all_selected + atoms
            positions = all_selected.positions
        else:
            print("    Warning: No atoms selected! Using all atoms.")
            positions = universe.atoms.positions
        
        print(f'    Total selected atoms: {len(positions)}')
        
        # Calculate relative positions from adsorbate COM (similar to calc_solute_solvent_displacements)
        relative_positions = positions - self.COM_adsorbate
        
        # Apply minimum image convention for periodic boundary conditions
        for dim in range(3):
            relative_positions[:, dim] = relative_positions[:, dim] - box_dimensions[dim] * np.round(relative_positions[:, dim] / box_dimensions[dim])
        
        # Filter atoms within the desired box range
        within_box_mask = np.all(np.abs(relative_positions) <= self.grid_half_box_length, axis=1)
        atoms_in_box = relative_positions[within_box_mask]
        
        # Use histogram method similar to calc_num_dist in generate_grid_interpolation.py
        arange = (self.r_range, self.r_range, self.r_range)
        
        # Generate histogram using numpy.histogramdd (same as generate_grid_interpolation.py)
        if len(atoms_in_box) > 0:
            voxel_occupancy_grid, edges = np.histogramdd(atoms_in_box,
                                                         bins = self.bin_array,
                                                         range = arange,
                                                         density = False)
        else:
            print("    No atoms found within the specified box range, creating empty grid")
            voxel_occupancy_grid = np.zeros(self.bin_array, dtype=float)
        
        # Convert to integer counts (histogramdd returns float64 by default)
        voxel_occupancy_grid = voxel_occupancy_grid.astype(int)
        
        # Print voxel grid shape and occupancy distribution
        print(f'\n    Created voxel grid shape: {voxel_occupancy_grid.shape}')
        
        max_occupancy = np.max(voxel_occupancy_grid)
        print(f'\n    Voxel occupancy distribution:')
        for i in range(max_occupancy + 1):
            count = np.sum(voxel_occupancy_grid == i)
            print(f'    --  {i} atoms: {count} voxels')
        
        return voxel_occupancy_grid

    
    # Keep the visualization methods but update them to work with the new structure
    def plot_voxel_occupancy_grids(self, voxel_grid=None):
        '''
        Visualization of the voxel grid showing water atoms in blue and adsorbate atoms in red.
        Uses existing voxel grid and atom position data to distinguish atom types.

        INPUTS:
            voxel_grid: numpy array, the voxel grid to visualize (optional, uses self.voxel_occupancy_grid if None)

        OUTPUTS:
            Displays a 3D plot of the voxel grid with color-coded atoms.
        '''
        if voxel_grid is None:
            voxel_grid = self.voxel_occupancy_grid
            
        print(f"\n--- Plotting Voxel Grid for {self.zeolite_type}-{self.adsorbate} snapshot {self.snapshot_index:02d}")
        
        # Get the universe from the snapshot
        universe = self.snapshot_mda.universe
        
        # Get solvent and adsorbate atoms separately
        water_atoms = universe.select_atoms('resname HOH')
        methanol_atoms = universe.select_atoms('resname MEO')
        adsorbate_atoms = universe.select_atoms('resname ADS')
        
        # Combine water and methanol for solvent visualization
        if len(water_atoms) > 0 and len(methanol_atoms) > 0:
            solvent_atoms = water_atoms + methanol_atoms
            print(f"    Solvent atoms: {len(solvent_atoms)} ({len(water_atoms)} water + {len(methanol_atoms)} methanol)")
        elif len(water_atoms) > 0:
            solvent_atoms = water_atoms
            print(f"    Solvent atoms: {len(solvent_atoms)} (water only)")
        elif len(methanol_atoms) > 0:
            solvent_atoms = methanol_atoms
            print(f"    Solvent atoms: {len(solvent_atoms)} (methanol only)")
        else:
            solvent_atoms = universe.select_atoms('name DUMMY')  # Empty selection
            print(f"    No solvent atoms found")
        
        print(f"    Adsorbate atoms: {len(adsorbate_atoms)}")
        
        # Initialize separate grids for solvent and adsorbate
        solvent_voxels = np.zeros(self.bin_array, dtype=bool)
        adsorbate_voxels = np.zeros(self.bin_array, dtype=bool)
        
        # Process solvent atoms (water + methanol)
        if len(solvent_atoms) > 0:
            solvent_positions = solvent_atoms.positions
            solvent_relative_positions = solvent_positions - self.COM_adsorbate
            
            # Apply minimum image convention
            box_dimensions = universe.dimensions[:3]
            for dim in range(3):
                solvent_relative_positions[:, dim] = solvent_relative_positions[:, dim] - box_dimensions[dim] * np.round(solvent_relative_positions[:, dim] / box_dimensions[dim])
            
            # Filter atoms within box and convert to voxel indices
            within_box_mask = np.all(np.abs(solvent_relative_positions) <= self.grid_half_box_length, axis=1)
            solvent_in_box = solvent_relative_positions[within_box_mask]
            
            if len(solvent_in_box) > 0:
                shifted_positions = solvent_in_box + self.grid_half_box_length
                voxel_indices = np.floor(shifted_positions / self.box_increment).astype(int)
                voxel_indices = np.clip(voxel_indices, 0, self.max_bin_num - 1)
                
                for indices in voxel_indices:
                    solvent_voxels[tuple(indices)] = True
        
        # Process adsorbate atoms
        if len(adsorbate_atoms) > 0:
            adsorbate_positions = adsorbate_atoms.positions
            adsorbate_relative_positions = adsorbate_positions - self.COM_adsorbate
            
            # Apply minimum image convention
            for dim in range(3):
                adsorbate_relative_positions[:, dim] = adsorbate_relative_positions[:, dim] - box_dimensions[dim] * np.round(adsorbate_relative_positions[:, dim] / box_dimensions[dim])
            
            # Filter atoms within box and convert to voxel indices
            within_box_mask = np.all(np.abs(adsorbate_relative_positions) <= self.grid_half_box_length, axis=1)
            adsorbate_in_box = adsorbate_relative_positions[within_box_mask]
            
            if len(adsorbate_in_box) > 0:
                shifted_positions = adsorbate_in_box + self.grid_half_box_length
                voxel_indices = np.floor(shifted_positions / self.box_increment).astype(int)
                voxel_indices = np.clip(voxel_indices, 0, self.max_bin_num - 1)
                
                for indices in voxel_indices:
                    adsorbate_voxels[tuple(indices)] = True
        
        # Coordinates for plotting
        x, y, z = np.indices(np.array(solvent_voxels.shape) + 1)
        
        # Set up the figure and axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Voxel Grid - Snapshot {self.snapshot_index}\nBlue: Solvent, Red: Adsorbate')

        # Calculate aspect ratios based on the shape of the voxel grid
        shape = np.array(solvent_voxels.shape)
        aspect_ratio = shape / shape.max()
        
        # Plot solvent atoms in blue
        if np.any(solvent_voxels):
            ax.voxels(x, y, z, solvent_voxels, facecolors='blue', alpha=0.7, edgecolor='none')
        
        # Plot adsorbate atoms in red
        if np.any(adsorbate_voxels):
            ax.voxels(x, y, z, adsorbate_voxels, facecolors='red', alpha=0.8, edgecolor='none')

        # Set the aspect of the plot to match the voxel dimensions
        ax.set_box_aspect(aspect_ratio)
        
        ax.set_xlabel('X dimension (voxels)')
        ax.set_ylabel('Y dimension (voxels)')
        ax.set_zlabel('Z dimension (voxels)')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='blue', alpha=0.7, label='Solvent'),
                           Patch(facecolor='red', alpha=0.8, label='Adsorbate')]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.show()
        
        return fig, ax
    
    
    def extract_element_types(self):
        '''
        Extract unique element types from the selected molecules (water and adsorbate) for atom_type features.
        Element names are extracted from atom names like "O_HOH", "H_ADS" by taking the part before underscore.
        '''
        print(f'\n--- Extracting Element Types for Atom Type Features')
        
        universe = self.snapshot_mda.universe
        
        # Select solvent (water and methanol) and adsorbate atoms (not zeolite for now)
        water_atoms = universe.select_atoms('resname HOH')
        methanol_atoms = universe.select_atoms('resname MEO')
        adsorbate_atoms = universe.select_atoms('resname ADS')
        
        # Combine selected atoms
        selected_atoms = water_atoms + methanol_atoms + adsorbate_atoms
        
        if self.verbose:
            print(f'    Selected {len(water_atoms)} water atoms, {len(methanol_atoms)} methanol atoms, and {len(adsorbate_atoms)} adsorbate atoms')
        
        # Extract element symbols from atom names (part before underscore)
        element_symbols = set()
        for atom in selected_atoms:
            element = atom.name.split('_')[0]  # Take part before underscore
            element_symbols.add(element)
        
        # Sort for consistent ordering
        self.element_types = sorted(list(element_symbols))
        
        if self.verbose:
            print(f'    Found element types in system: {self.element_types}')
            
            # Check which atom_type features are requested
            requested_atom_types = [f.split('_')[-1] for f in self.feature_list if f.startswith('atom_type_')]
            print(f'    Requested atom_type features: {requested_atom_types}')
            
            # Warn about mismatches
            for req_type in requested_atom_types:
                if req_type not in self.element_types:
                    print(f'    Warning: Requested atom_type_{req_type} not found in system - will be all zeros')
        
        return

    def create_voxel_features_grid(self):
        '''
        Create 4D voxel grid with atomic features separated into two channel groups:
        - First N channels: adsorbate features only (solvent atoms = 0 in these channels)
        - Last N channels: solvent features only (adsorbate atoms = 0 in these channels)
        
        Uses sum aggregation method for multiple atoms in same voxel.
        
        INPUTS:
            None (uses self.include_zeolite, self.include_solvent, self.include_adsorbate)
        
        OUTPUTS:
            voxel_features_grid: 4D numpy array with shape (max_bin_num, max_bin_num, max_bin_num, 2*num_atomic_features)
        '''
        print(f'\n--- Creating Separated Channel Groups Voxel Features Grid for snapshot {self.snapshot_index}')
        print(f'    Atomic features: {self.atomic_features}')
        
        # Calculate total number of feature channels: 2 * number of atomic features
        num_atomic_features = len(self.atomic_features)
        total_feature_channels = 2 * num_atomic_features
        
        # Create channel mapping for both groups
        self.feature_channel_mapping = {}
        
        # Adsorbate channels (first N channels)
        for i, feature_name in enumerate(self.atomic_features):
            adsorbate_channel_name = f"adsorbate_{feature_name}"
            self.feature_channel_mapping[adsorbate_channel_name] = [i]
            print(f'    {adsorbate_channel_name}: channel {i}')
        
        # Solvent channels (last N channels) 
        for i, feature_name in enumerate(self.atomic_features):
            solvent_channel_name = f"solvent_{feature_name}"
            channel_idx = num_atomic_features + i
            self.feature_channel_mapping[solvent_channel_name] = [channel_idx]
            print(f'    {solvent_channel_name}: channel {channel_idx}')
        
        print(f'    Total feature channels: {total_feature_channels} (adsorbate: {num_atomic_features}, solvent: {num_atomic_features})')
        
        # Get the universe from the snapshot
        universe = self.snapshot_mda.universe
        box_dimensions = universe.dimensions[:3]
        
        # Initialize 4D voxel grid: (x, y, z, features)
        voxel_features_grid = np.zeros((*self.bin_array, total_feature_channels), dtype=float)
        print(f'    Created 4D voxel grid shape: {voxel_features_grid.shape}')
        
        # Process each atom type separately to assign appropriate features
        atom_groups = []
        
        print('')
        if self.include_solvent:
            # Select both water and methanol atoms for mixed solvent systems
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            
            # Combine water and methanol atoms into one solvent group
            if len(water_atoms) > 0 and len(methanol_atoms) > 0:
                # Both water and methanol present
                solvent_atoms = water_atoms + methanol_atoms
                atom_groups.append(('water', solvent_atoms))  # Use 'water' label for combined solvent
                print(f'    Processing {len(solvent_atoms)} solvent atoms ({len(water_atoms)} water + {len(methanol_atoms)} methanol)')
            elif len(water_atoms) > 0:
                # Only water present
                atom_groups.append(('water', water_atoms))
                print(f'    Processing {len(water_atoms)} water atoms')
            elif len(methanol_atoms) > 0:
                # Only methanol present
                atom_groups.append(('water', methanol_atoms))  # Use 'water' label for consistency
                print(f'    Processing {len(methanol_atoms)} methanol atoms')
        
        if self.include_adsorbate:
            adsorbate_atoms = universe.select_atoms('resname ADS')
            if len(adsorbate_atoms) > 0:
                atom_groups.append(('adsorbate', adsorbate_atoms))
                print(f'    Processing {len(adsorbate_atoms)} adsorbate atoms')
        
        if self.include_zeolite:
            zeolite_atoms = universe.select_atoms(f'resname {self.zeolite_type}')
            if len(zeolite_atoms) > 0:
                atom_groups.append(('zeolite', zeolite_atoms))
                print(f'    Processing {len(zeolite_atoms)} zeolite atoms')
        
        # Process each atom group
        for atom_type, atoms in atom_groups:
            if len(atoms) == 0:
                continue
                
            # Get positions and apply periodic boundary conditions
            positions = atoms.positions
            relative_positions = positions - self.COM_adsorbate
            
            # Apply minimum image convention
            for dim in range(3):
                relative_positions[:, dim] = relative_positions[:, dim] - box_dimensions[dim] * np.round(relative_positions[:, dim] / box_dimensions[dim])
            
            # Filter atoms within the desired box range
            within_box_mask = np.all(np.abs(relative_positions) <= self.grid_half_box_length, axis=1)
            atoms_in_box = relative_positions[within_box_mask]
            atoms_in_box_objects = atoms[within_box_mask]  # Keep atom objects for feature extraction
            
            if len(atoms_in_box) == 0:
                print(f'    No {atom_type} atoms within box range')
                continue
            
            print(f'    {len(atoms_in_box)} {atom_type} atoms within box range')
            
            # Convert positions to voxel indices
            shifted_positions = atoms_in_box + self.grid_half_box_length
            voxel_indices = np.floor(shifted_positions / self.box_increment).astype(int)
            
            # Filter out atoms that would be outside the voxel grid bounds
            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < self.max_bin_num), axis=1)
            voxel_indices = voxel_indices[valid_mask]
            atoms_in_box_objects = atoms_in_box_objects[valid_mask]
            
            if len(voxel_indices) == 0:
                print(f'    No {atom_type} atoms within valid voxel grid bounds')
                continue
            
            if len(voxel_indices) != len(atoms_in_box):
                atoms_filtered = len(atoms_in_box) - len(voxel_indices)
                print(f'    Filtered out {atoms_filtered} {atom_type} atoms at grid boundaries')
            
            print(f'    {len(voxel_indices)} {atom_type} atoms within valid voxel grid bounds')
            
            # Determine which channel group to use based on atom type
            if atom_type == 'adsorbate':
                channel_prefix = 'adsorbate'
            elif atom_type in ['water', 'zeolite']:  # Both water and zeolite go to solvent channels
                channel_prefix = 'solvent'
            else:
                print(f'    Warning: Unknown atom type {atom_type}, skipping')
                continue
            
            # Fill features for each atom using sum aggregation
            for i, (indices, atom) in enumerate(zip(voxel_indices, atoms_in_box_objects)):
                ix, iy, iz = tuple(indices)
                
                # Extract features for this atom (only atomic features, no mol_type)
                features = self.get_atom_features(atom, atom_type, self.atomic_features)
                
                # Fill appropriate channel group with features
                for feature_name in self.atomic_features:
                    channel_name = f"{channel_prefix}_{feature_name}"
                    feat_idx = self.feature_channel_mapping[channel_name][0]
                    
                    # Sum values for atoms in same voxel
                    voxel_features_grid[ix, iy, iz, feat_idx] += features[feature_name]
        
        # Calculate and print voxel occupancy distribution
        total_atoms_per_voxel = np.zeros(self.bin_array, dtype=int)
        
        # Count total atoms per voxel by summing across all atom groups
        for atom_type, atoms in atom_groups:
            if len(atoms) == 0:
                continue
            
            # Get positions and apply periodic boundary conditions
            positions = atoms.positions
            relative_positions = positions - self.COM_adsorbate
            
            # Apply minimum image convention
            for dim in range(3):
                relative_positions[:, dim] = relative_positions[:, dim] - box_dimensions[dim] * np.round(relative_positions[:, dim] / box_dimensions[dim])
            
            # Filter atoms within the desired box range
            within_box_mask = np.all(np.abs(relative_positions) <= self.grid_half_box_length, axis=1)
            atoms_in_box = relative_positions[within_box_mask]
            
            if len(atoms_in_box) == 0:
                continue
            
            # Convert positions to voxel indices
            shifted_positions = atoms_in_box + self.grid_half_box_length
            voxel_indices = np.floor(shifted_positions / self.box_increment).astype(int)
            
            # Count atoms per voxel
            for indices in voxel_indices:
                if np.all((indices >= 0) & (indices < self.max_bin_num)):
                    ix, iy, iz = tuple(indices)
                    total_atoms_per_voxel[ix, iy, iz] += 1
        
        # Print voxel occupancy distribution
        max_occupancy = np.max(total_atoms_per_voxel)
        print(f'\n    Voxel occupancy distribution:')
        for i in range(max_occupancy + 1):
            count = np.sum(total_atoms_per_voxel == i)
            print(f'    --  {i} atoms: {count} voxels')
        
        # Print statistics for both channel groups
        print(f'\n    Voxel features grid statistics:')
        
        # Print adsorbate channel statistics
        print(f'    Adsorbate channels:')
        for feature_name in self.atomic_features:
            channel_name = f"adsorbate_{feature_name}"
            feat_idx = self.feature_channel_mapping[channel_name][0]
            feature_slice = voxel_features_grid[:, :, :, feat_idx]
            non_zero_count = np.count_nonzero(feature_slice)
            min_val = np.min(feature_slice)
            max_val = np.max(feature_slice)
            print(f'        {channel_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}]')
        
        # Print solvent channel statistics
        print(f'    Solvent channels:')
        for feature_name in self.atomic_features:
            channel_name = f"solvent_{feature_name}"
            feat_idx = self.feature_channel_mapping[channel_name][0]
            feature_slice = voxel_features_grid[:, :, :, feat_idx]
            non_zero_count = np.count_nonzero(feature_slice)
            min_val = np.min(feature_slice)
            max_val = np.max(feature_slice)
            print(f'        {channel_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}]')
        
        return voxel_features_grid


    def plot_voxel_features_grid(self, feature='atom_type_C', group='adsorbate', element_type=None, value_threshold=1e-6, legend=False):
        '''
        Visualization of a specific feature channel from the voxel features grid.
        Now supports separated channel groups for adsorbate and solvent.
        Adsorbate voxels shown in red, solvent voxels in blue, with intensity based on feature values.

        INPUTS:
            feature: str, the atomic feature name to visualize (e.g., 'atom_type_C', 'atom_mass', 'partial_charge')
            group: str, which channel group to visualize ('adsorbate' or 'solvent')
            element_type: str, deprecated parameter (kept for compatibility)
            value_threshold: float, minimum absolute value to consider as non-zero for visualization

        OUTPUTS:
            Displays a 3D plot of the specified feature channel.
        '''
        print(f"\n--- Plotting Voxel Features Grid for {self.zeolite_type}-{self.adsorbate} snapshot {self.snapshot_index:02d}")
        print(f"    Feature: {feature}")
        print(f"    Channel group: {group}")
        print(f"    Value threshold: {value_threshold}")
        
        # Check if feature exists in atomic features list
        if feature not in self.atomic_features:
            print(f"    Error: Feature '{feature}' not found in atomic features list")
            print(f"    Available atomic features: {self.atomic_features}")
            return None, None
        
        # Check if group is valid
        if group not in ['adsorbate', 'solvent']:
            print(f"    Error: Group '{group}' must be 'adsorbate' or 'solvent'")
            return None, None
        
        # Get the channel index
        channel_name = f"{group}_{feature}"
        if channel_name not in self.feature_channel_mapping:
            print(f"    Error: Channel '{channel_name}' not found in channel mapping")
            return None, None
        
        channel_idx = self.feature_channel_mapping[channel_name][0]
        
        if element_type is not None:
            print(f"    Warning: element_type parameter is deprecated, feature name should include element (e.g., 'atom_type_C')")
        
        print(f"    Channel index: {channel_idx}")
        
        # Extract the feature channel
        feature_grid = self.voxel_grid[:, :, :, channel_idx]
        
        # Show values above threshold (using absolute value for features that might have negative values)
        feature_voxels = np.abs(feature_grid) > value_threshold
        
        # Get statistics
        non_zero_count = np.count_nonzero(np.abs(feature_grid) > value_threshold)
        min_val = np.min(feature_grid)
        max_val = np.max(feature_grid)
        
        print(f"    Feature statistics:")
        print(f"        Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"        Voxels above threshold ({value_threshold}): {non_zero_count}")
        
        # Create the plot even if no non-zero voxels are found
        if non_zero_count == 0:
            print(f"    Note: No voxels found above threshold {value_threshold} for feature '{channel_name}'.")
            print(f"    Creating empty plot to show feature channel structure.")
            # Set feature_voxels to False everywhere for empty plot
            feature_voxels = np.zeros_like(feature_grid, dtype=bool)
        
        # Get voxel indices where feature values are above threshold
        voxel_indices = np.where(feature_voxels)
        
        # Create full-size boolean arrays for consistent plotting
        feature_voxel_plot = np.zeros(self.bin_array, dtype=bool)
        feature_alphas = np.zeros(self.bin_array, dtype=float)
        
        # Only process if there are non-zero voxels
        if non_zero_count > 0:
            for i, j, k in zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]):
                feature_val = feature_grid[i, j, k]
                
                # Calculate alpha based on normalized feature value
                if max_val > 0 or min_val < 0:
                    # Use absolute maximum for normalization to handle both positive and negative values
                    max_abs_val = max(abs(max_val), abs(min_val))
                    normalized_val = abs(feature_val) / max_abs_val
                else:
                    normalized_val = 1.0
                # Set alpha range from 0.3 to 0.9 for better visibility
                alpha = 0.3 + 0.6 * normalized_val
                
                feature_voxel_plot[i, j, k] = True
                feature_alphas[i, j, k] = alpha
        
        # Create coordinates for the FULL grid (not just occupied voxels)
        # This ensures consistent axis ranges for all plots
        x = np.arange(self.max_bin_num + 1)
        y = np.arange(self.max_bin_num + 1)
        z = np.arange(self.max_bin_num + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Set up the figure and axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create title with additional info for empty plots
        if non_zero_count == 0:
            title = f'Voxel Features Grid - Snapshot {self.snapshot_index}\nFeature: {channel_name} (EMPTY - No values above threshold)'
        else:
            title = f'Voxel Features Grid - Snapshot {self.snapshot_index}\nFeature: {channel_name}'
        ax.set_title(title, fontsize=12)
        
        # Choose color based on group
        if group == 'adsorbate':
            base_color = 'red'
            rgb_color = [1.0, 0.0, 0.0]
        else:  # solvent
            base_color = 'blue'
            rgb_color = [0.0, 0.0, 1.0]
        
        # Plot feature voxels (if any)
        if np.any(feature_voxel_plot):
            # Create color array with varying alpha
            feature_colors = np.zeros(feature_voxel_plot.shape + (4,), dtype=float)
            feature_indices = np.where(feature_voxel_plot)
            for i, j, k in zip(feature_indices[0], feature_indices[1], feature_indices[2]):
                alpha = feature_alphas[i, j, k]
                feature_colors[i, j, k] = rgb_color + [alpha]  # RGB color with varying alpha
            ax.voxels(X, Y, Z, feature_voxel_plot, facecolors=feature_colors, edgecolor='none')
        
        # For empty plots, add a subtle grid outline to show the voxel space
        if non_zero_count == 0:
            # Draw wire frame to show the voxel grid boundaries
            ax.plot([0, self.max_bin_num], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 0], [0, self.max_bin_num], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 0], [0, 0], [0, self.max_bin_num], 'k-', alpha=0.2, linewidth=0.5)
            
            # Add text annotation to explain the empty plot
            ax.text2D(0.5, 0.5, f'No {channel_name} features detected\nin this voxel grid', 
                     transform=ax.transAxes, ha='center', va='center',
                     fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # FORCE consistent axis limits for ALL plots
        ax.set_xlim(0, self.max_bin_num)
        ax.set_ylim(0, self.max_bin_num)
        ax.set_zlim(0, self.max_bin_num)
        
        # Set equal aspect ratio to ensure cubic voxels
        ax.set_box_aspect([1, 1, 1])
        
        ax.set_xlabel('X dimension (voxels)', fontsize=10)
        ax.set_ylabel('Y dimension (voxels)', fontsize=10)
        ax.set_zlabel('Z dimension (voxels)', fontsize=10)
        
        if legend:
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=base_color, alpha=0.7, label=f'{group.capitalize()}')
            ]
            if non_zero_count == 0:
                legend_elements.append(Patch(facecolor='lightgray', alpha=0.8, label='Empty feature'))
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax

    def get_atom_features(self, atom, atom_type: str, feature_list: list) -> dict:
        '''
        Extract atomic features for a given atom.
        
        INPUTS:
            atom: MDAnalysis atom object
            atom_type: str, type of atom ('water', 'adsorbate', 'zeolite')
            feature_list: list of feature names to extract
            
        OUTPUTS:
            features: dict with feature_name -> feature_value mapping
        '''
        features = {}
        
        for feature_name in feature_list:
            if feature_name.startswith('atom_type_'):
                # Extract requested element from feature name (e.g., 'atom_type_C' -> 'C')
                requested_element = feature_name.split('_')[-1]
                
                # Extract actual element symbol from atom name (part before underscore)
                actual_element = atom.name.split('_')[0]
                
                # Set to 1.0 if elements match, 0.0 otherwise
                if requested_element == actual_element:
                    features[feature_name] = 1.0
                else:
                    features[feature_name] = 0.0
            
            elif feature_name == 'is_hydrophobic':
                # Get hydrophobic status from atom ID mapping
                try:
                    atom_id = int(atom.id)  # Ensure atom ID is int
                    
                    if hasattr(self, 'atom_id_to_is_hydrophobic') and atom_id in self.atom_id_to_is_hydrophobic:
                        is_hydrophobic = self.atom_id_to_is_hydrophobic[atom_id]
                        features['is_hydrophobic'] = 1.0 if is_hydrophobic else 0.0
                    else:
                        # For atoms not in the hydrophobic mapping (e.g., zeolite atoms), set to 0
                        features['is_hydrophobic'] = 0.0
                except (AttributeError, ValueError, KeyError) as e:
                    if self.verbose:
                        print(f"    Warning: Could not extract hydrophobic status for atom {atom.name}: {e}, setting to 0.0")
                    features['is_hydrophobic'] = 0.0
                    
            elif feature_name == 'is_donor':
                # Get hydrogen bond donor potential from atom ID mapping
                try:
                    atom_id = int(atom.id)  # Ensure atom ID is int
                    
                    if hasattr(self, 'atom_id_to_is_donor') and atom_id in self.atom_id_to_is_donor:
                        is_donor = self.atom_id_to_is_donor[atom_id]
                        features['is_donor'] = 1.0 if is_donor else 0.0
                    else:
                        # For atoms not in the donor mapping (e.g., zeolite atoms), set to 0
                        features['is_donor'] = 0.0
                except (AttributeError, ValueError, KeyError) as e:
                    if self.verbose:
                        print(f"    Warning: Could not extract donor potential for atom {atom.name}: {e}, setting to 0.0")
                    features['is_donor'] = 0.0
            
            elif feature_name == 'is_acceptor':
                # Get hydrogen bond acceptor potential from atom ID mapping
                try:
                    atom_id = int(atom.id)  # Ensure atom ID is int
                    
                    if hasattr(self, 'atom_id_to_is_acceptor') and atom_id in self.atom_id_to_is_acceptor:
                        is_acceptor = self.atom_id_to_is_acceptor[atom_id]
                        features['is_acceptor'] = 1.0 if is_acceptor else 0.0
                    else:
                        # For atoms not in the acceptor mapping (e.g., zeolite atoms), set to 0
                        features['is_acceptor'] = 0.0
                except (AttributeError, ValueError, KeyError) as e:
                    if self.verbose:
                        print(f"    Warning: Could not extract acceptor potential for atom {atom.name}: {e}, setting to 0.0")
                    features['is_acceptor'] = 0.0
            
            elif feature_name == 'is_hbonded':
                # Get hydrogen bond status from atom ID mapping
                try:
                    atom_id = int(atom.id)  # Ensure atom ID is int
                    
                    if hasattr(self, 'atom_id_to_hbond_props') and atom_id in self.atom_id_to_hbond_props:
                        is_hbonded = self.atom_id_to_hbond_props[atom_id]['is_hbonded']
                        features['is_hbonded'] = 1.0 if is_hbonded else 0.0
                    else:
                        # For atoms not in the H-bond mapping (e.g., zeolite atoms), set to 0
                        features['is_hbonded'] = 0.0
                except (AttributeError, ValueError, KeyError) as e:
                    if self.verbose:
                        print(f"    Warning: Could not extract H-bond status for atom {atom.name}: {e}, setting to 0.0")
                    features['is_hbonded'] = 0.0
            
            elif feature_name == 'is_hbonded_donor':
                # Get hydrogen bond donor status from atom ID mapping
                try:
                    atom_id = int(atom.id)  # Ensure atom ID is int
                    
                    if hasattr(self, 'atom_id_to_hbond_props') and atom_id in self.atom_id_to_hbond_props:
                        is_hbonded_donor = self.atom_id_to_hbond_props[atom_id]['is_hbonded_donor']
                        features['is_hbonded_donor'] = 1.0 if is_hbonded_donor else 0.0
                    else:
                        # For atoms not in the H-bond mapping, set to 0
                        features['is_hbonded_donor'] = 0.0
                except (AttributeError, ValueError, KeyError) as e:
                    if self.verbose:
                        print(f"    Warning: Could not extract H-bond donor status for atom {atom.name}: {e}, setting to 0.0")
                    features['is_hbonded_donor'] = 0.0
            
            elif feature_name == 'is_hbonded_acceptor':
                # Get hydrogen bond acceptor status from atom ID mapping
                try:
                    atom_id = int(atom.id)  # Ensure atom ID is int
                    
                    if hasattr(self, 'atom_id_to_hbond_props') and atom_id in self.atom_id_to_hbond_props:
                        is_hbonded_acceptor = self.atom_id_to_hbond_props[atom_id]['is_hbonded_acceptor']
                        features['is_hbonded_acceptor'] = 1.0 if is_hbonded_acceptor else 0.0
                    else:
                        # For atoms not in the H-bond mapping, set to 0
                        features['is_hbonded_acceptor'] = 0.0
                except (AttributeError, ValueError, KeyError) as e:
                    if self.verbose:
                        print(f"    Warning: Could not extract H-bond acceptor status for atom {atom.name}: {e}, setting to 0.0")
                    features['is_hbonded_acceptor'] = 0.0
            
            elif feature_name == 'atom_mass':
                # Get atomic mass directly from MDAnalysis atom object
                try:
                    mass = atom.mass
                    features['atom_mass'] = float(mass)
                except (AttributeError, ValueError) as e:
                    print(f"    Warning: Could not extract mass for atom {atom.name}: {e}, setting to 0.0")
                    features['atom_mass'] = 0.0
            
            elif feature_name == 'partial_charge':
                # Get partial charge from MDAnalysis atom object
                try:
                    charge = atom.charge
                    features['partial_charge'] = float(charge)
                except (AttributeError, ValueError) as e:
                    print(f"    Warning: Could not extract charge for atom {atom.name}: {e}, setting to 0.0")
                    features['partial_charge'] = 0.0
            
            elif feature_name == 'valence':
                # Get valence from atom ID mapping
                try:
                    atom_id = atom.id  # Get atom ID from MDAnalysis atom object
                    
                    if hasattr(self, 'atom_id_to_valence') and atom_id in self.atom_id_to_valence:
                        valence = self.atom_id_to_valence[atom_id]
                        features['valence'] = float(valence)
                    else:
                        # For atoms not in the valence mapping (e.g., zeolite atoms), set default valence
                        if atom_type == 'zeolite':
                            features['valence'] = 0.0  # Default for zeolite atoms
                        else:
                            print(f"    Warning: Could not find valence for atom ID {atom_id} ({atom.name}), setting to 0.0")
                            features['valence'] = 0.0
                except (AttributeError, ValueError) as e:
                    print(f"    Warning: Could not extract valence for atom {atom.name}: {e}, setting to 0.0")
                    features['valence'] = 0.0
            
            elif feature_name == 'LJ_epsilon':
                # Get LJ epsilon parameter from atom type mapping
                try:
                    # Get atom type ID from MDAnalysis atom object
                    atom_type_id = getattr(atom, 'type', None)
                    
                    # If type attribute doesn't exist, try to extract from atom name
                    if atom_type_id is None:
                        # Try to get type ID from topology if available
                        try:
                            atom_type_id = atom.universe.atoms[atom.index].type
                        except:
                            atom_type_id = None
                    
                    # Convert to integer if it's a string
                    if isinstance(atom_type_id, str):
                        try:
                            atom_type_id = int(atom_type_id)
                        except ValueError:
                            atom_type_id = None
                    
                    if hasattr(self, 'atom_type_to_epsilon') and atom_type_id is not None and atom_type_id in self.atom_type_to_epsilon:
                        epsilon = self.atom_type_to_epsilon[atom_type_id]
                        features['LJ_epsilon'] = float(epsilon)
                    else:
                        if atom_type == 'zeolite':
                            features['LJ_epsilon'] = 0.0  # Default for zeolite atoms
                        else:
                            print(f"    Warning: Could not find LJ epsilon for atom type {atom_type_id} ({atom.name}), setting to 0.0")
                            features['LJ_epsilon'] = 0.0
                except (AttributeError, ValueError) as e:
                    print(f"    Warning: Could not extract LJ epsilon for atom {atom.name}: {e}, setting to 0.0")
                    features['LJ_epsilon'] = 0.0
            
            elif feature_name == 'LJ_sigma':
                # Get LJ sigma parameter from atom type mapping
                try:
                    # Get atom type ID from MDAnalysis atom object
                    atom_type_id = getattr(atom, 'type', None)
                    
                    # If type attribute doesn't exist, try to extract from atom name
                    if atom_type_id is None:
                        # Try to get type ID from topology if available
                        try:
                            atom_type_id = atom.universe.atoms[atom.index].type
                        except:
                            atom_type_id = None
                    
                    # Convert to integer if it's a string
                    if isinstance(atom_type_id, str):
                        try:
                            atom_type_id = int(atom_type_id)
                        except ValueError:
                            atom_type_id = None
                    
                    if hasattr(self, 'atom_type_to_sigma') and atom_type_id is not None and atom_type_id in self.atom_type_to_sigma:
                        sigma = self.atom_type_to_sigma[atom_type_id]
                        features['LJ_sigma'] = float(sigma)
                    else:
                        if atom_type == 'zeolite':
                            features['LJ_sigma'] = 0.0  # Default for zeolite atoms
                        else:
                            print(f"    Warning: Could not find LJ sigma for atom type {atom_type_id} ({atom.name}), setting to 0.0")
                            features['LJ_sigma'] = 0.0
                except (AttributeError, ValueError) as e:
                    print(f"    Warning: Could not extract LJ sigma for atom {atom.name}: {e}, setting to 0.0")
                    features['LJ_sigma'] = 0.0
            else:
                print(f"    Warning: Unknown feature '{feature_name}', setting to 0.0")
                features[feature_name] = 0.0
        
        return features

    def normalize_voxel_grid_minmax(self, voxel_grid=None):
        '''
        Normalize the voxel grid using min-max normalization to range [0, 1].
        This is a simple normalization that scales the values based on the minimum and maximum values in the grid.

        INPUTS:
            voxel_grid: numpy array, the voxel grid to normalize (optional, uses self.voxel_grid if None)

        OUTPUTS:
            normalized_grid: numpy array, the normalized voxel grid
        '''
        if voxel_grid is None:
            voxel_grid = self.voxel_grid
        
        # Perform min-max normalization
        min_val = np.min(voxel_grid)
        max_val = np.max(voxel_grid)
        
        # Avoid division by zero if max_val == min_val
        if max_val - min_val == 0:
            normalized_grid = np.zeros_like(voxel_grid)
        else:
            normalized_grid = (voxel_grid - min_val) / (max_val - min_val)
        
        return normalized_grid

    def store_pickle(self, file_name='voxel_grid'):
        '''
        Store the voxel grid and related information to a pickle file for later use.

        INPUTS:
            file_name: str, the base file name for the pickle file (without extension)

        OUTPUTS:
            Creates a pickle file with the voxel grid data
        '''
        
        # Create a dictionary to store all relevant data
        data_to_store = {
            'voxel_grid': self.voxel_grid,
            'voxel_occupancy_grid': self.voxel_occupancy_grid,
            'box_grids_size': self.box_grids_size,
            'box_increment': self.box_increment,
            'feature_list': self.feature_list,
            'snapshot_index': self.snapshot_index,
            'zeolite_type': self.zeolite_type,
            'solvent_type': self.solvent_type,
            'pore_type': self.pore_type,
            'adsorbate': self.adsorbate,
            'COM_adsorbate': self.COM_adsorbate,
        }
        
        # Create the file name with snapshot index
        file_name_full = f"{file_name}_snapshot{self.snapshot_index}.pkl"
        
        # Write to pickle file
        with open(file_name_full, 'wb') as f:
            pickle.dump(data_to_store, f)
        
        print(f"    Data stored to {file_name_full}")
        
        return file_name_full


    def normalize_voxel_features_grid(self, voxel_grid=None, normalization_method='max_abs'):
        '''
        Normalize the voxel features grid using sklearn preprocessing for continuous features only.
        Only normalizes voxels that are actually occupied by atoms (identified by non-zero values in any channel).
        Empty voxels remain at 0 to preserve sparsity.
        
        INPUTS:
            voxel_grid: numpy array, the voxel grid to normalize (optional, uses self.voxel_grid if None)
            normalization_method: str, normalization method to use:
                - 'max_abs': Max-Abs scaling to range [-1, 1] using sklearn.preprocessing.maxabs_scale
                - 'standard': StandardScaler (z-score normalization: mean=0, std=1) using sklearn.preprocessing.StandardScaler
        
        OUTPUTS:
            normalized_grid: numpy array, the normalized voxel grid
        '''
        if voxel_grid is None:
            voxel_grid = self.voxel_grid.copy()
        else:
            voxel_grid = voxel_grid.copy()
        
        print(f'\n--- Normalizing Voxel Features Grid')
        print(f'    Normalization method: {normalization_method}')
        print(f'    Only continuous features in occupied voxels will be normalized')
        print(f'    Discrete features and empty voxels will remain unchanged')
        
        # Get occupied voxels mask by checking if any channel has non-zero values
        occupied_mask = np.any(np.abs(voxel_grid) > 0, axis=3)
        
        num_occupied_voxels = np.sum(occupied_mask)
        total_voxels = np.prod(voxel_grid.shape[:3])
        
        print(f'    Total voxels: {total_voxels}')
        print(f'    Occupied voxels: {num_occupied_voxels}')
        print(f'    Empty voxels: {total_voxels - num_occupied_voxels}')
        
        if num_occupied_voxels == 0:
            print(f'    Warning: No occupied voxels found, skipping normalization')
            return voxel_grid
        
        # Use unified feature categories
        discrete_features = [f for f in self.feature_categories['discrete'] if f in self.atomic_features]
        continuous_features = [f for f in self.feature_categories['continuous'] if f in self.atomic_features]
        
        print(f'\n    Continuous features to normalize: {continuous_features}')
        print(f'    Discrete features (unchanged): {discrete_features}')
        
        # Normalize each continuous feature channel separately, only for occupied voxels
        # Process both adsorbate and solvent channels for each continuous feature
        for feature_name in continuous_features:
            for group in ['adsorbate', 'solvent']:
                channel_name = f"{group}_{feature_name}"
                if channel_name in self.feature_channel_mapping:
                    channel_idx = self.feature_channel_mapping[channel_name][0]
                    
                    # Extract the feature channel
                    feature_slice = voxel_grid[:, :, :, channel_idx]
                    
                    # Get values only from occupied voxels
                    occupied_values = feature_slice[occupied_mask]
                    
                    # Only normalize if there are non-zero values
                    if np.any(occupied_values != 0):
                        if normalization_method == 'max_abs':
                            # Max-abs scaling preserves sparsity (zeros remain zeros)
                            normalized_values = maxabs_scale(occupied_values)
                        
                        elif normalization_method == 'standard':
                            # Standard scaling (z-score normalization)
                            scaler = StandardScaler()
                            normalized_values = scaler.fit_transform(occupied_values.reshape(-1, 1)).flatten()
                        
                        else:
                            print(f'    Warning: Unknown normalization method {normalization_method}, skipping')
                            continue
                        
                        # Put normalized values back to occupied positions
                        voxel_grid[:, :, :, channel_idx][occupied_mask] = normalized_values
                        
                        # Calculate statistics
                        min_val = np.min(normalized_values)
                        max_val = np.max(normalized_values)
                        print(f'        {channel_name}: normalized to range [{min_val:.3f}, {max_val:.3f}]')
                    else:
                        print(f'        {channel_name}: no non-zero values, skipping normalization')
        
        # Print statistics for discrete features (unchanged)
        print(f'\n    Discrete features statistics (unchanged):')
        for feature_name in discrete_features:
            for group in ['adsorbate', 'solvent']:
                channel_name = f"{group}_{feature_name}"
                if channel_name in self.feature_channel_mapping:
                    channel_idx = self.feature_channel_mapping[channel_name][0]
                    feature_slice = voxel_grid[:, :, :, channel_idx]
                    non_zero_count = np.count_nonzero(feature_slice)
                    min_val = np.min(feature_slice)
                    max_val = np.max(feature_slice)
                    print(f'        {channel_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}]')
        
        print(f'\n    Normalization completed - empty voxels preserved as zeros')
        
        return voxel_grid

    def saturate_voxel_features_grid(self, voxel_grid=None):
        '''
        Apply van der Waals saturation to discrete features using PCMax algorithm.
        For each atom, calculate its continuous contribution n(r) to neighbor voxels within 2*r_vdw radius.
        At each voxel center, only the maximum effect from contributing atoms is kept.
        
        n = 1 - exp(-(r_vdw/r)^12)
        
        INPUTS:
            voxel_grid: numpy array, the voxel grid to saturate (optional, uses self.voxel_grid if None)
        
        OUTPUTS:
            saturated_grid: numpy array, the saturated voxel grid
        '''
        if voxel_grid is None:
            voxel_grid = self.voxel_grid.copy()
        else:
            voxel_grid = voxel_grid.copy()
        
        print(f'\n--- Applying van der Waals Saturation to Discrete Features')
        
        # Use unified feature categories
        discrete_features = [f for f in self.feature_categories['discrete'] if f in self.atomic_features]
        continuous_features = [f for f in self.feature_categories['continuous'] if f in self.atomic_features]
        
        print(f'    Discrete features (apply saturation): {discrete_features}')
        print(f'    Continuous features (keep unchanged): {continuous_features}')
        
        if not discrete_features:
            print(f'    No discrete features found, returning original grid')
            return voxel_grid
        
        # Get the universe from the snapshot
        universe = self.snapshot_mda.universe
        box_dimensions = universe.dimensions[:3]
        
        # Initialize new saturated grids for discrete features only
        saturated_grids = {}
        for feature_name in discrete_features:
            # Create saturated grids for both adsorbate and solvent channels
            for group in ['adsorbate', 'solvent']:
                channel_name = f"{group}_{feature_name}"
                if channel_name in self.feature_channel_mapping:
                    channel_idx = self.feature_channel_mapping[channel_name][0]
                    saturated_grids[channel_idx] = np.zeros(self.bin_array, dtype=float)
        
        # Process each atom type separately
        atom_groups = []
        
        if self.include_solvent:
            # Select both water and methanol atoms for mixed solvent systems
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            
            # Combine water and methanol atoms into one solvent group
            if len(water_atoms) > 0 and len(methanol_atoms) > 0:
                # Both water and methanol present
                solvent_atoms = water_atoms + methanol_atoms
                atom_groups.append(('water', solvent_atoms))  # Use 'water' label for combined solvent
            elif len(water_atoms) > 0:
                # Only water present
                atom_groups.append(('water', water_atoms))
            elif len(methanol_atoms) > 0:
                # Only methanol present
                atom_groups.append(('water', methanol_atoms))  # Use 'water' label for consistency
        
        if self.include_adsorbate:
            adsorbate_atoms = universe.select_atoms('resname ADS')
            if len(adsorbate_atoms) > 0:
                atom_groups.append(('adsorbate', adsorbate_atoms))
        
        if self.include_zeolite:
            zeolite_atoms = universe.select_atoms(f'resname {self.zeolite_type}')
            if len(zeolite_atoms) > 0:
                atom_groups.append(('zeolite', zeolite_atoms))
        
        # Process each atom group
        for atom_type, atoms in atom_groups:
            if len(atoms) == 0:
                continue
            
            if self.verbose:
                print(f'    Processing {len(atoms)} {atom_type} atoms')
            
            # Get positions and apply periodic boundary conditions
            positions = atoms.positions
            relative_positions = positions - self.COM_adsorbate
            
            # Apply minimum image convention
            for dim in range(3):
                relative_positions[:, dim] = relative_positions[:, dim] - box_dimensions[dim] * np.round(relative_positions[:, dim] / box_dimensions[dim])
            
            # Filter atoms within expanded range (2 * max van der Waals radius)
            max_vdw_radius = 2.1  # Maximum van der Waals radius in Angstrom (Si)
            expanded_range = self.grid_half_box_length + 2 * max_vdw_radius
            within_range_mask = np.all(np.abs(relative_positions) <= expanded_range, axis=1)
            atoms_in_range = relative_positions[within_range_mask]
            atoms_in_range_objects = atoms[within_range_mask]
            
            if len(atoms_in_range) == 0:
                print(f'        No {atom_type} atoms within expanded range')
                continue
            
            if self.verbose:
                print(f'        {len(atoms_in_range)} {atom_type} atoms within expanded range')
            
            # Process each atom and calculate its contribution to all voxels
            for atom_idx, (atom_pos, atom) in enumerate(zip(atoms_in_range, atoms_in_range_objects)):
                # Get van der Waals radius for this atom
                r_vdw = self.get_vdw_radius(atom)
                
                # Extract features for this atom (only atomic features, no mol_type)
                features = self.get_atom_features(atom, atom_type, discrete_features)
                
                # Determine which channel group to use based on atom type
                if atom_type == 'adsorbate':
                    channel_prefix = 'adsorbate'
                elif atom_type in ['water', 'zeolite']:  # Both water and zeolite go to solvent channels
                    channel_prefix = 'solvent'
                else:
                    print(f'    Warning: Unknown atom type {atom_type}, skipping saturation')
                    continue
                
                # Calculate which voxels this atom can influence (within 2 * r_vdw)
                influence_range = 2 * r_vdw
                
                # Find voxel range that this atom can influence
                min_voxel = np.floor((atom_pos - influence_range + self.grid_half_box_length) / self.box_increment).astype(int)
                max_voxel = np.ceil((atom_pos + influence_range + self.grid_half_box_length) / self.box_increment).astype(int)
                
                # Clip to valid voxel range
                min_voxel = np.clip(min_voxel, 0, self.max_bin_num)
                max_voxel = np.clip(max_voxel, 0, self.max_bin_num)
                
                # Iterate through all voxels within influence range
                for ix in range(min_voxel[0], max_voxel[0]):
                    for iy in range(min_voxel[1], max_voxel[1]):
                        for iz in range(min_voxel[2], max_voxel[2]):
                            # Calculate voxel center position
                            voxel_center = np.array([ix, iy, iz]) * self.box_increment - self.grid_half_box_length + self.box_increment/2
                            
                            # Calculate distance from atom to voxel center
                            r = np.linalg.norm(atom_pos - voxel_center)
                            
                            # Skip if outside influence range
                            if r > influence_range:
                                continue
                            
                            # Calculate softening factor using van der Waals formula
                            if r < 1e-6:  # Avoid division by zero
                                n = 1.0
                            else:
                                n = 1 - np.exp(-(r_vdw / r)**12)
                            
                            # Apply PCMax algorithm for discrete features
                            for feature_name in discrete_features:
                                channel_name = f"{channel_prefix}_{feature_name}"
                                if channel_name in self.feature_channel_mapping:
                                    channel_idx = self.feature_channel_mapping[channel_name][0]
                                    feature_val = features[feature_name]
                                    
                                    # Use PCMax for all binary features
                                    if feature_val > 0:  # Feature present
                                        saturated_grids[channel_idx][ix, iy, iz] = max(saturated_grids[channel_idx][ix, iy, iz], n)
                                        saturated_grids[channel_idx][ix, iy, iz] = max(saturated_grids[channel_idx][ix, iy, iz], n)
                                    # If feature_val == 0, don't update (stays at 0)
        

        # Replace discrete feature channels in the voxel grid with saturated versions
        for channel_idx, saturated_grid in saturated_grids.items():
            voxel_grid[:, :, :, channel_idx] = saturated_grid
        
        # Print statistics for features
        if self.verbose:
            # Print statistics for saturated features
            print(f'\n    Saturated discrete features statistics:')
            for feature_name in discrete_features:
                for group in ['adsorbate', 'solvent']:
                    channel_name = f"{group}_{feature_name}"
                    if channel_name in self.feature_channel_mapping:
                        channel_idx = self.feature_channel_mapping[channel_name][0]
                        feature_slice = voxel_grid[:, :, :, channel_idx]
                        non_zero_count = np.count_nonzero(feature_slice)
                        min_val = np.min(feature_slice)
                        max_val = np.max(feature_slice)
                        print(f'        {channel_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}] (saturated)')
        
            # Print statistics for continuous features (unchanged)
            print(f'\n    Continuous features statistics (unchanged):')
            for feature_name in continuous_features:
                for group in ['adsorbate', 'solvent']:
                    channel_name = f"{group}_{feature_name}"
                    if channel_name in self.feature_channel_mapping:
                        channel_idx = self.feature_channel_mapping[channel_name][0]
                        feature_slice = voxel_grid[:, :, :, channel_idx]
                        non_zero_count = np.count_nonzero(feature_slice)
                        min_val = np.min(feature_slice)
                        max_val = np.max(feature_slice)
                        print(f'        {channel_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}] (unchanged)')
        
        print(f'    van der Waals saturation completed')
        
        return voxel_grid
    
    def get_vdw_radius(self, atom):
        '''
        Get van der Waals radius for an atom based on its element type.
        
        INPUTS:
            atom: MDAnalysis atom object
            
        OUTPUTS:
            r_vdw: float, van der Waals radius in Angstrom
        '''
        # Extract element symbol from atom name (part before underscore)
        element = atom.name.split('_')[0]
        
        # Van der Waals radii in Angstrom (from literature)
        vdw_radii = {
            'H': 1.20,   # Hydrogen
            'C': 1.70,   # Carbon
            'O': 1.52,   # Oxygen
        }
        
        # Return van der Waals radius or default value
        return vdw_radii.get(element)  # Default to carbon radius if element not found

    def load_labels(self):
        """Load labels from CSV files for all environments."""
        if self.verbose:
            print(f"\n--- Loading Labels from CSV Files ---")
        
        database_path = get_paths('database_path')
        
        if self.verbose:
            print(f"    Database path: {database_path}")
            print(f"    Looking for CSV files: {LABEL_CSV_FILES}")
        
        # Load and merge all CSV files
        dfs = []
        for csv_file in LABEL_CSV_FILES:
            csv_path = os.path.join(database_path, csv_file)
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    dfs.append(df)
                    if self.verbose:
                        print(f"    ✅ Loaded {len(df)} records from {csv_file}")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"    ❌ Error reading {csv_file}: {e}")
                    continue
            else:
                if self.verbose:
                    print(f"    ❌ File not found: {csv_path}")
        
        if dfs:
            self.df_labels = pd.concat(dfs, ignore_index=True)
            if self.verbose:
                print(f"    ✅ Total records loaded: {len(self.df_labels)}")
                print(f"\n--- Available combinations:")
                
                # Show available combinations for debugging
                available_combinations = self.df_labels.groupby(['zeolite', 'environment', 'adsorbate']).size()
                for (zeolite, env, ads), count in available_combinations.head(5).items():
                    print(f"        {zeolite}-{env}-{ads}: {count} snapshots")
                
                if len(available_combinations) > 5:
                    print(f"        ... and {len(available_combinations) - 5} more combinations")
        else:
            self.df_labels = pd.DataFrame()
            if self.verbose:
                print(f"    ❌ No valid CSV files found or loaded")

    def get_target_interaction_energy(self) -> Optional[float]:
        """Get the target interaction energy for current configuration."""
        if self.df_labels is None or self.df_labels.empty:
            if self.verbose:
                print(f"    ⚠️  No labels loaded, returning None for target interaction energy")
            return None
        
        # Create environment string
        environment = f"{self.solvent_type}-{self.pore_type}"
        
        if self.verbose:
            print(f"\n--- Searching for Target Interaction Energy ---")
            print(f"    Looking for: {self.zeolite_type}-{environment}-{self.adsorbate}-{self.snapshot_index}")
        
        # Query the DataFrame for matching record
        mask = (
            (self.df_labels['zeolite'] == self.zeolite_type) &
            (self.df_labels['environment'] == environment) &
            (self.df_labels['adsorbate'] == self.adsorbate) &
            (self.df_labels['snapshot'] == self.snapshot_index)
        )
        
        matching_records = self.df_labels[mask]
        
        if len(matching_records) == 0:
            if self.verbose:
                print(f"    ❌ No matching record found")
                print(f"    Available zeolites: {sorted(self.df_labels['zeolite'].unique())}")
                print(f"    Available environments: {sorted(self.df_labels['environment'].unique())}")
                print(f"    Available adsorbates: {sorted(self.df_labels['adsorbate'].unique())}")
                
                # Show similar records for debugging
                partial_matches = self.df_labels[
                    (self.df_labels['zeolite'] == self.zeolite_type) &
                    (self.df_labels['environment'] == environment) &
                    (self.df_labels['adsorbate'] == self.adsorbate)
                ]
                if len(partial_matches) > 0:
                    print(f"    Similar records (same zeolite-env-adsorbate):")
                    for _, row in partial_matches.head(3).iterrows():
                        print(f"        Snapshot {row['snapshot']}: intE={row['intE']}")
            return None
        elif len(matching_records) > 1:
            if self.verbose:
                print(f"    ⚠️  Multiple matching records found ({len(matching_records)}), using first one")
            return float(matching_records.iloc[0]['intE'])
        else:
            target_interaction_energy = float(matching_records.iloc[0]['intE'])
            if self.verbose:
                print(f"    ✅ Target interaction energy found: {target_interaction_energy:.6f} (eV)")
            return target_interaction_energy

if __name__ == "__main__":
    
    FEATURE_LIST = [
                'atom_type_C',
                'atom_type_H',
                'atom_type_O',
                'is_hydrophobic',
                'is_donor',
                'is_acceptor',
                'is_hbonded',
                'is_hbonded_donor',
                'is_hbonded_acceptor',
                'atom_mass',
                'partial_charge',
                'valence',
                'LJ_epsilon',
                'LJ_sigma',
                ]
    
    # Create input variables dictionary
    input_vars = {
        'zeolite_type':         'FAU',          # Zeolite type (e.g. "FAU")
        'solvent_type':         'methanol_240_water_960',   # Solvent type (e.g. "water_pure")
        'pore_type':            'hydrophilic',  # Pore type (e.g. "hydrophilic", "hydrophobic")
        'adsorbate':            '08_01_ethene_glycol',  # Adsorbate type (e.g. "01_methanol", "02_01_02_propanol")
        'snapshot_index':       1,              # Single snapshot index
        'box_grids_size':       16.0,           # Box size in Angstrom (centered on adsorbate)
        'box_increment':        0.8,            # Voxel size in Angstrom
        'feature_list':         FEATURE_LIST,   # List of features to include in the voxel grid
        'include_solvent':      True,           # Whether to include solvent (water) atoms
        'include_zeolite':      False,          # Whether to include zeolite atoms
        'include_adsorbate':    True,           # Whether to include adsorbate atoms
        'verbose':              True,
    }
    
    # Process single snapshot
    generate_voxel_grids = GenerateVoxelGrids(**input_vars)
    target_interaction_energy = generate_voxel_grids.get_target_interaction_energy()
    
    # # Example plotting adsorbate channels
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_C', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_H', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_O', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hydrophobic', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_donor', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_acceptor', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded_donor', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded_acceptor', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_mass', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='partial_charge', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='valence', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='LJ_epsilon', group='adsorbate')
    # generate_voxel_grids.plot_voxel_features_grid(feature='LJ_sigma', group='adsorbate')
    
    
    # # Example plotting solvent channels
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_C', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_H', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_O', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hydrophobic', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_donor', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_acceptor', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded_donor', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded_acceptor', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='atom_mass', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='partial_charge', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='valence', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='LJ_epsilon', group='solvent')
    # generate_voxel_grids.plot_voxel_features_grid(feature='LJ_sigma', group='solvent')

    
    # Print final voxel grid shape and channel mapping
    print(f"\n=== Final Voxel Grid Information ===")
    print(f"    Voxel grid shape: {generate_voxel_grids.voxel_grid.shape}")
    print(f"    Total channels: {generate_voxel_grids.voxel_grid.shape[3]}")
    print(f"    Atomic features: {len(generate_voxel_grids.atomic_features)}")
    print(f"    Expected channels: {2 * len(generate_voxel_grids.atomic_features)} (2 channel groups × {len(generate_voxel_grids.atomic_features)} features)")
    
    print(f"\n    Channel mapping:")
    for channel_name, channel_indices in generate_voxel_grids.feature_channel_mapping.items():
        print(f"        {channel_name}: channel {channel_indices[0]}")
    
    # Test accessing specific channels
    test_voxel_1_adsorbate_atom_type_C = generate_voxel_grids.voxel_grid[:, :, :, 0]
    test_voxel_2_adsorbate_atom_type_H = generate_voxel_grids.voxel_grid[:, :, :, 1]
    test_voxel_3_adsorbate_atom_type_O = generate_voxel_grids.voxel_grid[:, :, :, 2]
    test_voxel_4_adsorbate_is_hydrophobic = generate_voxel_grids.voxel_grid[:, :, :, 3]  # Hydrophobic feature
    test_voxel_5_adsorbate_is_donor = generate_voxel_grids.voxel_grid[:, :, :, 4]        # New H-bond donor potential
    test_voxel_6_adsorbate_is_acceptor = generate_voxel_grids.voxel_grid[:, :, :, 5]     # New H-bond acceptor potential
    test_voxel_7_adsorbate_is_hbonded = generate_voxel_grids.voxel_grid[:, :, :, 6]      # H-bond feature
    test_voxel_8_adsorbate_is_hbonded_donor = generate_voxel_grids.voxel_grid[:, :, :, 7]  # H-bond feature
    test_voxel_9_adsorbate_is_hbonded_acceptor = generate_voxel_grids.voxel_grid[:, :, :, 8] # H-bond feature
    test_voxel_10_adsorbate_atom_mass = generate_voxel_grids.voxel_grid[:, :, :, 9]
    test_voxel_11_adsorbate_partial_charge = generate_voxel_grids.voxel_grid[:, :, :, 10]
    test_voxel_12_adsorbate_valence = generate_voxel_grids.voxel_grid[:, :, :, 11]          # Valence feature
    test_voxel_13_adsorbate_lj_epsilon = generate_voxel_grids.voxel_grid[:, :, :, 12]      # LJ epsilon feature
    test_voxel_14_adsorbate_lj_sigma = generate_voxel_grids.voxel_grid[:, :, :, 13]        # LJ sigma feature
    
    # Solvent channels (channels 14-27)
    test_voxel_15_solvent_atom_type_C = generate_voxel_grids.voxel_grid[:, :, :, 14]
    test_voxel_16_solvent_atom_type_H = generate_voxel_grids.voxel_grid[:, :, :, 15]
    test_voxel_17_solvent_atom_type_O = generate_voxel_grids.voxel_grid[:, :, :, 16]
    test_voxel_18_solvent_is_hydrophobic = generate_voxel_grids.voxel_grid[:, :, :, 17]     # Hydrophobic feature
    test_voxel_19_solvent_is_donor = generate_voxel_grids.voxel_grid[:, :, :, 18]           # H-bond donor potential
    test_voxel_20_solvent_is_acceptor = generate_voxel_grids.voxel_grid[:, :, :, 19]        # H-bond acceptor potential
    test_voxel_21_solvent_is_hbonded = generate_voxel_grids.voxel_grid[:, :, :, 20]         # H-bond feature
    test_voxel_22_solvent_is_hbonded_donor = generate_voxel_grids.voxel_grid[:, :, :, 21]   # H-bond donor feature
    test_voxel_23_solvent_is_hbonded_acceptor = generate_voxel_grids.voxel_grid[:, :, :, 22] # H-bond acceptor feature
    test_voxel_24_solvent_atom_mass = generate_voxel_grids.voxel_grid[:, :, :, 23]
    test_voxel_25_solvent_partial_charge = generate_voxel_grids.voxel_grid[:, :, :, 24]
    test_voxel_26_solvent_valence = generate_voxel_grids.voxel_grid[:, :, :, 25]            # Valence feature
    test_voxel_27_solvent_lj_epsilon = generate_voxel_grids.voxel_grid[:, :, :, 26]         # LJ epsilon feature
    test_voxel_28_solvent_lj_sigma = generate_voxel_grids.voxel_grid[:, :, :, 27]           # LJ sigma feature
    
    # Print channel statistics to verify the data
    print(f"\n=== Channel Data Verification ===")
    print(f"Adsorbate channels (0-13):")
    for i in range(14):
        channel_data = generate_voxel_grids.voxel_grid[:, :, :, i]
        non_zero_count = np.count_nonzero(channel_data)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        print(f"    Channel {i:2d}: {non_zero_count:4d} non-zero voxels, range [{min_val:8.3f}, {max_val:8.3f}]")
    
    print(f"\nSolvent channels (14-27):")
    for i in range(14, 28):
        channel_data = generate_voxel_grids.voxel_grid[:, :, :, i]
        non_zero_count = np.count_nonzero(channel_data)
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        print(f"    Channel {i:2d}: {non_zero_count:4d} non-zero voxels, range [{min_val:8.3f}, {max_val:8.3f}]")
    
    
    



