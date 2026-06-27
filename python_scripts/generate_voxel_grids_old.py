# -*- coding: utf-8 -*-
"""
generate_voxel_grids.py
    The purpose of this code is to generate voxel-based 3D grid representations from MD trajectory snapshots.
    This enables the conversion of molecular simulation data into structured formats suitable for 
    machine learning approaches, particularly 3D convolutional neural networks.

Main Class:
    GenerateVoxelGrids: Creates 3D voxel grids with atomic features for a single MD snapshot

Key Features:
    - Voxel-based spatial representation centered on adsorbate molecule
    - Multi-channel feature encoding (molecular types, atomic properties, hydrogen bonds, etc.)
    - Support for mixed solvent systems (water + methanol)
    - Feature normalization and saturation for ML training
    - Van der Waals saturation using PCMax algorithm for discrete features

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
from read_md_snapshot_mdanalysis import snapshotMDAnalysis
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

        # Define feature categories for consistent naming throughout the class
        self.feature_categories = {
            # Discrete features that use binary/one-hot encoding and should be saturated
            'discrete': ['mol_type',
                         'atom_type_C',
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
        atom_type_features = [f for f in self.feature_list if f.startswith('atom_type_')]
        if atom_type_features:
            self.extract_element_types()
        
        
        # Extract hydrophobic information if hydrophobic feature is requested
        if 'is_hydrophobic' in self.feature_list:
            if verbose:
                print(f"\n--- Extracting Hydrophobic Information ---")
            self.atom_id_to_is_hydrophobic = extract_is_hydrophobic_info(
                self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                verbose=verbose
            )
        
        
        # Extract hydrogen bond potential (donor/acceptor capacity) if features are requested
        donor_acceptor_features = ['is_donor', 'is_acceptor']
        if any(feature in self.feature_list for feature in donor_acceptor_features):
            if verbose:
                print(f"\n--- Extracting Hydrogen Bond Potential Information ---")
            
            # Use extract_is_donor_acceptor to get H-bond potential properties
            donor_acceptor_info = extract_is_donor_acceptor(
                zeolite_type=self.zeolite_type,
                solvent_type=self.solvent_type,
                pore_type=self.pore_type,
                adsorbate=self.adsorbate,
                snapshot_index=self.snapshot_index,
                r_cut=5.0,
                verbose=verbose
            )
            
            # Store the atom ID to donor/acceptor status mappings
            self.atom_id_to_is_donor = donor_acceptor_info['atom_id_to_is_donor']
            self.atom_id_to_is_acceptor = donor_acceptor_info['atom_id_to_is_acceptor']
        
                
        # Extract hydrogen bond information if HB features are requested
        hb_features = ['is_hbonded', 'is_hbonded_donor', 'is_hbonded_acceptor']
        if any(feature in self.feature_list for feature in hb_features):
            if verbose:
                print(f"\n--- Extracting Hydrogen Bond Information ---")
            
            # Get H-bond properties
            hbond_detector = HydrogenBondDetector(
                zeolite_type=self.zeolite_type,
                solvent_type=self.solvent_type,
                pore_type=self.pore_type,
                adsorbate=self.adsorbate,
                snapshot_index=self.snapshot_index,
                d_a_cutoff=3.5,
                d_h_cutoff=1.2,
                d_h_a_angle_cutoff=130,
                verbose=verbose
            )
            
            # Get atom ID to H-bond properties mapping directly from the detector
            self.atom_id_to_hbond_props = hbond_detector.atom_hbond_properties

        
        # Extract valence information if valence feature is requested
        if 'valence' in self.feature_list:
            if verbose:
                print(f"\n--- Extracting Valence Information ---")
            self.atom_id_to_valence = extract_total_valence_info(
                self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                verbose=verbose
            )
        
        # Extract LJ parameters if LJ features are requested
        if 'LJ_epsilon' in self.feature_list or 'LJ_sigma' in self.feature_list:
            if verbose:
                print(f"\n--- Extracting LJ Parameters ---")
            
            if 'LJ_epsilon' in self.feature_list:
                self.atom_type_to_epsilon = extract_LJ_parameter_info(
                    self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                    parameter='epsilon', verbose=verbose
                )
            
            if 'LJ_sigma' in self.feature_list:
                self.atom_type_to_sigma = extract_LJ_parameter_info(
                    self.zeolite_type, self.solvent_type, self.pore_type, self.adsorbate,
                    parameter='sigma', verbose=verbose
                )
        
        if verbose:
            discrete_in_list = [f for f in self.feature_categories['discrete'] if f in self.feature_list]
            continuous_in_list = [f for f in self.feature_categories['continuous'] if f in self.feature_list]
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
        
        # Create a copy for feature plotting before normalization and saturation
        # This preserves original feature attribution for accurate visualization
        self.voxel_grid_for_feature_plots = copy.deepcopy(self.voxel_grid)
        
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
            # Include both water and methanol molecules
            water_atoms = universe.select_atoms('resname HOH')  # Water molecules
            methanol_atoms = universe.select_atoms('resname MEO')  # Methanol molecules
            solvent_atoms = water_atoms + methanol_atoms
            selected_atoms.append(solvent_atoms)
            print(f'    Including solvent atoms: {len(water_atoms)} water + {len(methanol_atoms)} methanol = {len(solvent_atoms)} total')
            
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
    
    
    def extract_element_types(self):
        '''
        Extract unique element types from the selected molecules (water and adsorbate) for atom_type features.
        Element names are extracted from atom names like "O_HOH", "H_ADS" by taking the part before underscore.
        '''
        print(f'\n--- Extracting Element Types for Atom Type Features')
        
        universe = self.snapshot_mda.universe
        
        # Select water, methanol, and adsorbate atoms (not zeolite for now)
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
        Create 4D voxel grid with atomic features in the fourth dimension.
        Uses self.feature_list to determine which features to include.
        Uses sum aggregation method for multiple atoms in same voxel.
        
        INPUTS:
            None (uses self.include_zeolite, self.include_solvent, self.include_adsorbate)
        
        OUTPUTS:
            voxel_features_grid: 4D numpy array with shape (max_bin_num, max_bin_num, max_bin_num, num_features)
        '''
        print(f'\n--- Creating Voxel Features Grid for snapshot {self.snapshot_index}')
        print(f'    Features to include: {self.feature_list}')
        
        # Calculate total number of feature channels - each feature gets exactly one channel
        total_feature_channels = len(self.feature_list)
        self.feature_channel_mapping = {}  # Track which channels correspond to which features
        
        for channel_idx, feature_name in enumerate(self.feature_list):
            self.feature_channel_mapping[feature_name] = [channel_idx]
            print(f'    {feature_name}: channel {channel_idx}')
        
        print(f'    Total feature channels: {total_feature_channels}')
        
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
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            if len(water_atoms) > 0:
                atom_groups.append(('water', water_atoms))
                print(f'    Processing {len(water_atoms)} water atoms')
            if len(methanol_atoms) > 0:
                atom_groups.append(('methanol', methanol_atoms))
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
            
            # Fill features for each atom using sum aggregation
            for i, (indices, atom) in enumerate(zip(voxel_indices, atoms_in_box_objects)):
                ix, iy, iz = tuple(indices)
                
                # Extract features for this atom
                features = self.get_atom_features(atom, atom_type, self.feature_list)
                
                # Handle different aggregation for different feature types
                for feature_name in self.feature_list:
                    feat_idx = self.feature_channel_mapping[feature_name][0]
                    
                    if feature_name == 'mol_type':
                        # mol_type is a marker channel: set to +1 for adsorbate, -1 for water or methanol
                        if features[feature_name] > 0:  # adsorbate
                            voxel_features_grid[ix, iy, iz, feat_idx] = 1.0
                        elif features[feature_name] < 0:  # water or methanol
                            voxel_features_grid[ix, iy, iz, feat_idx] = -1.0
                        else:  # zeolite or unknown
                            voxel_features_grid[ix, iy, iz, feat_idx] = 0.0
                    else:
                        # All other features: sum values for atoms in same voxel
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
                ix, iy, iz = tuple(indices)
                total_atoms_per_voxel[ix, iy, iz] += 1
        
        # Print voxel occupancy distribution
        max_occupancy = np.max(total_atoms_per_voxel)
        print(f'\n    Voxel occupancy distribution:')
        for i in range(max_occupancy + 1):
            count = np.sum(total_atoms_per_voxel == i)
            print(f'    --  {i} atoms: {count} voxels')
        
        # Print statistics
        print(f'\n    Voxel features grid statistics:')
        for feature_name in self.feature_list:
            feat_idx = self.feature_channel_mapping[feature_name][0]
            feature_slice = voxel_features_grid[:, :, :, feat_idx]
            non_zero_count = np.count_nonzero(feature_slice)
            min_val = np.min(feature_slice)
            max_val = np.max(feature_slice)
            print(f'    {feature_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}]')
        
        return voxel_features_grid


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
            if feature_name == 'mol_type':
                if atom_type == 'adsorbate':
                    features['mol_type'] = 1.0
                elif atom_type == 'water' or atom_type == 'methanol':
                    # Both water and methanol are treated as -1 for mol_type feature
                    features['mol_type'] = -1.0
                elif atom_type == 'zeolite':
                    features['mol_type'] = 0.0
                else:
                    features['mol_type'] = 0.0
            
            elif feature_name.startswith('atom_type_'):
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
                atom_id = int(atom.id)  # Ensure atom ID is int
                is_donor = self.atom_id_to_is_donor[atom_id]
                features['is_donor'] = 1.0 if is_donor else 0.0
            
            elif feature_name == 'is_acceptor':
                # Get hydrogen bond acceptor potential from atom ID mapping
                atom_id = int(atom.id)  # Ensure atom ID is int
                is_acceptor = self.atom_id_to_is_acceptor[atom_id]
                features['is_acceptor'] = 1.0 if is_acceptor else 0.0
            
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
                mass = atom.mass
                features['atom_mass'] = float(mass)
            
            
            elif feature_name == 'partial_charge':
                # Get partial charge from MDAnalysis atom object
                charge = atom.charge
                features['partial_charge'] = float(charge)
            
            
            elif feature_name == 'valence':
                # Get valence from atom ID mapping
                atom_id = atom.id  # Get atom ID from MDAnalysis atom object
                valence = self.atom_id_to_valence[atom_id]
                features['valence'] = float(valence)

            
            elif feature_name == 'LJ_epsilon':
                # Get LJ epsilon parameter from atom type mapping
                # Get atom type ID from MDAnalysis atom object
                atom_type_id = getattr(atom, 'type', None)
                
                # If type attribute doesn't exist, try to extract from atom name
                if atom_type_id is None:
                    # get type ID from topology if available
                    atom_type_id = atom.universe.atoms[atom.index].type
                
                # Convert to integer if it's a string
                if isinstance(atom_type_id, str):
                    atom_type_id = int(atom_type_id)
                
                epsilon = self.atom_type_to_epsilon[atom_type_id]
                features['LJ_epsilon'] = float(epsilon)
            
            
            elif feature_name == 'LJ_sigma':
                # Get LJ sigma parameter from atom type mapping
                # Get atom type ID from MDAnalysis atom object
                atom_type_id = getattr(atom, 'type', None)
                
                # If type attribute doesn't exist, try to extract from atom name
                if atom_type_id is None:
                    # get type ID from topology if available
                    atom_type_id = atom.universe.atoms[atom.index].type
                
                # Convert to integer if it's a string
                if isinstance(atom_type_id, str):
                    atom_type_id = int(atom_type_id)
                
                sigma = self.atom_type_to_sigma[atom_type_id]
                features['LJ_sigma'] = float(sigma)

            else:
                print("    Warning: Unknown feature '{feature_name}', setting to 0.0")
                raise ValueError(f"Unknown feature '{feature_name}' requested")

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
        Only normalizes voxels that are actually occupied by atoms (identified by non-zero mol_type values).
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
        
        # Get occupied voxels mask from mol_type channel
        mol_type_channel_idx = self.feature_channel_mapping['mol_type'][0]
        occupied_mask = np.abs(voxel_grid[:, :, :, mol_type_channel_idx]) > 0
        
        num_occupied_voxels = np.sum(occupied_mask)
        total_voxels = np.prod(voxel_grid.shape[:3])
        
        print(f'    Total voxels: {total_voxels}')
        print(f'    Occupied voxels: {num_occupied_voxels}')
        print(f'    Empty voxels: {total_voxels - num_occupied_voxels}')
        
        if num_occupied_voxels == 0:
            print(f'    Warning: No occupied voxels found, skipping normalization')
            return voxel_grid
        
        # Use unified feature categories
        discrete_features = [f for f in self.feature_categories['discrete'] if f in self.feature_list]
        continuous_features = [f for f in self.feature_categories['continuous'] if f in self.feature_list]
        
        print(f'\n--- Continuous features to normalize: {continuous_features}')
        print(f'    Discrete features (unchanged): {discrete_features}')
        
        print(f'\n--- Normalizing Continuous Feature Channels, total channels: {len(continuous_features)} ---')
        # Normalize each continuous feature channel separately, only for occupied voxels
        for feature_name in continuous_features:
            if feature_name in self.feature_channel_mapping:
                feature_channels = self.feature_channel_mapping[feature_name]
                
                for channel_idx in feature_channels:
                    # Extract the feature channel
                    feature_slice = voxel_grid[:, :, :, channel_idx]
                    
                    # Get values only from occupied voxels
                    occupied_values = feature_slice[occupied_mask]
                    
                    if len(occupied_values) == 0:
                        print(f'    {feature_name} (channel {channel_idx}): no occupied voxels, skipping')
                        continue
                    
                    # Store original statistics
                    min_val = np.min(occupied_values)
                    max_val = np.max(occupied_values)
                    
                    # Apply selected normalization method using sklearn
                    if normalization_method == 'max_abs':
                        # Max-Abs scaling using sklearn
                        normalized_values = maxabs_scale(occupied_values.reshape(-1, 1)).flatten()
                        norm_range = "[-1, 1]"
                        
                    elif normalization_method == 'standard':
                        # StandardScaler using sklearn
                        scaler = StandardScaler()
                        normalized_values = scaler.fit_transform(occupied_values.reshape(-1, 1)).flatten()
                        norm_range = f"mean≈0, std≈1"
                        
                    else:
                        print(f'    Warning: Unknown normalization method {normalization_method}, using max_abs')
                        normalized_values = maxabs_scale(occupied_values.reshape(-1, 1)).flatten()
                        norm_range = "[-1, 1]"
                    
                    # Create normalized slice (start with copy of original)
                    normalized_slice = feature_slice.copy()
                    
                    # Update only occupied voxels
                    normalized_slice[occupied_mask] = normalized_values
                    
                    # Update the voxel grid
                    voxel_grid[:, :, :, channel_idx] = normalized_slice
                    
                    # Calculate statistics for reporting
                    norm_min = np.min(normalized_values)
                    norm_max = np.max(normalized_values)
                    norm_mean = np.mean(normalized_values)
                    norm_std = np.std(normalized_values)
                    
                    print(f'\n    {feature_name} (channel {channel_idx}): [{min_val:.6f}, {max_val:.6f}] -> [{norm_min:.6f}, {norm_max:.6f}] {norm_range}')
                    print(f'    Normalized stats: mean={norm_mean:.6f}, std={norm_std:.6f} (occupied voxels only)')

        print(f'\n--- Discrete Feature channels remain unchanged, total channels: {len(discrete_features)} ---')
        # Print statistics for discrete features (unchanged)
        for feature_name in discrete_features:
            if feature_name in self.feature_channel_mapping:
                channel_idx = self.feature_channel_mapping[feature_name][0]
                feature_slice = voxel_grid[:, :, :, channel_idx]
                min_val = np.min(feature_slice)
                max_val = np.max(feature_slice)
                non_zero_count = np.count_nonzero(feature_slice)
                occupied_non_zero_count = np.sum(np.abs(feature_slice[occupied_mask]) > 0)
                print(f'    {feature_name} (channel {channel_idx}): range [{min_val:.3f}, {max_val:.3f}], {non_zero_count} non-zero voxels, {occupied_non_zero_count} occupied non-zero voxels (unchanged)')
        
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
        discrete_features = [f for f in self.feature_categories['discrete'] if f in self.feature_list]
        continuous_features = [f for f in self.feature_categories['continuous'] if f in self.feature_list]
        
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
            channel_idx = self.feature_channel_mapping[feature_name][0]
            saturated_grids[channel_idx] = np.zeros(self.bin_array, dtype=float)
        
        # Process each atom type separately
        atom_groups = []
        
        if self.include_solvent:
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            if len(water_atoms) > 0:
                atom_groups.append(('water', water_atoms))
            if len(methanol_atoms) > 0:
                atom_groups.append(('methanol', methanol_atoms))
        
        if self.include_adsorbate:
            adsorbate_atoms = universe.select_atoms('resname ADS')
            if len(adsorbate_atoms) > 0:
                atom_groups.append(('adsorbate', adsorbate_atoms))
        
        if self.include_zeolite:
            zeolite_atoms = universe.select_atoms(f'resname {self.zeolite_type}')
            if len(zeolite_atoms) > 0:
                atom_groups.append(('zeolite', zeolite_atoms))
        
        # Process each atom group
        print(f'\n--- Processing Atom Groups for Saturation ---')
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
                
                # Extract features for this atom
                features = self.get_atom_features(atom, atom_type, discrete_features)
                
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
                                channel_idx = self.feature_channel_mapping[feature_name][0]
                                feature_val = features[feature_name]
                                
                                if feature_name == 'mol_type':
                                    if feature_val > 0:  # adsorbate
                                        saturated_grids[channel_idx][ix, iy, iz] = max(saturated_grids[channel_idx][ix, iy, iz], n)
                                    elif feature_val < 0:  # water
                                        saturated_grids[channel_idx][ix, iy, iz] = min(saturated_grids[channel_idx][ix, iy, iz], -n)
                                    # zeolite stays at 0
                                
                                elif feature_name.startswith('atom_type_'):
                                    # Use PCMax for specific atom type features
                                    if feature_val > 0:  # This element type present
                                        saturated_grids[channel_idx][ix, iy, iz] = max(saturated_grids[channel_idx][ix, iy, iz], n)
                                
                                elif feature_name in ['is_hydrophobic', 'is_donor', 'is_acceptor',
                                                      'is_hbonded', 'is_hbonded_donor', 'is_hbonded_acceptor']:
                                    # Use PCMax for binary features
                                    if feature_val > 0:  # Feature present
                                        saturated_grids[channel_idx][ix, iy, iz] = max(saturated_grids[channel_idx][ix, iy, iz], n)
                                    # If feature_val == 0, don't update (stays at 0)
        

        # Replace discrete feature channels in the voxel grid with saturated versions
        for channel_idx, saturated_grid in saturated_grids.items():
            voxel_grid[:, :, :, channel_idx] = saturated_grid
        
        # Print statistics for features
        if self.verbose:
            # Print statistics for saturated features
            print(f'\n--- Saturated discrete features statistics:')
            for feature_name in discrete_features:
                channel_idx = self.feature_channel_mapping[feature_name][0]
                feature_slice = voxel_grid[:, :, :, channel_idx]
                non_zero_count = np.count_nonzero(feature_slice)
                min_val = np.min(feature_slice)
                max_val = np.max(feature_slice)
                print(f'    {feature_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}] (saturated)')
        
            # Print statistics for continuous features (unchanged)
            print(f'\n--- Continuous features statistics (unchanged):')
            for feature_name in continuous_features:
                channel_idx = self.feature_channel_mapping[feature_name][0]
                feature_slice = voxel_grid[:, :, :, channel_idx]
                non_zero_count = np.count_nonzero(feature_slice)
                min_val = np.min(feature_slice)
                max_val = np.max(feature_slice)
                print(f'    {feature_name}: {non_zero_count} non-zero voxels, range [{min_val:.3f}, {max_val:.3f}] (unchanged)')
        
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

    
    # Keep the visualization methods but update them to work with the new structure
    def plot_voxel_occupancy_grids(self, voxel_grid=None, show_fig=False, show_title=True, save_fig=False, plot_saturate=False, plot_molecules='both', plot_clean=False):
        '''
        Visualization of the voxel grid showing water atoms in blue, methanol atoms in green, and adsorbate atoms in red.
        Uses existing voxel grid and atom position data to distinguish atom types.
        Supports both pure water and water-methanol co-solvent systems.

        INPUTS:
            voxel_grid: numpy array, the voxel grid to visualize (optional, uses self.voxel_occupancy_grid if None)
            show_fig: bool, whether to display the plot (default: False)
            show_title: bool, whether to show the plot title with system information (default: True)
            save_fig: bool, whether to save the figure to file (default: False)
            plot_saturate: bool, whether to apply van der Waals saturation for visualization (default: False)
                          - If True: applies saturate_voxel_features_grid method to create smooth, continuous representation
                          - If False: uses discrete atom positions for sharp voxel representation
                          - Saturation is applied only for visualization, doesn't modify original voxel generation
            plot_molecules: str, which molecules to visualize (default: 'both')
                          - 'both': show both solvent and adsorbate molecules
                          - 'solvent': show only solvent molecules (water and methanol)
                          - 'adsorbate': show only adsorbate molecules
            plot_clean: bool, whether to create a clean plot without axis labels (default: False)
                          - If True: removes axis labels and tick labels only
                          - If False: shows normal axis labels and tick labels

        OUTPUTS:
            fig, ax: matplotlib figure and axis objects
            Displays a 3D plot of the voxel grid with color-coded atoms.
            
        NOTES:
            - When plot_saturate=True, atoms influence surrounding voxels within 2*r_vdw radius
            - Creates more physically realistic representation with van der Waals effects
            - Saved figures include '_saturated' suffix when plot_saturate=True
            - plot_molecules parameter controls which molecule types are visualized
        '''
        if voxel_grid is None:
            voxel_grid = self.voxel_occupancy_grid
        
        # Validate plot_molecules parameter
        valid_options = ['both', 'solvent', 'adsorbate']
        if plot_molecules not in valid_options:
            raise ValueError(f"plot_molecules must be one of {valid_options}, got '{plot_molecules}'")
        
        print(f"\n--- Plotting Voxel Grid for {self.zeolite_type}-{self.solvent_type}-{self.pore_type}-{self.adsorbate} snapshot {self.snapshot_index:02d}")
        print(f"    Visualization mode: {plot_molecules}")
        
        # Apply saturation if requested (only for plotting, doesn't modify original data)
        if plot_saturate:
            print(f"    Applying van der Waals saturation for visualization...")
            # Use saturated feature grid for visualization with mixed colors
            saturated_grid = self.saturate_voxel_features_grid(self.voxel_grid.copy())
            # Extract mol_type channel for visualization
            if 'mol_type' in self.feature_list:
                mol_type_channel_idx = self.feature_channel_mapping['mol_type'][0]
                mol_type_grid = saturated_grid[:, :, :, mol_type_channel_idx]
                
                # Get the universe from the snapshot
                universe = self.snapshot_mda.universe
                
                # Get water, methanol, and adsorbate atoms separately
                water_atoms = universe.select_atoms('resname HOH')
                methanol_atoms = universe.select_atoms('resname MEO')
                adsorbate_atoms = universe.select_atoms('resname ADS')
                
                # Check if we have methanol atoms (for mixed solvent systems)
                has_methanol = len(methanol_atoms) > 0
                has_water = len(water_atoms) > 0
                
                if self.verbose:
                    print(f"    Water atoms found: {len(water_atoms)}")
                    print(f"    Methanol atoms found: {len(methanol_atoms)}")
                    print(f"    Adsorbate atoms found: {len(adsorbate_atoms)}")
                    print(f"    System type: {'Mixed (water+methanol)' if has_methanol and has_water else 'Pure water' if has_water else 'Unknown'}")
                    print(f"    Using mixed color approach for saturated visualization")
                
                # Calculate individual contributions from water and methanol
                water_contribution = np.zeros(self.bin_array, dtype=float)
                methanol_contribution = np.zeros(self.bin_array, dtype=float)
                adsorbate_voxels = (mol_type_grid > 0.01)  # Adsorbate has positive mol_type values
                
                # Get box dimensions for PBC
                box_dimensions = universe.dimensions[:3]
                
                # Calculate water contribution to each voxel
                if has_water and len(water_atoms) > 0 and plot_molecules in ['both', 'solvent']:
                    water_positions = water_atoms.positions
                    water_relative_positions = water_positions - self.COM_adsorbate
                    
                    # Apply minimum image convention
                    for dim in range(3):
                        water_relative_positions[:, dim] = water_relative_positions[:, dim] - box_dimensions[dim] * np.round(water_relative_positions[:, dim] / box_dimensions[dim])
                    
                    # Filter atoms within expanded range to account for saturation
                    max_vdw_radius = 2.1
                    expanded_range = self.grid_half_box_length + 2 * max_vdw_radius
                    within_range_mask = np.all(np.abs(water_relative_positions) <= expanded_range, axis=1)
                    water_in_range = water_relative_positions[within_range_mask]
                    
                    if len(water_in_range) > 0:
                        for atom_pos in water_in_range:
                            # Water van der Waals radius (approximately 1.4 Å)
                            r_vdw = 1.4
                            influence_range = 2 * r_vdw
                            
                            # Find voxel range that this atom can influence
                            min_voxel = np.floor((atom_pos - influence_range + self.grid_half_box_length) / self.box_increment).astype(int)
                            max_voxel = np.ceil((atom_pos + influence_range + self.grid_half_box_length) / self.box_increment).astype(int)
                            
                            # Clip to valid voxel range
                            min_voxel = np.clip(min_voxel, 0, self.max_bin_num)
                            max_voxel = np.clip(max_voxel, 0, self.max_bin_num)
                            
                            # Calculate contribution to each voxel
                            for ix in range(min_voxel[0], max_voxel[0]):
                                for iy in range(min_voxel[1], max_voxel[1]):
                                    for iz in range(min_voxel[2], max_voxel[2]):
                                        # Calculate voxel center position
                                        voxel_center = np.array([ix, iy, iz]) * self.box_increment - self.grid_half_box_length + self.box_increment/2
                                        
                                        # Calculate distance from atom to voxel center
                                        r = np.linalg.norm(atom_pos - voxel_center)
                                        
                                        # Skip if outside influence range or not a solvent voxel
                                        if r > influence_range or mol_type_grid[ix, iy, iz] >= -0.01:
                                            continue
                                        
                                        # Calculate softening factor
                                        if r < 1e-6:
                                            n = 1.0
                                        else:
                                            n = 1 - np.exp(-(r_vdw / r)**12)
                                        
                                        # Use PCMax - keep maximum contribution
                                        water_contribution[ix, iy, iz] = max(water_contribution[ix, iy, iz], n)
                
                # Calculate methanol contribution to each voxel
                if has_methanol and len(methanol_atoms) > 0 and plot_molecules in ['both', 'solvent']:
                    methanol_positions = methanol_atoms.positions
                    methanol_relative_positions = methanol_positions - self.COM_adsorbate
                    
                    # Apply minimum image convention
                    for dim in range(3):
                        methanol_relative_positions[:, dim] = methanol_relative_positions[:, dim] - box_dimensions[dim] * np.round(methanol_relative_positions[:, dim] / box_dimensions[dim])
                    
                    # Filter atoms within expanded range to account for saturation
                    within_range_mask = np.all(np.abs(methanol_relative_positions) <= expanded_range, axis=1)
                    methanol_in_range = methanol_relative_positions[within_range_mask]
                    
                    if len(methanol_in_range) > 0:
                        for atom_pos in methanol_in_range:
                            # Methanol van der Waals radius (approximately 1.7 Å)
                            r_vdw = 1.7
                            influence_range = 2 * r_vdw
                            
                            # Find voxel range that this atom can influence
                            min_voxel = np.floor((atom_pos - influence_range + self.grid_half_box_length) / self.box_increment).astype(int)
                            max_voxel = np.ceil((atom_pos + influence_range + self.grid_half_box_length) / self.box_increment).astype(int)
                            
                            # Clip to valid voxel range
                            min_voxel = np.clip(min_voxel, 0, self.max_bin_num)
                            max_voxel = np.clip(max_voxel, 0, self.max_bin_num)
                            
                            # Calculate contribution to each voxel
                            for ix in range(min_voxel[0], max_voxel[0]):
                                for iy in range(min_voxel[1], max_voxel[1]):
                                    for iz in range(min_voxel[2], max_voxel[2]):
                                        # Calculate voxel center position
                                        voxel_center = np.array([ix, iy, iz]) * self.box_increment - self.grid_half_box_length + self.box_increment/2
                                        
                                        # Calculate distance from atom to voxel center
                                        r = np.linalg.norm(atom_pos - voxel_center)
                                        
                                        # Skip if outside influence range or not a solvent voxel
                                        if r > influence_range or mol_type_grid[ix, iy, iz] >= -0.01:
                                            continue
                                        
                                        # Calculate softening factor
                                        if r < 1e-6:
                                            n = 1.0
                                        else:
                                            n = 1 - np.exp(-(r_vdw / r)**12)
                                        
                                        # Use PCMax - keep maximum contribution
                                        methanol_contribution[ix, iy, iz] = max(methanol_contribution[ix, iy, iz], n)
                
                # Create smooth color transition voxels based on contributions
                # Multiple transition levels for smoother color gradation
                pure_water_voxels = np.zeros(self.bin_array, dtype=bool)
                pure_methanol_voxels = np.zeros(self.bin_array, dtype=bool)
                
                # More granular transition levels
                water_rich_voxels = np.zeros(self.bin_array, dtype=bool)      # 75-90% water
                water_dominant_voxels = np.zeros(self.bin_array, dtype=bool)  # 60-75% water  
                balanced_voxels = np.zeros(self.bin_array, dtype=bool)        # 40-60% either
                methanol_dominant_voxels = np.zeros(self.bin_array, dtype=bool) # 60-75% methanol
                methanol_rich_voxels = np.zeros(self.bin_array, dtype=bool)     # 75-90% methanol
                
                # Smoothed color mixing parameters
                min_threshold = 0.05  # Lower threshold for detection
                
                for ix in range(self.bin_array[0]):
                    for iy in range(self.bin_array[1]):
                        for iz in range(self.bin_array[2]):
                            if mol_type_grid[ix, iy, iz] < -0.01:  # Solvent voxel
                                water_contrib = water_contribution[ix, iy, iz]
                                methanol_contrib = methanol_contribution[ix, iy, iz]
                                
                                # Calculate total contribution
                                total_contrib = water_contrib + methanol_contrib
                                
                                if total_contrib > min_threshold:
                                    # Calculate contribution ratios
                                    water_ratio = water_contrib / total_contrib
                                    methanol_ratio = methanol_contrib / total_contrib
                                    
                                    # Smooth classification based on ratios
                                    if water_ratio >= 0.95:  # 95%+ water
                                        pure_water_voxels[ix, iy, iz] = True
                                    elif water_ratio >= 0.75:  # 75-95% water
                                        water_rich_voxels[ix, iy, iz] = True
                                    elif water_ratio >= 0.60:  # 60-75% water
                                        water_dominant_voxels[ix, iy, iz] = True
                                    elif water_ratio >= 0.40:  # 40-60% (balanced)
                                        balanced_voxels[ix, iy, iz] = True
                                    elif water_ratio >= 0.25:  # 25-40% water (60-75% methanol)
                                        methanol_dominant_voxels[ix, iy, iz] = True
                                    elif water_ratio >= 0.05:  # 5-25% water (75-95% methanol)
                                        methanol_rich_voxels[ix, iy, iz] = True
                                    else:  # <5% water (95%+ methanol)
                                        pure_methanol_voxels[ix, iy, iz] = True
                                elif water_contrib > min_threshold:
                                    pure_water_voxels[ix, iy, iz] = True
                                elif methanol_contrib > min_threshold:
                                    pure_methanol_voxels[ix, iy, iz] = True
                
                # Store the voxel arrays for plotting
                water_voxels = pure_water_voxels
                methanol_voxels = pure_methanol_voxels
                
                if self.verbose:
                    pure_water_count = np.sum(pure_water_voxels)
                    water_rich_count = np.sum(water_rich_voxels)
                    water_dom_count = np.sum(water_dominant_voxels)
                    balanced_count = np.sum(balanced_voxels)
                    methanol_dom_count = np.sum(methanol_dominant_voxels)
                    methanol_rich_count = np.sum(methanol_rich_voxels)
                    pure_methanol_count = np.sum(pure_methanol_voxels)
                    total_solvent = np.sum(mol_type_grid < -0.01)
                    
                    print(f"    Smooth transition voxel classification:")
                    print(f"      Pure water (95%+): {pure_water_count}")
                    print(f"      Water-rich (75-95%): {water_rich_count}")
                    print(f"      Water-dominant (60-75%): {water_dom_count}")
                    print(f"      Balanced (40-60%): {balanced_count}")
                    print(f"      Methanol-dominant (60-75%): {methanol_dom_count}")
                    print(f"      Methanol-rich (75-95%): {methanol_rich_count}")
                    print(f"      Pure methanol (95%+): {pure_methanol_count}")
                    print(f"      Total solvent voxels: {total_solvent}")
                    total_classified = pure_water_count + water_rich_count + water_dom_count + balanced_count + methanol_dom_count + methanol_rich_count + pure_methanol_count
                    print(f"      Classified: {total_classified}")
                
                # Store additional voxel types for plotting
                smooth_water_rich_voxels = water_rich_voxels
                smooth_water_dominant_voxels = water_dominant_voxels
                smooth_balanced_voxels = balanced_voxels
                smooth_methanol_dominant_voxels = methanol_dominant_voxels
                smooth_methanol_rich_voxels = methanol_rich_voxels
            else:
                print(f"    Warning: mol_type feature not found, falling back to original method")
                plot_saturate = False
        
        if not plot_saturate:
            # Use original method based on atom positions
            # Get the universe from the snapshot
            universe = self.snapshot_mda.universe
            
            # Get water, methanol, and adsorbate atoms separately
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            adsorbate_atoms = universe.select_atoms('resname ADS')
            
            # Check if we have methanol atoms (for mixed solvent systems)
            has_methanol = len(methanol_atoms) > 0
            has_water = len(water_atoms) > 0
            
            if self.verbose:
                print(f"    Water atoms found: {len(water_atoms)}")
                print(f"    Methanol atoms found: {len(methanol_atoms)}")
                print(f"    Adsorbate atoms found: {len(adsorbate_atoms)}")
                print(f"    System type: {'Mixed (water+methanol)' if has_methanol and has_water else 'Pure water' if has_water else 'Unknown'}")
            
            # Initialize separate grids for each molecule type
            water_voxels = np.zeros(self.bin_array, dtype=bool)
            methanol_voxels = np.zeros(self.bin_array, dtype=bool)
            adsorbate_voxels = np.zeros(self.bin_array, dtype=bool)
            
            # Get box dimensions for PBC
            box_dimensions = universe.dimensions[:3]
            
            # Process water atoms
            if len(water_atoms) > 0 and plot_molecules in ['both', 'solvent']:
                water_positions = water_atoms.positions
                water_relative_positions = water_positions - self.COM_adsorbate
                
                # Apply minimum image convention
                for dim in range(3):
                    water_relative_positions[:, dim] = water_relative_positions[:, dim] - box_dimensions[dim] * np.round(water_relative_positions[:, dim] / box_dimensions[dim])
                
                # Filter atoms within box and convert to voxel indices
                within_box_mask = np.all(np.abs(water_relative_positions) <= self.grid_half_box_length, axis=1)
                water_in_box = water_relative_positions[within_box_mask]
                
                if len(water_in_box) > 0:
                    shifted_positions = water_in_box + self.grid_half_box_length
                    voxel_indices = np.floor(shifted_positions / self.box_increment).astype(int)
                    voxel_indices = np.clip(voxel_indices, 0, self.max_bin_num - 1)
                    
                    for indices in voxel_indices:
                        water_voxels[tuple(indices)] = True
            
            # Process methanol atoms
            if len(methanol_atoms) > 0 and plot_molecules in ['both', 'solvent']:
                methanol_positions = methanol_atoms.positions
                methanol_relative_positions = methanol_positions - self.COM_adsorbate
                
                # Apply minimum image convention
                for dim in range(3):
                    methanol_relative_positions[:, dim] = methanol_relative_positions[:, dim] - box_dimensions[dim] * np.round(methanol_relative_positions[:, dim] / box_dimensions[dim])
                
                # Filter atoms within box and convert to voxel indices
                within_box_mask = np.all(np.abs(methanol_relative_positions) <= self.grid_half_box_length, axis=1)
                methanol_in_box = methanol_relative_positions[within_box_mask]
                
                if len(methanol_in_box) > 0:
                    shifted_positions = methanol_in_box + self.grid_half_box_length
                    voxel_indices = np.floor(shifted_positions / self.box_increment).astype(int)
                    voxel_indices = np.clip(voxel_indices, 0, self.max_bin_num - 1)
                    
                    for indices in voxel_indices:
                        methanol_voxels[tuple(indices)] = True
            
            # Process adsorbate atoms
            if len(adsorbate_atoms) > 0 and plot_molecules in ['both', 'adsorbate']:
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
        
        # Coordinates for plotting - use consistent bin_array shape
        voxel_shape = self.bin_array
        x, y, z = np.indices(np.array(voxel_shape) + 1)
        
        # Set up the figure and axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create title with system information
        if show_title and not plot_clean:
            title_line1 = f'Voxel Grid - {self.zeolite_type} {self.pore_type.capitalize()}'
            title_line2 = f'Solvent: {self.solvent_type}, Adsorbate: {self.adsorbate}, Snapshot: {self.snapshot_index}'
            title_line3 = f'Visualization: {plot_molecules.capitalize()}'
            
            ax.set_title(f'{title_line1}\n{title_line2}\n{title_line3}', fontsize=11)

        # Calculate aspect ratios based on the shape of the voxel grid
        shape = np.array(voxel_shape)
        aspect_ratio = shape / shape.max()
        
        # Plot water atoms in blue with appropriate transparency
        if np.any(water_voxels) and plot_molecules in ['both', 'solvent']:
            water_count = np.sum(water_voxels)
            # Make water even more transparent - set lower alpha values
            water_alpha = max(0.15, min(0.4, 600.0 / max(water_count, 100)))
            ax.voxels(x, y, z, water_voxels, facecolors='blue', alpha=water_alpha, edgecolor='none')
            if self.verbose:
                print(f"    Plotting {water_count} pure water voxels with alpha={water_alpha:.2f}")
        
        # Plot methanol atoms in green with appropriate transparency
        if np.any(methanol_voxels) and plot_molecules in ['both', 'solvent']:
            methanol_count = np.sum(methanol_voxels)
            # Keep methanol transparency similar since it looks good
            methanol_alpha = max(0.3, min(0.6, 500.0 / max(methanol_count, 50)))
            ax.voxels(x, y, z, methanol_voxels, facecolors='green', alpha=methanol_alpha, edgecolor='none')
            if self.verbose:
                print(f"    Plotting {methanol_count} pure methanol voxels with alpha={methanol_alpha:.2f}")
        
        # Plot smooth transition voxels if we're in saturated mode
        if plot_saturate and 'mol_type' in self.feature_list and plot_molecules in ['both', 'solvent']:
            # Plot water-rich voxels in a slightly lighter blue
            if 'smooth_water_rich_voxels' in locals() and np.any(smooth_water_rich_voxels):
                water_rich_count = np.sum(smooth_water_rich_voxels)
                ax.voxels(x, y, z, smooth_water_rich_voxels, facecolors='cornflowerblue', alpha=0.35, edgecolor='none')
                if self.verbose:
                    print(f"    Plotting {water_rich_count} water-rich voxels (75-95% water)")
            
            # Plot water-dominant voxels in light blue
            if 'smooth_water_dominant_voxels' in locals() and np.any(smooth_water_dominant_voxels):
                water_dom_count = np.sum(smooth_water_dominant_voxels)
                ax.voxels(x, y, z, smooth_water_dominant_voxels, facecolors='lightblue', alpha=0.4, edgecolor='none')
                if self.verbose:
                    print(f"    Plotting {water_dom_count} water-dominant voxels (60-75% water)")
            
            # Plot balanced voxels in cyan-turquoise
            if 'smooth_balanced_voxels' in locals() and np.any(smooth_balanced_voxels):
                balanced_count = np.sum(smooth_balanced_voxels)
                ax.voxels(x, y, z, smooth_balanced_voxels, facecolors='turquoise', alpha=0.5, edgecolor='none')
                if self.verbose:
                    print(f"    Plotting {balanced_count} balanced voxels (40-60% each)")
            
            # Plot methanol-dominant voxels in light green
            if 'smooth_methanol_dominant_voxels' in locals() and np.any(smooth_methanol_dominant_voxels):
                methanol_dom_count = np.sum(smooth_methanol_dominant_voxels)
                ax.voxels(x, y, z, smooth_methanol_dominant_voxels, facecolors='lightgreen', alpha=0.45, edgecolor='none')
                if self.verbose:
                    print(f"    Plotting {methanol_dom_count} methanol-dominant voxels (60-75% methanol)")
            
            # Plot methanol-rich voxels in a slightly lighter green
            if 'smooth_methanol_rich_voxels' in locals() and np.any(smooth_methanol_rich_voxels):
                methanol_rich_count = np.sum(smooth_methanol_rich_voxels)
                ax.voxels(x, y, z, smooth_methanol_rich_voxels, facecolors='lightseagreen', alpha=0.4, edgecolor='none')
                if self.verbose:
                    print(f"    Plotting {methanol_rich_count} methanol-rich voxels (75-95% methanol)")
        
        # Plot adsorbate atoms in red (keep more opaque as it's the focus)
        if np.any(adsorbate_voxels) and plot_molecules in ['both', 'adsorbate']:
            adsorbate_count = np.sum(adsorbate_voxels)
            ax.voxels(x, y, z, adsorbate_voxels, facecolors='red', alpha=0.9, edgecolor='none')
            if self.verbose:
                print(f"    Plotting {adsorbate_count} adsorbate voxels with alpha=0.9")

        # Set the aspect of the plot to match the voxel dimensions
        ax.set_box_aspect(aspect_ratio)
        
        # Set consistent axis limits regardless of which molecules are displayed
        # This ensures the spatial scale remains consistent
        ax.set_xlim(0, voxel_shape[0])
        ax.set_ylim(0, voxel_shape[1])
        ax.set_zlim(0, voxel_shape[2])
        
        # Configure axis labels and appearance based on plot_clean parameter
        if not plot_clean:
            # Normal plot with axis labels and tick labels
            ax.set_xlabel('X dimension (voxels)')
            ax.set_ylabel('Y dimension (voxels)')
            ax.set_zlabel('Z dimension (voxels)')
        else:
            # Clean plot: only remove axis labels and tick labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            
            # Remove tick labels only (keep tick marks and grid)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
        # Add information about displayed molecules in verbose mode
        if self.verbose:
            total_voxels = np.prod(voxel_shape)
            if plot_molecules == 'both':
                water_count = np.sum(water_voxels) if np.any(water_voxels) else 0
                methanol_count = np.sum(methanol_voxels) if np.any(methanol_voxels) else 0
                adsorbate_count = np.sum(adsorbate_voxels) if np.any(adsorbate_voxels) else 0
                print(f"    Total voxels in grid: {total_voxels}")
                print(f"    Water voxels: {water_count} ({water_count/total_voxels*100:.2f}%)")
                print(f"    Methanol voxels: {methanol_count} ({methanol_count/total_voxels*100:.2f}%)")
                print(f"    Adsorbate voxels: {adsorbate_count} ({adsorbate_count/total_voxels*100:.2f}%)")
            elif plot_molecules == 'solvent':
                water_count = np.sum(water_voxels) if np.any(water_voxels) else 0
                methanol_count = np.sum(methanol_voxels) if np.any(methanol_voxels) else 0
                solvent_count = water_count + methanol_count
                print(f"    Total voxels in grid: {total_voxels}")
                print(f"    Displayed solvent voxels: {solvent_count} ({solvent_count/total_voxels*100:.2f}%)")
            elif plot_molecules == 'adsorbate':
                adsorbate_count = np.sum(adsorbate_voxels) if np.any(adsorbate_voxels) else 0
                print(f"    Total voxels in grid: {total_voxels}")
                print(f"    Displayed adsorbate voxels: {adsorbate_count} ({adsorbate_count/total_voxels*100:.2f}%)")
                print(f"    Note: Adsorbate typically occupies only a small fraction of the total volume")
        
        # Legend removed as requested

        if show_fig:
            plt.show()

        if save_fig:
            save_folder = os.path.join(get_paths('output_figure_path'), 'voxel_grids')
            saturate_suffix = '_saturated' if plot_saturate else ''
            molecule_suffix = f'_{plot_molecules}' if plot_molecules != 'both' else ''
            fig_filename = f'01_voxel_occupancy_{self.zeolite_type}_{self.solvent_type}_{self.pore_type}_{self.adsorbate}_snapshot{self.snapshot_index:02d}{saturate_suffix}{molecule_suffix}.png'
            fig.savefig(os.path.join(save_folder, fig_filename), dpi=1000, bbox_inches='tight')
            print(f"    Figure saved as {fig_filename}")
        
        return fig, ax


    def plot_voxel_features_grid(self, feature='mol_type', element_type=None, value_threshold=1e-6, legend=False, show_fig=False, show_title=True, save_fig=False, plot_clean=False):
        '''
        Visualization of a specific feature channel from the voxel features grid.
        Similar format to plot_voxel_occupancy_grids but for feature visualization.
        Adsorbate voxels shown in red, water voxels in blue, methanol voxels in green, with intensity based on feature values.
        Supports both pure water and water-methanol mixed solvent systems.
        Now shows empty plots for features with no non-zero values.
        
        Uses voxel_grid_for_feature_plots by default for accurate molecule-feature attribution,
        avoiding issues caused by PCMax saturation where features can spread to neighboring voxels.

        INPUTS:
            feature: str, the feature name to visualize (e.g., 'mol_type', 'atom_mass', 'atom_type_C')
            element_type: str, deprecated parameter (kept for compatibility)
            value_threshold: float, minimum absolute value to consider as non-zero for visualization
            legend: bool, whether to show the legend
            show_fig: bool, whether to display the plot (default: False)
            show_title: bool, whether to show the plot title with system information (default: True)
            save_fig: bool, whether to save the figure to file (default: False)
            plot_clean: bool, whether to create a clean plot without axis labels (default: False)
                          - If True: removes axis labels and tick labels only
                          - If False: shows normal axis labels and tick labels
            
        OUTPUTS:
            fig, ax: matplotlib figure and axis objects
            Displays a 3D plot of the specified feature channel.
        '''
        print(f"\n--- Plotting Voxel Features Grid for {self.zeolite_type}-{self.solvent_type}-{self.pore_type}-{self.adsorbate} snapshot {self.snapshot_index:02d}")
        print(f"    Feature: {feature}")
        print(f"    Value threshold: {value_threshold}")
        
        # Check if feature exists in feature list
        if feature not in self.feature_list:
            print(f"    Error: Feature '{feature}' not found in feature list")
            print(f"    Available features: {self.feature_list}")
            return None, None
        
        # Get the channel index
        channel_idx = self.feature_channel_mapping[feature][0]
        feature_name = feature
        
        if element_type is not None:
            print(f"    Warning: element_type parameter is deprecated, feature name should include element (e.g., 'atom_type_C')")
        
        print(f"    Channel index: {channel_idx}")
        
        # Extract the feature channel from selected grid
        feature_grid = self.voxel_grid_for_feature_plots[:, :, :, channel_idx]
        print(f"    Using original grid (voxel_grid_for_feature_plots) for accurate visualization")
        
        # Get the universe from the snapshot to identify different molecule types
        universe = self.snapshot_mda.universe
        
        # Get water, methanol, and adsorbate atoms separately
        water_atoms = universe.select_atoms('resname HOH')
        methanol_atoms = universe.select_atoms('resname MEO')
        adsorbate_atoms = universe.select_atoms('resname ADS')
        
        # Check if we have methanol atoms (for mixed solvent systems)
        has_methanol = len(methanol_atoms) > 0
        has_water = len(water_atoms) > 0
        
        if self.verbose:
            print(f"    Water atoms found: {len(water_atoms)}")
            print(f"    Methanol atoms found: {len(methanol_atoms)}")
            print(f"    Adsorbate atoms found: {len(adsorbate_atoms)}")
            print(f"    System type: {'Mixed (water+methanol)' if has_methanol and has_water else 'Pure water' if has_water else 'Unknown'}")
        
        # Show values above threshold (using absolute value for features that might have negative values)
        if feature == 'mol_type':
            # For mol_type, show all non-zero values (both +1 and -1)
            feature_voxels = np.abs(feature_grid) > value_threshold
        else:
            # For other features, use absolute value to catch both positive and negative values
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
            print(f"    Note: No voxels found above threshold {value_threshold} for feature '{feature_name}'.")
            print(f"    Creating empty plot to show feature channel structure.")
            # Set feature_voxels to False everywhere for empty plot
            feature_voxels = np.zeros_like(feature_grid, dtype=bool)
        
        # Get voxel indices where feature values are above threshold
        voxel_indices = np.where(feature_voxels)
        
        # Initialize separate grids for each molecule type
        water_voxels = np.zeros(self.bin_array, dtype=bool)
        methanol_voxels = np.zeros(self.bin_array, dtype=bool)
        adsorbate_voxels = np.zeros(self.bin_array, dtype=bool)
        water_alphas = np.zeros(self.bin_array, dtype=float)
        methanol_alphas = np.zeros(self.bin_array, dtype=float)
        adsorbate_alphas = np.zeros(self.bin_array, dtype=float)
        
        # Get box dimensions for PBC
        box_dimensions = universe.dimensions[:3]
        
        # Create voxel-to-molecule mapping by checking which atoms are in each voxel
        voxel_molecule_map = {}  # (i, j, k) -> 'water', 'methanol', or 'adsorbate'
        
        # Process each atom type separately to create voxel-to-molecule mapping
        atom_groups = [
            ('water', water_atoms),
            ('methanol', methanol_atoms), 
            ('adsorbate', adsorbate_atoms)
        ]
        
        for atom_type, atoms in atom_groups:
            if len(atoms) == 0:
                continue
                
            # Get positions and apply periodic boundary conditions
            positions = atoms.positions
            relative_positions = positions - self.COM_adsorbate
            
            # Apply minimum image convention
            for dim in range(3):
                relative_positions[:, dim] = relative_positions[:, dim] - box_dimensions[dim] * np.round(relative_positions[:, dim] / box_dimensions[dim])
            
            # Filter atoms within box and convert to voxel indices
            within_box_mask = np.all(np.abs(relative_positions) <= self.grid_half_box_length, axis=1)
            atoms_in_box = relative_positions[within_box_mask]
            
            if len(atoms_in_box) > 0:
                shifted_positions = atoms_in_box + self.grid_half_box_length
                voxel_indices_atoms = np.floor(shifted_positions / self.box_increment).astype(int)
                voxel_indices_atoms = np.clip(voxel_indices_atoms, 0, self.max_bin_num - 1)
                
                # Map each voxel to its molecule type
                for indices in voxel_indices_atoms:
                    voxel_key = tuple(indices)
                    # If voxel already has a molecule, prioritize adsorbate > methanol > water
                    if voxel_key not in voxel_molecule_map:
                        voxel_molecule_map[voxel_key] = atom_type
                    elif atom_type == 'adsorbate':
                        voxel_molecule_map[voxel_key] = atom_type
                    elif atom_type == 'methanol' and voxel_molecule_map[voxel_key] == 'water':
                        voxel_molecule_map[voxel_key] = atom_type
        
        # Only process if there are non-zero voxels
        if non_zero_count > 0:
            for i, j, k in zip(voxel_indices[0], voxel_indices[1], voxel_indices[2]):
                feature_val = feature_grid[i, j, k]
                voxel_key = (i, j, k)
                
                # Determine molecule type from the mapping we created
                molecule_type = voxel_molecule_map.get(voxel_key, None)
                
                # Skip voxels that don't correspond to any atoms (shouldn't happen for feature values > threshold)
                if molecule_type is None:
                    continue
                
                # Calculate alpha based on normalized feature value
                if feature == 'mol_type':
                    # For mol_type, use fixed alpha
                    alpha = 0.7
                else:
                    # For other features, normalize by absolute maximum for better visualization
                    if max_val > 0 or min_val < 0:
                        # Use absolute maximum for normalization to handle both positive and negative values
                        max_abs_val = max(abs(max_val), abs(min_val))
                        normalized_val = abs(feature_val) / max_abs_val
                    else:
                        normalized_val = 1.0
                    # Set alpha range from 0.3 to 0.9 for better visibility
                    alpha = 0.3 + 0.6 * normalized_val
                
                # Assign to appropriate molecule type
                if molecule_type == 'adsorbate':
                    adsorbate_voxels[i, j, k] = True
                    adsorbate_alphas[i, j, k] = alpha
                elif molecule_type == 'water':
                    water_voxels[i, j, k] = True
                    water_alphas[i, j, k] = alpha
                elif molecule_type == 'methanol':
                    methanol_voxels[i, j, k] = True
                    methanol_alphas[i, j, k] = alpha
        
        # Create coordinates for the FULL grid (not just occupied voxels)
        # This ensures consistent axis ranges for all plots
        x = np.arange(self.max_bin_num + 1)
        y = np.arange(self.max_bin_num + 1)
        z = np.arange(self.max_bin_num + 1)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Set up the figure and axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create title with system information
        if show_title:
            if non_zero_count == 0:
                title_line1 = f'Voxel Features Grid - {self.zeolite_type} {self.pore_type.capitalize()}'
                title_line2 = f'Solvent: {self.solvent_type}, Adsorbate: {self.adsorbate}, Snapshot: {self.snapshot_index}'
                title_line3 = f'Feature: {feature_name} (EMPTY - No values above threshold)'
                title = f'{title_line1}\n{title_line2}\n{title_line3}'
            else:
                title_line1 = f'Voxel Features Grid - {self.zeolite_type} {self.pore_type.capitalize()}'
                title_line2 = f'Solvent: {self.solvent_type}, Adsorbate: {self.adsorbate}, Snapshot: {self.snapshot_index}'
                title_line3 = f'Feature: {feature_name}'
                title = f'{title_line1}\n{title_line2}\n{title_line3}'
            
            ax.set_title(title, fontsize=11)
        
        # Plot adsorbate voxels in red (if any)
        if np.any(adsorbate_voxels):
            if feature == 'mol_type':
                ax.voxels(X, Y, Z, adsorbate_voxels, facecolors='red', alpha=0.8, edgecolor='none')
            else:
                # Create color array for adsorbate with varying alpha
                adsorbate_colors = np.zeros(adsorbate_voxels.shape + (4,), dtype=float)
                adsorbate_indices = np.where(adsorbate_voxels)
                for i, j, k in zip(adsorbate_indices[0], adsorbate_indices[1], adsorbate_indices[2]):
                    alpha = adsorbate_alphas[i, j, k]
                    adsorbate_colors[i, j, k] = [1.0, 0.0, 0.0, alpha]  # Red with varying alpha
                ax.voxels(X, Y, Z, adsorbate_voxels, facecolors=adsorbate_colors, edgecolor='none')
        
        # Plot water voxels in blue (if any)
        if np.any(water_voxels):
            if feature == 'mol_type':
                ax.voxels(X, Y, Z, water_voxels, facecolors='blue', alpha=0.7, edgecolor='none')
            else:
                # Create color array for water with varying alpha
                water_colors = np.zeros(water_voxels.shape + (4,), dtype=float)
                water_indices = np.where(water_voxels)
                for i, j, k in zip(water_indices[0], water_indices[1], water_indices[2]):
                    alpha = water_alphas[i, j, k]
                    water_colors[i, j, k] = [0.0, 0.0, 1.0, alpha]  # Blue with varying alpha
                ax.voxels(X, Y, Z, water_voxels, facecolors=water_colors, edgecolor='none')
        
        # Plot methanol voxels in green (if any)
        if np.any(methanol_voxels):
            if feature == 'mol_type':
                ax.voxels(X, Y, Z, methanol_voxels, facecolors='green', alpha=0.7, edgecolor='none')
            else:
                # Create color array for methanol with varying alpha
                methanol_colors = np.zeros(methanol_voxels.shape + (4,), dtype=float)
                methanol_indices = np.where(methanol_voxels)
                for i, j, k in zip(methanol_indices[0], methanol_indices[1], methanol_indices[2]):
                    alpha = methanol_alphas[i, j, k]
                    methanol_colors[i, j, k] = [0.0, 1.0, 0.0, alpha]  # Green with varying alpha
                ax.voxels(X, Y, Z, methanol_voxels, facecolors=methanol_colors, edgecolor='none')
        
        # For empty plots, add a subtle grid outline to show the voxel space
        if non_zero_count == 0:
            # Draw wire frame to show the voxel grid boundaries
            ax.plot([0, self.max_bin_num], [0, 0], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 0], [0, self.max_bin_num], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            ax.plot([0, 0], [0, 0], [0, self.max_bin_num], 'k-', alpha=0.2, linewidth=0.5)
            
            # Add text annotation to explain the empty plot
            ax.text2D(0.5, 0.5, f'No {feature_name} features detected\nin this voxel grid', 
                     transform=ax.transAxes, ha='center', va='center',
                     fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # FORCE consistent axis limits for ALL plots
        ax.set_xlim(0, self.max_bin_num)
        ax.set_ylim(0, self.max_bin_num)
        ax.set_zlim(0, self.max_bin_num)
        
        # Set equal aspect ratio to ensure cubic voxels
        ax.set_box_aspect([1, 1, 1])
        
        # Configure axis labels and appearance based on plot_clean parameter
        if not plot_clean:
            # Normal plot with axis labels and tick labels
            ax.set_xlabel('X dimension (voxels)', fontsize=10)
            ax.set_ylabel('Y dimension (voxels)', fontsize=10)
            ax.set_zlabel('Z dimension (voxels)', fontsize=10)
        else:
            # Clean plot: only remove axis labels and tick labels
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            
            # Remove tick labels only (keep tick marks and grid)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        
        if legend:
            # Add dynamic legend based on what's present in the system
            from matplotlib.patches import Patch
            legend_elements = []
            
            # Add legend entries based on what atoms are actually present
            if np.any(adsorbate_voxels):
                legend_elements.append(Patch(facecolor='red', alpha=0.8, label='Adsorbate'))
            
            if np.any(water_voxels):
                legend_elements.append(Patch(facecolor='blue', alpha=0.7, label='Water'))
                
            if np.any(methanol_voxels):
                legend_elements.append(Patch(facecolor='green', alpha=0.7, label='Methanol'))
            
            if non_zero_count == 0:
                legend_elements.append(Patch(facecolor='lightgray', alpha=0.8, label='Empty feature'))
            
            # Only show legend if there are elements to show
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        plt.tight_layout()

        if show_fig:
            plt.show()

        if save_fig:
            save_folder = os.path.join(get_paths('output_figure_path'), 'voxel_grids')
            feature_suffix = f'_{feature}' if feature != 'mol_type' else ''
            clean_suffix = '_clean' if plot_clean else ''
            fig_filename = f'02_voxel_features_{self.zeolite_type}_{self.solvent_type}_{self.pore_type}_{self.adsorbate}_snapshot{self.snapshot_index:02d}{feature_suffix}{clean_suffix}.png'
            fig.savefig(os.path.join(save_folder, fig_filename), dpi=1000, bbox_inches='tight')
            print(f"    Figure saved as {fig_filename}")
        
        return fig, ax


if __name__ == "__main__":
    
    # Create input variables dictionary for testing co-solvent system
    input_vars = {
        'zeolite_type':         'FAU',                      # Zeolite type (e.g. "FAU")
        'solvent_type':         'methanol_120_water_1080',   # water_pure methanol_240_water_960
        'pore_type':            'hydrophilic',              # Pore type (e.g. "hydrophilic", "hydrophobic")
        'adsorbate':            '02_01_02_propanol',        # Adsorbate type (e.g. "01_methanol", "02_01_02_propanol")
        'snapshot_index':       4,                          # Single snapshot index
        'box_grids_size':       20.0,                       # Box size in Angstrom (centered on adsorbate)
        'box_increment':        1.0,                        # Voxel size in Angstrom
        'feature_list':         FEATURE_LIST,               # List of features to include in the voxel grid
        'include_solvent':      True,                       # Whether to include solvent (water + methanol) atoms
        'include_zeolite':      False,                      # Whether to include zeolite atoms
        'include_adsorbate':    True,                       # Whether to include adsorbate atoms
        'verbose':              True,
    }
    
    # Process single snapshot
    generate_voxel_grids = GenerateVoxelGrids(**input_vars)
    target_interaction_energy = generate_voxel_grids.get_target_interaction_energy()

    # Visualize the voxel occupancy grid
    # print("\n=== Plotting voxel occupancy grid ===")
    generate_voxel_grids.plot_voxel_occupancy_grids(plot_saturate=False,
                                                    show_fig=False,
                                                    show_title=False,
                                                    save_fig=True,
                                                    plot_clean=True,
                                                    plot_molecules='both')
    # generate_voxel_grids.plot_voxel_occupancy_grids(plot_saturate=True,
    #                                                 show_fig=False,
    #                                                 show_title=False,
    #                                                 save_fig=True,
    #                                                 plot_clean=True,
    #                                                 plot_molecules='solvent')
    # generate_voxel_grids.plot_voxel_occupancy_grids(plot_saturate=True,
    #                                                 show_fig=False,
    #                                                 show_title=False,
    #                                                 save_fig=True,
    #                                                 plot_clean=True,
    #                                                 plot_molecules='adsorbate')


    # print("\n=== Plotting mol_type feature ===")
    # generate_voxel_grids.plot_voxel_features_grid(feature='mol_type', legend=True)
    
    # print("\n=== Plotting C atom distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_C', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)

    # print("\n=== Plotting H atom distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_H', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting O atom distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='atom_type_O', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting hydrophobic atoms ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='is_hydrophobic', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting H-bond donor potential ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='is_donor', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting H-bond acceptor potential ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='is_acceptor', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting H-bonded atoms ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting H-bond donors ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded_donor', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting H-bond acceptors ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='is_hbonded_acceptor', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting atom mass distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='atom_mass', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting partial charge distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='partial_charge', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting valence distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='valence', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting LJ epsilon distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='LJ_epsilon', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)
    
    # print("\n=== Plotting LJ sigma distribution ===")
    generate_voxel_grids.plot_voxel_features_grid(feature='LJ_sigma', legend=False, show_fig=False, show_title=False, save_fig=True, plot_clean=True)

    # Test the voxel grid with consistent channel indices
    test_voxel_0_mol_type = generate_voxel_grids.voxel_grid[:, :, :, 0]
    test_voxel_1_atom_type_C = generate_voxel_grids.voxel_grid[:, :, :, 1]
    test_voxel_2_atom_type_H = generate_voxel_grids.voxel_grid[:, :, :, 2]
    test_voxel_3_atom_type_O = generate_voxel_grids.voxel_grid[:, :, :, 3]
    test_voxel_4_is_hydrophobic = generate_voxel_grids.voxel_grid[:, :, :, 4]  # Hydrophobic feature
    test_voxel_5_is_donor = generate_voxel_grids.voxel_grid[:, :, :, 5]        # New H-bond donor potential
    test_voxel_6_is_acceptor = generate_voxel_grids.voxel_grid[:, :, :, 6]     # New H-bond acceptor potential
    test_voxel_7_is_hbonded = generate_voxel_grids.voxel_grid[:, :, :, 7]      # H-bond feature
    test_voxel_8_is_hbonded_donor = generate_voxel_grids.voxel_grid[:, :, :, 8]  # H-bond feature
    test_voxel_9_is_hbonded_acceptor = generate_voxel_grids.voxel_grid[:, :, :, 9] # H-bond feature
    test_voxel_10_atom_mass = generate_voxel_grids.voxel_grid[:, :, :, 10]
    test_voxel_11_partial_charge = generate_voxel_grids.voxel_grid[:, :, :, 11]
    test_voxel_12_valence = generate_voxel_grids.voxel_grid[:, :, :, 12]          # Valence feature
    test_voxel_13_lj_epsilon = generate_voxel_grids.voxel_grid[:, :, :, 13]      # LJ epsilon feature
    test_voxel_14_lj_sigma = generate_voxel_grids.voxel_grid[:, :, :, 14]        # LJ sigma feature



