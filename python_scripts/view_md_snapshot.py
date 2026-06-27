# -*- coding: utf-8 -*-
"""
This script is used to read and process MD configurations using the ASE library.
It loads a trajectory from a lammpstrj file and a topology from a lammpsdata file.
The trajectory is stored as a list of ASE Atoms objects, each representing a frame in the trajectory.
The element symbols of the atoms in the system are updated based on the specified adsorbate name.
"""

## Importing Modules
import os
import numpy as np
from ase.io import read
from ase.io import write
from ase.visualize import view
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

## Custom Functions
from core.path import get_paths


# Class Function To Generate Grid Interpolation Array Per Frame
class UniverseASE:
    """
    Load a single MD snapshot from LAMMPS data and dump files using ASE.
    """

    def __init__(self,
                 zeolite_type: str = 'FAU',
                 solvent_type: str = 'water_pure',
                 pore_type: str = 'hydrophilic',
                 adsorbate: str = '01_methanol',
                 snapshot_index: int = 1,
                 verbose: bool = False,
                 ):
        
        # Store basic settings
        self.verbose = verbose
        self.zeolite_type = zeolite_type
        self.solvent_type = solvent_type
        self.pore_type = pore_type
        self.adsorbate = adsorbate
        self.snapshot_index = snapshot_index
        
        # Parse solvent composition
        if self.verbose:
            print(f"\n--- Loading ASE object.")
            print(f"    Zeolite: {self.zeolite_type}")
            print(f"    Solvent: {self.solvent_type}")
            print(f"    Pore: {self.pore_type}")
            print(f"    Adsorbate: {self.adsorbate}")
            print(f"    Snapshot index: {self.snapshot_index}")
        
        self._parse_solvent_composition()
        
        # Build simulation directory path
        root = get_paths('simulation_path')
        folder = f"{self.solvent_type}-{self.pore_type}"
        self.sim_dir = os.path.join(root, self.zeolite_type, folder, self.adsorbate)
        
        # Define LAMMPS data file path
        self.path_lammpsdata = os.path.join(self.sim_dir,
                                            'data_nvt_samp_new.lammpsdata')

        # Load the specified snapshot
        self.ase_atoms = self.load_ase_atoms(self.snapshot_index) 
    
    
    def _parse_solvent_composition(self):
        """
        Parse solvent type and determine molecule ID ranges for different components.
        """
        print(f"\n--- Parsing solvent composition: {self.solvent_type}")
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
    
    
    
    # Function to Load Trajectories
    def load_ase_atoms(self, index: int):
        
        # Load topology from lammps data file
        self.atoms_topology = read(self.path_lammpsdata, format='lammps-data')
        if self.verbose:
            print(f'--- Loading topology for {self.adsorbate} from {self.path_lammpsdata}')
        
        # # Test: Print available arrays in the topology
        # print("Available arrays:", self.atoms_topology.arrays.keys())
        # self.atoms_numbers = self.atoms_topology.arrays['numbers']
        # self.atoms_positions = self.atoms_topology.arrays['positions']
        # self.atoms_masses = self.atoms_topology.arrays['masses']
        # self.atoms_id = self.atoms_topology.arrays['id']
        # self.atoms_type = self.atoms_topology.arrays['type']
        # self.atoms_mol_id = self.atoms_topology.arrays['mol-id']
        # self.atoms_initial_charges = self.atoms_topology.arrays['initial_charges']
        # self.atoms_mmcharges = self.atoms_topology.arrays['mmcharges']
        
        
        # Construct dump path
        sub = f"intE{index:02d}"
        traj_name = f"intE{index:02d}.traj"
        self.path_dump = os.path.join(self.sim_dir, sub, traj_name)

        # Read dump file (single frame)
        dump_atoms = read(self.path_dump, format='lammps-dump-text')
        if self.verbose:
            print(f'--- Loading snapshot for {self.adsorbate} from {self.path_dump}')
        
        # Create new Atoms by copying topology
        snapshot = self.atoms_topology.copy()
        
        # Assign coordinates from dump
        snapshot.set_positions(dump_atoms.get_positions())
        
        # Print system information if verbose
        if self.verbose:
            self._print_system_info(snapshot)
        
        return snapshot
    
    
    def _print_system_info(self, atoms):
        """
        Print detailed system information including atom counts, box size, etc.
        """
        print(f"\n=== System Information ===")
        
        # Basic counts
        total_atoms = len(atoms)
        print(f"    Total atoms: {total_atoms}")
        
        # Box dimensions
        cell = atoms.get_cell()
        if hasattr(cell, 'lengths'):
            lengths = cell.lengths()
            angles = cell.angles()
            print(f"    Box dimensions: {lengths[0]:.3f} x {lengths[1]:.3f} x {lengths[2]:.3f} Å")
            print(f"    Box angles: {angles[0]:.1f}° x {angles[1]:.1f}° x {angles[2]:.1f}°")
            volume = cell.volume
            print(f"    Box volume: {volume:.3f} Å³")
        
        # Component analysis
        if 'mol-id' in atoms.arrays:
            mol_ids = atoms.arrays['mol-id']
            unique_mol_ids = np.unique(mol_ids)
            
            # Count water molecules
            water_mask = ((mol_ids >= self.water_mol_range[0]) & (mol_ids <= self.water_mol_range[1]))
            water_atoms = np.sum(water_mask)
            water_molecules = len([mid for mid in unique_mol_ids if self.water_mol_range[0] <= mid <= self.water_mol_range[1]])
            print(f"    Water: {water_molecules} molecules ({water_atoms} atoms)")
            
            # Count methanol molecules (if present)
            if self.has_methanol:
                methanol_mask = ((mol_ids >= self.methanol_mol_range[0]) & (mol_ids <= self.methanol_mol_range[1]))
                methanol_atoms = np.sum(methanol_mask)
                methanol_molecules = len([mid for mid in unique_mol_ids if self.methanol_mol_range[0] <= mid <= self.methanol_mol_range[1]])
                print(f"    Methanol: {methanol_molecules} molecules ({methanol_atoms} atoms)")
            else:
                print(f"    Methanol: 0 molecules (0 atoms)")
            
            # Count zeolite atoms
            zeolite_mask = (mol_ids == self.zeolite_mol_id)
            zeolite_atoms = np.sum(zeolite_mask)
            print(f"    Zeolite: 1 framework ({zeolite_atoms} atoms)")
            
            # Count adsorbate atoms
            adsorbate_mask = (mol_ids == self.adsorbate_mol_id)
            adsorbate_atoms = np.sum(adsorbate_mask)
            print(f"    Adsorbate: 1 molecule ({adsorbate_atoms} atoms)")
        
    def view_system(self,
                    components: str = 'all',
                    center_on_adsorbate: bool = True,
                    box_size: float = 20.0,
                    view_raw: bool = False,
                    save_image: bool = False,
                    image_filename: str = None,
                    image_format: str = 'png',
                    rotation: str = None,
                    scale: float = None,
                    ):
        """
        Visualize the molecular system with options to show different components.
        Properly handles periodic boundary conditions and sets viewing box boundaries.
        
        Parameters:
        -----------
        components : str
            - 'all': Show adsorbate + water + methanol + zeolite (default)
            - 'adsorbate_solvent': Show only adsorbate + water + methanol
            - 'adsorbate_zeolite': Show only adsorbate + zeolite framework
            - 'adsorbate_water': Show only adsorbate + water  
            - 'adsorbate': Show only adsorbate
            - 'solvent': Show only water + methanol
            - 'water': Show only water
            - 'methanol': Show only methanol (if present)
            - 'zeolite': Show only zeolite
        center_on_adsorbate : bool
            If True, center the view on adsorbate center of mass
        box_size : float
            Size of the cubic viewing box in Angstroms
        view_raw : bool
            If True, view the entire raw system without any transformations, cropping, or component filtering (default: False)
        save_image : bool
            If True, save the visualization as an image file (default: False)
        image_filename : str
            Filename for saved image. If None, auto-generate based on system parameters
        image_format : str
            Image format: 'png', 'pov', 'eps', 'svg' (default: 'png')
        rotation : str
            Camera rotation angle. Examples: '0x,0y,0z', '45x,0y,0z', '0x,45y,0z', etc.
        scale : float
            View scale factor. Larger values = zoom out (farther view, thicker atom borders), smaller values = zoom in (closer view)
        """
        if not hasattr(self.ase_atoms, 'arrays') or 'mol-id' not in self.ase_atoms.arrays:
            if self.verbose:
                print("Warning: mol-id not available, showing all atoms")
            view(self.ase_atoms)
            return
        
        # If view_raw is True, just view the entire raw system directly
        # If view_raw is True, adjust z-box and center adsorbate COM in xyz
        if view_raw:
            mol_ids = self.ase_atoms.arrays['mol-id']
            
            # Create a copy for view modifications (don't modify original atoms)
            viewing_atoms = self.ase_atoms.copy()
            positions = viewing_atoms.get_positions()
            cell = viewing_atoms.get_cell()
            
            # Step 1: Find atom distribution and adjust box size
            min_z = np.min(positions[:, 2])
            max_z = np.max(positions[:, 2])
            if self.verbose:
                print(f"Original box z-dimension: {cell[2, 2]:.3f} Å")
                print(f"Atom z range: {min_z:.3f} - {max_z:.3f} Å")
            
            # Step 2: Adjust box - set bottom boundary slightly below lowest atom, top just above highest
            new_cell = cell.copy()
            bottom_boundary = 0.4  # Set bottom at 0.3 Å as suggested
            top_boundary = max_z # Add 1 Å buffer above highest atom
            new_cell[2, 2] = top_boundary
            
            # Step 3: Find adsorbate COM in all three dimensions
            adsorbate_mask = mol_ids == self.adsorbate_mol_id
            if np.any(adsorbate_mask):
                adsorbate_indices = np.where(adsorbate_mask)[0]
                adsorbate_com = np.mean(positions[adsorbate_indices], axis=0)  # COM in x, y, z
                
                # Calculate target position (center of the box in all dimensions)
                box_center = np.array([new_cell[0, 0], new_cell[1, 1], new_cell[2, 2]]) / 2.0
                
                # Calculate translation needed in all dimensions
                translation = box_center - adsorbate_com
                
                if self.verbose:
                    print(f"New box boundaries: {bottom_boundary:.3f} - {top_boundary:.3f} Å")
                    print(f"New box dimensions: {new_cell[0, 0]:.3f} x {new_cell[1, 1]:.3f} x {new_cell[2, 2]:.3f} Å")
                    print(f"Box center: ({box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}) Å")
                    print(f"Adsorbate COM: ({adsorbate_com[0]:.3f}, {adsorbate_com[1]:.3f}, {adsorbate_com[2]:.3f}) Å")
                    print(f"Translation needed: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) Å")
                
                # Step 4: Apply translation to all atoms in all dimensions
                positions += translation
                
                # Step 5: Apply periodic boundary conditions for all axes
                # Wrap atoms that went outside the box back into the box
                for i in range(3):
                    positions[:, i] = positions[:, i] % new_cell[i, i]
                
                # Update positions and cell
                viewing_atoms.set_positions(positions)
                viewing_atoms.set_cell(new_cell)
                
                # Step 6: Further crop to zeolite framework z-boundaries
                # Find zeolite atoms and their z-range
                zeolite_mask = mol_ids == self.zeolite_mol_id
                if np.any(zeolite_mask):
                    zeolite_positions = viewing_atoms.get_positions()[zeolite_mask]
                    zeolite_min_z = np.min(zeolite_positions[:, 2])
                    zeolite_max_z = np.max(zeolite_positions[:, 2])
                    
                    # Extend boundaries by 10 Å above and below zeolite framework
                    buffer_size = 1.0  # Å
                    extended_min_z = zeolite_min_z - buffer_size
                    extended_max_z = zeolite_max_z + buffer_size
                    
                    if self.verbose:
                        print(f"Zeolite framework z-range: {zeolite_min_z:.3f} - {zeolite_max_z:.3f} Å")
                        print(f"Extended z-range (±{buffer_size:.1f}Å): {extended_min_z:.3f} - {extended_max_z:.3f} Å")
                    
                    # Filter atoms to keep only those within extended zeolite z-boundaries
                    all_positions = viewing_atoms.get_positions()
                    z_filter_mask = ((all_positions[:, 2] >= extended_min_z) & 
                                    (all_positions[:, 2] <= extended_max_z))
                    
                    # Apply the filter
                    viewing_atoms = viewing_atoms[z_filter_mask]
                    
                    # Adjust box to extended boundaries
                    final_cell = new_cell.copy()
                    final_cell[2, 2] = extended_max_z - extended_min_z
                    
                    # Translate all atoms so extended region starts at z=0
                    final_positions = viewing_atoms.get_positions()
                    final_positions[:, 2] -= extended_min_z
                    viewing_atoms.set_positions(final_positions)
                    viewing_atoms.set_cell(final_cell)
                    
                    if self.verbose:
                        filtered_count = len(viewing_atoms)
                        final_z_range = [np.min(viewing_atoms.get_positions()[:, 2]), 
                                        np.max(viewing_atoms.get_positions()[:, 2])]
                        final_adsorbate_indices = np.where(viewing_atoms.arrays['mol-id'] == self.adsorbate_mol_id)[0]
                        if len(final_adsorbate_indices) > 0:
                            final_adsorbate_com = viewing_atoms[final_adsorbate_indices].get_center_of_mass()
                            print(f"Final adsorbate COM: ({final_adsorbate_com[0]:.3f}, {final_adsorbate_com[1]:.3f}, {final_adsorbate_com[2]:.3f}) Å")
                        print(f"Final zeolite-cropped z range: {final_z_range[0]:.3f} - {final_z_range[1]:.3f} Å")
                        print(f"Final box z-dimension: {final_cell[2, 2]:.3f} Å")
                        print(f"Atoms kept after zeolite cropping: {filtered_count}/{len(self.ase_atoms)}")
                else:
                    if self.verbose:
                        print("Warning: No zeolite atoms found for cropping")
            else:
                if self.verbose:
                    print("Warning: No adsorbate atoms found for centering")
                    print(f"Viewing raw system: {len(viewing_atoms)} atoms (box adjusted only)")
            
            # Handle image saving for raw view
            if save_image:
                # Generate filename if not provided
                if image_filename is None:
                    image_filename = f"{self.zeolite_type}_{self.solvent_type}_{self.pore_type}_{self.adsorbate}_snap{self.snapshot_index:02d}_raw.{image_format}"
                    
                # Create output directory if it doesn't exist
                output_dir = os.path.join(get_paths('output_figure_path'), "voxel_grids", "ase")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                image_path = os.path.join(output_dir, image_filename)
                
                if self.verbose:
                    print(f"Saving raw view image to: {image_path}")
                
                try:
                    # Use matplotlib and plot_atoms for better control
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Set up plot_atoms parameters
                    plot_kwargs = {
                        'radii': 0.6,  # Fixed smaller radii for raw view (more atoms)
                        'colors': None,  # Use default colors
                    }
                    
                    # Add rotation if specified (format: '45x,30y,0z')
                    if rotation is not None:
                        plot_kwargs['rotation'] = rotation
                    else:
                        plot_kwargs['rotation'] = '0x,0y,0z'  # Default rotation
                    
                    if self.verbose:
                        print(f"plot_kwargs: {plot_kwargs}")
                    
                    # Plot the atoms
                    plot_atoms(viewing_atoms, ax, **plot_kwargs)
                    
                    # Remove all visual elements for clean image
                    ax.set_aspect('equal')
                    ax.axis('off')  # Remove axes, ticks, and labels
                    
                    # Control "zoom level" by setting axis limits (like camera distance)
                    if scale is not None:
                        # Get current axis limits
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        
                        # Calculate center points
                        x_center = (xlim[0] + xlim[1]) / 2
                        y_center = (ylim[0] + ylim[1]) / 2
                        
                        # Calculate current ranges
                        x_range = xlim[1] - xlim[0]
                        y_range = ylim[1] - ylim[0]
                        
                        # Scale the view range (larger scale = zoom out = farther view)
                        new_x_range = x_range * scale
                        new_y_range = y_range * scale
                        
                        # Set new limits centered on the same point
                        ax.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
                        ax.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
                        
                        if self.verbose:
                            print(f"Applied scale {scale}: view range scaled to {new_x_range:.2f} x {new_y_range:.2f}")
                    
                    # Save with high DPI for better quality, no padding
                    if image_format.lower() == 'png':
                        fig.savefig(image_path, dpi=1000, bbox_inches='tight', pad_inches=0,
                                   facecolor='white', edgecolor='none')
                    elif image_format.lower() == 'svg':
                        fig.savefig(image_path, format='svg', bbox_inches='tight', pad_inches=0,
                                   facecolor='white', edgecolor='none')
                    elif image_format.lower() == 'eps':
                        fig.savefig(image_path, format='eps', bbox_inches='tight', pad_inches=0,
                                   facecolor='white', edgecolor='none')
                    else:
                        fig.savefig(image_path, bbox_inches='tight', pad_inches=0,
                                   facecolor='white', edgecolor='none')
                    
                    plt.close(fig)  # Clean up memory
                    
                    if self.verbose:
                        print(f"Raw view image saved successfully: {image_path}")
                        if rotation:
                            print(f"Applied rotation: {rotation}")
                        if scale:
                            print(f"Applied scale: {scale}")
                            
                except Exception as e:
                    if self.verbose:
                        print(f"Error saving raw view image: {e}")
                        print("Falling back to view() only")
                    view(viewing_atoms)
            else:
                view(viewing_atoms)
            return
        
        mol_ids = self.ase_atoms.arrays['mol-id']
        cell = self.ase_atoms.get_cell()
        
        # Create a copy for unwrapping coordinates
        atoms_unwrapped = self.ase_atoms.copy()
        
        # Unwrap adsorbate coordinates using minimum image convention
        if center_on_adsorbate:
            # Find adsorbate atoms (mol_id 1202)
            adsorbate_mask = mol_ids == 1202
            if not np.any(adsorbate_mask):
                if self.verbose:
                    print("Warning: No adsorbate atoms found, using geometric center")
                center = atoms_unwrapped.get_center_of_mass()
            else:
                # Get adsorbate positions and unwrap them
                adsorbate_indices = np.where(adsorbate_mask)[0]
                positions = atoms_unwrapped.get_positions()
                
                # Use first adsorbate atom as reference
                ref_pos = positions[adsorbate_indices[0]]
                
                # Unwrap other adsorbate atoms relative to the reference
                for i in adsorbate_indices[1:]:
                    delta = positions[i] - ref_pos
                    # Apply minimum image convention
                    delta_wrapped = delta - np.round(delta @ np.linalg.inv(cell)) @ cell
                    positions[i] = ref_pos + delta_wrapped
                
                atoms_unwrapped.set_positions(positions)
                adsorbate_atoms = atoms_unwrapped[adsorbate_mask]
                center = adsorbate_atoms.get_center_of_mass()
                
                if self.verbose:
                    print(f"Adsorbate COM (unwrapped): {center}")
        else:
            center = atoms_unwrapped.get_center_of_mass()
        
        # Now create viewing atoms by wrapping around the center
        viewing_atoms = atoms_unwrapped.copy()
        positions = viewing_atoms.get_positions()
        
        # Translate so adsorbate COM is near box center
        box_center = np.diag(cell) / 2
        translation = box_center - center
        positions += translation
        
        # Wrap all coordinates back into the simulation box
        positions = positions - np.floor(positions @ np.linalg.inv(cell)) @ cell
        viewing_atoms.set_positions(positions)
        
        # Recalculate center after wrapping
        if center_on_adsorbate:
            adsorbate_atoms_wrapped = viewing_atoms[adsorbate_mask]
            center_wrapped = adsorbate_atoms_wrapped.get_center_of_mass()
        else:
            center_wrapped = viewing_atoms.get_center_of_mass()
        
        # Define viewing box around the wrapped center
        half_box = box_size / 2.0
        box_min = center_wrapped - half_box
        box_max = center_wrapped + half_box
        
        # Filter atoms within the viewing box (considering PBC)
        positions_wrapped = viewing_atoms.get_positions()
        
        # For PBC, we need to consider atoms that might be just outside the box
        # but within the viewing region when wrapped
        in_box_mask = np.zeros(len(viewing_atoms), dtype=bool)
        
        for i in range(len(viewing_atoms)):
            pos = positions_wrapped[i]
            # Check if atom is in box, considering periodic images
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        image_pos = pos + dx * cell[0] + dy * cell[1] + dz * cell[2]
                        if np.all((image_pos >= box_min) & (image_pos <= box_max)):
                            in_box_mask[i] = True
                            # Use the image that's closest to center
                            if np.linalg.norm(image_pos - center_wrapped) < np.linalg.norm(positions_wrapped[i] - center_wrapped):
                                positions_wrapped[i] = image_pos
                            break
                if in_box_mask[i]:
                    break
                if in_box_mask[i]:
                    break
        
        # Update positions with the best periodic images
        viewing_atoms.set_positions(positions_wrapped)
        
        # Apply component filter
        if components == 'all':
            component_mask = np.ones(len(viewing_atoms), dtype=bool)
        
        elif components == 'adsorbate_solvent':
            # Show adsorbate + all solvent molecules (water + methanol)
            water_mask = ((mol_ids >= self.water_mol_range[0]) & (mol_ids <= self.water_mol_range[1]))
            if self.has_methanol:
                methanol_mask = ((mol_ids >= self.methanol_mol_range[0]) & (mol_ids <= self.methanol_mol_range[1]))
                solvent_mask = water_mask | methanol_mask
            else:
                solvent_mask = water_mask
            component_mask = solvent_mask | (mol_ids == self.adsorbate_mol_id)
        
        elif components == 'adsorbate_zeolite':
            # Show adsorbate + zeolite framework
            component_mask = (mol_ids == self.adsorbate_mol_id) | (mol_ids == self.zeolite_mol_id)
        
        elif components == 'adsorbate':
            component_mask = mol_ids == self.adsorbate_mol_id
        
        elif components == 'solvent':
            # Show all solvent molecules (water + methanol)
            water_mask = ((mol_ids >= self.water_mol_range[0]) & (mol_ids <= self.water_mol_range[1]))
            if self.has_methanol:
                methanol_mask = ((mol_ids >= self.methanol_mol_range[0]) & (mol_ids <= self.methanol_mol_range[1]))
                component_mask = water_mask | methanol_mask
            else:
                component_mask = water_mask
        
        elif components == 'zeolite':
            component_mask = mol_ids == self.zeolite_mol_id
        
        else:
            raise ValueError(f"Invalid components option: {components}. "
                           "Choose from 'all', 'adsorbate_solvent', 'adsorbate_zeolite', 'adsorbate_water', 'adsorbate', "
                           "'solvent', 'water', 'methanol', 'zeolite'")
        
        # Combine both masks
        final_mask = in_box_mask & component_mask
        
        if not np.any(final_mask):
            if self.verbose:
                print(f"Warning: No {components} atoms found in {box_size}Å box around adsorbate COM")
            # Show only atoms that are in the box, regardless of component type
            atoms_to_view = viewing_atoms[in_box_mask] if np.any(in_box_mask) else viewing_atoms[[]]
        else:
            atoms_to_view = viewing_atoms[final_mask]
        
        # Set the viewing box as the new cell boundaries
        # Translate coordinates to have the viewing box centered at origin
        final_positions = atoms_to_view.get_positions()
        final_positions -= center_wrapped
        atoms_to_view.set_positions(final_positions)
        
        # Create a cubic cell centered at origin
        new_cell = np.eye(3) * box_size
        atoms_to_view.set_cell(new_cell)
        atoms_to_view.set_pbc([True, True, True])
        
        # Translate to have box start at origin instead of centered
        final_positions += box_size / 2
        atoms_to_view.set_positions(final_positions)
        
        if self.verbose:
            print(f"Viewing {components}: {len(atoms_to_view)} atoms in {box_size}Å³ box")
            if np.any(mol_ids[final_mask] == 1202):
                final_center = atoms_to_view[mol_ids[final_mask] == 1202].get_center_of_mass()
                print(f"Final adsorbate COM in view: {final_center}")
            print(f"New cell dimensions: {box_size} x {box_size} x {box_size} Å")
        
        # Handle image saving
        if save_image:
            # Generate filename if not provided
            if image_filename is None:
                image_filename = f"{self.zeolite_type}_{self.solvent_type}_{self.pore_type}_{self.adsorbate}_snap{self.snapshot_index:02d}_{components}_box{box_size:.0f}A.{image_format}"
                
            # Create output directory if it doesn't exist
            output_dir = os.path.join(get_paths('output_figure_path'), "voxel_grids", "ase")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            image_path = os.path.join(output_dir, image_filename)
            
            if self.verbose:
                print(f"Saving image to: {image_path}")
            
            try:
                # Use matplotlib and plot_atoms for better control
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Set up plot_atoms parameters
                plot_kwargs = {
                    'radii': 0.8,  # Fixed atom size
                    'colors': None,  # Use default colors
                }
                
                # Add rotation if specified (format: '45x,30y,0z')
                if rotation is not None:
                    plot_kwargs['rotation'] = rotation
                else:
                    plot_kwargs['rotation'] = '0x,0y,0z'  # Default rotation
                
                if self.verbose:
                    print(f"plot_kwargs: {plot_kwargs}")
                
                # Plot the atoms
                plot_atoms(atoms_to_view, ax, **plot_kwargs)
                
                # Remove all visual elements for clean image
                ax.set_aspect('equal')
                ax.axis('off')  # Remove axes, ticks, and labels
                
                # Control "zoom level" by setting axis limits (like camera distance)
                if scale is not None:
                    # Get current axis limits
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    
                    # Calculate center points
                    x_center = (xlim[0] + xlim[1]) / 2
                    y_center = (ylim[0] + ylim[1]) / 2
                    
                    # Calculate current ranges
                    x_range = xlim[1] - xlim[0]
                    y_range = ylim[1] - ylim[0]
                    
                    # Scale the view range (larger scale = zoom out = farther view)
                    new_x_range = x_range * scale
                    new_y_range = y_range * scale
                    
                    # Set new limits centered on the same point
                    ax.set_xlim(x_center - new_x_range/2, x_center + new_x_range/2)
                    ax.set_ylim(y_center - new_y_range/2, y_center + new_y_range/2)
                    
                    if self.verbose:
                        print(f"Applied scale {scale}: view range scaled to {new_x_range:.2f} x {new_y_range:.2f}")
                
                # Save with high DPI for better quality, no padding
                if image_format.lower() == 'png':
                    fig.savefig(image_path, dpi=1000, bbox_inches='tight', pad_inches=0,
                               facecolor='white', edgecolor='none')
                elif image_format.lower() == 'svg':
                    fig.savefig(image_path, format='svg', bbox_inches='tight', pad_inches=0,
                               facecolor='white', edgecolor='none')
                elif image_format.lower() == 'eps':
                    fig.savefig(image_path, format='eps', bbox_inches='tight', pad_inches=0,
                               facecolor='white', edgecolor='none')
                else:
                    fig.savefig(image_path, bbox_inches='tight', pad_inches=0,
                               facecolor='white', edgecolor='none')
                
                plt.close(fig)  # Clean up memory
                
                if self.verbose:
                    print(f"Image saved successfully: {image_path}")
                    if rotation:
                        print(f"Applied rotation: {rotation}")
                    if scale:
                        print(f"Applied scale: {scale}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"Error saving image with matplotlib: {e}")
                    print("Falling back to view() only")
                view(atoms_to_view)
        else:
            # Just show the visualization
            view(atoms_to_view)

    

        
if __name__ == "__main__":
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'                        # e.g. "FAU", "BEA" or "MFI"
    solvent_type = 'methanol_240_water_960'     # e.g. "water_pure", "methanol_120_water_1080", "methanol_240_water_960", "methanol_600_water_600"
    pore_type = 'hydrophilic'                   # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '02_01_02_propanol'             # e.g. "01_methanol", "02_01_02_propanol"
    snapshot_index = 1
                    
    config_mda = UniverseASE(
                             zeolite_type = zeolite_type,
                             solvent_type = solvent_type,
                             pore_type = pore_type,
                             adsorbate = adsorbate,
                             snapshot_index = snapshot_index,
                             verbose = True,
                             )
    
    config_mda.view_system(
        view_raw=False,
        components='adsorbate_solvent',
        save_image=False,
        image_format='png',
        rotation='-67x,-27y,-11z',  # Custom rotation angles
        scale=1   # Scale factor: 1.0 = normal view, >1.0 = zoom out (farther), <1.0 = zoom in (closer)
    )