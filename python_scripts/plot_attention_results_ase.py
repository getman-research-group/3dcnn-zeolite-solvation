

"""
plot_3d_cnn_results_ase.py

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
class LoadUniverseASE:
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
                    box_size: float = 16.0,
                    ):
        """
        Visualize the molecular system with options to show different components.
        Properly handles periodic boundary conditions and sets viewing box boundaries.
        
        """
        if not hasattr(self.ase_atoms, 'arrays') or 'mol-id' not in self.ase_atoms.arrays:
            if self.verbose:
                print("Warning: mol-id not available, showing all atoms")
            view(self.ase_atoms)
            return
        
        mol_ids = self.ase_atoms.arrays['mol-id']
        cell = self.ase_atoms.get_cell()
        
        # Create a copy for unwrapping coordinates
        atoms_unwrapped = self.ase_atoms.copy()
        
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
        adsorbate_atoms_wrapped = viewing_atoms[adsorbate_mask]
        center_wrapped = adsorbate_atoms_wrapped.get_center_of_mass()
        
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
        
        # Show adsorbate + all solvent molecules (water + methanol)
        water_mask = ((mol_ids >= self.water_mol_range[0]) & (mol_ids <= self.water_mol_range[1]))
        if self.has_methanol:
            methanol_mask = ((mol_ids >= self.methanol_mol_range[0]) & (mol_ids <= self.methanol_mol_range[1]))
            solvent_mask = water_mask | methanol_mask
        else:
            solvent_mask = water_mask
        component_mask = solvent_mask | (mol_ids == self.adsorbate_mol_id)
        
        # Combine both masks
        final_mask = in_box_mask & component_mask
        
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
            if np.any(mol_ids[final_mask] == 1202):
                final_center = atoms_to_view[mol_ids[final_mask] == 1202].get_center_of_mass()
                print(f"Final adsorbate COM in view: {final_center}")
            print(f"New cell dimensions: {box_size} x {box_size} x {box_size} Å")

        # Just show the visualization
        view(atoms_to_view)
    
    
    def get_adsorbate_center(self):
        """
        Get adsorbate center of mass in original coordinates
        
        Note: ASE's get_center_of_mass() uses atomic mass weighting automatically
        This matches the voxel grid generation method
        """
        mol_ids = self.ase_atoms.arrays['mol-id']
        adsorbate_mask = mol_ids == self.adsorbate_mol_id
        adsorbate_atoms = self.ase_atoms[adsorbate_mask]
        return adsorbate_atoms.get_center_of_mass()
    
    
    def extract_region(self, box_size=16.0):
        """
        Extract 16Å box around adsorbate (adsorbate + solvent only, no zeolite)
        Coordinates centered at (8, 8, 8) matching voxel grid
        
        Important: 
        - Uses ASE's get_center_of_mass() which is atomic mass-weighted
        - Applies PBC correction for accurate relative positions (matches voxel generation)
        - Repositions so adsorbate COM is at (8, 8, 8) 
        - This matches exactly with voxel grid generation where adsorbate COM is centered
        """
        mol_ids = self.ase_atoms.arrays['mol-id']
        positions = self.ase_atoms.get_positions()
        center = self.get_adsorbate_center()
        
        # Get box dimensions for PBC (matching voxel generation method)
        cell = self.ase_atoms.get_cell()
        box_dimensions = cell.lengths()  # [Lx, Ly, Lz]
        
        if self.verbose:
            print(f"\n--- Extracting {box_size}Å region around adsorbate")
            print(f"    Adsorbate COM (original): {center}")
            print(f"    System box dimensions: [{box_dimensions[0]:.3f}, {box_dimensions[1]:.3f}, {box_dimensions[2]:.3f}] Å")
        
        # Calculate relative positions with PBC correction (matching voxel generation)
        relative_positions = positions - center
        
        # Apply minimum image convention for periodic boundary conditions
        # This matches the voxel generation method in generate_voxel_grids_type_2.py
        for dim in range(3):
            relative_positions[:, dim] = relative_positions[:, dim] - \
                box_dimensions[dim] * np.round(relative_positions[:, dim] / box_dimensions[dim])
        
        # Select atoms within box using PBC-corrected relative positions
        half_box = box_size / 2.0
        
        # Component mask (adsorbate + solvent)
        water_mask = (mol_ids >= self.water_mol_range[0]) & (mol_ids <= self.water_mol_range[1])
        if self.has_methanol:
            methanol_mask = (mol_ids >= self.methanol_mol_range[0]) & (mol_ids <= self.methanol_mol_range[1])
            solvent_mask = water_mask | methanol_mask
        else:
            solvent_mask = water_mask
        component_mask = solvent_mask | (mol_ids == self.adsorbate_mol_id)
        
        # Spatial mask using PBC-corrected positions
        within_box_mask = np.all(np.abs(relative_positions) <= half_box, axis=1)
        final_mask = component_mask & within_box_mask
        
        # Extract atoms
        region = self.ase_atoms[final_mask].copy()
        
        # Reposition using PBC-corrected relative positions
        # Add half_box to shift origin from adsorbate COM to (8, 8, 8)
        new_pos = relative_positions[final_mask] + half_box
        region.set_positions(new_pos)
        region.set_cell([box_size, box_size, box_size])
        region.set_pbc([False, False, False])
        
        # Verify adsorbate is centered at (8, 8, 8)
        if self.verbose:
            region_mol_ids = region.arrays['mol-id']
            region_adsorbate_mask = region_mol_ids == self.adsorbate_mol_id
            if np.any(region_adsorbate_mask):
                region_adsorbate = region[region_adsorbate_mask]
                new_center = region_adsorbate.get_center_of_mass()
                print(f"    Adsorbate COM (recentered): {new_center}")
                print(f"    Expected: [{half_box:.1f}, {half_box:.1f}, {half_box:.1f}]")
                deviation = np.linalg.norm(new_center - np.array([half_box, half_box, half_box]))
                print(f"    Deviation from center: {deviation:.6f} Å")
            print(f"    Final atom count in region: {len(region)}")
        
        return region
    
    
    def create_slice(self, region, plane='XY', slice_thickness=1.6):
        """
        Create 2D slice through center with adjustable thickness
        
        Args:
            region: ASE Atoms from extract_region()
            plane: 'XY', 'XZ', or 'YZ'
            slice_thickness: Total thickness of the slice in Angstroms (default: 1.6Å)
                           - 1.6Å matches voxel indices 9-10 (original behavior)
                           - 4.0Å would capture ±2.0Å from center
                           - Slice extends ±(slice_thickness/2) from center at 8.0Å
        Returns:
            ASE Atoms in slice
        """
        positions = region.get_positions()
        axis = {'YZ': 0, 'XZ': 1, 'XY': 2}[plane]
        
        # Slice at center (8.0Å) with ±(slice_thickness/2) thickness
        half_thickness = slice_thickness / 2.0
        slice_mask = np.abs(positions[:, axis] - 8.0) < half_thickness
        
        if self.verbose:
            num_atoms_in_slice = np.sum(slice_mask)
            print(f"    {plane} slice with thickness {slice_thickness:.1f}Å (±{half_thickness:.1f}Å): {num_atoms_in_slice} atoms")
        
        return region[slice_mask]
    
    
    def plot_slice(self, slice_atoms, plane='XY', molecule_type='all', 
                   transparent_bg=False, save_path=None, slice_thickness=1.6):
        """
        Plot 2D slice directly from coordinates (matching imshow convention exactly)
        
        Plane definitions (matching attention plots):
        - YZ plane: z(horizontal), y(vertical)
        - XZ plane: z(horizontal), x(vertical)  
        - XY plane: y(horizontal), x(vertical)
        
        Args:
            slice_atoms: ASE Atoms object
            plane: 'XY', 'XZ', or 'YZ'
            molecule_type: 'adsorbate', 'solvent', or 'all'
            transparent_bg: If True, set transparent background for overlay
            save_path: If provided, save figure to this path
            slice_thickness: Thickness used for slice creation (for filename)
        
        Uses scatter plot with ASE's default colors to ensure correct orientation
        """
        from ase.data import covalent_radii
        from ase.data.colors import jmol_colors
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Make background transparent if requested
        if transparent_bg:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
        
        # Get mol-id to filter atoms
        mol_ids = slice_atoms.arrays['mol-id']
        
        # Create mask based on molecule_type
        if molecule_type == 'adsorbate':
            mask = mol_ids == self.adsorbate_mol_id
        elif molecule_type == 'solvent':
            water_mask = (mol_ids >= self.water_mol_range[0]) & (mol_ids <= self.water_mol_range[1])
            if self.has_methanol:
                methanol_mask = (mol_ids >= self.methanol_mol_range[0]) & (mol_ids <= self.methanol_mol_range[1])
                mask = water_mask | methanol_mask
            else:
                mask = water_mask
        else:  # 'all'
            mask = np.ones(len(slice_atoms), dtype=bool)
        
        # Filter atoms by mask
        filtered_atoms = slice_atoms[mask]
        
        if len(filtered_atoms) == 0:
            print(f"Warning: No {molecule_type} atoms found in slice")
            return fig, ax
        
        # Get atom positions and atomic numbers
        positions = filtered_atoms.get_positions()
        atomic_numbers = filtered_atoms.get_atomic_numbers()
        
        # Extract 2D coordinates based on plane
        # Match exactly with imshow convention: array[i,j] where i→vertical, j→horizontal
        if plane == 'YZ':
            # YZ plane: z(horizontal), y(vertical)
            x_coords = positions[:, 2]  # z → horizontal (x-axis in plot)
            y_coords = positions[:, 1]  # y → vertical (y-axis in plot)
        elif plane == 'XZ':
            # XZ plane: z(horizontal), x(vertical)
            x_coords = positions[:, 2]  # z → horizontal (x-axis in plot)
            y_coords = positions[:, 0]  # x → vertical (y-axis in plot)
        else:  # XY
            # XY plane: y(horizontal), x(vertical)
            x_coords = positions[:, 1]  # y → horizontal (x-axis in plot)
            y_coords = positions[:, 0]  # x → vertical (y-axis in plot)
        
        # Plot atoms using scatter with ASE's default jmol colors
        for i in range(len(filtered_atoms)):
            atomic_num = atomic_numbers[i]
            color = jmol_colors[atomic_num]
            # Use covalent radius scaled for visualization
            radius = covalent_radii[atomic_num] * 1.0  # Larger atoms for better visibility
            
            circle = plt.Circle((x_coords[i],
                                 y_coords[i]),
                                 radius,
                                 color=color,
                                 ec='black',
                                 linewidth=3, # Outline with black border
                                 zorder=10,
                                 alpha=0.9)
            ax.add_patch(circle)
        
        # Set axis labels to match attention plot convention
        ax.set_xlabel('z' if plane != 'XY' else 'y', fontsize=14)
        ax.set_ylabel('y' if plane == 'YZ' else 'x', fontsize=14)
        title = f'{plane} - {molecule_type}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        ax.set_aspect('equal')
        
        if not transparent_bg:
            ax.set_facecolor('white')
        
        # Remove axis ticks and labels for cleaner overlay
        if transparent_bg:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            # Remove frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='none' if transparent_bg else 'white',
                       edgecolor='none')
            print(f"Saved: {save_path}")
        
        return fig, ax
    




if __name__ == "__main__":
    
    ## Define the simulation parameters
    zeolite_type = 'FAU'           # e.g. "FAU", "BEA" or "MFI"
    solvent_type = 'water_pure'    # e.g. "water_pure", "methanol_120_water_1080", "methanol_240_water_960", "methanol_600_water_600"
    pore_type = 'hydrophilic'      # e.g. "hydrophilic", "hydrophobic"
    adsorbate = '11_01_propylene_glycol'      # e.g. "01_methanol", "02_01_02_propanol"
    snapshot_index = 6
    voxel_id = 1  # Add voxel_id for filename consistency
    
    ## Slice thickness parameter (Angstroms)
    # Controls the thickness of the 2D slice extracted from the 3D structure
    # - 1.6Å: Default, matches voxel indices 9-10 (center ±0.8Å)
    # - 3.2Å: Captures voxel indices 8-11 (center ±1.6Å)
    # - 4.0Å: Captures center ±2.0Å
    # Note: Attention heatmap always uses 1.6Å (voxel indices 9-10)
    slice_thickness = 4  # Angstroms
    
    # Create output directory
    output_dir = os.path.join(get_paths("output_figure_path"), "ase_slices_overlay")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load configuration using MDAnalysis    
    config_mda = LoadUniverseASE(
                             zeolite_type = zeolite_type,
                             solvent_type = solvent_type,
                             pore_type = pore_type,
                             adsorbate = adsorbate,
                             snapshot_index = snapshot_index,
                             verbose = True,
                             )
    
    # Extract 16Å region around adsorbate
    print("\n" + "="*80)
    print("Extracting 16Å region and creating slices for overlay")
    print("="*80)
    region = config_mda.extract_region(box_size=16.0)
    print(f"Extracted {len(region)} atoms in 16Å box")
    
    # Generate 6 transparent overlay images: 3 planes × 2 molecule types
    molecule_types = ['adsorbate', 'solvent']
    planes = ['YZ', 'XZ', 'XY']
    
    print(f"\n{'='*80}")
    print(f"Generating 6 transparent overlay images (separate adsorbate & solvent)")
    print(f"Slice thickness: {slice_thickness:.1f} Å (center ±{slice_thickness/2:.1f} Å)")
    print(f"{'='*80}")
    
    for mol_type in molecule_types:
        for plane in planes:
            print(f"\n--- Creating {plane} plane - {mol_type} ---")
            
            # Create slice with specified thickness
            slice_atoms = config_mda.create_slice(region, plane=plane, slice_thickness=slice_thickness)
            print(f"{plane} slice: {len(slice_atoms)} total atoms")
            
            # Generate filename with thickness info
            filename = f"ase-{zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}-snap{snapshot_index:02d}-vox{voxel_id}-{plane}-{mol_type}.png"
            save_path = os.path.join(output_dir, filename)
            
            # Plot with transparent background
            fig, ax = config_mda.plot_slice(
                slice_atoms, 
                plane=plane,
                molecule_type=mol_type,
                transparent_bg=True,
                save_path=save_path,
                slice_thickness=slice_thickness
            )
            
            # Close figure to free memory
            plt.close(fig)
    
    # Generate 3 combined images: adsorbate + solvent together
    print(f"\n{'='*80}")
    print(f"Generating 3 combined images (adsorbate + solvent together)")
    print(f"Slice thickness: {slice_thickness:.1f} Å (center ±{slice_thickness/2:.1f} Å)")
    print(f"{'='*80}")
    
    for plane in planes:
        print(f"\n--- Creating {plane} plane - combined (adsorbate + solvent) ---")
        
        # Create slice with specified thickness
        slice_atoms = config_mda.create_slice(region, plane=plane, slice_thickness=slice_thickness)
        print(f"{plane} slice: {len(slice_atoms)} total atoms")
        
        # Generate filename with thickness info
        filename = f"ase-{zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}-snap{snapshot_index:02d}-vox{voxel_id}-{plane}-combined.png"
        save_path = os.path.join(output_dir, filename)
        
        # Plot with molecule_type='all' (adsorbate + solvent combined)
        fig, ax = config_mda.plot_slice(
            slice_atoms, 
            plane=plane,
            molecule_type='all',  # Show both adsorbate and solvent
            transparent_bg=True,
            save_path=save_path,
            slice_thickness=slice_thickness
        )
        
        # Close figure to free memory
        plt.close(fig)
    
    print("\n" + "="*80)
    print("✅ All images generated successfully!")
    print(f"📁 Location: {output_dir}")
    print("="*80)
    print("\nGenerated 9 images total:")
    print("\n1️⃣  Separate molecule types (6 images):")
    thickness_str = f"thick{slice_thickness:.1f}A".replace('.', 'p')
    print(f"  • YZ-adsorbate-{thickness_str}.png, YZ-solvent-{thickness_str}.png")
    print(f"  • XZ-adsorbate-{thickness_str}.png, XZ-solvent-{thickness_str}.png")
    print(f"  • XY-adsorbate-{thickness_str}.png, XY-solvent-{thickness_str}.png")
    print(f"\n2️⃣  Combined (adsorbate + solvent together, 3 images):")
    print(f"  • YZ-combined-{thickness_str}.png")
    print(f"  • XZ-combined-{thickness_str}.png")
    print(f"  • XY-combined-{thickness_str}.png")
    print(f"\n⚙️  Slice parameters:")
    print(f"   • Thickness: {slice_thickness:.1f} Å (center 8.0 Å ± {slice_thickness/2:.1f} Å)")
    print(f"   • Position range: {8.0 - slice_thickness/2:.1f} - {8.0 + slice_thickness/2:.1f} Å")
    print(f"   • Note: Attention heatmap uses fixed 1.6 Å thickness (voxel indices 9-10)")
    print("\n💡 All images have transparent backgrounds and can be directly")
    print("   overlaid on attention heatmaps in image editing software.")
    print("="*80)