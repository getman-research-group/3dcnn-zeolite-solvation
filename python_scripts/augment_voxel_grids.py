# -*- coding: utf-8 -*-
"""
augment_voxel_grids.py
    The purpose of this code is to perform data augmentation on voxel grids generated from MD snapshots.
    Since voxel grids are cubic and centered on adsorbate COM, we use rotation transformations
    which are physically meaningful and preserve the molecular interaction patterns.
    
    Now supports both pure water and water-methanol mixed solvent systems.

Functions:
    - augment_voxel_grid: Main function to augment a single voxel grid
    - rotate_90_degrees: Rotate voxel grid by 90 degrees around specified axis
    - apply_rotation_sequence: Apply sequence of rotations
    - plot_all_augmented_grids: Visualize all augmented grids in a grid layout
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from python_scripts_3dcnn.generate_voxel_grids import GenerateVoxelGrids
from core.path import get_paths
from core.global_vars import FEATURE_LIST
import os

## DEFINING ROTATION SEQUENCES (24 unique rotations including identity)
CUBE_ROTATION_SEQUENCES = [
    '',         # Identity (no rotation)
    'x',        # 90° around x-axis
    'y',        # 90° around y-axis
    'xx',       # 180° around x-axis
    'xy',       # 90° x then 90° y
    'yx',       # 90° y then 90° x
    'yy',       # 180° around y-axis
    'xxx',      # 270° around x-axis
    'xxy',      # 180° x then 90° y
    'xyx',      # 90° x then 90° y then 90° x
    'xyy',      # 90° x then 180° y
    'yxx',      # 90° y then 180° x
    'yyx',      # 180° y then 90° x
    'yyy',      # 270° around y-axis
    'xxxy',     # 270° x then 90° y
    'xxyx',     # 180° x then 90° y then 90° x
    'xxyy',     # same as zz - 180° around z-axis
    'xyxx',     # 90° x then 90° y then 180° x
    'xyyy',     # 90° x then 270° y
    'yxxx',     # 90° y then 270° x
    'yyyx',     # 270° y then 90° x
    'xxxyx',    # same as zzz - 270° around z-axis
    'xyxxx',    # same as z - 90° around z-axis
    'xyyyx',    # complex rotation sequence
]

def rotate_90_degrees(grid, axis):
    """
    Rotate a 4D voxel grid by 90 degrees around the specified axis.
    
    INPUTS:
        grid: numpy array, 4D voxel grid (x, y, z, features)
        axis: int, rotation axis (0=x, 1=y, 2=z)
    
    OUTPUTS:
        rotated_grid: numpy array, rotated voxel grid
    """
    if axis == 0:  # Rotate around x-axis (rotate in y-z plane)
        rotated_grid = np.rot90(grid, k=1, axes=(1, 2))
    elif axis == 1:  # Rotate around y-axis (rotate in z-x plane)
        rotated_grid = np.rot90(grid, k=1, axes=(2, 0))
    elif axis == 2:  # Rotate around z-axis (rotate in x-y plane)
        rotated_grid = np.rot90(grid, k=1, axes=(0, 1))
    else:
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z)")
    
    return rotated_grid


def apply_rotation_sequence(grid, sequence):
    """
    Apply a sequence of 90-degree rotations to a voxel grid.
    
    INPUTS:
        grid: numpy array, 4D voxel grid
        sequence: str, sequence of rotations (e.g., 'xyz', 'xx', 'zy')
    
    OUTPUTS:
        rotated_grid: numpy array, rotated voxel grid
    """
    rotated_grid = grid.copy()
    
    for rotation in sequence:
        if rotation == 'x':
            rotated_grid = rotate_90_degrees(rotated_grid, axis=0)
        elif rotation == 'y':
            rotated_grid = rotate_90_degrees(rotated_grid, axis=1)
        elif rotation == 'z':
            rotated_grid = rotate_90_degrees(rotated_grid, axis=2)
        else:
            raise ValueError(f"Invalid rotation '{rotation}'. Use 'x', 'y', or 'z'")
    
    return rotated_grid

def augment_voxel_grid(generate_voxel_grids, cube_rotation_sequences, include_identity=True):
    """
    Augment a single voxel grid with all 24 rotations.
    Now includes target_interaction_energy from the original GenerateVoxelGrids object.
    Also includes unsaturated voxel grid for better visualization.
    
    INPUTS:
        generate_voxel_grids: GenerateVoxelGrids object
            Object containing the voxel grid to augment and label information
        cube_rotation_sequences: list of strings
            List of rotation sequences to apply
        include_identity: bool, whether to include the original grid in the output
    
    OUTPUTS:
        augmented_data: list of dicts, each containing:
            - 'voxel_grid': rotated voxel grid (saturated)
            - 'voxel_grid_for_plots': rotated voxel grid (unsaturated, for visualization)
            - 'rotation_name': rotation sequence name
            - 'target_interaction_energy': target label (same for all rotations)
        rotation_names: list of strings, corresponding rotation names for each grid
    """
    
    print(f"\n\n--- Augmenting voxel grid with all {len(cube_rotation_sequences)} rotations...")
    
    # Get the voxel grids and target energy
    voxel_grid = generate_voxel_grids.voxel_grid  # Saturated version
    voxel_grid_for_plots = generate_voxel_grids.voxel_grid_for_feature_plots  # Unsaturated version
    target_energy = generate_voxel_grids.target_interaction_energy
    
    print(f"\n--- Original voxel grid shape: {voxel_grid.shape}")
    print(f"--- Original voxel grid for plots shape: {voxel_grid_for_plots.shape}")
    print(f"--- Target interaction energy: {target_energy} (eV)")
    
    augmented_data = []
    rotation_names = []
    
    for sequence in cube_rotation_sequences:
        if sequence == '':
            # Identity transformation
            if include_identity:
                augmented_data.append({
                    'voxel_grid': voxel_grid.copy(),
                    'voxel_grid_for_plots': voxel_grid_for_plots.copy(),
                    'rotation_name': 'identity',
                    'target_interaction_energy': target_energy
                })
                rotation_names.append('identity')
        else:
            # Apply rotation to both versions
            rotated_grid = apply_rotation_sequence(voxel_grid, sequence)
            rotated_grid_for_plots = apply_rotation_sequence(voxel_grid_for_plots, sequence)
            augmented_data.append({
                'voxel_grid': rotated_grid,
                'voxel_grid_for_plots': rotated_grid_for_plots,
                'rotation_name': f'rotation_{sequence}',
                'target_interaction_energy': target_energy
            })
            rotation_names.append(f'rotation_{sequence}')
    
    print(f"    Augmentation completed!")
    print(f"    Number of augmented grids: {len(augmented_data)}")
    print(f"    Each grid shape: {augmented_data[0]['voxel_grid'].shape}")
    print(f"    Each plot grid shape: {augmented_data[0]['voxel_grid_for_plots'].shape}")
    print(f"    Rotation names: {rotation_names[:5]}...")  # Show first 5 names
    print(f"    All grids have target_interaction_energy: {target_energy} (eV)")
    
    return augmented_data, rotation_names


def plot_all_augmented_grids(voxel_grids_augment_list,
                             rotation_names,
                             zeolite_type,
                             solvent_type,
                             pore_type,
                             adsorbate,
                             snapshot_index,
                             max_cols=6,
                             default_vox_rep=(4,4),
                             feature_channel=0,
                             ):
    '''
    This function plots all augmented arrays in a grid layout and saves the figure.
    For mixed solvent systems, automatically detects and color-codes different molecule types.
    
    INPUTS:
        voxel_grids_augment_list: list of numpy arrays
            List containing all augmented voxel grids
        rotation_names: list of strings
            Corresponding rotation names for each grid
        zeolite_type: str
            Zeolite type for filename
        solvent_type: str
            Solvent type for filename
        pore_type: str
            Pore type for filename
        adsorbate: str
            Adsorbate type for filename
        snapshot_index: int
            Snapshot index for filename
        max_cols: int
            Maximum number of columns in the subplot grid
        default_vox_rep: tuple
            Default voxel representation (figsize multiplier)
        feature_channel: int
            Which feature channel to visualize (default: 0 for mol_type)
    
    OUTPUTS:
        None (displays and saves the plot)
    '''
    print(f"\nPlotting all 24 augmented grids...")
    n_grids = len(voxel_grids_augment_list)
    
    if n_grids == 0:
        print("No grids to plot!")
        return None
    
    # Calculate grid layout
    n_cols = min(max_cols, n_grids)
    n_rows = int(np.ceil(n_grids / n_cols))
    
    # Create figure with subplots
    fig_width = default_vox_rep[0] * n_cols
    fig_height = default_vox_rep[1] * n_rows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    axes = []
    
    for i, (grid, name) in enumerate(zip(voxel_grids_augment_list, rotation_names)):
        # Create 3D subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        axes.append(ax)
        
        # Extract the specified feature channel
        if len(grid.shape) == 4:
            # 4D grid: (x, y, z, features)
            feature_grid = grid[:, :, :, feature_channel]
        else:
            # 3D grid: assume it's already the desired channel
            feature_grid = grid
        
        # Find non-zero voxels for mol_type channel (both positive and negative)
        if feature_channel == 0:  # mol_type channel
            non_zero_indices = np.where(np.abs(feature_grid) > 0)
        else:
            non_zero_indices = np.where(feature_grid > 0)
        
        if len(non_zero_indices[0]) > 0:
            # Create voxel representation
            voxel_array = np.zeros(feature_grid.shape, dtype=bool)
            voxel_array[non_zero_indices] = True
            
            # Create coordinates for plotting
            x, y, z = np.indices(np.array(voxel_array.shape) + 1)
            
            # Get feature values for color mapping
            feature_values = feature_grid[non_zero_indices]
            
            # Create colors array with correct shape for voxel plotting
            colors = np.zeros(voxel_array.shape + (4,), dtype=float)  # RGBA for each voxel
            
            if feature_channel == 0:  # mol_type channel - need to distinguish water from methanol
                # For mixed solvent systems, we need to check additional features to distinguish H2O from CH3OH
                if len(grid.shape) == 4 and grid.shape[3] >= 4:  # Check if we have enough channels
                    # Get the C atom channel (should be channel 1 based on FEATURE_LIST)
                    carbon_channel = grid[:, :, :, 1] if grid.shape[3] > 1 else None
                    
                    for j, (ix, iy, iz) in enumerate(zip(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2])):
                        feature_val = feature_values[j]
                        
                        if feature_val > 0:  # Adsorbate
                            colors[ix, iy, iz] = [1.0, 0.0, 0.0, 0.7]  # Red with alpha
                        elif feature_val < 0:  # Solvent (water or methanol)
                            # Check if this voxel has carbon atoms (indicates methanol)
                            if carbon_channel is not None and carbon_channel[ix, iy, iz] > 0:
                                colors[ix, iy, iz] = [0.0, 1.0, 0.0, 0.7]  # Green for methanol
                            else:
                                colors[ix, iy, iz] = [0.0, 0.0, 1.0, 0.7]  # Blue for water
                else:
                    # Fallback to original coloring if we don't have enough channels
                    for j, (ix, iy, iz) in enumerate(zip(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2])):
                        feature_val = feature_values[j]
                        if feature_val > 0:  # Adsorbate
                            colors[ix, iy, iz] = [1.0, 0.0, 0.0, 0.7]  # Red with alpha
                        elif feature_val < 0:  # Solvent (combined water + methanol)
                            colors[ix, iy, iz] = [0.0, 0.0, 1.0, 0.7]  # Blue with alpha
            else:
                # Use a colormap for other features
                # Normalize values for color mapping
                if np.max(feature_values) > np.min(feature_values):
                    norm_values = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
                else:
                    norm_values = np.ones_like(feature_values)
                
                cmap = plt.cm.viridis
                for j, (ix, iy, iz) in enumerate(zip(non_zero_indices[0], non_zero_indices[1], non_zero_indices[2])):
                    color = cmap(norm_values[j])
                    colors[ix, iy, iz] = [color[0], color[1], color[2], 0.7]  # Use colormap with alpha
            
            # Plot voxels
            ax.voxels(x, y, z, voxel_array, facecolors=colors, edgecolor='none')
        else:
            # Empty grid - just show empty space
            ax.text(0.5, 0.5, 0.5, 'Empty', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10)
        
        # Set title and labels
        ax.set_title(f'{name}', fontsize=10, pad=10)
        
        # Set equal aspect ratio and cubic appearance
        grid_shape = feature_grid.shape
        max_range = max(grid_shape)
        ax.set_xlim([0, max_range])
        ax.set_ylim([0, max_range])
        ax.set_zlim([0, max_range])
        
        # Set box aspect to make it cubic
        ax.set_box_aspect([1, 1, 1])
        
        # Remove tick labels and axis labels to save space
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    # Remove any unused subplots
    total_subplots = n_rows * n_cols
    for i in range(n_grids, total_subplots):
        fig.delaxes(fig.add_subplot(n_rows, n_cols, i+1))
    
    plt.tight_layout()
    
    # Save figure to specified path
    output_figure_path = get_paths('output_figure_path')
    feature_names = FEATURE_LIST
    feature_name = feature_names[feature_channel] if feature_channel < len(feature_names) else f'feature{feature_channel}'
    figure_filename = f'voxel_grids_augment_{zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}-snapshot{snapshot_index:02d}-{feature_name}.png'
    full_figure_path = os.path.join(output_figure_path, figure_filename)
    
    # Ensure the output directory exists
    os.makedirs(output_figure_path, exist_ok=True)
    
    # Save the figure with high DPI
    fig.savefig(full_figure_path, dpi=1000, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {full_figure_path}")
    
    # Show the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close(fig)

def check_augment_grids(original_grid, augmented_data, rotation_names):
    """
    Check the quality of augmented grids by verifying sum preservation and uniqueness.
    Updated to work with new augmented_data structure containing labels and plot grids.
    
    INPUTS:
        original_grid: numpy array, original voxel grid
        augmented_data: list of dicts, each containing 'voxel_grid', 'voxel_grid_for_plots', and 'target_interaction_energy'
        rotation_names: list of strings, corresponding rotation names
    
    OUTPUTS:
        similarity_matrix: numpy array, upper triangular matrix showing grid similarities
    """
    print(f"\n--- Checking augmented grids quality ---")
    
    # Extract voxel grids and target energies from augmented_data
    voxel_grids_augment = [item['voxel_grid'] for item in augmented_data]
    voxel_grids_for_plots = [item['voxel_grid_for_plots'] for item in augmented_data]
    target_energies = [item['target_interaction_energy'] for item in augmented_data]
    
    # Part 1: Verify augmentation preserves properties (sum check)
    print(f"\n    Part 1: Checking sum preservation...")
    original_sum = np.sum(original_grid)
    sum_warnings = 0
    
    for i, (grid, name) in enumerate(zip(voxel_grids_augment, rotation_names)):
        grid_sum = np.sum(grid)
        if not np.isclose(grid_sum, original_sum, atol=1e-10):
            print(f"    WARNING: {name} has different sum! Original: {original_sum:.6f}, Current: {grid_sum:.6f}")
            sum_warnings += 1
        if i == 0:  # Print original grid sum for reference
            print(f"    Original grid sum: {original_sum:.6f}")
            print(f"    Grid {name}: sum = {grid_sum:.6f}")
    
    if sum_warnings == 0:
        print(f"    ✓ All {len(voxel_grids_augment)} grids preserve the original sum")
    else:
        print(f"    ✗ {sum_warnings} grids have different sums!")
    
    # Part 2: Check target interaction energy consistency
    print(f"\n    Part 2: Checking target interaction energy consistency...")
    unique_energies = set(target_energies)
    if len(unique_energies) == 1:
        energy_value = list(unique_energies)[0]
        print(f"    ✓ All {len(target_energies)} grids have consistent target_interaction_energy: {energy_value} (eV)")
    else:
        print(f"    ✗ WARNING: Found {len(unique_energies)} different target energies: {unique_energies}")
    
    # Part 3: Check for duplicate/similar grids
    print(f"\n    Part 3: Checking grid uniqueness...")
    
    # Create similarity matrix (upper triangular)
    n_grids = len(voxel_grids_augment)
    similarity_matrix = np.full((n_grids, n_grids), False)
    
    duplicate_count = 0
    for i in range(n_grids):
        for j in range(i + 1, n_grids):
            # Check if grids are identical
            similarity = np.array_equal(voxel_grids_augment[i], voxel_grids_augment[j])
            similarity_matrix[i, j] = similarity
            
            if similarity:
                print(f"    WARNING: {rotation_names[i]} and {rotation_names[j]} are identical!")
                duplicate_count += 1
    
    total_comparisons = np.sum(similarity_matrix)
    if total_comparisons == 0:
        print(f"    ✓ All {n_grids} grids are unique (no duplicates found)")
    else:
        print(f"    ✗ Found {total_comparisons} duplicate pairs out of {n_grids*(n_grids-1)//2} comparisons")
    
    print(f"    Similarity matrix sum: {total_comparisons} (should be 0 for all unique grids)")
    
    # Part 4: Check plot grids consistency
    print(f"\n    Part 4: Checking plot grids availability...")
    if 'voxel_grid_for_plots' in augmented_data[0]:
        print(f"    ✓ All {len(voxel_grids_for_plots)} grids have voxel_grid_for_plots available for visualization")
        plot_grid_shape = voxel_grids_for_plots[0].shape
        print(f"    Plot grids shape: {plot_grid_shape}")
    else:
        print(f"    ✗ WARNING: voxel_grid_for_plots not found in augmented data")
    
    return similarity_matrix



if __name__ == "__main__":
    
    # Create input variables dictionary for testing mixed solvent system
    input_vars = {
        'zeolite_type':         'FAU',                      # Zeolite type (e.g. "FAU", "BEA", "MFI")
        'solvent_type':         'methanol_240_water_960',   # Solvent type (e.g. "water_pure", "methanol_240_water_960")
        'pore_type':            'hydrophilic',              # Pore type (e.g. "hydrophilic", "hydrophobic")
        'adsorbate':            '01_methanol',              # Adsorbate type (e.g. "01_methanol", "02_01_02_propanol")
        'snapshot_index':       1,                          # Single snapshot index
        'box_grids_size':       16.0,                       # Box size in Angstrom (centered on adsorbate)
        'box_increment':        0.8,                        # Voxel size in Angstrom
        'feature_list':         FEATURE_LIST,               # List of features to include in the voxel grid
        'include_solvent':      True,                       # Whether to include solvent (water + methanol) atoms
        'include_zeolite':      False,                      # Whether to include zeolite atoms
        'include_adsorbate':    True,                       # Whether to include adsorbate atoms
        'verbose':              True,
    }
    
    # Process single snapshot
    generate_voxel_grids = GenerateVoxelGrids(**input_vars)
    
    # Execute augmentation (now returns augmented_data with labels and plot grids)
    augmented_data, rotation_names = augment_voxel_grid(generate_voxel_grids,
                                                        cube_rotation_sequences = CUBE_ROTATION_SEQUENCES,
                                                        include_identity = True)
    
    # Check augmented grids quality
    similarity_matrix = check_augment_grids(original_grid = generate_voxel_grids.voxel_grid,
                                            augmented_data = augmented_data,
                                            rotation_names = rotation_names,
                                            )
    
    # Extract different versions for different purposes:
    # - voxel_grid: for ML training (saturated, normalized)
    # - voxel_grid_for_plots: for visualization (unsaturated, clear discrete features)
    print(f"\n--- Data Structure Summary ---")
    print(f"    For ML training: use item['voxel_grid'] - saturated and normalized")
    print(f"    For visualization: use item['voxel_grid_for_plots'] - unsaturated, discrete features")
    print(f"    For labels: use item['target_interaction_energy']")
    
    # Plot all augmented grids and save to file (using unsaturated grids for better visualization)
    # This creates 24 plots similar to generate_voxel_grids.plot_voxel_occupancy_grids()
    voxel_grids_augment_list_for_plots = [item['voxel_grid_for_plots'] for item in augmented_data]

    # # Plot all augmented grids
    # plot_all_augmented_grids(
    #     voxel_grids_augment_list=voxel_grids_augment_list_for_plots,
    #     rotation_names=rotation_names,
    #     zeolite_type=generate_voxel_grids.zeolite_type,
    #     solvent_type=generate_voxel_grids.solvent_type,
    #     pore_type=generate_voxel_grids.pore_type,
    #     adsorbate=generate_voxel_grids.adsorbate,
    #     snapshot_index=generate_voxel_grids.snapshot_index,
    #     max_cols = 6,
    #     default_vox_rep = (3, 3),
    #     feature_channel = 0,  # mol_type channel
    #     )
    