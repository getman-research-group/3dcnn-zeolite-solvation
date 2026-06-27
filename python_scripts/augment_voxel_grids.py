# -*- coding: utf-8 -*-
"""
augment_voxel_grids.py
    The purpose of this code is to perform data augmentation on voxel grids generated from MD snapshots.
    Specifically designed for format with separated channel groups:
    - First N channels: adsorbate features only (solvent atoms = 0 in these channels)
    - Last N channels: solvent features only (adsorbate atoms = 0 in these channels)
    
    Since voxel grids are cubic and centered on adsorbate COM, we use rotation transformations
    which are physically meaningful and preserve the molecular interaction patterns.

Functions:
    - augment_voxel_grid: Main function to augment a single voxel grid with format
    - rotate_90_degrees: Rotate voxel grid by 90 degrees around specified axis
    - apply_rotation_sequence: Apply sequence of rotations
    - plot_all_augmented_grids: Independently create and plot augmented occupancy grids
    - check_augment_grids: Verify augmentation quality for format
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_voxel_grids import GenerateVoxelGrids
import sys
import os
# Add the parent directory to sys.path to access core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.global_vars import FEATURE_LIST
from core.path import get_paths

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
    Augment a single voxel grid with all 24 rotations for the separated-channel format.
    The voxel representation uses separated channel groups where:
    - First N channels contain adsorbate features only
    - Last N channels contain solvent features only
    
    INPUTS:
        generate_voxel_grids: GenerateVoxelGrids object using separated channel groups
            Object containing the voxel grid to augment and label information
        cube_rotation_sequences: list of strings
            List of rotation sequences to apply
        include_identity: bool, whether to include the original grid in the output
    
    OUTPUTS:
        augmented_data: list of dicts, each containing:
            - 'voxel_grid': rotated voxel grid (main version for ML training)
            - 'rotation_name': rotation sequence name
            - 'target_interaction_energy': target label (same for all rotations)
            - 'data_format': 'separated_channels'
            - 'feature_channel_mapping': channel mapping information
        rotation_names: list of strings, corresponding rotation names for each grid
    """
    
    print(f"\n\n--- Augmenting Voxel Grid using {len(cube_rotation_sequences)} configured rotation sequences...")
    
    # Get the main voxel grid
    voxel_grid = generate_voxel_grids.voxel_grid
    
    # Get target energy and other metadata
    target_energy = generate_voxel_grids.target_interaction_energy
    feature_channel_mapping = generate_voxel_grids.feature_channel_mapping
    atomic_features = generate_voxel_grids.atomic_features
    
    print(f"--- Original voxel grid shape: {voxel_grid.shape}")
    print(f"--- Target interaction energy: {target_energy} (eV)")
    print(f"--- Atomic features: {len(atomic_features)}")
    print(f"--- Total channels: {voxel_grid.shape[3]} (2 × {len(atomic_features)} features)")
    print(f"--- Adsorbate channels: 0-{len(atomic_features)-1}")
    print(f"--- Solvent channels: {len(atomic_features)}-{voxel_grid.shape[3]-1}")
    
    # Verify the expected separated-channel structure
    expected_channels = 2 * len(atomic_features)
    if voxel_grid.shape[3] != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels for the separated-channel format, got {voxel_grid.shape[3]}")
    
    augmented_data = []
    rotation_names = []
    
    for sequence in cube_rotation_sequences:
        if sequence == '':
            # Identity transformation
            if include_identity:
                augmented_data.append({
                    'voxel_grid': voxel_grid.copy(),
                    'rotation_name': 'identity',
                    'target_interaction_energy': target_energy,
                    'data_format': 'separated_channels',
                    'feature_channel_mapping': feature_channel_mapping.copy(),
                    'atomic_features': atomic_features.copy()
                })
                rotation_names.append('identity')
        else:
            # Apply rotation
            rotated_grid = apply_rotation_sequence(voxel_grid, sequence)
            augmented_data.append({
                'voxel_grid': rotated_grid,
                'rotation_name': f'rotation_{sequence}',
                'target_interaction_energy': target_energy,
                'data_format': 'separated_channels',
                'feature_channel_mapping': feature_channel_mapping.copy(),
                'atomic_features': atomic_features.copy()
            })
            rotation_names.append(f'rotation_{sequence}')
    
    print(f"    Augmentation completed!")
    print(f"    Number of augmented grids: {len(augmented_data)}")
    print(f"    Each grid shape: {augmented_data[0]['voxel_grid'].shape}")
    print(f"    Data format: separated channel groups")
    print(f"    Rotation names: {rotation_names[:5]}...")  # Show first 5 names
    if (isinstance(target_energy, (int, float, np.integer, np.floating))
            and np.isfinite(target_energy)):
        print(f"    All generated grids inherit target_interaction_energy: {target_energy} (eV)")
    else:
        print(f"    WARNING: Generated grids do not have a valid target interaction energy")
    
    return augmented_data, rotation_names


def plot_all_augmented_grids(generate_voxel_grids,
                             cube_rotation_sequences=CUBE_ROTATION_SEQUENCES,
                             include_identity=True,
                             max_cols=6,
                             default_vox_rep=(3, 3),
                             water_alpha=0.35,
                             methanol_alpha=0.55,
                             adsorbate_alpha=0.9,
                             show_fig=False,
                             save_fig=True,
                             ):
    """
    Independently create and plot molecule-resolved augmented occupancy grids.

    This function does not use or modify the augmented training grids. It first
    builds a three-channel occupancy grid directly from the loaded MD snapshot,
    with channels for water, methanol, and adsorbate. It then applies the requested
    rotations only for visualization and arranges the results in one figure.

    Nothing in this function runs unless it is explicitly called, so normal data
    augmentation and dataset generation have no additional plotting overhead.

    INPUTS:
        generate_voxel_grids: initialized GenerateVoxelGrids object
        cube_rotation_sequences: rotation sequences to visualize
        include_identity: whether to include the original orientation
        max_cols: maximum number of subplot columns
        default_vox_rep: width and height multiplier for each subplot
        water_alpha: transparency of water voxels
        methanol_alpha: transparency of methanol voxels
        adsorbate_alpha: transparency of adsorbate voxels
        show_fig: whether to display the figure interactively
        save_fig: whether to save the figure in output_figures/voxel_grids

    OUTPUTS:
        fig: matplotlib figure containing the augmented occupancy plots
    """
    universe = generate_voxel_grids.snapshot_mda.universe
    box_dimensions = universe.dimensions[:3]

    # Plot-only channels: water, methanol, and adsorbate.
    occupancy_grid = np.zeros((*generate_voxel_grids.bin_array, 3), dtype=bool)
    atom_groups = [
        (universe.select_atoms('resname HOH'), 0),
        (universe.select_atoms('resname MEO'), 1),
        (universe.select_atoms('resname ADS'), 2),
    ]

    for atoms, channel_idx in atom_groups:
        if len(atoms) == 0:
            continue

        relative_positions = atoms.positions - generate_voxel_grids.COM_adsorbate

        # Apply the same periodic minimum image convention as voxel generation.
        for dim in range(3):
            relative_positions[:, dim] = (
                relative_positions[:, dim]
                - box_dimensions[dim]
                * np.round(relative_positions[:, dim] / box_dimensions[dim])
            )

        within_box_mask = np.all(
            np.abs(relative_positions) <= generate_voxel_grids.grid_half_box_length,
            axis=1,
        )
        positions_in_box = relative_positions[within_box_mask]

        if len(positions_in_box) == 0:
            continue

        shifted_positions = positions_in_box + generate_voxel_grids.grid_half_box_length
        voxel_indices = np.floor(
            shifted_positions / generate_voxel_grids.box_increment
        ).astype(int)
        voxel_indices = np.clip(
            voxel_indices, 0, generate_voxel_grids.max_bin_num - 1
        )

        for indices in voxel_indices:
            occupancy_grid[tuple(indices) + (channel_idx,)] = True

    # Rotate only the plot-specific occupancy grid.
    plot_grids = []
    rotation_names = []
    for sequence in cube_rotation_sequences:
        if sequence == '':
            if include_identity:
                plot_grids.append(occupancy_grid.copy())
                rotation_names.append('identity')
        else:
            plot_grids.append(apply_rotation_sequence(occupancy_grid, sequence))
            rotation_names.append(f'rotation_{sequence}')

    n_grids = len(plot_grids)
    print(f"\n--- Plotting {n_grids} augmented occupancy grids...")
    if n_grids == 0:
        print("    No augmented occupancy grids available for plotting")
        return None

    n_cols = min(max_cols, n_grids)
    n_rows = int(np.ceil(n_grids / n_cols))
    fig = plt.figure(
        figsize=(default_vox_rep[0] * n_cols, default_vox_rep[1] * n_rows)
    )

    for i, (grid, name) in enumerate(zip(plot_grids, rotation_names)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        x, y, z = np.indices(np.array(grid.shape[:3]) + 1)

        water_voxels = grid[:, :, :, 0]
        methanol_voxels = grid[:, :, :, 1]
        adsorbate_voxels = grid[:, :, :, 2]

        if np.any(water_voxels):
            ax.voxels(x, y, z, water_voxels, facecolors='blue',
                      alpha=water_alpha, edgecolor='none')
        if np.any(methanol_voxels):
            ax.voxels(x, y, z, methanol_voxels, facecolors='green',
                      alpha=methanol_alpha, edgecolor='none')
        if np.any(adsorbate_voxels):
            ax.voxels(x, y, z, adsorbate_voxels, facecolors='red',
                      alpha=adsorbate_alpha, edgecolor='none')

        ax.set_title(name, fontsize=9, pad=4)
        ax.set_xlim(0, grid.shape[0])
        ax.set_ylim(0, grid.shape[1])
        ax.set_zlim(0, grid.shape[2])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

    plt.tight_layout()

    if save_fig:
        output_folder = os.path.join(get_paths('output_figure_path'), 'voxel_grids')
        os.makedirs(output_folder, exist_ok=True)
        figure_filename = (
            f'voxel_grids_augment_{generate_voxel_grids.zeolite_type}-'
            f'{generate_voxel_grids.solvent_type}-{generate_voxel_grids.pore_type}-'
            f'{generate_voxel_grids.adsorbate}-'
            f'snapshot{generate_voxel_grids.snapshot_index:02d}.png'
        )
        full_figure_path = os.path.join(output_folder, figure_filename)
        fig.savefig(full_figure_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"    Figure saved to: {full_figure_path}")

    if show_fig:
        plt.show()

    return fig


def check_augment_grids(original_grid, augmented_data, rotation_names):
    """
    Check the quality of augmented grids by verifying sum preservation and uniqueness.
    Specifically designed for voxel data with separated channel groups.
    
    INPUTS:
        original_grid: numpy array, original voxel grid
        augmented_data: list of dicts, each containing augmented grid data
        rotation_names: list of strings, corresponding rotation names
    
    OUTPUTS:
        similarity_matrix: numpy array, upper triangular matrix showing grid similarities
    """
    print(f"\n--- Checking Augmented Voxel Grids Quality ---")
    
    # Extract voxel grids and metadata from augmented_data
    voxel_grids_augment = [item['voxel_grid'] for item in augmented_data]
    target_energies = [item['target_interaction_energy'] for item in augmented_data]
    data_formats = [item.get('data_format', 'unknown') for item in augmented_data]

    if len(voxel_grids_augment) != len(rotation_names):
        print(f"    WARNING: Found {len(voxel_grids_augment)} grids but {len(rotation_names)} rotation names")
    
    # Verify data format consistency
    unique_formats = set(data_formats)
    if len(unique_formats) == 1 and 'separated_channels' in unique_formats:
        print(f"    ✓ All grids use separated channel groups")
        print(f"    Original grid shape: {original_grid.shape}")
        num_features = len(augmented_data[0]['atomic_features'])
        print(f"    Channels 0-{num_features-1}: adsorbate features")
        print(f"    Channels {num_features}-{original_grid.shape[3]-1}: solvent features")
    else:
        print(f"    ⚠️  WARNING: Unexpected data formats found: {unique_formats}")
    
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
    
    if len(voxel_grids_augment) == 0:
        print(f"    WARNING: No augmented grids available for sum checking")
    elif sum_warnings == 0:
        print(f"    ✓ All {len(voxel_grids_augment)} grids preserve the original sum")
    else:
        print(f"    ✗ {sum_warnings} grids have different sums!")
    
    # Part 2: Check target interaction energy consistency
    print(f"\n    Part 2: Checking target interaction energy consistency...")
    valid_energies = [e for e in target_energies
                      if isinstance(e, (int, float, np.integer, np.floating)) and np.isfinite(e)]
    invalid_energy_count = len(target_energies) - len(valid_energies)

    if not target_energies:
        print(f"    WARNING: No augmented grids available for target-energy checking")
    elif invalid_energy_count > 0:
        print(f"    ✗ {invalid_energy_count} of {len(target_energies)} grids have missing or non-finite target energies")
    elif np.allclose(valid_energies, valid_energies[0], rtol=0.0, atol=1e-12):
        print(f"    ✓ All {len(valid_energies)} grids have consistent target_interaction_energy: {valid_energies[0]} (eV)")
    else:
        unique_energies = np.unique(valid_energies)
        print(f"    ✗ Found {len(unique_energies)} different target energies: {unique_energies.tolist()}")
    
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
    if n_grids == 0:
        print(f"    WARNING: No augmented grids available for uniqueness checking")
    elif total_comparisons == 0:
        print(f"    ✓ All {n_grids} grids are unique (no duplicates found)")
    else:
        print(f"    ✗ Found {total_comparisons} duplicate pairs out of {n_grids*(n_grids-1)//2} comparisons")
    
    # Part 4: Check channel-group dimensions and activity
    print(f"\n    Part 4: Checking channel-group structure and activity...")
    if len(augmented_data) > 0:
        num_features = len(augmented_data[0]['atomic_features'])
        expected_channels = 2 * num_features
        actual_channels = original_grid.shape[3]
        
        if actual_channels == expected_channels:
            print(f"    ✓ Correct channel structure: {actual_channels} channels (2 × {num_features} features)")
            
            # Check whether both channel groups contain data in the first few grids
            sample_grids = voxel_grids_augment[:min(3, len(voxel_grids_augment))]
            for i, grid in enumerate(sample_grids):
                # Check adsorbate- and solvent-channel activity
                adsorbate_channels = grid[:, :, :, :num_features]
                solvent_channels = grid[:, :, :, num_features:]
                
                adsorbate_nonzero = np.any(adsorbate_channels != 0)
                solvent_nonzero = np.any(solvent_channels != 0)
                
                if adsorbate_nonzero and solvent_nonzero:
                    print(f"    ✓ Grid {rotation_names[i]}: Both adsorbate and solvent channels have data")
                elif adsorbate_nonzero:
                    print(f"    ⚠️  Grid {rotation_names[i]}: Only adsorbate channels have data")
                elif solvent_nonzero:
                    print(f"    ⚠️  Grid {rotation_names[i]}: Only solvent channels have data")
                else:
                    print(f"    ✗ Grid {rotation_names[i]}: No data in any channels!")
        else:
            print(f"    ✗ WARNING: Expected {expected_channels} channels, got {actual_channels}")
    
    print(f"    Augmentation quality check completed")
    
    return similarity_matrix



if __name__ == "__main__":
    
    # Define atomic features for the separated-channel format (same as in generate_voxel_grids.py)
    ATOMIC_FEATURES = [
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
    
    # Create input variables dictionary for testing the voxel representation
    input_vars = {
        'zeolite_type':         'FAU',                      # Zeolite type (e.g. "FAU", "BEA", "MFI")
        'solvent_type':         'methanol_240_water_960',   # Solvent type (e.g. "water_pure", "methanol_240_water_960")
        'pore_type':            'hydrophilic',              # Pore type (e.g. "hydrophilic", "hydrophobic")
        'adsorbate':            '02_01_02_propanol',        # Adsorbate type (e.g. "01_methanol", "02_01_02_propanol")
        'snapshot_index':       1,                          # Single snapshot index
        'box_grids_size':       16.0,                       # Box size in Angstrom (centered on adsorbate)
        'box_increment':        0.8,                        # Voxel size in Angstrom
        'feature_list':         ATOMIC_FEATURES,            # List of features to include in the voxel grid
        'include_solvent':      True,                       # Whether to include solvent (water + methanol) atoms
        'include_zeolite':      False,                      # Whether to include zeolite atoms
        'include_adsorbate':    True,                       # Whether to include adsorbate atoms
        'verbose':              True,
    }
    
    print(f"=== Testing Voxel Grid Data Augmentation ===")
    print(f"Data format: Separated channel groups")
    print(f"- Channels 0-{len(ATOMIC_FEATURES)-1}: adsorbate features only")
    print(f"- Channels {len(ATOMIC_FEATURES)}-{2*len(ATOMIC_FEATURES)-1}: solvent features only")
    print(f"Expected total channels: {2 * len(ATOMIC_FEATURES)}")
    
    # Process one snapshot with separated channel groups
    generate_voxel_grids = GenerateVoxelGrids(**input_vars)
    
    print(f"\n=== Original Voxel Grid Information ===")
    print(f"Voxel grid shape: {generate_voxel_grids.voxel_grid.shape}")
    print(f"Target interaction energy: {generate_voxel_grids.target_interaction_energy} (eV)")
    print(f"Feature channel mapping keys: {list(generate_voxel_grids.feature_channel_mapping.keys())}")
    
    # Execute voxel-grid augmentation
    augmented_data, rotation_names = augment_voxel_grid(
        generate_voxel_grids,
        cube_rotation_sequences = CUBE_ROTATION_SEQUENCES,
        include_identity = True
    )
    
    # Check augmented-grid quality and channel separation
    similarity_matrix = check_augment_grids(
        original_grid = generate_voxel_grids.voxel_grid,
        augmented_data = augmented_data,
        rotation_names = rotation_names,
    )
    
    # Display final results
    print(f"\n=== Data Augmentation Summary ===")
    print(f"    Original grid shape: {generate_voxel_grids.voxel_grid.shape}")
    print(f"    Number of augmented grids: {len(augmented_data)}")
    print(f"    Data format: {augmented_data[0]['data_format']}")

    summary_energies = [item['target_interaction_energy'] for item in augmented_data]
    valid_summary_energies = [e for e in summary_energies
                              if isinstance(e, (int, float, np.integer, np.floating)) and np.isfinite(e)]
    if (len(valid_summary_energies) > 0
            and len(valid_summary_energies) == len(summary_energies)
            and np.allclose(valid_summary_energies, valid_summary_energies[0], rtol=0.0, atol=1e-12)):
        print(f"    Target interaction energy: {valid_summary_energies[0]} eV (consistent across all grids)")
    else:
        print(f"    Target interaction energy check: failed or incomplete")

    expected_channels = 2 * len(generate_voxel_grids.atomic_features)
    channel_structure_preserved = all(item['voxel_grid'].shape[3] == expected_channels
                                      for item in augmented_data)
    print(f"    Channel structure preserved: {'✓' if channel_structure_preserved else '✗'}")
    print(f"    Rotation sequences generated: {len(rotation_names)}")
    
    # Optional visualization; no plotting work occurs unless this is called.
    plot_all_augmented_grids(
        generate_voxel_grids=generate_voxel_grids,
        cube_rotation_sequences=CUBE_ROTATION_SEQUENCES,
        include_identity=True,
        max_cols=6,
        default_vox_rep=(3, 3),
        water_alpha=0.35,
        methanol_alpha=0.55,
        adsorbate_alpha=0.9,
        show_fig=False,
        save_fig=True,
    )