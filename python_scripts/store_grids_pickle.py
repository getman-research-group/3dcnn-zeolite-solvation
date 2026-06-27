"""Build and serialize the complete augmented voxel-grid dataset.

This module connects single-snapshot voxel generation with rotational data
augmentation and persistent storage. A pickle file is created for each unique
combination of zeolite, solvent/pore environment, and adsorbate directory. It
therefore represents one molecular simulation system rather than pooling the
same adsorbate across several environments.

For each system, snapshots ``intE01`` through ``intE10`` are processed. Every
snapshot is converted to a ``20 × 20 × 20 × 28`` voxel grid under the manuscript
defaults and expanded through the 24 proper rotations of a cube. The resulting
240 augmented grids retain the 10 snapshot-specific DFT interaction-energy
labels. The unaugmented grid for each snapshot is also stored separately for
inspection; because the augmentation list includes the identity rotation by
default, this produces 240 training grids plus 10 duplicate reference grids in
the serialized object.

The 28 channels are divided into two molecular groups:

- channels 0–13 contain the 14 adsorbate atomic features;
- channels 14–27 contain the same 14 features for solvent atoms.

Each pickle contains a top-level ``metadata`` dictionary and a ``snapshots``
dictionary keyed by snapshot number. Every snapshot record stores its original
grid, augmented grids, rotation names, and DFT target energy. Metadata records
the system identity, grid geometry, atomic-feature order, channel mapping,
component-inclusion settings, and total grid count.

Functions
---------
generate_and_store_adsorbate_grids(...)
    Process all 10 snapshots for one zeolite–environment–adsorbate system,
    generate and augment its voxel grids, optionally run an augmentation check,
    assemble the nested dataset structure, and write one pickle file.

check_all_pickle_files_exist(...)
    Construct the filenames expected from the configured zeolites,
    environments, adsorbates, and grid dimensions, then report which pickle
    files already exist and which are missing. File contents are not opened.

check_all_pickles_complete(...)
    Open every expected pickle and validate its structure, snapshot count,
    augmentation count, grid shape, channel metadata, labels, and rotation
    records. It returns a detailed completeness report and identifies missing,
    incomplete, or unreadable files.

generate_complete_dataset(...)
    Orchestrate dataset generation across all combinations in
    ``ADSORBATES_BY_ENV``. Existing files may be skipped or regenerated, after
    which the full collection is passed through the completeness validator.

Command-line examples
---------------------
Generate one pickle for a specified simulation system::

    python python_scripts/store_grids_pickle.py --test --zeolite FAU \
        --environment methanol_240_water_960-hydrophilic \
        --adsorbate 02_01_02_propanol

Generate every pickle configured in ``core/global_vars.py``::

    python python_scripts/store_grids_pickle.py --all

Existing complete files are skipped. Add ``--force-regenerate`` to either
command to rebuild the selected output. These commands let users reconstruct
the large pickle files locally instead of downloading them from GitHub.
"""
# Import standard libraries
import argparse
import os
import pickle
import numpy as np
import warnings

# Import voxel generation and rotational augmentation utilities.
from generate_voxel_grids import GenerateVoxelGrids
from augment_voxel_grids import augment_voxel_grid, check_augment_grids
from augment_voxel_grids import CUBE_ROTATION_SEQUENCES
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.path import get_paths
from core.global_vars import ZEOLITE_TYPES, ADSORBATES_BY_ENV
    
    
def generate_and_store_adsorbate_grids(
                                        zeolite_type='FAU',
                                        solvent_type='water_pure',
                                        pore_type='hydrophilic',
                                        adsorbate='01_methanol',
                                        box_grids_size=16.0,
                                        box_increment=0.8,
                                        feature_list=None,
                                        include_zeolite=False,
                                        include_solvent=True,
                                        include_adsorbate=True,
                                        output_dir=None,
                                        include_identity=True,
                                        check_quality=True,
                                        verbose=False,
                                        save_pickle=True,
                                    ):
    """
    Generate and optionally serialize all voxel grids for one molecular system.

    The representation contains separate adsorbate and solvent channel groups,
    giving 2 × N channels for N atomic features. Ten MD snapshots are processed,
    and each snapshot is expanded into 24 symmetry-equivalent rotations.
    
    Each adsorbate contains snapshots 1-10 (fixed range).
    
    INPUTS:
        save_pickle: bool, whether to save the results to pickle file (default: True)
        feature_list: list of atomic features; molecular identity is represented
            by channel separation rather than an additional ``mol_type`` channel
        ... (other parameters same as before)
    
    OUTPUTS:
        adsorbate_data: dict containing grids, labels, rotations, and metadata
        file_path: str, path to saved pickle file (None if save_pickle=False or saving failed)
    """
    # Use the manuscript's 14 atom-level features by default.
    if feature_list is None:
        # Keep this order aligned with GenerateVoxelGrids and the trained model.
        feature_list = [
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
    
    # Fixed snapshot range: 1-10 for all adsorbates
    snapshot_start, snapshot_end = 1, 11
    
    print(f"\n--- Processing {adsorbate}: snapshots {snapshot_start}-{snapshot_end-1}")
    
    # Initialize the system-level dataset and its reproducibility metadata.
    adsorbate_data = {
        'adsorbate': adsorbate,
        'snapshots': {},
        'metadata': {
            'data_format': 'separated_channels',
            'zeolite_type': zeolite_type,
            'solvent_type': solvent_type,
            'pore_type': pore_type,
            'box_grids_size': box_grids_size,
            'box_increment': box_increment,
            'atomic_features': feature_list,  # Renamed from 'features' for clarity
            'include_zeolite': include_zeolite,
            'include_solvent': include_solvent,
            'include_adsorbate': include_adsorbate,
        }
    }
    
    total_grids = 0
    
    # Process each snapshot (fixed range 1-10)
    for snapshot_index in range(snapshot_start, snapshot_end):
        print(f"\n--- Snapshot {snapshot_index}...")
        
        # Generate voxel grid
        input_vars = {
            'zeolite_type': zeolite_type,
            'solvent_type': solvent_type,
            'pore_type': pore_type,
            'adsorbate': adsorbate,
            'snapshot_index': snapshot_index,
            'box_grids_size': box_grids_size,
            'box_increment': box_increment,
            'feature_list': feature_list,
            'include_zeolite': include_zeolite,
            'include_solvent': include_solvent,
            'include_adsorbate': include_adsorbate,
            'verbose': verbose,
        }
        
        generate_voxel_grids = GenerateVoxelGrids(**input_vars)
        
        # Grid geometry and channel definitions are identical for all snapshots,
        # so record them once from the first generated sample.
        if snapshot_index == snapshot_start:
            adsorbate_data['metadata']['grid_shape'] = generate_voxel_grids.voxel_grid.shape
            adsorbate_data['metadata']['feature_channel_mapping'] = generate_voxel_grids.feature_channel_mapping
            adsorbate_data['metadata']['feature_categories'] = generate_voxel_grids.feature_categories
            adsorbate_data['metadata']['element_types'] = getattr(generate_voxel_grids, 'element_types', None)
            adsorbate_data['metadata']['target_interaction_energy'] = generate_voxel_grids.target_interaction_energy
            
            # Record the two channel-group ranges explicitly.
            total_channels = generate_voxel_grids.voxel_grid.shape[3]
            num_atomic_features = len(feature_list)
            adsorbate_data['metadata']['total_channels'] = total_channels
            adsorbate_data['metadata']['num_atomic_features'] = num_atomic_features
            adsorbate_data['metadata']['adsorbate_channels'] = f"0-{num_atomic_features-1}"
            adsorbate_data['metadata']['solvent_channels'] = f"{num_atomic_features}-{total_channels-1}"
            
            print(f"    Voxel metadata: {total_channels} channels "
                  f"({num_atomic_features} adsorbate + {num_atomic_features} solvent)")
        
        # Apply augmentation (now returns augmented_data with labels)
        augmented_data, rotation_names = augment_voxel_grid(
                                                            generate_voxel_grids,
                                                            cube_rotation_sequences = CUBE_ROTATION_SEQUENCES,
                                                            include_identity=include_identity
                                                            )
        
        # Quality check for first snapshot only
        if check_quality and snapshot_index == snapshot_start:
            similarity_matrix = check_augment_grids(
                                                    original_grid = generate_voxel_grids.voxel_grid,
                                                    augmented_data = augmented_data,
                                                    rotation_names = rotation_names
                                                    )
            if np.sum(similarity_matrix) > 0:
                print(f"Warning: found duplicate grids")
        
        # Retain the augmented grids together with the information needed to
        # interpret their rotations, labels, and channel ordering.
        augmented_data_for_storage = []
        for item in augmented_data:
            storage_item = {
                'voxel_grid': item['voxel_grid'],  # ML training grid
                'rotation_name': item['rotation_name'],
                'target_interaction_energy': item['target_interaction_energy'],
                'data_format': item['data_format'],
                'feature_channel_mapping': item['feature_channel_mapping'],
                'atomic_features': item['atomic_features'],
            }
            augmented_data_for_storage.append(storage_item)
        
        adsorbate_data['snapshots'][snapshot_index] = {
                                                    'original_grid': generate_voxel_grids.voxel_grid,
                                                    'augmented_data': augmented_data_for_storage,  # Contains only ML training data
                                                    'rotation_names': rotation_names,
                                                    'target_interaction_energy': generate_voxel_grids.target_interaction_energy,
                                                }
        
        total_grids += len(augmented_data)
        
    # Update metadata
    adsorbate_data['metadata']['total_grids'] = total_grids
    adsorbate_data['metadata']['snapshots_count'] = len(adsorbate_data['snapshots'])
    
    # Store to pickle file only if save_pickle is True
    if save_pickle:
        print (f"\n--- Saving {total_grids} grids to pickle file...")
        if output_dir is None:
            output_dir = get_paths('dataset_cnn')
        
        grid_shape = adsorbate_data['metadata']['grid_shape']
        voxel_shape_str = f"{grid_shape[0]}_{grid_shape[1]}_{grid_shape[2]}_{grid_shape[3]}"
        
        # Create subdirectory based on grid parameters (without num_snaps)
        subdir_name = f"size_{box_grids_size:.1f}-box_{box_increment:.1f}-shape_{voxel_shape_str}"
        subdir_path = os.path.join(output_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        
        filename = f'{zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}-size_{box_grids_size:.1f}-box_{box_increment:.1f}-shape_{voxel_shape_str}.pkl'
        file_path = os.path.join(subdir_path, filename)
        
        with open(file_path, 'wb') as f:
            pickle.dump(adsorbate_data, f, protocol=4)  # Changed from protocol=5 to protocol=4 for better numpy compatibility
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"    Pickle File Saved: {subdir_name}/{filename} ({file_size:.1f} MB, {total_grids} grids)")
    else:
        file_path = None
        print(f"\n--- Processing completed: {total_grids} grids (not saved to file)")
    
    return adsorbate_data, file_path


def check_all_pickle_files_exist(
                                    zeolite_types,
                                    adsorbates_by_env,
                                    box_grids_size=16.0,
                                    box_increment=0.8,
                                    feature_list=None,
                                    output_dir=None,
                                    verbose=False
                                ):
    """
    Check if all expected pickle files exist for given parameters.
    
    INPUTS:
        zeolite_types: list, zeolite types to check
        adsorbates_by_env: dict, environments and their adsorbates
        box_grids_size: float, box size in Angstrom
        box_increment: float, voxel size in Angstrom
        feature_list: list, list of features (used to calculate shape)
        output_dir: str, output directory (optional)
        verbose: bool, whether to print detailed information
    
    OUTPUTS:
        all_exist: bool, True if all expected files exist
        expected_files: list, list of all expected file paths
        missing_files: list, list of missing file paths
    """
    if feature_list is None:
        # Use atom-level features; molecular identity is encoded by channel group.
        feature_list = [
            'atom_type_C', 'atom_type_H', 'atom_type_O', 'is_hydrophobic', 'is_donor', 'is_acceptor', 
            'is_hbonded', 'is_hbonded_donor', 'is_hbonded_acceptor', 'atom_mass', 'partial_charge', 
            'valence', 'LJ_epsilon', 'LJ_sigma'
        ]
    
    if output_dir is None:
        output_dir = get_paths('dataset_cnn')
    
    # Calculate the expected spatial dimensions and channel count.
    max_bin_num = int(box_grids_size / box_increment)
    # Each atomic feature appears once in each molecular channel group.
    num_features = 2 * len(feature_list)  # Each atomic feature appears in both groups
    voxel_shape_str = f"{max_bin_num}_{max_bin_num}_{max_bin_num}_{num_features}"
    
    # Create subdirectory name
    subdir_name = f"size_{box_grids_size:.1f}-box_{box_increment:.1f}-shape_{voxel_shape_str}"
    subdir_path = os.path.join(output_dir, subdir_name)
    
    if verbose:
        print(f"\n--- Checking for existing pickle files ---")
        print(f"    Expected subdirectory: {subdir_name}")
        print(f"    Full path: {subdir_path}")
        
        # Print detailed dataset information
        print(f"\n--- Dataset Information ---")
        print(f"    Zeolite types ({len(zeolite_types)}): {zeolite_types}")
        print(f"    Environments ({len(adsorbates_by_env)}): {list(adsorbates_by_env.keys())}")
        
        # Count environment-adsorbate combinations
        env_adsorbate_combinations = []
        for env, adsorbates in adsorbates_by_env.items():
            print(f"        {env}: {len(adsorbates)} adsorbates")
            for ads in adsorbates:
                env_adsorbate_combinations.append(f"{env}-{ads}")
        
        print(f"    Total environment-adsorbate combinations: {len(env_adsorbate_combinations)}")
        
        # Calculate total expected files (zeolites × env-adsorbate combinations)
        total_expected_files = len(zeolite_types) * len(env_adsorbate_combinations)
        print(f"    Total expected pickle files: {len(zeolite_types)} zeolites × {len(env_adsorbate_combinations)} combinations = {total_expected_files}")
        
        # Print voxel grid information
        print(f"\n--- Voxel Grid Information ---")
        print(f"    Box size: {box_grids_size} Å")
        print(f"    Voxel size: {box_increment} Å")
        print(f"    Grid dimensions: {max_bin_num} × {max_bin_num} × {max_bin_num}")
        print(f"    Number of features: {num_features}")
        print(f"    Expected voxel shape: ({max_bin_num}, {max_bin_num}, {max_bin_num}, {num_features})")
        print(f"    Features: {feature_list}")
        print(f"")
    
    # Check if subdirectory exists
    if not os.path.exists(subdir_path):
        if verbose:
            print(f"    Subdirectory does not exist: {subdir_path}")
        return False, [], []
    
    # Generate list of all expected files
    expected_files = []
    existing_files = []
    missing_files = []
    
    for zeolite in zeolite_types:
        for env, adsorbates in adsorbates_by_env.items():
            solvent_type, pore_type = env.split('-')
            
            for adsorbate in adsorbates:
                filename = f'{zeolite}-{solvent_type}-{pore_type}-{adsorbate}-size_{box_grids_size:.1f}-box_{box_increment:.1f}-shape_{voxel_shape_str}.pkl'
                file_path = os.path.join(subdir_path, filename)
                expected_files.append(file_path)
                
                if os.path.exists(file_path):
                    existing_files.append(file_path)
                    if verbose:
                        print(f"    ✓ Found: {filename}")
                else:
                    missing_files.append(file_path)
                    if verbose:
                        print(f"    ✗ Missing: {filename}")
    
    all_exist = len(missing_files) == 0
    
    if verbose:
        print(f"\n--- Summary ---")
        print(f"    Total expected files: {len(expected_files)}")
        print(f"    Existing files: {len(existing_files)}")
        print(f"    Missing files: {len(missing_files)}")
        print(f"    All files exist: {all_exist}")
        
        if len(existing_files) > 0:
            # Calculate total file size
            total_size_mb = 0
            for file_path in existing_files:
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    total_size_mb += size_mb
                except:
                    pass
            print(f"    Total size of existing files: {total_size_mb:.1f} MB")
    
    return all_exist, expected_files, missing_files


def check_all_pickles_complete(
                                zeolite_types=None,
                                adsorbates_by_env=None,
                                box_grids_size=16.0,
                                box_increment=0.8,
                                feature_list=None,
                                output_dir=None,
                                verbose=True
                              ):
    """
    Check completeness and integrity of all pickle files in the dataset.
    
    INPUTS:
        zeolite_types: list, zeolite types to check (default: from global_vars)
        adsorbates_by_env: dict, environments and their adsorbates (default: from global_vars)
        box_grids_size: float, box size in Angstrom
        box_increment: float, voxel size in Angstrom
        feature_list: list, list of features (default: from global_vars)
        output_dir: str, output directory (optional)
        verbose: bool, whether to print detailed information
    
    OUTPUTS:
        dict: comprehensive report on all pickle files
    """
    if zeolite_types is None:
        zeolite_types = ZEOLITE_TYPES
    if adsorbates_by_env is None:
        adsorbates_by_env = ADSORBATES_BY_ENV
    if feature_list is None:
        # Use atom-level features; molecular identity is encoded by channel group.
        feature_list = [
            'atom_type_C', 'atom_type_H', 'atom_type_O', 'is_hydrophobic', 'is_donor', 'is_acceptor', 
            'is_hbonded', 'is_hbonded_donor', 'is_hbonded_acceptor', 'atom_mass', 'partial_charge', 
            'valence', 'LJ_epsilon', 'LJ_sigma'
        ]
    
    if output_dir is None:
        output_dir = get_paths('dataset_cnn')
    
    # Calculate the expected spatial dimensions and channel count.
    max_bin_num = int(box_grids_size / box_increment)
    # Each atomic feature appears once in each molecular channel group.
    num_features = 2 * len(feature_list)  # Each atomic feature appears in both groups
    voxel_shape_str = f"{max_bin_num}_{max_bin_num}_{max_bin_num}_{num_features}"
    subdir_name = f"size_{box_grids_size:.1f}-box_{box_increment:.1f}-shape_{voxel_shape_str}"
    subdir_path = os.path.join(output_dir, subdir_name)
    
    if verbose:
        print(f"\n=== Checking All Pickle Files Completeness ===")
        print(f"    Directory: {subdir_path}")
        print(f"    Expected grid shape: ({max_bin_num}, {max_bin_num}, {max_bin_num}, {num_features})")
    
    # Initialize results
    results = {
        'total_expected': 0,
        'files_found': 0,
        'files_complete': 0,
        'files_incomplete': 0,
        'files_corrupted': 0,
        'missing_files': [],
        'incomplete_files': [],
        'corrupted_files': [],
        'complete_files': [],
        'total_grids': 0,
        'expected_grids_per_file': 240,  # 10 snapshots × 24 augmentations
        'summary': {}
    }
    
    # Check each combination
    for zeolite in zeolite_types:
        for env, adsorbates in adsorbates_by_env.items():
            solvent_type, pore_type = env.split('-')
            
            for adsorbate in adsorbates:
                results['total_expected'] += 1
                
                # Generate expected filename
                filename = f'{zeolite}-{solvent_type}-{pore_type}-{adsorbate}-size_{box_grids_size:.1f}-box_{box_increment:.1f}-shape_{voxel_shape_str}.pkl'
                file_path = os.path.join(subdir_path, filename)
                
                file_key = f"{zeolite}-{env}-{adsorbate}"
                
                # Check if file exists
                if not os.path.exists(file_path):
                    results['missing_files'].append(file_key)
                    if verbose:
                        print(f"    ✗ Missing: {filename}")
                    continue
                
                results['files_found'] += 1
                
                # Try to load and validate pickle file
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Check basic structure
                    if 'snapshots' not in data or 'metadata' not in data:
                        results['corrupted_files'].append(file_key)
                        if verbose:
                            print(f"    ✗ Corrupted (missing keys): {filename}")
                        continue
                    
                    # Check number of snapshots
                    snapshots = data['snapshots']
                    if len(snapshots) != 10:
                        results['incomplete_files'].append(file_key)
                        if verbose:
                            print(f"    ⚠ Incomplete ({len(snapshots)}/10 snapshots): {filename}")
                        continue
                    
                    # Check each snapshot has required structure
                    total_grids_in_file = 0
                    snapshot_issues = []
                    
                    for snap_idx in range(1, 11):
                        if snap_idx not in snapshots:
                            snapshot_issues.append(f"missing snapshot {snap_idx}")
                            continue
                        
                        snap_data = snapshots[snap_idx]
                        required_keys = ['original_grid', 'augmented_data', 'rotation_names', 'target_interaction_energy']
                        
                        for key in required_keys:
                            if key not in snap_data:
                                snapshot_issues.append(f"snapshot {snap_idx} missing {key}")
                                break
                        
                        # Check augmented data
                        if 'augmented_data' in snap_data:
                            aug_data = snap_data['augmented_data']
                            if len(aug_data) != 24:
                                snapshot_issues.append(f"snapshot {snap_idx} has {len(aug_data)}/24 augmentations")
                            else:
                                total_grids_in_file += len(aug_data)
                                
                                # Check first augmentation structure
                                if len(aug_data) > 0:
                                    first_aug = aug_data[0]
                                    required_aug_keys = ['voxel_grid', 'rotation_name', 'target_interaction_energy']
                                    for key in required_aug_keys:
                                        if key not in first_aug:
                                            snapshot_issues.append(f"snapshot {snap_idx} augmentation missing {key}")
                                            break
                    
                    if snapshot_issues:
                        results['incomplete_files'].append(file_key)
                        if verbose:
                            print(f"    ⚠ Issues: {filename}")
                            for issue in snapshot_issues[:3]:  # Show first 3 issues
                                print(f"        - {issue}")
                            if len(snapshot_issues) > 3:
                                print(f"        - ... and {len(snapshot_issues)-3} more issues")
                        continue
                    
                    # Check total grids count
                    expected_total = 240
                    if total_grids_in_file != expected_total:
                        results['incomplete_files'].append(file_key)
                        if verbose:
                            print(f"    ⚠ Wrong grid count ({total_grids_in_file}/{expected_total}): {filename}")
                        continue
                    
                    # File is complete
                    results['complete_files'].append(file_key)
                    results['files_complete'] += 1
                    results['total_grids'] += total_grids_in_file
                    
                    if verbose:
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        print(f"    ✓ Complete ({file_size:.1f} MB): {filename}")
                
                except Exception as e:
                    results['corrupted_files'].append(file_key)
                    results['files_corrupted'] += 1
                    if verbose:
                        print(f"    ✗ Corrupted (error: {str(e)[:50]}...): {filename}")
    
    # Update counts
    results['files_incomplete'] = len(results['incomplete_files'])
    results['files_corrupted'] = len(results['corrupted_files'])
    
    # Generate summary
    results['summary'] = {
        'total_expected': results['total_expected'],
        'found': results['files_found'],
        'complete': results['files_complete'],
        'incomplete': results['files_incomplete'],
        'corrupted': results['files_corrupted'],
        'missing': len(results['missing_files']),
        'completion_rate': results['files_complete'] / results['total_expected'] * 100 if results['total_expected'] > 0 else 0,
        'total_grids': results['total_grids'],
        'expected_total_grids': results['total_expected'] * 240
    }
    
    if verbose:
        print(f"\n=== Summary ===")
        print(f"    Total expected files: {results['summary']['total_expected']}")
        print(f"    Files found: {results['summary']['found']}")
        print(f"    Complete files: {results['summary']['complete']}")
        print(f"    Incomplete files: {results['summary']['incomplete']}")
        print(f"    Corrupted files: {results['summary']['corrupted']}")
        print(f"    Missing files: {results['summary']['missing']}")
        print(f"    Completion rate: {results['summary']['completion_rate']:.1f}%")
        print(f"    Total grids in complete files: {results['summary']['total_grids']:,}")
        print(f"    Expected total grids: {results['summary']['expected_total_grids']:,}")
        
        if results['missing_files']:
            print(f"\n=== Missing Files ({len(results['missing_files'])}) ===")
            for file_key in results['missing_files'][:10]:  # Show first 10
                print(f"    - {file_key}")
            if len(results['missing_files']) > 10:
                print(f"    - ... and {len(results['missing_files'])-10} more")
        
        if results['incomplete_files']:
            print(f"\n=== Incomplete Files ({len(results['incomplete_files'])}) ===")
            for file_key in results['incomplete_files'][:10]:  # Show first 10
                print(f"    - {file_key}")
            if len(results['incomplete_files']) > 10:
                print(f"    - ... and {len(results['incomplete_files'])-10} more")
        
        if results['corrupted_files']:
            print(f"\n=== Corrupted Files ({len(results['corrupted_files'])}) ===")
            for file_key in results['corrupted_files']:
                print(f"    - {file_key}")
        
        # Print final status (integrated from main execution)
        if results['summary']['completion_rate'] == 100:
            print(f"\n🎉 SUCCESS: All {results['summary']['total_expected']} pickle files are complete!")
            print(f"   Total grids ready for ML: {results['summary']['total_grids']:,}")
        else:
            print(f"\n⚠️  INCOMPLETE: {results['summary']['complete']}/{results['summary']['total_expected']} files complete")
            if results['missing_files']:
                print(f"   - {len(results['missing_files'])} files missing")
            if results['incomplete_files']:
                print(f"   - {len(results['incomplete_files'])} files incomplete")
            if results['corrupted_files']:
                print(f"   - {len(results['corrupted_files'])} files corrupted")
    
    return results


def generate_complete_dataset(
                              zeolite_types=None,
                              adsorbates_by_env=None,
                              box_grids_size=16.0,
                              box_increment=0.8,
                              feature_list=None,
                              include_zeolite=False,
                              include_solvent=True,
                              include_adsorbate=True,
                              include_identity=True,
                              check_quality=True,
                              verbose=False,
                              save_pickle=True,
                              test=False,
                              force_regenerate=False,
                              output_dir=None
                             ):
    """
    Generate complete voxel grid dataset with augmentation for all zeolites and adsorbates.
    
    This is the main function that handles the entire dataset generation pipeline:
    1. Check existing files (unless force_regenerate=True)
    2. Generate missing files
    3. Validate all files for completeness
    
    INPUTS:
        zeolite_types: list, zeolite types to process (default: from global_vars)
        adsorbates_by_env: dict, environments and their adsorbates (default: from global_vars)
        box_grids_size: float, box size in Angstrom
        box_increment: float, voxel size in Angstrom
        feature_list: list, list of features (default: from global_vars)
        include_zeolite: bool, whether to include zeolite atoms
        include_solvent: bool, whether to include solvent atoms
        include_adsorbate: bool, whether to include adsorbate atoms
        include_identity: bool, whether to include identity rotation
        check_quality: bool, whether to check augmentation quality
        verbose: bool, whether to print detailed information during generation
        save_pickle: bool, whether to save results to pickle files
        test: bool, if True, use small test dataset (FAU + 1 adsorbate)
        force_regenerate: bool, if True, regenerate even if files exist
        output_dir: str, output directory (optional)
    
    OUTPUTS:
        dict: comprehensive report on dataset generation and validation
    """
    # Use global variables as defaults
    if zeolite_types is None:
        zeolite_types = ZEOLITE_TYPES
    if adsorbates_by_env is None:
        adsorbates_by_env = ADSORBATES_BY_ENV
    
    # Test mode limits execution to the explicitly supplied zeolite/environment/
    # adsorbate selection. The caller is responsible for passing a single-item
    # configuration when only one pickle should be generated.
    if test:
        print(f"\n=== TEST MODE ENABLED ===")
        print(f"    Using test dataset: {zeolite_types} with {adsorbates_by_env}")
    
    # Create common parameters for individual adsorbate processing
    common_params = {
        'box_grids_size': box_grids_size,
        'box_increment': box_increment,
        'feature_list': feature_list,
        'include_zeolite': include_zeolite,
        'include_solvent': include_solvent,
        'include_adsorbate': include_adsorbate,
        'include_identity': include_identity,
        'check_quality': check_quality,
        'verbose': verbose,
        'save_pickle': save_pickle,
        'output_dir': output_dir,
    }
    
    print(f"\n=== STARTING COMPLETE DATASET GENERATION ===")
    print(f"    Zeolite types: {zeolite_types}")
    print(f"    Environments: {list(adsorbates_by_env.keys())}")
    print(f"    Grid parameters: size={box_grids_size}, increment={box_increment}")
    print(f"    Force regenerate: {force_regenerate}")
    print(f"    Test mode: {test}")
    
    # Check if all expected pickle files already exist (unless force regeneration)
    if not force_regenerate:
        all_exist, expected_files, missing_files = check_all_pickle_files_exist(
            zeolite_types=zeolite_types,
            adsorbates_by_env=adsorbates_by_env,
            box_grids_size=box_grids_size,
            box_increment=box_increment,
            feature_list=feature_list,
            output_dir=output_dir,
            verbose=True
        )
    else:
        all_exist = False
        expected_files = []
        missing_files = []
        print(f"\n=== Force regenerate enabled - will recreate all files ===")
    
    # Decide whether to generate files
    if all_exist and not force_regenerate:
        print(f"\n=== All {len(expected_files)} pickle files already exist ===")
        print(f"    Skipping generation. If you want to regenerate, set force_regenerate=True or delete the files.")
        print(f"    Expected files directory: {os.path.dirname(expected_files[0]) if expected_files else 'N/A'}")
    else:
        if force_regenerate:
            print(f"\n=== Force regenerating all pickle files ===")
        else:
            print(f"\n=== Missing {len(missing_files)} out of {len(expected_files)} pickle files ===")
        print(f"    Proceeding with full generation...")
        
        # Iterate over each zeolite type
        for zeolite in zeolite_types:
            print(f"--- Processing zeolite: {zeolite}")
            
            # Iterate over each environment and its adsorbates
            for env, adsorbates in adsorbates_by_env.items():
                solvent_type, pore_type = env.split('-')
                print(f"    Environment: {env} (solvent: {solvent_type}, pore: {pore_type})")
                
                # Iterate over each adsorbate
                for ads in adsorbates:
                    print(f"    >> Processing {zeolite} {env} {ads}")
                    
                    adsorbate_data, file_path = generate_and_store_adsorbate_grids(
                                                                            zeolite_type=zeolite,
                                                                            solvent_type=solvent_type,
                                                                            pore_type=pore_type,
                                                                            adsorbate=ads,
                                                                            **common_params
                                                                        )
                    
                    if file_path:
                        print(f"    >> Successfully saved: {file_path}")
                    else:
                        print(f"    >> Processing completed (not saved)")
        
        print("\n--- All datasets processed ---")
    
    # Check completeness of all pickle files (with integrated final status reporting)
    print(f"\n=== FINAL VALIDATION ===")
    completeness_results = check_all_pickles_complete(
        zeolite_types=zeolite_types,
        adsorbates_by_env=adsorbates_by_env,
        box_grids_size=box_grids_size,
        box_increment=box_increment,
        feature_list=feature_list,
        output_dir=output_dir,
        verbose=True
    )
    
    return completeness_results


def parse_command_line():
    """Parse an explicit single-system or full-dataset generation request."""
    parser = argparse.ArgumentParser(
        description="Generate augmented voxel-grid pickle files from MD snapshots."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--test', action='store_true',
        help='Generate and validate one zeolite/environment/adsorbate pickle.'
    )
    mode.add_argument(
        '--all', action='store_true',
        help='Generate all pickles configured in core/global_vars.py.'
    )
    parser.add_argument('--zeolite', help='Zeolite name required with --test.')
    parser.add_argument(
        '--environment',
        help="Environment required with --test: '<solvent>-<pore_type>'."
    )
    parser.add_argument('--adsorbate', help='Adsorbate directory required with --test.')
    parser.add_argument(
        '--force-regenerate', action='store_true',
        help='Regenerate output even if the expected pickle already exists.'
    )
    args = parser.parse_args()

    if args.test:
        missing = [
            name for name in ('zeolite', 'environment', 'adsorbate')
            if getattr(args, name) is None
        ]
        if missing:
            parser.error('--test requires ' + ', '.join(f'--{name}' for name in missing))
    return args


def run_generation_and_validation(
    test=False,
    force_regenerate=False,
    test_zeolite=None,
    test_environment=None,
    test_adsorbate=None,
):
    """Run generation and, for a single-system test, detailed voxel validation."""

    # In normal mode, None selects the complete dataset from global_vars. In
    # test mode, these single-item containers restrict the existing generation
    # pipeline to exactly the combination configured above.
    selected_zeolites = [test_zeolite] if test else None
    selected_adsorbates_by_env = (
        {test_environment: [test_adsorbate]} if test else None
    )
    
    # Generate complete dataset using the integrated function
    completeness_results = generate_complete_dataset(
        zeolite_types=selected_zeolites,
        adsorbates_by_env=selected_adsorbates_by_env,
        box_grids_size=16.0,
        box_increment=0.8,
        feature_list=None,  # Use the standard 14 atomic features.
        include_zeolite=False,
        include_solvent=True,
        include_adsorbate=True,
        include_identity=True,
        check_quality=True,
        verbose=False,  # Set to True for detailed individual file processing
        save_pickle=True,
        test=test,
        force_regenerate=force_regenerate,
    )

    # Test data validation (run in test mode if files exist)
    if test:
        print(f"\n=== VOXEL DATA VALIDATION TEST ===")
        
        if completeness_results['files_complete'] > 0:
            # Calculate expected file path
            max_bin_num = int(16.0 / 0.8)  # box_grids_size / box_increment
            num_features = 28  # 14 atomic features × 2 molecular groups
            voxel_shape_str = f"{max_bin_num}_{max_bin_num}_{max_bin_num}_{num_features}"
            
            output_dir = get_paths('dataset_cnn')
            subdir_name = f"size_16.0-box_0.8-shape_{voxel_shape_str}"
            filename = f'{test_zeolite}-{test_environment}-{test_adsorbate}-size_16.0-box_0.8-shape_{voxel_shape_str}.pkl'
            test_file_path = os.path.join(output_dir, subdir_name, filename)
            
            try:
                # Load test data
                with open(test_file_path, 'rb') as f:
                    adsorbate_data = pickle.load(f)
                
                print(f"    ✓ Loaded test file: {filename}")
                print(f"    - Data format: {adsorbate_data['metadata'].get('data_format', 'unknown')}")
                print(f"    - Grid shape: {adsorbate_data['metadata']['grid_shape']}")
                print(f"    - Total channels: {adsorbate_data['metadata']['total_channels']}")
                print(f"    - Adsorbate channels: {adsorbate_data['metadata']['adsorbate_channels']}")
                print(f"    - Solvent channels: {adsorbate_data['metadata']['solvent_channels']}")
                
                # Test voxel data - compare original vs augmented first (identity)
                test_original_voxel = adsorbate_data['snapshots'][1]['original_grid'][:,:,:,0]  # Original first channel
                test_augmented_voxel = adsorbate_data['snapshots'][1]['augmented_data'][0]['voxel_grid'][:,:,:,0]  # First augmented (identity)
                
                print(f"\n--- Channel 0 (Adsorbate atom_type_C) Validation ---")
                print(f"    Original grid shape: {test_original_voxel.shape}, min: {test_original_voxel.min():.3f}, max: {test_original_voxel.max():.3f}, mean: {test_original_voxel.mean():.3f}")
                print(f"    Augmented[0] shape: {test_augmented_voxel.shape}, min: {test_augmented_voxel.min():.3f}, max: {test_augmented_voxel.max():.3f}, mean: {test_augmented_voxel.mean():.3f}")
                print(f"    Are they identical? {np.array_equal(test_original_voxel, test_augmented_voxel)}")
                
                # Verify that adsorbate and solvent features occupy distinct groups.
                print(f"\n--- Channel Separation Validation ---")
                adsorbate_channels = test_original_voxel  # Channel 0: adsorbate atom_type_C
                solvent_channels = adsorbate_data['snapshots'][1]['original_grid'][:,:,:,14]  # Channel 14: solvent atom_type_C
                
                print(f"    Adsorbate C atoms (channel 0): {np.sum(adsorbate_channels > 0)} non-zero voxels")
                print(f"    Solvent C atoms (channel 14): {np.sum(solvent_channels > 0)} non-zero voxels")
                
                # Check that adsorbate and solvent channels are properly separated
                adsorbate_nonzero_total = np.sum(adsorbate_data['snapshots'][1]['original_grid'][:,:,:,:14] > 0)
                solvent_nonzero_total = np.sum(adsorbate_data['snapshots'][1]['original_grid'][:,:,:,14:] > 0)
                print(f"    Total adsorbate data (channels 0-13): {adsorbate_nonzero_total} non-zero entries")
                print(f"    Total solvent data (channels 14-27): {solvent_nonzero_total} non-zero entries")
                
                # Validate augmentation consistency
                print(f"\n--- Augmentation Consistency Check ---")
                first_rotation = adsorbate_data['snapshots'][1]['augmented_data'][0]
                print(f"    First rotation: {first_rotation['rotation_name']}")
                print(f"    Target energy: {first_rotation['target_interaction_energy']:.3f} eV")
                print(f"    Data format: {first_rotation.get('data_format', 'not specified')}")
                
                print(f"    ✓ Voxel-data validation completed successfully!")
                
            except Exception as e:
                print(f"    ✗ Test validation failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    ✗ No complete files found for validation")
    else:
        print(f"\n--- Skipping validation test (test mode disabled) ---")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio")
    cli_args = parse_command_line()
    run_generation_and_validation(
        test=cli_args.test,
        force_regenerate=cli_args.force_regenerate,
        test_zeolite=cli_args.zeolite,
        test_environment=cli_args.environment,
        test_adsorbate=cli_args.adsorbate,
    )
