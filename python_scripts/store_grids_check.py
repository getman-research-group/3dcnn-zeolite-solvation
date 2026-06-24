import os
import pickle
import numpy as np
from core.path import get_paths


if __name__ == "__main__":
    
    # Set pickle file path
    pickle_folder = os.path.join(get_paths('dataset_cnn'), 'size_20.0-box_1.0-shape_20_20_20_28')
    pickle_name = 'FAU-methanol_240_water_960-hydrophilic-08_01_ethene_glycol-size_20.0-box_1.0-shape_20_20_20_28.pkl'
    pickle_file_path = os.path.join(pickle_folder, pickle_name)

    print(f"Loading pickle file: {pickle_file_path}")

    # Check if file exists
    if not os.path.exists(pickle_file_path):
        print(f"Error: Pickle file not found: {pickle_file_path}")
    else:
        # Load pickle file
        with open(pickle_file_path, 'rb') as f:
            adsorbate_data = pickle.load(f)
        
        print(f"Pickle file loaded successfully!")
        print(f"Data format: {adsorbate_data['metadata']['data_format']}")
        print(f"Available snapshots: {list(adsorbate_data['snapshots'].keys())}")
        
        # Get first voxel from first snapshot (first augmented voxel)
        first_snapshot_id = min(adsorbate_data['snapshots'].keys())
        first_voxel = adsorbate_data['snapshots'][first_snapshot_id]['augmented_data'][0]['voxel_grid']
        print(f"Grid shape: {first_voxel.shape}")
        
        print(f"First snapshot ID: {first_snapshot_id}")
        print(f"First voxel shape: {first_voxel.shape}")
        print(f"First voxel rotation: {adsorbate_data['snapshots'][first_snapshot_id]['augmented_data'][0]['rotation_name']}")
        
        # Generate channel variables for Spyder Variable Explorer
        # Adsorbate channels (channels 0-13)
        test_voxel_1_adsorbate_atom_type_C_pickle = first_voxel[:, :, :, 0]
        test_voxel_2_adsorbate_atom_type_H_pickle = first_voxel[:, :, :, 1]
        test_voxel_3_adsorbate_atom_type_O_pickle = first_voxel[:, :, :, 2]
        test_voxel_4_adsorbate_is_hydrophobic_pickle = first_voxel[:, :, :, 3]
        test_voxel_5_adsorbate_is_donor_pickle = first_voxel[:, :, :, 4]
        test_voxel_6_adsorbate_is_acceptor_pickle = first_voxel[:, :, :, 5]
        test_voxel_7_adsorbate_is_hbonded_pickle = first_voxel[:, :, :, 6]
        test_voxel_8_adsorbate_is_hbonded_donor_pickle = first_voxel[:, :, :, 7]
        test_voxel_9_adsorbate_is_hbonded_acceptor_pickle = first_voxel[:, :, :, 8]
        test_voxel_10_adsorbate_atom_mass_pickle = first_voxel[:, :, :, 9]
        test_voxel_11_adsorbate_partial_charge_pickle = first_voxel[:, :, :, 10]
        test_voxel_12_adsorbate_valence_pickle = first_voxel[:, :, :, 11]
        test_voxel_13_adsorbate_lj_epsilon_pickle = first_voxel[:, :, :, 12]
        test_voxel_14_adsorbate_lj_sigma_pickle = first_voxel[:, :, :, 13]

        # Solvent channels (channels 14-27)
        test_voxel_15_solvent_atom_type_C_pickle = first_voxel[:, :, :, 14]
        test_voxel_16_solvent_atom_type_H_pickle = first_voxel[:, :, :, 15]
        test_voxel_17_solvent_atom_type_O_pickle = first_voxel[:, :, :, 16]
        test_voxel_18_solvent_is_hydrophobic_pickle = first_voxel[:, :, :, 17]
        test_voxel_19_solvent_is_donor_pickle = first_voxel[:, :, :, 18]
        test_voxel_20_solvent_is_acceptor_pickle = first_voxel[:, :, :, 19]
        test_voxel_21_solvent_is_hbonded_pickle = first_voxel[:, :, :, 20]
        test_voxel_22_solvent_is_hbonded_donor_pickle = first_voxel[:, :, :, 21]
        test_voxel_23_solvent_is_hbonded_acceptor_pickle = first_voxel[:, :, :, 22]
        test_voxel_24_solvent_atom_mass_pickle = first_voxel[:, :, :, 23]
        test_voxel_25_solvent_partial_charge_pickle = first_voxel[:, :, :, 24]
        test_voxel_26_solvent_valence_pickle = first_voxel[:, :, :, 25]
        test_voxel_27_solvent_lj_epsilon_pickle = first_voxel[:, :, :, 26]
        test_voxel_28_solvent_lj_sigma_pickle = first_voxel[:, :, :, 27]
        
        # Additional useful variables
        metadata = adsorbate_data['metadata']
        target_interaction_energy = adsorbate_data['snapshots'][first_snapshot_id]['target_interaction_energy']
        feature_channel_mapping = adsorbate_data['metadata']['feature_channel_mapping']
        
        # Print channel information
        print("\n=== Channel Information ===")
        print(f"Adsorbate channels: {metadata['adsorbate_channels']}")
        print(f"Solvent channels: {metadata['solvent_channels']}")
        print(f"Target interaction energy: {target_interaction_energy}")
        
        # Print statistics for each channel
        print("\n=== Channel Statistics ===")
        for i in range(first_voxel.shape[3]):
            channel_data = first_voxel[:, :, :, i]
            non_zero_count = np.count_nonzero(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            mean_val = np.mean(channel_data)
            print(f"Channel {i:2d}: non-zero={non_zero_count:4d}, min={min_val:8.3f}, max={max_val:8.3f}, mean={mean_val:8.3f}")
        
        print(f"\n=== Variables created for Spyder Variable Explorer ===")
        print(f"All test_voxel_* variables are ready for inspection in Variable Explorer")
        print(f"Additional variables: metadata, target_interaction_energy, feature_channel_mapping")   