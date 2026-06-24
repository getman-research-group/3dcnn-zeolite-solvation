import os
import pickle
import torch
from core.path import get_paths

def load_pkl_file(pkl_name: str):
    """
    Simple function to load and inspect a pickle file
    
    Args:
        pkl_name: Name of the pkl file to load
    """
    model_dir = get_paths("output_model_cnn")
    pkl_path = os.path.join(model_dir, pkl_name)
    
    if not os.path.exists(pkl_path):
        print(f"Error: File not found: {pkl_path}")
        return None
    
    try:
        with open(pkl_path, 'rb') as f:
            # Handle GPU-trained models on CPU machines
            if torch.cuda.is_available():
                data = pickle.load(f)
            else:
                original_load = torch.load
                torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, map_location='cpu')
                try:
                    data = pickle.load(f)
                finally:
                    torch.load = original_load
        
        print(f"Successfully loaded: {pkl_name}")
        print(f"Keys in pickle file: {list(data.keys())}")
        
        # Show basic info
        if 'model_storage' in data:
            print(f"Number of folds: {len(data['model_storage'])}")
        
        if 'data_info' in data:
            print(f"Data info: {data['data_info']}")
        
        if 'training_config' in data:
            print(f"Training config: {data['training_config']}")
        
        return data
        
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

if __name__ == "__main__":
    # Specify the pickle file name here
    pkl_name = "model_2_8-pore_type-2518682-epochs_100-bs_32-lr_0.0002-grid_16.0_0.8.pkl"
    
    print(f"Loading pickle file: {pkl_name}")
    data = load_pkl_file(pkl_name)
