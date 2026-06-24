


import pickle

# Function to Load Pickle Given a Pickle Directory

def load_pickle(pickle_path):
    '''
    This function loads a pickle file.
    
    INPUTS:
        pickle_path: str
            Path to the pickle file.
    
    OUTPUTS:
        results: object
            Results from the pickle file.
    '''
   
    ## Loading The Data
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
        
    return results