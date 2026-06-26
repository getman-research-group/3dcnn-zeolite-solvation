# -*- coding: utf-8 -*-
"""
path.py
    This script contains all path functions.

Functions:
    get_base_paths: function to locate the base_path
    get_paths: function to find paths based on the operating system

┍━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━┑
│ System              │ Value               │
┝━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━┥
│ Linux               │ linux or linux2 (*) │
│ Windows             │ win32               │
│ Windows/Cygwin      │ cygwin              │
│ Windows/MSYS2       │ msys                │
│ Mac OS X            │ darwin              │
┕━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━┙

"""

## Importing Modules
import sys
import os
import getpass


def get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))



def get_base_paths():
    # Get the username and home directory
    username = getpass.getuser()
    
    homedir = os.path.expanduser('~')
    
    env_path = os.environ.get('ZEOLITE_SOLVATION_PATH')
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    # Set the base path based on the username and home directory
    if username == 'shi.1909' and homedir == r'C:\Users\shi.1909':    # Windows
        base_path = r"C:\Users\shi.1909\OneDrive - The Ohio State University\GitHub\3dcnn-zeolite-solvation"
    elif username == 'jiexin' and homedir == '/Users/jiexin':          # Mac OS
        base_path = "/Users/jiexin/Library/CloudStorage/OneDrive-Personal/GitHub/3dcnn-zeolite-solvation"
    elif username == 'jiexins' and homedir == '/users/PAS2536/jiexins':         # Linux
        base_path = "/fs/ess/PAS2536/jiexins/3dcnn-zeolite-solvation"
    else:
        base_path = get_repo_root()
    
    return base_path




def get_paths(path):
    
    base_path = get_base_paths()

    # Define the paths
    path_dic = {
        'simulation_path'       : os.path.join(base_path, 'md_simulations'),
        'database_path'         : os.path.join(base_path, 'database'),

        'dataset_cnn'           : os.path.join(base_path, 'dataset_cnn'),
        'dataset_gnn'           : os.path.join(base_path, 'dataset_gnn'),
        
        'output_model_cnn'      : os.path.join(base_path, 'output_model_cnn'),
        'output_model_fusion'   : os.path.join(base_path, 'output_model_fusion'),
        
        'csv_file_path'         : os.path.join(base_path, 'output_csv'),
        'output_figure_path'    : os.path.join(base_path, 'output_figures'),
    }
    
    return path_dic.get(path, "Path not found")


if __name__ == '__main__':
    print (get_paths('simulation_path'))
    print (sys.platform)
    print (get_base_paths())
    
