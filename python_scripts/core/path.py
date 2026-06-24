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



def get_base_paths():
    # Get the username and home directory
    username = getpass.getuser()
    
    homedir = os.path.expanduser('~')
    

    # Set the base path based on the username and home directory
    if username == 'shi.1909' and homedir == r'C:\Users\shi.1909':    # Windows
        base_path = r"C:\Users\shi.1909\OneDrive - The Ohio State University\GitHub\zeolite_ml_project"
    elif username == 'jiexin' and homedir == '/Users/jiexin':          # Mac OS
        base_path = "/Users/jiexin/Library/CloudStorage/OneDrive-TheOhioStateUniversity/GitHub/zeolite_ml_project"
    elif username == 'jiexins' and homedir == '/users/PAS2536/jiexins':         # Linux
        base_path = "/fs/ess/PAS2536/jiexins/zeolite_ml_project"
    else:
        raise ValueError("Unsupported user or home directory")
    
    return base_path




def get_paths(path):
    
    # Set the base path based on the operating system
    if sys.platform.startswith('linux'):
        base_path = "/fs/ess/PAS2536/jiexins/zeolite_ml_project"
    elif sys.platform.startswith('darwin'):
        base_path = "/Users/jiexin/Library/CloudStorage/OneDrive-TheOhioStateUniversity/GitHub/zeolite_ml_project"
    elif sys.platform.startswith('win'):
        base_path = r"C:\Users\shi.1909\OneDrive - The Ohio State University\GitHub\zeolite_ml_project"
    else:
        raise ValueError("Unsupported operating system")

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
    