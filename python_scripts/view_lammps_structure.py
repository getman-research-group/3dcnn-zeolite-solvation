#!/usr/bin/env python3
"""
Simple script to visualize LAMMPS data file using ASE
"""

from ase.io import read
from ase.visualize import view

# Path to LAMMPS data file
data_file = '/Users/jiexin/Library/CloudStorage/OneDrive-TheOhioStateUniversity/GitHub/zeolite_ml_project/md_simulations/fau_ad.lammpsdata'

# Read the structure
print(f"Reading structure from: {data_file}")

# Try different atom styles (the file has 10 fields, so try 'full' style)
try:
    # Try 'full' style first (most common for molecular systems)
    atoms = read(data_file, format='lammps-data', atom_style='full')
    print("Successfully read with 'full' atom style")
except Exception as e:
    print(f"Failed with 'full' style: {e}")
    try:
        # Try 'charge' style
        atoms = read(data_file, format='lammps-data', atom_style='charge')
        print("Successfully read with 'charge' atom style")
    except Exception as e2:
        print(f"Failed with 'charge' style: {e2}")
        # Try without specifying style (auto-detect)
        atoms = read(data_file, format='lammps-data')
        print("Successfully read with auto-detected style")

# Print basic information
print(f"\nStructure information:")
print(f"  Number of atoms: {len(atoms)}")
print(f"  Chemical formula: {atoms.get_chemical_formula()}")
print(f"  Cell dimensions: {atoms.cell.cellpar()}")
print(f"  Unique elements: {set(atoms.get_chemical_symbols())}")

# Visualize
print("\nOpening ASE viewer...")
view(atoms)
