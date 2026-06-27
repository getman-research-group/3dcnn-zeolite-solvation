# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np

# Use MDAnalysis distance calculation to find water molecules near adsorbate
from MDAnalysis.analysis import distances

# Custom imports
from core.path import get_paths
from read_md_snapshot import snapshotMDAnalysis


def _parse_solvent_composition(solvent_type):
    """
    Parse solvent type and determine molecule ID ranges for different components.
    Total solvent molecules is always 1200 (mol_ids 1-1200).
    Zeolite is always mol_id 1201, adsorbate is always mol_id 1202.
    
    INPUTS:
        solvent_type: str, solvent type (e.g. "water_pure", "methanol_120_water_1080")
        
    OUTPUTS:
        dict with keys:
            - water_mol_range: tuple (start, end) for water molecule IDs
            - methanol_mol_range: tuple (start, end) for methanol molecule IDs or None
            - has_methanol: bool indicating if methanol is present
            - zeolite_mol_id: int, molecule ID for zeolite (always 1201)
            - adsorbate_mol_id: int, molecule ID for adsorbate (always 1202)
    """
    # Fixed IDs for non-solvent components
    zeolite_mol_id = 1201
    adsorbate_mol_id = 1202
    
    if solvent_type == 'water_pure':
        # Pure water system: 1-1200 mol_ids are all water
        return {
            'water_mol_range': (1, 1200),
            'methanol_mol_range': None,
            'has_methanol': False,
            'zeolite_mol_id': zeolite_mol_id,
            'adsorbate_mol_id': adsorbate_mol_id
        }
    elif solvent_type == 'methanol_120_water_1080':
        # Mixed system: 1080 water + 120 methanol = 1200 total
        return {
            'water_mol_range': (1, 1080),
            'methanol_mol_range': (1081, 1200),
            'has_methanol': True,
            'zeolite_mol_id': zeolite_mol_id,
            'adsorbate_mol_id': adsorbate_mol_id
        }
    elif solvent_type == 'methanol_240_water_960':
        # Mixed system: 960 water + 240 methanol = 1200 total
        return {
            'water_mol_range': (1, 960),
            'methanol_mol_range': (961, 1200),
            'has_methanol': True,
            'zeolite_mol_id': zeolite_mol_id,
            'adsorbate_mol_id': adsorbate_mol_id
        }
    elif solvent_type == 'methanol_600_water_600':
        # Mixed system: 600 water + 600 methanol = 1200 total
        return {
            'water_mol_range': (1, 600),
            'methanol_mol_range': (601, 1200),
            'has_methanol': True,
            'zeolite_mol_id': zeolite_mol_id,
            'adsorbate_mol_id': adsorbate_mol_id
        }
    else:
        raise ValueError(f"Unsupported solvent_type: {solvent_type}. "
                       "Supported types: 'water_pure', 'methanol_120_water_1080', "
                       "'methanol_240_water_960', 'methanol_600_water_600'")


def _is_solvent_molecule(mol_id, solvent_composition):
    """
    Check if a molecule ID corresponds to a solvent molecule (water or methanol).
    
    INPUTS:
        mol_id: int, molecule ID to check
        solvent_composition: dict, output from _parse_solvent_composition()
        
    OUTPUTS:
        bool, True if molecule is a solvent molecule
    """
    water_start, water_end = solvent_composition['water_mol_range']
    if water_start <= mol_id <= water_end:
        return True
        
    if solvent_composition['has_methanol']:
        methanol_start, methanol_end = solvent_composition['methanol_mol_range']
        if methanol_start <= mol_id <= methanol_end:
            return True
            
    return False

def extract_LJ_parameter_info(zeolite_type,
                              solvent_type,
                              pore_type,
                              adsorbate,
                              parameter,
                              verbose = False):
    
    """
    Extract specific LJ parameter (epsilon or sigma) from LAMMPS data file.
    
    INPUTS:
        zeolite_type: str, zeolite type (e.g. "FAU", "BEA", "MFI")
        solvent_type: str, solvent type (e.g. "water_pure")
        pore_type: str, pore type (e.g. "hydrophilic", "hydrophobic")
        adsorbate: str, adsorbate type (e.g. "01_methanol")
        parameter: str, LJ parameter to extract ('epsilon' or 'sigma')
        verbose: bool, whether to print detailed information
        
    OUTPUTS:
        atom_type_to_parameter: dict, mapping atom type IDs to parameter values
                               {atom_type_id: float}
    """
    
    # Validate parameter input
    if parameter not in ['epsilon', 'sigma']:
        raise ValueError(f"Parameter must be 'epsilon' or 'sigma', got '{parameter}'")
    
    # Get root directory for simulations
    md_simulations = get_paths('simulation_path')
    
    # Construct simulation directory path
    folder_name = f"{solvent_type}-{pore_type}"
    sim_dir = os.path.join(md_simulations, zeolite_type, folder_name, adsorbate)
    
    # Define path to LAMMPS data file
    path_lammpsdata = os.path.join(sim_dir, 'data_nvt_samp_new.lammpsdata')
    
    if verbose:
        print(f"\n--- Extracting LJ Parameter: {parameter} ---")
        print(f"    Reading from: {path_lammpsdata}")
    
    atom_type_to_parameter = {}
    
    # Read the LAMMPS data file
    with open(path_lammpsdata, 'r', encoding="utf-8") as lammpsdata:
        lines = lammpsdata.readlines()
        
        # Find the "Pair Coeffs" section
        pair_coeffs_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Pair Coeffs"):
                pair_coeffs_start = i + 2  # Skip the header and empty line
                break
        
        if pair_coeffs_start is None:
            raise ValueError("Pair Coeffs section not found in LAMMPS data file")
        
        if verbose:
            print(f"    Found Pair Coeffs section at line {pair_coeffs_start}")
        
        # Find the end of Pair Coeffs section
        pair_coeffs_end = None
        for i, line in enumerate(lines[pair_coeffs_start:]):
            if line.strip() == "" or not line.strip()[0].isdigit():
                pair_coeffs_end = i + pair_coeffs_start
                break
        
        if pair_coeffs_end is None:
            pair_coeffs_end = len(lines)
        
        # Parse the Pair Coeffs lines
        for line_num in range(pair_coeffs_start, pair_coeffs_end):
            line = lines[line_num].strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse line format: "1      0.1521    3.1507    # O_HOH"
            parts = line.split()
            if len(parts) >= 4 and '#' in line:
                try:
                    atom_type_id = int(parts[0])
                    epsilon = float(parts[1])
                    sigma = float(parts[2])
                    
                    # Store only the requested parameter using atom type ID as key
                    if parameter == 'epsilon':
                        atom_type_to_parameter[atom_type_id] = epsilon
                    elif parameter == 'sigma':
                        atom_type_to_parameter[atom_type_id] = sigma
                    
                    if verbose:
                        param_value = epsilon if parameter == 'epsilon' else sigma
                        print(f"        Type {atom_type_id:2d}:  {parameter} = {param_value:.3f}")
                        
                except (ValueError, IndexError) as e:
                    if verbose:
                        print(f"    Warning: Could not parse line {line_num}: {line}")
                    continue
    
    if verbose:
        print(f"    Successfully extracted {parameter} for {len(atom_type_to_parameter)} atom types")
    
    return atom_type_to_parameter



def get_adsorbate_bonds_info(zeolite_type,
                             solvent_type,
                             pore_type,
                             adsorbate,
                             adsorbate_mol_id=None,  # Optional - will be determined from solvent_type
                             verbose=False):
    """
    Get the adsorbate bonds information using snapshotMDAnalysis.
    Based on atoms distances, it will determine the bonds between atoms.
    For adsorbate atoms only.
    
    Note: Since adsorbate is fixed during MD simulation, we always use snapshot 1
    as all snapshots have identical adsorbate structure.
    
    INPUTS:
        zeolite_type: str, zeolite type (e.g. "FAU", "BEA", "MFI")
        solvent_type: str, solvent type (e.g. "water_pure", "methanol_120_water_1080")
        pore_type: str, pore type (e.g. "hydrophilic", "hydrophobic")
        adsorbate: str, adsorbate type (e.g. "01_methanol")
        adsorbate_mol_id: int, molecule ID for adsorbate (if None, determined from solvent_type)
        verbose: bool, whether to print detailed information
        
    OUTPUTS:
        bonds_info: dict with keys:
                   'bonds': list of tuples (atom_id1, atom_id2)
                   'atom_id_to_valence': dict mapping atom IDs to valence counts
                   'atom_id_to_coords': dict mapping atom IDs to coordinates
                   'atom_id_to_name': dict mapping atom IDs to atom names
                   'atom_id_to_type': dict mapping atom IDs to atom type IDs
    """
    
    # Parse solvent composition to get adsorbate mol_id
    solvent_composition = _parse_solvent_composition(solvent_type)
    if adsorbate_mol_id is None:
        adsorbate_mol_id = solvent_composition['adsorbate_mol_id']
    
    if verbose:
        print(f"\n--- Extracting Adsorbate Bonds Information (using MDAnalysis) ---")
        print(f"    System: {zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}")
        print(f"    Note: Using snapshot 1 (adsorbate is fixed during MD simulation)")
        print(f"    Target molecule ID: {adsorbate_mol_id}")
    
    # Create snapshotMDAnalysis instance (always use snapshot 1 since adsorbate is fixed)
    snapshot_mda = snapshotMDAnalysis(
        zeolite_type=zeolite_type,
        solvent_type=solvent_type,
        pore_type=pore_type,
        adsorbate=adsorbate,
        snapshot_index=1,  # Fixed to 1 since adsorbate doesn't change
        verbose=False  # Keep MDAnalysis loading quiet
    )
    
    universe = snapshot_mda.universe
    
    if verbose:
        print(f"    Successfully loaded universe with {len(universe.atoms)} atoms")
    
    # Get adsorbate atoms using MDAnalysis selection
    adsorbate_atoms = universe.select_atoms('resname ADS')
    
    if len(adsorbate_atoms) == 0:
        raise ValueError("No adsorbate atoms found! Check that adsorbate residue is named 'ADS'")
    
    if verbose:
        print(f"    Found {len(adsorbate_atoms)} adsorbate atoms")
    
    # Comprehensive bond length cutoffs in Angstrom for organic molecules in dataset
    # Based on DFT calculations and experimental data for alcohols, aldehydes, diols, etc.
    # Slightly relaxed cutoffs to avoid missing bonds due to thermal motion in MD simulations
    bond_cutoffs = {
        # C-H bonds (different hybridizations)
        ('C', 'H'): 1.25,   # sp3 C-H: ~1.09 Å, sp2 C-H: ~1.08 Å, relaxed for MD
        ('H', 'C'): 1.25,
        
        # C-C bonds (single bonds in alcohols, diols, etc.)
        ('C', 'C'): 1.70,   # C-C single: ~1.54 Å, with margin for conformational flexibility
        
        # C-O bonds (alcohols, ethers, aldehydes)
        ('C', 'O'): 1.55,   # C-O single in alcohols: ~1.43 Å, C=O in aldehydes: ~1.21 Å
        ('O', 'C'): 1.55,   # Relaxed to catch both single and double bonds safely
        
        # O-H bonds (all alcohol/diol OH groups)
        ('O', 'H'): 1.15,   # O-H in alcohols: ~0.96 Å, relaxed for MD thermal motion
        ('H', 'O'): 1.15,
        
        # C=O double bonds (aldehydes: formaldehyde, acetaldehyde, 3c_aldehyde)
        # Note: C=O (~1.21 Å) is much shorter than C-O single bond, so same cutoff works
        
        # Exclude non-bonding interactions
        ('H', 'H'): 0.0,    # H-H never bond in organic molecules
        ('O', 'O'): 2.0,    # Allow for potential peroxide bonds or unusual conformations
    }
    
    # Extract atom information from MDAnalysis
    atom_id_to_coords = {}
    atom_id_to_name = {}
    atom_id_to_type = {}
    
    for atom in adsorbate_atoms:
        atom_id = int(atom.id)  # Convert to int for consistency
        atom_id_to_coords[atom_id] = atom.position.copy()  # Coordinates in Angstrom
        atom_id_to_name[atom_id] = atom.name  # Atom name (e.g., "C_C4", "O_O2", "H_H1")
        atom_id_to_type[atom_id] = atom.type  # Atom type (already set by snapshotMDAnalysis)
        
        if verbose:
            coords = atom_id_to_coords[atom_id]
            print(f"        Atom {atom_id} ({atom.name}): [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]")
    
    # Detect bonds between adsorbate atoms
    bonds = []
    atom_ids = list(atom_id_to_coords.keys())
    
    if verbose:
        print(f"    Detecting bonds using distance cutoffs...")
        print(f"    Bond cutoffs: {bond_cutoffs}")
    
    for i, atom_id1 in enumerate(atom_ids):
        for j, atom_id2 in enumerate(atom_ids[i+1:], i+1):
            
            # Get element symbols from atom names
            element1 = atom_id_to_name[atom_id1].split('_')[0]
            element2 = atom_id_to_name[atom_id2].split('_')[0]
            
            # Calculate distance
            coords1 = atom_id_to_coords[atom_id1]
            coords2 = atom_id_to_coords[atom_id2]
            distance = np.linalg.norm(coords1 - coords2)
            
            # Check if bond exists based on cutoff - try both element orders
            bond_key1 = (element1, element2)
            bond_key2 = (element2, element1)
            
            cutoff = bond_cutoffs.get(bond_key1, bond_cutoffs.get(bond_key2, None))
            
            if cutoff is not None and distance <= cutoff:
                bonds.append((atom_id1, atom_id2))
                if verbose:
                    print(f"        Bond detected: {atom_id1}({element1}) - {atom_id2}({element2}), distance: {distance:.3f} Å")
    
    # Calculate valence for adsorbate atoms
    atom_id_to_valence = {}
    for atom_id in atom_ids:
        atom_id_to_valence[atom_id] = 0
    
    # Count bonds for each atom
    for atom_id1, atom_id2 in bonds:
        atom_id_to_valence[atom_id1] += 1
        atom_id_to_valence[atom_id2] += 1
    
    if verbose:
        print(f"    Total bonds detected: {len(bonds)}")
        print(f"\n    Valence distribution:")
        for atom_id in sorted(atom_ids):
            element = atom_id_to_name[atom_id].split('_')[0]
            valence = atom_id_to_valence[atom_id]
            print(f"        Atom {atom_id} ({element}): valence = {valence}")
    
    bonds_info = {
        'bonds': bonds,
        'atom_id_to_valence': atom_id_to_valence,
        'atom_id_to_coords': atom_id_to_coords,
        'atom_id_to_name': atom_id_to_name,
        'atom_id_to_type': atom_id_to_type,
    }
    
    return bonds_info


def extract_total_valence_info(zeolite_type,
                               solvent_type,
                               pore_type,
                               adsorbate,
                               verbose=False):
    """
    Extract total valence information for each atom ID using MDAnalysis.
    Only considers solvent (water/methanol) and adsorbate atoms, not zeolite.
    
    For adsorbate: uses distance-based bond detection
    For solvent: uses MDAnalysis bond topology information
    
    INPUTS:
        zeolite_type: str, zeolite type (e.g. "FAU", "BEA", "MFI")
        solvent_type: str, solvent type (e.g. "water_pure", "methanol_120_water_1080")
        pore_type: str, pore type (e.g. "hydrophilic", "hydrophobic")
        adsorbate: str, adsorbate type (e.g. "01_methanol")
        verbose: bool, whether to print detailed information
        
    OUTPUTS:
        atom_id_to_valence: dict, mapping atom IDs to actual valence values
                           {atom_id: valence_value}
    """
    
    if verbose:
        print(f"\n--- Extracting Total Valence Information (using MDAnalysis) ---")
        print(f"    System: {zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}")
    
    # Parse solvent composition
    solvent_composition = _parse_solvent_composition(solvent_type)
    
    # Create snapshotMDAnalysis instance (use snapshot 1 since adsorbate is fixed)
    snapshot_mda = snapshotMDAnalysis(
        zeolite_type=zeolite_type,
        solvent_type=solvent_type,
        pore_type=pore_type,
        adsorbate=adsorbate,
        snapshot_index=1,
        verbose=False
    )
    
    universe = snapshot_mda.universe
    
    if verbose:
        print(f"    Successfully loaded universe with {len(universe.atoms)} atoms")
    
    # Initialize valence dictionary
    atom_id_to_valence = {}
    
    # ====== Part 1: Handle Adsorbate Atoms ======
    if verbose:
        print(f"\n--- Processing adsorbate atoms...")
    
    # Get adsorbate atoms
    adsorbate_atoms = universe.select_atoms('resname ADS')
    
    if len(adsorbate_atoms) == 0:
        raise ValueError("No adsorbate atoms found! Check that adsorbate residue is named 'ADS'")
    
    if verbose:
        print(f"    Found {len(adsorbate_atoms)} adsorbate atoms")
    
    # Use distance-based bond detection for adsorbate
    bond_cutoffs = {
        ('C', 'H'): 1.25, ('H', 'C'): 1.25,
        ('C', 'C'): 1.70,
        ('C', 'O'): 1.55, ('O', 'C'): 1.55,
        ('O', 'H'): 1.15, ('H', 'O'): 1.15,
        ('H', 'H'): 0.0, ('O', 'O'): 2.0,
    }
    
    # Extract adsorbate atom information
    adsorbate_atom_coords = {}
    adsorbate_atom_names = {}
    adsorbate_atom_ids = []
    
    for atom in adsorbate_atoms:
        atom_id = int(atom.id)
        adsorbate_atom_ids.append(atom_id)
        adsorbate_atom_coords[atom_id] = atom.position.copy()
        adsorbate_atom_names[atom_id] = atom.name
        atom_id_to_valence[atom_id] = 0  # Initialize valence
    
    # Detect bonds in adsorbate using distance cutoffs
    adsorbate_bonds = []
    for i, atom_id1 in enumerate(adsorbate_atom_ids):
        for j, atom_id2 in enumerate(adsorbate_atom_ids[i+1:], i+1):
            # Get element symbols
            element1 = adsorbate_atom_names[atom_id1].split('_')[0]
            element2 = adsorbate_atom_names[atom_id2].split('_')[0]
            
            # Calculate distance
            coords1 = adsorbate_atom_coords[atom_id1]
            coords2 = adsorbate_atom_coords[atom_id2]
            distance = np.linalg.norm(coords1 - coords2)
            
            # Check if bond exists
            bond_key1 = (element1, element2)
            bond_key2 = (element2, element1)
            cutoff = bond_cutoffs.get(bond_key1, bond_cutoffs.get(bond_key2, None))
            
            if cutoff is not None and distance <= cutoff:
                adsorbate_bonds.append((atom_id1, atom_id2))
                atom_id_to_valence[atom_id1] += 1
                atom_id_to_valence[atom_id2] += 1
    
    if verbose:
        print(f"    Detected {len(adsorbate_bonds)} bonds in adsorbate")
        for atom_id in sorted(adsorbate_atom_ids):
            element = adsorbate_atom_names[atom_id].split('_')[0]
            valence = atom_id_to_valence[atom_id]
            print(f"        Adsorbate atom {atom_id} ({element}): valence = {valence}")
    
    # ====== Part 2: Handle Solvent Atoms ======
    if verbose:
        print(f"\n--- Processing solvent atoms...")
    
    # Get solvent atoms (water and/or methanol)
    if solvent_composition['has_methanol']:
        solvent_atoms = universe.select_atoms('resname HOH or resname MEO')
        if verbose:
            water_atoms = universe.select_atoms('resname HOH')
            methanol_atoms = universe.select_atoms('resname MEO')
            print(f"    Found {len(water_atoms)} water atoms and {len(methanol_atoms)} methanol atoms")
    else:
        solvent_atoms = universe.select_atoms('resname HOH')
        if verbose:
            print(f"    Found {len(solvent_atoms)} water atoms")
    
    if len(solvent_atoms) == 0:
        if verbose:
            print(f"    Warning: No solvent atoms found")
    else:
        # Initialize solvent atom valences and create efficient lookup set
        solvent_atom_ids = set()
        for atom in solvent_atoms:
            atom_id = int(atom.id)
            atom_id_to_valence[atom_id] = 0
            solvent_atom_ids.add(atom_id)
        
        # Extract bonds from MDAnalysis topology
        if verbose:
            print(f"    Using MDAnalysis bond topology ({len(universe.bonds)} total bonds)")
            print(f"    Processing bonds for {len(solvent_atom_ids)} solvent atoms...")
        
        # Count bonds for each solvent atom - optimized version
        bond_count = 0
        for bond in universe.bonds:
            atom1_id = int(bond.atoms[0].id)
            atom2_id = int(bond.atoms[1].id)
            
            # Check if both atoms are solvent atoms using fast set lookup
            if atom1_id in solvent_atom_ids and atom2_id in solvent_atom_ids:
                atom_id_to_valence[atom1_id] += 1
                atom_id_to_valence[atom2_id] += 1
                bond_count += 1
        
        if verbose:
            print(f"    Processed {bond_count} solvent-solvent bonds from topology")
    
    # ====== Summary ======
    if verbose:
        print(f"    Valence summary:")
        
        # Create efficient atom lookup dictionary for faster access
        atom_lookup = {int(atom.id): atom for atom in universe.atoms}
        
        # Group by residue type for better display
        residue_valence_stats = {}
        for atom_id, valence in atom_id_to_valence.items():
            # Use fast dictionary lookup instead of array search
            atom = atom_lookup[atom_id]
            resname = atom.resname
            element = atom.name.split('_')[0]
            
            if resname not in residue_valence_stats:
                residue_valence_stats[resname] = {}
            if element not in residue_valence_stats[resname]:
                residue_valence_stats[resname][element] = []
            
            residue_valence_stats[resname][element].append(valence)
        
        for resname, element_stats in residue_valence_stats.items():
            print(f"        {resname}:")
            for element, valences in element_stats.items():
                avg_valence = np.mean(valences)
                print(f"            {element}: {len(valences)} atoms, average valence = {avg_valence:.1f}")
        
        print(f"\n    Successfully extracted valence for {len(atom_id_to_valence)} atoms")
    
    return atom_id_to_valence



def extract_is_hydrophobic_info(zeolite_type,
                                solvent_type,
                                pore_type,
                                adsorbate,
                                verbose=False):
    """
    Detect hydrophobic atoms only in adsorbate molecules.
    Solvent molecules (water/methanol) are excluded as they don't contain hydrophobic atoms.
    
    Hydrophobic detection rules:
    1. C atoms are hydrophobic if they have no direct O neighbors
    2. H atoms are hydrophobic if their only neighbor is a hydrophobic C
    3. O atoms are always hydrophilic (is_hydrophobic = 0)
    
    INPUTS:
        zeolite_type: str, zeolite type (e.g. "FAU", "BEA", "MFI")
        solvent_type: str, solvent type (e.g. "water_pure", "methanol_120_water_1080")
        pore_type: str, pore type (e.g. "hydrophilic", "hydrophobic")
        adsorbate: str, adsorbate type (e.g. "01_methanol")
        verbose: bool, whether to print detailed information
        
    OUTPUTS:
        atom_id_to_is_hydrophobic: dict, mapping atom IDs to hydrophobic status
                                  {atom_id: 0 or 1}
    """
    
    # Parse solvent composition
    solvent_composition = _parse_solvent_composition(solvent_type)
    
    if verbose:
        print(f"\n--- Extracting Hydrophobic Information (Adsorbate Only) ---")
        print(f"    System: {zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}")
        print(f"    Only checking adsorbate atoms (molid={solvent_composition['adsorbate_mol_id']})")
    
    # Create single MDAnalysis universe instance to avoid duplication
    if verbose:
        print(f"    Loading MDAnalysis universe...")
    
    snapshot_mda = snapshotMDAnalysis(
        zeolite_type=zeolite_type,
        solvent_type=solvent_type,
        pore_type=pore_type,
        adsorbate=adsorbate,
        snapshot_index=1,  # Fixed since adsorbate doesn't change
        verbose=False
    )
    
    universe = snapshot_mda.universe
    
    if verbose:
        print(f"    Successfully loaded universe with {len(universe.atoms)} atoms")
    
    # Get adsorbate bonds information for connectivity analysis (using existing universe)
    adsorbate_bonds_info = get_adsorbate_bonds_info(
        zeolite_type, solvent_type, pore_type, adsorbate, verbose=verbose
    )
    
    # Build connectivity map for adsorbate atoms only
    atom_id_to_neighbors = {}
    for atom_id1, atom_id2 in adsorbate_bonds_info['bonds']:
        if atom_id1 not in atom_id_to_neighbors:
            atom_id_to_neighbors[atom_id1] = []
        if atom_id2 not in atom_id_to_neighbors:
            atom_id_to_neighbors[atom_id2] = []
        atom_id_to_neighbors[atom_id1].append(atom_id2)
        atom_id_to_neighbors[atom_id2].append(atom_id1)
    
    # Initialize hydrophobic status for adsorbate atoms only
    atom_id_to_is_hydrophobic = {}
    
    # Get adsorbate atom IDs and names
    adsorbate_atom_ids = list(adsorbate_bonds_info['atom_id_to_name'].keys())
    atom_id_to_name = adsorbate_bonds_info['atom_id_to_name']
    
    # Step 1: Identify hydrophobic C atoms (no direct O neighbors)
    hydrophobic_c_atoms = set()
    
    if verbose:
        print(f"\n    Hydrophobic atoms information for adsorbate {adsorbate}:")
    
    for atom_id in adsorbate_atom_ids:
        element = atom_id_to_name[atom_id].split('_')[0]
        
        if element == 'C':
            # Check if this C atom has any O neighbors
            neighbors = atom_id_to_neighbors.get(atom_id, [])
            neighbor_elements = [atom_id_to_name[neighbor_id].split('_')[0] for neighbor_id in neighbors]
            
            has_oxygen_neighbor = 'O' in neighbor_elements
            
            if not has_oxygen_neighbor:
                # This C is hydrophobic
                hydrophobic_c_atoms.add(atom_id)
                atom_id_to_is_hydrophobic[atom_id] = 1
                if verbose:
                    print(f"        C atom {atom_id}: hydrophobic (no O neighbors), neighbors: {neighbor_elements}")
            else:
                # This C is hydrophilic
                atom_id_to_is_hydrophobic[atom_id] = 0
                if verbose:
                    print(f"        C atom {atom_id}: hydrophilic (has O neighbor), neighbors: {neighbor_elements}")
    
    # Step 2: Classify H atoms based on their neighbors
    for atom_id in adsorbate_atom_ids:
        element = atom_id_to_name[atom_id].split('_')[0]
        
        if element == 'H':
            neighbors = atom_id_to_neighbors.get(atom_id, [])
            
            if len(neighbors) == 1:
                # H should have exactly one neighbor
                neighbor_id = neighbors[0]
                
                if neighbor_id in hydrophobic_c_atoms:
                    # H connected to hydrophobic C → hydrophobic H
                    atom_id_to_is_hydrophobic[atom_id] = 1
                    if verbose:
                        neighbor_element = atom_id_to_name[neighbor_id].split('_')[0]
                        print(f"        H atom {atom_id}: hydrophobic (connected to hydrophobic C {neighbor_id})")
                else:
                    # H connected to O or hydrophilic C → hydrophilic H
                    atom_id_to_is_hydrophobic[atom_id] = 0
                    if verbose:
                        neighbor_element = atom_id_to_name[neighbor_id].split('_')[0]
                        print(f"        H atom {atom_id}: hydrophilic (connected to {neighbor_element} {neighbor_id})")
            else:
                # Unexpected number of neighbors for H
                atom_id_to_is_hydrophobic[atom_id] = 0
                if verbose:
                    print(f"        H atom {atom_id}: hydrophilic (unexpected {len(neighbors)} neighbors)")
    
    # Step 3: All O atoms are hydrophilic
    for atom_id in adsorbate_atom_ids:
        element = atom_id_to_name[atom_id].split('_')[0]
        
        if element == 'O':
            atom_id_to_is_hydrophobic[atom_id] = 0
            if verbose:
                print(f"        O atom {atom_id}: hydrophilic (oxygen is always hydrophilic)")
    
    # Add all solvent atoms as hydrophilic (using existing universe)
    if verbose:
        print(f"    Adding solvent atoms as hydrophilic...")
    
    # Get all solvent atoms (water and/or methanol) using existing universe
    if solvent_composition['has_methanol']:
        solvent_atoms = universe.select_atoms('resname HOH or resname MEO')
    else:
        solvent_atoms = universe.select_atoms('resname HOH')
    
    # Add all solvent atoms as hydrophilic
    for atom in solvent_atoms:
        atom_id = int(atom.id)
        atom_id_to_is_hydrophobic[atom_id] = 0
    
    if verbose:
        print(f"    Added {len(solvent_atoms)} solvent atoms as hydrophilic")
    
    # Print summary statistics
    if verbose:
        print(f"\n    Hydrophobic classification summary:")
        
        # Count by element and hydrophobic status for adsorbate only
        adsorbate_element_counts = {}
        for atom_id in adsorbate_atom_ids:
            element = atom_id_to_name[atom_id].split('_')[0]
            is_hydrophobic = atom_id_to_is_hydrophobic[atom_id]
            
            if element not in adsorbate_element_counts:
                adsorbate_element_counts[element] = {'hydrophobic': 0, 'hydrophilic': 0}
            
            if is_hydrophobic:
                adsorbate_element_counts[element]['hydrophobic'] += 1
            else:
                adsorbate_element_counts[element]['hydrophilic'] += 1
        
        print(f"    Adsorbate atoms:")
        for element, counts in adsorbate_element_counts.items():
            total = counts['hydrophobic'] + counts['hydrophilic']
            print(f"        {element}: {counts['hydrophobic']} hydrophobic, {counts['hydrophilic']} hydrophilic (total: {total})")
        
        # Count solvent atoms
        solvent_count = sum(1 for atom_id, is_hydrophobic in atom_id_to_is_hydrophobic.items()
                           if atom_id not in adsorbate_atom_ids)
        print(f"    Solvent atoms: 0 hydrophobic, {solvent_count} hydrophilic (total: {solvent_count})")
        
        total_atoms = len(atom_id_to_is_hydrophobic)
        hydrophobic_count = sum(atom_id_to_is_hydrophobic.values())
        hydrophilic_count = total_atoms - hydrophobic_count
        print(f"    Overall: {hydrophobic_count} hydrophobic, {hydrophilic_count} hydrophilic atoms (total: {total_atoms})")
    
    return atom_id_to_is_hydrophobic


def extract_is_donor_acceptor(zeolite_type,
                              solvent_type,
                              pore_type,
                              adsorbate,
                              snapshot_index=1,  # Changed from snapshot to snapshot_index for consistency
                              r_cut=5.0,  # Cutoff distance in Angstrom for solvent selection
                              verbose=False):
    """
    Detect H-bond donor and acceptor atoms in the system.
    
    ⚠️  IMPORTANT: This function identifies atoms with H-bond POTENTIAL, 
        not atoms currently involved in H-bonds. Perfect for voxel grid representation.
    
    For adsorbate: all atoms are considered
    For solvent: only solvent molecules (water/methanol) within r_cut distance from any adsorbate heavy atom
    
    Donor definition: H atom with only one covalent neighbor that is O
    Acceptor definition: O atom (in adsorbate or selected solvent molecules)
    
    Supports mixed solvent systems: water + methanol cosolvent mixtures
    
    INPUTS:
        zeolite_type: str, zeolite type (e.g. "FAU", "BEA", "MFI")
        solvent_type: str, solvent type (e.g. "water_pure", "methanol_120_water_1080")
        pore_type: str, pore type (e.g. "hydrophilic", "hydrophobic")
        adsorbate: str, adsorbate type (e.g. "01_methanol")
        snapshot_index: int, snapshot index to read coordinates from
        r_cut: float, cutoff distance for selecting solvent molecules near adsorbate (Angstrom)
        verbose: bool, whether to print detailed information
        
    OUTPUTS:
        result: dict with keys:
               'atom_id_to_is_donor': dict mapping atom IDs to donor status (0 or 1)
               'atom_id_to_is_acceptor': dict mapping atom IDs to acceptor status (0 or 1)
               'selected_solvent_mol_ids': set of selected solvent molecule IDs
               'r_cut_used': float, cutoff distance used
               'snapshot_index': int, snapshot index processed
               'statistics': dict with detailed breakdown:
                   - 'adsorbate': {'donors': int, 'acceptors': int}
                   - 'water': {'donors': int, 'acceptors': int}
                   - 'methanol': {'donors': int, 'acceptors': int}
                   - 'total_solvent': {'donors': int, 'acceptors': int}
                   - 'grand_total': {'donors': int, 'acceptors': int}
    """
    
    # Parse solvent composition
    solvent_composition = _parse_solvent_composition(solvent_type)
    
    if verbose:
        print(f"\n--- Extracting H-bond Donor and Acceptor Information ---")
        print(f"    System: {zeolite_type}-{solvent_type}-{pore_type}-{adsorbate}")
        print(f"    Snapshot: {snapshot_index}")
        print(f"    Solvent selection cutoff: {r_cut} Å from adsorbate heavy atoms")
    
    # Get adsorbate bonds information for connectivity analysis (using MDAnalysis)
    adsorbate_bonds_info = get_adsorbate_bonds_info(
        zeolite_type, solvent_type, pore_type, adsorbate, verbose=False
    )
    
    # Load MD snapshot using snapshotMDAnalysis
    if verbose:
        print(f"    Loading MD snapshot {snapshot_index} using snapshotMDAnalysis...")
    
    snapshot_mda = snapshotMDAnalysis(
        zeolite_type=zeolite_type,
        solvent_type=solvent_type,
        pore_type=pore_type,
        adsorbate=adsorbate,
        snapshot_index=snapshot_index,
        verbose=False  # Set to False to avoid too much output
    )
    
    universe = snapshot_mda.universe
    
    if verbose:
        print(f"    Successfully loaded universe with {len(universe.atoms)} atoms")
    
    # Get adsorbate atoms using MDAnalysis
    adsorbate_atoms = universe.select_atoms('resname ADS')
    
    if len(adsorbate_atoms) == 0:
        raise ValueError("No adsorbate atoms found! Check that adsorbate residue is named 'ADS'")
    
    if verbose:
        print(f"    Found {len(adsorbate_atoms)} adsorbate atoms")
    
    # Find adsorbate heavy atoms (non-H atoms) for distance calculation
    adsorbate_heavy_atoms = adsorbate_atoms.select_atoms('not name H*')
    
    if len(adsorbate_heavy_atoms) == 0:
        raise ValueError("No adsorbate heavy atoms found!")
    
    if verbose:
        print(f"    Found {len(adsorbate_heavy_atoms)} adsorbate heavy atoms for distance calculation")
    
    # Get all solvent molecules (water and/or methanol)
    if solvent_composition['has_methanol']:
        solvent_atoms = universe.select_atoms('resname HOH or resname MEO')
        if verbose:
            print(f"    Found {len(universe.select_atoms('resname HOH'))} water atoms and {len(universe.select_atoms('resname MEO'))} methanol atoms")
    else:
        solvent_atoms = universe.select_atoms('resname HOH')
        if verbose:
            print(f"    Found {len(solvent_atoms)} water atoms")
    
    if len(solvent_atoms) == 0:
        if verbose:
            print(f"    Warning: No solvent atoms found")
        selected_solvent_atom_ids = []
        solvent_mol_ids_near_adsorbate = set()
    else:
        if verbose:
            print(f"    Found {len(solvent_atoms)} total solvent atoms")
        
        # Select solvent molecules within cutoff distance using MDAnalysis
        # Get solvent oxygen atoms first (both water and methanol have oxygen)
        solvent_oxygens = solvent_atoms.select_atoms('name O*')
        
        if verbose:
            print(f"    Found {len(solvent_oxygens)} solvent oxygen atoms")
        
        # Calculate distances between all solvent oxygens and all adsorbate heavy atoms
        dist_array = distances.distance_array(solvent_oxygens.positions,
                                              adsorbate_heavy_atoms.positions,
                                              box=universe.dimensions)
        
        # Find solvent oxygens that are within cutoff distance of any adsorbate heavy atom
        min_distances = np.min(dist_array, axis=1)  # Minimum distance to any adsorbate heavy atom
        close_solvent_oxygens = solvent_oxygens[min_distances <= r_cut]
        
        if verbose:
            print(f"    Found {len(close_solvent_oxygens)} solvent oxygen atoms within {r_cut} Å")
        
        # Get the molecule IDs (residue IDs) of these solvent molecules and convert to int
        solvent_mol_ids_near_adsorbate = set(int(resid) for resid in close_solvent_oxygens.resids)
        
        # Get all atoms from selected solvent molecules
        if solvent_mol_ids_near_adsorbate:
            selected_solvent_atoms = solvent_atoms.select_atoms(f'resid {" ".join(map(str, solvent_mol_ids_near_adsorbate))}')
            # Convert atom IDs to int
            selected_solvent_atom_ids = [int(atom_id) for atom_id in selected_solvent_atoms.ids]
        else:
            selected_solvent_atom_ids = []
    
    if verbose:
        print(f"    Selected {len(solvent_mol_ids_near_adsorbate)} solvent molecules within {r_cut} Å")
        print(f"    Total selected solvent atoms: {len(selected_solvent_atom_ids)}")
    
    # Build connectivity for solvent molecules (water and methanol)
    solvent_connectivity = {}
    
    # For each selected solvent molecule, find O-H bonds
    for mol_id in solvent_mol_ids_near_adsorbate:
        # Get atoms in this solvent molecule using MDAnalysis
        solvent_mol_atoms = universe.select_atoms(f'resid {mol_id}')
        
        # Check if this is water or methanol based on number of atoms
        if len(solvent_mol_atoms) == 3:
            # Water molecule: 1 O + 2 H
            o_atoms = solvent_mol_atoms.select_atoms('name O*')
            h_atoms = solvent_mol_atoms.select_atoms('name H*')
            
            if len(o_atoms) == 1 and len(h_atoms) == 2:
                # Convert to int
                o_id = int(o_atoms.ids[0])
                h1_id = int(h_atoms.ids[0])
                h2_id = int(h_atoms.ids[1])
                
                # Build connectivity: H -> O
                solvent_connectivity[h1_id] = [o_id]
                solvent_connectivity[h2_id] = [o_id]
                solvent_connectivity[o_id] = [h1_id, h2_id]
                
        elif len(solvent_mol_atoms) == 6:
            # Methanol molecule: 1 C + 1 O + 4 H
            # Only the O-H bond is relevant for H-bonding, not C-H bonds
            c_atoms = solvent_mol_atoms.select_atoms('name C*')
            o_atoms = solvent_mol_atoms.select_atoms('name O*')
            h_atoms = solvent_mol_atoms.select_atoms('name H*')
            
            if len(c_atoms) == 1 and len(o_atoms) == 1 and len(h_atoms) == 4:
                # Convert to int
                c_id = int(c_atoms.ids[0])
                o_id = int(o_atoms.ids[0])
                h_ids = [int(h_id) for h_id in h_atoms.ids]
                
                # Determine which H is bonded to O (hydroxyl H) vs C (methyl H)
                # Use distance-based detection
                o_pos = o_atoms.positions[0]
                c_pos = c_atoms.positions[0]
                
                hydroxyl_h_id = None
                methyl_h_ids = []
                
                for h_id in h_ids:
                    h_atom = solvent_mol_atoms.select_atoms(f'id {h_id}')
                    h_pos = h_atom.positions[0]
                    
                    # Check distance to O vs C
                    dist_to_o = np.linalg.norm(h_pos - o_pos)
                    dist_to_c = np.linalg.norm(h_pos - c_pos)
                    
                    if dist_to_o < dist_to_c and dist_to_o < 1.15:  # O-H bond cutoff
                        hydroxyl_h_id = h_id
                    elif dist_to_c < 1.25:  # C-H bond cutoff
                        methyl_h_ids.append(h_id)
                
                # Build connectivity only for the hydroxyl H-O bond (H-bond relevant)
                if hydroxyl_h_id is not None:
                    solvent_connectivity[hydroxyl_h_id] = [o_id]
                    solvent_connectivity[o_id] = solvent_connectivity.get(o_id, []) + [hydroxyl_h_id]
                    
                    if verbose:
                        print(f"        Methanol mol {mol_id}: O {o_id} - H {hydroxyl_h_id} (hydroxyl donor)")
        
        else:
            if verbose:
                print(f"        Warning: Unknown solvent molecule {mol_id} with {len(solvent_mol_atoms)} atoms")
    
    # Initialize donor and acceptor status
    atom_id_to_is_donor = {}
    atom_id_to_is_acceptor = {}
    
    # Process all atoms in the universe and convert to int
    for atom_id in universe.atoms.ids:
        atom_id_int = int(atom_id)  # Convert numpy.int64 to int
        atom_id_to_is_donor[atom_id_int] = 0
        atom_id_to_is_acceptor[atom_id_int] = 0
    
    # Build adsorbate connectivity map
    adsorbate_connectivity = {}
    for atom_id1, atom_id2 in adsorbate_bonds_info['bonds']:
        if atom_id1 not in adsorbate_connectivity:
            adsorbate_connectivity[atom_id1] = []
        if atom_id2 not in adsorbate_connectivity:
            adsorbate_connectivity[atom_id2] = []
        adsorbate_connectivity[atom_id1].append(atom_id2)
        adsorbate_connectivity[atom_id2].append(atom_id1)
    
    # Detect donors and acceptors in adsorbate
    donor_count_ads = 0
    acceptor_count_ads = 0
    
    for atom in adsorbate_atoms:
        atom_id = int(atom.id)  # Convert to int
        element = atom.name.split('_')[0]
        
        # Check for donor: H with only one O neighbor
        if element == 'H':
            neighbors = adsorbate_connectivity.get(atom_id, [])
            if len(neighbors) == 1:
                neighbor_id = neighbors[0]
                # Find the neighbor atom in the universe to get its element
                neighbor_atom = universe.atoms[universe.atoms.ids == neighbor_id][0]
                neighbor_element = neighbor_atom.name.split('_')[0]
                if neighbor_element == 'O':
                    atom_id_to_is_donor[atom_id] = 1
                    donor_count_ads += 1
                    if verbose:
                        print(f"        Adsorbate donor: H atom {atom_id} bonded to O atom {neighbor_id}")
        
        # Check for acceptor: O atoms
        elif element == 'O':
            atom_id_to_is_acceptor[atom_id] = 1
            acceptor_count_ads += 1
            if verbose:
                print(f"        Adsorbate acceptor: O atom {atom_id}")
    
    # Detect donors and acceptors in selected solvent molecules
    donor_count_water = 0
    acceptor_count_water = 0
    donor_count_methanol = 0
    acceptor_count_methanol = 0
    
    for atom_id in selected_solvent_atom_ids:
        # Find the atom in the universe
        atom = universe.atoms[universe.atoms.ids == atom_id][0]
        element = atom.name.split('_')[0]
        
        # Determine if this atom belongs to water or methanol
        atom_resname = atom.resname
        is_water = (atom_resname == 'HOH')
        is_methanol = (atom_resname == 'MEO')
        
        # Check for donor: H with only one O neighbor
        if element == 'H':
            neighbors = solvent_connectivity.get(atom_id, [])
            if len(neighbors) == 1:
                neighbor_id = neighbors[0]
                neighbor_atom = universe.atoms[universe.atoms.ids == neighbor_id][0]
                neighbor_element = neighbor_atom.name.split('_')[0]
                if neighbor_element == 'O':
                    atom_id_to_is_donor[atom_id] = 1
                    if is_water:
                        donor_count_water += 1
                        if verbose:
                            print(f"        Water donor: H atom {atom_id} bonded to O atom {neighbor_id}")
                    elif is_methanol:
                        donor_count_methanol += 1
                        if verbose:
                            print(f"        Methanol donor: H atom {atom_id} bonded to O atom {neighbor_id}")
        
        # Check for acceptor: O atoms
        elif element == 'O':
            atom_id_to_is_acceptor[atom_id] = 1
            if is_water:
                acceptor_count_water += 1
                if verbose:
                    print(f"        Water acceptor: O atom {atom_id}")
            elif is_methanol:
                acceptor_count_methanol += 1
                if verbose:
                    print(f"        Methanol acceptor: O atom {atom_id}")
    
    # Calculate totals
    total_solvent_donors = donor_count_water + donor_count_methanol
    total_solvent_acceptors = acceptor_count_water + acceptor_count_methanol
    
    if verbose:
        print(f"\n    H-bond potential summary:")
        print(f"        Adsorbate: {donor_count_ads} donors, {acceptor_count_ads} acceptors")
        
        if solvent_composition['has_methanol']:
            # Mixed solvent system
            print(f"        Water: {donor_count_water} donors, {acceptor_count_water} acceptors")
            print(f"        Methanol: {donor_count_methanol} donors, {acceptor_count_methanol} acceptors")
            print(f"        Total solvent: {total_solvent_donors} donors, {total_solvent_acceptors} acceptors")
        else:
            # Pure water system
            print(f"        Water: {donor_count_water} donors, {acceptor_count_water} acceptors")
        
        print(f"        Grand total: {donor_count_ads + total_solvent_donors} donors, {acceptor_count_ads + total_solvent_acceptors} acceptors")
    
    result = {
        'atom_id_to_is_donor': atom_id_to_is_donor,
        'atom_id_to_is_acceptor': atom_id_to_is_acceptor,
        'selected_solvent_mol_ids': solvent_mol_ids_near_adsorbate,
        'r_cut_used': r_cut,
        'snapshot_index': snapshot_index,
        # Detailed statistics
        'statistics': {
            'adsorbate': {'donors': donor_count_ads, 'acceptors': acceptor_count_ads},
            'water': {'donors': donor_count_water, 'acceptors': acceptor_count_water},
            'methanol': {'donors': donor_count_methanol, 'acceptors': acceptor_count_methanol},
            'total_solvent': {'donors': total_solvent_donors, 'acceptors': total_solvent_acceptors},
            'grand_total': {'donors': donor_count_ads + total_solvent_donors, 'acceptors': acceptor_count_ads + total_solvent_acceptors}
        }
    }
    
    return result

if __name__ == "__main__":
    
    # Example usage - test with mixed solvent system
    zeolite_type = 'FAU'
    solvent_type = 'methanol_120_water_1080'  # Test with mixed solvent
    pore_type = 'hydrophilic'
    adsorbate = '01_methanol'
    
    # print("=== Testing LJ Parameter Extraction ===")
    # atom_type_to_epsilon = extract_LJ_parameter_info(zeolite_type,
    #                                                  solvent_type,
    #                                                  pore_type,
    #                                                  adsorbate,
    #                                                  parameter = 'epsilon',
    #                                                  verbose=True)
    
    # atom_type_to_sigma = extract_LJ_parameter_info(zeolite_type,
    #                                                solvent_type,
    #                                                pore_type,
    #                                                adsorbate,
    #                                                parameter = 'sigma',
    #                                                verbose=True)
    
        
    # # Get adsorbate bonds information for connectivity analysis (now using MDAnalysis)
    # adsorbate_bonds_info = get_adsorbate_bonds_info(zeolite_type, solvent_type, pore_type, adsorbate, verbose=True)
    
    
    # print("\n=== Testing Total Valence Information ===")
    # # Test total valence extraction
    # atom_id_to_valence = extract_total_valence_info(zeolite_type,
    #                                                solvent_type,
    #                                                pore_type,
    #                                                adsorbate,
    #                                                verbose=True)

    
    # print("\n=== Testing Hydrophobic Information ===")
    # # Test hydrophobic classification
    # atom_id_to_is_hydrophobic = extract_is_hydrophobic_info(zeolite_type,
    #                                                        solvent_type,
    #                                                        pore_type,
    #                                                        adsorbate,
    #                                                        verbose=True)
    
    print("\n=== Testing H-bond Donor/Acceptor Information ===")
    
    # Test 1: Pure water system
    print("\n--- Test 1: Pure Water System ---")
    solvent_type = 'water_pure'
    hbond_capacity_info = extract_is_donor_acceptor(zeolite_type,
                                           solvent_type,
                                           pore_type,
                                           adsorbate,
                                           snapshot_index=1,
                                           r_cut=5.0,
                                           verbose=True)
    
    # Test 2: Mixed solvent system
    print("\n--- Test 2: Mixed Solvent System ---")
    solvent_type = 'methanol_120_water_1080'
    hbond_capacity_info = extract_is_donor_acceptor(zeolite_type,
                                           solvent_type,
                                           pore_type,
                                           adsorbate,
                                           snapshot_index=1,
                                           r_cut=5.0,
                                           verbose=True)




