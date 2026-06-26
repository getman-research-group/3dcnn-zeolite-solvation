# -*- coding: utf-8 -*-
"""
extract_hbonds.py
Detect hydrogen bonds between adsorbate and water molecules using MDAnalysis HydrogenBondAnalysis.
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis import transformations
from MDAnalysis.lib.distances import distance_array

from core.path import get_paths
from read_md_snapshot import snapshotMDAnalysis
from core.global_vars import ZEOLITE_TYPES, ADSORBATES_BY_ENV

# Suppress specific warnings at the beginning
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.Application")

# Filter out specific MDAnalysis hydrogen bond warnings
warnings.filterwarnings("ignore",
                       message="No hydrogen bonds were found given angle*",
                       category=UserWarning,
                       module="MDAnalysis.analysis.hydrogenbonds.hbond_analysis")

class HydrogenBondDetector:
    """
    Detect hydrogen bonds between adsorbate and water molecules using MDAnalysis HydrogenBondAnalysis.
    """
    
    def __init__(self,
                 zeolite_type: str = 'FAU',
                 solvent_type: str = 'water_pure',
                 pore_type: str = 'hydrophilic',
                 adsorbate: str = '03_01_1_3_propanediol',
                 snapshot_index: int = 1,
                 d_a_cutoff: float = 3.5,
                 d_h_cutoff: float = 1.2,
                 d_h_a_angle_cutoff: float = 130.0,
                 verbose: bool = False):
        
        self.zeolite_type = zeolite_type
        self.solvent_type = solvent_type
        self.pore_type = pore_type
        self.adsorbate = adsorbate
        self.snapshot_index = snapshot_index
        self.verbose = verbose
        
        # H-bond parameters
        self.d_a_cutoff = d_a_cutoff
        self.d_h_cutoff = d_h_cutoff
        self.d_h_a_angle_cutoff = d_h_a_angle_cutoff
        
        # Load MD snapshot using existing class
        self.snapshot_mda = snapshotMDAnalysis(
            zeolite_type=zeolite_type,
            solvent_type=solvent_type,
            pore_type=pore_type,
            adsorbate=adsorbate,
            snapshot_index=snapshot_index,
            verbose=False,
        )
        
        self.universe = self.snapshot_mda.universe
        
        # Analyze solvent composition from MDAnalysis universe
        self.solvent_composition = self._analyze_solvent_composition()
        
        # Define atom selections
        self.define_atom_selections()
        
        # Detect hydrogen bonds (with warnings suppressed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
            self.hbond_results = self.detect_hydrogen_bonds()
        
        # Calculate and store hydrogen bond properties as instance variables
        self.atom_hbond_properties = self.get_atom_hbond_properties()
        self.hbonded_atoms = {atom_id: props for atom_id, props in self.atom_hbond_properties.items()
                             if props['is_hbonded']}
        
        if self.verbose:
            self.print_hbond_summary()

    def _analyze_solvent_composition(self):
        """
        Analyze solvent composition directly from MDAnalysis universe.
        This is more flexible than hardcoded molecule ID ranges.
        
        Returns:
            dict with keys:
                - has_water: bool indicating if water molecules are present
                - has_methanol: bool indicating if methanol molecules are present  
                - water_count: int number of water molecules
                - methanol_count: int number of methanol molecules
                - adsorbate_resnames: list of adsorbate residue names
                - zeolite_resnames: list of zeolite residue names
        """
        # Get all unique residue names in the system
        all_resnames = set(self.universe.residues.resnames)
        
        # Identify different molecular types based on residue names
        water_resnames = [name for name in all_resnames if 'HOH' in name]
        methanol_resnames = [name for name in all_resnames if 'MEO' in name] 
        adsorbate_resnames = [name for name in all_resnames if 'ADS' in name]
        zeolite_resnames = [name for name in all_resnames if name not in water_resnames + methanol_resnames + adsorbate_resnames]
        
        # Count molecules
        water_count = len(self.universe.select_atoms("resname " + " or resname ".join(water_resnames)).residues) if water_resnames else 0
        methanol_count = len(self.universe.select_atoms("resname " + " or resname ".join(methanol_resnames)).residues) if methanol_resnames else 0
        
        composition = {
            'has_water': water_count > 0,
            'has_methanol': methanol_count > 0,
            'water_count': water_count,
            'methanol_count': methanol_count,
            'adsorbate_resnames': adsorbate_resnames,
            'zeolite_resnames': zeolite_resnames,
            'water_resnames': water_resnames,
            'methanol_resnames': methanol_resnames
        }
        
        if self.verbose:
            print(f"\n--- Solvent Composition Analysis ---")
            print(f"    All residue names found: {sorted(all_resnames)}")
            print(f"    Water molecules: {water_count} (resnames: {water_resnames})")
            print(f"    Methanol molecules: {methanol_count} (resnames: {methanol_resnames})")
            print(f"    Adsorbate resnames: {adsorbate_resnames}")
            print(f"    Zeolite resnames: {zeolite_resnames}")
        
        return composition


    def define_atom_selections(self):
        """Define atom selections for solvent (water/methanol) and adsorbate molecules."""
        
        # Adsorbate selections (always ADS) - using wildcards
        self.adsorbate_donors_sel = "resname ADS and name O*"
        self.adsorbate_hydrogens_sel = "resname ADS and name H*"
        self.adsorbate_acceptors_sel = "resname ADS and name O*"
        
        # Water selections (always HOH) - using wildcards
        self.water_donors_sel = "resname HOH and name O*"
        self.water_hydrogens_sel = "resname HOH and name H*"
        self.water_acceptors_sel = "resname HOH and name O*"
        
        # Methanol selections (MEO if present) - using wildcards
        if self.solvent_composition['has_methanol']:
            self.methanol_donors_sel = "resname MEO and name O*"
            self.methanol_hydrogens_sel = "resname MEO and name H*"
            self.methanol_acceptors_sel = "resname MEO and name O*"
        else:
            self.methanol_donors_sel = "name NonExistent"
            self.methanol_hydrogens_sel = "name NonExistent"
            self.methanol_acceptors_sel = "name NonExistent"
        
        # Get atom groups
        self.adsorbate_atoms = self.universe.select_atoms("resname ADS")
        self.water_atoms = self.universe.select_atoms("resname HOH")
        
        if self.solvent_composition['has_methanol']:
            self.methanol_atoms = self.universe.select_atoms("resname MEO")
        else:
            self.methanol_atoms = self.universe.select_atoms("name NonExistent")
        
        # Get specific atom groups for debugging
        self.adsorbate_donors = self.universe.select_atoms(self.adsorbate_donors_sel)
        self.adsorbate_hydrogens = self.universe.select_atoms(self.adsorbate_hydrogens_sel)
        self.adsorbate_acceptors = self.universe.select_atoms(self.adsorbate_acceptors_sel)
        
        self.water_donors = self.universe.select_atoms(self.water_donors_sel)
        self.water_hydrogens = self.universe.select_atoms(self.water_hydrogens_sel)
        self.water_acceptors = self.universe.select_atoms(self.water_acceptors_sel)
        
        if self.verbose:
            print(f"\n--- Atom Selections Defined ---")
            print(f"    Solvent type: {self.solvent_type}")
            print(f"    Has water: {self.solvent_composition['has_water']} ({self.solvent_composition['water_count']} molecules)")
            print(f"    Has methanol: {self.solvent_composition['has_methanol']} ({self.solvent_composition['methanol_count']} molecules)")
            print(f"    Water atoms: {len(self.water_atoms)}")
            if self.solvent_composition['has_methanol']:
                print(f"    Methanol atoms: {len(self.methanol_atoms)}")
            print(f"    Adsorbate atoms: {len(self.adsorbate_atoms)}")
            print(f"    Adsorbate donors (O_ADS): {len(self.adsorbate_donors)}")
            print(f"    Adsorbate hydrogens (H_ADS): {len(self.adsorbate_hydrogens)}")
            print(f"    Adsorbate acceptors (O_ADS): {len(self.adsorbate_acceptors)}")
            
            # Show atom details for adsorbate
            print(f"\n    Adsorbate atom details:")
            for atom in self.adsorbate_atoms:
                print(f"        Atom {atom.id}: name={atom.name}, resname={atom.resname}, resid={atom.resid}")
    

    def detect_hydrogen_bonds(self) -> np.ndarray:
        """
        Detect hydrogen bonds between adsorbate and solvent molecules using HydrogenBondAnalysis.
        Split detection into separate donor-acceptor combinations for comprehensive coverage:
        1. Adsorbate as donor, solvent as acceptor
        2. Solvent as donor, adsorbate as acceptor
        Returns numpy array of hydrogen bond data.
        """
        if self.verbose:
            print(f"\n--- Detecting Hydrogen Bonds ---")
            print(f"    D-A distance cutoff: {self.d_a_cutoff} Å")
            print(f"    D-H distance cutoff: {self.d_h_cutoff} Å")
            print(f"    D-H-A angle cutoff: {self.d_h_a_angle_cutoff}°")
            print(f"    Solvent type: {self.solvent_type}")
        
        # Set current frame (snapshot)
        self.universe.trajectory[0]  # Use the first (and only) frame from the snapshot
        
        all_hbond_results = []
        
        # Helper function to detect H-bonds for a specific donor-acceptor pair
        def detect_specific_hbonds(donors_sel, hydrogens_sel, acceptors_sel, description):
            if self.verbose:
                print(f"\n    {description}    ")
                print(f"    Donors: {donors_sel}")
                print(f"    Hydrogens: {hydrogens_sel}")
                print(f"    Acceptors: {acceptors_sel}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
                
                hbonds = HydrogenBondAnalysis(
                    universe=self.universe,
                    donors_sel=donors_sel,
                    hydrogens_sel=hydrogens_sel,
                    acceptors_sel=acceptors_sel,
                    d_a_cutoff=self.d_a_cutoff,
                    d_h_cutoff=self.d_h_cutoff,
                    d_h_a_angle_cutoff=self.d_h_a_angle_cutoff,
                    update_selections=False
                )
                
                hbonds.run(start=0, stop=1, step=1, verbose=False)
            
            results = hbonds.results.hbonds
            if self.verbose:
                print(f"    Found: {len(results)} H-bonds")
            
            return results
        
        # 1. Water-related hydrogen bonds if water is present
        if self.solvent_composition['has_water']:
            if self.verbose:
                print(f"\n=== WATER HYDROGEN BONDS ===")
            
            # 1a. Adsorbate as donor, water as acceptor
            ads_donor_water_acceptor = detect_specific_hbonds(
                donors_sel=self.adsorbate_donors_sel,
                hydrogens_sel=self.adsorbate_hydrogens_sel,
                acceptors_sel=self.water_acceptors_sel,
                description="Adsorbate (donor) → Water (acceptor)"
            )
            if len(ads_donor_water_acceptor) > 0:
                all_hbond_results.extend(ads_donor_water_acceptor)
            
            # 1b. Water as donor, adsorbate as acceptor
            water_donor_ads_acceptor = detect_specific_hbonds(
                donors_sel=self.water_donors_sel,
                hydrogens_sel=self.water_hydrogens_sel,
                acceptors_sel=self.adsorbate_acceptors_sel,
                description="Water (donor) → Adsorbate (acceptor)"
            )
            if len(water_donor_ads_acceptor) > 0:
                all_hbond_results.extend(water_donor_ads_acceptor)
        
        # 2. Methanol-related hydrogen bonds if methanol is present
        if self.solvent_composition['has_methanol']:
            if self.verbose:
                print(f"\n=== METHANOL HYDROGEN BONDS ===")
            
            # 2a. Adsorbate as donor, methanol as acceptor
            ads_donor_methanol_acceptor = detect_specific_hbonds(
                donors_sel=self.adsorbate_donors_sel,
                hydrogens_sel=self.adsorbate_hydrogens_sel,
                acceptors_sel=self.methanol_acceptors_sel,
                description="Adsorbate (donor) → Methanol (acceptor)"
            )
            if len(ads_donor_methanol_acceptor) > 0:
                all_hbond_results.extend(ads_donor_methanol_acceptor)
            
            # 2b. Methanol as donor, adsorbate as acceptor
            methanol_donor_ads_acceptor = detect_specific_hbonds(
                donors_sel=self.methanol_donors_sel,
                hydrogens_sel=self.methanol_hydrogens_sel,
                acceptors_sel=self.adsorbate_acceptors_sel,
                description="Methanol (donor) → Adsorbate (acceptor)"
            )
            if len(methanol_donor_ads_acceptor) > 0:
                all_hbond_results.extend(methanol_donor_ads_acceptor)
        
        # 3. Convert to numpy array and summarize results
        if len(all_hbond_results) > 0:
            hbond_results = np.array(all_hbond_results)
        else:
            hbond_results = np.empty((0, 6))  # Empty array with correct shape
        
        if self.verbose:
            print(f"\n=== FINAL SUMMARY ===")
            print(f"    Total adsorbate-solvent H-bonds: {len(hbond_results)}")
            
            if self.solvent_composition['has_water']:
                water_count = len(ads_donor_water_acceptor) + len(water_donor_ads_acceptor)
                print(f"    Water-related H-bonds: {water_count}")
                print(f"        Adsorbate→Water: {len(ads_donor_water_acceptor)}")
                print(f"        Water→Adsorbate: {len(water_donor_ads_acceptor)}")
            
            if self.solvent_composition['has_methanol']:
                methanol_count = len(ads_donor_methanol_acceptor) + len(methanol_donor_ads_acceptor)
                print(f"    Methanol-related H-bonds: {methanol_count}")
                print(f"        Adsorbate→Methanol: {len(ads_donor_methanol_acceptor)}")
                print(f"        Methanol→Adsorbate: {len(methanol_donor_ads_acceptor)}")
        
        return hbond_results
    
    def get_atom_hbond_properties(self) -> Dict[int, Dict[str, bool]]:
        """
        Return dictionary mapping atom IDs to their hydrogen bond properties.
        
        Returns:
            Dict with structure: {atom_id: {'is_hbonded': bool, 'is_hbonded_donor': bool, 'is_hbonded_acceptor': bool}}
        """
        # Initialize all solvent and adsorbate atoms with no H-bond properties
        atom_hbond_dict = {}
        
        # Initialize all water atoms
        for atom in self.water_atoms:
            atom_hbond_dict[int(atom.id)] = {
                'is_hbonded': False,
                'is_hbonded_donor': False,
                'is_hbonded_acceptor': False
            }
        
        # Initialize all methanol atoms if present
        if self.solvent_composition['has_methanol']:
            for atom in self.methanol_atoms:
                atom_hbond_dict[int(atom.id)] = {
                    'is_hbonded': False,
                    'is_hbonded_donor': False,
                    'is_hbonded_acceptor': False
                }
        
        # Initialize all adsorbate atoms - convert np.int64 to int
        for atom in self.adsorbate_atoms:
            atom_hbond_dict[int(atom.id)] = {
                'is_hbonded': False,
                'is_hbonded_donor': False,
                'is_hbonded_acceptor': False
            }
        
        # Process detected hydrogen bonds
        if len(self.hbond_results) > 0:
            for hbond in self.hbond_results:
                # hbond format: [frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle]
                # Note: MDAnalysis uses 0-based indices, but atom IDs are 1-based
                donor_idx = int(hbond[1])
                hydrogen_idx = int(hbond[2])
                acceptor_idx = int(hbond[3])
                
                # Convert indices to atom IDs and ensure they are Python int
                donor_id = int(self.universe.atoms[donor_idx].id)
                hydrogen_id = int(self.universe.atoms[hydrogen_idx].id)
                acceptor_id = int(self.universe.atoms[acceptor_idx].id)
                
                # Mark donor atom (heavy atom bonded to hydrogen)
                if donor_id in atom_hbond_dict:
                    atom_hbond_dict[donor_id]['is_hbonded'] = True
                    atom_hbond_dict[donor_id]['is_hbonded_donor'] = True
                
                # Mark hydrogen atom (part of donor)
                if hydrogen_id in atom_hbond_dict:
                    atom_hbond_dict[hydrogen_id]['is_hbonded'] = True
                    # Hydrogen is part of donor, not donor itself in this classification
                
                # Mark acceptor atom
                if acceptor_id in atom_hbond_dict:
                    atom_hbond_dict[acceptor_id]['is_hbonded'] = True
                    atom_hbond_dict[acceptor_id]['is_hbonded_acceptor'] = True
        
        return atom_hbond_dict
    
    def print_hbond_summary(self):
        """Print summary of hydrogen bond detection results."""
        atom_props = self.get_atom_hbond_properties()
        
        # Count different categories
        total_hbonded = sum(1 for props in atom_props.values() if props['is_hbonded'])
        total_donors = sum(1 for props in atom_props.values() if props['is_hbonded_donor'])
        total_acceptors = sum(1 for props in atom_props.values() if props['is_hbonded_acceptor'])
        
        # Separate by molecule type
        water_atom_ids = set(atom.id for atom in self.water_atoms)
        adsorbate_atom_ids = set(atom.id for atom in self.adsorbate_atoms)
        
        water_hbonded = sum(1 for atom_id, props in atom_props.items()
                           if props['is_hbonded'] and atom_id in water_atom_ids)
        ads_hbonded = sum(1 for atom_id, props in atom_props.items()
                         if props['is_hbonded'] and atom_id in adsorbate_atom_ids)
        
        # Add methanol and other solvent statistics if present
        methanol_hbonded = 0
        if self.solvent_composition['has_methanol']:
            methanol_atom_ids = set(atom.id for atom in self.methanol_atoms)
            methanol_hbonded = sum(1 for atom_id, props in atom_props.items()
                                 if props['is_hbonded'] and atom_id in methanol_atom_ids)
        
        print(f"\n--- Hydrogen Bond Summary ---")
        print(f"    Solvent type: {self.solvent_type}")
        print(f"    Total H-bonds detected: {len(self.hbond_results)}")
        print(f"    Total atoms involved in H-bonds: {total_hbonded}")
        print(f"        Water atoms: {water_hbonded}")
        if self.solvent_composition['has_methanol']:
            print(f"        Methanol atoms: {methanol_hbonded}")
        print(f"        Adsorbate atoms: {ads_hbonded}")
        print(f"    Total donor atoms: {total_donors}")
        print(f"    Total acceptor atoms: {total_acceptors}")
        
        # Show detailed H-bond information
        if len(self.hbond_results) > 0:
            print(f"\n    Detailed H-bond information:")
            for i, hbond in enumerate(self.hbond_results):
                try:
                    # hbond format: [frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle]
                    donor_idx = int(hbond[1])
                    hydrogen_idx = int(hbond[2])
                    acceptor_idx = int(hbond[3])
                    distance = hbond[4]
                    angle = hbond[5]
                    
                    donor_atom = self.universe.atoms[donor_idx]
                    hydrogen_atom = self.universe.atoms[hydrogen_idx]
                    acceptor_atom = self.universe.atoms[acceptor_idx]
                    
                    # Determine molecule types for clarity
                    donor_moltype = donor_atom.resname
                    acceptor_moltype = acceptor_atom.resname
                    
                    print(f"        {i+1}. {donor_atom.name}({donor_atom.id},{donor_moltype})-{hydrogen_atom.name}({hydrogen_atom.id})...{acceptor_atom.name}({acceptor_atom.id},{acceptor_moltype})")
                    print(f"           Distance: {distance:.3f} Å, Angle: {angle:.1f}°")
                    
                except (IndexError, AttributeError) as e:
                    print(f"        {i+1}. Error displaying H-bond details: {e}")

        # Now these are instance variables, accessible directly
        print(f"\n=== Results ===")
        print(f"--- Atoms involved in hydrogen bonds: {len(self.hbonded_atoms)}")
        for atom_id, props in list(self.hbonded_atoms.items())[:10]:  # Show first 10
            print(f"    Atom {atom_id}: H-bonded={props['is_hbonded']}, "
                    f"Donor={props['is_hbonded_donor']}, Acceptor={props['is_hbonded_acceptor']}")


def extract_all_hbonds_counts(zeolite_type: str = 'FAU',
                              solvent_type: str = 'water_pure',
                              verbose: bool = True):
    """
    Extract hydrogen bond counts for all adsorbates and snapshots in the dataset.
    
    Parameters
    ----------
    zeolite_type : str
        Zeolite type (e.g., 'FAU', 'BEA', 'MFI')
    solvent_type : str
        Solvent type (default: 'water_pure')
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame with columns: adsorbate, snapshot, pore_type, hbonds
    """
    
    results = []
    total_combinations = 0
    completed_combinations = 0
    
    # Count total combinations for progress tracking
    for env, adsorbates in ADSORBATES_BY_ENV.items():
        if solvent_type in env:
            total_combinations += len(adsorbates) * 10  # 10 snapshots per adsorbate
    
    if verbose:
        print(f"=== Extracting H-bond counts for all adsorbates and snapshots ===")
        print(f"Zeolite: {zeolite_type}, Solvent: {solvent_type}")
        print(f"Total combinations to process: {total_combinations}")
    
    # Process each environment and adsorbate
    for env, adsorbates in ADSORBATES_BY_ENV.items():
        if solvent_type not in env:
            continue
            
        # Extract pore type from environment name
        pore_type = env.split('-')[1] if '-' in env else env.split('_')[1]
        
        if verbose:
            print(f"\n--- Processing environment: {env} (pore_type: {pore_type}) ---")
        
        for adsorbate in adsorbates:
            if verbose:
                print(f"  Processing adsorbate: {adsorbate}")
            
            # Process snapshots 1-10 for each adsorbate
            for snapshot_index in range(1, 11):
                try:
                    # Create HydrogenBondDetector instance
                    hbond_detector = HydrogenBondDetector(
                        zeolite_type=zeolite_type,
                        solvent_type=solvent_type,
                        pore_type=pore_type,
                        adsorbate=adsorbate,
                        snapshot_index=snapshot_index,
                        d_a_cutoff=3.5,
                        d_h_cutoff=1.0,
                        d_h_a_angle_cutoff=150.0,
                        verbose=False  # Suppress individual verbose output
                    )
                    
                    # Count hydrogen bonds
                    hbonds_count = len(hbond_detector.hbond_results)
                    
                    # Store result
                    results.append({
                        'adsorbate': adsorbate,
                        'snapshot': snapshot_index,
                        'pore_type': pore_type,
                        'hbonds': hbonds_count
                    })
                    
                    completed_combinations += 1
                    
                    if verbose:
                        print(f"    Snapshot {snapshot_index}: {hbonds_count} H-bonds "
                              f"({completed_combinations}/{total_combinations})")
                
                except Exception as e:
                    if verbose:
                        print(f"    Snapshot {snapshot_index}: ERROR - {str(e)}")
                    
                    # Store error result as 0 H-bonds
                    results.append({
                        'adsorbate': adsorbate,
                        'snapshot': snapshot_index,
                        'pore_type': pore_type,
                        'hbonds': 0
                    })
                    
                    completed_combinations += 1
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n=== Summary ===")
        print(f"Total records: {len(results_df)}")
        print(f"Adsorbates processed: {results_df['adsorbate'].nunique()}")
        print(f"Snapshots per adsorbate: {results_df['snapshot'].nunique()}")
        print(f"Pore types: {results_df['pore_type'].unique()}")
        print(f"\nH-bonds statistics:")
        print(results_df['hbonds'].describe())
    
    return results_df


if __name__ == "__main__":
    
    # Test 1: Pure water system
    print("\n" + "="*60)
    print("=== Test 1: Pure Water System ===")
    hbond_detector_water = HydrogenBondDetector(
        zeolite_type='FAU',
        solvent_type='water_pure',
        pore_type='hydrophilic',
        adsorbate='11_01_propylene_glycol',
        snapshot_index=1,
        d_a_cutoff=3.5,
        d_h_cutoff=1.2,
        d_h_a_angle_cutoff=130.0,
        verbose=True
    )
    
    # Test 2: Mixed solvent system
    print("\n" + "="*60)
    print("=== Test 2: Mixed Solvent System (Methanol + Water) ===")
    
    hbond_detector_mixed = HydrogenBondDetector(
        zeolite_type='FAU',
        solvent_type='methanol_120_water_1080',
        pore_type='hydrophilic',
        adsorbate='11_01_propylene_glycol',
        snapshot_index=1,
        d_a_cutoff=3.5,
        d_h_cutoff=1.2,
        d_h_a_angle_cutoff=130.0,
        verbose=True
    )
