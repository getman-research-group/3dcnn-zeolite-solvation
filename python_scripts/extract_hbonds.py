# -*- coding: utf-8 -*-
"""
extract_hbonds.py

Detect adsorbate-solvent hydrogen bonds in one sampled molecular dynamics (MD)
configuration using ``MDAnalysis.analysis.hydrogenbonds.HydrogenBondAnalysis``.

The script works with the LAMMPS systems organized under ``md_simulations``.
For a selected zeolite, solvent environment, pore type, adsorbate, and snapshot
index, :class:`HydrogenBondDetector` first delegates snapshot loading to
``snapshotMDAnalysis`` in ``read_md_snapshot.py``. That loader combines the
shared ``data_nvt_samp_new.lammpsdata`` topology with the coordinates in
``intE<index>/intE<index>.traj`` and assigns the residue names used here:

- ``ADS`` for the adsorbate;
- ``HOH`` for water;
- ``MEO`` for methanol when it is present.

Hydrogen bonds are evaluated in both directions between the adsorbate and each
available solvent component:

1. adsorbate donor to water acceptor;
2. water donor to adsorbate acceptor;
3. adsorbate donor to methanol acceptor;
4. methanol donor to adsorbate acceptor.

The donor-acceptor distance, donor-hydrogen distance, and donor-hydrogen-
acceptor angle thresholds are configurable. MDAnalysis uses the simulation-cell
information stored in the Universe when evaluating the molecular geometry. The
analysis intentionally excludes solvent-solvent, adsorbate-zeolite, and
zeolite-solvent hydrogen bonds.

For each detected bond, MDAnalysis returns the frame, donor, hydrogen, and
acceptor indices together with the donor-acceptor distance and D-H-A angle. The
script converts these results into per-atom Boolean properties:
``is_hbonded``, ``is_hbonded_donor``, and ``is_hbonded_acceptor``. These
properties are consumed by ``generate_voxel_grids.py`` as atom-level voxel
features. The module also provides ``extract_all_hbonds_counts`` for collecting
hydrogen-bond counts over all configured adsorbates and snapshots.

Running this file directly performs two verbose examples: one pure-water system
and one methanol-water system. The script reads simulation data but does not
modify the source trajectory or topology files.
"""

# Standard-library modules for path handling and warning control.
import os
import warnings

# Numerical arrays and tabular summaries.
import numpy as np
import pandas as pd

# Type annotations used by the public return values.
from typing import Dict, Tuple, List

# MDAnalysis tools for molecular selections, hydrogen-bond detection,
# coordinate transformations, and periodic-distance calculations.
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis import transformations
from MDAnalysis.lib.distances import distance_array

# Project utilities for path resolution, snapshot loading, and dataset metadata.
from core.path import get_paths
from read_md_snapshot import snapshotMDAnalysis
from core.global_vars import ZEOLITE_TYPES, ADSORBATES_BY_ENV

# Suppress progress-bar and optional BioPython deprecation messages that do not
# affect the hydrogen-bond analysis.
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="Bio.Application")

# An empty H-bond result is valid for an individual snapshot. Suppress the
# corresponding MDAnalysis warning and represent that case with an empty array.
warnings.filterwarnings("ignore",
                       message="No hydrogen bonds were found given angle*",
                       category=UserWarning,
                       module="MDAnalysis.analysis.hydrogenbonds.hbond_analysis")

class HydrogenBondDetector:
    """
    Detect hydrogen bonds between an adsorbate and the available solvents.

    Initialization performs the complete single-snapshot workflow: load the
    MDAnalysis Universe, identify solvent components, define atom selections,
    run directional hydrogen-bond searches, and generate per-atom properties.

    The detector treats oxygen atoms as potential heavy-atom donors and
    acceptors and hydrogen atoms as potential donor-bound hydrogens. The
    ``HydrogenBondAnalysis`` distance and angle criteria determine which of
    these candidates form hydrogen bonds in the selected snapshot.

    Parameters
    ----------
    zeolite_type : str
        Zeolite framework identifier used to locate the simulation directory.
    solvent_type : str
        Solvent composition, such as ``water_pure`` or
        ``methanol_240_water_960``.
    pore_type : str
        Pore environment, normally ``hydrophilic`` or ``hydrophobic``.
    adsorbate : str
        Adsorbate directory identifier.
    snapshot_index : int
        Sampled configuration number.
    d_a_cutoff : float
        Maximum donor-acceptor distance in angstrom.
    d_h_cutoff : float
        Maximum donor-hydrogen distance in angstrom used to associate donor and
        hydrogen candidates.
    d_h_a_angle_cutoff : float
        Minimum donor-hydrogen-acceptor angle in degrees.
    verbose : bool
        Print composition, selection, bond, and atom-level summaries when True.

    Attributes
    ----------
    hbond_results : numpy.ndarray
        Rows of ``[frame, donor_index, hydrogen_index, acceptor_index,
        distance, angle]`` returned by MDAnalysis.
    atom_hbond_properties : dict
        Mapping from LAMMPS atom ID to the three Boolean H-bond properties.
    hbonded_atoms : dict
        Subset of ``atom_hbond_properties`` containing only involved atoms.
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
        """Load and analyze one MD snapshot using the requested H-bond criteria."""

        # Store the system identifiers used by read_md_snapshot.py to resolve
        # the topology and sampled-coordinate paths.
        self.zeolite_type = zeolite_type
        self.solvent_type = solvent_type
        self.pore_type = pore_type
        self.adsorbate = adsorbate
        self.snapshot_index = snapshot_index
        self.verbose = verbose
        
        # Store the geometric criteria passed directly to HydrogenBondAnalysis.
        self.d_a_cutoff = d_a_cutoff
        self.d_h_cutoff = d_h_cutoff
        self.d_h_a_angle_cutoff = d_h_a_angle_cutoff
        
        # Load the shared LAMMPS topology and the selected one-frame trajectory.
        self.snapshot_mda = snapshotMDAnalysis(
            zeolite_type=zeolite_type,
            solvent_type=solvent_type,
            pore_type=pore_type,
            adsorbate=adsorbate,
            snapshot_index=snapshot_index,
            verbose=False,
        )
        
        # Reuse the annotated Universe created by snapshotMDAnalysis. Residue
        # names assigned there define all molecular selections below.
        self.universe = self.snapshot_mda.universe
        
        # Infer the components from the Universe instead of relying on molecule
        # ID ranges for each solvent composition.
        self.solvent_composition = self._analyze_solvent_composition()
        
        # Prepare reusable MDAnalysis selection strings and AtomGroups.
        self.define_atom_selections()
        
        # Run the four possible adsorbate-solvent donor/acceptor searches. A
        # snapshot with no qualifying bonds is a valid result.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
            self.hbond_results = self.detect_hydrogen_bonds()
        
        # Convert bond-level results into the atom-level flags required by the
        # voxel-generation workflow.
        self.atom_hbond_properties = self.get_atom_hbond_properties()
        self.hbonded_atoms = {atom_id: props for atom_id, props in self.atom_hbond_properties.items()
                             if props['is_hbonded']}
        
        if self.verbose:
            self.print_hbond_summary()

    def _analyze_solvent_composition(self):
        """
        Infer molecular components from residue names in the loaded Universe.

        Water, methanol, and adsorbate residues are recognized from ``HOH``,
        ``MEO``, and ``ADS`` in their residue names. Any remaining residue names
        are recorded as zeolite residues. Counting residues rather than atoms
        gives the number of solvent molecules and avoids hardcoded molecule-ID
        ranges for different compositions.
        
        Returns:
            dict with keys:
                - has_water: bool indicating if water molecules are present
                - has_methanol: bool indicating if methanol molecules are present  
                - water_count: int number of water molecules
                - methanol_count: int number of methanol molecules
                - adsorbate_resnames: list of adsorbate residue names
                - zeolite_resnames: list of zeolite residue names
        """
        # Residue names were assigned when read_md_snapshot.py built the
        # annotated Universe.
        all_resnames = set(self.universe.residues.resnames)
        
        # Classify each unique residue name by the naming convention used in
        # this repository.
        water_resnames = [name for name in all_resnames if 'HOH' in name]
        methanol_resnames = [name for name in all_resnames if 'MEO' in name] 
        adsorbate_resnames = [name for name in all_resnames if 'ADS' in name]
        zeolite_resnames = [name for name in all_resnames if name not in water_resnames + methanol_resnames + adsorbate_resnames]
        
        # Select all matching atoms and count their parent residues (molecules).
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
        """
        Define MDAnalysis selections for adsorbate, water, and methanol atoms.

        Oxygen atoms are candidate donors and acceptors, while hydrogen atoms
        are candidate donor-bound hydrogens. Wildcards allow the same selections
        to work with the component-specific atom names assigned by the snapshot
        loader. Empty ``NonExistent`` selections provide valid zero-length
        AtomGroups when methanol is absent.
        """
        
        # Adsorbate candidate donors, hydrogens, and acceptors.
        self.adsorbate_donors_sel = "resname ADS and name O*"
        self.adsorbate_hydrogens_sel = "resname ADS and name H*"
        self.adsorbate_acceptors_sel = "resname ADS and name O*"
        
        # Water candidate donors, hydrogens, and acceptors.
        self.water_donors_sel = "resname HOH and name O*"
        self.water_hydrogens_sel = "resname HOH and name H*"
        self.water_acceptors_sel = "resname HOH and name O*"
        
        # Methanol candidates are activated only for mixed-solvent systems.
        if self.solvent_composition['has_methanol']:
            self.methanol_donors_sel = "resname MEO and name O*"
            self.methanol_hydrogens_sel = "resname MEO and name H*"
            self.methanol_acceptors_sel = "resname MEO and name O*"
        else:
            self.methanol_donors_sel = "name NonExistent"
            self.methanol_hydrogens_sel = "name NonExistent"
            self.methanol_acceptors_sel = "name NonExistent"
        
        # Store complete component AtomGroups for later property initialization
        # and molecule-specific summary statistics.
        self.adsorbate_atoms = self.universe.select_atoms("resname ADS")
        self.water_atoms = self.universe.select_atoms("resname HOH")
        
        if self.solvent_composition['has_methanol']:
            self.methanol_atoms = self.universe.select_atoms("resname MEO")
        else:
            self.methanol_atoms = self.universe.select_atoms("name NonExistent")
        
        # Materialize the candidate groups for verbose selection diagnostics.
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
        Detect directional hydrogen bonds between adsorbate and solvent atoms.

        Separate ``HydrogenBondAnalysis`` runs cover adsorbate-to-solvent and
        solvent-to-adsorbate bonds for every solvent component present. Keeping
        these directions separate makes donor and acceptor roles explicit while
        allowing all results to be combined into one array.

        Returns
        -------
        numpy.ndarray
            Array with one row per detected bond and columns ``frame``,
            ``donor_index``, ``hydrogen_index``, ``acceptor_index``,
            ``donor_acceptor_distance``, and ``donor_hydrogen_acceptor_angle``.
            An empty result has shape ``(0, 6)``.
        """
        if self.verbose:
            print(f"\n--- Detecting Hydrogen Bonds ---")
            print(f"    D-A distance cutoff: {self.d_a_cutoff} Å")
            print(f"    D-H distance cutoff: {self.d_h_cutoff} Å")
            print(f"    D-H-A angle cutoff: {self.d_h_a_angle_cutoff}°")
            print(f"    Solvent type: {self.solvent_type}")
        
        # The loaded snapshot contains one frame; explicitly position the
        # Universe at that frame before analysis.
        self.universe.trajectory[0]  # Use the first (and only) frame from the snapshot
        
        all_hbond_results = []
        
        # Run one directional donor/acceptor search with shared geometric
        # criteria and return the raw MDAnalysis result array.
        def detect_specific_hbonds(donors_sel, hydrogens_sel, acceptors_sel, description):
            """Run HydrogenBondAnalysis for one molecular donor/acceptor pair."""
            if self.verbose:
                print(f"\n    {description}    ")
                print(f"    Donors: {donors_sel}")
                print(f"    Hydrogens: {hydrogens_sel}")
                print(f"    Acceptors: {acceptors_sel}")
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="MDAnalysis")
                
                # Explicit selections prevent MDAnalysis from guessing atom
                # roles and keep the definition consistent across systems.
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
                
                # Analyze only the single frame stored in this snapshot file.
                hbonds.run(start=0, stop=1, step=1, verbose=False)
            
            results = hbonds.results.hbonds
            if self.verbose:
                print(f"    Found: {len(results)} H-bonds")
            
            return results
        
        # 1. Evaluate both donor/acceptor directions involving water.
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
        
        # 2. Evaluate both donor/acceptor directions involving methanol.
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
        
        # 3. Combine all directional searches into the standard six-column
        # MDAnalysis result format.
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
        Convert bond-level results into per-atom hydrogen-bond properties.

        All water, methanol, and adsorbate atoms are initialized with False
        values. Atoms participating in a detected bond are then marked according
        to their donor, donor-bound hydrogen, or acceptor role. Zeolite atoms are
        not included because this analysis only evaluates adsorbate-solvent
        hydrogen bonds.

        Returns
        -------
        dict
            Mapping from the original LAMMPS atom ID to ``is_hbonded``,
            ``is_hbonded_donor``, and ``is_hbonded_acceptor`` Boolean values.
        """
        # Initialize every relevant atom so downstream code can perform a direct
        # atom-ID lookup even when the atom is not involved in a bond.
        atom_hbond_dict = {}
        
        # Initialize all water atoms.
        for atom in self.water_atoms:
            atom_hbond_dict[int(atom.id)] = {
                'is_hbonded': False,
                'is_hbonded_donor': False,
                'is_hbonded_acceptor': False
            }
        
        # Initialize all methanol atoms when present.
        if self.solvent_composition['has_methanol']:
            for atom in self.methanol_atoms:
                atom_hbond_dict[int(atom.id)] = {
                    'is_hbonded': False,
                    'is_hbonded_donor': False,
                    'is_hbonded_acceptor': False
                }
        
        # Initialize all adsorbate atoms and use native Python integer keys.
        for atom in self.adsorbate_atoms:
            atom_hbond_dict[int(atom.id)] = {
                'is_hbonded': False,
                'is_hbonded_donor': False,
                'is_hbonded_acceptor': False
            }
        
        # Update atom flags from each six-column MDAnalysis H-bond record.
        if len(self.hbond_results) > 0:
            for hbond in self.hbond_results:
                # hbond format: [frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle]
                # Note: MDAnalysis uses 0-based indices, but atom IDs are 1-based
                donor_idx = int(hbond[1])
                hydrogen_idx = int(hbond[2])
                acceptor_idx = int(hbond[3])
                
                # Convert zero-based Universe indices into the original atom IDs
                # used throughout the voxel-generation code.
                donor_id = int(self.universe.atoms[donor_idx].id)
                hydrogen_id = int(self.universe.atoms[hydrogen_idx].id)
                acceptor_id = int(self.universe.atoms[acceptor_idx].id)
                
                # Mark the heavy-atom donor.
                if donor_id in atom_hbond_dict:
                    atom_hbond_dict[donor_id]['is_hbonded'] = True
                    atom_hbond_dict[donor_id]['is_hbonded_donor'] = True
                
                # Mark the donor-bound hydrogen as involved in a bond. It is not
                # classified as the heavy-atom donor.
                if hydrogen_id in atom_hbond_dict:
                    atom_hbond_dict[hydrogen_id]['is_hbonded'] = True
                
                # Mark the heavy-atom acceptor.
                if acceptor_id in atom_hbond_dict:
                    atom_hbond_dict[acceptor_id]['is_hbonded'] = True
                    atom_hbond_dict[acceptor_id]['is_hbonded_acceptor'] = True
        
        return atom_hbond_dict
    
    def print_hbond_summary(self):
        """
        Print composition-level, atom-level, and bond-level diagnostics.

        The detailed section reports atom names and IDs, molecular components,
        donor-acceptor distances, and D-H-A angles. This method is diagnostic
        only and does not modify the stored results.
        """
        atom_props = self.get_atom_hbond_properties()
        
        # Count atoms carrying each Boolean property.
        total_hbonded = sum(1 for props in atom_props.values() if props['is_hbonded'])
        total_donors = sum(1 for props in atom_props.values() if props['is_hbonded_donor'])
        total_acceptors = sum(1 for props in atom_props.values() if props['is_hbonded_acceptor'])
        
        # Build component-specific atom-ID sets for summary counts.
        water_atom_ids = set(atom.id for atom in self.water_atoms)
        adsorbate_atom_ids = set(atom.id for atom in self.adsorbate_atoms)
        
        water_hbonded = sum(1 for atom_id, props in atom_props.items()
                           if props['is_hbonded'] and atom_id in water_atom_ids)
        ads_hbonded = sum(1 for atom_id, props in atom_props.items()
                         if props['is_hbonded'] and atom_id in adsorbate_atom_ids)
        
        # Include a separate methanol count for mixed-solvent systems.
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
        
        # Print the geometry and molecular identity of each detected bond.
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
                    
                    # Residue names identify the donor and acceptor components.
                    donor_moltype = donor_atom.resname
                    acceptor_moltype = acceptor_atom.resname
                    
                    print(f"        {i+1}. {donor_atom.name}({donor_atom.id},{donor_moltype})-{hydrogen_atom.name}({hydrogen_atom.id})...{acceptor_atom.name}({acceptor_atom.id},{acceptor_moltype})")
                    print(f"           Distance: {distance:.3f} Å, Angle: {angle:.1f}°")
                    
                except (IndexError, AttributeError) as e:
                    print(f"        {i+1}. Error displaying H-bond details: {e}")

        # Show a compact preview of the atom-property mapping retained on the
        # detector instance.
        print(f"\n=== Results ===")
        print(f"--- Atoms involved in hydrogen bonds: {len(self.hbonded_atoms)}")
        for atom_id, props in list(self.hbonded_atoms.items())[:10]:  # Show first 10
            print(f"    Atom {atom_id}: H-bonded={props['is_hbonded']}, "
                    f"Donor={props['is_hbonded_donor']}, Acceptor={props['is_hbonded_acceptor']}")


def extract_all_hbonds_counts(zeolite_type: str = 'FAU',
                              solvent_type: str = 'water_pure',
                              verbose: bool = True):
    """
    Collect hydrogen-bond counts over configured environments and snapshots.

    Environments in ``ADSORBATES_BY_ENV`` are filtered by ``solvent_type``.
    Every configured adsorbate is evaluated for snapshots 1 through 10 using a
    new ``HydrogenBondDetector`` instance. This function returns the collected
    table in memory and does not write a CSV file.
    
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
        DataFrame with columns ``adsorbate``, ``snapshot``, ``pore_type``, and
        ``hbonds``. Under the existing error-handling behavior, a failed
        combination is recorded with a count of zero.
    """
    
    results = []
    total_combinations = 0
    completed_combinations = 0
    
    # Count expected adsorbate/snapshot combinations for progress reporting.
    for env, adsorbates in ADSORBATES_BY_ENV.items():
        if solvent_type in env:
            total_combinations += len(adsorbates) * 10  # 10 snapshots per adsorbate
    
    if verbose:
        print(f"=== Extracting H-bond counts for all adsorbates and snapshots ===")
        print(f"Zeolite: {zeolite_type}, Solvent: {solvent_type}")
        print(f"Total combinations to process: {total_combinations}")
    
    # Process only configured environments containing the requested solvent key.
    for env, adsorbates in ADSORBATES_BY_ENV.items():
        if solvent_type not in env:
            continue
            
        # Environment names encode the pore type after the hyphen.
        pore_type = env.split('-')[1] if '-' in env else env.split('_')[1]
        
        if verbose:
            print(f"\n--- Processing environment: {env} (pore_type: {pore_type}) ---")
        
        for adsorbate in adsorbates:
            if verbose:
                print(f"  Processing adsorbate: {adsorbate}")
            
            # Process the ten sampled configurations available for each system.
            for snapshot_index in range(1, 11):
                try:
                    # Constructing the detector performs the complete analysis
                    # for this system and snapshot.
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
                    
                    # Each result row represents one detected hydrogen bond.
                    hbonds_count = len(hbond_detector.hbond_results)
                    
                    # Store the snapshot-level count and identifying metadata.
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
                    
                    # Preserve the existing batch behavior by recording failed
                    # combinations as zero H-bonds and continuing.
                    results.append({
                        'adsorbate': adsorbate,
                        'snapshot': snapshot_index,
                        'pore_type': pore_type,
                        'hbonds': 0
                    })
                    
                    completed_combinations += 1
    
    # Return a tabular summary suitable for downstream analysis.
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
    
    # Example 1: analyze one propylene-glycol snapshot in pure water.
    print("\n" + "="*60)
    print("=== Test 1: Pure Water System ===")
    hbond_detector_water = HydrogenBondDetector(
        zeolite_type='FAU',
        solvent_type='water_pure',
        pore_type='hydrophilic',
        adsorbate='11_01_propylene_glycol',
        snapshot_index=1,
        d_a_cutoff=3.5,
        d_h_cutoff=1.0,
        d_h_a_angle_cutoff=150.0,
        verbose=True
    )
    
    # Example 2: analyze one 2-propanol snapshot in methanol-water solvent.
    print("\n" + "="*60)
    print("=== Test 2: Mixed Solvent System (Methanol + Water) ===")
    
    hbond_detector_mixed = HydrogenBondDetector(
        zeolite_type='FAU',
        solvent_type='methanol_240_water_960',
        pore_type='hydrophilic',
        adsorbate='02_01_02_propanol',
        snapshot_index=1,
        d_a_cutoff=3.5,
        d_h_cutoff=1.0,
        d_h_a_angle_cutoff=150.0,
        verbose=True
    )
