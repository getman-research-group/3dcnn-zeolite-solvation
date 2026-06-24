import os
import sys
from typing import List, Dict
from core.path import get_paths
from python_scripts_3dcnn.core.global_vars import ZEOLITE_TYPES, ADSORBATES_BY_ENV

# Map atomic weight (as '###.###') to element symbol
atom_weight_to_element = {
    '15.999': 'O',
    '1.008' : 'H',
    '47.867': 'Ti',
    '28.085': 'Si',
    '12.011': 'C',
    # add other mappings as needed
}

class LammpsDataProcessor:
    """
    Process a LAMMPS data file: read, reorder atoms by ID, parse bonds and masses,
    assign molecule IDs based on bonds, annotate Masses, and write a new data file.
    """

    def __init__(self,
                 input_file: str,
                 output_file: str,
                 verbose: bool = False,
                 zeolite: str = '',
                 env: str = '',
                 ads: str = '',
                 ):
        
        
        ## Initialize parameters
        self.input_file = input_file
        self.output_file = output_file
        self.verbose = verbose
        self.zeolite = zeolite
        self.env = env
        self.ads = ads

        self.header: List[str] = []
        self.atom_lines: List[List[str]] = []
        self.footer: List[str] = []
        self.bonds: List[tuple] = []
        self.atom_type_to_element: Dict[int,str] = {}

        # Step 1: Read header, atom entries, and footer
        self.read_file()
        
        # Step 2: Sort atom entries by their atom IDs
        self.reorder_atoms()
        
        # Step 3: Parse bond connectivity
        self.parse_bonds()
        
        # Step 4: Build atom_type -> element mapping from Masses section
        self.parse_masses()
        
        # Step 5: Assign molecule IDs based on bonds
        self.assign_mol_ids()
        
        # Step 6: Annotate Masses section with element categories
        self.annotate_masses()
        
        # Step 7: Write out modified data file
        self.write_file()

    
    # Step 1: Read header, atom entries, and footer
    def read_file(self):
        lines = open(self.input_file, 'r').readlines()
        # locate Atoms section
        idx = next(i for i,l in enumerate(lines) if l.strip().startswith('Atoms'))
        start = idx + 1
        while start < len(lines) and (not lines[start].split() or lines[start].strip().startswith('#')):
            start += 1
        end = start
        while end < len(lines):
            parts = lines[end].split()
            if not parts:
                break
            try:
                _ = int(parts[0]); end += 1
            except ValueError:
                break
        # Header is everything before Atoms, footer is everything after
        self.header = lines[:start]
        self.atom_raw = lines[start:end]
        self.footer = lines[end:]
        # Atoms section is a list of lists
        self.atom_lines = [ln.split() for ln in self.atom_raw]
        if self.verbose:
            print(f"Read {len(self.atom_lines)} atom entries and footer length {len(self.footer)}")

    
    # Step 2: Sort atom entries by their atom IDs
    def reorder_atoms(self):
        self.atom_lines.sort(key=lambda cols: int(cols[0]))
        if self.verbose:
            print("Atoms sorted by ID")
    
    
    # Step 3: Parse bond connectivity
    def parse_bonds(self):
        bond_idx = next((i for i,l in enumerate(self.footer) if l.strip().startswith('Bonds')), None)
        if bond_idx is None:
            raise ValueError('Bonds section not found')
        i = bond_idx + 2
        while i < len(self.footer):
            parts = self.footer[i].split()
            if not parts:
                break
            a1, a2 = int(parts[2]), int(parts[3])
            self.bonds.append((a1, a2))
            i += 1
        # build adjacency
        self.adj: Dict[int, List[int]] = {}
        for a1, a2 in self.bonds:
            self.adj.setdefault(a1, []).append(a2)
            self.adj.setdefault(a2, []).append(a1)
        if self.verbose:
            print(f"Parsed {len(self.bonds)} bonds")

    
    # Step 4: Build atom_type -> element mapping from Masses section
    def parse_masses(self):
        """
        Parse the Masses section to map atom types to element symbols.
        """
        in_masses = False
        # clear any previous mappings
        self.atom_type_to_element.clear()
        for line in self.header:
            stripped = line.strip()
            if stripped.startswith('Masses'):
                in_masses = True
                continue
            if in_masses:
                parts = stripped.split()
                # skip blank lines or comments
                if not parts or stripped.startswith('#'):
                    continue
                # stop if this line is not a mass entry
                if not parts[0].isdigit():
                    break
                # parse valid mass entry
                atype = int(parts[0])
                weight = f"{float(parts[1]):.3f}"
                element = atom_weight_to_element.get(weight, '')
                self.atom_type_to_element[atype] = element
        if self.verbose:
            print(f"Mapped {len(self.atom_type_to_element)} atom types to elements")

    
    # Step 5: Assign molecule IDs based on bonds
    def assign_mol_ids(self):
        mol_map: Dict[int,int] = {}
        water_counter = 1
        # assign water by O-H bonds
        for cols in self.atom_lines:
            aid = int(cols[0]); atype = int(cols[2])
            if aid <= 1200:
                neighbors = [n for n in self.adj.get(aid, []) if 1201 <= n <= 3600]
                if len(neighbors) != 2:
                    raise ValueError(f"Atom {aid} expects 2 H neighbors, got {neighbors}")
                mol_map[aid] = water_counter
                mol_map[neighbors[0]] = water_counter
                mol_map[neighbors[1]] = water_counter
                water_counter += 1
        # zeolite
        for aid in range(3601, 4756):
            mol_map[aid] = water_counter
        # adsorbate
        for cols in self.atom_lines:
            aid = int(cols[0])
            if aid >= 4756:
                mol_map[aid] = water_counter + 1
        # write back
        for cols in self.atom_lines:
            aid = int(cols[0])
            cols[1] = str(mol_map.get(aid, 0))
        if self.verbose:
            print(f"Assigned molecule IDs; water={water_counter-1}, zeolite=1, adsorbate=1")
    
    
    # Step 6: Annotate Masses section with element categories
    def annotate_masses(self):
        """
        Annotate Masses and Pair Coeffs entries with element and category comments.
        """
        new_header = []
        in_masses = False
        in_pair_coeffs = False
        
        for line in self.header:
            stripped = line.strip()
            
            # Handle Masses section
            if stripped.startswith('Masses'):
                in_masses = True
                in_pair_coeffs = False
                new_header.append(line)
                continue
            
            # Handle Pair Coeffs section
            if stripped.startswith('Pair Coeffs'):
                in_pair_coeffs = True
                in_masses = False
                new_header.append(line)
                continue
            
            if in_masses:
                if stripped == '':
                    new_header.append(line)
                    continue
                parts = stripped.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    atype = int(parts[0])
                    weight = float(parts[1])
                    element = self.atom_type_to_element.get(atype, '')
                    # determine category by any atom_lines lookup
                    molid = next((int(c[1]) for c in self.atom_lines if int(c[2])==atype), None)
                    if molid is None:
                        category = 'unknown'
                    elif 1 <= molid <= 1200:
                        category = 'HOH'
                    elif molid == 1201:
                        category = self.zeolite
                    elif molid == 1202:
                        category = 'ADS'
                    comment = f"# {element}_{category}"
                    annotated = f"{atype:<5d}{weight:>8.3f}    {comment}\n"
                    new_header.append(annotated)
                    continue
                in_masses = False
            
            if in_pair_coeffs:
                if stripped == '':
                    new_header.append(line)
                    continue
                parts = stripped.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    atype = int(parts[0])
                    param1 = float(parts[1])
                    param2 = float(parts[2])
                    element = self.atom_type_to_element.get(atype, '')
                    # determine category by any atom_lines lookup
                    molid = next((int(c[1]) for c in self.atom_lines if int(c[2])==atype), None)
                    if molid is None:
                        category = 'unknown'
                    elif 1 <= molid <= 1200:
                        category = 'HOH'
                    elif molid == 1201:
                        category = self.zeolite
                    elif molid == 1202:
                        category = 'ADS'
                    comment = f"# {element}_{category}"
                    annotated = f"{atype:<3d}{param1:>10.4f}{param2:>10.4f}    {comment}\n"
                    new_header.append(annotated)
                    continue
                in_pair_coeffs = False
            
            new_header.append(line)
        
        self.header = new_header

    
    # Step 7: Write out modified data file
    def write_file(self):
        """Write processed data with aligned columns and atom annotations."""
        with open(self.output_file, 'w') as f:
            f.writelines(self.header)
            for cols in self.atom_lines:
                atom_id, mol_id, atype = map(int, cols[:3])
                charge, x, y, z = map(float, cols[3:7])
                nx = int(cols[7]) if len(cols)>7 else 0
                ny = int(cols[8]) if len(cols)>8 else 0
                nz = int(cols[9]) if len(cols)>9 else 0
                element = self.atom_type_to_element.get(atype, '')
                if 1 <= mol_id <= 1200:
                    category = 'HOH'
                elif mol_id == 1201:
                    category = self.zeolite
                elif mol_id == 1202:
                    category = 'ADS'
                comment = f"# {element}_{category}"
                line = (
                    f"{atom_id:<6d}{mol_id:<6d}{atype:<6d}  "
                    f"{charge:>10.6f}{x:>12.6f}{y:>12.6f}{z:>12.6f}  "
                    f"{nx:>4d}{ny:>4d}{nz:>4d}  {comment}\n"
                )
                f.write(line)
            f.writelines(self.footer)
        if self.verbose:
            print(f"Wrote processed data to {self.output_file}")


if __name__ == '__main__':
    
    # The path to the simulation folder
    base = get_paths("simulation_path")
    
    # Use one adsorbate for testing
    test = False
    
    # test zeolite types
    if test:
        ZEOLITE_TYPES = ['FAU']
        ADSORBATES_BY_ENV = {'water_pure-hydrophobic': ['05_02_13_propanediol']}
    
    # Iterate over each zeolite type
    for zeolite in ZEOLITE_TYPES:
        
        zeolite_dir = os.path.join(base, zeolite)
        print(f"--- Processing zeolite: {zeolite}")
        
        # Iterate over each environment and its adsorbates
        for env, adsorbates in ADSORBATES_BY_ENV.items():
            env_dir = os.path.join(zeolite_dir, env)
            print(f"    Environment: {env}")
            # Iterate over each adsorbate
            for ads in adsorbates:
                print('ads: ', ads)
                ads_dir = os.path.join(env_dir, ads)
                print('ads_dir: ', ads_dir)

                inp = os.path.join(ads_dir, 'data_nvt_samp.lammpsdata')
                out = os.path.join(ads_dir, 'data_nvt_samp_new.lammpsdata')
                
                print(f"    >> Modifying {zeolite} {env} {ads}")
                lammps = LammpsDataProcessor(inp,
                                             out,
                                             verbose=True,
                                             zeolite=zeolite,
                                             env=env,
                                             ads=ads,
                                             )
