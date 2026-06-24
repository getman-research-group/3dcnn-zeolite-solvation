import os
import sys
from typing import List, Dict
from core.path import get_paths

# Map atomic weight to element symbol with tolerance
# Each element has one reference mass, and we'll use tolerance matching
atom_weight_to_element = {
    15.999: 'O',   # Oxygen
    1.008: 'H',    # Hydrogen
    47.867: 'Ti',  # Titanium
    28.085: 'Si',  # Silicon
    12.011: 'C',   # Carbon
}

# Tolerance for mass matching
MASS_TOLERANCE = 0.5

class CosolventLammpsDataProcessor:
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
        
        # Step 2: Correct molecule IDs for cosolvent systems
        self.correct_molecule_ids()
        
        # Step 3: Write the corrected file
        self.write_file()
        

    
    # Step 1: Read header, atom entries, and footer
    def read_file(self):
        """
        Read the LAMMPS data file and parse header, atom data, bonds, masses, and footer.
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file {self.input_file} not found")
        
        if self.verbose:
            print(f"\n--- Reading LAMMPS data file: {self.input_file}")
        
        with open(self.input_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the file sections
        in_atoms_section = False
        in_bonds_section = False
        in_masses_section = False
        atoms_header_found = False
        bonds_header_found = False
        masses_header_found = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for section headers
            if line_stripped.startswith("Atoms"):
                in_atoms_section = True
                atoms_header_found = True
                in_masses_section = False  # End masses section
                self.header.append(line)
                continue
            elif line_stripped.startswith("Bonds"):
                in_bonds_section = True
                bonds_header_found = True
                in_atoms_section = False
                in_masses_section = False  # End masses section
                self.footer.append(line)
                continue
            elif line_stripped.startswith("Masses"):
                in_masses_section = True
                masses_header_found = True
                self.header.append(line)
                continue
            elif line_stripped.startswith("Pair Coeffs") or line_stripped.startswith("Bond Coeffs") or line_stripped.startswith("Angle Coeffs") or line_stripped.startswith("Dihedral Coeffs"):
                # End masses section when we hit other coefficient sections
                in_masses_section = False
                self.header.append(line)
                continue
            elif line_stripped == "":
                # Empty line handling - preserve all empty lines in their original locations
                if in_atoms_section:
                    # Empty lines within atoms section are rare, but if they exist, skip them for atom parsing
                    # but don't add to header/footer as they'll be handled in write_file
                    pass
                elif not atoms_header_found:
                    # Before atoms section - goes to header
                    self.header.append(line)
                else:
                    # After atoms section - goes to footer  
                    self.footer.append(line)
                continue
            
            # Parse content based on current section
            if in_masses_section:
                self._parse_masses_line(line_stripped)
                self.header.append(line)
            elif in_atoms_section:
                self._parse_atom_line(line_stripped)
            elif in_bonds_section:
                self._parse_bonds_line(line_stripped)
                self.footer.append(line)
            else:
                # Before atoms section or after bonds section
                if not atoms_header_found:
                    self.header.append(line)
                else:
                    self.footer.append(line)
        
        if self.verbose:
            print(f"    Successfully read {len(self.atom_lines)} atoms")
            print(f"    Successfully read {len(self.bonds)} bonds") 
            print(f"    Successfully read {len(self.atom_type_to_element)} atom types")
            
            # Print sample of read data
            if len(self.atom_lines) > 0:
                print(f"    Sample atom line: {self.atom_lines[0]}")
            if len(self.bonds) > 0:
                print(f"    Sample bond: {self.bonds[0]}")

            # Analyze mol IDs to understand the structure
            mol_ids = [int(atom[1]) for atom in self.atom_lines]
            atom_ids = [int(atom[0]) for atom in self.atom_lines]
            min_mol_id = min(mol_ids)
            max_mol_id = max(mol_ids)
            unique_mol_ids = sorted(set(mol_ids))
            print(f"    Molecule IDs range: {min_mol_id} to {max_mol_id}")
            print(f"    Total unique molecules: {len(unique_mol_ids)}")
            print(f"    Atom IDs range: {min(atom_ids)} to {max(atom_ids)}")
            
            # Show some mol id statistics for debugging
            mol_id_counts = {}
            for mol_id in mol_ids:
                mol_id_counts[mol_id] = mol_id_counts.get(mol_id, 0) + 1
            
            # Get environment config to understand expected structure
            env_config = self._get_environment_config()
            
            if env_config:
                # Count water molecules based on mol_id range
                water_mol_ids = [mid for mid in unique_mol_ids if env_config['water_mol_start'] <= mid <= env_config['water_mol_end']]
                water_count = len(water_mol_ids)
                
                # Count methanol molecules (before correction)
                methanol_mol_ids = [mid for mid in unique_mol_ids if env_config['methanol_mol_current_start'] <= mid <= env_config['methanol_mol_current_end']]
                methanol_count = len(methanol_mol_ids)
                
                # Count zeolite molecules (usually mol_id 1201)
                zeolite_mol_ids = [mid for mid in unique_mol_ids if mid == 1201]
                zeolite_count = len(zeolite_mol_ids)
                
                # Count adsorbate molecules (usually mol_id 1202)
                adsorbate_mol_ids = [mid for mid in unique_mol_ids if mid == 1202]
                adsorbate_count = len(adsorbate_mol_ids)
                
                print(f"\n--- Component analysis:")
                print(f"    Water molecules: {water_count} molecules (mol_id {env_config['water_mol_start']}-{env_config['water_mol_end']})")
                print(f"    Methanol solvent molecules: {methanol_count} molecules (mol_id {env_config['methanol_mol_current_start']}-{env_config['methanol_mol_current_end']}, will be changed to {env_config['methanol_mol_target_start']}-{env_config['methanol_mol_target_end']})")
                print(f"    Zeolite: {zeolite_count} molecules (mol_id {zeolite_mol_ids[0] if zeolite_mol_ids else 'not found'})")
                print(f"    Adsorbate: {adsorbate_count} molecules (mol_id {adsorbate_mol_ids[0] if adsorbate_mol_ids else 'not found'})")
            else:
                # Fallback for unknown environments
                mol_sizes_3 = [mid for mid in unique_mol_ids if mol_id_counts[mid] == 3]
                mol_sizes_6 = [mid for mid in unique_mol_ids if mol_id_counts[mid] == 6] 
                mol_sizes_other = [mid for mid in unique_mol_ids if mol_id_counts[mid] not in [3, 6]]
                
                print(f"    Molecules with 3 atoms (likely water): {len(mol_sizes_3)} molecules")
                if len(mol_sizes_6) > 0:
                    print(f"    Molecules with 6 atoms (likely methanol): {len(mol_sizes_6)} molecules")
                if mol_sizes_other:
                    print(f"    Other molecules: {len(mol_sizes_other)} molecules with sizes: {[(mid, mol_id_counts[mid]) for mid in mol_sizes_other[:5]]}")
                
            # Check specific environment expectations
            env_config = self._get_environment_config()
            if env_config:
                print(f"\n--- For {self.env}:")
                print(f"    Expected structure:")
                print(f"    - Atoms {env_config['water_atom_start']}-{env_config['water_atom_end']}: {env_config['n_water']} water molecules (mol_id {env_config['water_mol_start']}-{env_config['water_mol_end']})")
                print(f"    - Atoms {env_config['methanol_atom_start']}-{env_config['methanol_atom_end']}: {env_config['n_methanol']} methanol molecules (mol_id should be {env_config['methanol_mol_target_start']}-{env_config['methanol_mol_target_end']}, currently {env_config['methanol_mol_current_start']}-{env_config['methanol_mol_current_end']})")
                print(f"    - Zeolite: mol_id 1201")
                print(f"    - Adsorbate: mol_id 1202")
                
                # Identify methanol molecules that need mol_id correction
                water_atoms = [atom for atom in self.atom_lines if env_config['water_atom_start'] <= int(atom[0]) <= env_config['water_atom_end']]
                methanol_atoms = [atom for atom in self.atom_lines if env_config['methanol_atom_start'] <= int(atom[0]) <= env_config['methanol_atom_end']]
                
                print(f"\n--- Actual structure:")
                print(f"    - Water atoms ({env_config['water_atom_start']}-{env_config['water_atom_end']}): {len(water_atoms)} atoms")
                print(f"    - Methanol atoms ({env_config['methanol_atom_start']}-{env_config['methanol_atom_end']}): {len(methanol_atoms)} atoms")
                
                if water_atoms:
                    water_mol_ids = set(int(atom[1]) for atom in water_atoms)
                    print(f"    - Water mol_ids: {min(water_mol_ids)}-{max(water_mol_ids)} ({len(water_mol_ids)} molecules)")
                
                if methanol_atoms:
                    methanol_mol_ids = set(int(atom[1]) for atom in methanol_atoms)
                    print(f"    - Methanol original mol_ids: {min(methanol_mol_ids)}-{max(methanol_mol_ids)} ({len(methanol_mol_ids)} molecules)")
                    print(f"    - Methanol mol_ids need to be changed to: {env_config['methanol_mol_target_start']}-{env_config['methanol_mol_target_end']}")
    
    def _get_environment_config(self):
        """Get configuration parameters for different cosolvent environments"""
        env_configs = {
            'methanol_120_water_1080-hydrophilic': {
                'n_water': 1080,
                'n_methanol': 120,
                'water_atom_start': 1,
                'water_atom_end': 3240,
                'methanol_atom_start': 3241,
                'methanol_atom_end': 3960,
                'water_mol_start': 1,
                'water_mol_end': 1080,
                'methanol_mol_current_start': 1,
                'methanol_mol_current_end': 120,
                'methanol_mol_target_start': 1081,
                'methanol_mol_target_end': 1200
            },
            'methanol_120_water_1080-hydrophobic': {
                'n_water': 1080,
                'n_methanol': 120,
                'water_atom_start': 1,
                'water_atom_end': 3240,
                'methanol_atom_start': 3241,
                'methanol_atom_end': 3960,
                'water_mol_start': 1,
                'water_mol_end': 1080,
                'methanol_mol_current_start': 1,
                'methanol_mol_current_end': 120,
                'methanol_mol_target_start': 1081,
                'methanol_mol_target_end': 1200
            },
            'methanol_240_water_960-hydrophilic': {
                'n_water': 960,
                'n_methanol': 240,
                'water_atom_start': 1,
                'water_atom_end': 2880,
                'methanol_atom_start': 2881,
                'methanol_atom_end': 4320,
                'water_mol_start': 1,
                'water_mol_end': 960,
                'methanol_mol_current_start': 1,
                'methanol_mol_current_end': 240,
                'methanol_mol_target_start': 961,
                'methanol_mol_target_end': 1200
            },
            'methanol_240_water_960-hydrophobic': {
                'n_water': 960,
                'n_methanol': 240,
                'water_atom_start': 1,
                'water_atom_end': 2880,
                'methanol_atom_start': 2881,
                'methanol_atom_end': 4320,
                'water_mol_start': 1,
                'water_mol_end': 960,
                'methanol_mol_current_start': 1,
                'methanol_mol_current_end': 240,
                'methanol_mol_target_start': 961,
                'methanol_mol_target_end': 1200
            },
            'methanol_600_water_600-hydrophilic': {
                'n_water': 600,
                'n_methanol': 600,
                'water_atom_start': 1,
                'water_atom_end': 1800,
                'methanol_atom_start': 1801,
                'methanol_atom_end': 5400,
                'water_mol_start': 1,
                'water_mol_end': 600,
                'methanol_mol_current_start': 1,
                'methanol_mol_current_end': 600,
                'methanol_mol_target_start': 601,
                'methanol_mol_target_end': 1200
            },
            'methanol_600_water_600-hydrophobic': {
                'n_water': 600,
                'n_methanol': 600,
                'water_atom_start': 1,
                'water_atom_end': 1800,
                'methanol_atom_start': 1801,
                'methanol_atom_end': 5400,
                'water_mol_start': 1,
                'water_mol_end': 600,
                'methanol_mol_current_start': 1,
                'methanol_mol_current_end': 600,
                'methanol_mol_target_start': 601,
                'methanol_mol_target_end': 1200
            }
        }
        return env_configs.get(self.env)
    
    def validate_molecule_structure(self):
        """
        Validate that molecule IDs follow expected patterns and match atom annotations
        """
        env_config = self._get_environment_config()
        if not env_config:
            if self.verbose:
                print("\n--- Skipping structure validation: unknown environment")
            return
        
        if self.verbose:
            print(f"\n--- Validating molecule structure for {self.env}...")
        
        # Collect validation results
        water_issues = []
        methanol_issues = []
        annotation_issues = []
        
        # Group atoms by mol_id for water molecules
        water_mol_groups = {}
        for atom in self.atom_lines:
            atom_id = int(atom[0])
            mol_id = int(atom[1])
            
            # Check if this is a water atom
            if (env_config['water_atom_start'] <= atom_id <= env_config['water_atom_end'] and
                env_config['water_mol_start'] <= mol_id <= env_config['water_mol_end']):
                
                if mol_id not in water_mol_groups:
                    water_mol_groups[mol_id] = []
                water_mol_groups[mol_id].append(atom)
        
        # Group atoms by mol_id for methanol molecules (current mol_ids)
        methanol_mol_groups = {}
        for atom in self.atom_lines:
            atom_id = int(atom[0])
            mol_id = int(atom[1])
            
            # Check if this is a methanol atom
            if (env_config['methanol_atom_start'] <= atom_id <= env_config['methanol_atom_end'] and
                env_config['methanol_mol_current_start'] <= mol_id <= env_config['methanol_mol_current_end']):
                
                if mol_id not in methanol_mol_groups:
                    methanol_mol_groups[mol_id] = []
                methanol_mol_groups[mol_id].append(atom)
        
        # Validate water molecules (should have 3 atoms each)
        for mol_id, atoms in water_mol_groups.items():
            if len(atoms) != 3:
                water_issues.append(f"Water mol_id {mol_id} has {len(atoms)} atoms (expected 3)")
        
        # Validate methanol molecules (should have 6 atoms each)
        for mol_id, atoms in methanol_mol_groups.items():
            if len(atoms) != 6:
                methanol_issues.append(f"Methanol mol_id {mol_id} has {len(atoms)} atoms (expected 6)")
        
        # Check annotation consistency for all atoms
        for atom in self.atom_lines:
            mol_id = int(atom[1])
            atom_type = int(atom[2])
            
            # Get expected molecule type based on mol_id
            expected_mol_type = None
            if mol_id == 1201:
                expected_mol_type = 'FAU'
            elif mol_id == 1202:
                expected_mol_type = 'ADS'
            elif env_config['water_mol_start'] <= mol_id <= env_config['water_mol_end']:
                expected_mol_type = 'HOH'
            elif env_config['methanol_mol_current_start'] <= mol_id <= env_config['methanol_mol_current_end']:
                expected_mol_type = 'MEO'
            
            # Get element from atom type
            element = self.atom_type_to_element.get(atom_type, 'Unknown')
            
            # Check if atom has annotation (if it's from original file)
            if len(atom) > 7:
                # Parse existing annotation
                comment_parts = ' '.join(atom[7:])
                if comment_parts.startswith('#'):
                    comment_parts = comment_parts[1:].strip()
                
                # Extract element and mol type from comment
                if '_' in comment_parts:
                    comment_element, comment_mol_type = comment_parts.split('_', 1)
                    
                    # Check element consistency
                    if comment_element != element:
                        annotation_issues.append(f"Atom {atom[0]}: element mismatch - atom_type {atom_type} is {element} but comment says {comment_element}")
                    
                    # Check mol type consistency
                    if expected_mol_type and comment_mol_type != expected_mol_type:
                        annotation_issues.append(f"Atom {atom[0]}: mol_type mismatch - mol_id {mol_id} should be {expected_mol_type} but comment says {comment_mol_type}")
        
        # Print validation results
        if self.verbose:
            print(f"    Water molecules validation:")
            print(f"      Expected: {env_config['n_water']} molecules with 3 atoms each")
            print(f"      Found: {len(water_mol_groups)} molecules")
            if water_issues:
                print(f"      Issues found: {len(water_issues)}")
                for issue in water_issues[:5]:  # Show first 5 issues
                    print(f"        - {issue}")
                if len(water_issues) > 5:
                    print(f"        ... and {len(water_issues) - 5} more issues")
            else:
                print(f"      ✅ All water molecules have correct structure")
            
            print(f"    Methanol molecules validation:")
            print(f"      Expected: {env_config['n_methanol']} molecules with 6 atoms each")
            print(f"      Found: {len(methanol_mol_groups)} molecules")
            if methanol_issues:
                print(f"      Issues found: {len(methanol_issues)}")
                for issue in methanol_issues[:5]:  # Show first 5 issues
                    print(f"        - {issue}")
                if len(methanol_issues) > 5:
                    print(f"        ... and {len(methanol_issues) - 5} more issues")
            else:
                print(f"      ✅ All methanol molecules have correct structure")
            
            print(f"    Annotation consistency validation:")
            if annotation_issues:
                print(f"      Issues found: {len(annotation_issues)}")
                for issue in annotation_issues[:5]:  # Show first 5 issues
                    print(f"        - {issue}")
                if len(annotation_issues) > 5:
                    print(f"        ... and {len(annotation_issues) - 5} more issues")
            else:
                print(f"      ✅ All atom annotations are consistent")
    
    def correct_molecule_ids(self):
        """
        Correct the molecule IDs for cosolvent methanol molecules
        """
        env_config = self._get_environment_config()
        
        # First, validate the current molecule structure
        self.validate_molecule_structure()
        
        if self.verbose:
            print(f"\n--- Correcting molecule IDs for {self.env}...")
        
        corrections_made = 0
        
        # Correct methanol molecule IDs
        for atom in self.atom_lines:
            atom_id = int(atom[0])
            current_mol_id = int(atom[1])
            
            # Check if this atom is in the methanol region
            if (env_config['methanol_atom_start'] <= atom_id <= env_config['methanol_atom_end'] and
                env_config['methanol_mol_current_start'] <= current_mol_id <= env_config['methanol_mol_current_end']):
                
                # Calculate new mol_id: shift by the offset
                offset = env_config['methanol_mol_target_start'] - env_config['methanol_mol_current_start']
                new_mol_id = current_mol_id + offset
                
                # Update the mol_id in the atom line
                atom[1] = str(new_mol_id)
                corrections_made += 1
        
        if self.verbose:
            print(f"    Corrected {corrections_made} atom mol_ids")
            print(f"    Methanol mol_ids changed from {env_config['methanol_mol_current_start']}-{env_config['methanol_mol_current_end']} to {env_config['methanol_mol_target_start']}-{env_config['methanol_mol_target_end']}")
    
    def write_file(self):
        """
        Write the corrected LAMMPS data file
        """
        if self.verbose:
            print(f"\n--- Writing corrected LAMMPS data file: {self.output_file}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(self.output_file, 'w') as f:
            # Write header
            for line in self.header:
                f.write(line)
            
            # Add empty line after Atoms header (like in original file)
            f.write("\n")
            
            # Write atoms
            for atom in self.atom_lines:
                # Format the atom line
                atom_line = f"{atom[0]:>8} {atom[1]:>8} {atom[2]:>3} {atom[3]:>12} {atom[4]:>12} {atom[5]:>12} {atom[6]:>12}"
                
                # Generate new comment format: # Element_MOL
                mol_id = int(atom[1])
                atom_type = int(atom[2])
                
                # Get element from atom type
                element = self.atom_type_to_element.get(atom_type, 'X')  # 'X' as fallback
                
                # Determine molecule type based on mol_id
                if mol_id == 1201:
                    mol_name = 'FAU'
                elif mol_id == 1202:
                    mol_name = 'ADS'
                else:
                    # For water and methanol, use existing logic
                    env_config = self._get_environment_config()
                    if env_config:
                        if env_config['water_mol_start'] <= mol_id <= env_config['water_mol_end']:
                            mol_name = 'HOH'
                        elif env_config['methanol_mol_target_start'] <= mol_id <= env_config['methanol_mol_target_end']:
                            mol_name = 'MEO'
                        else:
                            mol_name = 'UNK'  # Unknown
                    else:
                        mol_name = 'UNK'
                
                # Create comment in new format
                comment = f"# {element}_{mol_name}"
                atom_line += f" {comment}"
                
                f.write(atom_line + "\n")
            
            # Add empty line after atoms section (like in original file)
            f.write("\n")
            
            # Write footer (bonds, etc.)
            for line in self.footer:
                f.write(line)
        
        if self.verbose:
            print(f"    Successfully wrote {len(self.atom_lines)} atoms to {self.output_file}")
    
    def _parse_masses_line(self, line: str):
        """Parse a line from the Masses section"""
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    atom_type = int(parts[0])
                    mass = float(parts[1])
                    
                    # Find matching element within tolerance
                    element = None
                    for ref_mass, ref_element in atom_weight_to_element.items():
                        if abs(mass - ref_mass) <= MASS_TOLERANCE:
                            element = ref_element
                            break
                    
                    if element:
                        self.atom_type_to_element[atom_type] = element
                    else:
                        self.atom_type_to_element[atom_type] = f"Unknown_{mass:.3f}"
                        
                except ValueError:
                    pass
    
    def _parse_atom_line(self, line: str):
        """Parse a line from the Atoms section"""
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 7:  # atom_id mol_id atom_type charge x y z
                self.atom_lines.append(parts)
    
    def _parse_bonds_line(self, line: str):
        """Parse a line from the Bonds section"""
        if line and not line.startswith("#"):
            parts = line.split()
            if len(parts) >= 4:  # bond_id bond_type atom1 atom2
                try:
                    bond_id = int(parts[0])
                    bond_type = int(parts[1])
                    atom1 = int(parts[2])
                    atom2 = int(parts[3])
                    self.bonds.append((bond_id, bond_type, atom1, atom2))
                except ValueError:
                    pass



if __name__ == '__main__':
    
    # Use one adsorbate for testing
    test = False # True False

    # test zeolite types
    if test:
        ZEOLITE_TYPES = ['FAU']
        ADSORBATES_BY_ENV = {'methanol_120_water_1080-hydrophilic': ['01_methanol'],
                             'methanol_120_water_1080-hydrophobic': ['01_methanol'],
                             'methanol_240_water_960-hydrophilic': ['02_01_02_propanol'],
                             'methanol_240_water_960-hydrophobic': ['02_propanol'],
                             'methanol_600_water_600-hydrophilic': ['08_01_ethene_glycol'],
                             'methanol_600_water_600-hydrophobic': ['07_01_ethene_glycol'],
                             }
    else:
        ZEOLITE_TYPES = ['FAU']
        ADSORBATES_BY_ENV = {'methanol_120_water_1080-hydrophilic': [
                                                '01_methanol',
                                                '02_01_02_propanol',
                                                '04_05_glycerol',
                                                '05_3c_aldehyde',
                                                '08_01_ethene_glycol',
                                                '11_01_propylene_glycol',
                                                ],

                            'methanol_120_water_1080-hydrophobic': [
                                                '01_methanol',
                                                '02_propanol',
                                                '04_3c_aldehyde',
                                                '06_02_glycerol',
                                                '07_01_ethene_glycol',
                                                '08_05_propylene_glycol',
                                                ],

                            'methanol_240_water_960-hydrophilic': [
                                                '01_methanol',
                                                '02_01_02_propanol',
                                                '04_05_glycerol',
                                                '05_3c_aldehyde',
                                                '08_01_ethene_glycol',
                                                '11_01_propylene_glycol',
                                                ],

                            'methanol_240_water_960-hydrophobic': [
                                                '01_methanol',
                                                '02_propanol',
                                                '04_3c_aldehyde',
                                                '06_02_glycerol',
                                                '07_01_ethene_glycol',
                                                '08_05_propylene_glycol',
                                                ],

                            'methanol_600_water_600-hydrophilic': [
                                                '01_methanol',
                                                '02_01_02_propanol',
                                                '04_05_glycerol',
                                                '05_3c_aldehyde',
                                                '08_01_ethene_glycol',
                                                '11_01_propylene_glycol',
                                                ],

                            'methanol_600_water_600-hydrophobic': [
                                                '01_methanol',
                                                '02_propanol',
                                                '04_3c_aldehyde',
                                                '06_02_glycerol',
                                                '07_01_ethene_glycol',
                                                '08_05_propylene_glycol',
                                                ],
                            }
    
    # The path to the simulation folder
    base = get_paths("simulation_path")
    
    # Iterate over each zeolite type
    for zeolite in ZEOLITE_TYPES:
        
        zeolite_dir = os.path.join(base, zeolite)
        
        # Iterate over each environment and its adsorbates
        for env, adsorbates in ADSORBATES_BY_ENV.items():
            env_dir = os.path.join(zeolite_dir, env)
            print(f"\n\n--- Environment: {env}")
            # Iterate over each adsorbate
            for ads in adsorbates:
                print('    adsorbate: ', ads)
                ads_dir = os.path.join(env_dir, ads)
                print('    adsorbate_dir: ', ads_dir)

                inp = os.path.join(ads_dir, 'fau_ad_co.lammpsdata')
                out = os.path.join(ads_dir, 'data_nvt_samp_new.lammpsdata')
                
                lammps = CosolventLammpsDataProcessor(inp,
                                             out,
                                             verbose=True,
                                             zeolite=zeolite,
                                             env=env,
                                             ads=ads,
                                             )
