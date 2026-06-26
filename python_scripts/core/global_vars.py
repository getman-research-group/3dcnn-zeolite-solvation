# -*- coding: utf-8 -*-
"""
global_vars.py
This code contains all global variables.

"""
## Importing Functions
try:
    from path import get_base_paths
except:
    from core.path import get_base_paths
    

ATOMIC_RADII = {'H'   : 0.120, 'He'  : 0.140, 'Li'  : 0.076, 'Be' : 0.059,
                'B'   : 0.192, 'C'   : 0.170, 'N'   : 0.155, 'O'  : 0.152,
                'F'   : 0.147, 'Ne'  : 0.154, 'Na'  : 0.102, 'Mg' : 0.086,
                'Al'  : 0.184, 'Si'  : 0.210, 'P'   : 0.180, 'S'  : 0.180,
                'Cl'  : 0.181, 'Ar'  : 0.188, 'K'   : 0.138, 'Ca' : 0.114,
                'Sc'  : 0.211, 'Ti'  : 0.200, 'V'   : 0.200, 'Cr' : 0.200,
                'Mn'  : 0.200, 'Fe'  : 0.200, 'Co'  : 0.200, 'Ni' : 0.163,
                'Cu'  : 0.140, 'Zn'  : 0.139, 'Ga'  : 0.187, 'Ge' : 0.211,
                'As'  : 0.185, 'Se'  : 0.190, 'Br'  : 0.185, 'Kr' : 0.202,
                'Rb'  : 0.303, 'Sr'  : 0.249, 'Y'   : 0.200, 'Zr' : 0.200,
                'Nb'  : 0.200, 'Mo'  : 0.200, 'Tc'  : 0.200, 'Ru' : 0.200,
                'Rh'  : 0.200, 'Pd'  : 0.163, 'Ag'  : 0.172, 'Cd' : 0.158,
                'In'  : 0.193, 'Sn'  : 0.217, 'Sb'  : 0.206, 'Te' : 0.206,
                'I'   : 0.198, 'Xe'  : 0.216, 'Cs'  : 0.167, 'Ba' : 0.149,
                'La'  : 0.200, 'Ce'  : 0.200, 'Pr'  : 0.200, 'Nd' : 0.200,
                'Pm'  : 0.200, 'Sm'  : 0.200, 'Eu'  : 0.200, 'Gd' : 0.200,
                'Tb'  : 0.200, 'Dy'  : 0.200, 'Ho'  : 0.200, 'Er' : 0.200,
                'Tm'  : 0.200, 'Yb'  : 0.200, 'Lu'  : 0.200, 'Hf' : 0.200,
                'Ta'  : 0.200, 'W'   : 0.200, 'Re'  : 0.200, 'Os' : 0.200,
                'Ir'  : 0.200, 'Pt'  : 0.175, 'Au'  : 0.166, 'Hg' : 0.155,
                'Tl'  : 0.196, 'Pb'  : 0.202, 'Bi'  : 0.207, 'Po' : 0.197,
                'At'  : 0.202, 'Rn'  : 0.220, 'Fr'  : 0.348, 'Ra' : 0.283,
                'Ac'  : 0.200, 'Th'  : 0.200, 'Pa'  : 0.200, 'U'  : 0.186,
                'Np'  : 0.200, 'Pu'  : 0.200, 'Am'  : 0.200, 'Cm' : 0.200,
                'Bk'  : 0.200, 'Cf'  : 0.200, 'Es'  : 0.200, 'Fm' : 0.200,
                'Md'  : 0.200, 'No'  : 0.200, 'Lr'  : 0.200, 'Rf' : 0.200,
                'Db'  : 0.200, 'Sg'  : 0.200, 'Bh'  : 0.200, 'Hs' : 0.200,
                'Mt'  : 0.200, 'Ds'  : 0.200, 'Rg'  : 0.200, 'Cn' : 0.200,
                'Uut' : 0.200, 'Fl'  : 0.200, 'Uup' : 0.200, 'Lv' : 0.200,
                'Uus' : 0.200, 'Uuo' : 0.200,
                }

## Defining Adsorbates, Create a list contains all the possible adsorbates.
ADSORBATES_BY_ENV = {

    'methanol_120_water_1080-hydrophilic': [
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

    'water_pure-hydrophilic': [
                                '01_methanol',
                                '02_propanol',
                                '03_01_1_3_propanediol',
                                '03_02_1_3_propanediol',
                                '03_04_1_3_propanediol',
                                '03_06_1_3_propanediol',
                                '03_07_1_3_propanediol',
                                '03_08_1_3_propanediol',
                                '04_04_glycerol',
                                '04_05_glycerol',
                                '04_06_glycerol',
                                '05_3c_aldehyde',
                                '06_glycerol_intermediate',
                                '07_ethanol',
                                '08_01_ethene_glycol',
                                '08_02_ethene_glycol',
                                '09_01_propylene_glycol',
                                '09_02_propylene_glycol',
                                '10_01_isopropanol',
                                '11_01_propylene_glycol',
                                '12_formaldehyde',
                                '13_acetaldehyde',
                            ],
    
    'water_pure-hydrophobic': [
                                '01_methanol',
                                '02_propanol',
                                '03_ethanol',
                                '04_3c_aldehyde',
                                '05_01_13_propanediol',
                                '05_02_13_propanediol',
                                '05_07_13_propanediol',
                                '05_08_13_propanediol',
                                '06_01_glycerol',
                                '06_02_glycerol',
                                '06_05_glycerol',
                                '07_01_ethene_glycol',
                                '07_02_ethene_glycol',
                                '08_01_propylene_glycol',
                                '08_02_propylene_glycol',
                                '08_05_propylene_glycol_mid',
                                '09_formaldehyde',
                                '10_acetaldehyde',
                                '11_01_glycerol_intermediate',
                                '11_02_glycerol_intermediate',
                                '12_01_isopropanol',
                                '12_02_isopropanol'
    ],
}

## Define zeolite types
ZEOLITE_TYPES = [
                #  'BEA',
                #  'CHA',
                 'FAU',
                #  'MFI',
                 ]

## Define solvent types
SOLVENTS = ['HOH','MeCN','MeOH']

## Define adsorbate environments
ENVIRONMENTS = list(ADSORBATES_BY_ENV.keys())



FEATURE_LIST = [
                'atom_type_C',
                'atom_type_H',
                'atom_type_O',
                'is_hydrophobic',
                'is_donor',
                'is_acceptor',
                'is_hbonded',
                'is_hbonded_donor',
                'is_hbonded_acceptor',
                'atom_mass',
                'partial_charge',
                'valence',
                'LJ_epsilon',
                'LJ_sigma',
                ]


## CSV files containing target interaction energy labels
LABEL_CSV_FILES = [
                     'methanol_120_water_1080-hydrophilic.csv',
                     'methanol_120_water_1080-hydrophobic.csv',
                     'methanol_240_water_960-hydrophilic.csv',
                     'methanol_240_water_960-hydrophobic.csv',
                     'methanol_600_water_600-hydrophilic.csv',
                     'methanol_600_water_600-hydrophobic.csv',
                     'water_pure-hydrophilic.csv',
                     'water_pure-hydrophobic.csv',
                     ]



## Dictionary for Default Runs For CNN Networks
SPLIT_DICT = {}


## Defining Path to Main Data
PATH_MAIN_PROJECT = get_base_paths()


if __name__ == "__main__":
    print(PATH_MAIN_PROJECT)
    