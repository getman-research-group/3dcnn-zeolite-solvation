# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 17:56:55 2022

@author: xiutinc
"""

import pandas as pd
import numpy as np
import datetime
import ase
import sys, math
from scipy.spatial import distance
from ase.data import chemical_symbols
from ase.data import atomic_masses
from ase import Atoms

if len(sys.argv) < 4:
    print ("dft intE calc")
    print ("usage: traj_2_intE_cosol_3.0.py traj_file converged_xyz_file cutoff ")
    sys.exit()


traj_file = sys.argv[1]
converged_xyz_file = sys.argv[2]
cutoff = int(sys.argv[3])

# traj_file = 'intE01.traj'
# converged_xyz_file = 'converged_one_unit.xyz' 
# cutoff = 10



######  solvent trajactory information ############################################ 

with open (traj_file, 'r') as traj:
    traj_lines = traj.readlines()
traj.close() 

box =[float(i.split()[1]) for i in traj_lines[5:8]]

traj_coords=[]
for traj_line in traj_lines[9:]:
    traj_coords.append(traj_line.split())
      
df_traj = pd.DataFrame(traj_coords,
                        columns = ['atom_id', 'mol_id', 'atom_type', 'charge', 'x', 'y', 'z'] )
df_traj[['atom_id', 'mol_id', 'atom_type']] = df_traj[['atom_id', 'mol_id', 'atom_type']].astype(int)
df_traj[['charge', 'x', 'y', 'z']] = df_traj[['charge', 'x', 'y', 'z']].astype(float)

##################################################################################
###### mass and bond information extract from lammps file #########################
with open('data_nvt_samp.lammpsdata','r') as lammpsfile:
    lines = lammpsfile.readlines()
lammpsfile.close()

num_atom_type = int(lines[3].split()[0])
print (num_atom_type)


for line in lines:
    
    if line.startswith("Masses"):
        mass_line = lines.index(line)                       
############# adding mass information
# add mass to each atom
masses = []
for item in lines[mass_line+2 : num_atom_type + mass_line+2]:
    mass = item.split()
    masses.append(mass)
    
# print(masses)

masses =np.reshape(masses, (num_atom_type,2)).T
df_masses = pd.DataFrame(masses[1], columns = [ 'mass' ], index = masses[0])
# print(df_masses)

list_mass=[]
for at in df_traj['atom_type']:
    # print(at)
    mass = df_masses.loc[str(at) ,'mass']
    # print (at, mass)
    list_mass.append(mass)
# print(list_mass)
    
df_traj["mass"] = list_mass

df_traj=df_traj.astype('float')
####### type range of zeolite, water, methanol and adsorbate  #################
df_traj_water  = df_traj[df_traj['atom_type'].isin([2,5])].copy()   
df_traj_meoh   = df_traj[df_traj['atom_type'].isin([1,3,4,6])].copy()
df_traj_zeo    = df_traj[df_traj['mol_id']==1201].copy()
df_traj_ad     = df_traj[df_traj['mol_id']==1202].copy()   
#####################################################################
######### mass_of_center_ad   ####################
mol_coords = []     
for atom_id in df_traj_ad['atom_id'].values[:]:
    # print(atom_id)
    x =    df_traj_ad.loc[df_traj_ad['atom_id'] == atom_id, 'x'   ].values[0]
    y =    df_traj_ad.loc[df_traj_ad['atom_id'] == atom_id, 'y'   ].values[0]
    z =    df_traj_ad.loc[df_traj_ad['atom_id'] == atom_id, 'z'   ].values[0]
    mass = df_traj_ad.loc[df_traj_ad['atom_id'] == atom_id, 'mass'].values[0]
    mol_coords.append([atom_id, x, y, z, mass])
# print('adsorbate coords:', mol_coords)

np_mol_coords = np.array(mol_coords).astype(np.float16)
np_mol_x = np_mol_coords[:,1]
np_mol_y = np_mol_coords[:,2]
np_mol_z = np_mol_coords[:,3]
np_mol_mass = np_mol_coords[:,4]
# df_mol = pd.DataFrame(mol_coords)
x_ave = np.average(np_mol_x, weights= np_mol_mass)
y_ave = np.average(np_mol_y, weights= np_mol_mass)
z_ave = np.average(np_mol_z, weights= np_mol_mass)
mass_of_center_ad = [x_ave, y_ave, z_ave]
# print('adsorbate mass center coords:', mass_of_center_ad[:])

# ############### add distance to solvent information ###############################

### instead of using center of mass to pick solvent molecules , we use position of O atom to fine solvent molecule
# def com(solvent_dataframe):
#     mol_info = solvent_dataframe.mol_id.unique()
#     for mol_id in mol_info[:]:
#         df_mol = solvent_dataframe[solvent_dataframe['mol_id']==mol_id]
#         # print (df_mol)
#         com_x = np.average(df_mol['x'], weights= df_mol['mass'].astype(float)) 
#         com_y = np.average(df_mol['y'], weights= df_mol['mass'].astype(float)) 
#         com_z = np.average(df_mol['z'], weights= df_mol['mass'].astype(float)) 
#         # print(com_x, com_y, com_z)
#         ### distance from solvent molecule to adsorbate
#         ### period information pbc, and box size is [24,24,150] and using mic=True to make sure distance is periodic.
#         cube_cell = Atoms('H2', positions=[mass_of_center_ad, [com_x, com_y, com_z]], cell=box, pbc=True)
#         distance_mic = cube_cell.get_distance(0, 1, mic=True)
#         solvent_dataframe.loc[solvent_dataframe['mol_id']==mol_id, 'r2ad'] = distance_mic

# ### add r_com solvent to each solvent row
# com(df_traj_water)
# com(df_traj_meoh)   

def r_Oad(solvent_dataframe):
    mol_info = solvent_dataframe.mol_id.unique()
    for mol_id in mol_info[:]:
        df_mol = solvent_dataframe[solvent_dataframe['mol_id']==mol_id]
        # print (df_mol)
        O_sol = df_mol[df_mol['mass']==15.999][['x', 'y', 'z']].to_numpy()[0]
        # print(O_sol)
        ### distance from solvent molecule to adsorbate
        ### period information pbc, and box size is [24,24,150] and using mic=True to make sure distance is periodic.
        cube_cell = Atoms('H2', positions=[mass_of_center_ad, O_sol], cell=box, pbc=True)
        distance_mic = cube_cell.get_distance(0, 1, mic=True)
        solvent_dataframe.loc[solvent_dataframe['mol_id']==mol_id, 'r2ad'] = distance_mic

### add r_com solvent to each solvent row
r_Oad(df_traj_water)
r_Oad(df_traj_meoh)    

df_sphere_water = df_traj_water[df_traj_water['r2ad']<= cutoff]
df_sphere_meoh  = df_traj_meoh[df_traj_meoh['r2ad']<= cutoff]
#################################################################################
####################### add atom symbol ##################################
### generate mass to symbol database
df_mass2symbol=pd.DataFrame({'symbol':chemical_symbols[1:], 
                                      'mass':  atomic_masses[1:]})
### create a function to adding symbol when mass match to database
def mass_get_symbol(x):
    df_match = df_mass2symbol[df_mass2symbol['mass'] == x['mass']]
    return df_match['symbol'].values[0] if not df_match.empty else None    

# Apply the function row-wise
df_sphere_water = df_sphere_water.copy()
df_sphere_water['symbol'] = df_sphere_water.apply(mass_get_symbol, axis=1)
df_sphere_meoh = df_sphere_meoh.copy()
df_sphere_meoh['symbol']  = df_sphere_meoh.apply(mass_get_symbol, axis=1)

df_sphere_solvent = pd.concat([df_sphere_water, df_sphere_meoh], axis=0)[['symbol', 'x', 'y', 'z']]
################################################################################
########### zeolite and adosrbate geometry in gas phase

with open(converged_xyz_file,'r') as gas_xyz:
    readlines = gas_xyz.readlines()
gas_xyz.close()


list_gas_xyz = []
for line in readlines[2:]:
    coord = line.split()
    list_gas_xyz.append(coord)

### geometry of zeolite with adsorbate and zeolite refreence in dataframe    
df_gas_xyz = pd.DataFrame(list_gas_xyz, columns = ['symbol', 'x', 'y', 'z'])
df_gas_xyz_zeo = df_gas_xyz.iloc[:-df_traj_ad.shape[0]]   

# print("df_traj_ad:", df_traj_ad)


####################################################################################
##############  add liquid phase solvent molecule in lammps file to gas phase zeolite in xyz file 
##############  to generate zeo+ad+sol and zeo+sol file
### slvents need to move along z axis to make it fix zeolite box in gas phase
Ti_sol_coords_z = df_traj.loc[df_traj['atom_type']== 7, 'z'].item() 
Ti_gas_coords_z = df_gas_xyz.loc[df_gas_xyz['symbol']=='Ti', 'z'].item()
### coordinate of zeolite framework atoms of system moved in NPT simulaiton
### so we move all solvent molecules in liquid phase back using ti in gas phase as orgin
zshift = float(Ti_sol_coords_z) - float(Ti_gas_coords_z)
df_sphere_solvent_zshift = df_sphere_solvent.copy()
df_sphere_solvent_zshift['z'] = df_sphere_solvent['z'] - zshift

### xyz file of zeolite with solvent and adsorbate molecules
df_aq_xyz = pd.concat([df_gas_xyz, # zeolite w/ ad
                        df_sphere_solvent_zshift]
                        ,axis = 0)
### xyz file of zeolite with solvent without adsorbate molecules
df_aq_xyz_zeo = pd.concat([df_gas_xyz_zeo, # zeolite w/o ad
                          df_sphere_solvent_zshift],
                          axis = 0)

cartesian_coords_aq     = df_aq_xyz.to_string(header=False, index=False)
cartesian_coords_aq_zeo = df_aq_xyz_zeo.to_string(header=False, index=False)


aq_file_name = traj_file.split('.')[0] + '_aq.xyz'
aq_zeo_file_name = traj_file.split('.')[0] + '_aq_zeo.xyz'
#####################################################################################
############# write two files, one is zeo+ad+sol , ohter one is zeo+sol
with open(aq_file_name,"w") as xyz_file:
    # vasp.write(OUT_FILENAME)
    
    xyz_file.write(str(df_aq_xyz.shape[0]))
    xyz_file.write(''.join("\n\n"))
    xyz_file.write(cartesian_coords_aq)
        
xyz_file.close()
        
with open(aq_zeo_file_name,"w") as xyz_file:
    # vasp.write(OUT_FILENAME)
    
    xyz_file.write(str(df_aq_xyz_zeo.shape[0]))
    xyz_file.write(''.join("\n\n"))

    xyz_file.write(cartesian_coords_aq_zeo)
        
xyz_file.close()