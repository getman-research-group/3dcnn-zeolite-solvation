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

if len(sys.argv) < 4:
    print ("dft intE calc")
    print ("usage: traj_2_intE.py traj_file converged_xyz_file cutoff ")
    sys.exit()


traj_file = sys.argv[1]
converged_xyz_file = sys.argv[2]
cutoff = int(sys.argv[3])

# traj_file = 'intE01.traj'
# converged_xyz_file = 'converged_one_unit.xyz' 
# cutoff = 7



####  water trajactory information 

with open (traj_file, 'r') as traj:
    traj_lines = traj.readlines()
traj.close() 

traj_coords=[]
for traj_line in traj_lines[9:]:
    traj_coords.append(traj_line.split())
      
df_traj = pd.DataFrame(traj_coords,
                        columns = ['atom_id', 'mol_id', 'atom_type', 'charge', 'x', 'y', 'z'] )
df_traj[['atom_id', 'mol_id', 'atom_type']] = df_traj[['atom_id', 'mol_id', 'atom_type']].astype(int)
df_traj[['charge', 'x', 'y', 'z']] = df_traj[['charge', 'x', 'y', 'z']].astype(float)


###### mass and bond information extract from lammps file

with open('data_nvt_samp.lammpsdata','r') as lammpsfile:
    lines = lammpsfile.readlines()
lammpsfile.close()
   
for i, line in enumerate(lines):
    
    if line.startswith("Masses"):
        mass_line = i        
    if line.startswith("Pair"):
        pair_line = i              
    if line.startswith("Angles"):
        angle_line = i
        # print(lines[angle_line])  
        


############# adding mass information
# add mass to each atom
masses = []
for item in lines[mass_line+2 : pair_line-1]:
    mass = item.split()
    masses.append(mass)
    
# print(masses)
masses =np.reshape(masses, (16,2)).T
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

df_traj_ad     = df_traj[df_traj['atom_type']>10].copy()     # atom_type of adosrbate is from 11
df_traj_water  = df_traj[df_traj['atom_type']<3].copy()   # atom_type of adosrbate is 1,2

# mass_of_center_ad
mol_coords = []     
for atom_id in df_traj_ad['atom_id'].values[:]:
    # print(atom_id)
    x =    df_traj_ad .loc[df_traj_ad['atom_id'] == atom_id, 'x'   ].values[0]
    y =    df_traj_ad .loc[df_traj_ad['atom_id'] == atom_id, 'y'   ].values[0]
    z =    df_traj_ad .loc[df_traj_ad['atom_id'] == atom_id, 'z'   ].values[0]
    mass = df_traj_ad .loc[df_traj_ad['atom_id'] == atom_id, 'mass'].values[0]
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
print('adsorbate mass center coords:',
      [mass_of_center_ad[0],
      mass_of_center_ad[1],
      mass_of_center_ad[2]],
      )



############### angle information

angles = []

for item in lines[angle_line+2:]:
    angle = item.split()
    angles.append(angle[2:5])
# angles

df_angles = pd.DataFrame(angles, columns = ['H1', 'O', 'H2' ])
df_angles = df_angles.astype('int').sort_values(by=['O']).reset_index(drop=True)
# print (df_angles)


# # mass_of_center_water
mass_of_center_water = []
for mol in df_angles.loc[:].values.tolist():
    # print(mol)
    mol_coords = []
    for atom_id in mol:
        # print(atom_id)
        x =    df_traj_water.loc[df_traj_water['atom_id'] == atom_id, 'x'   ].item()
        y =    df_traj_water.loc[df_traj_water['atom_id'] == atom_id, 'y'   ].item()
        z =    df_traj_water.loc[df_traj_water['atom_id'] == atom_id, 'z'   ].item()
        mass = df_traj_water.loc[df_traj_water['atom_id'] == atom_id, 'mass'].item()
        mol_coords.append([atom_id, x, y, z, mass])
    # print(mol_coords)
    np_mol_coords = np.array(mol_coords).astype(np.float64)
    np_mol_x = np_mol_coords[:,1]
    np_mol_y = np_mol_coords[:,2]
    np_mol_z = np_mol_coords[:,3]
    np_mol_mass = np_mol_coords[:,4]
    
    ad_x = mass_of_center_ad[0]
    ad_y = mass_of_center_ad[1]
    ad_z = mass_of_center_ad[2]
    x_ave = np.average(np_mol_x, weights= np_mol_mass)         
    y_ave = np.average(np_mol_y, weights= np_mol_mass)
    z_ave = np.average(np_mol_z, weights= np_mol_mass)
    

    if abs(ad_x - x_ave)> 24.443/2:
        delta_x = 24.443 + min(ad_x, x_ave) - max(ad_x, x_ave)
        # print (delta_x)
    else:
        delta_x = abs(ad_x - x_ave)
        
    if abs(ad_y - y_ave)> 24.443/2:
        delta_y = 24.443 + min(ad_y, y_ave) - max(ad_y, y_ave)
        # print (delta_y)
    else:
        delta_y = abs(ad_y - y_ave)
    
    delta_z  = abs(ad_z - z_ave)
        
    r_0 = distance.euclidean(mass_of_center_ad, [x_ave, y_ave, z_ave])
    r = math.sqrt(delta_x**2 + delta_y**2 + delta_z**2 )
    mass_of_center_water.append([x_ave, y_ave, z_ave, r, r_0])
     
    
df_mass_of_center_water = pd.DataFrame(mass_of_center_water)
df_angles_center_water = pd.concat([df_angles, df_mass_of_center_water],
                                        axis = 1 )
# print(df_angles_center_water)
# cutoff = 10
df_sphere_water = df_angles_center_water[df_angles_center_water[3]<= cutoff ] ###################### cutoff #############

np_sphere_water = df_sphere_water.iloc[:,:3].to_numpy()

sphere_water = []
for mol in np_sphere_water[:,1]:
        oxygen_coords = df_traj.loc[df_traj['atom_id'] == mol, 
                                  ['atom_type', 'x', 'y', 'z']].values[0].tolist()
        # print(oxygen_coords)
        sphere_water.append(oxygen_coords)
        
for mol in np_sphere_water[:,[0,2]]:
    for atom_id in mol: 
            hydrogen_coords = df_traj.loc[df_traj['atom_id'] == atom_id, 
                                      ['atom_type', 'x', 'y', 'z']].values[0].tolist()
            sphere_water.append(hydrogen_coords)
        
# print(sphere_water)          

df_sphere_water = pd.DataFrame(sphere_water, columns = ['atom_type', 'x', 'y', 'z'])
num_water_sphere = df_sphere_water.loc[df_sphere_water['atom_type']==1].shape[0]
print ('Number of water molecules in cutoff %d is %d' % (cutoff, num_water_sphere ))

df_symbol_mass= pd.DataFrame(atomic_masses[1:], index=chemical_symbols[1:])    

atom_names = []
for atom_type in df_sphere_water['atom_type'].astype(int):
    # print(atom_type)
    atom_mass = df_masses.loc[str(atom_type)].item()
    # print (atom_mass)
    atom_name = df_symbol_mass[df_symbol_mass[0] == float(atom_mass)].index.item()
    atom_names.append(atom_name)

    
df_sphere_water['atom_name']=atom_names


########### zeolite and adosrbate geometry in gas phase

with open(converged_xyz_file,'r') as gas_xyz:
    readlines = gas_xyz.readlines()
gas_xyz.close()


list_gas_xyz = []
for line in readlines[2:]:
    coord = line.split()
    list_gas_xyz.append(coord)

### geometry of zeolite with adsorbate and zeolite refreence in dataframe    
df_gas_xyz = pd.DataFrame(list_gas_xyz, columns = ['atom_name', 'x', 'y', 'z'])
df_gas_xyz_zeo = df_gas_xyz.iloc[:-df_traj_ad.shape[0]]   




################### water needs to move along z axis to make it fix original zeolite box 
Ti_traj_coords_z = df_traj.loc[df_traj['atom_type']== 3, 'z'].item() 
Ti_gas_coords_z = df_gas_xyz.loc[df_gas_xyz['atom_name']=='Ti', 'z'].item()
z_shift = float(Ti_traj_coords_z) - float(Ti_gas_coords_z)   # geometry of system moved in NPT simulaiton

df_sphere_water_shift = df_sphere_water.copy()
df_sphere_water_shift['z'] = df_sphere_water['z'] - z_shift


df_aq_xyz = pd.concat([df_gas_xyz,
                        df_sphere_water_shift[['atom_name', 'x', 'y', 'z']]]
                      , axis = 0)

df_aq_xyz_zeo = pd.concat([df_gas_xyz_zeo,
                            df_sphere_water_shift[['atom_name', 'x', 'y', 'z']]],
                          axis = 0)

cartesian_coords_aq     = df_aq_xyz.to_string(header=False, index=False)
cartesian_coords_aq_zeo = df_aq_xyz_zeo.to_string(header=False, index=False)





aq_file_name = traj_file.split('.')[0] + '_aq.xyz'
aq_zeo_file_name = traj_file.split('.')[0] + '_aq_zeo.xyz'

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