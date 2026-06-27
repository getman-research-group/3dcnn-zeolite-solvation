# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:12:15 2022

@author: xiutinc
"""

import pandas as pd
import numpy as np
import sys


if len(sys.argv) < 3:
    print('remove one unit cell from gas phase zeolite geometry')
    print('usage: zeolite_ad_g.py converged.xyz one_unit.xyz')
    sys.exit()

initial_file = sys.argv[1]
final_file =sys.argv[2]



    
# initial_file =   'glycerol_04_converged.xyz' 
# final_file = 'glycerol_04_converged_one_unit.xyz'
    

with open (initial_file, 'r') as ini_xyz:
    readlines = ini_xyz.readlines()
ini_xyz.close()

coords = []

for line in readlines[2:]:
    coord = line.split()
    coords.append(coord)
    
df_xyz = pd.DataFrame(coords, columns=['atom', 'x', 'y', 'z'])
df_xyz.dropna(inplace=True)    

df_xyz.iloc[:,1:4] = df_xyz.iloc[:,1:4].astype(float)



df_xyz_shift = df_xyz.copy()
df_xyz_shift['z'] = df_xyz['z'] - 12.137

df_xyz_one_unit = df_xyz_shift[( df_xyz_shift['z'] < 24.433 ) & ( df_xyz_shift['z'] >= 0 )]

with open(final_file, 'w') as finalxyz:
    finalxyz.write(str(df_xyz_one_unit.shape[0]))
    finalxyz.write('\n\n')
    finalxyz.write(df_xyz_one_unit.to_string(index=False, header=False))
    
finalxyz.close()
        