# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:03:48 2022

@author: xiutinc
"""

import numpy as np
import pandas as pd
import  matplotlib.pyplot  as plt

with open('traj_intE.txt','r') as traj_intE:
    lines=traj_intE.readlines()
    for i, line in enumerate(lines):
        if line.startswith('intE01/aq '):
            index_intE = i
            # print(index_intE)
traj_intE.close()

calc_intE=[]
for line in lines[index_intE:]:
    if line != []:
        calc_intE.append(line.split())
        
        
calc_intE_arr = np.array(list( filter( None, calc_intE) ) )
    

calc_intE_arr_reshape = calc_intE_arr[:,1].reshape(10,3).astype(float)

# zeo_ref_g = -13940.4115797905
zeo_ref_g = -6996.49192847317590

intE_frame = []

for x in calc_intE_arr_reshape:
    
    intE = ( x[0] - x[1] - ( x[2] - zeo_ref_g ) ) * 2625.5 / 96.487
    
    intE_frame.append(intE.round(decimals = 3))
    
intE_average = np.average(intE_frame).round(decimals = 4)
print (intE_frame)
print ('intE_dft:', intE_average)    

# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])    
# # ax.bar(range(10),intE_frame)


# bars = ax.barh(list(i for i in range(1,11)), intE_frame)
# ax.bar_label(bars)
# ax.axvline(intE_average, color='g', linewidth=2)


# ax.yaxis.tick_right()
# # plt.xlim(np.amin(intE_frame)-0.1, np.amax(intE_frame)+0.1)
# plt.xlim(np.amin(intE_frame)-0.1, 0)
# plt.show()   
# plt.savefig('intE_frames.png')    
