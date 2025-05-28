#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:38:26 2025

@author: ivanbijamov
"""
import numpy as np
import pandas as pd
#Must modify file path to match where your version of the file is stored
#Ensure that you correct the headers of the file you use, as by default only
# 6 out of the 9 have headers, which causes it to break
dataImport = pd.read_csv('/Users/ivanbijamov/Library/Mobile Documents/com~apple~CloudDocs/SuperluminalJets/testbehemoth/data_3957413210177_xx_1.5_tx_0.5.csv')
radii = []
#from x
thetas = []
#from z
phis = []
for i in range(len(dataImport)):
    r=np.sqrt(dataImport.iloc[i,0]**2+dataImport.iloc[i,1]**2+dataImport.iloc[i,2]**2)
    phi = np.arctan2(dataImport.iloc[i,1],dataImport.iloc[i,0])
    theta = np.arccos(np.arctan(dataImport.iloc[i,2]/r))
    phis.append(phi)
    thetas.append(theta)
    radii.append(r)
newList = [0]*len(dataImport)
i=0
#returns radius, theta, phi, and velocity
for radius,theta,phi in zip(radii,thetas,phis):
    newList[i]=[radius,theta,phi,dataImport.iloc[i,8]]
    i+=1
print(newList)
