#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:05:13 2025

@author: xuanqianyu
"""

import torch
import time
import numpy as np
import pandas as pd

from Two_Bubble_Thick_Calculation import Two_Bubble_Thick_Calculation
from Two_Bubble_Classic_Calculation import Two_Bubble_Classic_Calculation


def profile_configuration_classic(para, device, dtype):
    
    para_tensor = torch.tensor(para, device=device, dtype=dtype)
    
    w = para_tensor[0]
    A_by_A0 = para_tensor[1]
    R_by_R0 = para_tensor[2]
    d = para_tensor[3]
    T = para_tensor[4]
    Z = para_tensor[5]
    R = para_tensor[6]
    NT = para_tensor[7]
    NZ = para_tensor[8]
    NR = para_tensor[9]
    X = Two_Bubble_Classic_Calculation(w,A_by_A0,R_by_R0,d,T,Z,R,NT,NZ,NR,device,dtype)
    
    X.profile_configuration()
    
    return X.slice_phi.cpu()


def profile_configuration_quantum(para, initial_profile, device, dtype):
    
    para_tensor = torch.tensor(para, device=device, dtype=dtype)
    initial = torch.tensor(initial_profile, device=device, dtype=dtype)
    
    w = para_tensor[0]
    d = para_tensor[1]
    T = para_tensor[2]
    Z = para_tensor[3]
    R = para_tensor[4]
    NT = para_tensor[5]
    NZ = para_tensor[6]
    NR = para_tensor[7]
    X = Two_Bubble_Thick_Calculation(w,initial,d,T,Z,R,NT,NZ,NR,device,dtype)
    
    X.profile_configuration()
    
    return X.slice_phi.cpu()



w_value = (0.6, 0.8, 1.0)


Aw_value = (1/1, 1/1, 1/1)

Rw_value = (1/1, 1/1, 1/1)


d_value = (29.7406, 11.0455, 7.60744)


T_classic = (40, 17, 45)

Z_classic = (55, 20, 20)

R_classic = (55, 20, 20)


T_quantum = (25, 10, 7)

Z_quantum = (45, 20, 12)

R_quantum = (45, 20, 12)


NT = 2*10**5

NZ = 2*10**3

NR = 2*10**3


para0 = (w_value[0], Aw_value[0], Rw_value[0], d_value[0], T_classic[0], Z_classic[0], R_classic[0], NT, NZ, NR)
para1 = (w_value[1], Aw_value[1], Rw_value[1], d_value[1], T_classic[1], Z_classic[1], R_classic[1], NT, NZ, NR)
para2 = (w_value[2], Aw_value[2], Rw_value[2], d_value[2], T_classic[2], Z_classic[2], R_classic[2], NT, NZ, NR)
para3 = (w_value[0], d_value[0], T_quantum[0], Z_quantum[0], R_quantum[0], NT, NZ, NR)
para4 = (w_value[1], d_value[1], T_quantum[1], Z_quantum[1], R_quantum[1], NT, NZ, NR)
para5 = (w_value[2], d_value[2], T_quantum[2], Z_quantum[2], R_quantum[2], NT, NZ, NR)


Para = (para0,para1,para2,para3,para4,para5)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float32


RangeChoice = np.arange(3)

Ini = [0, 4, 8]


for i in RangeChoice:
    
    j = i + 3
    
    k = Ini[i]
    
    slice_filename_classic = f"./data_fit_show/phi_configuration_{i}.pt"
    slice_filename_quantum = f"./data_fit_show/phi_configuration_{j}.pt"
    
    
    PARA_classic = Para[i]
    print(PARA_classic)
    
    start = time.time()            
    profile_total_classic = profile_configuration_classic(PARA_classic, device, dtype)
    end = time.time()
    print(end-start)
    
    
    
    initial_profile = pd.read_csv(f'./InitialProfiles/Profile_{k}.csv', header=None).values
    
    PARA_quantum = Para[j]
    print(PARA_quantum)
    
    start = time.time()            
    profile_total_quantum = profile_configuration_quantum(PARA_quantum, initial_profile, device, dtype)
    end = time.time()
    print(end-start)
    
    
    
    slice_file_data_classic = profile_total_classic.clone().detach()
    
    slice_file_data_quantum = profile_total_quantum.clone().detach()
    

    torch.save(slice_file_data_classic,slice_filename_classic)
    torch.save(slice_file_data_quantum,slice_filename_quantum)
    
    
    print(f"{i} : download finished")
    
    
    
    
    
    
    
    
    