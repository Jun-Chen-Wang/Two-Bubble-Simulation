#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:08:25 2025

@author: xuanqianyu
"""

import torch
import time
import numpy as np

from gw_integration_Simpson import gw_integration_Simpson
from Two_Bubble_Classic_Calculation import Two_Bubble_Classic_Calculation


def profile_configuration(para, device, dtype):
    
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
    
    return X.slice_phi.cpu(), X.slice_kinetic.cpu(), X.slice_gradient.cpu(), X.slice_potential.cpu(), X.slice_total.cpu(),\
        X.profile_phi_z_inte, X.profile_phi_r_inte, X.t_lattice_inte,\
            X.z_lattice_inte, X.r_lattice_inte, X.d




w_value = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 0.1)


Aw_value = (1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 3/2)

Rw_value = (1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1, 1/1)


d_value = (29.7406, 20.9798, 16.0205, 13.0646, 11.0455, 9.72099, 8.81836, 8.12678, 7.60744, 35)


T_value = (30, 23, 20, 18, 17, 17, 25, 45, 45, 30)

Z_value = (50, 35, 30, 25, 24, 20, 20, 15, 15, 50)

R_value = (50, 35, 30, 25, 24, 20, 20, 15, 15, 50)


NT_value = (1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5)

NZ_value = (8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2)

NR_value = (8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2)


para0 = (w_value[0], Aw_value[0], Rw_value[0], d_value[0], T_value[0], Z_value[0], R_value[0], NT_value[0], NZ_value[0], NR_value[0])
para1 = (w_value[1], Aw_value[1], Rw_value[1], d_value[1], T_value[1], Z_value[1], R_value[1], NT_value[1], NZ_value[1], NR_value[1])
para2 = (w_value[2], Aw_value[2], Rw_value[2], d_value[2], T_value[2], Z_value[2], R_value[2], NT_value[2], NZ_value[2], NR_value[2])
para3 = (w_value[3], Aw_value[3], Rw_value[3], d_value[3], T_value[3], Z_value[3], R_value[3], NT_value[3], NZ_value[3], NR_value[3])
para4 = (w_value[4], Aw_value[4], Rw_value[4], d_value[4], T_value[4], Z_value[4], R_value[4], NT_value[4], NZ_value[4], NR_value[4])
para5 = (w_value[5], Aw_value[5], Rw_value[5], d_value[5], T_value[5], Z_value[5], R_value[5], NT_value[5], NZ_value[5], NR_value[5])
para6 = (w_value[6], Aw_value[6], Rw_value[6], d_value[6], T_value[6], Z_value[6], R_value[6], NT_value[6], NZ_value[6], NR_value[6])
para7 = (w_value[7], Aw_value[7], Rw_value[7], d_value[7], T_value[7], Z_value[7], R_value[7], NT_value[7], NZ_value[7], NR_value[7])
para8 = (w_value[8], Aw_value[8], Rw_value[8], d_value[8], T_value[8], Z_value[8], R_value[8], NT_value[8], NZ_value[8], NR_value[8])
para9 = (w_value[9], Aw_value[9], Rw_value[9], d_value[9], T_value[9], Z_value[9], R_value[9], NT_value[9], NZ_value[9], NR_value[9])


Para = (para0,para1,para2,para3,para4,para5,para6,para7,para8,para9)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float32


RangeChoice = np.arange(9)


for i in RangeChoice:
    
    PARA = Para[i]
    print(PARA)

    slice_filename = f"./data_fit_classic/phi_configuration_{i}.pt"
    kinetic_filename = f"./data_fit_classic/kinetic_configuration_{i}.pt"
    gradient_filename = f"./data_fit_classic/gradient_configuration_{i}.pt"
    potential_filename = f"./data_fit_classic/potential_configuration_{i}.pt"
    total_filename = f"./data_fit_classic/total_configuration_{i}.pt"
    
    start = time.time()            
    profile_total = profile_configuration(PARA, device, dtype)
    end = time.time()
    print(end-start)
    
    
    slice_file_data = profile_total[0].clone().detach()
    kinetic_file_data = profile_total[1].clone().detach()
    gradient_file_data = profile_total[2].clone().detach()
    potential_file_data = profile_total[3].clone().detach()
    total_file_data = profile_total[4].clone().detach()

    slice_phi_z_data = profile_total[5].clone().detach()
    slice_phi_r_data = profile_total[6].clone().detach()
    slice_t_data = profile_total[7].clone().detach()
    slice_z_data = profile_total[8].clone().detach()
    slice_r_data = profile_total[9].clone().detach()
    slice_d = profile_total[10].clone().detach()

    torch.save(slice_file_data,slice_filename)
    torch.save(kinetic_file_data,kinetic_filename)
    torch.save(gradient_file_data,gradient_filename)
    torch.save(potential_file_data,potential_filename)
    torch.save(total_file_data,total_filename)
    
    print(f"{i} : Solve PDE finished")
    
    
    start = time.time()       
    GW = gw_integration_Simpson(slice_phi_z_data, slice_phi_r_data, slice_t_data, slice_z_data, slice_r_data, slice_d, device, dtype)
    GW.GW_configuration_New()
    end = time.time()
    print(end-start)

    gw_spectrum = GW.GW_spectrum_profile_New.cpu()
    GW_filename = f"./data_fit_classic/gw_spectrum_{i}.pt"
    torch.save(gw_spectrum,GW_filename)

    print(f"{i} : GW calculation finished")
    
    
    
    
    
    
    