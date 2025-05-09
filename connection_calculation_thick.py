#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:07:09 2025

@author: xuanqianyu
"""

import torch
import time
import pandas as pd

from Two_Bubble_Thick_Calculation import Two_Bubble_Thick_Calculation

def profile_configuration(para, initial_profile):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    dtype = torch.float32
    
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
    
    return X.slice_phi.cpu(), X.slice_kinetic.cpu(), X.slice_gradient.cpu(), X.slice_potential.cpu(), X.slice_total.cpu(),\
        X.profile_phi_z_inte, X.profile_phi_r_inte, X.t_lattice_inte,\
            X.z_lattice_inte, X.r_lattice_inte, X.d


w_value = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)

d_value = (29.7406, 16.0205, 11.0455, 8.81836, 7.60744, 6.90004, 6.50339, 6.29604, 6.17939, 6.12687)

T_value = (25, 14, 10, 8, 7, 6, 6, 6, 6, 6)

Z_value = (45, 24, 18, 13.5, 12, 10.5, 10.5, 9.75, 9.75, 9.75)

R_value = (45, 24, 18, 13.5, 12, 10.5, 10.5, 9.75, 9.75, 9.75)

NT_value = (1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5, 1*10**5)

NZ_value = (8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2)

NR_value = (8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2, 8*10**2)

para0 = (w_value[0], d_value[0], T_value[0], Z_value[0], R_value[0], NT_value[0], NZ_value[0], NR_value[0])
para1 = (w_value[1], d_value[1], T_value[1], Z_value[1], R_value[1], NT_value[1], NZ_value[1], NR_value[1])
para2 = (w_value[2], d_value[2], T_value[2], Z_value[2], R_value[2], NT_value[2], NZ_value[2], NR_value[2])
para3 = (w_value[3], d_value[3], T_value[3], Z_value[3], R_value[3], NT_value[3], NZ_value[3], NR_value[3])
para4 = (w_value[4], d_value[4], T_value[4], Z_value[4], R_value[4], NT_value[4], NZ_value[4], NR_value[4])
para5 = (w_value[5], d_value[5], T_value[5], Z_value[5], R_value[5], NT_value[5], NZ_value[5], NR_value[5])
para6 = (w_value[6], d_value[6], T_value[6], Z_value[6], R_value[6], NT_value[6], NZ_value[6], NR_value[6])
para7 = (w_value[7], d_value[7], T_value[7], Z_value[7], R_value[7], NT_value[7], NZ_value[7], NR_value[7])
para8 = (w_value[8], d_value[8], T_value[8], Z_value[8], R_value[8], NT_value[8], NZ_value[8], NR_value[8])
para9 = (w_value[9], d_value[9], T_value[9], Z_value[9], R_value[9], NT_value[9], NZ_value[9], NR_value[9])

Para = (para0,para1,para2,para3,para4,para5,para6,para7,para8,para9)

i = 0
PARA = Para[i]
print(PARA)

initial_profile = pd.read_csv(f'./InitialProfiles/Profile_{i}.csv', header=None).values

slice_filename = f"./data_fit/phi_configuration_{i}.pt"
kinetic_filename = f"./data_fit/kinetic_configuration_{i}.pt"
gradient_filename = f"./data_fit/gradient_configuration_{i}.pt"
potential_filename = f"./data_fit/potential_configuration_{i}.pt"
total_filename = f"./data_fit/total_configuration_{i}.pt"

slice_phi_z_filename = f"./data_pre_fit/phi_z_profile_{i}.pt"
slice_phi_r_filename = f"./data_pre_fit/phi_r_profile_{i}.pt"
slice_t_filename_gw = f"./data_pre_fit/t_lattice_{i}.pt"
slice_z_filename_gw = f"./data_pre_fit/z_lattice_{i}.pt"
slice_r_filename_gw = f"./data_pre_fit/r_lattice_{i}.pt"
d_filename = f"./data_pre_fit/d_{i}.pt"

start = time.time()            
profile_total = profile_configuration(PARA, initial_profile)
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

torch.save(slice_phi_z_data,slice_phi_z_filename)
torch.save(slice_phi_r_data,slice_phi_r_filename)
torch.save(slice_t_data,slice_t_filename_gw)
torch.save(slice_z_data,slice_z_filename_gw)
torch.save(slice_r_data,slice_r_filename_gw)
torch.save(slice_d,d_filename)

print("download finished")



