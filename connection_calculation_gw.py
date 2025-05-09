#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 09:57:09 2025

@author: xuanqianyu
"""

import torch
import time
from gw_integration_Simpson import gw_integration_Simpson

i = 0

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float64

phi_z = torch.load(f"./data_pre_fit/phi_z_profile_{i}.pt", weights_only=True)
phi_r = torch.load(f"./data_pre_fit/phi_r_profile_{i}.pt", weights_only=True)
t_lattice = torch.load(f"./data_pre_fit/t_lattice_{i}.pt", weights_only=True)
z_lattice = torch.load(f"./data_pre_fit/z_lattice_{i}.pt", weights_only=True)
r_lattice = torch.load(f"./data_pre_fit/r_lattice_{i}.pt", weights_only=True)
d = torch.load(f"./data_pre_fit/d_{i}.pt", weights_only=True)

start = time.time()       
GW = gw_integration_Simpson(phi_z,phi_r,t_lattice,z_lattice,r_lattice,d,device,dtype)
GW.GW_configuration_New()
end = time.time()
print(end-start)

gw_spectrum = GW.GW_spectrum_profile_New.cpu()
GW_filename = f"./data_fit/gw_spectrum_{i}.pt"
torch.save(gw_spectrum,GW_filename)

print("download finished")