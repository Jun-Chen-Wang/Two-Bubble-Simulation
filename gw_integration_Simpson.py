#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 09:23:06 2025

@author: xuanqianyu
"""

import torch
from functools import cached_property

class gw_integration_Simpson(object):
    G = 1
    Num_omega = 100
    Num_th_k = 500
    N_print_omega = 2
    delta = 10**(-12)
    omega_min_time_scale_d = 2*10**(-1)
    omega_max_time_scale_d = 2*10**(1)
    
    def __init__(self,phi_z,phi_r,t_lattice,z_lattice,r_lattice,d,device,dtype):
        self.device = device
        self.dtype = dtype
        self.phi_z = phi_z
        self.phi_r = phi_r
        self.dth_k = torch.pi/self.Num_th_k
        self.th_k_lattice = torch.tensor([i*self.dth_k for i in torch.arange(self.Num_th_k)],device=self.device,dtype=self.dtype)
        self.d = d
        self.omega_min = self.omega_min_time_scale_d/d
        self.omega_max = self.omega_max_time_scale_d/d
        self.omega_lattice = torch.logspace(torch.log10(self.omega_min),torch.log10(self.omega_max),self.Num_omega)
        self.t_lattice_inte = t_lattice
        self.z_lattice_inte = z_lattice
        self.r_lattice_inte = r_lattice
        self.NumT_inte = len(self.t_lattice_inte)
        self.NumZ_inte = len(self.z_lattice_inte)
        self.NumR_inte = len(self.r_lattice_inte)
        self.dt_inte = t_lattice[1]-t_lattice[0]
        self.dz_inte = z_lattice[1]-z_lattice[0]
        self.dr_inte = r_lattice[1]-r_lattice[0]
        
    @cached_property
    def r_tensor(self):
        r_tensor = torch.unsqueeze(torch.unsqueeze(self.r_lattice_inte, 0), 0)
        r_tensor = r_tensor.expand(self.NumT_inte,self.NumZ_inte,self.NumR_inte)
        return r_tensor
    
    @cached_property
    def z_tensor(self):
        z_tensor = torch.unsqueeze(torch.unsqueeze(self.z_lattice_inte, -1), 0)
        z_tensor = z_tensor.expand(self.NumT_inte,self.NumZ_inte,self.NumR_inte)
        return z_tensor
    
    @cached_property
    def t_tensor(self):
        t_tensor = torch.unsqueeze(torch.unsqueeze(self.t_lattice_inte, -1), -1)
        t_tensor = t_tensor.expand(self.NumT_inte,self.NumZ_inte,self.NumR_inte)
        return t_tensor
        
    def r_integration(self,configuration):
        integrand_odd = configuration[:,:,1:self.NumR_inte:2]
        integrand_even = configuration[:,:,2:self.NumR_inte-1:2]
        value_main = 4*torch.sum(integrand_odd,axis=2)*self.dr_inte/3 + 2*torch.sum(integrand_even,axis=2)*self.dr_inte/3
        value_min = (configuration[:,:,0]+configuration[:,:,-1])*self.dr_inte/3
        return value_main + value_min
    
    def z_integration(self,configuration):
        integrand_odd = configuration[:,1:self.NumZ_inte:2]
        integrand_even = configuration[:,2:self.NumZ_inte-1:2]
        value_main = 4*torch.sum(integrand_odd,axis=1)*self.dz_inte/3 + 2*torch.sum(integrand_even,axis=1)*self.dz_inte/3
        value_min = (configuration[:,0]+configuration[:,-1])*self.dz_inte/3
        return value_main + value_min
    
    def t_integration(self,configuration):
        integrand_odd = configuration[1:self.NumT_inte:2]
        integrand_even = configuration[2:self.NumT_inte-1:2]
        value_main = 4*torch.sum(integrand_odd,axis=0)*self.dt_inte/3 + 2*torch.sum(integrand_even,axis=0)*self.dt_inte/3
        value_min = (configuration[0]+configuration[-1])*self.dt_inte/3
        return value_main + value_min
    
    def th_k_integration(self,configuration):
        integrand_odd = configuration[1:self.Num_th_k:2]
        integrand_even = configuration[2:self.Num_th_k-1:2]
        value_main = 4*torch.sum(integrand_odd,axis=0)*self.dth_k/3 + 2*torch.sum(integrand_even,axis=0)*self.dth_k/3
        value_min = (configuration[0]+configuration[-1])*self.dth_k/3
        return value_main + value_min
    
    def integrand(self,omega,th_k):
        s_t = torch.sin(th_k)
        c_t = torch.cos(th_k)
        
        T_rr = -torch.exp(1j*omega*self.t_tensor)*torch.cos(omega*self.z_tensor*c_t)*self.r_tensor*((s_t**2)*torch.special.bessel_j0(omega*self.r_tensor*s_t)+(c_t**2+1)*(2*torch.special.bessel_j1(omega*self.r_tensor*s_t+self.delta)/(omega*self.r_tensor*s_t+self.delta)-torch.special.bessel_j0(omega*self.r_tensor*s_t+self.delta)))*(self.phi_r**2)
        T_xz = -2*torch.exp(1j*omega*self.t_tensor)*torch.sin(omega*self.z_tensor*c_t)*self.r_tensor*torch.special.bessel_j1(omega*self.r_tensor*s_t)*(self.phi_r*self.phi_z)
        T_zz = 2*torch.exp(1j*omega*self.t_tensor)*torch.cos(omega*self.z_tensor*c_t)*self.r_tensor*torch.special.bessel_j0(omega*self.r_tensor*s_t)*(self.phi_z**2)
        
        T_RR = self.t_integration(self.z_integration(self.r_integration(T_rr)))
        T_XZ = self.t_integration(self.z_integration(self.r_integration(T_xz)))
        T_ZZ = self.t_integration(self.z_integration(self.r_integration(T_zz)))
        
        return T_RR, T_XZ, T_ZZ
    
    def integration(self,omega):
        value = torch.zeros(self.Num_th_k,device=self.device,dtype=self.dtype)
        for i in torch.arange(self.Num_th_k):
            th_k = self.th_k_lattice[i]
            T_rr, T_xz, T_zz = self.integrand(omega,th_k)
            
            s_k = torch.sin(th_k)
            c_k = torch.cos(th_k)
                
            T_kk = T_rr + (s_k**2)*T_zz - 2*s_k*c_k*T_xz
            T_kk = torch.abs(T_kk)**2
            T_kk = T_kk*s_k
            
            value[i] = T_kk
        result = self.th_k_integration(value)
        result = 2*torch.pi*self.G*(omega)**2*result
        return result
    
    def GW_configuration_New(self):
        configuration = torch.zeros((self.Num_omega,2),device=self.device,dtype=self.dtype)
        print(f"GPU: {configuration.is_cuda}")
        for i in torch.arange(self.Num_omega):
            omega = self.omega_lattice[i]
            configuration[i][0] = omega*self.d/2
            configuration[i][1] = self.integration(omega)*omega
            if torch.fmod(i,self.N_print_omega)==0:
                print("%.3f%s" % (100*i/self.Num_omega, '% has been finished'))
        self.GW_spectrum_profile_New = configuration
        
        
        
        
        
        