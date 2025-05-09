#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 08:53:01 2025

@author: xuanqianyu
"""

import torch
import torch.fft
from functools import cached_property

from Define_Potential import Potential_torch

class Two_Bubble_Classic_Calculation(object):
    delta_r = 10**(-3)
    nT_sample_plot = 400
    nZ_sample_plot = 400
    nR_sample_plot = 5
    nT_sample_inte = 500
    nZ_sample_inte = 500
    nR_sample_inte = 500
    N_print_confi = 10**3
    order = 2
    N_iteration = 100
    iteration_error = 10**(-3)
    
    def __init__(self,w,A_by_A0,R_by_R0,d,T,Z,R,NT,NZ,NR,device,dtype):
        self.device = device
        self.dtype = dtype
        
        self.w = w
        self.Potential = Potential_torch(self.w,self.device,self.dtype)
        
        self.Aini = self.Potential.A() * A_by_A0
        self.Rini = self.Potential.R() * R_by_R0
        
        self.d = d
        
        self.T = T
        self.Z = 2*Z
        self.R = 2*R
        
        self.NumT = int(NT)
        self.NumZ = int(2*NZ)
        self.NumR = int(2*NR)
        
        self.dt = self.T/self.NumT
        self.dz = self.Z/self.NumZ
        self.dr = self.R/self.NumR
        
        self.t_lattice = torch.tensor([i*self.dt for i in torch.arange(self.NumT)],device=self.device,dtype=self.dtype)
        self.z_lattice = torch.tensor([-self.Z/2 + i*self.dz for i in torch.arange(self.NumZ)],device=self.device,dtype=self.dtype)
        #self.r_lattice = torch.tensor([-self.R/2 + self.delta_r + i*self.dr for i in torch.arange(self.NumR)],device=self.device,dtype=self.dtype)
        self.r_lattice = torch.tensor([-self.R/2 + i*self.dr for i in torch.arange(self.NumR)],device=self.device,dtype=self.dtype)
        
        self.index_Z0 = int(NZ)
        self.Z_ini = self.z_lattice[self.index_Z0]
        self.index_R0 = int(NR)
        self.R_ini = self.r_lattice[self.index_R0]
        
        self.nT_space_plot = int(torch.ceil(torch.tensor(self.NumT/self.nT_sample_plot,device=self.device,dtype=self.dtype)))
        self.nZ_space_plot = int(torch.ceil(torch.tensor(self.index_Z0/self.nZ_sample_plot,device=self.device,dtype=self.dtype)))
        self.nR_space_plot = int(torch.ceil(torch.tensor(self.index_R0/self.nR_sample_plot,device=self.device,dtype=self.dtype)))
        
        self.nT_space_inte = int(torch.ceil(torch.tensor(self.NumT/self.nT_sample_inte,device=self.device,dtype=self.dtype)))
        self.nZ_space_inte = int(torch.ceil(torch.tensor(self.index_Z0/self.nZ_sample_inte,device=self.device,dtype=self.dtype)))
        self.nR_space_inte = int(torch.ceil(torch.tensor(self.index_R0/self.nR_sample_inte,device=self.device,dtype=self.dtype)))
        
        self.NumT_plot = int(torch.ceil(torch.tensor(self.NumT/self.nT_space_plot,device=self.device,dtype=self.dtype)))
        self.NumZ_plot = int(torch.ceil(torch.tensor(self.index_Z0/self.nZ_space_plot,device=self.device,dtype=self.dtype)))
        self.NumR_plot = int(torch.ceil(torch.tensor(self.index_R0/self.nR_space_plot,device=self.device,dtype=self.dtype)))
        
        self.NumT_inte = int(torch.ceil(torch.tensor(self.NumT/self.nT_space_inte,device=self.device,dtype=self.dtype)))
        self.NumZ_inte = int(torch.ceil(torch.tensor(self.index_Z0/self.nZ_space_inte,device=self.device,dtype=self.dtype)))
        self.NumR_inte = int(torch.ceil(torch.tensor(self.index_R0/self.nR_space_inte,device=self.device,dtype=self.dtype)))
        
        self.t_lattice_plot = self.t_lattice[0::self.nT_space_plot]
        self.z_lattice_plot = (self.z_lattice[self.index_Z0:])[0::self.nZ_space_plot]
        self.r_lattice_plot = (self.r_lattice[self.index_R0:])[0::self.nR_space_plot]
        
        self.t_lattice_inte = self.t_lattice[0::self.nT_space_inte]
        self.z_lattice_inte = (self.z_lattice[self.index_Z0:])[0::self.nZ_space_inte]
        self.r_lattice_inte = (self.r_lattice[self.index_R0:])[0::self.nR_space_inte]
        
    def change_profile(self,phi):
        phi_new = phi[self.index_Z0:,self.index_R0:]
        return phi_new
    
    @cached_property
    def Judge_zero(self):
        return torch.any(self.r_lattice == 0)
    
    @cached_property
    def r_tensor(self):
        r_tensor = torch.unsqueeze(self.r_lattice, 0)
        r_tensor = r_tensor.expand(self.NumZ,self.NumR)
        return r_tensor
    
    @cached_property
    def z_tensor(self):
        z_tensor = torch.unsqueeze(self.z_lattice, -1)
        z_tensor = z_tensor.expand(self.NumZ,self.NumR)
        return z_tensor
    
    @cached_property
    def  kr_tensor(self):
         kr = torch.fft.fftfreq(self.NumR, d=(self.dr)) * 2 * torch.pi
         kr_tensor = torch.unsqueeze(kr, 0)
         kr_tensor = kr_tensor.expand(self.NumZ,self.NumR)
         return kr_tensor.to(self.device)
     
    @cached_property
    def  kz_tensor(self):
         kz = torch.fft.fftfreq(self.NumZ, d=(self.dz)) * 2 * torch.pi
         kz_tensor = torch.unsqueeze(kz, -1)
         kz_tensor = kz_tensor.expand(self.NumZ,self.NumR)
         return kz_tensor.to(self.device)
    
    @cached_property
    def laplace_tensor(self):
        return -(self.kr_tensor**2+self.kz_tensor**2)
    
    @cached_property
    def initial_phi_t(self):
        dis_1 = torch.sqrt(self.r_tensor**2+(self.z_tensor-self.d/2)**2)
        dis_2 = torch.sqrt(self.r_tensor**2+(self.z_tensor+self.d/2)**2)
        value_1 = self.Aini * torch.exp(-dis_1**2 / (2 * self.Rini**2))
        value_2 = self.Aini * torch.exp(-dis_2**2 / (2 * self.Rini**2))
        return value_1 + value_2
    
    @cached_property
    def initial_phi(self):
        return torch.zeros((self.NumZ,self.NumR),device=self.device,dtype=self.dtype)
    
    def phi_R(self,phi0):
        phi0_ft = torch.fft.fftn(phi0)
        phir_ft = 1j*self.kr_tensor*phi0_ft
        phir = torch.fft.ifftn(phir_ft)
        return phir.real
    
    def phi_Z(self,phi0):
        phi0_ft = torch.fft.fftn(phi0)
        phiz_ft = 1j*self.kz_tensor*phi0_ft
        phiz = torch.fft.ifftn(phiz_ft)
        return phiz.real
    
    def phi_RR(self,phi0):
        phi0_ft = torch.fft.fftn(phi0)
        phir_ft = -self.kr_tensor**2*phi0_ft
        phir = torch.fft.ifftn(phir_ft)
        return phir.real
    
    def phi_ZZ(self,phi0):
        phi0_ft = torch.fft.fftn(phi0)
        phiz_ft = -self.kz_tensor**2*phi0_ft
        phiz = torch.fft.ifftn(phiz_ft)
        return phiz.real
    
    def phi_Laplace(self,phi0):
        phi0_ft = torch.fft.fftn(phi0)
        phil_ft = self.laplace_tensor*phi0_ft
        phil = torch.fft.ifftn(phil_ft)
        return phil.real
    
    def r_integration(self,configuration):
        integrand_odd = configuration[:,1:self.index_R0:2]
        integrand_even = configuration[:,2:self.index_R0-1:2]
        value_main = 4*torch.sum(integrand_odd,axis=1)*self.dr/3 + 2*torch.sum(integrand_even,axis=1)*self.dr/3
        value_min = (configuration[:,0]+configuration[:,-1])*self.dr/3
        return value_main + value_min
    
    def z_integration(self,configuration):
        integrand_odd = configuration[1:self.index_Z0:2]
        integrand_even = configuration[2:self.index_Z0-1:2]
        value_main = 4*torch.sum(integrand_odd,axis=0)*self.dz/3 + 2*torch.sum(integrand_even,axis=0)*self.dz/3
        value_min = (configuration[0]+configuration[-1])*self.dz/3
        return value_main + value_min
    
    def define_time_slice(self,profile,T_lattice):
        NT = len(T_lattice)
        for i in torch.arange(NT):
            t = T_lattice[i]
            array_i = torch.tensor([t,profile[i]],device=self.device,dtype=self.dtype)
            array_i = array_i.reshape(1,2)
            if i==0:
                array = array_i
            else:
                array = torch.cat((array,array_i),axis=0)
        return array
    
    def define_spacetime_slice(self,profile,T_lattice,Z_lattice,R_lattice):
        NT = len(T_lattice)
        NZ = len(Z_lattice)
        NR = len(R_lattice)
        for i in torch.arange(NT):
            t = T_lattice[i]
            for j in torch.arange(NZ):
                z = Z_lattice[j]
                for k in torch.arange(NR):
                    r = R_lattice[k]
                    array_k = torch.tensor([t,z,r,profile[i][j][k]],device=self.device,dtype=self.dtype)
                    array_k = array_k.reshape(1,4)
                    if k==0:
                        array_j = array_k
                    else:
                        array_j = torch.cat((array_j,array_k),axis=0)
                array_j = array_j.reshape(1,NR,4)
                if j==0:
                    array_i = array_j
                else:
                    array_i = torch.cat((array_i,array_j),axis=0)
            array_i = array_i.reshape(1,NZ,NR,4)
            if i==0:
                origin = array_i
            else:
                origin = torch.cat((origin,array_i),axis=0)
            
        for i in torch.arange(NT):
            t = T_lattice[i]
            for j in torch.arange(1,NZ):
                z = -Z_lattice[-j]
                for k in torch.arange(NR):
                    r = R_lattice[k]
                    array_k = torch.tensor([t,z,r,profile[i][-j][k]],device=self.device,dtype=self.dtype)
                    array_k = array_k.reshape(1,4)
                    if k==0:
                        array_j = array_k
                    else:
                        array_j = torch.cat((array_j,array_k),axis=0)
                array_j = array_j.reshape(1,NR,4)
                if j==1:
                    array_i = array_j
                else:
                    array_i = torch.cat((array_i,array_j),axis=0)
            array_i = array_i.reshape(1,NZ-1,NR,4)
            if i==0:
                mirror = array_i
            else:
                mirror = torch.cat((mirror,array_i),axis=0)
    
        return torch.cat((mirror,origin), axis=1)
    
    def define_spacetime_slice_2d(self,profile,T_lattice,Z_lattice):
        NT = len(T_lattice)
        NZ = len(Z_lattice)
        for i in torch.arange(NT):
            t = T_lattice[i]
            for j in torch.arange(NZ):
                z = Z_lattice[j]
                array_j = torch.tensor([t,z,profile[i][j]],device=self.device,dtype=self.dtype)
                array_j = array_j.reshape(1,3)
                if j==0:
                    array_i = array_j
                else:
                    array_i = torch.cat((array_i,array_j),axis=0)
            array_i = array_i.reshape(1,NZ,3)
            if i==0:
                origin = array_i
            else:
                origin = torch.cat((origin,array_i),axis=0)
            
        for i in torch.arange(NT):
            t = T_lattice[i]
            for j in torch.arange(1,NZ):
                z = -Z_lattice[-j]
                array_j = torch.tensor([t,z,profile[i][-j]],device=self.device,dtype=self.dtype)
                array_j = array_j.reshape(1,3)
                if j==1:
                    array_i = array_j
                else:
                    array_i = torch.cat((array_i,array_j),axis=0)
            array_i = array_i.reshape(1,NZ-1,3)
            if i==0:
                mirror = array_i
            else:
                mirror = torch.cat((mirror,array_i),axis=0)
    
        return torch.cat((mirror,origin), axis=1)

    

    def profile_configuration(self):
        PHI_plot = self.reduce_plot_ft_2d(self.initial_phi_ft)
        
        PHI_z_inte = self.reduce_inte_ft(1j*self.kz_tensor*self.initial_phi_ft)
        
        PHI_r_inte = self.reduce_inte_ft(1j*self.kr_tensor*self.initial_phi_ft)
        
        phi = self.initial_phi_ft
        phi_t = self.initial_phi_t_ft
        
        E_k, E_g, E_p, E_t = self.E_total_ft(phi_t, 1j*self.kz_tensor*self.initial_phi_ft, 1j*self.kr_tensor*self.initial_phi_ft, phi)
        
        E_Kinetic = torch.tensor([E_k],device=self.device,dtype=self.dtype)
        E_Gradient = torch.tensor([E_g],device=self.device,dtype=self.dtype)
        E_Potential = torch.tensor([E_p],device=self.device,dtype=self.dtype)
        E_Total = torch.tensor([E_t],device=self.device,dtype=self.dtype)
        
        print(f"GPU: {phi_t.is_cuda}")
        print(phi_t.dtype)
        print(self.R_ini)
        print(self.Z_ini)
        print(self.Judge_zero)
        print((E_k,E_g,E_p,E_t))
        
        for i in torch.arange(1,self.NumT):
            t = (i-1)*self.dt
            phi, phi_t = self.FORWARD_ft(t,phi,phi_t)
            
            if torch.fmod(i,self.nT_space_plot)==0:
                phi_z = 1j*self.kz_tensor*phi
                phi_r = 1j*self.kr_tensor*phi
                
                e_kinetic, e_gradient, e_potential, e_total = self.E_total_ft(phi_t, phi_z, phi_r, phi)
                
                PHI_plot = self.add_plot_ft_2d(PHI_plot,phi)
                
                E_Kinetic = torch.cat((E_Kinetic, torch.tensor([e_kinetic],device=self.device,dtype=self.dtype)))
                E_Gradient = torch.cat((E_Gradient, torch.tensor([e_gradient],device=self.device,dtype=self.dtype)))
                E_Potential = torch.cat((E_Potential, torch.tensor([e_potential],device=self.device,dtype=self.dtype)))
                E_Total = torch.cat((E_Total, torch.tensor([e_total],device=self.device,dtype=self.dtype)))
            
            if torch.fmod(i,self.nT_space_inte)==0:
                phi_z = 1j*self.kz_tensor*phi
                phi_r = 1j*self.kr_tensor*phi
                
                PHI_z_inte = self.add_inte_ft(PHI_z_inte,phi_z)
                PHI_r_inte = self.add_inte_ft(PHI_r_inte,phi_r)
            
            if torch.fmod(i,self.N_print_confi)==0:
                print("%.3f%s" % (100*i/self.NumT, '% has been done'))
                if torch.any(torch.isnan(phi)) == True:
                    print("nan appear!")
            
                
        self.slice_phi = self.define_spacetime_slice_2d(PHI_plot,self.t_lattice_plot,self.z_lattice_plot)
        
        self.slice_kinetic = self.define_time_slice(E_Kinetic,self.t_lattice_plot)
        self.slice_gradient = self.define_time_slice(E_Gradient,self.t_lattice_plot)
        self.slice_potential = self.define_time_slice(E_Potential,self.t_lattice_plot)
        self.slice_total = self.define_time_slice(E_Total,self.t_lattice_plot)
        
        self.profile_phi_z_inte = PHI_z_inte
        self.profile_phi_r_inte = PHI_r_inte
    
    @cached_property
    def initial_phi_ft(self):
        return torch.fft.fftn(self.initial_phi)
        
    @cached_property
    def initial_phi_t_ft(self):
        return torch.fft.fftn(self.initial_phi_t)
    
    def phi_Rterm_ft(self,phi0_ft):
        phir_ft = 1j*self.kr_tensor*phi0_ft
        phir = torch.fft.ifftn(phir_ft).real
        value = phir/self.r_tensor
        if self.Judge_zero:
            phirr = torch.fft.ifftn(-self.kr_tensor**2*phi0_ft).real
            value[:,self.index_R0] = phirr[:,self.index_R0]
        return torch.fft.fftn(value)
        
    def fun_ft(self,phi0_ft):
        phi0 = torch.fft.ifftn(phi0_ft).real
        phir_term_ft = self.phi_Rterm_ft(phi0_ft)
        phiV = self.Potential.Vphi(phi0)
        phiV_ft = torch.fft.fftn(phiV)
        phiK_ft = self.laplace_tensor*phi0_ft
        return phiK_ft+phir_term_ft-phiV_ft
    
    def reduce_plot_ft(self,phi_ft):
        phi = torch.fft.ifftn(phi_ft).real
        value = self.change_profile(phi)
        value = value[0::self.nZ_space_plot,0::self.nR_space_plot]
        value = value.reshape(1,self.NumZ_plot,self.NumR_plot)
        return value
    
    def reduce_plot_ft_2d(self,phi_ft):
        phi = torch.fft.ifftn(phi_ft).real
        value = self.change_profile(phi)
        value = value[0::self.nZ_space_plot,0]
        value = value.reshape(1,self.NumZ_plot)
        return value
    
    def reduce_inte_ft(self,phi_ft):
        phi = torch.fft.ifftn(phi_ft).real
        value = self.change_profile(phi)
        value = value[0::self.nZ_space_inte,0::self.nR_space_inte]
        value = value.reshape(1,self.NumZ_inte,self.NumR_inte)
        return value
    
    def add_plot_ft(self,PHI,phi_new):
        value = self.reduce_plot_ft(phi_new)
        return torch.cat((PHI,value),axis=0)
    
    def add_plot_ft_2d(self,PHI,phi_new):
        value = self.reduce_plot_ft_2d(phi_new)
        return torch.cat((PHI,value),axis=0)
    
    def add_inte_ft(self,PHI,phi_new):
        value = self.reduce_inte_ft(phi_new)
        return torch.cat((PHI,value),axis=0)
    
    def E_total_ft(self,phit0,phiz0,phir0,phi0):
        phi = self.change_profile(torch.fft.ifftn(phi0).real)
        phit = self.change_profile(torch.fft.ifftn(phit0).real)
        phiz = self.change_profile(torch.fft.ifftn(phiz0).real)
        phir = self.change_profile(torch.fft.ifftn(phir0).real)
        
        r_fun = self.change_profile(self.r_tensor)
        
        configuration_kinetic = r_fun * phit**2/2
        value_kinetic = 4*torch.pi*self.z_integration(self.r_integration(configuration_kinetic))
        
        configuration_gradient = r_fun * (phir**2 + phiz**2)/2
        value_gradient = 4*torch.pi*self.z_integration(self.r_integration(configuration_gradient))
        
        configuration_potential = r_fun * self.Potential.V(phi)
        value_potential = 4*torch.pi*self.z_integration(self.r_integration(configuration_potential))
        
        value = value_kinetic + value_gradient + value_potential
    
        return value_kinetic, value_gradient, value_potential, value

    
    def FORWARD_ft(self,t,phi0,phit0):    # 6th Runge-Kutta method + more precise derivative
        K1 = self.dt * phit0
        G1 = self.dt * self.fun_ft(phi0)
        K2 = self.dt * (phit0 + G1/4)
        G2 = self.dt * self.fun_ft(phi0 + K1/4)
        K3 = self.dt * (phit0 + (G1 + G2)/8)
        G3 = self.dt * self.fun_ft(phi0 + (K1 + K2)/8)
        K4 = self.dt * (phit0 + (-5*G2 + 8*G3)/6)
        G4 = self.dt * self.fun_ft(phi0 + (-5*K2 + 8*K3)/6)
        K5 = self.dt * (phit0 + (G1 + G2 + 4*G4)/8)
        G5 = self.dt * self.fun_ft(phi0 + (K1 + K2 + 4*K4)/8)
        K6 = self.dt * (phit0 + (3*G2 + 2*G3 - G4 + 2*G5)/8)
        G6 = self.dt * self.fun_ft(phi0 + (3*K2 + 2*K3 - K4 + 2*K5)/8)
        K7 = self.dt * (phit0 + (G1 - 2*G2 + 4*G3 + 4*G6)/7)
        G7 = self.dt * self.fun_ft(phi0 + (K1 - 2*K2 + 4*K3 + 4*K6)/7)
        phi_new = phi0 + (7*K1 + 32*K3 + 12*K4 + 16*K5 + 16*K6 + 7*K7)/90
        phit_new = phit0 + (7*G1 + 32*G3 + 12*G4 + 16*G5 + 16*G6 + 7*G7)/90
        return phi_new, phit_new