#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 08:49:27 2025

@author: xuanqianyu
"""

from scipy import integrate
import torch

class Potential_torch(object):
    
    def __init__(self,omega,device,dtype):
        self.device = device
        self.dtype = dtype
        self.omega = omega
    
    def V(self,phi):
        return phi**2-2*(1+2*self.omega)*phi**3+(1+3*self.omega)*phi**4
    
    def Vphi(self,phi):
        return 2*phi-6*phi**2*(1+2*self.omega)+4*phi**3*(1+3*self.omega)
    
    def phim(self):
        return 1/(2+6*self.omega)
    
    def Vm(self):
        return self.V(self.phim())
    
    def A(self):
        return torch.sqrt(2*torch.exp(torch.tensor(1,device=self.device,dtype=self.dtype))*(self.Vm()-self.V(0)))
    
    def delta(self):
        return self.phim()/(torch.sqrt(torch.tensor(self.Vm(),device=self.device,dtype=self.dtype)))
    
    def R(self):
        return 2*(torch.sqrt((1+3*self.omega))*torch.sqrt((1+6*self.omega))\
                  *(-self.omega**1.5*torch.sqrt((1+6*self.omega))+1+6*self.omega+12*self.omega**2)\
                      -3*self.omega*(1+2*self.omega)*(1+4*self.omega)\
                          *torch.log((self.omega+torch.sqrt((self.omega))*torch.sqrt((1+3*self.omega)))\
                                  /(1+4*self.omega+torch.sqrt((1+3*self.omega))*torch.sqrt((1+6*self.omega)))))\
            /(3*torch.sqrt(torch.tensor(2,device=self.device,dtype=self.dtype))*(1+3*self.omega)**2.5*self.omega)
            
    def sigma(self):
        self.sigma, self.error = integrate.quad(lambda x:torch.sqrt(2*(self.V(x)-self.V(1))),0,1)
        return self.sigma
    
    def Rc(self):
        return 2*self.sigma()/(self.V(0)-self.V(1))