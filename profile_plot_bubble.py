#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:03:44 2025

@author: xuanqianyu
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


data1 = torch.load('./data_fit_show/phi_configuration_0.pt', weights_only=True)
data2 = torch.load('./data_fit_show/phi_configuration_1.pt', weights_only=True)
data3 = torch.load('./data_fit_show/phi_configuration_2.pt', weights_only=True)
data4 = torch.load('./data_fit_show/phi_configuration_3.pt', weights_only=True)
data5 = torch.load('./data_fit_show/phi_configuration_4.pt', weights_only=True)
data6 = torch.load('./data_fit_show/phi_configuration_5.pt', weights_only=True)


x1 = data1[0,:,1]
y1 = data1[:,0,0]
density1 = data1[:,:,2]


x2 = data2[0,:,1]
y2 = data2[:,0,0]
density2 = data2[:,:,2]


x3 = data3[0,:,1]
y3 = data3[:,0,0]
density3 = data3[:,:,2]


x4 = data4[0,:,1]
y4 = data4[:,0,0]
density4 = data4[:,:,2]


x5 = data5[0,:,1]
y5 = data5[:,0,0]
density5 = data5[:,:,2]

x6 = data6[0,:,1]
y6 = data6[:,0,0]
density6 = data6[:,:,2]


X = [x1, x2, x3, x4, x5, x6]
Y = [y1, y2, y3, y4, y5, y6]
Density = [density1, density2, density3, density4, density5, density6]

fig = plt.figure(figsize=(9, 6))

Title = ['Semiclassical Phase Transition', 'Semiclassical Phase Transition', 'Semiclassical Phase Transition', 'Quantum Phase Transition', 'Quantum Phase Transition', 'Quantum Phase Transition']

i = 5

x = X[i]
y = Y[i]
density = Density[i].numpy()
title = Title[i]


implot = plt.imshow(density, cmap='jet', vmax=1.1, vmin=-0.1, origin='lower', extent=[np.floor(x[0]), np.ceil(x[-1]), np.floor(y[0]), np.ceil(y[-1])], aspect='auto')  # origin='lower' 是为了使坐标原点在左下角
plt.tick_params(width=2,labelsize=20)
plt.xlabel(r'$z$',fontweight='normal',fontsize=25)
plt.ylabel(r'$t$',fontweight='normal',fontsize=25)
plt.title(title,fontsize=25,color='k')


cbar = fig.colorbar(implot, orientation='vertical')

cbar.set_label(r'$\phi(t,z)$', fontsize=25)
cbar.ax.yaxis.set_label_position('right')



plt.show()