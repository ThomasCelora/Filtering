# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle

with open('Coeffs.pickle', 'rb') as filehandle:
    Coeffs = pickle.load(filehandle)[0]
# print(Coeffs[0])

zetas, kappas, etas = Coeffs[:,:,0],Coeffs[:,:,1],Coeffs[:,:,2]
# print(kappas)
plt.imshow(np.transpose(zetas),vmin=1e-3,vmax=1e1,label='Zeta')
plt.colorbar()
plt.title('Zeta')
plt.legend()
plt.show()
# plt.imshow(np.transpose(kappas),vmin=1e-3,vmax=1e0)
# plt.colorbar()
# plt.show()
# plt.imshow(np.transpose(etas),vmin=1e-3,vmax=1e0)
# plt.colorbar()
# plt.show()