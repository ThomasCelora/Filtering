# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:02:05 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import pickle
import seaborn as sns

with open('Coeffs_2998_31919.pickle', 'rb') as filehandle:
    Coeffs = pickle.load(filehandle)[0]
print(Coeffs.shape)

def rearrange(raw_coeffs):
    # return raw_coeffs
    return np.abs(np.transpose(raw_coeffs))

def clip(coeffs):
    for i in range(coeffs.shape[0]):
        for j in range(coeffs.shape[1]):
            if coeffs[i,j] == np.inf:
                coeffs[i,j] = 0
            coeffs[i,j] = np.log(coeffs[i,j])
            if coeffs[i,j] == -np.inf:
                coeffs[i,j] = 0

    return coeffs

zetas, kappas, etas = rearrange(Coeffs[:,:])#,0],Coeffs[:,:,1],Coeffs[:,:,2]

fig, axes = plt.subplots(1,3,figsize=(12,12))
# print(kappas)
axes[0].imshow(np.transpose(zetas),vmin=-1e-3,vmax=1e3,label='Zeta')
axes[0].set_title('Zeta')

axes[1].imshow(np.transpose(kappas),vmin=-1e-2,vmax=1e2,label='Kappa')
axes[1].set_title('Kappa')

axes[2].imshow(np.transpose(etas),vmin=-1e-2,vmax=1e2,label='Eta')
axes[2].set_title('Eta')

# plt.legend()
plt.show()

clip(zetas),clip(kappas),clip(etas)
sns.displot(zetas)#, x="log(zeta)")
sns.displot(kappas)#, x="log(kappa)")
sns.displot(etas)#, x="log(eta)")