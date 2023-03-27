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

# with open('Coeffs_998_31919.pickle', 'rb') as filehandle:
# with open('Coeffs_1998_34121.pickle', 'rb') as filehandle:
with open('Coeffs_2998_32626_x0203_y0405.pickle', 'rb') as filehandle:
    Coeffs = pickle.load(filehandle)[0]
# print(Coeffs.shape)

Nx = 26 # Number of observers in x & y directions
Ny = 26
# obs_filename = 'obs998_31919.txt'
# coords_filename = 'coords998_31919.txt'
obs_filename = 'obs2998_32626_x0203_y0405.txt'
coords_filename = 'coords2998_32626_x0203_y0405.txt'
coords = np.loadtxt(coords_filename)#[361:-360]
coords = coords.reshape(3,Nx,Ny,3)
obs = np.loadtxt(obs_filename)#[361:-360]
obs = obs.reshape(3,Nx,Ny,3)
Ws, vxs, vys = obs[1,:,:,0], obs[1,:,:,1], obs[1,:,:,2] # 1 to pick out central slice of 3
# print(Ws.shape)

def rearrange(raw_coeffs):
    # return raw_coeffs
    return np.abs(np.transpose(raw_coeffs))

def clip(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] == np.inf:
                array[i,j] = 1e5
            array[i,j] = np.log10(abs(array[i,j]))
            if array[i,j] == -np.inf:
                array[i,j] = 0
    return array

def clip2(array, maxx, minn):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > maxx:
                array[i,j] = maxx
            if array[i,j] < minn:
                array[i,j] = minn
            # array[i,j] = np.log10(abs(array[i,j]))
    return array

def clip3(array, maxx, minn):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > maxx:
                array[i,j] = 0
            if array[i,j] < minn:
                array[i,j] = 0
            # array[i,j] = np.log10(abs(array[i,j]))
    return array

def cull(arr1, arr2, maxx, minn):
    i = 0
    while i < len(arr1):
        if arr1[i] > maxx:
            arr1 = np.delete(arr1,i)
            arr2 = np.delete(arr2,i)
            continue
        if arr1[i] < minn:
            arr1 = np.delete(arr1,i)
            arr2 = np.delete(arr2,i)
            continue
        else:
            i+=1
    return arr1, arr2

# zetas, kappas, etas = rearrange(Coeffs[:,:])#,0],Coeffs[:,:,1],Coeffs[:,:,2]

# fig, axes = plt.subplots(1,3,figsize=(12,12))
# # print(kappas)
# axes[0].imshow(np.transpose(zetas),vmin=1e-1,vmax=1e2,label='Zeta')
# axes[0].set_title('Zeta')

# axes[1].imshow(np.transpose(kappas),vmin=-1e-1,vmax=1e1,label='Kappa')
# axes[1].set_title('Kappa')

# axes[2].imshow(np.transpose(etas),vmin=-1e1,vmax=1e1,label='Eta')
# axes[2].set_title('Eta')
# # plt.legend()
# plt.show()


# ##########################

# clip(zetas),clip(kappas),clip(etas)

# zeta_plot = sns.displot(zetas)#, x="log(zeta)")
# zeta_plot.set(xlabel ="log(zeta)", title ='Zeta distribution')

# kappa_plot = sns.displot(kappas)#, x="log(kappa)")
# kappa_plot.set(xlabel ="log(kappa)", title ='Kappa distribution')

# eta_plot = sns.displot(etas)#, x="log(eta)")
# eta_plot.set(xlabel ="log(eta)", title ='Eta distribution')

# Ws_trimmed = Ws[1:-1,1:-1]
# sns.regplot(x=Ws_trimmed.flatten(),y=etas.flatten())

# ##########################

# Pickle_Files = ['Zetas_998_31919.pickle','Kappas_998_31919.pickle','Etas_998_31919.pickle']
Pickle_Files = ['Zetas_1998_34121.pickle','Kappas_1998_34121.pickle','Etas_1998_34121.pickle']
# Pickle_Files = ['Zetas_2998_31919.pickle','Kappas_2998_31919.pickle','Etas_2998_31919.pickle']
n_x = 39
n_y = 19

# with open(Pickle_Files[0], 'rb') as filehandle:
#     Zetas = np.array(pickle.load(filehandle))
# # print(Zetas.shape)
# # print(Zetas)
# Zetas = Zetas.reshape(n_x,n_y)

# with open(Pickle_Files[1], 'rb') as filehandle:
#     Kappas = np.array(pickle.load(filehandle))
# Kappas = Kappas.reshape(n_x,n_y,3)
# Kappas0, Kappas1, Kappas2 = Kappas[:,:,0], Kappas[:,:,1], Kappas[:,:,2]
# Kappa_Components = [Kappas0, Kappas1, Kappas2]
# Kappa_strs = ['Kappa0', 'Kappa1', 'Kappa2']
# print(Kappas)

with open(Pickle_Files[2], 'rb') as filehandle:
    Etas = np.array(pickle.load(filehandle))
# print(Etas.shape)
Etas = Etas.reshape(n_x,n_y,3,3)
# print(Etas)
Etas00, Etas01, Etas02 = Etas[:,:,0,0], Etas[:,:,0,1], Etas[:,:,0,2]
Etas11, Etas12, Etas22 = Etas[:,:,1,1], Etas[:,:,1,2], Etas[:,:,2,2]
Eta_Components = [Etas00, Etas01, Etas02, Etas11, Etas12, Etas22]
Eta_strs = ['Etas00', 'Etas01', 'Etas02', 'Etas11', 'Etas12', 'Etas22']

# print(kappas)
# plt.figure()
# im = plt.imshow(np.transpose(Zetas),vmin=-5e1,vmax=5e1,label='Zeta')
# plt.colorbar(im)
# plt.title('Zeta')
# plt.show()

# fig, axes = plt.subplots(3,1,figsize=(12,12))
# axes = axes.flatten()
# for Kappa_Component, Kappa_str, ax in zip(Kappa_Components, Kappa_strs, axes):
#     im = ax.imshow(np.transpose(Kappa_Component),vmin=-0.5e0,vmax=0.5e0)#,label=Kappa_str)
#     ax.set_title(Kappa_str)
#     fig.colorbar(im, ax=ax)
# #plt.legend()
# # plt.colorbar(im)
# plt.show()

fig, axes = plt.subplots(3,2,figsize=(12,12))
axes = axes.flatten()
for Eta_component, Eta_str, ax in zip(Eta_Components, Eta_strs, axes):
    im = ax.imshow(np.transpose(Eta_component),vmin=-1e1,vmax=1e1)#,label=Eta_str)
    ax.set_title(Eta_str)
    fig.colorbar(im, ax=ax)
#plt.legend()
plt.show()


# clip(Zetas)
# zeta_plot = sns.displot(Zetas)#, x="log(zeta)")
# zeta_plot.set(xlabel ="log(Zeta)", title ='Zeta distribution')

# clip(Kappas0)
# kappa_plot = sns.displot(Kappas0)#, x="log(kappa)")
# kappa_plot.set(xlabel ="log(Kappa0)", title ='Kappa0 distribution')

# clip(Etas01)
# eta_plot = sns.displot(Etas01)#, x="log(eta)")
# eta_plot.set(xlabel ="log(eta)", title ='Eta distribution')


Ws_trimmed = Ws[1:-1,1:-1]
Ws_trimmed = Ws_trimmed[1:-1,1:-1]
Etas01 = Etas01[1:-1,1:-1]
Etas00_Regression, Ws_trimmed = cull(np.abs(np.transpose(Etas01)).flatten(), Ws_trimmed.flatten(), 5e1, -5e1)
#Etas00_Regression = np.log(np.transpose(Etas01))
#Etas00_Regression = clip2(np.transpose(Etas01),1e3,-1e3)
# fig, axes = plt.subplots(1,2,figsize=(12,6))
# axes[0].imshow(Etas00_Regression)
# axes[1].imshow(np.transpose(Ws_trimmed))
# plt.figure()
# plt.show()

sns.displot(Etas00_Regression.flatten())#, x="log(eta)")
plt.show()
sns.regplot(x=Ws_trimmed.flatten()-1,y=Etas00_Regression.flatten())
plt.show()
sns.jointplot(x=Ws_trimmed.flatten()-1,y=Etas00_Regression.flatten(), kind="hex", color="#4CB391")
plt.show()



















