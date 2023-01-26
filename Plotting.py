# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:58:00 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py

n_files = 5
nx = 400
ny = 800
fs = []
for n in range(n_files):
    # fs.append(h5py.File('./Data/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
    # fs.append(h5py.File('./Data/KH/Ideal/t_998_1002/dp_400x800x0_'+str(n)+'.hdf5','r'))
    # fs.append(h5py.File('./Data/KH/Ideal/t_1998_2002/dp_400x800x0_'+str(n)+'.hdf5','r'))
    fs.append(h5py.File('./Data/KH/Ideal/t_2998_3002/dp_400x800x0_'+str(n)+'.hdf5','r'))
    
# coords = np.loadtxt('coords1998.txt').reshape(3,41,21,3)
# obs = np.loadtxt('obs1998.txt').reshape(3,41,21,3)
# coords = np.loadtxt('coords1998_tef2.txt').reshape(3,3,17,3)
# obs = np.loadtxt('obs1998_tef2.txt').reshape(3,3,17,3)
coords = np.loadtxt('coords2998_tef2.txt').reshape(3,19,19,3)
obs = np.loadtxt('obs2998_tef2.txt').reshape(3,19,19,3)
Us = obs[1,:,:]
UWs = obs[1,:,:,0]
Uxs = obs[1,:,:,1]
Uys = obs[1,:,:,2]


# print(Uxs.shape)
# Xs = np.linspace(-0.4,0.4,41)
# Ys = np.linspace(-0.2,0.2,21)
# Xs = np.linspace(-0.1,0.1,3)
# Ys = np.linspace(-0.8,0.8,17)
Xs = np.linspace(-0.45,0.45,19)
Ys = np.linspace(-0.9,0.9,19)
Extent = (Xs[0],Xs[-1],Ys[0],Ys[-1])
# ts = np.linspace(9.98,10.02,n_files) # Need to actually get these
# ts = np.linspace(19.98,20.02,n_files) # Need to actually get these
ts = np.linspace(9.98,10.02,n_files) # Need to actually get these
xs = np.linspace(-0.5,0.5,nx)
ys =  np.linspace(-1.0,1.0,ny)
extent = (xs[0],xs[-1],ys[0],ys[-1])
# points = (ts,xs,ys)
# points = np.meshgrid(xs,ys)
# self.dt get this from files...
dx = (xs[-1] - xs[0])/nx # actual grid-resolution
dy = (ys[-1] - ys[0])/ny


Ws = np.zeros((n_files, nx, ny))
vxs = np.zeros((n_files, nx, ny))
vys = np.zeros((n_files, nx, ny))
ns = np.zeros((n_files, nx, ny))

for f, counter in zip(fs, range(n_files)):
    Ws[counter] = f['Auxiliary/W'][:]
    vxs[counter] = f['Primitive/v1'][:]
    vys[counter] = f['Primitive/v2'][:]
    ns[counter] = f['Primitive/n'][:]


# fig, axes = plt.subplots(2,3)
# for n, ax in enumerate(axes.flatten()):
#     # ax.imshow(vxs[n][:])
#     ax.imshow(ns[n][:])

print(np.min(Uxs),np.max(Uxs))
print(np.min(vxs[2]),np.max(vxs[2]))

# fig, axes = plt.subplots(4,2,figsize=(2,25))
# fig, axes = plt.subplots(4,2,figsize=(2,12))
fig, axes = plt.subplots(4,2,figsize=(8,16))
axes[0,0].imshow(np.transpose(Ws[2]),extent=extent,vmin=np.min(UWs),vmax=np.max(UWs))
axes[0,1].imshow(np.transpose(UWs[:]),extent=Extent,vmin=np.min(UWs),vmax=np.max(UWs))
axes[1,0].imshow(np.transpose(vxs[2]),extent=extent,vmin=np.min(Uxs),vmax=np.max(Uxs))
axes[1,1].imshow(np.transpose(Uxs[:]),extent=Extent,vmin=np.min(Uxs),vmax=np.max(Uxs))
axes[2,0].imshow(np.transpose(vys[2]),extent=extent,vmin=np.min(Uys),vmax=np.max(Uys))
axes[2,1].imshow(np.transpose(Uys[:]),extent=Extent,vmin=np.min(Uys),vmax=np.max(Uys))
# axes[0].imshow(vxs[2],extent=extent)
# axes[1].imshow(Uxs[:],extent=Extent)
# axes[0].imshow(vxs[2],vmin=np.min(Uxs),vmax=np.max(Uxs),extent=extent)
# axes[1].imshow(Uxs[:],vmin=np.min(Uxs),vmax=np.max(Uxs),extent=Extent)
axes[0,0].set_ylabel('Ws')
axes[1,0].set_ylabel('Vxs')
axes[2,0].set_ylabel('Vys')
axes[3,0].set_ylabel('ns')
axes[3,0].imshow(np.transpose(ns[2][:]),extent=extent)
axes[3,1].imshow(np.transpose(UWs[:]),extent=Extent,vmin=np.min(UWs),vmax=np.max(UWs))

for i in range(4):
    axes[i,0].set_xlim(*Extent[0:2])
    axes[i,0].set_ylim(*Extent[2:])
fig.tight_layout()
plt.show()












