# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:58:00 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable

n_files = 5
nx = 400
ny = 800
fs = []
for n in range(n_files):
    # fs.append(h5py.File('./Data/KH/Ideal/dp_200x200x0_'+str(n)+'.hdf5','r'))
    # fs.append(h5py.File('./Data/KH/Ideal/t_998_1002/dp_400x800x0_'+str(n)+'.hdf5','r'))
    #fs.append(h5py.File('./Data/KH/Ideal/t_1998_2002/dp_400x800x0_'+str(n)+'.hdf5','r'))
    fs.append(h5py.File('./Data/KH/Ideal/t_2998_3002/dp_400x800x0_'+str(n)+'.hdf5','r'))

n_T = 3
# n_X = 26
# n_Y = 26
# X_lims = [0.2,0.3]
# Y_lims = [0.4,0.5]
n_X = 19
n_Y = 19
X_lims = [-0.4,0.4]
Y_lims = [-0.8,0.8]
t_lims = [29.98,30.02]
x_lims = [-0.5,0.5]
y_lims = [-1.0,1.0]
coords_filename = './Output/coords2998_32626_x0203_y0405.txt'
obs_filename = './Output/obs2998_32626_x0203_y0405.txt'
coords = np.loadtxt(coords_filename)
# coords = np.append(coords,[0.0,0.0,0.0]) # hack for 998_31919
coords = coords.reshape(n_T,n_X,n_Y,3)
obs = np.loadtxt(obs_filename)
# obs = np.append(obs,[0.0,0.0,0.0]) # hack for 998_31919
obs = obs.reshape(n_T,n_X,n_Y,3)
coords = np.loadtxt(coords_filename).reshape(n_T,n_X,n_Y,3)
obs = np.loadtxt(obs_filename).reshape(n_T,n_X,n_Y,3)
Us = obs[1,:,:]
UWs = obs[1,:,:,0]
Uxs = obs[1,:,:,1]
Uys = obs[1,:,:,2]


Xs = np.linspace(X_lims[0],X_lims[1],n_X)
Ys = np.linspace(Y_lims[0],Y_lims[1],n_Y)
Extent = (Xs[0],Xs[-1],Ys[0],Ys[-1])
ts = np.linspace(t_lims[0],t_lims[1],n_files) # Need to actually get these
xs = np.linspace(x_lims[0],x_lims[1],nx)
ys =  np.linspace(y_lims[0],y_lims[1],ny)
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
Ts = np.zeros((n_files, nx, ny))

for f, counter in zip(fs, range(n_files)):
    Ws[counter] = f['Auxiliary/W'][:]
    vxs[counter] = f['Primitive/v1'][:]
    vys[counter] = f['Primitive/v2'][:]
    ns[counter] = f['Primitive/n'][:]
    Ts[counter] = f['Auxiliary/T'][:]


# fig, axes = plt.subplots(2,3)
# for n, ax in enumerate(axes.flatten()):
#     # ax.imshow(vxs[n][:])
#     ax.imshow(ns[n][:])

# fig, axes = plt.subplots(4,2,figsize=(2,25))
# fig, axes = plt.subplots(4,2,figsize=(2,12))

import math

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
    
def find_nearest_cell(point):
    x_pos = find_nearest(xs,point[0])
    y_pos = find_nearest(ys,point[1])
    return [np.where(xs==x_pos)[0][0], np.where(ys==y_pos)[0][0]]

def find_fluctuations(vs,Vs):

    fluctuations = np.zeros((n_X,n_Y))
    
    for X, x_count in zip(Xs, range(n_X)):
        for Y, y_count in zip(Ys, range(n_Y)):
            x_cell, y_cell = find_nearest_cell([X,Y])[0], find_nearest_cell([X,Y])[1]
            #x, y = xs[x_cell], ys[y_cell]
            v = vs[x_cell][y_cell]
            U = Vs[x_count][y_count]
            fluctuations[x_count][y_count] = v - U

    return fluctuations

###################################################################


def plot2(z,Z,str1,str2,fig_name):
    fig, axes = plt.subplots(1,2)#,figsize=(8,16))
    ax0 = axes[0].imshow(np.transpose(z),extent=Extent)#,vmin=np.min(Z),vmax=np.max(Z))
    ax1 = axes[1].imshow(np.transpose(Z[:]),extent=Extent)#,vmin=np.min(Z),vmax=np.max(Z))
    axes[0].set_title(str1)
    axes[1].set_title(str2)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax0, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax1, cax=cax, orientation='vertical')
    # for i in range(1):
    #     axes[0].set_xlim(*Extent[0:2])
    #     axes[1].set_ylim(*Extent[2:])
    axes[0].axis('off')
    axes[1].axis('off')
    fig.tight_layout()
    plt.savefig(fig_name,bbox_inches='tight',dpi=1200)
    plt.show()

def plot3(z,Z,diff,str1,str2,str3,fig_name):
    fig, axes = plt.subplots(1,3)#,figsize=(8,20))
    ax0 = axes[0].imshow(np.transpose(z),extent=Extent,vmin=np.min(Z),vmax=np.max(Z))
    ax1 = axes[1].imshow(np.transpose(Z),extent=Extent,vmin=np.min(Z),vmax=np.max(Z))
    ax2 = axes[2].imshow(np.transpose(diff[:]),extent=Extent)
    axes[0].set_title(str1)
    axes[1].set_title(str2)
    axes[2].set_title(str3)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax0, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax1, cax=cax, orientation='vertical')
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(ax2, cax=cax, orientation='vertical')
    for i in range(1):
        axes[0].set_xlim(*Extent[0:2])
        axes[1].set_ylim(*Extent[2:])
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    fig.tight_layout()
    plt.savefig(fig_name,bbox_inches='tight',dpi=1200)
    plt.show()
    
###################################################################

fluctuations = find_fluctuations

plot3(Ws[2],UWs,find_fluctuations(Ws[2],UWs),r'$W(v$)',r'$W(U$)',r'$W(v) - W(U)$','W_comparison.png')

# fig, axes = plt.subplots(1,2,figsize=(8,16))
# ax0 = axes[0].imshow(np.transpose(Ws[2]),extent=Extent,vmin=np.min(UWs),vmax=np.max(UWs))
# ax1 = axes[1].imshow(np.transpose(UWs[:]),extent=Extent,vmin=np.min(UWs),vmax=np.max(UWs))
# axes[0].set_ylabel(r'$W(v$)')
# axes[1].set_ylabel(r'$W(U$)')
# divider = make_axes_locatable(axes[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax0, cax=cax, orientation='vertical')
# divider = make_axes_locatable(axes[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax1, cax=cax, orientation='vertical')
# for i in range(1):
#     axes[0].set_xlim(*Extent[0:2])
#     axes[1].set_ylim(*Extent[2:])
# fig.tight_layout()
# plt.show()

plot3(vxs[2],Uxs,find_fluctuations(vxs[2],Uxs),r'$v_x$',r'$U_x$',r'$v_x - U_x$','vx_comparison.png')


# fig, axes = plt.subplots(1,3,figsize=(8,20))
# ax0 = axes[0].imshow(np.transpose(vxs[2]),extent=Extent,vmin=np.min(Uxs),vmax=np.max(Uxs))
# ax1 = axes[1].imshow(np.transpose(Uxs[:]),extent=Extent,vmin=np.min(Uxs),vmax=np.max(Uxs))
# ax2 = axes[2].imshow(np.transpose(fluctuations[:]),extent=Extent)
# axes[0].set_title(r'$v_x$')
# axes[1].set_title(r'$U_x$')
# axes[2].set_title(r'$v_x - U_x$')
# divider = make_axes_locatable(axes[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax0, cax=cax, orientation='vertical')
# divider = make_axes_locatable(axes[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax1, cax=cax, orientation='vertical')
# divider = make_axes_locatable(axes[2])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax2, cax=cax, orientation='vertical')
# for i in range(1):
#     axes[0].set_xlim(*Extent[0:2])
#     axes[1].set_ylim(*Extent[2:])
# axes[0].axis('off')
# axes[1].axis('off')
# axes[2].axis('off')
# fig.tight_layout()
# plt.savefig('vx_comparison.png',bbox_inches='tight')
# plt.show()


plot3(vys[2],Uys,find_fluctuations(vys[2],Uys),r'$v_y$',r'$U_y$',r'$v_y - U_y$','vy_comparison.png')

# fig, axes = plt.subplots(1,2,figsize=(8,16))
# axes[0].imshow(np.transpose(vys[2]),extent=Extent,vmin=np.min(Uys),vmax=np.max(Uys))
# axes[1].imshow(np.transpose(Uys[:]),extent=Extent,vmin=np.min(Uys),vmax=np.max(Uys))
# axes[0].set_ylabel(r'$v_y$')
# axes[1].set_ylabel(r'$U_y$')
# divider = make_axes_locatable(axes[0])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax0, cax=cax, orientation='vertical')
# divider = make_axes_locatable(axes[1])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(ax1, cax=cax, orientation='vertical')
# for i in range(1):
#     axes[0].set_xlim(*Extent[0:2])
#     axes[1].set_ylim(*Extent[2:])
# fig.tight_layout()
# plt.show()

# =============================================================================

fig, axes = plt.subplots(1,2,figsize=(8,16))
axes[0].imshow(np.transpose(Ws[2]),extent=extent,vmin=np.min(UWs),vmax=np.max(UWs))
axes[1].imshow(np.transpose(UWs[:]),extent=Extent,vmin=np.min(UWs),vmax=np.max(UWs))
axes[0].set_title('Ws')
for i in range(1):
    axes[0].set_xlim(*Extent[0:2])
    axes[1].set_ylim(*Extent[2:])
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1,2,figsize=(8,16))
axes[0].imshow(np.transpose(vxs[2]),extent=extent,vmin=np.min(Uxs),vmax=np.max(Uxs))
axes[1].imshow(np.transpose(Uxs[:]),extent=Extent,vmin=np.min(Uxs),vmax=np.max(Uxs))
axes[0].set_title('Vxs')
for i in range(1):
    axes[0].set_xlim(*Extent[0:2])
    axes[1].set_ylim(*Extent[2:])
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(1,2,figsize=(8,16))
axes[0].imshow(np.transpose(vys[2]),extent=extent,vmin=np.min(Uys),vmax=np.max(Uys))
axes[1].imshow(np.transpose(Uys[:]),extent=Extent,vmin=np.min(Uys),vmax=np.max(Uys))
axes[0].set_title('Vys')
for i in range(1):
    axes[0].set_xlim(*Extent[0:2])
    axes[1].set_ylim(*Extent[2:])
fig.tight_layout()
plt.show()

# axes[0,0].set_title('Ws')
# axes[1,0].set_title('Vxs')
# axes[2,0].set_title('Vys')
# for i in range(1):
#     axes[i,0].set_xlim(*Extent[0:2])
#     axes[i,0].set_ylim(*Extent[2:])

plot2(ns[2],Ts[2],r'$n$',r'$T$','n_T_full.png')

# fig, axes = plt.subplots(1,2,figsize=(8,16))
# axes[0].set_title('n')
# axes[0].imshow(np.transpose(ns[2][:]),extent=extent)

# axes[1].set_title('T')
# axes[1].imshow(np.transpose(Ts[2][:]),extent=extent)
# fig.tight_layout()
# plt.show()



fig, axes = plt.subplots(1,2,figsize=(8,16))
axes[0].set_title('ns')
axes[0].imshow(np.transpose(ns[2][:]),extent=extent)

axes[1].set_title('Ts')
axes[1].imshow(np.transpose(Ts[2][:]),extent=extent)
fig.tight_layout()
plt.show()













