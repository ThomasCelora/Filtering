import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time
import math
from scipy import stats
from scipy.interpolate import SmoothBivariateSpline

from matplotlib.ticker import FuncFormatter

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##################################################################
    # EXTRACTING THE 1ST ADIABATIC COEFFICIENT LOCALLY
    # ################################################################## 
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO MODEL
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    MesoModelLoadFile = pickle_directory + meso_pickled_filename

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    micro_model = meso_model.micro_model

    nx, ny = micro_model.domain_vars['nx'], micro_model.domain_vars['ny']
    central_slice_idx = int((micro_model.domain_vars['nt']-1)/2)
    time_micro = micro_model.domain_vars['t'][central_slice_idx]
    x_range = micro_model.domain_vars['xmin'], micro_model.domain_vars['xmax']
    y_range = micro_model.domain_vars['ymin'], micro_model.domain_vars['ymax']

    visualizer = Plotter_2D()

    p_micro, micro_extent = visualizer.get_var_data(micro_model, 'p', time_micro, x_range, y_range)
    n_micro, _ = visualizer.get_var_data(micro_model, 'n', time_micro, x_range, y_range)
    int_en_micro, _ = visualizer.get_var_data(micro_model, 'e', time_micro, x_range, y_range)

    eps_micro = n_micro + np.multiply(n_micro, int_en_micro)

    p_micro_flat = p_micro.flatten()
    n_micro_flat = n_micro.flatten()
    eps_micro_flat = eps_micro.flatten()

    micro_SBS_p = SmoothBivariateSpline(n_micro_flat, eps_micro_flat, p_micro_flat)
    micro_SBS_p_n = micro_SBS_p.partial_derivative(1, 0)
    micro_SBS_p_eps = micro_SBS_p.partial_derivative(0, 1)

    def compute_Gamma(p, n,eps, partial_n_p, partial_eps_p): 
        """
        Given some values for the number density, the energy density and the pressure
        Compute the corresponding adiabatic index (1st) using the cubic spline interpolations above
        """
        Gamma = partial_n_p + (p+eps)/n * partial_eps_p 
        Gamma *= n/p

        return Gamma
    
    gamma_micro = np.zeros_like(p_micro)
    rel_diff_gamma_micro = np.zeros_like(p_micro)
    for i in range(len(p_micro[:,0])):
        for j in range(len(p_micro[0,:])):
            p = p_micro[i, j] 
            n = n_micro[i, j]
            eps =  eps_micro[i, j]

            partial_n_p = micro_SBS_p_n(n, eps)
            partial_eps_p = micro_SBS_p_eps(n, eps)   
            gamma_micro[i,j] = compute_Gamma(p, n , eps, partial_n_p, partial_eps_p) -4./3.
            rel_diff_gamma_micro[i,j] = np.abs(gamma_micro[i,j]) / (4/3)


    Nt, Nx, Ny = meso_model.domain_vars['Nt'], meso_model.domain_vars['Nx'], meso_model.domain_vars['Ny'] 
    central_slice_idx = int((Nt-1)/2)

    time_meso = meso_model.domain_vars['T'][central_slice_idx]
    x_range = meso_model.domain_vars['Xmin'], meso_model.domain_vars['Xmax']
    y_range = meso_model.domain_vars['Ymin'], meso_model.domain_vars['Ymax']

    p_filt, meso_extent = visualizer.get_var_data(meso_model, 'p_filt', time_meso, x_range, y_range)
    n_tilde, _ = visualizer.get_var_data(meso_model, 'n_tilde', time_meso, x_range, y_range)
    eps_tilde, _ = visualizer.get_var_data(meso_model, 'eps_tilde', time_meso, x_range, y_range)

    p_filt_flat = p_filt.flatten()
    n_tilde_flat = n_tilde.flatten()
    eps_tilde_flat = eps_tilde.flatten()

    meso_SBS_p = SmoothBivariateSpline(n_tilde_flat, eps_tilde_flat, p_filt_flat)
    meso_SBS_p_n = meso_SBS_p.partial_derivative(1, 0)
    meso_SBS_p_eps = meso_SBS_p.partial_derivative(0, 1)
    
    gamma_meso = np.zeros_like(p_filt)
    rel_diff_gamma_meso = np.zeros_like(p_filt)
    for i in range(len(p_filt[:,0])):
        for j in range(len(p_filt[0,:])):
            p = p_filt[i, j] 
            n = n_tilde[i, j]
            eps =  eps_tilde[i, j]

            partial_n_p = meso_SBS_p_n(n, eps)
            partial_eps_p = meso_SBS_p_eps(n, eps)   

            gamma_meso[i,j] = compute_Gamma(p, n , eps, partial_n_p, partial_eps_p) -4./3.
            rel_diff_gamma_meso[i,j] = np.abs(gamma_meso[i,j]) / (4/3.)

    
    ############################################################################################### 
    # PLOTTING THE DIFFERENCE BETWEEN THE EXTRACTED ADIABATIC INDEX AND THE EXPECTED VALUE OF 4/3
    ############################################################################################### 
    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[9,4])
    axes = axes.flatten()

    im = axes[0].imshow(gamma_micro, extent=micro_extent, origin='lower', cmap='Spectral_r', norm='symlog')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[1].imshow(gamma_meso, extent=meso_extent, origin='lower', cmap='Spectral_r', norm='symlog')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    axes[0].set_title(r"$\Gamma_{micro} - 4/3$", fontsize =10)
    axes[1].set_title(r"$\Gamma_{meso} - 4/3$", fontsize =10)

    for ax in axes:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'Interp_Gamma'
    format = 'png'
    dpi = 300
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    plt.close()


    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[9,4])
    axes = axes.flatten()

    im = axes[0].imshow(rel_diff_gamma_micro, extent=micro_extent, origin='lower', cmap='plasma', norm='log')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[1].imshow(rel_diff_gamma_meso, extent=meso_extent, origin='lower', cmap='plasma', norm='log')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    axes[0].set_title(r"$\frac{|\Gamma_{micro} - 4/3|}{4/3}$", fontsize =12)
    axes[1].set_title(r"$\frac{|\Gamma_{meso} - 4/3|}{4/3}$", fontsize =12)

    for ax in axes:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'Interp_Gamma_reldiff'
    format = 'png'
    dpi = 300
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    plt.close()


    ##################################################################################
    # PLOTTING THE INTERPOLATED P AND ITS DERIVATIVES: CHECK ON IMPACT OF ARTIFACTS
    ##################################################################################

    p_filt_interp = np.zeros_like(p_filt)
    d_n_p =  np.zeros_like(p_filt)
    d_eps_p =  np.zeros_like(p_filt)
    for i in range(len(p_filt[:,0])):
        for j in range(len(p_filt[0,:])):
            n = n_tilde[i, j]
            eps =  eps_tilde[i, j]

            p_filt_interp[i,j]  = meso_SBS_p(n, eps)
            d_n_p[i,j] = meso_SBS_p_n(n, eps)
            d_eps_p[i,j] = meso_SBS_p_eps(n, eps)   

    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")


    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[13,8], sharex = True, sharey = True)
    axes = axes.flatten()

    im = axes[0].imshow(p_filt, extent=meso_extent, origin='lower', cmap='plasma')
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[1].imshow(n_tilde, extent=meso_extent, origin='lower', cmap='plasma')
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[2].imshow(eps_tilde, extent=meso_extent, origin='lower', cmap='plasma')
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[3].imshow(p_filt_interp, extent=meso_extent, origin='lower', cmap='plasma')
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[4].imshow(d_n_p, extent=meso_extent, origin='lower', cmap='plasma')
    divider = make_axes_locatable(axes[4])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[5].imshow(d_eps_p, extent=meso_extent, origin='lower', cmap='plasma')
    divider = make_axes_locatable(axes[5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


    for ax in axes:
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')


    axes[0].set_title(r"$<p>$", fontsize=12)
    axes[1].set_title(r"$\tilde n$", fontsize=12)
    axes[2].set_title(r"$\tilde \varepsilon$", fontsize=12)
    axes[3].set_title(r"$<p>_{interpolated}$", fontsize=12)
    axes[4].set_title(r"$\partial_{\tilde{n}} <p>$", fontsize=14)
    axes[5].set_title(r"$\partial_{\tilde{\varepsilon}} <p>$", fontsize=14)


    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = 'Interpolation_test'
    format = 'png'
    dpi = 300
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)
    plt.close()