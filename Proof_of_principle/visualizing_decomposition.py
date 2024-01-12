import sys
sys.path.append('../master_files/')
import pickle
import time 
import configparser
import json

from FileReaders import *
from MicroModels import *
from Filters import *
from MesoModels import *
from Visualization import *

if __name__ == '__main__':
    
    

    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()

    config = configparser.ConfigParser()
    config.read(sys.argv[1])

    # LOADING THE DECOMPOSED MESO MODEL
    directory = config['Directories']['pickled_files_dir']
    meso_decomposed_filename = config['Filenames']['meso_decomposed_pickle_name']
    MesoModelLoadFile = directory + meso_decomposed_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)


    ##########################################################
    # CODE FOR TESTING THE ALGEBRAIC DECOMPOSITION OF SET
    ##########################################################

    # WORKING OUT THE ALGEBRAIC CONSTRAINTS
    heat_flux = meso_model.meso_vars['q_res']
    aniso_stresses = meso_model.meso_vars['pi_res']
    favre_vel = meso_model.meso_vars['u_tilde']

    Nt = meso_model.domain_vars['Nt']
    Nx = meso_model.domain_vars['Nx']
    Ny = meso_model.domain_vars['Ny']

    trace_free_constr = np.zeros((Nt, Nx, Ny))
    stress_ort_1st_constr = np.zeros((Nt, Nx, Ny))
    stress_ort_2ns_constr = np.zeros((Nt, Nx, Ny))
    heat_flux_ort_constr = np.zeros((Nt, Nx, Ny))

    metric = meso_model.metric

    for h in range(Nt): 
        for i in range(Nx): 
            for j in range(Ny):
                favre_vel_covector = np.einsum('ij,j', metric, favre_vel[h,i,j])

                trace_free_constr[h,i,j] = np.einsum('ij,ji->', aniso_stresses[h,i,j], metric)
                stress_ort_1st_constr[h,i,j] = np.einsum('i,ij->', favre_vel_covector, aniso_stresses[h,i,j])
                stress_ort_2ns_constr[h,i,j] = np.einsum('ij,j->', aniso_stresses[h,i,j], favre_vel_covector)
                heat_flux_ort_constr[h,i,j] = np.einsum('i,i->', favre_vel_covector, heat_flux[h,i,j])

    # Now choosing one particular slice: it should not matter which
    slice_num = 0
    trace_free_check = trace_free_constr[slice_num,:,:]
    stress_ort_1st_check = stress_ort_1st_constr[slice_num,:,:]
    stress_ort_2nd_check = stress_ort_2ns_constr[slice_num,:,:]
    heat_flux_ort_check = stress_ort_1st_constr[slice_num,:,:]

    # NOW PLOTTING ALL THESE QUANTITIES
    fig, axes = plt.subplots(2,2)

    im = axes[0,0].imshow(trace_free_check)
    divider = make_axes_locatable(axes[0,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[0,1].imshow(stress_ort_1st_check)
    divider = make_axes_locatable(axes[0,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[1,0].imshow(stress_ort_2nd_check)
    divider = make_axes_locatable(axes[1,0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axes[1,1].imshow(heat_flux_ort_check)
    divider = make_axes_locatable(axes[1,1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Setting titles and labelling axes
    axes[0,0].set_title(r'$\pi^a_a$')
    axes[0,1].set_title(r'$u_a \pi^{ab}$')
    axes[1,0].set_title(r'$\pi^{ab}u_{b}$')
    axes[1,1].set_title(r'$q^a u_a$')
    fig.suptitle('Checking algebraic constraints in decomposing filtered SET')
    fig.tight_layout()
    
    for i in range(2):
        for j in range(2):
            axes[i,j].set_xlabel(r'$y$')
            axes[i,j].set_ylabel(r'$x$')

    
    saving_directory = config['Directories']['figures_dir']
    figurename = '/SET_algebraic_constraints.pdf'
    plt.savefig(saving_directory + figurename, format = 'pdf')
    

    ##########################################################
    # CODE FOR TESTING THE ALGEBRAIC DECOMPOSITION OF SET
    ##########################################################

    # shear = meso_model.domain_vars['shear_tilde']
    # acceleration  = meso_model.domain_vars['acc_tilde']