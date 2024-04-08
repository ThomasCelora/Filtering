import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ##############################################################
    # # SCRIPT TO COMPARE THE COMPUTED RESIDUALS WITH THEIR CONSTANT 
    # # COEFFICIENTS MODELLING
    # ##############################################################
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO AND MICRO MODELS 
    pickle_directory = config['Directories']['pickled_files_dir']
    meso_pickled_filename = config['Filenames']['meso_pickled_filename']
    MesoModelLoadFile = pickle_directory + meso_pickled_filename

    print('=========================================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('=========================================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)
    micro_model = meso_model.micro_model

    print('Finished reading pickled data\n')


    num_slices_meso = int(config['Models_settings']['num_T_slices'])
    central_slice_num = int((num_slices_meso-1)/2)


    # BUILDING THE VARIOUS MODELS WITH CONSTANT COEFF
    eta = meso_model.meso_vars['eta']
    pi_res = meso_model.meso_vars['pi_res']
    shear_tilde = meso_model.meso_vars['shear_tilde']
    eta_const = np.mean(np.abs(eta))
    pi_res_model = np.multiply(eta_const, shear_tilde)
    # print(f'data shape: {pi_res.shape} model shape: {pi_res_model.shape}')

    zeta = meso_model.meso_vars['zeta']
    Pi_res = meso_model.meso_vars['Pi_res']
    exp_tilde = meso_model.meso_vars['exp_tilde']
    zeta_const = np.mean(np.abs(zeta))
    Pi_res_model = np.multiply(zeta_const, np.abs(exp_tilde))
    # print(f'data shape: {Pi_res.shape} model shape: {Pi_res_model.shape}')

    # # Adding EOS residual to Pi_res
    EOS_res = meso_model.meso_vars['eos_res']
    Pi_res = Pi_res + EOS_res

    kappa = meso_model.meso_vars['kappa']
    q_res = meso_model.meso_vars['q_res']
    Theta_tilde = meso_model.meso_vars['Theta_tilde']
    kappa_const = np.mean(np.abs(kappa))
    q_res_model = np.multiply(kappa_const, Theta_tilde)
    # print(f'data shape: {q_res.shape} model shape: {q_res_model.shape}')

    # SQUARING THE TENSORS
    metric = np.zeros((3,3))
    metric[0,0] = -1
    metric[1,1] = metric[2,2] = +1

    Nx, Ny = meso_model.domain_vars['Nx'], meso_model.domain_vars['Ny']
    pi_res_sq = np.zeros((Nx, Ny))
    pi_res_sq_mod = np.zeros((Nx, Ny))
    q_res_sq = np.zeros((Nx, Ny))
    q_res_sq_mod = np.zeros((Nx, Ny))

    h = central_slice_num
    Pi_res = np.log10(Pi_res[h,:,:])
    Pi_res_model = np.log10(Pi_res_model[h,:,:])

    for i in range(Nx):
        for j in range(Ny):
            temp = np.einsum('ij,kl,kj,li->', pi_res[h,i,j], pi_res[h,i,j], metric, metric)
            temp = np.log10(np.sqrt(temp))
            pi_res_sq[i,j] = temp

            temp = np.einsum('ij,kl,kj,li->',pi_res_model[h,i,j], pi_res_model[h,i,j], metric, metric)
            temp = np.log10(np.sqrt(temp))
            pi_res_sq_mod[i,j] = temp

            temp = np.einsum('i,ij,j->', q_res[h,i,j], metric, q_res[h,i,j])
            temp = np.log10(np.sqrt(temp))
            q_res_sq[i,j] = temp

            temp = np.einsum('i,ij,j->', q_res_model[h,i,j], metric, q_res_model[h,i,j])
            temp = np.log10(np.sqrt(temp))
            q_res_sq_mod[i,j] = temp

    # CREATING SUBPLOTS WITH CORRESPONDING DISTRIBUTIONS
    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(1,3, figsize=[13,4])
    axes = axes.flatten()

    data = [[Pi_res, Pi_res_model], [pi_res_sq, pi_res_sq_mod], [q_res_sq, q_res_sq_mod],]

    x_axis_labels = []
    x_axis_labels.append(r'$\log(\tilde\Pi)$')
    x_axis_labels.append(r'$\log(\sqrt{\tilde\pi_{ab}\tilde\pi^{ab}})$')
    x_axis_labels.append(r'$\log(\sqrt{\tilde q_{a}\tilde q^{a}})$')
    


    for i in range(len(axes)):
        X = data[i][0]
        Y = data[i][1]
        if X.shape != Y.shape:
            print(f'Careful: shapes not compatible. i={i}')

        X = X.flatten()
        Y = Y.flatten()

        stat = 'probability'
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='is_categorical_dtype is deprecated')
            warnings.filterwarnings('ignore', message='use_inf_as_na option is deprecated')
            sns.histplot(X, stat=stat, kde=True, color='firebrick', ax=axes[i], label='sim. data')
            sns.histplot(Y, stat=stat, kde=True, color='steelblue', ax=axes[i], label='model')
        
        axes[i].legend(loc = 'best', prop={'size': 10})
        axes[i].set_xlabel(x_axis_labels[i], fontsize=10)
        axes[i].set_ylabel(stat, fontsize=10)


    fig.tight_layout()

    fig_directory = config['Directories']['figures_dir'] 
    filename = f'/Const_coeff_models_' + f'{stat}' + 'EOS'
    format = 'png'
    dpi = 300
    filename += "." + format
    plt.savefig(fig_directory + filename, format=format, dpi=dpi)

    
