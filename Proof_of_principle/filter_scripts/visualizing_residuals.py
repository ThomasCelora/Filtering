import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time
import math

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # # ##########################################################################################
    # # SCRIPT TO VISUALIZE THE RESIDUALS: PLOT THEM AGAINST THE CLOSURE INGREDIENTS AND THE
    # # WEIGHING FUNCTIONS THAT CAN BE CONSIDERED FOR CALIBRATING THE MODEL
    # ############################################################################################

    
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

    print('================================================')
    print(f'Starting job on data from {MesoModelLoadFile}')
    print('================================================\n\n')

    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)

    statistical_tool = CoefficientsAnalysis()  
    correlation_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = correlation_ranges['x_range']
    y_range = correlation_ranges['y_range']
    meso_grid_info = json.loads(config['Meso_model_settings']['meso_grid'])
    num_slices_meso = meso_grid_info['num_T_slices']
    time_of_central_slice = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)]
    ranges = [[time_of_central_slice, time_of_central_slice], x_range, y_range]
    model_points = meso_model.domain_vars['Points']
    saving_directory = config['Directories']['figures_dir']

    # print('Producing correlation plot for pi_res')
    # x = meso_model.meso_vars['shear_sq']
    # y = meso_model.meso_vars['pi_res_sq']
    # data = [x,y]    
    # data = statistical_tool.trim_dataset(data, ranges, model_points)
    # preprocess_data = {"value_ranges": [[None, None], [None, None]], "log_abs": [1, 1]}
    # data = statistical_tool.preprocess_data(data, preprocess_data)
    # data = statistical_tool.extract_randomly(data, 30000)
    # x, y = data[0], data[1]
    # xlabel = r'$\log(\tilde{\sigma}_{ab}\tilde{\sigma}^{ab})$'
    # ylabel = r'$\log(\tilde{\pi}_{ab}\tilde{\pi}^{ab})$'
    # g2=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    # saving_directory = config['Directories']['figures_dir']
    # filename = '/pi_aniso_correlation.pdf'
    # plt.savefig(saving_directory + filename, format='pdf')
    # print(f'Finished correlation plot for pi_res, saved as {filename}\n\n')


    # print('Producing correlation plot for q_res')
    # x = meso_model.meso_vars['q_res_sq']
    # y = meso_model.meso_vars['Theta_sq']
    # data = [x,y]    
    # data = statistical_tool.trim_dataset(data, ranges, model_points)
    # preprocess_data = {"value_ranges": [[None, None], [None, None]], "log_abs": [1, 1]}
    # data = statistical_tool.preprocess_data(data, preprocess_data)
    # data = statistical_tool.extract_randomly(data, 30000)
    # x, y = data[0], data[1]
    # xlabel = r'$\log(\tilde{q}_a\tilde{q}^a)$'
    # ylabel = r'$\log(\tilde{\Theta}_a \tilde{\Theta}^a)$'
    # model_points = meso_model.domain_vars['Points']
    # g2=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    # saving_directory = config['Directories']['figures_dir']
    # filename = '/q_res_correlation.pdf'
    # plt.savefig(saving_directory + filename, format='pdf')
    # print(f'Finished correlation plot for q_res, saved as {filename}\n\n')


    # print('Producing correlation plot for Pi_res')
    # x = np.power(meso_model.meso_vars['Pi_res'], 2)
    # y = np.power(meso_model.meso_vars['exp_tilde'], 2)
    # data = [x,y]    
    # data = statistical_tool.trim_dataset(data, ranges, model_points)
    # preprocess_data = {"value_ranges": [[None, None], [None, None]], "log_abs": [1, 1]}
    # data = statistical_tool.preprocess_data(data, preprocess_data)
    # data = statistical_tool.extract_randomly(data, 30000)
    # x, y = data[0], data[1]
    # xlabel = r'$\log(\tilde{\Pi}^2)$'
    # ylabel = r'$\log(\tilde{\theta}^2)$'
    # model_points = meso_model.domain_vars['Points']
    # g2=statistical_tool.visualize_correlation(x, y, xlabel=xlabel, ylabel=ylabel)
    # saving_directory = config['Directories']['figures_dir']
    # filename = '/Pi_res_correlation.pdf'
    # plt.savefig(saving_directory + filename, format='pdf')
    # print(f'Finished correlation plot for Pi_res, saved as {filename}\n\n')

    
    # #############################################################
    # PLOTTING RESIDUALS VS CORRESPONDING CLOSURE INGREDIENTS 
    # #############################################################

    meso_grid_info = json.loads(config['Meso_model_settings']['meso_grid'])
    num_slices_meso = meso_grid_info['num_T_slices']
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    visualizer = Plotter_2D()  

    vars_strs = ['zeta', 'exp_tilde', 'Pi_res']
    norms = ['mysymlog', 'symlog', 'log']
    cmaps = ['Spectral', 'Spectral', 'plasma']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Bulk_viscosity.png"
    plt.savefig(saving_directory + filename, format = 'png', dpi=300)
    print('Finished bulk viscosity')

    vars_strs = ['eta', 'shear_sq', 'pi_res_sq']
    norms = ['mysymlog', 'log', 'log']
    cmaps = ['Spectral', 'plasma', 'plasma']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Shear_viscosity.png"
    plt.savefig(saving_directory + filename, format = 'png', dpi=300)
    print('Finished shear viscosity')

    vars_strs = ['kappa', 'Theta_sq', 'q_res_sq']
    norms = ['mysymlog', 'log', 'log']
    cmaps = ['Spectral', 'plasma', 'plasma']
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/Heat_conductivity.png"
    plt.savefig(saving_directory + filename, format = 'png', dpi=300)
    print('Finished heat conductivity')


    # det_s = meso_model.meso_vars['det_shear']
    # vars_strs = ['eta', 'det_shear']
    # norms = ['mysymlog', 'mysymlog']
    # cmaps = ['seismic', 'seismic']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # time_for_filename = str(round(time_meso,2))
    # filename = "/eta_vs_newdet.png"
    # plt.savefig(saving_directory + filename, format = 'png', dpi = 400)
    # print(f'Finished figure {filename}')


    # ############################################################
    # # PLOTTING Q1 AND Q2 AGAINST THE RESIDUALS SQUARED
    # ############################################################

    # meso_grid_info = json.loads(config['Meso_model_settings']['meso_grid'])
    # num_slices_meso = meso_grid_info['num_T_slices']
    # time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    # visualizer = Plotter_2D()  

    # vars_strs = ['Pi_res_sq', 'Q1', 'Q2']
    # norms = ['log', 'mysymlog', 'log']
    # cmaps = ['plasma', 'coolwarm', 'plasma']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename = "/Pi_res_Q.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished Pi_res')

    # vars_strs = ['pi_res_sq', 'Q1', 'Q2']
    # norms = ['log', 'mysymlog', 'log']
    # cmaps = ['plasma', 'coolwarm', 'plasma']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename = "/an_pi_res_Q.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished pi_res')

    # vars_strs = ['q_res_sq', 'Q1', 'Q2']
    # norms = ['log', 'mysymlog', 'log']
    # cmaps = ['plasma', 'coolwarm', 'plasma']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename = "/q_res_Q.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished q_res')

    # ############################################################
    # # PLOTTING WEIGHING FUNCTIONS
    # ############################################################

    # meso_grid_info = json.loads(config['Meso_model_settings']['meso_grid'])
    # num_slices_meso = meso_grid_info['num_T_slices']
    # time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    # visualizer = Plotter_2D()  

    # d = {'Q1' : r'$\tilde{\sigma}_{ab}\tilde{\sigma}^{ab} - \tilde{\omega}_{ab}\tilde{\omega}^{ab}$'}
    # meso_model.update_labels_dict(d)
    # d = {'Q2' : r'$\tilde{\sigma}_{ab}\tilde{\sigma}^{ab}/\tilde{\omega}_{ab}\tilde{\omega}^{ab}$'}
    # meso_model.update_labels_dict(d)
    # d = {'weights' : r'$w$'}
    # meso_model.update_labels_dict(d)

    # print('Building skew weights based on Q1')
    # meso_model.weights_Q1_skew()
    # print('Finished re-building the weights\n')

    # vars_strs = ['weights', 'Q1']
    # norms = [None, 'mysymlog']
    # cmaps = ['plasma', 'coolwarm']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename = "/Q1_skew.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished Q1weights_plot\n')

    # print('Building non-negative weights based on Q1')
    # meso_model.weights_Q1_non_neg()
    # print('Finished re-building the weights\n')

    # vars_strs = ['weights', 'Q1']
    # norms = [None, 'mysymlog']
    # cmaps = ['plasma', 'coolwarm']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename = "/Q1_non_neg.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished Q1weights_plot')


    # print('Building weights based on Q2')
    # meso_model.weights_Q2()
    # print('Finished re-building the weights\n')

    # vars_strs = ['weights', 'Q2']
    # norms = [None, 'log']
    # cmaps = ['plasma', 'plasma']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename = "/Q2_weights.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished Q1weights_plot\n')