#!/bin/bash

import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
from matplotlib.ticker import LogLocator

from FileReaders import *
from MicroModels import *
from MesoModels import *
from Visualization import *

if __name__ == '__main__':

    ####################################################################################################
    # SCRIPT TO VISUALIZE THE FILTERING OBSERVERS: COMPARE WITH EITHER MICRO OR FAVRE VEL
    ####################################################################################################

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
    
    print(f'Finished reading data from {MesoModelLoadFile}')

    # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE 
    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    meso_grid_info = json.loads(config['Meso_model_settings']['meso_grid'])
    num_slices_meso = meso_grid_info['num_T_slices']
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    if time_meso != time_micro:
        print("Slices of meso and micro model do not coincide. Careful!")
    else: 
        print("Comparing data at same time-slice, hurray!")
    

    # PLOT SETTINGS
    saving_directory = config['Directories']['figures_dir']
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    diff_plot_settings = json.loads(config['Plot_settings']['diff_plot_settings'])
    diff_method = diff_plot_settings['method']
    interp_dims = diff_plot_settings['interp_dims']    
    visualizer = Plotter_2D([10, 3])
    
    # label_2_update = {'U' : r'$U^a$'}
    # meso_model.upgrade_labels_dict(label_2_update)

    # FINALLY, PLOTTING 
    # models = [micro_model, meso_model]
    # vars = [['bar_vel', 'bar_vel', 'bar_vel'],['u_tilde', 'u_tilde', 'u_tilde']]
    # components_indices= [[(0,),(1,),(2,)], [(0,), (1,), (2,)]]
    # norms = [['log','mysymlog','mysymlog'],['log','mysymlog','mysymlog'],['log','log','log']] 
    # cmaps = [['plasma','seismic','seismic'],['plasma','seismic','seismic'], ['plasma','plasma','plasma']] 
    # fig = visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices=components_indices, method=diff_method,
    #                                             interp_dims=interp_dims, diff_plot=False, rel_diff=True, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # filename="/favreVSmicro.pdf"
    # plt.savefig(saving_directory + filename, format="pdf")
    # print('Finished plotting the favre velocities')


    models = [micro_model, meso_model]
    # vars = [['bar_vel', 'bar_vel', 'bar_vel'],['U', 'U', 'U']]
    # norms = [['log','mysymlog','mysymlog'],['log','mysymlog','mysymlog'],['log','log','log']] 
    # cmaps = [['plasma','seismic','seismic'], ['plasma','seismic','seismic'], ['plasma','plasma','plasma']] 

    # vars = [['bar_vel'], ['U']]
    # norms = [['log'], ['log'], ['log']]
    # cmaps = [['plasma'],['plasma'], ['plasma']]
    # components_indices= [[(0,)], [(0,)]]
    # fig = visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices=components_indices, method=diff_method,
    #                                             interp_dims=interp_dims, diff_plot=False, rel_diff=True, norms=norms, cmaps=cmaps)
    
    # axes = np.array(fig.axes)
    # axes = axes.flatten()
    # axes[0].set_title(micro_model.labels_var_dict['bar_vel'] + r"$,$ $a=0$")
    # axes[1].set_title(meso_model.labels_var_dict['U'] + r"$,$ $a=0$")
    # axes[2].set_title(r"$Relative$ $difference$")

    comp = (1,)
    bar_vel_data, extent_micro = visualizer.get_var_data(micro_model, 'bar_vel', time_meso, x_range, y_range, comp)
    obs_data, extent_meso = visualizer.get_var_data(meso_model, 'U', time_meso, x_range, y_range, comp)
    ar_mean = (np.abs(bar_vel_data) + np.abs(obs_data))/2
    rel_diff = np.abs(bar_vel_data - obs_data)/ar_mean


    fig = plt.figure(figsize=[13,4])
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3, sharey=ax2)
    plt.setp(ax3.get_yticklabels(), visible=False)
    axes = [ax1, ax2, ax3]
    axesRight = [ax2, ax3]
    

    im = ax1.imshow(rel_diff, extent=extent_meso, origin='lower', cmap = 'plasma', norm='log')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.minorticks_on()

    images = []
    im = ax2.imshow(bar_vel_data, extent=extent_micro, origin='lower', cmap='plasma') #, norm='log')    
    images.append(im)
    im = ax3.imshow(obs_data, extent=extent_meso, origin='lower', cmap='plasma') #, norm='log')    
    images.append(im)


    for i in range(len(axes)):
        axes[i].set_xlabel(r'$x$')
        axes[i].set_ylabel(r'$y$')

    title = r'$Rel.$ $difference$'
    ax1.set_title(title)

    title = micro_model.labels_var_dict['bar_vel']
    title += r'$,$ $a=0$'
    ax2.set_title(title)

    title = meso_model.labels_var_dict['U']
    title += r'$,$ $a=0$'
    ax3.set_title(title)

    # fig.tight_layout()
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_axis_off()

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # fig.colorbar(images[0], ax=axesRight, orientation='vertical', location='right', shrink=.8, pad=0.025 )
    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely! 
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacks.connect('changed', update)

    
    fig.tight_layout()
    filename=f"/ObsVSmicro_{comp[0]}."
    format = 'png'
    dpi=400
    plt.savefig(saving_directory + filename + format, format=format, dpi=dpi)
    print('Finished plotting the observers')

