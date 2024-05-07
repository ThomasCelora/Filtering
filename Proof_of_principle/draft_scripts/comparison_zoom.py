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
    # # SCRIPT TO COMPARE MICRO AND FILTERED DATA
    # # one figure shows the relative difference over the full grid, 
    # # other two are produced to compare the data in a zoomed-in patch
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

    print('=========================================================================')
    print(f'Starting job on data from {pickle_directory+ meso_pickled_filename}')
    print('=========================================================================\n\n')

    MesoModelLoadFile = pickle_directory + meso_pickled_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_model = pickle.load(filehandle)    
    micro_model = meso_model.micro_model

    print('Finished reading pickled data\n')

    # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE
    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    num_slices_meso = int(config['Models_settings']['num_T_slices'])
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    if time_meso != time_micro:
        print("Slices of meso and micro model do not coincide. Careful!\n")
    else: 
        print("Comparing data at same time-slice, hurray!\n")
        
    # # PLOT SETTINGS
    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2D()

    inset_ranges = json.loads(config['Plot_settings']['inset_ranges'])
    inset_x_range = inset_ranges['x_range']
    inset_y_range = inset_ranges['y_range']

    rel_diff_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    rel_diff_x_range = rel_diff_ranges['x_range']
    rel_diff_y_range = rel_diff_ranges['y_range']


    # Building the data for showing the relative difference between the various plots
    var_str = 'BC'
    comp = (0,)

    micro_data_zoom, extent_micro_zoom = visualizer.get_var_data(micro_model, var_str, time_meso, inset_x_range, inset_y_range, comp)
    meso_data_zoom, extent_meso_zoom = visualizer.get_var_data(meso_model, var_str, time_meso, inset_x_range, inset_y_range, comp)

    micro_data, extent_micro = visualizer.get_var_data(micro_model, var_str, time_meso, rel_diff_x_range, rel_diff_y_range, comp)
    meso_data, extent_meso = visualizer.get_var_data(meso_model, var_str, time_meso, rel_diff_x_range, rel_diff_y_range, comp)
    ar_mean = (np.abs(micro_data) + np.abs(meso_data))/2
    rel_diff = np.abs(meso_data - micro_data)/ar_mean

    # now plotting    
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
    fig.colorbar(im, cax=cax, orientation='vertical')

    images = []
    im = ax2.imshow(micro_data_zoom, extent=extent_micro_zoom, origin='lower', cmap='plasma') #, norm='log')    
    images.append(im)
    im = ax3.imshow(meso_data_zoom, extent=extent_meso_zoom, origin='lower', cmap='plasma') #, norm='log')    
    images.append(im)


    for i in range(len(axes)):
        axes[i].set_xlabel(r'$x$')
        axes[i].set_ylabel(r'$y$')

    title = r'$Rel.$ $difference$'
    ax1.set_title(title, fontsize=10)

    title = micro_model.labels_var_dict[var_str]
    title += r'$,$ $a=0$'
    ax2.set_title(title, fontsize=10)

    title = meso_model.labels_var_dict[var_str]
    title += r'$,$ $a=0$'
    ax3.set_title(title, fontsize=10)

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


    # Adding boundaries of zoomed in regions 
    A = np.array([inset_x_range[0], inset_y_range[0]])
    B = np.array([inset_x_range[1], inset_y_range[0]])
    C = np.array([inset_x_range[1], inset_y_range[1]])
    D = np.array([inset_x_range[0], inset_y_range[1]])
    arrows_starts = [A, B, C, D]
    arrows_increments = [arrows_starts[i] - arrows_starts[i-1] for i in range(1, len(arrows_starts))]
    arrows_increments.append(A-D)
    for i in range(len(arrows_starts)):
        ax1.arrow(arrows_starts[i][0], arrows_starts[i][1], arrows_increments[i][0], arrows_increments[i][1], \
                      width=0.005,color='white',head_length=0.0,head_width=0.0)

    fig.tight_layout()

    format='png'
    dpi=400
    filename = f"/Compairing_zoom_{var_str}_{comp}." + format 
    plt.savefig(saving_directory + filename, format = format, dpi=dpi)
    
    
