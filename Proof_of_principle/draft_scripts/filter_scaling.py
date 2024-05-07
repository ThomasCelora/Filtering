import sys
# import os
sys.path.append('../../master_files/')
import pickle
import configparser
import json
import time
from matplotlib import colors 
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *
from Analysis import *

if __name__ == '__main__':

    # ###############################################################################################
    # # SCRIPT TO SHOW HOW THE IMPACT OF FILTERING SCALES WITH THE FILTER-SIZE: REL DIFFERENCES
    # ###############################################################################################
    
    # READING SIMULATION SETTINGS FROM CONFIG FILE
    if len(sys.argv) == 1:
        print(f"You must pass the configuration file for the simulations.")
        raise Exception()
    
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    
    # LOADING MESO AND MICRO MODELS 
    pickle_directory = config['Directories']['pickled_files_dir']
    

    print('=========================================================================')
    print(f'Starting job on data from {pickle_directory}')
    print('=========================================================================\n\n')

    meso_pickled_filename = '/rHD2d_nocg_fw=2_bl=8dx.pickle'
    MesoModelLoadFile = pickle_directory + meso_pickled_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_2 = pickle.load(filehandle)
    micro_model = meso_2.micro_model

    meso_pickled_filename = '/rHD2d_nocg_fw=4_bl=8dx.pickle'
    MesoModelLoadFile = pickle_directory + meso_pickled_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_4 = pickle.load(filehandle)

    meso_pickled_filename = '/rHD2d_nocg_fw=bl=8dx.pickle'
    MesoModelLoadFile = pickle_directory + meso_pickled_filename
    with open(MesoModelLoadFile, 'rb') as filehandle: 
        meso_8 = pickle.load(filehandle)    


    print('Finished reading pickled data\n')

    # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE
    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    num_slices_meso = int(config['Models_settings']['num_T_slices'])
    time_meso2 = meso_2.domain_vars['T'][int((num_slices_meso-1)/2)] 
    time_meso4 = meso_2.domain_vars['T'][int((num_slices_meso-1)/2)] 
    time_meso8 = meso_2.domain_vars['T'][int((num_slices_meso-1)/2)] 

    compatible = True
    if time_micro != time_meso2 or time_meso2 != time_meso4 or time_meso4 != time_meso8:
        compatible = False
    
    if not compatible:
        print("Slices of meso and micro models do not coincide. Careful!\n")
    else: 
        print("Comparing data at same time-slice, hurray!\n")

    # # PLOT SETTINGS
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2D()
    # diff_plot_settings =json.loads(config['Plot_settings']['diff_plot_settings']) 

    # Building the data for showing the relative difference between the various plots
    models = [micro_model, meso_2, meso_4, meso_8]  
    var_str = 'BC'
    comp = (0,)

    # full_range = [0.04,0.96]
    datamicro, extentmicro = visualizer.get_var_data(micro_model, var_str, time_micro, x_range, y_range, comp)
    data2, extent2 = visualizer.get_var_data(meso_2, var_str, time_meso2, x_range, y_range, comp)
    data4, extent4 = visualizer.get_var_data(meso_4, var_str, time_meso4, x_range, y_range, comp)
    data8, extent8 = visualizer.get_var_data(meso_8, var_str, time_meso8, x_range, y_range, comp)


    abs_diff2 = data2 - datamicro
    abs_diff4 = data4 - datamicro
    abs_diff8 = data8 - datamicro

    ar_mean = (np.abs(data2) + np.abs(datamicro))/2
    rel_diff2 = np.abs(datamicro -data2)/ar_mean 
    rel_diff2  = rel_diff2 / 2. 

    ar_mean = (np.abs(data4) + np.abs(datamicro))/2
    rel_diff4 = np.abs(data4 -datamicro)/ar_mean 
    rel_diff4 = rel_diff4 / 4

    ar_mean = (np.abs(data8) + np.abs(datamicro))/2
    rel_diff8 = np.abs(data8 - datamicro)/ar_mean 
    rel_diff8 = rel_diff8 /8


    # PLOTTING 
    fig = plt.figure(figsize=[13,4])
    ax1 = fig.add_subplot(1, 4, 1)
    ax2 = fig.add_subplot(1, 4, 2, sharey=ax1)
    ax3 = fig.add_subplot(1, 4, 3, sharey=ax1)
    ax4 = fig.add_subplot(1, 4, 4, sharey=ax1)
    axes = [ax1, ax2, ax3, ax4]
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    

    images =[]

    im = ax1.imshow(datamicro, extent=extentmicro, origin='lower', cmap='plasma') #, norm='log')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = ax2.imshow(rel_diff2, extent=extent2, origin='lower', cmap='plasma') #, norm='log')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_axis_off()
    images.append(im)

    im = ax3.imshow(rel_diff4, extent=extent4, origin='lower', cmap='plasma') #, norm='log')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_axis_off()
    images.append(im)

    im = ax4.imshow(rel_diff8, extent=extent8, origin='lower', cmap='plasma') #, norm='log')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    images.append(im)

    for i in range(len(axes)):
        axes[i].set_xlabel(r'$x$')
        axes[i].set_ylabel(r'$y$')

    title = micro_model.labels_var_dict[var_str]
    title += r'$,$ $a=0$'
    axes[0].set_title(title, fontsize=10)

    title = r'$Scaled$ $rel.$ $diff.,$ $L=2dx$'
    # title = meso_2.labels_var_dict[var_str]
    # title += r'$,$ $a=0,$ $L=2dx$'
    axes[1].set_title(title, fontsize=10)

    title = r'$Scaled$ $rel.$ $diff.,$ $L=4dx$'
    # title = meso_4.labels_var_dict[var_str]
    # title += r'$,$ $a=0,$ $L=4dx$'
    axes[2].set_title(title, fontsize=10)

    title = r'$Scaled$ $rel.$ $diff.,$ $L=8dx$'
    # title = meso_8.labels_var_dict[var_str]
    # title += r'$,$ $a=0,$ $L=8dx$'
    axes[3].set_title(title, fontsize=10)

    fig.tight_layout()

    # COMMON COLORMAP
    # Finding the global min and max and setting the shared colormap based on these.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    # fig.colorbar(images[0], ax=axes, orientation='vertical', location='right', shrink=0.52, pad=0.025 )

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


    format='png'
    dpi=400
    filename = f"/Lin_scale_filt_{var_str}_{comp[0]}." + format 
    plt.savefig(saving_directory + filename, format = format, dpi=dpi)
    
    

    # # ##################################################################################################
    # # # SCRIPT TO SHOW HOW THE IMPACT OF FILTERING SCALES WITH THE FILTER-SIZE: COARSE-GRAINING
    # # ##################################################################################################
    
    # # READING SIMULATION SETTINGS FROM CONFIG FILE
    # if len(sys.argv) == 1:
    #     print(f"You must pass the configuration file for the simulations.")
    #     raise Exception()
    
    # config = configparser.ConfigParser()
    # config.read(sys.argv[1])
    
    # # LOADING MESO AND MICRO MODELS 
    # pickle_directory = config['Directories']['pickled_files_dir']
    

    # print('=========================================================================')
    # print(f'Starting job on data from {pickle_directory}')
    # print('=========================================================================\n\n')

    # meso_pickled_filename = '/rHD2d_cg=fw=bl=2dx.pickle'
    # MesoModelLoadFile = pickle_directory + meso_pickled_filename
    # with open(MesoModelLoadFile, 'rb') as filehandle: 
    #     meso_2 = pickle.load(filehandle)
    # micro_model = meso_2.micro_model

    # meso_pickled_filename = '/rHD2d_cg=fw=bl=4dx.pickle'
    # MesoModelLoadFile = pickle_directory + meso_pickled_filename
    # with open(MesoModelLoadFile, 'rb') as filehandle: 
    #     meso_4 = pickle.load(filehandle)

    # meso_pickled_filename = '/rHD2d_cg=fw=bl=8dx.pickle'
    # MesoModelLoadFile = pickle_directory + meso_pickled_filename
    # with open(MesoModelLoadFile, 'rb') as filehandle: 
    #     meso_8 = pickle.load(filehandle)    


    # print('Finished reading pickled data\n')

    # # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE
    # num_snaps = micro_model.domain_vars['nt']
    # central_slice_num = int(num_snaps/2.)
    # time_micro = micro_model.domain_vars['t'][central_slice_num]
    # num_slices_meso = int(config['Models_settings']['num_T_slices'])
    # time_meso2 = meso_2.domain_vars['T'][int((num_slices_meso-1)/2)] 
    # time_meso4 = meso_2.domain_vars['T'][int((num_slices_meso-1)/2)] 
    # time_meso8 = meso_2.domain_vars['T'][int((num_slices_meso-1)/2)] 

    # compatible = True
    # if time_micro != time_meso2 or time_meso2 != time_meso4 or time_meso4 != time_meso8:
    #     compatible = False
    
    # if not compatible:
    #     print("Slices of meso and micro models do not coincide. Careful!\n")
    # else: 
    #     print("Comparing data at same time-slice, hurray!\n")

    # # # PLOT SETTINGS
    # plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    # x_range = plot_ranges['x_range']
    # y_range = plot_ranges['y_range']
    # saving_directory = config['Directories']['figures_dir']
    # visualizer = Plotter_2D()
    # # diff_plot_settings =json.loads(config['Plot_settings']['diff_plot_settings']) 

    # # Building the data for showing the relative difference between the various plots
    # models = [micro_model, meso_2, meso_4, meso_8]  
    # var_str = 'BC'
    # comp = (0,)

    # inset_ranges = json.loads(config['Plot_settings']['inset_ranges'])
    # inset_x_range = inset_ranges['x_range']
    # inset_y_range = inset_ranges['y_range']

    # # full_range = [0.04,0.96]
    # datamicro, extentmicro = visualizer.get_var_data(micro_model, var_str, time_micro, x_range, y_range, comp)
    # data2, extent2 = visualizer.get_var_data(meso_2, var_str, time_meso2, inset_x_range, inset_y_range, comp)
    # data4, extent4 = visualizer.get_var_data(meso_4, var_str, time_meso4, inset_x_range, inset_y_range, comp)
    # data8, extent8 = visualizer.get_var_data(meso_8, var_str, time_meso8, inset_x_range, inset_y_range, comp)


    # # PLOTTING 
    # fig = plt.figure(figsize=[13,4])
    # ax1 = fig.add_subplot(1, 4, 1)
    # ax2 = fig.add_subplot(1, 4, 2, sharey=ax1)
    # ax3 = fig.add_subplot(1, 4, 3, sharey=ax1)
    # ax4 = fig.add_subplot(1, 4, 4, sharey=ax1)
    # axes = [ax1, ax2, ax3, ax4]
    # plt.setp(ax3.get_yticklabels(), visible=False)
    # plt.setp(ax4.get_yticklabels(), visible=False)

    
    # images =[]

    # im = ax1.imshow(datamicro, extent=extentmicro, origin='lower', cmap='plasma') #, norm='log')
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # images.append(im)

    # im = ax2.imshow(rel_diff2, extent=extent2, origin='lower', cmap='plasma') #, norm='log')
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.set_axis_off()
    # images.append(im)

    # im = ax3.imshow(rel_diff4, extent=extent4, origin='lower', cmap='plasma') #, norm='log')
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.set_axis_off()
    # images.append(im)

    # im = ax4.imshow(rel_diff8, extent=extent8, origin='lower', cmap='plasma') #, norm='log')
    # divider = make_axes_locatable(ax4)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im, cax=cax, orientation='vertical')
    # images.append(im)

    # for i in range(len(axes)):
    #     axes[i].set_xlabel(r'$x$')
    #     axes[i].set_ylabel(r'$y$')

    # title = micro_model.labels_var_dict[var_str]
    # title += r'$,$ $a=0$'
    # axes[0].set_title(title)

    # # title = r'$Scaled$ $rel.$ $diff.,$ $L=2dx$'
    # title = meso_2.labels_var_dict[var_str]
    # title += r'$,$ $a=0,$ $L=2dx$'
    # axes[1].set_title(title)

    # # title = r'$Scaled$ $rel.$ $diff.,$ $L=4dx$'
    # title = meso_4.labels_var_dict[var_str]
    # title += r'$,$ $a=0,$ $L=4dx$'
    # axes[2].set_title(title)

    # # title = r'$Scaled$ $rel.$ $diff.,$ $L=8dx$'
    # title = meso_8.labels_var_dict[var_str]
    # title += r'$,$ $a=0,$ $L=8dx$'
    # axes[3].set_title(title)

    # fig.tight_layout()

    # # COMMON COLORMAP
    # # Finding the global min and max and setting the shared colormap based on these.
    # vmin = min(image.get_array().min() for image in images)
    # vmax = max(image.get_array().max() for image in images)
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # for im in images:
    #     im.set_norm(norm)

    # # fig.colorbar(images[0], ax=axes, orientation='vertical', location='right', shrink=0.52, pad=0.025 )

    # # Make images respond to changes in the norm of other images (e.g. via the
    # # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # # recurse infinitely! 
    # def update(changed_image):
    #     for im in images:
    #         if (changed_image.get_cmap() != im.get_cmap()
    #                 or changed_image.get_clim() != im.get_clim()):
    #             im.set_cmap(changed_image.get_cmap())
    #             im.set_clim(changed_image.get_clim())

    # for im in images:
    #     im.callbacks.connect('changed', update)


    # # Adding boundaries of zoomed in regions 
    # A = np.array([inset_x_range[0], inset_y_range[0]])
    # B = np.array([inset_x_range[1], inset_y_range[0]])
    # C = np.array([inset_x_range[1], inset_y_range[1]])
    # D = np.array([inset_x_range[0], inset_y_range[1]])
    # arrows_starts = [A, B, C, D]
    # arrows_increments = [arrows_starts[i] - arrows_starts[i-1] for i in range(1, len(arrows_starts))]
    # arrows_increments.append(A-D)
    # for i in range(len(arrows_starts)):
    #     ax1.arrow(arrows_starts[i][0], arrows_starts[i][1], arrows_increments[i][0], arrows_increments[i][1], \
    #                   width=0.005,color='white',head_length=0.0,head_width=0.0)

    # fig.tight_layout()

    # format='png'
    # dpi=400
    # filename = f"/Filter_scale_BC_bl=fw." + format 
    # plt.savefig(saving_directory + filename, format = format, dpi=dpi)