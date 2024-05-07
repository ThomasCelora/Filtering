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

    # ###################################################################
    # SCRIPT TO VISUALIZE VARIOUS MESO QUANTITIES AND COMPARE WITH MICRO 
    # ###################################################################
    
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

    # CHECKING WE ARE COMPARING DATA FROM THE SAME TIME-SLICE
    num_snaps = micro_model.domain_vars['nt']
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    meso_grid_info = json.loads(config['Meso_model_settings']['meso_grid'])
    num_slices_meso = meso_grid_info['num_T_slices']
    time_meso = meso_model.domain_vars['T'][int((num_slices_meso-1)/2)] 
    if time_meso != time_micro:
        print("Slices of meso and micro model do not coincide. Careful!\n")
    else: 
        print("Comparing data at same time-slice, hurray!\n")

    # # PLOT SETTINGS
    plot_ranges = json.loads(config['Plot_settings']['plot_ranges'])
    x_range = plot_ranges['x_range']
    y_range = plot_ranges['y_range']
    saving_directory = config['Directories']['figures_dir']
    visualizer = Plotter_2D([12, 8])
    diff_plot_settings =json.loads(config['Plot_settings']['diff_plot_settings']) 
    diff_method = diff_plot_settings['method']
    interp_dims = diff_plot_settings['interp_dims']

    # # PLOTTING MICRO VS FILTERED BC AND SET
    # # #####################################
    # start_time = time.perf_counter()
    # vars = [['BC'],['BC']]
    # models = [micro_model, meso_model]
    # comp = 1
    # components= [[(comp,)],[(comp,)]]
    # # norms = [['log'], ['log'], ['log']]
    # norms = [['symlog'], ['symlog'], ['log']]
    # # cmaps = [['plasma'],['plasma'],['plasma']]
    # cmaps = [['coolwarm'],['coolwarm'],['plasma']]
    # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices = components, 
    #                                            interp_dims = interp_dims, method = diff_method, diff_plot=False, rel_diff=True, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # format='png'
    # dpi=400
    # filename = "/Comparing_BC" +f"_{comp}." + format
    # plt.savefig(saving_directory + filename, format = format, dpi=dpi)

    # vars = [['SET'], ['SET']]
    # models = [micro_model, meso_model]
    # comp = 1
    # components = [[(comp,comp)], [(comp,comp)]]
    # norms = [['log'], ['log'], ['log']]
    # # norms = [['symlog'], ['symlog'], ['log']]
    # cmaps = [['plasma'],['plasma'],['plasma']]
    # # cmaps = [['coolwarm'],['coolwarm'],['plasma']]
    # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, x_range, y_range, components_indices = components, 
    #                                            interp_dims = interp_dims, method = diff_method, diff_plot=False, rel_diff=True, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # format='png'
    # dpi=400
    # filename = "/Comparing_SET" +f"_({comp},{comp})."+format
    # plt.savefig(saving_directory + filename, format = format, dpi=dpi)
    # time_taken = time.perf_counter() - start_time
    # print(f'Finished plotting model comparison: time taken (X2) ={time_taken}\n')

    # # # PLOTTING THE DECOMPOSED SET 
    # # #############################
    vars_strs = ['pi_res', 'pi_res', 'pi_res', 'pi_res', 'pi_res', 'pi_res']
    norms = ['mysymlog', 'mysymlog', 'mysymlog', 'mysymlog', 'mysymlog', 'mysymlog']
    cmaps = ['Spectral_r','Spectral_r','Spectral_r','Spectral_r','Spectral_r','Spectral_r']
    components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/DecomposedSET_1.pdf" 
    plt.savefig(saving_directory + filename, format = 'pdf')

    vars_strs = ['q_res', 'q_res', 'q_res', 'Pi_res', 'p_tilde', 'p_filt']
    norms = ['mysymlog', 'mysymlog', 'mysymlog', 'log', 'log', 'log']
    cmaps = ['Spectral_r','Spectral_r','Spectral_r', 'plasma', 'plasma', 'plasma']
    components = [(0,), (1,), (2,), (), (), ()]
    fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    fig.tight_layout()
    time_for_filename = str(round(time_meso,2))
    filename = "/DecomposedSET_2.pdf"
    plt.savefig(saving_directory + filename, format = 'pdf')
    print('Finished plotting decomposition of SET\n')

    # # # # PLOTTING THE DERIVATIVES OF FAVRE VEL AND TEMPERATURE
    # # ######################################################### 
    # favre_vel_components = [0,1,2]
    # for i in range(len(favre_vel_components)):
    #     components = [tuple([favre_vel_components[i]])]
    #     for j in range(3):
    #         components.append(tuple([j,favre_vel_components[i]]))
    #     if i==0: 
    #         norms = ['log', 'mysymlog', 'mysymlog', 'mysymlog']
    #         cmaps = ['plasma', 'seismic', 'seismic', 'seismic']
    #     else: 
    #         norms = ['mysymlog', 'mysymlog', 'mysymlog', 'mysymlog']
    #         cmaps = ['seismic', 'seismic', 'seismic', 'seismic']

    #     vars_strs = ['u_tilde', 'D_u_tilde', 'D_u_tilde', 'D_u_tilde']
    #     fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    #     fig.tight_layout()
    #     time_for_filename = str(round(time_meso,2))     
    #     filename = "/D_favre_{}.pdf".format(favre_vel_components[i])
    #     plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished plotting derivatives of favre velocity\n', flush=True)

    # vars_strs = ['T_tilde', 'D_T_tilde', 'D_T_tilde', 'D_T_tilde']
    # components = [(), (0,), (1,), (2,)]
    # norms = ['log', 'mysymlog', 'mysymlog', 'mysymlog']
    # cmaps = ['plasma', 'seismic', 'seismic', 'seismic']
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # time_for_filename = str(round(time_meso,2))
    # filename = "/D_T.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished plotting temperature derivatives\n', flush=True)

    # # # PLOTTING SHEAR, ACCELERATION, EXPANSION AND TEMPERATURE DERIVATIVES
    # ########################################################################
    # vars_strs = ['shear_tilde', 'shear_tilde', 'shear_tilde', 'shear_tilde', 'shear_tilde', 'shear_tilde']
    # components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    # norms = ['mysymlog','mysymlog','mysymlog','mysymlog','mysymlog','mysymlog']
    # cmaps = ['seismic','seismic','seismic','seismic','seismic','seismic'] 
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # time_for_filename = str(round(time_meso,2))
    # filename = "/Shear_comps.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished plotting shear\n', flush=True)

    # vars_strs = ['acc_tilde', 'acc_tilde', 'acc_tilde', 'Theta_tilde', 'Theta_tilde', 'Theta_tilde']
    # components = [(0,), (1,), (2,), (0,), (1,), (2,)]
    # norms = ['mysymlog','mysymlog','mysymlog','mysymlog','mysymlog','mysymlog']
    # cmaps = ['seismic','seismic','seismic','seismic','seismic','seismic'] 
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # time_for_filename = str(round(time_meso,2))
    # filename = "/Acc+Tderivs.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished plotting DaT\n', flush=True)

    # vars_strs = ['vort_tilde', 'vort_tilde', 'vort_tilde']
    # components = [(0,1), (0,2), (1,2)]
    # norms = [None, None, None] #['mysymlog','mysymlog','mysymlog']
    # cmaps = ['plasma','plasma','plasma'] 
    # fig = visualizer.plot_vars(meso_model, vars_strs, time_meso, x_range, y_range, components_indices=components, norms=norms, cmaps=cmaps)
    # fig.tight_layout()
    # time_for_filename = str(round(time_meso,2))
    # filename = "/Vorticity_comps.pdf"
    # plt.savefig(saving_directory + filename, format = 'pdf')
    # print('Finished plotting vorticity\n', flush=True)

    # # SUMMARY PLOT OF THE RESIDUALS
    vars_strs = ['Pi_res', 'pi_res_sq', 'q_res_sq']
    vars = []
    extents = []
    for var_str in vars_strs:
        temp_var, temp_extent = visualizer.get_var_data(meso_model, var_str, time_meso, x_range, y_range) 
        if var_str != 'Pi_res':
            temp_var = np.sqrt(temp_var)
        vars.append(temp_var)
        extents.append(temp_extent)



    norms = ['log', 'log', 'log']
    cmaps = ['plasma', 'plasma', 'plasma']


    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(1,3, squeeze=False, figsize=[12,4], sharey=True)
    axes = axes.flatten()

    images = []

    for i in range(len(axes)):

        if norms[i] == 'mysymlog': 
            ticks, labels, nodes = MySymLogPlotting.get_mysymlog_var_ticks(vars[i])
            data_to_plot = MySymLogPlotting.symlog_var(vars[i])
            mynorm = MyThreeNodesNorm(nodes)
            im = axes[i].imshow(data_to_plot, extent=extents[i], origin='lower', norm=mynorm, cmap=cmaps[i])
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels(labels)

        else:
            im = axes[i].imshow(vars[i], extent=extents[i], origin='lower', cmap=cmaps[i], norm=norms[i])
            # divider = make_axes_locatable(axes[i])
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # fig.colorbar(im, cax=cax, orientation='vertical')
            images.append(im)

        axes[i].set_xlabel(r'$x$', fontsize=10)
        axes[i].set_ylabel(r'$y$', fontsize=10)

    axes[0].set_title(r'$\tilde{\Pi}$', fontsize=14)
    axes[1].set_title(r'$\sqrt{\tilde{\pi}_{ab}\tilde{\pi}^{ab}}$', fontsize=14)
    axes[2].set_title(r'$\sqrt{\tilde{q}_{a}\tilde{q}^{a}}$', fontsize=14)

    fig.tight_layout()


    ###########################################################
    # Adapt the following if you want to have a single colormap 
    ###########################################################
    # SETTING UP A COMMON COLORMAP 
    # Finding the global min and max and setting the colormap to be based on these.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axes.ravel().tolist(), orientation='vertical', location='right', shrink=0.9)

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
    filename = f"/Residuals." + format 
    plt.savefig(saving_directory + filename, format = format, dpi=dpi)


    # # NOW SUMMARY PLOT FOR THE CLOSURE INGREDIENTS

    vars_strs = ['exp_tilde', 'shear_sq', 'Theta_sq']
    vars = []
    extents = []
    for var_str in vars_strs:
        temp_var, temp_extent = visualizer.get_var_data(meso_model, var_str, time_meso, x_range, y_range) 
        if var_str != 'exp_tilde':
            temp_var = np.sqrt(temp_var)
        vars.append(temp_var)
        extents.append(temp_extent)



    norms = ['symlog', 'log', 'log']
    cmaps = ['Spectral_r', 'plasma', 'plasma']


    plt.rc("font",family="serif")
    plt.rc("mathtext",fontset="cm")
    fig, axes = plt.subplots(1,3, squeeze=False, figsize=[12,4], sharey=True)
    axes = axes.flatten()

    images = []

    for i in range(len(axes)):

        if norms[i] == 'mysymlog': 
            ticks, labels, nodes = MySymLogPlotting.get_mysymlog_var_ticks(vars[i])
            data_to_plot = MySymLogPlotting.symlog_var(vars[i])
            mynorm = MyThreeNodesNorm(nodes)
            im = axes[i].imshow(data_to_plot, extent=extents[i], origin='lower', norm=mynorm, cmap=cmaps[i])
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels(labels)

        else:
            im = axes[i].imshow(vars[i], extent=extents[i], origin='lower', cmap=cmaps[i], norm=norms[i])
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            images.append(im)

        axes[i].set_xlabel(r'$x$', fontsize=10)
        axes[i].set_ylabel(r'$y$', fontsize=10)

    axes[0].set_title(r'$\tilde{\theta}$', fontsize=14)
    axes[1].set_title(r'$\sqrt{\tilde{\sigma}_{ab}\tilde{\sigma}^{ab}}$', fontsize=14)
    axes[2].set_title(r'$\sqrt{\tilde{\Theta}_{a}\tilde{\Theta}^{a}}$', fontsize=14)

    fig.tight_layout()

    format='png'
    dpi=400
    filename = f"/Gradients." + format 
    plt.savefig(saving_directory + filename, format = format, dpi=dpi)
    
