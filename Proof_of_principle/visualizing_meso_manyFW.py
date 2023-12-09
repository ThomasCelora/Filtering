import sys
# import os
sys.path.append('/home/tc2m23/Filtering/master_files/')
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
import pickle

from FileReaders import *
from MicroModels import *
from MesoModels import * 
from Visualization import *

if __name__ == '__main__':

    ###############################################################
    # BLOCK TO COMPARE MICRO AND MESO WITH DIFF FILTERING SIZES
    ###############################################################
    # Loading the models
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
    ET = sys.argv[1]
    # ET = "1.0"
    MicroModelLoadFile = directory + "IdealHD_2D_ET_" + ET + "_micro.pickle"
    with open(MicroModelLoadFile, 'rb') as filehandle:
        micro_model = pickle.load(filehandle)

    models = [micro_model]
    FW = ['2', '4', '8']
    if ET == "2.0":
        FW = ['2', '4']
    for fw in FW: 
        MesoModelLoadFile = directory + "resHD_2D_ET_" + ET + "_FW_"+ fw +"dx.pickle"
        with open(MesoModelLoadFile, 'rb') as filehandle: 
            models.append(pickle.load(filehandle))

    saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/Figures/400X400/"
    # saving_directory = "./"

    # Checking for alignment of slices
    num_snaps = 11
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]
    for i in range(1, len(models)):
        time_meso = models[i].domain_vars['T'][0]
        if time_meso != time_micro:
            print('Careful: micro and meso slices appear not to be aligned!')


    ranges_x = [0.04, 0.96]
    ranges_y = [0.04, 0.96]
    visualizer = Plotter_2D([11.97, 8.36])

    # Plot for baryon current
    to_be_plotted = ['BC', 'BC', 'BC']
    comp_to_plot = [(0,), (1,), (2,)]
    vars = [to_be_plotted for _ in range(len(models))]
    components = [comp_to_plot for _ in range(len(models))]
    fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components)
    fig.tight_layout()
    # plt.show()
    filename = "manyFW_BC_ET_" + ET + ".svg"
    plt.savefig(saving_directory + filename, format = "svg")

    # plot for SET-diagonal 
    to_be_plotted = ['SET', 'SET', 'SET']
    comp_to_plot = [(0,0), (1,1), (2,2)]
    vars = [to_be_plotted for _ in range(len(models))]
    components = [comp_to_plot for _ in range(len(models))]
    fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components)
    fig.tight_layout()
    # plt.show()
    filename = "manyFW_SETdiag_ET_" + ET + ".svg"
    plt.savefig(saving_directory + filename, format = "svg")

    # plot for SET off-diagonal 
    to_be_plotted = ['SET', 'SET', 'SET']
    comp_to_plot = [(0,1), (0,2), (1,2)]
    vars = [to_be_plotted for _ in range(len(models))]
    components = [comp_to_plot for _ in range(len(models))]
    fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components)
    fig.tight_layout()
    # plt.show()
    filename = "manyFW_SEToffdiag_ET_" + ET + ".svg"
    plt.savefig(saving_directory + filename, format = "svg")


    ###############################################################
    # BLOCK TO PLOT THE DIFFERENCE BETWEEN TWO FILTER WIDTHS
    ###############################################################
    # directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
    # ET = sys.argv[1]
    # models = []
    # FW = ['2', '8']
    # if ET == "2.0":
    #     FW = ['2', '4']
    # for fw in FW: 
    #     MesoModelLoadFile = directory + "resHD_2D_ET_" + ET + "_FW_"+ fw +"dx.pickle"
    #     with open(MesoModelLoadFile, 'rb') as filehandle: 
    #         models.append(pickle.load(filehandle))

    # saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/Figures/400X400/"
    # # saving_directory = "./"

    # time_meso = models[0].domain_vars['T'][0]
    # ranges_x = [0.04, 0.96]
    # ranges_y = [0.04, 0.96]
    # visualizer = Plotter_2D([11.97, 8.36])

    # # Plot for baryon current
    # to_be_plotted = ['BC', 'BC', 'BC']
    # comp_to_plot = [(0,), (1,), (2,)]
    # vars = [to_be_plotted for _ in range(len(models))]
    # components = [comp_to_plot for _ in range(len(models))]
    # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True)
    # # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, \
    # #                                            method = "interpolate", interp_dims = (150, 150), diff_plot=True)
    # fig.tight_layout()
    # # plt.show()
    # filename = "contrastFW_BC_ET_" + ET + ".svg"
    # plt.savefig(saving_directory + filename, format = "svg")

    # # plot for SET-diagonal 
    # to_be_plotted = ['SET', 'SET', 'SET']
    # comp_to_plot = [(0,0), (1,1), (2,2)]
    # vars = [to_be_plotted for _ in range(len(models))]
    # components = [comp_to_plot for _ in range(len(models))]
    # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True)
    # # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, \
    # #                                            method = "interpolate", interp_dims = (150, 150), diff_plot=True)
    # fig.tight_layout()
    # # plt.show()
    # filename = "contrastFW_SETdiag_ET_" + ET + ".svg"
    # plt.savefig(saving_directory + filename, format = "svg")

    # # plot for SET off-diagonal 
    # to_be_plotted = ['SET', 'SET', 'SET']
    # comp_to_plot = [(0,1), (0,2), (1,2)]
    # vars = [to_be_plotted for _ in range(len(models))]
    # components = [comp_to_plot for _ in range(len(models))]
    # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True)
    # # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, \
    # #                                            method = "interpolate", interp_dims = (150, 150), diff_plot=True)
    # fig.tight_layout()
    # # plt.show()
    # filename = "constrastFW_SEToffdiag_ET_" + ET + ".svg"
    # plt.savefig(saving_directory + filename, format = "svg")