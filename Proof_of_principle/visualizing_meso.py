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
    # PRODUCE MANY DIFF PLOTS TO COMPARE MICRO AND MESO AT DIFFERENT
    # ENDTIME AND DIFFERENT WIDTHS. FOR EACH PLOT THE DIFFERENCE 
    # (ABSOLUTE AND RELATIVE). 
    ###############################################################

    # Loading the models
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_3.5_step0.5/20dx/pickled_files/800X800/"
    ET = sys.argv[1]
    MicroModelLoadFile = directory + "HD_2D_ET_" + ET + "_micro.pickle"
    with open(MicroModelLoadFile, 'rb') as filehandle:
        micro_model = pickle.load(filehandle)

    num_snaps = 21
    central_slice_num = int(num_snaps/2.)
    time_micro = micro_model.domain_vars['t'][central_slice_num]

    # saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/Figures/400X400/"
    saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_3.5_step0.5/20dx/Figures/800X800/"

    FW = ['2', '4', '8']
    for fw in FW:
        MesoModelLoadFile = directory + "resHD_2D_ET_" + ET + "_FW_"+ fw +"dx.pickle"
        with open(MesoModelLoadFile, 'rb') as filehandle: 
            meso_model = pickle.load(filehandle)

        meso_central_slice = 1
        time_meso = meso_model.domain_vars['T'][meso_central_slice]
        if time_meso != time_micro:
            print("Slices of meso and micro model do not coincide. Careful!")
        else: 
            print("Comparing data at same time-slice, hurray!")

        ranges_x = [0.02, 0.98]
        ranges_y = [0.02, 0.98]
        visualizer = Plotter_2D([11.97, 8.36])

        ######################################################
        # CODE BLOCKS TO VISUALIZE SEPARATE QUANTITIES
        ###################################################### 
        # # Plot for baryon current
        # vars = [['BC', 'BC', 'BC'], ['BC', 'BC', 'BC']]
        # models = [micro_model, meso_model]
        # components = [[(0,), (1,), (2,)], [(0,), (1,), (2,)]]
        # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True, rel_diff=True)
        # fig.tight_layout()
        # # plt.show()
        # filename = "comparing_BC_ET_" + ET + "_FW_"+ fw + ".svg"
        # plt.savefig(saving_directory + filename, format = "svg")


        # # # Plots for SET - diag and off-diagonal
        # vars = [['SET', 'SET', 'SET'], ['SET', 'SET', 'SET']]
        # models = [micro_model, meso_model]
        # components = [[(0,0), (1,1), (2,2)], [(0,0), (1,1), (2,2)]]
        # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True, rel_diff=True)
        # fig.tight_layout()
        # # plt.show()
        # filename = "comparing_SET_diagonal" + ET + "_FW_"+ fw + ".svg" 
        # plt.savefig(saving_directory + filename, format = "svg")

        # # vars = [['SET', 'SET', 'SET'], ['SET', 'SET', 'SET']]
        # models = [micro_model, meso_model]
        # components = [[(0,1), (0,2), (1,2)], [(0,1), (0,2), (1,2)]]
        # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True, rel_diff=True)
        # fig.tight_layout()
        # # plt.show()
        # filename = "comparing_SET_off_diag" + ET + "_FW_"+ fw + ".svg" 
        # plt.savefig(saving_directory + filename, format = "svg")


        # # Plots to compare observer with Favre-velocity (need to decompose structures first!)
        # meso_model.decompose_structures()
        # vars = [['U', 'U', 'U'], ['u_tilde', 'u_tilde', 'u_tilde']]
        # models = [meso_model, meso_model]
        # components = [[(0,), (1,), (2,)], [(0,), (1,), (2,)]]
        # fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True, rel_diff=True)
        # fig.tight_layout()
        # filename = "comparing_obs_VS_Favre" + ET + "_FW_" + fw + ".svg"
        # plt.savefig(saving_directory + filename, format = 'svg')


        ######################################################
        # CODE FOR SUMMARY PLOTS
        ###################################################### 
        vars = [['BC', 'SET'], ['BC', 'SET',]]
        models = [micro_model, meso_model]
        components = [[(0,), (0,0)], [(0,), (0,0)]]
        fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True, rel_diff=True)
        fig.tight_layout()
        filename = "ET_"+ ET+ "_fw_" + fw + "_scaling.pdf"
        plt.savefig(saving_directory + filename, format = 'pdf')

        vars = [['BC', 'SET'], ['BC', 'SET',]]
        models = [micro_model, meso_model]
        components = [[(2,), (0,2)], [(2,), (0,2)]]
        fig=visualizer.plot_vars_models_comparison(models, vars, time_meso, ranges_x, ranges_y, components_indices = components, diff_plot=True, rel_diff=True)
        fig.tight_layout()
        filename = "ET_"+ ET+ "_fw_" + fw + "_not_scaling.pdf"
        plt.savefig(saving_directory + filename, format = 'pdf')
    