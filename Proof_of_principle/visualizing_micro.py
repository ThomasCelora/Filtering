import sys
# import os
sys.path.append('/home/tc2m23/Filtering/master_files/')
# sys.path.append('/Users/thomas/Dropbox/Work/projects/Filtering/master_files')
import pickle

from FileReaders import *
from MicroModels import *
from Visualization import *

if __name__ == '__main__':

    # READING DATA
    # CPU_start_time = time.perf_counter()
    directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/pickled_files/400X400/"
    ET=str(sys.argv[1])
    MicroModelLoadFile = directory + "IdealHD_2D_ET_" + ET+ "_micro.pickle"

    with open(MicroModelLoadFile, 'rb') as filehandle: 
        micro_model = pickle.load(filehandle)

    ranges = [0.01, 0.99]
    visualizer = Plotter_2D([11.97, 8.36])
    num_snaps = 11
    central_slice = int(num_snaps/2)
    print("Plotting data from central slice, num: {}".format(central_slice))
    saving_directory = "/scratch/tc2m23/KHIRandom/hydro/ET_1_2_2.5_3_3.5/10dx_after/Figures/400X400/"
    # saving_directory = "./"

    vars = ['BC', 'BC', 'BC', 'n', 'W', 'vx']
    components = [(0,), (1,), (2,), (), (), ()]
    fig=visualizer.plot_vars(micro_model, vars, micro_model.domain_vars['t'][central_slice], ranges, ranges, components_indices = components)
    fig.tight_layout()
    # plt.show()
    filename = "micro_ET_"+ET+"_BC.svg"
    plt.savefig(saving_directory + filename, format = "svg")

    # vars = ['SET', 'SET', 'SET', 'SET', 'SET', 'SET']
    # components = [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
    # fig=visualizer.plot_vars(micro_model, vars, micro_model.domain_vars['t'][central_slice], ranges, ranges, components_indices = components)
    # fig.tight_layout()
    # # plt.show()
    # filename = "micro_ET_"+ET+"_SET.svg"
    # plt.savefig(saving_directory + filename, format = "svg")

    # vars = ['W', 'vx', 'vy', 'n', 'p', 'e']
    # fig=visualizer.plot_vars(micro_model, vars, micro_model.domain_vars['t'][central_slice], ranges, ranges)
    # fig.tight_layout()
    # # plt.show()
    # filename = "micro_ET_"+ET+"_prims.svg"
    # plt.savefig(saving_directory + filename, format = "svg")

