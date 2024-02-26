#!/bin/bash
####################################
# JOB INFO: 
# parallel, single node, max num of cores
####################################

#SBATCH --output=outputs/Meso-pickling-%A.out  
#SBATCH --nodes=1      
#SBATCH --ntasks=40                
#SBATCH --time=03:00:00 

# uncomment if job array
##SBATCH --array=1-6
##SBATCH --output=outputs/slurm-%A_%a.out 

#SBATCH --mail-type=begin        
#SBATCH --mail-type=fail
#SBATCH --mail-type=end          
#SBATCH --mail-user=celora@ice.csic.es


####################################
# LOADING THE CONDA ENVIRONMENT 
####################################

module load conda/py3-latest
# I get warnings to use 'conda deactivate' instead but that doesn't work
source deactivate 
conda activate myenv

####################################
# RETRIEVING PARAMETERS FOR RUNS: OLD, IGNORE
####################################

# ET=$(awk -v var=${SLURM_ARRAY_TASK_ID} '$1==var {print $2}' config.txt)

####################################
# LAUNCHING THE JOBS
####################################

python3 -u pickling_meso.py config_meso.txt
# python3 -u pickling_meso_setup_array.py config_meso_array.txt




