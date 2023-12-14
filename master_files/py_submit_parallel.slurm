#!/bin/bash
####################################
# JOB INFO: single node allocating max core available
####################################

#SBATCH --output=outputs/slurm-%A.out        
#SBATCH --nodes=1           
#SBATCH --ntasks=1 #number of MPI processes         
#SBATCH --cpus-per-task=40 #number of threads per MPI process
#SBATCH --time=00:30:00 

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
# LAUNCHING THE JOBS
####################################

# python3 -u Filters.py 
python3 -u MesoModels.py
