#!/bin/bash
####################################
# JOB INFO: parallel, single node
####################################

#SBATCH --output=outputs/parallel_%A.out  
#SBATCH --nodes=1      
#SBATCH --ntasks=40                
#SBATCH --time=01:00:00 

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

python3 -u find_best_fit.py config_calibration.txt




