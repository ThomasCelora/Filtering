#!/bin/bash
####################################
# JOB INFO: serial
####################################

    
#SBATCH --nodes=1                
#SBATCH --ntasks=1   
#SBATCH --time=00:10:00 
#SBATCH --output=outputs/serial_%A.out 


#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=fail
#SBATCH --mail-type=end          # send email when job ends
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
#####################################
# python3 -u regress+residual_check.py config_calibration.txt
# python3 -u visualizing_correlations.py config_calibration.txt
# python3 -u reducing_regressors.py config_calibration.txt 
python3 -u hunting_correlations.py config_calibration.txt
# python3 -u regressing_residual.py config_calibration.txt




