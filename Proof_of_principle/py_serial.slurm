#!/bin/bash
####################################
# JOB INFO:
####################################

    
#SBATCH --nodes=1                
#SBATCH --ntasks=1   
#SBATCH --time=00:05:00 
#SBATCH --output=outputs/visualize_slurm-%A.out 
# #SBATCH --output=outputs/calibration_slurm-%A.out 
# #SBATCH --output=outputs/reducing_slurm-%A.out  

# uncomment if array:
##SBATCH --array=1-3
##SBATCH --output=outputs/slurm-%A_%a.out  


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
# RETRIEVING THE PARAMETERS FOR THE RUNS: OLD
####################################

# ET=$(awk -v var=${SLURM_ARRAY_TASK_ID} '$1==var {print $2}' config.txt)


####################################
# LAUNCHING THE JOBS
#####################################

# python3 -u visualizing_micro.py config_visualizing.txt
# python3 -u visualizing_obs.py config_visualizing.txt
# python3 -u visualizing_meso.py config_visualizing.txt
# python3 -u visualizing_residuals.py config_calibration.txt 

python3 -u visualizing_correlations.py config_calibration.txt
python3 -u reducing_regressors.py config_calibration.txt 
python3 -u hunting_correlations.py config_calibration.txt
python3 -u regressing_residual.py config_calibration.txt

