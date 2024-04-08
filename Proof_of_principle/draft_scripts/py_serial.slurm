#!/bin/bash
####################################
# JOB INFO:
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
python3 -u filter_scaling.py config_draft.txt
python3 -u comparison_zoom.py config_draft.txt
python3 -u const_coeff_all.py config_draft.txt



