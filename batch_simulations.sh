#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --output=output/d_%a.out
#SBATCH --partition=small-g  
#SBATCH --account=project_462000736
#SBATCH --gres=gpu:1
#SBATCH --array=0-200

module load LUMI/24.03 partition/G
module load CuPy/12.2.0-cpeGNU-24.03-rocm
module load cray-python

python3 run_jneuro_subjects_sim_sparse.py --kl_idx=${SLURM_ARRAY_TASK_ID} --grid_size=5
#   squeue -u $USER | grep run | awk '{print $1}' | xargs -n 1 scancel