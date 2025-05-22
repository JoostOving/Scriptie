#!/bin/bash
#SBATCH --job-name=run_models
#SBATCH --output=zero_shot_output.log
#SBATCH --error=zero_shot_error.log
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=20000
#SBATCH --cpus-per-task=4

VENV_PATH="/home3/s5251370/venvs/first_env"

# Load modules
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source "$VENV_PATH/bin/activate"

export CUDA_VISIBLE_DEVICES=0,1

export HF_HOME=/scratch/$USER/huggingface_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Run your script
python ./eval_tester.py

