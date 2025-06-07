#!/bin/bash
#SBATCH --job-name=run_models
#SBATCH --output=zero_shot_output.log
#SBATCH --error=zero_shot_error.log
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH --cpus-per-task=4

module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

python ./run_models.py

