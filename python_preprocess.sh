#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20000
#SBATCH --output=preprocess_output.log
#SBATCH --error=preprocess_error.log
module load Python/3.10.4-GCCcore-11.3.0
python ./preprocess_data.py
