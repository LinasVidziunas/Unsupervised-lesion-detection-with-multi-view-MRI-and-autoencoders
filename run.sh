#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=gpuA100 
#SBATCH --time=20:00:00
#SBATCH --job-name=cAE
#SBATCH --output=cAE_01.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4

# Run the Python script that uses the GPU
python3 -u cAE.py
