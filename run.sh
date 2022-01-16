#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=gpuA100 
#SBATCH --time=02:00:00
#SBATCH --job-name=vae_mnist
#SBATCH --output=vae_mnist_01.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4

# Run the Python script that uses the GPU
python -u vae_mnist.py
