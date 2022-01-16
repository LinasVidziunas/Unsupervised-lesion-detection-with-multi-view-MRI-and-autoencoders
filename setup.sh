#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:00:00
#SBATCH --job-name=tf_mnist_test_setup
#SBATCH --output=tf_mnist_test_setup.out
 
# Set up environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda create -n tf_mnist_test_env --file requirements.txt -y
