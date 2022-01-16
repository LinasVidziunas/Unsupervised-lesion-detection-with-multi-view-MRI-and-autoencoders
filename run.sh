#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=00:30:00
#SBATCH --job-name=tf_mnist_test
#SBATCH --output=tf_mnist_test_01.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda-python39
conda activate tf_mnist_test_env
# Run the Python script that uses the GPU
python -u tf_mnist_test.py
