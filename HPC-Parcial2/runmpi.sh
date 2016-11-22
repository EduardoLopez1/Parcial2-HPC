#!/bin/bash

#SBATCH --job-name TestGPU
#SBATCH --output TestGPU.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun Main
