#!/bin/bash
#$ -l h_rt=1:0:0
#$ -l h_vmem=11G
#$ -pe smp 12
#$ -l gpu_type=ampere
#$ -l gpu=1
#$ -wd /data/home/ec23594/3DGS/new_Head_Avatar/Gaussian-Head-Avatar/preprocess
#$ -cwd
#$ -j y
#$ -m ea
#$ -o logs/

# Load modules

module load anaconda3
module load cmake
module load gcc/7.1.0
module load cuda/11.3
# Clone 
#git clone https://github.com/YuelangX/Gaussian-Head-Avatar.git --recursive


export CUDA_HOME=/share/apps/centos7/cuda/11.3
conda activate preprocess
conda install -c conda-forge opencv

python preprocess_nersemble.py