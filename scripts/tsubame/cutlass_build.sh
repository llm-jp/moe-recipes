#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=1:00:00
#$ -p -5

# priotiry: -5: normal, -4: high, -3: highest

# Load modules
module use /gs/fs/tga-NII-LLM/modules/modulefiles

module load ylab/cuda/12.1
module load ylab/cudnn/8.9.7
module load ylab/nccl/cuda-12.2/2.20.5
module load ylab/hpcx/2.17.1
module load ninja/1.11.1

# git clone
git clone git@github.com:NVIDIA/cutlass.git
cd cutlass

# set environment variables
export CUDACXX=${CUDA_HOME}/bin/nvcc

# build for H100 (see: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=90


