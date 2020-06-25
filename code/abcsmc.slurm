#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=SMC_cmp_m
#SBATCH --time=6:00:0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=36

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=d171-s1898201
# Replace [partition name] below with your partition name (e.g. standard,gpu-skylake)
#SBATCH --partition=standard
# Replace [qos name] below with your qos name (e.g. standard,long,gpu)
#SBATCH --qos=standard

# Set the number of threads to the CPUs per task
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch the parallel job
#   Using 36 threads per node
#   srun picks up the distribution from the sbatch options

source ~/miniconda3/bin/activate

srun python3 abcsmc3_m_log.py