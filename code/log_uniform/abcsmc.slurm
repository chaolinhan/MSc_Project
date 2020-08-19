#!/bin/bash

# Slurm job options (name, compute nodes, job time)
# Set job name and wall time
#SBATCH --job-name=cmp_l
#SBATCH --time=23:20:00
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

# Lauch the miniconda environment
source ~/miniconda3/bin/activate

# Launch the parallel job
# Set python file
sleep 5
srun --cpu-bind=cores ~/miniconda3/bin/python3 abcsmc5_m_log.py