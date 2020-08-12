#!/bin/bash

# Slurm job options (name, compute nodes, job time)
# TODO: set job name and wall time
#SBATCH --job-name=ib1f2
#SBATCH --time=1:59:0
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
# TODO: set python file
srun --cpu-bind=cores ~/miniconda3/bin/python3 inferBack_base.py