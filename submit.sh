#!/bin/bash
#SBATCH --job-name=spectral_function
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-10
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --partition=short
#SBATCH --output=logs/spec__%A_%a.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

file=spectral-function.py

task_id=$SLURM_ARRAY_TASK_ID

L=8
J=1.0
t_max=1000.0
N_ensemble=$SLURM_CPUS_PER_TASK

if [ $task_id -le 5 ]; then
    g=0.0
else
    g=2.0
fi

python3 $file $L $J $g $t_max $N_ensemble $task_id