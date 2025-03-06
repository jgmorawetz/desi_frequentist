#!/bin/bash
#SBATCH --job-name=MAPchains
#SBATCH --qos=regular
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

module load julia

srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="BAO" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="BAO" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="CMB" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="CMB" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB+DESY5SN" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB+DESY5SN" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB+PantheonPlusSN" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB+PantheonPlusSN" --variation="w0waCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB+Union3SN" --variation="LCDM" & 
srun -n 1 -c 1 julia MAP_DESIY1FullShape.jl --n_runs=50 --dataset="FS+BAO+CMB+Union3SN" --variation="w0waCDM" & 

wait