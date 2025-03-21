#!/bin/bash
#SBATCH --job-name=MAPchain
#SBATCH --account=rrg-wperciva
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --time=96:00:00
#SBATCH --array=0-11


PARAMS=(
    "FS LCDM"
    "FS w0waCDM"
    "FS+BAO LCDM"
    "FS+BAO w0waCDM"
    "FS+BAO+CMB LCDM"
    "FS+BAO+CMB w0waCDM"
    "FS+BAO+CMB+DESY5SN LCDM"
    "FS+BAO+CMB+DESY5SN w0waCDM"
    "FS+BAO+CMB+PantheonPlusSN LCDM"
    "FS+BAO+CMB+PantheonPlusSN w0waCDM"
    "FS+BAO+CMB+Union3SN LCDM"
    "FS+BAO+CMB+Union3SN w0waCDM"
)

PARAM_SET=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
n_runs=100
dataset="${PARAM_SET[0]}"
variation="${PARAM_SET[1]}"

julia MAP_DESIY1FullShape.jl --n_runs $n_runs --dataset "$dataset" --variation "$variation"