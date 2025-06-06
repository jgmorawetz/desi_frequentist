#!/bin/bash
#SBATCH --job-name=MAP
#SBATCH --qos=shared
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
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
chains_path="/global/homes/j/jgmorawe/FrequentistExample1/FrequentistExample1/MCMC_results_paper/${dataset}_${variation}_5000_1000_0.65_x6merged_chain.npy"

julia MAP_DESIY1FullShape.jl --n_runs $n_runs --dataset "$dataset" --variation "$variation" --chains_path "$chains_path"