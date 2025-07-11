#!/bin/bash
#SBATCH --job-name=MCMC_correct
#SBATCH --qos=shared
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --array=43,45,46,47,79,80,81

module load julia

PARAMS=(
    "FS LCDM"
    "FS w0waCDM"
    "BAO LCDM"
    "BAO w0waCDM"
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

PARAM_INDEX=$((SLURM_ARRAY_TASK_ID/6))
RUN_INDEX=$((SLURM_ARRAY_TASK_ID%6+1))

PARAM_SET=(${PARAMS[$PARAM_INDEX]})
dataset="${PARAM_SET[0]}"
variation="${PARAM_SET[1]}"
save_dir="/global/homes/j/jgmorawe/FrequentistExample1/FrequentistExample1/MCMC_results_paper_corrected/"

julia run_MCMC_chains.jl --n_steps=5000 --n_burn=1000 --acceptance=0.65 --chain_index="$RUN_INDEX" --dataset="$dataset" --variation="$variation" --save_dir="$save_dir"